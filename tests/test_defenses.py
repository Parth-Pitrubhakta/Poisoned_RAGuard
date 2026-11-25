"""Step 5 defenses: faithfulness of the TrustRAG port, filter semantics, and end-to-end runs in the CPU toy world."""
import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest

from prag.config import ROOT, RunConfig, load_registry
from prag.defenses.base import DefenseContext, available, build_defense
from prag.defenses.trustrag import (document_list, final_prompt, group_ngram_filtering, kmeans_filter, rouge_l_f,
                                    stage_one_prompt, stage_two_prompt)
from prag.llm.backends import EchoLLM
from prag.pipeline import Components, execute
from prag.retrieval.encoders import HashEncoder
from prag.retrieval.index import FaissRetriever, PrecomputedRetriever
from prag.retrieval.merge import Passage
from prag.tracing.schema import trace_errors
from toyworld import build_dense_index, config_dict, make_world

REF = ROOT / "third_party" / "trustrag_ref" / "defend_module.py"


# ---- TrustRAG prompts == reference f-strings (parsed with ast, never executed) ------------------------------------
def _ref_fstrings(func: str) -> dict[str, ast.JoinedStr]:
    tree = ast.parse(REF.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
    return {n.targets[0].id: n.value for n in ast.walk(fn)
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.JoinedStr)}


def _render(js: ast.JoinedStr, env: dict[str, str]) -> str:
    out = []
    for v in js.values:
        out.append(v.value if isinstance(v, ast.Constant) else env[ast.unparse(v.value)])
    return "".join(out)


@pytest.mark.skipif(not REF.exists(), reason="reference copy not present")
def test_trustrag_prompts_match_reference_verbatim():
    fs = _ref_fstrings("conflict_query")
    q, ic, ext, ik = "who wrote x {weird}", "DOCS\nMemorized Documents:IK", "EXTERNAL {x}", "INTERNAL"
    assert stage_one_prompt(q) == _render(fs["stage_one_prompt"], {"questions[i]": q})
    docs = ["alpha", "beta"]
    assert document_list(docs) == "Externally Retrieved Document0:alpha\nExternally Retrieved Document1:beta\n"
    assert stage_two_prompt(q, docs, "IK") == _render(fs["stage_two_prompt"], {
        "initial_context": document_list(docs) + "Memorized Documents:IK", "questions[i]": q})
    assert final_prompt(q, ext, ik) == _render(fs["final_prompt"], {
        "questions[i]": q, "stage_two_responses[i]": ext, "internal_knowledges[i]": ik})
    assert ic  # silence unused


def test_reference_thresholds_unchanged():
    src = REF.read_text()
    assert "threshold=0.88" in src and src.count("> 0.25") >= 2
    assert "KMeans(n_clusters=2,n_init=10,max_iter=500, random_state=0)" in src


# ---- ROUGE-L and k-means branches --------------------------------------------------------------------------------
def test_rouge_l():
    assert rouge_l_f("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert rouge_l_f("a b c d", "a x c y") == pytest.approx(0.5)
    assert rouge_l_f("Hello, World!", "hello world") == pytest.approx(1.0)  # tokenizer: lowercase, punctuation dropped
    assert rouge_l_f("", "x") == 0.0


def test_group_ngram_filtering_drops_near_duplicates():
    texts = ["who wrote moby dick. Mark Twain wrote it", "who wrote moby dick. Twain is the author",
             "pasta needs salted water and patience"]
    assert group_ngram_filtering(texts) == {0, 1}


def test_kmeans_drops_dense_poison_cluster():
    rng = np.random.default_rng(0)
    base = rng.normal(size=16)
    poison = [base + 0.01 * rng.normal(size=16) for _ in range(3)]          # near-identical embeddings
    clean = [rng.normal(size=16) for _ in range(2)]                         # unrelated
    emb = np.stack(poison + clean)
    texts = ["q? Mark Twain wrote it"] * 0 + [f"who wrote moby dick. Mark Twain {i}" for i in range(3)] + \
            ["pasta needs salted water", "the moon orbits the earth"]
    keep, branch = kmeans_filter(emb, texts)
    assert set(keep) == {3, 4}, branch


def test_kmeans_ngram_precheck_keeps_all():
    emb = np.eye(4)
    keep, branch = kmeans_filter(emb, ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"])
    assert keep == [0, 1, 2, 3] and branch == "ngram_precheck_keep_all"


def test_defenses_never_read_is_poison():
    for f in (ROOT / "src" / "prag" / "defenses").glob("*.py"):
        code = re.sub(r'""".*?"""', "", f.read_text(), flags=re.S)  # docstrings may mention the flag
        assert "is_poison" not in code, f"{f.name} reads is_poison"


def test_duplicate_filter():
    d = build_defense("duplicate_filter")
    ps = [Passage("a", 3.0, rank=1), Passage("b", 2.0, rank=2), Passage("a", 1.0, rank=3)]
    kept, info = d.filter("q", ps)
    assert [p.text for p in kept] == ["a", "b"] and info["duplicates_removed_idx"] == [2]


def test_registry_lists_all():
    assert set(available()) == {"none", "duplicate_filter", "knowledge_expansion", "paraphrase", "perplexity_filter", "trustrag"}


# ---- end-to-end in the toy world ---------------------------------------------------------------------------------
@pytest.fixture
def world(tmp_path):
    w = make_world(tmp_path)
    w["index"] = build_dense_index(w, tmp_path / "index")
    w["tmp"] = tmp_path
    w["reg"] = load_registry(w["registry"])
    return w


def _run(w, cfg_d, name):
    cfg = RunConfig.model_validate(cfg_d)
    ctx = DefenseContext(registry=w["reg"], dataset_dir=Path(cfg.dataset.dir), seed=cfg.seed, top_k=cfg.rag.top_k)
    comps = Components(make_encoder=lambda: HashEncoder(256), clean=FaissRetriever(w["index"]), make_reranker=None,
                       make_llm=lambda: EchoLLM(), defense=build_defense(cfg.defense.name, cfg.defense.params, ctx),
                       llm_spec={"engine": "echo"})
    d = w["tmp"] / "experiments" / name
    d.mkdir(parents=True, exist_ok=True)
    s = execute(cfg, comps, d, name, w["tmp"] / "experiments" / "index.json")
    traces = [json.loads(l) for l in open(d / "traces.jsonl")]
    for t in traces:
        assert trace_errors(t) == []
    return s, traces


def test_none_vs_duplicate_filter_identical(world):
    base = config_dict(world, index_dir=world["index"])
    s0, t0 = _run(world, base, "none")
    s1, t1 = _run(world, {**base, "defense": "duplicate_filter"}, "dup")
    assert [t["output_poisoned"] for t in t0] == [t["output_poisoned"] for t in t1]
    assert s1["defense"] == "duplicate_filter" and s1["defense_stats"]["mean_removed"] == 0.0


def test_perplexity_filter_toy(world):
    cfg = {**config_dict(world, index_dir=world["index"]),
           "defense": {"name": "perplexity_filter", "params": {"lm": "echo", "calib_n": 10, "fpr": 0.2}}}
    s, traces = _run(world, cfg, "ppl")
    t = traces[0]
    assert len(t["defense"]["info"]["ppl"]) == len(t["retrieved_topk"])
    assert s["defense_params"]["threshold"] is not None
    assert all(len(t["defense"]["kept_ranks"]) == len(t["retrieved_topk"]) - t["defense"]["removed"] for t in traces)


def test_trustrag_end_to_end(world):
    cfg = {**config_dict(world, index_dir=world["index"]), "defense": {"name": "trustrag", "params": {"embedder": "hash"}}}
    s, traces = _run(world, cfg, "trust")
    t = traces[0]
    g = t["defense"]["generation"]
    assert g["poisoned"]["internal_knowledge"] == g["clean"]["internal_knowledge"]  # stage 1 is question-only, shared
    assert t["prompt_poisoned"].lstrip().startswith("You will be given a question")
    assert t["defense"]["info"]["branch"]
    # poison in the toy world shares the question text -> high ROUGE-L -> TrustRAG removes at least some of it
    assert s["defense_stats"]["mean_poison_kept"] <= s["defense_stats"]["mean_poison_in_topk"]


def test_knowledge_expansion_requires_matching_k(world):
    base = config_dict(world, index_dir=world["index"])
    with pytest.raises(ValueError):
        RunConfig.model_validate({**base, "defense": {"name": "knowledge_expansion", "params": {"k": 9}}})
    s, traces = _run(world, {**base, "rag": {"top_k": 4}, "defense": {"name": "knowledge_expansion", "params": {"k": 4}}}, "ke")
    assert len(traces[0]["retrieved_topk"]) == 4


def test_paraphrase_rewrites_retrieval_and_prompt(world):
    targets = json.loads(Path(world["targets"]).read_text())
    para = {"meta": {"model": "toy"}, "items": {q: {"question": v["question"], "paraphrases": [v["question"].upper() + " please"]}
                                                for q, v in targets.items()}}
    f = world["tmp"] / "para.json"
    f.write_text(json.dumps(para))
    base = config_dict(world, index_dir=world["index"])
    s, traces = _run(world, {**base, "defense": {"name": "paraphrase", "params": {"file": str(f)}}}, "para")
    t = traces[0]
    assert t["defense"]["query_used"].endswith(" please")
    assert "Query: " + t["defense"]["query_used"] in t["prompt_poisoned"]
    pre = world["tmp"] / "pre.json"
    pre.write_text("{}")
    with pytest.raises(ValueError):  # precomputed results exist only for the original queries
        RunConfig.model_validate({**config_dict(world, precomputed=pre), "defense": {"name": "paraphrase", "params": {"file": str(f)}}})
