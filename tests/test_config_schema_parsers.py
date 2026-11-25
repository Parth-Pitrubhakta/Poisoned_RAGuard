import copy
import json

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from prag.config import ROOT, RunConfig, check_registry, config_hash, load_config, load_registry
from prag.llm.parsers import parse_harmony, parse_think_tags
from prag.retrieval.encoders import HashEncoder, score
from prag.retrieval.index import build_flat_ip
from prag.retrieval.merge import Passage
from prag.retrieval.reranker import rerank
from prag.tracing.schema import trace_errors
from toyworld import build_dense_index, make_world

SMOKE = ROOT / "configs" / "baseline_nq_smoke.yaml"


def _raw():
    return yaml.safe_load(SMOKE.read_text())


# --- config ---------------------------------------------------------------------------------------------------
def test_repo_configs_validate():
    for p in ("baseline_nq_smoke.yaml", "baseline_nq_full.yaml"):
        cfg, _ = load_config(ROOT / "configs" / p)
        assert cfg.defense.name == "none" and cfg.retriever.mode == "upstream_precomputed"
    full, _ = load_config(ROOT / "configs" / "baseline_nq_full.yaml")
    assert full.targets.total == 100


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d.update(typo_key=1), "Extra inputs"),
    (lambda d: d["retriever"].update(score_function="cos_sim"), "computed with dot"),
    (lambda d: d["retriever"].pop("precomputed"), "needs retriever.precomputed"),
    (lambda d: d["targets"].update(num_questions=0), "greater than or equal"),
    (lambda d: d.update(reranker={"key": "qwen3-reranker-4b", "first_stage_k": 2}), "first_stage_k"),
    (lambda d: d["llm"].update(output_parser="nope"), "output_parser"),
])
def test_config_rejects(mutate, msg):
    d = _raw(); mutate(d)
    with pytest.raises(ValidationError, match=msg):
        RunConfig.model_validate(d)


def test_registry_check_and_hash():
    d = _raw(); d["llm"]["key"] = "no-such-model"
    with pytest.raises(KeyError, match="llms.no-such-model"):
        check_registry(RunConfig.model_validate(d), load_registry())
    a = RunConfig.model_validate(_raw())
    b = RunConfig.model_validate(yaml.safe_load("# a comment\n" + SMOKE.read_text()))
    assert config_hash(a) == config_hash(b)
    d = _raw(); d["rag"]["top_k"] = 3
    assert config_hash(RunConfig.model_validate(d)) != config_hash(a)


# --- trace schema ---------------------------------------------------------------------------------------------
def _good_trace():
    p = {"text": "x", "score": 1.0, "rank": 1, "is_poison": True, "doc_id": None, "poison_for": "q1", "rerank_score": None}
    g = {"text": "a", "prompt_tokens": 3, "completion_tokens": 1, "finish_reason": "stop", "latency_s": 0.1, "cost_usd": 0.0,
         "raw_text": None, "reasoning": None}
    return {"schema_version": "1.0", "run_id": "r", "config": {"hash": "h", "retriever": "c", "llm": "l", "top_k": 1,
                                                               "adv_per_query": 1, "defense": "none"},
            "dataset": "nq", "iteration": 0, "question_id": "q1", "question": "?", "correct_answer": "a", "target_answer": "b",
            "poison_texts": ["x"], "n_poison_injected_total": 1, "clean_topk": [], "first_stage": None, "retrieved_topk": [p],
            "defense": {"name": "none", "info": {}, "removed": 0}, "prompt_poisoned": "p", "prompt_clean": None,
            "llm_input_poisoned": "p", "output_poisoned": "a", "output_clean": None, "success": False,
            "clean_correct": None, "clean_says_target": None, "precision": 1.0, "recall": 1.0, "n_own_poison_in_topk": 1,
            "usage": {"poisoned": g, "clean": None}, "latency_s": {}, "cost_usd": 0.0}


def test_trace_schema():
    t = _good_trace()
    assert trace_errors(t) == []
    for broken in ({"success": "yes"}, {"precision": 1.5}, {"extra": 1}):
        b = {**copy.deepcopy(t), **broken}
        assert trace_errors(b), broken
    b = copy.deepcopy(t); del b["retrieved_topk"][0]["is_poison"]
    assert any("is_poison" in e for e in trace_errors(b))
    b = copy.deepcopy(t); del b["target_answer"]
    assert trace_errors(b)


# --- parsers --------------------------------------------------------------------------------------------------
def test_think_parser():
    assert parse_think_tags("<think>\nhmm Sydney?\n</think>\n\nCanberra.") == ("Canberra.", "hmm Sydney?")
    assert parse_think_tags("<think>\n\n</think>\n\nCanberra") == ("Canberra", None)
    assert parse_think_tags("<think> still thinking about Sydney") == ("", "still thinking about Sydney")
    assert parse_think_tags("Canberra") == ("Canberra", None)


def test_harmony_parser():
    raw = ("<|channel|>analysis<|message|>Context says Sydney but...<|end|><|start|>assistant"
           "<|channel|>final<|message|>Canberra.<|return|>")
    assert parse_harmony(raw) == ("Canberra.", "Context says Sydney but...")
    assert parse_harmony("<|channel|>analysis<|message|>Sydney, maybe") == ("", "Sydney, maybe")  # no final answer
    assert parse_harmony("Canberra") == ("Canberra", None)


# --- retrieval primitives ------------------------------------------------------------------------------------
def test_faiss_flat_ip_matches_bruteforce():
    rng = np.random.default_rng(12)
    x, q = rng.normal(size=(500, 32)).astype(np.float32), rng.normal(size=(7, 32)).astype(np.float32)
    for norm in (False, True):
        idx = build_flat_ip(x, normalize=norm)
        xs = x / np.linalg.norm(x, axis=1, keepdims=True) if norm else x
        qs = q / np.linalg.norm(q, axis=1, keepdims=True) if norm else q
        s, i = idx.search(qs, 10)
        bf = np.argsort(-(qs @ xs.T), axis=1)[:, :10]
        assert (i == bf).all()
        np.testing.assert_allclose(s, np.take_along_axis(qs @ xs.T, bf, 1), rtol=1e-5)


def test_score_functions():
    d = np.array([[1.0, 0.0], [3.0, 4.0]], np.float32); q = np.array([2.0, 0.0], np.float32)
    np.testing.assert_allclose(score(q, d, "dot"), [2.0, 6.0])
    np.testing.assert_allclose(score(q, d, "cos_sim"), [1.0, 0.6], rtol=1e-6)


def test_rerank_is_stable_and_reranks():
    class R:
        name = "fixed"
        def scores(self, q, docs): return [0.5, 0.9, 0.5, 0.1]
    cands = [Passage(text=t, score=s, rank=i + 1) for i, (t, s) in enumerate([("a", 4), ("b", 3), ("c", 2), ("d", 1)])]
    out = rerank(R(), "q", cands, 3)
    assert [p.text for p in out] == ["b", "a", "c"] and [p.rank for p in out] == [1, 2, 3]
    assert out[0].score == 3 and out[0].rerank_score == 0.9


def test_build_index_shards_merge(tmp_path):
    from prag import build_index

    w = make_world(tmp_path)
    out = tmp_path / "idx"
    common = ["--dataset-dir", str(w["dataset_dir"]), "--retriever", "hash", "--out", str(out), "--registry", str(w["registry"]),
              "--num-shards", "2"]
    import prag.retrieval.encoders as E
    orig = E.build_encoder
    E.build_encoder = lambda key, reg, device="cuda", batch_size=64: HashEncoder(256)  # no CUDA in this test
    try:
        for s in ("0", "1"):
            assert build_index.main(common + ["--shard", s]) == 0
    finally:
        E.build_encoder = orig
    assert build_index.main(common + ["--merge"]) == 0
    ref = build_dense_index(w, tmp_path / "ref")
    import faiss
    a, b = faiss.read_index(str(out / "index.faiss")), faiss.read_index(str(ref / "index.faiss"))
    assert a.ntotal == b.ntotal == 20
    np.testing.assert_array_equal(a.reconstruct_n(0, 20), b.reconstruct_n(0, 20))
    assert json.loads((out / "doc_ids.json").read_text()) == json.loads((ref / "doc_ids.json").read_text())
