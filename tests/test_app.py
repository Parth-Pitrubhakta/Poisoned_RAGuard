"""Dashboard tests: every page renders against a toy experiments/ dir (no GPU, no models), plus the read-only contract."""
import json
import re
import sys
from pathlib import Path

import pytest

from prag.config import RunConfig
from prag.pipeline import execute
from test_pipeline_cpu import comps
from toyworld import build_dense_index, config_dict, make_world

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["Home.py"] + sorted(p.name for p in (APP / "pages").glob("*.py"))


@pytest.fixture(scope="module")
def exp_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("dash")
    w = make_world(root)
    w["index"], w["tmp"] = build_dense_index(w, root / "index"), root
    exp = root / "experiments"
    for name, extra in (("nq_contriever_vicuna7b_full-20990101-000000", {}),
                        ("nq_qwen3emb8b-rerank4b_vicuna7b_full-20990101-000001", {"reranker": {"key": "overlap", "first_stage_k": 6}})):
        cfg = RunConfig.model_validate(config_dict(w, index_dir=w["index"], **extra))
        d = exp / name
        d.mkdir(parents=True)
        execute(cfg, comps(cfg, w), d, name, exp / "index.json")
    # a TrustRAG-defended run (hash embedder) so Attack flow renders the defense details
    from prag.config import load_registry
    from prag.defenses.base import DefenseContext, build_defense
    from prag.llm.backends import EchoLLM
    from prag.pipeline import Components
    from prag.retrieval.encoders import HashEncoder
    from prag.retrieval.index import FaissRetriever

    name = "nq_contriever_vicuna7b_trustrag_full-20990101-000002"
    cfg = RunConfig.model_validate({**config_dict(w, index_dir=w["index"]), "defense": {"name": "trustrag", "params": {"embedder": "hash"}}})
    dfn = build_defense("trustrag", {"embedder": "hash"}, DefenseContext(registry=load_registry(w["registry"]), top_k=cfg.rag.top_k))
    d = exp / name
    d.mkdir(parents=True)
    execute(cfg, Components(make_encoder=lambda: HashEncoder(256), clean=FaissRetriever(w["index"]), make_reranker=None,
                            make_llm=lambda: EchoLLM(), defense=dfn, llm_spec={"engine": "echo"}), d, name, exp / "index.json")
    (exp / "defenses").mkdir()
    (exp / "defenses" / "defense_table.json").write_text(json.dumps(_toy_defense_table(name)))
    (exp / "datasets.json").write_text(json.dumps({"nq": {
        "name": "nq", "split": "test", "corpus_passages": 20, "queries": 4, "qrels_test_pairs": 4, "qrels_test_queries": 4,
        "passage_words": {"mean": 8.0, "median": 8}, "passages_with_title_share": 0.2, "target_questions": 4,
        "adv_texts_per_target": 3, "adv_text_words": {"mean": 9.0, "median": 9}, "targets_source": "toy"}}))
    (exp / "retrieval_quality.json").write_text(json.dumps({"nq": {"n_queries": 4, "retrievers": {
        "hash": {"recall@5": 1.0, "recall@20": 1.0, "recall@100": 1.0, "ndcg@10": 0.9}}}}))
    return exp


def _toy_defense_table(run_id: str) -> dict:
    order = ["duplicate_filter", "perplexity_filter", "paraphrase", "knowledge_expansion@10", "knowledge_expansion@50", "trustrag",
             "trustrag@contriever"]
    rows = []
    for i, dfn in enumerate(order):
        for llm, nm in (("vicuna-7b-v1.3", "Vicuna-7B"), ("qwen3.5-27b", "Qwen3.5-27B"), ("gpt-oss-120b", "gpt-oss-120b")):
            if dfn.startswith("knowledge") and llm == "vicuna-7b-v1.3":
                continue
            for setup in ("Contriever (paper)", "Qwen3-Emb-8B", "Qwen3-Emb-8B + rerank"):
                v = 0.8 - 0.1 * i
                rows.append({"defense": dfn, "retrieval": setup, "llm": llm, "llm_name": nm, "k": 50 if dfn.endswith("50") else 5,
                             "asr_none": 0.8, "asr_def": v, "d_asr": v - 0.8, "asr_norm_none": 0.85, "asr_norm_def": v,
                             "asr_answered_none": 0.9, "asr_answered_def": v, "robust_none": 0.85, "robust_def": v, "robust_metric": "exact",
                             "clean_none": 0.6, "clean_def": 0.62, "d_clean": 0.02, "poison_in_topk": 4.6, "poison_kept": 0.5 if "trust" in dfn else 4.6,
                             "clean_removed": 0.8, "empty_ctx": 0.7, "empty_final": 0.5, "empty_final_none": 0.2, "discordant_asr": 3,
                             "p_mcnemar_asr": 0.01, "run_id": f"{run_id}-{dfn}-{llm}-{setup}", "twin_run_id": "twin"})
    names = {d: d.replace("_", " ") for d in order}
    noise = {l: {"cells": 3, "max_abs_d_asr": 0.01, "max_abs_d_robust": 0.01, "max_abs_d_clean": 0.05, "max_discordant": 7}
             for l in ("vicuna-7b-v1.3", "qwen3.5-27b", "gpt-oss-120b")}
    return {"rows": rows, "noise_floor": noise, "order": order, "names": names}


def _app(exp_dir, monkeypatch, page):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PRAG_EXPERIMENTS_DIR", str(exp_dir))
    sys.modules.pop("lib", None)  # lib reads PRAG_EXPERIMENTS_DIR at import
    path = APP / page if page == "Home.py" else APP / "pages" / page
    return AppTest.from_file(str(path), default_timeout=60)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(exp_dir, monkeypatch, page):
    at = _app(exp_dir, monkeypatch, page).run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("Research artifact" in w.value and "synthetic" in w.value for w in at.warning), "banner missing"


def test_attack_flow_shows_poison_and_verdict(exp_dir, monkeypatch):
    at = _app(exp_dir, monkeypatch, "3_Attack_Flow.py").run()
    assert not at.exception
    verdicts = [e.value for e in at.error] + [s.value for s in at.success]
    assert any("ATTACK SUCCEEDED" in v or "ATTACK FAILED" in v for v in verdicts)
    assert any("POISON" in str(df.value.to_string()) if hasattr(df.value, "to_string") else False for df in at.dataframe) or \
        any("POISON" in m.value for m in at.markdown)


def test_defenses_page_all_views(exp_dir, monkeypatch):
    at = _app(exp_dir, monkeypatch, "5_Defenses.py").run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("truncation-confounded" in w.value for w in at.warning)
    assert any("no loss" in w.value for w in at.warning)
    assert any("RobustRAG" in i.value for i in at.info)
    for metric in at.radio(key="def_metric").options:
        at.radio(key="def_metric").set_value(metric).run()
        assert not at.exception, metric
    for s_ in at.selectbox(key="def_setup").options:
        at.selectbox(key="def_setup").set_value(s_).run()
        assert not at.exception, s_
    at.toggle(key="def_sub").set_value(True).run()
    assert not at.exception


def test_attack_flow_shows_defense(exp_dir, monkeypatch):
    at = _app(exp_dir, monkeypatch, "3_Attack_Flow.py")
    at.session_state["run_id"] = "nq_contriever_vicuna7b_trustrag_full-20990101-000002"
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("Defense `trustrag`" in i.value for i in at.info)


def test_results_all_runs_scope(exp_dir, monkeypatch):
    at = _app(exp_dir, monkeypatch, "4_Results.py").run()
    assert not at.exception
    at.radio[1].set_value("all complete runs").run()
    assert not at.exception


def test_dashboard_is_read_only():
    forbidden = re.compile(r"^\s*(import|from)\s+(vllm|torch|faiss|sentence_transformers|transformers|subprocess|"
                           r"prag\.(pipeline|run|retrieval|llm|attack|build_index|baseline))\b", re.M)
    for p in [APP / "lib.py", *APP.glob("*.py"), *(APP / "pages").glob("*.py")]:
        assert not forbidden.search(p.read_text()), f"{p.name} imports model/pipeline code"
        assert ".write_text(" not in p.read_text() and "open(" not in p.read_text().replace("open(path, encoding", ""), p.name
