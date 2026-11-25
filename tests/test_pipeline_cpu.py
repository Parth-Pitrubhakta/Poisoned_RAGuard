"""End-to-end pipeline on CPU with test doubles: traces, summary, metrics, resume, reranker, CLI."""
import json
import shutil

import numpy as np
import pytest
import yaml

from prag.config import RunConfig, load_registry
from prag.defenses.base import NoDefense
from prag.eval.metrics import aggregate
from prag.llm.backends import EchoLLM
from prag.llm.base import EmptyOutputError, Generation
from prag.pipeline import Components, execute
from prag.retrieval.encoders import HashEncoder
from prag.retrieval.index import FaissRetriever, PrecomputedRetriever
from prag.retrieval.reranker import OverlapReranker
from prag.tracing.schema import trace_errors
from toyworld import build_dense_index, config_dict, make_world


class CountingEcho(EchoLLM):
    def __init__(self):
        super().__init__("echo"); self.calls = 0

    def generate(self, msgs, max_new_tokens=None):
        self.calls += len(msgs)
        return super().generate(msgs)


@pytest.fixture
def world(tmp_path):
    w = make_world(tmp_path)
    w["index"] = build_dense_index(w, tmp_path / "index")
    w["tmp"] = tmp_path
    return w


def comps(cfg, w, llm=None, clean=None):
    return Components(make_encoder=lambda: HashEncoder(256), clean=clean or FaissRetriever(w["index"]),
                      make_reranker=(lambda: OverlapReranker()) if cfg.reranker else None,
                      make_llm=lambda: llm or EchoLLM(), defense=NoDefense(), llm_spec={"engine": "echo"})


def run(w, cfg_d, name="run", llm=None, clean=None):
    cfg = RunConfig.model_validate(cfg_d)
    d = w["tmp"] / "experiments" / name
    d.mkdir(parents=True, exist_ok=True)
    s = execute(cfg, comps(cfg, w, llm, clean), d, name, w["tmp"] / "experiments" / "index.json")
    traces = [json.loads(l) for l in open(d / "traces.jsonl")]
    return cfg, s, traces, d


def _strip_latency(t):
    t = dict(t); t.pop("latency_s"); u = json.loads(json.dumps(t["usage"]))
    for g in u.values():
        if g: g["latency_s"] = 0
    t["usage"] = u
    return t


def test_end_to_end_traces_and_metrics(world):
    cfg, s, traces, d = run(world, config_dict(world, index_dir=world["index"]))
    assert len(traces) == 4 and s["status"] == "complete" and s["n_questions"] == 4
    assert all(trace_errors(t) == [] for t in traces)
    for t in traces:
        assert [p["rank"] for p in t["retrieved_topk"]] == [1, 2, 3]
        scores = [p["score"] for p in t["retrieved_topk"]]
        assert scores == sorted(scores, reverse=True)
        assert t["n_poison_injected_total"] == 2 * 2           # M * N per iteration
        assert all(p["poison_for"] in {t["question_id"], None} or p["is_poison"] for p in t["retrieved_topk"])
        assert t["prompt_poisoned"].count("\n") >= 2 and all(p["text"] in t["prompt_poisoned"] for p in t["retrieved_topk"])
        assert all(p["text"] in t["prompt_clean"] for p in t["clean_topk"])
        assert not any(p["is_poison"] for p in t["clean_topk"])
        own = sum(p["is_poison"] and p["poison_for"] == t["question_id"] for p in t["retrieved_topk"])
        assert own == t["n_own_poison_in_topk"] and t["precision"] == own / 3 and t["recall"] == own / 2
        # echo LLM repeats the top passage, so success <=> top passage contains the target
        top = t["retrieved_topk"][0]["text"].lower()
        assert t["success"] == (t["target_answer"].lower() in top)
    by_it = [[t for t in traces if t["iteration"] == i] for i in (0, 1)]
    want = aggregate([sum(t["success"] for t in v) for v in by_it], [[t["n_own_poison_in_topk"] for t in v] for v in by_it],
                     M=2, top_k=3, adv_per_query=2)
    assert s["metrics"]["rounded"] == want["rounded"] and s["metrics"]["asr_mean"] > 0
    idx = json.loads((world["tmp"] / "experiments" / "index.json").read_text())
    assert [r["run_id"] for r in idx["runs"]] == ["run"]


def test_iteration_pools_are_separate(world):
    _, _, traces, _ = run(world, config_dict(world, index_dir=world["index"]))
    it_q = {t["question_id"]: t["iteration"] for t in traces}
    for t in traces:  # a poison from another iteration must never appear
        assert all(it_q[p["poison_for"]] == t["iteration"] for p in t["retrieved_topk"] if p["is_poison"])


def test_precomputed_mode_matches_dense(world):
    fr = FaissRetriever(world["index"])
    enc = HashEncoder(256)
    qs = {q: v["question"] for q, v in json.loads(world["targets"].read_text()).items()}
    top = fr.topk(list(qs), np.concatenate([enc.encode_queries([t]) for t in qs.values()]), 10)
    pre = world["tmp"] / "pre.json"
    pre.write_text(json.dumps({q: dict(v) for q, v in top.items()}))
    _, s1, t1, _ = run(world, config_dict(world, index_dir=world["index"]), "dense")
    _, s2, t2, _ = run(world, config_dict(world, precomputed=pre), "pre", clean=PrecomputedRetriever(pre))
    assert [x["retrieved_topk"] for x in t1] == [x["retrieved_topk"] for x in t2]
    assert s1["metrics"]["rounded"] == s2["metrics"]["rounded"]


def test_resume_regenerates_only_missing(world):
    _, s_full, full, d = run(world, config_dict(world, index_dir=world["index"]), "a")
    d2 = world["tmp"] / "experiments" / "b"
    shutil.copytree(d, d2)
    lines = (d2 / "traces.jsonl").read_text().splitlines(keepends=True)
    (d2 / "traces.jsonl").write_text("".join(lines[:2]) + lines[2][:40])  # iteration 0 done + torn line
    llm = CountingEcho()
    cfg = RunConfig.model_validate(config_dict(world, index_dir=world["index"]))
    s = execute(cfg, comps(cfg, world, llm), d2, "a", None)
    resumed = [json.loads(l) for l in open(d2 / "traces.jsonl")]
    assert llm.calls == 1 + 2 * 2                      # smoke + (poisoned + clean) for the 2 missing questions
    assert [_strip_latency(t) for t in resumed] == [_strip_latency(t) for t in full]
    assert s["metrics"] == s_full["metrics"]


def test_resume_complete_run_never_loads_llm(world):
    cfg_d = config_dict(world, index_dir=world["index"])
    _, _, _, d = run(world, cfg_d, "c")
    cfg = RunConfig.model_validate(cfg_d)
    c = comps(cfg, world)
    c.make_llm = lambda: (_ for _ in ()).throw(AssertionError("LLM loaded on a complete run"))
    assert execute(cfg, c, d, "c", None)["status"] == "complete"


def test_reranker_path(world):
    cfg_d = config_dict(world, index_dir=world["index"], reranker={"key": "overlap", "first_stage_k": 6})
    _, s, traces, _ = run(world, cfg_d, "rr")
    for t in traces:
        assert len(t["first_stage"]) == 6 and len(t["retrieved_topk"]) == 3
        assert [p["rank"] for p in t["retrieved_topk"]] == [1, 2, 3]
        rs = [p["rerank_score"] for p in t["retrieved_topk"]]
        assert rs == sorted(rs, reverse=True) and None not in rs
        assert {p["text"] for p in t["retrieved_topk"]} <= {p["text"] for p in t["first_stage"]}
        assert all(p["rerank_score"] is not None for p in t["clean_topk"])
    assert s["reranker"]["key"] == "overlap"


def test_attack_none_has_no_poison(world):
    _, s, traces, _ = run(world, config_dict(world, index_dir=world["index"], attack={"method": "none", "adv_per_query": 2}), "none")
    assert all(not any(p["is_poison"] for p in t["retrieved_topk"]) and t["poison_texts"] == [] for t in traces)
    assert s["metrics"]["precision_mean"] == 0 and s["metrics"]["asr_mean"] == 0


def test_empty_output_raises(world):
    class Mute(EchoLLM):
        def generate(self, msgs):
            return [Generation("", 1, 0, "stop", 0.0) for _ in msgs]
    with pytest.raises(EmptyOutputError):
        run(world, config_dict(world, index_dir=world["index"]), "mute", llm=Mute())


def test_cli_end_to_end_and_dry_run(world, capsys):
    from prag import run as cli

    cfg_path = world["tmp"] / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config_dict(world, index_dir=world["index"])))
    exp = world["tmp"] / "exp"
    assert cli.main(["--config", str(cfg_path), "--registry", str(world["registry"]), "--dry-run"]) == 0
    plan = capsys.readouterr().out
    assert '"llm_calls": 9' in plan and "within guardrails" in plan
    assert cli.main(["--config", str(cfg_path), "--registry", str(world["registry"]), "--experiments-dir", str(exp)]) == 0
    (run_dir,) = [p for p in exp.iterdir() if p.is_dir()]
    assert {"config.yaml", "traces.jsonl", "summary.json", "run.log"} <= {p.name for p in run_dir.iterdir()}
    log_lines = [json.loads(l) for l in (run_dir / "run.log").read_text().splitlines()]
    assert {"start", "retrieval_done", "smoke_test", "question", "done"} <= {l.get("event") for l in log_lines}
    # resume of a finished run is a no-op that keeps the same metrics
    before = json.loads((run_dir / "summary.json").read_text())["metrics"]
    assert cli.main(["--resume", str(run_dir), "--registry", str(world["registry"]), "--experiments-dir", str(exp)]) == 0
    assert json.loads((run_dir / "summary.json").read_text())["metrics"] == before
    assert len(json.loads((exp / "index.json").read_text())["runs"]) == 1


def test_real_configs_dry_run(capsys):
    from prag import run as cli

    for c in ("configs/baseline_nq_smoke.yaml", "configs/baseline_nq_full.yaml"):
        assert cli.main(["--config", c, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert '"llm_calls": 5' in out and '"llm_calls": 201' in out
