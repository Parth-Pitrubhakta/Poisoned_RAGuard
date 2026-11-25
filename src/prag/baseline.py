"""Step 1 — faithful PoisonedRAG baseline (port of upstream main.py, attack_method=LM_targeted, use_truth=False).

Differences from upstream, all deliberate and logged:
  * `targets.num_questions` = upstream M (questions per iteration), `targets.iterations` = upstream repeat_times
    (default 1). Iteration i uses targets[start + i*M : start + (i+1)*M]; all M*N poison texts of that iteration are
    injected and scored against each of its queries, exactly as upstream. Full protocol: M=10, iterations=10.
  * vLLM instead of HF generate (same prompt tokens and sampling params; see prag.llm.vllm_backend).
  * a 1-question non-empty-output smoke test before the batch (upstream issue #1).
  * an extra *no-attack* generation per question (clean top-k) so traces hold clean-vs-poisoned answers.
    It is not used in ASR / P / R / F1.

usage:  uv run python -m prag.baseline --config configs/baseline_nq_smoke.yaml [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("prag.baseline")


def _setup_logging(run_dir: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if run_dir is not None:
        handlers.append(logging.FileHandler(run_dir / "run.log"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s", handlers=handlers, force=True)


def pick_gpu(min_free_gb: float) -> tuple[str, float]:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    gpus = [(i.strip(), int(f) / 1024) for i, f in (l.split(",") for l in out.strip().splitlines())]
    idx, free = max(gpus, key=lambda t: t[1])
    if free < min_free_gb:
        raise RuntimeError(f"no GPU with >= {min_free_gb} GB free (best: GPU{idx} {free:.1f} GB) — shared box, not starting")
    return idx, free


def setup_seeds(seed: int) -> None:  # upstream src/utils.setup_seeds
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg_text = Path(args.config).read_text()
    cfg = yaml.safe_load(cfg_text)
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    mcfg = models["llms"][cfg["llm"]["key"]]
    M, top_k, N = cfg["targets"]["num_questions"], cfg["rag"]["top_k"], cfg["attack"]["adv_per_query"]
    n_iter = cfg["targets"].get("iterations", 1)
    n_q = M * n_iter

    if args.dry_run:
        _setup_logging(None)
        weights = Path(os.environ.get("HF_HOME", "")) / "hub" / ("models--" + mcfg["hf_id"].replace("/", "--"))
        have = any(weights.glob("blobs/*")) and sum(p.stat().st_size for p in weights.glob("blobs/*")) > 1e9
        log.info("DRY RUN %s", cfg["run_name"])
        log.info("  iterations=%d x M=%d", n_iter, M)
        log.info("  dataset=%s  questions=%d  top_k=%d  N=%d  retriever=%s(%s)  llm=%s@%s",
                 cfg["dataset"]["name"], n_q, top_k, N, cfg["retriever"]["key"], cfg["retriever"]["mode"], mcfg["hf_id"], mcfg.get("revision"))
        log.info("  LLM calls: %d (1 smoke + %d poisoned + %d clean)  | API cost: $0.00 (local)", 1 + 2 * n_q, n_q, n_q)
        log.info("  weights cached: %s  (download if not: ~%.1f GB into %s)", have, mcfg["size_gb"], weights)
        log.info("  est. time: ~1-2 min vLLM load + ~%d s generation; outputs ~%.1f MB", max(30, n_q), 0.02 * n_q)
        return 0

    run_id = f"{cfg['run_name']}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = ROOT / "experiments" / "baseline" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(cfg_text)
    _setup_logging(run_dir)
    gpu, free = pick_gpu(cfg["llm"].get("min_free_gb", 30))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    log.info("run_id=%s  GPU%s (%.1f GB free)", run_id, gpu, free)

    import torch
    from .attack.lm_targeted import lm_targeted_poison
    from .data.beir import corpus_text, load_corpus
    from .data.targets import load_targets
    from .eval.metrics import aggregate, attack_success, clean_str
    from .llm.base import EmptyOutputError
    from .llm.prompts import wrap_prompt
    from .llm.vllm_backend import VLLMTarget
    from .retrieval.contriever import ContrieverEncoder
    from .retrieval.merge import Passage, count_own_poison, load_precomputed, poisoned_topk

    setup_seeds(cfg["seed"])
    s = cfg["targets"]["start_index"]
    targets = load_targets(ROOT / cfg["targets"]["source"])[s:s + n_q]
    pre = load_precomputed(ROOT / cfg["retriever"]["precomputed"])
    corpus = load_corpus(ROOT / cfg["dataset"]["dir"], ids={d for t in targets for d, _ in pre[t.id][:top_k]})
    log.info("targets=%s", [t.id for t in targets])

    if len(targets) < n_q:
        raise ValueError(f"only {len(targets)} targets available from start_index {s}, need {n_q}")

    # --- retrieval + poisoning (upstream main.py), one poison pool per iteration ---
    enc = ContrieverEncoder(device="cuda")
    per_q = []
    for it in range(n_iter):
        it_targets = targets[it * M:(it + 1) * M]
        groups = [lm_targeted_poison(t, N) for t in it_targets]
        adv_list = [(t.id, p) for t, g in zip(it_targets, groups) for p in g]
        adv_embs = enc.encode([p for _, p in adv_list])
        for t, own in zip(it_targets, groups):
            q_emb = enc.encode([t.question])[0]
            clean = [Passage(text=corpus_text(corpus[d]), score=sc, doc_id=d, rank=i + 1) for i, (d, sc) in enumerate(pre[t.id][:top_k])]
            sims = enc.dot(q_emb, adv_embs).tolist()
            poison = [Passage(text=p, score=sc, is_poison=True, poison_for=qid) for (qid, p), sc in zip(adv_list, sims)]
            topk = poisoned_topk(clean, poison, top_k)
            per_q.append(dict(it=it, t=t, own=own, clean=clean, topk=topk, n_injected=len(adv_list),
                              prompt_poison=wrap_prompt(t.question, [p.text for p in topk], prompt_id=4),
                              prompt_clean=wrap_prompt(t.question, [p.text for p in clean], prompt_id=4)))
    del enc; torch.cuda.empty_cache()

    # --- target LLM ---
    L = cfg["llm"]
    llm = VLLMTarget(mcfg["hf_id"], mcfg.get("revision"), mcfg["chat_format"], dtype=mcfg.get("dtype", "auto"),
                     temperature=L["temperature"], top_p=L.get("top_p", 1.0), top_k=L.get("top_k", 50),
                     max_new_tokens=L["max_new_tokens"], repetition_penalty=L["repetition_penalty"], seed=L["seed"],
                     max_model_len=mcfg.get("max_model_len"), gpu_memory_utilization=L.get("gpu_memory_utilization", 0.30))
    if cfg["eval"].get("smoke_test_nonempty", True):
        g = llm.generate([per_q[0]["prompt_poison"]])[0]
        log.info("smoke test (%s): %d completion tokens, finish=%s, output=%r", per_q[0]["t"].id, g.completion_tokens, g.finish_reason, g.text[:200])
        if not g.text.strip():
            raise EmptyOutputError("empty LLM output on smoke test — model/tokenizer/device mismatch (upstream issue #1)")

    gens_p = llm.generate([q["prompt_poison"] for q in per_q])
    gens_c = llm.generate([q["prompt_clean"] for q in per_q])

    # --- eval + traces ---
    asr_cnts, ret_lists, traces = [0] * n_iter, [[] for _ in range(n_iter)], []
    upstream_fmt = [[] for _ in range(n_iter)]
    cfg_hash = hashlib.sha256(cfg_text.encode()).hexdigest()[:12]
    for q, gp, gc in zip(per_q, gens_p, gens_c):
        t = q["t"]
        n_own = count_own_poison(q["topk"], q["own"])
        ok = attack_success(t.incorrect_answer, gp.text)
        it = q["it"]
        asr_cnts[it] += ok; ret_lists[it].append(n_own)
        traces.append({
            "run_id": run_id, "config_hash": cfg_hash, "dataset": cfg["dataset"]["name"], "iteration": it, "question_id": t.id,
            "question": t.question, "correct_answer": t.correct_answer, "target_answer": t.incorrect_answer,
            "poison_texts": q["own"], "n_poison_injected_total": q["n_injected"],
            "clean_topk": [p.to_dict() for p in q["clean"]], "retrieved_topk": [p.to_dict() for p in q["topk"]],
            "prompt_poisoned": q["prompt_poison"], "prompt_clean": q["prompt_clean"],
            "llm_input_poisoned": llm.format_prompt(q["prompt_poison"]),
            "output_poisoned": gp.text, "output_clean": gc.text,
            "success": bool(ok), "clean_correct": attack_success(t.correct_answer, gc.text),
            "clean_says_target": attack_success(t.incorrect_answer, gc.text),
            "precision": n_own / top_k, "recall": n_own / N, "defense": cfg.get("defense", "none"),
            "usage": {"poisoned": gp.to_dict(), "clean": gc.to_dict()},
        })
        upstream_fmt[it].append({"id": t.id, "question": t.question, "injected_adv": [p.text for p in q["topk"] if p.text in set(q["own"])],
                             "input_prompt": q["prompt_poison"], "output_poison": gp.text,
                             "incorrect_answer": t.incorrect_answer, "answer": t.correct_answer})
        log.info("iter%d %s | own poison in top-%d: %d | success=%s | clean_correct=%s\n    poisoned: %r\n    clean:    %r",
                 it, t.id, top_k, n_own, ok, traces[-1]["clean_correct"], clean_str(gp.text)[:160], clean_str(gc.text)[:160])

    metrics = aggregate(asr_cnts, ret_lists, M=M, top_k=top_k, adv_per_query=N)
    summary = {"run_id": run_id, "kind": "baseline", "protocol": ("full upstream protocol" if (M, n_iter) == (10, 10) else "subset") + f" ({n_iter} iter x M={M} = {n_q} questions)",
               "config": cfg, "llm": llm.params, "retriever": {"key": cfg["retriever"]["key"], "mode": cfg["retriever"]["mode"]},
               "gpu": gpu, "metrics": metrics,
               "clean_accuracy": sum(t["clean_correct"] for t in traces) / len(traces),
               "total_cost_usd": 0.0, "created": dt.datetime.now().isoformat(timespec="seconds")}
    with open(run_dir / "traces.jsonl", "w") as f:
        for tr in traces:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (run_dir / "upstream_format_results.json").write_text(json.dumps([{f"iter_{i}": r} for i, r in enumerate(upstream_fmt)], ensure_ascii=False))
    idx_path = ROOT / "experiments" / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {"runs": []}
    index["runs"].append({"run_id": run_id, "path": str(run_dir.relative_to(ROOT)), "kind": "baseline",
                          "dataset": cfg["dataset"]["name"], "retriever": cfg["retriever"]["key"], "llm": cfg["llm"]["key"],
                          "defense": cfg.get("defense", "none"), "n_questions": n_q, "metrics": metrics["rounded"]})
    idx_path.write_text(json.dumps(index, indent=2))
    log.info("ASR=%.2f  P=%.2f  R=%.2f  F1=%.2f  clean_acc=%.2f  -> %s", metrics["asr_mean"], metrics["precision_mean"],
             metrics["recall_mean"], metrics["f1_mean"], summary["clean_accuracy"], run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
