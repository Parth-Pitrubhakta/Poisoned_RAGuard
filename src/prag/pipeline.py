"""Config-driven PoisonedRAG pipeline (Step 2). Generalizes prag.baseline with the same upstream semantics:

  per iteration i (upstream repeat_times), targets[start + i*M : start + (i+1)*M]:
    poison pool  = for each target: question + "." + adv_text  (N per target; M*N per iteration)
    clean top-n  = upstream precomputed results | FAISS IndexFlatIP over the corpus   (n = top_k, or first_stage_k)
    merged       = sort(clean top-n + every poison of the iteration scored by the retriever, desc)[:n]  (stable)
    final top-k  = merged[:k]  |  rerank(merged)[:k]  when a reranker is configured
    defense      -> optional query rewrite before retrieval; filter on the final top-k (poisoned and clean paths);
                    optional multi-stage answer replacing the single MULTIPLE_PROMPT (prompt_id=4) call
    P / R        = own-poison-in-top-k / k, / N   (pre-defense, as upstream)
    success      = clean_str(target) in clean_str(answer)

Components are built lazily and released in order (encoder -> reranker -> LLM) so a single GPU suffices.
traces.jsonl is the checkpoint: finished (iteration, question) pairs are never regenerated.
"""
from __future__ import annotations

import datetime as dt
import gc
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .attack.lm_targeted import lm_targeted_poison
from .config import RunConfig, config_hash
from .data.beir import corpus_text, load_corpus
from .data.targets import TargetQuestion, load_targets
from .defenses.base import Defense
from .eval.metrics import aggregate, attack_success, attack_success_normalized
from .llm.base import EmptyOutputError, Generation, TargetLLM, cost_usd
from .llm.prompts import wrap_prompt
from .retrieval.encoders import Encoder, score
from .retrieval.index import CleanRetriever
from .retrieval.merge import Passage, count_own_poison, poisoned_topk
from .retrieval.reranker import Reranker, rerank
from .tracing.schema import SCHEMA_VERSION, validate_summary
from .tracing.writer import TraceWriter, upsert_index

log = logging.getLogger("prag.pipeline")


@dataclass
class Components:
    make_encoder: Callable[[], Encoder] | None      # None only if attack=none and the clean retriever needs no queries
    clean: CleanRetriever
    make_reranker: Callable[[], Reranker] | None
    make_llm: Callable[[], TargetLLM]
    defense: Defense
    llm_spec: dict                                  # registry entry, for pricing
    device: str = "cpu"                             # where a defense's own model (PPL LM, embedder) runs


def setup_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _release() -> None:
    """gc + empty the CUDA cache. Callers must drop their own references first (x = None)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def iteration_targets(cfg: RunConfig) -> list[list[TargetQuestion]]:
    t = cfg.targets
    allt = load_targets(cfg.path(t.source))[t.start_index:t.start_index + t.total]
    if len(allt) < t.total:
        raise ValueError(f"only {len(allt)} targets from start_index {t.start_index}, need {t.total}")
    M = t.num_questions
    return [allt[i * M:(i + 1) * M] for i in range(t.iterations)]


def retrieve(cfg: RunConfig, comps: Components, iters: list[list[TargetQuestion]]) -> list[dict]:
    """Everything up to (not including) the LLM, for every question. Cheap relative to generation, so it is
    recomputed on resume, which also makes resumed traces identical to uninterrupted ones."""
    k, N = cfg.rag.top_k, cfg.attack.adv_per_query
    n = cfg.reranker.first_stage_k if cfg.reranker else k
    flat = [t for it in iters for t in it]
    attack = cfg.attack.method != "none"
    t0 = time.perf_counter()
    enc = comps.make_encoder() if comps.make_encoder and (attack or comps.clean.needs_query_embedding) else None
    # the query actually used (a defense may paraphrase it); upstream encodes each query on its own
    queries = [comps.defense.rewrite_query(t.id, t.question) for t in flat]
    q_embs = np.concatenate([enc.encode_queries([q]) for q in queries]) if enc is not None else None
    clean_ids = comps.clean.topk([t.id for t in flat], q_embs, n)
    corpus = load_corpus(cfg.path(cfg.dataset.dir), ids={d for v in clean_ids.values() for d, _ in v})

    per_q, qi = [], 0
    for it, targets in enumerate(iters):
        groups = [lm_targeted_poison(t, N) if attack else [] for t in targets]
        pool = [(t.id, p) for t, g in zip(targets, groups) for p in g]
        pool_embs = enc.encode_docs([p for _, p in pool]) if pool else None
        for t, own in zip(targets, groups):
            clean = [Passage(text=corpus_text(corpus[d]), score=float(s), doc_id=d, rank=i + 1)
                     for i, (d, s) in enumerate(clean_ids[t.id])]
            poison = []
            if pool:
                sims = score(q_embs[qi], pool_embs, cfg.retriever.score_function)
                poison = [Passage(text=p, score=float(s), is_poison=True, poison_for=pid) for (pid, p), s in zip(pool, sims)]
            first = poisoned_topk(clean, poison, n)
            per_q.append(dict(it=it, t=t, query=queries[qi], own=own, clean_n=clean, first=first, n_injected=len(pool)))
            qi += 1
    t_ret = (time.perf_counter() - t0) / max(1, len(per_q))
    enc = None  # drop the last reference *here*; _release(x) cannot free a name bound in the caller
    _release()

    t1 = time.perf_counter()
    rr = comps.make_reranker() if (cfg.reranker and comps.make_reranker) else None
    for q in per_q:
        if rr is not None:
            q["topk"] = rerank(rr, q["query"], q["first"], k)
            q["clean_topk"] = rerank(rr, q["query"], q["clean_n"], k)
        else:
            q["topk"], q["clean_topk"] = q["first"][:k], q["clean_n"][:k]
        q["lat_retrieval"] = t_ret
    t_rr = (time.perf_counter() - t1) / max(1, len(per_q)) if rr is not None else 0.0
    rr = None
    _release()

    t2 = time.perf_counter()
    d = comps.defense
    d.prepare(comps.device)
    res = d.filter_batch([(q["query"], q["topk"]) for q in per_q] + [(q["query"], q["clean_topk"]) for q in per_q])
    d.release()
    _release()
    t_def = (time.perf_counter() - t2) / max(1, len(per_q))
    for q, (ctx, info), (cctx, cinfo) in zip(per_q, res[:len(per_q)], res[len(per_q):]):
        q.update(lat_rerank=t_rr, lat_defense=t_def, ctx=ctx, cctx=cctx, defense_info=info, clean_defense_info=cinfo,
                 prompt_poison=wrap_prompt(q["query"], [p.text for p in ctx], prompt_id=cfg.rag.prompt_id),
                 prompt_clean=wrap_prompt(q["query"], [p.text for p in cctx], prompt_id=cfg.rag.prompt_id))
    return per_q


def generate_batch(cfg: RunConfig, comps: Components, llm: TargetLLM, batch: list[dict]):
    """-> (poisoned gens, clean gens or [None]*n, poisoned stage infos, clean stage infos).
    Poisoned and clean prompts go in one call; per-request seeds make outputs independent of batching."""
    n, clean = len(batch), cfg.eval.clean_run
    if comps.defense.has_answer:
        qs = [q["query"] for q in batch] * (2 if clean else 1)
        ctxs = [[p.text for p in q["ctx"]] for q in batch] + ([[p.text for p in q["cctx"]] for q in batch] if clean else [])
        gens, infos = comps.defense.answer(llm, qs, ctxs)
    else:
        gens = llm.generate([q["prompt_poison"] for q in batch] + ([q["prompt_clean"] for q in batch] if clean else []))
        infos = [None] * len(gens)
    if clean:
        return gens[:n], gens[n:], infos[:n], infos[n:]
    return gens, [None] * n, infos, [None] * n


def _price(gens: list[Generation], spec: dict) -> None:
    for g in gens:
        if not g.cost_usd:
            g.cost_usd = cost_usd(spec, g.prompt_tokens, g.completion_tokens)


def make_trace(cfg: RunConfig, run_id: str, chash: str, q: dict, llm: TargetLLM, gp: Generation, gc_: Generation | None,
               ip: dict | None = None, ic: dict | None = None) -> dict:
    t, k, N = q["t"], cfg.rag.top_k, cfg.attack.adv_per_query
    n_own = count_own_poison(q["topk"], q["own"])
    prompt_p = ip["final_prompt"] if ip else q["prompt_poison"]
    prompt_c = (ic["final_prompt"] if ic else q["prompt_clean"]) if gc_ else None
    strip = lambda i: {a: b for a, b in i.items() if a != "final_prompt"} if i else None  # noqa: E731
    defense = {"name": cfg.defense.name, "info": q["defense_info"], "removed": len(q["topk"]) - len(q["ctx"]),
               "kept_ranks": [p.rank for p in q["ctx"]], "poison_kept": sum(p.is_poison for p in q["ctx"]),
               "own_poison_kept": count_own_poison(q["ctx"], q["own"]),
               "query_used": q["query"] if q["query"] != t.question else None,
               "clean_info": q["clean_defense_info"], "clean_removed": len(q["clean_topk"]) - len(q["cctx"]),
               "clean_kept_ranks": [p.rank for p in q["cctx"]],
               "generation": {"poisoned": strip(ip), "clean": strip(ic)} if ip else None}
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "config": {"hash": chash, "retriever": f"{cfg.retriever.key}:{cfg.retriever.mode}:{cfg.retriever.score_function}",
                   "reranker": cfg.reranker.key if cfg.reranker else None, "llm": cfg.llm.key, "attack": cfg.attack.method,
                   "top_k": k, "adv_per_query": N, "defense": cfg.defense.name},
        "dataset": cfg.dataset.name, "iteration": q["it"], "question_id": t.id, "question": t.question,
        "correct_answer": t.correct_answer, "target_answer": t.incorrect_answer,
        "poison_texts": list(q["own"]), "n_poison_injected_total": q["n_injected"],
        "clean_topk": [p.to_dict() for p in q["clean_topk"]],
        "first_stage": [p.to_dict() for p in q["first"]] if cfg.reranker else None,
        "retrieved_topk": [p.to_dict() for p in q["topk"]],
        "defense": defense,
        "prompt_poisoned": prompt_p, "prompt_clean": prompt_c,
        "llm_input_poisoned": llm.format_prompt(prompt_p),
        "output_poisoned": gp.text, "output_clean": gc_.text if gc_ else None,
        "success": bool(attack_success(t.incorrect_answer, gp.text)),
        "clean_correct": bool(attack_success(t.correct_answer, gc_.text)) if gc_ else None,
        "clean_says_target": bool(attack_success(t.incorrect_answer, gc_.text)) if gc_ else None,
        "precision": n_own / k, "recall": n_own / N, "n_own_poison_in_topk": n_own,
        "usage": {"poisoned": gp.to_dict(), "clean": gc_.to_dict() if gc_ else None},
        "latency_s": {"retrieval": q["lat_retrieval"], "rerank": q["lat_rerank"], "defense": q["lat_defense"], "llm_poisoned": gp.latency_s,
                      "llm_clean": gc_.latency_s if gc_ else None},
        "cost_usd": gp.cost_usd + (gc_.cost_usd if gc_ else 0.0),
    }


def summarize(cfg: RunConfig, run_id: str, chash: str, traces: list[dict], llm_params: dict, timing: dict,
              defense_params: dict | None = None) -> dict:
    M, k, N = cfg.targets.num_questions, cfg.rag.top_k, cfg.attack.adv_per_query
    by_it: dict[int, list[dict]] = {}
    for tr in sorted(traces, key=lambda r: (r["iteration"], r["question_id"])):
        by_it.setdefault(tr["iteration"], []).append(tr)
    complete = len(by_it) == cfg.targets.iterations and all(len(v) == M for v in by_it.values())
    if complete:
        metrics = aggregate([sum(r["success"] for r in v) for v in by_it.values()],
                            [[r["n_own_poison_in_topk"] for r in v] for v in by_it.values()], M=M, top_k=k, adv_per_query=N)
    else:  # partial: flat means over finished questions (not upstream-comparable)
        p = np.array([r["precision"] for r in traces] or [0.0]); rc = np.array([r["recall"] for r in traces] or [0.0])
        f1 = np.divide(2 * p * rc, p + rc, out=np.zeros_like(p), where=(p + rc) > 0)
        raw = {"asr_mean": float(np.mean([r["success"] for r in traces] or [0.0])), "precision_mean": float(p.mean()),
               "recall_mean": float(rc.mean()), "f1_mean": float(f1.mean())}
        metrics = {**raw, "rounded": {a: round(b, 2) for a, b in raw.items()},
                   "asr_per_iter": [sum(r["success"] for r in v) / len(v) for v in by_it.values()],
                   "ret_counts": [[r["n_own_poison_in_topk"] for r in v] for v in by_it.values()]}
    cc = [r["clean_correct"] for r in traces if r["clean_correct"] is not None]
    # secondary, NOT upstream: typography-normalized scoring (see eval.metrics.normalize_typography)
    norm = {"asr": float(np.mean([attack_success_normalized(r["target_answer"], r["output_poisoned"]) for r in traces] or [0.0])),
            "clean_accuracy": (float(np.mean([attack_success_normalized(r["correct_answer"], r["output_clean"]) for r in traces
                                              if r["output_clean"] is not None])) if cc else None),
            "empty_final_answer_rate": float(np.mean([not r["output_poisoned"].strip() for r in traces] or [0.0])),
            "truncated_rate": float(np.mean([r["usage"]["poisoned"]["finish_reason"] == "length" for r in traces] or [0.0])),
            # TrustRAG's definition (main_trustrag.py): target in answer AND correct answer not in answer (exact clean_str)
            "asr_target_and_not_correct": float(np.mean([r["success"] and not attack_success(r["correct_answer"], r["output_poisoned"])
                                                         for r in traces] or [0.0]))}
    dfs = [r["defense"] for r in traces if isinstance(r.get("defense"), dict)]
    dstats = {"mean_removed": float(np.mean([d.get("removed", 0) for d in dfs] or [0.0])),
              "mean_poison_kept": float(np.mean([d.get("poison_kept", 0) for d in dfs] or [0.0])),
              "mean_poison_in_topk": float(np.mean([sum(p["is_poison"] for p in r["retrieved_topk"]) for r in traces] or [0.0])),
              "mean_clean_removed": float(np.mean([d.get("clean_removed", 0) for d in dfs] or [0.0])),
              "empty_context_rate": float(np.mean([len(d.get("kept_ranks", [1])) == 0 for d in dfs] or [0.0]))}
    full = (M, cfg.targets.iterations) == (10, 10)
    usage = [r["usage"][s] for r in traces for s in ("poisoned", "clean") if r["usage"].get(s)]
    s = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "kind": "pipeline",
        "protocol": ("full upstream protocol" if full else "subset") + f" ({cfg.targets.iterations} iter x M={M} = {cfg.targets.total} questions)",
        "status": "complete" if complete else "partial",
        "config": cfg.model_dump(mode="json"), "config_hash": chash, "dataset": cfg.dataset.name,
        "retriever": {"key": cfg.retriever.key, "mode": cfg.retriever.mode, "score_function": cfg.retriever.score_function},
        "reranker": cfg.reranker.model_dump() if cfg.reranker else None, "llm": llm_params, "defense": cfg.defense.name,
        "defense_params": defense_params or {}, "defense_stats": dstats,
        "n_questions": len(traces), "metrics": metrics,
        "clean_accuracy": (sum(cc) / len(cc)) if cc else None,
        "clean_says_target_rate": (sum(bool(r["clean_says_target"]) for r in traces) / len(cc)) if cc else None,
        "secondary_metrics": norm,
        "total_cost_usd": float(sum(r["cost_usd"] for r in traces)),
        "tokens": {"prompt": sum(u["prompt_tokens"] for u in usage), "completion": sum(u["completion_tokens"] for u in usage)},
        "timing_s": timing, "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    validate_summary(s)
    return s


def execute(cfg: RunConfig, comps: Components, run_dir: Path, run_id: str, index_path: Path | None) -> dict:
    t_start = time.perf_counter()
    setup_seeds(cfg.seed)
    chash = config_hash(cfg)
    writer = TraceWriter(run_dir / "traces.jsonl")
    iters = iteration_targets(cfg)
    log.info("run %s: %d iterations x M=%d, %d already done", run_id, len(iters), cfg.targets.num_questions, len(writer.done),
             extra={"data": {"event": "start", "run_id": run_id, "config_hash": chash, "resume_done": len(writer.done)}})

    per_q = retrieve(cfg, comps, iters)
    t_ret = time.perf_counter() - t_start
    pending = [q for q in per_q if (q["it"], q["t"].id) not in writer.done]
    log.info("retrieval done in %.1fs; %d questions pending generation", t_ret, len(pending),
             extra={"data": {"event": "retrieval_done", "seconds": t_ret, "pending": len(pending)}})

    llm_params: dict = {"key": cfg.llm.key}
    t_gen = 0.0
    if pending:
        t1 = time.perf_counter()
        llm = comps.make_llm()
        llm_params = {"key": cfg.llm.key, **llm.params}
        if cfg.eval.smoke_test_nonempty:
            g = llm.generate([pending[0]["prompt_poison"]])[0]
            log.info("smoke test %s: %d tokens, finish=%s, %r", pending[0]["t"].id, g.completion_tokens, g.finish_reason, g.text[:120],
                     extra={"data": {"event": "smoke_test", "completion_tokens": g.completion_tokens, "finish": g.finish_reason}})
            if not g.text.strip():
                if g.completion_tokens > 0 and (g.reasoning or g.raw_text):
                    # a reasoning model that used its whole budget thinking (finish=length): a real, scored outcome
                    # (empty final answer), not a broken template. T and max_new_tokens are held fixed by design.
                    log.warning("smoke test %s: no final answer after %d tokens (finish=%s); reasoning truncation, continuing",
                                pending[0]["t"].id, g.completion_tokens, g.finish_reason,
                                extra={"data": {"event": "smoke_test_truncated", "completion_tokens": g.completion_tokens}})
                else:
                    raise EmptyOutputError("empty LLM output on smoke test: model/tokenizer/template mismatch (upstream issue #1)")
        B = cfg.llm.batch_questions  # checkpoint unit. Per-request seeds make outputs independent of batching.
        for c0 in range(0, len(pending), B):
            batch = pending[c0:c0 + B]
            gp, gcl, ip, ic = generate_batch(cfg, comps, llm, batch)
            _price(gp, comps.llm_spec); _price([g for g in gcl if g], comps.llm_spec)
            for q, a, b, x, y in zip(batch, gp, gcl, ip, ic):
                tr = make_trace(cfg, run_id, chash, q, llm, a, b, x, y)
                writer.write(tr)
                log.info("iter%d %s own=%d/%d success=%s clean_correct=%s | %r", q["it"], tr["question_id"], tr["n_own_poison_in_topk"],
                         cfg.rag.top_k, tr["success"], tr["clean_correct"], a.text[:100],
                         extra={"data": {"event": "question", "iteration": q["it"], "qid": tr["question_id"], "success": tr["success"],
                                         "n_own": tr["n_own_poison_in_topk"], "clean_correct": tr["clean_correct"]}})
        t_gen = time.perf_counter() - t1
        llm = None
        _release()
    else:
        prev = run_dir / "summary.json"
        if prev.exists():
            llm_params = json.loads(prev.read_text()).get("llm", llm_params)

    traces = writer.read_all()
    s = summarize(cfg, run_id, chash, traces, llm_params,
                  {"retrieval": round(t_ret, 2), "generation": round(t_gen, 2), "total": round(time.perf_counter() - t_start, 2)},
                  comps.defense.describe())
    (run_dir / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
    if index_path is not None:
        base = index_path.parent.parent  # paths in index.json are relative to the project root
        upsert_index(index_path, {"run_id": run_id, "path": str(run_dir.relative_to(base) if run_dir.is_relative_to(base) else run_dir),
                                  "kind": "pipeline", "status": s["status"], "dataset": cfg.dataset.name,
                                  "retriever": cfg.retriever.key, "reranker": cfg.reranker.key if cfg.reranker else None,
                                  "llm": cfg.llm.key, "defense": cfg.defense.name, "attack": cfg.attack.method,
                                  "top_k": cfg.rag.top_k, "adv_per_query": cfg.attack.adv_per_query,
                                  "n_questions": s["n_questions"], "protocol": s["protocol"], "metrics": s["metrics"]["rounded"],
                                  "clean_accuracy": s["clean_accuracy"], "total_cost_usd": s["total_cost_usd"], "created": s["created"]})
    m = s["metrics"]
    log.info("ASR=%.2f P=%.2f R=%.2f F1=%.2f clean_acc=%s status=%s -> %s", m["asr_mean"], m["precision_mean"], m["recall_mean"],
             m["f1_mean"], s["clean_accuracy"], s["status"], run_dir, extra={"data": {"event": "done", "metrics": m["rounded"]}})
    return s
