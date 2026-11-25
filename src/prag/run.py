"""CLI for the Step 2 pipeline.

  uv run python -m prag.run --config configs/<cfg>.yaml --dry-run     # plan + time/cost/disk estimate, loads nothing
  uv run python -m prag.run --config configs/<cfg>.yaml               # new run -> experiments/<run_id>/
  uv run python -m prag.run --resume experiments/<run_id>             # continue an interrupted run (same config)
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

from .config import ROOT, RunConfig, check_registry, config_hash, load_config, load_registry

log = logging.getLogger("prag.run")

# Measured in Step 1 (NQ, prompt_id 4, top_k 5): ~420 prompt tokens, ~25 completion tokens per call.
EST_PROMPT_TOKENS, EST_COMPLETION_TOKENS = 450, 40
GPU_TOTAL_GB = 141.0


def _hub_dir(hf_id: str) -> Path:
    return Path(os.environ.get("HF_HOME", ROOT / ".cache" / "hf")) / "hub" / ("models--" + hf_id.replace("/", "--"))


def cached_gb(hf_id: str) -> float:
    d = _hub_dir(hf_id) / "blobs"
    return sum(p.stat().st_size for p in d.glob("*")) / 1e9 if d.exists() else 0.0


def gpu_free() -> list[tuple[str, float]]:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [(i.strip(), int(f) / 1024) for i, f in (l.split(",") for l in out.strip().splitlines())]


def needs_gpu(cfg: RunConfig, reg: dict) -> bool:
    return (reg["llms"][cfg.llm.key].get("engine") == "vllm"
            or reg["retrievers"][cfg.retriever.key].get("kind") not in ("hash",)
            or bool(cfg.reranker and reg["rerankers"][cfg.reranker.key].get("kind") != "overlap"))


def pick_gpus(n: int, min_free_gb: float) -> list[tuple[str, float]]:
    gpus = sorted(gpu_free(), key=lambda t: -t[1])[:n]
    if len(gpus) < n or gpus[-1][1] < min_free_gb:
        raise RuntimeError(f"need {n} GPU(s) with >= {min_free_gb} GB free; have {gpu_free()} (shared box, not starting)")
    return gpus


def estimate(cfg: RunConfig, reg: dict) -> dict:
    """Plan for --dry-run. Pure bookkeeping: reads metadata from disk, loads no model."""
    spec = reg["llms"][cfg.llm.key]
    n_q = cfg.targets.total
    paths = 2 if cfg.eval.clean_run else 1
    # TrustRAG conflict resolution: 1 shared internal-knowledge call + 2 calls per path
    per_q = (1 + 2 * paths) if (cfg.defense.name == "trustrag" and cfg.defense.params.get("conflict", True)) else paths
    calls = (1 if cfg.eval.smoke_test_nonempty else 0) + n_q * per_q
    ptok_per = EST_PROMPT_TOKENS * max(1.0, cfg.rag.top_k / 5)
    ptok, ctok_est, ctok_max = int(calls * ptok_per), calls * EST_COMPLETION_TOKENS, calls * cfg.llm.max_new_tokens
    pin, pout = spec.get("price_in_per_mtok", 0.0), spec.get("price_out_per_mtok", 0.0)
    cost_est = (ptok * pin + ctok_est * pout) / 1e6
    cost_max = (ptok * pin + ctok_max * pout) / 1e6

    models = [("llm", cfg.llm.key, spec)]
    models.append(("retriever", cfg.retriever.key, reg["retrievers"][cfg.retriever.key]))
    if cfg.reranker:
        models.append(("reranker", cfg.reranker.key, reg["rerankers"][cfg.reranker.key]))
    weights, download_gb = [], 0.0
    for role, key, s in models:
        if "hf_id" not in s:
            weights.append({"role": role, "key": key, "source": s.get("engine") or s.get("kind"), "cached": True, "download_gb": 0.0})
            continue
        have = cached_gb(s["hf_id"])
        need = max(0.0, s.get("size_gb", 0.0) - have) if have < 0.9 * s.get("size_gb", 0.0) else 0.0
        download_gb += need
        weights.append({"role": role, "key": key, "hf_id": s["hf_id"], "cached_gb": round(have, 2), "download_gb": round(need, 1)})

    files = {"targets": cfg.path(cfg.targets.source), "corpus": cfg.path(cfg.dataset.dir) / "corpus.jsonl"}
    if cfg.retriever.mode == "upstream_precomputed":
        files["precomputed"] = cfg.path(cfg.retriever.precomputed)
    else:
        files["index"] = cfg.path(cfg.retriever.index_dir) / "index.faiss"
    missing = {k: str(v) for k, v in files.items() if not v.exists()}

    engine = spec.get("engine")
    size = spec.get("size_gb", 0.0)
    load_s = 0 if engine in ("echo", "ollama") else 60 + 4 * size / max(1, cfg.llm.tensor_parallel_size)
    gen_s = 0.0 if engine == "echo" else (calls * 0.25 if engine == "vllm" else calls * 6.0)  # vLLM batches; Ollama is serial
    total_s = 30 + load_s + gen_s + (60 if cfg.reranker else 0)
    gpu_budget = cfg.llm.gpu_memory_utilization * GPU_TOTAL_GB * cfg.llm.tensor_parallel_size
    flags = []
    if download_gb > 2:
        flags.append(f"download {download_gb:.1f} GB > 2 GB")
    if cost_max > 5:
        flags.append(f"worst-case API cost ${cost_max:.2f} > $5")
    if total_s > 600:
        flags.append(f"est. {total_s / 60:.0f} min > 10 min")
    if missing:
        flags.append("missing inputs: " + ", ".join(missing))
    if engine == "vllm" and size > 0.85 * gpu_budget:
        flags.append(f"weights {size} GB vs vLLM budget {gpu_budget:.0f} GB: raise gpu_memory_utilization or tensor_parallel_size")
    if engine == "ollama":
        flags.append("uses the shared system Ollama service")
    return {"run_name": cfg.run_name, "protocol": f"{cfg.targets.iterations} iter x M={cfg.targets.num_questions} = {n_q} questions",
            "dataset": cfg.dataset.name, "retriever": f"{cfg.retriever.key} ({cfg.retriever.mode}, {cfg.retriever.score_function})",
            "reranker": cfg.reranker.key if cfg.reranker else None, "llm": f"{cfg.llm.key} ({engine})", "defense": cfg.defense.name,
            "llm_calls": calls, "tokens_est": {"prompt": ptok, "completion": ctok_est, "completion_max": ctok_max},
            "cost_usd": {"est": round(cost_est, 4), "max": round(cost_max, 4)}, "weights": weights,
            "download_gb": round(download_gb, 1), "missing": missing, "time_est_s": round(total_s),
            "disk_out_mb": round(0.02 * n_q, 1), "gpus_needed": cfg.llm.tensor_parallel_size if needs_gpu(cfg, reg) else 0,
            "gpu_free_gb": {i: round(f, 1) for i, f in gpu_free()}, "needs_approval": flags}


def build_components(cfg: RunConfig, reg: dict, device: str):
    from .defenses.base import DefenseContext, build_defense
    from .llm.backends import build_llm
    from .pipeline import Components
    from .retrieval.encoders import build_encoder
    from .retrieval.index import FaissRetriever, PrecomputedRetriever
    from .retrieval.reranker import build_reranker

    clean = (PrecomputedRetriever(cfg.path(cfg.retriever.precomputed)) if cfg.retriever.mode == "upstream_precomputed"
             else FaissRetriever(cfg.path(cfg.retriever.index_dir)))
    rr = cfg.reranker
    return Components(
        make_encoder=lambda: build_encoder(cfg.retriever.key, reg, device=device, batch_size=cfg.retriever.batch_size),
        clean=clean,
        make_reranker=(lambda: build_reranker(rr.key, reg, device=device, batch_size=rr.batch_size, max_length=rr.max_length)) if rr else None,
        make_llm=lambda: build_llm(cfg.llm, reg),
        defense=build_defense(cfg.defense.name, cfg.defense.params,
                              DefenseContext(registry=reg, dataset_dir=cfg.path(cfg.dataset.dir), seed=cfg.seed,
                                             top_k=cfg.rag.top_k, batch_size=cfg.retriever.batch_size)),
        llm_spec=reg["llms"][cfg.llm.key],
        device=device,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--resume", help="existing experiments/<run_id> directory")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--experiments-dir", default=str(ROOT / "experiments"))
    ap.add_argument("--registry", default=None, help="models.yaml (default configs/models.yaml)")
    ap.add_argument("--gpu", default=None, help="pin GPU id(s), e.g. 2 or 1,2 (overrides llm.gpu=auto; free memory still checked)")
    args = ap.parse_args(argv)
    if not (args.config or args.resume):
        ap.error("--config or --resume is required")

    from .tracing.writer import setup_logging

    reg = load_registry(args.registry)
    exp = Path(args.experiments_dir)
    if args.resume:
        run_dir = Path(args.resume).resolve()
        cfg_text = (run_dir / "config.yaml").read_text()
        cfg = RunConfig.model_validate(yaml.safe_load(cfg_text)); check_registry(cfg, reg)
        if args.config:
            other, _ = load_config(args.config, reg)
            if config_hash(other) != config_hash(cfg):
                raise SystemExit(f"--config differs from {run_dir}/config.yaml (hash {config_hash(other)} != {config_hash(cfg)})")
        run_id = run_dir.name
    else:
        cfg, cfg_text = load_config(args.config, reg)

    if args.dry_run:
        setup_logging(None)
        import json
        plan = estimate(cfg, reg)
        print(json.dumps(plan, indent=2))
        print("NEEDS APPROVAL: " + "; ".join(plan["needs_approval"]) if plan["needs_approval"] else "within guardrails (no approval needed)")
        return 0

    from .config import force_offline

    force_offline()
    if not args.resume:
        run_id = f"{cfg.run_name}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_dir = exp / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text(cfg_text)
    setup_logging(run_dir)

    device = "cpu"
    if needs_gpu(cfg, reg):
        if args.gpu is not None:
            free = dict(gpu_free())
            gpus = [(g, free.get(g, -1.0)) for g in args.gpu.split(",")]
            low = [g for g, f in gpus if f < cfg.llm.min_free_gb]
            if low:
                raise RuntimeError(f"--gpu {args.gpu}: GPU(s) {low} have < {cfg.llm.min_free_gb} GB free")
        elif cfg.llm.gpu == "auto":
            gpus = pick_gpus(cfg.llm.tensor_parallel_size, cfg.llm.min_free_gb)
        else:
            gpus = [(str(cfg.llm.gpu), -1.0)]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(g for g, _ in gpus)
        device = "cuda"
        log.info("GPUs %s", gpus, extra={"data": {"event": "gpu", "gpus": gpus}})

    from .pipeline import execute

    execute(cfg, build_components(cfg, reg, device), run_dir, run_id, exp / "index.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
