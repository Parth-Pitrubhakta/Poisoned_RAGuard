"""Embed a BEIR corpus and build an exact FAISS IndexFlatIP (retriever.mode: dense_index).

Docs are embedded as title + " " + text (as upstream's beir_results were computed). The RAG context uses `text` only.

  # plan only (rows, dim, disk, rough time), loads nothing
  uv run python -m prag.build_index --dataset-dir data/beir/nq --retriever qwen3-emb-8b --out data/indexes/nq-qwen3-emb-8b --dry-run
  # embed shard i of n (run n of these in parallel, one per GPU), then merge
  CUDA_VISIBLE_DEVICES=0 uv run python -m prag.build_index ... --shard 0 --num-shards 4
  uv run python -m prag.build_index ... --merge --num-shards 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .config import load_registry, resolve_path

log = logging.getLogger("prag.build_index")

# rough single-H200 throughput (docs/s), length-sorted batches. Contriever fp32 measured ~440 docs/s UNsorted (2026-11-19).
THROUGHPUT = {"contriever_meanpool": 1500, "qwen3_embedding": {"0.6B": 2500, "8B": 450}, "sentence_transformer": 2000}


def _corpus_ids_texts(dataset_dir: Path):
    from .data.beir import contriever_index_text, load_corpus

    corpus = load_corpus(dataset_dir)
    ids = list(corpus)
    return ids, [contriever_index_text(corpus[i]) for i in ids]


def _shard_range(n: int, shard: int, num: int) -> tuple[int, int]:
    per = -(-n // num)
    return shard * per, min(n, (shard + 1) * per)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--retriever", required=True, help="configs/models.yaml retrievers.<key>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--score-function", choices=["dot", "cos_sim"], default="dot")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--registry", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    reg = load_registry(a.registry)
    spec = reg["retrievers"][a.retriever]
    ds, out = resolve_path(a.dataset_dir), resolve_path(a.out)

    if a.dry_run:
        with open(ds / "corpus.jsonl", "rb") as f:
            n = sum(1 for _ in f)
        dim = spec.get("dim", 768)
        tp = THROUGHPUT.get(spec["kind"], 1000)
        if isinstance(tp, dict):
            tp = tp["8B"] if "8B" in spec.get("hf_id", "") else tp["0.6B"]
        gb = n * dim * 4 / 1e9
        print(json.dumps({"rows": n, "dim": dim, "disk_gb": {"shards_fp32": round(gb, 1), "index_faiss": round(gb, 1),
                                                             "total_peak": round(2 * gb, 1)},
                          "time_est_min_per_gpu_total": round(n / tp / 60), "with_num_shards": a.num_shards,
                          "wall_est_min": round(n / tp / 60 / a.num_shards),
                          "weights": {"hf_id": spec.get("hf_id"), "size_gb": spec.get("size_gb")}}, indent=2))
        return 0

    from .config import force_offline

    force_offline()
    ids, texts = _corpus_ids_texts(ds)
    out.mkdir(parents=True, exist_ok=True)
    if a.merge:
        from .retrieval.index import write_index

        parts = [np.load(out / f"emb_{i:03d}-of-{a.num_shards:03d}.npy") for i in range(a.num_shards)]
        embs = np.concatenate(parts)
        if embs.shape[0] != len(ids):
            raise ValueError(f"shards hold {embs.shape[0]} rows, corpus has {len(ids)}")
        write_index(out, embs, ids, {"retriever": a.retriever, "hf_id": spec.get("hf_id"), "revision": spec.get("revision"),
                                     "score_function": a.score_function, "text_fn": "title + ' ' + text",
                                     "dataset": str(a.dataset_dir), "built": time.strftime("%Y-%m-%dT%H:%M:%S")})
        log.info("index written: %s (%d x %d)", out, *embs.shape)
        return 0

    from .retrieval.encoders import build_encoder

    lo, hi = _shard_range(len(ids), a.shard, a.num_shards)
    path = out / f"emb_{a.shard:03d}-of-{a.num_shards:03d}.npy"
    if path.exists():
        log.info("shard exists, skipping: %s", path)
        return 0
    enc = build_encoder(a.retriever, reg, device="cuda", batch_size=a.batch_size)
    # Encode longest-first (as BEIR's DenseRetrievalExactSearch does) so batches have similar lengths and little
    # padding. Padding is masked, so embeddings don't depend on the order; rows are scattered back to corpus order.
    order = sorted(range(lo, hi), key=lambda i: len(texts[i]), reverse=True)
    t0, chunk, out_rows = time.perf_counter(), 50_000, None
    for s in range(0, len(order), chunk):
        idx = order[s:s + chunk]
        e = enc.encode_docs([texts[i] for i in idx])
        if out_rows is None:
            out_rows = np.zeros((hi - lo, e.shape[1]), dtype=np.float32)
        out_rows[np.asarray(idx) - lo] = e
        done = min(len(order), s + chunk)
        rate = done / (time.perf_counter() - t0)
        log.info("shard %d: %d/%d docs, %.0f docs/s, eta %.1f min", a.shard, done, hi - lo, rate, (hi - lo - done) / rate / 60)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, out_rows)
    tmp.replace(path)
    log.info("shard %d done: rows %d..%d -> %s", a.shard, lo, hi, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
