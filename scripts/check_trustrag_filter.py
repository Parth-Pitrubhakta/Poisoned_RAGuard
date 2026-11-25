"""Offline check of TrustRAG stage 1 (k-means + n-gram filter) on an existing run's retrieved top-k. No LLM.

    uv run python scripts/check_trustrag_filter.py --run experiments/<run_id> [--embedder contriever] [--gpu 0]

Reports, for the poisoned and the clean top-k: poison kept / removed, clean passages removed, branch counts, and the
within-poison vs within-clean cosine similarity under the embedder (the reference threshold 0.88 was tuned for SimCSE).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--embedder", default="contriever")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--sim-threshold", type=float, default=0.88)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from prag.config import force_offline, load_registry

    force_offline()
    from prag.defenses.trustrag import kmeans_filter
    from prag.retrieval.encoders import build_encoder

    traces = [json.loads(l) for l in open(Path(args.run) / "traces.jsonl")]
    enc = build_encoder(args.embedder, load_registry(), device="cuda" if args.gpu is not None else "cpu")
    texts = sorted({p["text"] for t in traces for key in ("retrieved_topk", "clean_topk") for p in t[key]})
    E = dict(zip(texts, enc.encode_docs(texts)))
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))  # noqa: E731
    report = {}
    for key in ("retrieved_topk", "clean_topk"):
        branches, kept_p, tot_p, rem_c, tot_c, pp, cc = collections.Counter(), 0, 0, 0, 0, [], []
        for t in traces:
            ps = t[key]
            keep, br = kmeans_filter(np.stack([E[p["text"]] for p in ps]), [p["text"] for p in ps], sim_t=args.sim_threshold)
            branches[br] += 1
            tot_p += sum(p["is_poison"] for p in ps)
            kept_p += sum(ps[i]["is_poison"] for i in keep)
            tot_c += sum(not p["is_poison"] for p in ps)
            rem_c += sum(not p["is_poison"] for i, p in enumerate(ps) if i not in keep)
            P = [E[p["text"]] for p in ps if p["is_poison"]]
            C = [E[p["text"]] for p in ps if not p["is_poison"]]
            pp += [cos(P[i], P[j]) for i in range(len(P)) for j in range(i + 1, len(P))]
            cc += [cos(C[i], C[j]) for i in range(len(C)) for j in range(i + 1, len(C))]
        report[key] = {"questions": len(traces), "poison_total": tot_p, "poison_kept": kept_p, "clean_total": tot_c,
                       "clean_removed": rem_c, "branches": dict(branches.most_common()),
                       "cos_poison_pairs": {"n": len(pp), "median": float(np.median(pp)) if pp else None,
                                            "p10": float(np.quantile(pp, 0.1)) if pp else None},
                       "cos_clean_pairs": {"n": len(cc), "median": float(np.median(cc)) if cc else None,
                                           "p90": float(np.quantile(cc, 0.9)) if cc else None}}
    out = json.dumps({"run": Path(args.run).name, "embedder": args.embedder, "sim_threshold": args.sim_threshold, **report}, indent=2)
    print(out)
    if args.out:
        Path(args.out).write_text(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
