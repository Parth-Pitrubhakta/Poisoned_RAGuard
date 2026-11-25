"""Faithfulness check for retriever.mode=dense_index: does our FAISS index reproduce upstream's precomputed top-k?

Gate = the target questions used by the attack (default: first 100 of the targets file). For every target query:
  * top-k doc ids must match the precomputed list, order included, except where adjacent docs are near-ties
    (|score gap| <= --tie-eps) and only swap order;
  * scores must agree to within --score-tol.
Information = the same statistics over every query in the precomputed file.

  uv run python scripts/check_dense_vs_precomputed.py --index data/indexes/nq-contriever \
      --precomputed upstream/results/beir_results/nq-contriever.json --targets upstream/results/adv_targeted_results/nq.json
"""
import argparse
import json
import sys

from prag.config import force_offline

force_offline()  # before any HF import

import numpy as np

from prag.config import load_registry, resolve_path
from prag.data.beir import load_queries
from prag.data.targets import load_targets
from prag.retrieval.encoders import build_encoder
from prag.retrieval.index import FaissRetriever
from prag.retrieval.merge import load_precomputed


def compare(ours, theirs, k, tie_eps):
    a, b = [d for d, _ in ours[:k]], [d for d, _ in theirs[:k]]
    if a == b:
        return "exact"
    th = dict(theirs)
    if set(a) == set(b):
        return "tie_reorder" if all(abs(th[x] - th[y]) <= tie_eps for x, y in zip(a, b) if x != y) else "reorder"
    # a doc swapped in/out at the boundary on a near-tie
    kth = theirs[k - 1][1]
    ours_d = dict(ours)
    if all(abs(ours_d[d] - kth) <= tie_eps for d in set(a) - set(b)):
        return "tie_boundary"
    return "mismatch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--precomputed", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--dataset-dir", default="data/beir/nq")
    ap.add_argument("--retriever", default="contriever")
    ap.add_argument("--n-targets", type=int, default=100)
    ap.add_argument("--ks", default="5,20,100")
    ap.add_argument("--tie-eps", type=float, default=1e-4)
    ap.add_argument("--score-tol", type=float, default=1e-3)
    ap.add_argument("--all-queries", action="store_true")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    pre = load_precomputed(resolve_path(a.precomputed))
    targets = load_targets(resolve_path(a.targets))[:a.n_targets]
    queries = load_queries(resolve_path(a.dataset_dir), None)
    qids = [t.id for t in targets]
    if a.all_queries:
        qids += [q for q in pre if q not in set(qids)]
    enc = build_encoder(a.retriever, load_registry(), device="cuda")
    q_embs = np.concatenate([enc.encode_queries([queries[q]]) for q in qids])  # one query at a time, like upstream
    ours = FaissRetriever(resolve_path(a.index)).topk(qids, q_embs, max(ks))

    report = {}
    for name, sel in (("targets", qids[:len(targets)]), ("all_queries", qids)):
        if name == "all_queries" and not a.all_queries:
            continue
        r = {}
        for k in ks:
            cats = [compare(ours[q], pre[q], k, a.tie_eps) for q in sel]
            r[f"top{k}"] = {c: cats.count(c) for c in sorted(set(cats))}
        diffs = [abs(s - dict(pre[q]).get(d, s)) for q in sel for d, s in ours[q][:max(ks)] if d in dict(pre[q])]
        r["max_abs_score_diff"] = float(max(diffs))
        r["n"] = len(sel)
        report[name] = r
    gate = report["targets"]
    bad5 = gate["top5"].get("mismatch", 0) + gate["top5"].get("reorder", 0)
    ok = bad5 == 0 and gate["max_abs_score_diff"] <= a.score_tol
    report["gate"] = {"pass": ok, "rule": f"top-5 of all {len(targets)} targets match (near-ties <= {a.tie_eps} allowed), "
                                         f"|score diff| <= {a.score_tol}"}
    mism = [q for q in qids[:len(targets)] if compare(ours[q], pre[q], 5, a.tie_eps) in ("mismatch", "reorder")]
    report["target_top5_mismatches"] = {q: {"ours": ours[q][:5], "upstream": pre[q][:5]} for q in mism[:5]}
    print(json.dumps(report, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
