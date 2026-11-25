"""Clean retrieval quality (no poison) on NQ qrels: Recall@k and nDCG@10, for a FAISS index vs upstream precomputed.

  CUDA_VISIBLE_DEVICES=<gpu> uv run python scripts/retrieval_quality.py --index data/indexes/nq-qwen3-emb-8b --retriever qwen3-emb-8b
"""
from prag.config import force_offline

force_offline()

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from prag.config import load_registry, resolve_path  # noqa: E402
from prag.data.beir import load_qrels, load_queries  # noqa: E402
from prag.retrieval.encoders import build_encoder  # noqa: E402
from prag.retrieval.index import FaissRetriever  # noqa: E402
from prag.retrieval.merge import load_precomputed  # noqa: E402


def metrics(res, qrels, ks=(5, 20, 100)):
    out = {f"recall@{k}": float(np.mean([len(set(d for d, _ in res[q][:k]) & set(qrels[q])) / len(qrels[q]) for q in qrels])) for k in ks}
    nd = []
    for q, rel in qrels.items():
        dcg = sum(1 / np.log2(i + 2) for i, (d, _) in enumerate(res[q][:10]) if d in rel)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(10, len(rel))))
        nd.append(dcg / idcg)
    out["ndcg@10"] = float(np.mean(nd))
    return {k: round(v, 4) for k, v in out.items()}


ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--retriever", required=True)
ap.add_argument("--dataset-dir", default="data/beir/nq")
ap.add_argument("--precomputed", default="upstream/results/beir_results/nq-contriever.json")
ap.add_argument("--out", default=None, help="also write the result JSON here (e.g. experiments/retrieval_quality.json)")
a = ap.parse_args()
qrels = load_qrels(resolve_path(a.dataset_dir), "test")
queries = load_queries(resolve_path(a.dataset_dir), "test")
qids = sorted(qrels)
enc = build_encoder(a.retriever, load_registry(), device="cuda")
q = enc.encode_queries([queries[x] for x in qids])
fr = FaissRetriever(resolve_path(a.index))
print("index vector norms (first 1000):", np.round(np.linalg.norm(fr.index.reconstruct_n(0, 1000), axis=1)[[0, 1, -1]], 4))
ours = fr.topk(qids, q, 100)
res = {"dataset": str(a.dataset_dir), "split": "test", "n_queries": len(qids),
       "retrievers": {a.retriever: metrics(ours, qrels), "contriever": metrics(load_precomputed(resolve_path(a.precomputed)), qrels)},
       "notes": {a.retriever: f"FAISS IndexFlatIP {a.index}", "contriever": f"upstream precomputed {a.precomputed}"}}
print(json.dumps(res, indent=2))
if a.out:
    out = resolve_path(a.out)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[Path(a.dataset_dir).name] = res
    out.write_text(json.dumps(prev, indent=2))
