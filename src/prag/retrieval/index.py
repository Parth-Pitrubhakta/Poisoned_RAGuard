"""Clean top-k providers: upstream precomputed BEIR results, or an exact FAISS IndexFlatIP over the corpus.

Index directory layout (written by prag.build_index):
    meta.json      {retriever, hf_id, revision, dim, n, score_function, normalized, text_fn, dataset}
    doc_ids.json   row -> BEIR doc id
    index.faiss    IndexFlatIP (exact inner product; vectors L2-normalized at build time iff score_function=cos_sim)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .merge import load_precomputed


def import_faiss():
    """Import torch *before* faiss. faiss-cpu bundles its own libgomp; if it loads first and torch/CUDA come up later,
    IndexFlatIP.search segfaults on large indexes (reproduced 2026-11-19 with the 2.68M x 768 NQ index, exit 139)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    import faiss

    return faiss


class CleanRetriever(Protocol):
    def topk(self, qids: list[str], q_embs: np.ndarray | None, k: int) -> dict[str, list[tuple[str, float]]]: ...
    needs_query_embedding: bool


class PrecomputedRetriever:
    needs_query_embedding = False

    def __init__(self, path: str | Path):
        self._pre = load_precomputed(path)

    def topk(self, qids, q_embs, k):
        missing = [q for q in qids if q not in self._pre]
        if missing:
            raise KeyError(f"query ids not in precomputed results: {missing[:5]}")
        have = min(len(self._pre[q]) for q in qids)
        if have < k:
            raise ValueError(f"precomputed results hold only top-{have}, asked for {k}")
        return {q: self._pre[q][:k] for q in qids}


class FaissRetriever:
    needs_query_embedding = True

    def __init__(self, index_dir: str | Path):
        faiss = import_faiss()
        d = Path(index_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        self.doc_ids = json.loads((d / "doc_ids.json").read_text())
        self.index = faiss.read_index(str(d / "index.faiss"))
        if self.index.ntotal != len(self.doc_ids):
            raise ValueError(f"index has {self.index.ntotal} rows but doc_ids has {len(self.doc_ids)}")

    def topk(self, qids, q_embs, k):
        q = np.ascontiguousarray(q_embs, dtype=np.float32)
        if self.meta.get("normalized"):
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        scores, rows = self.index.search(q, k)
        return {qid: [(self.doc_ids[r], float(s)) for r, s in zip(rr, ss) if r >= 0]
                for qid, rr, ss in zip(qids, rows, scores)}


def build_flat_ip(embs: np.ndarray, normalize: bool):
    faiss = import_faiss()

    x = np.ascontiguousarray(embs, dtype=np.float32)
    if normalize:
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    return index


def write_index(out_dir: str | Path, embs: np.ndarray, doc_ids: list[str], meta: dict) -> Path:
    faiss = import_faiss()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    normalize = meta.get("score_function") == "cos_sim"
    faiss.write_index(build_flat_ip(embs, normalize), str(out / "index.faiss"))
    (out / "doc_ids.json").write_text(json.dumps(doc_ids))
    (out / "meta.json").write_text(json.dumps({**meta, "dim": int(embs.shape[1]), "n": len(doc_ids),
                                               "normalized": normalize}, indent=2))
    return out
