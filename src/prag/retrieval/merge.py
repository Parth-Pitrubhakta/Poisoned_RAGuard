"""Upstream-faithful construction of the poisoned top-k (main.py, LM_targeted branch) and precomputed results."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Passage:
    text: str
    score: float
    doc_id: str | None = None          # BEIR doc id for clean passages, None for poison
    is_poison: bool = False
    poison_for: str | None = None      # target-question id the poison was written for
    rank: int | None = None            # 1-based rank in the final top-k
    rerank_score: float | None = None  # set when a reranker produced the final order

    def to_dict(self) -> dict:
        return asdict(self)


def load_precomputed(path: str | Path) -> dict[str, list[tuple[str, float]]]:
    """upstream results/beir_results/<ds>-<retriever>.json: {qid: {doc_id: score}} in descending-score order."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {qid: list(docs.items()) for qid, docs in raw.items()}


def poisoned_topk(clean: list[Passage], poison: list[Passage], top_k: int) -> list[Passage]:
    """upstream:
        topk_results = [clean top-k] ; topk_results += [every poison text of the iteration with its sim]
        topk_results = sorted(topk_results, key=lambda x: float(x['score']), reverse=True)[:top_k]
    Python's sort is stable, so on exact ties clean passages (appended first) stay ahead of poison.
    """
    merged = sorted(list(clean[:top_k]) + list(poison), key=lambda p: float(p.score), reverse=True)[:top_k]
    return [Passage(**{**p.to_dict(), "rank": i + 1}) for i, p in enumerate(merged)]


def count_own_poison(topk: list[Passage], own_poison_texts: list[str]) -> int:
    """upstream: cnt_from_adv = sum([i in adv_text_set for i in topk_contents]) — membership by *text*,
    counting only the poison written for this question."""
    own = set(own_poison_texts)
    return sum(p.text in own for p in topk)
