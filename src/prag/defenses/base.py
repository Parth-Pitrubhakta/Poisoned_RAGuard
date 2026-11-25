"""Defense hook (Step 5).

A defense can act at three points of the pipeline; each is optional (the base class is the identity):

  rewrite_query(qid, question) -> str     before retrieval (query paraphrasing). The rewritten query is used for
                                          retrieval, reranking and the RAG prompt, as in the paper (Sec. 7.1).
  filter_batch(items) -> [(kept, info)]   after retrieval / reranking, on the final top-k of every question
                                          (poisoned and clean paths alike). `prepare(device)` / `release()` load and
                                          free any model it needs; the pipeline calls them around the filter step,
                                          after the retriever and reranker are freed and before the target LLM loads.
  answer(llm, questions, contexts)        replaces the single MULTIPLE_PROMPT call (multi-stage generation, e.g.
                                          TrustRAG conflict resolution). Returns (generations, per-item info).

A defense never reads `Passage.is_poison`. That flag exists for evaluation only, and the pipeline records
how many poison passages a defense kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..retrieval.merge import Passage


@dataclass
class DefenseContext:
    registry: dict = field(default_factory=dict)
    dataset_dir: Path | None = None
    seed: int = 12
    top_k: int = 5
    batch_size: int = 64


class Defense:
    name = "none"
    rewrites_query = False
    has_answer = False

    def rewrite_query(self, qid: str, question: str) -> str:
        return question

    def prepare(self, device: str) -> None:
        pass

    def release(self) -> None:
        pass

    def filter(self, question: str, passages: list[Passage]) -> tuple[list[Passage], dict]:
        return passages, {}

    def filter_batch(self, items: list[tuple[str, list[Passage]]]) -> list[tuple[list[Passage], dict]]:
        return [self.filter(q, ps) for q, ps in items]

    def describe(self) -> dict:
        """Parameters recorded in summary.json (defense_params)."""
        return {}


class NoDefense(Defense):
    name = "none"

    def __init__(self, ctx: DefenseContext | None = None):
        pass


def _registry() -> dict[str, type]:
    from .filters import DuplicateFilter, KnowledgeExpansion, Paraphrase, PerplexityFilter
    from .trustrag import TrustRAG

    return {c.name: c for c in (NoDefense, DuplicateFilter, KnowledgeExpansion, Paraphrase, PerplexityFilter, TrustRAG)}


def available() -> list[str]:
    return sorted(_registry())


def build_defense(name: str, params: dict | None = None, ctx: DefenseContext | None = None) -> Defense:
    reg = _registry()
    if name not in reg:
        raise ValueError(f"unknown defense {name!r}; available: {sorted(reg)}")
    cls = reg[name]
    return cls(ctx or DefenseContext(), **(params or {}))


