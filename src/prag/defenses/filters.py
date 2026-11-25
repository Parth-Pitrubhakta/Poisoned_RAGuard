"""The four defenses evaluated in the PoisonedRAG paper (Sec. 7), as pipeline defenses.

  paraphrase           query paraphrasing (7.1). Paraphrases are produced once by scripts/make_paraphrases.py
                       (paper prompt, Appendix G) and read here. Used for retrieval *and* generation, as in the paper.
  perplexity_filter    perplexity-based detection (7.2), applied as a filter: drop retrieved passages whose PPL exceeds
                       the (1 - fpr) quantile of PPL on clean corpus passages (threshold calibrated on the KB only).
  duplicate_filter     duplicate text filtering (7.3): SHA-256 of each text, keep the first (highest-ranked) copy.
  knowledge_expansion  knowledge expansion (7.4): a larger k. The work is done by `rag.top_k`; this defense only
                       labels the run and checks that `params.k == rag.top_k`.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..config import resolve_path
from ..retrieval.merge import Passage
from .base import Defense, DefenseContext


class DuplicateFilter(Defense):
    name = "duplicate_filter"

    def __init__(self, ctx: DefenseContext):
        pass

    def filter(self, question, passages):
        seen, kept, dup = set(), [], []
        for i, p in enumerate(passages):
            h = hashlib.sha256(p.text.encode("utf-8")).hexdigest()
            if h in seen:
                dup.append(i)
                continue
            seen.add(h)
            kept.append(p)
        return kept, {"duplicates_removed_idx": dup}

    def describe(self):
        return {"hash": "sha256", "keep": "first (highest-ranked) copy"}


class KnowledgeExpansion(Defense):
    name = "knowledge_expansion"

    def __init__(self, ctx: DefenseContext, k: int):
        if ctx.top_k != k:
            raise ValueError(f"knowledge_expansion k={k} must equal rag.top_k={ctx.top_k}")
        self.k = k

    def describe(self):
        return {"k": self.k}


class Paraphrase(Defense):
    name = "paraphrase"
    rewrites_query = True

    def __init__(self, ctx: DefenseContext, file: str, which: int = 0):
        d = json.loads(resolve_path(file).read_text())
        self.meta, self.items, self.which, self.file = d.get("meta", {}), d["items"], which, file

    def rewrite_query(self, qid, question):
        it = self.items.get(qid)
        if it is None or len(it.get("paraphrases", [])) <= self.which:
            raise KeyError(f"no paraphrase #{self.which} for {qid} in {self.file}")
        if it["question"] != question:
            raise ValueError(f"paraphrase file question mismatch for {qid}")
        return it["paraphrases"][self.which]

    def describe(self):
        return {"file": self.file, "which": self.which, "paraphraser": self.meta.get("model"), "prompt": self.meta.get("prompt")}


class PerplexityFilter(Defense):
    name = "perplexity_filter"

    def __init__(self, ctx: DefenseContext, lm: str = "vicuna-7b-v1.3", calib_n: int = 1000, fpr: float = 0.05,
                 max_length: int = 512, batch_size: int = 32):
        self.ctx, self.lm, self.calib_n, self.fpr = ctx, lm, calib_n, fpr
        self.max_length, self.batch_size = max_length, batch_size
        self.scorer, self.threshold, self.calib = None, None, {}

    def prepare(self, device):
        from .perplexity import build_ppl_scorer, sample_corpus_texts

        self.scorer = build_ppl_scorer(self.lm, self.ctx.registry, device=device, max_length=self.max_length,
                                       batch_size=self.batch_size)
        texts = sample_corpus_texts(self.ctx.dataset_dir, self.calib_n, self.ctx.seed)
        ppl = np.array(self.scorer.score(texts))
        self.threshold = float(np.quantile(ppl, 1 - self.fpr))
        self.calib = {"n": len(texts), "seed": self.ctx.seed,
                      "quantiles": {str(q): float(np.quantile(ppl, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}

    def release(self):
        self.scorer = None

    def filter_batch(self, items):
        uniq = sorted({p.text for _, ps in items for p in ps})
        ppl = dict(zip(uniq, self.scorer.score(uniq))) if uniq else {}
        out = []
        for _, ps in items:
            vals = [ppl[p.text] for p in ps]
            kept = [p for p, v in zip(ps, vals) if v <= self.threshold]
            out.append((kept, {"ppl": vals, "threshold": self.threshold,
                               "removed_idx": [i for i, v in enumerate(vals) if v > self.threshold]}))
        return out

    def describe(self):
        return {"lm": self.lm, "fpr_target": self.fpr, "threshold": self.threshold, "calibration": self.calib,
                "max_length": self.max_length, "note": "paper's 'cl100k_base from tiktoken' is a tokenizer, not an LM"}
