"""BEIR-format dataset loading (corpus.jsonl / queries.jsonl / qrels/<split>.tsv).

Mirrors `beir.datasets.data_loader.GenericDataLoader` semantics used by upstream PoisonedRAG:
  corpus[doc_id]  = {"title": str, "text": str}
  queries[qid]    = str            (only queries that appear in qrels for the split)
  qrels[qid][doc] = int
We avoid the `beir` package itself (it drags in pytrec_eval); the format is trivial.

`load_corpus(..., ids=...)` streams the 1.5 GB NQ corpus and keeps only the requested doc ids, so the
small-subset baseline does not have to hold 2.68M passages in memory.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

UPSTREAM_SPLIT_OVERRIDE = {"msmarco": "train"}  # upstream main.py loads msmarco with split="train"


def corpus_text(doc: dict) -> str:
    """Passage string placed in the RAG context. Upstream main.py uses `corpus[id]["text"]` (no title)."""
    return doc["text"]


def contriever_index_text(doc: dict) -> str:
    """Passage string used when upstream built `beir_results` (contriever beir_utils.encode_corpus)."""
    return doc["title"] + " " + doc["text"] if len(doc["title"]) > 0 else doc["text"]


def load_corpus(dataset_dir: str | Path, ids: Iterable[str] | None = None) -> dict[str, dict]:
    want = set(ids) if ids is not None else None
    out: dict[str, dict] = {}
    with open(Path(dataset_dir) / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = str(row["_id"])
            if want is not None and doc_id not in want:
                continue
            out[doc_id] = {"title": row.get("title") or "", "text": row.get("text") or ""}
            if want is not None and len(out) == len(want):
                break
    if want is not None:
        missing = want - out.keys()
        if missing:
            raise KeyError(f"{len(missing)} doc ids not found in corpus, e.g. {sorted(missing)[:5]}")
    return out


def load_qrels(dataset_dir: str | Path, split: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(Path(dataset_dir) / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)  # header: query-id corpus-id score
        for qid, doc_id, score in reader:
            qrels.setdefault(qid, {})[doc_id] = int(score)
    return qrels


def load_queries(dataset_dir: str | Path, split: str | None = None) -> dict[str, str]:
    qrels = load_qrels(dataset_dir, split) if split else None
    queries: dict[str, str] = {}
    with open(Path(dataset_dir) / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = str(row["_id"])
            if qrels is None or qid in qrels:
                queries[qid] = row["text"]
    return queries
