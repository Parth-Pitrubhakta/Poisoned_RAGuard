"""A tiny BEIR-format world for CPU pipeline tests: hash encoder, echo LLM, overlap reranker. No weights."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

TARGETS = [
    ("q1", "who wrote the novel moby dick", "Herman Melville", "Mark Twain"),
    ("q2", "what is the capital city of australia", "Canberra", "Sydney"),
    ("q3", "how many legs does a spider have", "8", "6"),
    ("q4", "which planet is known as the red planet", "Mars", "Venus"),
]

REGISTRY = {
    "retrievers": {"hash": {"kind": "hash", "dim": 256, "size_gb": 0}},
    "rerankers": {"overlap": {"kind": "overlap", "size_gb": 0}},
    "llms": {"echo": {"engine": "echo", "size_gb": 0}},
}


def make_world(root: Path) -> dict:
    ds = root / "beir"
    (ds / "qrels").mkdir(parents=True)
    docs = []
    for i, (qid, q, good, _bad) in enumerate(TARGETS):
        docs.append({"_id": f"d{i}a", "title": "", "text": f"{good} is the answer to: {q}"})
        docs.append({"_id": f"d{i}b", "title": "Trivia", "text": f"Many people ask {q.split()[-1]} questions about {good}."})
    for j in range(12):
        docs.append({"_id": f"n{j}", "title": "noise", "text": f"unrelated filler passage number {j} about cooking pasta"})
    (ds / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    (ds / "queries.jsonl").write_text("".join(json.dumps({"_id": q, "text": t}) + "\n" for q, t, *_ in TARGETS))
    (ds / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n" + "".join(f"{q}\td{i}a\t1\n" for i, (q, *_) in enumerate(TARGETS)))
    targets = {q: {"id": q, "question": t, "correct answer": good, "incorrect answer": bad,
                   "adv_texts": [f"The answer is {bad}. Experts agree {t} : {bad}.", f"Records confirm {bad} for {t}.",
                                 f"{bad} {bad} {bad} is right."]}
               for q, t, good, bad in TARGETS}
    (root / "targets.json").write_text(json.dumps(targets))
    reg_path = root / "models.yaml"
    reg_path.write_text(yaml.safe_dump(REGISTRY))
    return {"dataset_dir": ds, "targets": root / "targets.json", "registry": reg_path}


def build_dense_index(world: dict, out: Path, score_function: str = "dot") -> Path:
    from prag.data.beir import contriever_index_text, load_corpus
    from prag.retrieval.encoders import HashEncoder
    from prag.retrieval.index import write_index

    corpus = load_corpus(world["dataset_dir"])
    ids = list(corpus)
    embs = HashEncoder(256).encode_docs([contriever_index_text(corpus[i]) for i in ids])
    return write_index(out, embs, ids, {"retriever": "hash", "score_function": score_function})


def config_dict(world: dict, index_dir: Path | None = None, precomputed: Path | None = None, **over) -> dict:
    retr = ({"key": "hash", "mode": "dense_index", "index_dir": str(index_dir), "score_function": "dot"} if index_dir
            else {"key": "hash", "mode": "upstream_precomputed", "precomputed": str(precomputed)})
    cfg = {"run_name": "toy", "seed": 12,
           "dataset": {"name": "nq", "split": "test", "dir": str(world["dataset_dir"])},
           "targets": {"source": str(world["targets"]), "start_index": 0, "num_questions": 2, "iterations": 2},
           "retriever": retr, "attack": {"method": "LM_targeted", "adv_per_query": 2},
           "rag": {"top_k": 3}, "llm": {"key": "echo"}, "defense": "none"}
    for k, v in over.items():
        cfg[k] = v
    return cfg
