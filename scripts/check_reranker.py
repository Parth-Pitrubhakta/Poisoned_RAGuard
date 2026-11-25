"""Sanity check for retrieval/reranker.py::Qwen3Reranker (model-card style example + one NQ question).

  CUDA_VISIBLE_DEVICES=<free gpu> uv run python scripts/check_reranker.py
"""
from prag.config import force_offline

force_offline()

from prag.config import load_registry  # noqa: E402
from prag.retrieval.reranker import build_reranker  # noqa: E402

rr = build_reranker("qwen3-reranker-4b", load_registry(), device="cuda")
cases = [
    ("What is the capital of China?", ["The capital of China is Beijing.", "Gravity is a force that attracts two bodies towards each other."]),
    ("who recorded i can't help falling in love with you",
     ["\"Can't Help Falling in Love\" is a pop ballad originally recorded by American singer Elvis Presley and published by Gladys Music, Elvis Presley's publishing company.",
      "The Chicago Fire Department is the department of the City of Chicago that provides fire suppression and emergency medical services."]),
]
ok = True
for q, docs in cases:
    s = rr.scores(q, docs)
    print(q, "->", [round(x, 4) for x in s])
    ok &= s[0] > 0.5 > s[1]
print("PASS" if ok else "FAIL")
