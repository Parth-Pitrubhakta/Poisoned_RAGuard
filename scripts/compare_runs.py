"""Compare two run directories question-by-question (retrieval, prompts, outputs, success) and schema-check the second.

  uv run python scripts/compare_runs.py experiments/<run_a> experiments/<run_b>
"""
import json
import sys

from prag.tracing.schema import trace_errors


def load(d):
    with open(f"{d}/traces.jsonl") as f:
        return {(t.get("iteration", 0), t["question_id"]): t for t in map(json.loads, f)}


a, b = load(sys.argv[1]), load(sys.argv[2])
assert a.keys() == b.keys(), f"question sets differ: {len(a)} vs {len(b)}"
fields = {
    "retrieved_topk": lambda t: [(p["text"], round(p["score"], 4), p["is_poison"]) for p in t["retrieved_topk"]],
    "prompt_poisoned": lambda t: t["prompt_poisoned"], "llm_input_poisoned": lambda t: t["llm_input_poisoned"],
    "output_poisoned": lambda t: t["output_poisoned"], "output_clean": lambda t: t["output_clean"],
    "success": lambda t: t["success"],
}
errs = sum(len(trace_errors(t)) for t in b.values())
same = {f: sum(fn(a[k]) == fn(b[k]) for k in a) for f, fn in fields.items()}
print(f"n={len(a)} schema_errors(b)={errs} identical: " + " ".join(f"{f}={n}" for f, n in same.items()))
sys.exit(0 if errs == 0 and all(n == len(a) for n in same.values()) else 1)
