"""Write experiments/datasets.json: corpus / query / target statistics for the dashboard's Datasets page.
(The dashboard reads only experiments/*, so dataset facts are exported here once.)

  uv run python scripts/make_dataset_card.py --dataset nq
"""
import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {  # recorded in RUNLOG.md at download time
    "nq": {"beir_url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip",
           "zip_sha256": "2553bf7bfbab47b1436ca00a34bce57320e18e611fd00999a2a3a1b4714be752", "zip_bytes": 498307926},
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="nq")
a = ap.parse_args()
d = ROOT / "data" / "beir" / a.dataset
n, words, titled = 0, [], 0
with open(d / "corpus.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        n += 1
        if i % 100 == 0:  # 1% sample for length stats
            row = json.loads(line)
            words.append(len(row.get("text", "").split()))
            titled += bool(row.get("title"))
with open(d / "queries.jsonl", encoding="utf-8") as f:
    n_queries = sum(1 for _ in f)
with open(d / "qrels" / "test.tsv", encoding="utf-8") as f:
    rows = [l.split("\t") for l in f.read().splitlines()[1:]]
targets = json.loads((ROOT / "upstream" / "results" / "adv_targeted_results" / f"{a.dataset}.json").read_text())
adv_words = [len(t.split()) for v in targets.values() for t in v["adv_texts"]]
card = {
    "name": a.dataset, "split": "test", **SOURCES.get(a.dataset, {}),
    "corpus_passages": n, "queries": n_queries, "qrels_test_pairs": len(rows), "qrels_test_queries": len({r[0] for r in rows}),
    "passage_words": {"mean": round(st.mean(words), 1), "median": st.median(words), "sample": len(words)},
    "passages_with_title_share": round(titled / len(words), 3),
    "target_questions": len(targets), "adv_texts_per_target": len(next(iter(targets.values()))["adv_texts"]),
    "adv_text_words": {"mean": round(st.mean(adv_words), 1), "median": st.median(adv_words)},
    "targets_source": f"upstream/results/adv_targeted_results/{a.dataset}.json (PoisonedRAG release, LM_targeted)",
}
out = ROOT / "experiments" / "datasets.json"
allc = json.loads(out.read_text()) if out.exists() else {}
allc[a.dataset] = card
out.write_text(json.dumps(allc, indent=2))
print(json.dumps(card, indent=2))
