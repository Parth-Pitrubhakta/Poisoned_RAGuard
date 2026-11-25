"""Write experiments/trace_aggregates.json: the per-run secondary metrics the dashboard computes from traces.

  uv run python scripts/export_trace_aggregates.py

traces.jsonl files hold the benchmark poison texts and are not committed to the public repo (see .gitignore). This file
holds only numbers (typography-normalized ASR / clean accuracy, answered-only ASR, empty-final and truncation rates),
so the Results page works on a clone without traces. It uses the dashboard's own function (app/lib.py::normalized_stats),
so the numbers are identical to what the dashboard shows when traces are present. Re-run after any new run.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import lib  # noqa: E402  (dashboard loaders; read-only)


def main() -> int:
    idx = json.loads((lib.EXP / "index.json").read_text())
    out = {}
    for r in idx["runs"]:
        if (lib.BASE / r["path"] / "traces.jsonl").exists():
            out[r["path"]] = lib.normalized_stats(r["path"])
    path = lib.EXP / "trace_aggregates.json"
    path.write_text(json.dumps({"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "note": "numbers only; from traces.jsonl via app/lib.py::normalized_stats",
                                "runs": out}, indent=1) + "\n")
    print(f"{path.relative_to(ROOT)}: {len(out)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
