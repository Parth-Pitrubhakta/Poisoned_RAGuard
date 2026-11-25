"""Run directory I/O: schema-checked append-only traces (the checkpoint), structured logging, and the index."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from .schema import validate_trace


class TraceWriter:
    """traces.jsonl is the checkpoint: every record is validated, then appended and fsynced.
    On resume, records already present (keyed by (iteration, question_id)) are skipped."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.done: set[tuple[int, str]] = set()
        if self.path.exists():
            good = []
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:  # torn last line from a crash: drop it
                        continue
                    good.append(line if line.endswith("\n") else line + "\n")
                    self.done.add((rec["iteration"], rec["question_id"]))
            self.path.write_text("".join(good), encoding="utf-8")

    def write(self, rec: dict) -> None:
        validate_trace(rec)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.done.add((rec["iteration"], rec["question_id"]))

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]


class JsonLineFormatter(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        d = {"ts": self.formatTime(r, "%Y-%m-%dT%H:%M:%S"), "level": r.levelname, "logger": r.name, "msg": r.getMessage()}
        d.update(getattr(r, "data", {}) or {})
        if r.exc_info:
            d["exc"] = self.formatException(r.exc_info)
        return json.dumps(d, ensure_ascii=False, default=str)


def setup_logging(run_dir: Path | None) -> None:
    """Human-readable console, JSON lines in <run_dir>/run.log. Use log.info(msg, extra={'data': {...}})."""
    con = logging.StreamHandler(sys.stdout)
    con.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S"))
    handlers: list[logging.Handler] = [con]
    if run_dir is not None:
        fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        fh.setFormatter(JsonLineFormatter())
        handlers.append(fh)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def upsert_index(index_path: Path, entry: dict) -> None:
    index = json.loads(index_path.read_text()) if index_path.exists() else {"runs": []}
    index["runs"] = [r for r in index["runs"] if r["run_id"] != entry["run_id"]] + [entry]
    tmp = index_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, indent=2))
    tmp.replace(index_path)
