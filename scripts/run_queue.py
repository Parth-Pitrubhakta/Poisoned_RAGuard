"""Tiny file-backed work queue: each worker owns one GPU and pops config paths until the queue is empty.

  uv run python scripts/run_queue.py --queue configs/defenses/queue_big.txt --gpu 1

The queue file holds one config path per line. Popping takes an exclusive flock, so several workers can share a queue.
Each job runs `python -m prag.run --config <cfg> --gpu <gpu>` (offline, own run dir). Outcomes are appended to
<queue>.done as "<iso time>\t<gpu>\t<rc>\t<seconds>\t<config>". A failed job is not retried. Fix it, then `--resume`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pop(queue: Path) -> str | None:
    with open(queue, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        lines = [l for l in f.read().splitlines() if l.strip()]
        if not lines:
            return None
        f.seek(0); f.truncate(); f.write("".join(l + "\n" for l in lines[1:]))
        return lines[0].strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--gpu", required=True)
    args = ap.parse_args()
    q = Path(args.queue)
    done = q.with_suffix(q.suffix + ".done")
    while (cfg := pop(q)) is not None:
        t0 = time.time()
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} GPU{args.gpu} start {cfg}", flush=True)
        rc = subprocess.call([sys.executable, "-m", "prag.run", "--config", cfg, "--gpu", args.gpu], cwd=ROOT)
        line = f"{dt.datetime.now().isoformat(timespec='seconds')}\t{args.gpu}\t{rc}\t{time.time() - t0:.0f}\t{cfg}"
        with open(done, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
