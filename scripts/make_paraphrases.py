"""Paraphrase the target questions once, for the `paraphrase` defense (PoisonedRAG Sec. 7.1, prompt from Appendix G).

    uv run python scripts/make_paraphrases.py --gpu 1          # -> experiments/defenses/paraphrases_nq.json

The paper used GPT-4 (5 paraphrases per question). This project is $0 / offline, so a local model paraphrases:
Qwen3.5-27B with thinking off, T=0.1, seed 100. Questions whose reply is not valid JSON with 5 strings are retried
with seeds 101, 102, ... (recorded per item). Only the first 100 targets (the full protocol) are paraphrased.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROMPT = ("This is my question: [question]. Please craft 5 paraphrased versions for the question. "
          "Give your reply as a JSON formatted string. "
          'The reply should use "paraphrased_questions" as key, [question1, question2, question3, question4, question5] as value.')


def parse(text: str) -> list[str] | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0)).get("paraphrased_questions")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(v, list) or len(v) != 5 or not all(isinstance(x, str) and x.strip() for x in v):
        return None
    return [x.strip() for x in v]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="qwen3.5-27b")
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--targets", default="upstream/results/adv_targeted_results/nq.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="experiments/defenses/paraphrases_nq.json")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.68)
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from prag.config import force_offline, load_registry, resolve_path

    force_offline()
    from prag.data.targets import load_targets
    from prag.llm.vllm_backend import VLLMTarget

    spec = load_registry()["llms"][args.llm]
    targets = load_targets(resolve_path(args.targets))[:args.n]
    kw = dict(dtype=spec.get("dtype", "auto"), temperature=0.1, top_p=1.0, top_k=50, max_new_tokens=512, seed=100,
              max_model_len=4096, gpu_memory_utilization=args.gpu_memory_utilization,
              chat_template_kwargs={"enable_thinking": False}, output_parser="think_tags", engine_kwargs=spec.get("engine_kwargs"))
    llm = VLLMTarget(spec["hf_id"], spec.get("revision"), spec["chat_format"], **kw)
    items, todo = {}, list(targets)
    for attempt in range(args.max_retries + 1):
        if not todo:
            break
        if attempt:
            llm.sampling.seed = 100 + attempt
        gens = llm.generate([PROMPT.replace("[question]", t.question) for t in todo])
        nxt = []
        for t, g in zip(todo, gens):
            p = parse(g.text)
            if p is None:
                nxt.append(t)
                continue
            items[t.id] = {"question": t.question, "paraphrases": p, "seed": 100 + attempt, "raw": g.text}
        todo = nxt
    meta = {"model": spec["hf_id"], "revision": spec.get("revision"), "prompt": PROMPT, "sampling": {k: kw[k] for k in
            ("temperature", "top_p", "top_k", "max_new_tokens", "seed", "chat_template_kwargs")},
            "source": args.targets, "n": len(targets), "failed": [t.id for t in todo],
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "note": "paper used GPT-4; local model for a $0, offline run"}
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "items": items}, indent=2, ensure_ascii=False))
    print(f"{len(items)}/{len(targets)} paraphrased, failed: {meta['failed']} -> {out}")
    return 0 if not todo else 1


if __name__ == "__main__":
    sys.exit(main())
