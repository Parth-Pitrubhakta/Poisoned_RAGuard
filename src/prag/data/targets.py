"""Upstream released target questions + poison texts (`results/adv_targeted_results/<dataset>.json`)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TargetQuestion:
    id: str
    question: str
    correct_answer: str
    incorrect_answer: str          # the attacker-chosen target answer
    adv_texts: tuple[str, ...] = field(default_factory=tuple)  # released "S" texts (before query prefix)


def load_targets(path: str | Path) -> list[TargetQuestion]:
    """Ordered exactly like upstream: `list(load_json(path).values())` (JSON object order)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for key, v in raw.items():
        assert v["id"] == key, f"id mismatch for {key}"
        out.append(TargetQuestion(id=v["id"], question=v["question"], correct_answer=v["correct answer"],
                                  incorrect_answer=v["incorrect answer"], adv_texts=tuple(v["adv_texts"])))
    return out
