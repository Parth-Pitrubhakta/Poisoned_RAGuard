from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass
class Generation:
    text: str                         # final answer (after the output parser); ASR is computed on this
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    latency_s: float
    cost_usd: float = 0.0
    raw_text: str | None = None       # untouched completion, when it differs from `text`
    reasoning: str | None = None      # reasoning / analysis channel, if any

    def to_dict(self) -> dict:
        return asdict(self)


class TargetLLM(Protocol):
    name: str
    params: dict

    def format_prompt(self, user_msg: str) -> str: ...
    def generate(self, user_msgs: list[str], max_new_tokens: int | None = None) -> list[Generation]: ...


class EmptyOutputError(RuntimeError):
    """Raised instead of silently returning "" (upstream Vicuna.query swallows exceptions -> ASR 0.0, issue #1)."""


def cost_usd(spec: dict, prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * spec.get("price_in_per_mtok", 0.0) + completion_tokens * spec.get("price_out_per_mtok", 0.0)) / 1e6
