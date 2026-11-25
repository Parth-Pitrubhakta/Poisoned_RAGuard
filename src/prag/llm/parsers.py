"""Split a raw completion into (final answer, reasoning) for reasoning models.

ASR is computed on the *final answer* only. Reasoning is kept in the trace, but a target string that appears only in
the reasoning does not count as attack success.
"""
from __future__ import annotations

import re

_HARMONY_FINAL = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.S)
_HARMONY_ANALYSIS = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)", re.S)


def parse_none(raw: str) -> tuple[str, str | None]:
    return raw, None


def parse_think_tags(raw: str) -> tuple[str, str | None]:
    """Qwen3 / Qwen3.5 style: optional '<think>' ... '</think>' then the answer. With enable_thinking=False the
    template emits an empty think block, which this also handles."""
    if "</think>" in raw:
        reasoning, answer = raw.split("</think>", 1)
        return answer.strip(), reasoning.replace("<think>", "").strip() or None
    if "<think>" in raw:  # truncated mid-reasoning: no final answer
        return "", raw.replace("<think>", "").strip()
    return raw.strip(), None


def parse_harmony(raw: str) -> tuple[str, str | None]:
    """gpt-oss harmony output decoded with special tokens kept. Missing final channel -> empty answer."""
    analysis = " ".join(m.strip() for m in _HARMONY_ANALYSIS.findall(raw)) or None
    finals = _HARMONY_FINAL.findall(raw)
    if finals:
        return finals[-1].strip(), analysis
    if "<|channel|>" not in raw:  # plain text (e.g. served through a parser already)
        return raw.strip(), None
    return "", analysis


PARSERS = {"none": parse_none, "think_tags": parse_think_tags, "harmony": parse_harmony}
