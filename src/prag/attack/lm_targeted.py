"""PoisonedRAG black-box attack (`attack_method="LM_targeted"`), faithful to upstream src/attack.py.

Each poison text P = S + I, where S = the target question (+ ".") makes P retrievable for that question,
and I = an LLM-written passage asserting the attacker's target answer (released by upstream; we reuse it).
Confined to local benchmark corpora — see CLAUDE.md.
"""
from __future__ import annotations

from ..data.targets import TargetQuestion


def lm_targeted_poison(target: TargetQuestion, adv_per_query: int) -> list[str]:
    """upstream: adv_texts = [question + "." + t for t in all_adv_texts[id]['adv_texts'][:adv_per_query]]"""
    if adv_per_query > len(target.adv_texts):
        raise ValueError(f"{target.id}: only {len(target.adv_texts)} released poison texts, asked for {adv_per_query}")
    prefix = target.question + "."
    return [prefix + t for t in target.adv_texts[:adv_per_query]]
