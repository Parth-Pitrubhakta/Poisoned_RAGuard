"""Passage perplexity under a causal LM (for the paper's perplexity-based detection, Sec. 7.2).

The paper says it uses "the cl100k_base model from OpenAI tiktoken to calculate perplexity". cl100k_base is a
tokenizer, not a language model, so that cannot be reproduced literally. We use a local causal LM from the registry
(default: Vicuna-7B v1.3, the paper-era target model) through HF transformers. PPL = exp(mean token NLL),
with BOS prepended, over the first `max_length` tokens.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..data.beir import corpus_text


class HFPerplexity:
    def __init__(self, hf_id: str, revision: str | None, device: str = "cuda", dtype: str = "float16",
                 max_length: int = 512, batch_size: int = 32):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.unk_token or self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision, dtype=getattr(torch, dtype)).to(device).eval()
        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self.name = hf_id

    def score(self, texts: list[str]) -> list[float]:
        torch = self.torch
        out: list[float] = []
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # length-sorted batches, scattered back
        res = [0.0] * len(texts)
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_length, add_special_tokens=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = enc.input_ids[:, 1:]
            mask = enc.attention_mask[:, 1:].float()
            nll = -(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask).sum(1) / mask.sum(1).clamp(min=1)
            for i, v in zip(idx, nll.tolist()):
                res[i] = float(math.exp(v))
        out.extend(res)
        return out


class ToyPerplexity:
    """CPU test double: 'perplexity' = 1 + mean word length (longer words = more surprising). Deterministic."""
    name = "toy"

    def score(self, texts):
        return [1.0 + float(np.mean([len(w) for w in t.split()] or [0])) for t in texts]


def build_ppl_scorer(llm_key: str, registry: dict, device: str = "cuda", max_length: int = 512, batch_size: int = 32):
    spec = registry["llms"][llm_key]
    if spec.get("engine") == "echo":
        return ToyPerplexity()
    return HFPerplexity(spec["hf_id"], spec.get("revision"), device=device, dtype=spec.get("dtype", "float16"),
                        max_length=max_length, batch_size=batch_size)


def sample_corpus_texts(dataset_dir: str | Path, n: int, seed: int) -> list[str]:
    """`n` passages drawn uniformly (without replacement, numpy default_rng(seed)) from corpus.jsonl,
    in file order. Used to calibrate the PPL threshold on clean knowledge-base text only."""
    path = Path(dataset_dir) / "corpus.jsonl"
    with open(path, "rb") as f:
        total = sum(1 for _ in f)
    pick = set(np.random.default_rng(seed).choice(total, size=min(n, total), replace=False).tolist())
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in pick:
                out.append(corpus_text(json.loads(line)))
    return out
