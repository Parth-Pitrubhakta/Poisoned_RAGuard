"""Optional second stage: rerank the first-stage candidates (clean + poison) and keep the top-k.

Qwen3-Reranker follows the model card's reference usage: a causal LM scores P("yes") for the given query/doc pair.
NOTE: weights are not downloaded yet. This path is covered by a CPU test double only and gets exercised in Step 3.
"""
from __future__ import annotations

from typing import Protocol

from .merge import Passage

QWEN3_RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_QWEN3_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
                 "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Reranker(Protocol):
    name: str

    def scores(self, query: str, docs: list[str]) -> list[float]: ...


class Qwen3Reranker:
    def __init__(self, hf_id: str, revision: str | None = None, device: str = "cuda", batch_size: int = 16,
                 max_length: int = 2048, instruction: str = QWEN3_RERANK_INSTRUCTION):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name, self.device, self.batch_size, self.max_length, self.instruction = hf_id, device, batch_size, max_length, instruction
        self.tok = AutoTokenizer.from_pretrained(hf_id, revision=revision, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision, torch_dtype=torch.bfloat16,
                                                          attn_implementation="sdpa").to(device).eval()
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")
        self._pre = self.tok.encode(_QWEN3_PREFIX, add_special_tokens=False)
        self._suf = self.tok.encode(_QWEN3_SUFFIX, add_special_tokens=False)

    def scores(self, query, docs):
        import torch

        out = []
        body_len = self.max_length - len(self._pre) - len(self._suf)
        for i in range(0, len(docs), self.batch_size):
            pairs = [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {d}" for d in docs[i:i + self.batch_size]]
            enc = self.tok(pairs, padding=False, truncation="longest_first", return_attention_mask=False, max_length=body_len)
            enc["input_ids"] = [self._pre + ids + self._suf for ids in enc["input_ids"]]
            batch = self.tok.pad(enc, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**batch).logits[:, -1, :]
            two = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out += torch.log_softmax(two, dim=1)[:, 1].exp().tolist()
        return out


class CrossEncoderReranker:
    """bge-reranker-v2-m3 and similar sentence-transformers CrossEncoders."""

    def __init__(self, hf_id: str, revision: str | None = None, device: str = "cuda", batch_size: int = 16, max_length: int = 1024):
        from sentence_transformers import CrossEncoder

        self.name, self.batch_size = hf_id, batch_size
        self._m = CrossEncoder(hf_id, revision=revision, device=device, max_length=max_length)

    def scores(self, query, docs):
        return [float(s) for s in self._m.predict([(query, d) for d in docs], batch_size=self.batch_size)]


class OverlapReranker:
    """CPU test double: fraction of query words present in the doc."""
    name = "overlap"

    def scores(self, query, docs):
        q = set(query.lower().split())
        return [len(q & set(d.lower().split())) / max(1, len(q)) for d in docs]


def build_reranker(key: str, registry: dict, device: str = "cuda", batch_size: int = 16, max_length: int = 2048) -> Reranker:
    spec = registry["rerankers"][key]
    hf_id = spec.get("hf_id", "")
    if spec.get("kind") == "overlap":
        return OverlapReranker()
    if "Qwen3-Reranker" in hf_id:
        return Qwen3Reranker(hf_id, spec.get("revision"), device=device, batch_size=batch_size, max_length=max_length)
    return CrossEncoderReranker(hf_id, spec.get("revision"), device=device, batch_size=batch_size)


def rerank(reranker: Reranker, query: str, candidates: list[Passage], top_k: int) -> list[Passage]:
    """Stable sort by reranker score (ties keep first-stage order), then re-rank 1..k. Retriever scores stay in `score`."""
    s = reranker.scores(query, [p.text for p in candidates])
    order = sorted(range(len(candidates)), key=lambda i: -s[i])[:top_k]
    return [Passage(**{**candidates[i].to_dict(), "rerank_score": float(s[i]), "rank": r + 1}) for r, i in enumerate(order)]
