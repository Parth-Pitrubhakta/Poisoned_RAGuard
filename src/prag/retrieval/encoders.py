"""Dense encoders behind one interface.

    enc.encode_queries(list[str]) -> float32 np.ndarray [n, dim]
    enc.encode_docs(list[str])    -> float32 np.ndarray [n, dim]

Poison texts are always encoded with `encode_docs` (they are inserted into the knowledge base as documents).
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    name: str
    dim: int

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...
    def encode_docs(self, texts: list[str]) -> np.ndarray: ...


class ContrieverDense:
    """Paper retriever. Same model for queries and docs, mean pooling, no normalization."""

    def __init__(self, hf_id: str, revision: str | None, device: str = "cuda", batch_size: int = 64):
        from .contriever import ContrieverEncoder

        self._enc = ContrieverEncoder(hf_id=hf_id, revision=revision, device=device)
        self.name, self.dim, self.batch_size = hf_id, 768, batch_size

    def encode_queries(self, texts):
        return self._enc.encode(texts, batch_size=self.batch_size).numpy()

    encode_docs = encode_queries


class SentenceTransformerDense:
    """Qwen3-Embedding-* and BGE-M3 through sentence-transformers.

    `encode_query` applies the model's own "query" prompt from its sentence-transformers config. For Qwen3-Embedding
    that is "Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery:".
    Documents get no prompt. Both models L2-normalize, so dot == cos_sim.
    """

    def __init__(self, hf_id: str, revision: str | None, device: str = "cuda", batch_size: int = 64,
                 dtype: str = "bfloat16", max_seq_length: int = 512):
        import torch
        from sentence_transformers import SentenceTransformer

        kw = {"torch_dtype": getattr(torch, dtype), "attn_implementation": "sdpa"}
        tok_kw = {"padding_side": "left"} if "Qwen3-Embedding" in hf_id else {}
        self._m = SentenceTransformer(hf_id, revision=revision, device=device, model_kwargs=kw, tokenizer_kwargs=tok_kw)
        self._m.max_seq_length = max_seq_length
        self.name, self.batch_size = hf_id, batch_size
        self.dim = self._m.get_sentence_embedding_dimension()

    def encode_queries(self, texts):
        return np.asarray(self._m.encode_query(texts, batch_size=self.batch_size, convert_to_numpy=True), dtype=np.float32)

    def encode_docs(self, texts):
        return np.asarray(self._m.encode_document(texts, batch_size=self.batch_size, convert_to_numpy=True), dtype=np.float32)


class SimCSECLS:
    """TrustRAG's embedder (reference defend_module.get_sentence_embedding): tokenizer(sentence, truncation=True,
    padding=True), model(..., output_hidden_states=True).hidden_states[-1][:, 0, :], fp32, one sentence at a time."""

    def __init__(self, hf_id: str, revision: str | None, device: str = "cuda", batch_size: int = 1):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device, self.batch_size = torch, device, batch_size
        self.tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        self.model = AutoModel.from_pretrained(hf_id, revision=revision).to(device).eval()
        self.name, self.dim = hf_id, self.model.config.hidden_size
        self.max_length = min(self.tok.model_max_length, self.model.config.max_position_embeddings)

    def encode_docs(self, texts):
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i:i + self.batch_size], return_tensors="pt", truncation=True, padding=True,
                           max_length=self.max_length).to(self.device)
            with self.torch.no_grad():
                h = self.model(**enc, output_hidden_states=True, return_dict=True).hidden_states[-1][:, 0, :]
            out.append(h.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    encode_queries = encode_docs


class HashEncoder:
    """CPU test double: bag-of-words hashed into `dim` buckets. Deterministic, no weights."""

    def __init__(self, dim: int = 64):
        self.name, self.dim = "hash", dim

    def _one(self, t: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in t.lower().replace(".", " ").replace("?", " ").split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def encode_queries(self, texts):
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    encode_docs = encode_queries


def build_encoder(key: str, registry: dict, device: str = "cuda", batch_size: int = 64) -> Encoder:
    spec = registry["retrievers"][key]
    kind = spec["kind"]
    if kind == "contriever_meanpool":
        return ContrieverDense(spec["hf_id"], spec.get("revision"), device=device, batch_size=batch_size)
    if kind in ("qwen3_embedding", "sentence_transformer"):
        return SentenceTransformerDense(spec["hf_id"], spec.get("revision"), device=device, batch_size=batch_size)
    if kind == "simcse_cls":
        return SimCSECLS(spec["hf_id"], spec.get("revision"), device=device, batch_size=1)
    if kind == "hash":
        return HashEncoder(spec.get("dim", 64))
    raise ValueError(f"unknown retriever kind {kind!r}")


def score(q: np.ndarray, d: np.ndarray, fn: str) -> np.ndarray:
    """q [dim], d [n, dim] -> [n]. 'dot' = upstream torch.mm; 'cos_sim' = torch.cosine_similarity."""
    s = d @ q
    if fn == "cos_sim":
        s = s / (np.linalg.norm(d, axis=1) * np.linalg.norm(q) + 1e-12)
    return s
