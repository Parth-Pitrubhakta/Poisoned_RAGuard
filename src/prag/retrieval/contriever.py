"""Contriever encoder, equivalent to upstream src/contriever_src/contriever.Contriever (BertModel without pooler,
masked mean pooling over last_hidden_state, no normalization). Upstream uses the same model for query and doc."""
from __future__ import annotations

import torch
from transformers import AutoTokenizer, BertModel

CONTRIEVER_REVISION = "2bd46a25019aeea091fd42d1f0fd4801675cf699"  # facebook/contriever, recorded 2026-11-19


class ContrieverEncoder:
    def __init__(self, hf_id: str = "facebook/contriever", revision: str | None = CONTRIEVER_REVISION,
                 device: str = "cuda", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        self.model = BertModel.from_pretrained(hf_id, revision=revision, add_pooling_layer=False,
                                               attn_implementation="eager").to(device).eval()
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        """Returns float32 [n, 768] on CPU. Padding is masked out, so batching does not change embeddings."""
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"]
            hidden = hidden.masked_fill(~mask[..., None].bool(), 0.0)
            out.append((hidden.sum(dim=1) / mask.sum(dim=1)[..., None]).float().cpu())
        return torch.cat(out)

    @staticmethod
    def dot(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """score_function='dot' (upstream torch.mm(adv_emb, query_emb.T))."""
        return d @ q.reshape(-1)
