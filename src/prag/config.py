"""Pydantic-validated run config + model registry (configs/models.yaml).

Paths in a config are relative to the project root unless absolute.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[2]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetCfg(_Strict):
    name: Literal["nq", "hotpotqa", "msmarco"]
    split: str = "test"
    dir: str
    beir_url: str | None = None


class TargetsCfg(_Strict):
    source: str                                  # upstream results/adv_targeted_results/<ds>.json
    start_index: int = Field(0, ge=0)
    num_questions: int = Field(..., ge=1)        # upstream M (questions per iteration)
    iterations: int = Field(1, ge=1)             # upstream repeat_times

    @property
    def total(self) -> int:
        return self.num_questions * self.iterations


class RetrieverCfg(_Strict):
    key: str                                     # configs/models.yaml retrievers.<key>
    score_function: Literal["dot", "cos_sim"] = "dot"
    mode: Literal["upstream_precomputed", "dense_index"]
    precomputed: str | None = None               # mode=upstream_precomputed
    index_dir: str | None = None                 # mode=dense_index (built by prag.build_index)
    batch_size: int = Field(64, ge=1)

    @model_validator(mode="after")
    def _mode_fields(self):
        if self.mode == "upstream_precomputed":
            if not self.precomputed:
                raise ValueError("retriever.mode=upstream_precomputed needs retriever.precomputed")
            if self.score_function != "dot":
                raise ValueError("upstream precomputed beir_results were computed with dot; use score_function: dot")
        if self.mode == "dense_index" and not self.index_dir:
            raise ValueError("retriever.mode=dense_index needs retriever.index_dir")
        return self


class RerankerCfg(_Strict):
    key: str                                     # configs/models.yaml rerankers.<key>
    first_stage_k: int = Field(20, ge=1)         # candidates (clean + poison, merged by retriever score) to rerank
    batch_size: int = Field(16, ge=1)
    max_length: int = Field(2048, ge=64)


class AttackCfg(_Strict):
    method: Literal["LM_targeted", "none"] = "LM_targeted"
    adv_per_query: int = Field(5, ge=1)          # N


class RagCfg(_Strict):
    top_k: int = Field(5, ge=1)
    prompt_id: Literal[4] = 4


class LLMCfg(_Strict):
    key: str                                     # configs/models.yaml llms.<key>
    temperature: float = Field(0.1, ge=0)
    do_sample: bool = True
    top_p: float = Field(1.0, gt=0, le=1)
    top_k: int = 50                              # HF generate implicit default
    max_new_tokens: int = Field(150, ge=1)
    max_model_len: int | None = Field(None, ge=256)  # override the registry value (e.g. knowledge expansion, k=50)
    repetition_penalty: float = 1.0
    seed: int = 100
    gpu: Literal["auto"] | int = "auto"
    min_free_gb: float = 30
    gpu_memory_utilization: float = Field(0.30, gt=0, le=0.95)
    tensor_parallel_size: int = Field(1, ge=1)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)  # e.g. {enable_thinking: false}
    output_parser: Literal["none", "think_tags", "harmony"] = "none"
    batch_questions: int = Field(50, ge=1)       # questions per generate call (poisoned+clean together) = checkpoint unit


class EvalCfg(_Strict):
    asr: Literal["clean_str_substring"] = "clean_str_substring"
    smoke_test_nonempty: bool = True
    clean_run: bool = True                       # also answer with the clean top-k (no attack) for traces


class DefenseCfg(_Strict):
    name: str = "none"
    params: dict[str, Any] = Field(default_factory=dict)


class RunConfig(_Strict):
    run_name: str = Field(..., pattern=r"^[A-Za-z0-9_.\-]+$")
    seed: int = 12
    dataset: DatasetCfg
    targets: TargetsCfg
    retriever: RetrieverCfg
    reranker: RerankerCfg | None = None
    attack: AttackCfg = AttackCfg()
    rag: RagCfg = RagCfg()
    llm: LLMCfg
    eval: EvalCfg = EvalCfg()
    defense: DefenseCfg = DefenseCfg()

    @field_validator("defense", mode="before")
    @classmethod
    def _defense_str(cls, v):
        return {"name": v} if isinstance(v, str) else v

    @model_validator(mode="after")
    def _reranker_k(self):
        if self.reranker and self.reranker.first_stage_k < self.rag.top_k:
            raise ValueError("reranker.first_stage_k must be >= rag.top_k")
        return self

    @model_validator(mode="after")
    def _defense_fit(self):
        d = self.defense
        if d.name == "paraphrase" and self.retriever.mode != "dense_index":
            raise ValueError("defense paraphrase changes the query, so it needs retriever.mode=dense_index "
                             "(upstream precomputed results exist only for the original queries)")
        if d.name == "knowledge_expansion" and d.params.get("k") != self.rag.top_k:
            raise ValueError("defense knowledge_expansion: params.k must equal rag.top_k")
        return self

    def path(self, p: str) -> Path:
        return resolve_path(p)


def force_offline() -> None:
    """Runs never download. vLLM's loader otherwise fetches extra repo files (it pulled gpt-oss-120b's 65 GB
    original/ checkpoint on 2026-11-19). Weights come only from scripts/download_model.py. Call before any HF import;
    vLLM's engine subprocess inherits the environment."""
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_registry(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or ROOT / "configs" / "models.yaml").read_text())


def check_registry(cfg: RunConfig, reg: dict) -> None:
    missing = []
    if cfg.retriever.key not in reg.get("retrievers", {}):
        missing.append(f"retrievers.{cfg.retriever.key}")
    if cfg.reranker and cfg.reranker.key not in reg.get("rerankers", {}):
        missing.append(f"rerankers.{cfg.reranker.key}")
    if cfg.llm.key not in reg.get("llms", {}):
        missing.append(f"llms.{cfg.llm.key}")
    if missing:
        raise KeyError(f"not in configs/models.yaml: {', '.join(missing)}")


def load_config(path: str | Path, registry: dict | None = None) -> tuple[RunConfig, str]:
    """Returns (validated config, raw YAML text). Raises on unknown keys or registry mismatches."""
    text = Path(path).read_text()
    cfg = RunConfig.model_validate(yaml.safe_load(text))
    check_registry(cfg, registry if registry is not None else load_registry())
    return cfg, text


def config_hash(cfg: RunConfig) -> str:
    """Hash of the *validated* config (defaults filled), so comments/whitespace don't change it."""
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:12]
