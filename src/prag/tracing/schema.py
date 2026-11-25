"""JSON Schemas for experiments/<run_id>/traces.jsonl (one record per target question) and summary.json.

Bump SCHEMA_VERSION on breaking changes. The dashboard keys off it.
"""
from __future__ import annotations

from jsonschema import Draft202012Validator

SCHEMA_VERSION = "1.0"

_num = {"type": "number"}
_str = {"type": "string"}
_nstr = {"type": ["string", "null"]}
_nnum = {"type": ["number", "null"]}

PASSAGE = {
    "type": "object",
    "required": ["text", "score", "rank", "is_poison", "doc_id", "poison_for"],
    "properties": {
        "text": _str, "score": _num, "rank": {"type": ["integer", "null"], "minimum": 1},
        "is_poison": {"type": "boolean"}, "doc_id": _nstr, "poison_for": _nstr, "rerank_score": _nnum,
    },
    "additionalProperties": False,
}

GENERATION = {
    "type": "object",
    "required": ["text", "prompt_tokens", "completion_tokens", "finish_reason", "latency_s", "cost_usd"],
    "properties": {
        "text": _str, "prompt_tokens": {"type": "integer", "minimum": 0}, "completion_tokens": {"type": "integer", "minimum": 0},
        "finish_reason": _nstr, "latency_s": {"type": "number", "minimum": 0}, "cost_usd": {"type": "number", "minimum": 0},
        "raw_text": _nstr, "reasoning": _nstr,
    },
    "additionalProperties": False,
}

TRACE = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "PoisonedRAG per-question trace",
    "type": "object",
    "required": ["schema_version", "run_id", "config", "dataset", "iteration", "question_id", "question",
                 "correct_answer", "target_answer", "poison_texts", "n_poison_injected_total", "clean_topk",
                 "retrieved_topk", "defense", "prompt_poisoned", "llm_input_poisoned", "output_poisoned",
                 "success", "precision", "recall", "n_own_poison_in_topk", "usage", "latency_s", "cost_usd"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "run_id": _str,
        "config": {"type": "object", "required": ["hash", "retriever", "llm", "top_k", "adv_per_query", "defense"]},
        "dataset": _str, "iteration": {"type": "integer", "minimum": 0},
        "question_id": _str, "question": _str, "correct_answer": _str, "target_answer": _str,
        "poison_texts": {"type": "array", "items": _str},
        "n_poison_injected_total": {"type": "integer", "minimum": 0},
        "clean_topk": {"type": "array", "items": PASSAGE},
        "first_stage": {"type": ["array", "null"], "items": PASSAGE},
        "retrieved_topk": {"type": "array", "items": PASSAGE},
        "defense": {"type": "object", "required": ["name"], "properties": {"name": _str, "info": {"type": "object"},
                                                                           "removed": {"type": "integer"}}},
        "prompt_poisoned": _str, "prompt_clean": _nstr,
        "llm_input_poisoned": _str,
        "output_poisoned": _str, "output_clean": _nstr,
        "success": {"type": "boolean"},
        "clean_correct": {"type": ["boolean", "null"]}, "clean_says_target": {"type": ["boolean", "null"]},
        "precision": {"type": "number", "minimum": 0, "maximum": 1},
        "recall": {"type": "number", "minimum": 0, "maximum": 1},
        "n_own_poison_in_topk": {"type": "integer", "minimum": 0},
        "usage": {"type": "object", "required": ["poisoned"],
                  "properties": {"poisoned": GENERATION, "clean": {"anyOf": [GENERATION, {"type": "null"}]}}},
        "latency_s": {"type": "object"},
        "cost_usd": {"type": "number", "minimum": 0},
    },
    "additionalProperties": False,
}

SUMMARY = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "PoisonedRAG run summary",
    "type": "object",
    "required": ["schema_version", "run_id", "kind", "protocol", "status", "config", "config_hash", "dataset",
                 "retriever", "reranker", "llm", "defense", "n_questions", "metrics", "clean_accuracy",
                 "total_cost_usd", "tokens", "timing_s", "created"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "status": {"enum": ["complete", "partial"]},
        "metrics": {"type": "object", "required": ["asr_mean", "precision_mean", "recall_mean", "f1_mean", "rounded",
                                                   "asr_per_iter", "ret_counts"]},
        "n_questions": {"type": "integer", "minimum": 0},
    },
}

_TRACE_V = Draft202012Validator(TRACE)
_SUMMARY_V = Draft202012Validator(SUMMARY)


def trace_errors(rec: dict) -> list[str]:
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in _TRACE_V.iter_errors(rec)]


def validate_trace(rec: dict) -> None:
    errs = trace_errors(rec)
    if errs:
        raise ValueError("trace failed schema: " + "; ".join(errs[:5]))


def validate_summary(rec: dict) -> None:
    errs = [e.message for e in _SUMMARY_V.iter_errors(rec)]
    if errs:
        raise ValueError("summary failed schema: " + "; ".join(errs[:5]))
