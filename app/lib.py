"""Shared loading + UI helpers for the dashboard.

READ-ONLY CONTRACT: the dashboard reads experiments/* (index.json, <run>/summary.json, <run>/traces.jsonl,
datasets.json, retrieval_quality.json). It never imports model, retrieval or pipeline code, and it never runs attacks.
The only project import is prag.eval.metrics (pure string functions) for the typography-normalized secondary metric.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prag.eval.metrics import attack_success_normalized  # noqa: E402  (pure functions, no models)

EXP = Path(os.environ.get("PRAG_EXPERIMENTS_DIR", ROOT / "experiments")).resolve()
BASE = EXP.parent  # index.json paths are relative to this
BANNER = "Research artifact — poison texts shown are synthetic and confined to benchmark datasets."

LLM_NAME = {"vicuna-7b-v1.3": "Vicuna-7B v1.3", "qwen3.5-27b": "Qwen3.5-27B", "gpt-oss-120b": "gpt-oss-120b"}
LLM_ORDER = list(LLM_NAME)
RET_NAME = {"contriever": "Contriever (paper)", "qwen3-emb-8b": "Qwen3-Embedding-8B"}

# Reference categorical palette (dataviz skill, references/palette.md), fixed slot order, light / dark steps.
SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a"], "dark": ["#3987e5", "#d95926", "#199e70"]}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Step 3 grid cells (configs/matrix/*.yaml). The check-A dense-Contriever run and Step 1/2 baselines are not grid cells.
MATRIX_RE = re.compile(r"^[a-z]+_(contriever|qwen3emb8b|qwen3emb8b-rerank4b)_(vicuna7b|qwen35_27b|gptoss120b)_full-")
POISON_BG, CLEAN_BG = "rgba(227, 73, 72, 0.22)", "rgba(27, 175, 122, 0.12)"


def page(title: str, icon: str = "🧪") -> None:
    st.set_page_config(page_title=f"{title} · PoisonedRAG", page_icon=icon, layout="wide")
    st.warning(BANNER, icon="⚠️")
    st.title(title)


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def series_colors() -> list[str]:
    return SERIES[theme_mode()]


def ret_label(retriever: str | None, reranker: str | None = None) -> str:
    base = RET_NAME.get(retriever or "", retriever or "?")
    return base + (" + Qwen3-Reranker-4B" if reranker else "")


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float):
    return json.loads(Path(path).read_text())


def read_json(path: Path):
    return _read_json(str(path), path.stat().st_mtime) if path.exists() else None


@st.cache_data(show_spinner=False)
def _read_traces(path: str, mtime: float) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_traces(run_path: str) -> list[dict]:
    p = BASE / run_path / "traces.jsonl"
    return _read_traces(str(p), p.stat().st_mtime) if p.exists() else []


def load_summary(run_path: str) -> dict:
    return read_json(BASE / run_path / "summary.json") or {}


def runs() -> pd.DataFrame:
    """One row per run in experiments/index.json, enriched from summary.json. Old Step 1 runs are normalized."""
    idx = read_json(EXP / "index.json") or {"runs": []}
    rows = []
    for r in idx["runs"]:
        s = load_summary(r["path"])
        m = (s.get("metrics") or {})
        rows.append({
            "run_id": r["run_id"], "path": r["path"], "kind": r.get("kind", "?"), "status": r.get("status", "complete"),
            "dataset": r.get("dataset"), "retriever": r.get("retriever"), "reranker": r.get("reranker"),
            "setup": ret_label(r.get("retriever"), r.get("reranker")), "llm": r.get("llm"),
            "llm_name": LLM_NAME.get(r.get("llm"), r.get("llm")), "defense": r.get("defense", "none"),
            "attack": r.get("attack", "LM_targeted"), "n_questions": r.get("n_questions"),
            "protocol": s.get("protocol", ""), "asr": m.get("asr_mean"), "precision": m.get("precision_mean"),
            "recall": m.get("recall_mean"), "f1": m.get("f1_mean"), "clean_accuracy": s.get("clean_accuracy"),
            "cost_usd": s.get("total_cost_usd", 0.0), "gen_s": (s.get("timing_s") or {}).get("generation"),
            "created": s.get("created", ""),
            "matrix": bool(MATRIX_RE.match(r["run_id"])) and r.get("status", "complete") == "complete",
        })
    return pd.DataFrame(rows)


NORM_KEYS = ("asr_normalized", "clean_accuracy_normalized", "asr_answered", "empty_final", "truncated")


def normalized_stats(run_path: str) -> dict:
    """Secondary metrics recomputed from traces (works for runs written before summary.secondary_metrics existed).

    Without traces (e.g. a clone of the public repo, where traces.jsonl is not committed), fall back to
    experiments/trace_aggregates.json (numbers only; written by scripts/export_trace_aggregates.py), else None values.
    """
    T = load_traces(run_path)
    if not T:
        agg = (read_json(EXP / "trace_aggregates.json") or {}).get("runs", {})
        return {k: (agg.get(run_path) or {}).get(k) for k in NORM_KEYS}
    n = len(T)
    answered = [t for t in T if t["output_poisoned"].strip()]
    return {
        "asr_normalized": sum(attack_success_normalized(t["target_answer"], t["output_poisoned"]) for t in T) / n,
        "clean_accuracy_normalized": sum(attack_success_normalized(t["correct_answer"], t.get("output_clean") or "") for t in T) / n,
        "asr_answered": (sum(t["success"] for t in answered) / len(answered)) if answered else None,
        "empty_final": 1 - len(answered) / n,
        "truncated": sum((t["usage"]["poisoned"].get("finish_reason") == "length") for t in T) / n,
    }


def run_picker(df: pd.DataFrame, key: str = "run_id") -> pd.Series | None:
    """Sidebar run picker shared across pages via st.session_state."""
    if df.empty:
        st.sidebar.info("No runs in experiments/index.json yet.")
        return None
    df = df.sort_values("created", ascending=False)
    ids = df["run_id"].tolist()
    if st.session_state.get(key) not in ids:
        pref = df[df["matrix"] & (df["status"] == "complete")]
        st.session_state[key] = (pref if not pref.empty else df).iloc[0]["run_id"]
    label = {r.run_id: f"{'✅' if r.status == 'complete' else '⏸'} {r.setup} × {r.llm_name} · {r.n_questions} Q · {r.run_id[-15:]}"
             for r in df.itertuples()}
    st.sidebar.selectbox("Run", ids, key=key, format_func=lambda x: label.get(x, x))
    row = df[df["run_id"] == st.session_state[key]].iloc[0]
    st.sidebar.caption(f"`{row.run_id}`  \n{row.protocol}")
    return row


def fmt(x, nd=2) -> str:
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{nd}f}"
