"""PoisonedRAG reproduction dashboard — Overview + run picker.

  uv run streamlit run app/Home.py --server.headless true --server.address 127.0.0.1 --server.port 8502
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st  # noqa: E402

from lib import EXP, fmt, normalized_stats, page, run_picker, runs  # noqa: E402

page("PoisonedRAG — original vs. modern RAG", "🧪")
st.markdown(
    "Reproduction and modernization of **PoisonedRAG** (Zou et al., USENIX Security 2025, "
    "[arXiv:2402.07867](https://arxiv.org/abs/2402.07867)). "
    "An attacker injects **N = 5** synthetic passages per target question into the knowledge base. "
    "The attack succeeds if the RAG system then answers with the attacker's chosen answer. "
    "Everything here is read from `experiments/`; the dashboard never runs models or attacks."
)

df = runs()
row = run_picker(df)
if row is None:
    st.info(f"No runs found in `{EXP}`. Run `uv run python -m prag.run --config configs/baseline_nq_smoke.yaml` first.")
    st.stop()

# --- headline: paper vs reproduction vs modern grid ---------------------------------------------------------
grid = df[df["matrix"]]
st.subheader("Headline")
c = st.columns(4)
c[0].metric("Runs recorded", len(df), help="Every entry in experiments/index.json, including smoke and check runs.")
paper = df[(df["retriever"] == "contriever") & (df["llm"] == "vicuna-7b-v1.3") & (df["n_questions"] == 100)]
if not paper.empty:
    c[1].metric("Paper stack ASR (Contriever × Vicuna-7B)", fmt(paper.iloc[-1]["asr"]), delta=f"paper: 0.88",
                delta_color="off", help="Full protocol: 10 iterations × 10 questions. Paper Table 1, NQ, black-box.")
if not grid.empty:
    lo = grid.loc[grid["asr"].idxmin()]
    hi = grid.loc[grid["asr"].idxmax()]
    c[2].metric("Lowest ASR in the grid (exact)", fmt(lo["asr"]), delta=f"{lo['setup']} × {lo['llm_name']}", delta_color="off")
    c[3].metric("Highest ASR in the grid (exact)", fmt(hi["asr"]), delta=f"{hi['setup']} × {hi['llm_name']}", delta_color="off")
st.caption("ASR = attack success rate, upstream's exact substring metric on the final answer. "
           "The Results page also shows a typography-normalized variant; it matters for gpt-oss-120b.")

# --- selected run ------------------------------------------------------------------------------------------
st.subheader("Selected run")
st.markdown(f"**{row.setup} × {row.llm_name}** · dataset `{row.dataset}` · defense `{row.defense}` · {row.protocol or f'{row.n_questions} questions'}")
if row.n_questions and row.n_questions < 100:
    st.info(f"Subset run ({row.n_questions} questions). Numbers are not comparable to the 100-question protocol.", icon="ℹ️")
ns = normalized_stats(row.path)
m = st.columns(6)
m[0].metric("ASR (exact)", fmt(row.asr), help="Share of questions whose final answer contains the attacker's target answer.")
m[1].metric("ASR (typography-normalized)", fmt(ns.get("asr_normalized")), help="Secondary metric, not upstream: NFKC, ASCII quotes, collapsed whitespace.")
m[2].metric("Poison precision", fmt(row.precision), help="Own poison passages in the top-k / k.")
m[3].metric("Poison recall", fmt(row.recall), help="Own poison passages in the top-k / N.")
m[4].metric("Clean accuracy", fmt(row.clean_accuracy), help="No poison: correct answer appears in the answer.")
m[5].metric("Empty final answers", fmt(ns.get("empty_final")), help="Reasoning truncated before an answer (counts as attack failure).")

# --- all runs ----------------------------------------------------------------------------------------------
st.subheader("All runs")
show = df.sort_values("created", ascending=False)[
    ["run_id", "status", "dataset", "setup", "llm_name", "defense", "n_questions", "asr", "precision", "recall", "f1",
     "clean_accuracy", "cost_usd", "created"]]
st.dataframe(show, hide_index=True, width="stretch",
             column_config={"asr": st.column_config.NumberColumn("ASR", format="%.2f"),
                            "precision": st.column_config.NumberColumn(format="%.2f"),
                            "recall": st.column_config.NumberColumn(format="%.2f"),
                            "f1": st.column_config.NumberColumn("F1", format="%.2f"),
                            "clean_accuracy": st.column_config.NumberColumn("clean acc", format="%.2f"),
                            "cost_usd": st.column_config.NumberColumn("cost $", format="%.2f"),
                            "setup": "retrieval", "llm_name": "LLM", "n_questions": "Qs"})
st.caption("Pages: **Architecture** (pipeline diagram) · **Datasets** · **Attack flow** (one question end to end) · "
           "**Results** (grid charts + download) · **Defenses**.")
