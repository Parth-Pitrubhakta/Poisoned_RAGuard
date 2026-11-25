import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st  # noqa: E402

from lib import EXP, load_summary, page, read_json, run_picker, runs  # noqa: E402

page("Architecture", "🧭")
row = run_picker(runs())
if row is None:
    st.stop()
s = load_summary(row.path)
cfg = s.get("config", {})
retr = cfg.get("retriever", {})
llm = s.get("llm", {}) or {}
rr = cfg.get("reranker")
k = cfg.get("rag", {}).get("top_k", 5)
N = cfg.get("attack", {}).get("adv_per_query", 5)
M = cfg.get("targets", {}).get("num_questions", 10)

card = (read_json(EXP / "datasets.json") or {}).get(row.dataset, {})
n_corpus = f"{card['corpus_passages']:,} passages" if card else "BEIR corpus"
POISON, CLEAN, MODEL, NEUTRAL = "#e34948", "#1baf7a", "#2a78d6", "#8a8984"
mode = retr.get("mode", "upstream_precomputed")
retr_label = (f"{row.setup}\\n{'upstream precomputed top-100' if mode == 'upstream_precomputed' else 'FAISS IndexFlatIP (exact)'}"
              f"\\nscore: {retr.get('score_function', 'dot')}")
rr_label = f"Reranker\\nQwen3-Reranker-4B\\nfirst-stage top-{rr.get('first_stage_k')} → top-{k}" if rr else "Reranker\\n(none)"
llm_label = f"Target LLM\\n{row.llm_name}\\nT={llm.get('temperature', '?')}, max {llm.get('max_new_tokens', '?')} tok"
if llm.get("output_parser") not in (None, "none"):
    llm_label += f"\\nanswer split: {llm['output_parser']}"

dot = f"""
digraph G {{
  rankdir=LR; bgcolor="transparent"; fontname="Helvetica"; nodesep=0.35; ranksep=0.45;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, penwidth=1.4, fontcolor="#0b0b0b"];
  edge [color="{NEUTRAL}", fontname="Helvetica", fontsize=9, fontcolor="{NEUTRAL}", penwidth=1.2];

  subgraph cluster_attacker {{ label="Attacker (black-box, LM_targeted)"; fontcolor="{POISON}"; color="{POISON}"; style="dashed,rounded";
    tq [label="Target question\\n+ attacker's target answer", fillcolor="#fde8e8", color="{POISON}"];
    adv [label="Poison texts\\nquestion + '.' + adv_text\\nN={N} per question, M×N={M*N} per iteration", fillcolor="#fde8e8", color="{POISON}"];
    tq -> adv;
  }}
  subgraph cluster_kb {{ label="Knowledge base (BEIR {row.dataset}, local copy)"; fontcolor="{CLEAN}"; color="{CLEAN}"; style="rounded";
    corpus [label="Clean corpus\\n{n_corpus}", fillcolor="#e3f6ee", color="{CLEAN}"];
  }}
  q [label="User query", fillcolor="#f0efec", color="{NEUTRAL}"];
  ret [label="{retr_label}", fillcolor="#e1ecfa", color="{MODEL}"];
  merge [label="Merge clean top-n + every poison\\nsort by score, keep top", fillcolor="#f0efec", color="{NEUTRAL}"];
  rr [label="{rr_label}", fillcolor="{'#e1ecfa' if rr else '#f7f7f5'}", color="{MODEL if rr else NEUTRAL}", style="{'rounded,filled' if rr else 'rounded,filled,dashed'}"];
  def [label="Defense\\n{row.defense}", fillcolor="#f7f7f5", color="{NEUTRAL}", style="rounded,filled,dashed"];
  prompt [label="MULTIPLE_PROMPT (id 4)\\ntop-{k} contexts + query", fillcolor="#f0efec", color="{NEUTRAL}"];
  llm [label="{llm_label}", fillcolor="#e1ecfa", color="{MODEL}"];
  ev [label="Eval\\nASR: target ⊂ answer\\nP / R / F1 of own poison", fillcolor="#f0efec", color="{NEUTRAL}"];

  adv -> merge [label="injected", color="{POISON}", fontcolor="{POISON}", penwidth=2];
  corpus -> ret [label="clean passages", color="{CLEAN}", fontcolor="{CLEAN}"];
  q -> ret; ret -> merge [label="clean top-n"]; merge -> rr -> def -> prompt -> llm -> ev;
  tq -> q [style=dotted, label="same question"];
}}
"""
st.graphviz_chart(dot, width="stretch")
st.markdown(
    f"""
**Legend.** <span style="color:{POISON}">■ red = attacker / poison</span> ·
<span style="color:{CLEAN}">■ green = clean knowledge</span> ·
<span style="color:{MODEL}">■ blue = models in this run</span> · gray = deterministic glue; dashed = inactive in this run.

**Faithful to upstream `main.py`.**
- Every poison text of the iteration is scored against each query by the same retriever. It is merged with the clean
  top-n and stably sorted.
- P/R count only the question's *own* poison.
- With a reranker, the merged top-{rr.get('first_stage_k') if rr else 'n'} is reranked and the top-{k} is kept.
- The defense sees the passages but never the `is_poison` flag.
""", unsafe_allow_html=True)
with st.expander("Run config (from summary.json)"):
    st.json(cfg)
