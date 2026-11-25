import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from lib import EXP, load_traces, page, read_json, run_picker, runs, series_colors  # noqa: E402

page("Datasets", "📚")
row = run_picker(runs())
cards = read_json(EXP / "datasets.json") or {}
quality = read_json(EXP / "retrieval_quality.json") or {}
if not cards:
    st.info("No `experiments/datasets.json`. Create it with `uv run python scripts/make_dataset_card.py --dataset nq`.")

for name, c in cards.items():
    st.subheader(f"BEIR {name.upper()} ({c['split']})")
    cols = st.columns(4)
    cols[0].metric("Corpus passages", f"{c['corpus_passages']:,}")
    cols[1].metric("Test queries", f"{c['queries']:,}")
    cols[2].metric("Attack target questions", c["target_questions"], help=c["targets_source"])
    cols[3].metric("Poison texts per target", c["adv_texts_per_target"])
    st.markdown(
        f"- Passages: mean {c['passage_words']['mean']} words, median {c['passage_words']['median']} (1% sample). "
        f"{c['passages_with_title_share']:.0%} have a Wikipedia title.\n"
        f"- Released poison texts (`adv_text`): mean {c['adv_text_words']['mean']} words. Each is prefixed with the question "
        f"(`question + '.' + adv_text`).\n"
        f"- qrels (test): {c['qrels_test_pairs']:,} relevant pairs for {c['qrels_test_queries']:,} queries.\n"
        f"- Source: `{c.get('beir_url', '?')}` · zip sha256 `{c.get('zip_sha256', '?')[:16]}…`")

    q = quality.get(name)
    if q:
        st.markdown(f"**Clean retrieval quality (no poison, all {q['n_queries']:,} test queries)**")
        rets = list(q["retrievers"])
        metrics = ["recall@5", "recall@20", "recall@100", "ndcg@10"]
        colors = series_colors()
        fig = go.Figure()
        for i, r in enumerate(rets):
            fig.add_bar(name={"qwen3-emb-8b": "Qwen3-Embedding-8B", "contriever": "Contriever"}.get(r, r), x=metrics,
                        y=[q["retrievers"][r][m] for m in metrics], marker_color=colors[i % len(colors)],
                        marker_line_width=0, hovertemplate="%{x}: %{y:.3f}<extra>%{fullData.name}</extra>",
                        text=[f"{q['retrievers'][r][m]:.2f}" for m in metrics], textposition="outside")
        fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, height=330, yaxis_range=[0, 1.08],
                          yaxis_title=None, margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", y=1.12),
                          yaxis=dict(gridcolor="rgba(128,128,128,0.15)"))
        st.plotly_chart(fig, width="stretch", theme="streamlit")
        st.caption("Contriever's nDCG@10 matches BEIR's published unsupervised-Contriever NQ score (0.254). "
                   "Clean retrieval is much better with the modern embedder; the Results page shows that this barely changes poison retrieval.")

if row is not None:
    T = load_traces(row.path)
    if T:
        st.subheader("Target questions in the selected run")
        tq = pd.DataFrame([{"iteration": t.get("iteration", 0), "id": t["question_id"], "question": t["question"],
                            "correct answer": t["correct_answer"], "attacker target": t["target_answer"]} for t in T])
        st.dataframe(tq, hide_index=True, width="stretch", height=360)
        st.caption("Questions and target answers come from upstream's released `adv_targeted_results`; this project generated no new ones.")
