import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from lib import LLM_NAME, LLM_ORDER, SEQ_BLUE, load_summary, normalized_stats, page, run_picker, runs, series_colors  # noqa: E402

page("Results", "📊")
df = runs()
run_picker(df)
if df.empty:
    st.stop()

SETUP_ORDER = ["Contriever (paper)", "Qwen3-Embedding-8B", "Qwen3-Embedding-8B + Qwen3-Reranker-4B"]
GRID = dict(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
LAYOUT = dict(margin=dict(t=30, b=10, l=10, r=10), legend=dict(orientation="h", y=1.15, x=0), hoverlabel=dict(namelength=-1))

# ---- filters (one row above the charts) ---------------------------------------------------------------------
f = st.columns([1, 2, 2])
ds = f[0].selectbox("Dataset", sorted(df["dataset"].dropna().unique()))
metric = f[1].radio("ASR metric", ["exact (upstream)", "typography-normalized", "answered only"], horizontal=True,
                    help="exact = upstream clean_str substring. normalized = NFKC + ASCII quotes + collapsed whitespace (secondary). "
                         "answered only = exact, restricted to questions with a non-empty final answer.")
scope = f[2].radio("Runs", ["Step 3 grid (full protocol)", "all complete runs"], horizontal=True)

d = df[(df["dataset"] == ds) & (df["status"] == "complete")]
if scope.startswith("Step 3"):
    d = d[d["matrix"]]
if d.empty:
    st.info("No complete runs for this selection.")
    st.stop()
d = d.sort_values("created").drop_duplicates(["setup", "llm"], keep="last").copy()  # latest per cell
extra = pd.DataFrame([normalized_stats(p) for p in d["path"]], index=d.index)
d = pd.concat([d, extra], axis=1)
d["asr_view"] = {"exact (upstream)": d["asr"], "typography-normalized": d["asr_normalized"], "answered only": d["asr_answered"]}[metric]
d["per_iter"] = [(load_summary(p).get("metrics") or {}).get("asr_per_iter", []) for p in d["path"]]
setups = [s for s in SETUP_ORDER if s in set(d["setup"])] + sorted(set(d["setup"]) - set(SETUP_ORDER))
llms = [l for l in LLM_ORDER if l in set(d["llm"])] + sorted(set(d["llm"]) - set(LLM_ORDER))
colors = series_colors()
cell = lambda s, l, col: (d[(d["setup"] == s) & (d["llm"] == l)][col].tolist() or [None])[0]  # noqa: E731

# ---- 1. heatmap -----------------------------------------------------------------------------------------------
st.subheader(f"ASR by retrieval setup × LLM — {metric}")
z = [[cell(s, l, "asr_view") for l in llms] for s in setups]
fig = go.Figure(go.Heatmap(
    z=z, x=[LLM_NAME.get(l, l) for l in llms], y=setups, zmin=0, zmax=1, xgap=2, ygap=2,
    colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)], colorbar=dict(title="ASR", thickness=10),
    text=[["—" if v is None else f"{v:.2f}" for v in r] for r in z], texttemplate="%{text}", textfont=dict(size=15),
    hovertemplate="%{y} × %{x}<br>ASR %{z:.2f}<extra></extra>"))
fig.update_layout(height=120 + 70 * len(setups), yaxis=dict(autorange="reversed"), **LAYOUT)
st.plotly_chart(fig, width="stretch", theme="streamlit")
st.caption("n = 100 questions per cell → 1 SE ≈ ±0.04. Paired McNemar tests per step are not significant on NQ alone (see experiments/RESULTS.md).")

# ---- 2. grouped bars: retrieval effect per LLM ---------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.markdown("**ASR per LLM, by retrieval setup**")
    fig = go.Figure()
    for i, s in enumerate(setups):
        ys = [cell(s, l, "asr_view") for l in llms]
        fig.add_bar(name=s, x=[LLM_NAME.get(l, l) for l in llms], y=ys, marker_color=colors[i % len(colors)], marker_line_width=0,
                    text=[None if v is None else f"{v:.2f}" for v in ys], textposition="outside",
                    hovertemplate="%{x}<br>" + s + ": %{y:.2f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06, height=380, yaxis=dict(range=[0, 1.1], **GRID), **LAYOUT)
    st.plotly_chart(fig, width="stretch", theme="streamlit")

# ---- 3. dumbbell: exact vs normalized ---------------------------------------------------------------------
with right:
    st.markdown("**Exact vs typography-normalized ASR** (the gap is a scoring artifact)")
    fig = go.Figure()
    labels = [f"{LLM_NAME.get(l, l)} · {s.replace('Qwen3-Embedding-8B', 'Q3-Emb-8B').replace(' + Qwen3-Reranker-4B', ' + rerank').replace(' (paper)', '')}"
              for l in llms for s in setups if cell(s, l, "asr") is not None]
    ex = [cell(s, l, "asr") for l in llms for s in setups if cell(s, l, "asr") is not None]
    nm = [cell(s, l, "asr_normalized") for l in llms for s in setups if cell(s, l, "asr") is not None]
    for lab, a, b in zip(labels, ex, nm):
        fig.add_scatter(x=[a, b], y=[lab, lab], mode="lines", line=dict(color="rgba(128,128,128,0.45)", width=2),
                        showlegend=False, hoverinfo="skip")
    fig.add_scatter(name="exact (upstream)", x=ex, y=labels, mode="markers", marker=dict(size=11, color=colors[0], line=dict(width=2, color="white")),
                    hovertemplate="%{y}<br>exact %{x:.2f}<extra></extra>")
    fig.add_scatter(name="typography-normalized", x=nm, y=labels, mode="markers", marker=dict(size=11, color=colors[1], line=dict(width=2, color="white")),
                    hovertemplate="%{y}<br>normalized %{x:.2f}<extra></extra>")
    fig.update_layout(height=380, xaxis=dict(range=[0, 1.02], title="ASR", **GRID), yaxis=dict(autorange="reversed"), **LAYOUT)
    st.plotly_chart(fig, width="stretch", theme="streamlit")

# ---- 4. clean accuracy + poison retrieval ------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.markdown("**Clean accuracy (no poison)** — exact metric")
    fig = go.Figure()
    for i, s in enumerate(setups):
        ys = [cell(s, l, "clean_accuracy") for l in llms]
        fig.add_bar(name=s, x=[LLM_NAME.get(l, l) for l in llms], y=ys, marker_color=colors[i % len(colors)], marker_line_width=0,
                    text=[None if v is None else f"{v:.2f}" for v in ys], textposition="outside",
                    hovertemplate="%{x}<br>" + s + ": %{y:.2f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06, height=360, yaxis=dict(range=[0, 1.1], **GRID), **LAYOUT)
    st.plotly_chart(fig, width="stretch", theme="streamlit")
with right:
    st.markdown("**Poison retrieval F1** (LLM-independent; P = R = F1 here)")
    f1s = [d[d["setup"] == s]["f1"].iloc[0] for s in setups]
    fig = go.Figure(go.Bar(x=setups, y=f1s, marker_color=colors[0], marker_line_width=0, text=[f"{v:.2f}" for v in f1s],
                           textposition="outside", hovertemplate="%{x}<br>F1 %{y:.2f}<extra></extra>"))
    fig.update_layout(height=360, yaxis=dict(range=[0, 1.1], **GRID), bargap=0.45, **LAYOUT)
    st.plotly_chart(fig, width="stretch", theme="streamlit")
st.caption("Clean retrieval improves a lot with the modern embedder (Datasets page), yet 5 poison passages still take ≈4.5–4.8 of the 5 slots.")

# ---- 5. per-iteration spread -------------------------------------------------------------------------------
st.markdown("**ASR per iteration** (10 iterations × 10 questions; exact metric) — shows the spread behind each mean")
fig = go.Figure()
for i, s in enumerate(setups):
    xs, ys = [], []
    for l in llms:
        for v in (cell(s, l, "per_iter") or []):
            xs.append(LLM_NAME.get(l, l)); ys.append(v)
    fig.add_box(name=s, x=xs, y=ys, marker_color=colors[i % len(colors)], boxpoints="all", jitter=0.4, pointpos=0,
                marker=dict(size=8, opacity=0.75, line=dict(width=1, color="white")), line=dict(width=2),
                hovertemplate="%{x}<br>" + s + "<br>iteration ASR %{y:.1f}<extra></extra>")
fig.update_layout(boxmode="group", height=380, yaxis=dict(range=[-0.05, 1.05], title="ASR", **GRID), **LAYOUT)
st.plotly_chart(fig, width="stretch", theme="streamlit")

# ---- table + download --------------------------------------------------------------------------------------
st.subheader("Table")
cols = ["setup", "llm_name", "n_questions", "asr", "asr_normalized", "asr_answered", "precision", "recall", "f1",
        "clean_accuracy", "clean_accuracy_normalized", "empty_final", "truncated", "gen_s", "cost_usd", "run_id"]
tbl = d[cols].rename(columns={"setup": "retrieval", "llm_name": "LLM", "n_questions": "Qs", "gen_s": "generation s"})
st.dataframe(tbl, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
             ["asr", "asr_normalized", "asr_answered", "precision", "recall", "f1", "clean_accuracy", "clean_accuracy_normalized",
              "empty_final", "truncated", "cost_usd"]})
st.download_button("Download CSV", tbl.to_csv(index=False).encode(), file_name=f"poisonedrag_results_{ds}.csv", mime="text/csv")
