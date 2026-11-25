import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from lib import EXP, LLM_NAME, LLM_ORDER, page, read_json, run_picker, runs, series_colors  # noqa: E402

page("Defenses", "🛡️")
df = runs()
run_picker(df)
D = read_json(EXP / "defenses" / "defense_table.json")
if not D or not D.get("rows"):
    st.info("No defense table yet. Run the Step 5 configs, then `uv run python scripts/make_defense_results.py` "
            "(writes `experiments/defenses/defense_table.json`).", icon="ℹ️")
    st.stop()

R = pd.DataFrame(D["rows"])
NAMES, NOISE = D["names"], D.get("noise_floor", {})
SETUPS = ["Contriever (paper)", "Qwen3-Emb-8B", "Qwen3-Emb-8B + rerank"]
NEUTRAL, BAND = "#8a8984", "rgba(138, 137, 132, 0.22)"
colors = series_colors()
GRID = dict(gridcolor="rgba(128,128,128,0.15)", zeroline=False)

st.markdown(
    "Each defended run repeats one Step 3 cell (same 100 NQ targets, same poison, same retrieval × LLM, T = 0.1). Only the "
    "defense changes, and the run is compared with its undefended Step 3 **twin**. A defense filters the poisoned *and* the clean "
    "(no-attack) top-k, so the clean-accuracy change is its utility cost. It never sees the `is_poison` flag.")

# ---- headline -------------------------------------------------------------------------------------------------
tr = R[R["defense"] == "trustrag"]
head = st.columns(4)
if not tr.empty:
    head[0].metric("TrustRAG (SimCSE) — robust ASR", f"{tr['robust_def'].mean():.2f}",
                   f"{tr['robust_def'].mean() - tr['robust_none'].mean():+.2f} vs none ({tr['robust_none'].mean():.2f})",
                   delta_color="inverse", help="Mean over all 9 cells; robust metric per LLM (see the metric selector).")
    tr_v = tr[tr["llm"] == "vicuna-7b-v1.3"]
    head[1].metric("TrustRAG — clean accuracy change", f"{tr['d_clean'].mean():+.2f}",
                   f"Vicuna-7B {tr_v['d_clean'].mean():+.2f}" if not tr_v.empty else None,
                   help="Mean over all 9 cells. The mean hides a split: Vicuna-7B loses clean accuracy, and the reasoning models' "
                        "rise is length-inflated, not a real gain (see caveats).")
ke = R[(R["defense"] == "knowledge_expansion@50") & (R["llm"] == "gpt-oss-120b")]
if not ke.empty:
    head[2].metric("Best paper defense: k = 50 (gpt-oss)", f"{ke['robust_def'].mean():.2f}",
                   f"{ke['robust_def'].mean() - ke['robust_none'].mean():+.2f} vs none", delta_color="inverse",
                   help="Knowledge expansion, typography-normalized ASR, mean over 3 retrieval setups.")
nf = max((v["max_discordant"] for k, v in NOISE.items() if k != "vicuna-7b-v1.3"), default=None)
head[3].metric("Noise floor (reasoning LLMs)", "±0.05", f"≤ {nf} of 100 answers flip on identical prompts" if nf is not None else None,
               delta_color="off", help="Measured with the duplicate-filter runs (≤ 1 prompt changed per 100). Vicuna is exactly reproducible.")

# ---- filters (one row) ----------------------------------------------------------------------------------------
f = st.columns([2, 1.3, 1.2])
METRICS = {"robust (recommended)": ("robust_none", "robust_def"), "exact (upstream)": ("asr_none", "asr_def"),
           "typography-normalized": ("asr_norm_none", "asr_norm_def"), "answered only": ("asr_answered_none", "asr_answered_def")}
metric = f[0].radio("ASR metric", list(METRICS), horizontal=True, key="def_metric",
                    help="robust = exact for Vicuna, answered-only for Qwen3.5 (truncated reasoning would otherwise score as 'failed'), "
                         "typography-normalized for gpt-oss (U+202F spaces hide exact matches).")
setup = f[1].selectbox("Retrieval setup", ["mean over setups"] + SETUPS, key="def_setup")
show_sub = f[2].toggle("Show TrustRAG with Contriever substitute", value=False, key="def_sub",
                       help="The first Step 5 run used local Contriever instead of TrustRAG's SimCSE embedder.")

order = [d for d in D["order"] if d in set(R["defense"]) and (show_sub or d != "trustrag@contriever")]
V = R if setup == "mean over setups" else R[R["retrieval"] == setup]
c0, c1 = METRICS[metric]


def agg(col: str, dfn: str, llm: str):
    x = V[(V["defense"] == dfn) & (V["llm"] == llm)][col].dropna()
    return float(x.mean()) if len(x) else None


def label(d: str) -> str:
    return NAMES.get(d, d).replace("Knowledge expansion", "Knowl. expansion")


ylab = [label(d) for d in order][::-1]  # first defense at the top


def dumbbell(before: str, after: str, title: str, xtitle: str):
    llms = [l for l in LLM_ORDER if l in set(R["llm"])]
    fig = make_subplots(rows=1, cols=len(llms), shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[LLM_NAME.get(l, l) for l in llms])
    for j, l in enumerate(llms, 1):
        noise = 0.05 if l != "vicuna-7b-v1.3" else 0.0
        xs0 = [agg(before, d, l) for d in order][::-1]
        xs1 = [agg(after, d, l) for d in order][::-1]
        for y, a, b in zip(ylab, xs0, xs1):
            if a is None or b is None:
                continue
            if noise:  # ±0.05 noise band around the undefended value
                fig.add_scatter(x=[a - noise, a + noise], y=[y, y], mode="lines", line=dict(color=BAND, width=14),
                                hoverinfo="skip", showlegend=False, row=1, col=j)
            fig.add_scatter(x=[a, b], y=[y, y], mode="lines", line=dict(color="rgba(128,128,128,0.5)", width=2),
                            hoverinfo="skip", showlegend=False, row=1, col=j)
        ok = [(y, a, b) for y, a, b in zip(ylab, xs0, xs1) if a is not None and b is not None]
        na = [y for y, a, b in zip(ylab, xs0, xs1) if a is None or b is None]
        if ok:
            fig.add_scatter(x=[a for _, a, _ in ok], y=[y for y, _, _ in ok], mode="markers", name="none (Step 3 twin)",
                            marker=dict(size=10, color=NEUTRAL, line=dict(width=2, color="white")), showlegend=j == 1,
                            hovertemplate="%{y}<br>none: %{x:.2f}<extra>" + LLM_NAME.get(l, l) + "</extra>", row=1, col=j)
            fig.add_scatter(x=[b for _, _, b in ok], y=[y for y, _, _ in ok], mode="markers+text", name="defended",
                            text=[f"{b:.2f}" for _, _, b in ok], textposition="top center", textfont=dict(size=10),
                            marker=dict(size=11, color=colors[0], line=dict(width=2, color="white")), showlegend=j == 1,
                            hovertemplate="%{y}<br>defended: %{x:.2f}<extra>" + LLM_NAME.get(l, l) + "</extra>", row=1, col=j)
        for y in na:
            fig.add_annotation(x=0.5, y=y, text="N/A (2048-token context)" if l == "vicuna-7b-v1.3" else "not run",
                               showarrow=False, font=dict(size=10, color=NEUTRAL), row=1, col=j)
        fig.update_xaxes(range=[-0.02, 1.02], title_text=xtitle if j == 2 else None, **GRID, row=1, col=j)
    fig.update_yaxes(categoryorder="array", categoryarray=ylab)
    fig.update_layout(height=110 + 48 * len(order), margin=dict(t=50, b=10, l=10, r=10), hoverlabel=dict(namelength=-1),
                      legend=dict(orientation="h", y=1.18, x=0), title=dict(text=title, x=0, y=0.995, font=dict(size=15)))
    return fig


# ---- 1. ASR before / after ----------------------------------------------------------------------------------
st.subheader("Attack success: undefended → defended, per LLM")
st.plotly_chart(dumbbell(c0, c1, f"ASR ({metric}); {setup}", "ASR"), width="stretch", theme="streamlit")
st.caption("Gray band = ±0.05 run-to-run noise around the undefended value for the reasoning models (measured: identical prompts "
           f"flip up to {nf} of 100 Qwen3.5 / gpt-oss answers). Movement inside the band is not a defense effect. n = 100 per cell, 1 SE ≈ ±0.04.")

# ---- 2. clean accuracy -----------------------------------------------------------------------------------------
st.subheader("Utility: clean accuracy (no attack), undefended → defended")
st.plotly_chart(dumbbell("clean_none", "clean_def", f"clean accuracy (exact); {setup}", "clean accuracy"), width="stretch", theme="streamlit")
st.caption("A defense that lowers ASR by also lowering clean accuracy is not a win. Exact substring metric. See the TrustRAG caveat below.")

# ---- 3. what each defense removed ------------------------------------------------------------------------------
st.subheader("What each defense removed (per question, mean over cells)")
g = V[V["defense"].isin(order)].groupby("defense").agg(poison_in=("poison_in_topk", "mean"), poison_kept=("poison_kept", "mean"),
                                                       clean_removed=("clean_removed", "mean"), k=("k", "max")).reindex(order)
fig = go.Figure()
for i, (col, name) in enumerate([("poison_in", "poison retrieved into top-k"), ("poison_kept", "poison reaching the LLM"),
                                 ("clean_removed", "clean passages removed (no-attack top-k)")]):
    fig.add_bar(y=[label(d) for d in g.index], x=g[col], name=name, orientation="h", marker_color=colors[i], marker_line_width=0,
                text=[f"{v:.2f}" for v in g[col]], textposition="outside",
                hovertemplate="%{y}<br>" + name + ": %{x:.2f}<extra></extra>")
fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05, height=120 + 70 * len(order), xaxis=dict(range=[0, 5.9], title="passages per question", **GRID),
                  yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=30, b=10, l=10, r=10))
st.plotly_chart(fig, width="stretch", theme="streamlit")
st.caption("Top-k is 5, except knowledge expansion (k = 10 / 50), where the same ~5 poison passages share the prompt with 5–45 extra clean "
           "ones. Only TrustRAG removes poison. Paraphrasing retrieves less of it (the poison starts with the *original* question).")

# ---- 4. caveats --------------------------------------------------------------------------------------------------
st.subheader("Read these before quoting a number")
q_ke = R[R["defense"].str.startswith("knowledge_expansion") & (R["llm"] == "qwen3.5-27b")]
if not q_ke.empty:
    st.warning(f"**Knowledge expansion × Qwen3.5-27B is truncation-confounded.** With 10–50 passages, {q_ke['empty_final'].min():.0%}–"
               f"{q_ke['empty_final'].max():.0%} of its poisoned prompts end with no final answer (undefended: {q_ke['empty_final_none'].mean():.0%}). "
               "The exact metric counts that as 'attack failed'. Use the answered-only (robust) view: "
               f"{q_ke['asr_answered_none'].mean():.2f} → {q_ke['asr_answered_def'].mean():.2f}.", icon="⚠️")
if not tr.empty:
    q_tr = tr[tr["llm"] == "qwen3.5-27b"]
    v_mod = tr[(tr["llm"] == "vicuna-7b-v1.3") & (tr["retrieval"] != "Contriever (paper)")]
    st.warning("**TrustRAG's clean accuracy: no loss for the reasoning models, a real loss for Vicuna-7B.** "
               + (f"With the modern retrievers Vicuna-7B falls {v_mod['clean_none'].mean():.2f} → {v_mod['clean_def'].mean():.2f} "
                  "(pooled over the two cells, 42 answers lost vs 21 gained, exact McNemar p = 0.011, post hoc; Vicuna has no run-to-run noise). "
                  "Stage 1 removes 1.1–1.2 of 5 clean passages per question there, and the 3-stage prompting adds its own changes. " if not v_mod.empty else "")
               + "For Qwen3.5 / gpt-oss the rise is not a gain: the 3-stage answers are long (median ~50–100 words vs ~3), the "
               "substring metric rewards length, and Qwen3.5 also truncates less on the short stage prompts"
               + (f" (clean accuracy {q_tr['clean_none'].mean():.2f} → {q_tr['clean_def'].mean():.2f})" if not q_tr.empty else "")
               + ". The same length effect inflates ASR, so TrustRAG's ASR drop is if anything conservative.", icon="⚠️")
st.info("**Noise floor.** vLLM is not batch-invariant, and 4096-token reasoning amplifies tiny numeric differences. On (almost) identical "
        "prompts: " + "; ".join(f"{LLM_NAME.get(l, l)} up to {v['max_discordant']} flipped answers, |ΔASR| ≤ {v['max_abs_d_asr']:.2f}, "
                                f"|Δclean| ≤ {v['max_abs_d_clean']:.2f}" for l, v in NOISE.items()) + ".", icon="ℹ️")
st.info("**Not applicable / not run.** Knowledge expansion × Vicuna-7B: 8/100 prompts exceed its 2048-token context already at k = 10. "
        "**RobustRAG**: out of scope. Its guarantee assumes a *minority* of retrieved passages are poisoned, but here 4.5–4.8 of 5 are, "
        "so aggregation would side with the poison by construction. Deviations (paraphraser, PPL model) are listed in RESULTS.md.", icon="ℹ️")

both = R[R["defense"].isin(["trustrag", "trustrag@contriever"])]
if both["defense"].nunique() == 2:
    st.markdown("**TrustRAG: faithful SimCSE embedder vs the Contriever substitute** (robust ASR / clean accuracy, mean over setups)")
    t = both.groupby(["defense", "llm"]).agg(asr=("robust_def", "mean"), clean=("clean_def", "mean"), kept=("poison_kept", "mean"),
                                             removed=("clean_removed", "mean")).reset_index()
    t["defense"] = t["defense"].map(lambda d: NAMES.get(d, d))
    t["llm"] = t["llm"].map(lambda l: LLM_NAME.get(l, l))
    st.dataframe(t.rename(columns={"defense": "variant", "llm": "LLM", "asr": "robust ASR", "clean": "clean acc",
                                   "kept": "poison kept / q", "removed": "clean removed / q"}),
                 hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                                                  ["robust ASR", "clean acc", "poison kept / q", "clean removed / q"]})

# ---- 5. every run + drill-down -----------------------------------------------------------------------------------
st.subheader("Every defended run")
cols = ["defense", "retrieval", "llm_name", "k", "asr_none", "asr_def", "d_asr", "p_mcnemar_asr", "robust_none", "robust_def", "robust_metric",
        "clean_none", "clean_def", "d_clean", "poison_in_topk", "poison_kept", "clean_removed", "empty_ctx", "empty_final_none", "empty_final",
        "discordant_asr", "run_id", "twin_run_id"]
T = V[V["defense"].isin(order)][[c for c in cols if c in V]].copy()
T["defense"] = T["defense"].map(lambda d: NAMES.get(d, d))
num = [c for c in T.columns if T[c].dtype.kind == "f"]
st.dataframe(T.rename(columns={"llm_name": "LLM", "p_mcnemar_asr": "p (McNemar)"}), hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.3f" if c == "p_mcnemar_asr" else "%.2f") for c in num})
st.download_button("Download CSV", T.to_csv(index=False).encode(), file_name="poisonedrag_defenses_nq.csv", mime="text/csv")

pick = st.selectbox("Inspect a defended run in Attack flow", T["run_id"].tolist(),
                    format_func=lambda r: f"{NAMES.get(R.set_index('run_id').loc[r, 'defense'], '')} · "
                                          f"{R.set_index('run_id').loc[r, 'retrieval']} × {R.set_index('run_id').loc[r, 'llm_name']}")
if st.button("Open in Attack flow →"):
    st.session_state["run_id"] = pick
    st.switch_page("pages/3_Attack_Flow.py")
st.caption("Generated by `scripts/make_defense_results.py` from the run traces (`experiments/defenses/defense_table.json`). "
           "Full protocol, deviations and findings: `experiments/RESULTS.md` → Step 5.")
