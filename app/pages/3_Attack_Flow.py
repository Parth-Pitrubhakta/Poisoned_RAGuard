import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib import CLEAN_BG, POISON_BG, load_traces, page, run_picker, runs  # noqa: E402
from prag.eval.metrics import attack_success_normalized  # noqa: E402

page("Attack flow", "🔎")
row = run_picker(runs())
if row is None:
    st.stop()
T = load_traces(row.path)
if not T:
    st.info("This run has no traces yet.")
    st.stop()

# ---- question picker + filters (one row) -------------------------------------------------------------------
f1, f2, f3 = st.columns([1, 1, 3])
outcome = f1.selectbox("Outcome", ["all", "attack succeeded", "attack failed"])
iters = sorted({t.get("iteration", 0) for t in T})
it = f2.selectbox("Iteration", ["all"] + iters)
sel = [t for t in T if (it == "all" or t.get("iteration", 0) == it)
       and (outcome == "all" or (t["success"] if outcome == "attack succeeded" else not t["success"]))]
if not sel:
    st.info("No question matches these filters.")
    st.stop()
qlabel = {t["question_id"]: f"{'❌' if t['success'] else '🛡️'} {t['question_id']} — {t['question']}" for t in sel}
qid = f3.selectbox(f"Question ({len(sel)} shown)", list(qlabel), format_func=qlabel.get, key=f"q_{row.run_id}")
t = next(x for x in sel if x["question_id"] == qid)

# ---- verdict ------------------------------------------------------------------------------------------------
k = len(t["retrieved_topk"])
n_poison = len(t["poison_texts"])
n_own = t.get("n_own_poison_in_topk", round(t["recall"] * max(1, n_poison)))  # Step 1 traces lack this field
if t["success"]:
    st.error(f"**ATTACK SUCCEEDED** — the poisoned answer contains the attacker's target “{t['target_answer']}”.", icon="❌")
else:
    norm_hit = attack_success_normalized(t["target_answer"], t["output_poisoned"])
    extra = " (but it *does* match after typography normalization — a metric artifact)" if norm_hit else ""
    empty = " The model produced **no final answer** (reasoning truncated)." if not t["output_poisoned"].strip() else ""
    st.success(f"**ATTACK FAILED** — the target answer is not in the final answer{extra}.{empty}", icon="🛡️")

c = st.columns(4)
c[0].markdown(f"**Question**  \n{t['question']}")
c[1].markdown(f"**Correct answer**  \n✅ {t['correct_answer']}")
c[2].markdown(f"**Attacker target**  \n❌ {t['target_answer']}")
c[3].markdown(f"**Own poison retrieved**  \n{n_own} / {k} slots · "
              f"P {t['precision']:.2f} · R {t['recall']:.2f}  \n{t['n_poison_injected_total']} poison texts injected this iteration")

# ---- defense summary (Step 5 runs) ------------------------------------------------------------------------
dfd = t.get("defense") if isinstance(t.get("defense"), dict) else None
defended = bool(dfd and dfd.get("name", "none") != "none")
if defended:
    msg = (f"**Defense `{dfd['name']}`**: removed {dfd.get('removed', 0)} of {k} retrieved passages; "
           f"{dfd.get('poison_kept', '?')} poison passage(s) reached the LLM; on the no-attack path it removed {dfd.get('clean_removed', 0)} clean passage(s).")
    if (dfd.get("info") or {}).get("branch"):
        msg += f"  \nTrustRAG stage-1 branch: `{dfd['info']['branch']}`."
    if dfd.get("query_used"):
        msg += f"  \nQuery actually used (paraphrased): *{dfd['query_used']}*"
    st.info(msg, icon="🛡️")

# ---- answers: clean vs poisoned ---------------------------------------------------------------------------
st.subheader("Answers")
a, b = st.columns(2)
with a:
    ok = t.get("clean_correct")
    st.markdown(f"**✅ Clean RAG (no poison)** — {'correct' if ok else 'not correct' if ok is not None else 'not run'}")
    st.info(t.get("output_clean") or "_(empty)_", icon="✅" if ok else "➖")
with b:
    st.markdown(f"**❌ Poisoned RAG** — {'attacker answer' if t['success'] else 'resisted / no answer'}")
    (st.error if t["success"] else st.warning)(t["output_poisoned"] or "_(empty final answer)_", icon="❌" if t["success"] else "🛡️")


def passages_table(ps: list[dict], own_qid: str, kept_ranks: list | None = None) -> None:
    rows = [{"rank": p.get("rank"), "label": ("POISON" if p["is_poison"] else "clean"),
             **({"defense": "kept" if p.get("rank") in set(kept_ranks) else "removed"} if kept_ranks is not None else {}),
             "for": ("this question" if p.get("poison_for") == own_qid else f"other ({p.get('poison_for')})") if p["is_poison"] else "",
             "score": p["score"], "rerank": p.get("rerank_score"), "doc_id": p.get("doc_id") or "", "text": p["text"]} for p in ps]
    df = pd.DataFrame(rows)
    if df["rerank"].isna().all():
        df = df.drop(columns=["rerank"])

    def shade(r):
        return [f"background-color: {POISON_BG if r['label'] == 'POISON' else CLEAN_BG}"] * len(r)

    st.dataframe(df.style.apply(shade, axis=1).format({"score": "{:.4f}", **({"rerank": "{:.3f}"} if "rerank" in df else {})}),
                 hide_index=True, width="stretch",
                 column_config={"text": st.column_config.TextColumn(width="large"), "label": st.column_config.TextColumn(width="small")})


st.subheader(f"Retrieved top-{k} under attack")
passages_table(t["retrieved_topk"], qid, dfd.get("kept_ranks") if defended else None)
st.caption("Red rows = injected POISON (synthetic, written for this benchmark question). Green rows = genuine corpus passages. "
           "`score` = retriever similarity; `rerank` = reranker P(relevant) when a reranker is used.")
if t.get("first_stage"):
    with st.expander(f"First stage before reranking ({len(t['first_stage'])} candidates)"):
        passages_table(t["first_stage"], qid)
with st.expander(f"Clean top-{len(t['clean_topk'])} (no attack) — what the model saw for the clean answer"):
    passages_table(t["clean_topk"], qid, dfd.get("clean_kept_ranks") if defended else None)
with st.expander(f"The {n_poison} poison texts written for this question"):
    for i, p in enumerate(t["poison_texts"], 1):
        hit = any(r["is_poison"] and r["text"] == p for r in t["retrieved_topk"])
        st.markdown(f"**POISON {i}** {'· retrieved' if hit else '· not retrieved'}")
        st.code(p, language=None, wrap_lines=True)

gen = (dfd or {}).get("generation") or {}
if gen.get("poisoned"):
    with st.expander("TrustRAG conflict resolution: intermediate stages (poisoned path)"):
        gp = gen["poisoned"]
        st.markdown("**Stage 1: internal knowledge** (question only; shared with the clean path)")
        st.code(gp.get("internal_knowledge") or "(empty)", language=None, wrap_lines=True)
        st.markdown("**Stage 2: consolidated information** (retrieved docs that survived stage 1 + internal knowledge)")
        st.code(gp.get("consolidated") or "(empty)", language=None, wrap_lines=True)
        st.caption("Stage 3 (final answer) is the poisoned answer above. Tokens per stage: "
                   + ", ".join(f"{n} {v['prompt_tokens']}→{v['completion_tokens']} ({v['finish_reason']})" for n, v in gp.get("stages", {}).items()))

st.subheader("What the LLM received")
with st.expander("Prompt sent (final stage)" if gen.get("poisoned") else "Prompt sent (MULTIPLE_PROMPT, poisoned contexts)", expanded=False):
    st.code(t["prompt_poisoned"], language=None, wrap_lines=True)
with st.expander("Exact model input (chat template applied)"):
    st.code(t["llm_input_poisoned"], language=None, wrap_lines=True)
u = t["usage"]["poisoned"]
if u.get("reasoning") or u.get("raw_text"):
    with st.expander("Reasoning / raw output (not scored — ASR uses the final answer only)"):
        if u.get("reasoning"):
            st.code(u["reasoning"], language=None, wrap_lines=True)
        if u.get("raw_text"):
            st.caption("Raw completion")
            st.code(u["raw_text"], language=None, wrap_lines=True)

st.subheader("Usage")
uc = t["usage"].get("clean") or {}
st.dataframe(pd.DataFrame([
    {"call": "poisoned", "prompt tokens": u["prompt_tokens"], "completion tokens": u["completion_tokens"],
     "finish": u.get("finish_reason"), "latency s": round(u["latency_s"], 3), "cost $": u.get("cost_usd", 0.0)},
    {"call": "clean", "prompt tokens": uc.get("prompt_tokens"), "completion tokens": uc.get("completion_tokens"),
     "finish": uc.get("finish_reason"), "latency s": round(uc.get("latency_s", 0.0) or 0.0, 3), "cost $": uc.get("cost_usd", 0.0)},
]), hide_index=True, width="stretch")
dfn = t.get("defense") or "none"
dfn = dfn.get("name", "none") if isinstance(dfn, dict) else dfn  # Step 1 traces store a plain string
st.caption(f"Defense: `{dfn}` · latency is the batch time divided by the batch size (vLLM batches requests).")
