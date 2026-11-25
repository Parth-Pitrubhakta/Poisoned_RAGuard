"""Pre-Checkpoint-1 check: precomputed doc ids -> BEIR passage text join is correct, and clean passages
actually reach the RAG context (not just poison).

Checks, for the first `--n` target questions:
  1. every precomputed top-k doc id resolves to a non-empty corpus passage;
  2. re-encoding those passages with Contriever (title + " " + text, as upstream's evaluate_beir did)
     reproduces the precomputed dot scores -> proves the id->text join is the *right* text;
  3. after merging with poison (upstream logic), the prompt contains every surviving clean passage verbatim.

usage: uv run python scripts/check_corpus_join.py [--n 2] [--top_k 5] [--adv_per_query 5]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def freest_gpu() -> str:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    idx, free = max((l.split(", ") for l in out.strip().splitlines()), key=lambda t: int(t[1]))
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--adv_per_query", type=int, default=5)
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", freest_gpu())

    import torch
    from prag.attack.lm_targeted import lm_targeted_poison
    from prag.data.beir import contriever_index_text, corpus_text, load_corpus
    from prag.data.targets import load_targets
    from prag.llm.prompts import wrap_prompt
    from prag.retrieval.contriever import ContrieverEncoder
    from prag.retrieval.merge import Passage, load_precomputed, poisoned_topk

    targets = load_targets(f"{ROOT}/upstream/results/adv_targeted_results/nq.json")[: args.n]
    pre = load_precomputed(f"{ROOT}/upstream/results/beir_results/nq-contriever.json")
    need = {d for t in targets for d, _ in pre[t.id][: args.top_k]}
    print(f"streaming corpus for {len(need)} doc ids ...", flush=True)
    corpus = load_corpus(f"{ROOT}/data/beir/nq", ids=need)

    enc = ContrieverEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
    # upstream: all poison texts of the iteration are injected and scored against every query
    all_poison = [(t.id, p) for t in targets for p in lm_targeted_poison(t, args.adv_per_query)]
    poison_emb = enc.encode([p for _, p in all_poison])

    ok = True
    for t in targets:
        print(f"\n=== {t.id}: {t.question!r}  (correct={t.correct_answer!r}, target={t.incorrect_answer!r})")
        ids_scores = pre[t.id][: args.top_k]
        q_emb = enc.encode([t.question])[0]
        d_emb = enc.encode([contriever_index_text(corpus[d]) for d, _ in ids_scores])
        recomputed = enc.dot(q_emb, d_emb).tolist()
        clean = []
        for (d, s), r in zip(ids_scores, recomputed):
            text = corpus_text(corpus[d])
            good = bool(text.strip()) and abs(r - s) <= args.tol
            ok &= good
            print(f"  [{'OK ' if good else 'BAD'}] {d:<12} pre={s:.6f} recomputed={r:.6f} |Δ|={abs(r - s):.2e}  "
                  f"title={corpus[d]['title'][:30]!r} text={text[:70]!r}...")
            clean.append(Passage(text=text, score=s, doc_id=d))
        psims = enc.dot(q_emb, poison_emb).tolist()
        poison = [Passage(text=p, score=sc, is_poison=True, poison_for=qid) for (qid, p), sc in zip(all_poison, psims)]
        topk = poisoned_topk(clean, poison, args.top_k)
        prompt = wrap_prompt(t.question, [p.text for p in topk], prompt_id=4)
        print("  poisoned top-k:")
        for p in topk:
            tag = f"POISON(for {p.poison_for})" if p.is_poison else f"clean {p.doc_id}"
            in_prompt = p.text in prompt
            ok &= in_prompt
            print(f"    #{p.rank} {p.score:.4f} {tag:<22} in_prompt={in_prompt} {p.text[:60]!r}...")
        n_clean = sum(not p.is_poison for p in topk)
        print(f"  -> {n_clean} clean + {args.top_k - n_clean} poison passages in poisoned context")
        # no-attack context: the clean top-k alone must reach the prompt verbatim
        clean_prompt = wrap_prompt(t.question, [p.text for p in clean], prompt_id=4)
        clean_ok = all(p.text in clean_prompt for p in clean) and not any(p.text in clean_prompt for p in poison)
        ok &= clean_ok
        print(f"  -> clean (no-attack) context: {len(clean)} BEIR passages in prompt, 0 poison: {clean_ok}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
