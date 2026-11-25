"""README figures (light + dark SVG) from finished results. No models, no GPU, no downloads.

  uv run python scripts/make_readme_figures.py      # -> docs/figures/{asr_matrix,defenses,trustrag_overlap}_{light,dark}.svg

Inputs (read-only): experiments/defenses/defense_table.json (Step 3 twins + Step 5 defenses), the Step 3 traces of
Qwen3-Emb-8B x Vicuna-7B (clean no-attack top-5, for clean-pair overlap), and upstream's released poison texts
(upstream/results/adv_targeted_results/nq.json, for poison-pair overlap). The figures show aggregate numbers only,
never poison text. Colours follow the dashboard palette (app/lib.py); the README picks the variant with <picture>.
"""
from __future__ import annotations

import itertools
import json
from html import escape
from pathlib import Path

from prag.defenses.trustrag import rouge_l_f

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, 'Liberation Sans', sans-serif"

THEMES = {
    "light": dict(bg="#ffffff", ink="#1f1e1c", ink2="#55534e", muted="#7a7873", grid="#e6e4df", axis="#c9c6bf",
                  band="#ebe9e4", hl="#f3f7fd", neutral="#8a8780", blue="#2a78d6", orange="#eb6834", green="#1baf7a",
                  ramp=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"], ramp_dark_from=3,
                  slot_empty="#e6e4df"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", ink2="#b1bac4", muted="#8b949e", grid="#262c35", axis="#3a414b",
                 band="#1f252e", hl="#121b2a", neutral="#8b949e", blue="#3987e5", orange="#d95926", green="#199e70",
                 ramp=["#0f2747", "#143a6a", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"], ramp_dark_from=None,
                 slot_empty="#262c35"),
}
LLMS = [("vicuna-7b-v1.3", "Vicuna-7B v1.3", "2023, paper's LLM"), ("qwen3.5-27b", "Qwen3.5-27B", "reasoning, dense"),
        ("gpt-oss-120b", "gpt-oss-120b", "reasoning, MoE")]
SETUPS = [("Contriever (paper)", "Contriever", "paper's retriever"), ("Qwen3-Emb-8B", "Qwen3-Embedding-8B", "modern dense"),
          ("Qwen3-Emb-8B + rerank", "+ Qwen3-Reranker-4B", "modern, reranked")]
DEFENSES = [("duplicate_filter", "Duplicate filter"), ("perplexity_filter", "Perplexity filter"), ("paraphrase", "Query paraphrasing"),
            ("knowledge_expansion@10", "Knowledge expansion, k = 10"), ("knowledge_expansion@50", "Knowledge expansion, k = 50"),
            ("trustrag", "TrustRAG")]


def t(x, y, s, size=13, fill="ink", anchor="start", weight=400, T=None, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{T[fill]}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{extra}>{escape(str(s))}</text>')


def doc(w, h, body, T, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}" '
            f'role="img" aria-labelledby="t d"><title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>'
            f'<rect width="{w}" height="{h}" fill="{T["bg"]}"/>' + "".join(body) + "</svg>\n")


# ---------------------------------------------------------------------------------------------------- data
def load():
    D = json.loads((ROOT / "experiments/defenses/defense_table.json").read_text())
    rows = D["rows"]
    cell = {(r["retrieval"], r["llm"]): r for r in rows if r["defense"] == "trustrag"}  # twin values = Step 3 cell
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    dfn = {(d, l): (mean([r["robust_none"] for r in rows if r["defense"] == d and r["llm"] == l]),
                    mean([r["robust_def"] for r in rows if r["defense"] == d and r["llm"] == l]))
           for d, _ in DEFENSES for l, _, _ in LLMS}
    # pairwise ROUGE-L among each question's 5 poison texts (first 100 targets = full protocol), and among clean top-5
    adv = list(json.loads((ROOT / "upstream/results/adv_targeted_results/nq.json").read_text()).values())[:100]
    inj, strip = [], []
    for it in adv:
        q, xs = it["question"], it["adv_texts"][:5]
        inj.append([rouge_l_f(q + "." + a, q + "." + b) for a, b in itertools.combinations(xs, 2)])
        strip.append([rouge_l_f(a, b) for a, b in itertools.combinations(xs, 2)])
    run = next(p for p in sorted((ROOT / "experiments").glob("nq_qwen3emb8b_vicuna7b_full-*")) if (p / "traces.jsonl").exists())
    clean = []
    for line in (run / "traces.jsonl").read_text().splitlines():
        tr = json.loads(line)
        xs = [p["text"] for p in tr["clean_topk"]]
        clean.append([rouge_l_f(a, b) for a, b in itertools.combinations(xs, 2)])
    return cell, dfn, (inj, strip, clean), D


# ---------------------------------------------------------------------------------------------------- figure 1
def fig_asr_matrix(cell, T):
    W, H = 980, 392
    x0, y0, cw, ch, gap = 250, 112, 150, 68, 4
    body = [t(24, 36, "No defense: the attack survives a modern RAG stack", 20, weight=700, T=T),
            t(24, 60, "Attack success rate (robust metric), NQ, 100 target questions per cell, 5 poison texts in a 2.68M-passage corpus", 13, "ink2", T=T)]
    for j, (_, name, sub) in enumerate(LLMS):
        cx = x0 + j * (cw + gap) + cw / 2
        body += [t(cx, y0 - 26, name, 14, anchor="middle", weight=700, T=T), t(cx, y0 - 10, sub, 11.5, "muted", anchor="middle", T=T)]
    sx = x0 + 3 * (cw + gap) + 34
    body += [t(sx + 82, y0 - 26, "Poison in the top-5", 14, anchor="middle", weight=700, T=T),
             t(sx + 82, y0 - 10, "retrieved slots (same for all LLMs)", 11.5, "muted", anchor="middle", T=T)]
    for i, (key, name, sub) in enumerate(SETUPS):
        y = y0 + i * (ch + gap)
        body += [t(x0 - 16, y + ch / 2 - 2, name, 14, anchor="end", weight=700, T=T),
                 t(x0 - 16, y + ch / 2 + 15, sub, 11.5, "muted", anchor="end", T=T)]
        for j, (l, _, _) in enumerate(LLMS):
            r = cell[(key, l)]
            v, ex = r["robust_none"], r["asr_none"]
            k = max(0, min(6, int((v - 0.5) / 0.5 * 7)))
            dark_text = (T["ramp_dark_from"] is not None and k >= T["ramp_dark_from"]) or (T["ramp_dark_from"] is None and k >= 5)
            fg = ("#ffffff" if T["ramp_dark_from"] is not None else T["bg"]) if dark_text else T["ink"]
            x = x0 + j * (cw + gap)
            body.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="4" fill="{T["ramp"][k]}"/>')
            body.append(f'<text x="{x + cw / 2:.1f}" y="{y + 36:.1f}" font-size="24" font-weight="700" fill="{fg}" text-anchor="middle">{v:.2f}</text>')
            if abs(ex - v) > 0.005:
                body.append(f'<text x="{x + cw / 2:.1f}" y="{y + 55:.1f}" font-size="11.5" fill="{fg}" text-anchor="middle" opacity="0.85">exact metric {ex:.2f}</text>')
        p = cell[(key, "vicuna-7b-v1.3")]["poison_in_topk"]
        for s in range(5):
            fx = sx + s * 34
            frac = max(0.0, min(1.0, p - s))
            body.append(f'<rect x="{fx}" y="{y + 14}" width="30" height="30" rx="4" fill="{T["slot_empty"]}"/>')
            if frac > 0:
                body.append(f'<rect x="{fx}" y="{y + 14}" width="{30 * frac:.1f}" height="30" rx="4" fill="{T["orange"]}"/>')
        body.append(t(sx + 82, y + 62, f"{p:.2f} of 5 are poison", 12, "ink2", anchor="middle", T=T))
    yb = y0 + 3 * (ch + gap) + 22
    body += [t(24, yb, "Robust metric: exact substring for Vicuna; answered-only for Qwen3.5 (it truncates ~20% of answers at T = 0.1); "
                       "typography-normalized for gpt-oss", 11.5, "muted", T=T),
             t(24, yb + 17, "(it writes U+202F spaces, which hide exact matches). The paper's exact metric is shown where it differs. "
                            "1 SE ≈ ±0.04 per cell.", 11.5, "muted", T=T)]
    return doc(W, H, body, T, "Attack success with no defense, by retrieval setup and LLM",
               "Heatmap of robust attack success rate for 3 retrieval setups by 3 LLMs, between 0.70 and 0.99, "
               "plus the share of the top-5 retrieved passages that are poison: 4.79, 4.53 and 4.62 of 5.")


# ---------------------------------------------------------------------------------------------------- figure 2
def fig_defenses(dfn, D, T):
    W = 1000
    lx, px0, pw, pg, top, rh = 24, 232, 236, 22, 150, 44
    H = top + rh * len(DEFENSES) + 92
    sc = lambda j, v: px0 + j * (pw + pg) + 12 + v * (pw - 24)
    body = [t(lx, 36, "Defenses: only TrustRAG cuts the attack below 10%", 20, weight=700, T=T),
            t(lx, 60, "Robust attack success rate, no defense → with defense. Mean over 3 retrieval setups; 100 questions per cell; 57 defended runs.", 13, "ink2", T=T)]
    # legend
    ly = 92
    body += [f'<circle cx="{lx + 6}" cy="{ly - 4}" r="6" fill="{T["bg"]}" stroke="{T["neutral"]}" stroke-width="2.5"/>',
             t(lx + 18, ly, "no defense", 12.5, "ink2", T=T),
             f'<circle cx="{lx + 116}" cy="{ly - 4}" r="6.5" fill="{T["blue"]}"/>', t(lx + 128, ly, "with defense", 12.5, "ink2", T=T),
             f'<rect x="{lx + 232}" y="{ly - 10}" width="34" height="12" rx="2" fill="{T["band"]}"/>',
             t(lx + 274, ly, "±0.05 run-to-run noise (reasoning models; Vicuna is exactly reproducible)", 12.5, "ink2", T=T)]
    # TrustRAG highlight row
    ti = [d for d, _ in DEFENSES].index("trustrag")
    body.append(f'<rect x="{lx - 8}" y="{top + ti * rh - 2}" width="{W - 2 * lx + 16}" height="{rh}" rx="6" fill="{T["hl"]}"/>')
    for j, (l, name, sub) in enumerate(LLMS):
        x = px0 + j * (pw + pg)
        body += [t(x + pw / 2, top - 22, name, 14, anchor="middle", weight=700, T=T)]
        for g in (0, 0.25, 0.5, 0.75, 1.0):
            gx = sc(j, g)
            body.append(f'<line x1="{gx:.1f}" y1="{top - 8}" x2="{gx:.1f}" y2="{top + rh * len(DEFENSES) - 6}" stroke="{T["grid"]}" stroke-width="1"/>')
            body.append(t(gx, top + rh * len(DEFENSES) + 12, f"{g:g}", 11, "muted", anchor="middle", T=T))
        for i, (d, _) in enumerate(DEFENSES):
            cy = top + i * rh + rh / 2 - 2
            a, b = dfn[(d, l)]
            if a is None or b is None:
                body.append(t(x + pw / 2, cy + 4, "N/A: 2048-token context", 11.5, "muted", anchor="middle", T=T,
                              extra=' font-style="italic"'))
                continue
            if l != "vicuna-7b-v1.3":
                body.append(f'<rect x="{sc(j, a - 0.05):.1f}" y="{cy - 9}" width="{sc(j, 0.1) - sc(j, 0):.1f}" height="18" rx="3" fill="{T["band"]}"/>')
            body.append(f'<line x1="{sc(j, a):.1f}" y1="{cy}" x2="{sc(j, b):.1f}" y2="{cy}" stroke="{T["neutral"]}" stroke-width="2" opacity="0.7"/>')
            body.append(f'<circle cx="{sc(j, a):.1f}" cy="{cy}" r="6" fill="{T["bg"]}" stroke="{T["neutral"]}" stroke-width="2.5"/>')
            body.append(f'<circle cx="{sc(j, b):.1f}" cy="{cy}" r="6.5" fill="{T["blue"]}" stroke="{T["bg"]}" stroke-width="2"/>')
            lab_x = min(sc(j, a), sc(j, b)) - 11
            body.append(t(lab_x, cy + 4.5, f"{b:.2f}", 12, "ink", anchor="end", weight=700 if d == "trustrag" else 400, T=T))
        body.append(t(x + pw / 2, top + rh * len(DEFENSES) + 30, "attack success rate", 11.5, "muted", anchor="middle", T=T))
    for i, (d, name) in enumerate(DEFENSES):
        cy = top + i * rh + rh / 2 + 2.5
        body.append(t(lx, cy, name, 13.5, "ink", weight=700 if d == "trustrag" else 400, T=T))
    yb = H - 26
    body.append(t(lx, yb, "Robust metric: exact for Vicuna, answered-only for Qwen3.5, typography-normalized for gpt-oss. Knowledge expansion "
                          "for Qwen3.5 is partly truncation (45–67% empty answers).", 11.5, "muted", T=T))
    body.append(t(lx, yb + 16, "TrustRAG = k-means + n-gram filter (SimCSE) and 3-stage conflict resolution. Clean-accuracy cost: "
                               "Vicuna-7B −0.07; no loss for the reasoning models.", 11.5, "muted", T=T))
    return doc(W, H, body, T, "Attack success under each defense, per LLM",
               "Dumbbell chart, one panel per LLM. Duplicate and perplexity filters change nothing; paraphrasing trims 0.07 to 0.12; "
               "knowledge expansion to k = 50 roughly halves attack success for gpt-oss (0.79 to 0.39); TrustRAG reduces it to 0.08-0.09 for all three LLMs.")


# ---------------------------------------------------------------------------------------------------- figure 3
def fig_overlap(pairs, T):
    inj, strip, clean = pairs  # per question: list of 10 pair scores
    W, lx, plot_x, plot_w, rowh, top = 1000, 24, 300, 500, 96, 100
    H = top + 3 * rowh + 70
    nb, lo, hi = 40, 0.0, 1.0
    sx = lambda v: plot_x + (v - lo) / (hi - lo) * plot_w
    rows = [("Poison pairs, as injected", "question + \".\" + adv_text (the published texts)", inj, "orange"),
            ("Poison pairs, question prefix removed", "adv_text only", strip, "orange"),
            ("Clean pairs, no attack", "top-5 from Qwen3-Embedding-8B", clean, "blue")]
    body = [t(lx, 36, "Why TrustRAG works here: the 5 poison texts are near-duplicates", 20, weight=700, T=T),
            t(lx, 60, "ROUGE-L F between every pair of passages in a question's set (100 questions, 1,000 pairs per row). "
                      "TrustRAG's filter only acts if some pair exceeds 0.25.", 13, "ink2", T=T)]
    gate = sx(0.25)
    for i, (name, sub, qs, col) in enumerate(rows):
        xs = [v for q in qs for v in q]
        tripped = sum(any(v > 0.25 for v in q) for q in qs)
        y = top + i * rowh
        base = y + rowh - 26
        counts = [0] * nb
        for v in xs:
            counts[min(nb - 1, int((v - lo) / (hi - lo) * nb))] += 1
        peak = max(counts)
        bw = plot_w / nb
        body.append(f'<line x1="{plot_x}" y1="{base}" x2="{plot_x + plot_w}" y2="{base}" stroke="{T["axis"]}" stroke-width="1"/>')
        for b, c in enumerate(counts):
            if c:
                h = (rowh - 40) * c / peak
                body.append(f'<rect x="{plot_x + b * bw + 1:.1f}" y="{base - h:.1f}" width="{bw - 2:.1f}" height="{h:.1f}" rx="1.5" fill="{T[col]}"/>')
        above = sum(v > 0.25 for v in xs) / len(xs)
        body += [t(lx, base - 30, name, 14, weight=700, T=T), t(lx, base - 13, sub, 11.5, "muted", T=T),
                 t(plot_x + plot_w + 28, base - 40, f"{above:.0%}", 22, "ink", weight=700, T=T),
                 t(plot_x + plot_w + 28, base - 23, "of pairs above the gate", 11.5, "muted", T=T),
                 t(plot_x + plot_w + 28, base - 6, f"{tripped}/{len(qs)} questions trip it", 11.5, "ink2", weight=700, T=T)]
    body.append(f'<line x1="{gate:.1f}" y1="{top - 8}" x2="{gate:.1f}" y2="{top + 3 * rowh - 26}" stroke="{T["ink"]}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    body.append(t(gate + 6, top - 12, "TrustRAG n-gram gate, 0.25", 12, "ink", weight=700, T=T))
    ya = top + 3 * rowh - 8
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        body.append(t(sx(g), ya, f"{g:g}", 11, "muted", anchor="middle", T=T))
    body.append(t(plot_x + plot_w / 2, ya + 18, "ROUGE-L F of a passage pair", 11.5, "muted", anchor="middle", T=T))
    body.append(t(lx, H - 14, "An adaptive attacker who keeps every pair under the gate (and under SimCSE cosine 0.88) would pass TrustRAG's filter. "
                              "Untested here.", 12, "ink2", T=T))
    frac = lambda qs: sum(v > 0.25 for q in qs for v in q) / sum(len(q) for q in qs)
    return doc(W, H, body, T, "Pairwise overlap of poison vs clean passages",
               f"Histograms of ROUGE-L between passage pairs. Poison as injected: {frac(inj):.0%} of pairs above 0.25; "
               f"prefix removed: {frac(strip):.0%}; clean top-5 pairs: {frac(clean):.0%}.")


def main():
    cell, dfn, pairs, D = load()
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, T in THEMES.items():
        for name, svg in (("asr_matrix", fig_asr_matrix(cell, T)), ("defenses", fig_defenses(dfn, D, T)),
                          ("trustrag_overlap", fig_overlap(pairs, T))):
            p = OUT / f"{name}_{mode}.svg"
            p.write_text(svg)
            print(p.relative_to(ROOT), f"{len(svg) / 1024:.1f} KB")
    for name, qs in zip(("injected", "stripped", "clean"), pairs):
        print(f"{name}: pairs >0.25 {sum(v > .25 for q in qs for v in q) / sum(map(len, qs)):.3f}, "
              f"questions tripping {sum(any(v > .25 for v in q) for q in qs)}/{len(qs)}")


if __name__ == "__main__":
    main()
