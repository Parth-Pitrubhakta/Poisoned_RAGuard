# Figures

**TrustRAG architecture** (`trustrag_architecture.{svg,tex,dot}`): the Step 5 TrustRAG defense as it runs in this
project's pipeline. The SVG is the primary figure. It is hand-written on a 1200 × 796 px canvas, with colours as
literals in a `<style>` block, and it matches the dashboard palette. The TikZ file is a 1:1 port that uses the same
coordinates (`x=1pt, y=-1pt`) and the same text sizes; for a paper, compile it and `\includegraphics[width=\textwidth]`
the PDF, or set `\figcaptionfalse` to drop the built-in caption and use `\caption{}`. The DOT file has the same content,
but Graphviz chooses its layout. To edit, change the SVG and make the same change in the `.tex` (the coordinates are
identical); re-port the DOT only if the content changes. Every number comes from `experiments/RESULTS.md`
(Step 5, Tables D1–D3 and Limitations) and `experiments/defenses/trustrag_filter_simcse_*.json`, and the stage
logic follows `src/prag/defenses/trustrag.py`, so update the figure whenever those files change. After any edit, run
`source scripts/env.sh && /usr/bin/python3 -I scripts/figure_check.py`. It parses the SVG and checks its `#id`
references, renders it with the system librsvg (1× and 2× PNG), builds the TikZ PDF and fails on font warnings or
bitmap (Type 3) fonts, and structurally checks the DOT. Previews go to `.cache/figcheck/`, never into this folder.
Graphviz `dot` is not installed here, so the DOT is checked but has not been rendered; render it on another machine
with `dot -Tsvg trustrag_architecture.dot -o trustrag_architecture.dot.svg`.

**README charts** (`asr_matrix`, `defenses`, `trustrag_overlap`, each `_light.svg` + `_dark.svg`; the README picks one
with `<picture>`) are generated, not hand-drawn: `uv run python scripts/make_readme_figures.py`. They read only
`experiments/defenses/defense_table.json`, one Step 3 trace file (clean top-5 overlap) and upstream's released poison
file (poison-pair overlap), and show aggregate numbers only, never poison text. Re-run after the results change.
