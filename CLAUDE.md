# CLAUDE.md — rules for working in poisonedrag-modern

Defensive trustworthy-ML research: reproduce and modernize **PoisonedRAG** (Zou et al., USENIX Security 2025,
arXiv:2402.07867, upstream https://github.com/sleeepeer/PoisonedRAG) to measure how vulnerable modern RAG
stacks are to knowledge-base poisoning and to evaluate defenses.

## Scope (non-negotiable)
- All poisoning is confined to **local copies of benchmark corpora** (BEIR NQ / HotpotQA / MS-MARCO).
  Never target real systems, live indexes, or third parties. Never publish poison texts outside this repo.
- Poison texts come from upstream's released `results/adv_targeted_results/*.json`. Generate new ones only
  if explicitly asked, and only for these benchmark questions.
- White-box HotFlip is an optional, clearly-flagged extension (`attack.method: hotflip`), off by default.

## Guardrails
1. Write only inside this directory (`poisonedrag-modern/`). The venv is `./venv_PoisonedRAG`.
2. **Always `source scripts/env.sh` first.** It redirects uv, Python, HF, vLLM, Triton, Inductor, and FlashInfer
   caches into `./.cache/`. Never use the shared `/DATA2/shared/hf-cache`.
3. No OS / driver / system-package changes. No sudo. No Docker (no access anyway).
4. Ask before: any download > ~2 GB, any paid API call or run that could exceed $5 total, any new
   pip/uv install (show the resolved packages), and any run > ~10 min or > 5 GB disk.
5. Secrets: never print, log, or write API keys. Report only "set / not set".
6. Reproducibility: global seed 12, pinned deps (`uv.lock`), and every command logged in `RUNLOG.md`.
7. Stop at each ⛔ checkpoint (see README "Status") and wait for approval.
8. The user handles all git themselves. Do not commit. `upstream/` is a pristine clone; never edit it.

## Machine (see SURVEY.md)
- Shared server, 4× H200 141 GB, CUDA driver 13.0, 1 TB RAM, 95 TB on /DATA2. Headless over SSH.
- **Shared GPUs:** before any GPU job, check `nvidia-smi` free memory, pick the freest GPU, and set
  `CUDA_VISIBLE_DEVICES` to a single id (unless TP>1 is approved).
- No provider API keys are set, so all targets are local ($0).

## Environment
```bash
source scripts/env.sh
uv sync                        # creates ./venv_PoisonedRAG from uv.lock
uv run pytest                  # CPU unit tests
uv run python -m prag.run --config configs/<cfg>.yaml [--dry-run]
uv run streamlit run app/Home.py --server.headless true --server.address 127.0.0.1 --server.port 8502
```
Stack: Python 3.12 (uv-managed), torch 2.13.0 (cu130), vLLM 0.32.0, transformers 5.19.0,
sentence-transformers 6.2.0, faiss-cpu 1.15.1, streamlit 1.67.0, plotly 7.1.0. **Do not** reuse upstream's torch 1.13 pins.
`beir`, `fschat`, and `datasets` are intentionally not dependencies: we read BEIR's simple jsonl/tsv format
directly and reproduce FastChat's vicuna_v1.1 template as a string.

## Model choices (configs/models.yaml)
| Role | Model |
|---|---|
| Paper retriever | `facebook/contriever` (mean pooling, dot score) + upstream precomputed `beir_results` |
| Modern retriever | `Qwen/Qwen3-Embedding-8B` (primary), `Qwen/Qwen3-Embedding-0.6B` (fast), `BAAI/bge-m3` (alt) |
| Reranker (flag) | `Qwen/Qwen3-Reranker-4B` |
| Original-style LLM | `lmsys/vicuna-7b-v1.3` (in the paper, ungated) |
| Modern LLMs | `Qwen/Qwen3.5-27B` (dense), `openai/gpt-oss-120b` (MoE) |
| Vector store | FAISS `IndexFlatIP` (exact dot product, matches paper) |

## Faithfulness notes (from reading upstream @ f660d72)
- Poison text = `question + "." + adv_text` (`src/attack.py`, LM_targeted). N = `adv_per_query` (default 5).
- Retrieval: clean top-k from precomputed BEIR results, then poison texts are scored against the query
  embedding with the same retriever, merged, sorted desc, and the top-k kept. Only `corpus[id]["text"]` is used (no title).
- Prompt: `MULTIPLE_PROMPT` (prompt_id=4), contexts joined with `"\n"`. Copy it byte-exact.
- Vicuna: FastChat `vicuna_v1.1` conversation template, `do_sample=True`, temperature 0.1, 150 new tokens,
  repetition_penalty 1.0. Model config reseeds with **seed 100** after the global `setup_seeds(12)`.
- Upstream `Vicuna.query` swallows every exception and returns `""`. That is the cause of "ASR 0.0" (issue #1).
  We never swallow errors; the smoke test asserts non-empty output.
- Decoder-only tokenizers use `padding_side='left'` for batched generation.
- Metrics (main.py): ASR = mean over iters of (#q with `clean_str(target) in clean_str(output)`) / M;
  Precision = poison-in-top-k / k; Recall = poison-in-top-k / N; F1 = `2PR/(P+R)` element-wise (0 where P+R=0),
  then the mean. Upstream rounds means to 2 decimals; we store the unrounded values and also report the rounded ones.
- `clean_str`: `str(s).strip()`, drop one trailing ".", lowercase.

## Code layout
- `src/prag/config.py`: pydantic `RunConfig` (extra keys forbidden) plus `models.yaml` registry checks.
- `src/prag/pipeline.py`: retrieve (encoder → clean top-n → merge per-iteration poison pool → optional rerank → defense), then generate and trace.
  Components are built lazily and freed in order (encoder → reranker → LLM).
- `src/prag/run.py`: the CLI (`--dry-run` estimate, `--resume`, GPU pick). `src/prag/build_index.py`: sharded FAISS IndexFlatIP build.
- `src/prag/baseline.py`: frozen Step 1 runner. Keep it for reference; `prag.run` reproduces it exactly (100/100 traces).
- `src/prag/defenses/`: Step 5. `base.py` hook (rewrite_query / prepare+filter_batch+release / answer), `filters.py` (paper's four),
  `trustrag.py` (port of HuichiZhou/TrustRAG@11dcea0; `third_party/trustrag_ref/` is a read-only reference, never imported). Configs:
  `scripts/make_defense_configs.py` → `configs/defenses/<tag>/`; table: `scripts/make_defense_results.py` (RESULTS DEFENSE block).
  `pipeline._release()` frees nothing by itself: set the caller's reference to None first (shared GPUs).
- `src/prag/{data,retrieval,attack,llm,eval,defenses,tracing}`: the modules. `llm/parsers.py` splits off reasoning
  (think tags / harmony); ASR is computed on the final answer only.
- `experiments/<run_id>/{config.yaml,traces.jsonl,summary.json,run.log}` and `experiments/index.json` (Step 1 runs are under `experiments/baseline/`).
  `traces.jsonl` is the checkpoint: records are validated against `tracing/schema.py` before append.
- `app/`: Streamlit dashboard (`Home.py` + `pages/`; `lib.py` = loading). **Reads only `experiments/*`.** It must never run attacks or models.
  It imports only `prag.eval.metrics`; `tests/test_app.py` checks every page with AppTest and checks the import ban.
  Without traces (public clone) it falls back to `experiments/trace_aggregates.json` (numbers only; `scripts/export_trace_aggregates.py`).
- Public repo: `.gitignore` keeps `traces.jsonl`, `upstream_format_results.json` (both hold poison texts), `run.log`, `upstream/`, `third_party/` local.
  README charts: `scripts/make_readme_figures.py` → `docs/figures/*_{light,dark}.svg`.
- `tests/`: CPU-only. `tests/toyworld.py` provides a hash encoder, an echo LLM and an overlap reranker for end-to-end pipeline tests.

## Conventions
- Structured logging (JSON lines to `run.log`, human-readable to the console). Runs are checkpointed and
  resumable per question. `--dry-run` prints the plan plus time/cost/disk estimates without loading models.
- Label every result **subset** vs the **full 100-question protocol**.
- Keep this file under 200 lines.
