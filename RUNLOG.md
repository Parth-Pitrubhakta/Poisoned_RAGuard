# RUNLOG.md

Every command that changes state, downloads, or runs an experiment is logged here (newest at the bottom).
Times are server-local. Secrets are never logged.

## 2026-11-19 — Phase 1: system survey (read-only)

Inspection only; nothing installed or downloaded. Network use was HTTP HEAD plus HF/GitHub/PyPI metadata GETs.
- `cat /etc/os-release; uname -a; systemd-detect-virt; lscpu; free -h; df -h`
- `nvidia-smi; nvidia-smi --query-gpu=...; nvidia-smi topo -m; nvcc --version`
- `<tool> --version` for python3, conda, mamba, uv, pip, git, git-lfs, gcc, g++, clang, cmake, docker, podman, ollama, nvcc
- `conda env list; <env>/bin/python -m pip list | grep ...` (system, base, ml-base, cp-vllm, venv)
- `grep` of the cp-vllm vLLM 0.28 model registry for supported architectures
- `curl -I` reachability: google, huggingface.co, github.com, pypi.org, download.pytorch.org, api.anthropic.com, api.openai.com, BEIR host
- `curl -I` BEIR `{nq,hotpotqa,msmarco}.zip` → 475 / 623 / 1032 MiB
- HF `/api/models/<id>` metadata for the candidate embedders, rerankers, and LLMs; HF `/api/models?search=` for newer families
- GitHub API `/repos/sleeepeer/PoisonedRAG` and `/contents/{,results,results/beir_results,model_configs}`
- env-var presence checks (keys: none set), `ss -ltn` (8501 free), `ollama list`, `docker ps` (no access), `who`, `uptime`
- Wrote `SURVEY.md`.

## 2026-11-19 — Phase 2 / Step 0: scaffold

Approved by user ("yes start the implementation", venv name `venv_PoisonedRAG`). Survey decisions taken as the
recommended defaults: project stays at `Cybersecurity/PoisonedRAG_implementation/poisonedrag-modern/`; uv
bootstrapped inside the project; project-local `HF_HOME`; one GPU per job with a free-VRAM check; recommended stack.

- `git clone https://github.com/sleeepeer/PoisonedRAG.git upstream` → commit `f660d72174f06b13fae5163ce656e7b235db858f`
  (2026-01-27, "fix results loading bug"), 55 MB including .git. Unmodified.
- PyPI JSON metadata lookups (no installs) for: uv, vllm (0.28–0.34 torch/transformers pins), torch,
  transformers, sentence-transformers, faiss-cpu, beir, datasets, streamlit, plotly, graphviz, pandas, numpy,
  pytest, pydantic, jsonschema, pyyaml, tqdm, orjson, huggingface-hub, pyarrow, scipy.
- Checked `cp-vllm` env: PyPI `torch==2.13.0` ships with `nvidia_cuda_runtime 13.0` (cu130), which is compatible with driver 580 / CUDA 13.0.
- Created scaffold: `CLAUDE.md README.md RUNLOG.md pyproject.toml .python-version .gitignore scripts/env.sh
  configs/{models.yaml,baseline_nq_smoke.yaml} src/prag/{data,retrieval,attack,llm,eval,defenses,tracing}/ experiments/ app/ tests/ data/`.
- ⛔ Checkpoint 0 presented; user replied "approve install" (+ notes: verify vLLM archs, verify corpus join, verbatim fastchat template, df -h before weights).

## 2026-11-19 — Step 0: install (approved)

- `python3 -m venv .bootstrap` → **failed**: system Python lacks ensurepip (fixing it needs `apt install python3.10-venv` + sudo, which is not allowed).
  Removed the partial `.bootstrap/`; instead: `python3 -m pip install --no-user --target .bootstrap/pkgs uv==0.13.7`
  and `ln -s ../pkgs/bin/uv .bootstrap/bin/uv` (same uv 0.13.7, still project-local, no system change).
- `uv python install 3.12` → CPython 3.12.16 in `.cache/uv-python`
- `uv lock` → 220 packages; all 14 top-level pins locked exactly as pinned (0 mismatches). CUDA runtime 13.0.96, cuDNN 9.20, NCCL 2.29.7.
- `uv sync` → `venv_PoisonedRAG/` (8.6 GB), 3m09s
- Verify: torch 2.13.0+cu130, cuda 13.0, 4× H200 visible (~139 GiB free each), bf16 GPU matmul OK. vLLM 0.32.0 registers
  GptOssForCausalLM, Qwen3_5ForConditionalGeneration, Gemma4UnifiedForConditionalGeneration (+ Gemma4ForConditionalGeneration), LlamaForCausalLM, Qwen3ForCausalLM (391 archs total).

## 2026-11-19 — Step 1 prep

- `df -h` before any weights: project, venv, and every cache path resolve to `/DATA2` (95 TB free). `$HOME` is also on /DATA2. TRANSFORMERS_CACHE unset.
- `curl -fL .../BEIR/datasets/nq.zip` → 498,307,926 B, sha256 `2553bf7bfbab47b1436ca00a34bce57320e18e611fd00999a2a3a1b4714be752`;
  unzipped to `data/beir/nq` (corpus 2,681,468 / queries 3,452 / qrels test 4,202 lines; 1.5 GB). Zip removed.
- `snapshot_download facebook/contriever@2bd46a25019aeea091fd42d1f0fd4801675cf699` (pytorch_model.bin 0.438 GB, sha256 matches the LFS oid).
- Fetched FastChat v0.2.36 `conversation.py` + `model/model_adapter.py` (scratchpad, read-only reference) → vicuna_v1.1 copied verbatim into `src/prag/llm/chat_formats.py`.
- `snapshot_download lmsys/vicuna-7b-v1.3@236eeeab… allow=[*.json, tokenizer.model]` (tokenizer/config only, < 1 MB).
  Transformers-5 tokenizer == [BOS]+SentencePiece on 100/100 NQ poisoned prompts → token ids are identical to upstream's slow tokenizer.
- `uv run python scripts/check_corpus_join.py --n 2` (N=5 and N=1) → PASS. Precomputed doc ids → BEIR text join reproduces upstream
  Contriever dot scores to |Δ| ≤ 6e-7 (title+" "+text, max_len 512). Clean passages reach the prompt (no-attack 5/5; poisoned N=1: 4 clean + 1 poison).
- `uv run pytest` → 20 passed (clean_str / f1_score / MULTIPLE_PROMPT / wrap_prompt cross-checked against upstream's own code).
- `uv run python -m prag.baseline --config configs/baseline_nq_smoke.yaml --dry-run` → OK.

## 2026-11-19 — Step 1: faithful baseline (approved "approve Vicuna")

- `snapshot_download lmsys/vicuna-7b-v1.3@236eeeab96f0dc2e463f2bebb7bb49809279c6d6 allow=[*.json, tokenizer.model, pytorch_model-*.bin]`
  → 149 s, shards 9,976,634,558 B + 3,500,315,539 B (match HF metadata). `df -h /DATA2` before/after: 95 TB free.
- `uv run python -m prag.baseline --config configs/baseline_nq_smoke.yaml` → 64 s wall, GPU0 (auto-picked, 139.8 GB free), freed afterwards.
  run_id `baseline_nq_contriever_vicuna7b_smoke-20261119-090636`. Smoke test non-empty (14 tokens, finish=stop).
  **Subset (2 Qs, 1 iter), NQ, Contriever dot, top_k=5, N=5, Vicuna-7B v1.3: ASR 1.00, Precision 0.80, Recall 0.80, F1 0.80**; clean (no-attack) accuracy 0.50.
- ⛔ Checkpoint 1 presented.

## 2026-11-19 — Step 1b: full 100-question NQ baseline (user chose option a)

- `src/prag/baseline.py`: added `targets.iterations` (= upstream repeat_times; one poison pool of M*N texts per iteration, as upstream). New `configs/baseline_nq_full.yaml` (M=10, iterations=10). Smoke config unchanged (iterations defaults to 1).
- `uv run pytest` → 20 passed; `--dry-run` → 201 LLM calls, $0, weights cached.
- `uv run python -m prag.baseline --config configs/baseline_nq_full.yaml` → 57 s wall, GPU0, freed afterwards. run_id `baseline_nq_contriever_vicuna7b_full-20261119-120126`.
  **Full protocol (10 iter × M=10), NQ, Contriever dot, top_k=5, N=5, Vicuna-7B v1.3: ASR 0.89, Precision 0.96, Recall 0.96, F1 0.96**; clean accuracy 0.57.
  ASR per iteration [1.0, 1.0, 0.9, 0.9, 0.9, 0.8, 0.7, 0.8, 0.9, 1.0]. Paper Table 1 (NQ black-box, Vicuna-7B): ASR 0.88, F1 0.96 (arXiv HTML, fetched 2026-11-19).
  11 failures: 2 are substring-metric false negatives ("two seasons" vs "2"; "100 years after the events of Return of the Jedi"); 9 are parametric-knowledge resistance on famous facts (5 of those 9 had all 5 own poisons in the top-5). 0 empty outputs.

## 2026-11-19 — Step 2: modular reimplementation

- New modules: `config.py` (pydantic), `pipeline.py`, `run.py` (CLI: --dry-run / --resume), `build_index.py` (sharded FAISS),
  `retrieval/{encoders,index,reranker}.py`, `llm/{backends,parsers}.py`, `defenses/base.py` (only `none`), `tracing/{schema,writer}.py`.
  Edited: `llm/base.py` (Generation.raw_text/reasoning, pricing), `llm/vllm_backend.py` (chat_template_kwargs, output parser),
  `retrieval/merge.py` (Passage.rerank_score), `configs/models.yaml` (Contriever revision pinned). No new dependencies.
- `CUDA_VISIBLE_DEVICES="" uv run pytest` → 45 passed (20 existing + 25 new: config validation, trace schema, parsers,
  FAISS vs brute force, rerank, sharded index build, end-to-end CPU pipeline, resume, CLI).
- `uv run python -m prag.run --config configs/baseline_nq_smoke.yaml --dry-run` → 5 calls, $0, 0 GB download, within guardrails.
- `uv run python -m prag.run --config configs/baseline_nq_smoke.yaml` → 46 s, GPU0. run_id `baseline_nq_contriever_vicuna7b_smoke-20261119-121220`.
  ASR 1.00, P/R/F1 0.80 (same as Step 1). Traces: 0 schema errors.
- `uv run python -m prag.run --config configs/baseline_nq_full.yaml` → 55 s, GPU0. run_id `baseline_nq_contriever_vicuna7b_full-20261119-121322`.
  ASR 0.89, P/R/F1 0.96, clean acc 0.57. `scripts/compare_runs.py` vs the Step 1 run: 100/100 identical retrieved top-k, prompts,
  model input, poisoned + clean outputs, and success.
- Resume check: copied that run to `experiments/_resume_check/`, kept 47 traces plus a torn 48th line, `prag.run --resume` → regenerated 53,
  100/100 identical to the uninterrupted run. Removed `experiments/_resume_check/`.
- (A one-off comparison script was briefly written to the session scratchpad outside the project, then moved to `scripts/compare_runs.py`.)
- ⛔ Checkpoint 2 presented.

## 2026-11-19 — Step 3 (approved: A, B, C, D in order; A gates B)

User notes: C includes Vicuna-7B (2 retrievers × 3 LLMs); temperature 0.1 for all LLMs (flag degradation, don't tune);
ASR on the post-split final answer for Qwen3.5-27B / gpt-oss; check that all 4 GPUs are idle before a sharded embed.

- Prep: `scripts/check_dense_vs_precomputed.py`, `scripts/make_matrix_configs.py` (→ `configs/matrix/*.yaml`, 9 configs, all validate),
  `scripts/make_results.py`, `configs/nq_contriever-dense_vicuna7b_full.yaml`. `vllm_backend`: stop on the model's generation_config
  EOS ids (gpt-oss `<|return|>`; Vicuna eos=2 = tokenizer eos, so no change), `engine_kwargs` from the registry, and Qwen `<think>`
  opened in the prompt is restored before parsing. Registry: pinned revisions for Qwen3.5-27B fc05dae…, gpt-oss-120b b5c939d…,
  Qwen3-Embedding-8B 1d8ad4c…, Qwen3-Reranker-4B 22e6836… (HF API, all ungated, Apache-2.0).
- Upstream `evaluate_beir.py` check: DenseEncoderModel default max_length 512 (the CLI `--max_length 128` is never passed), title+" "+text, no norm, dot.
- A: `nvidia-smi` → all 4 GPUs idle (14 MiB, 0%). Launched 4 × `CUDA_VISIBLE_DEVICES=i prag.build_index --retriever contriever --num-shards 4 --shard i --batch-size 128`.
  Measured ~440 docs/s/GPU (estimate was 4000) → ~24 min. Killed at the first chunk, added longest-first length sorting (as BEIR DRES), relaunched 12:23.
  Still ~550–700 docs/s: NQ passages are uniform ~100-word DPR chunks, so fp32 compute is the bottleneck, not padding. Projected ~18 min (> 10 min guardrail);
  asked the user → "let it finish" (keeps exact fp32; TF32/bf16 would weaken the faithfulness check).
- Note: transformers 5 loads Contriever weights from the HF auto-conversion `refs/pr/2/model.safetensors` (same tensors as the pinned .bin; Step 1 scores matched upstream to 6e-7).
- A: shards finished 12:34:50 (11.5 min wall), `prag.build_index ... --merge` → `data/indexes/nq-contriever/index.faiss` (2,681,468 × 768, 7.7 GB; shards 4 × 2.0 GB kept).
- A: `CUDA_VISIBLE_DEVICES=0 uv run python scripts/check_dense_vs_precomputed.py --index data/indexes/nq-contriever --precomputed upstream/results/beir_results/nq-contriever.json --targets upstream/results/adv_targeted_results/nq.json --all-queries`
  → **PASS**. 100 target queries: top-5 100/100 exact, top-20 100/100 exact, top-100 93 exact + 7 near-tie reorders (|Δ| ≤ 1e-4); max |score diff| 1.1e-6.
  All 3,452 NQ queries: 0 mismatches; top-5 3432 exact + 20 near-ties; max |score diff| 3.3e-6.
- A: first `prag.run --config configs/nq_contriever-dense_vicuna7b_full.yaml` → **segfault (exit 139) in faiss search**. Reproduced: faiss imported before torch,
  then CUDA encoder → crash; torch first → OK (faiss-cpu bundles its own libgomp). Fix: `retrieval/index.py::import_faiss()` always imports torch first. Verified, then removed the repro script.
- A: `prag.run --config configs/nq_contriever-dense_vicuna7b_full.yaml` → 2m29s, GPU1. ASR 0.89, P/R/F1 0.96, clean 0.57.
  `scripts/compare_runs.py` vs the precomputed run → **100/100 identical** (top-k, prompts, model input, both outputs, success). A gates B: passed.
- B: `df -h` → project, $HOME, HF_HOME on /DATA2 (95 TB free). `scripts/download_model.py llms qwen3.5-27b` and `... gpt-oss-120b` (pinned revisions;
  ignore original/*, metal/*) started 12:44 in parallel.
- (Guardrail slip: the faulthandler debug log for the segfault was written to the session scratchpad `dense.log`, outside the project; deleted.)
- B: downloads done: Qwen3.5-27B 55.6 GB / 766 s; gpt-oss-120b 65.3 GB / 993 s (top-level only).
  Templates: Qwen3.5 thinking ON by default (prompt ends "<think>\n"); gpt-oss reasoning_effort default "medium". Vendor sampling defaults (Qwen 0.6/top_k 20/top_p 0.95) NOT used: held at 0.1/50/1.0 per user.
- `pipeline`: generation now batches `llm.batch_questions` (default 50) questions, poisoned + clean prompts in one call (checkpoint unit); was 10/iteration.
  Re-verified: `prag.run --config configs/matrix/nq_contriever_vicuna7b.yaml` → 100/100 identical to the Step 2 run (also confirms the explicit EOS stop ids are a no-op for Vicuna).
- B: `prag.run --config configs/matrix/nq_contriever_qwen35_27b.yaml` (GPU0, 12:59) and `... nq_contriever_gptoss120b.yaml` (13:01) running in parallel.
- B: Contriever × Qwen3.5-27B done 13:11 (12 min incl. load): ASR 0.79, P/R/F1 0.96, clean acc 0.42.
- **INCIDENT (unapproved download).** The gpt-oss run stalled after model init. Its vLLM loader was downloading the repo's `original/` checkpoint
  (7 shards, 65.2 GB) even though `download_model.py` had skipped it. Found 13:12 via `.incomplete` blobs + HTTPS connections; killed the run (pid 198770 / EngineCore 202634).
  Deleted the `original/` snapshot links, the model blob entries, and the 7 orphaned shards in the shared store (`.cache/hf/hub/blobs/`, 65.2 GB). Store 187 → 126 GB
  (= contriever 0.8 + vicuna 13 + gpt-oss 61 GiB + Qwen3.5 52 GiB). 0 broken links, 0 .incomplete files. Removed the aborted run dir (config.yaml only).
  Prevention: `config.force_offline()` sets HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 in `prag.run`, `prag.build_index` and `check_dense_vs_precomputed.py`.
  Runs can no longer download; weights come only from `scripts/download_model.py`. Tests: 45 passed.
- B: Contriever × gpt-oss-120b (offline relaunch, GPU1, 13:15–13:20): ASR 0.51 (exact), P/R/F1 0.96, clean acc 0.36. No downloads.
  Finding: gpt-oss emits U+202F narrow no-break spaces between words (55/100 poisoned outputs), so upstream's exact substring check misses multi-word targets.
  Added a secondary, clearly labelled `eval.metrics.normalize_typography` (NFKC + ASCII quotes/dashes + whitespace collapse; no number words) → typography-normalized ASR 0.89, clean acc 0.62.
  Upstream-exact stays the headline. Qwen3.5-27B degradation at T=0.1: 20% poisoned / 26% clean hit the 4096-token cap in "Wait, I'll check…" loops (empty final answer);
  ASR on answered questions 0.99. Flagged, not tuned. Tests: 46 passed. `scripts/make_results.py` regenerates the RESULTS tables from traces.
- C: `scripts/download_model.py retrievers qwen3-emb-8b` → 15.2 GB in 187 s. GPU check: GPU0 has a foreign process (~5 GB, pid not visible to us) → sharding across GPUs 1–3 only (user rule).
  `CUDA_VISIBLE_DEVICES={1,2,3} prag.build_index --retriever qwen3-emb-8b --num-shards 3 --shard {0,1,2} --batch-size 128` started.
- C: measured ~64 docs/s/GPU on the longest-first chunk (~21k tok/s/GPU, compute-bound bf16). NQ sample (1%): mean 113 tokens, median 97, 0.23% at the 512 cap
  → ~80 min total on 3 GPUs (estimate was ~25). Asked the user → "let it finish". GPU0's foreign process ended during the build (GPU0 now free; not added mid-job).
- D: `download_model.py rerankers qwen3-reranker-4b` → 8.1 GB / 84 s. `scripts/check_reranker.py` on GPU0 (a foreign process was using 19.8 GB there) → PASS (relevant 0.998 / 0.995, irrelevant 0.000).
- C: merged index 2,681,468 × 4096 (44 GB; shards 44 GB kept). `scripts/retrieval_quality.py` (GPU1, no poison, 3,452 NQ queries): Qwen3-Emb-8B nDCG@10 0.644 / R@5 0.758 / R@100 0.974 vs Contriever 0.254 / 0.293 / 0.771 (matches BEIR's published unsupervised-Contriever NQ 0.254). Added `prag.run --gpu`. Launched C (3 runs) on GPUs 1/2/3 in parallel; GPU0 still has a foreign 19.8 GB process.
- C done: Qwen3-Emb-8B × Vicuna ASR 0.85 / F1 0.91 / clean 0.74 (2m13s); × gpt-oss ASR 0.43 / clean 0.49 (4m27s); × Qwen3.5 ASR 0.72 / clean 0.66 (10m21s). D launched: rerank × Vicuna (GPU1) + × gpt-oss (GPU3) 15:57, × Qwen3.5 (GPU2) 15:58.
- D done: Qwen3-Emb-8B + Qwen3-Reranker-4B × Vicuna ASR 0.81 / F1 0.92 / clean 0.72; × Qwen3.5 0.66 / clean 0.67; × gpt-oss 0.39 / clean 0.51. No downloads (0 .incomplete; HF store 148 GB = previous 126 + emb-8B + reranker).
- `scripts/make_results.py` → experiments/RESULTS.md tables. Analysis (inline python, read-only on traces): exact McNemar per step (none significant at 0.05);
  reranker gives kept poison P(yes) 0.999 vs clean 0.998, poison is top-1 for 89/100. Wrote RESULTS Findings, Cost and time; README status + usage.
- ⛔ Checkpoint 3 presented.
- User: "delete the shards" → `rm data/indexes/{nq-contriever,nq-qwen3-emb-8b}/emb_*.npy` (4 × 2.0 GB + 3 × 14.6 GB = 52 GB). index.faiss / doc_ids.json / meta.json kept; rebuilding shards would need a full re-embed.

## 2026-11-19 — Step 4: Streamlit dashboard (approved)

- `scripts/make_dataset_card.py --dataset nq` → `experiments/datasets.json`; `scripts/retrieval_quality.py ... --out experiments/retrieval_quality.json` (GPU1, 2.4 min; same numbers as Step 3).
- `app/Home.py`, `app/pages/{1_Architecture,2_Datasets,3_Attack_Flow,4_Results,5_Defenses}.py`, `app/lib.py`, `.streamlit/config.toml`.
  Charts use the dataviz skill's documented default palette (slots 1–3; documented as validated all-pairs in light + dark).
  The skill's validator could not run: system node v12 lacks `??=`, and a newer node would be a system install, so not done.
- `tests/test_app.py` (AppTest, toy experiments dir): every page renders + banner present; attack flow shows the verdict; Results scope toggle; read-only import ban → 9 passed.
  AppTest over the real `experiments/`: 14 runs × 6 pages, 0 failures (after handling Step 1 traces whose `defense` is a plain string).
- `uv run streamlit run app/Home.py --server.headless true --server.address 127.0.0.1 --server.port 8501` (pid 766039) →
  `/_stcore/health` ok; Home/Architecture/Datasets/Attack_Flow/Results/Defenses HTTP 200; bound to 127.0.0.1 only. `~/.streamlit/machine_id_v4` predates the project (2026-09-16).
- Headless Firefox screenshot attempt (profile/HOME under `.cache/`) produced no image; removed those dirs. No visual check beyond AppTest.
- ⛔ Checkpoint 4 presented.

## 2026-11-20 — Step 5: defenses (approved: paper's four + TrustRAG; RobustRAG only if cheap; $0, offline)

- GPU check 00:3x: a foreign process (pid 1000236, not ours) holds 64 / 40 / 40 / 36 GB on GPUs 0–3, 0% util → ~79 / 103 / 103 / 107 GB free.
- Read the paper's Sec. 7 (arxiv.org/html/2402.07867). Paraphrasing: GPT-4, 5 paraphrases, used for retrieval + generation; prompt in App. G.
  PPL: "cl100k_base model from OpenAI tiktoken" (a tokenizer, not an LM). Duplicate filter: SHA-256. Knowledge expansion: k=50.
- TrustRAG reference: `curl` of `defend_module.py` + `main_trustrag.py` from github.com/HuichiZhou/TrustRAG @ 11dcea0 → `third_party/trustrag_ref/` (47 KB, reference only, never imported/executed).
  (First attempt with `run.py` + tree listing hung on GitHub; stopped, only those 2 files kept.)
- Code: `src/prag/defenses/{base,filters,perplexity,trustrag}.py`; pipeline hooks (rewrite_query / prepare+filter_batch+release / answer);
  `LLMCfg.max_model_len` override; per-call `max_new_tokens`; prompt-length guard in the vLLM backend; config validators
  (paraphrase ⇒ dense_index; knowledge_expansion ⇒ params.k == rag.top_k). `tests/test_defenses.py` (TrustRAG prompts == reference f-strings via ast) → 69 passed.
- Deviations forced by no-download: TrustRAG embedder = local Contriever (ref: SimCSE, not cached); ROUGE-L re-implemented without Porter stemming
  (`rouge_score`/`nltk` not installed); PPL LM = local Vicuna-7B; paraphraser = local Qwen3.5-27B (thinking off).
- `scripts/check_trustrag_filter.py --run experiments/nq_contriever_vicuna7b_full-20261119-125748 --gpu 0` (no LLM): stage 1 keeps 44/479 poison,
  removes 9/21 clean under attack and 44/500 clean passages without attack; poison-pair cosine median 0.85 (ref threshold 0.88 was tuned for SimCSE).
- `scripts/make_paraphrases.py --gpu 1` (00:50–00:51) → `experiments/defenses/paraphrases_nq.json`: 100/100 parsed on seed 100, 0 retries. 0 `.incomplete` blobs.
- `scripts/make_defense_configs.py` → 51 configs in `configs/defenses/{dup,ppl,para,ke10,ke50,trustrag}/`; all pass `--dry-run`, download 0 GB.
  Large-model configs use gpu_memory_utilization 0.68 / min_free_gb 97 because of the foreign process.
- Queues: `configs/defenses/queue_small.txt` (15 Vicuna jobs → GPU0), `queue_big.txt` (36 Qwen3.5 / gpt-oss jobs → GPUs 1–3, longest first),
  run by `scripts/run_queue.py`. Estimate ≈ 400 GPU-min, ~2–2.5 h wall.
- 00:53 GPU0 worker started on queue_small (pilot: first jobs validate every code path on Vicuna).
- 00:58 Pilots OK (dup = Step 3 on 100/100 outputs; ppl, trustrag ran clean). GPU1/2/3 workers started on queue_big.
- 00:58 two failures (no generation happened):
  (1) `trustrag/nq_qwen3emb8b_qwen35_27b` on GPU2: vLLM "free memory 84.87 GiB < 0.68 × 139.8". Cause: `pipeline._release(enc)` only deleted its own
      local name, so the Qwen3-Emb-8B encoder (~15 GB) and the reranker stayed alive in `retrieve()` until it returned. This was harmless on idle GPUs in Step 3 and fatal now.
      Fix: `enc = None` / `rr = None` / `llm = None` in the caller, then `_release()` (gc + empty_cache). Tests 69 passed. Removed the failed run dir (config.yaml + run.log only) and re-queued it.
  (2) `ke10/nq_contriever_vicuna7b` on GPU0: the new length guard caught 8/100 prompts > Vicuna's 2048-token context (longest 2949). Knowledge expansion
      is therefore N/A for Vicuna-7B: dropped the 3 Vicuna ke10 jobs and removed the failed run dir. KE runs only for Qwen3.5-27B / gpt-oss-120b (k=10, k=50).
- 01:00 same memory failure for `trustrag/nq_qwen3emb8b-rerank4b_qwen35_27b` (GPU3; started before the fix). Removed its dir, re-queued.
- 01:14 `ke50/nq_qwen3emb8b_qwen35_27b` (GPU2) stopped by the smoke test: Qwen3.5-27B spent all 4096 tokens reasoning on test1 with 50 passages
  (finish=length, empty final answer). This is the known T=0.1 truncation degradation, not a template mismatch, so the smoke test was a false positive.
  Narrowed it: raise only when nothing was generated (no tokens, no reasoning, no raw text); otherwise log `smoke_test_truncated` and continue. Sampling unchanged. Removed dir, re-queued.
- 01:23 `ke50/nq_qwen3emb8b-rerank4b_qwen35_27b` (GPU2; started before the smoke-test fix) hit the same false positive. Removed its dir, re-queued.
- Noise-floor control (dup filter changes ≤1 prompt): Contriever × Vicuna-7B reproduces Step 3 on 100/100 outputs; Contriever × Qwen3.5-27B gives
  99 identical prompts but only 85 identical final answers (46 identical raw completions), with 7 attack-success flips and 2 clean-correct flips.
  Cause: vLLM is not batch-invariant (KV size 0.68 vs 0.75 util, batch composition) and 4096-token reasoning amplifies tiny numeric differences
  before a seeded sample diverges. ⇒ For reasoning models, |ΔASR| ≲ 0.05 between runs is noise, not a defense effect.
- 02:53 queues empty: 48/48 Step 5 runs complete (100 Q each). 0 `.incomplete` blobs, HF store 148 GB (unchanged), GPUs back to the foreign process only.
  Queue jobs used 19,650 GPU-s (5.5 GPU-h) including 5 aborted attempts; wall time 00:53–02:54.
- `scripts/make_defense_results.py` → `experiments/defenses/defense_table.{json,csv}` + RESULTS DEFENSE block (Tables D1, D2, D2b, D3, PPL AUC).
  Added answered-only ASR / empty-final columns after seeing that Qwen3.5 truncates on 45–67% of prompts at k = 10–50.
  `scripts/make_results.py` now skips defended runs. Its TABLES block was verified byte-identical to the pre-Step-5 version.
- Guardrail slip: a temporary copy of RESULTS.md for that comparison went to the session scratchpad (outside the project). Deleted.
- RESULTS: Step 5 protocol, deviations, findings, limits, cost. README status → Checkpoint 5a; CLAUDE.md code layout.
- Not run: RobustRAG (needs a spaCy/NLTK install → approval). Dashboard not wired yet (user: stop before wiring).
- ⛔ Checkpoint 5a presented.

## 2026-11-20 — Step 5b: faithful TrustRAG embedder, RobustRAG note, Defenses page (approved)

- Registry: `simcse-sup-bert` = princeton-nlp/sup-simcse-bert-base-uncased @ 2d82fab (HF API, 03:3x). The repo has only `pytorch_model.bin` + Flax,
  so `download_model.py` now honours a per-entry `download_ignore`. `df -h /DATA2`: 95 TB free.
- `scripts/download_model.py retrievers simcse-sup-bert` → 6 files, 438 MB in 14 s (store +438,262,007 B), 0 `.incomplete`. No other downloads.
- `retrieval/encoders.SimCSECLS`: last-layer [CLS], fp32, one sentence at a time (as the reference `get_sentence_embedding`).
- Offline stage-1 check (`check_trustrag_filter.py --embedder simcse-sup-bert`, GPU0; outputs `experiments/defenses/trustrag_filter_simcse_*.json`):
  Contriever top-k keeps 5/479 poison (Contriever embedder: 44/479) and removes 47/500 clean without attack. Qwen3-Emb-8B: 12/453, 114/500;
  rerank: 9/462, 124/500. Poison-pair cosine median 0.93 vs clean 0.74–0.81, so the reference 0.88 threshold works as tuned.
- `make_defense_configs.py` → `configs/defenses/trustrag-simcse/` (9 configs, embedder only changed); all dry-run clean, 0 GB download.
  `make_defense_results.py` now labels TrustRAG by embedder (SimCSE = headline, Contriever = substitute row).
- 03:38 4 workers started on queue_simcse_{small,big} (3 Vicuna on GPU0; 6 Qwen3.5/gpt-oss on GPUs 1–3).
- 03:39 GPU1–3 workers had exited 127 (`uv` not on PATH: env.sh was sourced inside the backgrounded subshell). Nothing was popped; relaunched.
- 04:07 SimCSE TrustRAG: 9/9 runs complete (100 Q each), 1.4 GPU-h, 03:37–04:07. No further downloads (0 `.incomplete`, store 148 GB). GPUs back to the foreign process only.
  Result (robust ASR, mean over setups): Vicuna 0.85 → 0.09, Qwen3.5 0.90 → 0.08, gpt-oss 0.79 → 0.09 (Contriever substitute: 0.16 / 0.10 / 0.15).
  Paired per cell vs the Contriever pass: ASR lower in 9/9, significant in 2 (Contriever × Vicuna p = 0.039, rerank × Vicuna p = 0.002); clean accuracy differs in none (p ≥ 0.11).
- RESULTS: TrustRAG headline = SimCSE; Contriever pass kept as a comparison row; RobustRAG out-of-scope paragraph (minority-poison precondition fails at 4.5–4.8/5).
  `make_defense_results.py`: robust ASR per row, discordant counts, measured noise floor in `defense_table.json`.
- Dashboard: `app/pages/5_Defenses.py` rewritten. Headline metrics; metric / setup / substitute filters; ASR and clean-accuracy dumbbells per LLM with a ±0.05 noise band;
  poison-removal bars; caveat boxes (KE × Qwen3.5 truncation, TrustRAG clean = no loss, noise floor, N/A + RobustRAG); SimCSE vs Contriever table;
  every run + CSV; jump to Attack flow. `3_Attack_Flow.py`: kept/removed column, paraphrased query, TrustRAG stages.
  Tests: 2 new app tests (toy defended run + synthetic table) → 71 passed. AppTest on the real experiments/: 6 pages, all Defenses views, 0 exceptions.
  Live server (pid 766039) still healthy and serves the new pages; no restart needed.
- First launch of the SimCSE GPU1–3 workers exited 127 (env.sh sourced inside a backgrounded subshell); nothing popped; relaunched.
- ⛔ Checkpoint 5b presented.

## 2026-11-20 — Final write-up: NQ study complete (approved "finalize the NQ study")

- Read-only analysis (CPU, no downloads): ROUGE-L (`prag.defenses.trustrag.rouge_l_f`) over each question's 5 released poison texts (first 100 NQ targets):
  all 1,000 pairs > 0.25 (median 0.49, min 0.28) → 100/100 questions trip TrustRAG's n-gram precheck; with the question prefix stripped, 79% of pairs
  still > 0.25. Stage-1 branch counts from `experiments/defenses/trustrag_filter_simcse_*.json`: "both dense → drop all" on 65–78% of questions; poison-pair SimCSE cosine p10 = 0.88 (= threshold).
- Pooled exact McNemar (post hoc) on Vicuna-7B clean accuracy, TrustRAG SimCSE vs Step 3 twins, two modern-retriever cells: 0.73 → 0.625, 42 lost / 21 gained, p = 0.011
  (Contriever substitute: p = 0.022). **Correction:** the earlier "TrustRAG costs no clean accuracy" holds only for the reasoning models. Fixed in RESULTS Step 5 Finding 1
  and in the Defenses page (headline metric delta + caveat box).
- RESULTS: new "Limitations and future work (NQ study)" section (headline: TrustRAG works only against near-duplicate poison; adaptive diversified poison untested).
- README: final pass. Status = NQ study complete; headline-result table; exact end-to-end reproduction commands (env → upstream/BEIR → weights → indexes → Steps 1–5 → dashboard).
- `uv run pytest` → 71 passed. AppTest on `5_Defenses.py` with real data: 0 exceptions. Live server (pid 766039) healthy.
- ✅ NQ study complete. No further work started.

## 2026-11-20 — TrustRAG architecture figure (docs/figures/; read-only w.r.t. results, $0, CPU, no downloads)

- Grounded in `src/prag/defenses/trustrag.py`, `src/prag/pipeline.py`, `configs/defenses/trustrag-simcse/*.yaml`, RESULTS Step 5 + Limitations,
  and `experiments/defenses/trustrag_filter_simcse_*.json` (read only). Clean pairs under attack: 21/47/38 per 100 q (mean 0.35), 22/106 dropped.
- Spec corrections, taken from the code: the 0.88 cutoff applies to a K-means cluster's *mean* pairwise SimCSE cosine, not to single pairs; ROUGE-L 0.25 is an any-pair gate plus an in-cluster filter;
  "0.95 clean dropped" is the no-attack path (under attack: 0.07 of 0.35); Contriever cells use upstream's precomputed top-100, not FAISS;
  poison is scored by the same encoder and merged, not indexed; Stage 2 = 3 calls to the target LLM itself.
- New: `docs/figures/trustrag_architecture.svg` (primary, hand-authored), `.tex` (TikZ 1:1 port), `.dot` (Graphviz), `docs/figures/README.md`,
  `scripts/figure_check.py` (SVG parse + ref check + librsvg render; DOT structural check; TikZ build + Type 3 font check).
- Tools: Firefox is only a snap stub and Graphviz `dot` is absent, so the SVG renders via system librsvg 2.52 (python3-gi). Nothing was installed.
  `source scripts/env.sh && /usr/bin/python3 -I scripts/figure_check.py` → SVG well-formed, 7/7 refs resolve, 1×/2× PNG in `.cache/figcheck/`;
  TikZ: pdflatex exit 0, 0 font warnings, fonts all Type 1 (EC typewriter bitmap replaced by `courier`), page 1199.5 × 797 pt; DOT: 20 nodes, 21 edges, balanced, **not rendered**.
- Fixes found by rendering: XML comments containing `--`; connectors drawn under the panel fills (moved last); one caveat line overflowed.
- TeX font cache (mktexpk ectt1000, before the switch to Courier) went to `.cache/texmf-var/`. RESULTS and the dashboard are unchanged.

## 2026-11-22 — Dashboard shared on the campus network (user approved "Campus network link")

- Stopped the localhost-only server (pid 766039). Relaunched: `nohup setsid uv run streamlit run app/Home.py --server.headless true --server.address 0.0.0.0 --server.port 8501`
  (pid 1924748, log `.cache/streamlit_lan.log`). Reachable at http://<campus-ip>:8501 (private campus address; no password; Attack flow shows the benchmark poison texts).
  `.streamlit/config.toml` unchanged (still 127.0.0.1), so any restart without the flag is localhost-only again. HTTP 200 on / and /Defenses via the LAN IP.

## 2026-11-23 — Dashboard moved to port 8502 (user request: 8501 taken)

- The only listener on 8501 was our own dashboard (pid 1924748, started 2026-11-22). Stopping it was blocked by the permission check, so it is still running.
- Default port changed 8501 → 8502 in `.streamlit/config.toml`, `README.md` (launch + port-forward lines), `CLAUDE.md`, `app/Home.py` docstring. SURVEY.md left as a historical record.
- Launched: `nohup setsid uv run streamlit run app/Home.py --server.headless true --server.address 0.0.0.0 --server.port 8502` (pid 2683804, log `.cache/streamlit_lan_8502.log`).
  HTTP 200 on / via 127.0.0.1 and http://<campus-ip>:8502 (also /Defenses). Same campus-network exposure as before; no password.

## 2026-11-25 — GitHub preparation + README rewrite (user: "give me steps, I'll push"; no git commands run by Claude)

- `scripts/make_readme_figures.py` (CPU, read-only on results) → `docs/figures/{asr_matrix,defenses,trustrag_overlap}_{light,dark}.svg`.
  New number: ROUGE-L > 0.25 for 43/100 *clean* no-attack top-5 sets (Qwen3-Emb-8B), vs 100/100 poison sets (as injected and prefix-stripped). Previews rendered to `.cache/figcheck/`.
- Dashboard without traces (simulated public clone, copy under `.cache/`): Results page crashed (KeyError `asr_normalized`).
  Fix: `app/lib.py::normalized_stats` falls back to `experiments/trace_aggregates.json` (numbers only) from new `scripts/export_trace_aggregates.py` (71 runs).
  All 6 pages + every Results metric view: 0 exceptions without traces. `pytest` → 71 passed.
- `.gitignore`: + `upstream/`, `third_party/`, `experiments/**/traces.jsonl`, `experiments/**/upstream_format_results.json` (embeds `injected_adv`),
  `experiments/**/run.log`, queue/paraphrase logs. Dry run with a throwaway git dir in `.cache/` (deleted): 325 files, 2.4 MB;
  scanned against all 500 released NQ poison texts → 0 files contain any.
- Checked via web: HuichiZhou/TrustRAG is MIT; arXiv 2501.00879 author list (citation); raw URLs of the two reference files at 11dcea0 → 200.
- README rewritten (findings, figures, mermaid diagrams, dashboard quick start, reproduction in collapsible steps; pytest moved after the upstream clone).
  CLAUDE.md + docs/figures/README.md updated. Headless Firefox screenshots not possible (snap needs $HOME under /home); screenshots left to the user.
- User added `docs/img/dashboard_{defenses,results}.png` and enabled them in the README (checked: no poison text, address bar shows localhost).
- Redaction before publishing (user-approved step 2): server IP → `<campus-ip>` in RUNLOG (2 lines); SURVEY hostname → `(redacted)`, username → `<user>`.
  Rescan of every file that would be committed: no IP, hostname, home path or username left.
- README clone links → github.com/Parth-Pitrubhakta/Poisoned_RAGuard (user-provided).
- README title → "Poisoned RAGuard" (user request); tagline "six defenses" corrected to five.
- GitHub metadata: `CITATION.cff` (author from the GitHub username; PoisonedRAG + TrustRAG as references; YAML parses) and
  `docs/img/social_preview.{svg,png}` (1280×640, rendered with system librsvg). Numbers from RESULTS; no poison text.
