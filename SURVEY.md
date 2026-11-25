# SURVEY.md — Phase 1 System Survey (read-only)

Project: Reproduce & modernize **PoisonedRAG** (Zou et al., USENIX Security 2025, arXiv:2402.07867), defensive research only.
Survey date: 2026-11-19. Nothing was installed or downloaded. The only network calls were HTTP HEAD requests and Hugging Face / GitHub **metadata** API reads.

> **Hardware Tier: TIER 3.** 4× NVIDIA H200 (141 GB HBM3e each, 564 GB total), all four linked to each other by NVLink (NV6), 1 TB RAM, 95 TB free disk.
> Hardware won't limit anything here. The real limits are (a) this is a **shared** machine, (b) **no API keys** are set, and (c) **uv is not installed**.

---

## A. OS / platform

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS (jammy) |
| Kernel | 6.8.0-138-generic, x86_64 |
| Host type | Bare metal (`systemd-detect-virt` = none, no `/.dockerenv`, not WSL) |
| Hostname | (redacted) |
| Session | Remote over SSH, headless (`DISPLAY` unset, tty session), editor = VS Code Remote |
| User groups | `<user>`, `students` → **shared multi-user server** |

## B. CPU / RAM

| Item | Value |
|---|---|
| CPU | 2× Intel Xeon Platinum 8558 (48 cores/socket, HT) = 96 cores / 192 threads, 4 NUMA nodes |
| RAM | 1.0 TiB total, ~994 GiB available |
| Swap | 15 GiB |

## C. Disk

| Mount | Size | Free | Note |
|---|---|---|---|
| `/DATA2` (working FS, multipath SAN) | 109 TB | **95 TB** | ✅ well above the 50 GB threshold |
| `/` | 861 GB | 290 GB | keep project data off root |
| HF cache `$HF_HOME=/DATA2/shared/hf-cache` | — | 46 GB used | **shared** cache (see Decisions below) |

## D. Accelerators

| GPU | Model | VRAM | Used at survey | Compute cap |
|---|---|---|---|---|
| 0–3 | NVIDIA H200 | 143,771 MiB each | 14 MiB (idle) | 9.0 (Hopper) |

| Item | Value |
|---|---|
| Driver | 580.159.03 |
| CUDA (driver max) | 13.0 |
| System `nvcc` | 11.5 (`/usr/bin/nvcc`, old; not needed because pip wheels ship their own CUDA runtime) |
| Interconnect | NV6 NVLink between every GPU pair; GPU0–1 on NUMA 0, GPU2–3 on NUMA 2 |
| AMD ROCm / Apple | not found / n/a |

## E. Toolchain

| Tool | Status |
|---|---|
| python3 | 3.10.12 (`/usr/bin/python3`) |
| conda / mamba | conda 26.3.2, mamba 2.6.0 (`/DATA2/conda`) |
| **uv** | **not found**, so it has to be bootstrapped (Checkpoint 0) |
| pip | 22.0.2 (system) |
| git / git-lfs | 2.34.1 / 3.0.2 |
| gcc / g++ | 11.4.0 |
| clang | not found |
| cmake | 3.22.1 |
| nvcc | 11.5 |
| Docker | 29.1.3 installed, but **no access** (`permission denied` on the socket, user not in the `docker` group). Not needed. |
| Podman | not found |
| Ollama | 0.30.6, server running on `127.0.0.1:11434` (shared service) |
| Graphviz `dot` binary | **not found**. Not needed, because `st.graphviz_chart` renders DOT in the browser. |

**Libraries already installed** (each in its own env; we won't reuse them, but they show which versions work on this box):

| Env | Python | Relevant packages |
|---|---|---|
| system `/usr/bin/python3` | 3.10 | torch 2.13.0, transformers 5.16.1, vllm 0.28.0, openai 3.3.1, anthropic 1.1.0 |
| `cp-vllm` (conda) | 3.11 | torch **2.13.0+cu130**, transformers 5.16.1, **vllm 0.28.0**, openai, anthropic |
| `ml-base` (conda) | 3.11 | torch 2.5.1+cu121, transformers 5.8.0, datasets 4.8.5, pandas 3.0.2, accelerate, bitsandbytes |
| any env | — | sentence-transformers, faiss, beir, streamlit, plotly, graphviz, sglang: **not installed** |

The installed vLLM 0.28.0 registry supports `GptOssForCausalLM`, `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Gemma4*`, and `LlamaForCausalLM`. It does **not** support `Qwen4ExpForConditionalGeneration` (Qwen3.8-Flash-Next).

**Ollama models already on disk** (no download needed): `gpt-oss:120b` (65 GB), `bge-m3` (1.2 GB), `mxbai-embed-large`, `nomic-embed-text`, `qwen2.5:7b`, `qwen2.5:3b`, `gemma3`, `llama3.2`, `gemma3:270m`.

**HF cache, already present:** Qwen2.5-0.5B/1.5B/7B-Instruct, Qwen3-VL-8B, bge-small-en-v1.5, all-MiniLM-L6-v2, and others. **Not present:** Contriever, Qwen3-Embedding, Vicuna, gpt-oss, and the BEIR datasets.

## F. Network / secrets / UI

| Check | Result |
|---|---|
| Internet (google.com) | ✅ 200 |
| huggingface.co | ✅ 200 |
| github.com / api.github.com | ✅ 200 |
| PyPI / download.pytorch.org | ✅ 200 / ✅ 200 |
| BEIR host (public.ukp.informatik.tu-darmstadt.de) | ✅ 200 |
| api.anthropic.com / api.openai.com | reachable (no key, so unusable) |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` / cached token | **not set / absent**, so gated repos (Llama-2/3) are unavailable |
| `OPENAI_API_KEY` | not set |
| `ANTHROPIC_API_KEY` | not set |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | not set |
| `COHERE_API_KEY` | not set |
| `HF_HOME`, `TRANSFORMERS_CACHE` | set (→ `/DATA2/shared/hf-cache`) |
| `CUDA_VISIBLE_DEVICES` | not set |
| Port 8501 | **free** (also listening: 11434 = Ollama) |
| Headless / remote | **Yes.** Launch Streamlit with `--server.headless true --server.address 127.0.0.1`, then open it through VS Code's auto port-forward or `ssh -L 8501:localhost:8501 <host>`. |

**API spend: $0.** With no provider keys present, every target model runs locally.

---

## G. Verified model IDs (Hugging Face metadata, read on 2026-11-19)

"Weights" is the approximate download size (params × dtype). "Repo storage" from the API also counts git history, so it overstates the download.

### Embedders / rerankers

| HF ID | Params | Weights ≈ | Gated | Role |
|---|---|---|---|---|
| `facebook/contriever` | 110M | 0.44 GB | no | **Paper baseline** (upstream uses `contriever`, dot score) |
| `facebook/contriever-msmarco` | 110M | 0.44 GB | no | Paper ablation |
| `Qwen/Qwen3-Embedding-0.6B` | 0.60B | 1.2 GB | no | Modern, small |
| `Qwen/Qwen3-Embedding-4B` | 4.02B | 8 GB | no | Modern, mid |
| `Qwen/Qwen3-Embedding-8B` | 7.57B | 15 GB | no | **Modern, strongest** (dim 4096, MRL-truncatable) |
| `BAAI/bge-m3` | 0.57B | 2.3 GB | no | Modern alternative (also in Ollama) |
| `BAAI/bge-reranker-v2-m3` | 0.57B | 2.3 GB (F32) | no | Optional reranker |
| `Qwen/Qwen3-Reranker-4B` | 4.02B | 8 GB | no | Optional reranker (stronger) |

### Target LLMs

| HF ID | Params | Weights ≈ | Gated | Fits | Notes |
|---|---|---|---|---|---|
| `lmsys/vicuna-7b-v1.3` | 6.7B | 13.5 GB | no | 1 GPU | **In the paper**, ungated → faithful "original-style" model |
| `lmsys/vicuna-13b-v1.3` | 13B | 26 GB | no | 1 GPU | In the paper |
| `meta-llama/Llama-2-7b-chat-hf` | 6.74B | 13.5 GB | **manual** | 1 GPU | In the paper; needs HF_TOKEN + Meta approval |
| `Qwen/Qwen3-8B` | 8.19B | 16 GB | no | 1 GPU | Cheap modern, thinking-mode toggle |
| `Qwen/Qwen3.5-9B` | 9.65B | 19 GB | no | 1 GPU | Newer small; Apache-2.0; multimodal arch, used text-only |
| `Qwen/Qwen3.5-27B` | 27.8B | 56 GB BF16 | no | 1 GPU, BF16 | **Recommended modern dense** model |
| `google/gemma-4-31B-it` | 31.3B | 63 GB BF16 | no | 1 GPU, BF16 | Alternative modern dense (Apache-2.0) |
| `google/gemma-4-12B-it` | 12.0B | 24 GB | no | 1 GPU | Cheap modern alternative |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | 30.5B (3B active) | 61 GB | no | 1 GPU | Fast MoE |
| `openai/gpt-oss-120b` | 116.8B MoE | ~65 GB (MXFP4) | no | 1 GPU | **Recommended frontier-class open model** (already in Ollama) |
| `openai/gpt-oss-20b` | 20.9B MoE | ~13 GB | no | 1 GPU | Cheap reasoning model |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | 235B MoE | 470 GB BF16 | no | 4 GPUs (TP=4) | Possible stretch goal |
| `Qwen/Qwen3.8-Flash-Next` | 180B | 360 GB | no | — | ⚠️ **Excluded**: license "other", not supported by vLLM 0.28, released only weeks ago |

The Gemma "12B" size you named exists as `google/gemma-4-12B-it`. The list also shows a newer `Qwen3.5` line, so at Tier 3 I'd use Qwen3.5-27B in place of the older Qwen3-32B.

### Datasets / upstream

| Item | Size | Note |
|---|---|---|
| BEIR `nq.zip` | 475 MiB (~2.68M passages) | HEAD-checked |
| BEIR `hotpotqa.zip` | 623 MiB (~5.23M passages) | HEAD-checked |
| BEIR `msmarco.zip` | 1,032 MiB (~8.84M passages) | HEAD-checked |
| Upstream `sleeepeer/PoisonedRAG` | ~17 MB repo, MIT, last push 2026-01-27 | |
| `results/adv_targeted_results/{nq,hotpotqa,msmarco}.json` | 120 / 110 / 123 KB | released target Qs, target answers, poison texts |
| `results/beir_results/{nq,hotpotqa,msmarco}-contriever.json` | 11 / 23 / 2.5 MB | **precomputed Contriever top-k**, so the baseline needs **no corpus embedding** |
| `model_configs/` | vicuna7b/13b/33b, llama7b/13b, gpt3.5/4, palm2 | |

---

## H. Proposed plan (not built)

### Recommended stack (Tier 3, all local, $0 API)

| Layer | Choice | Why |
|---|---|---|
| Env manager | **uv** in a project-local venv (`./poisonedrag-modern/.venv`), Python 3.11 | uv is missing, so bootstrap it *inside* the project (see Checkpoint 0 options) |
| Torch | `torch==2.13.0` (cu130 wheels) | Matches the driver's CUDA 13.0, and `cp-vllm` shows this combination works here |
| Engine | **vLLM 0.28.0** (offline `LLM.generate` + embedding/pooling mode) | Hopper-optimized; supports every chosen arch; batched and deterministic with a seed |
| Baseline retriever | `facebook/contriever` + upstream precomputed `nq-contriever.json`, score=dot | Matches the paper exactly |
| Modern embedder | `Qwen/Qwen3-Embedding-8B` (primary) and `Qwen3-Embedding-0.6B` (fast) | Best open embedder family; 0.6B gives a quick smoke run |
| Reranker (flag) | `Qwen/Qwen3-Reranker-4B` | Tests whether reranking amplifies or dampens poisoning |
| Vector store | **FAISS `IndexFlatIP`** (faiss-gpu-cu12, or faiss-cpu on 1 TB RAM) | Exact dot product, same as the paper. NQ × 4096-d fp32 ≈ 44 GB, which fits in RAM. MRL→1024-d ≈ 11 GB. |
| Original-style LLM | `lmsys/vicuna-7b-v1.3` | In the paper, ungated, 13.5 GB |
| Modern LLMs | `Qwen/Qwen3.5-27B` (dense) + `openai/gpt-oss-120b` (MoE reasoning) | One strong dense model and one frontier-class MoE, each on its own GPU |
| Frontend | Streamlit + Plotly + `st.graphviz_chart` (browser rendering, no `dot` binary needed) | Works headless over SSH port-forward |
| Datasets | NQ first, then HotpotQA, then MS-MARCO | NQ has the smallest corpus of the three |

### Alternatives

1. **Zero-download LLMs through Ollama.** Use the already-pulled `gpt-oss:120b` and `bge-m3` via `localhost:11434`. Pros: no 65 GB download, works immediately. Cons: Ollama is a *shared system service* (other users' load and model swaps), it serves quantized GGUF, it's less deterministic, and Vicuna isn't there.
2. **Lighter footprint.** Vicuna-7B + `Qwen3-8B`/`gemma-4-12B-it` + `Qwen3-Embedding-0.6B`/`bge-m3`. Total ≈ 45 GB of downloads, and every run finishes in minutes. Good if the GPUs turn out to be contended.
3. **Faithful-plus.** Add `meta-llama/Llama-2-7b-chat-hf` (requires you to set `HF_TOKEN` and accept Meta's license) and a stretch run on `Qwen3-235B-A22B-Instruct-2507` with TP=4 (470 GB download, ~1 h). API targets (Claude / GPT / Gemini) only if you add keys later. I'd check live pricing then and get your approval before any spend.

### First baseline subset (approved by you before it runs)

NQ, **2 target questions**, `top_k=5`, `N=5` poison texts/question (released upstream texts), Contriever precomputed results, `score=dot`, `seed=12`, Vicuna-7B on 1 GPU, temperature 0.1.
Expected: < 5 min after downloads. Includes a 1-question non-empty-output smoke test (upstream issue #1) and `padding_side='left'`.

### Rough cost / time estimates (before any approval)

| Item | Download | Time (est.) | Needs approval? |
|---|---|---|---|
| Upstream clone + NQ BEIR zip | ~0.5 GB | ~1 min | no (< 2 GB) |
| Contriever + Vicuna-7B | ~14 GB | ~3–5 min | **yes** (> 2 GB) |
| Qwen3-Embedding-0.6B / 8B | 1.2 / 15 GB | — | **yes** (8B) |
| Embed NQ corpus (2.68M) with Qwen3-Emb-0.6B | — | ~15 min, 1 GPU | **yes** (> 10 min) |
| Embed NQ corpus with Qwen3-Emb-8B | — | ~3 h on 1 GPU / ~50 min on 4 GPUs (rough FLOP estimate) | **yes** |
| Qwen3.5-27B + gpt-oss-120b | ~56 + ~65 GB | — | **yes** |
| Full 100-question NQ, 1 retriever × 3 LLMs | — | ~10–20 min of generation | **yes** |
| **API spend** | — | **$0** (no keys) | — |
| **Total disk if everything above runs** | **~165 GB** (+ ~60 GB indexes) | — | fits easily |

---

## Decisions I need from you at "APPROVED: PHASE 2"

1. **Project location.** I put this at `Cybersecurity/PoisonedRAG_implementation/poisonedrag-modern/`, next to the empty `main.py` you had open. Should I move it to `~/poisonedrag-modern/`?
2. **uv bootstrap.** uv isn't installed. Options (all user-space, no sudo):
   (a) `python3 -m venv .venv-bootstrap && .venv-bootstrap/bin/pip install uv`, entirely inside the project **(recommended)**;
   (b) `pip install --user uv` → `~/.local/bin` (outside the project);
   (c) the astral standalone installer.
3. **HF cache location.** `$HF_HOME` points at the **shared** `/DATA2/shared/hf-cache`. Either set `HF_HOME=./poisonedrag-modern/.cache/hf` (keeps to the guardrail, no effect on other users; **recommended**) or use the shared cache (other users could reuse the downloads).
4. **GPU etiquette.** This is a shared box. I plan to pin `CUDA_VISIBLE_DEVICES` to one GPU per job and check free VRAM before each run. Do you have assigned GPUs?
5. **Stack.** Recommended / Alt 1 (Ollama) / Alt 2 (light) / Alt 3 (+Llama-2, needs HF_TOKEN).

## Appendix: commands run in Phase 1 (all read-only)

`cat /etc/os-release`, `uname -a`, `systemd-detect-virt`, `lscpu`, `free -h`, `df -h`, `nvidia-smi` (+ `--query-gpu`, `topo -m`), `nvcc --version`, `<tool> --version` for each toolchain item, `conda env list`, `<env>/bin/python -m pip list`, `grep` of the vLLM model registry, `curl -I` reachability + BEIR HEAD requests, HF `/api/models` metadata GETs, GitHub `/repos/.../contents` GETs, env-var presence checks, `ss -ltn`, `ollama list`, `docker ps`, `who`, `uptime`. These will be copied into RUNLOG.md at Step 0.
