# Source this before every command:   source scripts/env.sh
# Keeps the venv and every cache (uv, Python, HF, vLLM, Triton, Inductor, FlashInfer) inside the project,
# so nothing is written to the shared /DATA2/shared/hf-cache or to ~/.cache.

PRAG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRAG_ROOT

# --- environment / package manager ---
export UV_PROJECT_ENVIRONMENT="$PRAG_ROOT/venv_PoisonedRAG"
export UV_CACHE_DIR="$PRAG_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PRAG_ROOT/.cache/uv-python"
export UV_PYTHON_PREFERENCE=only-managed
export PATH="$PRAG_ROOT/.bootstrap/bin:$PATH"          # project-local uv binary

# --- model / kernel caches (override shared defaults) ---
export HF_HOME="$PRAG_ROOT/.cache/hf"
unset TRANSFORMERS_CACHE                                # deprecated; points at the shared cache
export XDG_CACHE_HOME="$PRAG_ROOT/.cache/xdg"
export VLLM_CACHE_ROOT="$PRAG_ROOT/.cache/vllm"
export TRITON_CACHE_DIR="$PRAG_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PRAG_ROOT/.cache/inductor"
export FLASHINFER_WORKSPACE_BASE="$PRAG_ROOT/.cache/flashinfer"

# --- privacy / reproducibility ---
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export PYTHONHASHSEED=12
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$PRAG_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
