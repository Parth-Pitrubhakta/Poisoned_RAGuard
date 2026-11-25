"""Download one registry model at its pinned revision into the project HF cache ($HF_HOME, set by scripts/env.sh).

  uv run python scripts/download_model.py llms qwen3.5-27b
  uv run python scripts/download_model.py llms gpt-oss-120b      # top-level MXFP4 only (skips original/, metal/)
"""
import os
import sys
import time

from huggingface_hub import snapshot_download

from prag.config import load_registry

IGNORE = ["original/*", "metal/*", "*.bin", "*.pt", "*.pth", "*.h5", "*.msgpack", "*.onnx", "onnx/*", "*.gguf"]

section, key = sys.argv[1], sys.argv[2]
spec = load_registry()[section][key]
assert os.environ.get("HF_HOME", "").endswith(".cache/hf"), "source scripts/env.sh first (project-local HF_HOME)"
t0 = time.time()
# a registry entry may override the ignore list (e.g. SimCSE ships only pytorch_model.bin)
path = snapshot_download(spec["hf_id"], revision=spec["revision"], ignore_patterns=spec.get("download_ignore", IGNORE), max_workers=8)
size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path, followlinks=True) for f in fs)
print(f"{key}: {spec['hf_id']}@{spec['revision'][:7]} -> {path}  {size / 1e9:.1f} GB in {time.time() - t0:.0f}s")
