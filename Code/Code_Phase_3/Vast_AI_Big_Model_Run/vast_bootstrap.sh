#!/usr/bin/env bash
# vast_bootstrap.sh — prepare a rented Vast.ai box for the 70B bf16 probe.
#
# Fails fast and loudly. GPU time is billed by the second, so every check that
# can be done before the 132 GB download starts is done before it starts.
#
#   bash vast_bootstrap.sh

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
MIN_GPUS="${MIN_GPUS:-2}"
MIN_VRAM_GB="${MIN_VRAM_GB:-78}"       # an 80 GB card reports ~81559 MiB
MIN_DISK_GB="${MIN_DISK_GB:-180}"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

say "1/6  GPUs"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found — is this a GPU instance?"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$GPU_COUNT" -ge "$MIN_GPUS" ] || fail \
  "need $MIN_GPUS GPUs for 70B in bf16 (131.5 GB of weights), found $GPU_COUNT.
   A single 80 GB card cannot hold it. Either rent 2 cards, or accept a
   quantised model — but a quantised model has a different output
   distribution, which is the quantity this probe measures."

SMALLEST=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
SMALLEST_GB=$((SMALLEST / 1024))
[ "$SMALLEST_GB" -ge "$MIN_VRAM_GB" ] || fail \
  "smallest GPU is ${SMALLEST_GB} GB; need >= ${MIN_VRAM_GB} GB per card."
say "    $GPU_COUNT GPUs, smallest ${SMALLEST_GB} GB — enough for bf16 tensor-parallel."

say "2/6  Disk"
AVAIL_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
[ "$AVAIL_GB" -ge "$MIN_DISK_GB" ] || fail \
  "only ${AVAIL_GB} GB free; the checkpoint alone is ~132 GB (need >= ${MIN_DISK_GB} GB)."
say "    ${AVAIL_GB} GB free."

say "3/6  Python and CUDA"
python3 --version
python3 - <<'PY'
import sys
if sys.version_info < (3, 9):
    raise SystemExit("Python 3.9+ required")
PY
nvcc --version 2>/dev/null | tail -2 || echo "    (nvcc absent; the wheel ships its own CUDA runtime)"

say "4/6  Dependencies"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements_big.txt
python3 - <<'PY'
import torch, vllm
print(f"    torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"devices={torch.cuda.device_count()}")
print(f"    vllm  {vllm.__version__}")
PY

say "5/6  Hugging Face access"
# Llama-3.1-70B is gated: without an accepted licence the download 401s AFTER
# the instance is already billing. Check now.
if [ -z "${HUGGINGFACE_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
  fail "set HUGGINGFACE_TOKEN (or HF_TOKEN) — Llama-3.1-70B is a gated repo."
fi
export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_TOKEN}"
python3 - <<PY
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
model = "${MODEL}"
try:
    info = api.model_info(model)
    print(f"    access OK: {model} @ {info.sha[:12]}")
except Exception as exc:
    sys.exit(f"cannot access {model}: {exc}\\n"
             f"Accept the licence at https://huggingface.co/{model}")
PY

say "6/6  Pre-download the checkpoint"
# Done as its own step so a slow link fails here rather than inside vLLM,
# and so the download is resumable.
python3 - <<PY
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    "${MODEL}", token=os.environ["HF_TOKEN"],
    allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
    max_workers=8, resume_download=True,
)
print(f"    checkpoint at {path}")
PY

say "Ready. Next: python3 run_big_probe.py --tensor-parallel-size $GPU_COUNT"
echo "     Remember to DESTROY the instance when the results are pulled."
