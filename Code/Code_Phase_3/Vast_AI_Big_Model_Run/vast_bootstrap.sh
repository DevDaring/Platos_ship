#!/usr/bin/env bash
# vast_bootstrap.sh — prepare a rented Vast.ai box for the 70B bf16 probe.
#
# Fails fast and loudly. GPU time is billed by the second, so every check that
# can be done before the 132 GB download starts is done before it starts.
#
#   HUGGINGFACE_TOKEN=... bash vast_bootstrap.sh

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
MIN_GPUS="${MIN_GPUS:-2}"
MIN_VRAM_GB="${MIN_VRAM_GB:-78}"       # an 80 GB card reports ~81559 MiB
MIN_DISK_GB="${MIN_DISK_GB:-180}"
HERE="$(cd "$(dirname "$0")" && pwd)"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

say "1/7  GPUs"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found — is this a GPU instance?"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
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

say "2/7  Disk"
AVAIL_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
[ "$AVAIL_GB" -ge "$MIN_DISK_GB" ] || fail \
  "only ${AVAIL_GB} GB free; the checkpoint alone is ~132 GB (need >= ${MIN_DISK_GB} GB)."
say "    ${AVAIL_GB} GB free."

say "3/7  Python"
python3 --version
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')
[ "$PYVER" -ge 310 ] || fail "Python 3.10+ required (found $(python3 --version))."
python3 -m pip --version >/dev/null 2>&1 || fail "pip missing on this image."
nvcc --version 2>/dev/null | tail -2 || echo "    (nvcc absent; the wheel ships its own CUDA runtime)"

say "4/7  Dependencies"
# Vast.ai images are already ML images with a system python that owns torch,
# so we install in place rather than building a venv that would re-download
# torch. --no-cache-dir keeps a 132 GB checkpoint and a pip cache from
# competing for the same disk.
python3 -m pip install --quiet --no-cache-dir --upgrade pip
python3 -m pip install --quiet --no-cache-dir -r "$HERE/requirements_big.txt"
python3 - <<'PY'
import torch, vllm
print(f"    torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"devices={torch.cuda.device_count()}")
print(f"    vllm  {vllm.__version__}")
major, minor = torch.cuda.get_device_capability(0)
print(f"    compute capability sm{major}{minor}")
if (major, minor) < (8, 0):
    raise SystemExit("FlashAttention-2 needs sm80+.")
PY
# Every module the probe imports, checked HERE rather than after the download.
# The first version of this payload omitted analysis/stats.py (imported by
# corrected_probe at contrast time, i.e. after all GPU work was paid for) and
# openai_compatible_agent.py (imported at start-up via src/extraction.py).
say "    import check"
# deploy_vast.sh flattens the payload INTO $HERE, so GPU_Only, src and
# analysis are siblings of run_big_probe.py, not one level up. The old
# push_inputs.sh used the other layout; looking in the wrong place made
# this check fail on a payload that was actually complete.
cd "$HERE" && python3 - <<'PY'
import sys
sys.path.insert(0, ".")
import importlib
for name in ("GPU_Only.src.corrected_probe", "src.contexts", "src.extraction",
             "src.seeding", "analysis.stats"):
    importlib.import_module(name)
    print(f"      ok  {name}")
PY
cd "$HERE"

say "5/7  FlashAttention"
# vLLM ships FlashAttention kernels (vllm-flash-attn) and both A100 (sm80) and
# H100 (sm90) support FlashAttention-2. Forcing the backend makes the choice
# explicit and logged rather than left to a heuristic. A standalone wheel is
# installed only if a prebuilt one matches this torch/CUDA pair: a source
# build takes ~30 minutes on hardware that is already billing.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
if python3 -c "import flash_attn" 2>/dev/null; then
  python3 -c "import flash_attn; print(f'    flash-attn {flash_attn.__version__} already present')"
else
  echo "    attempting a prebuilt flash-attn wheel (no source build)…"
  if python3 -m pip install --quiet --no-cache-dir --only-binary=:all: flash-attn 2>/dev/null; then
    python3 -c "import flash_attn; print(f'    flash-attn {flash_attn.__version__} installed')"
  else
    echo "    no prebuilt wheel for this torch/CUDA pair — skipping."
    echo "    vLLM's bundled FlashAttention kernels are used regardless."
  fi
fi
python3 - <<'PY'
import os
try:
    import vllm.vllm_flash_attn  # noqa: F401
    print("    vllm-flash-attn kernels available")
except Exception as exc:
    print(f"    vllm-flash-attn not importable ({exc}); vLLM will pick a backend")
print(f"    VLLM_ATTENTION_BACKEND={os.environ.get('VLLM_ATTENTION_BACKEND')}")
PY

say "6/7  Hugging Face access"
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

say "7/7  Pre-download the checkpoint (~132 GB)"
# Done as its own step so a slow link fails here rather than inside vLLM,
# and so the download is resumable. hf_transfer uses many parallel
# connections, which on a multi-Gbps box turns ~40 minutes into ~10.
python3 -m pip install --quiet --no-cache-dir hf_transfer 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 - <<PY
import os, time
from huggingface_hub import snapshot_download
start = time.time()
try:
    path = snapshot_download(
        "${MODEL}", token=os.environ["HF_TOKEN"],
        allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
        max_workers=16,
    )
except Exception:
    # hf_transfer can fail on some networks; fall back to plain requests.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    path = snapshot_download(
        "${MODEL}", token=os.environ["HF_TOKEN"],
        allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
        max_workers=8,
    )
print(f"    checkpoint at {path}  ({(time.time()-start)/60:.1f} min)")
PY

cat > "$HOME/platos_env.sh" <<ENV
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
# FlashInfer's JIT compiled against mismatched headers on the L4 run and
# aborted sampling. vLLM's default sampler is correct and costs nothing
# measurable here, so the JIT path stays off.
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="${HF_TOKEN}"
ENV
chmod 600 "$HOME/platos_env.sh"

say "Ready. Next: bash smoke_test_vast.sh   (2 questions, all three conditions)"
echo "     Then:  bash run_full_vast.sh"
echo "     Remember to DESTROY the instance when the results are pulled."
