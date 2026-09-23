#!/usr/bin/env bash
# vm_bootstrap.sh — prepare the L4 VM. Runs ON the VM.
#
# Fails fast: every check that can be done before the checkpoint download
# starts is done before it starts, because the GPU bills by the second.
#
#   export HUGGINGFACE_TOKEN=...
#   bash vm_bootstrap.sh
#
# EVERYTHING IS INSTALLED INTO A VENV, NOT THE SYSTEM PYTHON.
# The first run of this probe hit five separate toolchain failures on a bare
# image and each was fixed by hand on the box, so the fixes died with the VM
# and this script still assumed a system pip that did not exist. The venv is
# created here, its path is exported through platos_env.sh, and both
# smoke_test.sh and run_full.sh pick it up by sourcing that file. autopush.sh
# already expects the interpreter at $HOME/platos/.venv/bin/python.

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MIN_VRAM_GB="${MIN_VRAM_GB:-22}"      # an L4 reports ~23034 MiB
MIN_DISK_GB="${MIN_DISK_GB:-60}"
VENV="${VENV:-$HOME/platos/.venv}"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

say "1/7  GPU"
command -v nvidia-smi >/dev/null || fail "nvidia-smi missing — driver not installed yet."
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$((VRAM / 1024))
[ "$VRAM_GB" -ge "$MIN_VRAM_GB" ] || fail \
  "GPU has ${VRAM_GB}GB; Llama-3.1-8B in bf16 needs ~15GB of weights plus KV
   cache (>= ${MIN_VRAM_GB}GB). Do NOT quantise to fit: this probe measures the
   output distribution, and quantisation changes exactly that."
say "    ${VRAM_GB}GB VRAM — enough for 8B in bf16."

say "2/7  Disk"
AVAIL_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
[ "$AVAIL_GB" -ge "$MIN_DISK_GB" ] || fail "only ${AVAIL_GB}GB free (need >= ${MIN_DISK_GB}GB)."
say "    ${AVAIL_GB}GB free."

say "3/7  Python and virtual environment"
python3 --version
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')
[ "$PYVER" -ge 310 ] || fail "Python 3.10+ required (found $(python3 --version))."

# The image carries a CUDA runtime and a compiler but no pip and no venv
# module. Install both before anything tries to use them.
if ! python3 -c "import ensurepip, venv" 2>/dev/null; then
  say "    installing python3-venv / python3-pip"
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3-venv python3-pip build-essential
fi

mkdir -p "$(dirname "$VENV")"
[ -d "$VENV" ] || python3 -m venv "$VENV"
PY="$VENV/bin/python"
[ -x "$PY" ] || fail "venv interpreter missing at $PY"
"$PY" -m pip install --quiet --upgrade pip wheel setuptools
say "    venv at $VENV ($("$PY" --version))"

say "4/7  vLLM and dependencies"
"$PY" -m pip install --quiet -r requirements_l4.txt
"$PY" - <<'PY'
import torch, vllm
print(f"    torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"device={torch.cuda.get_device_name(0)}")
print(f"    vllm  {vllm.__version__}")
major, minor = torch.cuda.get_device_capability(0)
print(f"    compute capability sm{major}{minor}")
if (major, minor) < (8, 0):
    raise SystemExit("FlashAttention-2 needs sm80+; an L4 is sm89.")
PY

say "5/7  FlashAttention"
# vLLM ships its own FlashAttention kernels (vllm-flash-attn) and selects them
# automatically on Ada (sm89). Forcing the backend makes the choice explicit
# and logged, rather than left to a heuristic. A standalone `flash-attn` wheel
# is installed too for any HF-transformers path, but it is optional: if no
# prebuilt wheel matches this torch/CUDA pair we skip it rather than trigger a
# 30-minute source build on a billing GPU.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
if "$PY" -c "import flash_attn" 2>/dev/null; then
  "$PY" -c "import flash_attn; print(f'    flash-attn {flash_attn.__version__} already present')"
else
  echo "    attempting a prebuilt flash-attn wheel (no source build)…"
  if "$PY" -m pip install --quiet --only-binary=:all: flash-attn 2>/dev/null; then
    "$PY" -c "import flash_attn; print(f'    flash-attn {flash_attn.__version__} installed')"
  else
    echo "    no prebuilt wheel for this torch/CUDA pair — skipping."
    echo "    vLLM's bundled FlashAttention kernels are used regardless."
  fi
fi
"$PY" - <<'PY'
import os
try:
    import vllm.vllm_flash_attn  # noqa: F401
    print("    vllm-flash-attn kernels available")
except Exception as exc:
    print(f"    vllm-flash-attn not importable ({exc}); vLLM will pick a backend")
print(f"    VLLM_ATTENTION_BACKEND={os.environ.get('VLLM_ATTENTION_BACKEND')}")
PY

say "6/7  Hugging Face access"
if [ -z "${HUGGINGFACE_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
  fail "set HUGGINGFACE_TOKEN — Llama-3.1-8B is a gated repo and the download
   would 401 only after the instance is already billing."
fi
export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_TOKEN}"
"$PY" - <<PY
import os, sys
from huggingface_hub import HfApi
try:
    info = HfApi(token=os.environ["HF_TOKEN"]).model_info("${MODEL}")
    print(f"    access OK: ${MODEL} @ {info.sha[:12]}")
except Exception as exc:
    sys.exit(f"cannot access ${MODEL}: {exc}\\n"
             f"Accept the licence at https://huggingface.co/${MODEL}")
PY

say "7/7  Pre-download the checkpoint"
# Its own step so a slow link fails here rather than inside vLLM, and so the
# download resumes instead of restarting.
"$PY" - <<PY
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    "${MODEL}", token=os.environ["HF_TOKEN"],
    allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
    max_workers=8,
)
print(f"    checkpoint at {path}")
PY

cat > "$HOME/platos_env.sh" <<ENV
# Put the venv first so plain \`python3\` in smoke_test.sh and run_full.sh is
# the interpreter that actually has vLLM installed.
export PATH="$VENV/bin:\$PATH"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
# FlashInfer's JIT compiles against headers that did not match this image on
# the first run and aborted sampling. vLLM's default sampler is correct and
# costs nothing measurable here, so the JIT path stays off.
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_TOKEN="${HF_TOKEN}"
ENV
chmod 600 "$HOME/platos_env.sh"

say "Ready. Next: bash smoke_test.sh   (2 questions, ~2 minutes)"
