#!/usr/bin/env bash
# run_sweep_vast.sh — the X8 probe across the remaining open-weight models,
# to turn a two-point contrast into a capability gradient. Runs ON the box.
#
#   GITHUB_TOKEN=... bash run_sweep_vast.sh
#
# WHY A SWEEP AND NOT ONE MORE MODEL.
# Llama-3.1-8B and Llama-3.1-70B give two points: enough to say the large
# model feels more distributional pull while adopting less, not enough to
# call it a gradient. The paper's behavioural gradient rests on eight models
# and a Spearman correlation against solo accuracy. This run puts the same
# shape of evidence behind the distributional measure.
#
# EACH MODEL IS SMOKE-TESTED BEFORE ITS OWN FULL RUN. A model that fails its
# smoke test is SKIPPED, not fatal: one broken chat template must not cost
# the other three their results on hardware that is already billing.
#
# Checked before renting anything, because each would have failed only after
# the checkpoint was downloaded:
#   - Mistral-Small-3.2-24B-2506 ships no HF chat template at all (it uses
#     mistral_common), so the 2501 checkpoint is used instead and the
#     difference is recorded here rather than discovered later.
#   - Gemma templates were verified to accept a system role; run_big_probe.py
#     now falls back to folding it into the user turn if any model's does not.
#   - Qwen2.5-72B in bf16 needs both cards; everything else fits one.

set -euo pipefail

HOME_DIR="${HOME_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SWEEP_OUT="${SWEEP_OUT:-$HOME_DIR/sweep_results}"
QUESTIONS="${QUESTIONS:-300}"
REPLICATES="${REPLICATES:-3}"
SEED="${SEED:-20260502}"
TP="${TP:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"

# Smallest first: a cheap model surfaces a pipeline fault before an
# expensive one has spent forty minutes of GPU time proving the same thing.
MODELS=(
  "google/gemma-3-4b-it"
  "mistralai/Mistral-Small-24B-Instruct-2501"
  "google/gemma-3-27b-it"
  "Qwen/Qwen2.5-72B-Instruct"
)

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m!!  %s\033[0m\n' "$*"; }

[ -n "${GITHUB_TOKEN:-}" ] || { echo "set GITHUB_TOKEN for the autopush" >&2; exit 1; }

cd "$HOME_DIR"
mkdir -p logs "$SWEEP_OUT"
# shellcheck disable=SC1090
[ -f "$HOME/platos_env.sh" ] && source "$HOME/platos_env.sh"

# The sweep MUST push to its own remote folder. The first version passed
# RESULTS_DIR=sweep_results but kept autopush's default destination, which is
# the 70B's results/ folder. autopush syncs with `rsync --delete`, so every
# 70B parquet and JSON file there was absent from the source and was DELETED
# from main. The data survived locally and in history, but main was wrong.
# Output also goes to a log now; sending it to /dev/null is why the deletion
# went unnoticed until the results were checked by hand.
push() { GITHUB_TOKEN="$GITHUB_TOKEN" RESULTS_DIR="$SWEEP_OUT" \
         REMOTE_SUBDIR="Code/Code_Phase_3/Vast_AI_Big_Model_Run/sweep_results" \
         bash autopush_vast.sh --once >> logs/sweep_autopush.log 2>&1 || true; }

STATUS_FILE="$SWEEP_OUT/sweep_status.txt"
: > "$STATUS_FILE"

for MODEL in "${MODELS[@]}"; do
  SLUG="$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')"
  OUT="$SWEEP_OUT/$SLUG"
  LOG="logs/${SLUG}.log"

  if [ -f "$OUT/big_probe_contrast.parquet" ]; then
    say "$MODEL — already complete, skipping"
    echo "$SLUG  SKIPPED (already complete)" >> "$STATUS_FILE"
    continue
  fi

  say "[$MODEL]  1/3 smoke test on 2 questions"
  rm -rf "${OUT}_smoke"
  if ! python3 run_big_probe.py --model "$MODEL" --tensor-parallel-size "$TP" \
        --dry-run --include-numeric --out "${OUT}_smoke" >> "$LOG" 2>&1; then
    warn "$MODEL failed its smoke test — SKIPPING (see $LOG)"
    echo "$SLUG  FAILED (smoke test)" >> "$STATUS_FILE"
    push; continue
  fi

  say "[$MODEL]  2/3 checking smoke output"
  if ! python3 - "$OUT" <<'PY' >> "$LOG" 2>&1
import sys
from pathlib import Path
import pandas as pd
out = Path(sys.argv[1] + "_smoke")
d = pd.read_parquet(out / "big_probe_trials.parquet")
problems = []
if d.empty:
    problems.append("no rows")
if set(d.condition.unique()) != {"R", "E", "WR"}:
    problems.append(f"conditions {sorted(d.condition.unique())}")
for column in ("round0_text", "round1_text"):
    if column not in d.columns:
        problems.append(f"{column} missing — raw generations discarded")
    elif d[column].astype(str).str.strip().eq("").all():
        problems.append(f"{column} empty on every row")
mass = pd.to_numeric(d["prob_mass_round1_on_target"], errors="coerce")
if mass.notna().sum() == 0:
    problems.append("probability mass entirely NaN")
elif ((mass.dropna() < -1e-9) | (mass.dropna() > 1 + 1e-9)).any():
    problems.append("probability mass outside [0,1]")
g = d.groupby(["question_identifier", "replicate"]).fixed_target.nunique()
if (g > 1).any():
    problems.append("target differs across conditions within a cell")
if problems:
    print("SMOKE PROBLEMS: " + "; ".join(problems))
    raise SystemExit(1)
print("smoke output OK")
PY
  then
    warn "$MODEL smoke output failed its checks — SKIPPING (see $LOG)"
    echo "$SLUG  FAILED (smoke checks)" >> "$STATUS_FILE"
    push; continue
  fi
  rm -rf "${OUT}_smoke"

  say "[$MODEL]  3/3 full run"
  if python3 run_big_probe.py --model "$MODEL" --tensor-parallel-size "$TP" \
       --questions "$QUESTIONS" --replicates "$REPLICATES" --seed "$SEED" \
       --include-numeric --out "$OUT" >> "$LOG" 2>&1; then
    ROWS=$(python3 -c "import pandas as pd;print(len(pd.read_parquet('$OUT/big_probe_trials.parquet')))" 2>/dev/null || echo "?")
    say "[$MODEL] finished — $ROWS rows"
    echo "$SLUG  OK ($ROWS rows)" >> "$STATUS_FILE"
  else
    warn "$MODEL full run failed — continuing with the rest (see $LOG)"
    echo "$SLUG  FAILED (full run)" >> "$STATUS_FILE"
  fi
  push

  # The checkpoints total ~255 GB. Freeing each one after its run keeps the
  # disk from deciding which of the later models gets to exist.
  python3 - "$MODEL" <<'PY' >> "$LOG" 2>&1 || true
import shutil, sys
from huggingface_hub import scan_cache_dir
target = sys.argv[1]
for repo in scan_cache_dir().repos:
    if repo.repo_id == target:
        shutil.rmtree(repo.repo_path, ignore_errors=True)
        print(f"freed {repo.size_on_disk / 2**30:.1f} GB for {target}")
PY
done

say "Sweep complete"
cat "$STATUS_FILE"
push
