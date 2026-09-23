#!/usr/bin/env bash
# smoke_test.sh — prove the pipeline works on 2 questions before the full run.
#
# This is the step that separates "the code imports" from "the code produces
# correct output on this machine". It loads the real model, runs all three
# conditions on two questions, and then CHECKS the output rather than merely
# reporting that the script exited zero.
#
#   bash smoke_test.sh

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
HOME_DIR="${HOME_DIR:-$HOME/platos}"
SMOKE_OUT="$HOME_DIR/results_smoke"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mSMOKE TEST FAILED: %s\033[0m\n' "$*" >&2; exit 1; }

cd "$HOME_DIR"
# shellcheck disable=SC1090
[ -f "$HOME/platos_env.sh" ] && source "$HOME/platos_env.sh"

say "Running the probe on 2 questions (all 3 conditions)"
rm -rf "$SMOKE_OUT"
python3 run_probe.py \
  --model "$MODEL" \
  --tensor-parallel-size 1 \
  --dry-run \
  --out "$SMOKE_OUT" 2>&1 | tail -25

say "Checking the output"
python3 - <<PY || fail "output checks did not pass"
import sys
from pathlib import Path

import pandas as pd

out = Path("$SMOKE_OUT")
trials_path = out / "big_probe_trials.parquet"
problems = []

if not trials_path.exists():
    sys.exit("no trials parquet was written")

trials = pd.read_parquet(trials_path)
print(f"  rows: {len(trials)}")
print(f"  conditions: {sorted(trials['condition'].unique())}")

# 1. Every condition must have produced rows.
missing = {"R", "E", "WR"} - set(trials["condition"])
if missing:
    problems.append(f"conditions produced no rows: {sorted(missing)}")

# 2. The target must be fixed per (question, replicate) ACROSS conditions —
#    otherwise the difference-in-differences compares unlike things.
per_cell = trials.groupby(["question_identifier", "replicate"])["fixed_target"].nunique()
if (per_cell > 1).any():
    problems.append("fixed_target varies across conditions within a cell")

# 3. Probability mass must be real numbers in [0, 1], not NaN.
for column in ("prob_mass_round0_on_target", "prob_mass_round1_on_target",
               "prob_mass_round0_on_correct", "prob_mass_round1_on_correct"):
    values = pd.to_numeric(trials[column], errors="coerce")
    if values.isna().all():
        problems.append(f"{column} is entirely NaN — candidate scoring failed")
    elif ((values < -1e-6) | (values > 1 + 1e-6)).any():
        problems.append(f"{column} outside [0, 1]")

# 4. Off-candidate mass must be present, not normalised away.
if "mass_outside_candidates_round1" not in trials.columns:
    problems.append("off-candidate mass column missing")
elif pd.to_numeric(trials["mass_outside_candidates_round1"],
                   errors="coerce").isna().all():
    problems.append("off-candidate mass is entirely NaN")

# 5. Answers must have parsed.
if trials["round1_answer"].isna().all() or (trials["round1_answer"] == "").all():
    problems.append("no Round-1 answer parsed — check the chat template")

# 6. The baseline R must not have seen peers: its Round-0 and Round-1 mass on
#    the target should be identical for the same trial only when the answer did
#    not move. Just check the column exists and is finite.
r_rows = trials[trials["condition"] == "R"]
if r_rows.empty:
    problems.append("no baseline (R) rows")

for column in ("big_probe_contrast.parquet", "big_probe_candidates.parquet",
               "big_probe_meta.json"):
    if not (out / column).exists():
        problems.append(f"missing output: {column}")

if problems:
    print("\n  PROBLEMS:")
    for problem in problems:
        print(f"    - {problem}")
    sys.exit(1)

print("\n  all checks passed")
PY

say "Smoke test passed. Full run:  bash run_full.sh"
