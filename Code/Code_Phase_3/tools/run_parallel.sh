#!/usr/bin/env bash
# run_parallel.sh — the whole Phase 3 API run, one process per focal model.
#
#   bash tools/run_parallel.sh              # real run
#   bash tools/run_parallel.sh --dry-run    # one unit per cell, every code path
#
# Stages, each gated on the previous one:
#   1. --prepare, ONCE and sequentially. It builds every shared input (honest
#      bank, hedged pool, GSM-Symbolic pool and personas). Shards refuse to
#      start without them (src/sharding.py), so none can race to build one.
#   2. One process per focal model, all at once. The eight models bill to
#      different providers, so they do not share rate limits. A model is never
#      split: its cached Round-0 answer is shared by all of its conditions.
#      Each shard writes only to results/outputs/shards/<model>/.
#   3. A shard exiting 3 left failed API calls unrecorded (src/call_guard.py).
#      It is re-run, which pays only for the missing units, until it exits 0.
#      Any other non-zero exit is a real error: that shard stops and the
#      merge does not happen.
#   4. tools/merge_shards.py, which refuses unless every check passes, then
#      the offline analysis (including X5, which only reads the caches).
#
# Inside each process, units also run on a few threads (src/concurrency.py);
# results are identical to one thread (tests/test_concurrency.py). Worker
# counts follow what each route sustained when measured on 24 Sept 2026:
# nano-gpt took 8 concurrent calls without error, OpenRouter's upstream
# returned 429s at 8 for Gemma-3-4B, so OpenRouter models get 4-6.
#
# Environment: PYTHON (default python3), MAX_ATTEMPTS (default 20),
# RETRY_WAIT seconds between attempts (default 300), PREPARE_WORKERS (12).

set -u
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-20}   # 20 x 5 min: rides out long 429 episodes
RETRY_WAIT=${RETRY_WAIT:-300}
EXTRA=("$@")
FOCALS=(deepseek_primary gpt4o_mini sweep_llama_3_1_70b sweep_qwen_2_5_72b
        sweep_gemma_3_27b sweep_mistral_small sweep_llama_3_1_8b_focal
        sweep_gemma_3_4b_focal)
PREPARE_WORKERS=${PREPARE_WORKERS:-12}
declare -A WORKERS=(
    [deepseek_primary]=6          # DeepSeek direct
    [gpt4o_mini]=6                # LinkAPI (Azure group)
    [sweep_llama_3_1_70b]=4       # OpenRouter
    [sweep_qwen_2_5_72b]=6        # nano-gpt
    [sweep_gemma_3_27b]=6         # OpenRouter; 0 429s at 4 in the smoke run
    [sweep_mistral_small]=6       # nano-gpt
    [sweep_llama_3_1_8b_focal]=12 # nano-gpt; slowest model (27 s mean, loops to the cap)
    [sweep_gemma_3_4b_focal]=4    # OpenRouter; 429s seen at 8
)
LOGS=logs/launcher
mkdir -p "$LOGS"
EXITS="$LOGS/exits.txt"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Run one command until it exits 0; exit 3 means "retry", anything else stops.
until_complete() {
    local name=$1; shift
    local attempt rc=1
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        "$@" > "$LOGS/$name.attempt$attempt.log" 2>&1
        rc=$?
        echo "$(stamp) $name attempt=$attempt exit=$rc" >> "$EXITS"
        [ "$rc" -eq 0 ] && return 0
        [ "$rc" -ne 3 ] && return "$rc"
        sleep "$RETRY_WAIT"
    done
    return "$rc"
}

echo "$(stamp) prepare: start" >> "$EXITS"
if ! until_complete prepare "$PY" run_all.py --prepare \
        --workers "$PREPARE_WORKERS" "${EXTRA[@]}"; then
    echo "$(stamp) prepare did not complete; no shard started" | tee -a "$EXITS"
    exit 1
fi

declare -A PIDS
for focal in "${FOCALS[@]}"; do
    until_complete "$focal" "$PY" run_all.py --focal "$focal" --shard "$focal" \
        --workers "${WORKERS[$focal]}" "${EXTRA[@]}" &
    PIDS[$focal]=$!
done

failed=()
for focal in "${FOCALS[@]}"; do
    if ! wait "${PIDS[$focal]}"; then
        failed+=("$focal")
    fi
done
if [ "${#failed[@]}" -gt 0 ]; then
    echo "$(stamp) incomplete shards: ${failed[*]}; merge NOT run" | tee -a "$EXITS"
    exit 1
fi

echo "$(stamp) all ${#FOCALS[@]} shards complete; merging" | tee -a "$EXITS"
"$PY" tools/merge_shards.py > "$LOGS/merge.log" 2>&1 || {
    echo "$(stamp) merge refused; see $LOGS/merge.log" | tee -a "$EXITS"; exit 1; }
"$PY" run_all.py --analyse > "$LOGS/analyse.log" 2>&1 || {
    echo "$(stamp) analysis failed; see $LOGS/analyse.log" | tee -a "$EXITS"; exit 1; }

# 5. Verify before calling the run done (tools/verify_run.py). A dry run has
#    one unit per cell by design, so its coverage check does not apply.
VERIFY_ARGS=()
DRY=0
for ((i = 0; i < ${#EXTRA[@]}; i++)); do
    [ "${EXTRA[$i]}" = "--max-questions" ] && VERIFY_ARGS=(--max-questions "${EXTRA[$((i + 1))]}")
    [ "${EXTRA[$i]}" = "--dry-run" ] && DRY=1
done
if [ "$DRY" = 0 ]; then
    "$PY" tools/verify_run.py "${VERIFY_ARGS[@]}" > "$LOGS/verify.log" 2>&1 || {
        echo "$(stamp) VERIFICATION FAILED; see $LOGS/verify.log" | tee -a "$EXITS"; exit 1; }
fi
echo "$(stamp) DONE" | tee -a "$EXITS"
