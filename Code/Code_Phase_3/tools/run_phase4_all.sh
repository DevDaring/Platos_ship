#!/usr/bin/env bash
# run_phase4_all.sh — run every pre-registered Phase 4 experiment (PREREG_PHASE4.md).
#
# One chain per provider-bound model, so no provider sees more concurrency than
# the Phase 3 run measured as safe (tools/run_parallel.sh):
#   DeepSeek-v4-flash : A -> C -> (wait for B inputs) -> B
#   Gemma-3-27B       : A -> C -> (wait for B inputs) -> B
#   weak peers        : B bank -> eligible cohort -> B Round-0 for three models
#   Llama-3.1-8B      : (wait for B inputs) -> B
# Every step resumes from its checkpoint; exit code 3 (failed calls) is retried.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=results/phase4/launcher
mkdir -p "$LOG"
MAX_ATTEMPTS=${MAX_ATTEMPTS:-12}

step() {   # step <name> <args...>
    local name=$1; shift
    local attempt code
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        "$PY" tools/run_phase4.py "$@" >> "$LOG/$name.log" 2>&1
        code=$?
        echo "$(date -u +%FT%TZ) $name attempt=$attempt exit=$code" >> "$LOG/exits.txt"
        [ "$code" -eq 0 ] && return 0
        [ "$code" -ne 3 ] && return "$code"
        sleep 60
    done
    return 3
}

wait_for_b_inputs() {
    until [ -f results/phase4/B/READY ]; do sleep 60; done
}

(
    step bank bank --workers 12 &&
    step eligible eligible &&
    { step r0_deepseek r0 --focal deepseek_primary --workers 6 &
      step r0_gemma27 r0 --focal sweep_gemma_3_27b --workers 4 &
      step r0_llama8 r0 --focal sweep_llama_3_1_8b_focal --workers 12 &
      wait; } &&
    touch results/phase4/B/READY
) &

(
    step A_deepseek run --exp A --focal deepseek_primary --workers 2 &&
    step C_deepseek run --exp C --focal deepseek_primary --workers 3 &&
    wait_for_b_inputs &&
    step B_deepseek run --exp B --focal deepseek_primary --workers 2
) &

(
    step A_gemma27 run --exp A --focal sweep_gemma_3_27b --workers 2 &&
    step C_gemma27 run --exp C --focal sweep_gemma_3_27b --workers 3 &&
    wait_for_b_inputs &&
    step B_gemma27 run --exp B --focal sweep_gemma_3_27b --workers 2
) &

(
    wait_for_b_inputs &&
    step B_llama8 run --exp B --focal sweep_llama_3_1_8b_focal --workers 3
) &

wait
echo "$(date -u +%FT%TZ) ALL DONE" >> "$LOG/exits.txt"
