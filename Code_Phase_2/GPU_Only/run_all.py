#!/usr/bin/env python3
"""
run_all.py — GPU_Only entry point (Experiment E9, mechanistic logprob probe).

    python3 run_all.py                      # probe C1/C2/C4 on Llama-3.1-8B
    python3 run_all.py --model <hf_repo>    # different open focal
    python3 run_all.py --max-questions 30   # quick pilot
    python3 run_all.py --conditions C1_smart_solo,C4_one_smart_two_dumb

Hardware: one 24 GB card (RTX 4090) is enough for an 8B focal in bf16 (~10–15 h
for 300 q x 3 reps x 3 conditions). See README.md and API_AND_RECHARGE.md §GPU.

This is the ONLY component that needs a GPU. Everything else is API-only and
lives in ../CPU_Only. It reuses ../CPU_Only's question pool + personas + prompt
builders so the setup is identical to the main experiment (symmetry).
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

GPU_ROOT = Path(__file__).parent.resolve()
CPU_ROOT = (GPU_ROOT.parent / "CPU_Only").resolve()


def setup_logging():
    logs = GPU_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(logs / "gpu_probe.log"); fh.setFormatter(fmt)
    ch = logging.StreamHandler(); ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root = logging.getLogger(); root.setLevel(logging.INFO); root.addHandler(fh); root.addHandler(ch)


def main():
    ap = argparse.ArgumentParser(description="Phase 2 GPU logprob probe (E9)")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--conditions", default="C1_smart_solo,C2_three_smart,C4_one_smart_two_dumb")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="2 questions x 1 replication, no stage cache — full code path")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore the stage cache and recompute every stage")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--eager", action="store_true",
                    help="Skip torch.compile/CUDA graphs (avoids a long cold-start compile)")
    args = ap.parse_args()

    if args.dry_run:
        args.max_questions = args.max_questions or 2
        args.reps = 1

    setup_logging()
    log = logging.getLogger("platos_ship.gpu_run")

    # Load env (HF token for gated models) from Code_Phase_2/.env
    parent_env = GPU_ROOT.parent / ".env"
    if parent_env.exists():
        load_dotenv(parent_env)
    if os.getenv("HUGGINGFACE_TOKEN"):
        os.environ.setdefault("HF_TOKEN", os.getenv("HUGGINGFACE_TOKEN"))

    if not (CPU_ROOT / "data" / "processed" / "question_pool.parquet").exists():
        log.error(
            "CPU_Only/data/processed/question_pool.parquet not found. Run the CPU_Only "
            "pipeline first (it builds the question pool + personas this probe reuses)."
        )
        sys.exit(1)

    from src.logprob_probe import run_logprob_probe
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    run_logprob_probe(
        gpu_project_root=GPU_ROOT,
        cpu_project_root=CPU_ROOT,
        conditions=conditions,
        trials_per_question=args.reps,
        model_repo=args.model,
        max_questions=args.max_questions,
        resume=not (args.no_resume or args.dry_run),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=args.eager,
    )
    log.info("GPU probe done. Outputs in GPU_Only/data/outputs/.")


if __name__ == "__main__":
    main()
