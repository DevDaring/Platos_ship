#!/usr/bin/env python3
"""
run_phase4.py — the pre-registered Phase 4 experiments (PREREG_PHASE4.md).

Every write goes under results/phase4/; the canonical Protocol-B outputs are
only READ (the Round-0 cache for Experiments A and C). One process per focal
model; inside a process the arms of an experiment run CONCURRENTLY, each with
its own worker pool, so the arms share one collection window.

    # Experiment A (visibility factorial) and C (round control): cached R0
    python tools/run_phase4.py run --exp A --focal deepseek_primary --workers 3
    python tools/run_phase4.py run --exp C --focal sweep_gemma_3_27b --workers 3

    # Experiment B (natural errors, held-out pool)
    python tools/build_heldout_pool.py
    python tools/run_phase4.py bank --workers 8          # natural weak-peer bank
    python tools/run_phase4.py eligible                   # freeze the eligible cohort
    python tools/run_phase4.py r0 --focal deepseek_primary --workers 6
    python tools/run_phase4.py run --exp B --focal deepseek_primary --workers 2

Exit code 3 when some calls failed (re-run the same command; it resumes).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_all  # noqa: E402
from src.agents import build_agents  # noqa: E402
from src.call_guard import failure_counts  # noqa: E402
from src.peer_pools import load_pools  # noqa: E402
from src.r0_cache import build_r0_cache, load_r0_cache, r0_lookup  # noqa: E402
from src.revision_runner import run_condition  # noqa: E402
from src.snapshots import SnapshotAuditor  # noqa: E402
from src.store import Checkpoint, IncrementalWriter  # noqa: E402

P4 = ROOT / "results/phase4"
CANONICAL_R0 = ROOT / "results/outputs/r0_cache.parquet"
PEER_KEYS = ["unit_id", "peer_display_name", "shown_to_focal"]

# Frozen in PREREG_PHASE4.md.
DESIGN = {
    "A": {"conditions": ["A_R", "A_WR", "A_Rhid", "A_WRhid"], "pool": "main300",
          "rounds": 1, "focal": ["deepseek_primary", "sweep_gemma_3_27b"]},
    "C": {"conditions": ["C_R", "C_WR"], "pool": "mitigation100", "rounds": 3,
          "focal": ["deepseek_primary", "sweep_gemma_3_27b"]},
    "B": {"conditions": ["B_R", "B_P0", "B_P1", "B_P2"], "pool": "heldout",
          "rounds": 1,
          "focal": ["deepseek_primary", "sweep_gemma_3_27b", "sweep_llama_3_1_8b_focal"]},
}
BANK_SAMPLES_PER_MODEL = 4


def _setup(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    run_all.setup_logging(folder / "logs", verbose=False)
    run_all.load_env(ROOT)
    configs = run_all.load_configs(ROOT)
    return configs["experiment"], dict(configs["paths"])


def _heldout_pool() -> pd.DataFrame:
    return pd.read_parquet(P4 / "B/heldout_pool.parquet")


def _eligible_pool() -> pd.DataFrame:
    ids = json.loads((P4 / "B/eligible.json").read_text(encoding="utf-8"))["eligible"]
    pool = _heldout_pool()
    return pool[pool["question_identifier"].isin(set(ids))].reset_index(drop=True)


def cmd_bank(workers: int) -> int:
    from src.honest_bank import build_honest_bank

    folder = P4 / "B"
    experiment, paths = _setup(folder)
    agents = build_agents(ROOT, focal_keys=[], need_weak=True, need_judge=True,
                          need_verifier=False)
    defaults = agents["models_config"].get("request_defaults", {})
    build_honest_bank(
        questions=_heldout_pool(), weak_agents=agents["weak_agents"],
        weak_specs=agents["weak_specs"], judge_cascade=agents["judge_cascade"],
        bank_path=folder / "natural_bank.parquet",
        checkpoint=Checkpoint(folder / "checkpoint_bank.parquet"),
        replicates=BANK_SAMPLES_PER_MODEL,
        temperature=float(defaults.get("weak_temperature", 0.9)),
        max_output_tokens=int(defaults.get("weak_max_output_tokens", 1024)),
        workers=workers)
    return 3 if failure_counts() else 0


def cmd_eligible() -> int:
    """Freeze the eligible cohort: >= 2 correct and >= 2 wrong usable answers."""
    from src.call_guard import FAILED_STATUSES

    bank = pd.read_parquet(P4 / "B/natural_bank.parquet")
    usable = bank[~bank["error_status"].astype(str).isin(FAILED_STATUSES)
                  & bank["message_text"].fillna("").str.strip().ne("")
                  & bank["extracted_answer"].fillna("").astype(str).str.strip().ne("")]
    counts = usable.groupby("question_identifier")["is_correct"].agg(
        n_correct=lambda s: int(s.astype(bool).sum()),
        n_wrong=lambda s: int((~s.astype(bool)).sum()))
    eligible = counts[(counts["n_correct"] >= 2) & (counts["n_wrong"] >= 2)]
    pool = _heldout_pool()
    report = {
        "bank_rows": int(len(bank)), "usable_rows": int(len(usable)),
        "questions": int(len(pool)),
        "questions_with_usable_answers": int(len(counts)),
        "eligible": sorted(eligible.index.tolist()),
        "n_eligible": int(len(eligible)),
        "natural_error_rate": float(1 - usable["is_correct"].astype(bool).mean()),
        "error_rate_by_model": usable.groupby("weak_model_key")["is_correct"].apply(
            lambda s: float(1 - s.astype(bool).mean())).to_dict(),
        "eligible_by_subject": pool[pool["question_identifier"].isin(eligible.index)]
        ["subject_category"].value_counts().to_dict(),
    }
    (P4 / "B/eligible.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print({k: v for k, v in report.items() if k != "eligible"})
    return 0


def cmd_r0(focal: str, workers: int) -> int:
    folder = P4 / "B" / focal
    experiment, paths = _setup(folder)
    agents = build_agents(ROOT, focal_keys=[focal], need_weak=False, need_judge=True,
                          need_verifier=False)
    defaults = agents["models_config"].get("request_defaults", {})
    auditor = SnapshotAuditor()
    build_r0_cache(
        questions=_eligible_pool(), focal_keys=[focal],
        focal_agents=agents["focal_agents"], focal_specs=agents["focal_specs"],
        judge_cascade=agents["judge_cascade"], auditor=auditor,
        cache_path=folder / "r0_cache.parquet",
        checkpoint=Checkpoint(folder / "checkpoint_r0.parquet"),
        replicates=int(experiment["replicates_per_question"]),
        temperature=float(experiment["r0_cache"]["temperature"]),
        max_output_tokens=int(experiment["r0_cache"]["max_output_tokens"]),
        workers=workers)
    auditor.save(folder / "snapshot_audit_r0.parquet")
    return 3 if failure_counts() else 0


def cmd_run(exp: str, focal: str, workers: int) -> int:
    design = DESIGN[exp]
    if focal not in design["focal"]:
        print(f"refused: {focal} is not a pre-registered model for Experiment {exp}")
        return 1
    folder = P4 / exp / focal
    experiment, paths = _setup(folder)
    if exp == "B":
        questions = _eligible_pool()
        r0 = load_r0_cache(P4 / "B" / focal / "r0_cache.parquet")
        paths["natural_bank_file"] = str(P4 / "B/natural_bank.parquet")
    else:
        questions = run_all.load_pool(design["pool"], experiment, paths, ROOT,
                                      int(experiment["random_seed"]))
        r0 = load_r0_cache(CANONICAL_R0)
    agents = build_agents(ROOT, focal_keys=[focal], need_weak=True, need_judge=True,
                          need_verifier=False)
    defaults = agents["models_config"].get("request_defaults", {})
    pools = load_pools(paths, ROOT)
    index = r0_lookup(r0)
    checkpoint = Checkpoint(folder / "checkpoint.parquet")
    revisions = IncrementalWriter(folder / "revision_log.parquet", flush_every=50,
                                  checkpoint=checkpoint)
    peers = IncrementalWriter(folder / "peer_message_log.parquet", flush_every=100)
    auditor = SnapshotAuditor()
    calls: dict = {}

    def arm(condition: str) -> None:
        calls[condition] = run_condition(
            condition_name=condition, condition=experiment["conditions"][condition],
            focal_key=focal, focal_agent=agents["focal_agents"][focal],
            focal_spec=agents["focal_specs"][focal], questions=questions,
            r0_index=index, pools=pools, weak_specs=agents["weak_specs"],
            weak_agents=agents["weak_agents"], judge_cascade=agents["judge_cascade"],
            auditor=auditor, revision_writer=revisions, peer_writer=peers,
            checkpoint=checkpoint, master_seed=int(experiment["random_seed"]),
            replicates=int(experiment["replicates_per_question"]),
            rounds=design["rounds"], rounds_config=experiment.get("rounds", {}),
            temperature=float(defaults.get("focal_temperature", 0.7)),
            max_output_tokens=int(defaults.get("focal_max_output_tokens", 2048)),
            experiment_name=f"Phase4_{exp}", workers=workers,
            prior_rounds=_prior_rounds(folder))

    threads = [threading.Thread(target=arm, args=(c,), name=f"arm-{c}")
               for c in design["conditions"]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    revisions.consolidate(dedup_on=["unit_id"])
    peers.consolidate(dedup_on=PEER_KEYS)
    checkpoint.save()
    auditor.save(folder / "snapshot_audit.parquet")
    failed = failure_counts()
    (folder / "run_metadata.json").write_text(json.dumps({
        "experiment": exp, "focal": focal, "conditions": design["conditions"],
        "revision_calls": calls, "failed_calls_not_recorded": failed,
        "prereg": "PREREG_PHASE4.md",
        "run_timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
    }, indent=2), encoding="utf-8")
    logging.getLogger("phase4").info("%s/%s: %s calls, failures %s", exp, focal, calls, failed)
    return 3 if failed else 0


def _prior_rounds(folder: Path):
    from src.revision_runner import load_prior_rounds

    return load_prior_rounds(folder / "revision_log.parquet")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank")
    b.add_argument("--workers", type=int, default=4)
    sub.add_parser("eligible")
    r0 = sub.add_parser("r0")
    r0.add_argument("--focal", required=True)
    r0.add_argument("--workers", type=int, default=4)
    run = sub.add_parser("run")
    run.add_argument("--exp", required=True, choices=sorted(DESIGN))
    run.add_argument("--focal", required=True)
    run.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.cmd == "bank":
        return cmd_bank(args.workers)
    if args.cmd == "eligible":
        return cmd_eligible()
    if args.cmd == "r0":
        return cmd_r0(args.focal, args.workers)
    return cmd_run(args.exp, args.focal, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
