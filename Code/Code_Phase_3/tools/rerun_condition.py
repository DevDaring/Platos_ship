#!/usr/bin/env python3
"""
rerun_condition.py — re-run ONE condition on the merged outputs, then splice.

Written for WRh after the hand audit found rewrite preambles in its peer
messages (tools/clean_hedged_pool.py). Everything else is held identical to
the main run: the same merged Round-0 cache (so the initial answer each unit
starts from is the very record every other condition used), the same prompt,
temperature, output cap, judge, peer seeding (src/peer_pools.peer_rng) and
experiment label. Only the hedged messages differ, and only by the removed
preamble line.

    # 1. run (resumable; writes only under results/outputs/rerun_<COND>/)
    python3 tools/rerun_condition.py --condition WRh --experiment X2_message_decomposition \
        --focal deepseek_primary --workers 6
    # 2. after every model is done, move the old rows to an archive and splice
    python3 tools/rerun_condition.py --condition WRh --splice

Exit code 3 when some calls failed (re-run the same command).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_all  # noqa: E402
from src.agents import build_agents  # noqa: E402
from src.call_guard import failure_counts  # noqa: E402
from src.peer_pools import load_pools  # noqa: E402
from src.r0_cache import load_r0_cache, r0_lookup  # noqa: E402
from src.revision_runner import run_condition  # noqa: E402
from src.snapshots import SnapshotAuditor  # noqa: E402
from src.store import Checkpoint, IncrementalWriter  # noqa: E402

OUT = ROOT / "results/outputs"
PEER_KEYS = ["unit_id", "peer_display_name", "shown_to_focal"]


def run(condition: str, experiment_name: str, focal: str, workers: int) -> int:
    configs = run_all.load_configs(ROOT)
    experiment, paths = configs["experiment"], configs["paths"]
    folder = OUT / f"rerun_{condition}" / focal
    folder.mkdir(parents=True, exist_ok=True)
    run_all.setup_logging(folder / "logs", verbose=False)
    run_all.load_env(ROOT)

    spec = experiment["experiments"][experiment_name]
    questions = run_all.load_pool(spec.get("pool", "main300"), experiment, paths,
                                  ROOT, int(experiment["random_seed"]))
    agents = build_agents(ROOT, focal_keys=[focal], need_weak=True,
                          need_judge=True, need_verifier=False)
    defaults = agents["models_config"].get("request_defaults", {})
    r0 = load_r0_cache(OUT / "r0_cache.parquet")
    checkpoint = Checkpoint(folder / "checkpoint.parquet")
    revisions = IncrementalWriter(folder / "revision_log.parquet", flush_every=50,
                                  checkpoint=checkpoint)
    peers = IncrementalWriter(folder / "peer_message_log.parquet", flush_every=100)
    auditor = SnapshotAuditor()

    calls = run_condition(
        condition_name=condition, condition=experiment["conditions"][condition],
        focal_key=focal, focal_agent=agents["focal_agents"][focal],
        focal_spec=agents["focal_specs"][focal], questions=questions,
        r0_index=r0_lookup(r0), pools=load_pools(paths, ROOT),
        weak_specs=agents["weak_specs"], weak_agents=agents["weak_agents"],
        judge_cascade=agents["judge_cascade"], auditor=auditor,
        revision_writer=revisions, peer_writer=peers, checkpoint=checkpoint,
        master_seed=int(experiment["random_seed"]),
        replicates=int(experiment["replicates_per_question"]),
        rounds=1, rounds_config=experiment.get("rounds", {}),
        temperature=float(defaults.get("focal_temperature", 0.7)),
        max_output_tokens=int(defaults.get("focal_max_output_tokens", 2048)),
        experiment_name=experiment_name, workers=workers)
    revisions.consolidate(dedup_on=["unit_id"])
    peers.consolidate(dedup_on=PEER_KEYS)
    checkpoint.save()
    auditor.save(folder / "snapshot_audit.parquet")
    failed = failure_counts()
    (folder / "rerun_metadata.json").write_text(json.dumps({
        "condition": condition, "experiment": experiment_name, "focal": focal,
        "revision_calls": calls, "failed_calls_not_recorded": failed,
        "reason": "hedged messages cleaned of rewrite preambles "
                  "(tools/clean_hedged_pool.py)",
        "run_timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
    }, indent=2), encoding="utf-8")
    logging.getLogger("rerun").info("%s/%s: %d calls, failures %s", condition, focal,
                                    calls, failed)
    return 3 if failed else 0


def splice(condition: str) -> int:
    """Archive the old rows of the condition and put the re-run rows in."""
    rerun_root = OUT / f"rerun_{condition}"
    metas = [json.loads(p.read_text(encoding="utf-8"))
             for p in rerun_root.glob("*/rerun_metadata.json")]
    if not metas or any(m["failed_calls_not_recorded"] for m in metas):
        print("refused: some model has not finished or has unretried failures")
        return 1
    new_rev = pd.concat([pd.read_parquet(p) for p in rerun_root.glob("*/revision_log.parquet")])
    new_peer = pd.concat([pd.read_parquet(p) for p in rerun_root.glob("*/peer_message_log.parquet")])

    rev = pd.read_parquet(OUT / "revision_log.parquet")
    peer = pd.read_parquet(OUT / "peer_message_log.parquet")
    done = pd.read_parquet(OUT / "completed_units.parquet")
    old = rev["condition"] == condition
    if set(rev.loc[old, "focal_key"]) != set(new_rev["focal_key"]):
        print("refused: re-run models differ from the models that ran the condition")
        return 1
    if len(new_rev) < 0.97 * old.sum():
        print(f"refused: re-run has {len(new_rev)} rows vs {int(old.sum())} before")
        return 1

    archive = OUT / "archive"
    archive.mkdir(exist_ok=True)
    old_units = set(rev.loc[old, "unit_id"])
    rev[old].to_parquet(archive / f"{condition}_v1_preamble_revision_log.parquet", index=False)
    peer[peer["unit_id"].isin(old_units)].to_parquet(
        archive / f"{condition}_v1_preamble_peer_message_log.parquet", index=False)
    for name in ("revision_log", "peer_message_log", "completed_units"):
        shutil.copy2(OUT / f"{name}.parquet", archive / f"{name}.before_{condition}_splice.parquet")

    rev = pd.concat([rev[~old], new_rev], ignore_index=True)
    peer = pd.concat([peer[~peer["unit_id"].isin(old_units)], new_peer], ignore_index=True)
    done = pd.concat([done[~done["unit_id"].isin(old_units)],
                      new_rev[["unit_id"]]], ignore_index=True).drop_duplicates("unit_id")
    assert rev["unit_id"].is_unique
    rev.to_parquet(OUT / "revision_log.parquet", index=False)
    peer.to_parquet(OUT / "peer_message_log.parquet", index=False)
    done.to_parquet(OUT / "completed_units.parquet", index=False)
    print(f"spliced {condition}: {int(old.sum())} old rows archived, {len(new_rev)} new rows in")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--experiment")
    parser.add_argument("--focal")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--splice", action="store_true")
    args = parser.parse_args()
    if args.splice:
        return splice(args.condition)
    return run(args.condition, args.experiment, args.focal, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
