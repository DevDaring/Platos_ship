#!/usr/bin/env python3
"""
verify_run.py — check a finished Phase 3 run before anyone reads a number from it.

    python3 tools/verify_run.py                  # full run: 300 / 100 / 200 items
    python3 tools/verify_run.py --max-questions 4

Reads the merged outputs only (no API calls). Exits 1 if any HARD check
fails; SOFT findings are printed for the write-up but do not fail the run.

HARD
  * no row from a failed call (failure / snapshot_unavailable)
  * every focal response served by its pinned model
  * every (experiment, condition, model) cell has its designed units, less
    only units the design allows to skip (agreement infeasible, hedged
    rewrite missing, honest message missing), and none beyond them
  * WR, W and SF show the same personas per (question, replicate); WR1's
    persona is WR's first and WR's pair is WR4's first two
  * WRh shows the hedged rewrites of WR's personas
  * X6 verified only X6's models, never the verifier's own model
  * the analysis wrote paper_numbers.json and it passed the design check

SOFT
  * per model: truncated, unparsed, empty-text shares; serving routes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import resolve_focal_selector  # noqa: E402
from src.call_guard import FAILED_STATUSES  # noqa: E402

# Units a condition may legitimately skip, and the reason prefix it logs.
SKIPPABLE = {"WRagree", "WRdiff", "WRh", "H", "Hfilt"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    experiment = yaml.safe_load((ROOT / "config/experiment.yaml").read_text(encoding="utf-8"))
    models = yaml.safe_load((ROOT / "config/models.yaml").read_text(encoding="utf-8"))
    paths = yaml.safe_load((ROOT / "config/paths.yaml").read_text(encoding="utf-8"))
    out = ROOT / "results/outputs"
    focal_specs = models["focal_agents"]
    reps = int(experiment["replicates_per_question"])

    hard, soft = [], []
    rev = pd.read_parquet(out / "revision_log.parquet")
    r0 = pd.read_parquet(out / "r0_cache.parquet")
    peers = pd.read_parquet(out / "peer_message_log.parquet")

    # ── failed calls ──────────────────────────────────────────────────────
    for name, frame in (("revision", rev), ("round0", r0)):
        n = int(frame["error_status"].astype(str).isin(FAILED_STATUSES).sum())
        if n:
            hard.append(f"{name}: {n} rows from failed calls")

    # ── provenance ────────────────────────────────────────────────────────
    for name, frame, col in (("revision", rev, "focal_served_model"),
                             ("round0", r0, "focal_served_model")):
        for focal, group in frame.groupby("focal_key"):
            prefix = str(focal_specs[focal]["expected_served_prefix"]).lower()
            bad = group[~group[col].astype(str).str.lower().str.startswith(prefix)]
            if len(bad):
                hard.append(f"{name}/{focal}: {len(bad)} rows served by "
                            f"{sorted(bad[col].astype(str).unique())[:3]}, pin {prefix}")

    # ── coverage ──────────────────────────────────────────────────────────
    def n_questions(pool: str) -> int:
        spec = experiment["pools"][pool]
        n = int(spec.get("n_questions", 0)) * (2 if spec.get("pair_with_originals") else 1)
        return min(n, args.max_questions) if args.max_questions else n

    round1 = rev[rev["round_index"] == 1]
    counts = round1.groupby(["experiment", "condition", "focal_key"]).size()
    for exp_name, spec in experiment["experiments"].items():
        if not spec.get("enabled") or spec.get("offline") or exp_name.startswith("X6"):
            continue
        designed = n_questions(spec.get("pool", "main300")) * reps
        conditions = list(spec.get("conditions", []))
        if spec.get("round_sweep"):
            conditions += [f"{spec['round_sweep']['condition']}_rounds{r}"
                           for r in spec["round_sweep"]["rounds"]]
        for focal in resolve_focal_selector(spec.get("focal"), focal_specs):
            for condition in conditions:
                got = int(counts.get((exp_name, condition, focal), 0))
                if got > designed:
                    hard.append(f"{exp_name}/{condition}/{focal}: {got} units > designed {designed}")
                elif got < designed:
                    (soft if condition in SKIPPABLE else hard).append(
                        f"{exp_name}/{condition}/{focal}: {got} of {designed} units")
    for condition in [c for c in rev["condition"].unique() if c.startswith("WR_rounds")]:
        n_rounds = int(condition.rsplit("rounds", 1)[1])
        per_unit = rev[rev["condition"] == condition].groupby(
            ["focal_key", "question_identifier", "replicate"])["round_index"].nunique()
        if (per_unit != n_rounds).any():
            hard.append(f"{condition}: {(per_unit != n_rounds).sum()} units missing rounds")

    # ── persona matching ──────────────────────────────────────────────────
    meta = rev[rev["round_index"] == 1][["unit_id", "condition", "focal_key",
                                         "question_identifier", "replicate"]]
    shown = peers[peers["shown_to_focal"].astype(bool)].merge(meta, on="unit_id")
    sets = (shown.dropna(subset=["persona_identifier"])
            .groupby(["condition", "question_identifier", "replicate", "focal_key"])
            ["persona_identifier"].apply(lambda s: tuple(sorted(set(s)))))
    by_cell = sets.groupby(level=[0, 1, 2]).apply(lambda s: set(s))
    if (by_cell.map(len) > 1).any():
        hard.append("some (condition, question, replicate) cells showed different "
                    "personas to different models")
    first = sets.groupby(level=[0, 1, 2]).first()
    for other in ("W", "SF", "WR_rounds2", "WR_rounds3", "WRh"):
        for (q, r), wr in first.xs("WR", level=0).items():
            if (other, q, r) in first.index and first[(other, q, r)] != wr:
                hard.append(f"{other} differs from WR at ({q}, r{r})")
                break
    for (q, r), wr in first.xs("WR", level=0).items() if "WR" in first.index.get_level_values(0) else []:
        for dose, need in (("WR1", 1), ("WR4", 4)):
            if (dose, q, r) in first.index:
                got = set(first[(dose, q, r)])
                if (need == 1 and not got <= set(wr)) or (need == 4 and not set(wr) <= got):
                    hard.append(f"{dose} not nested with WR at ({q}, r{r})")
    hedged_modes = shown[shown["condition"] == "WRh"]["anchor_mode"].unique().tolist()
    if hedged_modes and hedged_modes != ["hedged"]:
        hard.append(f"WRh showed non-hedged peers: {hedged_modes}")

    # ── X6 scope ──────────────────────────────────────────────────────────
    safeguard = out / "safeguard_results.parquet"
    if safeguard.exists():
        verified = pd.read_parquet(safeguard)
        x6 = experiment["experiments"]["X6_verification_safeguard"]
        allowed = set(resolve_focal_selector(x6.get("focal"), focal_specs))
        units = rev.set_index("unit_id")["focal_key"]
        seen = set(units.reindex(verified["unit_id"]).dropna())
        if seen - allowed:
            hard.append(f"X6 verified models outside its design: {sorted(seen - allowed)}")
        slug = models["verifier_agent"]["model_slug"]
        if any(focal_specs[f]["model_slug"] == slug for f in seen):
            hard.append("X6 verified a model with itself")

    # ── analysis ──────────────────────────────────────────────────────────
    numbers = out / "paper_numbers.json"
    if not numbers.exists():
        hard.append("paper_numbers.json missing: analysis did not run")
    else:
        pn = json.loads(numbers.read_text(encoding="utf-8"))
        if not pn.get("design_check", {}).get("passed"):
            hard.append(f"design check failed: {pn.get('design_check')}")

    # ── soft: parse quality and routes ────────────────────────────────────
    for name, frame in (("round0", r0), ("revision", rev)):
        g = frame.assign(truncated=frame["finish_reason"].astype(str).eq("length"),
                         unparsed=frame["extracted_answer"].isna(),
                         empty=frame["raw_response_text"].fillna("").str.strip().eq(""))
        table = g.groupby("focal_key")[["truncated", "unparsed", "empty"]].mean().round(3)
        soft.append(f"{name} quality:\n{table.to_string()}")
        if "served_route" in frame.columns:
            routes = frame.groupby(["focal_key", "served_route"]).size().unstack(fill_value=0)
            soft.append(f"{name} routes:\n{routes.to_string()}")

    print("=" * 72)
    print(f"verify_run: {len(hard)} HARD failure(s), {len(soft)} note(s)")
    print("=" * 72)
    for line in hard:
        print("HARD  ", line)
    for line in soft:
        print("note  ", line)
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
