#!/usr/bin/env python3
"""
clean_hedged_pool.py — strip rewrite preambles from the hedged pool, offline.

The 50-pair hand audit (25 Sept 2026) found that 63.9% of hedged rewrites
opened with an instruction echo ("Here's a rewritten version of the message
with tentative wording:"), and that 85.6% of WRh units showed at least one
such message to the focal model. That line announces a rewrite, so WR vs WRh
compared confident text with visibly rewritten text, not only confident with
hedged wording.

This removes the echo with src.hedged_personas.clean_rewrite, keeps the
rewrite itself untouched, and re-validates every row under the same rules
plus the new meta-text rule. No API call is made.

    python3 tools/clean_hedged_pool.py

Writes results/processed/hedged_personas.parquet (cleaned) and keeps the
first version as results/processed/hedged_personas_v1_preamble.parquet.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hedged_personas import clean_rewrite, validate_hedged  # noqa: E402


def main() -> int:
    config = yaml.safe_load((ROOT / "config/experiment.yaml").read_text(encoding="utf-8"))
    validation = config["hedged_pool"]["validation"]
    pool_path = ROOT / "results/processed/hedged_personas.parquet"
    keep_path = ROOT / "results/processed/hedged_personas_v1_preamble.parquet"
    if keep_path.exists():
        print(f"{keep_path.name} exists: the pool was already cleaned; refusing to run twice")
        return 1
    shutil.copy2(pool_path, keep_path)

    pool = pd.read_parquet(pool_path)
    before = pool["validation_pass_status"].eq("passed")
    changed = passed = 0
    for index, row in pool[before].iterrows():
        original = row["generated_persona_text"] or ""
        cleaned = clean_rewrite(original)
        ok, reason = validate_hedged(
            cleaned, row["source_persona_text"] or "",
            str(row["assigned_wrong_answer_letter_or_value"]), validation)
        changed += cleaned != original
        pool.at[index, "generated_persona_text"] = cleaned if ok else ""
        pool.at[index, "validation_pass_status"] = "passed" if ok else "failed"
        pool.at[index, "validation_failure_reason"] = "" if ok else f"after cleaning: {reason}"
        passed += ok
    pool["cleaned_of_preamble"] = before
    pool.to_parquet(pool_path, index=False)

    record = {
        "rows": int(len(pool)),
        "passed_before": int(before.sum()),
        "preamble_removed": int(changed),
        "passed_after_cleaning": int(passed),
        "dropped_by_cleaning": int(before.sum() - passed),
        "kept_first_version_as": keep_path.name,
    }
    (ROOT / "results/processed/hedged_cleaning_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
