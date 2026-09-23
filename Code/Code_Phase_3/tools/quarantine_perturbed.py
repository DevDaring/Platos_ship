#!/usr/bin/env python3
"""
quarantine_perturbed.py — isolate the invalid perturbed-GSM8K artefacts.

What is wrong
-------------
`Code_Phase_2/CPU_Only/src/perturbed_gsm8k.py` builds the contamination probe by

    perturb_question_text()  : multiply EVERY standalone integer by 4
    recompute_linear_answer(): multiply the gold answer by 4

which is correct only when the answer is a linear function of every scaled
operand. It is not for percentages, fractions or ratio quantities, and the
pool contains all three: 12 of 97 items carry a scaled `%`, 10 carry a scaled
fraction. Verified by hand:

    gsm8k_perturbed_0005  saved gold     10,400   true    41,600
    gsm8k_perturbed_0007  saved gold  1,540,000   true 7,840,000
    gsm8k_perturbed_0000  "4/8 an ounce" — the fraction itself was scaled

The pool's own probe agreed with the generated label on 11 of 97 items.

Consequence for the paper
-------------------------
Section 4.5 of the reviewed manuscript reports solo accuracy "falling" from
76.9% to 21.6% on perturbed items and a residual +5.4-point gain. Those
numbers measure label corruption, not difficulty, so they cannot support any
claim about contamination or recomputation. Every sentence resting on them is
withdrawn; X4 (GSM-Symbolic) replaces the probe with template-computed labels.

This script does not delete anything. It copies the affected artefacts into a
quarantine directory with a README stating the defect, and writes a machine-
readable record the registry reads so those trials can never re-enter an
analysis by accident.

    python3 tools/quarantine_perturbed.py --dry-run
    python3 tools/quarantine_perturbed.py
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent               # Code/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("platos_ship3.quarantine")

QUARANTINE_DIRNAME = "QUARANTINE_perturbed_invalid"

TARGETS = [
    "Code_Phase_2/results/processed/perturbed_gsm8k_pool.parquet",
    "Code_Phase_2/results/processed/perturbed_personas.parquet",
]

# Stale analysis output: it reports Spearman -0.22 and DeepSeek solo accuracy
# 0.2165 because the perturbed trials leaked into the sweep grouping. The
# manuscript states -0.95 and 0.762. Both cannot be right; this one is wrong.
STALE_ANALYSIS = "Code_Phase_2/results/outputs/capability_sweep_analysis.json"

README = """# QUARANTINE — perturbed-GSM8K artefacts (invalid gold answers)

These files are retained for provenance. **Do not use them in any analysis.**

## The defect

`Code_Phase_2/CPU_Only/src/perturbed_gsm8k.py`:

* `perturb_question_text()` multiplies every standalone integer in the question
  by 4, including the numerator and denominator of fractions and the operand of
  a percentage;
* `recompute_linear_answer()` multiplies the original gold answer by 4.

Scaling the answer by the same factor is valid only when the answer is a linear
function of every scaled operand. Counter-examples from the saved pool:

| Item | Saved gold | Correct answer |
|---|---:|---:|
| `gsm8k_perturbed_0005` | 10,400 | 41,600 |
| `gsm8k_perturbed_0007` | 1,540,000 | 7,840,000 |
| `gsm8k_perturbed_0000` | 8 | question is infeasible as rewritten ("4/8 an ounce") |

Of 97 items, 12 contain a scaled `%` and 10 a scaled fraction. The pool's own
verification probe agreed with the generated label on 11 of 97.

## What this invalidates

Section 4.5 of the reviewed manuscript: the 76.9% -> 21.6% solo-accuracy drop
and the residual +5.4-point gain. Those quantities reflect corrupted labels,
not item difficulty, so they support no conclusion about contamination or about
answer recomputation.

## Replacement

Phase 3 experiment **X4** uses GSM-Symbolic (Mirzadeh et al., ICLR 2025,
arXiv:2410.05229), whose instances are generated from symbolic templates, so
the answer is computed rather than inferred. See
`Code_Phase_3/src/gsm_symbolic.py`. The claim it licenses is a comparison of
the treatment effect on regenerated items against matched originals — not a
statement that contamination has been ruled out.
"""


def quarantine(dry_run: bool = False) -> Dict[str, Any]:
    destination = REPO_ROOT / "Code_Phase_2" / "results" / QUARANTINE_DIRNAME
    record: Dict[str, Any] = {
        "quarantine_directory": str(destination),
        "moved": [],
        "absent": [],
        "stale_analysis": None,
        "reason": "perturbed-GSM8K gold answers invalid; see README.md",
    }

    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "README.md").write_text(README, encoding="utf-8")

    for relative in TARGETS:
        source = REPO_ROOT / relative
        if not source.exists():
            record["absent"].append(relative)
            logger.info("absent (nothing to quarantine): %s", relative)
            continue
        target = destination / Path(relative).name
        logger.info("%s %s -> %s", "WOULD copy" if dry_run else "copying",
                    source, target)
        if not dry_run:
            shutil.copy2(source, target)
        record["moved"].append({"from": relative, "to": str(target)})

    stale = REPO_ROOT / STALE_ANALYSIS
    if stale.exists():
        target = destination / "capability_sweep_analysis.STALE.json"
        logger.info("%s stale analysis %s -> %s",
                    "WOULD copy" if dry_run else "copying", stale, target)
        if not dry_run:
            shutil.copy2(stale, target)
            stale.unlink()
        record["stale_analysis"] = {
            "from": STALE_ANALYSIS,
            "to": str(target),
            "why": ("reports Spearman -0.22 and DeepSeek solo accuracy 0.2165 "
                    "because perturbed trials leaked into the sweep grouping; "
                    "regenerate with analysis/run_analysis.py"),
        }

    if not dry_run:
        with open(destination / "quarantine_record.json", "w",
                  encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        logger.info("Quarantine complete -> %s", destination)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would happen and change nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    quarantine(args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
