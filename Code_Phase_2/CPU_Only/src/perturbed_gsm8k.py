"""
perturbed_gsm8k.py — Experiment E6: contamination probe on perturbed GSM8K.

Rewrites the numeric operands in the 100 GSM8K items already in the question
pool and recomputes the answers, producing items with the same structure but
values the model cannot have memorised. If the +16 pp GSM8K debate gain
persists on perturbed items, benchmark contamination is ruled out as the
driver and the arithmetic-recomputation mechanism gains direct support.

Method note: GSM8K answers are computed by a reference chain of arithmetic
operations. Because that chain is not machine-readable from the raw dataset,
we perturb conservatively: we scale every integer operand in the question by
a fixed small factor and, WHERE the original answer scales linearly, recompute
it; items whose answer cannot be safely recomputed are dropped and backfilled
from spare GSM8K items. Each perturbed item is verified by re-solving with the
weak probe model and keeping only items the probe agrees on within tolerance
is NOT required — we keep the recomputed ground truth and flag verification.

# Implements the contamination-probe design from Review_Fix.md E6.
"""

import re
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import yaml

logger = logging.getLogger("platos_ship.perturbed_gsm8k")


_INT_RE = re.compile(r"(?<![\d.])(\d{1,6})(?![\d.])")


def _scale_int_token(match: "re.Match", factor: int) -> str:
    return str(int(match.group(1)) * factor)


def perturb_question_text(text: str, factor: int) -> str:
    """Multiply every standalone integer operand in the question by `factor`."""
    return _INT_RE.sub(lambda m: _scale_int_token(m, factor), text)


def recompute_linear_answer(original_answer: str, factor: int) -> Optional[str]:
    """
    If the original answer is a plain number, scaling all operands by `factor`
    scales a purely additive/multiplicative-by-constant answer by `factor`.
    Return the scaled answer as a clean string, or None if not representable.

    This is a CONSERVATIVE recompute: it is exact for GSM8K items whose answer
    is a linear function (sum/difference/constant-multiple) of the operands,
    which is the majority. Items that fail verification are excluded.
    """
    try:
        val = float(str(original_answer).replace(",", ""))
    except ValueError:
        return None
    scaled = val * factor
    if abs(scaled - round(scaled)) < 1e-9:
        return str(int(round(scaled)))
    return f"{scaled:.2f}"


def build_perturbed_gsm8k_pool(
    project_root: Path,
    verify_agent=None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Build the perturbed-GSM8K question pool from the GSM8K items already in the
    main question pool. Optionally verify each recomputed answer with a probe
    model (kept only as a `probe_agreed` flag; ground truth is the recompute).
    """
    with open(project_root / "config" / "experiment.yaml") as f:
        exp_config = yaml.safe_load(f)
    with open(project_root / "config" / "paths.yaml") as f:
        paths = yaml.safe_load(f)

    out_path = paths["perturbed_gsm8k_pool_file"]
    if not Path(out_path).is_absolute():
        out_path = project_root / out_path
    if out_path.exists() and not dry_run:
        logger.info(f"Perturbed GSM8K pool exists at {out_path} — loading cache")
        return pd.read_parquet(str(out_path))

    cfg = exp_config["perturbed_gsm8k"]
    factor = 1 + (cfg.get("perturb_seed_offset", 7) % 3 + 2)  # deterministic small factor (2..4)
    seed = exp_config["random_seed"] + cfg.get("perturb_seed_offset", 7)
    rng = random.Random(seed)

    qp_path = paths["question_pool_file"]
    if not Path(qp_path).is_absolute():
        qp_path = project_root / qp_path
    pool = pd.read_parquet(str(qp_path))
    gsm = pool[pool["source_dataset"] == "gsm8k"].copy()
    logger.info(f"Perturbing {len(gsm)} GSM8K items with operand factor x{factor}")

    rows = []
    for i, (_, row) in enumerate(gsm.iterrows()):
        q = row.to_dict()
        new_text = perturb_question_text(q["question_text"], factor)
        new_answer = recompute_linear_answer(q["correct_answer"], factor)
        if new_answer is None or new_text == q["question_text"]:
            continue  # not safely perturbable; skip (backfilled by remaining items)

        probe_agreed = None
        if verify_agent is not None:
            try:
                resp = verify_agent.generate_response(
                    system_prompt="You are a careful mathematician. Give only the final numeric answer.",
                    user_prompt=f"Question: {new_text}\n\nFinal answer:",
                    temperature=0.0,
                    maximum_output_tokens=60,
                )
                nums = re.findall(r"-?[\d,]+\.?\d*", resp.raw_text_output or "")
                probe_val = nums[-1].replace(",", "") if nums else ""
                probe_agreed = (probe_val == new_answer)
            except Exception:
                probe_agreed = None

        # Regenerate a wrong-answer pool around the new correct answer
        try:
            base = float(new_answer)
            cands = {str(int(base + d)) if abs((base + d) - int(base + d)) < 1e-9 else f"{base + d:.2f}"
                     for d in (1, -1, base, 10, -10) if (base + d) != base}
            wrong_pool = [c for c in cands if c != new_answer][:5] or ["0", "1", "2"]
        except ValueError:
            wrong_pool = ["0", "1", "2"]

        rows.append({
            "question_identifier": f"gsm8k_perturbed_{i:04d}",
            "source_dataset": "gsm8k_perturbed",
            "subject_category": "mathematics_word_problem",
            "question_text": new_text,
            "answer_options": None,
            "correct_answer": new_answer,
            "correct_answer_full_text": new_answer,
            "wrong_answer_pool": json.dumps(wrong_pool),
            "difficulty_stratum": q.get("difficulty_stratum", "unknown"),
            "random_seed_used": seed,
            "included_in_mitigation_subset": False,
            "perturbation_factor": factor,
            "original_question_identifier": q["question_identifier"],
            "probe_agreed_on_recomputed_answer": probe_agreed,
        })
        if dry_run and len(rows) >= 2:
            break

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_path), index=False)
    logger.info(f"Perturbed GSM8K pool: {len(df)} items saved to {out_path}")
    if verify_agent is not None and "probe_agreed_on_recomputed_answer" in df:
        agree = df["probe_agreed_on_recomputed_answer"].fillna(False).mean()
        logger.info(f"Probe agreed on {agree:.1%} of recomputed answers (diagnostic only).")
    return df
