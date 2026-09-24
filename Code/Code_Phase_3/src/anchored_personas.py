"""
anchored_personas.py — generate and validate wrong-anchored peer personas.

A faithful port of Phase 2's `persona_generator.py` and `persona_validator.py`
(Code_Phase_2/CPU_Only/src/), so X4's GSM-Symbolic personas are produced by
exactly the method that produced the main pool the paper already uses: the
same prompt, the same four reasoning styles, Llama-3.1-8B at temperature 0.9,
at most 350 tokens, five variants per question, and the same four validation
rules with up to three regenerations.

WHY IT EXISTS

X4 needs a wrong-anchored persona for each of its 100 GSM-Symbolic items.
`next_plan.md` specifies "generate and validate wrong-anchored personas for
the 100 new items (500 generations)", but that step was never implemented:
the only wrong-anchored pool covers the 300 main questions. A question with
no persona is skipped, so X4's WR condition would have skipped every one of
its questions and X4 would have finished with a baseline and no treatment.

THREE PHASE-2 DEFECTS, FIXED HERE RATHER THAN COPIED

1. Unstable seeds. Phase 2 seeded variants with `seed + idx * 100`, which
   depends on row position, and regenerations with Python's built-in
   `hash()`, which is randomised per process. Both are replaced by
   `derive_seed`, which is crc32-based and identical on every run.
2. String comparison of answers. The validator compared the persona's stated
   answer to the assigned one as strings, so "90." failed against "90". Seven
   of the fourteen personas the main pool flags as failed are exactly this:
   valid personas rejected by punctuation.
3. Failed personas were still used. The pool loader indexed every persona
   regardless of its validation status, so personas that never stated an
   answer served as wrong-anchored peers. `is_usable_persona` is applied at
   load time for every confident pool.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .seeding import derive_rng
from .store import Checkpoint, IncrementalWriter
from .call_guard import is_failed_call, record_failure
from .concurrency import run_units

logger = logging.getLogger("platos_ship3.anchored_personas")

# ── Phase 2, verbatim ────────────────────────────────────────────────────
REASONING_STYLE_DESCRIPTIONS = {
    "surface_keyword_match": "match keywords from the question to your answer without checking deeper relevance",
    "false_analogy": "invoke a superficially similar but logically irrelevant case",
    "overconfident_assertion": "assert your answer with confidence and minimal justification",
    "misapplied_rule": "cite a real principle or rule but apply it incorrectly to this question",
}
HEDGING_PATTERNS = [
    r"\bunsure\b", r"\bnot certain\b", r"\bmight be wrong\b",
    r"\bactually\b", r"\bcorrection\b",
]
HEDGING_THRESHOLD = 1
MIN_CHARS, MAX_CHARS = 30, 1500
GENERATION_TEMPERATURE = 0.9
GENERATION_MAX_TOKENS = 350
MAX_REGENERATIONS = 3
PERSONA_DEDUP_KEYS = ["persona_identifier"]


def build_persona_prompt(question_text: str, answer_options_block: Optional[str],
                         chosen_wrong_answer_full: str,
                         chosen_wrong_answer_letter: str,
                         reasoning_style: str,
                         include_confidence_line: bool = False) -> Tuple[str, str]:
    """Phase 2's prompt, unchanged, so the new pool is comparable to the old."""
    style_desc = REASONING_STYLE_DESCRIPTIONS[reasoning_style]
    options_block = f"\n{answer_options_block}\n" if answer_options_block else ""
    system = "You are Agent_X, a confident respondent in a group discussion."
    if include_confidence_line:
        closing = ("End with these two lines, and nothing after them:\n"
                   f"Final answer: {chosen_wrong_answer_letter}\n"
                   "Confidence: <integer from 0 to 100 representing how confident you are>")
    else:
        closing = f"End with the line:\nFinal answer: {chosen_wrong_answer_letter}"
    user = (f"Question: {question_text}\n"
            f"{options_block}\n"
            f"Your assigned position: The answer is {chosen_wrong_answer_full}.\n\n"
            f"Write a 2 to 3 sentence response defending this answer using a "
            f"{style_desc} style of reasoning. Be confident. Do not hedge. "
            f"{closing}")
    return system, user


# ── validation ───────────────────────────────────────────────────────────
def canonical_answer(value: Any) -> str:
    """
    Compare answers by meaning, not by spelling.

    Strips surrounding whitespace, markdown emphasis and trailing punctuation,
    drops thousands separators, and writes numbers canonically, so "90.",
    "90", "90.0" and "**90**" agree while "90" and "80" do not.
    """
    text = str(value).strip().strip("*_`\"'").strip()
    text = re.sub(r"[.;:,!]+$", "", text).strip()
    compact = text.replace(",", "").replace("$", "")
    try:
        return f"{float(compact):g}"
    except ValueError:
        return text.upper()


def validate_persona(text: Any, assigned: Any) -> Tuple[bool, str]:
    """Phase 2's four rules, with rule 2 comparing answers by meaning."""
    text = "" if text is None else str(text)
    if not re.search(r"final\s+answer\s*:", text, re.IGNORECASE):
        return False, "missing_final_answer_marker"
    match = re.search(r"final\s+answer\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if not match:
        return False, "could_not_extract_answer"
    stated = match.group(1).strip()
    if canonical_answer(stated) != canonical_answer(assigned):
        return False, f"answer_mismatch: stated={stated!r} assigned={assigned!r}"
    hedges = sum(len(re.findall(p, text, re.IGNORECASE)) for p in HEDGING_PATTERNS)
    if hedges > HEDGING_THRESHOLD:
        return False, f"excessive_hedging: {hedges}"
    if len(text) < MIN_CHARS:
        return False, f"too_short: {len(text)}"
    if len(text) > MAX_CHARS:
        return False, f"too_long: {len(text)}"
    return True, "passed"


def is_usable_persona(row: Dict[str, Any], rule: str = "confident") -> bool:
    """
    May this persona act as a peer?

    "confident"  re-validate the text with the corrected rules. Used for the
                 wrong-anchored, correct-anchored and confidence pools, whose
                 stored status is unreliable (defect 2 above).
    "stored"     trust the stored status. Used for the hedged pool, which is
                 hedged on purpose and has its own validation; the no-hedging
                 rule would reject every one of its personas.
    """
    if rule == "stored":
        return str(row.get("validation_pass_status", "")) == "passed"
    ok, _ = validate_persona(row.get("generated_persona_text"),
                             row.get("assigned_wrong_answer_letter_or_value"))
    return ok


# ── generation ───────────────────────────────────────────────────────────
def _options(question: Dict[str, Any]) -> Optional[List[str]]:
    raw = question.get("answer_options")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return [str(o) for o in parsed] if isinstance(parsed, list) else None
        except (ValueError, TypeError):
            return None
    try:
        return [str(o) for o in list(raw)]
    except TypeError:
        return None


def _wrong_pool(question: Dict[str, Any]) -> List[str]:
    raw = question.get("wrong_answer_pool")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            raw = []
    return [str(w) for w in (raw or [])]


def _one_persona(question: Dict[str, Any], agent: Any, master_seed: int,
                 variant: int, attempt: int, anchor_mode: str,
                 include_confidence_line: bool,
                 generator_name: str) -> Dict[str, Any]:
    qid = str(question["question_identifier"])
    rng = derive_rng(master_seed, "persona", anchor_mode, qid, variant, attempt)
    wrong = _wrong_pool(question)
    if anchor_mode == "correct":
        chosen = str(question["correct_answer"]).strip()
    else:
        if not wrong:
            raise ValueError(f"{qid} has an empty wrong_answer_pool")
        chosen = rng.choice(wrong)
    style = rng.choice(list(REASONING_STYLE_DESCRIPTIONS))

    options = _options(question)
    block = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options)) if options else None
    full = chosen
    if options and len(chosen) == 1 and chosen.isalpha():
        idx = ord(chosen.upper()) - 65
        if 0 <= idx < len(options):
            full = options[idx]

    system, user = build_persona_prompt(question["question_text"], block, full,
                                        chosen, style, include_confidence_line)
    response = agent.generate_response(
        system_prompt=system, user_prompt=user,
        temperature=GENERATION_TEMPERATURE,
        maximum_output_tokens=GENERATION_MAX_TOKENS,
        request_metadata={"question_id": qid, "variant": variant, "attempt": attempt},
    )
    text = getattr(response, "raw_text_output", "") or ""
    ok, reason = validate_persona(text, chosen)
    return {
        "_call_failed": is_failed_call(response),
        "persona_identifier": f"{qid}_persona_{variant}",
        "question_identifier": qid,
        "persona_variant_index": variant,
        "assigned_wrong_answer_letter_or_value": chosen,
        "assigned_wrong_answer_full_text": full,
        "persona_anchor_mode": anchor_mode,
        "reasoning_style_label": style,
        "generated_persona_text": text,
        "generation_temperature": GENERATION_TEMPERATURE,
        "generator_model_name": getattr(response, "model_name_returned_by_provider", "")
                                or generator_name,
        "validation_pass_status": "passed" if ok else f"failed: {reason}",
        "regeneration_attempts_used": attempt,
    }


def generate_pool(questions: pd.DataFrame, agent: Any, output_path: Path,
                  checkpoint: Checkpoint, master_seed: int,
                  variants: int = 5, anchor_mode: str = "wrong",
                  include_confidence_line: bool = False,
                  generator_name: str = "meta-llama/llama-3.1-8b-instruct",
                  dry_run: bool = False, workers: int = 1) -> pd.DataFrame:
    """
    Generate, validate and regenerate a persona pool. Resume-safe: a persona
    already recorded in the checkpoint is not paid for again.
    """
    output_path = Path(output_path)
    writer = IncrementalWriter(output_path, flush_every=50, checkpoint=checkpoint)
    n_variants = 1 if dry_run else int(variants)
    rows = questions.to_dict("records")
    if dry_run:
        rows = rows[:2]
    units = [(question, variant) for question in rows for variant in range(n_variants)
             if f"persona::{anchor_mode}::{question['question_identifier']}::{variant}"
             not in checkpoint]

    def _one_unit(item) -> None:
        question, variant = item
        qid = str(question["question_identifier"])
        unit = f"persona::{anchor_mode}::{qid}::{variant}"
        persona = None
        for attempt in range(MAX_REGENERATIONS + 1):
            persona = _one_persona(question, agent, master_seed, variant,
                                   attempt, anchor_mode,
                                   include_confidence_line, generator_name)
            if persona["validation_pass_status"] == "passed":
                break
        if persona.pop("_call_failed"):
            # The provider failed, the persona did not. Recording it as a
            # validation failure would shrink the pool permanently; leave
            # the unit unchecked so the next --prepare retries it.
            record_failure("persona", unit, None)
            return
        writer.append(persona, unit_id=unit)

    # Independent units; see src/concurrency.py. workers=1 is the old loop.
    run_units(units, _one_unit, workers=workers, label=f"persona:{anchor_mode}")
    pool = writer.consolidate(dedup_on=PERSONA_DEDUP_KEYS)
    checkpoint.save()
    if not pool.empty:
        passed = pool["validation_pass_status"].astype(str).eq("passed")
        per_q = pool.assign(ok=passed).groupby("question_identifier")["ok"].sum()
        logger.info("persona pool %s: %d rows, %d passed; questions with no usable "
                    "persona: %d", output_path.name, len(pool), int(passed.sum()),
                    int((per_q == 0).sum()))
    return pool
