"""
verifier.py — X6: verify a proposed answer change before adopting it.

The reviewed paper's only mitigation was a confidence-weighted peer filter,
which failed (oEjr W3 calls the corrected version post-hoc). X6 offers a
mitigation that can actually work, and tests it honestly.

Rule under test
---------------
Keep the focal model's INITIAL answer unless an independent verifier supports
the proposed change.

Design rules that keep it honest:
  * the verifier is a DIFFERENT model from every focal, so the safeguard is
    not a model grading itself;
  * the verifier solves the question first, then sees the two candidate
    answers in RANDOMISED order with no indication of which is the initial
    answer and which the peer-supported one;
  * gold answers NEVER reach the decision rule. They only score the outcome
    afterwards. A gold-aware rule would be an oracle upper bound and is
    reported separately, clearly labelled;
  * the verifier's own token cost is reported, and the comparison baselines
    are budget-matched, because "spend another call" is itself an alternative
    use of the budget.

Baselines it is compared against (analysis/safeguard.py):
  always_keep | always_revise | random_retention_at_matched_coverage |
  confidence_retention | oracle (upper bound, labelled)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .extraction import answers_equal, extract_answer, options_to_string
from .seeding import derive_rng
from .store import Checkpoint, IncrementalWriter
from .call_guard import is_failed_call, record_failure
from .concurrency import run_units

logger = logging.getLogger("platos_ship3.verifier")

VERIFIER_DEDUP_KEYS = ["verify_unit_id"]

VERIFIER_SYSTEM = (
    "You are a careful, independent problem solver. You check work; you do "
    "not defer to anyone."
)

VERIFIER_USER = """Question: {question}{options}

Solve this problem yourself, step by step.

Then consider these two candidate answers:
Option 1: {candidate_1}
Option 2: {candidate_2}

State which candidate your own solution supports, or state NEITHER if your
solution supports neither. End with these two lines, and nothing after them:
Final answer: <your own answer>
Supported option: <1, 2, or NEITHER>"""


def _parse_supported_option(raw_text: str) -> Optional[str]:
    """Read the 'Supported option:' line; None when absent or malformed."""
    import re

    match = re.search(r"Supported\s+option\s*:\s*(1|2|NEITHER)",
                      raw_text or "", re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def verify_changes(
    revisions: pd.DataFrame,
    questions: pd.DataFrame,
    verifier_agent,
    verifier_spec: Dict[str, Any],
    judge_cascade,
    auditor,
    output_path: Path,
    checkpoint: Checkpoint,
    master_seed: int,
    dry_run: bool = False,
    workers: int = 1,
) -> pd.DataFrame:
    """
    Run the verifier on every revision that PROPOSED A CHANGE.

    Unchanged answers need no verification — the rule only fires on a proposed
    change, which is also why its coverage is low and its token cost modest.
    """
    writer = IncrementalWriter(Path(output_path), flush_every=50,
                               checkpoint=checkpoint)
    question_index = {
        row["question_identifier"]: row for row in questions.to_dict("records")
    }

    changed = revisions[
        revisions["answer_changed"].fillna(False)
        & revisions["extracted_answer"].notna()
        & revisions["r0_answer"].notna()
    ].copy()
    if dry_run:
        changed = changed.head(2)

    todo = [r for r in changed.to_dict("records")
            if f"VERIFY|{r['unit_id']}" not in checkpoint]
    logger.info("Verifier: %d proposed changes, %d to verify.",
                len(changed), len(todo))

    def _one_verification(item) -> None:
        index, revision = item
        unit = f"VERIFY|{revision['unit_id']}"
        question = question_index.get(revision["question_identifier"])
        if question is None:
            return

        initial_answer = str(revision["r0_answer"])
        revised_answer = str(revision["extracted_answer"])

        # Blind the order: the verifier must not learn which is which.
        rng = derive_rng(master_seed, "verify", revision["unit_id"])
        initial_is_first = rng.random() < 0.5
        candidate_1 = initial_answer if initial_is_first else revised_answer
        candidate_2 = revised_answer if initial_is_first else initial_answer

        options_str = options_to_string(question.get("answer_options"))
        options_block = f"\n\n{options_str}" if options_str else ""

        started = time.time()
        response = verifier_agent.generate_response(
            system_prompt=VERIFIER_SYSTEM,
            user_prompt=VERIFIER_USER.format(
                question=question["question_text"],
                options=options_block,
                candidate_1=candidate_1,
                candidate_2=candidate_2,
            ),
            temperature=float(verifier_spec.get("temperature", 0.0)),
            maximum_output_tokens=int(verifier_spec.get("max_output_tokens", 2048)),
            request_metadata={"stage": "verify", "unit_id": revision["unit_id"]},
        )
        if is_failed_call(response):
            record_failure("verifier", str(revision["unit_id"]), response)
            return

        auditor.check(
            agent_key="verifier",
            expected_prefix=verifier_spec.get("expected_served_prefix"),
            served_model=response.model_name_returned_by_provider,
            role="verifier",
            strict=True,
            context={"unit_id": revision["unit_id"]},
        )

        verifier_answer, method = extract_answer(
            response.raw_text_output, question["question_text"], options_str,
            judge_cascade, answer_options=question.get("answer_options"),
        )
        supported_option = _parse_supported_option(response.raw_text_output)

        # Map the blinded option back to initial / revised.
        supported_side = None
        if supported_option == "1":
            supported_side = "initial" if initial_is_first else "revised"
        elif supported_option == "2":
            supported_side = "revised" if initial_is_first else "initial"
        elif supported_option == "NEITHER":
            supported_side = "neither"

        # The decision rule: adopt the change only when the verifier's OWN
        # solution matches the revised answer, or it explicitly backs it.
        own_supports_revised = answers_equal(verifier_answer, revised_answer)
        own_supports_initial = answers_equal(verifier_answer, initial_answer)
        adopt_change = bool(own_supports_revised) or supported_side == "revised"
        if own_supports_initial and not own_supports_revised:
            adopt_change = False

        final_answer = revised_answer if adopt_change else initial_answer

        writer.append(
            {
                "verify_unit_id": unit,
                "unit_id": revision["unit_id"],
                "protocol": revision.get("protocol"),
                "experiment": revision.get("experiment"),
                "condition": revision.get("condition"),
                "focal_key": revision.get("focal_key"),
                "question_identifier": revision["question_identifier"],
                "source_dataset": revision.get("source_dataset"),
                "replicate": revision.get("replicate"),
                "initial_answer": initial_answer,
                "revised_answer": revised_answer,
                "initial_was_first_shown": initial_is_first,
                "verifier_served_model": response.model_name_returned_by_provider,
                "verifier_raw_text": response.raw_text_output,
                "verifier_own_answer": verifier_answer,
                "verifier_answer_method": method,
                "verifier_supported_option": supported_option,
                "verifier_supported_side": supported_side,
                "adopt_change": adopt_change,
                "safeguarded_final_answer": final_answer,
                # Scoring columns — used AFTER the decision, never inside it.
                "correct_answer": revision.get("correct_answer"),
                "r0_is_correct": revision.get("r0_is_correct"),
                "unsafeguarded_is_correct": revision.get("is_correct"),
                "total_input_tokens": response.total_input_tokens,
                "total_output_tokens": response.total_output_tokens,
                "wall_clock_latency_seconds": response.wall_clock_latency_seconds,
                "error_status": response.error_status,
                "finish_reason": response.finish_reason,
                "served_route": response.served_route,
                "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
                "elapsed_seconds": round(time.time() - started, 3),
            },
            unit_id=unit,
        )
        if index % 100 == 0:
            logger.info("Verifier: %d/%d done.", index, len(todo))

    # Independent units; see src/concurrency.py. workers=1 is the old loop.
    run_units(list(enumerate(todo, start=1)), _one_verification, workers=workers, label="verifier")

    return writer.consolidate(dedup_on=VERIFIER_DEDUP_KEYS)
