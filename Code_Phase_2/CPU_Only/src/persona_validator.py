"""
persona_validator.py — Validates generated dumb personas per §5.3.

Discards personas that fail validation and regenerates up to 3 times.
Writes validation report to logs/persona_validation_report.txt.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import yaml

logger = logging.getLogger("platos_ship.persona_validator")

# Hedging phrases to detect (persona should sound confident)
HEDGING_PATTERNS = [
    r"\bunsure\b", r"\bnot certain\b", r"\bmight be wrong\b",
    r"\bactually\b", r"\bcorrection\b",
]
HEDGING_THRESHOLD = 1  # Max number of hedging phrases allowed


def validate_single_persona(persona: Dict) -> Tuple[bool, str]:
    """
    Validate a single persona per §5.3 rules.
    
    Returns (passed: bool, reason: str).
    """
    text = persona.get("generated_persona_text", "")
    assigned_answer = persona.get("assigned_wrong_answer_letter_or_value", "")

    # Rule 1: Must contain "Final answer:" (case-insensitive)
    if not re.search(r"final\s+answer\s*:", text, re.IGNORECASE):
        return False, "missing_final_answer_marker"

    # Rule 2: Extracted final answer must equal assigned wrong answer
    match = re.search(r"final\s+answer\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip()
        # Normalize for comparison
        if extracted.upper() != assigned_answer.upper():
            return False, f"answer_mismatch: extracted='{extracted}' vs assigned='{assigned_answer}'"
    else:
        return False, "could_not_extract_answer"

    # Rule 3: Hedging phrases below threshold
    hedge_count = 0
    for pattern in HEDGING_PATTERNS:
        hedge_count += len(re.findall(pattern, text, re.IGNORECASE))
    if hedge_count > HEDGING_THRESHOLD:
        return False, f"excessive_hedging: {hedge_count} hedging phrases found"

    # Rule 4: Length bounds
    if len(text) < 30:
        return False, f"too_short: {len(text)} chars"
    if len(text) > 1500:
        return False, f"too_long: {len(text)} chars"

    return True, "passed"


def validate_and_regenerate(
    personas_df: pd.DataFrame,
    llama_agent,
    questions_df: pd.DataFrame,
    max_regeneration_attempts: int = 3,
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Validate all personas and regenerate failed ones up to max_regeneration_attempts.
    """
    from .persona_generator import generate_personas_for_question

    total = len(personas_df)
    passed_count = 0
    failed_count = 0
    regenerated_count = 0
    discard_reasons = {}

    validated_rows = []

    # Group by question
    for qid, group in personas_df.groupby("question_identifier"):
        question_row = questions_df[questions_df["question_identifier"] == qid]
        if question_row.empty:
            logger.warning(f"Question {qid} not found in pool, skipping")
            continue
        question = question_row.iloc[0].to_dict()

        for _, persona in group.iterrows():
            persona_dict = persona.to_dict()
            passed, reason = validate_single_persona(persona_dict)

            if passed:
                persona_dict["validation_pass_status"] = "passed"
                validated_rows.append(persona_dict)
                passed_count += 1
                continue

            # Try regeneration
            success = False
            for attempt in range(1, max_regeneration_attempts + 1):
                logger.debug(f"Regenerating {persona_dict['persona_identifier']}, attempt {attempt}: {reason}")

                new_personas = generate_personas_for_question(
                    question, llama_agent, num_variants=1,
                    seed=hash(f"{qid}_{persona_dict['persona_variant_index']}_{attempt}") % (2**31),
                )

                if new_personas:
                    new_persona = new_personas[0]
                    new_persona["persona_identifier"] = persona_dict["persona_identifier"]
                    new_persona["persona_variant_index"] = persona_dict["persona_variant_index"]

                    new_passed, new_reason = validate_single_persona(new_persona)
                    if new_passed:
                        new_persona["validation_pass_status"] = "passed"
                        new_persona["regeneration_attempts_used"] = attempt
                        validated_rows.append(new_persona)
                        regenerated_count += 1
                        success = True
                        break

            if not success:
                # Keep the original but mark as failed
                persona_dict["validation_pass_status"] = f"failed: {reason}"
                validated_rows.append(persona_dict)
                failed_count += 1
                discard_reasons[reason] = discard_reasons.get(reason, 0) + 1

    # Build result DataFrame
    result_df = pd.DataFrame(validated_rows)

    # Retention rate
    total_checked = passed_count + regenerated_count + failed_count
    retention_rate = (passed_count + regenerated_count) / max(total_checked, 1) * 100

    logger.info(f"Persona validation complete:")
    logger.info(f"  Total checked: {total_checked}")
    logger.info(f"  Passed first try: {passed_count}")
    logger.info(f"  Regenerated successfully: {regenerated_count}")
    logger.info(f"  Failed (kept with flag): {failed_count}")
    logger.info(f"  Retention rate: {retention_rate:.1f}%")

    # Write validation report
    if project_root:
        report_path = project_root / "logs" / "persona_validation_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write("Persona Validation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total personas checked: {total_checked}\n")
            f.write(f"Passed (first attempt): {passed_count}\n")
            f.write(f"Regenerated successfully: {regenerated_count}\n")
            f.write(f"Failed after max attempts: {failed_count}\n")
            f.write(f"Retention rate: {retention_rate:.1f}%\n\n")
            f.write("Discard reasons:\n")
            for reason, count in sorted(discard_reasons.items(), key=lambda x: -x[1]):
                f.write(f"  {reason}: {count}\n")
        logger.info(f"Validation report written to {report_path}")

    # Save validated personas
    if project_root:
        with open(project_root / "config" / "paths.yaml") as fp:
            paths = yaml.safe_load(fp)
        output_path = paths["dumb_personas_file"]
        if not Path(output_path).is_absolute():
            output_path = project_root / output_path
        result_df.to_parquet(str(output_path), index=False)
        logger.info(f"Validated personas saved to {output_path}")

    return result_df
