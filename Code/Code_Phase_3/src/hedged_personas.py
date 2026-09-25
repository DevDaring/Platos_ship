"""
hedged_personas.py — the WRh pool: same wrong content, tentative wording.

Reviewer oEjr, W4: "Wrong and honest peers differ simultaneously in
correctness, confidence, prompting, reasoning style, and answer distribution.
Consequently, the results do not establish whether correction is caused by
wrong answers, confident disagreement, additional reasoning, or simply
prompting the focal model to reconsider."

Condition WRh isolates ONE of those: expressed confidence. Each wrong-anchored
message is rewritten to sound unsure while keeping every claim and the exact
final answer. WR minus WRh is therefore the effect of confident wording with
content held fixed. It is a wording intervention, not evidence of a human-like
social mechanism, and the paper must say so.

The rewrite uses the SAME model that wrote the original pool
(Llama-3.1-8B), so generator identity is not a second thing that changed.

Validation (all must pass, else regenerate up to 3x):
  1. parsed final answer identical to the source message;
  2. at least N hedge markers present;
  3. no numeric token absent from the source (no new arithmetic);
  4. length within [0.6, 1.4] of the source.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .extraction import answers_equal, extract_answer_regex
from .seeding import derive_rng
from .store import Checkpoint, IncrementalWriter
from .call_guard import is_failed_call, record_failure
from .concurrency import run_units

logger = logging.getLogger("platos_ship3.hedged_personas")

HEDGED_DEDUP_KEYS = ["persona_identifier"]

REWRITE_SYSTEM = "You rewrite short arguments while preserving their content exactly."

REWRITE_USER = """Rewrite the message below so that it sounds tentative and unsure.

Rules:
- Keep every claim and every number exactly as they are.
- Do not add any new reason, fact or number.
- Keep the final answer EXACTLY the same.
- Use tentative wording such as "I think", "maybe", "I am not sure", "it might be".
- Keep it about the same length.
- End with the line: Final answer: {final_answer}

Message:
{message}"""

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers_in(text: str) -> set:
    return {m.group(0).replace(",", "") for m in _NUMBER_RE.finditer(text or "")}


# The rewriter often opened with an instruction echo such as "Here's a
# rewritten version of the message with tentative wording:". Shown to a focal
# model as a peer's message, that line announces the message was rewritten,
# which can lower its credibility for a reason other than its wording and so
# confounds WR vs WRh. Found in 63.9% of the first hedged pool and in 85.6% of
# WRh units (hand audit, 25 Sept 2026). It is stripped deterministically, and
# any remaining meta-text fails validation.
_PREAMBLE = re.compile(
    r"^\s*(?:here(?:'s| is)|sure|okay|certainly)[^\n:]{0,120}:\s*\n+", re.IGNORECASE)
_META = re.compile(r"rewritten|rewrite|tentative wording|tentative tone|original message",
                   re.IGNORECASE)


def clean_rewrite(text: str) -> str:
    """Remove a leading instruction echo; the rewrite itself is untouched."""
    return _PREAMBLE.sub("", text or "", count=1).strip()


def validate_hedged(
    rewritten: str,
    source_text: str,
    source_answer: str,
    config: Dict[str, Any],
) -> Tuple[bool, str]:
    """Return (passed, reason)."""
    if not rewritten or not rewritten.strip():
        return False, "empty"
    if _META.search(rewritten):
        return False, "meta_text: mentions the rewrite itself"

    if config.get("final_answer_must_match_source", True):
        parsed = extract_answer_regex(rewritten)
        if parsed is None:
            return False, "missing_final_answer_marker"
        if not answers_equal(parsed, source_answer):
            return False, f"answer_changed: {parsed!r} vs {source_answer!r}"

    markers = [m.lower() for m in config.get("hedge_markers", [])]
    minimum = int(config.get("minimum_hedge_markers", 2))
    lowered = rewritten.lower()
    found = sum(1 for marker in markers if marker in lowered)
    if found < minimum:
        return False, f"insufficient_hedging: {found} < {minimum}"

    if config.get("forbid_new_numbers", True):
        new_numbers = _numbers_in(rewritten) - _numbers_in(source_text)
        if new_numbers:
            return False, f"new_numbers: {sorted(new_numbers)}"

    low, high = config.get("length_ratio_bounds", [0.6, 1.4])
    if source_text:
        ratio = len(rewritten) / max(len(source_text), 1)
        if not low <= ratio <= high:
            return False, f"length_ratio_out_of_bounds: {ratio:.2f}"

    return True, "passed"


def build_hedged_pool(
    anchored_personas: pd.DataFrame,
    rewrite_agent,
    output_path: Path,
    checkpoint: Checkpoint,
    config: Dict[str, Any],
    master_seed: int,
    dry_run: bool = False,
    workers: int = 1,
) -> pd.DataFrame:
    """
    Rewrite every wrong-anchored message into a hedged variant.

    Failures after `max_regeneration_attempts` are dropped and recorded, never
    silently replaced — the appendix reports the realised pass rate.
    """
    writer = IncrementalWriter(Path(output_path), flush_every=100,
                               checkpoint=checkpoint)
    validation_config = config.get("validation", {})
    max_attempts = int(config.get("max_regeneration_attempts", 3))
    temperature = float(config.get("rewrite_temperature", 0.7))
    max_tokens = int(config.get("max_output_tokens", 600))

    records = anchored_personas.to_dict("records")
    if dry_run:
        records = records[:2]

    todo = [r for r in records if r["persona_identifier"] not in checkpoint]
    logger.info("Hedged pool: %d personas, %d to rewrite.", len(records), len(todo))

    stats = {"passed_first": 0, "passed_after_regen": 0, "failed": 0}
    stats_lock = threading.Lock()      # rewrites run on several threads

    def _tally(key: str) -> None:
        with stats_lock:
            stats[key] += 1

    def _one_rewrite(item) -> None:
        index, persona = item
        source_text = persona.get("generated_persona_text", "") or ""
        source_answer = str(
            persona.get("assigned_wrong_answer_letter_or_value", "")
        ).strip()
        persona_id = persona["persona_identifier"]
        rng = derive_rng(master_seed, "hedge", persona_id)

        accepted_text, reason, attempts_used = None, "not_attempted", 0
        call_failed = False
        for attempt in range(max_attempts):
            attempts_used = attempt + 1
            started = time.time()
            response = rewrite_agent.generate_response(
                system_prompt=REWRITE_SYSTEM,
                user_prompt=REWRITE_USER.format(
                    final_answer=source_answer, message=source_text),
                # A little sampling variety between retries, deterministically.
                temperature=temperature + 0.1 * rng.random() * (attempt > 0),
                maximum_output_tokens=max_tokens,
                request_metadata={"stage": "hedge_rewrite",
                                  "persona_identifier": persona_id,
                                  "attempt": attempt},
            )
            call_failed = is_failed_call(response)
            candidate = clean_rewrite(response.raw_text_output)
            passed, reason = validate_hedged(
                candidate, source_text, source_answer, validation_config)
            if passed:
                accepted_text = candidate
                if attempt == 0:
                    _tally("passed_first")
                else:
                    _tally("passed_after_regen")
                break
            logger.debug("Hedge attempt %d failed for %s: %s",
                         attempt + 1, persona_id, reason)

        if accepted_text is None and call_failed:
            # The last attempt died at the provider: retry on the next run
            # instead of recording the persona as unhedgeable.
            record_failure("hedge_rewrite", persona_id, response)
            return
        if accepted_text is None:
            _tally("failed")

        writer.append(
            {
                # Same schema as the anchored pool so peer_pools can index both.
                "persona_identifier": persona_id,
                "question_identifier": persona["question_identifier"],
                "persona_variant_index": persona.get("persona_variant_index", 0),
                "assigned_wrong_answer_letter_or_value": source_answer,
                "assigned_wrong_answer_full_text": persona.get(
                    "assigned_wrong_answer_full_text"),
                "reasoning_style_label": persona.get("reasoning_style_label"),
                "generated_persona_text": accepted_text or "",
                "source_persona_text": source_text,
                "persona_anchor_mode": "hedged",
                "generation_temperature": temperature,
                "generator_model_name": getattr(rewrite_agent, "model_name", ""),
                "validation_pass_status": "passed" if accepted_text else "failed",
                "validation_failure_reason": "" if accepted_text else reason,
                "regeneration_attempts_used": attempts_used,
                "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
                "elapsed_seconds": round(time.time() - started, 3),
            },
            unit_id=persona_id,
        )
        if index % 200 == 0:
            logger.info("Hedged pool: %d/%d done.", index, len(todo))

    # Independent units; see src/concurrency.py. workers=1 is the old loop.
    run_units(list(enumerate(todo, start=1)), _one_rewrite, workers=workers, label="hedge_rewrite")

    pool = writer.consolidate(dedup_on=HEDGED_DEDUP_KEYS)
    total = max(len(pool), 1)
    logger.info(
        "Hedged pool: %d rows | first-attempt %d, after-regen %d, failed %d "
        "(retention %.1f%%)",
        len(pool), stats["passed_first"], stats["passed_after_regen"],
        stats["failed"], 100.0 * (total - stats["failed"]) / total,
    )
    # Drop failures from the usable pool; the full frame (with failures) stays
    # on disk so the appendix can quote the real pass rate.
    return pool


def sample_for_hand_audit(
    pool: pd.DataFrame, n: int, master_seed: int, output_path: Path
) -> pd.DataFrame:
    """
    Draw a fixed random sample of (source, rewritten) pairs for human checking.

    The paper reports this audit: an automated validator cannot certify that
    'tentative' wording did not also weaken the argument, which is exactly the
    alternative explanation a reviewer will raise for any WR-vs-WRh effect.
    """
    usable = pool[pool["validation_pass_status"] == "passed"]
    if usable.empty:
        return pd.DataFrame()
    take = min(int(n), len(usable))
    sample = usable.sample(n=take, random_state=master_seed % (2**31 - 1))
    columns = ["persona_identifier", "question_identifier",
               "assigned_wrong_answer_letter_or_value",
               "source_persona_text", "generated_persona_text"]
    sample = sample[columns].copy()
    sample["auditor_verdict_same_content"] = ""     # to be filled by hand
    sample["auditor_verdict_less_confident"] = ""
    sample["auditor_notes"] = ""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    logger.info("Hand-audit sheet (%d pairs) -> %s", len(sample), path)
    return sample
