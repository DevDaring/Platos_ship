"""
r0_cache.py — Stage 0 of Protocol B: the cached initial answer.

One Round-0 call per (focal model, question, replicate). Every revision
condition then clones this exact record, so a treatment contrast compares
revisions *of the same initial state*. In Phase 1/2 each condition generated
its own fresh Round-0 answer, so matching on (question, replicate) did not
match the initial answer, and part of every measured "flip" was ordinary
temperature-0.7 resampling.

Cost note: the cache is also what makes the study affordable. Eight focal
models x 300 questions x 3 replicates = 7,200 calls, reused by every one of
the revision conditions instead of being re-paid per condition.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .contexts import build_round0_prompt
from .extraction import extract_answer, extract_confidence, grade, options_to_string
from .seeding import r0_unit_id
from .store import Checkpoint, IncrementalWriter
from .call_guard import drop_failed_calls, is_failed_call, record_failure
from .concurrency import run_units
from .scopes import main_scope_only

logger = logging.getLogger("platos_ship3.r0_cache")

R0_DEDUP_KEYS = ["r0_unit_id"]


def build_r0_cache(
    questions: pd.DataFrame,
    focal_keys: List[str],
    focal_agents: Dict[str, Any],
    focal_specs: Dict[str, Any],
    judge_cascade,
    auditor,
    cache_path: Path,
    checkpoint: Checkpoint,
    replicates: int,
    temperature: float = 0.7,
    max_output_tokens: int = 600,
    flush_every: int = 100,
    dry_run: bool = False,
    workers: int = 1,
) -> pd.DataFrame:
    """
    Populate (or extend) the Round-0 cache.

    Resume-safe: any (focal, question, replicate) already in the checkpoint is
    skipped, so a crashed run costs nothing to restart.
    """
    writer = IncrementalWriter(Path(cache_path), flush_every=flush_every,
                               checkpoint=checkpoint)

    planned = [
        (focal_key, question, replicate)
        for focal_key in focal_keys
        for _, question in questions.iterrows()
        for replicate in range(replicates)
    ]
    if dry_run:
        planned = planned[: len(focal_keys)]

    todo = [p for p in planned if r0_unit_id(p[0], p[1]["question_identifier"], p[2])
            not in checkpoint]
    logger.info(
        "R0 cache: %d planned, %d already cached, %d to run.",
        len(planned), len(planned) - len(todo), len(todo),
    )

    def _one_r0(item) -> None:
        index, (focal_key, question, replicate) = item
        question_id = question["question_identifier"]
        unit = r0_unit_id(focal_key, question_id, replicate)
        agent = focal_agents[focal_key]
        spec = focal_specs.get(focal_key, {})

        system_prompt, user_prompt = build_round0_prompt(
            question_text=question["question_text"],
            answer_options=question.get("answer_options"),
        )

        started = time.time()
        response = agent.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            maximum_output_tokens=max_output_tokens,
            request_metadata={"stage": "r0", "question_id": question_id,
                              "replicate": replicate, "focal": focal_key},
        )
        if is_failed_call(response):
            # Not cached, not checkpointed: a failed Round-0 would be shown as
            # an empty previous answer in EVERY condition. Retried next run.
            record_failure("r0", unit, response)
            return

        # A focal response must come from the pinned snapshot, or the run stops.
        auditor.check(
            agent_key=focal_key,
            expected_prefix=spec.get("expected_served_prefix"),
            served_model=response.model_name_returned_by_provider,
            role="focal_r0",
            strict=True,
            context={"question_id": question_id, "replicate": replicate},
        )

        options_str = options_to_string(question.get("answer_options"))
        answer, method = extract_answer(
            response.raw_text_output, question["question_text"], options_str,
            judge_cascade, answer_options=question.get("answer_options"),
        )
        confidence, confidence_status = extract_confidence(response.raw_text_output)
        is_correct = grade(answer, question["correct_answer"])

        writer.append(
            {
                "r0_unit_id": unit,
                "focal_key": focal_key,
                "focal_served_model": response.model_name_returned_by_provider,
                "question_identifier": question_id,
                "source_dataset": question.get("source_dataset"),
                "subject_category": question.get("subject_category"),
                "replicate": replicate,
                "raw_response_text": response.raw_text_output,
                "extracted_answer": answer,
                "answer_extraction_method": method,
                "extracted_confidence": confidence,
                "confidence_parse_status": confidence_status,
                "correct_answer": question["correct_answer"],
                "is_correct": is_correct,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "total_input_tokens": response.total_input_tokens,
                "total_output_tokens": response.total_output_tokens,
                "wall_clock_latency_seconds": response.wall_clock_latency_seconds,
                "error_status": response.error_status,
                "finish_reason": response.finish_reason,
                "retry_attempts_used": response.retry_attempts_used,
                "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
                "elapsed_seconds": round(time.time() - started, 3),
            },
            unit_id=unit,
        )

        if index % 200 == 0:
            logger.info("R0 cache: %d/%d done.", index, len(todo))

    # Independent units; see src/concurrency.py. workers=1 is the old loop.
    run_units(list(enumerate(todo, start=1)), _one_r0, workers=workers, label="r0")

    return writer.consolidate(dedup_on=R0_DEDUP_KEYS)


def load_r0_cache(cache_path: Path) -> pd.DataFrame:
    """Read the cache, or return an empty frame with the expected columns."""
    path = Path(cache_path)
    if not path.exists():
        return pd.DataFrame(
            columns=["r0_unit_id", "focal_key", "question_identifier", "replicate",
                     "raw_response_text", "extracted_answer", "is_correct"]
        )
    return drop_failed_calls(pd.read_parquet(path), "Round-0 cache")


def r0_lookup(cache: pd.DataFrame) -> Dict[tuple, Dict[str, Any]]:
    """Index the cache by (focal_key, question_identifier, replicate)."""
    return {
        (row["focal_key"], row["question_identifier"], int(row["replicate"])): row
        for row in cache.to_dict("records")
    }


def solo_accuracy_by_focal(cache: pd.DataFrame) -> Dict[str, float]:
    """
    Per-model solo (Round-0) accuracy — the x-axis of the capability gradient.

    Measured on the main 300 items only, the items every model answers. The
    cache also holds GSM-Symbolic Round-0 for the X4 models; averaging over it
    gave those models a different denominator from the rest.
    """
    cache = main_scope_only(cache)
    if cache.empty:
        return {}
    return cache.groupby("focal_key")["is_correct"].mean().to_dict()
