"""
honest_bank.py — natural weak-peer messages, generated once and shared.

Condition H shows the focal model what the weak peers actually say when asked
the question normally (no persona, no assigned answer). Generating the bank
once per (question, replicate) and reusing it across all eight focal models
means:

  * every focal model sees the SAME natural messages, so a cross-model
    difference is the focal model's behaviour, not peer sampling noise;
  * the bank costs 300 x 3 x 2 = 1,800 calls instead of 8 x that.

Two estimands are kept separate (next_plan.md §5 / v1 plan §B2):
  1. natural peer-policy effect — every generated message is included,
     whatever it says, so the realised correct/wrong mix is the weak models'
     own error rate on this pool;
  2. controlled exposure — the analysis may later condition on peer
     correctness, but that is an observational stratification and is labelled
     as such. The bank records `is_correct` per message so both are possible.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .contexts import build_round0_prompt
from .extraction import extract_answer, extract_confidence, grade, options_to_string
from .store import Checkpoint, IncrementalWriter
from .call_guard import is_failed_call, record_failure
from .concurrency import run_units

logger = logging.getLogger("platos_ship3.honest_bank")

HONEST_DEDUP_KEYS = ["honest_unit_id"]


def honest_unit_id(question_id: str, replicate: int, weak_key: str) -> str:
    return f"HONEST|{weak_key}|{question_id}|r{replicate}"


def build_honest_bank(
    questions: pd.DataFrame,
    weak_agents: Dict[str, Any],
    weak_specs: Dict[str, Any],
    judge_cascade,
    bank_path: Path,
    checkpoint: Checkpoint,
    replicates: int,
    temperature: float = 0.9,
    max_output_tokens: int = 350,
    dry_run: bool = False,
    workers: int = 1,
) -> pd.DataFrame:
    """Generate (or extend) the honest weak-peer message bank."""
    writer = IncrementalWriter(Path(bank_path), flush_every=100, checkpoint=checkpoint)

    planned = [
        (weak_key, question, replicate)
        for weak_key in weak_agents
        for _, question in questions.iterrows()
        for replicate in range(replicates)
    ]
    if dry_run:
        # One (question, replicate) served by EVERY weak model, so condition H
        # has its designed peers in a dry run. Taking the first len(weak)
        # entries of a list ordered weak-model-first gave one model twice and
        # the other none, so H could never be exercised before a real run.
        first_q = planned[0][1]["question_identifier"] if planned else None
        planned = [p for p in planned
                   if p[1]["question_identifier"] == first_q and p[2] == 0]

    todo = [
        p for p in planned
        if honest_unit_id(p[1]["question_identifier"], p[2], p[0]) not in checkpoint
    ]
    logger.info("Honest bank: %d planned, %d to generate.", len(planned), len(todo))

    def _one_message(item) -> None:
        index, (weak_key, question, replicate) = item
        question_id = question["question_identifier"]
        unit = honest_unit_id(question_id, replicate, weak_key)
        agent = weak_agents[weak_key]

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
            request_metadata={"stage": "honest_bank", "question_id": question_id,
                              "replicate": replicate, "weak": weak_key},
        )
        if is_failed_call(response):
            # An empty honest message would still count toward H's peers.
            record_failure("honest_bank", unit, response)
            return

        options_str = options_to_string(question.get("answer_options"))
        answer, method = extract_answer(
            response.raw_text_output, question["question_text"], options_str,
            judge_cascade, answer_options=question.get("answer_options"),
        )
        confidence, confidence_status = extract_confidence(response.raw_text_output)

        writer.append(
            {
                "honest_unit_id": unit,
                "question_identifier": question_id,
                "source_dataset": question.get("source_dataset"),
                "replicate": replicate,
                "weak_model_key": weak_key,
                "weak_model_paper_name": weak_specs.get(weak_key, {}).get(
                    "paper_name", weak_key),
                "served_model": response.model_name_returned_by_provider,
                "message_text": response.raw_text_output,
                "extracted_answer": answer,
                "answer_extraction_method": method,
                "extracted_confidence": confidence,
                "confidence_parse_status": confidence_status,
                "correct_answer": question["correct_answer"],
                "is_correct": grade(answer, question["correct_answer"]),
                "temperature": temperature,
                "total_input_tokens": response.total_input_tokens,
                "total_output_tokens": response.total_output_tokens,
                "error_status": response.error_status,
                "finish_reason": response.finish_reason,
                "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
                "elapsed_seconds": round(time.time() - started, 3),
            },
            unit_id=unit,
        )
        if index % 200 == 0:
            logger.info("Honest bank: %d/%d done.", index, len(todo))

    # Independent units; see src/concurrency.py. workers=1 is the old loop.
    run_units(list(enumerate(todo, start=1)), _one_message, workers=workers, label="honest_bank")

    bank = writer.consolidate(dedup_on=HONEST_DEDUP_KEYS)
    if not bank.empty:
        by_model = bank.groupby("weak_model_key")["is_correct"].agg(["mean", "size"])
        logger.info("Honest bank accuracy by weak model:\n%s", by_model.to_string())
    return bank


def bank_composition_summary(bank: pd.DataFrame) -> pd.DataFrame:
    """
    Per (question, replicate): how many honest peers were correct.

    The released C4H showed harmful revision of 25.9% / 7.9% / 1.4% when both,
    one or neither honest peer was wrong. That stratification is observational,
    so this table exists to report it as such rather than as an assigned
    treatment.
    """
    if bank.empty:
        return pd.DataFrame()
    grouped = (
        bank.groupby(["question_identifier", "replicate"])["is_correct"]
        .agg(["sum", "size"])
        .reset_index()
        .rename(columns={"sum": "n_peers_correct", "size": "n_peers"})
    )
    grouped["composition"] = grouped["n_peers_correct"].map(
        {0: "both_wrong", 1: "split", 2: "both_right"}
    ).fillna("other")
    return grouped
