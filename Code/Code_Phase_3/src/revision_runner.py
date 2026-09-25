"""
revision_runner.py — Protocol B execution engine.

Runs one (focal model, condition, pool) cell: for every (question, replicate)
it clones the cached Round-0 record, builds that condition's CONTEXT_BLOCK,
issues ONE revision call with the shared template, and records the outcome.

Multi-round (X3): rounds 2+ feed the focal its own previous-round text.
Anchored peers re-assert their message each round (an adversary stays
committed, matching Nilayam et al. 2026); honest and self peers make their own
stateful revision call, so they can move.

Every row records the peers actually shown, their provenance, the filter
decisions, and the peer-asserted target — the analysis needs all four.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .contexts import (
    PeerMessage,
    apply_confidence_filter,
    build_revision_prompt,
    filtered_placeholder_peer,
    order_peers,
)
from .extraction import extract_answer, extract_confidence, grade, options_to_string
from .peer_pools import PersonaPools, build_peers, primary_wrong_target
from .seeding import unit_id
from .store import Checkpoint, IncrementalWriter
from .call_guard import is_failed_call, record_failure
from .concurrency import run_units

logger = logging.getLogger("platos_ship3.revision_runner")

REVISION_DEDUP_KEYS = ["unit_id"]


def _self_samples_for(
    r0_index: Dict[tuple, Dict[str, Any]],
    focal_key: str,
    question_id: str,
    replicate: int,
    n_peers: int,
    n_replicates: int,
) -> List[Dict[str, Any]]:
    """
    Condition E: extra Round-0 samples of the focal model itself.

    Reuses the OTHER cached replicates of the same question, so the homogeneous
    control costs nothing extra and is exactly "three samples of one model" —
    the self-consistency comparison Choi et al. (NeurIPS 2025) and Zhang et al.
    (2025) ask for.
    """
    samples = []
    for offset in range(1, n_replicates):
        other = (replicate + offset) % n_replicates
        record = r0_index.get((focal_key, question_id, other))
        if record is not None:
            samples.append(record)
        if len(samples) == n_peers:
            break
    return samples


def _wrong_targets(peers: List[PeerMessage]) -> List[str]:
    return sorted({p.assigned_target for p in peers
                   if p.assigned_target and p.anchor_mode in ("wrong", "hedged")})


def _peer_rows(unit: str, peers: List[PeerMessage], shown: bool) -> List[Dict[str, Any]]:
    """One row per peer message, for the separate peer-message log."""
    return [{"unit_id": unit, "shown_to_focal": shown, **peer.to_row()} for peer in peers]


def run_condition(
    condition_name: str,
    condition: Dict[str, Any],
    focal_key: str,
    focal_agent,
    focal_spec: Dict[str, Any],
    questions: pd.DataFrame,
    r0_index: Dict[tuple, Dict[str, Any]],
    pools: PersonaPools,
    weak_specs: Dict[str, Any],
    weak_agents: Dict[str, Any],
    judge_cascade,
    auditor,
    revision_writer: IncrementalWriter,
    peer_writer: IncrementalWriter,
    checkpoint: Checkpoint,
    master_seed: int,
    replicates: int,
    protocol: str = "B",
    rounds: int = 1,
    rounds_config: Optional[Dict[str, Any]] = None,
    temperature: float = 0.7,
    max_output_tokens: int = 600,
    experiment_name: str = "",
    dry_run: bool = False,
    prior_rounds: Optional[Dict[str, Dict[str, Any]]] = None,
    workers: int = 1,
) -> int:
    """
    Execute one condition cell. Returns the number of revision calls made.

    `prior_rounds` maps a unit id to the row already written for it
    ({raw_response_text, extracted_answer}). It is only consulted when resuming
    a multi-round condition, to restore what the focal model saw as its own
    previous answer. Built by `load_prior_rounds`.
    """
    rounds_config = rounds_config or {}
    peer_source = condition.get("peer_source", "none")
    source_framing = condition.get("source_framing", "peer_attributed")
    filter_config = condition.get("confidence_filter") or {}

    rows = list(questions.iterrows())
    if dry_run:
        rows = rows[:1]

    cells = [(question, replicate)
             for _, question in rows
             for replicate in range(1 if dry_run else replicates)]

    def _run_cell(cell) -> int:
        """One (question, replicate) unit, all of its rounds. Returns calls made."""
        question, replicate = cell
        question_id = question["question_identifier"]
        options_str = options_to_string(question.get("answer_options"))
        calls_made = 0

        # Peers are chosen once per (question, replicate) cell and held
        # fixed across its rounds, so the peer log keys on round 1's id.
        cell_unit = unit_id(protocol, focal_key, condition_name,
                            question_id, replicate, 1)

        # A unit is done only when every one of its rounds is done.
        if all(unit_id(protocol, focal_key, condition_name, question_id,
                       replicate, r) in checkpoint
               for r in range(1, rounds + 1)):
            return calls_made

        r0 = r0_index.get((focal_key, question_id, replicate))
        if r0 is None:
            logger.warning("No cached R0 for (%s, %s, r%d); skipping.",
                           focal_key, question_id, replicate)
            return calls_made

        self_samples = _self_samples_for(
            r0_index, focal_key, question_id, replicate,
            int(condition.get("n_peers", 0)), replicates,
        )
        peers, diagnostics = build_peers(
            condition={**condition, "_name": condition_name},
            question=question.to_dict(),
            replicate=replicate,
            pools=pools,
            weak_specs=weak_specs,
            master_seed=master_seed,
            focal_key=focal_key,
            self_samples=self_samples,
        )
        # A unit runs only with EXACTLY the designed number of peers.
        # Skipping only on zero peers let a unit run with one peer where
        # the design says two, which silently changes the treatment.
        n_designed = int(condition.get("n_peers", 0))
        if peer_source not in ("none", "generic") and len(peers) != n_designed:
            logger.warning(
                "Condition %s: skipped (%s, %s, r%d) — %s",
                condition_name, focal_key, question_id, replicate,
                diagnostics.get("skipped_reason")
                or f"built {len(peers)} of {n_designed} designed peers",
            )
            return calls_made

        # Order seeded like the draw: the same personas appear in the same
        # order in WR, WRh, W and SF, and for every focal model.
        peers = order_peers(peers, master_seed, question_id, replicate)

        # The deployed retain-high-confidence rule, when this condition uses it.
        filter_diagnostics: Dict[str, Any] = {}
        if filter_config.get("enabled"):
            kept, filter_diagnostics = apply_confidence_filter(
                peers,
                threshold=int(filter_config.get("threshold", 60)),
                unparseable_counts_as=filter_config.get(
                    "unparseable_counts_as", "dropped"),
            )
            peer_writer.extend(_peer_rows(cell_unit, peers, shown=False))
            shown_peers = kept or [filtered_placeholder_peer()]
            peer_writer.extend(_peer_rows(cell_unit, kept, shown=True))
        else:
            shown_peers = peers
            peer_writer.extend(_peer_rows(cell_unit, peers, shown=True))

        # Targets of the peers the model was SHOWN (after any filter): a
        # model cannot adopt an answer it never saw. Wrong targets only:
        # in CR the correct-anchored peer also carries an assigned answer,
        # and moving to it is a correct revision, not adoption.
        target = primary_wrong_target(shown_peers)
        all_targets = _wrong_targets(shown_peers)
        targets_before_filter = _wrong_targets(peers)

        # ── rounds ────────────────────────────────────────────────────
        # `previous_text` is what the focal model is shown as its own
        # previous answer: the cached Round-0 text in round 1, and the
        # preceding round's output thereafter. Protocol B depends on this
        # being right, so a resumed run must restore it rather than skip
        # past it — skipping a completed round without restoring its output
        # would feed the next round the Round-0 text and silently change
        # the experiment.
        previous_text = r0["raw_response_text"]
        previous_answer = r0.get("extracted_answer")
        round_peers = list(shown_peers)

        for round_index in range(1, rounds + 1):
            unit = unit_id(protocol, focal_key, condition_name,
                           question_id, replicate, round_index)
            if unit in checkpoint and round_index < rounds:
                recovered = prior_rounds.get(unit) if prior_rounds else None
                if recovered and (recovered.get("raw_response_text") or "").strip():
                    previous_text = recovered["raw_response_text"]
                    previous_answer = recovered.get("extracted_answer")
                    continue
                # Checkpointed but unrecoverable: re-run this round rather
                # than hand the next one the wrong context. One wasted call
                # is cheaper than a corrupted unit.
                logger.warning(
                    "Round %d of %s is checkpointed but its text could not "
                    "be recovered; re-running the round so the next one "
                    "receives the correct previous answer.",
                    round_index, unit,
                )

            # Experiment A: `show_initial: false` hides the cached first answer
            # in round 1. The hidden answer still defines the unit's initial
            # state (r0_is_correct), so transitions are resampling transitions.
            shown_previous = (None if (round_index == 1 and
                                       condition.get("show_initial", True) is False)
                              else previous_text)
            system_prompt, user_prompt = build_revision_prompt(
                question_text=question["question_text"],
                answer_options=question.get("answer_options"),
                own_previous_text=shown_previous,
                peers=round_peers,
                peer_source=peer_source,
                source_framing=source_framing,
            )

            started = time.time()
            response = focal_agent.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                maximum_output_tokens=max_output_tokens,
                request_metadata={
                    "stage": "revision", "condition": condition_name,
                    "question_id": question_id, "replicate": replicate,
                    "round": round_index, "focal": focal_key,
                },
            )
            calls_made += 1
            if is_failed_call(response):
                # Not graded as a wrong answer, not checkpointed. Later
                # rounds of this unit would be fed a missing answer, so
                # stop here; the resume logic restores earlier rounds.
                record_failure("revision", unit, response)
                break

            auditor.check(
                agent_key=focal_key,
                expected_prefix=focal_spec.get("expected_served_prefix"),
                served_model=response.model_name_returned_by_provider,
                role="focal_revision",
                strict=True,
                context={"condition": condition_name,
                         "question_id": question_id, "replicate": replicate,
                         "round": round_index},
            )

            answer, method = extract_answer(
                response.raw_text_output, question["question_text"],
                options_str, judge_cascade, answer_options=question.get("answer_options"),
            )
            confidence, confidence_status = extract_confidence(
                response.raw_text_output)
            is_correct = grade(answer, question["correct_answer"])
            r0_correct = bool(r0["is_correct"])

            revision_writer.append(
                {
                    "unit_id": unit,
                    "protocol": protocol,
                    "experiment": experiment_name,
                    "condition": condition_name,
                    "peer_source": peer_source,
                    "source_framing": source_framing,
                    "focal_key": focal_key,
                    "focal_served_model": response.model_name_returned_by_provider,
                    "question_identifier": question_id,
                    "source_dataset": question.get("source_dataset"),
                    "subject_category": question.get("subject_category"),
                    "replicate": replicate,
                    "round_index": round_index,
                    "n_rounds_total": rounds,
                    "initial_answer_shown": shown_previous is not None,
                    # ── initial state (identical across conditions) ──
                    "r0_unit_id": r0["r0_unit_id"],
                    # Always the Round-0 answer, matching r0_is_correct.
                    # It used to hold the previous ROUND's answer from
                    # round 2 on, while r0_is_correct stayed Round 0's.
                    "r0_answer": r0.get("extracted_answer"),
                    "previous_round_answer": previous_answer,
                    "r0_is_correct": r0_correct,
                    "r0_confidence": r0.get("extracted_confidence"),
                    # ── revision outcome ────────────────────────────
                    "raw_response_text": response.raw_text_output,
                    "extracted_answer": answer,
                    "answer_extraction_method": method,
                    "extracted_confidence": confidence,
                    "confidence_parse_status": confidence_status,
                    "correct_answer": question["correct_answer"],
                    "is_correct": is_correct,
                    # ── derived revision events (denominators applied
                    #     in analysis/metrics.py, never here) ─────────
                    "flip_correct_to_incorrect": r0_correct and not is_correct,
                    "flip_incorrect_to_correct": (not r0_correct) and is_correct,
                    "answer_changed": not _same(answer, r0.get("extracted_answer")),
                    # ── peer exposure ───────────────────────────────
                    # The "all peers filtered" notice is not a peer.
                    "n_peers_shown": sum(1 for p in round_peers
                                         if p.peer_source != "filtered_empty"),
                    "peer_asserted_target": target,
                    "peer_asserted_targets_all": json.dumps(all_targets),
                    "peer_targets_before_filter": json.dumps(
                        targets_before_filter),
                    # Adoption = the revised answer is ANY target a peer
                    # argued for. Scoring only the modal target counted a
                    # switch to the other peer's wrong answer as "not
                    # adopted" whenever the two peers disagreed.
                    "adopted_peer_target": any(
                        _same(answer, t) for t in all_targets),
                    "adopted_primary_peer_target": bool(
                        target is not None and _same(answer, target)),
                    "peers_agree_on_target": diagnostics.get(
                        "peers_agree_on_target"),
                    "n_distinct_peer_targets": diagnostics.get(
                        "n_distinct_targets"),
                    "n_honest_peers_correct": diagnostics.get(
                        "n_honest_peers_correct"),
                    # ── filter ──────────────────────────────────────
                    "filter_enabled": bool(filter_config.get("enabled")),
                    "n_peers_before_filter": filter_diagnostics.get(
                        "n_peers_before_filter"),
                    "n_peers_retained": filter_diagnostics.get("n_peers_retained"),
                    "filter_decisions_json": json.dumps(
                        filter_diagnostics.get("filter_decisions", [])),
                    # ── cost / provenance ───────────────────────────
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "total_input_tokens": response.total_input_tokens,
                    "total_output_tokens": response.total_output_tokens,
                    "wall_clock_latency_seconds":
                        response.wall_clock_latency_seconds,
                    "error_status": response.error_status,
                    "finish_reason": response.finish_reason,
                    "served_route": response.served_route,
                    "retry_attempts_used": response.retry_attempts_used,
                    "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
                    "elapsed_seconds": round(time.time() - started, 3),
                },
                unit_id=unit,
            )

            # Prepare the next round.
            if round_index < rounds:
                previous_text = response.raw_text_output
                previous_answer = answer
                round_peers = _advance_peers(
                    round_peers, question, peer_source, rounds_config,
                    weak_agents, weak_specs, judge_cascade, auditor,
                    temperature, max_output_tokens, round_index,
                )
        return calls_made

    # Units are independent (see src/concurrency.py), so they may run on
    # several threads; with workers=1 this is the sequential loop it replaced.
    return sum(run_units(cells, _run_cell, workers=workers,
                         label=f"{focal_key}:{condition_name}"))


def _same(left: Any, right: Any) -> bool:
    from .extraction import answers_equal

    return answers_equal(left, right)


def _advance_peers(
    peers: List[PeerMessage],
    question: pd.Series,
    peer_source: str,
    rounds_config: Dict[str, Any],
    weak_agents: Dict[str, Any],
    weak_specs: Dict[str, Any],
    judge_cascade,
    auditor,
    temperature: float,
    max_output_tokens: int,
    round_index: int,
) -> List[PeerMessage]:
    """
    Produce the peer messages for the next round.

    Anchored peers re-assert the same message when
    `rounds.anchored_peers_reassert` is true: a committed adversary does not
    change its mind, which is the assumption the scaling study states openly.
    Honest and self peers are not re-queried here — round 2+ for those peers
    would need their own cached state, and X3 only sweeps rounds under the
    anchored condition, so this keeps the design honest rather than inventing
    peer behaviour the data cannot support.
    """
    if peer_source in ("anchored_wrong", "anchored_hedged", "anchored_split",
                       "anchored_confidence", "bare_answer"):
        if rounds_config.get("anchored_peers_reassert", True):
            return peers
    return peers


def load_prior_rounds(revision_log_path) -> Dict[str, Dict[str, Any]]:
    """
    Index an existing revision log by unit id, for multi-round resume.

    Only the columns needed to reconstruct what the next round should see are
    kept, so this stays cheap even on a large log.
    """
    path = Path(revision_log_path)
    if not path.exists():
        return {}
    try:
        frame = pd.read_parquet(
            path, columns=["unit_id", "raw_response_text", "extracted_answer"])
    except Exception as exc:                 # a truncated shard must not abort
        logger.warning("Could not read prior rounds from %s (%s); affected "
                       "multi-round units will re-run.", path, exc)
        return {}
    return {
        row["unit_id"]: {"raw_response_text": row["raw_response_text"],
                         "extracted_answer": row["extracted_answer"]}
        for row in frame.to_dict("records")
    }
