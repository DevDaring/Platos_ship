"""
test_multiround_resume.py — a resumed multi-round run must equal an
uninterrupted one.

Protocol B's guarantee is that the focal model always sees its OWN previous
answer. In a multi-round condition (X3's round sweep) "previous" means the
previous ROUND, not the cached Round-0 answer. If a run is interrupted between
rounds and resumed, the restarted round must still receive the earlier round's
text.

This is the regression test for a resume path that skipped a completed round
without restoring its output, so round 2 silently received the Round-0 text
instead. Nothing in the logs would have shown it: the row looks complete, the
snapshot pin holds, the answer parses. It would simply be a different
experiment from the one the paper describes.

Offline: a stub agent records what it was shown.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.agent_wrappers.base_agent import AgentResponse
from src.peer_pools import PersonaPools
from src.revision_runner import run_condition
from src.seeding import r0_unit_id, unit_id
from src.snapshots import SnapshotAuditor
from src.store import Checkpoint, IncrementalWriter


class RecordingAgent:
    """Returns a per-round marker and records every prompt it was shown."""

    provider = "stub"
    model_name = "stub-model"

    def __init__(self):
        self.prompts = []
        self.round = 0

    def generate_response(self, system_prompt, user_prompt, temperature,
                          maximum_output_tokens, request_metadata=None):
        self.round += 1
        self.prompts.append(user_prompt)
        return AgentResponse(
            raw_text_output=(f"Reasoning for round {self.round}.\n"
                             f"Final answer: A\nConfidence: 80"),
            model_name_returned_by_provider="stub-model",
            error_status="success",
        )


QUESTIONS = pd.DataFrame([{
    "question_identifier": "q1",
    "question_text": "What is 2 + 2?",
    "answer_options": None,
    "correct_answer": "4",
    "source_dataset": "gsm8k",
    "subject_category": "math",
}])

R0_TEXT = "Initial thinking.\nFinal answer: B\nConfidence: 70"


def _r0_index():
    return {
        ("focal", "q1", 0): {
            "r0_unit_id": r0_unit_id("focal", "q1", 0),
            "focal_key": "focal",
            "question_identifier": "q1",
            "replicate": 0,
            "raw_response_text": R0_TEXT,
            "extracted_answer": "B",
            "extracted_confidence": 70,
            "is_correct": False,
        }
    }


def _run(tmp_path: Path, checkpoint: Checkpoint, agent: RecordingAgent,
         prior_rounds=None, rounds: int = 2):
    revision_writer = IncrementalWriter(tmp_path / "rev.parquet",
                                        flush_every=1, checkpoint=checkpoint)
    peer_writer = IncrementalWriter(tmp_path / "peers.parquet", flush_every=1)
    run_condition(
        condition_name="R",
        condition={"peer_source": "none", "n_peers": 0},
        focal_key="focal",
        focal_agent=agent,
        focal_spec={},
        questions=QUESTIONS,
        r0_index=_r0_index(),
        pools=PersonaPools({}, {}, {}, {}),
        weak_specs={},
        weak_agents={},
        judge_cascade=None,
        auditor=SnapshotAuditor(),
        revision_writer=revision_writer,
        peer_writer=peer_writer,
        checkpoint=checkpoint,
        master_seed=20260502,
        replicates=1,
        rounds=rounds,
        rounds_config={},
        experiment_name="test",
        prior_rounds=prior_rounds or {},
    )
    revision_writer.consolidate(dedup_on=["unit_id"])
    peer_writer.consolidate(dedup_on=["unit_id", "peer_display_name",
                                      "shown_to_focal"])
    return revision_writer.target


class TestMultiRoundResume:

    def test_uninterrupted_round2_sees_round1_output(self, tmp_path):
        agent = RecordingAgent()
        _run(tmp_path, Checkpoint(tmp_path / "ck.parquet"), agent)
        assert len(agent.prompts) == 2
        assert R0_TEXT in agent.prompts[0], "round 1 must see the Round-0 answer"
        assert "Reasoning for round 1." in agent.prompts[1], (
            "round 2 must see round 1's output, not the Round-0 answer"
        )

    def test_resumed_round2_also_sees_round1_output(self, tmp_path):
        # Simulate a crash after round 1: its unit is checkpointed and its row
        # is on disk, but round 2 never ran.
        checkpoint = Checkpoint(tmp_path / "ck.parquet")
        round1_unit = unit_id("B", "focal", "R", "q1", 0, 1)
        checkpoint.add(round1_unit)
        checkpoint.save()

        prior = {round1_unit: {"raw_response_text": "Reasoning for round 1.\n"
                                                   "Final answer: A\n"
                                                   "Confidence: 80",
                               "extracted_answer": "A"}}

        agent = RecordingAgent()
        _run(tmp_path, checkpoint, agent, prior_rounds=prior)

        assert len(agent.prompts) == 1, "only the missing round should re-run"
        prompt = agent.prompts[0]
        assert "Reasoning for round 1." in prompt, (
            "THE BUG: the resumed round 2 received the Round-0 text instead of "
            "round 1's output, silently changing the experiment"
        )
        assert R0_TEXT not in prompt

    def test_resume_without_recoverable_text_reruns_rather_than_guesses(
            self, tmp_path):
        # Checkpointed but the text cannot be recovered (log lost/truncated).
        # Re-running costs a call; guessing corrupts the experiment.
        checkpoint = Checkpoint(tmp_path / "ck.parquet")
        checkpoint.add(unit_id("B", "focal", "R", "q1", 0, 1))
        checkpoint.save()

        agent = RecordingAgent()
        _run(tmp_path, checkpoint, agent, prior_rounds={})
        assert len(agent.prompts) == 2, (
            "with no recoverable prior-round text the unit must re-run from "
            "round 1 rather than feed round 2 the wrong context"
        )
        assert "Reasoning for round 1." in agent.prompts[1]

    def test_fully_completed_unit_is_skipped(self, tmp_path):
        checkpoint = Checkpoint(tmp_path / "ck.parquet")
        for r in (1, 2):
            checkpoint.add(unit_id("B", "focal", "R", "q1", 0, r))
        checkpoint.save()

        agent = RecordingAgent()
        _run(tmp_path, checkpoint, agent)
        assert agent.prompts == [], "a completed unit must cost nothing"

    def test_single_round_conditions_are_unaffected(self, tmp_path):
        agent = RecordingAgent()
        _run(tmp_path, Checkpoint(tmp_path / "ck.parquet"), agent, rounds=1)
        assert len(agent.prompts) == 1
        assert R0_TEXT in agent.prompts[0]
