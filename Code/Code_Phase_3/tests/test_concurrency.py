"""
Tests for in-process concurrency (src/concurrency.py).

The claim being tested: running a shard's units on several threads changes
the wall-clock time and nothing else. So the same condition run with one
worker and with four must produce the same rows, the same peers and the same
checkpoint; the writers must lose nothing under contention; and an error in
any worker must stop the run as it did when the loop was sequential.
"""

from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

from src.agent_wrappers.base_agent import AgentResponse
from src.concurrency import run_units
from src.peer_pools import PersonaPools
from src.revision_runner import run_condition
from src.seeding import r0_unit_id
from src.snapshots import SnapshotAuditor
from src.store import Checkpoint, IncrementalWriter


# ── run_units ─────────────────────────────────────────────────────────────
def test_results_come_back_in_input_order():
    def slow_square(x):
        time.sleep(0.01 * (5 - x % 5))
        return x * x
    assert run_units(range(20), slow_square, workers=6) == [x * x for x in range(20)]


def test_first_error_stops_the_run_and_cancels_what_has_not_started():
    started = []
    lock = threading.Lock()

    def work(x):
        with lock:
            started.append(x)
        if x == 3:
            raise RuntimeError("provenance violation")
        time.sleep(0.05)
        return x

    with pytest.raises(RuntimeError, match="provenance violation"):
        run_units(range(200), work, workers=4)
    assert len(started) < 200, "units queued after the failure must not start"


def test_one_worker_is_the_plain_loop():
    order = []
    run_units(range(5), order.append, workers=1)
    assert order == [0, 1, 2, 3, 4]


# ── writers under contention ──────────────────────────────────────────────
def test_writer_and_checkpoint_lose_nothing_under_eight_threads(tmp_path):
    checkpoint = Checkpoint(tmp_path / "ck.parquet")
    writer = IncrementalWriter(tmp_path / "out.parquet", flush_every=7,
                               checkpoint=checkpoint)

    def write(i):
        writer.append({"unit_id": f"u{i}", "value": i}, unit_id=f"u{i}")

    run_units(range(4000), write, workers=8)
    frame = writer.consolidate(dedup_on=["unit_id"])
    assert len(frame) == 4000 and frame["unit_id"].nunique() == 4000
    assert Checkpoint(tmp_path / "ck.parquet").count == 4000


# ── a real condition, 1 worker vs 4 ───────────────────────────────────────
class EchoAgent:
    """Deterministic in the prompt, so thread scheduling cannot change output."""
    provider = "stub"
    model_name = "stub-model"

    def generate_response(self, system_prompt, user_prompt, temperature,
                          maximum_output_tokens, request_metadata=None):
        time.sleep(0.002)
        answer = "7" if "Agent_" in user_prompt else "4"
        return AgentResponse(
            raw_text_output=f"seen {abs(hash(user_prompt)) % 10**8}\nFinal answer: {answer}",
            model_name_returned_by_provider="stub-model", error_status="success")


N_Q, N_REP = 12, 3
QUESTIONS = pd.DataFrame([{
    "question_identifier": f"q{i}", "question_text": f"What is {i} + 4?",
    "answer_options": None, "correct_answer": "4", "source_dataset": "gsm8k",
    "subject_category": "math"} for i in range(N_Q)])


def _persona(qid, i, target):
    return {"persona_identifier": f"{qid}_persona_{i}", "question_identifier": qid,
            "persona_variant_index": i, "assigned_wrong_answer_letter_or_value": target,
            "generated_persona_text": f"Because {i}.\nFinal answer: {target}",
            "generator_model_name": "llama"}


def _run(tmp_path, workers):
    pools = PersonaPools(
        anchored_wrong={f"q{i}": [_persona(f"q{i}", v, t) for v, t in enumerate("75798")]
                        for i in range(N_Q)},
        anchored_correct={}, anchored_hedged={}, honest_bank={})
    r0 = {("focal", f"q{i}", r): {
        "r0_unit_id": r0_unit_id("focal", f"q{i}", r), "focal_key": "focal",
        "question_identifier": f"q{i}", "replicate": r,
        "raw_response_text": "Thinking.\nFinal answer: 4", "extracted_answer": "4",
        "extracted_confidence": None, "is_correct": True}
        for i in range(N_Q) for r in range(N_REP)}
    checkpoint = Checkpoint(tmp_path / "ck.parquet")
    revisions = IncrementalWriter(tmp_path / "rev.parquet", flush_every=5,
                                  checkpoint=checkpoint)
    peers = IncrementalWriter(tmp_path / "peers.parquet", flush_every=5)
    calls = run_condition(
        condition_name="WR", condition={"peer_source": "anchored_wrong", "n_peers": 2},
        focal_key="focal", focal_agent=EchoAgent(), focal_spec={},
        questions=QUESTIONS, r0_index=r0, pools=pools,
        weak_specs={"w1": {"paper_name": "W1"}, "w2": {"paper_name": "W2"}},
        weak_agents={}, judge_cascade=None, auditor=SnapshotAuditor(),
        revision_writer=revisions, peer_writer=peers, checkpoint=checkpoint,
        master_seed=3, replicates=N_REP, rounds=2, rounds_config={},
        experiment_name="t", prior_rounds={}, workers=workers)
    rev = revisions.consolidate(dedup_on=["unit_id"])
    peer = peers.consolidate(dedup_on=["unit_id", "peer_display_name", "shown_to_focal"])
    keep = ["unit_id", "round_index", "raw_response_text", "extracted_answer",
            "is_correct", "adopted_peer_target", "peer_asserted_targets_all"]
    return (calls, rev[keep].sort_values("unit_id").reset_index(drop=True),
            peer[["unit_id", "persona_identifier", "peer_display_name"]]
            .sort_values(["unit_id", "peer_display_name"]).reset_index(drop=True),
            Checkpoint(tmp_path / "ck.parquet").count)


def test_four_workers_equal_one_worker(tmp_path):
    one = _run(tmp_path / "one", workers=1)
    four = _run(tmp_path / "four", workers=4)
    assert one[0] == four[0] == N_Q * N_REP * 2
    pd.testing.assert_frame_equal(one[1], four[1])
    pd.testing.assert_frame_equal(one[2], four[2])
    assert one[3] == four[3] == N_Q * N_REP * 2
