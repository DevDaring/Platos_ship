"""
test_pipeline_integration.py — the whole pipeline, offline, with stub agents.

The unit tests pin individual repairs; this one checks that the pieces actually
compose: Round-0 cache -> peer assembly -> revision conditions -> registry ->
metrics -> contrasts -> design check. Nothing here touches the network, so it
runs in CI and before every real run.

What it would catch that the unit tests would not: a renamed column between the
runner and the registry, a checkpoint key that does not round-trip, a condition
that silently produces no rows, a metric computed on the wrong denominator once
real frames flow through it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.design_check import check_collected
from analysis.metrics import harmful, summarise_by
from analysis.registry import _scope_for_question
from src.agent_wrappers.base_agent import AgentResponse
from src.contexts import build_round0_prompt
from src.peer_pools import PersonaPools
from src.r0_cache import build_r0_cache, load_r0_cache, r0_lookup, solo_accuracy_by_focal
from src.revision_runner import REVISION_DEDUP_KEYS, run_condition
from src.snapshots import SnapshotAuditor
from src.store import Checkpoint, IncrementalWriter

N_QUESTIONS = 6
N_REPLICATES = 3


class ScriptedAgent:
    """
    A focal stand-in whose correctness depends on the condition it is in.

    Deterministic by design: the same prompt always yields the same answer, so
    the pipeline's own determinism is what the test measures.
    """

    provider = "stub"

    def __init__(self, name: str, served: str, flip_under_peers: bool):
        self.agent_name = name
        self.model_name = served
        self.served = served
        self.flip_under_peers = flip_under_peers
        self.calls = 0

    def generate_response(self, system_prompt, user_prompt, temperature,
                          maximum_output_tokens, request_metadata=None):
        self.calls += 1
        meta = request_metadata or {}
        qid = str(meta.get("question_id", "q0"))
        index = int(qid.split("_")[-1]) if "_" in qid else 0

        if meta.get("stage") == "r0":
            # Two-thirds of questions start correct.
            answer = "A" if index % 3 != 2 else "B"
        else:
            saw_peers = "Other agents' responses:" in user_prompt
            if saw_peers and self.flip_under_peers and index % 2 == 0:
                answer = "B"          # abandons the correct answer
            elif saw_peers and index % 3 == 2:
                answer = "A"          # recovers a wrong one
            else:
                answer = "A" if index % 3 != 2 else "B"

        return AgentResponse(
            raw_text_output=f"Working.\nFinal answer: {answer}\nConfidence: 85",
            model_name_returned_by_provider=self.served,
            total_input_tokens=100, total_output_tokens=40,
            error_status="success",
        )


@pytest.fixture
def questions() -> pd.DataFrame:
    return pd.DataFrame([{
        "question_identifier": f"mmlupro_{i:04d}",
        "question_text": f"Question number {i}?",
        "answer_options": json.dumps(["alpha", "beta", "gamma"]),
        "correct_answer": "A",
        "source_dataset": "mmlu_pro",
        "subject_category": "math",
        "wrong_answer_pool": json.dumps(["B", "C"]),
    } for i in range(N_QUESTIONS)])


@pytest.fixture
def pools(questions) -> PersonaPools:
    anchored = {}
    for qid in questions["question_identifier"]:
        anchored[qid] = [{
            "persona_identifier": f"{qid}_persona_{v}",
            "question_identifier": qid,
            "persona_variant_index": v,
            "assigned_wrong_answer_letter_or_value": "B" if v % 2 == 0 else "C",
            "generated_persona_text": (f"It is clearly {'B' if v % 2 == 0 else 'C'}.\n"
                                       f"Final answer: {'B' if v % 2 == 0 else 'C'}\n"
                                       f"Confidence: 95"),
            "generator_model_name": "stub-llama",
        } for v in range(4)]
    return PersonaPools(anchored_wrong=anchored, anchored_correct={},
                        anchored_hedged={}, honest_bank={})


def _agents():
    return {
        "strong": ScriptedAgent("strong", "stub-strong", flip_under_peers=False),
        "weak": ScriptedAgent("weak", "stub-weak", flip_under_peers=True),
    }


def _run_pipeline(tmp_path: Path, questions, pools):
    agents = _agents()
    specs = {k: {"expected_served_prefix": f"stub-{k}", "paper_name": k}
             for k in agents}
    auditor = SnapshotAuditor()

    r0_cache = build_r0_cache(
        questions=questions, focal_keys=list(agents), focal_agents=agents,
        focal_specs=specs, judge_cascade=None, auditor=auditor,
        cache_path=tmp_path / "r0.parquet",
        checkpoint=Checkpoint(tmp_path / "ck_r0.parquet"),
        replicates=N_REPLICATES,
    )

    checkpoint = Checkpoint(tmp_path / "ck.parquet")
    revision_writer = IncrementalWriter(tmp_path / "rev.parquet",
                                        flush_every=10, checkpoint=checkpoint)
    peer_writer = IncrementalWriter(tmp_path / "peers.parquet", flush_every=10)
    index = r0_lookup(r0_cache)

    for focal_key, agent in agents.items():
        for condition_name, condition in [
            ("R", {"peer_source": "none", "n_peers": 0}),
            ("WR", {"peer_source": "anchored_wrong", "n_peers": 2}),
        ]:
            run_condition(
                condition_name=condition_name, condition=condition,
                focal_key=focal_key, focal_agent=agent,
                focal_spec=specs[focal_key], questions=questions,
                r0_index=index, pools=pools, weak_specs={"w": {"paper_name": "W"}},
                weak_agents={}, judge_cascade=None, auditor=auditor,
                revision_writer=revision_writer, peer_writer=peer_writer,
                checkpoint=checkpoint, master_seed=20260502,
                replicates=N_REPLICATES, experiment_name="X1_common_matrix",
            )

    revisions = revision_writer.consolidate(dedup_on=REVISION_DEDUP_KEYS)
    peers = peer_writer.consolidate(
        dedup_on=["unit_id", "peer_display_name", "shown_to_focal"])
    return r0_cache, revisions, peers, auditor, agents


class TestPipeline:

    def test_r0_cache_is_complete_and_shared(self, tmp_path, questions, pools):
        r0, revisions, _, _, _ = _run_pipeline(tmp_path, questions, pools)
        assert len(r0) == 2 * N_QUESTIONS * N_REPLICATES
        assert r0["r0_unit_id"].is_unique
        # Every revision must point at a cached initial answer, and each
        # (focal, question, replicate) must reuse the SAME one across
        # conditions — that is the whole point of Protocol B.
        merged = revisions.merge(r0[["r0_unit_id", "raw_response_text"]],
                                 on="r0_unit_id", how="left",
                                 suffixes=("", "_cached"))
        assert merged["raw_response_text_cached"].notna().all()
        per_cell = revisions.groupby(
            ["focal_key", "question_identifier", "replicate"])["r0_unit_id"].nunique()
        assert (per_cell == 1).all(), "conditions diverged on the initial answer"

    def test_every_cell_is_populated(self, tmp_path, questions, pools):
        _, revisions, _, _, _ = _run_pipeline(tmp_path, questions, pools)
        counts = revisions.groupby(["focal_key", "condition"]).size()
        assert set(counts.index) == {("strong", "R"), ("strong", "WR"),
                                     ("weak", "R"), ("weak", "WR")}
        assert (counts == N_QUESTIONS * N_REPLICATES).all()

    def test_unit_ids_unique_and_checkpoint_round_trips(self, tmp_path,
                                                        questions, pools):
        _, revisions, _, _, _ = _run_pipeline(tmp_path, questions, pools)
        assert revisions["unit_id"].is_unique
        checkpoint = Checkpoint(tmp_path / "ck.parquet")
        assert checkpoint.count == len(revisions)
        assert all(u in checkpoint for u in revisions["unit_id"])

    def test_rerun_is_free(self, tmp_path, questions, pools):
        _run_pipeline(tmp_path, questions, pools)
        # Second pass over the same directory: everything is checkpointed.
        agents = _agents()
        specs = {k: {"expected_served_prefix": f"stub-{k}"} for k in agents}
        checkpoint = Checkpoint(tmp_path / "ck.parquet")
        writer = IncrementalWriter(tmp_path / "rev.parquet", flush_every=10,
                                   checkpoint=checkpoint)
        index = r0_lookup(load_r0_cache(tmp_path / "r0.parquet"))
        calls = run_condition(
            condition_name="WR",
            condition={"peer_source": "anchored_wrong", "n_peers": 2},
            focal_key="strong", focal_agent=agents["strong"],
            focal_spec=specs["strong"], questions=questions, r0_index=index,
            pools=pools, weak_specs={"w": {"paper_name": "W"}}, weak_agents={},
            judge_cascade=None, auditor=SnapshotAuditor(),
            revision_writer=writer,
            peer_writer=IncrementalWriter(tmp_path / "peers.parquet"),
            checkpoint=checkpoint, master_seed=20260502,
            replicates=N_REPLICATES, experiment_name="X1_common_matrix",
        )
        assert calls == 0, "a completed condition must cost nothing to re-run"

    def test_peer_provenance_is_recorded(self, tmp_path, questions, pools):
        _, _, peers, _, _ = _run_pipeline(tmp_path, questions, pools)
        anchored = peers[peers["peer_source"] == "anchored_wrong"]
        assert not anchored.empty
        # The Phase-2 defect: a message logged under a slot that did not write
        # it. Generator and slot must be separately recorded.
        assert (anchored["message_generator_model"] == "stub-llama").all()
        assert anchored["nominal_peer_slot_model"].notna().all()
        assert anchored["assigned_target"].isin(["B", "C"]).all()

    def test_snapshot_audit_covers_every_call(self, tmp_path, questions, pools):
        _, revisions, _, auditor, agents = _run_pipeline(tmp_path, questions, pools)
        audit = auditor.to_frame()
        assert len(audit) == sum(a.calls for a in agents.values())
        assert audit["matched"].all(), "stub models must satisfy their own pins"

    def test_metrics_flow_through_with_right_denominators(self, tmp_path,
                                                          questions, pools):
        _, revisions, _, _, _ = _run_pipeline(tmp_path, questions, pools)
        summary = summarise_by(revisions, ["focal_key", "condition"])
        assert not summary.empty
        for _, row in summary.iterrows():
            assert row["harmful_revision_denominator"] == row["initial_accuracy_n"]
            assert row["joint_loss_denominator"] == row["n_units"]
        # The weak stub abandons correct answers under peers; the strong one
        # does not. The gradient the paper reports should be visible here.
        weak = revisions[(revisions.focal_key == "weak") & (revisions.condition == "WR")]
        strong = revisions[(revisions.focal_key == "strong") & (revisions.condition == "WR")]
        assert harmful(weak).value > harmful(strong).value

    def test_registry_scoping_and_design_check(self, tmp_path, questions, pools):
        _, revisions, _, _, _ = _run_pipeline(tmp_path, questions, pools)
        revisions = revisions.copy()
        revisions["protocol"] = "B"
        revisions["dataset_scope"] = revisions["question_identifier"].map(
            _scope_for_question)
        assert (revisions["dataset_scope"] == "main300").all()
        # Both treatments have their baseline on both models.
        assert check_collected(revisions, protocol="B") == []

    def test_design_check_catches_a_missing_baseline(self, tmp_path, questions,
                                                     pools):
        _, revisions, _, _, _ = _run_pipeline(tmp_path, questions, pools)
        revisions = revisions.copy()
        revisions["protocol"] = "B"
        revisions["dataset_scope"] = "main300"
        # Drop the weak model's baseline: exactly the Phase-2 gap.
        broken = revisions[~((revisions.focal_key == "weak")
                             & (revisions.condition == "R"))]
        findings = check_collected(broken, protocol="B")
        assert any(f.severity == "error" and "weak" in f.focal_models
                   for f in findings)

    def test_solo_accuracy_is_measurable_per_model(self, tmp_path, questions,
                                                   pools):
        r0, _, _, _, _ = _run_pipeline(tmp_path, questions, pools)
        solo = solo_accuracy_by_focal(r0)
        assert set(solo) == {"strong", "weak"}
        assert all(0.0 <= v <= 1.0 for v in solo.values())

    def test_determinism(self, tmp_path, questions, pools):
        a = _run_pipeline(tmp_path / "a", questions, pools)[1]
        b = _run_pipeline(tmp_path / "b", questions, pools)[1]
        key = ["unit_id"]
        a = a.sort_values(key).reset_index(drop=True)
        b = b.sort_values(key).reset_index(drop=True)
        assert a["unit_id"].tolist() == b["unit_id"].tolist()
        # Peer selection is seeded, so the same targets must be chosen.
        assert a["peer_asserted_target"].tolist() == b["peer_asserted_target"].tolist()
