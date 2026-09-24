"""
Regression tests for the ten defects an independent audit found before the
Phase 3 API run (September 2026). Each test names the defect it pins down.

  1  peer draw seeded with the condition name  -> WR/WRh/W/SF saw different personas
  2  paired contrasts across unequal model sets -> 3-model mean vs 8-model mean
  3  solo accuracy pooled GSM-Symbolic Round-0  -> x-axis on unequal items
  4  failed API calls saved and checkpointed    -> outages became harmful flips
  5  adoption scored against one target only    -> the other peer's target missed
  6  option values never mapped to letters      -> correct answers scored wrong
  7  X6 verified every model, Qwen by Qwen      -> scoring set mostly unverified
  8  gate counted reused messages many times
  10 rounds pooled, budget means unweighted, dose points on different items
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pytest

from src import call_guard
from src.agent_wrappers.base_agent import AgentResponse
from src.agent_wrappers.judge_agent import extract_answer_regex
from src.contexts import PeerMessage
from src.extraction import extract_answer
from src.peer_pools import PersonaPools, build_peers
from src.r0_cache import build_r0_cache, solo_accuracy_by_focal
from src.revision_runner import run_condition
from src.seeding import r0_unit_id, unit_id
from src.snapshots import SnapshotAuditor
from src.store import Checkpoint, IncrementalWriter


@pytest.fixture(autouse=True)
def _clean_failures():
    call_guard.reset_failures()
    yield
    call_guard.reset_failures()


# ── shared stubs ───────────────────────────────────────────────────────────
class StubAgent:
    provider = "stub"
    model_name = "stub-model"

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.calls = 0
        self.prompts = []

    def generate_response(self, system_prompt, user_prompt, temperature,
                          maximum_output_tokens, request_metadata=None):
        self.calls += 1
        self.prompts.append(user_prompt)
        if self.calls in self.fail_on:
            return AgentResponse(raw_text_output="",
                                 model_name_returned_by_provider="stub-model",
                                 error_status="failure")
        return AgentResponse(raw_text_output=f"Round {self.calls}.\nFinal answer: 4",
                             model_name_returned_by_provider="stub-model",
                             error_status="success")


QUESTIONS = pd.DataFrame([{
    "question_identifier": "q1", "question_text": "What is 2 + 2?",
    "answer_options": None, "correct_answer": "4",
    "source_dataset": "gsm8k", "subject_category": "math",
}])


def _r0_index(answer="4", correct=True):
    return {("focal", "q1", 0): {
        "r0_unit_id": r0_unit_id("focal", "q1", 0), "focal_key": "focal",
        "question_identifier": "q1", "replicate": 0,
        "raw_response_text": f"Thinking.\nFinal answer: {answer}",
        "extracted_answer": answer, "extracted_confidence": None,
        "is_correct": correct}}


def _run(tmp_path, agent, checkpoint, condition=None, rounds=1, pools=None):
    revision_writer = IncrementalWriter(tmp_path / "rev.parquet", flush_every=1,
                                        checkpoint=checkpoint)
    peer_writer = IncrementalWriter(tmp_path / "peers.parquet", flush_every=1)
    run_condition(
        condition_name="R", condition=condition or {"peer_source": "none", "n_peers": 0},
        focal_key="focal", focal_agent=agent, focal_spec={}, questions=QUESTIONS,
        r0_index=_r0_index(), pools=pools or PersonaPools({}, {}, {}, {}),
        weak_specs={"w1": {"paper_name": "W1"}, "w2": {"paper_name": "W2"}},
        weak_agents={}, judge_cascade=None, auditor=SnapshotAuditor(),
        revision_writer=revision_writer, peer_writer=peer_writer,
        checkpoint=checkpoint, master_seed=1, replicates=1, rounds=rounds,
        rounds_config={}, experiment_name="t", prior_rounds={},
    )
    return revision_writer.consolidate(dedup_on=["unit_id"])


# ── 4: failed calls are never recorded ─────────────────────────────────────
class TestFailedCalls:

    def test_failed_revision_is_not_written_or_checkpointed(self, tmp_path):
        checkpoint = Checkpoint(tmp_path / "ck.parquet")
        rows = _run(tmp_path, StubAgent(fail_on={1}), checkpoint)
        assert rows.empty
        assert unit_id("B", "focal", "R", "q1", 0, 1) not in checkpoint
        assert call_guard.failure_counts() == {"revision": 1}

    def test_the_next_run_retries_the_failed_unit(self, tmp_path):
        checkpoint = Checkpoint(tmp_path / "ck.parquet")
        _run(tmp_path, StubAgent(fail_on={1}), checkpoint)
        rows = _run(tmp_path, StubAgent(), checkpoint)
        assert len(rows) == 1 and bool(rows["is_correct"].iloc[0])

    def test_failure_mid_rounds_stops_the_unit_after_saving_earlier_rounds(self, tmp_path):
        checkpoint = Checkpoint(tmp_path / "ck.parquet")
        agent = StubAgent(fail_on={2})
        rows = _run(tmp_path, agent, checkpoint, rounds=3)
        assert agent.calls == 2, "round 3 must not run on a missing round 2"
        assert list(rows["round_index"]) == [1]
        assert unit_id("B", "focal", "R", "q1", 0, 2) not in checkpoint

    def test_empty_text_with_success_status_is_data_not_failure(self, tmp_path):
        class Empty(StubAgent):
            def generate_response(self, *a, **k):
                return AgentResponse(raw_text_output="",
                                     model_name_returned_by_provider="stub-model",
                                     error_status="success")
        rows = _run(tmp_path, Empty(), Checkpoint(tmp_path / "ck.parquet"))
        assert len(rows) == 1 and call_guard.failure_counts() == {}

    def test_failed_round0_is_not_cached(self, tmp_path):
        checkpoint = Checkpoint(tmp_path / "r0ck.parquet")
        cache = build_r0_cache(
            questions=QUESTIONS, focal_keys=["focal"],
            focal_agents={"focal": StubAgent(fail_on={1})},
            focal_specs={"focal": {}}, judge_cascade=None,
            auditor=SnapshotAuditor(), cache_path=tmp_path / "r0.parquet",
            checkpoint=checkpoint, replicates=2)
        assert list(cache["replicate"]) == [1]
        assert r0_unit_id("focal", "q1", 0) not in checkpoint
        assert call_guard.failure_counts() == {"r0": 1}

    def test_drop_failed_calls_removes_stale_rows(self):
        frame = pd.DataFrame({"error_status": ["success", "failure",
                                               "api_error_recovered"]})
        assert len(call_guard.drop_failed_calls(frame, "t")) == 2

    def test_honest_index_ignores_failed_and_empty_messages(self):
        from src.peer_pools import _index_honest
        bank = pd.DataFrame([
            {"question_identifier": "q", "replicate": 0, "weak_model_key": "a",
             "message_text": "Final answer: 3", "error_status": "success"},
            {"question_identifier": "q", "replicate": 0, "weak_model_key": "b",
             "message_text": "", "error_status": "failure"},
            {"question_identifier": "q", "replicate": 0, "weak_model_key": "c",
             "message_text": "   ", "error_status": "success"},
        ])
        assert len(_index_honest(bank)[("q", 0)]) == 1

    def test_run_exits_incomplete_when_calls_failed(self):
        import run_all
        assert run_all._exit_status() == 0
        call_guard.record_failure("r0", "u", None)
        assert run_all._exit_status() == run_all.EXIT_INCOMPLETE


# ── 1: every condition sees the same personas ──────────────────────────────
def _persona(qid, i, target):
    return {"persona_identifier": f"{qid}_persona_{i}", "question_identifier": qid,
            "persona_variant_index": i, "assigned_wrong_answer_letter_or_value": target,
            "generated_persona_text": f"Because {i}.\nFinal answer: {target}",
            "generator_model_name": "llama"}


def _pools(n=5, hedge_missing=()):
    wrong = {"q1": [_persona("q1", i, t) for i, t in enumerate("BCBDE"[:n])]}
    hedged = {"q1": [{**p, "generated_persona_text": "I think maybe " + p["generated_persona_text"]}
                     for p in wrong["q1"] if p["persona_variant_index"] not in hedge_missing]}
    return PersonaPools(anchored_wrong=wrong, anchored_correct={},
                        anchored_hedged=hedged, honest_bank={})


def _ids(condition, focal="f", replicate=0, pools=None):
    peers, diag = build_peers(condition={**condition, "_name": "x"},
                              question={"question_identifier": "q1"},
                              replicate=replicate, pools=pools or _pools(),
                              weak_specs={"w": {"paper_name": "W"}},
                              master_seed=7, focal_key=focal)
    return [p.persona_identifier for p in peers], peers, diag


class TestMatchedPersonas:
    WR = {"peer_source": "anchored_wrong", "n_peers": 2}

    def test_wr_w_sf_share_personas(self):
        wr = _ids(self.WR)[0]
        assert _ids({"peer_source": "bare_answer", "n_peers": 2})[0] == wr
        assert _ids({**self.WR, "source_framing": "anonymous_candidate"})[0] == wr

    def test_wrh_is_the_hedged_rewrite_of_wr(self):
        wr_ids, wr_peers, _ = _ids(self.WR)
        wrh_ids, wrh_peers, _ = _ids({"peer_source": "anchored_hedged", "n_peers": 2})
        assert wrh_ids == wr_ids
        assert all(h.text.startswith("I think maybe") for h in wrh_peers)
        assert [h.assigned_target for h in wrh_peers] == [w.assigned_target for w in wr_peers]
        assert all(h.anchor_mode == "hedged" for h in wrh_peers)

    def test_wrh_skips_rather_than_substituting_when_a_rewrite_is_missing(self):
        wr_ids = _ids(self.WR)[0]
        missing = int(wr_ids[0].rsplit("_", 1)[1])
        ids, _, diag = _ids({"peer_source": "anchored_hedged", "n_peers": 2},
                            pools=_pools(hedge_missing={missing}))
        assert ids == [] and diag["skipped_reason"].startswith("anchored_hedged_infeasible")

    def test_every_focal_model_sees_the_same_personas(self):
        assert _ids(self.WR, focal="a")[0] == _ids(self.WR, focal="b")[0]

    def test_replicates_still_differ(self):
        draws = {tuple(_ids(self.WR, replicate=r)[0]) for r in range(10)}
        assert len(draws) > 1

    def test_dose_conditions_are_nested(self):
        pools = _pools()
        one = _ids({"peer_source": "anchored_wrong", "n_peers": 1}, pools=pools)[0]
        two = _ids(self.WR, pools=pools)[0]
        four = _ids({"peer_source": "anchored_wrong", "n_peers": 4}, pools=pools)[0]
        assert one == two[:1] and two == four[:2]

    def test_order_is_the_same_across_conditions(self):
        from src.contexts import order_peers
        _, wr, _ = _ids(self.WR)
        _, wrh, _ = _ids({"peer_source": "anchored_hedged", "n_peers": 2})
        a = [p.persona_identifier for p in order_peers(wr, 7, "q1", 0)]
        b = [p.persona_identifier for p in order_peers(wrh, 7, "q1", 0)]
        assert a == b


# ── 5: adoption against any wrong target the model saw ─────────────────────
class TestAdoption:

    def _pools(self):
        wrong = {"q1": [_persona("q1", 0, "7"), _persona("q1", 1, "9")]}
        return PersonaPools(anchored_wrong=wrong, anchored_correct={},
                            anchored_hedged={}, honest_bank={})

    def test_adopting_either_disagreeing_target_counts(self, tmp_path):
        class Says9(StubAgent):
            def generate_response(self, *a, **k):
                return AgentResponse(raw_text_output="Final answer: 9",
                                     model_name_returned_by_provider="stub-model",
                                     error_status="success")
        rows = _run(tmp_path, Says9(), Checkpoint(tmp_path / "ck.parquet"),
                    condition={"peer_source": "anchored_wrong", "n_peers": 2},
                    pools=self._pools())
        assert bool(rows["adopted_peer_target"].iloc[0])

    def test_wrong_targets_exclude_correct_anchored_peers(self):
        from src.revision_runner import _wrong_targets
        peers = [PeerMessage(display_name="a", text="", anchor_mode="wrong", assigned_target="7"),
                 PeerMessage(display_name="b", text="", anchor_mode="correct", assigned_target="4")]
        assert _wrong_targets(peers) == ["7"]


# ── 6: answer extraction ──────────────────────────────────────────────────
class TestExtraction:
    OPTIONS = ["1", "-42", "3.5", "7"]

    def test_option_value_maps_to_its_letter(self):
        assert extract_answer("Final answer: -42", "q", "", None,
                              answer_options=self.OPTIONS) == ("B", "regex_success+value_to_letter")

    def test_unmatched_value_is_left_alone(self):
        assert extract_answer("Final answer: 9", "q", "", None,
                              answer_options=self.OPTIONS)[0] == "9"

    def test_numeric_questions_are_untouched(self):
        assert extract_answer("Final answer: 12", "q", "", None)[0] == "12"

    @pytest.mark.parametrize("text", ["Final answer: I think it is B",
                                      "The answer is A bit unclear"])
    def test_pronoun_is_not_an_option(self, text):
        assert extract_answer_regex(text) is None

    @pytest.mark.parametrize("text,letter", [("Final answer: I", "I"),
                                             ("Final answer: A", "A"),
                                             ("Final answer: B because x", "B"),
                                             ("Final answer: A.", "A")])
    def test_real_letters_still_parse(self, text, letter):
        assert extract_answer_regex(text) == letter


# ── 3: solo accuracy on the main pool only ────────────────────────────────
def test_solo_accuracy_ignores_gsm_symbolic_round0():
    cache = pd.DataFrame({
        "focal_key": ["m"] * 4,
        "question_identifier": ["mmlu_1", "mmlu_2", "gsmsym_1", "gsmorig_1"],
        "is_correct": [True, True, False, False]})
    assert solo_accuracy_by_focal(cache) == {"m": 1.0}


def test_protocol_a_gradient_reads_protocol_a_first_answers():
    from analysis.contrasts import protocol_solo_accuracy
    registry = pd.DataFrame({
        "protocol": ["A", "A", "B"], "dataset_scope": ["main300"] * 3,
        "focal_key": ["m", "m", "m"], "r0_is_correct": [True, False, True]})
    assert protocol_solo_accuracy(registry, "A") == {"m": 0.5}


# ── 2: contrasts on matched units ─────────────────────────────────────────
def test_contrast_uses_only_units_in_both_arms():
    from analysis.contrasts import paired_condition_contrast
    rows = []
    for model in ["a", "b", "c"]:
        for q in range(12):
            for condition in ["WR", "G"] if model == "a" else ["WR"]:
                correct = condition == "G" or model != "a"
                rows.append({"protocol": "B", "dataset_scope": "main300",
                             "condition": condition, "focal_key": model,
                             "question_identifier": f"q{q}", "replicate": 0,
                             "is_correct": correct, "r0_is_correct": True})
    result = paired_condition_contrast(pd.DataFrame(rows), "t", "accuracy_delta",
                                       "WR", "G", n_resamples=200)
    # Model a only: WR 0% vs G 100%. Pooling b and c into WR gave -33 points.
    assert result.estimate == pytest.approx(-1.0)
    assert result.extra["models_compared"] == "a"


# ── 7: X6 never verifies a model with itself ──────────────────────────────
def test_x6_excludes_a_focal_that_is_the_verifier():
    from run_all import x6_focal_keys
    focal = {"deepseek": {"model_slug": "ds", "in_x2": True},
             "qwen": {"model_slug": "qwen/qwen-2.5-72b-instruct", "in_x2": True},
             "gemma": {"model_slug": "g", "in_x2": False}}
    keys = x6_focal_keys({"focal": "TIER_X2"}, focal,
                         {"model_slug": "qwen/qwen-2.5-72b-instruct"})
    assert keys == ["deepseek"]


# ── 8: the gate counts each message once ─────────────────────────────────
def test_gate_counts_each_distinct_message_once():
    from analysis.gate import distinct_messages
    log = pd.DataFrame({
        "unit_id": [f"u{i}" for i in range(8)] + ["u8", "u8"],
        "peer_source": ["honest"] * 10,
        "peer_text": ["m1"] * 8 + ["m2", "m2"],
        "shown_to_focal": [True] * 8 + [False, True]})
    assert sorted(distinct_messages(log)["peer_text"]) == ["m1", "m2"]


# ── 10: rounds, budget, dose ──────────────────────────────────────────────
def test_cell_metrics_keep_rounds_apart():
    from analysis.registry import cell_metrics
    frame = pd.DataFrame({
        "protocol": "B", "dataset_scope": "main300", "condition": "WR_rounds2",
        "focal_key": "m", "question_identifier": ["q1", "q1"], "replicate": 0,
        "round_index": [1, 2], "is_correct": [True, False],
        "r0_is_correct": [True, True], "answer_changed": [False, True],
        "flip_correct_to_incorrect": [False, True],
        "flip_incorrect_to_correct": [False, False],
        "peer_asserted_target": ["7", "7"], "adopted_peer_target": [False, False],
        "total_input_tokens": 1, "total_output_tokens": 1})
    metrics = cell_metrics(frame)
    assert sorted(metrics["round_index"]) == [1, 2]


def test_budget_table_weights_tasks_by_questions_and_bills_e_four_calls():
    from analysis.voting import budget_matched_comparison
    voting = pd.DataFrame({
        "focal_key": ["m", "m"], "source_dataset": ["big", "small"],
        "single_sample_accuracy": [0.9, 0.1], "plurality_vote_accuracy": [0.9, 0.1],
        "n_replicates": [3, 3], "n_questions": [90, 10]})
    metrics = pd.DataFrame({
        "protocol": "B", "dataset_scope": "main300", "condition": ["E", "WR"],
        "focal_key": "m", "round_index": 1, "accuracy": [0.5, 0.5]})
    table = budget_matched_comparison(voting, metrics).set_index("strategy")
    assert table.loc["single_sample", "accuracy"] == pytest.approx(0.82)
    assert table.loc["debate:E", "focal_calls"] == 4
    assert table.loc["debate:WR", "focal_calls"] == 2


def test_dose_points_use_the_same_units():
    from analysis.make_figures import dose_response_table
    rows = []
    for condition in ["R", "WR1", "WR", "WR4"]:
        n_questions = 3 if condition in ("R", "WR") else 1
        for q in range(n_questions):
            rows.append({"protocol": "B", "dataset_scope": "main300",
                         "condition": condition, "focal_key": "m",
                         "question_identifier": f"q{q}", "replicate": 0,
                         "round_index": 1, "is_correct": True,
                         "r0_is_correct": True})
    table = dose_response_table(pd.DataFrame(rows))
    assert set(table["n_units"]) == {1}


# ── smoke-run findings: DeepSeek thinking, truncation ──────────────────────
def test_provider_extra_body_reaches_the_request_and_finish_reason_is_kept():
    """deepseek-v4-flash thinks unless told not to; its answers came back empty."""
    from types import SimpleNamespace
    from src.agent_wrappers.openai_compatible_agent import build_agent_from_config

    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Final answer: B"),
                                     finish_reason="length")],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7),
            model="deepseek-flash")

    import os
    os.environ["_TEST_DS_KEY"] = "k"
    providers = {"deepseek": {"base_url_env": "_UNSET", "base_url_default": "http://x",
                              "api_key_envs": ["_TEST_DS_KEY"],
                              "extra_body": {"thinking": {"type": "disabled"}}}}
    agent = build_agent_from_config("ds", "deepseek", "deepseek-v4-flash", providers)
    agent._create_client = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    response = agent.generate_response("s", "u", 0.7, 2048)
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}
    assert sent["max_tokens"] == 2048
    assert response.finish_reason == "length"


def test_the_shipped_config_disables_deepseek_thinking_and_raises_the_caps():
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    models = yaml.safe_load((root / "config/models.yaml").read_text(encoding="utf-8"))
    experiment = yaml.safe_load((root / "config/experiment.yaml").read_text(encoding="utf-8"))
    assert models["providers"]["deepseek"]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert models["request_defaults"]["focal_max_output_tokens"] >= 2048
    assert experiment["r0_cache"]["max_output_tokens"] == \
        models["request_defaults"]["focal_max_output_tokens"]
    assert models["verifier_agent"]["max_output_tokens"] >= 2048


# ── smoke-run finding: answers stated as an option's value ────────────────
class TestOptionValueAnswers:
    OPTS = '["[1, 2]", "[3, 2]", "[0, 7]"]'

    def test_final_line_value_maps_to_its_option_letter(self):
        assert extract_answer("work\nFinal answer: [3, 2]\nConfidence: 90", "q", "",
                              None, answer_options=self.OPTS) == \
            ("B", "final_line+value_to_letter")

    def test_emphasis_is_ignored_and_no_option_is_invented(self):
        assert extract_answer("Final answer: **[0, 7]**", "q", "", None,
                              answer_options=self.OPTS)[0] == "C"
        assert extract_answer("Final answer: [9, 9]", "q", "", None,
                              answer_options=self.OPTS)[0] is None

    def test_judge_grounding_reads_the_phase3_option_listing(self):
        """Phase 3 sends 'A. foo' lines; the guard parsed only 'A) foo'."""
        from src.agent_wrappers.judge_agent import JudgeCascade
        from src.extraction import options_to_string
        listing = options_to_string(self.OPTS)
        assert JudgeCascade._option_text_for_letter("B", listing) == "[3, 2]"
        assert JudgeCascade._is_grounded("B", "Final answer: [3, 2]", listing)
        assert not JudgeCascade._is_grounded("C", "Final answer: [3, 2]", listing)

    def test_options_from_a_numpy_array_reach_the_judge(self):
        import numpy as np
        from src.extraction import options_to_string
        assert options_to_string(np.array(["x", "y"], dtype=object)) == "A. x\nB. y"


def test_a_model_can_carry_its_own_retry_policy():
    """Gemma-3-4B has one upstream on OpenRouter; its 429 episodes last minutes."""
    from pathlib import Path
    import yaml
    from src.agents import _build_one
    root = Path(__file__).resolve().parent.parent
    models = yaml.safe_load((root / "config/models.yaml").read_text(encoding="utf-8"))
    import os
    os.environ.setdefault("OPENROUTER_API_KEY_1", "k")
    os.environ.setdefault("HUGGINGFACE_TOKEN", "k")
    spec = models["focal_agents"]["sweep_gemma_3_4b_focal"]
    agent = _build_one("g", spec, models["providers"], models["request_defaults"],
                       allow_fallback=True, pin_model=False)
    hf, openrouter = agent._chain
    # One fast try on Hugging Face, then patient retries on OpenRouter.
    assert (hf.provider, hf.max_retries) == ("hf_router", 0)
    assert hf.model_name == "google/gemma-3-4b-it:deepinfra"
    assert (openrouter.provider, openrouter.max_retries) == ("openrouter", 10)
    assert openrouter.retry_backoff[-1] == 60


def test_a_pinned_chain_with_no_serving_link_counts_as_a_failed_call(tmp_path):
    """PinnedFallbackAgent reports 'snapshot_unavailable'; it must not become data."""
    class Unavailable(StubAgent):
        def generate_response(self, *a, **k):
            return AgentResponse(raw_text_output="", error_status="snapshot_unavailable",
                                 model_name_returned_by_provider="stub-model")
    checkpoint = Checkpoint(tmp_path / "ck.parquet")
    rows = _run(tmp_path, Unavailable(), checkpoint)
    assert rows.empty and call_guard.failure_counts() == {"revision": 1}
    frame = pd.DataFrame({"error_status": ["success", "snapshot_unavailable", "failure"]})
    assert len(call_guard.drop_failed_calls(frame, "t")) == 1


def test_the_serving_route_is_recorded_per_row(tmp_path):
    class Routed(StubAgent):
        def generate_response(self, *a, **k):
            return AgentResponse(raw_text_output="Final answer: 4", served_route="hf_router",
                                 model_name_returned_by_provider="stub-model",
                                 error_status="success", finish_reason="stop")
    rows = _run(tmp_path, Routed(), Checkpoint(tmp_path / "ck.parquet"))
    assert rows["served_route"].iloc[0] == "hf_router"
    assert rows["finish_reason"].iloc[0] == "stop"
