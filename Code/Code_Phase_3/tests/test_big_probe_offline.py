"""
test_big_probe_offline.py — verify the 70B probe before renting a GPU.

Everything here runs without vLLM, without CUDA and without the 132 GB
checkpoint: candidate construction, exact tokenised scoring, target fixing,
peer assembly and the difference-in-differences. A fault found here costs
nothing; the same fault found on Vast.ai costs GPU-hours, because the failure
would land *after* the model is loaded and billing has started.

The tokenizer is stubbed with a deterministic character-level encoder that
reproduces the property that matters: single-token candidates versus
multi-token ones, and first-token collisions between numeric answers.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from GPU_Only.src.corrected_probe import (
    ExactCandidateScorer,
    build_probe_targets,
    difference_in_differences,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "Vast_AI_Big_Model_Run"))
from run_big_probe import build_peers_for, candidates_for  # noqa: E402


class StubTokenizer:
    """
    Character-level encoder: each character is its own token.

    That makes a single letter ("A") one token and a multi-digit number
    ("150") three, and it makes "12" and "150" share a first token — the exact
    collision the Phase-2 first-character matcher merged silently.
    """

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text.strip()]


@pytest.fixture
def scorer():
    return ExactCandidateScorer(StubTokenizer())


class TestCandidateScoring:

    def test_mcq_letters_are_single_token(self, scorer):
        single = scorer.single_token_candidates(["A", "B", "C", "D"])
        assert set(single) == {"A", "B", "C", "D"}

    def test_numeric_prefix_collisions_are_reported_not_merged(self, scorer):
        # "12", "150" and "1" all begin with the token for "1". The released
        # probe silently merged them; this one must name them.
        collisions = scorer.collision_report(["1", "12", "150", "9"])
        assert collisions, "a first-token collision must be reported"
        colliding = {c for group in collisions.items() for c in (group[0], *group[1])}
        assert {"1", "12", "150"} <= colliding
        assert "9" not in colliding

    def test_off_candidate_mass_is_retained(self, scorer):
        # The model puts 0.5 on "A", 0.2 on "B" and 0.3 somewhere else. The
        # released probe renormalised that 0.3 away; it must survive here.
        logprobs = {
            ord("A"): float(np.log(0.5)),
            ord("B"): float(np.log(0.2)),
            ord("Z"): float(np.log(0.3)),
        }
        scores, outside = scorer.score_from_first_position(logprobs, ["A", "B"])
        by_candidate = {s.candidate: s.probability for s in scores}
        assert by_candidate["A"] == pytest.approx(0.5, abs=1e-6)
        assert by_candidate["B"] == pytest.approx(0.2, abs=1e-6)
        assert outside == pytest.approx(0.3, abs=1e-6), (
            "mass outside the candidate set must be reported, not normalised away"
        )

    def test_probabilities_are_not_renormalised(self, scorer):
        logprobs = {ord("A"): float(np.log(0.4)), ord("B"): float(np.log(0.1))}
        scores, _ = scorer.score_from_first_position(logprobs, ["A", "B"])
        total = sum(s.probability for s in scores)
        assert total == pytest.approx(0.5, abs=1e-6), (
            "candidate probabilities must stay absolute, not sum to 1"
        )

    def test_candidate_below_top_k_is_zero_not_missing(self, scorer):
        scores, _ = scorer.score_from_first_position(
            {ord("A"): float(np.log(0.9))}, ["A", "B"])
        by_candidate = {s.candidate: s for s in scores}
        assert by_candidate["B"].probability == 0.0
        assert by_candidate["B"].scoring_method == "below_top_k"

    def test_multi_token_candidate_is_flagged_for_teacher_forcing(self, scorer):
        scores, _ = scorer.score_from_first_position(
            {ord("1"): float(np.log(0.8))}, ["150"])
        assert scores[0].scoring_method == "needs_teacher_forcing"
        assert scores[0].n_tokens == 3


class TestTargets:

    @staticmethod
    def _inputs():
        questions = [{"question_identifier": f"q{i}"} for i in range(4)]
        personas = {
            f"q{i}": [{"assigned_wrong_answer_letter_or_value": c} for c in "BCD"]
            for i in range(4)
        }
        return questions, personas

    def test_one_target_per_question_replicate(self):
        questions, personas = self._inputs()
        targets = build_probe_targets(questions, personas, 3, 20260502)
        assert len(targets) == 4 * 3
        assert all(v in {"B", "C", "D"} for v in targets.values())

    def test_targets_are_deterministic(self):
        questions, personas = self._inputs()
        a = build_probe_targets(questions, personas, 3, 20260502)
        b = build_probe_targets(questions, personas, 3, 20260502)
        assert a == b, "the target must be reproducible across processes"

    def test_question_without_personas_is_skipped_not_guessed(self):
        questions, personas = self._inputs()
        personas.pop("q0")
        targets = build_probe_targets(questions, personas, 2, 20260502)
        assert all(q != "q0" for q, _ in targets)


class TestPeerAssembly:

    QUESTION = {"question_identifier": "q0", "correct_answer": "A",
                "answer_options": json.dumps(["a", "b", "c"])}
    PERSONAS = [
        {"assigned_wrong_answer_letter_or_value": "B",
         "generated_persona_text": "It is B.\nFinal answer: B",
         "generator_model_name": "llama"},
        {"assigned_wrong_answer_letter_or_value": "C",
         "generated_persona_text": "It is C.\nFinal answer: C",
         "generator_model_name": "llama"},
    ]

    def test_baseline_has_no_peers(self):
        assert build_peers_for("R", self.QUESTION, "B", self.PERSONAS, []) == []

    def test_homogeneous_uses_self_samples(self):
        peers = build_peers_for("E", self.QUESTION, "B", self.PERSONAS,
                                ["mine 1\nFinal answer: A", "mine 2\nFinal answer: A"])
        assert len(peers) == 2
        assert all(p.anchor_mode == "self" for p in peers)

    def test_treatment_prefers_peers_arguing_the_fixed_target(self):
        peers = build_peers_for("WR", self.QUESTION, "B", self.PERSONAS, [])
        assert len(peers) == 2
        assert peers[0].assigned_target == "B", (
            "the peers must argue the target the contrast tracks"
        )

    def test_generator_provenance_is_carried(self):
        peers = build_peers_for("WR", self.QUESTION, "B", self.PERSONAS, [])
        assert all(p.message_generator_model == "llama" for p in peers)


class TestCandidatesFor:

    def test_mcq_returns_option_letters(self):
        q = {"answer_options": json.dumps(["x", "y", "z"]), "correct_answer": "A"}
        assert candidates_for(q) == ["A", "B", "C"]

    def test_numeric_returns_correct_plus_wrong_pool(self):
        q = {"answer_options": None, "correct_answer": "42",
             "wrong_answer_pool": json.dumps(["84", "21"])}
        assert candidates_for(q) == ["42", "84", "21"]


class TestDifferenceInDifferences:

    @staticmethod
    def _frame(shift_wr: float, shift_e: float, n: int = 30):
        """
        Independent noise per condition.

        Sharing one noise draw across both conditions would make every paired
        difference identical, and a percentile bootstrap over a constant is
        degenerate — it returns a zero-width interval whatever the code does.
        Real trials never look like that, so the fixture must not either.
        """
        rows = []
        rng = np.random.default_rng(7)
        for i in range(n):
            for condition, shift in (("WR", shift_wr), ("E", shift_e)):
                rows.append({
                    "question_identifier": f"q{i}", "replicate": 0,
                    "condition": condition,
                    "delta_prob_mass_toward_target": shift + rng.normal(0, 0.01),
                    "round0_answer": "A", "round1_answer": "A",
                })
        return pd.DataFrame(rows)

    def test_detects_a_real_shift(self):
        result = difference_in_differences(self._frame(0.08, -0.02))
        assert result["estimate"] == pytest.approx(0.10, abs=0.02)
        assert result["ci_low"] > 0
        assert result["p_value"] < 0.01
        assert result["n_pairs"] == 30

    def test_reports_no_shift_when_there_is_none(self):
        result = difference_in_differences(self._frame(0.02, 0.02))
        assert abs(result["estimate"]) < 0.01
        assert result["ci_low"] < 0 < result["ci_high"]

    def test_same_answer_subset_is_reported_separately(self):
        # The claim that matters: movement that is not an artefact of a flip.
        result = difference_in_differences(self._frame(0.08, -0.02))
        assert result["n_pairs_stated_answer_unchanged"] == 30
        assert result["estimate_stated_answer_unchanged"] == pytest.approx(
            result["estimate"], abs=1e-9)

    def test_empty_input_does_not_raise(self):
        empty = pd.DataFrame(columns=["question_identifier", "replicate",
                                      "condition", "delta_prob_mass_toward_target",
                                      "round0_answer", "round1_answer"])
        assert difference_in_differences(empty)["n_pairs"] == 0


class RealisticTokenizer:
    """
    Closer to a BPE tokenizer than the character stub: a leading space plus a
    short token is ONE token (as Llama encodes " A" or " 42"), while a longer
    number splits. That is the distinction the scoring path turns on.
    """

    SINGLE = {" A", " B", " C", " D", " 1", " 9", " 42"}

    def encode(self, text, add_special_tokens=False):
        if text in self.SINGLE:
            return [hash(text) % 90000 + 1000]
        return [ord(c) for c in text]


class TestTeacherForcedScoring:
    """Multi-token numeric answers must be scored exactly, not by first char."""

    @pytest.fixture
    def scorer(self):
        return ExactCandidateScorer(RealisticTokenizer())

    def test_single_token_candidates_skip_teacher_forcing(self, scorer):
        assert scorer.needs_teacher_forcing(["A", "B", "C"]) == []

    def test_multi_token_numerics_are_flagged(self, scorer):
        # "150" and "12" are multi-token; "42" is single.
        flagged = scorer.needs_teacher_forcing(["42", "150", "12"])
        assert "150" in flagged and "12" in flagged
        assert "42" not in flagged

    def test_continuation_span_locates_the_candidate(self, scorer):
        prefix = "Final answer:"
        start, end = scorer.continuation_span(prefix, "150")
        prefix_len = len(scorer.tokenizer.encode(prefix))
        assert start == prefix_len
        assert end - start == len(scorer.tokenizer.encode(" 150"))

    def test_prompt_logprob_sum_is_the_sequence_logprob(self, scorer):
        class LP:
            def __init__(self, v): self.logprob = v
        # vLLM ids include a BOS at index 0. The candidate is the last 2 tokens.
        ids = [128000, 25, 220, 17]
        table = [None, {25: LP(-0.1)}, {220: LP(-0.5)}, {17: LP(-0.5)}]
        total = scorer.score_from_prompt_logprobs(table, ids, 2)
        assert total == pytest.approx(-1.0)
        assert math.exp(total) == pytest.approx(0.3678794, abs=1e-5)

    def test_different_candidates_get_different_probabilities(self, scorer):
        """THE bug: reading a neighbouring token made every candidate that
        shared a leading space score identically (all 0.666)."""
        class LP:
            def __init__(self, v): self.logprob = v
        results = {}
        for digit_id, lp in ((17, -0.5), (16, -0.9), (19, -1.3)):
            ids = [128000, 25, 220, digit_id]
            table = [None, {25: LP(-0.02)}, {220: LP(-0.02)}, {digit_id: LP(lp)}]
            results[digit_id] = scorer.score_from_prompt_logprobs(table, ids, 2)
        assert len(set(results.values())) == 3, (
            "candidates differing only in their final token must not collapse "
            "to one value"
        )

    def test_probabilities_sum_to_at_most_one(self, scorer):
        """The symptom that exposed the bug: the sum was 3.28, not <= 1."""
        class LP:
            def __init__(self, v): self.logprob = v
        total = 0.0
        for digit_id, lp in ((17, -1.2), (16, -1.6), (19, -2.0), (18, -2.4)):
            ids = [128000, 25, 220, digit_id]
            table = [None, {25: LP(-0.05)}, {220: LP(-0.05)}, {digit_id: LP(lp)}]
            total += math.exp(scorer.score_from_prompt_logprobs(table, ids, 2))
        assert total <= 1.0 + 1e-9, f"candidate probabilities sum to {total}"

    def test_missing_logprob_yields_none_not_zero(self, scorer):
        # Unavailable must not be silently scored as probability zero.
        ids = [128000, 25, 220, 17]
        assert scorer.score_from_prompt_logprobs(
            [None, {}, {}, {}], ids, 2) is None
        assert scorer.score_from_prompt_logprobs([], [], 2) is None

    def test_wrong_token_at_position_is_not_silently_accepted(self, scorer):
        """If the table holds a different token than vLLM's id, fail loudly."""
        class LP:
            def __init__(self, v): self.logprob = v
        ids = [128000, 25, 220, 17]
        table = [None, {25: LP(-0.1)}, {220: LP(-0.5)}, {99999: LP(-0.5)}]
        assert scorer.score_from_prompt_logprobs(table, ids, 2) is None

    def test_merge_keeps_probabilities_absolute(self, scorer):
        scores, _ = scorer.score_from_first_position(
            {scorer.tokenizer.encode(" 42")[0]: math.log(0.30)}, ["42", "150"])
        merged, outside = scorer.merge_teacher_forced(scores, {"150": math.log(0.20)})
        by_candidate = {s.candidate: s for s in merged}
        assert by_candidate["42"].probability == pytest.approx(0.30, abs=1e-6)
        assert by_candidate["150"].probability == pytest.approx(0.20, abs=1e-6)
        assert by_candidate["150"].scoring_method == "teacher_forced"
        # Absolute, not renormalised: they must NOT sum to 1.
        assert outside == pytest.approx(0.50, abs=1e-6)

    def test_unavailable_candidate_is_marked_not_zeroed(self, scorer):
        scores, _ = scorer.score_from_first_position({}, ["150"])
        merged, outside = scorer.merge_teacher_forced(scores, {"150": None})
        assert merged[0].probability is None
        assert merged[0].scoring_method == "teacher_forcing_unavailable"
        assert outside == pytest.approx(1.0)

    def test_collision_that_broke_the_released_probe(self, scorer):
        # "1", "12", "150" share a first character. The released probe merged
        # them; scoring each full sequence separates them.
        forced = {"12": math.log(0.10), "150": math.log(0.05)}
        scores, _ = scorer.score_from_first_position(
            {scorer.tokenizer.encode(" 1")[0]: math.log(0.40)}, ["1", "12", "150"])
        merged, _ = scorer.merge_teacher_forced(scores, forced)
        probs = {s.candidate: s.probability for s in merged}
        assert probs["1"] == pytest.approx(0.40, abs=1e-6)
        assert probs["12"] == pytest.approx(0.10, abs=1e-6)
        assert probs["150"] == pytest.approx(0.05, abs=1e-6)
        assert len({probs["1"], probs["12"], probs["150"]}) == 3


class TestPerTaskContrast:
    """MCQ and GSM8K must be reported separately, never pooled."""

    @staticmethod
    def _mixed_frame():
        import numpy as np
        rng = np.random.default_rng(11)
        rows = []
        # MCQ: mass on a single option letter, O(0.1).
        for i in range(25):
            for cond, shift in (("WR", 0.09), ("E", -0.01)):
                rows.append({
                    "question_identifier": f"mmlupro_{i:04d}", "replicate": 0,
                    "condition": cond, "source_dataset": "mmlu_pro",
                    "delta_prob_mass_toward_target": shift + rng.normal(0, 0.01),
                    "round0_answer": "A", "round1_answer": "A"})
        # GSM8K: mass on an exact digit string, ~100x smaller.
        for i in range(25):
            for cond, shift in (("WR", 0.0008), ("E", -0.0001)):
                rows.append({
                    "question_identifier": f"gsm8k_{i:04d}", "replicate": 0,
                    "condition": cond, "source_dataset": "gsm8k",
                    "delta_prob_mass_toward_target": shift + rng.normal(0, 0.0001),
                    "round0_answer": "1", "round1_answer": "1"})
        return pd.DataFrame(rows)

    def test_one_row_per_task_family_plus_pooled(self):
        from GPU_Only.src.corrected_probe import difference_in_differences_by_task
        rows = difference_in_differences_by_task(self._mixed_frame())
        datasets = {r["source_dataset"] for r in rows}
        assert datasets == {"mmlu_pro", "gsm8k", "POOLED"}

    def test_pooled_row_is_flagged_not_to_report(self):
        from GPU_Only.src.corrected_probe import difference_in_differences_by_task
        rows = difference_in_differences_by_task(self._mixed_frame())
        pooled = [r for r in rows if r["source_dataset"] == "POOLED"][0]
        assert pooled["report"] is False
        assert "why_not" in pooled
        assert all(r["report"] for r in rows if r["source_dataset"] != "POOLED")

    def test_each_family_recovers_its_own_effect(self):
        from GPU_Only.src.corrected_probe import difference_in_differences_by_task
        rows = {r["source_dataset"]: r
                for r in difference_in_differences_by_task(self._mixed_frame())}
        assert rows["mmlu_pro"]["estimate"] == pytest.approx(0.10, abs=0.01)
        assert rows["gsm8k"]["estimate"] == pytest.approx(0.0009, abs=0.0003)

    def test_pooling_would_have_buried_the_small_family(self):
        """The reason the split exists: the pooled mean is ~half the MCQ
        effect and ~55x the GSM8K effect, representing neither."""
        from GPU_Only.src.corrected_probe import difference_in_differences_by_task
        rows = {r["source_dataset"]: r
                for r in difference_in_differences_by_task(self._mixed_frame())}
        pooled = rows["POOLED"]["estimate"]
        gsm = rows["gsm8k"]["estimate"]
        assert pooled > 20 * gsm, (
            "pooled estimate should be dominated by the larger-scale family"
        )

    def test_single_family_input_still_works(self):
        from GPU_Only.src.corrected_probe import difference_in_differences_by_task
        frame = self._mixed_frame()
        rows = difference_in_differences_by_task(frame[frame.source_dataset == "gsm8k"])
        assert {r["source_dataset"] for r in rows} == {"gsm8k", "POOLED"}


class TestUnchangedSubsetExcludesUnparsed:
    """An unparsed answer must not count as 'the answer stayed the same'."""

    @staticmethod
    def _frame(r0_t, r1_t, r0_b, r1_b, n=20):
        rows = []
        for i in range(n):
            rows.append({"question_identifier": f"q{i}", "replicate": 0,
                         "condition": "WR", "source_dataset": "gsm8k",
                         "delta_prob_mass_toward_target": 0.05,
                         "round0_answer": r0_t, "round1_answer": r1_t})
            rows.append({"question_identifier": f"q{i}", "replicate": 0,
                         "condition": "E", "source_dataset": "gsm8k",
                         "delta_prob_mass_toward_target": -0.01,
                         "round0_answer": r0_b, "round1_answer": r1_b})
        return pd.DataFrame(rows)

    def test_both_empty_is_not_counted_as_unchanged(self):
        r = difference_in_differences(self._frame("", "", "", ""))
        assert r["n_pairs_stated_answer_unchanged"] == 0, (
            "two unparsed answers compare equal but are not evidence the "
            "answer held"
        )

    def test_genuinely_unchanged_is_counted(self):
        r = difference_in_differences(self._frame("A", "A", "A", "A"))
        assert r["n_pairs_stated_answer_unchanged"] == 20

    def test_one_side_unparsed_is_excluded(self):
        r = difference_in_differences(self._frame("A", "", "A", "A"))
        assert r["n_pairs_stated_answer_unchanged"] == 0

    def test_changed_answer_is_excluded(self):
        r = difference_in_differences(self._frame("A", "B", "A", "A"))
        assert r["n_pairs_stated_answer_unchanged"] == 0
