"""
test_repairs.py — the four defects that produced wrong numbers, each pinned.

Every test runs offline. Together they are the evidence that the Phase-3
repairs do what the revision claims, which is the part a reviewer can check
without re-running a single API call.

    python3 -m pytest tests/ -q
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.gate import retention_gap
from analysis.metrics import harmful, joint_loss, summarise
from src.contexts import (
    PeerMessage,
    apply_confidence_filter,
    build_revision_prompt,
    build_round0_prompt,
)
from src.extraction import answers_equal, grade, normalise_answer
from src.hedged_personas import validate_hedged
from src.seeding import derive_seed, stable_hash, unit_id


# ──────────────────────────────────────────────────────────────────────────
# DEFECT 1 — process-dependent seeding
# ──────────────────────────────────────────────────────────────────────────

class TestSeeding:
    """`hash()` on str is salted per process; crc32 is not."""

    def test_stable_hash_is_deterministic_in_process(self):
        assert stable_hash("mmlupro_0000", 3) == stable_hash("mmlupro_0000", 3)

    def test_stable_hash_survives_a_new_interpreter(self):
        # The actual regression: Phase 1/2 seeds changed between processes
        # because PYTHONHASHSEED is randomised. Run the same derivation in two
        # fresh interpreters with different salts and require agreement.
        code = (
            "import sys; sys.path.insert(0, r'{root}');"
            "from src.seeding import derive_seed;"
            "print(derive_seed(20260502, 'mmlupro_0000', 2))"
        ).format(root=Path(__file__).resolve().parent.parent)

        outputs = []
        for salt in ("0", "1"):
            env = {**dict(__import__("os").environ), "PYTHONHASHSEED": salt}
            result = subprocess.run([sys.executable, "-c", code], env=env,
                                    capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stderr
            outputs.append(result.stdout.strip())
        assert outputs[0] == outputs[1], (
            "derived seed changed across processes — the Phase-1/2 bug is back"
        )

    def test_builtin_hash_would_have_failed(self):
        # Documents why the repair was needed: builtin hash() of the same
        # string differs across interpreters with different salts.
        code = "print(hash('mmlupro_0000'))"
        outputs = []
        for salt in ("0", "1"):
            env = {**dict(__import__("os").environ), "PYTHONHASHSEED": salt}
            result = subprocess.run([sys.executable, "-c", code], env=env,
                                    capture_output=True, text=True, timeout=60)
            outputs.append(result.stdout.strip())
        assert outputs[0] != outputs[1]

    def test_unit_ids_are_unique_per_cell(self):
        ids = {
            unit_id("B", focal, condition, "q1", replicate, 1)
            for focal in ("a", "b")
            for condition in ("R", "WR")
            for replicate in (0, 1, 2)
        }
        assert len(ids) == 2 * 2 * 3

    def test_derived_seeds_are_in_range(self):
        for parts in [("q", 0), ("q", 1), ("zzz", 99)]:
            seed = derive_seed(20260502, *parts)
            assert 0 <= seed < 2**31 - 1


# ──────────────────────────────────────────────────────────────────────────
# DEFECT 2 — flip-rate denominator
# ──────────────────────────────────────────────────────────────────────────

class TestDenominators:
    """
    Phase 2's `_flip_rate()` returned P(A0=1 AND Y=0); the paper defined
    P(Y=0 | A0=1). On real data those differed by a factor of ~1.5.
    """

    @staticmethod
    def _frame():
        # 10 units: 6 initially correct (2 of them end wrong), 4 initially
        # wrong (1 ends correct).
        return pd.DataFrame({
            "question_identifier": [f"q{i}" for i in range(10)],
            "r0_is_correct": [True] * 6 + [False] * 4,
            "is_correct": [True, True, True, True, False, False,
                           True, False, False, False],
            "answer_changed": [False] * 10,
            "peer_asserted_target": [None] * 10,
            "adopted_peer_target": [False] * 10,
            "extracted_answer": ["A"] * 10,
            "correct_answer": ["A"] * 10,
        })

    def test_harmful_uses_conditional_denominator(self):
        rate = harmful(self._frame())
        assert rate.denominator == 6, "denominator must be initially-correct units"
        assert rate.numerator == 2
        assert rate.value == pytest.approx(2 / 6)

    def test_joint_loss_differs_from_conditional(self):
        frame = self._frame()
        conditional = harmful(frame).value       # 2/6 = 0.333
        joint = joint_loss(frame).value          # 2/10 = 0.200
        assert conditional != pytest.approx(joint)
        assert joint == pytest.approx(0.2)

    def test_summarise_reports_both_with_denominators(self):
        row = summarise(self._frame(), condition="WR")
        assert row["harmful_revision_denominator"] == 6
        assert row["joint_loss_denominator"] == 10
        assert row["harmful_revision"] != row["joint_loss"]

    def test_empty_denominator_is_nan_not_zero(self):
        frame = self._frame()
        frame["r0_is_correct"] = False
        assert np.isnan(harmful(frame).value)


# ──────────────────────────────────────────────────────────────────────────
# DEFECT 3 — retention-gap sign and single-class substrate
# ──────────────────────────────────────────────────────────────────────────

class TestRetentionGap:
    """
    Phase 2's corrected_gate computed P(loud|wrong) - P(loud|correct) and
    passed when positive — the sign of the HARMFUL case for a retain-high
    filter. It also substituted zero for an absent class, producing a
    spurious gap of 0.99 and verdict "passed" on the wrong-anchored substrate.
    """

    def test_perfect_discrimination_passes(self):
        messages = pd.DataFrame({
            "peer_confidence": [95, 98, 10, 5],
            "peer_is_correct": [True, True, False, False],
        })
        result = retention_gap(messages, "synthetic_perfect", threshold=60)
        assert result.delta_retention == pytest.approx(1.0)
        assert result.auroc_confidence_vs_correct == pytest.approx(1.0)
        assert result.verdict == "passed"

    def test_inverted_discrimination_fails(self):
        # Confident when WRONG: exactly the case a retain-high filter must
        # not be told is good. The old sign convention would have passed it.
        messages = pd.DataFrame({
            "peer_confidence": [10, 5, 95, 98],
            "peer_is_correct": [True, True, False, False],
        })
        result = retention_gap(messages, "synthetic_inverted", threshold=60)
        assert result.delta_retention == pytest.approx(-1.0)
        assert result.verdict == "failed"

    def test_single_class_is_undefined_not_zero(self):
        # The wrong-anchored substrate: every peer is wrong by construction.
        messages = pd.DataFrame({
            "peer_confidence": [95, 98],
            "peer_is_correct": [False, False],
        })
        result = retention_gap(messages, "anchored_wrong", threshold=60)
        assert result.verdict == "undefined_single_class"
        assert result.delta_retention is None
        assert result.auroc_confidence_vs_correct is None

    def test_uninformative_confidence_fails(self):
        # The real finding: weak peers sound equally confident either way.
        messages = pd.DataFrame({
            "peer_confidence": [95, 96, 95, 97],
            "peer_is_correct": [True, True, False, False],
        })
        result = retention_gap(messages, "synthetic_flat", threshold=60)
        assert abs(result.delta_retention) < 0.10
        assert result.verdict == "failed"

    def test_unparseable_accounting_is_reported_both_ways(self):
        # The released C5H case: every peer whose confidence failed to parse
        # happened to be wrong. Dropping them therefore flatters the filter,
        # and the flattery comes from a parsing failure rather than from
        # confidence — so both accountings must be reported.
        messages = pd.DataFrame({
            "peer_confidence": [95, 95, 95, None, None],
            "peer_is_correct": [True, True, False, False, False],
        })
        dropped = retention_gap(messages, "s", unparseable_counts_as="dropped")
        retained = retention_gap(messages, "s", unparseable_counts_as="retained")
        assert dropped.n_unparseable_confidence == 2
        # dropped:  P(ret|correct) = 2/2 = 1; P(ret|wrong) = 1/3; gap = 2/3.
        # retained: every peer is kept, so both rates are 1 and the gap is 0.
        # Scoring unparseable peers as dropped removes only wrong peers here,
        # so the apparent gap rises — from parsing, not from confidence.
        assert dropped.delta_retention == pytest.approx(2 / 3)
        assert retained.delta_retention == pytest.approx(0.0)
        assert dropped.delta_retention > retained.delta_retention


# ──────────────────────────────────────────────────────────────────────────
# DEFECT 4 — unmatched revision prompts (the Protocol B repair)
# ──────────────────────────────────────────────────────────────────────────

class TestProtocolB:
    """
    In Phase 1/2 the debate Round-1 prompt omitted the focal's own Round-0
    text while the re-answer control included it. Protocol B shows it in
    EVERY condition, so the peer block is the only difference.
    """

    QUESTION = "What is 2 + 2?"
    OWN = "I computed 2 + 2 = 4.\nFinal answer: 4\nConfidence: 90"

    def _prompt(self, peers, peer_source, framing="peer_attributed"):
        return build_revision_prompt(
            question_text=self.QUESTION, answer_options=None,
            own_previous_text=self.OWN, peers=peers,
            peer_source=peer_source, source_framing=framing,
        )[1]

    @staticmethod
    def _peers(n=2):
        return [
            PeerMessage(display_name=f"Agent_{i}", text=f"It is {5 + i}.",
                        final_answer=str(5 + i), confidence=95,
                        peer_source="anchored_wrong", anchor_mode="wrong",
                        assigned_target=str(5 + i))
            for i in range(n)
        ]

    def test_own_answer_present_in_every_condition(self):
        for peer_source, peers in [
            ("none", []), ("generic", []), ("bare_answer", self._peers()),
            ("anchored_wrong", self._peers()), ("honest", self._peers()),
            ("self_samples", self._peers()),
        ]:
            prompt = self._prompt(peers, peer_source)
            assert "Your previous response:" in prompt, peer_source
            assert self.OWN in prompt, peer_source

    def test_only_the_context_block_differs(self):
        baseline = self._prompt([], "none")
        treatment = self._prompt(self._peers(), "anchored_wrong")
        # Everything before the peer block, and the closing instruction, match
        # except for the single clause naming the peer responses.
        assert baseline.startswith(f"Question: {self.QUESTION}")
        assert treatment.startswith(f"Question: {self.QUESTION}")
        assert "Reconsider the problem carefully" in baseline
        assert "Reconsider the problem carefully" in treatment
        assert "Other agents' responses:" not in baseline
        assert "Other agents' responses:" in treatment

    def test_generic_challenge_carries_no_answer(self):
        # The condition-G block must name no alternative answer and give no
        # reasoning; otherwise "WR beats G" would not isolate message content.
        from src.contexts import build_context_block

        block = build_context_block([], "generic")
        assert "may be wrong but gave no details" in block
        assert "Final answer" not in block
        assert not any(character.isdigit() for character in block)
        assert block in self._prompt([], "generic")

    def test_bare_answer_condition_has_no_rationale(self):
        prompt = self._prompt(self._peers(), "bare_answer")
        assert "Final answer: 5" in prompt
        assert "It is 5." not in prompt      # the rationale is withheld

    def test_source_framing_keeps_content_drops_attribution(self):
        attributed = self._prompt(self._peers(), "anchored_wrong")
        anonymous = self._prompt(self._peers(), "anchored_wrong",
                                 framing="anonymous_candidate")
        assert "Agent_0 said:" in attributed
        assert "Agent_0 said:" not in anonymous
        assert "Candidate solution 1:" in anonymous
        for peer in self._peers():
            assert peer.text in attributed and peer.text in anonymous

    def test_round0_prompt_matches_the_published_one(self):
        _, user = build_round0_prompt(self.QUESTION)
        assert "Provide your reasoning in 2 to 3 sentences." in user
        assert "Final answer:" in user and "Confidence:" in user


# ──────────────────────────────────────────────────────────────────────────
# Supporting validators
# ──────────────────────────────────────────────────────────────────────────

class TestConfidenceFilter:

    @staticmethod
    def _peer(confidence):
        return PeerMessage(display_name="p", text="t", final_answer="A",
                           confidence=confidence)

    def test_retains_above_threshold(self):
        kept, diagnostics = apply_confidence_filter(
            [self._peer(90), self._peer(30)], threshold=60)
        assert len(kept) == 1
        assert diagnostics["n_peers_dropped"] == 1

    def test_unparseable_dropped_by_default(self):
        kept, _ = apply_confidence_filter([self._peer(None)], threshold=60)
        assert kept == []

    def test_unparseable_can_be_retained_for_the_alternative_accounting(self):
        kept, _ = apply_confidence_filter(
            [self._peer(None)], threshold=60, unparseable_counts_as="retained")
        assert len(kept) == 1


class TestHedgedValidation:

    SOURCE = "The answer is clearly 42 because the rule applies.\nFinal answer: 42"
    CONFIG = {
        "final_answer_must_match_source": True,
        "minimum_hedge_markers": 2,
        "hedge_markers": ["i think", "maybe", "not sure", "might", "possibly"],
        "forbid_new_numbers": True,
        "length_ratio_bounds": [0.6, 1.4],
    }

    def test_accepts_a_proper_hedge(self):
        text = ("I think the answer might be 42, though I am not sure the rule "
                "applies.\nFinal answer: 42")
        passed, reason = validate_hedged(text, self.SOURCE, "42", self.CONFIG)
        assert passed, reason

    def test_rejects_a_changed_answer(self):
        text = "I think it might be 43, maybe.\nFinal answer: 43"
        passed, reason = validate_hedged(text, self.SOURCE, "42", self.CONFIG)
        assert not passed and "answer_changed" in reason

    def test_rejects_insufficient_hedging(self):
        text = "The answer is 42.\nFinal answer: 42"
        passed, reason = validate_hedged(text, self.SOURCE, "42", self.CONFIG)
        assert not passed and "insufficient_hedging" in reason

    def test_rejects_new_numbers(self):
        text = ("I think maybe 42 follows from step 7 of the rule.\n"
                "Final answer: 42")
        passed, reason = validate_hedged(text, self.SOURCE, "42", self.CONFIG)
        assert not passed and "new_numbers" in reason


class TestGrading:

    def test_numeric_forms_are_equivalent(self):
        assert answers_equal("1,540,000", "1540000")
        assert answers_equal("$1540000.00", "1540000")
        assert answers_equal("42.0", "42")

    def test_letters_are_case_insensitive(self):
        assert answers_equal("b", "B")

    def test_unparsed_answer_is_incorrect(self):
        assert grade(None, "B") is False
        assert grade("", "B") is False

    def test_normalisation_is_idempotent(self):
        for value in ["1,540,000", "B", "b ", "42.0", "$5"]:
            once = normalise_answer(value)
            assert normalise_answer(once) == once
