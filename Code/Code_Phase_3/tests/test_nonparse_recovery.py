"""
Tests for the non-parse bounds and the judge-cascade recovery driver.

The bounds are checked against brute-force enumeration rather than against
remembered numbers, so the test verifies the mathematics and not a previous
run of the same code.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.nonparse_bounds import (  # noqa: E402
    _prefix_extreme, bounded_unchanged_subset, classify_pairs, sharp_bounds,
)
from analysis.recover_unparsed import (  # noqa: E402
    _assert_blind, _needs_recovery, calibrate, recover,
)


# -- bounds --------------------------------------------------------------
@pytest.mark.parametrize("seed", range(25))
def test_prefix_scan_matches_exhaustive_search(seed):
    rng = np.random.default_rng(seed)
    known = rng.normal(0, 1, int(rng.integers(1, 5)))
    ambiguous = rng.normal(0, 1, int(rng.integers(0, 8)))
    best_low, best_high = np.inf, -np.inf
    for size in range(ambiguous.size + 1):
        for combo in itertools.combinations(range(ambiguous.size), size):
            values = (np.concatenate([known, ambiguous[list(combo)]])
                      if combo else known)
            best_low = min(best_low, values.mean())
            best_high = max(best_high, values.mean())
    low, _ = _prefix_extreme(known, ambiguous, maximise=False)
    high, _ = _prefix_extreme(known, ambiguous, maximise=True)
    assert low == pytest.approx(best_low)
    assert high == pytest.approx(best_high)


def test_bounds_collapse_when_nothing_is_ambiguous():
    out = sharp_bounds([1.0, 2.0, 3.0], [])
    assert out["bound_low"] == pytest.approx(2.0)
    assert out["bound_high"] == pytest.approx(2.0)
    assert out["bound_width"] == pytest.approx(0.0)


def test_bounds_always_contain_the_complete_case_estimate():
    rng = np.random.default_rng(3)
    for _ in range(50):
        known = rng.normal(0, 1, 6)
        ambiguous = rng.normal(0, 1, 5)
        out = sharp_bounds(known, ambiguous)
        assert out["bound_low"] <= out["estimate_complete_case"] + 1e-12
        assert out["estimate_complete_case"] <= out["bound_high"] + 1e-12
        assert out["bound_low"] <= out["estimate_all_ambiguous_held"] + 1e-12
        assert out["estimate_all_ambiguous_held"] <= out["bound_high"] + 1e-12


def test_unknown_is_not_treated_as_moved():
    """An unreadable generation is absence of evidence, not a changed answer."""
    merged = pd.DataFrame({
        "round0_answer_treatment": ["A", "A", "A"],
        "round1_answer_treatment": ["A", "",  "B"],
        "round0_answer_baseline":  ["A", "A", "A"],
        "round1_answer_baseline":  ["A", "A", "A"],
    })
    assert list(classify_pairs(merged)) == ["in", "ambiguous", "out"]


def test_two_blank_answers_do_not_count_as_held():
    """The regression that inflated the subset: '' == '' is not agreement."""
    merged = pd.DataFrame({
        "round0_answer_treatment": [""], "round1_answer_treatment": [""],
        "round0_answer_baseline": ["A"], "round1_answer_baseline": ["A"],
    })
    assert list(classify_pairs(merged)) == ["ambiguous"]


def _toy_trials():
    rows = []
    for question in range(8):
        for replicate in range(2):
            for condition in ("E", "WR"):
                rows.append({
                    "question_identifier": f"q{question}",
                    "source_dataset": "mmlu_pro",
                    "replicate": replicate,
                    "condition": condition,
                    "round0_answer": "A",
                    "round1_answer": "A" if question % 3 else "",
                    "delta_prob_mass_toward_target":
                        0.1 if condition == "WR" else 0.0,
                })
    return pd.DataFrame(rows)


def test_driver_reports_a_bracketing_interval():
    rows = bounded_unchanged_subset(_toy_trials())
    assert len(rows) == 1
    row = rows[0]
    assert row["bound_low"] <= row["bound_high"]
    assert row["n_known_in"] + row["n_ambiguous"] + row["n_pairs_out"] \
        == row["n_pairs_total"]
    assert row["sign_robust_to_nonparse"] is True   # every pair shifts +0.1


# -- recovery driver -----------------------------------------------------
class _StubCascade:
    """Returns a fixed answer and records what it was shown."""

    def __init__(self, answer="C"):
        self.answer = answer
        self.seen = []
        self.usage_stats = {"total_judge_calls": 0}

    def _build_user_prompt(self, question_text, answer_options, raw_text):
        return (f"The question was: {question_text}\n\n"
                f"The valid answer options were: {answer_options}\n\n"
                f"The model's response was:\n{raw_text}")

    def extract_answer(self, question_text, answer_options, raw_text):
        self.seen.append(raw_text)
        return self.answer, "judge_tier1"


def _trials_with_text():
    return pd.DataFrame({
        "question_identifier": ["q0", "q1", "q2"],
        "condition": ["WR", "E", "WR"],
        "correct_answer": ["A", "B", "C"],
        "fixed_target": ["D", "D", "D"],
        "round1_answer": ["A", "", ""],
        "round1_text": ["The answer is A", "a long ramble", ""],
    })


def test_recovery_targets_only_blank_rows_that_have_text():
    frame = _trials_with_text()
    assert list(_needs_recovery(frame)) == [False, True, False]


def test_recovery_fills_only_the_targeted_row():
    cascade = _StubCascade("C")
    out = recover(_trials_with_text(), cascade)
    assert list(out["round1_answer_judged"]) == ["", "C", ""]
    assert list(out["round1_extraction_method"]) == ["", "judge_tier1", ""]
    assert cascade.seen == ["a long ramble"]


def test_recovery_preserves_the_original_regex_column():
    out = recover(_trials_with_text(), _StubCascade("Z"))
    assert list(out["round1_answer"]) == ["A", "", ""]


def test_abstention_leaves_the_row_unrecovered():
    out = recover(_trials_with_text(), _StubCascade("UNPARSEABLE"))
    assert list(out["round1_answer_judged"]) == ["", "", ""]


def test_recovery_refuses_a_parquet_without_raw_text():
    """The exact situation of the completed L4 run: nothing to judge."""
    frame = _trials_with_text().drop(columns=["round1_text"])
    with pytest.raises(ValueError, match="round1_text"):
        recover(frame, _StubCascade())


def test_blindness_guard_rejects_a_leaking_prompt():
    for leak in ("the condition was WR",
                 "the other agent said D",
                 "correct_answer: A"):
        with pytest.raises(AssertionError):
            _assert_blind(leak, {"condition": "WR", "correct_answer": "A",
                                 "fixed_target": "D"})


def test_blindness_guard_accepts_a_clean_prompt():
    prompt = _StubCascade()._build_user_prompt(
        "What is 2+2?", "a number", "I think it is four.")
    _assert_blind(prompt, {"condition": "WR", "correct_answer": "4",
                           "fixed_target": "7"})


def test_calibration_scores_judge_against_regex():
    frame = pd.DataFrame({
        "question_identifier": ["q0", "q1"],
        "round1_answer": ["C", "D"],
        "round1_text": ["answer C", "answer D"],
    })
    report = calibrate(frame, _StubCascade("C"), sample_size=2)
    assert report["n"] == 2
    assert report["n_agree"] == 1
    assert report["agreement"] == pytest.approx(0.5)
