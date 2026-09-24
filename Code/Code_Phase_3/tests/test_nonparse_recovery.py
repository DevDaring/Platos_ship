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
    _assert_blind, _needs_recovery, _options_for, add_effective_answers,
    calibrate, recover, recovery_summary,
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
    """Row 0 parsed, row 1 unreadable but has text, row 2 has no text at all."""
    return pd.DataFrame({
        "question_identifier": ["q0", "q1", "q2"],
        "condition": ["WR", "E", "WR"],
        "correct_answer": ["A", "B", "C"],
        "fixed_target": ["D", "D", "D"],
        "round0_answer": ["A", "B", "C"],
        "round0_text": ["I say A", "I say B", "I say C"],
        "round1_answer": ["A", "", ""],
        "round1_text": ["The answer is A", "a long ramble", ""],
    })


def test_recovery_targets_only_blank_rows_that_have_text():
    frame = _trials_with_text()
    assert list(_needs_recovery(frame, "round1")) == [False, True, False]
    # Round 0 parsed everywhere, so nothing there needs a judge.
    assert list(_needs_recovery(frame, "round0")) == [False, False, False]


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
    # and the effective answer stays blank, so the pair remains ambiguous
    assert list(out["round1_answer_effective"]) == ["A", "", ""]


def test_effective_answer_prefers_regex_and_falls_back_to_judge():
    out = recover(_trials_with_text(), _StubCascade("C"))
    assert list(out["round1_answer_effective"]) == ["A", "C", ""]
    # the regex column is never overwritten, so provenance stays auditable
    assert list(out["round1_answer"]) == ["A", "", ""]


def test_round_zero_is_recovered_too():
    """
    A Round-0 failure removes a trial from the subset just as a Round-1
    failure does, so both rounds must be judged.
    """
    frame = _trials_with_text()
    frame.loc[0, "round0_answer"] = ""
    cascade = _StubCascade("Q")
    out = recover(frame, cascade)
    assert out.loc[0, "round0_answer_judged"] == "Q"
    assert out.loc[0, "round0_answer_effective"] == "Q"
    assert "I say A" in cascade.seen


def test_recovery_summary_counts_what_was_resolved():
    out = recover(_trials_with_text(), _StubCascade("C"))
    summary = recovery_summary(out)
    assert summary["round1"]["unparsed_by_regex"] == 2
    assert summary["round1"]["recovered"] == 1
    assert summary["round1"]["still_unresolved"] == 1


def test_recovery_refuses_a_parquet_without_raw_text():
    """The exact situation of the first L4 run: nothing to judge."""
    frame = _trials_with_text().drop(columns=["round1_text"])
    with pytest.raises(ValueError, match="round1_text"):
        recover(frame, _StubCascade())


# -- option labelling ----------------------------------------------------
# The focal model answers "Final answer: G". Handing the judge the bare
# option texts gave it no letters to answer in, so it abstained on clear
# answers or replied with the option text. Agreement with the regex on rows
# the regex had already read was 29/40; with letters it is 39/40.
def test_options_are_labelled_with_letters():
    import json
    rendered = _options_for({"answer_options": json.dumps(["42", "0", "15"])})
    assert rendered.startswith("A) 42; B) 0; C) 15")
    assert "single capital letter" in rendered


def test_options_parse_a_json_string_not_just_a_list():
    """
    The pool stores the options as a JSON STRING.

    Labelling only the list case was a silent no-op: the function fell
    through and handed the judge raw JSON, which is what the first
    calibration run actually measured.
    """
    import json
    as_string = _options_for({"answer_options": json.dumps(["x", "y"])})
    as_list = _options_for({"answer_options": ["x", "y"]})
    assert as_string == as_list
    assert as_string.startswith("A) x; B) y")


def test_letter_order_matches_the_extractor():
    """Index 0 is A, which is the mapping the probe's own regex assumes."""
    import json
    rendered = _options_for({
        "answer_options": json.dumps([str(i) for i in range(10)])})
    assert "G) 6" in rendered          # 7th option is G
    assert "J) 9" in rendered


def test_numeric_questions_get_no_letters():
    assert _options_for({"answer_options": None}) == "a number"
    assert _options_for({"answer_options": float("nan")}) == "a number"


def test_malformed_option_json_does_not_crash():
    out = _options_for({"answer_options": "[not valid json"})
    assert isinstance(out, str) and out


class _LeakyCascade(_StubCascade):
    """A builder that puts experimental metadata into the template."""

    def __init__(self, leak):
        super().__init__()
        self.leak = leak

    def _build_user_prompt(self, question_text, answer_options, raw_text):
        return (f"{self.leak}\n"
                + super()._build_user_prompt(question_text, answer_options,
                                             raw_text))


_ROW = {"condition": "WR", "correct_answer": "banana", "fixed_target": "cherry"}


@pytest.mark.parametrize("leak", [
    "this trial came from condition WR",
    "the other agent said cherry",
    "correct_answer for reference: banana",
])
def test_blindness_guard_rejects_a_leaking_template(leak):
    cascade = _LeakyCascade(leak)
    prompt = cascade._build_user_prompt("What is 2+2?", "a number", "four.")
    with pytest.raises(AssertionError):
        _assert_blind(cascade, prompt, _ROW, "What is 2+2?", "a number",
                      "four.")


def test_blindness_guard_accepts_a_clean_prompt():
    cascade = _StubCascade()
    question, options, raw = "What is 2+2?", "a number", "I think it is four."
    prompt = cascade._build_user_prompt(question, options, raw)
    _assert_blind(cascade, prompt, _ROW, question, options, raw)


def test_blindness_guard_tolerates_content_that_looks_like_a_leak():
    """
    The content is arbitrary natural language and is NOT searched.

    Three earlier versions of this guard aborted real runs here: on a
    response containing the correct answer, on the ordinary word
    "condition" inside an exam question, and on a response that quoted the
    question back.
    """
    cascade = _StubCascade()
    question = "Which condition causes this, given the other agent's report?"
    options = "A) banana; B) cherry"
    raw = ("Restating: Which condition causes this, given the other agent's "
           "report? I say banana.")
    prompt = cascade._build_user_prompt(question, options, raw)
    _assert_blind(cascade, prompt, _ROW, question, options, raw)


def test_blindness_guard_rejects_an_unexpected_interpolation():
    """If the prompt is not the template filled with these inputs, stop."""
    cascade = _StubCascade()
    prompt = cascade._build_user_prompt("q", "o", "r") + "\nextra smuggled text"
    with pytest.raises(AssertionError, match="not the question/options"):
        _assert_blind(cascade, prompt, _ROW, "q", "o", "r")


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


# -- calibration must not penalise a correct judge ----------------------
class _ScriptedCascade(_StubCascade):
    """Replies with a scripted answer per response text."""

    def __init__(self, replies):
        super().__init__()
        self.replies = replies

    def extract_answer(self, question_text, answer_options, raw_text):
        return self.replies[raw_text], "judge_tier1"


def test_calibration_separates_formatting_letters_and_abstentions():
    """
    Strict equality counted three non-errors as errors. Formatting (14000 vs
    14,000), the judge giving the LETTER where the regex kept the option
    VALUE (-42 is option G), and an abstention. None is a wrong answer.
    """
    frame = pd.DataFrame({
        "question_identifier": ["q1", "q2", "q3", "q4"],
        "round1_answer": ["14000", "-42", "B", "C"],
        "round1_text": ["t1", "t2", "t3", "t4"],
        "answer_options": [None,
                           '["42","0","15","30","60","-20","-42"]',
                           '["x","y","z"]', '["x","y","z"]'],
    })
    cascade = _ScriptedCascade({"t1": "14,000", "t2": "G",
                                "t3": "UNPARSEABLE", "t4": "A"})
    r = calibrate(frame, cascade, sample_size=4)
    assert r["n_abstained"] == 1
    assert r["n_committed"] == 3
    # 14,000 == 14000 and G == -42 agree; C vs A is the one real error
    assert r["n_agree_normalised"] == 2
    assert r["precision_when_committed"] == pytest.approx(2 / 3)
    real = [d for d in r["example_disagreements"]]
    assert len(real) == 1 and real[0]["judge"] == "A"
    # strict agreement is kept for comparison with earlier runs
    assert r["n_agree"] == 0
