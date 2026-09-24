"""
Tests for mapping option values onto letters.

The observed failure: a model answers "Final answer: -42" where -42 is option
G. The regex keeps "-42", which never equals "G", so a correct answer is
scored wrong. The rate differed by model family (Gemma 7-9%, Llama ~1%),
which is exactly the kind of bias that can manufacture a family effect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.answer_normalise import normalise_frame, parse_options, to_letter  # noqa: E402

OPTS = ["42", "0", "15", "30", "60", "-20", "-42", "-30", "20", "-15"]


@pytest.mark.parametrize("answer,expected,how", [
    # the observed case: -42 is the seventh option, G
    ("-42", "G", "mapped"),
    # already a letter: untouched, and case is normalised
    ("G", "G", "letter"),
    ("g", "G", "letter"),
    # numeric canonical forms agree
    ("15.0", "C", "mapped"),
    ("15.00", "C", "mapped"),
    # a near miss is the model's own wrong answer, NOT a formatting slip
    ("14.9", "14.9", "unmatched"),
    ("99", "99", "unmatched"),
    # nothing stated
    ("", "", "blank"),
    (None, "", "blank"),
])
def test_to_letter(answer, expected, how):
    assert to_letter(answer, OPTS) == (expected, how)


def test_ambiguous_match_is_left_alone():
    """Two options with the same value: guessing either would invent data."""
    assert to_letter("5", ["5", "5.0", "7"]) == ("5", "ambiguous")


def test_text_options_match_case_insensitively():
    opts = ["bonobos", "orangutans", "gibbons"]
    assert to_letter("Bonobos", opts) == ("A", "mapped")
    assert to_letter("  gibbons ", opts) == ("C", "mapped")


def test_thousands_separator_is_ignored():
    assert to_letter("14,000", ["14000", "1400"]) == ("A", "mapped")


def test_letter_beyond_the_option_count_is_not_a_letter():
    """With four options, 'J' is not a valid choice and must not pass as one."""
    assert to_letter("J", ["1", "2", "3", "4"]) == ("J", "unmatched")


def test_numeric_questions_are_untouched():
    """GSM8K has no option list; its numeric answers must pass through."""
    assert to_letter("42", None) == ("42", "no_options")


def test_parse_options_handles_the_pool_json_string():
    assert parse_options('["a", "b"]') == ["a", "b"]
    assert parse_options(["a", "b"]) == ["a", "b"]
    assert parse_options(float("nan")) is None
    assert parse_options("[not json") is None


def test_normalise_frame_keeps_the_raw_column_for_audit():
    pool = pd.DataFrame({"question_identifier": ["q1", "q2"],
                         "answer_options": ['["10", "20"]', None]})
    frame = pd.DataFrame({"question_identifier": ["q1", "q1", "q2"],
                          "round0_answer": ["20", "A", "20"]})
    out, tally = normalise_frame(frame, pool, ("round0_answer",))
    assert list(out["round0_answer"]) == ["B", "A", "20"]
    assert list(out["round0_answer_raw"]) == ["20", "A", "20"]
    assert tally["round0_answer"] == {"mapped": 1, "letter": 1, "no_options": 1}
