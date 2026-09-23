"""
Regression tests for two judge-cascade defects that fabricated answers.

Both were found by running the cascade on responses that state no answer.

DEFECT 1 - ABSTENTION ESCALATED LIKE AN OUTAGE.
  `extract_answer` fell through to the next tier on UNPARSEABLE exactly as
  it fell through on a 500. A response that never committed was therefore
  handed to a second, equally competent model, which answered the question.
  Observed: "I need to think about this more carefully" on "What is 8 * 7?"
  came back as 56.

DEFECT 2 - THE PROMPT NEVER FORBADE SOLVING.
  Hardening the wording was not enough: tier 1 still returned 56 for the
  same input. The guard that works is mechanical. An answer that appears
  nowhere in the response was not extracted from it.

Both matter because a fabricated answer does not look like missing data. It
enters the analysis as an observation, and it carries the JUDGE's competence
rather than the focal model's behaviour, which is the quantity under study.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agent_wrappers.base_agent import AgentResponse  # noqa: E402
from src.agent_wrappers.judge_agent import JudgeCascade  # noqa: E402


# -- the mechanical guard ------------------------------------------------
@pytest.mark.parametrize("answer,text,expected", [
    # the exact fabrication that prompted the guard
    ("56", "I need to think about this more carefully", False),
    ("42", "Adding again: 17+25 = 42.", True),
    ("14", "Actually I made an error before. It is 14, not 15.", True),
    # an option letter must be its own token, not a letter inside a word
    ("B", "Mercury is nearest. I'll go with B.", True),
    ("B", "Also, about that, nothing here", False),
    # Echoing an option line is NOT choosing it. "A)" here is the model
    # restating the choices, so it must not ground option A.
    ("A", "A) Venus is hot", False),
    # numeric normalisation
    ("3", "The result is 3.0 exactly", True),
    ("1000", "I get 1,000 in total", True),
    ("3.50", "the cost is 3.5 dollars", True),
    # a digit inside a longer number is not that number
    ("7", "There are 17 apples", False),
    ("2", "the value 25 appears here", False),
    # nothing at all
    ("", "anything", False),
    ("42", "", False),
])
def test_grounding_guard(answer, text, expected):
    assert JudgeCascade._is_grounded(answer, text) is expected


def test_guard_rejects_a_correct_answer_that_was_never_stated():
    """
    Being right is not the same as being present.

    The judge must not supply the answer the response failed to give, even
    when it supplies the correct one.
    """
    assert JudgeCascade._is_grounded("56", "Let me work out 8 times 7 later") is False


def test_guard_accepts_a_wrong_answer_that_was_stated():
    """A response's wrong answer is exactly what the probe needs recorded."""
    assert JudgeCascade._is_grounded("15", "8 times 7 is 15, I'm confident") is True


# -- cascade control flow ------------------------------------------------
class _Tier:
    """Minimal stand-in for a configured tier."""

    def __init__(self, name, reply, status="success"):
        self.name = name
        self.calls = 0

        class _Agent:
            def generate_response(_self, **kwargs):
                self.calls += 1
                return AgentResponse(raw_text_output=reply, error_status=status)

        self.spec = {"name": name, "agent": _Agent(), "max_tokens": 16,
                     "label": name}


def _cascade(*tiers):
    """Build a JudgeCascade without touching config or the network."""
    cascade = JudgeCascade.__new__(JudgeCascade)
    cascade._tiers = [t.spec for t in tiers]
    cascade._total_calls = 0
    cascade._tier_usage = {t.name: 0 for t in tiers}
    cascade._tier_usage["all_failed"] = 0
    return cascade


def test_abstention_stops_the_cascade():
    """The second tier must never see a response the first declared unreadable."""
    first = _Tier("primary", "UNPARSEABLE")
    second = _Tier("secondary", "56")
    answer, method = _cascade(first, second).extract_answer(
        "What is 8 * 7?", "a number", "I need to think about this more")
    assert answer == "UNPARSEABLE"
    assert method == "abstain_primary"
    assert second.calls == 0, "abstention escalated; the next tier solved it"


def test_provider_failure_does_escalate():
    """An outage is not a verdict, so the next tier should be tried."""
    first = _Tier("primary", "", status="failure")
    second = _Tier("secondary", "42")
    answer, method = _cascade(first, second).extract_answer(
        "What is 17 + 25?", "a number", "the total is 42")
    assert answer == "42"
    assert method == "judge_secondary"
    assert second.calls == 1


def test_empty_reply_escalates():
    first = _Tier("primary", "   ")
    second = _Tier("secondary", "42")
    answer, _ = _cascade(first, second).extract_answer(
        "q", "a number", "the total is 42")
    assert answer == "42"


def test_ungrounded_answer_abstains_without_escalating():
    """
    A tier that solves is refused, not escalated.

    Escalating would only hand the same solvable question to another model
    just as willing to answer it.
    """
    first = _Tier("primary", "56")
    second = _Tier("secondary", "56")
    answer, method = _cascade(first, second).extract_answer(
        "What is 8 * 7?", "a number", "I need to think about this more")
    assert answer == "UNPARSEABLE"
    assert method == "ungrounded_primary"
    assert second.calls == 0


def test_grounded_answer_is_returned_from_the_first_tier():
    first = _Tier("primary", "42")
    second = _Tier("secondary", "99")
    answer, method = _cascade(first, second).extract_answer(
        "What is 17 + 25?", "a number", "Adding again: 17+25 = 42.")
    assert answer == "42"
    assert method == "judge_primary"
    assert second.calls == 0


def test_all_tiers_failing_returns_parse_failure():
    first = _Tier("primary", "", status="failure")
    second = _Tier("secondary", "", status="failure")
    answer, method = _cascade(first, second).extract_answer("q", "o", "t")
    assert answer == "UNPARSEABLE"
    assert method == "parse_failure"

@pytest.mark.parametrize("reply", ["UNPARSEABLE", "UNPARSE", "unparseable",
                                   "  UNPARSEABLE  ", "**UNPARSEABLE**"])
def test_any_spelling_of_the_abstention_token_is_an_abstention(reply):
    """
    A tier replied "UNPARSE". Strict equality missed it, so the answer fell
    through to the grounding guard and was logged as a fabrication rather
    than as the abstention it plainly was.
    """
    first = _Tier("primary", reply)
    second = _Tier("secondary", "56")
    answer, method = _cascade(first, second).extract_answer(
        "What is 8 * 7?", "a number", "I need to think about this more")
    assert answer == "UNPARSEABLE"
    assert method == "abstain_primary"
    assert second.calls == 0

# -- option-letter grounding --------------------------------------------
# Letter-only matching was wrong in both directions. Case-insensitive
# matching meant "a" and "I" grounded options A and I on almost any text,
# so those two letters were never really checked. And a response that named
# the option without writing its letter was rejected although the judge had
# read it correctly. These are the rows whose "Final answer:" line the regex
# could not find, so naming without lettering is exactly the common case.
_OPTIONS = ("A) bonobos; B) orangutans; C) gibbons; D) chimpanzees; "
            "I) lemurs  -- answer with the single capital letter")


@pytest.mark.parametrize("letter,text,expected,why", [
    ("A", "closely related to bonobos, so that one", True,
     "the option body appears, so the judge read rather than solved"),
    ("A", "Final answer: A Confidence: 90", True, "stated as a choice"),
    ("A", "a quick note about nothing", False, "lowercase article"),
    ("A", "A careful reading shows the result", False, "sentence-initial"),
    ("I", "I think we should wait", False, "pronoun, not an option"),
    ("I", "I will go with I here", True, "stated as a choice"),
    ("I", "the lemurs are the answer", True, "option body appears"),
    ("D", "I pick D here", True, "unambiguous letter as a token"),
    ("J", "no letter here", False, "absent"),
    ("C", "nothing relevant", False, "neither letter nor body"),
])
def test_letter_grounding(letter, text, expected, why):
    assert JudgeCascade._is_grounded(letter, text, _OPTIONS) is expected, why


def test_last_option_body_excludes_the_trailing_instruction():
    """
    The listing ends with an instruction. Without treating "--" as a
    terminator the final option absorbed it, so its body matched nothing
    and that option could never be grounded by its text.
    """
    assert JudgeCascade._option_text_for_letter("I", _OPTIONS) == "lemurs"
    assert JudgeCascade._option_text_for_letter("A", _OPTIONS) == "bonobos"


def test_letter_grounding_without_an_options_listing():
    """With no listing, fall back to letter matching alone."""
    assert JudgeCascade._is_grounded("D", "I pick D", "") is True
    assert JudgeCascade._is_grounded("D", "nothing", "") is False
