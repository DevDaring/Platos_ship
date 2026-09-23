"""
extraction.py — answer/confidence parsing and grading.

Thin layer over the proven Phase-2 cascade (regex first, then a three-tier
judge). Adds:
  * `normalise_answer` / `answers_equal`, so a numeric answer written as
    "1,540,000", "1540000.0" or "$1540000" grades identically;
  * `grade`, the single place correctness is decided;
  * explicit parse-status strings that the registry can count.

Unrecovered parse failures are scored INCORRECT — the conservative choice, and
the same convention as Phase 1/2, so the two protocols stay comparable.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Tuple

from .agent_wrappers.judge_agent import (  # noqa: F401  (re-exported)
    extract_answer_regex,
    extract_confidence_regex,
)

logger = logging.getLogger("platos_ship3.extraction")

_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_STRIP_CHARS = " \t\n\r.,;:!?'\"()[]{}*$%"


def normalise_answer(answer: Any) -> str:
    """
    Canonical form for comparison.

    Single letters upper-case ("b" -> "B"). Numbers lose thousands separators,
    currency/percent symbols and trailing zeros ("1,540,000.00" -> "1540000").
    Everything else is lower-cased and whitespace-collapsed.
    """
    if answer is None:
        return ""
    text = str(answer).strip().strip(_STRIP_CHARS)
    if not text:
        return ""

    if len(text) == 1 and text.isalpha():
        return text.upper()

    candidate = text.replace(",", "").replace("$", "").replace("%", "").strip()
    if _NUMERIC_RE.match(candidate):
        value = float(candidate)
        if value == int(value):
            return str(int(value))
        return f"{value:g}"

    return " ".join(text.lower().split())


def answers_equal(left: Any, right: Any) -> bool:
    """True when two answers are the same after normalisation."""
    left_n, right_n = normalise_answer(left), normalise_answer(right)
    if not left_n or not right_n:
        return False
    return left_n == right_n


def grade(extracted_answer: Any, correct_answer: Any) -> bool:
    """
    Decide correctness. An unparsed answer ("" / None) is INCORRECT.

    This is the only place in Phase 3 where correctness is decided, so a
    change to grading cannot silently disagree between metrics and tables.
    """
    return answers_equal(extracted_answer, correct_answer)


def extract_answer(
    raw_text: str,
    question_text: str,
    answer_options_str: str,
    judge_cascade=None,
) -> Tuple[Optional[str], str]:
    """
    Regex first; judge cascade only when regex fails.

    Returns (answer_or_None, method). `method` is one of
    "regex_success", a judge tier name, or "unrecovered_parse_failure".
    """
    answer = extract_answer_regex(raw_text)
    if answer:
        return answer, "regex_success"

    if judge_cascade is None:
        return None, "unrecovered_parse_failure"

    try:
        answer, method = judge_cascade.extract_answer(
            question_text=question_text,
            answer_options=answer_options_str,
            raw_text=raw_text,
        )
    except Exception as exc:                      # judge outage must not crash a run
        logger.warning("Judge cascade raised %s; scoring as parse failure.", exc)
        return None, "unrecovered_parse_failure"

    if not answer or str(answer).strip().upper() == "UNPARSEABLE":
        return None, "unrecovered_parse_failure"
    return answer, method


def extract_confidence(raw_text: str) -> Tuple[Optional[int], str]:
    """
    Confidence integer plus a parse status.

    Status is one of "parsed", "missing_line", "out_of_range" or
    "unparseable". The retention-gap analysis needs "confidence absent" kept
    distinct from "confidence low" — conflating them is one of the two faults
    in the original filter.
    """
    confidence, status = extract_confidence_regex(raw_text)
    if confidence is None:
        return None, status or "missing_line"
    try:
        value = int(confidence)
    except (TypeError, ValueError):
        return None, "unparseable"
    if not 0 <= value <= 100:
        return None, "out_of_range"
    return value, "parsed"


def options_to_string(answer_options: Any) -> str:
    """Render the option list for the judge prompt ('' for numeric items)."""
    import json

    if answer_options is None or not isinstance(answer_options, (str, list)):
        return ""
    if isinstance(answer_options, str):
        try:
            answer_options = json.loads(answer_options)
        except (ValueError, TypeError):
            return ""
    if not answer_options:
        return ""
    return "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(answer_options))
