"""
Map multiple-choice answers stated as an option's VALUE onto its LETTER.

WHY THIS EXISTS

MMLU-Pro stores the correct answer and the wrong peer's target as letters
A-J. Many MMLU-Pro options are numbers, and models sometimes answer with the
option's value rather than its letter: "Final answer: -42" where -42 is
option G. The regex faithfully extracts "-42", which never equals "G", so the
trial is scored as a WRONG answer even when G is correct.

The rate differs sharply by model family, measured on the X8 runs:

    Gemma-3-4B     8.6% of MMLU-Pro answers are values, not letters
    Gemma-3-27B    6.9%   (121 correct answers silently scored wrong)
    Mistral-24B    3.7%
    Qwen2.5-72B    2.5%
    Llama-3.1-8B   1.3%
    Llama-3.1-70B  1.0%

A bias that differs by family can manufacture a family effect in any
comparison that uses solo accuracy or adoption, so it must be removed before
either is read.

THE RULE, WHICH IS DELIBERATELY STRICT

  - Only questions with a list of options are touched.
  - An answer that is already a letter within the option count is kept.
  - Otherwise the answer and every option are normalised (whitespace and
    thousands separators removed; numbers put in canonical form so 7, 7.0
    and 7.00 agree; text lower-cased) and compared EXACTLY.
  - Exactly one matching option: replace with that letter.
  - None, or several: leave the answer untouched. No fuzzy matching and no
    guessing, because a near-miss number (0.74 against an option of 0.75)
    is the model's own wrong answer, not a formatting difference.

The probability-mass read-out is unaffected: it scores letter tokens from
the logprobs and never passes through the extractor.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

LETTERS = "ABCDEFGHIJ"


def _canonical(value: Any) -> str:
    text = str(value).strip().replace(",", "")
    if text.endswith("."):
        text = text[:-1]
    try:
        number = float(text)
        return f"{number:g}"
    except ValueError:
        return text.lower()


def parse_options(raw: Any) -> Optional[List[str]]:
    if isinstance(raw, (list, tuple)):
        return [str(o) for o in raw]
    if hasattr(raw, "tolist") and not isinstance(raw, str):
        try:
            return [str(o) for o in raw.tolist()]
        except Exception:
            return None
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(o) for o in parsed]
        except (ValueError, TypeError):
            return None
    return None


def to_letter(answer: Any, options: Optional[List[str]]) -> Tuple[str, str]:
    """
    Returns (answer, how). `how` is one of:
      'letter'     already a valid letter, unchanged
      'mapped'     a value matching exactly one option, now its letter
      'ambiguous'  matched several options, left unchanged
      'unmatched'  matched none, left unchanged
      'blank'      no answer
      'no_options' not a multiple-choice question
    """
    text = "" if answer is None else str(answer).strip()
    if not text or text.lower() == "nan":
        return "", "blank"
    if not options:
        return text, "no_options"
    if len(text) == 1 and text.upper() in LETTERS[:len(options)]:
        return text.upper(), "letter"
    target = _canonical(text)
    hits = [LETTERS[i] for i, option in enumerate(options[:len(LETTERS)])
            if _canonical(option) == target]
    if len(hits) == 1:
        return hits[0], "mapped"
    return text, ("ambiguous" if hits else "unmatched")


def normalise_frame(frame: pd.DataFrame, pool: pd.DataFrame,
                    columns: Tuple[str, ...]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Apply `to_letter` to each named answer column. The original column is
    kept alongside as `<column>_raw` so the change is auditable row by row.
    """
    options = {str(r["question_identifier"]): parse_options(r.get("answer_options"))
               for r in pool.to_dict("records")}
    out = frame.copy()
    tally: Dict[str, Dict[str, int]] = {}
    qids = out["question_identifier"].astype(str)
    for column in columns:
        if column not in out.columns:
            continue
        out[f"{column}_raw"] = out[column]
        results = [to_letter(a, options.get(q)) for a, q in zip(out[column], qids)]
        out[column] = [r[0] for r in results]
        counts: Dict[str, int] = {}
        for _, how in results:
            counts[how] = counts.get(how, 0) + 1
        tally[column] = counts
    return out, tally
