"""
Tests for persona generation and validation (src/anchored_personas.py) and
for the pool loader's filtering (src/peer_pools.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.anchored_personas import (  # noqa: E402
    canonical_answer, generate_pool, is_usable_persona, validate_persona,
)
from src.peer_pools import _index_personas, _merge_disjoint  # noqa: E402
from src.store import Checkpoint  # noqa: E402


# -- answer comparison ---------------------------------------------------
@pytest.mark.parametrize("a,b,same", [
    ("90.", "90", True),         # the defect: 7 valid personas rejected on this
    ("90.0", "90", True),
    ("**90**", "90", True),
    ("1,248", "1248", True),
    ("12.50", "12.5", True),
    ("H.", "H", True),
    ("h", "H", True),
    ("90", "80", False),
    ("B", "C", False),
])
def test_canonical_answer(a, b, same):
    assert (canonical_answer(a) == canonical_answer(b)) is same


# -- the four Phase-2 rules ----------------------------------------------
GOOD = "It is clearly 90, since the rate doubles each step.\nFinal answer: 90."


def test_valid_persona_passes_despite_trailing_full_stop():
    assert validate_persona(GOOD, "90") == (True, "passed")


def test_missing_answer_line_fails():
    ok, reason = validate_persona("It is clearly 90 without doubt.", "90")
    assert not ok and reason == "missing_final_answer_marker"


def test_stating_a_different_answer_fails():
    ok, reason = validate_persona("Clearly so.\nFinal answer: 80", "90")
    assert not ok and reason.startswith("answer_mismatch")


def test_hedging_fails():
    text = "I am unsure, and actually I might be wrong.\nFinal answer: 90"
    ok, reason = validate_persona(text, "90")
    assert not ok and reason.startswith("excessive_hedging")


def test_length_bounds():
    assert validate_persona("Final answer: 9", "9")[1].startswith("too_short")
    assert validate_persona("x" * 1600 + "\nFinal answer: 9", "9")[1].startswith("too_long")


# -- which pools use which rule ------------------------------------------
def test_confident_rule_revalidates_rather_than_trusting_the_flag():
    row = {"generated_persona_text": GOOD,
           "assigned_wrong_answer_letter_or_value": "90",
           "validation_pass_status": "failed: answer_mismatch: extracted='90.'"}
    assert is_usable_persona(row, "confident") is True


def test_stored_rule_trusts_the_hedged_pools_own_verdict():
    """The hedged pool is hedged on purpose; the no-hedging rule would reject all of it."""
    hedged = {"generated_persona_text": "I might be wrong, but maybe 90.\nFinal answer: 90",
              "assigned_wrong_answer_letter_or_value": "90",
              "validation_pass_status": "passed"}
    assert is_usable_persona(hedged, "stored") is True
    assert is_usable_persona({**hedged, "validation_pass_status": "failed"}, "stored") is False


def test_loader_drops_unusable_personas_and_keeps_valid_ones():
    frame = pd.DataFrame([
        {"question_identifier": "q1", "persona_variant_index": 0,
         "generated_persona_text": GOOD, "assigned_wrong_answer_letter_or_value": "90"},
        {"question_identifier": "q1", "persona_variant_index": 1,
         "generated_persona_text": "no answer line here at all",
         "assigned_wrong_answer_letter_or_value": "90"},
    ])
    index = _index_personas(frame, "confident", "t")
    assert [p["persona_variant_index"] for p in index["q1"]] == [0]


def test_overlapping_pools_are_refused_rather_than_mixed():
    with pytest.raises(ValueError, match="overlap"):
        _merge_disjoint({"q1": [{}]}, {"q1": [{}]})
    assert set(_merge_disjoint({"q1": [{}]}, {"q2": [{}]})) == {"q1", "q2"}


# -- generation ----------------------------------------------------------
class _Resp:
    def __init__(self, text):
        self.raw_text_output = text
        self.model_name_returned_by_provider = "meta-llama/llama-3.1-8b-instruct"


class _ScriptedAgent:
    """Answers with whatever it was told to defend; optionally fails first."""

    def __init__(self, fail_first=0):
        self.calls = []
        self.fail_first = fail_first

    def generate_response(self, system_prompt, user_prompt, temperature,
                          maximum_output_tokens, request_metadata=None):
        self.calls.append(user_prompt)
        assigned = user_prompt.split("End with the line:\nFinal answer: ")[-1].strip()
        if len(self.calls) <= self.fail_first:
            return _Resp("This one forgets to commit to anything.")
        return _Resp(f"It is obviously {assigned}, no question about it.\n"
                     f"Final answer: {assigned}")


def _questions():
    return pd.DataFrame([
        {"question_identifier": "gsmsym_1", "question_text": "What is 3 + 4?",
         "correct_answer": "7", "wrong_answer_pool": '["6", "8", "14"]',
         "answer_options": None},
        {"question_identifier": "gsmsym_2", "question_text": "What is 2 * 5?",
         "correct_answer": "10", "wrong_answer_pool": '["9", "11", "20"]',
         "answer_options": None},
    ])


def test_every_persona_defends_a_wrong_answer(tmp_path):
    pool = generate_pool(_questions(), _ScriptedAgent(), tmp_path / "p.parquet",
                         Checkpoint(tmp_path / "c.parquet"), master_seed=20260502,
                         variants=5)
    assert len(pool) == 10
    assert (pool.validation_pass_status == "passed").all()
    correct = {"gsmsym_1": "7", "gsmsym_2": "10"}
    for _, r in pool.iterrows():
        assert r.assigned_wrong_answer_letter_or_value != correct[r.question_identifier]


def test_generation_is_reproducible(tmp_path):
    """Seeds come from crc32, not row position or Python's salted hash()."""
    a = generate_pool(_questions(), _ScriptedAgent(), tmp_path / "a.parquet",
                      Checkpoint(tmp_path / "ca.parquet"), master_seed=20260502)
    b = generate_pool(_questions().iloc[::-1].reset_index(drop=True),
                      _ScriptedAgent(), tmp_path / "b.parquet",
                      Checkpoint(tmp_path / "cb.parquet"), master_seed=20260502)
    key = ["persona_identifier", "assigned_wrong_answer_letter_or_value",
           "reasoning_style_label"]
    assert a[key].sort_values(key[0]).reset_index(drop=True).equals(
        b[key].sort_values(key[0]).reset_index(drop=True))


def test_a_failed_persona_is_regenerated(tmp_path):
    agent = _ScriptedAgent(fail_first=1)
    pool = generate_pool(_questions().head(1), agent, tmp_path / "p.parquet",
                         Checkpoint(tmp_path / "c.parquet"), master_seed=1,
                         variants=1)
    assert pool.validation_pass_status.iloc[0] == "passed"
    assert int(pool.regeneration_attempts_used.iloc[0]) == 1
    assert len(agent.calls) == 2


def test_resume_does_not_pay_twice(tmp_path):
    ck = tmp_path / "c.parquet"
    first = _ScriptedAgent()
    generate_pool(_questions(), first, tmp_path / "p.parquet", Checkpoint(ck),
                  master_seed=1, variants=5)
    again = _ScriptedAgent()
    pool = generate_pool(_questions(), again, tmp_path / "p.parquet", Checkpoint(ck),
                         master_seed=1, variants=5)
    assert again.calls == []
    assert len(pool) == 10
