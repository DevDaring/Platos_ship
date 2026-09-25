"""
Tests for the secondary analyses and the pre-paper fixes of 25 Sept 2026:
worst-case parse bounds, adoption over its matched chance rate, the hedged
preamble cleaner, and per-model request fields (the Bedrock pin).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from analysis.secondary import adoption_excess, parse_failure_bounds
from src.hedged_personas import clean_rewrite, validate_hedged


def _unit(condition, q, correct, r0=True, answer="A", targets=None, adopted=False):
    return {"protocol": "B", "dataset_scope": "main300", "round_index": 1,
            "condition": condition, "focal_key": "m", "question_identifier": q,
            "replicate": 0, "is_correct": correct, "r0_is_correct": r0,
            "extracted_answer": answer, "adopted_peer_target": adopted,
            "peer_asserted_targets_all": json.dumps(targets or [])}


def test_bounds_contain_the_scored_estimate_and_widen_with_unparsed():
    rows = [_unit("WR", f"q{i}", i % 2 == 0) for i in range(10)]
    rows += [_unit("R", f"q{i}", True) for i in range(10)]
    rows[1]["extracted_answer"] = None      # one unparsed WR answer, scored wrong
    families = {"F": [{"name": "c", "stat": "accuracy_delta", "a": "WR", "b": "R"}]}
    b = parse_failure_bounds(pd.DataFrame(rows), families).iloc[0]
    assert b["worst_case_low"] <= b["estimate_unparsed_scored_wrong"] <= b["worst_case_high"]
    assert b["worst_case_high"] - b["worst_case_low"] == pytest.approx(0.1)
    assert b["sign_robust"]                  # -0.5 .. -0.4: negative either way


def test_adoption_excess_uses_the_same_units_R_answer_as_chance():
    rows = []
    for i in range(6):
        # WR: adopts the target on 3 of 6; R lands on the same target on 1 of 6
        rows.append(_unit("WR", f"q{i}", False, targets=["C"], adopted=i < 3, answer="C" if i < 3 else "A"))
        rows.append(_unit("R", f"q{i}", i != 0, answer="C" if i == 0 else "A"))
    out = adoption_excess(pd.DataFrame(rows), {"m": 0.5}, n_resamples=200)
    row = out["per_model"][0]
    assert row["adoption"] == pytest.approx(0.5)
    assert row["chance"] == pytest.approx(1 / 6)
    assert row["excess"] == pytest.approx(0.5 - 1 / 6)


def test_clean_rewrite_strips_only_a_leading_instruction_echo():
    echoed = "Here's a rewritten version of the message with tentative wording:\n\nI think it is 5.\nFinal answer: 5"
    assert clean_rewrite(echoed) == "I think it is 5.\nFinal answer: 5"
    plain = "Here is what I think: maybe 5.\nFinal answer: 5"
    assert clean_rewrite(plain) == plain            # no newline after the colon: content, not echo
    config = {"hedge_markers": ["i think", "maybe"], "minimum_hedge_markers": 2}
    ok, reason = validate_hedged("I think maybe this is rewritten. Final answer: 5",
                                 "It is 5. Final answer: 5", "5", config)
    assert not ok and reason.startswith("meta_text")


def test_model_request_fields_merge_over_the_providers():
    from src.agent_wrappers.openai_compatible_agent import build_agent_from_config
    import os
    os.environ["_T_KEY"] = "k"
    providers = {"p": {"base_url_env": "_NONE", "base_url_default": "http://x",
                       "api_key_envs": ["_T_KEY"], "extra_body": {"a": 1, "b": 1}}}
    agent = build_agent_from_config("x", "p", "m", providers,
                                    extra_body={"b": 2, "provider": {"order": ["Amazon Bedrock"]}})
    assert agent.extra_body == {"a": 1, "b": 2, "provider": {"order": ["Amazon Bedrock"]}}
