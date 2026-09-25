"""Tests for the Phase 4 additions: hidden-answer prompts, natural panels,
and the exact-expectation random safeguard comparator."""

import pandas as pd

from src.contexts import PeerMessage, build_revision_prompt
from src.peer_pools import PersonaPools, build_peers


def test_hidden_prompt_omits_first_answer_and_keeps_format():
    _, shown = build_revision_prompt("Q?", None, "MY FIRST ANSWER", [], "none")
    _, hidden = build_revision_prompt("Q?", None, None, [], "none")
    assert "MY FIRST ANSWER" in shown and "Your previous response" in shown
    assert "Your previous response" not in hidden and "previous" not in hidden.lower()
    assert "Final answer: <your answer>" in hidden


def test_hidden_prompt_with_peers_keeps_the_peer_block():
    peers = [PeerMessage(display_name="Agent_Beta", text="It is 48. Final answer: B")]
    _, hidden = build_revision_prompt("Q?", None, None, peers, "anchored_wrong")
    assert "Agent_Beta said: It is 48" in hidden
    assert hidden.index("Agent_Beta") < hidden.index("Review the other responses")


def _bank(n_correct, n_wrong):
    rows = []
    for i in range(n_correct):
        rows.append({"question_identifier": "q1", "weak_model_key": "a", "replicate": i,
                     "message_text": f"c{i}. Final answer: A", "extracted_answer": "A",
                     "is_correct": True, "honest_unit_id": f"c{i}"})
    for i in range(n_wrong):
        rows.append({"question_identifier": "q1", "weak_model_key": "b", "replicate": i,
                     "message_text": f"w{i}. Final answer: {'BC'[i % 2]}",
                     "extracted_answer": "BC"[i % 2], "is_correct": False,
                     "honest_unit_id": f"w{i}"})
    return {"q1": rows}


def _pools(bank):
    return PersonaPools(anchored_wrong={}, anchored_correct={}, anchored_hedged={},
                        honest_bank={}, anchored_confidence={}, natural_bank=bank)


def _panel(n_wrong, bank):
    cond = {"peer_source": "natural_panel", "n_peers": 2, "n_wrong": n_wrong}
    peers, diag = build_peers(cond, {"question_identifier": "q1"}, 0, _pools(bank), {},
                              master_seed=7, focal_key="f")
    return peers, diag


def test_natural_panels_are_nested_and_composed_as_designed():
    bank = _bank(3, 3)
    p0, _ = _panel(0, bank)
    p1, _ = _panel(1, bank)
    p2, _ = _panel(2, bank)
    assert [p.anchor_mode for p in p0] == ["correct", "correct"]
    assert sorted(p.anchor_mode for p in p1) == ["correct", "wrong"]
    assert [p.anchor_mode for p in p2] == ["wrong", "wrong"]
    assert p1[0].text == p0[0].text           # same first correct message
    assert p1[1].text == p2[0].text           # same first wrong message
    assert all(p.assigned_target == p.final_answer for p in p2)
    assert all(p.assigned_target is None for p in p0)


def test_ineligible_question_is_skipped_not_degraded():
    peers, diag = _panel(1, _bank(1, 5))
    assert peers == [] and "natural_panel_ineligible" in diag["skipped_reason"]


def test_random_comparator_is_matched_exactly():
    from analysis.safeguard import score_policies

    units = pd.DataFrame({
        "unit_id": [f"u{i}" for i in range(8)],
        "focal_key": ["f"] * 8,
        "question_identifier": [f"q{i}" for i in range(8)],
        "replicate": [0] * 8,
        "answer_changed": [True, True, True, True, False, False, False, False],
        "extracted_answer": ["A", "B", "A", "B", "A", "A", "B", "B"],
        "r0_answer": ["B", "A", "B", "A", "A", "A", "B", "B"],
        "correct_answer": ["A"] * 8,
        "r0_is_correct": [False, True, False, True, True, True, False, False],
        "extracted_confidence": [95] * 8,
    })
    verified = pd.DataFrame({"unit_id": ["u0", "u1", "u2", "u3"],
                             "adopt_change": [True, False, False, False]})
    table = score_policies(verified, units).set_index("policy")
    assert table.loc["random_matched", "adoption_rate_of_changes"] == \
        table.loc["verifier", "adoption_rate_of_changes"] == 0.25
