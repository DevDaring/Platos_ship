"""
Pin the confidence filter's direction.

The deployed filter RETAINS a peer when confidence >= threshold and drops it
otherwise (missing confidence counts as below threshold). The pre-flight gate
is signed to match. These tests exist because an earlier version of the paper
described the gate with the opposite sign, which would have implied that
keeping *wrong* peers was the useful outcome.

    python3 -m pytest Code_Phase_2/CPU_Only/tests/test_filter_semantics.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.confidence_weighted_protocol import filter_peers_by_confidence  # noqa: E402
from src.corrected_gate import compute_corrected_gate  # noqa: E402

THRESHOLD = 60


def _peer(name, conf):
    return {"agent_identifier": name, "extracted_self_reported_confidence_integer": conf}


def test_boundary_values_retained_or_dropped():
    """0, 59 and missing are dropped; 60, 61, 100 are retained."""
    peers = [_peer("focal", 100), _peer("p0", 0), _peer("p59", 59), _peer("p60", 60),
             _peer("p61", 61), _peer("p100", 100), _peer("pNone", None)]
    kept, dropped = filter_peers_by_confidence(peers, "focal", THRESHOLD)
    kept_ids = {p["agent_identifier"] for p in kept}
    assert kept_ids == {"p60", "p61", "p100"}, kept_ids
    assert dropped == 3          # p0, p59, pNone
    assert "focal" not in kept_ids  # the focal agent is never a peer of itself


def test_gate_is_positive_when_correct_peers_are_more_confident():
    """A useful retain-high filter keeps correct peers more often than wrong ones."""
    df = pd.DataFrame({
        "condition_identifier": ["C"] * 20,
        "responding_agent_role": ["dumb"] * 20,
        "debate_round_index": [0] * 20,
        "extracted_self_reported_confidence_integer": [90] * 10 + [10] * 10,
        "extracted_answer_matches_ground_truth": [True] * 10 + [False] * 10,
    })
    r = compute_corrected_gate(df, ["C"], ["dumb"], THRESHOLD, 0.10, 0.60, rounds=[0])
    assert r["retained_when_correct"] == 1.0
    assert r["retained_when_wrong"] == 0.0
    assert r["retention_gap_correct_minus_wrong"] == 1.0
    assert r["gate_decision"] == "passed"


def test_gate_is_negative_when_wrong_peers_are_more_confident():
    """The harmful direction must produce a negative gap and fail."""
    df = pd.DataFrame({
        "condition_identifier": ["C"] * 20,
        "responding_agent_role": ["dumb"] * 20,
        "debate_round_index": [0] * 20,
        "extracted_self_reported_confidence_integer": [10] * 10 + [90] * 10,
        "extracted_answer_matches_ground_truth": [True] * 10 + [False] * 10,
    })
    r = compute_corrected_gate(df, ["C"], ["dumb"], THRESHOLD, 0.10, 0.60, rounds=[0])
    assert r["retention_gap_correct_minus_wrong"] == -1.0
    assert r["gate_decision"] == "failed"


def test_single_class_stays_undefined():
    """Wrong-anchored peers are never correct; the gap must not become 0."""
    df = pd.DataFrame({
        "condition_identifier": ["C"] * 10,
        "responding_agent_role": ["dumb"] * 10,
        "debate_round_index": [0] * 10,
        "extracted_self_reported_confidence_integer": [90] * 10,
        "extracted_answer_matches_ground_truth": [False] * 10,
    })
    r = compute_corrected_gate(df, ["C"], ["dumb"], THRESHOLD, 0.10, 0.60, rounds=[0])
    assert r["retention_gap_correct_minus_wrong"] is None
    assert r["retained_when_correct"] is None
    assert r["gate_decision"] == "undefined_single_class"
