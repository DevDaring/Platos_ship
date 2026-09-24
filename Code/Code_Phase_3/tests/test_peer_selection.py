"""
Tests for peer selection (src/peer_pools.py).

Found by reading the code during the parallel rehearsal: peers were drawn
WITH replacement, so 20.1% of WR trials on the main pool showed two
supposedly independent peers posting identical text; WRagree duplicated one
persona when it could not find two agreeing ones; WRdiff topped up at random;
and a unit with fewer peers than designed ran anyway. No test covered any of
this, which is why it survived.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.peer_pools import PersonaPools, _pick_variants, build_peers, load_pools  # noqa: E402


def _personas(targets):
    return [{"persona_identifier": f"p{i}", "persona_variant_index": i,
             "assigned_wrong_answer_letter_or_value": t,
             "generated_persona_text": f"It is {t}.\nFinal answer: {t}",
             "generator_model_name": "llama"} for i, t in enumerate(targets)]


FIVE = _personas(["B", "C", "B", "D", "E"])


# -- distinctness --------------------------------------------------------
@pytest.mark.parametrize("count", [1, 2, 4])
def test_default_never_repeats_a_persona(count):
    for seed in range(500):
        picked = _pick_variants(FIVE, count, random.Random(seed))
        ids = [p["persona_identifier"] for p in picked]
        assert len(ids) == count and len(set(ids)) == count


def test_same_target_gives_distinct_personas_sharing_one_target():
    for seed in range(300):
        picked = _pick_variants(FIVE, 2, random.Random(seed), force_same_target=True)
        assert len({p["persona_identifier"] for p in picked}) == 2
        assert len({p["assigned_wrong_answer_letter_or_value"] for p in picked}) == 1


def test_distinct_targets_gives_different_targets():
    for seed in range(300):
        picked = _pick_variants(FIVE, 2, random.Random(seed), force_distinct_targets=True)
        assert len({p["assigned_wrong_answer_letter_or_value"] for p in picked}) == 2


# -- infeasible conditions refuse instead of faking ---------------------
def test_same_target_refuses_rather_than_duplicating_a_persona():
    """Previously this returned one persona twice: identical text, two names."""
    assert _pick_variants(_personas(["B", "C", "D"]), 2, random.Random(1),
                          force_same_target=True) == []


def test_distinct_targets_refuses_rather_than_topping_up():
    assert _pick_variants(_personas(["B", "B", "B"]), 2, random.Random(1),
                          force_distinct_targets=True) == []


def test_too_few_personas_refuses():
    assert _pick_variants(_personas(["B", "C"]), 4, random.Random(1)) == []


# -- build_peers records why ---------------------------------------------
def _pools(wrong=None, honest=None):
    return PersonaPools(anchored_wrong=wrong or {}, anchored_correct={},
                        anchored_hedged={}, honest_bank=honest or {})


def _build(condition, pools, self_samples=None):
    return build_peers(condition={**condition, "_name": "t"},
                       question={"question_identifier": "q1"}, replicate=0,
                       pools=pools, weak_specs={"w": {"paper_name": "W"}},
                       master_seed=1, focal_key="f", self_samples=self_samples)


def test_infeasible_agreement_records_a_reason():
    peers, diag = _build({"peer_source": "anchored_wrong", "n_peers": 2,
                          "force_same_target": True},
                         _pools(wrong={"q1": _personas(["B", "C", "D"])}))
    assert peers == [] and diag["skipped_reason"].startswith("anchored_wrong_infeasible")


def test_honest_with_one_message_where_two_are_designed_is_skipped():
    """Previously ran with one honest peer, silently."""
    one = {("q1", 0): [{"message_text": "It is 7.\nFinal answer: 7",
                        "weak_model_key": "w", "extracted_answer": "7"}]}
    peers, diag = _build({"peer_source": "honest", "n_peers": 2}, _pools(honest=one))
    assert peers == [] and diag["skipped_reason"].startswith("honest_bank_insufficient")


def test_self_samples_short_are_skipped_with_a_reason():
    peers, diag = _build({"peer_source": "self_samples", "n_peers": 2}, _pools(),
                         self_samples=[{"raw_response_text": "x", "extracted_answer": "A"}])
    assert peers == [] and diag["skipped_reason"].startswith("self_samples_insufficient")


def test_feasible_unit_gets_exactly_the_designed_peers():
    peers, _ = _build({"peer_source": "anchored_wrong", "n_peers": 2},
                      _pools(wrong={"q1": FIVE}))
    assert len(peers) == 2
    assert len({p.persona_identifier for p in peers}) == 2


# -- on the real pool ----------------------------------------------------
@pytest.mark.skipif(not (_ROOT / "results/processed/dumb_personas.parquet").exists(),
                    reason="persona pool not present")
def test_no_identical_peers_on_the_real_pool():
    """20.1% of WR trials had identical peers before; now exactly none."""
    logging.disable(logging.CRITICAL)
    try:
        paths = yaml.safe_load((_ROOT / "config/paths.yaml").read_text(encoding="utf-8"))
        pools = load_pools(paths, _ROOT)
    finally:
        logging.disable(logging.NOTSET)
    duplicates = trials = 0
    for qid, variants in pools.anchored_wrong.items():
        for rep in range(3):
            picked = _pick_variants(variants, 2, random.Random(hash((qid, rep)) & 0xFFFF))
            trials += 1
            duplicates += len({p["persona_identifier"] for p in picked}) < 2
    assert trials >= 900 and duplicates == 0


# -- the confidence filter must see peers that state a confidence ---------
@pytest.mark.skipif(not (_ROOT / "results/processed/confidence_personas.parquet").exists(),
                    reason="confidence pool not present")
def test_filter_conditions_use_a_pool_whose_confidence_is_readable():
    """
    WRfilt used the main wrong pool, whose confidence was readable on 0 of
    1,500 personas, so the filter dropped every peer and WRfilt became R.
    """
    from src.contexts import apply_confidence_filter
    logging.disable(logging.CRITICAL)
    try:
        paths = yaml.safe_load((_ROOT / "config/paths.yaml").read_text(encoding="utf-8"))
        pools = load_pools(paths, _ROOT)
    finally:
        logging.disable(logging.NOTSET)
    cond = yaml.safe_load((_ROOT / "config/experiment.yaml").read_text(encoding="utf-8"))["conditions"]
    assert cond["WRfilt"]["peer_source"] == "anchored_confidence"
    assert cond["WRconf"]["peer_source"] == "anchored_confidence"
    assert "confidence_filter" not in cond["WRconf"]

    kept_any = units = 0
    for qid in list(pools.anchored_confidence)[:100]:
        peers, _ = build_peers(condition={**cond["WRfilt"], "_name": "WRfilt"},
                               question={"question_identifier": qid}, replicate=0,
                               pools=pools, weak_specs={"w": {"paper_name": "W"}},
                               master_seed=1, focal_key="f")
        assert len(peers) == 2
        kept, _ = apply_confidence_filter(peers, threshold=60, unparseable_counts_as="dropped")
        units += 1
        kept_any += bool(kept)
    assert kept_any / units > 0.9, f"filter left peers on only {kept_any}/{units} units"
