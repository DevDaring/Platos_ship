"""
gate.py — the retention gap, defined to match the filter that is deployed.

Two faults in the original pre-flight test, both stated in the reviewed paper
and both repaired here.

FAULT 1 — wrong quantity. The original gate asked
    P(confidence >= threshold | wrong)
i.e. "are wrong peers loud?". It observed 0.995 and passed. But the partner
rate P(loud | correct) was 0.998, so confidence separated nothing. Loudness is
not discrimination.

FAULT 2 — wrong sign. `Code_Phase_2/CPU_Only/src/corrected_gate.py:83` computes
    gap = P(loud | wrong) - P(loud | correct)
and passes when that is sufficiently POSITIVE. The deployed filter RETAINS
high-confidence peers, so a positive value of that quantity means the filter
preferentially keeps WRONG peers — the harmful case. The manuscript defines the
useful quantity with the opposite sign:

    delta_ret = P(retained | correct) - P(retained | wrong)

A filter helps only when delta_ret is positive. This module computes that, and
nothing else, so code and paper cannot disagree again.

Also fixed: a substrate with only one correctness class (the wrong-anchored
pool, where every peer is wrong by construction) has an UNDEFINED gap. The
Phase-2 implementation substituted zero for the empty class and reported a
spurious gap of 0.99 with verdict "passed" on exactly the substrate where the
filter has nothing to work with. Here it returns `undefined_single_class`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("platos_ship3.gate")


@dataclass
class GateResult:
    substrate: str
    n_messages: int
    n_correct: int
    n_wrong: int
    n_unparseable_confidence: int
    retained_given_correct: Optional[float]
    retained_given_wrong: Optional[float]
    delta_retention: Optional[float]
    auroc_confidence_vs_correct: Optional[float]
    threshold: int
    delta_threshold: float
    auroc_floor: float
    verdict: str
    unparseable_counts_as: str
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """
    AUROC of confidence against correctness, `correct` as the positive class.

    Rank-based (handles ties). Returns None when either class is empty, which
    is the honest answer for a single-class substrate.
    """
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    mask = np.isfinite(score_array)
    score_array, label_array = score_array[mask], label_array[mask]
    n_positive = int((label_array == 1).sum())
    n_negative = int((label_array == 0).sum())
    if n_positive == 0 or n_negative == 0:
        return None

    from scipy.stats import rankdata

    ranks = rankdata(score_array)
    rank_sum_positive = ranks[label_array == 1].sum()
    return float(
        (rank_sum_positive - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def retention_gap(
    messages: pd.DataFrame,
    substrate: str,
    threshold: int = 60,
    delta_threshold: float = 0.10,
    auroc_floor: float = 0.60,
    unparseable_counts_as: str = "dropped",
    confidence_column: str = "peer_confidence",
    correct_column: str = "peer_is_correct",
) -> GateResult:
    """
    Compute delta_ret on one substrate.

    `unparseable_counts_as="dropped"` reproduces what the deployed rule does
    (a peer with no parseable confidence is removed). That accounting matters:
    in the released C5H all 22 unparseable peers happened to be wrong, which
    lifts delta_ret to +0.072 — still below the 0.10 threshold, and arising
    from a parsing failure rather than from confidence, so the rule cannot be
    tuned to exploit it. Both accountings are reported.
    """
    frame = messages.copy()
    if frame.empty:
        return GateResult(
            substrate=substrate, n_messages=0, n_correct=0, n_wrong=0,
            n_unparseable_confidence=0, retained_given_correct=None,
            retained_given_wrong=None, delta_retention=None,
            auroc_confidence_vs_correct=None, threshold=threshold,
            delta_threshold=delta_threshold, auroc_floor=auroc_floor,
            verdict="no_data", unparseable_counts_as=unparseable_counts_as,
        )

    confidence = pd.to_numeric(frame[confidence_column], errors="coerce")
    is_correct = frame[correct_column].astype("boolean")
    unparseable = confidence.isna()

    if unparseable_counts_as == "dropped":
        retained = confidence.fillna(-1) >= threshold
    else:
        retained = confidence.isna() | (confidence >= threshold)

    n_correct = int((is_correct == True).sum())      # noqa: E712 (nullable bool)
    n_wrong = int((is_correct == False).sum())       # noqa: E712

    if n_correct == 0 or n_wrong == 0:
        return GateResult(
            substrate=substrate, n_messages=int(len(frame)), n_correct=n_correct,
            n_wrong=n_wrong, n_unparseable_confidence=int(unparseable.sum()),
            retained_given_correct=None, retained_given_wrong=None,
            delta_retention=None, auroc_confidence_vs_correct=None,
            threshold=threshold, delta_threshold=delta_threshold,
            auroc_floor=auroc_floor, verdict="undefined_single_class",
            unparseable_counts_as=unparseable_counts_as,
            note=(
                "Only one correctness class present, so the retention gap is "
                "undefined rather than zero. The wrong-anchored substrate is "
                "single-class by construction: every peer is wrong."
            ),
        )

    retained_given_correct = float(retained[is_correct == True].mean())   # noqa: E712
    retained_given_wrong = float(retained[is_correct == False].mean())    # noqa: E712
    delta = retained_given_correct - retained_given_wrong

    parsed = ~unparseable
    auroc = _auroc(
        confidence[parsed].to_numpy(),
        (is_correct[parsed] == True).astype(int).to_numpy(),   # noqa: E712
    )

    passes_delta = delta > delta_threshold
    passes_auroc = auroc is not None and auroc > auroc_floor
    verdict = "passed" if (passes_delta and passes_auroc) else "failed"

    return GateResult(
        substrate=substrate, n_messages=int(len(frame)), n_correct=n_correct,
        n_wrong=n_wrong, n_unparseable_confidence=int(unparseable.sum()),
        retained_given_correct=retained_given_correct,
        retained_given_wrong=retained_given_wrong, delta_retention=delta,
        auroc_confidence_vs_correct=auroc, threshold=threshold,
        delta_threshold=delta_threshold, auroc_floor=auroc_floor,
        verdict=verdict, unparseable_counts_as=unparseable_counts_as,
        note=(
            "delta_ret = P(retained | correct) - P(retained | wrong). Positive "
            "means the filter preferentially keeps correct peers. Both the "
            "gap and the AUROC must clear their thresholds to pass."
        ),
    )


def legacy_loudness_gate(
    messages: pd.DataFrame,
    threshold: int = 60,
    confidence_column: str = "peer_confidence",
    correct_column: str = "peer_is_correct",
) -> Dict[str, Any]:
    """
    The ORIGINAL gate, reproduced so the paper can show what it would have
    licensed. Reported only as a contrast with the corrected test.
    """
    confidence = pd.to_numeric(messages[confidence_column], errors="coerce")
    is_correct = messages[correct_column].astype("boolean")
    loud = confidence.fillna(-1) >= threshold
    wrong_mask = is_correct == False       # noqa: E712
    loud_when_wrong = float(loud[wrong_mask].mean()) if wrong_mask.any() else None
    return {
        "metric": "P(confidence >= threshold | wrong)",
        "value": loud_when_wrong,
        "activation_threshold": 0.40,
        "verdict": ("passed" if (loud_when_wrong is not None
                                 and loud_when_wrong >= 0.40) else "failed"),
        "why_this_is_misspecified": (
            "It measures loudness, not discrimination, and its sign does not "
            "match a retain-high-confidence filter."
        ),
    }


def build_gate_report(
    peer_messages: pd.DataFrame,
    threshold: int = 60,
    delta_threshold: float = 0.10,
    auroc_floor: float = 0.60,
) -> pd.DataFrame:
    """
    Run the corrected gate on every substrate present, under both accountings
    of unparseable confidence.
    """
    if peer_messages.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for substrate, group in peer_messages.groupby("peer_source"):
        for accounting in ("dropped", "retained"):
            result = retention_gap(
                group, substrate=str(substrate), threshold=threshold,
                delta_threshold=delta_threshold, auroc_floor=auroc_floor,
                unparseable_counts_as=accounting,
            )
            rows.append(result.to_dict())
    return pd.DataFrame(rows)
