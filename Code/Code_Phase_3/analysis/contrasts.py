"""
contrasts.py — execute the pre-registered contrast families.

Reads the families declared in config/experiment.yaml (`analysis.contrast_families`)
and computes each one at the question level with a paired bootstrap CI and a
sign-flip permutation p-value, then Holm-corrects within the family.

Two design points the reviewers will check:

  * The capability gradient is computed on the EXCESS harmful-revision rate,
    H(WR) - H(R), not on H(WR) alone. Weak models revise more under any
    revision prompt; subtracting the matched re-answer baseline removes that
    common churn, so what remains is wrong-peer-specific. The raw version is
    reported alongside, as in the reviewed paper.
  * Both protocols are run. If the gradient survives under Protocol B it is
    the paper's headline; if it does not, the finding is that the published
    gradient was largely resampling noise, and the paper says so. The analysis
    is written so either outcome is reportable without changing the code.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import per_question_rate
from .stats import (
    ContrastResult,
    apply_family,
    paired_bootstrap_difference,
    paired_permutation_p,
    spearman_exact,
)

logger = logging.getLogger("platos_ship3.contrasts")

STAT_TO_RATE = {
    "accuracy_delta": "accuracy",
    "harmful_delta": "harmful_revision",
    "beneficial_delta": "beneficial_revision",
    "adoption_delta": "peer_target_adoption",
}


def _cell(registry: pd.DataFrame, protocol: str, scope: str, condition: str,
          focal_key: Optional[str] = None) -> pd.DataFrame:
    frame = registry[
        (registry["protocol"] == protocol)
        & (registry["dataset_scope"] == scope)
        & (registry["condition"] == condition)
    ]
    if focal_key is not None:
        frame = frame[frame["focal_key"] == focal_key]
    return frame


UNIT_KEYS = ["focal_key", "question_identifier", "replicate"]


def _match_units(left: pd.DataFrame, right: pd.DataFrame):
    """Restrict both arms to the units present in both."""
    def unit_index(frame):
        return pd.MultiIndex.from_frame(frame[UNIT_KEYS].astype(str))

    left_units, right_units = unit_index(left), unit_index(right)
    common = left_units.intersection(right_units)
    matched_left = left[left_units.isin(common)]
    matched_right = right[right_units.isin(common)]
    info = {
        "models_compared": ",".join(sorted(matched_left["focal_key"].unique())),
        "n_units_matched": int(len(common)),
        "n_units_dropped_a": int(len(left) - len(matched_left)),
        "n_units_dropped_b": int(len(right) - len(matched_right)),
    }
    return matched_left, matched_right, info


def paired_condition_contrast(
    registry: pd.DataFrame,
    name: str,
    statistic: str,
    condition_a: str,
    condition_b: str,
    protocol: str = "B",
    scope: str = "main300",
    focal_key: Optional[str] = None,
    n_resamples: int = 5000,
    seed: int = 20260502,
) -> ContrastResult:
    """One paired contrast A minus B at the question level."""
    rate_name = STAT_TO_RATE[statistic]
    left_frame = _cell(registry, protocol, scope, condition_a, focal_key)
    right_frame = _cell(registry, protocol, scope, condition_b, focal_key)

    if left_frame.empty or right_frame.empty:
        return ContrastResult(
            name=name, statistic=statistic, estimate=float("nan"),
            ci_low=float("nan"), ci_high=float("nan"), n_pairs=0,
            p_value=float("nan"), p_value_method="no_data",
            extra={"condition_a": condition_a, "condition_b": condition_b,
                   "protocol": protocol, "dataset_scope": scope,
                   "focal_key": focal_key or "ALL"},
        )

    # Both arms on the SAME (model, question, replicate) units. Conditions run
    # on different model sets (X2's conditions on three models, WR on eight),
    # and pairing by question alone compared a three-model mean with an
    # eight-model mean. Matching the units also keeps the shared Round-0
    # answer, so harmful-revision denominators are identical on both sides.
    left_frame, right_frame, matching = _match_units(left_frame, right_frame)
    left = per_question_rate(left_frame, rate_name)
    right = per_question_rate(right_frame, rate_name)
    estimate, low, high, n_pairs, differences = paired_bootstrap_difference(
        left, right, n_resamples=n_resamples, seed=seed)
    p_value = paired_permutation_p(differences, seed=seed)

    return ContrastResult(
        name=name, statistic=statistic, estimate=estimate, ci_low=low,
        ci_high=high, n_pairs=n_pairs, p_value=p_value,
        p_value_method="sign_flip_permutation",
        extra={
            "condition_a": condition_a, "condition_b": condition_b,
            "protocol": protocol, "dataset_scope": scope,
            "focal_key": focal_key or "ALL",
            **matching,
            "rate_a": float(np.nanmean(left)) if len(left) else float("nan"),
            "rate_b": float(np.nanmean(right)) if len(right) else float("nan"),
        },
    )


def capability_gradient(
    registry: pd.DataFrame,
    solo_accuracy: Dict[str, float],
    protocol: str = "B",
    scope: str = "main300",
    treatment: str = "WR",
    baseline: str = "R",
    subtract_baseline: bool = True,
) -> Dict[str, Any]:
    """
    Spearman(solo accuracy, harmful-revision excess) across focal models.

    `solo_accuracy` comes from the Round-0 cache — measured on the same items
    and replicates as the treatments, so "stronger model" has one operational
    definition rather than a parameter count or a role name.
    """
    treatment_frame = registry[
        (registry["protocol"] == protocol)
        & (registry["dataset_scope"] == scope)
        & (registry["condition"] == treatment)
    ]
    baseline_frame = registry[
        (registry["protocol"] == protocol)
        & (registry["dataset_scope"] == scope)
        & (registry["condition"] == baseline)
    ]
    if treatment_frame.empty:
        return {"rho": float("nan"), "n": 0, "p_value_method": "no_data"}

    rows = []
    for focal_key, group in treatment_frame.groupby("focal_key"):
        from .metrics import harmful

        treatment_harm = harmful(group).value
        baseline_group = baseline_frame[baseline_frame["focal_key"] == focal_key]
        baseline_harm = (
            harmful(baseline_group).value if not baseline_group.empty else np.nan
        )
        excess = (treatment_harm - baseline_harm) if subtract_baseline else treatment_harm
        rows.append(
            {
                "focal_key": focal_key,
                "solo_accuracy": solo_accuracy.get(focal_key, np.nan),
                "harmful_treatment": treatment_harm,
                "harmful_baseline": baseline_harm,
                "harmful_excess": excess,
            }
        )

    table = pd.DataFrame(rows).dropna(subset=["solo_accuracy"])
    if len(table) < 3:
        return {"rho": float("nan"), "n": int(len(table)),
                "p_value_method": "insufficient_n", "per_model": table}

    statistic_column = "harmful_excess" if subtract_baseline else "harmful_treatment"
    result = spearman_exact(table["solo_accuracy"], table[statistic_column])
    result.update(
        protocol=protocol,
        dataset_scope=scope,
        treatment=treatment,
        baseline=baseline if subtract_baseline else None,
        statistic_column=statistic_column,
        per_model=table.sort_values("solo_accuracy", ascending=False),
    )
    return result


def protocol_solo_accuracy(registry: pd.DataFrame, protocol: str,
                           scope: str = "main300") -> Dict[str, float]:
    """Per-model Round-0 accuracy from one protocol's own rows."""
    rows = registry[(registry["protocol"] == protocol)
                    & (registry["dataset_scope"] == scope)].dropna(
                        subset=["r0_is_correct"])
    if rows.empty:
        return {}
    return rows.groupby("focal_key")["r0_is_correct"].apply(
        lambda s: float(s.astype(bool).mean())).to_dict()


def run_family(
    registry: pd.DataFrame,
    family_name: str,
    specifications: List[Dict[str, Any]],
    solo_accuracy: Dict[str, float],
    scope: str = "main300",
    alpha: float = 0.05,
    n_resamples: int = 5000,
    seed: int = 20260502,
) -> tuple[List[ContrastResult], Dict[str, Any]]:
    """
    Run one declared family. Correlation entries are reported separately from
    the Holm family, because a correlation across models and a paired
    difference across questions are not exchangeable tests.
    """
    paired: List[ContrastResult] = []
    gradients: Dict[str, Any] = {}

    for spec in specifications:
        statistic = spec["stat"]
        if statistic in STAT_TO_RATE:
            paired.append(
                paired_condition_contrast(
                    registry, name=spec["name"], statistic=statistic,
                    condition_a=spec["a"], condition_b=spec["b"],
                    protocol=spec.get("protocol", "B"), scope=scope,
                    focal_key=spec.get("focal_key"),
                    n_resamples=n_resamples, seed=seed,
                )
            )
        elif statistic == "spearman_solo_vs_harmful_excess":
            protocol = spec.get("protocol", "B")
            # Each protocol's x-axis is measured within that protocol. The
            # Phase-3 Round-0 cache describes Protocol B's runs; Protocol A's
            # gradient reads Protocol A's own first answers instead.
            solo = (solo_accuracy if protocol == "B"
                    else protocol_solo_accuracy(registry, protocol, scope))
            gradients[spec["name"]] = capability_gradient(
                registry, solo, protocol=protocol, scope=scope,
                subtract_baseline=True,
            )
            gradients[f"{spec['name']}_raw"] = capability_gradient(
                registry, solo, protocol=protocol, scope=scope,
                subtract_baseline=False,
            )
        else:
            logger.warning("Unknown statistic '%s' in family %s.",
                           statistic, family_name)

    apply_family(paired, family_name, alpha=alpha)
    return paired, gradients


def per_model_contrasts(
    registry: pd.DataFrame,
    condition_a: str,
    condition_b: str,
    statistic: str = "accuracy_delta",
    protocol: str = "B",
    scope: str = "main300",
    n_resamples: int = 5000,
    seed: int = 20260502,
) -> pd.DataFrame:
    """
    The same contrast within each focal model — the per-model column of the
    main results table, with a CI per model rather than one pooled number.
    """
    focal_keys = sorted(
        registry[(registry["protocol"] == protocol)
                 & (registry["condition"] == condition_a)]["focal_key"]
        .dropna().unique()
    )
    results = []
    for focal_key in focal_keys:
        result = paired_condition_contrast(
            registry, name=f"{condition_a}_minus_{condition_b}:{focal_key}",
            statistic=statistic, condition_a=condition_a, condition_b=condition_b,
            protocol=protocol, scope=scope, focal_key=focal_key,
            n_resamples=n_resamples, seed=seed,
        )
        results.append(result.to_dict())
    return pd.DataFrame(results)
