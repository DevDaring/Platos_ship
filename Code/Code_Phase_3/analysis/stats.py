"""
stats.py — inference at the question level.

Rules this module enforces (ANALYSIS_PLAN.md):

  1. The independent unit is the QUESTION. 1,500 trials are not 1,500
     independent observations; replicates of one question are collapsed
     first, then paired across conditions.
  2. Paired contrasts use the SAME questions on both sides, dropping a
     question when either side has an empty denominator (so a harmful-revision
     contrast is computed over questions with an initially-correct replicate
     in both conditions).
  3. Confidence intervals are question-level bootstrap percentile intervals.
  4. Multiplicity is Holm, applied within a declared family; every raw p-value
     is also returned so the correction can be redone under another grouping.
  5. Small-n correlations (8 focal models) use an EXACT permutation p-value,
     never the asymptotic one.
  6. No non-inferiority or "safety" claim is available from this module by
     construction — there is no such function.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("platos_ship3.stats")


@dataclass
class ContrastResult:
    name: str
    statistic: str
    estimate: float
    ci_low: float
    ci_high: float
    n_pairs: int
    p_value: float
    p_value_method: str
    family: str = ""
    p_holm: Optional[float] = None
    significant_after_holm: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "statistic": self.statistic,
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_pairs": self.n_pairs,
            "p_value": self.p_value,
            "p_value_method": self.p_value_method,
            "family": self.family,
            "p_holm": self.p_holm,
            "significant_after_holm": self.significant_after_holm,
        }
        payload.update(self.extra)
        return payload


# ──────────────────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 20260502,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> Tuple[float, float, float]:
    """Percentile bootstrap over the question-level values. (est, low, high)."""
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                       dtype=float)
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    if array.size == 1:
        value = float(statistic(array))
        return value, value, value

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(int(n_resamples), array.size))
    draws = np.apply_along_axis(statistic, 1, array[indices])
    return (
        float(statistic(array)),
        float(np.percentile(draws, 100 * alpha / 2)),
        float(np.percentile(draws, 100 * (1 - alpha / 2))),
    )


def paired_bootstrap_difference(
    left: pd.Series,
    right: pd.Series,
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 20260502,
) -> Tuple[float, float, float, int, np.ndarray]:
    """
    Question-level paired difference (left - right).

    Returns (estimate, ci_low, ci_high, n_pairs, per_question_differences).
    Questions missing from either side, or NaN on either side, are dropped.
    """
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if joined.empty:
        return float("nan"), float("nan"), float("nan"), 0, np.array([])
    differences = (joined["left"] - joined["right"]).to_numpy(dtype=float)
    estimate, low, high = bootstrap_ci(differences, n_resamples, alpha, seed)
    return estimate, low, high, int(len(differences)), differences


def paired_permutation_p(
    differences: np.ndarray, n_resamples: int = 10000, seed: int = 20260502
) -> float:
    """
    Two-sided sign-flip permutation p-value for a paired difference.

    Exchangeability under the null is 'the sign of each question's difference
    is arbitrary', which is the right null for a within-question paired design.
    """
    array = np.asarray(differences, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    observed = abs(array.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(int(n_resamples), array.size))
    null = np.abs((signs * array).mean(axis=1))
    # +1 in numerator and denominator: never report p = 0 from a finite sample.
    return float((np.sum(null >= observed) + 1) / (n_resamples + 1))


# ──────────────────────────────────────────────────────────────────────────
# Correlation with an exact permutation p-value (the 8-model gradient)
# ──────────────────────────────────────────────────────────────────────────

def spearman_exact(
    x: Sequence[float], y: Sequence[float], max_exact_n: int = 10,
    n_resamples: int = 100000, seed: int = 20260502,
) -> Dict[str, Any]:
    """
    Spearman rho with an EXACT permutation p-value when n is small.

    With eight focal models the asymptotic p-value is not trustworthy; 8! =
    40,320 permutations is cheap, so enumerate them. Also returns
    leave-one-out rho, because a correlation over eight points must be shown
    not to hinge on a single model.
    """
    from scipy.stats import rankdata, spearmanr

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[mask], y_array[mask]
    n = x_array.size
    if n < 3:
        return {"rho": float("nan"), "p_value": float("nan"), "n": int(n),
                "p_value_method": "insufficient_n"}

    rho = float(spearmanr(x_array, y_array).statistic)
    x_ranks, y_ranks = rankdata(x_array), rankdata(y_array)

    def _rho_from(perm_ranks: np.ndarray) -> float:
        return float(np.corrcoef(x_ranks, perm_ranks)[0, 1])

    if n <= max_exact_n:
        null = [_rho_from(np.asarray(perm))
                for perm in itertools.permutations(y_ranks)]
        null = np.asarray(null)
        p_value = float(np.mean(np.abs(null) >= abs(rho)))
        method = f"exact_permutation_{len(null)}"
    else:
        rng = np.random.default_rng(seed)
        null = np.asarray([
            _rho_from(rng.permutation(y_ranks)) for _ in range(int(n_resamples))
        ])
        p_value = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_resamples + 1))
        method = f"monte_carlo_permutation_{n_resamples}"

    leave_one_out = []
    for index in range(n):
        keep = np.ones(n, dtype=bool)
        keep[index] = False
        if keep.sum() >= 3:
            leave_one_out.append(float(spearmanr(x_array[keep], y_array[keep]).statistic))

    return {
        "rho": rho,
        "p_value": p_value,
        "p_value_method": method,
        "n": int(n),
        "leave_one_out_min": float(np.min(leave_one_out)) if leave_one_out else float("nan"),
        "leave_one_out_max": float(np.max(leave_one_out)) if leave_one_out else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────────────
# Multiplicity
# ──────────────────────────────────────────────────────────────────────────

def holm_correct(p_values: Sequence[float], alpha: float = 0.05
                 ) -> Tuple[List[float], List[bool]]:
    """
    Holm-Bonferroni step-down. Returns (adjusted_p, reject_flags) in the
    original order. NaN p-values pass through as NaN and never reject.
    """
    values = list(p_values)
    indexed = [(i, p) for i, p in enumerate(values)
               if p is not None and np.isfinite(p)]
    adjusted = [float("nan")] * len(values)
    reject = [False] * len(values)
    if not indexed:
        return adjusted, reject

    indexed.sort(key=lambda kv: kv[1])
    m = len(indexed)
    running_max = 0.0
    for rank, (original_index, p) in enumerate(indexed):
        candidate = min(1.0, (m - rank) * p)
        running_max = max(running_max, candidate)   # enforce monotonicity
        adjusted[original_index] = running_max
        reject[original_index] = running_max <= alpha
    return adjusted, reject


def apply_family(results: List[ContrastResult], family: str,
                 alpha: float = 0.05) -> List[ContrastResult]:
    """Tag a list of contrasts as one family and Holm-correct within it."""
    adjusted, reject = holm_correct([r.p_value for r in results], alpha)
    for result, p_holm, is_significant in zip(results, adjusted, reject):
        result.family = family
        result.p_holm = p_holm
        result.significant_after_holm = bool(is_significant)
    return results


# ──────────────────────────────────────────────────────────────────────────
# GEE (secondary; reported with odds ratios and intervals, not p alone)
# ──────────────────────────────────────────────────────────────────────────

def gee_logistic(
    frame: pd.DataFrame,
    outcome: str = "is_correct",
    predictor: str = "condition",
    cluster: str = "question_identifier",
    reference_level: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    GEE logistic regression clustered by question, exchangeable correlation.

    Secondary to the paired bootstrap. Returns None (with a log line) when
    statsmodels is unavailable or the fit does not converge, rather than
    letting the pipeline die on an optional analysis.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        logger.warning("statsmodels not installed; skipping GEE.")
        return None

    data = frame[[outcome, predictor, cluster]].dropna().copy()
    if data.empty or data[predictor].nunique() < 2:
        return None
    data[outcome] = data[outcome].astype(int)

    if reference_level and reference_level in set(data[predictor]):
        formula = (f"{outcome} ~ C({predictor}, "
                   f"Treatment(reference='{reference_level}'))")
    else:
        formula = f"{outcome} ~ C({predictor})"

    try:
        model = smf.gee(
            formula, groups=cluster, data=data,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        fit = model.fit()
    except Exception as exc:
        logger.warning("GEE fit failed (%s); skipping.", exc)
        return None

    summary = pd.DataFrame({
        "term": fit.params.index,
        "coefficient": fit.params.values,
        "std_error": fit.bse.values,
        "z": fit.tvalues.values,
        "p_value": fit.pvalues.values,
        "odds_ratio": np.exp(fit.params.values),
        "or_ci_low": np.exp(fit.conf_int()[0].values),
        "or_ci_high": np.exp(fit.conf_int()[1].values),
    })
    summary["n_clusters"] = data[cluster].nunique()
    summary["n_observations"] = len(data)
    return summary


def mcnemar_question_level(left: pd.Series, right: pd.Series) -> Dict[str, Any]:
    """
    Continuity-corrected McNemar on question-level majority correctness.

    Kept because the reviewed paper reports it and the revision must be
    comparable; the bootstrap contrast remains primary.
    """
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if joined.empty:
        return {"n_pairs": 0, "b": 0, "c": 0, "chi2": float("nan"),
                "p_value": float("nan")}
    left_correct = joined["left"] >= 0.5
    right_correct = joined["right"] >= 0.5
    b = int((left_correct & ~right_correct).sum())
    c = int((~left_correct & right_correct).sum())
    if b + c == 0:
        return {"n_pairs": int(len(joined)), "b": b, "c": c,
                "chi2": float("nan"), "p_value": 1.0}

    from scipy.stats import chi2 as chi2_dist

    chi2_statistic = (abs(b - c) - 1) ** 2 / (b + c)
    return {
        "n_pairs": int(len(joined)),
        "b": b,
        "c": c,
        "chi2": float(chi2_statistic),
        "p_value": float(chi2_dist.sf(chi2_statistic, 1)),
    }
