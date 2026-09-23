"""
make_tables.py — emit the paper's LaTeX data tables from the registry.

The manuscript \\input{}s these, so a table cannot drift from the logs. House
LaTeX rules (CLAUDE.md): booktabs rules only, no vertical bars, caption above
the table, wrapped in adjustbox, \\small or \\footnotesize past 7 columns, and
every \\label has a matching \\ref in the prose.

Tables emitted:
  tab_coverage      the common condition matrix (Reviewer YVDD W1)
  tab_main          per-model accuracy and revision rates under each condition
  tab_contrasts     the pre-registered families with CIs and Holm-adjusted p
  tab_gradient      the eight-model capability gradient, both protocols
  tab_decomposition X2: what in the message moves the model
  tab_voting        budget-matched baselines (Choi 2025, Zhang 2025)
  tab_gate          the retention gap, corrected definition
  tab_safeguard     X6 policies against their baselines
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger("platos_ship3.make_tables")

CONDITION_LABELS = {
    "R": "Re-answer (no peers)",
    "G": "Generic challenge",
    "W": "Bare wrong answers",
    "WR": "Two wrong rationales",
    "WRh": "Two hedged wrong rationales",
    "SF": "Wrong rationales, unattributed",
    "H": "Two honest weak peers",
    "E": "Two self-samples",
    "CR": "One wrong, one correct",
    "WR1": "One wrong rationale",
    "WR4": "Four wrong rationales",
    "WRagree": "Two wrong, same target",
    "WRdiff": "Two wrong, different targets",
    "WRfilt": "Two wrong, confidence-filtered",
    "Hfilt": "Two honest, confidence-filtered",
}


def _escape(text: Any) -> str:
    out = str(text)
    for old, new in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        out = out.replace(old, new)
    return out


def _fmt(value: Any, digits: int = 1, percent: bool = True) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    if percent:
        return f"{100 * float(value):.{digits}f}"
    return f"{float(value):.{digits}f}"


def _ci(low: Any, high: Any, digits: int = 1, percent: bool = True) -> str:
    if low is None or high is None:
        return "--"
    if not (np.isfinite(low) and np.isfinite(high)):
        return "--"
    return f"[{_fmt(low, digits, percent)}, {_fmt(high, digits, percent)}]"


def _wrap(body: str, caption: str, label: str, columns: int,
          use_adjustbox: bool = True) -> str:
    size = ""
    if columns > 9:
        size = "\\footnotesize\n"
    elif columns > 7:
        size = "\\small\n"
    open_box = "\\begin{adjustbox}{max width=\\linewidth}\n" if use_adjustbox else ""
    close_box = "\\end{adjustbox}\n" if use_adjustbox else ""
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"{size}"
        f"\\caption{{{caption}\\label{{{label}}}}}\n"
        f"{open_box}{body}{close_box}"
        "\\end{table}\n"
    )


def _tabular(header: List[str], rows: List[List[str]], align: str) -> str:
    lines = [f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _write(tables_dir: Path, name: str, content: str) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / f"{name}.tex"
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote %s", path)


# ──────────────────────────────────────────────────────────────────────────

def table_coverage(coverage: pd.DataFrame, tables_dir: Path) -> None:
    """
    The condition matrix, per protocol.

    Emitted for whichever protocols are present. Under Protocol A it is the
    evidence for Reviewer YVDD's W1: the re-answer control exists for two
    models and the honest-peer control for one, while the sweep models have
    neither. Under Protocol B every cell is filled identically.
    """
    if coverage.empty:
        return
    for protocol in sorted(coverage["protocol"].dropna().unique()):
        _table_coverage_one(coverage, str(protocol), tables_dir)


def _table_coverage_one(coverage: pd.DataFrame, protocol: str,
                        tables_dir: Path) -> None:
    protocol_b = coverage[coverage["protocol"] == protocol]
    if protocol_b.empty:
        return
    pivot = protocol_b.pivot_table(
        index="focal_key", columns="condition",
        values="n_units", aggfunc="sum", fill_value=0,
    )
    header = ["Focal model"] + [_escape(c) for c in pivot.columns] + ["Reps"]
    reps = protocol_b.groupby("focal_key")["n_replicates"].max()
    rows = [
        [_escape(model)] + [str(int(v)) for v in pivot.loc[model]]
        + [str(int(reps.get(model, 0)))]
        for model in pivot.index
    ]
    body = _tabular(header, rows, "l" + "r" * (len(header) - 1))
    if protocol == "B":
        caption = ("Condition matrix under Protocol~B. Every focal model "
                   "receives the same questions, the same replication count, "
                   "and the same revision template; only the peer block "
                   "differs.")
        name, label = "tab_coverage", "tab:coverage"
    else:
        caption = ("Condition matrix in the previously reported design "
                   "(Protocol~A). Coverage is unequal: the re-answer control "
                   "exists for two focal models, the honest-peer and "
                   "split-peer controls for one, and the remaining models are "
                   "run on a reduced protocol at a lower replication count.")
        name, label = "tab_coverage_legacy", "tab:coverage_legacy"
    _write(tables_dir, name, _wrap(body, caption, label, len(header)))


def table_main(metrics: pd.DataFrame, tables_dir: Path) -> None:
    """
    Accuracy, harmful and beneficial revision per (model, condition).

    Protocol B when it exists; otherwise the legacy Protocol-A numbers, so the
    table is always available and the A-vs-B comparison can be made directly.
    """
    if metrics.empty:
        return
    protocol = "B" if (metrics["protocol"] == "B").any() else "A"
    selected = metrics[
        (metrics["protocol"] == protocol)
        & (metrics["dataset_scope"] == "main300")
    ]
    if selected.empty:
        return
    header = ["Focal model", "Condition", "Acc.", "H (C$\\to$I)", "B (I$\\to$C)",
              "Adopt", "$n$ units"]
    rows = []
    for _, row in selected.sort_values(["focal_key", "condition"]).iterrows():
        rows.append([
            _escape(row["focal_key"]),
            _escape(CONDITION_LABELS.get(row["condition"], row["condition"])),
            _fmt(row["accuracy"]),
            f"{_fmt(row['harmful_revision'])} ({int(row['harmful_revision_denominator'])})",
            f"{_fmt(row['beneficial_revision'])} ({int(row['beneficial_revision_denominator'])})",
            _fmt(row.get("peer_target_adoption")),
            str(int(row["n_units"])),
        ])
    body = _tabular(header, rows, "ll" + "r" * 5)
    _write(tables_dir, "tab_main", _wrap(
        body,
        "Protocol~" + protocol
        + " outcomes. H is $P(\\text{final wrong}\\mid\\text{initial "
        "correct})$ and B is $P(\\text{final correct}\\mid\\text{initial "
        "wrong})$; each denominator is given in brackets. Adopt is the rate "
        "of revising to the peer-asserted answer among initially correct units.",
        "tab:main", len(header)))


def table_contrasts(contrasts: pd.DataFrame, tables_dir: Path) -> None:
    """The pre-registered families with Holm-adjusted p-values."""
    if contrasts.empty:
        return
    header = ["Family", "Contrast", "Estimate (pp)", "95\\% CI", "$n$ pairs",
              "$p$ raw", "$p$ Holm"]
    rows = []
    for _, row in contrasts.iterrows():
        rows.append([
            _escape(row.get("family", "")),
            _escape(row["name"]),
            _fmt(row["estimate"]),
            _ci(row["ci_low"], row["ci_high"]),
            str(int(row["n_pairs"])),
            _fmt(row["p_value"], 4, percent=False),
            _fmt(row.get("p_holm"), 4, percent=False),
        ])
    body = _tabular(header, rows, "ll" + "r" * 5)
    _write(tables_dir, "tab_contrasts", _wrap(
        body,
        "Pre-registered contrasts. Estimates are question-level paired "
        "differences in percentage points with percentile bootstrap intervals "
        "(5{,}000 resamples); $p$ values are sign-flip permutation tests, "
        "Holm-corrected within the declared family.",
        "tab:contrasts", len(header)))


def table_gradient(gradients: Dict[str, Any], tables_dir: Path) -> None:
    """The eight-model gradient under both protocols."""
    rows, header = [], ["Focal model", "Solo acc.", "H under WR",
                        "H under R", "Excess", "Protocol"]
    for name, gradient in gradients.items():
        table = gradient.get("per_model")
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        protocol = gradient.get("protocol", "?")
        for _, row in table.sort_values("solo_accuracy", ascending=False).iterrows():
            rows.append([
                _escape(row["focal_key"]),
                _fmt(row["solo_accuracy"]),
                _fmt(row["harmful_treatment"]),
                _fmt(row["harmful_baseline"]),
                _fmt(row["harmful_excess"]),
                _escape(protocol),
            ])
    if not rows:
        return
    body = _tabular(header, rows, "l" + "r" * 4 + "c")
    caption_parts = []
    for name, gradient in gradients.items():
        if "rho" in gradient and np.isfinite(gradient.get("rho", np.nan)):
            caption_parts.append(
                f"{_escape(name)}: $\\rho = {gradient['rho']:.2f}$, "
                f"$p = {gradient['p_value']:.3f}$ ({_escape(gradient['p_value_method'])})"
            )
    _write(tables_dir, "tab_gradient", _wrap(
        body,
        "Harmful revision against focal ability. Excess is H under two wrong "
        "rationales minus H under the matched re-answer baseline, which "
        "removes the revision churn common to every condition. "
        + "; ".join(caption_parts) + ".",
        "tab:gradient", len(header)))


def table_decomposition(metrics: pd.DataFrame, tables_dir: Path) -> None:
    """X2: which part of the peer message does the work. Protocol B only."""
    order = ["R", "G", "W", "WR", "WRh", "SF", "H"]
    if metrics.empty or not (metrics["protocol"] == "B").any():
        return
    selected = metrics[
        (metrics["protocol"] == "B")
        & (metrics["dataset_scope"] == "main300")
        & (metrics["condition"].isin(order))
    ]
    if selected.empty:
        return
    pivot_accuracy = selected.pivot_table(index="focal_key", columns="condition",
                                          values="accuracy")
    pivot_harm = selected.pivot_table(index="focal_key", columns="condition",
                                      values="harmful_revision")
    columns = [c for c in order if c in pivot_accuracy.columns]
    header = ["Focal model", "Metric"] + [_escape(c) for c in columns]
    rows = []
    for model in pivot_accuracy.index:
        rows.append([_escape(model), "Accuracy"]
                    + [_fmt(pivot_accuracy.loc[model, c]) for c in columns])
        rows.append(["", "H (C$\\to$I)"]
                    + [_fmt(pivot_harm.loc[model, c]) if c in pivot_harm.columns
                       else "--" for c in columns])
    body = _tabular(header, rows, "ll" + "r" * len(columns))
    _write(tables_dir, "tab_decomposition", _wrap(
        body,
        "Decomposing the peer message. R is a matched re-answer, G a "
        "content-free challenge, W bare wrong answers, WR wrong rationales, "
        "WRh the same rationales in tentative wording, SF the same content "
        "without peer attribution, and H natural weak peers.",
        "tab:decomposition", len(header)))


def table_voting(voting: pd.DataFrame, tables_dir: Path) -> None:
    """Self-consistency against debate, with the call budget attached."""
    if voting.empty:
        return
    header = ["Focal model", "Task", "Single sample", "Plurality vote",
              "$\\Delta$", "Calls"]
    rows = []
    for _, row in voting.iterrows():
        rows.append([
            _escape(row["focal_key"]),
            _escape(row.get("source_dataset", "all")),
            _fmt(row["single_sample_accuracy"]),
            _fmt(row["plurality_vote_accuracy"]),
            _fmt(row["vote_minus_single"]),
            str(int(row["n_calls_vote"])),
        ])
    body = _tabular(header, rows, "ll" + "r" * 4)
    _write(tables_dir, "tab_voting", _wrap(
        body,
        "Self-consistency baseline computed from the cached initial answers, "
        "reported per task rather than pooled, with the number of focal calls "
        "each strategy spends.",
        "tab:voting", len(header)))


def table_gate(gate_report: pd.DataFrame, tables_dir: Path) -> None:
    """The retention gap under both accountings of unparseable confidence."""
    if gate_report.empty:
        return
    header = ["Substrate", "Unparsed counted as", "$P(\\text{ret}\\mid C)$",
              "$P(\\text{ret}\\mid W)$", "$\\Delta_{\\mathrm{ret}}$", "AUROC",
              "Verdict"]
    rows = []
    for _, row in gate_report.iterrows():
        rows.append([
            _escape(row["substrate"]),
            _escape(row["unparseable_counts_as"]),
            _fmt(row["retained_given_correct"], 1),
            _fmt(row["retained_given_wrong"], 1),
            _fmt(row["delta_retention"], 3, percent=False),
            _fmt(row["auroc_confidence_vs_correct"], 2, percent=False),
            _escape(row["verdict"]),
        ])
    body = _tabular(header, rows, "ll" + "r" * 4 + "l")
    _write(tables_dir, "tab_gate", _wrap(
        body,
        "Retention gap $\\Delta_{\\mathrm{ret}} = P(\\text{retained}\\mid"
        "\\text{correct}) - P(\\text{retained}\\mid\\text{wrong})$, which is "
        "the quantity a retain-high-confidence filter needs to be positive. "
        "A substrate with one correctness class is undefined rather than zero.",
        "tab:gate", len(header)))


def table_safeguard(safeguard: pd.DataFrame, tables_dir: Path) -> None:
    """X6 policies, with the oracle marked as an upper bound."""
    if safeguard.empty:
        return
    header = ["Policy", "Deployable", "Accuracy", "95\\% CI", "H (C$\\to$I)",
              "Adoption of changes"]
    rows = []
    for _, row in safeguard.iterrows():
        rows.append([
            _escape(row["policy"]),
            "yes" if row["is_deployable"] else "no (bound)",
            _fmt(row["accuracy"]),
            _ci(row["accuracy_ci_low"], row["accuracy_ci_high"]),
            _fmt(row["harmful_revision"]),
            _fmt(row["adoption_rate_of_changes"]),
        ])
    body = _tabular(header, rows, "ll" + "r" * 4)
    _write(tables_dir, "tab_safeguard", _wrap(
        body,
        "Verification safeguard against its baselines. The random policy "
        "adopts changes at the verifier's own coverage, so adopting fewer "
        "changes cannot by itself explain a difference. The oracle row is an "
        "upper bound and is not a deployable rule.",
        "tab:safeguard", len(header)))


def make_all_tables(
    project_root: Path,
    registry: pd.DataFrame,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    contrasts: pd.DataFrame,
    gradients: Dict[str, Any],
    voting: pd.DataFrame,
    gate_report: pd.DataFrame,
    safeguard: pd.DataFrame,
) -> None:
    import yaml

    with open(project_root / "config" / "paths.yaml") as handle:
        paths = yaml.safe_load(handle)
    tables_dir = Path(paths["tables_directory"])
    if not tables_dir.is_absolute():
        tables_dir = project_root / paths["tables_directory"]

    table_coverage(coverage, tables_dir)
    table_main(metrics, tables_dir)
    table_contrasts(contrasts, tables_dir)
    table_gradient(gradients, tables_dir)
    table_decomposition(metrics, tables_dir)
    table_voting(voting, tables_dir)
    table_gate(gate_report, tables_dir)
    table_safeguard(safeguard, tables_dir)
    logger.info("Tables written to %s", tables_dir)
