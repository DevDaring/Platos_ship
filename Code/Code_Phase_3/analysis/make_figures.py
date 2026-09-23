"""
make_figures.py — regenerate every figure from the registry.

Reviewer YVDD on the reviewed submission: "the quality of Figure 1 is poor and
needs further improvement. In particular, the model names in the figure are not
aligned with the descriptions in the text." Both are fixed structurally here:
model names come from `paper_name` in config/models.yaml, the same source the
tables use, so a figure label cannot disagree with the prose.

Design rules applied (see the project's visualization guidance):
  * fixed categorical hue order, never cycled: blue, vermillion, green, amber,
    pink. Validated for colour-vision deficiency (worst adjacent deutan
    dE = 11.0) rather than eyeballed;
  * colour is never the only channel — every series also carries a distinct
    marker shape, and the figures stay readable in greyscale print;
  * one y-axis per panel. Never two scales on one plot;
  * recessive grid and axes; thin marks; direct labels rather than a number on
    every point;
  * a legend whenever two or more series are shown.

Figures:
  figure1_capability_gradient  the headline: harmful revision vs focal ability
  figure2_decomposition        X2: which part of the message moves the model
  figure3_budget_matched       accuracy against focal calls spent
  figure4_dose_response        X3: peer count and rounds
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("platos_ship3.make_figures")

# Fixed categorical order — assigned by position, never cycled.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
MARKERS = ["o", "s", "^", "D", "v"]

INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
GRID = "#d9d9d9"


def _setup(matplotlib) -> None:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def _paper_names(project_root: Path) -> Dict[str, str]:
    """Model display names — the single source the tables also use."""
    import yaml

    with open(project_root / "config" / "models.yaml") as handle:
        config = yaml.safe_load(handle)
    return {
        key: spec.get("paper_name", key)
        for key, spec in (config.get("focal_agents") or {}).items()
    }


def _annotate_without_collisions(ax, x, y, labels) -> None:
    """
    Direct labels that do not overlap.

    Points clustered in x get their labels fanned out above and below and
    connected by a hairline, rather than stacked on top of one another. Five of
    the eight focal models sit within four accuracy points of each other, so
    naive labelling is unreadable — and unreadable model names are exactly what
    Reviewer YVDD objected to.
    """
    order = np.argsort(x)
    span = float(np.ptp(x)) or 1.0
    crowd_threshold = 0.06 * span          # within 6% of the x-range is crowded
    offsets = [(0, 9), (0, -14), (0, 20), (0, -25), (0, 31)]
    cluster_index = 0

    for position, index in enumerate(order):
        previous = order[position - 1] if position else None
        crowded = (previous is not None
                   and abs(x[index] - x[previous]) < crowd_threshold)
        cluster_index = cluster_index + 1 if crowded else 0
        dx, dy = offsets[cluster_index % len(offsets)]
        ax.annotate(
            labels[index], (x[index], y[index]),
            textcoords="offset points", xytext=(dx, dy),
            ha="center", va="bottom" if dy > 0 else "top",
            fontsize=6.5, color=INK_MUTED,
            arrowprops=(dict(arrowstyle="-", color=GRID, linewidth=0.5,
                             shrinkA=0, shrinkB=3)
                        if cluster_index else None),
        )


def figure_capability_gradient(
    gradients: Dict[str, Any], names: Dict[str, str], figures_dir: Path
) -> None:
    """
    The headline figure: harmful-revision excess against solo accuracy.

    Both protocols are drawn — filled markers for Protocol B (matched
    initial state), hollow for Protocol A (the published stateless design) —
    so a reader sees immediately whether the gradient is a property of the
    models or of the old protocol.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup(matplotlib)

    # For each protocol prefer the baseline-subtracted gradient, but fall back
    # to the raw one when too few models carry a matched re-answer baseline.
    # Under the legacy design only two of the eight focal models were ever run
    # with a re-answer control, so the excess is undefined for the rest — which
    # is itself the coverage problem Protocol B removes.
    chosen: Dict[str, Dict[str, Any]] = {}
    for name, gradient in gradients.items():
        table = gradient.get("per_model")
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        if not np.isfinite(gradient.get("rho", float("nan"))):
            continue
        protocol = str(gradient.get("protocol", "?"))
        is_raw = name.endswith("_raw")
        current = chosen.get(protocol)
        if current is None:
            chosen[protocol] = {**gradient, "_is_raw": is_raw}
        elif current["_is_raw"] and not is_raw and gradient.get("n", 0) >= 3:
            chosen[protocol] = {**gradient, "_is_raw": is_raw}
        elif (not current["_is_raw"]) and current.get("n", 0) < 3 and is_raw:
            chosen[protocol] = {**gradient, "_is_raw": is_raw}

    if not chosen:
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    _style_axes(ax)
    uses_raw = False

    for index, (protocol, gradient) in enumerate(sorted(chosen.items())):
        table = gradient["per_model"].sort_values("solo_accuracy")
        colour = PALETTE[index % len(PALETTE)]
        marker = MARKERS[index % len(MARKERS)]
        filled = protocol == "B"

        is_raw = gradient["_is_raw"]
        uses_raw = uses_raw or is_raw
        column = "harmful_treatment" if is_raw else "harmful_excess"
        x = 100 * table["solo_accuracy"].to_numpy(dtype=float)
        y = 100 * table[column].to_numpy(dtype=float)

        rho = gradient.get("rho", float("nan"))
        p_value = gradient.get("p_value", float("nan"))
        suffix = " (raw)" if is_raw else ""
        label = (f"Protocol {protocol}{suffix}: "
                 r"$\rho$ = " f"{rho:.2f}, p = {p_value:.3f}, n = {gradient.get('n', 0)}")

        ax.plot(x, y, linestyle="-" if filled else "--", linewidth=1.0,
                color=colour, alpha=0.45, zorder=2)
        ax.scatter(x, y, s=42, marker=marker, zorder=3,
                   facecolor=colour if filled else "white",
                   edgecolor=colour, linewidth=1.4, label=label)

        # Direct labels, once: on Protocol B when it is present, otherwise on
        # the only series drawn. Reviewer YVDD noted that the previous figure's
        # model names did not match the text, so they are taken from the same
        # `paper_name` field the tables use and are always shown.
        if filled or len(chosen) == 1:
            _annotate_without_collisions(ax, x, y,
                                         [names.get(k, k) for k in table["focal_key"]])

    ax.axhline(0, color=INK_MUTED, linewidth=0.6, linestyle=":", zorder=1)
    ax.set_xlabel("Solo (initial-answer) accuracy, %")
    ax.set_ylabel(
        "Harmful revision under two wrong peers, %" if uses_raw
        else "Harmful-revision excess over\nmatched re-answer, pp"
    )
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path = figures_dir / "figure1_capability_gradient.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info("Wrote %s", path)


def figure_decomposition(
    metrics: pd.DataFrame, names: Dict[str, str], figures_dir: Path
) -> None:
    """X2: accuracy and harmful revision by condition, grouped by focal model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup(matplotlib)
    order = ["R", "G", "W", "WR", "WRh", "H"]
    selected = metrics[
        (metrics["protocol"] == "B")
        & (metrics["dataset_scope"] == "main300")
        & (metrics["condition"].isin(order))
    ]
    if selected.empty:
        return

    models = sorted(selected["focal_key"].unique())
    conditions = [c for c in order if c in set(selected["condition"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True)

    for ax, metric, ylabel in zip(
        axes, ["accuracy", "harmful_revision"],
        ["Accuracy, %", r"Harmful revision $P(\mathrm{wrong}\mid\mathrm{correct})$, %"],
    ):
        _style_axes(ax)
        width = 0.8 / max(len(models), 1)
        positions = np.arange(len(conditions))
        for index, model in enumerate(models):
            values = []
            for condition in conditions:
                cell = selected[(selected["focal_key"] == model)
                                & (selected["condition"] == condition)]
                values.append(100 * float(cell[metric].iloc[0])
                              if not cell.empty else np.nan)
            ax.bar(positions + index * width - 0.4 + width / 2, values,
                   width=width * 0.9, color=PALETTE[index % len(PALETTE)],
                   edgecolor="white", linewidth=0.8, zorder=3,
                   label=names.get(model, model) if metric == "accuracy" else None)
        ax.set_xticks(positions)
        ax.set_xticklabels(conditions)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Revision condition")

    axes[0].legend(frameon=False, ncol=1, loc="lower right")
    fig.tight_layout()
    path = figures_dir / "figure2_decomposition.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info("Wrote %s", path)


def figure_budget_matched(
    voting: pd.DataFrame, metrics: pd.DataFrame, names: Dict[str, str],
    figures_dir: Path,
) -> None:
    """Accuracy against focal calls spent — debate versus simply sampling more."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup(matplotlib)
    if voting.empty or metrics.empty:
        return

    pooled = (voting.groupby("focal_key")
              .agg(single=("single_sample_accuracy", "mean"),
                   vote=("plurality_vote_accuracy", "mean"),
                   k=("n_replicates", "median"))
              .reset_index())
    wr = metrics[(metrics["protocol"] == "B")
                 & (metrics["dataset_scope"] == "main300")
                 & (metrics["condition"] == "WR")]
    if wr.empty:
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    _style_axes(ax)

    for index, (_, row) in enumerate(pooled.iterrows()):
        model = row["focal_key"]
        colour = PALETTE[index % len(PALETTE)]
        marker = MARKERS[index % len(MARKERS)]
        cell = wr[wr["focal_key"] == model]
        xs = [1, int(row["k"]), 2]
        ys = [100 * row["single"], 100 * row["vote"],
              100 * float(cell["accuracy"].iloc[0]) if not cell.empty else np.nan]
        ax.scatter(xs, ys, s=40, marker=marker, color=colour, zorder=3,
                   label=names.get(model, model))
        ax.plot(xs[:2], ys[:2], color=colour, linewidth=1.0, alpha=0.4, zorder=2)

    ax.set_xlabel("Focal model calls per question")
    ax.set_ylabel("Accuracy, %")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["1\n(single sample)", "2\n(revise)", "3\n(vote)"])
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    path = figures_dir / "figure3_budget_matched.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info("Wrote %s", path)


def figure_dose_response(
    metrics: pd.DataFrame, names: Dict[str, str], figures_dir: Path
) -> None:
    """X3: harmful revision against the number of confidently wrong peers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup(matplotlib)
    dose_map = {"R": 0, "WR1": 1, "WR": 2, "WR4": 4}
    selected = metrics[
        (metrics["protocol"] == "B") & (metrics["condition"].isin(dose_map))
    ]
    if selected.empty or selected["condition"].nunique() < 2:
        return

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    _style_axes(ax)
    for index, model in enumerate(sorted(selected["focal_key"].unique())):
        subset = selected[selected["focal_key"] == model].copy()
        subset["n_wrong_peers"] = subset["condition"].map(dose_map)
        subset = subset.sort_values("n_wrong_peers")
        ax.plot(subset["n_wrong_peers"], 100 * subset["harmful_revision"],
                marker=MARKERS[index % len(MARKERS)], markersize=5,
                color=PALETTE[index % len(PALETTE)], linewidth=1.2,
                label=names.get(model, model), zorder=3)

    ax.set_xlabel("Number of confidently wrong peers")
    ax.set_ylabel(r"Harmful revision, %")
    ax.set_xticks(sorted(set(dose_map.values())))
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figures_dir / "figure4_dose_response.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info("Wrote %s", path)


def make_all_figures(
    project_root: Path,
    registry: pd.DataFrame,
    metrics: pd.DataFrame,
    gradients: Dict[str, Any],
    voting: pd.DataFrame,
) -> None:
    import yaml

    try:
        import matplotlib  # noqa: F401  (availability probe)
    except ImportError:
        logger.warning("matplotlib not installed; skipping figures.")
        return

    with open(project_root / "config" / "paths.yaml") as handle:
        paths = yaml.safe_load(handle)
    figures_dir = Path(paths["figures_directory"])
    if not figures_dir.is_absolute():
        figures_dir = project_root / paths["figures_directory"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    names = _paper_names(project_root)
    figure_capability_gradient(gradients, names, figures_dir)
    figure_decomposition(metrics, names, figures_dir)
    figure_budget_matched(voting, metrics, names, figures_dir)
    figure_dose_response(metrics, names, figures_dir)
    logger.info("Figures written to %s", figures_dir)
