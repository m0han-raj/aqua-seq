"""Metrics, risk-aware threshold selection, bootstrap intervals and figures.

The positive class is UNSAFE throughout (see :data:`config.POSITIVE_LABEL`), because
a missed unsafe sample is the costly error this project optimises against.

Plot styling follows a validated categorical palette: hues are assigned in fixed slot
order and never cycled, every multi-series figure carries a legend, grids are
recessive, and bar charts are directly labelled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config

# --------------------------------------------------------------------------------------
# Styling -- validated categorical palette, fixed slot order, never cycled
# --------------------------------------------------------------------------------------
SERIES_COLORS: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8982"
GRID_COLOR = "#e3e2de"
SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "RdBu_r"


def apply_style() -> None:
    """Apply the project-wide matplotlib style (recessive grid, thin marks, ink text)."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK_PRIMARY,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "figure.dpi": config.FIGURE_DPI,
            "savefig.dpi": config.FIGURE_DPI,
            "savefig.bbox": "tight",
            "font.size": 10,
        }
    )
    for spine in ("top", "right"):
        plt.rcParams[f"axes.spines.{spine}"] = False


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save ``fig`` as a PNG into the figures directory and close it.

    Args:
        fig: The figure to save.
        name: File stem (no extension).

    Returns:
        The path the figure was written to.
    """
    config.ensure_output_dirs()
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ThresholdChoice:
    """The outcome of risk-aware threshold selection.

    Attributes:
        threshold: The probability cut-off applied to the UNSAFE score.
        target_recall: The recall on UNSAFE that was required.
        achieved_recall: Recall actually achieved at ``threshold`` in cross-validation.
        achieved_precision: Precision at ``threshold`` in cross-validation.
        flag_rate: Fraction of all samples flagged UNSAFE at ``threshold``.
        target_met: Whether ``target_recall`` was reachable at all.
    """

    threshold: float
    target_recall: float
    achieved_recall: float
    achieved_precision: float
    flag_rate: float
    target_met: bool


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the full metric set for UNSAFE-as-positive predictions.

    Args:
        y_true: True labels (1 == UNSAFE).
        y_score: Predicted probability of the UNSAFE class.
        threshold: Cut-off applied to ``y_score``.

    Returns:
        A dict of threshold-free metrics (ROC-AUC, PR-AUC, Brier) and
        threshold-dependent metrics (precision/recall/F1/balanced accuracy, counts).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0))),
        "threshold": float(threshold),
        "precision_unsafe": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_unsafe": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_unsafe": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()),
        "flag_rate": float(y_pred.mean()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "n_samples": int(len(y_true)),
    }


def select_threshold(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray,
    target_recall: float = config.TARGET_UNSAFE_RECALL,
) -> ThresholdChoice:
    """Pick the operating threshold that meets a recall target at the best precision.

    This must be called with CROSS-VALIDATED predictions on the TRAINING set only.
    Among all thresholds whose UNSAFE recall is at least ``target_recall``, the one
    with the highest precision is chosen. If the target is unreachable, the threshold
    maximising recall is returned and ``target_met`` is False.

    Args:
        y_true: True labels (1 == UNSAFE).
        y_score: Predicted probability of the UNSAFE class.
        target_recall: Minimum acceptable recall on the UNSAFE class.

    Returns:
        The chosen :class:`ThresholdChoice`.
    """
    y_true = np.asarray(y_true).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one more point than thresholds; drop the last.
    precision, recall = precision[:-1], recall[:-1]

    eligible = recall >= target_recall
    if eligible.any():
        best = int(np.argmax(np.where(eligible, precision, -np.inf)))
        target_met = True
    else:
        best = int(np.argmax(recall))
        target_met = False

    threshold = float(thresholds[best])
    flag_rate = float((y_score >= threshold).mean())
    return ThresholdChoice(
        threshold=threshold,
        target_recall=float(target_recall),
        achieved_recall=float(recall[best]),
        achieved_precision=float(precision[best]),
        flag_rate=flag_rate,
        target_met=target_met,
    )


def bootstrap_metric_cis(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray,
    threshold: float,
    n_boot: int = config.N_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = config.RANDOM_SEED,
) -> dict[str, dict[str, float]]:
    """Bootstrap percentile confidence intervals for the headline metrics.

    Args:
        y_true: True labels (1 == UNSAFE).
        y_score: Predicted probability of the UNSAFE class.
        threshold: Cut-off applied to ``y_score``.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level; 0.05 gives a 95% interval.
        seed: Seed for the resampling RNG.

    Returns:
        A mapping metric name -> ``{"lower": float, "upper": float}``.
    """
    y_true = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(seed)
    metric_names = (
        "roc_auc",
        "pr_auc",
        "precision_unsafe",
        "recall_unsafe",
        "f1_unsafe",
        "balanced_accuracy",
    )
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores = classification_metrics(y_true[idx], y_score[idx], threshold)
        for name in metric_names:
            samples[name].append(scores[name])

    lower_q, upper_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        name: {
            "lower": float(np.percentile(values, lower_q)),
            "upper": float(np.percentile(values, upper_q)),
        }
        for name, values in samples.items()
        if values
    }


def threshold_sweep(
    y_true: np.ndarray | pd.Series, y_score: np.ndarray, n_points: int = 200
) -> pd.DataFrame:
    """Tabulate precision, recall and flag rate across a grid of thresholds.

    Args:
        y_true: True labels (1 == UNSAFE).
        y_score: Predicted probability of the UNSAFE class.
        n_points: Number of thresholds to evaluate.

    Returns:
        A dataframe with one row per threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    grid = np.linspace(float(y_score.min()), float(y_score.max()), n_points)
    rows = []
    for threshold in grid:
        pred = (y_score >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "recall_unsafe": float(recall_score(y_true, pred, zero_division=0)),
                "precision_unsafe": float(precision_score(y_true, pred, zero_division=0)),
                "f1_unsafe": float(f1_score(y_true, pred, zero_division=0)),
                "flag_rate": float(pred.mean()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def plot_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], name: str) -> Path:
    """Plot ROC curves for several models on one axis.

    Args:
        curves: Mapping model name -> ``(y_true, y_score)``.
        name: Figure file stem.

    Returns:
        Path to the saved PNG.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, (label, (y_true, y_score)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), y_score)
        auc = roc_auc_score(np.asarray(y_true).astype(int), y_score)
        ax.plot(fpr, tpr, color=SERIES_COLORS[i % len(SERIES_COLORS)], label=f"{label} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linestyle="--", linewidth=1.2, label="Chance (AUC 0.500)")
    ax.set_xlabel("False positive rate (safe samples flagged unsafe)")
    ax.set_ylabel("True positive rate (unsafe samples caught)")
    ax.set_title("AquaSentinel - ROC curves, UNSAFE as positive class")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    return save_figure(fig, name)


def plot_pr_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], prevalence: float, name: str
) -> Path:
    """Plot precision-recall curves for several models on one axis.

    Args:
        curves: Mapping model name -> ``(y_true, y_score)``.
        prevalence: Base rate of the UNSAFE class, drawn as the no-skill line.
        name: Figure file stem.

    Returns:
        Path to the saved PNG.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, (label, (y_true, y_score)) in enumerate(curves.items()):
        y_true = np.asarray(y_true).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color=SERIES_COLORS[i % len(SERIES_COLORS)], label=f"{label} (AP {ap:.3f})")
    ax.axhline(
        prevalence,
        color=INK_MUTED,
        linestyle="--",
        linewidth=1.2,
        label=f"No skill = UNSAFE base rate ({prevalence:.3f})",
    )
    ax.set_xlabel("Recall on UNSAFE")
    ax.set_ylabel("Precision on UNSAFE")
    ax.set_title("AquaSentinel - Precision-recall curves, UNSAFE as positive class")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    return save_figure(fig, name)


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray, title: str, name: str
) -> Path:
    """Plot a 2x2 confusion matrix with counts and row percentages.

    Args:
        y_true: True labels (1 == UNSAFE).
        y_pred: Binary predictions.
        title: Figure title.
        name: Figure file stem.

    Returns:
        Path to the saved PNG.
    """
    apply_style()
    matrix = confusion_matrix(np.asarray(y_true).astype(int), y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.imshow(matrix, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=matrix.max())
    ax.grid(False)

    labels = [config.NEGATIVE_CLASS_NAME, config.POSITIVE_CLASS_NAME]
    ax.set_xticks([0, 1], labels=[f"Predicted\n{label}" for label in labels])
    ax.set_yticks([0, 1], labels=[f"Actual\n{label}" for label in labels])

    row_totals = matrix.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            pct = matrix[i, j] / row_totals[i, 0] * 100 if row_totals[i, 0] else 0.0
            ax.text(
                j,
                i,
                f"{matrix[i, j]:,}\n{pct:.1f}% of row",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=INK_PRIMARY if matrix[i, j] < matrix.max() * 0.6 else SURFACE,
            )
    ax.set_title(title)
    return save_figure(fig, name)


def plot_threshold_tradeoff(
    sweep: pd.DataFrame, chosen: ThresholdChoice, name: str
) -> Path:
    """Plot recall, precision and flag rate against the decision threshold.

    Args:
        sweep: Output of :func:`threshold_sweep`.
        chosen: The selected threshold, marked with a vertical rule.
        name: Figure file stem.

    Returns:
        Path to the saved PNG.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(sweep["threshold"], sweep["recall_unsafe"], color=SERIES_COLORS[0], label="Recall on UNSAFE")
    ax.plot(sweep["threshold"], sweep["precision_unsafe"], color=SERIES_COLORS[1], label="Precision on UNSAFE")
    ax.plot(
        sweep["threshold"],
        sweep["flag_rate"],
        color=SERIES_COLORS[2],
        linestyle=":",
        label="Share of all samples flagged for lab testing",
    )
    ax.axhline(chosen.target_recall, color=INK_MUTED, linestyle="--", linewidth=1.2,
               label=f"Recall target ({chosen.target_recall:.2f})")
    ax.axvline(chosen.threshold, color=INK_PRIMARY, linewidth=1.5,
               label=f"Chosen threshold ({chosen.threshold:.3f})")
    ax.set_xlabel("Decision threshold on predicted P(UNSAFE)")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("AquaSentinel - Risk-aware threshold trade-off (cross-validated, training set)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    return save_figure(fig, name)


def plot_calibration(
    y_true: np.ndarray | pd.Series, y_score: np.ndarray, label: str, name: str
) -> Path:
    """Plot a reliability (calibration) curve.

    Args:
        y_true: True labels (1 == UNSAFE).
        y_score: Predicted probability of the UNSAFE class.
        label: Series label for the legend.
        name: Figure file stem.

    Returns:
        Path to the saved PNG.
    """
    apply_style()
    prob_true, prob_pred = calibration_curve(
        np.asarray(y_true).astype(int), np.clip(y_score, 0, 1), n_bins=10, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linestyle="--", linewidth=1.2, label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, color=SERIES_COLORS[0], marker="o", markersize=8, label=label)
    ax.set_xlabel("Mean predicted P(UNSAFE)")
    ax.set_ylabel("Observed fraction UNSAFE")
    ax.set_title("AquaSentinel - Calibration of predicted unsafe probability")
    ax.legend(loc="upper left")
    return save_figure(fig, name)


def plot_model_comparison(frame: pd.DataFrame, name: str) -> Path:
    """Plot a bar chart of cross-validated ROC-AUC with standard-deviation error bars.

    Args:
        frame: Columns ``model``, ``cv_roc_auc_mean``, ``cv_roc_auc_std``.
        name: Figure file stem.

    Returns:
        Path to the saved PNG.
    """
    apply_style()
    ordered = frame.sort_values("cv_roc_auc_mean")
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(ordered) + 2.2))
    positions = np.arange(len(ordered))
    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(ordered))]
    ax.barh(
        positions,
        ordered["cv_roc_auc_mean"],
        xerr=ordered["cv_roc_auc_std"],
        color=colors,
        height=0.62,
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.2, "capsize": 3},
    )
    ax.axvline(0.5, color=INK_MUTED, linestyle="--", linewidth=1.2)
    ax.text(0.5, len(ordered) - 0.35, " chance", color=INK_MUTED, fontsize=9, va="center")
    for pos, (mean, std) in enumerate(
        zip(ordered["cv_roc_auc_mean"], ordered["cv_roc_auc_std"])
    ):
        ax.text(mean + std + 0.012, pos, f"{mean:.3f} ± {std:.3f}", va="center",
                fontsize=9, color=INK_PRIMARY)
    ax.set_yticks(positions, labels=ordered["model"])
    ax.set_xlabel("Cross-validated ROC-AUC (5-fold x 3 repeats, training set)")
    ax.set_xlim(0.4, max(0.78, float(ordered["cv_roc_auc_mean"].max()) + 0.12))
    ax.set_title("AquaSentinel - Model comparison")
    ax.grid(axis="y", visible=False)
    return save_figure(fig, name)


def plot_target_balance(y: pd.Series, name: str) -> Path:
    """Plot the class balance of the UNSAFE target.

    Args:
        y: The UNSAFE-encoded target series.
        name: Figure file stem.

    Returns:
        Path to the saved PNG.
    """
    apply_style()
    counts = y.value_counts().sort_index()
    labels = [config.NEGATIVE_CLASS_NAME, config.POSITIVE_CLASS_NAME]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bars = ax.bar(labels, counts.to_numpy(), color=[SERIES_COLORS[0], SERIES_COLORS[1]], width=0.55)
    total = int(counts.sum())
    for bar, count in zip(bars, counts.to_numpy()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.012,
            f"{count:,}\n({count / total:.1%})",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("Number of samples")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_title("AquaSentinel - Class balance (UNSAFE = Potability 0)")
    ax.grid(axis="x", visible=False)
    return save_figure(fig, name)


def plot_missingness(report: pd.DataFrame, name: str) -> Path:
    """Plot percentage of missing values per column.

    Args:
        report: Output of :func:`aquasentinel.data.missingness_report`.
        name: Figure file stem.

    Returns:
        Path to the saved PNG.
    """
    apply_style()
    present = report[report["n_missing"] > 0].sort_values("pct_missing")
    fig, ax = plt.subplots(figsize=(7, 0.55 * max(len(present), 3) + 2))
    positions = np.arange(len(present))
    ax.barh(positions, present["pct_missing"], color=SERIES_COLORS[0], height=0.55)
    for pos, (count, pct) in enumerate(zip(present["n_missing"], present["pct_missing"])):
        ax.text(pct + 0.4, pos, f"{pct:.1f}%  ({count:,} rows)", va="center", fontsize=9)
    ax.set_yticks(positions, labels=present.index)
    ax.set_xlabel("Missing values (% of rows)")
    ax.set_xlim(0, float(present["pct_missing"].max()) * 1.45)
    ax.set_title("AquaSentinel - Missing values by column")
    ax.grid(axis="y", visible=False)
    return save_figure(fig, name)


def plot_feature_distributions(
    frame: pd.DataFrame, y: pd.Series, columns: Sequence[str], name: str
) -> Path:
    """Plot per-feature density split by class.

    Args:
        frame: Feature dataframe.
        y: UNSAFE-encoded target.
        columns: Features to plot, one panel each.
        name: Figure file stem.

    Returns:
        Path to the saved PNG.
    """
    apply_style()
    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.1 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, column in zip(axes, columns):
        for cls, color, label in (
            (0, SERIES_COLORS[0], config.NEGATIVE_CLASS_NAME),
            (1, SERIES_COLORS[1], config.POSITIVE_CLASS_NAME),
        ):
            values = frame.loc[y == cls, column].dropna()
            ax.hist(values, bins=40, density=True, alpha=0.55, color=color, label=label)
        ax.set_title(f"{column}  [{config.FEATURE_UNITS.get(column, '')}]", fontsize=10)
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    for ax in axes[len(columns) :]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        "AquaSentinel - Feature distributions by class (density)", y=1.045, fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return save_figure(fig, name)


def plot_correlation_heatmap(frame: pd.DataFrame, name: str) -> Path:
    """Plot the feature correlation matrix on a diverging scale.

    Args:
        frame: Numeric dataframe to correlate.
        name: Figure file stem.

    Returns:
        Path to the saved PNG.
    """
    apply_style()
    corr = frame.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6.8))
    image = ax.imshow(corr, cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(corr)), labels=corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), labels=corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color=SURFACE if abs(value) > 0.55 else INK_PRIMARY,
            )
    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("AquaSentinel - Feature correlation matrix")
    return save_figure(fig, name)


def plot_permutation_importance(frame: pd.DataFrame, name: str) -> Path:
    """Plot permutation importance with standard-deviation error bars.

    Args:
        frame: Columns ``feature``, ``importance_mean``, ``importance_std``.
        name: Figure file stem.

    Returns:
        Path to the saved PNG.
    """
    apply_style()
    ordered = frame.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(ordered) + 2))
    positions = np.arange(len(ordered))
    ax.barh(
        positions,
        ordered["importance_mean"],
        xerr=ordered["importance_std"],
        color=SERIES_COLORS[0],
        height=0.58,
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.2, "capsize": 3},
    )
    ax.axvline(0, color=INK_MUTED, linewidth=1.2)
    ax.set_yticks(positions, labels=ordered["feature"])
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled (test set)")
    ax.set_title("AquaSentinel - Permutation importance of the final model")
    ax.grid(axis="y", visible=False)
    return save_figure(fig, name)


def threshold_choice_to_dict(choice: ThresholdChoice) -> dict[str, float | bool]:
    """Convert a :class:`ThresholdChoice` into a JSON-serialisable dict."""
    return asdict(choice)
