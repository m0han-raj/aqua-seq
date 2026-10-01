"""Explainability: permutation importance and SHAP for the final tree model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from . import config, evaluate


def permutation_importance_frame(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 20,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """Compute permutation importance on held-out data.

    Features are permuted in their RAW form, before the preprocessing step, so the
    reported importance is attributable to the named physicochemical parameter.

    Args:
        pipeline: A fitted estimator pipeline.
        x: Held-out features.
        y: Held-out UNSAFE-encoded labels.
        n_repeats: Number of shuffles per feature.
        scoring: Scorer name passed to scikit-learn.

    Returns:
        A dataframe with ``feature``, ``importance_mean`` and ``importance_std``,
        sorted most to least important.
    """
    result = permutation_importance(
        pipeline,
        x,
        y,
        n_repeats=n_repeats,
        random_state=config.RANDOM_SEED,
        scoring=scoring,
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "feature": list(x.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _transformed_frame(pipeline: Pipeline, x: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted preprocessing step and return a named dataframe."""
    preprocessor = pipeline.named_steps["preprocess"]
    matrix = preprocessor.transform(x)
    try:
        names = list(preprocessor.get_feature_names_out())
    except (AttributeError, ValueError):  # pragma: no cover - defensive
        names = [f"f{i}" for i in range(matrix.shape[1])]
    return pd.DataFrame(matrix, columns=names, index=x.index)


def shap_summary(
    pipeline: Pipeline,
    x: pd.DataFrame,
    max_samples: int = 600,
) -> tuple[Any, pd.DataFrame] | None:
    """Compute SHAP values for a tree-based final model.

    Args:
        pipeline: A fitted pipeline whose ``model`` step is a tree ensemble.
        x: Raw features to explain.
        max_samples: Cap on the number of rows explained, for runtime.

    Returns:
        ``(shap_values, transformed_frame)`` where ``shap_values`` is the matrix for
        the UNSAFE class, or ``None`` if SHAP is unavailable or the model is not
        supported by :class:`shap.TreeExplainer`.
    """
    try:
        import shap
    except ImportError:  # pragma: no cover
        return None

    model = pipeline.named_steps["model"]
    sample = x.sample(
        n=min(max_samples, len(x)), random_state=config.RANDOM_SEED
    ).sort_index()
    transformed = _transformed_frame(pipeline, sample)

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
    except Exception:  # pragma: no cover - model type not supported by TreeExplainer
        return None

    values = np.asarray(values)
    if values.ndim == 3:
        # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        else:
            values = values[1]
    return values, transformed


def plot_shap_summary(values: np.ndarray, transformed: pd.DataFrame, name: str) -> Path:
    """Save a SHAP beeswarm summary plot.

    Args:
        values: SHAP values for the UNSAFE class.
        transformed: The preprocessed feature frame the values correspond to.
        name: Figure file stem.

    Returns:
        Path to the saved PNG.
    """
    import shap

    evaluate.apply_style()
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(values, transformed, show=False, plot_size=None)
    plt.title("AquaSentinel - SHAP summary (impact on predicted P(UNSAFE))", fontsize=12,
              fontweight="bold")
    plt.tight_layout()
    return evaluate.save_figure(fig, name)


def plot_shap_dependence(
    values: np.ndarray, transformed: pd.DataFrame, features: list[str], name: str
) -> Path:
    """Save SHAP dependence plots for selected features.

    Args:
        values: SHAP values for the UNSAFE class.
        transformed: The preprocessed feature frame the values correspond to.
        features: Column names to plot, one panel each.
        name: Figure file stem.

    Returns:
        Path to the saved PNG.
    """
    evaluate.apply_style()
    fig, axes = plt.subplots(1, len(features), figsize=(5.2 * len(features), 4.2))
    axes = np.atleast_1d(axes)
    for ax, feature in zip(axes, features):
        position = list(transformed.columns).index(feature)
        ax.scatter(
            transformed[feature],
            values[:, position],
            s=12,
            alpha=0.5,
            color=evaluate.SERIES_COLORS[0],
            edgecolors="none",
        )
        ax.axhline(0, color=evaluate.INK_MUTED, linewidth=1.2)
        ax.set_xlabel(f"{feature} (standardised / imputed)")
        ax.set_ylabel("SHAP value -> P(UNSAFE)")
        ax.set_title(feature, fontsize=11)
    fig.suptitle(
        "AquaSentinel - SHAP dependence for the two most important features",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return evaluate.save_figure(fig, name)
