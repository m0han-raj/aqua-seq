"""Leakage-safe preprocessing pipelines and the guideline-based rule baseline.

All preprocessing lives inside scikit-learn ``Pipeline`` objects so that imputation
statistics and scaling parameters are fitted on training folds only and applied
unchanged to held-out folds.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

ImputerName = Literal["median", "median_indicator", "knn", "knn_indicator"]

IMPUTER_CHOICES: tuple[ImputerName, ...] = (
    "median",
    "median_indicator",
    "knn",
    "knn_indicator",
)


def build_imputer(name: ImputerName) -> SimpleImputer | KNNImputer:
    """Create an imputer by name.

    Args:
        name: One of ``median``, ``median_indicator``, ``knn`` or ``knn_indicator``.
            The ``*_indicator`` variants append binary missing-indicator columns.

    Returns:
        An unfitted scikit-learn imputer.

    Raises:
        ValueError: If ``name`` is not a recognised strategy.
    """
    add_indicator = name.endswith("_indicator")
    if name.startswith("median"):
        return SimpleImputer(strategy="median", add_indicator=add_indicator)
    if name.startswith("knn"):
        return KNNImputer(n_neighbors=5, weights="distance", add_indicator=add_indicator)
    raise ValueError(f"Unknown imputer strategy: {name!r}")


def build_preprocessor(
    imputer: ImputerName = "median",
    scale: bool = True,
) -> ColumnTransformer:
    """Build the numeric preprocessing ColumnTransformer.

    Args:
        imputer: Imputation strategy to use.
        scale: Whether to standardise features. Tree ensembles do not need it;
            logistic regression and the KNN imputer do.

    Returns:
        An unfitted ``ColumnTransformer`` covering all nine physicochemical features.
    """
    steps: list[tuple[str, object]] = [("impute", build_imputer(imputer))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[("numeric", Pipeline(steps=steps), config.FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------------------
# Guideline-based rule baseline
# --------------------------------------------------------------------------------------
def rule_violation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag, per parameter, whether each sample breaches its verified limit.

    Missing measurements cannot breach a limit, so they are recorded as ``False``.
    Only parameters in :data:`config.PARAMETER_LIMITS` are considered; parameters in
    :data:`config.UNVERIFIED_PARAMETERS` are excluded by design.

    Args:
        frame: A dataframe containing the physicochemical feature columns.

    Returns:
        A boolean dataframe with one column per checked parameter.
    """
    violations = {}
    for name, limit in config.PARAMETER_LIMITS.items():
        values = frame[name]
        breach = pd.Series(False, index=frame.index)
        if limit.lower is not None:
            breach |= values < limit.lower
        if limit.upper is not None:
            breach |= values > limit.upper
        violations[name] = breach.fillna(False)
    return pd.DataFrame(violations, index=frame.index)


def rule_baseline_predict(frame: pd.DataFrame) -> np.ndarray:
    """Predict UNSAFE (1) when any verified guideline limit is breached.

    Args:
        frame: A dataframe containing the physicochemical feature columns.

    Returns:
        An integer array of predictions in the UNSAFE-as-positive encoding.
    """
    return rule_violation_matrix(frame).any(axis=1).astype(int).to_numpy()


def rule_baseline_score(frame: pd.DataFrame) -> np.ndarray:
    """Score samples by how many verified limits they breach.

    Used as a continuous "score" so the rule baseline can be placed on ROC and
    precision-recall curves alongside the models.

    Args:
        frame: A dataframe containing the physicochemical feature columns.

    Returns:
        A float array: the fraction of checked parameters that are breached.
    """
    matrix = rule_violation_matrix(frame)
    return (matrix.sum(axis=1) / matrix.shape[1]).to_numpy(dtype=float)
