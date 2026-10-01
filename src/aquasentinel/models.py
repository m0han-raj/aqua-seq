"""Model zoo and hyperparameter search spaces.

Every estimator is wrapped in a ``Pipeline`` whose first step is the preprocessing
``ColumnTransformer``, so imputation and scaling are refitted inside every CV fold.
Search spaces include the imputation strategy itself, so the imputer comparison is
performed under cross-validation rather than on the full training set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config, features

try:  # pragma: no cover - exercised implicitly by run_all
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    HAS_LIGHTGBM = False

try:  # pragma: no cover
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False


@dataclass
class ModelSpec:
    """A candidate model together with its randomised-search space.

    Attributes:
        name: Human-readable model name used in results files and the README.
        pipeline: The unfitted estimator pipeline.
        param_distributions: Search space keyed by pipeline parameter path.
        needs_scaling: Whether the model benefits from standardised features.
    """

    name: str
    pipeline: Pipeline
    param_distributions: dict[str, Any] = field(default_factory=dict)
    needs_scaling: bool = True


def _wrap(estimator: object, scale: bool) -> Pipeline:
    """Attach the preprocessing transformer in front of ``estimator``."""
    return Pipeline(
        steps=[
            ("preprocess", features.build_preprocessor(imputer="median", scale=scale)),
            ("model", estimator),
        ]
    )


# Imputation strategies are tuned as a hyperparameter, so the median-vs-KNN comparison
# happens inside cross-validation on the training set only.
_IMPUTER_SPACE = {"preprocess__numeric__impute": [features.build_imputer(n) for n in features.IMPUTER_CHOICES]}


def build_model_specs() -> list[ModelSpec]:
    """Build the full list of candidate models with their search spaces.

    Returns:
        A list of :class:`ModelSpec`, ordered from simplest to most complex. XGBoost
        and LightGBM entries are included only if the libraries are installed.
    """
    seed = config.RANDOM_SEED
    specs: list[ModelSpec] = []

    specs.append(
        ModelSpec(
            name="Logistic Regression",
            pipeline=_wrap(
                LogisticRegression(max_iter=5000, random_state=seed, solver="liblinear"),
                scale=True,
            ),
            param_distributions={
                **_IMPUTER_SPACE,
                "model__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
            },
        )
    )

    specs.append(
        ModelSpec(
            name="Random Forest",
            pipeline=_wrap(
                RandomForestClassifier(random_state=seed, n_jobs=-1), scale=False
            ),
            param_distributions={
                **_IMPUTER_SPACE,
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [None, 8, 12, 16, 24],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", 0.5],
                "model__class_weight": [None, "balanced", "balanced_subsample"],
            },
            needs_scaling=False,
        )
    )

    specs.append(
        ModelSpec(
            name="Gradient Boosting",
            pipeline=_wrap(GradientBoostingClassifier(random_state=seed), scale=False),
            param_distributions={
                **_IMPUTER_SPACE,
                "model__n_estimators": [200, 400, 600],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__max_depth": [2, 3, 4, 5],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__min_samples_leaf": [1, 5, 20],
            },
            needs_scaling=False,
        )
    )

    if HAS_XGBOOST:
        specs.append(
            ModelSpec(
                name="XGBoost",
                pipeline=_wrap(
                    XGBClassifier(
                        random_state=seed,
                        n_jobs=-1,
                        eval_metric="logloss",
                        tree_method="hist",
                    ),
                    scale=False,
                ),
                param_distributions={
                    **_IMPUTER_SPACE,
                    "model__n_estimators": [300, 500, 800],
                    "model__learning_rate": [0.02, 0.05, 0.1],
                    "model__max_depth": [3, 4, 6, 8],
                    "model__subsample": [0.7, 0.85, 1.0],
                    "model__colsample_bytree": [0.7, 0.85, 1.0],
                    "model__min_child_weight": [1, 3, 7],
                    # ~0.64 is the SAFE:UNSAFE ratio; 1.0 leaves the data as-is.
                    "model__scale_pos_weight": [1.0, 0.64],
                },
                needs_scaling=False,
            )
        )

    if HAS_LIGHTGBM:
        specs.append(
            ModelSpec(
                name="LightGBM",
                pipeline=_wrap(
                    LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1), scale=False
                ),
                param_distributions={
                    **_IMPUTER_SPACE,
                    "model__n_estimators": [300, 500, 800],
                    "model__learning_rate": [0.02, 0.05, 0.1],
                    "model__num_leaves": [15, 31, 63],
                    "model__max_depth": [-1, 4, 8],
                    "model__min_child_samples": [5, 20, 50],
                    "model__subsample": [0.7, 0.85, 1.0],
                    "model__colsample_bytree": [0.7, 0.85, 1.0],
                    "model__class_weight": [None, "balanced"],
                },
                needs_scaling=False,
            )
        )

    return specs


def build_dummy_classifier() -> Pipeline:
    """Build the majority-class dummy baseline.

    Returns:
        A pipeline that always predicts the most frequent training class.
    """
    return _wrap(
        DummyClassifier(strategy="most_frequent", random_state=config.RANDOM_SEED),
        scale=False,
    )
