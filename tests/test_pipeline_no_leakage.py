"""Tests proving that preprocessing never sees data outside its training fold.

These are the tests that make the headline numbers trustworthy: if imputation or
scaling statistics were computed over the full dataset, every reported metric would
be optimistically biased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from aqua_seq import config, data, features


@pytest.fixture(scope="module")
def split() -> data.SplitData:
    return data.make_split()


def _fitted_imputer(pipeline_or_ct: object) -> object:
    """Pull the fitted imputer out of a fitted ColumnTransformer."""
    return pipeline_or_ct.named_transformers_["numeric"].named_steps["impute"]


def test_median_imputer_statistics_come_from_training_data_only(
    split: data.SplitData,
) -> None:
    preprocessor = features.build_preprocessor(imputer="median", scale=False)
    preprocessor.fit(split.x_train)
    learned = _fitted_imputer(preprocessor).statistics_

    train_medians = split.x_train[config.FEATURES].median().to_numpy()
    np.testing.assert_allclose(learned, train_medians, rtol=1e-9)

    full = pd.concat([split.x_train, split.x_test])[config.FEATURES].median().to_numpy()
    assert not np.allclose(
        learned, full
    ), "Imputer statistics match the full-dataset medians, which indicates leakage."


def test_scaler_statistics_come_from_training_data_only(split: data.SplitData) -> None:
    preprocessor = features.build_preprocessor(imputer="median", scale=True)
    preprocessor.fit(split.x_train)
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]

    imputed_train = _fitted_imputer(preprocessor).transform(split.x_train[config.FEATURES])
    np.testing.assert_allclose(scaler.mean_, imputed_train.mean(axis=0), rtol=1e-9)


def test_transforming_test_data_does_not_update_fitted_statistics(
    split: data.SplitData,
) -> None:
    """Calling transform() must be read-only with respect to learned parameters."""
    preprocessor = features.build_preprocessor(imputer="median", scale=True)
    preprocessor.fit(split.x_train)
    before = _fitted_imputer(preprocessor).statistics_.copy()

    preprocessor.transform(split.x_test)
    after = _fitted_imputer(preprocessor).statistics_

    np.testing.assert_array_equal(before, after)


def test_extreme_test_values_cannot_influence_the_fitted_pipeline(
    split: data.SplitData,
) -> None:
    """Poison the test set; a leakage-free pipeline is completely unaffected."""
    preprocessor = features.build_preprocessor(imputer="median", scale=True)
    preprocessor.fit(split.x_train)
    clean = preprocessor.transform(split.x_test)
    statistics_before = _fitted_imputer(preprocessor).statistics_.copy()

    poisoned = split.x_test.copy()
    poisoned.loc[:, config.FEATURES] = poisoned[config.FEATURES] * 1e6

    preprocessor.transform(poisoned)
    np.testing.assert_array_equal(statistics_before, _fitted_imputer(preprocessor).statistics_)
    np.testing.assert_allclose(clean, preprocessor.transform(split.x_test))


def test_preprocessing_is_inside_the_estimator_pipeline() -> None:
    """Every candidate model must carry its preprocessing, so CV refits it per fold."""
    from aqua_seq import models

    for spec in models.build_model_specs():
        assert isinstance(spec.pipeline, Pipeline)
        assert spec.pipeline.steps[0][0] == "preprocess", (
            f"{spec.name} does not start with a preprocessing step; "
            "fitting it outside the pipeline would leak across CV folds."
        )


def test_imputation_strategy_is_tuned_inside_cross_validation() -> None:
    """The median-vs-KNN comparison must be a pipeline parameter, not a pre-step."""
    from aqua_seq import models

    for spec in models.build_model_specs():
        assert (
            "preprocess__numeric__impute" in spec.param_distributions
        ), f"{spec.name} does not search the imputer inside CV."


def test_cross_val_score_runs_without_touching_the_test_set(
    split: data.SplitData,
) -> None:
    """A sanity check that the pipeline is CV-ready and produces plausible scores."""
    from sklearn.linear_model import LogisticRegression

    pipeline = Pipeline(
        steps=[
            ("preprocess", features.build_preprocessor(imputer="median", scale=True)),
            ("model", LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)),
        ]
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_SEED)
    scores = cross_val_score(pipeline, split.x_train, split.y_train, cv=cv, scoring="roc_auc")
    assert len(scores) == 3
    assert np.all(
        (scores > 0.3) & (scores < 0.9)
    ), f"Implausible CV scores {scores}; a value near 1.0 would suggest leakage."
