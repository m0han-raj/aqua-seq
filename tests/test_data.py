"""Tests for schema validation, label semantics and the train/test split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquasentinel import config, data


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return data.load_raw()


def test_raw_file_exists_and_loads(raw: pd.DataFrame) -> None:
    assert len(raw) > 0
    assert set(raw.columns) == set(config.FEATURES) | {config.RAW_TARGET}


def test_expected_shape(raw: pd.DataFrame) -> None:
    # The published Kaggle dataset has 3,276 rows and 9 features plus the target.
    assert raw.shape == (3276, 10)


def test_target_is_binary(raw: pd.DataFrame) -> None:
    assert set(raw[config.RAW_TARGET].unique()) == {0, 1}
    assert not raw[config.RAW_TARGET].isna().any()


def test_validate_schema_rejects_missing_column(raw: pd.DataFrame) -> None:
    with pytest.raises(data.SchemaError, match="Missing expected columns"):
        data.validate_schema(raw.drop(columns=["ph"]))


def test_validate_schema_rejects_extra_column(raw: pd.DataFrame) -> None:
    with pytest.raises(data.SchemaError, match="Unexpected extra columns"):
        data.validate_schema(raw.assign(surprise=1.0))


def test_validate_schema_rejects_non_binary_target(raw: pd.DataFrame) -> None:
    broken = raw.copy()
    broken.loc[broken.index[0], config.RAW_TARGET] = 7
    with pytest.raises(data.SchemaError, match="must be binary"):
        data.validate_schema(broken)


def test_unsafe_label_is_the_complement_of_potability(raw: pd.DataFrame) -> None:
    """UNSAFE must be 1 exactly where Potability is 0. This mapping drives every metric."""
    unsafe = data.to_unsafe_label(raw)
    assert unsafe.name == config.TARGET
    assert (unsafe == (1 - raw[config.RAW_TARGET])).all()
    assert (unsafe[raw[config.RAW_TARGET] == 0] == 1).all()
    assert (unsafe[raw[config.RAW_TARGET] == 1] == 0).all()


def test_unsafe_is_the_majority_class(raw: pd.DataFrame) -> None:
    """A deliberate guard: 'unsafe' being the majority class shapes every baseline."""
    unsafe = data.to_unsafe_label(raw)
    assert unsafe.mean() > 0.5


def test_split_sizes_and_disjointness(raw: pd.DataFrame) -> None:
    split = data.make_split(raw)
    assert split.n_train + split.n_test == len(raw)
    assert abs(split.n_test / len(raw) - config.TEST_SIZE) < 0.01
    assert not set(split.x_train.index) & set(split.x_test.index)


def test_split_is_stratified(raw: pd.DataFrame) -> None:
    split = data.make_split(raw)
    overall = data.to_unsafe_label(raw).mean()
    assert abs(split.y_train.mean() - overall) < 0.01
    assert abs(split.y_test.mean() - overall) < 0.01


def test_split_is_deterministic(raw: pd.DataFrame) -> None:
    first, second = data.make_split(raw), data.make_split(raw)
    assert list(first.x_test.index) == list(second.x_test.index)
    np.testing.assert_array_equal(first.y_test.to_numpy(), second.y_test.to_numpy())


def test_missingness_report_matches_pandas(raw: pd.DataFrame) -> None:
    report = data.missingness_report(raw)
    for column in raw.columns:
        assert report.loc[column, "n_missing"] == raw[column].isna().sum()
    assert report["n_missing"].is_monotonic_decreasing
