"""Data loading, schema validation and the single train/test split.

The split is created once, deterministically, from ``config.RANDOM_SEED``. Every
downstream step (cross-validation, tuning, threshold selection) must use the TRAIN
frame only; the TEST frame is touched exactly once, in the final evaluation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


class SchemaError(ValueError):
    """Raised when the raw CSV does not match the expected Water Potability schema."""


@dataclass(frozen=True)
class SplitData:
    """The one and only train/test split used by the project.

    ``y_train`` / ``y_test`` use the UNSAFE-as-positive encoding (1 = unsafe).
    """

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def n_train(self) -> int:
        """Number of training rows."""
        return int(len(self.x_train))

    @property
    def n_test(self) -> int:
        """Number of test rows."""
        return int(len(self.x_test))


def load_raw(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the raw Water Potability CSV and validate its schema.

    Args:
        csv_path: Override for the CSV location. Defaults to ``config.RAW_CSV``.

    Returns:
        The raw dataframe, exactly as stored on disk.

    Raises:
        FileNotFoundError: If the CSV is missing, with instructions on how to get it.
        SchemaError: If the columns do not match the expected schema.
    """
    path = csv_path or config.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            f"Download it from {config.DATASET_URL} and place "
            f"water_potability.csv in {config.RAW_DATA_DIR}.\n"
            "See data/README.md for step-by-step instructions."
        )
    frame = pd.read_csv(path)
    validate_schema(frame)
    return frame


def validate_schema(frame: pd.DataFrame) -> None:
    """Check that ``frame`` has the expected columns and dtypes.

    Args:
        frame: Candidate raw dataframe.

    Raises:
        SchemaError: If a column is missing, unexpected, or not numeric, or if the
            target is not binary 0/1.
    """
    expected = set(config.FEATURES) | {config.RAW_TARGET}
    actual = set(frame.columns)
    if missing := expected - actual:
        raise SchemaError(f"Missing expected columns: {sorted(missing)}")
    if unexpected := actual - expected:
        raise SchemaError(f"Unexpected extra columns: {sorted(unexpected)}")

    for column in config.FEATURES:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise SchemaError(f"Feature {column!r} is not numeric ({frame[column].dtype}).")

    target_values = set(frame[config.RAW_TARGET].dropna().unique().tolist())
    if not target_values <= {0, 1}:
        raise SchemaError(
            f"{config.RAW_TARGET} must be binary 0/1, found {sorted(target_values)}."
        )
    if frame[config.RAW_TARGET].isna().any():
        raise SchemaError(f"{config.RAW_TARGET} contains missing values.")


def to_unsafe_label(frame: pd.DataFrame) -> pd.Series:
    """Convert the raw ``Potability`` column into the UNSAFE-as-positive target.

    ``Potability`` is 1 for potable (safe) water, so the unsafe label is its complement.

    Args:
        frame: A dataframe containing the raw target column.

    Returns:
        An integer series named ``config.TARGET`` where 1 == UNSAFE.
    """
    return (1 - frame[config.RAW_TARGET]).astype(int).rename(config.TARGET)


def make_split(frame: pd.DataFrame | None = None) -> SplitData:
    """Build the stratified 80/20 train/test split with a fixed seed.

    Args:
        frame: Raw dataframe. Loaded from disk when omitted.

    Returns:
        A :class:`SplitData` holding features and UNSAFE-encoded targets.
    """
    data = load_raw() if frame is None else frame
    features = data[config.FEATURES].copy()
    target = to_unsafe_label(data)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=target,
    )
    return SplitData(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def missingness_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing values per column.

    Args:
        frame: Any dataframe.

    Returns:
        A dataframe indexed by column with ``n_missing`` and ``pct_missing`` columns,
        sorted from most to least missing.
    """
    counts = frame.isna().sum()
    report = pd.DataFrame(
        {
            "n_missing": counts.astype(int),
            "pct_missing": (counts / len(frame) * 100).round(2),
        }
    )
    return report.sort_values("n_missing", ascending=False)
