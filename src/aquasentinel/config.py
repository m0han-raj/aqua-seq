"""Central configuration: paths, seeds, label semantics and guideline limits.

Every guideline limit below carries the source name and URL it was taken from.
Limits that could not be verified against an authoritative source are deliberately
absent (see ``UNVERIFIED_PARAMETERS``) and are therefore excluded from the rule
baseline. Nothing here is guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
RAW_CSV: Final[Path] = RAW_DATA_DIR / "water_potability.csv"

RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
METRICS_DIR: Final[Path] = RESULTS_DIR / "metrics"
FIGURES_DIR: Final[Path] = RESULTS_DIR / "figures"

RANDOM_SEED: Final[int] = 42
TEST_SIZE: Final[float] = 0.20
CV_FOLDS: Final[int] = 5
CV_REPEATS: Final[int] = 3
N_SEARCH_ITER: Final[int] = 40
N_BOOTSTRAP: Final[int] = 2000
FIGURE_DPI: Final[int] = 150

RAW_TARGET: Final[str] = "Potability"
TARGET: Final[str] = "unsafe"
POSITIVE_LABEL: Final[int] = 1
POSITIVE_CLASS_NAME: Final[str] = "UNSAFE"
NEGATIVE_CLASS_NAME: Final[str] = "SAFE"

TARGET_UNSAFE_RECALL: Final[float] = 0.90

FEATURES: Final[list[str]] = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

FEATURE_UNITS: Final[dict[str, str]] = {
    "ph": "pH units",
    "Hardness": "mg/L (as CaCO3)",
    "Solids": "ppm (total dissolved solids)",
    "Chloramines": "ppm",
    "Sulfate": "mg/L",
    "Conductivity": "uS/cm",
    "Organic_carbon": "ppm",
    "Trihalomethanes": "ug/L",
    "Turbidity": "NTU",
}


@dataclass(frozen=True)
class ParameterLimit:
    """A verified drinking-water limit for a single physicochemical parameter.

    Attributes:
        lower: Inclusive lower bound of the acceptable range, or ``None`` if unbounded.
        upper: Inclusive upper bound of the acceptable range, or ``None`` if unbounded.
        unit: Unit the bounds are expressed in.
        source: Short name of the standard the limit was taken from.
        url: URL the limit was verified against.
        note: Any caveat a reader needs in order to interpret the limit correctly.
    """

    lower: float | None
    upper: float | None
    unit: str
    source: str
    url: str
    note: str = ""

    def is_violation(self, value: float) -> bool:
        """Return True if ``value`` falls outside the acceptable range."""
        if self.lower is not None and value < self.lower:
            return True
        return self.upper is not None and value > self.upper


PARAMETER_LIMITS: Final[dict[str, ParameterLimit]] = {
    "ph": ParameterLimit(
        lower=6.5,
        upper=8.5,
        unit="pH units",
        source="BIS IS 10500:2012 (acceptable limit; no relaxation permitted)",
        url="https://infralens.in/code/IS-10500-2012",
        note=(
            "WHO GDWQ 4th ed. Annex 3 lists pH as 'Not of health concern at levels "
            "found in drinking-water' and sets no health-based guideline value, but "
            "6.5-8.5 is recommended for operational/disinfection reasons."
        ),
    ),
    "Hardness": ParameterLimit(
        lower=None,
        upper=600.0,
        unit="mg/L (as CaCO3)",
        source="BIS IS 10500:2012 (permissible limit in absence of alternate source)",
        url="https://infralens.in/code/IS-10500-2012",
        note=(
            "WHO GDWQ 4th ed. Annex 3: 'Hardness - Not of health concern at levels "
            "found in drinking-water'; no WHO guideline value exists."
        ),
    ),
    "Solids": ParameterLimit(
        lower=None,
        upper=2000.0,
        unit="mg/L (total dissolved solids)",
        source="BIS IS 10500:2012 (permissible limit in absence of alternate source)",
        url="https://infralens.in/code/IS-10500-2012",
        note=(
            "WHO GDWQ 4th ed. Annex 3: 'Total dissolved solids - Not of health concern "
            "at levels found in drinking-water'. NOTE: the Kaggle column is labelled "
            "ppm but has a median near 21,000, i.e. roughly 10x the BIS permissible "
            "limit for essentially every row, so this rule term fires almost always. "
            "See the README limitations section."
        ),
    ),
    "Chloramines": ParameterLimit(
        lower=None,
        upper=4.0,
        unit="mg/L (as Cl2)",
        source="US EPA National Primary Drinking Water Regulations (MRDL)",
        url=(
            "https://www.epa.gov/ground-water-and-drinking-water/"
            "national-primary-drinking-water-regulations"
        ),
        note=(
            "US EPA maximum residual disinfectant level for chloramines (as Cl2) is "
            "4.0 mg/L. WHO GDWQ Annex 3 gives 3 mg/L for MONOchloramine specifically; "
            "the dataset column is the aggregate, so the EPA aggregate MRDL is used."
        ),
    ),
    "Sulfate": ParameterLimit(
        lower=None,
        upper=400.0,
        unit="mg/L (as SO4)",
        source="BIS IS 10500:2012 (permissible limit in absence of alternate source)",
        url="https://infralens.in/code/IS-10500-2012",
        note=(
            "WHO GDWQ 4th ed. Annex 3: 'Sulfate - Not of health concern at levels "
            "found in drinking-water'; no WHO guideline value exists."
        ),
    ),
    "Trihalomethanes": ParameterLimit(
        lower=None,
        upper=80.0,
        unit="ug/L (total THMs)",
        source="US EPA National Primary Drinking Water Regulations (MCL for TTHMs)",
        url=(
            "https://www.epa.gov/ground-water-and-drinking-water/"
            "national-primary-drinking-water-regulations"
        ),
        note=(
            "US EPA MCL for total trihalomethanes is 0.080 mg/L = 80 ug/L. WHO does "
            "NOT publish a single total-THM value: Annex 3 states 'Trihalomethanes - "
            "The sum of the ratio of the concentration of each to its respective "
            "guideline value should not exceed 1', which cannot be applied here "
            "because the dataset reports only the aggregate."
        ),
    ),
    "Turbidity": ParameterLimit(
        lower=None,
        upper=5.0,
        unit="NTU",
        source="BIS IS 10500:2012 (permissible limit in absence of alternate source)",
        url="https://infralens.in/code/IS-10500-2012",
        note=(
            "BIS acceptable limit is 1 NTU, permissible 5 NTU. WHO GDWQ sets no "
            "health-based guideline value for turbidity."
        ),
    ),
}

UNVERIFIED_PARAMETERS: Final[dict[str, str]] = {
    "Conductivity": (
        "Neither WHO GDWQ 4th ed. Annex 3 nor BIS IS 10500:2012 publishes a numeric "
        "drinking-water limit for electrical conductivity; conductivity is normally "
        "regulated indirectly via total dissolved solids. Excluded from the rule."
    ),
    "Organic_carbon": (
        "WHO GDWQ 4th ed. Annex 3 lists no guideline value for total organic carbon, "
        "and the US EPA regulates TOC through required percentage removal under the "
        "Disinfectants/Disinfection By-products Rule rather than a concentration MCL. "
        "No single threshold could be verified, so it is excluded from the rule."
    ),
}

DATASET_URL: Final[str] = "https://www.kaggle.com/datasets/adityakadiwal/water-potability"
DATASET_LICENSE: Final[str] = "CC0: Public Domain"
DATASET_DOWNLOAD_DATE: Final[str] = "2026-10-01"


def ensure_output_dirs() -> None:
    """Create the results directories if they do not already exist."""
    for directory in (METRICS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)
