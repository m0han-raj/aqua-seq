# Data

## Source

| | |
|---|---|
| **Dataset** | Water Quality ("Water Potability") |
| **Author** | Aditya Kadiwal |
| **URL** | https://www.kaggle.com/datasets/adityakadiwal/water-potability |
| **License** | `CC0: Public Domain` — verified on 2026-10-01 via the Kaggle dataset API (`licenseNameNullable: "CC0: Public Domain"`) |
| **Downloaded** | 2026-10-01, using `kagglehub` (dataset version 3) |
| **File** | `raw/water_potability.csv` |
| **Size** | 525,187 bytes |
| **SHA-256** | `904004bde729bfe3d2e195f46343bceead09e32a0eb95bb8184e7e20e029b2bf` |
| **Rows** | 3,276 data rows (plus a header row) |
| **Columns** | 9 physicochemical features + 1 binary target |

Because the dataset is released under **CC0: Public Domain**, redistribution is
permitted and `raw/water_potability.csv` **is committed to this repository**. No
download step is required to reproduce the results.

## Provenance caveat

The Kaggle dataset page does not identify the water bodies, the sampling period, the
measuring laboratory, or how the `Potability` label was determined. The data is widely
believed to be synthetic or semi-synthetic. Treat it as a *modelling exercise on a
public benchmark*, not as evidence about any real water supply. This caveat drives the
limitations section of the top-level README.

## Column dictionary

| Column | Type | Unit (as documented on Kaggle) | Description |
|---|---|---|---|
| `ph` | float | pH units | Acid–base balance of the water. Physically bounded to 0–14. |
| `Hardness` | float | mg/L as CaCO₃ | Capacity of the water to precipitate soap; driven by calcium and magnesium salts. |
| `Solids` | float | ppm | Total dissolved solids (TDS): dissolved inorganic/organic minerals and salts. |
| `Chloramines` | float | ppm | Chloramine disinfectant residual. |
| `Sulfate` | float | mg/L | Dissolved sulfate. |
| `Conductivity` | float | μS/cm | Electrical conductivity, a proxy for dissolved ionic content. |
| `Organic_carbon` | float | ppm | Total organic carbon. |
| `Trihalomethanes` | float | μg/L | Total trihalomethanes, a chlorination by-product. |
| `Turbidity` | float | NTU | Cloudiness caused by suspended solids. |
| `Potability` | int | — | **Target.** `1` = potable (safe to drink), `0` = not potable (unsafe). |

### Target encoding used in this project

Aqua-Seq models the **unsafe** class as the positive class, because a missed unsafe
sample is the costly error:

```
unsafe = 1 - Potability      # 1 == UNSAFE == (Potability == 0)
```

This mapping is defined once in `src/aqua_seq/config.py` and is covered by a test
(`tests/test_data.py::test_unsafe_label_is_the_complement_of_potability`).

### Missing values

Three columns contain missing values; the other six are complete. Exact counts are
regenerated into `results/metrics/data_audit.json` by `scripts/run_all.py`:

| Column | Missing | % of rows |
|---|---|---|
| `Sulfate` | 781 | 23.8% |
| `ph` | 491 | 15.0% |
| `Trihalomethanes` | 162 | 4.9% |

Missingness is handled inside the modelling pipeline (median vs. KNN imputation, with
and without missing-indicator features, compared under cross-validation), never by
dropping rows.

## Directory layout

```
data/
  README.md    <- this file
  raw/         <- the original CSV, never modified
  processed/   <- train/test splits written by scripts/run_all.py (git-ignored)
```

`processed/` is regenerated on every run and is excluded from version control, since it
is fully derived from `raw/` plus the fixed random seed.

## How to obtain the data yourself

The CSV is already in `raw/`. If you want to re-download it from the original source:

**Option A — `kagglehub` (no credentials needed for this public dataset):**

```powershell
python -c "import kagglehub, shutil, pathlib; p = kagglehub.dataset_download('adityakadiwal/water-potability'); shutil.copy(pathlib.Path(p) / 'water_potability.csv', 'data/raw/water_potability.csv')"
```

**Option B — Kaggle CLI (requires a `kaggle.json` API token):**

```powershell
kaggle datasets download -d adityakadiwal/water-potability -p data/raw --unzip
```

**Option C — manual:** download from the dataset URL above and save the file as
`data/raw/water_potability.csv`.

The pipeline validates the schema on load and raises a clear error if the file is
missing or its columns do not match (`src/aqua_seq/data.py`).
