"""Run the entire AquaSentinel pipeline end to end, deterministically.

Regenerates every file in ``results/metrics`` and ``results/figures`` from the raw
CSV with no manual steps:

    python scripts/run_all.py

Stages: data audit -> EDA -> rule baseline -> split -> leakage-safe CV tuning ->
risk-aware threshold selection -> single final test evaluation -> explainability ->
error analysis -> README results sections.
"""

from __future__ import annotations

import json
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

from aquasentinel import config, data, evaluate, explain, features, models, report  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def set_global_seeds(seed: int = config.RANDOM_SEED) -> None:
    """Seed every RNG the pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)


def write_json(payload: Any, name: str) -> Path:
    """Write ``payload`` as pretty JSON into the metrics directory."""
    config.ensure_output_dirs()
    path = config.METRICS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_csv(frame: pd.DataFrame, name: str, index: bool = False) -> Path:
    """Write ``frame`` as CSV into the metrics directory."""
    config.ensure_output_dirs()
    path = config.METRICS_DIR / f"{name}.csv"
    frame.to_csv(path, index=index)
    return path


def log(message: str) -> None:
    """Print a progress line with a stage marker."""
    print(f"[aquasentinel] {message}", flush=True)


# --------------------------------------------------------------------------------------
# Step 1-2: audit and EDA
# --------------------------------------------------------------------------------------
def run_audit_and_eda(raw: pd.DataFrame) -> dict[str, Any]:
    """Produce the data-audit and EDA artefacts.

    Args:
        raw: The raw dataframe straight from the CSV.

    Returns:
        The audit summary that is also written to ``data_audit.json``.
    """
    log("Step 1-2: data audit and EDA")
    target = data.to_unsafe_label(raw)
    feature_frame = raw[config.FEATURES]
    missing = data.missingness_report(raw)

    # Is missingness related to the target or to other features?
    missing_vs_target = {}
    for column in missing[missing["n_missing"] > 0].index:
        flag = raw[column].isna()
        rate_unsafe = float(target[flag].mean())
        rate_observed = float(target[~flag].mean())
        _, p_value = mannwhitneyu(target[flag], target[~flag], alternative="two-sided")
        missing_vs_target[column] = {
            "unsafe_rate_when_missing": rate_unsafe,
            "unsafe_rate_when_present": rate_observed,
            "difference": rate_unsafe - rate_observed,
            "mannwhitney_p": float(p_value),
        }

    # Statistical tests: each feature against the target, with rank-biserial effect size.
    tests = []
    for column in config.FEATURES:
        unsafe_values = feature_frame.loc[target == 1, column].dropna()
        safe_values = feature_frame.loc[target == 0, column].dropna()
        statistic, p_value = mannwhitneyu(
            unsafe_values, safe_values, alternative="two-sided"
        )
        # Rank-biserial correlation: 2*AUC - 1, bounded [-1, 1]; 0 means no separation.
        auc = statistic / (len(unsafe_values) * len(safe_values))
        tests.append(
            {
                "feature": column,
                "n_unsafe": int(len(unsafe_values)),
                "n_safe": int(len(safe_values)),
                "median_unsafe": float(unsafe_values.median()),
                "median_safe": float(safe_values.median()),
                "mannwhitney_u": float(statistic),
                "p_value": float(p_value),
                "rank_biserial_effect_size": float(2 * auc - 1),
                "abs_effect_size": float(abs(2 * auc - 1)),
            }
        )
    tests_frame = pd.DataFrame(tests).sort_values("abs_effect_size", ascending=False)
    write_csv(tests_frame, "eda_feature_tests")

    summary_stats = feature_frame.describe().T
    summary_stats.index.name = "feature"
    write_csv(summary_stats.round(4), "eda_summary_statistics", index=True)

    # Figures
    evaluate.plot_target_balance(target, "01_target_balance")
    evaluate.plot_missingness(missing, "02_missingness")
    evaluate.plot_feature_distributions(
        feature_frame, target, config.FEATURES, "03_feature_distributions"
    )
    evaluate.plot_correlation_heatmap(
        raw[config.FEATURES + [config.RAW_TARGET]], "04_correlation_heatmap"
    )

    audit = {
        "n_rows": int(len(raw)),
        "n_features": len(config.FEATURES),
        "n_duplicate_rows": int(raw.duplicated().sum()),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "class_counts": {
            config.NEGATIVE_CLASS_NAME: int((target == 0).sum()),
            config.POSITIVE_CLASS_NAME: int((target == 1).sum()),
        },
        "unsafe_prevalence": float(target.mean()),
        "missingness": missing.to_dict(orient="index"),
        "missingness_vs_target": missing_vs_target,
        "impossible_values": {
            "ph_outside_0_14": int(
                ((raw["ph"] < 0) | (raw["ph"] > 14)).sum()
            ),
            "negative_values_any_feature": int((feature_frame < 0).sum().sum()),
        },
        "strongest_univariate_effects": tests_frame.head(3)[
            ["feature", "rank_biserial_effect_size", "p_value"]
        ].to_dict(orient="records"),
    }
    write_json(audit, "data_audit")
    return audit


# --------------------------------------------------------------------------------------
# Step 3: rule baseline
# --------------------------------------------------------------------------------------
def run_rule_baseline(raw: pd.DataFrame) -> dict[str, Any]:
    """Evaluate the guideline-based rule baseline and record which limits fire."""
    log("Step 3: guideline-based rule baseline")
    target = data.to_unsafe_label(raw)
    matrix = features.rule_violation_matrix(raw[config.FEATURES])
    predictions = features.rule_baseline_predict(raw[config.FEATURES])

    per_parameter = {}
    for column in matrix.columns:
        limit = config.PARAMETER_LIMITS[column]
        per_parameter[column] = {
            "lower": limit.lower,
            "upper": limit.upper,
            "unit": limit.unit,
            "source": limit.source,
            "url": limit.url,
            "note": limit.note,
            "n_violations": int(matrix[column].sum()),
            "pct_violations": float(matrix[column].mean() * 100),
            "n_measured": int(raw[column].notna().sum()),
        }

    payload = {
        "parameters_used": per_parameter,
        "parameters_excluded": config.UNVERIFIED_PARAMETERS,
        "metrics_full_dataset": evaluate.classification_metrics(
            target, predictions.astype(float), threshold=0.5
        ),
    }
    write_json(payload, "rule_baseline")
    return payload


# --------------------------------------------------------------------------------------
# Steps 4-6: split, leakage-safe CV, tuning
# --------------------------------------------------------------------------------------
def run_modeling(split: data.SplitData) -> dict[str, Any]:
    """Tune every candidate model with CV on the training set only."""
    log("Steps 4-6: cross-validated tuning (training set only)")
    search_cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )
    report_cv = RepeatedStratifiedKFold(
        n_splits=config.CV_FOLDS,
        n_repeats=config.CV_REPEATS,
        random_state=config.RANDOM_SEED,
    )

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    # --- Baseline 1: majority-class dummy -------------------------------------------
    dummy = models.build_dummy_classifier()
    dummy_scores = cross_val_score(
        dummy, split.x_train, split.y_train, cv=report_cv, scoring="roc_auc", n_jobs=-1
    )
    rows.append(
        {
            "model": "Dummy (majority class)",
            "kind": "baseline",
            "cv_roc_auc_mean": float(dummy_scores.mean()),
            "cv_roc_auc_std": float(dummy_scores.std()),
            "best_params": {},
        }
    )
    fitted["Dummy (majority class)"] = clone(dummy).fit(split.x_train, split.y_train)

    # --- Baseline 2: guideline rule, scored through the same CV folds ----------------
    rule_aucs, rule_aps = [], []
    for _, valid_idx in report_cv.split(split.x_train, split.y_train):
        x_valid = split.x_train.iloc[valid_idx]
        y_valid = split.y_train.iloc[valid_idx]
        scores = features.rule_baseline_score(x_valid)
        metrics = evaluate.classification_metrics(y_valid, scores, threshold=1e-9)
        rule_aucs.append(metrics["roc_auc"])
        rule_aps.append(metrics["pr_auc"])
    rows.append(
        {
            "model": "Guideline rule baseline",
            "kind": "baseline",
            "cv_roc_auc_mean": float(np.mean(rule_aucs)),
            "cv_roc_auc_std": float(np.std(rule_aucs)),
            "best_params": {},
        }
    )

    # --- Candidate models ------------------------------------------------------------
    for spec in models.build_model_specs():
        started = time.time()
        log(f"  tuning {spec.name} ({config.N_SEARCH_ITER} candidates x {config.CV_FOLDS} folds)")
        search = RandomizedSearchCV(
            estimator=spec.pipeline,
            param_distributions=spec.param_distributions,
            n_iter=config.N_SEARCH_ITER,
            scoring="roc_auc",
            cv=search_cv,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(split.x_train, split.y_train)

        # Re-score the chosen configuration under the richer repeated CV.
        scores = cross_val_score(
            clone(search.best_estimator_),
            split.x_train,
            split.y_train,
            cv=report_cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        elapsed = time.time() - started
        log(f"    CV ROC-AUC {scores.mean():.4f} +/- {scores.std():.4f}  ({elapsed:.0f}s)")

        chosen_imputer = search.best_params_.get("preprocess__numeric__impute")
        rows.append(
            {
                "model": spec.name,
                "kind": "model",
                "cv_roc_auc_mean": float(scores.mean()),
                "cv_roc_auc_std": float(scores.std()),
                "search_best_roc_auc": float(search.best_score_),
                "chosen_imputer": type(chosen_imputer).__name__ if chosen_imputer else None,
                "chosen_imputer_add_indicator": bool(
                    getattr(chosen_imputer, "add_indicator", False)
                ),
                "fit_seconds": round(elapsed, 1),
                "best_params": {
                    k: (type(v).__name__ if k.endswith("impute") else v)
                    for k, v in search.best_params_.items()
                },
            }
        )
        fitted[spec.name] = search.best_estimator_

    leaderboard = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    write_csv(leaderboard.drop(columns=["best_params"]), "model_leaderboard")
    write_json(rows, "model_leaderboard_full")
    evaluate.plot_model_comparison(leaderboard, "05_model_comparison")
    return {"leaderboard": leaderboard, "fitted": fitted}


def run_imputer_comparison(split: data.SplitData) -> pd.DataFrame:
    """Compare imputation strategies head-to-head inside cross-validation."""
    log("Step 5: imputation strategy comparison (cross-validated)")
    cv = RepeatedStratifiedKFold(
        n_splits=config.CV_FOLDS, n_repeats=1, random_state=config.RANDOM_SEED
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    rows = []
    for name in features.IMPUTER_CHOICES:
        pipeline = Pipeline(
            steps=[
                ("preprocess", features.build_preprocessor(imputer=name, scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400, random_state=config.RANDOM_SEED, n_jobs=-1
                    ),
                ),
            ]
        )
        scores = cross_val_score(
            pipeline, split.x_train, split.y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        rows.append(
            {
                "imputer": name,
                "cv_roc_auc_mean": float(scores.mean()),
                "cv_roc_auc_std": float(scores.std()),
            }
        )
        log(f"  {name:18s} ROC-AUC {scores.mean():.4f} +/- {scores.std():.4f}")

    frame = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    write_csv(frame, "imputer_comparison")
    return frame


# --------------------------------------------------------------------------------------
# Step 7: threshold selection and the single final test evaluation
# --------------------------------------------------------------------------------------
def run_final_evaluation(
    split: data.SplitData, leaderboard: pd.DataFrame, fitted: dict[str, Any]
) -> dict[str, Any]:
    """Select the threshold on CV predictions, then evaluate ONCE on the test set."""
    log("Step 7: threshold selection and final test evaluation")
    best_name = str(
        leaderboard[leaderboard["kind"] == "model"].iloc[0]["model"]
    )
    best_pipeline = fitted[best_name]
    log(f"  best model by CV ROC-AUC: {best_name}")

    # --- Threshold chosen on cross-validated TRAINING predictions only ---------------
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )
    oof_scores = cross_val_predict(
        clone(best_pipeline),
        split.x_train,
        split.y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    choice = evaluate.select_threshold(split.y_train, oof_scores)
    sweep = evaluate.threshold_sweep(split.y_train, oof_scores)
    write_csv(sweep, "threshold_sweep")
    evaluate.plot_threshold_tradeoff(sweep, choice, "06_threshold_tradeoff")
    log(
        f"  chosen threshold {choice.threshold:.4f} "
        f"(CV recall {choice.achieved_recall:.3f}, precision {choice.achieved_precision:.3f})"
    )

    # --- Retrain on the full training set, then touch the test set exactly once ------
    final_model = clone(best_pipeline).fit(split.x_train, split.y_train)
    test_scores = final_model.predict_proba(split.x_test)[:, 1]

    at_chosen = evaluate.classification_metrics(split.y_test, test_scores, choice.threshold)
    at_default = evaluate.classification_metrics(split.y_test, test_scores, 0.5)
    cis = evaluate.bootstrap_metric_cis(split.y_test, test_scores, choice.threshold)

    # --- Baselines on the same test set ----------------------------------------------
    rule_scores = features.rule_baseline_score(split.x_test)
    rule_pred = features.rule_baseline_predict(split.x_test)
    rule_metrics = evaluate.classification_metrics(split.y_test, rule_scores, 1e-9)
    rule_metrics["recall_unsafe_hard_rule"] = float(
        (rule_pred[np.asarray(split.y_test) == 1] == 1).mean()
    )

    dummy_pipeline = fitted["Dummy (majority class)"]
    dummy_scores = dummy_pipeline.predict_proba(split.x_test)[:, 1]
    dummy_metrics = evaluate.classification_metrics(split.y_test, dummy_scores, 0.5)

    # --- Curves, confusion matrix, calibration ---------------------------------------
    curves = {
        name: (split.y_test, fitted[name].predict_proba(split.x_test)[:, 1])
        for name in leaderboard[leaderboard["kind"] == "model"]["model"]
    }
    curves["Guideline rule baseline"] = (split.y_test, rule_scores)
    evaluate.plot_roc_curves(curves, "07_roc_curves")
    evaluate.plot_pr_curves(curves, float(split.y_test.mean()), "08_pr_curves")
    evaluate.plot_confusion_matrix(
        split.y_test,
        (test_scores >= choice.threshold).astype(int),
        f"AquaSentinel - {best_name} on the test set at threshold {choice.threshold:.3f}",
        "09_confusion_matrix",
    )
    evaluate.plot_calibration(split.y_test, test_scores, best_name, "10_calibration")

    payload = {
        "best_model": best_name,
        "label_mapping": {
            "positive_class": config.POSITIVE_CLASS_NAME,
            "definition": "unsafe = 1 = (Potability == 0)",
        },
        "threshold_choice": evaluate.threshold_choice_to_dict(choice),
        "n_train": split.n_train,
        "n_test": split.n_test,
        "test_unsafe_prevalence": float(split.y_test.mean()),
        "final_model_test_at_chosen_threshold": at_chosen,
        "final_model_test_at_default_threshold": at_default,
        "final_model_test_bootstrap_ci_95": cis,
        "rule_baseline_test": rule_metrics,
        "dummy_baseline_test": dummy_metrics,
    }
    write_json(payload, "final_evaluation")
    return {"payload": payload, "model": final_model, "test_scores": test_scores,
            "best_name": best_name, "choice": choice}


# --------------------------------------------------------------------------------------
# Step 8-9: explainability and error analysis
# --------------------------------------------------------------------------------------
def run_explainability(final: dict[str, Any], split: data.SplitData) -> dict[str, Any]:
    """Permutation importance and SHAP for the final model."""
    log("Step 8: explainability")
    importance = explain.permutation_importance_frame(
        final["model"], split.x_test, split.y_test
    )
    write_csv(importance, "permutation_importance")
    evaluate.plot_permutation_importance(importance, "11_permutation_importance")

    shap_status = "skipped"
    result = explain.shap_summary(final["model"], split.x_test)
    if result is not None:
        values, transformed = result
        explain.plot_shap_summary(values, transformed, "12_shap_summary")
        mean_abs = np.abs(values).mean(axis=0)
        top_two = [transformed.columns[i] for i in np.argsort(mean_abs)[::-1][:2]]
        explain.plot_shap_dependence(values, transformed, top_two, "13_shap_dependence")
        shap_status = "computed"
        write_csv(
            pd.DataFrame(
                {"feature": transformed.columns, "mean_abs_shap": mean_abs}
            ).sort_values("mean_abs_shap", ascending=False),
            "shap_importance",
        )

    payload = {
        "permutation_importance": importance.to_dict(orient="records"),
        "shap_status": shap_status,
        "parameters_in_rule": sorted(config.PARAMETER_LIMITS),
        "parameters_excluded_from_rule": sorted(config.UNVERIFIED_PARAMETERS),
    }
    write_json(payload, "explainability")
    return payload


def run_error_analysis(final: dict[str, Any], split: data.SplitData) -> dict[str, Any]:
    """Characterise which samples the final model gets wrong."""
    log("Step 9: error analysis")
    threshold = final["choice"].threshold
    predictions = (final["test_scores"] >= threshold).astype(int)
    frame = split.x_test.copy()
    frame["y_true"] = split.y_test.to_numpy()
    frame["y_pred"] = predictions
    frame["score"] = final["test_scores"]
    frame["outcome"] = np.select(
        [
            (frame.y_true == 1) & (frame.y_pred == 1),
            (frame.y_true == 1) & (frame.y_pred == 0),
            (frame.y_true == 0) & (frame.y_pred == 1),
        ],
        ["true_positive", "false_negative (MISSED UNSAFE)", "false_positive"],
        default="true_negative",
    )
    write_csv(
        frame.groupby("outcome")[config.FEATURES].median().round(3),
        "error_analysis_group_medians",
        index=True,
    )

    missed = frame[frame.outcome.str.startswith("false_negative")]
    caught = frame[frame.outcome == "true_positive"]
    comparison = []
    for column in config.FEATURES:
        comparison.append(
            {
                "feature": column,
                "median_missed_unsafe": float(missed[column].median()),
                "median_caught_unsafe": float(caught[column].median()),
                "median_gap": float(missed[column].median() - caught[column].median()),
            }
        )
    write_csv(pd.DataFrame(comparison), "error_analysis_missed_vs_caught")

    payload = {
        "outcome_counts": frame["outcome"].value_counts().to_dict(),
        "n_missed_unsafe": int(len(missed)),
        "missed_unsafe_score_median": float(missed["score"].median()) if len(missed) else None,
        "caught_unsafe_score_median": float(caught["score"].median()) if len(caught) else None,
        "score_overlap_note": (
            "Scores for missed and caught unsafe samples overlap heavily, which is the "
            "signature of a weak feature-label relationship rather than a tuning problem."
        ),
    }
    write_json(payload, "error_analysis")
    return payload


def main() -> int:
    """Run every stage in order and write the README results sections."""
    started = time.time()
    set_global_seeds()
    config.ensure_output_dirs()

    log(f"AquaSentinel pipeline, seed={config.RANDOM_SEED}")
    raw = data.load_raw()
    log(f"Loaded {len(raw):,} rows x {raw.shape[1]} columns from {config.RAW_CSV}")

    run_audit_and_eda(raw)
    run_rule_baseline(raw)

    split = data.make_split(raw)
    log(f"Split: {split.n_train:,} train / {split.n_test:,} test (stratified, seed {config.RANDOM_SEED})")
    split.x_train.assign(**{config.TARGET: split.y_train}).to_csv(
        config.PROCESSED_DATA_DIR / "train.csv", index=False
    )
    split.x_test.assign(**{config.TARGET: split.y_test}).to_csv(
        config.PROCESSED_DATA_DIR / "test.csv", index=False
    )

    run_imputer_comparison(split)
    modeling = run_modeling(split)
    final = run_final_evaluation(split, modeling["leaderboard"], modeling["fitted"])
    run_explainability(final, split)
    run_error_analysis(final, split)

    runtime = time.time() - started
    write_json(
        {
            "runtime_seconds": round(runtime, 1),
            "python_version": sys.version.split()[0],
            "random_seed": config.RANDOM_SEED,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "run_metadata",
    )
    log(f"Pipeline finished in {runtime / 60:.1f} min")

    report.build_readme_sections()
    log("README results sections written to results/metrics/readme_sections.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
