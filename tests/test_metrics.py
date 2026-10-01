"""Tests for metrics, risk-aware threshold selection and the rule baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aqua_seq import config, evaluate, features


def test_metrics_on_a_perfect_classifier() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.0, 0.1, 0.9, 1.0])
    metrics = evaluate.classification_metrics(y_true, y_score, threshold=0.5)
    assert metrics["roc_auc"] == 1.0
    assert metrics["recall_unsafe"] == 1.0
    assert metrics["precision_unsafe"] == 1.0
    assert metrics["false_negatives"] == 0


def test_metrics_count_unsafe_as_the_positive_class() -> None:
    """Recall must be measured on label 1, which this project defines as UNSAFE."""
    y_true = np.array([1, 1, 1, 0])
    y_score = np.array([0.9, 0.9, 0.1, 0.1])
    metrics = evaluate.classification_metrics(y_true, y_score, threshold=0.5)
    assert metrics["true_positives"] == 2
    assert metrics["false_negatives"] == 1
    assert metrics["recall_unsafe"] == pytest.approx(2 / 3)
    assert metrics["precision_unsafe"] == 1.0


def test_confusion_counts_sum_to_sample_count() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(200)
    metrics = evaluate.classification_metrics(y_true, y_score, threshold=0.5)
    total = (
        metrics["true_positives"]
        + metrics["false_positives"]
        + metrics["true_negatives"]
        + metrics["false_negatives"]
    )
    assert total == metrics["n_samples"] == 200


def test_lowering_the_threshold_cannot_reduce_recall() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=400)
    y_score = rng.random(400)
    recalls = [
        evaluate.classification_metrics(y_true, y_score, t)["recall_unsafe"]
        for t in np.linspace(0.05, 0.95, 20)
    ]
    assert all(a >= b - 1e-12 for a, b in zip(recalls, recalls[1:], strict=False))


def test_select_threshold_meets_the_recall_target() -> None:
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=1000)
    y_score = np.clip(y_true * 0.45 + rng.normal(0.3, 0.18, size=1000), 0, 1)
    choice = evaluate.select_threshold(y_true, y_score, target_recall=0.90)
    assert choice.target_met
    assert choice.achieved_recall >= 0.90
    achieved = evaluate.classification_metrics(y_true, y_score, choice.threshold)
    assert achieved["recall_unsafe"] >= 0.90 - 1e-9


def test_select_threshold_reports_failure_when_target_unreachable() -> None:
    """With a target of 1.0 recall at a reachable precision the flag must stay honest."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_score = np.array([0.9, 0.1, 0.2, 0.8, 0.3])
    choice = evaluate.select_threshold(y_true, y_score, target_recall=1.01)
    assert not choice.target_met


def test_select_threshold_prefers_the_best_precision_among_eligible() -> None:
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=800)
    y_score = np.clip(y_true * 0.4 + rng.normal(0.3, 0.2, size=800), 0, 1)
    choice = evaluate.select_threshold(y_true, y_score, target_recall=0.85)
    sweep = evaluate.threshold_sweep(y_true, y_score, n_points=300)
    eligible = sweep[sweep["recall_unsafe"] >= 0.85]
    assert choice.achieved_precision >= eligible["precision_unsafe"].max() - 0.02


def test_bootstrap_intervals_bracket_the_point_estimate() -> None:
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 2, size=500)
    y_score = np.clip(y_true * 0.3 + rng.normal(0.35, 0.2, size=500), 0, 1)
    point = evaluate.classification_metrics(y_true, y_score, 0.5)
    cis = evaluate.bootstrap_metric_cis(y_true, y_score, 0.5, n_boot=200)
    for metric in ("roc_auc", "recall_unsafe"):
        assert cis[metric]["lower"] <= point[metric] <= cis[metric]["upper"]
        assert cis[metric]["lower"] < cis[metric]["upper"]


def test_threshold_sweep_is_monotonic_in_flag_rate() -> None:
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 2, size=300)
    y_score = rng.random(300)
    sweep = evaluate.threshold_sweep(y_true, y_score, n_points=50)
    assert sweep["flag_rate"].is_monotonic_decreasing


def test_every_limit_cites_a_source_and_url() -> None:
    """No guideline limit may exist without a verifiable provenance."""
    for name, limit in config.PARAMETER_LIMITS.items():
        assert limit.source, f"{name} has no source"
        assert limit.url.startswith("http"), f"{name} has no URL"
        assert limit.lower is not None or limit.upper is not None


def test_unverified_parameters_are_excluded_from_the_rule() -> None:
    overlap = set(config.PARAMETER_LIMITS) & set(config.UNVERIFIED_PARAMETERS)
    assert not overlap, f"{overlap} are both limited and marked unverified"
    for name in config.UNVERIFIED_PARAMETERS:
        assert name in config.FEATURES


def test_rule_flags_a_clear_violation() -> None:
    frame = pd.DataFrame(
        [dict.fromkeys(config.FEATURES, 0.0)],
    )
    frame.loc[0, "ph"] = 7.0
    frame.loc[0, "Turbidity"] = 99.0
    assert features.rule_baseline_predict(frame)[0] == 1


def test_rule_passes_a_compliant_sample() -> None:
    compliant = {
        "ph": 7.0,
        "Hardness": 150.0,
        "Solids": 400.0,
        "Chloramines": 2.0,
        "Sulfate": 100.0,
        "Conductivity": 400.0,
        "Organic_carbon": 10.0,
        "Trihalomethanes": 40.0,
        "Turbidity": 1.0,
    }
    frame = pd.DataFrame([compliant])
    assert features.rule_baseline_predict(frame)[0] == 0


def test_missing_measurements_do_not_count_as_violations() -> None:
    frame = pd.DataFrame([dict.fromkeys(config.FEATURES, np.nan)])
    matrix = features.rule_violation_matrix(frame)
    assert not matrix.to_numpy().any()
    assert features.rule_baseline_predict(frame)[0] == 0


def test_rule_score_is_the_fraction_of_breached_parameters() -> None:
    frame = pd.DataFrame(
        [
            {
                "ph": 3.0,
                "Hardness": 150.0,
                "Solids": 400.0,
                "Chloramines": 2.0,
                "Sulfate": 100.0,
                "Conductivity": 400.0,
                "Organic_carbon": 10.0,
                "Trihalomethanes": 40.0,
                "Turbidity": 99.0,
            }
        ]
    )
    score = features.rule_baseline_score(frame)[0]
    assert score == pytest.approx(2 / len(config.PARAMETER_LIMITS))
