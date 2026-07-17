from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from src.model.risk_adjusted_common import (
    RiskAdjustedGateSpec,
    RiskMetrics,
    SegmentWindow,
    WalkForwardAggregate,
    aggregate_walk_forward,
    build_risk_adjusted_labels,
    compute_risk_metrics,
    evaluate_promotion,
    purge_segment_tail,
    soft_budget_position,
)


def test_segment_is_an_immutable_inclusive_contract() -> None:
    segment = SegmentWindow("train", "2024-01-02", "2024-01-31")

    assert segment.start == pd.Timestamp("2024-01-02")
    assert segment.end == pd.Timestamp("2024-01-31")
    with pytest.raises(FrozenInstanceError):
        segment.end = pd.Timestamp("2024-02-01")  # type: ignore[misc]
    with pytest.raises(ValueError, match="start must not be after end"):
        SegmentWindow("bad", "2024-02-01", "2024-01-31")


def test_purge_segment_tail_removes_horizon_plus_one_decision_rows() -> None:
    index = pd.bdate_range("2024-01-02", periods=12)
    frame = pd.DataFrame({"value": np.arange(12)}, index=index)

    purged = purge_segment_tail(frame, horizon=5)

    assert purged.index.tolist() == index[:6].tolist()


def test_five_day_labels_use_compounded_next_open_returns_and_purge_six_rows() -> None:
    index = pd.bdate_range("2024-01-02", periods=24)
    next_open_returns = pd.Series(np.linspace(-0.03, 0.04, len(index)), index=index)
    segments = (
        SegmentWindow("train", index[0], index[11]),
        SegmentWindow("valid", index[12], index[23]),
    )

    result = build_risk_adjusted_labels(next_open_returns, segments=segments)

    assert result.horizon == 5
    assert result.purge_rows == 6
    assert result.purged_boundary_rows == {"train": 6, "valid": 6}
    assert result.labels.groupby("segment").size().to_dict() == {"train": 6, "valid": 6}
    expected = float(np.prod(1.0 + next_open_returns.iloc[:5]) - 1.0)
    assert result.labels.loc[index[0], "future_return_5d"] == pytest.approx(expected)
    assert result.labels.loc[index[0], "future_downside_5d"] == pytest.approx(max(-expected, 0.0))
    assert index[6] not in result.labels.index
    assert index[18] not in result.labels.index


@pytest.mark.parametrize("horizon", [2, 5.5])
def test_risk_adjusted_label_horizon_is_fixed_at_five_days(horizon: float) -> None:
    index = pd.bdate_range("2024-01-02", periods=12)
    returns = pd.Series(np.linspace(-0.02, 0.02, len(index)), index=index)

    with pytest.raises(ValueError, match="horizon is fixed at five"):
        build_risk_adjusted_labels(
            returns,
            segments=(SegmentWindow("train", index[0], index[-1]),),
            horizon=horizon,
        )


def test_crash_threshold_is_fit_on_train_only_and_frozen_for_other_segments() -> None:
    index = pd.bdate_range("2024-01-02", periods=28)
    train = pd.Series(np.linspace(-0.04, 0.03, 14), index=index[:14])
    valid = pd.Series(np.linspace(-0.01, 0.02, 14), index=index[14:])
    returns = pd.concat([train, valid])
    segments = (
        SegmentWindow("train", index[0], index[13]),
        SegmentWindow("valid", index[14], index[27]),
    )

    original = build_risk_adjusted_labels(returns, segments=segments)
    perturbed_returns = returns.copy()
    perturbed_returns.loc[index[14:]] = np.linspace(-0.90, 0.90, 14)
    perturbed = build_risk_adjusted_labels(perturbed_returns, segments=segments)

    assert perturbed.train_crash_threshold == pytest.approx(original.train_crash_threshold)
    pd.testing.assert_frame_equal(
        original.labels.query("segment == 'train'"),
        perturbed.labels.query("segment == 'train'"),
    )
    expected_threshold = original.labels.query("segment == 'train'")["future_return_5d"].quantile(0.10)
    assert original.train_crash_threshold == pytest.approx(expected_threshold)
    expected_crash = (
        (original.labels["future_return_5d"] <= original.train_crash_threshold)
        | (original.labels["future_min_daily_return_5d"] <= -0.08)
    ).astype("int8")
    pd.testing.assert_series_equal(
        original.labels["future_crash_5d"],
        expected_crash.rename("future_crash_5d"),
    )


def test_label_at_a_decision_is_invariant_to_returns_after_its_horizon() -> None:
    index = pd.bdate_range("2024-01-02", periods=16)
    returns = pd.Series(np.linspace(-0.02, 0.02, len(index)), index=index)
    segment = SegmentWindow("train", index[0], index[-1])

    original = build_risk_adjusted_labels(returns, segments=(segment,))
    perturbed_returns = returns.copy()
    perturbed_returns.loc[index[5:]] = 0.75
    perturbed = build_risk_adjusted_labels(perturbed_returns, segments=(segment,))

    columns = ["future_return_5d", "future_downside_5d", "future_min_daily_return_5d"]
    pd.testing.assert_series_equal(original.labels.loc[index[0], columns], perturbed.labels.loc[index[0], columns])


def test_label_builder_rejects_overlap_hidden_inside_purged_segment_tails() -> None:
    index = pd.bdate_range("2024-01-02", periods=18)
    returns = pd.Series(np.linspace(-0.02, 0.02, len(index)), index=index)
    segments = (
        SegmentWindow("train", index[0], index[11]),
        SegmentWindow("valid", index[6], index[17]),
    )

    with pytest.raises(ValueError, match="segments must not overlap"):
        build_risk_adjusted_labels(returns, segments=segments)


def test_label_builder_requires_train_to_precede_every_other_segment() -> None:
    index = pd.bdate_range("2024-01-02", periods=24)
    returns = pd.Series(np.linspace(-0.02, 0.02, len(index)), index=index)
    segments = (
        SegmentWindow("valid", index[0], index[11]),
        SegmentWindow("train", index[12], index[23]),
    )

    with pytest.raises(ValueError, match="train segment must precede"):
        build_risk_adjusted_labels(returns, segments=segments)


def test_label_builder_records_skipped_nonfinite_candidates_as_incomplete() -> None:
    index = pd.bdate_range("2024-01-02", periods=24)
    returns = pd.Series(np.linspace(-0.02, 0.02, len(index)), index=index)
    returns.loc[index[4]] = np.nan
    segments = (
        SegmentWindow("train", index[0], index[11]),
        SegmentWindow("valid", index[12], index[23]),
    )

    result = build_risk_adjusted_labels(returns, segments=segments)

    assert result.skipped_invalid_rows == {"train": 5, "valid": 0}
    assert result.labels_complete is False


def test_label_builder_marks_declared_segments_without_complete_labels_missing() -> None:
    index = pd.bdate_range("2024-01-02", periods=18)
    returns = pd.Series(np.linspace(-0.02, 0.02, len(index)), index=index)
    segments = (
        SegmentWindow("train", index[0], index[11]),
        SegmentWindow("valid", index[12], index[17]),
    )

    result = build_risk_adjusted_labels(returns, segments=segments)

    assert result.missing_label_segments == ("valid",)
    assert result.labels_complete is False


def test_soft_budget_respects_floor_cap_and_fails_closed_on_bad_data() -> None:
    scores = pd.Series([-100.0, 100.0, 100.0, np.nan], index=list("abcd"))
    positions = soft_budget_position(
        mu_score=scores,
        tail_score=pd.Series([0.0, 0.0, 2.0, 0.0], index=scores.index),
        crash_probability=pd.Series(0.0, index=scores.index),
        data_quality_valid=pd.Series([True, True, True, True], index=scores.index),
    )

    assert isinstance(positions, pd.Series)
    assert positions.index.equals(scores.index)
    assert positions.loc["a"] == pytest.approx(0.15)
    assert positions.loc["b"] == pytest.approx(0.75)
    assert positions.loc["c"] == pytest.approx(0.15)
    assert positions.loc["d"] == 0.0
    assert soft_budget_position(100.0, 0.0, 0.0, data_quality_valid=False) == 0.0
    with pytest.raises(ValueError, match="floor must be smaller than cap"):
        soft_budget_position(0.0, 0.0, 0.0, data_quality_valid=True, floor=0.75, cap=0.15)


def test_soft_budget_treats_nonfinite_data_quality_as_invalid() -> None:
    positions = soft_budget_position(
        pd.Series([100.0, 100.0]),
        pd.Series([0.0, 0.0]),
        pd.Series([0.0, 0.0]),
        data_quality_valid=pd.Series([True, np.nan]),
    )

    assert positions.tolist() == pytest.approx([0.75, 0.0])
    assert soft_budget_position(100.0, 0.0, 0.0, data_quality_valid=np.nan) == 0.0


def test_risk_metrics_cover_drawdown_tail_participation_and_turnover() -> None:
    returns = pd.Series([0.10, -0.10, -0.20, 0.05, 0.02, -0.01])
    positions = pd.Series([0.50, 0.50, 0.00, 0.25, -0.25, -0.25])

    metrics = compute_risk_metrics(returns, positions)

    nav = pd.Series([1.0, *(1.0 + returns).cumprod().tolist()])
    assert metrics.total_return == pytest.approx(float(nav.iloc[-1] - 1.0))
    assert metrics.max_drawdown == pytest.approx(float((nav / nav.cummax() - 1.0).min()))
    assert metrics.cvar_5pct == pytest.approx(-0.20)
    assert metrics.worst_five_day_mean == pytest.approx(float(returns.nsmallest(5).mean()))
    assert metrics.extreme_loss_day_count == 2
    assert metrics.avg_abs_position == pytest.approx(float(positions.abs().mean()))
    assert metrics.zero_position_day_ratio == pytest.approx(1.0 / 6.0)
    assert metrics.total_turnover == pytest.approx(1.75)
    assert metrics.observations == 6
    assert metrics.return_to_drawdown == pytest.approx(metrics.total_return / abs(metrics.max_drawdown))


def test_risk_metrics_include_initial_nav_peak_and_compound_offsetting_returns() -> None:
    metrics = compute_risk_metrics(
        pd.Series([-0.25, 0.25]),
        pd.Series([0.50, 0.50]),
    )

    assert metrics.total_return == pytest.approx(-0.0625)
    assert metrics.max_drawdown == pytest.approx(-0.25)


def test_walk_forward_aggregate_is_weighted_and_counts_positive_folds() -> None:
    fold_returns = ([0.02, -0.01], [-0.03, -0.01, 0.01], [0.01, 0.01])
    fold_positions = ([0.2, 0.2], [0.4, 0.0, 0.4], [0.3, 0.3])
    first, second, third = (
        compute_risk_metrics(pd.Series(returns), pd.Series(positions))
        for returns, positions in zip(fold_returns, fold_positions)
    )

    aggregate = aggregate_walk_forward((first, second, third))
    combined_returns = tuple(value for fold in fold_returns for value in fold)
    combined_positions = tuple(value for fold in fold_positions for value in fold)
    expected = compute_risk_metrics(combined_returns, combined_positions)

    assert aggregate.fold_count == 3
    assert aggregate.positive_return_folds == 2
    assert aggregate.metrics == expected
    assert aggregate.metrics.daily_returns == combined_returns
    assert aggregate.metrics.positions == combined_positions


def _metrics(**overrides: float | int) -> RiskMetrics:
    base = RiskMetrics(
        total_return=0.12,
        max_drawdown=-0.10,
        cvar_5pct=-0.03,
        worst_five_day_mean=-0.04,
        extreme_loss_day_count=1,
        avg_abs_position=0.40,
        zero_position_day_ratio=0.10,
        total_turnover=3.0,
        observations=100,
    )
    return replace(base, **overrides)


def test_walk_forward_aggregate_rejects_summary_only_metrics() -> None:
    with pytest.raises(ValueError, match="daily return and position paths"):
        aggregate_walk_forward((_metrics(), _metrics(), _metrics()))


def test_walk_forward_aggregate_preserves_one_finite_extreme_loss_threshold() -> None:
    fold = compute_risk_metrics(
        [-0.10, 0.01],
        [0.30, 0.30],
        extreme_loss_threshold=-0.08,
    )

    aggregate = aggregate_walk_forward((fold, fold, fold))

    assert aggregate.metrics.extreme_loss_threshold == -0.08
    assert aggregate.fold_extreme_loss_thresholds == (-0.08, -0.08, -0.08)


def test_walk_forward_aggregate_retains_immutable_folds_and_derives_counts() -> None:
    positive = compute_risk_metrics([0.01], [0.30])
    negative = compute_risk_metrics([-0.01], [0.30])

    aggregate = aggregate_walk_forward((positive, negative, positive))

    assert aggregate.folds == (positive, negative, positive)
    assert aggregate.fold_count == 3
    assert aggregate.positive_return_folds == 2
    coerced = WalkForwardAggregate(
        metrics=aggregate.metrics,
        folds=[positive, negative, positive],  # type: ignore[arg-type]
    )
    assert isinstance(coerced.folds, tuple)
    with pytest.raises(FrozenInstanceError):
        aggregate.folds = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        WalkForwardAggregate(  # type: ignore[call-arg]
            metrics=aggregate.metrics,
            positive_return_folds=99,
            fold_count=99,
        )


def test_walk_forward_aggregate_rejects_mixed_or_nonfinite_extreme_loss_thresholds() -> None:
    standard = compute_risk_metrics([-0.10, 0.01], [0.30, 0.30], extreme_loss_threshold=-0.08)
    different = compute_risk_metrics([-0.10, 0.01], [0.30, 0.30], extreme_loss_threshold=-2.0)
    nonfinite = compute_risk_metrics([-0.10, 0.01], [0.30, 0.30], extreme_loss_threshold=np.nan)

    with pytest.raises(ValueError, match="one finite extreme-loss threshold"):
        aggregate_walk_forward((standard, different))
    with pytest.raises(ValueError, match="one finite extreme-loss threshold"):
        aggregate_walk_forward((standard, nonfinite))


def test_promotion_requires_test_and_walk_forward_to_beat_every_shared_gate() -> None:
    candidate_fold = compute_risk_metrics([0.0015] * 10, [0.40] * 10)
    legacy_fold = compute_risk_metrics([0.001] * 9 + [0.01], [0.60] * 10)
    candidate_test = compute_risk_metrics(
        candidate_fold.daily_returns * 3,
        candidate_fold.positions * 3,
    )
    legacy_test = compute_risk_metrics(legacy_fold.daily_returns * 3, legacy_fold.positions * 3)
    candidate_walk_forward = aggregate_walk_forward((candidate_fold, candidate_fold, candidate_fold))
    legacy_walk_forward = aggregate_walk_forward((legacy_fold, legacy_fold, legacy_fold))

    result = evaluate_promotion(
        candidate_test,
        legacy_test,
        candidate_walk_forward,
        legacy_walk_forward,
        spec=RiskAdjustedGateSpec(max_total_turnover=20.0),
        data_quality_valid=True,
        label_leakage_free=True,
        test_time_selection_free=True,
        labels_complete=True,
    )

    assert result.promoted is True
    assert result.decision == "promoted_risk_adjusted_candidate"
    assert result.failed_checks == ()
    assert all(result.checks.values())


def test_promotion_integrity_flags_default_to_fail_closed() -> None:
    metrics = _metrics()
    walk_forward = WalkForwardAggregate(metrics=metrics, folds=())

    result = evaluate_promotion(metrics, metrics, walk_forward, walk_forward)

    assert result.promoted is False
    assert result.checks["data_quality_valid"] is False
    assert result.checks["label_leakage_free"] is False
    assert result.checks["test_time_selection_free"] is False
    assert result.checks["labels_complete"] is False


def test_promotion_integrity_flags_only_accept_literal_true() -> None:
    metrics = _metrics()
    walk_forward = WalkForwardAggregate(metrics=metrics, folds=())

    result = evaluate_promotion(
        metrics,
        metrics,
        walk_forward,
        walk_forward,
        data_quality_valid=np.nan,  # type: ignore[arg-type]
        label_leakage_free=1,  # type: ignore[arg-type]
        test_time_selection_free=np.bool_(True),  # type: ignore[arg-type]
        labels_complete="true",  # type: ignore[arg-type]
    )

    assert result.promoted is False
    assert result.checks["data_quality_valid"] is False
    assert result.checks["label_leakage_free"] is False
    assert result.checks["test_time_selection_free"] is False
    assert result.checks["labels_complete"] is False


def test_promotion_fails_closed_and_reports_every_failed_gate() -> None:
    bad = _metrics(
        total_return=0.01,
        max_drawdown=-0.25,
        cvar_5pct=-0.08,
        worst_five_day_mean=-0.09,
        extreme_loss_day_count=5,
        avg_abs_position=0.10,
        zero_position_day_ratio=0.80,
        total_turnover=25.0,
    )
    legacy = _metrics(total_return=0.20, max_drawdown=-0.10)
    bad_walk_forward = WalkForwardAggregate(metrics=bad, folds=())
    legacy_walk_forward = WalkForwardAggregate(metrics=legacy, folds=())

    result = evaluate_promotion(
        bad,
        legacy,
        bad_walk_forward,
        legacy_walk_forward,
        spec=RiskAdjustedGateSpec(max_total_turnover=20.0),
        data_quality_valid=False,
        label_leakage_free=False,
        test_time_selection_free=False,
        labels_complete=False,
    )

    assert result.promoted is False
    assert result.decision == "not_promoted_failed_risk_adjusted_gates"
    assert set(result.failed_checks) == {
        "data_quality_valid",
        "label_leakage_free",
        "test_time_selection_free",
        "labels_complete",
        "test_candidate_metrics_valid",
        "test_legacy_metrics_valid",
        "walk_forward_candidate_metrics_valid",
        "walk_forward_legacy_metrics_valid",
        "walk_forward_candidate_structure_valid",
        "walk_forward_legacy_structure_valid",
        "candidate_walk_forward_fold_count",
        "legacy_walk_forward_fold_count",
        "test_max_drawdown",
        "test_participation",
        "test_zero_position_day_ratio",
        "test_positive_return_capture",
        "test_return_to_drawdown",
        "test_cvar_5pct",
        "test_worst_five_day_mean",
        "test_extreme_loss_day_count",
        "test_total_turnover",
        "walk_forward_max_drawdown",
        "walk_forward_participation",
        "walk_forward_zero_position_day_ratio",
        "walk_forward_positive_return_capture",
        "walk_forward_return_to_drawdown",
        "walk_forward_cvar_5pct",
        "walk_forward_worst_five_day_mean",
        "walk_forward_extreme_loss_day_count",
        "walk_forward_total_turnover",
        "walk_forward_positive_folds",
    }


def test_promotion_rejects_nonfinite_and_structurally_invalid_metrics() -> None:
    candidate_test = replace(_metrics(), total_return=float("inf"))
    legacy_test = replace(_metrics(), zero_position_day_ratio=1.10)
    candidate_walk_forward = WalkForwardAggregate(
        metrics=replace(_metrics(), observations=-1),
        folds=(replace(_metrics(), observations=-1),) * 3,
    )
    legacy_walk_forward = WalkForwardAggregate(
        metrics=replace(_metrics(), avg_abs_position=-0.10),
        folds=(replace(_metrics(), avg_abs_position=-0.10),) * 3,
    )

    result = evaluate_promotion(
        candidate_test,
        legacy_test,
        candidate_walk_forward,
        legacy_walk_forward,
    )

    assert result.promoted is False
    assert result.checks["test_candidate_metrics_valid"] is False
    assert result.checks["test_legacy_metrics_valid"] is False
    assert result.checks["walk_forward_candidate_metrics_valid"] is False
    assert result.checks["walk_forward_legacy_metrics_valid"] is False
    assert result.checks["walk_forward_candidate_structure_valid"] is False
    assert result.checks["walk_forward_legacy_structure_valid"] is False


@pytest.mark.parametrize(
    ("field_name", "delta"),
    [
        ("total_return", 0.01),
        ("max_drawdown", -0.01),
        ("cvar_5pct", -0.01),
        ("worst_five_day_mean", -0.01),
        ("extreme_loss_day_count", 1),
        ("avg_abs_position", 0.10),
        ("zero_position_day_ratio", 0.10),
        ("total_turnover", 0.10),
        ("observations", 1),
    ],
)
def test_promotion_recomputes_path_backed_metrics_before_accepting_them(
    field_name: str,
    delta: float | int,
) -> None:
    metrics = compute_risk_metrics(
        [0.01, -0.01, 0.02, -0.02, 0.03, -0.03],
        [0.20, 0.20, 0.00, 0.30, 0.30, 0.10],
    )
    tampered = replace(metrics, **{field_name: getattr(metrics, field_name) + delta})
    walk_forward = aggregate_walk_forward((metrics, metrics, metrics))

    result = evaluate_promotion(
        tampered,
        metrics,
        walk_forward,
        walk_forward,
        data_quality_valid=True,
        label_leakage_free=True,
        test_time_selection_free=True,
        labels_complete=True,
    )

    assert result.promoted is False
    assert result.checks["test_candidate_metrics_valid"] is False


def test_promotion_rejects_incomparable_extreme_loss_thresholds() -> None:
    candidate_fold = compute_risk_metrics(
        [-0.10, 0.01],
        [0.30, 0.30],
        extreme_loss_threshold=-2.0,
    )
    legacy_fold = compute_risk_metrics(
        [-0.10, 0.01],
        [0.30, 0.30],
        extreme_loss_threshold=-0.08,
    )
    candidate_walk_forward = aggregate_walk_forward((candidate_fold, candidate_fold, candidate_fold))
    legacy_walk_forward = aggregate_walk_forward((legacy_fold, legacy_fold, legacy_fold))

    result = evaluate_promotion(
        candidate_fold,
        legacy_fold,
        candidate_walk_forward,
        legacy_walk_forward,
        data_quality_valid=True,
        label_leakage_free=True,
        test_time_selection_free=True,
        labels_complete=True,
    )

    assert result.promoted is False
    assert result.checks["test_extreme_loss_threshold_comparable"] is False
    assert result.checks["walk_forward_extreme_loss_threshold_comparable"] is False


def test_promotion_requires_matched_test_and_walk_forward_observation_evidence() -> None:
    candidate_fold = compute_risk_metrics([0.01, -0.01], [0.30, 0.30])
    legacy_fold = compute_risk_metrics([0.01, -0.01, 0.02], [0.30, 0.30, 0.30])
    candidate_walk_forward = aggregate_walk_forward((candidate_fold, candidate_fold))
    legacy_walk_forward = aggregate_walk_forward((legacy_fold, legacy_fold, legacy_fold))

    result = evaluate_promotion(
        candidate_fold,
        legacy_fold,
        candidate_walk_forward,
        legacy_walk_forward,
        data_quality_valid=True,
        label_leakage_free=True,
        test_time_selection_free=True,
        labels_complete=True,
    )

    assert result.promoted is False
    assert result.checks["test_observation_count_matched"] is False
    assert result.checks["walk_forward_fold_counts_matched"] is False
    assert result.checks["candidate_walk_forward_fold_count"] is False
    assert result.checks["legacy_walk_forward_fold_count"] is True
    assert result.checks["walk_forward_observation_count_matched"] is False
