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


def test_risk_metrics_cover_drawdown_tail_participation_and_turnover() -> None:
    returns = pd.Series([0.10, -0.10, -0.20, 0.05, 0.02, -0.01])
    positions = pd.Series([0.50, 0.50, 0.00, 0.25, -0.25, -0.25])

    metrics = compute_risk_metrics(returns, positions)

    cumulative_return = returns.cumsum()
    assert metrics.total_return == pytest.approx(float(returns.sum()))
    assert metrics.max_drawdown == pytest.approx(float((cumulative_return - cumulative_return.cummax()).min()))
    assert metrics.cvar_5pct == pytest.approx(-0.20)
    assert metrics.worst_five_day_mean == pytest.approx(float(returns.nsmallest(5).mean()))
    assert metrics.extreme_loss_day_count == 2
    assert metrics.avg_abs_position == pytest.approx(float(positions.abs().mean()))
    assert metrics.zero_position_day_ratio == pytest.approx(1.0 / 6.0)
    assert metrics.total_turnover == pytest.approx(1.75)
    assert metrics.observations == 6
    assert metrics.return_to_drawdown == pytest.approx(metrics.total_return / abs(metrics.max_drawdown))


def test_walk_forward_aggregate_is_weighted_and_counts_positive_folds() -> None:
    first = compute_risk_metrics(pd.Series([0.02, -0.01]), pd.Series([0.2, 0.2]))
    second = compute_risk_metrics(pd.Series([-0.03, -0.01, 0.01]), pd.Series([0.4, 0.0, 0.4]))
    third = compute_risk_metrics(pd.Series([0.01, 0.01]), pd.Series([0.3, 0.3]))

    aggregate = aggregate_walk_forward((first, second, third))

    assert aggregate.fold_count == 3
    assert aggregate.positive_return_folds == 2
    assert aggregate.metrics.observations == 7
    assert aggregate.metrics.total_return == pytest.approx(
        first.total_return + second.total_return + third.total_return
    )
    assert aggregate.metrics.max_drawdown == min(
        first.max_drawdown,
        second.max_drawdown,
        third.max_drawdown,
    )
    assert aggregate.metrics.total_turnover == pytest.approx(
        first.total_turnover + second.total_turnover + third.total_turnover
    )


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


def test_promotion_requires_test_and_walk_forward_to_beat_every_shared_gate() -> None:
    candidate_test = _metrics()
    legacy_test = _metrics(
        total_return=0.15,
        max_drawdown=-0.20,
        cvar_5pct=-0.04,
        worst_five_day_mean=-0.05,
        extreme_loss_day_count=2,
    )
    candidate_walk_forward = WalkForwardAggregate(_metrics(total_return=0.18, max_drawdown=-0.12), 2, 3)
    legacy_walk_forward = WalkForwardAggregate(
        _metrics(
            total_return=0.22,
            max_drawdown=-0.25,
            cvar_5pct=-0.04,
            worst_five_day_mean=-0.05,
            extreme_loss_day_count=2,
        ),
        3,
        3,
    )

    result = evaluate_promotion(
        candidate_test,
        legacy_test,
        candidate_walk_forward,
        legacy_walk_forward,
        spec=RiskAdjustedGateSpec(max_total_turnover=20.0),
    )

    assert result.promoted is True
    assert result.decision == "promoted_risk_adjusted_candidate"
    assert result.failed_checks == ()
    assert all(result.checks.values())


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
    bad_walk_forward = WalkForwardAggregate(bad, 1, 3)
    legacy_walk_forward = WalkForwardAggregate(legacy, 3, 3)

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
