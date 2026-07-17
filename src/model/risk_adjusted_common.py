from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SegmentWindow:
    """Inclusive, immutable boundaries for one model-evaluation segment."""

    name: str
    start: pd.Timestamp | str
    end: pd.Timestamp | str

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        start = pd.Timestamp(self.start)
        end = pd.Timestamp(self.end)
        if not name:
            raise ValueError("segment name must not be empty")
        if start > end:
            raise ValueError("segment start must not be after end")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True)
class RiskAdjustedLabels:
    labels: pd.DataFrame
    train_crash_threshold: float
    horizon: int
    purge_rows: int
    purged_boundary_rows: Mapping[str, int]
    skipped_invalid_rows: Mapping[str, int]
    missing_label_segments: tuple[str, ...]

    @property
    def labels_complete(self) -> bool:
        return not self.missing_label_segments and all(
            count == 0 for count in self.skipped_invalid_rows.values()
        )


@dataclass(frozen=True)
class RiskMetrics:
    total_return: float
    max_drawdown: float
    cvar_5pct: float
    worst_five_day_mean: float
    extreme_loss_day_count: int
    avg_abs_position: float
    zero_position_day_ratio: float
    total_turnover: float
    observations: int
    daily_returns: tuple[float, ...] = field(default=(), repr=False, compare=False)
    positions: tuple[float, ...] = field(default=(), repr=False, compare=False)
    daily_turnover: tuple[float, ...] = field(default=(), repr=False, compare=False)
    turnover_is_explicit: bool = field(default=False, repr=False, compare=False)
    extreme_loss_threshold: float = field(default=-0.08, repr=False, compare=False)

    @property
    def return_to_drawdown(self) -> float:
        drawdown = abs(float(self.max_drawdown))
        if drawdown > np.finfo("float64").eps:
            return float(self.total_return) / drawdown
        if self.total_return > 0.0:
            return float("inf")
        if self.total_return < 0.0:
            return float("-inf")
        return 0.0


@dataclass(frozen=True)
class WalkForwardAggregate:
    metrics: RiskMetrics
    folds: tuple[RiskMetrics, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "folds", tuple(self.folds))

    @property
    def positive_return_folds(self) -> int:
        return sum(fold.total_return > 0.0 for fold in self.folds)

    @property
    def fold_count(self) -> int:
        return len(self.folds)

    @property
    def fold_extreme_loss_thresholds(self) -> tuple[float, ...]:
        return tuple(float(fold.extreme_loss_threshold) for fold in self.folds)


@dataclass(frozen=True)
class RiskAdjustedGateSpec:
    max_drawdown: float = -0.20
    min_avg_abs_position: float = 0.15
    max_zero_position_day_ratio: float = 0.70
    min_positive_walk_forward_folds: int = 2
    expected_walk_forward_folds: int = 3
    min_return_to_drawdown_multiple: float = 1.25
    min_legacy_return_capture: float = 0.50
    max_total_turnover: float = 20.0

    def __post_init__(self) -> None:
        if self.max_drawdown > 0.0:
            raise ValueError("max_drawdown must be non-positive")
        if not 0.0 <= self.min_avg_abs_position <= 1.0:
            raise ValueError("min_avg_abs_position must be between zero and one")
        if not 0.0 <= self.max_zero_position_day_ratio <= 1.0:
            raise ValueError("max_zero_position_day_ratio must be between zero and one")
        if not 0 <= self.min_positive_walk_forward_folds <= self.expected_walk_forward_folds:
            raise ValueError("positive fold gate must fit within the expected fold count")
        if self.min_return_to_drawdown_multiple < 0.0:
            raise ValueError("return-to-drawdown multiple must be non-negative")
        if self.min_legacy_return_capture < 0.0:
            raise ValueError("legacy return capture must be non-negative")
        if self.max_total_turnover < 0.0:
            raise ValueError("max_total_turnover must be non-negative")


@dataclass(frozen=True)
class PromotionEvaluation:
    promoted: bool
    decision: str
    checks: Mapping[str, bool]
    failed_checks: tuple[str, ...]


def purge_segment_tail(frame: pd.Series | pd.DataFrame, *, horizon: int = 5) -> pd.Series | pd.DataFrame:
    """Remove decision rows whose labels could touch a segment boundary."""

    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    keep = max(len(frame) - (horizon + 1), 0)
    return frame.iloc[:keep].copy()


def build_risk_adjusted_labels(
    next_open_returns: pd.Series,
    *,
    segments: Sequence[SegmentWindow],
    train_segment: str = "train",
    horizon: int = 5,
    crash_quantile: float = 0.10,
    daily_crash_threshold: float = -0.08,
) -> RiskAdjustedLabels:
    """Build boundary-safe future labels with one train-fitted crash cutoff."""

    if horizon != 5:
        raise ValueError("risk-adjusted label horizon is fixed at five trading days")
    horizon = 5
    if not 0.0 < crash_quantile < 1.0:
        raise ValueError("crash_quantile must be between zero and one")
    returns = pd.to_numeric(next_open_returns, errors="coerce").astype("float64")
    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise ValueError("next_open_returns index must be sorted and unique")
    windows = tuple(segments)
    names = tuple(window.name for window in windows)
    if not windows:
        raise ValueError("at least one segment is required")
    if len(set(names)) != len(names):
        raise ValueError("segment names must be unique")
    if train_segment not in names:
        raise ValueError(f"train segment is missing: {train_segment}")
    ordered_windows = sorted(windows, key=lambda window: window.start)
    if any(right.start <= left.end for left, right in zip(ordered_windows, ordered_windows[1:])):
        raise ValueError("segments must not overlap")
    train_window = next(window for window in windows if window.name == train_segment)
    if any(window.start <= train_window.end for window in windows if window.name != train_segment):
        raise ValueError("train segment must precede every non-train segment")

    rows: list[dict[str, object]] = []
    seen_dates: set[object] = set()
    purged: dict[str, int] = {}
    skipped_invalid: dict[str, int] = {}
    for window in windows:
        segment_returns = returns.loc[window.start : window.end]
        purged[window.name] = min(len(segment_returns), horizon + 1)
        skipped_invalid[window.name] = 0
        eligible = purge_segment_tail(segment_returns, horizon=horizon)
        for offset, decision_date in enumerate(eligible.index):
            future = segment_returns.iloc[offset : offset + horizon]
            values = future.to_numpy(dtype="float64")
            if len(values) != horizon or not np.isfinite(values).all():
                skipped_invalid[window.name] += 1
                continue
            if decision_date in seen_dates:
                raise ValueError("segments must not overlap")
            seen_dates.add(decision_date)
            future_return = float(np.prod(1.0 + values) - 1.0)
            rows.append(
                {
                    "date": decision_date,
                    "segment": window.name,
                    "future_return_5d": future_return,
                    "future_downside_5d": max(-future_return, 0.0),
                    "future_min_daily_return_5d": float(values.min()),
                }
            )

    if not rows:
        raise ValueError("segments do not contain any complete labels")
    labels = pd.DataFrame(rows).set_index("date")
    present_segments = set(labels["segment"])
    missing_segments = tuple(name for name in names if name not in present_segments)
    train_labels = labels.loc[labels["segment"] == train_segment, "future_return_5d"]
    if train_labels.empty:
        raise ValueError("train segment does not contain any complete labels")
    crash_threshold = float(train_labels.quantile(crash_quantile))
    labels["future_crash_5d"] = (
        (labels["future_return_5d"] <= crash_threshold)
        | (labels["future_min_daily_return_5d"] <= float(daily_crash_threshold))
    ).astype("int8")
    return RiskAdjustedLabels(
        labels=labels,
        train_crash_threshold=crash_threshold,
        horizon=horizon,
        purge_rows=horizon + 1,
        purged_boundary_rows=MappingProxyType(purged),
        skipped_invalid_rows=MappingProxyType(skipped_invalid),
        missing_label_segments=missing_segments,
    )


def soft_budget_position(
    mu_score: float | Sequence[float] | pd.Series,
    tail_score: float | Sequence[float] | pd.Series,
    crash_probability: float | Sequence[float] | pd.Series,
    *,
    data_quality_valid: bool | Sequence[bool] | pd.Series,
    floor: float = 0.15,
    cap: float = 0.75,
) -> float | np.ndarray | pd.Series:
    """Map opportunity and tail scores to a bounded continuous allocation."""

    floor = float(floor)
    cap = float(cap)
    if not 0.0 <= floor < cap <= 1.0:
        raise ValueError("floor must be smaller than cap and both must be between zero and one")

    series_inputs = [
        value
        for value in (mu_score, tail_score, crash_probability, data_quality_valid)
        if isinstance(value, pd.Series)
    ]
    index = series_inputs[0].index if series_inputs else None

    def values(value: object, *, dtype: str) -> np.ndarray:
        if index is not None:
            if isinstance(value, pd.Series):
                return value.reindex(index).to_numpy(dtype=dtype)
            return np.full(len(index), value, dtype=dtype)
        return np.asarray(value, dtype=dtype)

    mu, tail, crash, quality = np.broadcast_arrays(
        values(mu_score, dtype="float64"),
        values(tail_score, dtype="float64"),
        values(crash_probability, dtype="float64"),
        values(data_quality_valid, dtype="float64"),
    )
    sigmoid = np.empty_like(mu, dtype="float64")
    non_negative = mu >= 0.0
    sigmoid[non_negative] = 1.0 / (1.0 + np.exp(-mu[non_negative]))
    exp_mu = np.exp(mu[~non_negative])
    sigmoid[~non_negative] = exp_mu / (1.0 + exp_mu)
    risk_multiplier = np.clip(1.0 - crash - 0.5 * tail, 0.0, 1.0)
    position = np.clip(floor + (cap - floor) * sigmoid * risk_multiplier, floor, cap)
    finite = np.isfinite(mu) & np.isfinite(tail) & np.isfinite(crash) & np.isfinite(position)
    valid = np.isfinite(quality) & (quality == 1.0)
    position = np.where(valid & finite, position, 0.0)
    if index is not None:
        return pd.Series(position, index=index, dtype="float64", name="target_position")
    if position.ndim == 0:
        return float(position)
    return position


def compute_risk_metrics(
    daily_returns: pd.Series | Sequence[float],
    positions: pd.Series | Sequence[float],
    *,
    extreme_loss_threshold: float = -0.08,
    daily_turnover: pd.Series | Sequence[float] | None = None,
) -> RiskMetrics:
    """Compute the shared candidate/legacy metrics on decimal daily returns."""

    returns = pd.Series(daily_returns, dtype="float64")
    position = pd.Series(positions, dtype="float64")
    if returns.empty or len(returns) != len(position):
        raise ValueError("daily_returns and positions must have the same non-zero length")
    if not np.isfinite(returns.to_numpy()).all() or not np.isfinite(position.to_numpy()).all():
        raise ValueError("daily_returns and positions must be finite")
    turnover_is_explicit = daily_turnover is not None
    if daily_turnover is None:
        turnover = position.diff().abs().fillna(position.abs())
    else:
        turnover = pd.Series(daily_turnover, dtype="float64")
        if len(turnover) != len(position) or not np.isfinite(turnover.to_numpy()).all():
            raise ValueError("daily_turnover must be finite and aligned with positions")
        if turnover.lt(0.0).any():
            raise ValueError("daily_turnover must be non-negative")
    nav = pd.Series(np.r_[1.0, (1.0 + returns.to_numpy()).cumprod()])
    drawdown = nav / nav.cummax() - 1.0
    tail_count = max(1, int(np.ceil(len(returns) * 0.05)))
    return RiskMetrics(
        total_return=float(nav.iloc[-1] - 1.0),
        max_drawdown=float(drawdown.min()),
        cvar_5pct=float(returns.nsmallest(tail_count).mean()),
        worst_five_day_mean=float(returns.nsmallest(min(5, len(returns))).mean()),
        extreme_loss_day_count=int(returns.le(float(extreme_loss_threshold)).sum()),
        avg_abs_position=float(position.abs().mean()),
        zero_position_day_ratio=float(position.abs().le(1e-12).mean()),
        total_turnover=float(turnover.sum()),
        observations=int(len(returns)),
        daily_returns=tuple(float(value) for value in returns),
        positions=tuple(float(value) for value in position),
        daily_turnover=tuple(float(value) for value in turnover),
        turnover_is_explicit=turnover_is_explicit,
        extreme_loss_threshold=float(extreme_loss_threshold),
    )


def aggregate_walk_forward(folds: Sequence[RiskMetrics]) -> WalkForwardAggregate:
    """Create one observation-weighted, conservative three-fold summary."""

    metrics = tuple(folds)
    if not metrics:
        raise ValueError("at least one walk-forward fold is required")
    if any(
        len(item.daily_returns) != item.observations
        or len(item.positions) != item.observations
        or (item.daily_turnover and len(item.daily_turnover) != item.observations)
        for item in metrics
    ):
        raise ValueError("walk-forward metrics must include daily return and position paths")
    thresholds = tuple(float(item.extreme_loss_threshold) for item in metrics)
    if not all(np.isfinite(threshold) for threshold in thresholds) or not all(
        np.isclose(threshold, thresholds[0], rtol=1e-12, atol=1e-12)
        for threshold in thresholds[1:]
    ):
        raise ValueError("walk-forward folds must use one finite extreme-loss threshold")
    combined_returns = tuple(value for item in metrics for value in item.daily_returns)
    combined_positions = tuple(value for item in metrics for value in item.positions)
    combined_turnover = (
        tuple(value for item in metrics for value in item.daily_turnover)
        if all(item.turnover_is_explicit for item in metrics)
        else None
    )
    aggregate = compute_risk_metrics(
        combined_returns,
        combined_positions,
        extreme_loss_threshold=thresholds[0],
        daily_turnover=combined_turnover,
    )
    return WalkForwardAggregate(
        metrics=aggregate,
        folds=metrics,
    )


def _risk_metrics_are_valid(metrics: RiskMetrics) -> bool:
    scalar_values = (
        metrics.total_return,
        metrics.max_drawdown,
        metrics.cvar_5pct,
        metrics.worst_five_day_mean,
        metrics.avg_abs_position,
        metrics.zero_position_day_ratio,
        metrics.total_turnover,
        metrics.extreme_loss_threshold,
    )
    if not all(np.isfinite(float(value)) for value in scalar_values):
        return False
    integer_types = (int, np.integer)
    if isinstance(metrics.observations, bool) or not isinstance(metrics.observations, integer_types):
        return False
    if isinstance(metrics.extreme_loss_day_count, bool) or not isinstance(
        metrics.extreme_loss_day_count,
        integer_types,
    ):
        return False
    if metrics.observations <= 0:
        return False
    if not 0 <= metrics.extreme_loss_day_count <= metrics.observations:
        return False
    if not -1.0 <= metrics.total_return:
        return False
    if not -1.0 <= metrics.max_drawdown <= 0.0:
        return False
    if not -1.0 <= metrics.cvar_5pct or not -1.0 <= metrics.worst_five_day_mean:
        return False
    if not 0.0 <= metrics.avg_abs_position <= 1.0:
        return False
    if not 0.0 <= metrics.zero_position_day_ratio <= 1.0:
        return False
    if metrics.total_turnover < 0.0:
        return False
    has_returns = bool(metrics.daily_returns)
    has_positions = bool(metrics.positions)
    if not has_returns or not has_positions:
        return False
    if len(metrics.daily_returns) != metrics.observations or len(metrics.positions) != metrics.observations:
        return False
    returns = np.asarray(metrics.daily_returns, dtype="float64")
    positions = np.asarray(metrics.positions, dtype="float64")
    if not np.isfinite(returns).all() or not np.isfinite(positions).all():
        return False
    if (returns <= -1.0).any() or (np.abs(positions) > 1.0).any():
        return False
    recomputed = compute_risk_metrics(
        metrics.daily_returns,
        metrics.positions,
        extreme_loss_threshold=metrics.extreme_loss_threshold,
        daily_turnover=metrics.daily_turnover or None,
    )
    float_fields = (
        "total_return",
        "max_drawdown",
        "cvar_5pct",
        "worst_five_day_mean",
        "avg_abs_position",
        "zero_position_day_ratio",
        "total_turnover",
    )
    if not all(
        np.isclose(
            float(getattr(metrics, name)),
            float(getattr(recomputed, name)),
            rtol=1e-9,
            atol=1e-12,
        )
        for name in float_fields
    ):
        return False
    return (
        metrics.extreme_loss_day_count == recomputed.extreme_loss_day_count
        and metrics.observations == recomputed.observations
    )


def _walk_forward_structure_is_valid(aggregate: WalkForwardAggregate) -> bool:
    if not aggregate.folds:
        return False
    if not all(_risk_metrics_are_valid(fold) for fold in aggregate.folds):
        return False
    thresholds = aggregate.fold_extreme_loss_thresholds
    if not all(np.isfinite(float(threshold)) for threshold in thresholds):
        return False
    if not all(
        np.isclose(
            float(threshold),
            float(aggregate.metrics.extreme_loss_threshold),
            rtol=1e-12,
            atol=1e-12,
        )
        for threshold in thresholds
    ):
        return False
    canonical = aggregate_walk_forward(aggregate.folds)
    return (
        aggregate.metrics == canonical.metrics
        and aggregate.metrics.daily_returns == canonical.metrics.daily_returns
        and aggregate.metrics.positions == canonical.metrics.positions
        and np.isclose(
            aggregate.metrics.extreme_loss_threshold,
            canonical.metrics.extreme_loss_threshold,
            rtol=1e-12,
            atol=1e-12,
        )
    )


def _extreme_loss_thresholds_are_comparable(left: RiskMetrics, right: RiskMetrics) -> bool:
    left_threshold = float(left.extreme_loss_threshold)
    right_threshold = float(right.extreme_loss_threshold)
    return bool(
        np.isfinite(left_threshold)
        and np.isfinite(right_threshold)
        and np.isclose(left_threshold, right_threshold, rtol=1e-12, atol=1e-12)
    )


def evaluate_promotion(
    candidate_test: RiskMetrics,
    legacy_test: RiskMetrics,
    candidate_walk_forward: WalkForwardAggregate,
    legacy_walk_forward: WalkForwardAggregate,
    *,
    spec: RiskAdjustedGateSpec = RiskAdjustedGateSpec(),
    data_quality_valid: bool | None = None,
    label_leakage_free: bool | None = None,
    test_time_selection_free: bool | None = None,
    labels_complete: bool | None = None,
) -> PromotionEvaluation:
    """Evaluate every untouched-test and walk-forward promotion requirement."""

    checks: dict[str, bool] = {
        "data_quality_valid": data_quality_valid is True,
        "label_leakage_free": label_leakage_free is True,
        "test_time_selection_free": test_time_selection_free is True,
        "labels_complete": labels_complete is True,
        "test_candidate_metrics_valid": _risk_metrics_are_valid(candidate_test),
        "test_legacy_metrics_valid": _risk_metrics_are_valid(legacy_test),
        "walk_forward_candidate_metrics_valid": _risk_metrics_are_valid(candidate_walk_forward.metrics),
        "walk_forward_legacy_metrics_valid": _risk_metrics_are_valid(legacy_walk_forward.metrics),
        "walk_forward_candidate_structure_valid": _walk_forward_structure_is_valid(candidate_walk_forward),
        "walk_forward_legacy_structure_valid": _walk_forward_structure_is_valid(legacy_walk_forward),
        "test_extreme_loss_threshold_comparable": _extreme_loss_thresholds_are_comparable(
            candidate_test,
            legacy_test,
        ),
        "walk_forward_extreme_loss_threshold_comparable": _extreme_loss_thresholds_are_comparable(
            candidate_walk_forward.metrics,
            legacy_walk_forward.metrics,
        ),
        "test_observation_count_matched": candidate_test.observations == legacy_test.observations,
        "walk_forward_fold_counts_matched": (
            candidate_walk_forward.fold_count == legacy_walk_forward.fold_count
        ),
        "candidate_walk_forward_fold_count": (
            candidate_walk_forward.fold_count == spec.expected_walk_forward_folds
        ),
        "legacy_walk_forward_fold_count": (
            legacy_walk_forward.fold_count == spec.expected_walk_forward_folds
        ),
        "walk_forward_observation_count_matched": (
            candidate_walk_forward.metrics.observations
            == legacy_walk_forward.metrics.observations
        ),
    }

    def add_metric_checks(scope: str, candidate: RiskMetrics, legacy: RiskMetrics) -> None:
        checks[f"{scope}_max_drawdown"] = candidate.max_drawdown >= spec.max_drawdown
        checks[f"{scope}_participation"] = candidate.avg_abs_position >= spec.min_avg_abs_position
        checks[f"{scope}_zero_position_day_ratio"] = (
            candidate.zero_position_day_ratio <= spec.max_zero_position_day_ratio
        )
        checks[f"{scope}_positive_return_capture"] = (
            candidate.total_return >= spec.min_legacy_return_capture * legacy.total_return
        )
        checks[f"{scope}_return_to_drawdown"] = (
            candidate.return_to_drawdown
            >= spec.min_return_to_drawdown_multiple * legacy.return_to_drawdown
        )
        checks[f"{scope}_cvar_5pct"] = candidate.cvar_5pct >= legacy.cvar_5pct
        checks[f"{scope}_worst_five_day_mean"] = (
            candidate.worst_five_day_mean >= legacy.worst_five_day_mean
        )
        checks[f"{scope}_extreme_loss_day_count"] = (
            candidate.extreme_loss_day_count <= legacy.extreme_loss_day_count
        )
        checks[f"{scope}_total_turnover"] = candidate.total_turnover <= spec.max_total_turnover

    add_metric_checks("test", candidate_test, legacy_test)
    add_metric_checks("walk_forward", candidate_walk_forward.metrics, legacy_walk_forward.metrics)
    checks["walk_forward_positive_folds"] = (
        candidate_walk_forward.positive_return_folds >= spec.min_positive_walk_forward_folds
    )
    failed = tuple(name for name, passed in checks.items() if not passed)
    promoted = not failed
    return PromotionEvaluation(
        promoted=promoted,
        decision=(
            "promoted_risk_adjusted_candidate"
            if promoted
            else "not_promoted_failed_risk_adjusted_gates"
        ),
        checks=MappingProxyType(checks),
        failed_checks=failed,
    )


__all__ = [
    "PromotionEvaluation",
    "RiskAdjustedGateSpec",
    "RiskAdjustedLabels",
    "RiskMetrics",
    "SegmentWindow",
    "WalkForwardAggregate",
    "aggregate_walk_forward",
    "build_risk_adjusted_labels",
    "compute_risk_metrics",
    "evaluate_promotion",
    "purge_segment_tail",
    "soft_budget_position",
]
