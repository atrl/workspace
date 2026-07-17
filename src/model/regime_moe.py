from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from .risk_adjusted_common import soft_budget_position


REGIMES = ("trend", "reversal", "stress")


@dataclass(frozen=True)
class RegimeMoESpec:
    """Frozen search contract for the three-expert SOXL mixture route."""

    train_segment: str = "train"
    selection_segment: str = "valid"
    test_segment: str = "test"
    return_label: str = "future_return_5d"
    downside_label: str = "future_downside_5d"
    crash_label: str = "future_crash_5d"
    regime_return_feature: str = "lagged_return"
    regime_trend_feature: str = "trend"
    regime_volatility_feature: str = "volatility"
    trend_features: tuple[str, ...] = ("trend", "breadth", "cross_asset_momentum")
    reversal_features: tuple[str, ...] = (
        "oversold",
        "residual",
        "range",
        "volatility_gap",
    )
    stress_features: tuple[str, ...] = (
        "volatility",
        "volatility_gap",
        "range",
        "lagged_return",
    )
    gate_features: tuple[str, ...] = (
        "lagged_return",
        "trend",
        "volatility",
        "breadth",
        "volatility_gap",
    )
    ridge_alpha: float = 1.0
    catboost_iterations: int = 32
    min_regime_observations: int = 40
    floor_cap_candidates: tuple[tuple[float, float], ...] = (
        (0.15, 0.60),
        (0.15, 0.75),
        (0.20, 0.75),
    )
    random_state: int = 1729

    def __post_init__(self) -> None:
        if self.selection_segment != "valid":
            raise ValueError("selection_segment must be the validation segment")
        if self.test_segment in {self.train_segment, self.selection_segment}:
            raise ValueError("test segment must be separate from train and validation")
        if self.ridge_alpha <= 0.0 or self.catboost_iterations <= 0:
            raise ValueError("model parameters must be positive")
        if self.min_regime_observations < 1:
            raise ValueError("min_regime_observations must be positive")
        if not self.floor_cap_candidates:
            raise ValueError("floor/cap candidates must not be empty")
        for floor, cap in self.floor_cap_candidates:
            if not 0.0 <= float(floor) < float(cap) <= 1.0:
                raise ValueError("floor/cap candidates must satisfy 0 <= floor < cap <= 1")
        for feature_group in (
            self.trend_features,
            self.reversal_features,
            self.stress_features,
            self.gate_features,
        ):
            if not feature_group:
                raise ValueError("expert and gate feature groups must not be empty")


@dataclass(frozen=True)
class RegimeMoEResult:
    model_comparison: pd.DataFrame
    selected_candidate: str
    regime_thresholds: Mapping[str, float]
    regime_occupancy: Mapping[str, int]
    gate_probabilities_by_segment: Mapping[str, pd.DataFrame]
    expert_predictions_by_segment: Mapping[str, pd.DataFrame]
    predictions_by_segment: Mapping[str, pd.DataFrame]
    positions_by_segment: Mapping[str, pd.Series]
    failure_rows: pd.DataFrame
    selection_evidence: Mapping[str, Any]
    floor: float
    cap: float
    fitted_policy: "FittedRegimeMoEPolicy | None"


@dataclass(frozen=True)
class FittedRegimeMoEPolicy:
    """Train/validation-frozen, pickle-safe mixture policy."""

    feature_columns: tuple[str, ...]
    preprocessing_median: pd.Series
    preprocessing_mean: pd.Series
    preprocessing_scale: pd.Series
    gate_model: Any
    gate_features: tuple[str, ...]
    trend_model: Any
    trend_features: tuple[str, ...]
    reversal_model: Any
    reversal_features: tuple[str, ...]
    stress_model: Any
    stress_features: tuple[str, ...]
    expert_score_center: Mapping[str, float]
    expert_score_scale: Mapping[str, float]
    downside_scale: float
    floor: float
    cap: float

    def predict(
        self,
        features: pd.DataFrame,
        *,
        data_quality_valid: bool | pd.Series | Sequence[bool] | None = None,
        segment: str = "inference",
    ) -> pd.DataFrame:
        raw = _coerce_feature_frame(features).reindex(columns=list(self.feature_columns))
        intrinsic = pd.Series(
            np.isfinite(raw.to_numpy(dtype="float64")).all(axis=1),
            index=raw.index,
        )
        quality = _combine_quality(intrinsic, data_quality_valid)
        return _predict_policy_segment(self, raw, quality, segment=segment)


@dataclass
class _SegmentData:
    features: pd.DataFrame
    labels: pd.DataFrame
    quality: pd.Series


class _ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.value, dtype="float64")


class _ConstantGate:
    def __init__(self, probabilities: Sequence[float]) -> None:
        values = np.asarray(probabilities, dtype="float64")
        self.probabilities = values / values.sum()
        self.classes_ = np.arange(len(REGIMES), dtype="int64")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile(self.probabilities, (len(features), 1))


class _NonFinitePredictionError(ValueError):
    def __init__(self, *, segment: str, first_offending_date: str) -> None:
        super().__init__(f"non-finite prediction in {segment} at {first_offending_date}")
        self.segment = segment
        self.first_offending_date = first_offending_date


def _coerce_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("features must be a non-empty DataFrame")
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex) and "feature" in result.columns.get_level_values(0):
        result = result["feature"].copy()
    result.columns = [str(column) for column in result.columns]
    return result.apply(pd.to_numeric, errors="coerce").astype("float64")


def _prepare_learning_segment(dataset, segment: str, spec: RegimeMoESpec) -> _SegmentData:
    from qlib.data.dataset.handler import DataHandlerLP

    frame = dataset.prepare(
        segment,
        col_set=["feature", "label"],
        data_key=DataHandlerLP.DK_L,
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"segment {segment!r} is empty")
    labels_needed = (spec.return_label, spec.downside_label, spec.crash_label)
    if isinstance(frame.columns, pd.MultiIndex) and "feature" in frame.columns.get_level_values(0):
        features = _coerce_feature_frame(frame["feature"])
        if "label" not in frame.columns.get_level_values(0):
            raise ValueError(f"segment {segment!r} does not include qlib labels")
        labels = frame["label"].copy()
    else:
        missing = [label for label in labels_needed if label not in frame.columns]
        if missing:
            raise ValueError(f"segment {segment!r} is missing labels: {missing}")
        labels = frame.loc[:, list(labels_needed)].copy()
        features = _coerce_feature_frame(frame.drop(columns=list(labels_needed)))
    labels.columns = [str(column) for column in labels.columns]
    missing = [label for label in labels_needed if label not in labels.columns]
    if missing:
        raise ValueError(f"segment {segment!r} is missing labels: {missing}")
    labels = labels.loc[:, list(labels_needed)].apply(pd.to_numeric, errors="coerce").astype("float64")
    quality = pd.Series(
        np.isfinite(features.to_numpy(dtype="float64")).all(axis=1),
        index=features.index,
        name="data_quality_valid",
    )
    return _SegmentData(features=features, labels=labels, quality=quality)


def _prepare_inference_segment(dataset, segment: str) -> _SegmentData:
    from qlib.data.dataset.handler import DataHandlerLP

    features = _coerce_feature_frame(
        dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
    )
    quality = pd.Series(
        np.isfinite(features.to_numpy(dtype="float64")).all(axis=1),
        index=features.index,
        name="data_quality_valid",
    )
    return _SegmentData(features, pd.DataFrame(index=features.index), quality)


def _require_features(features: pd.DataFrame, names: Sequence[str], *, role: str) -> None:
    missing = [name for name in names if name not in features.columns]
    if missing:
        raise ValueError(f"{role} is missing required features: {missing}")


def _fit_preprocessor(train: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    median = train.replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    filled = train.replace([np.inf, -np.inf], np.nan).fillna(median)
    mean = filled.mean()
    scale = filled.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return median, mean, scale


def _transform(
    frame: pd.DataFrame, median: pd.Series, mean: pd.Series, scale: pd.Series
) -> pd.DataFrame:
    aligned = frame.reindex(columns=median.index).replace([np.inf, -np.inf], np.nan).fillna(median)
    return ((aligned - mean) / scale).fillna(0.0).astype("float64")


def _regime_thresholds(train: pd.DataFrame, spec: RegimeMoESpec) -> dict[str, float]:
    names = (
        spec.regime_return_feature,
        spec.regime_trend_feature,
        spec.regime_volatility_feature,
    )
    _require_features(train, names, role="regime labeling")
    return {
        "return_median": float(train[spec.regime_return_feature].median()),
        "trend_median": float(train[spec.regime_trend_feature].median()),
        "volatility_stress_quantile": float(train[spec.regime_volatility_feature].quantile(2.0 / 3.0)),
    }


def _assign_regimes(
    features: pd.DataFrame, thresholds: Mapping[str, float], spec: RegimeMoESpec
) -> pd.Series:
    stress = features[spec.regime_volatility_feature] >= thresholds["volatility_stress_quantile"]
    trend = (~stress) & (
        (features[spec.regime_trend_feature] >= thresholds["trend_median"])
        | (features[spec.regime_return_feature] >= thresholds["return_median"])
    )
    labels = np.where(stress, "stress", np.where(trend, "trend", "reversal"))
    return pd.Series(labels, index=features.index, name="regime")


def _finite_training_rows(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    mask = np.isfinite(features.to_numpy(dtype="float64")).all(axis=1) & np.isfinite(
        target.to_numpy(dtype="float64")
    )
    if not bool(mask.any()):
        raise ValueError("candidate has no finite training rows")
    return features.loc[mask], target.loc[mask].astype("float64")


def _fit_trend_expert(features: pd.DataFrame, target: pd.Series, spec: RegimeMoESpec) -> Any:
    x, y = _finite_training_rows(features, target)
    return Ridge(alpha=spec.ridge_alpha).fit(x, y)


def _fit_reversal_expert(features: pd.DataFrame, target: pd.Series, spec: RegimeMoESpec) -> Any:
    from catboost import CatBoostRegressor

    x, y = _finite_training_rows(features, target)
    model = CatBoostRegressor(
        iterations=spec.catboost_iterations,
        depth=3,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=spec.random_state,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x, y)
    return model


def _fit_stress_expert(features: pd.DataFrame, target: pd.Series, spec: RegimeMoESpec) -> Any:
    x, y = _finite_training_rows(features, target)
    return Ridge(alpha=spec.ridge_alpha).fit(x, y)


def _fit_gate(features: pd.DataFrame, labels: pd.Series, spec: RegimeMoESpec) -> Any:
    encoded = labels.map({name: position for position, name in enumerate(REGIMES)}).astype("int64")
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=spec.random_state,
        solver="lbfgs",
    ).fit(features, encoded)


def _prediction_date(index: pd.Index, position: int) -> str:
    value = index[position]
    if isinstance(value, tuple):
        level = index.names.index("datetime") if "datetime" in index.names else len(value) - 1
        value = value[level]
    try:
        return pd.Timestamp(value).isoformat()
    except (TypeError, ValueError):
        return str(value)


def _validate_values(values: Any, index: pd.Index, *, segment: str) -> np.ndarray:
    array = np.asarray(values, dtype="float64").reshape(-1)
    if len(array) != len(index):
        raise ValueError(f"candidate produced {len(array)} predictions for {len(index)} rows")
    finite = np.isfinite(array)
    if not finite.all():
        position = int(np.flatnonzero(~finite)[0])
        raise _NonFinitePredictionError(
            segment=segment, first_offending_date=_prediction_date(index, position)
        )
    return array


def _gate_probabilities(model: Any, features: pd.DataFrame, *, segment: str) -> pd.DataFrame:
    raw = np.asarray(model.predict_proba(features), dtype="float64")
    if raw.shape != (len(features), len(REGIMES)):
        raise ValueError("gate must predict all three regime probabilities")
    classes = tuple(int(value) for value in model.classes_)
    if classes != tuple(range(len(REGIMES))):
        order = [classes.index(position) for position in range(len(REGIMES))]
        raw = raw[:, order]
    if not np.isfinite(raw).all() or (raw < 0.0).any():
        flat = int(np.flatnonzero(~np.isfinite(raw) | (raw < 0.0))[0])
        raise _NonFinitePredictionError(
            segment=segment,
            first_offending_date=_prediction_date(features.index, flat // len(REGIMES)),
        )
    row_sum = raw.sum(axis=1)
    if (row_sum <= 0.0).any():
        raise ValueError("gate probability row has non-positive mass")
    raw = raw / row_sum[:, None]
    return pd.DataFrame(raw, index=features.index, columns=list(REGIMES), dtype="float64")


def _predict_policy_segment(
    policy: FittedRegimeMoEPolicy,
    features: pd.DataFrame,
    quality: pd.Series,
    *,
    segment: str,
) -> pd.DataFrame:
    transformed = _transform(
        features, policy.preprocessing_median, policy.preprocessing_mean, policy.preprocessing_scale
    )
    gate = _gate_probabilities(
        policy.gate_model, transformed.loc[:, list(policy.gate_features)], segment=segment
    )
    trend_raw = _validate_values(
        policy.trend_model.predict(transformed.loc[:, list(policy.trend_features)]),
        transformed.index,
        segment=segment,
    )
    reversal_raw = _validate_values(
        policy.reversal_model.predict(transformed.loc[:, list(policy.reversal_features)]),
        transformed.index,
        segment=segment,
    )
    stress_downside = _validate_values(
        policy.stress_model.predict(transformed.loc[:, list(policy.stress_features)]),
        transformed.index,
        segment=segment,
    )
    expert_raw = pd.DataFrame(
        {
            "trend": trend_raw,
            "reversal": reversal_raw,
            "stress": -np.maximum(stress_downside, 0.0),
        },
        index=transformed.index,
        dtype="float64",
    )
    expert_scores = pd.DataFrame(index=transformed.index)
    for regime in REGIMES:
        expert_scores[regime] = (
            expert_raw[regime] - float(policy.expert_score_center[regime])
        ) / float(policy.expert_score_scale[regime])
    expected_score = (gate * expert_scores).sum(axis=1)
    tail_score = (
        pd.Series(np.maximum(stress_downside, 0.0), index=transformed.index)
        / float(policy.downside_scale)
    ).clip(0.0, 1.0)
    target = soft_budget_position(
        expected_score,
        tail_score * gate["stress"],
        gate["stress"],
        data_quality_valid=quality,
        floor=policy.floor,
        cap=policy.cap,
    )
    return pd.DataFrame(
        {
            "trend_probability": gate["trend"],
            "reversal_probability": gate["reversal"],
            "stress_probability": gate["stress"],
            "trend_expert_raw": expert_raw["trend"],
            "reversal_expert_raw": expert_raw["reversal"],
            "stress_expert_raw": expert_raw["stress"],
            "expected_score": expected_score,
            "tail_score": tail_score,
            "target_position": target,
        },
        index=transformed.index,
    ).astype("float64")


def _combine_quality(
    intrinsic: pd.Series,
    external: bool | pd.Series | Sequence[bool] | None,
) -> pd.Series:
    if external is None:
        return intrinsic.astype(bool)
    if isinstance(external, pd.Series):
        supplied = external.reindex(intrinsic.index).fillna(False).astype(bool)
    elif np.isscalar(external):
        supplied = pd.Series(bool(external), index=intrinsic.index)
    else:
        array = np.asarray(external)
        if len(array) != len(intrinsic):
            raise ValueError("data quality input must align with its segment")
        supplied = pd.Series(array.astype(bool), index=intrinsic.index)
    return intrinsic.astype(bool) & supplied


def _quality_with_evidence(
    segment: _SegmentData,
    external: bool | pd.Series | Sequence[bool] | None,
) -> tuple[pd.Series, dict[str, Any]]:
    quality = _combine_quality(segment.quality, external)
    invalid = ~quality
    external_invalid = bool((~_combine_quality(pd.Series(True, index=quality.index), external)).any()) \
        if external is not None else False
    return quality, {
        "invalid_count": int(invalid.sum()),
        "first_offending_date": (
            _prediction_date(quality.index, int(np.flatnonzero(invalid.to_numpy())[0]))
            if invalid.any()
            else None
        ),
        "reason_code": (
            "missing_or_stale_required_source"
            if external_invalid
            else "non_finite_required_feature" if invalid.any() else ""
        ),
    }


def _budget_utility(realized: pd.Series, position: pd.Series) -> float:
    aligned = pd.concat([realized.rename("return"), position.rename("position")], axis=1).dropna()
    if aligned.empty:
        return float("-inf")
    strategy_return = aligned["return"].to_numpy(dtype="float64") * aligned[
        "position"
    ].to_numpy(dtype="float64")
    downside = np.minimum(strategy_return, 0.0)
    downside_scale = float(np.sqrt(np.mean(np.square(downside))))
    return float(np.mean(strategy_return) / max(downside_scale, 1e-6))


def _success_row(candidate: str, role: str) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "model": candidate,
        "role": role,
        "status": "ok",
        "validation_score": np.nan,
        "selected": False,
        "reason_code": "",
        "failure_type": "",
        "detail": "",
    }


def _failure_row(candidate: str, role: str, exc: Exception) -> dict[str, Any]:
    non_finite = isinstance(exc, _NonFinitePredictionError)
    return {
        "candidate": candidate,
        "model": candidate,
        "role": role,
        "status": "failed",
        "validation_score": np.nan,
        "selected": False,
        "reason_code": "non_finite_prediction" if non_finite else "fit_failed",
        "failure_type": type(exc).__name__,
        "detail": str(exc),
    }


def _zero_predictions(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        0.0,
        index=index,
        columns=(
            "trend_probability",
            "reversal_probability",
            "stress_probability",
            "trend_expert_raw",
            "reversal_expert_raw",
            "stress_expert_raw",
            "expected_score",
            "tail_score",
            "target_position",
        ),
    )


def _sparse_result(
    *,
    train: _SegmentData,
    valid: _SegmentData,
    test: _SegmentData,
    thresholds: Mapping[str, float],
    occupancy: Mapping[str, int],
    spec: RegimeMoESpec,
) -> RegimeMoEResult:
    rows = []
    for candidate, role in (
        ("trend_ridge_expert", "trend_expert"),
        ("reversal_catboost_expert", "reversal_expert"),
        ("stress_downside_expert", "stress_expert"),
        ("multinomial_soft_gate", "regime_gate"),
    ):
        rows.append(
            {
                **_failure_row(candidate, role, RuntimeError("blocked_sparse_regime")),
                "reason_code": "blocked_sparse_regime",
                "failure_type": "SparseRegime",
            }
        )
    comparison = pd.DataFrame(rows)
    predictions = {
        spec.train_segment: _zero_predictions(train.features.index),
        spec.selection_segment: _zero_predictions(valid.features.index),
        spec.test_segment: _zero_predictions(test.features.index),
    }
    priors = np.asarray([occupancy[name] for name in REGIMES], dtype="float64")
    priors = priors / priors.sum()
    gate = {
        name: pd.DataFrame(
            np.tile(priors, (len(frame), 1)), index=frame.index, columns=list(REGIMES)
        )
        for name, frame in predictions.items()
    }
    experts = {
        name: pd.DataFrame(0.0, index=frame.index, columns=list(REGIMES))
        for name, frame in predictions.items()
    }
    return RegimeMoEResult(
        model_comparison=comparison,
        selected_candidate="blocked_sparse_regime",
        regime_thresholds=dict(thresholds),
        regime_occupancy=dict(occupancy),
        gate_probabilities_by_segment=gate,
        expert_predictions_by_segment=experts,
        predictions_by_segment=predictions,
        positions_by_segment={
            name: frame["target_position"].copy() for name, frame in predictions.items()
        },
        failure_rows=comparison.copy(),
        selection_evidence={
            "route_status": "blocked_sparse_regime",
            "blocked_reason": "blocked_sparse_regime",
            "threshold_fit_segment": spec.train_segment,
            "selection_segment": spec.selection_segment,
            "test_labels_used_for_selection": False,
            "test_accessed_after_freeze": True,
            "min_regime_observations": spec.min_regime_observations,
        },
        floor=float(spec.floor_cap_candidates[0][0]),
        cap=float(spec.floor_cap_candidates[0][1]),
        fitted_policy=None,
    )


def run_regime_moe_models(
    dataset,
    *,
    spec: RegimeMoESpec | None = None,
    data_quality_by_segment: Mapping[
        str, bool | pd.Series | Sequence[bool]
    ] | None = None,
) -> RegimeMoEResult:
    """Fit specialist experts and freeze a validation-selected soft mixture."""

    spec = spec or RegimeMoESpec()
    external_quality = dict(data_quality_by_segment or {})
    train_raw = _prepare_learning_segment(dataset, spec.train_segment, spec)
    valid_raw = _prepare_learning_segment(dataset, spec.selection_segment, spec)
    required = set(
        (
            spec.regime_return_feature,
            spec.regime_trend_feature,
            spec.regime_volatility_feature,
            *spec.trend_features,
            *spec.reversal_features,
            *spec.stress_features,
            *spec.gate_features,
        )
    )
    _require_features(train_raw.features, sorted(required), role="regime mixture")
    _require_features(valid_raw.features, sorted(required), role="regime mixture")

    thresholds = _regime_thresholds(train_raw.features, spec)
    train_regime = _assign_regimes(train_raw.features, thresholds, spec)
    occupancy = {name: int(train_regime.eq(name).sum()) for name in REGIMES}
    if min(occupancy.values()) < spec.min_regime_observations:
        test_raw = _prepare_inference_segment(dataset, spec.test_segment)
        return _sparse_result(
            train=train_raw,
            valid=valid_raw,
            test=test_raw,
            thresholds=thresholds,
            occupancy=occupancy,
            spec=spec,
        )

    median, mean, scale = _fit_preprocessor(train_raw.features)
    train_x = _transform(train_raw.features, median, mean, scale)
    valid_x = _transform(valid_raw.features, median, mean, scale)
    train_return = train_raw.labels[spec.return_label]
    train_downside = train_raw.labels[spec.downside_label]
    valid_return = valid_raw.labels[spec.return_label]
    rows: list[dict[str, Any]] = []
    degraded = False

    expert_definitions = (
        (
            "trend",
            "trend_ridge_expert",
            "trend_train_mean_fallback",
            spec.trend_features,
            train_return,
            _fit_trend_expert,
        ),
        (
            "reversal",
            "reversal_catboost_expert",
            "reversal_train_mean_fallback",
            spec.reversal_features,
            train_return,
            _fit_reversal_expert,
        ),
        (
            "stress",
            "stress_downside_expert",
            "stress_train_mean_fallback",
            spec.stress_features,
            train_downside,
            _fit_stress_expert,
        ),
    )
    expert_models: dict[str, Any] = {}
    expert_names: dict[str, str] = {}
    for regime, candidate, fallback_name, features, target, fit_function in expert_definitions:
        mask = train_regime.eq(regime)
        try:
            candidate_model = fit_function(
                train_x.loc[mask, list(features)], target.loc[mask], spec
            )
            _validate_values(
                candidate_model.predict(train_x.loc[:, list(features)]),
                train_x.index,
                segment=spec.train_segment,
            )
            _validate_values(
                candidate_model.predict(valid_x.loc[:, list(features)]),
                valid_x.index,
                segment=spec.selection_segment,
            )
            expert_models[regime] = candidate_model
            expert_names[regime] = candidate
            rows.append(_success_row(candidate, f"{regime}_expert"))
        except Exception as exc:
            degraded = True
            rows.append(_failure_row(candidate, f"{regime}_expert", exc))
            finite = target.loc[mask].replace([np.inf, -np.inf], np.nan).dropna()
            fallback_value = float(finite.mean()) if not finite.empty else 0.0
            expert_models[regime] = _ConstantRegressor(fallback_value)
            expert_names[regime] = fallback_name
            rows.append(_success_row(fallback_name, f"{regime}_expert_fallback"))

    try:
        gate_model = _fit_gate(train_x.loc[:, list(spec.gate_features)], train_regime, spec)
        _gate_probabilities(
            gate_model,
            train_x.loc[:, list(spec.gate_features)],
            segment=spec.train_segment,
        )
        _gate_probabilities(
            gate_model,
            valid_x.loc[:, list(spec.gate_features)],
            segment=spec.selection_segment,
        )
        gate_name = "multinomial_soft_gate"
        rows.append(_success_row(gate_name, "regime_gate"))
    except Exception as exc:
        degraded = True
        rows.append(_failure_row("multinomial_soft_gate", "regime_gate", exc))
        gate_name = "train_regime_prior_gate"
        gate_model = _ConstantGate([occupancy[name] for name in REGIMES])
        rows.append(_success_row(gate_name, "regime_gate_fallback"))

    train_expert_raw = {
        "trend": _validate_values(
            expert_models["trend"].predict(train_x.loc[:, list(spec.trend_features)]),
            train_x.index,
            segment=spec.train_segment,
        ),
        "reversal": _validate_values(
            expert_models["reversal"].predict(train_x.loc[:, list(spec.reversal_features)]),
            train_x.index,
            segment=spec.train_segment,
        ),
        "stress": -np.maximum(
            _validate_values(
                expert_models["stress"].predict(train_x.loc[:, list(spec.stress_features)]),
                train_x.index,
                segment=spec.train_segment,
            ),
            0.0,
        ),
    }
    centers = {name: float(np.mean(values)) for name, values in train_expert_raw.items()}
    scales = {}
    for name, values in train_expert_raw.items():
        value = float(np.std(values))
        scales[name] = value if np.isfinite(value) and value > np.finfo("float64").eps else 1.0
    train_stress_downside = np.maximum(-train_expert_raw["stress"], 0.0)
    downside_scale = float(np.quantile(train_stress_downside, 0.95))
    if not np.isfinite(downside_scale) or downside_scale <= np.finfo("float64").eps:
        downside_scale = 1.0

    train_quality, train_quality_evidence = _quality_with_evidence(
        train_raw, external_quality.get(spec.train_segment)
    )
    valid_quality, valid_quality_evidence = _quality_with_evidence(
        valid_raw, external_quality.get(spec.selection_segment)
    )
    policies: list[tuple[float, float, float, FittedRegimeMoEPolicy, pd.DataFrame, pd.DataFrame]] = []
    for floor, cap in spec.floor_cap_candidates:
        policy = FittedRegimeMoEPolicy(
            feature_columns=tuple(str(column) for column in median.index),
            preprocessing_median=median.copy(),
            preprocessing_mean=mean.copy(),
            preprocessing_scale=scale.copy(),
            gate_model=gate_model,
            gate_features=spec.gate_features,
            trend_model=expert_models["trend"],
            trend_features=spec.trend_features,
            reversal_model=expert_models["reversal"],
            reversal_features=spec.reversal_features,
            stress_model=expert_models["stress"],
            stress_features=spec.stress_features,
            expert_score_center=dict(centers),
            expert_score_scale=dict(scales),
            downside_scale=downside_scale,
            floor=float(floor),
            cap=float(cap),
        )
        train_prediction = _predict_policy_segment(
            policy, train_raw.features, train_quality, segment=spec.train_segment
        )
        valid_prediction = _predict_policy_segment(
            policy, valid_raw.features, valid_quality, segment=spec.selection_segment
        )
        utility = _budget_utility(valid_return, valid_prediction["target_position"])
        policies.append((utility, float(floor), float(cap), policy, train_prediction, valid_prediction))

    selected = max(policies, key=lambda item: (item[0], -item[1], -item[2]))
    selected_utility, floor, cap, fitted_policy, train_prediction, valid_prediction = selected
    rows.append(
        {
            **_success_row("validation_selected_regime_moe", "allocation_policy"),
            "validation_score": selected_utility,
            "selected": True,
        }
    )

    # First test access occurs only after thresholds, models and position policy are frozen.
    test_raw = _prepare_inference_segment(dataset, spec.test_segment)
    missing_test_features = sorted(required.difference(test_raw.features.columns))
    if missing_test_features:
        test_raw.quality[:] = False
    test_quality, test_quality_evidence = _quality_with_evidence(
        test_raw, external_quality.get(spec.test_segment)
    )
    if missing_test_features:
        test_quality_evidence["reason_code"] = "missing_or_stale_required_source"
        test_quality_evidence["missing_features"] = tuple(missing_test_features)
    test_failure = None
    try:
        test_prediction = _predict_policy_segment(
            fitted_policy, test_raw.features, test_quality, segment=spec.test_segment
        )
    except Exception as exc:
        reason = "non_finite_prediction" if isinstance(exc, _NonFinitePredictionError) else "prediction_failed"
        test_failure = {
            "reason_code": reason,
            "failure_type": type(exc).__name__,
            "segment": spec.test_segment,
            "first_offending_date": (
                exc.first_offending_date
                if isinstance(exc, _NonFinitePredictionError)
                else _prediction_date(test_raw.features.index, 0)
            ),
        }
        row = _failure_row("frozen_test_prediction", "allocation_policy", exc)
        row["reason_code"] = reason
        rows.append(row)
        test_prediction = _zero_predictions(test_raw.features.index)

    predictions = {
        spec.train_segment: train_prediction,
        spec.selection_segment: valid_prediction,
        spec.test_segment: test_prediction,
    }
    for name, prediction in predictions.items():
        if not np.isfinite(prediction.to_numpy(dtype="float64")).all():
            raise RuntimeError(f"non-finite frozen prediction in {name}")
    comparison = pd.DataFrame(rows).sort_values(["role", "candidate"], kind="stable").reset_index(drop=True)
    failures = comparison.loc[comparison["status"] != "ok"].reset_index(drop=True)
    quality_evidence = {
        spec.train_segment: train_quality_evidence,
        spec.selection_segment: valid_quality_evidence,
        spec.test_segment: test_quality_evidence,
    }
    quality_blocked = any(value["invalid_count"] > 0 for value in quality_evidence.values())
    if test_failure:
        route_status = (
            "blocked_non_finite_prediction"
            if test_failure["reason_code"] == "non_finite_prediction"
            else "blocked_prediction_failed"
        )
    elif quality_blocked:
        route_status = "blocked_data_quality"
    elif degraded:
        route_status = "ok_degraded_candidate_failure"
    else:
        route_status = "ok"
    gate_by_segment = {
        name: frame.loc[
            :, ["trend_probability", "reversal_probability", "stress_probability"]
        ].rename(
            columns={
                "trend_probability": "trend",
                "reversal_probability": "reversal",
                "stress_probability": "stress",
            }
        )
        for name, frame in predictions.items()
    }
    expert_by_segment = {
        name: frame.loc[
            :, ["trend_expert_raw", "reversal_expert_raw", "stress_expert_raw"]
        ].rename(
            columns={
                "trend_expert_raw": "trend",
                "reversal_expert_raw": "reversal",
                "stress_expert_raw": "stress",
            }
        )
        for name, frame in predictions.items()
    }
    frozen = {
        "thresholds": dict(thresholds),
        "expert_candidates": dict(expert_names),
        "gate_candidate": gate_name,
        "floor": floor,
        "cap": cap,
        "expert_score_center": dict(centers),
        "expert_score_scale": dict(scales),
        "downside_scale": downside_scale,
    }
    return RegimeMoEResult(
        model_comparison=comparison,
        selected_candidate="validation_selected_regime_moe",
        regime_thresholds=dict(thresholds),
        regime_occupancy=dict(occupancy),
        gate_probabilities_by_segment=gate_by_segment,
        expert_predictions_by_segment=expert_by_segment,
        predictions_by_segment=predictions,
        positions_by_segment={
            name: frame["target_position"].copy() for name, frame in predictions.items()
        },
        failure_rows=failures,
        selection_evidence={
            "route_status": route_status,
            "threshold_fit_segment": spec.train_segment,
            "selection_segment": spec.selection_segment,
            "test_labels_used_for_selection": False,
            "test_accessed_after_freeze": True,
            "data_keys": {
                spec.train_segment: "DataHandlerLP.DK_L",
                spec.selection_segment: "DataHandlerLP.DK_L",
                spec.test_segment: "DataHandlerLP.DK_I",
            },
            "regime_occupancy": dict(occupancy),
            "min_regime_observations": spec.min_regime_observations,
            "policy_grid_evaluations": tuple(
                {
                    "floor": item[1],
                    "cap": item[2],
                    "validation_utility": item[0],
                }
                for item in policies
            ),
            "selected_validation_utility": selected_utility,
            "frozen_policy": frozen,
            "test_prediction_failure": test_failure,
            "data_quality_failures_by_segment": quality_evidence,
        },
        floor=floor,
        cap=cap,
        fitted_policy=fitted_policy,
    )


__all__ = [
    "FittedRegimeMoEPolicy",
    "RegimeMoEResult",
    "RegimeMoESpec",
    "run_regime_moe_models",
]
