from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_pinball_loss, mean_squared_error
from sklearn.model_selection import train_test_split

from .risk_adjusted_common import soft_budget_position


@dataclass(frozen=True)
class TailBudgetSpec:
    """Small deterministic search contract for the tail-first model route."""

    train_segment: str = "train"
    selection_segment: str = "valid"
    test_segment: str = "test"
    return_label: str = "future_return_5d"
    downside_label: str = "future_downside_5d"
    crash_label: str = "future_crash_5d"
    ridge_alphas: tuple[float, ...] = (1.0,)
    lightgbm_rounds: int = 24
    catboost_iterations: int = 32
    densemble_epochs: int = 16
    downside_quantile: float = 0.90
    blend_weights: tuple[float, ...] = (0.25, 0.50, 0.75)
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
        if not self.ridge_alphas or any(float(alpha) <= 0.0 for alpha in self.ridge_alphas):
            raise ValueError("ridge_alphas must contain positive values")
        if self.lightgbm_rounds <= 0 or self.catboost_iterations <= 0 or self.densemble_epochs <= 0:
            raise ValueError("model iteration counts must be positive")
        if not 0.0 < self.downside_quantile < 1.0:
            raise ValueError("downside_quantile must be between zero and one")
        if any(not 0.0 < float(weight) < 1.0 for weight in self.blend_weights):
            raise ValueError("blend_weights must be strictly between zero and one")
        if not self.floor_cap_candidates:
            raise ValueError("floor_cap_candidates must not be empty")
        for floor, cap in self.floor_cap_candidates:
            if not 0.0 <= float(floor) < float(cap) <= 1.0:
                raise ValueError("floor/cap candidates must satisfy 0 <= floor < cap <= 1")


@dataclass(frozen=True)
class TailBudgetResult:
    model_comparison: pd.DataFrame
    selected_candidate: str
    predictions_by_segment: Mapping[str, pd.DataFrame]
    positions_by_segment: Mapping[str, pd.Series]
    failure_rows: pd.DataFrame
    selection_evidence: Mapping[str, Any]
    floor: float
    cap: float


@dataclass
class _SegmentData:
    features: pd.DataFrame
    labels: pd.DataFrame
    quality: pd.Series


@dataclass
class _Candidate:
    name: str
    role: str
    model: Any
    train_prediction: pd.Series
    valid_prediction: pd.Series
    component_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _PolicyChoice:
    name: str
    expected: _Candidate
    downside: _Candidate
    crash: _Candidate
    floor: float
    cap: float
    validation_utility: float
    mu_center: float
    mu_scale: float
    mu_direction: float
    tail_scale: float
    train_mu_score: pd.Series
    valid_mu_score: pd.Series
    train_tail_score: pd.Series
    valid_tail_score: pd.Series
    train_position: pd.Series
    valid_position: pd.Series
    weights: dict[str, float] = field(default_factory=dict)


class _BlendModel:
    def __init__(self, left: Any, right: Any, weight: float) -> None:
        self.left = left
        self.right = right
        self.weight = float(weight)

    def predict_segment(self, features: pd.DataFrame, segment: str) -> np.ndarray:
        left = _predict_model(self.left, features, segment=segment)
        right = _predict_model(self.right, features, segment=segment)
        return self.weight * left + (1.0 - self.weight) * right


class _DEnsemblePredictor:
    """Use qlib's public DEnsembleModel.predict(dataset, segment) API."""

    def __init__(self, model: Any, dataset_view: "_DEnsembleDatasetView") -> None:
        self.model = model
        self.dataset_view = dataset_view

    def predict_segment(self, features: pd.DataFrame, segment: str) -> np.ndarray:
        placeholder = pd.Series(0.0, index=features.index, name="target")
        self.dataset_view.frames[segment] = (features, placeholder)
        return np.asarray(self.model.predict(self.dataset_view, segment=segment), dtype="float64")


class _NonFinitePredictionError(ValueError):
    def __init__(self, *, segment: str, first_offending_date: str) -> None:
        super().__init__(f"non-finite prediction in {segment} at {first_offending_date}")
        self.segment = str(segment)
        self.first_offending_date = str(first_offending_date)


class _CalibratedCrashModel:
    def __init__(self, base_model: Any, calibrator: LogisticRegression) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.base_model.predict_proba(features), dtype="float64")[:, 1]
        logits = _logit(raw).reshape(-1, 1)
        return np.asarray(self.calibrator.predict_proba(logits), dtype="float64")[:, 1]


class _ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.value, dtype="float64")


class _DEnsembleDatasetView:
    """Present one target and preprocessed frames through qlib's Dataset API."""

    def __init__(self, frames: Mapping[str, tuple[pd.DataFrame, pd.Series]]) -> None:
        self.frames = dict(frames)

    def prepare(self, segments, col_set="__all", data_key="infer", **kwargs):
        if isinstance(segments, (list, tuple)):
            return [self.prepare(segment, col_set=col_set, data_key=data_key, **kwargs) for segment in segments]
        features, label = self.frames[str(segments)]
        if col_set == "feature":
            return features.copy()
        label_frame = label.rename("target").to_frame()
        if col_set == "label":
            return label_frame
        if col_set == "__all__" or (
            isinstance(col_set, Sequence) and {"feature", "label"}.issubset(set(col_set))
        ):
            feature_frame = features.copy()
            feature_frame.columns = pd.MultiIndex.from_product([["feature"], feature_frame.columns])
            label_frame.columns = pd.MultiIndex.from_product([["label"], label_frame.columns])
            return pd.concat([feature_frame, label_frame], axis=1)
        raise KeyError(f"unsupported col_set: {col_set!r}")


def _prepare_segment(dataset, segment: str, spec: TailBudgetSpec) -> _SegmentData:
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
        features = frame["feature"].copy()
        if "label" not in frame.columns.get_level_values(0):
            raise ValueError(f"segment {segment!r} does not include qlib labels")
        labels = frame["label"].copy()
    else:
        missing = [label for label in labels_needed if label not in frame.columns]
        if missing:
            raise ValueError(f"segment {segment!r} is missing labels: {missing}")
        labels = frame.loc[:, list(labels_needed)].copy()
        features = frame.drop(columns=list(labels_needed)).copy()
    missing = [label for label in labels_needed if label not in labels.columns]
    if missing:
        raise ValueError(f"segment {segment!r} is missing labels: {missing}")
    if features.empty:
        raise ValueError(f"segment {segment!r} does not include features")
    features = features.apply(pd.to_numeric, errors="coerce").astype("float64")
    labels = labels.loc[:, list(labels_needed)].apply(pd.to_numeric, errors="coerce").astype("float64")
    quality = pd.Series(
        np.isfinite(features.to_numpy(dtype="float64")).all(axis=1),
        index=features.index,
        name="data_quality_valid",
    )
    return _SegmentData(features=features, labels=labels, quality=quality)


def _fit_preprocessor(train: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    median = train.replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    filled = train.replace([np.inf, -np.inf], np.nan).fillna(median)
    mean = filled.mean()
    scale = filled.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return median, mean, scale


def _transform(frame: pd.DataFrame, median: pd.Series, mean: pd.Series, scale: pd.Series) -> pd.DataFrame:
    aligned = frame.reindex(columns=median.index).replace([np.inf, -np.inf], np.nan).fillna(median)
    return ((aligned - mean) / scale).fillna(0.0).astype("float64")


def _finite_target(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    mask = np.isfinite(target.to_numpy(dtype="float64"))
    if not bool(mask.any()):
        raise ValueError("target does not contain finite observations")
    return features.loc[mask], target.loc[mask].astype("float64")


def _prediction_date(index: pd.Index, position: int) -> str:
    value = index[position]
    if isinstance(value, tuple):
        datetime_level = index.names.index("datetime") if "datetime" in index.names else len(value) - 1
        value = value[datetime_level]
    try:
        return pd.Timestamp(value).isoformat()
    except (TypeError, ValueError):
        return str(value)


def _validate_prediction_values(values: np.ndarray, index: pd.Index, *, segment: str) -> np.ndarray:
    values = np.asarray(values, dtype="float64").reshape(-1)
    if len(values) != len(index):
        raise ValueError(f"candidate produced {len(values)} predictions for {len(index)} rows")
    finite = np.isfinite(values)
    if not finite.all():
        position = int(np.flatnonzero(~finite)[0])
        raise _NonFinitePredictionError(
            segment=segment,
            first_offending_date=_prediction_date(index, position),
        )
    return values


def _predict_model(model: Any, features: pd.DataFrame, *, segment: str = "unknown") -> np.ndarray:
    if hasattr(model, "predict_segment"):
        values = model.predict_segment(features, segment)
    else:
        values = np.asarray(model.predict(features), dtype="float64").reshape(-1)
    return _validate_prediction_values(values, features.index, segment=segment)


def _series(values: np.ndarray, index: pd.Index, name: str) -> pd.Series:
    return pd.Series(np.asarray(values, dtype="float64"), index=index, name=name)


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1.0 - 1e-6)
    return np.log(probability / (1.0 - probability))


def _fit_crash_model(
    features: pd.DataFrame,
    target: pd.Series,
    spec: TailBudgetSpec,
) -> _CalibratedCrashModel:
    from catboost import CatBoostClassifier

    x, y = _finite_target(features, target)
    y = y.astype("int64")
    counts = y.value_counts()
    if len(counts) != 2 or int(counts.min()) < 2:
        raise ValueError("crash calibration requires both classes with at least two observations")
    x_fit, x_calibrate, y_fit, y_calibrate = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=spec.random_state,
        stratify=y,
    )
    base = CatBoostClassifier(
        iterations=spec.catboost_iterations,
        depth=3,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=spec.random_state,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )
    base.fit(x_fit, y_fit)
    raw = np.asarray(base.predict_proba(x_calibrate), dtype="float64")[:, 1]
    calibrator = LogisticRegression(random_state=spec.random_state, solver="lbfgs")
    calibrator.fit(_logit(raw).reshape(-1, 1), y_calibrate)
    return _CalibratedCrashModel(base, calibrator)


def _quality_for_segment(
    segment: _SegmentData,
    external: bool | pd.Series | Sequence[bool] | None,
) -> pd.Series:
    quality = segment.quality.copy()
    if external is None:
        return quality
    if isinstance(external, pd.Series):
        values = external.reindex(quality.index).fillna(False).astype(bool)
    elif np.isscalar(external):
        values = pd.Series(bool(external), index=quality.index)
    else:
        array = np.asarray(external)
        if len(array) != len(quality):
            raise ValueError("data quality input must align with its segment")
        values = pd.Series(array.astype(bool), index=quality.index)
    return quality & values


def _quality_with_evidence(
    segment: _SegmentData,
    external: bool | pd.Series | Sequence[bool] | None,
) -> tuple[pd.Series, dict[str, Any]]:
    quality = _quality_for_segment(segment, external)
    invalid = ~quality
    invalid_count = int(invalid.sum())
    external_invalid = False
    if external is not None:
        if isinstance(external, pd.Series):
            external_invalid = bool((~external.reindex(quality.index).fillna(False).astype(bool)).any())
        elif np.isscalar(external):
            external_invalid = not bool(external)
        else:
            external_invalid = bool((~np.asarray(external).astype(bool)).any())
    first_offending_date = None
    if invalid_count:
        first_offending_date = _prediction_date(quality.index, int(np.flatnonzero(invalid.to_numpy())[0]))
    evidence = {
        "invalid_count": invalid_count,
        "first_offending_date": first_offending_date,
        "reason_code": (
            "missing_or_stale_required_source"
            if external_invalid
            else "non_finite_required_feature" if invalid_count else ""
        ),
    }
    return quality, evidence


def _budget_utility(realized_return: pd.Series, position: pd.Series) -> float:
    aligned = pd.concat([realized_return.rename("return"), position.rename("position")], axis=1).dropna()
    if aligned.empty:
        return float("-inf")
    strategy_return = aligned["return"].to_numpy(dtype="float64") * aligned["position"].to_numpy(dtype="float64")
    if not np.isfinite(strategy_return).all():
        return float("-inf")
    downside = np.minimum(strategy_return, 0.0)
    downside_scale = float(np.sqrt(np.mean(np.square(downside))))
    return float(np.mean(strategy_return) / max(downside_scale, 1e-6))


def _failure_row(name: str, role: str, exc: Exception) -> dict[str, Any]:
    non_finite = isinstance(exc, _NonFinitePredictionError)
    return {
        "candidate": name,
        "model": name,
        "role": role,
        "status": "failed",
        "validation_score": np.nan,
        "selected": False,
        "reason_code": "non_finite_prediction" if non_finite else "fit_failed",
        "failure_type": type(exc).__name__,
        "detail": type(exc).__name__,
        "segment": exc.segment if non_finite else "",
        "first_offending_date": exc.first_offending_date if non_finite else "",
        "blend_weight": np.nan,
        "floor": np.nan,
        "cap": np.nan,
    }


def _success_row(candidate: _Candidate) -> dict[str, Any]:
    return {
        "candidate": candidate.name,
        "model": candidate.name,
        "role": candidate.role,
        "status": "ok",
        "validation_score": float(candidate.component_score),
        "selected": False,
        "reason_code": "",
        "failure_type": "",
        "detail": "",
        "segment": "",
        "first_offending_date": "",
        "blend_weight": float(candidate.metadata.get("blend_weight", np.nan)),
        "floor": np.nan,
        "cap": np.nan,
    }


def _unavailable_row(name: str, role: str) -> dict[str, Any]:
    return {
        "candidate": name,
        "model": name,
        "role": role,
        "status": "unavailable",
        "validation_score": np.nan,
        "selected": False,
        "reason_code": "base_candidate_unavailable",
        "failure_type": "DependencyUnavailable",
        "detail": "DependencyUnavailable",
        "segment": "",
        "first_offending_date": "",
        "blend_weight": np.nan,
        "floor": np.nan,
        "cap": np.nan,
    }


def _blend_candidate(
    name: str,
    role: str,
    left: _Candidate,
    right: _Candidate,
    weight: float,
) -> _Candidate:
    weight = float(weight)
    return _Candidate(
        name=name,
        role=role,
        model=_BlendModel(left.model, right.model, weight),
        train_prediction=weight * left.train_prediction + (1.0 - weight) * right.train_prediction,
        valid_prediction=weight * left.valid_prediction + (1.0 - weight) * right.valid_prediction,
        component_score=float("nan"),
        metadata={"blend_weight": weight},
    )


def _policy_choice(
    *,
    name: str,
    expected: _Candidate,
    downside: _Candidate,
    crash: _Candidate,
    floor: float,
    cap: float,
    train_return: pd.Series,
    valid_return: pd.Series,
    train_quality: pd.Series,
    valid_quality: pd.Series,
    weights: Mapping[str, float] | None = None,
) -> _PolicyChoice:
    mu_center = float(expected.train_prediction.mean())
    mu_scale = float(expected.train_prediction.std(ddof=0))
    if not np.isfinite(mu_scale) or mu_scale <= np.finfo("float64").eps:
        mu_scale = 1.0
    mu_direction = 1.0
    direction = pd.concat([expected.train_prediction, train_return], axis=1).dropna()
    if len(direction) >= 2:
        correlation = float(direction.corr().iloc[0, 1])
        if np.isfinite(correlation) and correlation < 0.0:
            mu_direction = -1.0
    tail_scale = float(downside.train_prediction.clip(lower=0.0).quantile(0.95))
    if not np.isfinite(tail_scale) or tail_scale <= np.finfo("float64").eps:
        tail_scale = 1.0
    train_mu_score = (expected.train_prediction - mu_center) / mu_scale * mu_direction
    valid_mu_score = (expected.valid_prediction - mu_center) / mu_scale * mu_direction
    train_tail_score = (downside.train_prediction.clip(lower=0.0) / tail_scale).clip(0.0, 1.0)
    valid_tail_score = (downside.valid_prediction.clip(lower=0.0) / tail_scale).clip(0.0, 1.0)
    train_position = soft_budget_position(
        train_mu_score,
        train_tail_score,
        crash.train_prediction.clip(0.0, 1.0),
        data_quality_valid=train_quality,
        floor=floor,
        cap=cap,
    )
    valid_position = soft_budget_position(
        valid_mu_score,
        valid_tail_score,
        crash.valid_prediction.clip(0.0, 1.0),
        data_quality_valid=valid_quality,
        floor=floor,
        cap=cap,
    )
    return _PolicyChoice(
        name=name,
        expected=expected,
        downside=downside,
        crash=crash,
        floor=float(floor),
        cap=float(cap),
        validation_utility=_budget_utility(valid_return, valid_position),
        mu_center=mu_center,
        mu_scale=mu_scale,
        mu_direction=mu_direction,
        tail_scale=tail_scale,
        train_mu_score=train_mu_score,
        valid_mu_score=valid_mu_score,
        train_tail_score=train_tail_score,
        valid_tail_score=valid_tail_score,
        train_position=train_position,
        valid_position=valid_position,
        weights=dict(weights or {}),
    )


def _choice_key(choice: _PolicyChoice) -> tuple[float, float, float, float, float]:
    return (
        float(choice.validation_utility),
        -float(choice.weights.get("expected", 0.0)),
        -float(choice.weights.get("downside", 0.0)),
        -float(choice.floor),
        -float(choice.cap),
    )


def _policy_evidence(choice: _PolicyChoice) -> dict[str, Any]:
    return {
        "route_candidate": choice.name,
        "expected_candidate": choice.expected.name,
        "downside_candidate": choice.downside.name,
        "crash_candidate": choice.crash.name,
        "expected_weight": choice.weights.get("expected"),
        "downside_weight": choice.weights.get("downside"),
        "floor": choice.floor,
        "cap": choice.cap,
        "validation_utility": float(choice.validation_utility),
    }


def run_tail_budget_models(
    dataset,
    spec: TailBudgetSpec = TailBudgetSpec(),
    *,
    data_quality_by_segment: Mapping[str, bool | pd.Series | Sequence[bool]] | None = None,
) -> TailBudgetResult:
    """Fit and validation-select tail-first models through a qlib Dataset surface.

    Test data is prepared only after every model, direction, blend weight, and
    floor/cap choice has been frozen from train and validation evidence.
    """

    data_quality_by_segment = data_quality_by_segment or {}
    train_raw = _prepare_segment(dataset, spec.train_segment, spec)
    valid_raw = _prepare_segment(dataset, spec.selection_segment, spec)
    median, mean, scale = _fit_preprocessor(train_raw.features)
    train_x = _transform(train_raw.features, median, mean, scale)
    valid_x = _transform(valid_raw.features, median, mean, scale)
    train_return = train_raw.labels[spec.return_label]
    valid_return = valid_raw.labels[spec.return_label]
    train_downside = train_raw.labels[spec.downside_label]
    valid_downside = valid_raw.labels[spec.downside_label]
    train_crash = train_raw.labels[spec.crash_label]
    valid_crash = valid_raw.labels[spec.crash_label]

    rows: list[dict[str, Any]] = []
    expected: list[_Candidate] = []
    downside: list[_Candidate] = []

    for alpha in spec.ridge_alphas:
        name = "ridge_expected" if len(spec.ridge_alphas) == 1 else f"ridge_expected_alpha_{alpha:g}"
        try:
            fit_x, fit_y = _finite_target(train_x, train_return)
            model = Ridge(alpha=float(alpha)).fit(fit_x, fit_y)
            train_pred = _series(_predict_model(model, train_x, segment=spec.train_segment), train_x.index, name)
            valid_pred = _series(
                _predict_model(model, valid_x, segment=spec.selection_segment), valid_x.index, name
            )
            score = -float(mean_squared_error(valid_return, valid_pred))
            candidate = _Candidate(name, "expected_return", model, train_pred, valid_pred, score)
            expected.append(candidate)
            rows.append(_success_row(candidate))
        except Exception as exc:  # optional candidate isolation is intentional
            rows.append(_failure_row(name, "expected_return", exc))

    try:
        from lightgbm import LGBMRegressor

        fit_x, fit_y = _finite_target(train_x, train_return)
        model = LGBMRegressor(
            n_estimators=spec.lightgbm_rounds,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            min_child_samples=8,
            random_state=spec.random_state,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        ).fit(fit_x, fit_y)
        train_pred = _series(
            _predict_model(model, train_x, segment=spec.train_segment), train_x.index, "lightgbm_expected"
        )
        valid_pred = _series(
            _predict_model(model, valid_x, segment=spec.selection_segment),
            valid_x.index,
            "lightgbm_expected",
        )
        score = -float(mean_squared_error(valid_return, valid_pred))
        candidate = _Candidate("lightgbm_expected", "expected_return", model, train_pred, valid_pred, score)
        expected.append(candidate)
        rows.append(_success_row(candidate))
    except Exception as exc:
        rows.append(_failure_row("lightgbm_expected", "expected_return", exc))

    try:
        from catboost import CatBoostRegressor

        fit_x, fit_y = _finite_target(train_x, train_downside)
        model = CatBoostRegressor(
            iterations=spec.catboost_iterations,
            depth=3,
            learning_rate=0.05,
            loss_function=f"Quantile:alpha={spec.downside_quantile}",
            random_seed=spec.random_state,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(fit_x, fit_y)
        train_pred = _series(
            _predict_model(model, train_x, segment=spec.train_segment), train_x.index, "catboost_downside"
        )
        valid_pred = _series(
            _predict_model(model, valid_x, segment=spec.selection_segment),
            valid_x.index,
            "catboost_downside",
        )
        score = -float(mean_pinball_loss(valid_downside, valid_pred, alpha=spec.downside_quantile))
        candidate = _Candidate("catboost_downside", "downside_budget", model, train_pred, valid_pred, score)
        downside.append(candidate)
        rows.append(_success_row(candidate))
    except Exception as exc:
        rows.append(_failure_row("catboost_downside", "downside_budget", exc))

    try:
        from qlib.contrib.model.double_ensemble import DEnsembleModel

        view = _DEnsembleDatasetView(
            {
                spec.train_segment: (train_x, train_downside),
                spec.selection_segment: (valid_x, valid_downside),
            }
        )
        qlib_model = DEnsembleModel(
            num_models=1,
            enable_sr=False,
            enable_fs=False,
            epochs=spec.densemble_epochs,
            early_stopping_rounds=None,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            min_data_in_leaf=8,
            verbosity=-1,
            num_threads=1,
            seed=spec.random_state,
            deterministic=True,
            force_col_wise=True,
        )
        qlib_model.fit(view)
        model = _DEnsemblePredictor(qlib_model, view)
        train_pred = _series(
            _predict_model(model, train_x, segment=spec.train_segment),
            train_x.index,
            "qlib_densemble_downside",
        )
        valid_pred = _series(
            _predict_model(model, valid_x, segment=spec.selection_segment),
            valid_x.index,
            "qlib_densemble_downside",
        )
        score = -float(mean_pinball_loss(valid_downside, valid_pred, alpha=spec.downside_quantile))
        candidate = _Candidate("qlib_densemble_downside", "downside_budget", model, train_pred, valid_pred, score)
        downside.append(candidate)
        rows.append(_success_row(candidate))
    except Exception as exc:
        rows.append(_failure_row("qlib_densemble_downside", "downside_budget", exc))

    crash_prior = float(np.clip(train_crash.dropna().mean(), 0.0, 1.0))
    prior_model = _ConstantModel(crash_prior)
    prior_candidate = _Candidate(
        "train_crash_prior",
        "crash_probability_fallback",
        prior_model,
        _series(prior_model.predict(train_x), train_x.index, "train_crash_prior"),
        _series(prior_model.predict(valid_x), valid_x.index, "train_crash_prior"),
        float("nan"),
    )
    crash_candidate: _Candidate | None = None
    try:
        crash_model = _fit_crash_model(train_x, train_crash, spec)
        train_crash_pred = _series(
            _predict_model(crash_model, train_x, segment=spec.train_segment), train_x.index, "catboost_crash"
        )
        valid_crash_pred = _series(
            _predict_model(crash_model, valid_x, segment=spec.selection_segment),
            valid_x.index,
            "catboost_crash",
        )
        score = -float(log_loss(valid_crash.astype("int64"), valid_crash_pred, labels=[0, 1]))
        crash_candidate = _Candidate(
            "catboost_crash",
            "crash_probability",
            crash_model,
            train_crash_pred,
            valid_crash_pred,
            score,
        )
        rows.append(_success_row(crash_candidate))
    except Exception as exc:
        rows.append(_failure_row("catboost_crash", "crash_probability", exc))

    if not expected:
        raise RuntimeError("all expected-return candidates failed")
    if not downside:
        raise RuntimeError("all downside candidates failed")
    train_quality, train_quality_evidence = _quality_with_evidence(
        train_raw, data_quality_by_segment.get(spec.train_segment)
    )
    valid_quality, valid_quality_evidence = _quality_with_evidence(
        valid_raw, data_quality_by_segment.get(spec.selection_segment)
    )
    active_crash = crash_candidate if crash_candidate is not None else prior_candidate
    ridge_candidates = [candidate for candidate in expected if candidate.name.startswith("ridge_expected")]
    lgb_candidates = [candidate for candidate in expected if candidate.name == "lightgbm_expected"]
    catboost_downside = [candidate for candidate in downside if candidate.name == "catboost_downside"]
    densemble_downside = [candidate for candidate in downside if candidate.name == "qlib_densemble_downside"]
    expected_options = list(expected)
    if ridge_candidates and lgb_candidates:
        expected_left = max(ridge_candidates, key=lambda candidate: (candidate.component_score, candidate.name))
        expected_right = lgb_candidates[0]
        for expected_weight in spec.blend_weights:
            expected_options.append(
                _blend_candidate(
                    "validation_blend_expected",
                    "expected_return_blend",
                    expected_left,
                    expected_right,
                    expected_weight,
                )
            )

    downside_options = list(downside)
    if catboost_downside and densemble_downside:
        for downside_weight in spec.blend_weights:
            downside_options.append(
                _blend_candidate(
                    "validation_blend_downside",
                    "downside_budget_blend",
                    catboost_downside[0],
                    densemble_downside[0],
                    downside_weight,
                )
            )

    policy_choices: list[_PolicyChoice] = []
    for expected_candidate in expected_options:
        for downside_candidate in downside_options:
            is_blend = (
                expected_candidate.name == "validation_blend_expected"
                or downside_candidate.name == "validation_blend_downside"
            )
            weights = {}
            if expected_candidate.name == "validation_blend_expected":
                weights["expected"] = float(expected_candidate.metadata["blend_weight"])
            if downside_candidate.name == "validation_blend_downside":
                weights["downside"] = float(downside_candidate.metadata["blend_weight"])
            for candidate_floor, candidate_cap in spec.floor_cap_candidates:
                policy_choices.append(
                    _policy_choice(
                        name=("validation_selected_blend" if is_blend else "validation_selected_individual"),
                        expected=expected_candidate,
                        downside=downside_candidate,
                        crash=active_crash,
                        floor=candidate_floor,
                        cap=candidate_cap,
                        train_return=train_return,
                        valid_return=valid_return,
                        train_quality=train_quality,
                        valid_quality=valid_quality,
                        weights=weights,
                    )
                )

    individual_choices = [choice for choice in policy_choices if choice.name == "validation_selected_individual"]
    blend_choices = [choice for choice in policy_choices if choice.name == "validation_selected_blend"]
    for route_name, role, choices in (
        ("validation_selected_individual", "allocation_policy", individual_choices),
        ("validation_selected_blend", "allocation_policy_blend", blend_choices),
    ):
        if choices:
            route_best = max(choices, key=_choice_key)
            rows.append(
                {
                    "candidate": route_name,
                    "model": route_name,
                    "role": role,
                    "status": "ok",
                    "validation_score": float(route_best.validation_utility),
                    "selected": False,
                    "reason_code": "",
                    "failure_type": "",
                    "detail": "",
                    "blend_weight": np.nan,
                    "floor": route_best.floor,
                    "cap": route_best.cap,
                }
            )
        else:
            rows.append(_unavailable_row(route_name, role))

    expected_blend_choices = [
        choice for choice in policy_choices if choice.expected.name == "validation_blend_expected"
    ]
    downside_blend_choices = [
        choice for choice in policy_choices if choice.downside.name == "validation_blend_downside"
    ]
    for candidate_name, role, choices, component in (
        ("validation_blend_expected", "expected_return_blend", expected_blend_choices, "expected"),
        ("validation_blend_downside", "downside_budget_blend", downside_blend_choices, "downside"),
    ):
        if choices:
            component_best = max(choices, key=_choice_key)
            candidate = getattr(component_best, component)
            row = _success_row(candidate)
            row["validation_score"] = component_best.validation_utility
            row["floor"] = component_best.floor
            row["cap"] = component_best.cap
            rows.append(row)
        else:
            rows.append(_unavailable_row(candidate_name, role))

    blend_evaluations = [_policy_evidence(choice) for choice in blend_choices]
    two_sided_blend_choices = [
        choice for choice in blend_choices if "expected" in choice.weights and "downside" in choice.weights
    ]
    two_sided_blend_evaluations = [_policy_evidence(choice) for choice in two_sided_blend_choices]
    policy_grid_evaluations = tuple(_policy_evidence(choice) for choice in policy_choices)
    utility_best_blend = _policy_evidence(max(blend_choices, key=_choice_key)) if blend_choices else None
    utility_best_two_sided_blend = (
        _policy_evidence(max(two_sided_blend_choices, key=_choice_key))
        if two_sided_blend_choices
        else None
    )
    selected_policy = max(policy_choices, key=_choice_key)
    selected_name = selected_policy.name
    for row in rows:
        component_choices = [
            choice
            for choice in policy_choices
            if row["candidate"]
            in {choice.expected.name, choice.downside.name, choice.crash.name}
        ]
        if component_choices and row["status"] == "ok":
            component_best = max(component_choices, key=_choice_key)
            row["validation_score"] = float(component_best.validation_utility)
        row["selected"] = row["candidate"] == selected_name

    frozen_policy = {
        "expected_candidate": selected_policy.expected.name,
        "downside_candidate": selected_policy.downside.name,
        "crash_candidate": selected_policy.crash.name,
        "expected_blend_weight": selected_policy.weights.get("expected"),
        "downside_blend_weight": selected_policy.weights.get("downside"),
        "mu_center": selected_policy.mu_center,
        "mu_scale": selected_policy.mu_scale,
        "mu_direction": selected_policy.mu_direction,
        "tail_scale": selected_policy.tail_scale,
        "floor": selected_policy.floor,
        "cap": selected_policy.cap,
    }
    train_predictions = pd.DataFrame(
        {
            "mu_raw": selected_policy.expected.train_prediction,
            "mu_score": selected_policy.train_mu_score,
            "tail_raw": selected_policy.downside.train_prediction,
            "tail_score": selected_policy.train_tail_score,
            "crash_probability": selected_policy.crash.train_prediction.clip(0.0, 1.0),
            "target_position": selected_policy.train_position,
        },
        index=train_x.index,
    ).astype("float64")
    valid_predictions = pd.DataFrame(
        {
            "mu_raw": selected_policy.expected.valid_prediction,
            "mu_score": selected_policy.valid_mu_score,
            "tail_raw": selected_policy.downside.valid_prediction,
            "tail_score": selected_policy.valid_tail_score,
            "crash_probability": selected_policy.crash.valid_prediction.clip(0.0, 1.0),
            "target_position": selected_policy.valid_position,
        },
        index=valid_x.index,
    ).astype("float64")

    # This is the first test-segment access: the validation-selected policy is frozen above.
    test_raw = _prepare_segment(dataset, spec.test_segment, spec)
    test_x = _transform(test_raw.features, median, mean, scale)
    test_quality, test_quality_evidence = _quality_with_evidence(
        test_raw, data_quality_by_segment.get(spec.test_segment)
    )
    test_prediction_failure: dict[str, str] | None = None
    try:
        test_mu_values = _validate_prediction_values(
            _predict_model(selected_policy.expected.model, test_x, segment=spec.test_segment),
            test_x.index,
            segment=spec.test_segment,
        )
        test_tail_values = _validate_prediction_values(
            _predict_model(selected_policy.downside.model, test_x, segment=spec.test_segment),
            test_x.index,
            segment=spec.test_segment,
        )
        test_crash_values = _validate_prediction_values(
            _predict_model(selected_policy.crash.model, test_x, segment=spec.test_segment),
            test_x.index,
            segment=spec.test_segment,
        )
        test_mu_raw = _series(test_mu_values, test_x.index, "mu_raw")
        test_tail_raw = _series(test_tail_values, test_x.index, "tail_raw")
        test_crash_probability = _series(
            test_crash_values, test_x.index, "crash_probability"
        ).clip(0.0, 1.0)
        test_mu_score = (
            (test_mu_raw - selected_policy.mu_center)
            / selected_policy.mu_scale
            * selected_policy.mu_direction
        )
        test_tail_score = (test_tail_raw.clip(lower=0.0) / selected_policy.tail_scale).clip(0.0, 1.0)
        test_position = soft_budget_position(
            test_mu_score,
            test_tail_score,
            test_crash_probability,
            data_quality_valid=test_quality,
            floor=selected_policy.floor,
            cap=selected_policy.cap,
        )
        test_predictions = pd.DataFrame(
            {
                "mu_raw": test_mu_raw,
                "mu_score": test_mu_score,
                "tail_raw": test_tail_raw,
                "tail_score": test_tail_score,
                "crash_probability": test_crash_probability,
                "target_position": test_position,
            },
            index=test_x.index,
        ).astype("float64")
    except _NonFinitePredictionError as exc:
        test_prediction_failure = {
            "reason_code": "non_finite_prediction",
            "segment": exc.segment,
            "first_offending_date": exc.first_offending_date,
        }
        rows.append(_failure_row("frozen_test_prediction", "allocation_policy", exc))
        test_predictions = pd.DataFrame(
            0.0,
            index=test_x.index,
            columns=(
                "mu_raw",
                "mu_score",
                "tail_raw",
                "tail_score",
                "crash_probability",
                "target_position",
            ),
        )
    for segment_name, predictions in (
        (spec.train_segment, train_predictions),
        (spec.selection_segment, valid_predictions),
        (spec.test_segment, test_predictions),
    ):
        if not np.isfinite(predictions.to_numpy(dtype="float64")).all():
            raise RuntimeError(f"non-finite frozen prediction in {segment_name}")

    comparison = pd.DataFrame(rows).sort_values(["role", "candidate"], kind="stable").reset_index(drop=True)
    failures = comparison.loc[comparison["status"] != "ok"].reset_index(drop=True)
    data_quality_failures_by_segment = {
        spec.train_segment: train_quality_evidence,
        spec.selection_segment: valid_quality_evidence,
        spec.test_segment: test_quality_evidence,
    }
    data_quality_blocked = any(
        evidence["invalid_count"] > 0 for evidence in data_quality_failures_by_segment.values()
    )
    evidence = MappingProxyType(
        {
            "selection_segment": spec.selection_segment,
            "test_labels_used_for_selection": False,
            "test_accessed_after_freeze": True,
            "route_status": (
                "blocked_non_finite_prediction"
                if test_prediction_failure
                else "blocked_data_quality" if data_quality_blocked else "ok"
            ),
            "test_prediction_failure": test_prediction_failure,
            "data_quality_failures_by_segment": MappingProxyType(data_quality_failures_by_segment),
            "preprocessing_fit_segment": spec.train_segment,
            "calibration_fit_segment": spec.train_segment,
            "densemble_prediction_api": "DEnsembleModel.predict(dataset, segment)",
            "blend_selection_metric": "validation_allocation_utility",
            "blend_grid_evaluations": tuple(blend_evaluations),
            "two_sided_blend_grid_evaluations": tuple(two_sided_blend_evaluations),
            "policy_grid_evaluations": policy_grid_evaluations,
            "utility_best_blend": utility_best_blend,
            "utility_best_two_sided_blend": utility_best_two_sided_blend,
            "selected_validation_utility": float(selected_policy.validation_utility),
            "frozen_policy": MappingProxyType(frozen_policy),
        }
    )
    predictions_by_segment = MappingProxyType(
        {
            spec.train_segment: train_predictions,
            spec.selection_segment: valid_predictions,
            spec.test_segment: test_predictions,
        }
    )
    positions_by_segment = MappingProxyType(
        {
            spec.train_segment: train_predictions["target_position"].copy(),
            spec.selection_segment: valid_predictions["target_position"].copy(),
            spec.test_segment: test_predictions["target_position"].copy(),
        }
    )
    return TailBudgetResult(
        model_comparison=comparison,
        selected_candidate=selected_name,
        predictions_by_segment=predictions_by_segment,
        positions_by_segment=positions_by_segment,
        failure_rows=failures,
        selection_evidence=evidence,
        floor=selected_policy.floor,
        cap=selected_policy.cap,
    )

__all__ = ["TailBudgetResult", "TailBudgetSpec", "run_tail_budget_models"]
