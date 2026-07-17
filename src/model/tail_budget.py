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


class _BlendModel:
    def __init__(self, left: Any, right: Any, weight: float) -> None:
        self.left = left
        self.right = right
        self.weight = float(weight)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        left = _predict_model(self.left, features)
        right = _predict_model(self.right, features)
        return self.weight * left + (1.0 - self.weight) * right


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


def _predict_model(model: Any, features: pd.DataFrame) -> np.ndarray:
    if model.__class__.__name__ == "DEnsembleModel" and hasattr(model, "ensemble"):
        values = np.zeros(len(features), dtype="float64")
        weights = np.asarray(model.sub_weights, dtype="float64")
        for submodel, sub_features, weight in zip(model.ensemble, model.sub_features, weights):
            values += np.asarray(
                submodel.predict(features.loc[:, sub_features].to_numpy(dtype="float64")),
                dtype="float64",
            ) * float(weight)
        values /= float(weights.sum())
    else:
        values = np.asarray(model.predict(features), dtype="float64").reshape(-1)
    if len(values) != len(features) or not np.isfinite(values).all():
        raise ValueError("candidate produced non-finite or misaligned predictions")
    return values


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
    return {
        "candidate": name,
        "model": name,
        "role": role,
        "status": "failed",
        "validation_score": np.nan,
        "selected": False,
        "reason_code": "fit_failed",
        "failure_type": type(exc).__name__,
        "detail": type(exc).__name__,
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
        "blend_weight": float(candidate.metadata.get("blend_weight", np.nan)),
        "floor": np.nan,
        "cap": np.nan,
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
            train_pred = _series(_predict_model(model, train_x), train_x.index, name)
            valid_pred = _series(_predict_model(model, valid_x), valid_x.index, name)
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
        train_pred = _series(_predict_model(model, train_x), train_x.index, "lightgbm_expected")
        valid_pred = _series(_predict_model(model, valid_x), valid_x.index, "lightgbm_expected")
        score = -float(mean_squared_error(valid_return, valid_pred))
        candidate = _Candidate("lightgbm_expected", "expected_return", model, train_pred, valid_pred, score)
        expected.append(candidate)
        rows.append(_success_row(candidate))
    except Exception as exc:
        rows.append(_failure_row("lightgbm_expected", "expected_return", exc))

    if len(expected) >= 2:
        left, right = expected[0], expected[1]
        blend_options: list[_Candidate] = []
        for weight in spec.blend_weights:
            model = _BlendModel(left.model, right.model, weight)
            train_pred = weight * left.train_prediction + (1.0 - weight) * right.train_prediction
            valid_pred = weight * left.valid_prediction + (1.0 - weight) * right.valid_prediction
            score = -float(mean_squared_error(valid_return, valid_pred))
            blend_options.append(
                _Candidate(
                    "validation_blend_expected",
                    "expected_return_blend",
                    model,
                    train_pred,
                    valid_pred,
                    score,
                    {"blend_weight": float(weight)},
                )
            )
        best = max(blend_options, key=lambda item: (item.component_score, -item.metadata["blend_weight"]))
        expected.append(best)
        rows.append(_success_row(best))

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
        train_pred = _series(_predict_model(model, train_x), train_x.index, "catboost_downside")
        valid_pred = _series(_predict_model(model, valid_x), valid_x.index, "catboost_downside")
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
        model = DEnsembleModel(
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
        model.fit(view)
        train_pred = _series(
            np.asarray(model.predict(view, segment=spec.train_segment)),
            train_x.index,
            "qlib_densemble_downside",
        )
        valid_pred = _series(
            np.asarray(model.predict(view, segment=spec.selection_segment)),
            valid_x.index,
            "qlib_densemble_downside",
        )
        if not np.isfinite(train_pred).all() or not np.isfinite(valid_pred).all():
            raise ValueError("DEnsemble produced non-finite predictions")
        score = -float(mean_pinball_loss(valid_downside, valid_pred, alpha=spec.downside_quantile))
        candidate = _Candidate("qlib_densemble_downside", "downside_budget", model, train_pred, valid_pred, score)
        downside.append(candidate)
        rows.append(_success_row(candidate))
    except Exception as exc:
        rows.append(_failure_row("qlib_densemble_downside", "downside_budget", exc))

    if len(downside) >= 2:
        left, right = downside[0], downside[1]
        blend_options = []
        for weight in spec.blend_weights:
            model = _BlendModel(left.model, right.model, weight)
            train_pred = weight * left.train_prediction + (1.0 - weight) * right.train_prediction
            valid_pred = weight * left.valid_prediction + (1.0 - weight) * right.valid_prediction
            score = -float(mean_pinball_loss(valid_downside, valid_pred, alpha=spec.downside_quantile))
            blend_options.append(
                _Candidate(
                    "validation_blend_downside",
                    "downside_budget_blend",
                    model,
                    train_pred,
                    valid_pred,
                    score,
                    {"blend_weight": float(weight)},
                )
            )
        best = max(blend_options, key=lambda item: (item.component_score, -item.metadata["blend_weight"]))
        downside.append(best)
        rows.append(_success_row(best))

    crash_prior = float(np.clip(train_crash.dropna().mean(), 0.0, 1.0))
    crash_model: Any = _ConstantModel(crash_prior)
    try:
        crash_model = _fit_crash_model(train_x, train_crash, spec)
        train_crash_pred = _series(crash_model.predict(train_x), train_x.index, "catboost_crash")
        valid_crash_pred = _series(crash_model.predict(valid_x), valid_x.index, "catboost_crash")
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
        train_crash_pred = _series(crash_model.predict(train_x), train_x.index, "crash_prior")
        valid_crash_pred = _series(crash_model.predict(valid_x), valid_x.index, "crash_prior")

    if not expected:
        raise RuntimeError("all expected-return candidates failed")
    if not downside:
        raise RuntimeError("all downside candidates failed")
    selected_expected = max(expected, key=lambda item: (item.component_score, item.name))
    mu_center = float(selected_expected.train_prediction.mean())
    mu_scale = float(selected_expected.train_prediction.std(ddof=0))
    if not np.isfinite(mu_scale) or mu_scale <= np.finfo("float64").eps:
        mu_scale = 1.0
    mu_direction = 1.0
    direction_frame = pd.concat([selected_expected.train_prediction, train_return], axis=1).dropna()
    if len(direction_frame) >= 2:
        correlation = float(direction_frame.corr().iloc[0, 1])
        if np.isfinite(correlation) and correlation < 0.0:
            mu_direction = -1.0
    valid_mu_score = (selected_expected.valid_prediction - mu_center) / mu_scale * mu_direction
    valid_quality = _quality_for_segment(valid_raw, data_quality_by_segment.get(spec.selection_segment))

    allocation_choices: list[tuple[float, str, _Candidate, float, float, pd.Series, pd.Series]] = []
    for candidate in downside:
        tail_scale = float(candidate.train_prediction.clip(lower=0.0).quantile(0.95))
        if not np.isfinite(tail_scale) or tail_scale <= np.finfo("float64").eps:
            tail_scale = 1.0
        valid_tail_score = (candidate.valid_prediction.clip(lower=0.0) / tail_scale).clip(0.0, 1.0)
        for floor, cap in spec.floor_cap_candidates:
            positions = soft_budget_position(
                valid_mu_score,
                valid_tail_score,
                valid_crash_pred.clip(0.0, 1.0),
                data_quality_valid=valid_quality,
                floor=floor,
                cap=cap,
            )
            utility = _budget_utility(valid_return, positions)
            allocation_choices.append(
                (utility, candidate.name, candidate, float(floor), float(cap), positions, valid_tail_score)
            )
    selected = max(allocation_choices, key=lambda item: (item[0], item[1], -item[3], -item[4]))
    selected_utility, selected_name, selected_downside, floor, cap, valid_position, valid_tail_score = selected

    for row in rows:
        if row["candidate"] in {item.name for item in downside}:
            choices = [choice for choice in allocation_choices if choice[1] == row["candidate"]]
            best_choice = max(choices, key=lambda item: (item[0], -item[3], -item[4]))
            row["validation_score"] = float(best_choice[0])
            row["floor"] = float(best_choice[3])
            row["cap"] = float(best_choice[4])
        row["selected"] = row["candidate"] == selected_name

    frozen_policy = {
        "expected_candidate": selected_expected.name,
        "downside_candidate": selected_downside.name,
        "downside_blend_weight": selected_downside.metadata.get("blend_weight"),
        "mu_center": mu_center,
        "mu_scale": mu_scale,
        "mu_direction": mu_direction,
        "tail_scale": float(selected_downside.train_prediction.clip(lower=0.0).quantile(0.95)),
        "crash_model": "catboost_crash" if isinstance(crash_model, _CalibratedCrashModel) else "train_crash_prior",
        "floor": floor,
        "cap": cap,
    }
    if not np.isfinite(frozen_policy["tail_scale"]) or frozen_policy["tail_scale"] <= np.finfo("float64").eps:
        frozen_policy["tail_scale"] = 1.0

    train_mu_score = (selected_expected.train_prediction - mu_center) / mu_scale * mu_direction
    train_tail_score = (
        selected_downside.train_prediction.clip(lower=0.0) / float(frozen_policy["tail_scale"])
    ).clip(0.0, 1.0)
    train_quality = _quality_for_segment(train_raw, data_quality_by_segment.get(spec.train_segment))
    train_position = soft_budget_position(
        train_mu_score,
        train_tail_score,
        train_crash_pred.clip(0.0, 1.0),
        data_quality_valid=train_quality,
        floor=floor,
        cap=cap,
    )
    train_predictions = pd.DataFrame(
        {
            "mu_raw": selected_expected.train_prediction,
            "mu_score": train_mu_score,
            "tail_raw": selected_downside.train_prediction,
            "tail_score": train_tail_score,
            "crash_probability": train_crash_pred.clip(0.0, 1.0),
            "target_position": train_position,
        },
        index=train_x.index,
    ).astype("float64")
    valid_predictions = pd.DataFrame(
        {
            "mu_raw": selected_expected.valid_prediction,
            "mu_score": valid_mu_score,
            "tail_raw": selected_downside.valid_prediction,
            "tail_score": valid_tail_score,
            "crash_probability": valid_crash_pred.clip(0.0, 1.0),
            "target_position": valid_position,
        },
        index=valid_x.index,
    ).astype("float64")

    # This is the first test-segment access: the validation-selected policy is frozen above.
    test_raw = _prepare_segment(dataset, spec.test_segment, spec)
    test_x = _transform(test_raw.features, median, mean, scale)
    test_mu_raw = _series(_predict_model(selected_expected.model, test_x), test_x.index, "mu_raw")
    test_tail_values = _predict_model(selected_downside.model, test_x)
    test_tail_raw = _series(test_tail_values, test_x.index, "tail_raw")
    test_crash_probability = _series(
        _predict_model(crash_model, test_x),
        test_x.index,
        "crash_probability",
    ).clip(0.0, 1.0)
    test_mu_score = (test_mu_raw - mu_center) / mu_scale * mu_direction
    test_tail_score = (test_tail_raw.clip(lower=0.0) / float(frozen_policy["tail_scale"])).clip(0.0, 1.0)
    test_quality = _quality_for_segment(test_raw, data_quality_by_segment.get(spec.test_segment))
    test_position = soft_budget_position(
        test_mu_score,
        test_tail_score,
        test_crash_probability,
        data_quality_valid=test_quality,
        floor=floor,
        cap=cap,
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
    for segment_name, predictions in (
        (spec.train_segment, train_predictions),
        (spec.selection_segment, valid_predictions),
        (spec.test_segment, test_predictions),
    ):
        if not np.isfinite(predictions.to_numpy(dtype="float64")).all():
            raise RuntimeError(f"non-finite frozen prediction in {segment_name}")

    comparison = pd.DataFrame(rows).sort_values(["role", "candidate"], kind="stable").reset_index(drop=True)
    failures = comparison.loc[comparison["status"] != "ok"].reset_index(drop=True)
    evidence = MappingProxyType(
        {
            "selection_segment": spec.selection_segment,
            "test_labels_used_for_selection": False,
            "test_accessed_after_freeze": True,
            "preprocessing_fit_segment": spec.train_segment,
            "calibration_fit_segment": spec.train_segment,
            "selected_validation_utility": float(selected_utility),
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
        floor=floor,
        cap=cap,
    )

__all__ = ["TailBudgetResult", "TailBudgetSpec", "run_tail_budget_models"]
