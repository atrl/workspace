from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .risk_adjusted_common import soft_budget_position


SEQUENCE_FAMILIES = ("tcn", "gru_alstm", "transformer")


@dataclass(frozen=True)
class PanelSequenceSpec:
    """Frozen, compact search contract for SOXL multivariate sequences."""

    train_segment: str = "train"
    selection_segment: str = "valid"
    test_segment: str = "test"
    return_label: str = "future_return_5d"
    downside_label: str = "future_downside_5d"
    crash_label: str = "future_crash_5d"
    label_horizon: int = 5
    windows: tuple[int, ...] = (20, 60)
    model_families: tuple[str, ...] = SEQUENCE_FAMILIES
    hidden_width: int = 16
    temporal_layers: int = 1
    dropout: float = 0.10
    max_epochs: int = 24
    early_stopping_patience: int = 4
    learning_rate: float = 0.01
    batch_size: int = 64
    floor_cap_candidates: tuple[tuple[float, float], ...] = (
        (0.15, 0.60),
        (0.15, 0.75),
        (0.20, 0.75),
    )
    random_state: int = 1729

    def __post_init__(self) -> None:
        if self.selection_segment != "valid":
            raise ValueError("selection_segment must be the validation segment")
        if len({self.train_segment, self.selection_segment, self.test_segment}) != 3:
            raise ValueError("train, validation, and test segments must be distinct")
        if self.label_horizon != 5:
            raise ValueError("sequence label horizon is fixed at five trading days")
        if tuple(self.windows) != (20, 60):
            raise ValueError("sequence windows must be exactly 20 and 60 sessions")
        if not self.model_families or any(
            family not in SEQUENCE_FAMILIES for family in self.model_families
        ):
            raise ValueError(f"model_families must be chosen from {SEQUENCE_FAMILIES}")
        if not 1 <= int(self.hidden_width) <= 32:
            raise ValueError("hidden width must be positive and at most 32")
        if not 1 <= int(self.temporal_layers) <= 2:
            raise ValueError("temporal layers must be positive and at most two")
        if not 0.10 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be at least 0.10 and below one")
        if self.max_epochs < 1 or self.early_stopping_patience < 1:
            raise ValueError("epoch and patience values must be positive")
        if self.learning_rate <= 0.0 or self.batch_size < 1:
            raise ValueError("learning rate and batch size must be positive")
        if not self.floor_cap_candidates:
            raise ValueError("floor/cap candidates must not be empty")
        for floor, cap in self.floor_cap_candidates:
            if not 0.0 <= float(floor) < float(cap) <= 1.0:
                raise ValueError("floor/cap candidates must satisfy 0 <= floor < cap <= 1")


@dataclass(frozen=True)
class SequenceWindowBatch:
    values: np.ndarray
    targets: np.ndarray | None
    endpoints: pd.Index
    source_starts: tuple[Any, ...]
    source_endpoints: tuple[Any, ...]


@dataclass(frozen=True)
class _FittedTorchSequenceEstimator:
    family: str
    input_width: int
    hidden_width: int
    temporal_layers: int
    dropout: float
    random_state: int
    state: Mapping[str, np.ndarray]
    best_epoch: int
    validation_loss: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        with _isolated_torch_process_state():
            torch, _ = _torch_modules()
            model = _build_torch_model(
                family=self.family,
                input_width=self.input_width,
                hidden_width=self.hidden_width,
                temporal_layers=self.temporal_layers,
                dropout=self.dropout,
            )
            model.load_state_dict(
                {name: torch.as_tensor(value).clone() for name, value in self.state.items()}
            )
            model.eval()
            with torch.no_grad():
                prediction = model(torch.as_tensor(values, dtype=torch.float32))
        return prediction.detach().cpu().numpy().reshape(-1).astype("float64")


@dataclass(frozen=True)
class FittedPanelSequencePolicy:
    """Pickle-safe validation-frozen sequence policy."""

    feature_columns: tuple[str, ...]
    preprocessing_median: pd.Series
    preprocessing_mean: pd.Series
    preprocessing_scale: pd.Series
    estimator: _FittedTorchSequenceEstimator
    family: str
    window: int
    mu_center: float
    mu_scale: float
    floor: float
    cap: float
    history_tail: pd.DataFrame

    def predict(
        self,
        features: pd.DataFrame,
        *,
        data_quality_valid: bool | pd.Series | Sequence[bool] | None = None,
        segment: str = "inference",
    ) -> pd.DataFrame:
        raw = _coerce_feature_frame(features).reindex(columns=list(self.feature_columns))
        _validate_datetime_index(raw.index, role=f"{segment} inference")
        intrinsic = pd.Series(
            np.isfinite(raw.to_numpy(dtype="float64")).all(axis=1), index=raw.index
        )
        quality = _combine_quality(intrinsic, data_quality_valid)
        return _predict_policy_segment(self, raw, quality, segment=segment)


@dataclass(frozen=True)
class PanelSequenceResult:
    model_comparison: pd.DataFrame
    selected_candidate: str
    predictions_by_segment: Mapping[str, pd.DataFrame]
    positions_by_segment: Mapping[str, pd.Series]
    failure_rows: pd.DataFrame
    selection_evidence: Mapping[str, Any]
    mode_evidence: Mapping[str, Mapping[str, Any]]
    blocked_mode_positions: Mapping[str, pd.Series]
    floor: float
    cap: float
    fitted_policy: FittedPanelSequencePolicy | None


@dataclass
class _SegmentData:
    features: pd.DataFrame
    labels: pd.DataFrame
    intrinsic_quality: pd.Series


@dataclass(frozen=True)
class _FrozenRouteDecision:
    """Capability token required before the route may read DK_I test data."""

    selected_candidate: str
    policy: FittedPanelSequencePolicy | None
    frozen: bool = True

    def __post_init__(self) -> None:
        if not str(self.selected_candidate).strip():
            raise ValueError("a frozen route decision requires a candidate or block reason")
        if self.frozen is not True:
            raise ValueError("route decision token must be frozen")


def _freeze_route_decision(
    selected_candidate: str,
    policy: FittedPanelSequencePolicy | None,
    *,
    event_trace: list[str],
) -> _FrozenRouteDecision:
    decision = _FrozenRouteDecision(
        selected_candidate=selected_candidate,
        policy=policy,
    )
    event_trace.append("candidate_selection_frozen")
    return decision


def _torch_modules():
    try:
        import torch
        from torch import nn
    except (ImportError, OSError) as exc:
        raise ModuleNotFoundError("PyTorch sequence backend is unavailable") from exc
    return torch, nn


def _build_torch_model(
    *, family: str, input_width: int, hidden_width: int, temporal_layers: int, dropout: float
):
    torch, nn = _torch_modules()

    class TCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            in_channels = input_width
            for _ in range(temporal_layers):
                layers.extend(
                    [
                        nn.Conv1d(in_channels, hidden_width, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                in_channels = hidden_width
            self.temporal = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_width, 1)

        def forward(self, values):
            encoded = self.temporal(values.transpose(1, 2))
            return self.head(encoded[:, :, -1]).squeeze(-1)

    class GRUALSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_width,
                hidden_width,
                num_layers=temporal_layers,
                dropout=dropout if temporal_layers > 1 else 0.0,
                batch_first=True,
            )
            self.attention = nn.Linear(hidden_width, 1)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_width, 1)

        def forward(self, values):
            encoded, _ = self.gru(values)
            weights = torch.softmax(self.attention(encoded).squeeze(-1), dim=1)
            context = (encoded * weights.unsqueeze(-1)).sum(dim=1)
            return self.head(self.dropout(context)).squeeze(-1)

    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(input_width, hidden_width)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_width,
                nhead=1,
                dim_feedforward=hidden_width * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=temporal_layers, enable_nested_tensor=False
            )
            self.head = nn.Linear(hidden_width, 1)

        def forward(self, values):
            length = values.shape[1]
            positions = torch.arange(length, device=values.device, dtype=values.dtype)
            scales = torch.arange(
                1, hidden_width + 1, device=values.device, dtype=values.dtype
            )
            positional = torch.sin(positions[:, None] / scales[None, :])
            encoded = self.encoder(self.projection(values) + positional.unsqueeze(0))
            return self.head(encoded[:, -1, :]).squeeze(-1)

    constructors = {"tcn": TCN, "gru_alstm": GRUALSTM, "transformer": Transformer}
    if family not in constructors:
        raise ValueError(f"unknown sequence family: {family}")
    return constructors[family]()


def _coerce_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("features must be a non-empty DataFrame")
    result = frame.copy()
    if (
        isinstance(result.columns, pd.MultiIndex)
        and "feature" in result.columns.get_level_values(0)
    ):
        result = result["feature"].copy()
    result.columns = [str(column) for column in result.columns]
    return result.apply(pd.to_numeric, errors="coerce").astype("float64")


def _validate_datetime_index(index: pd.Index, *, role: str) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        if "datetime" not in index.names:
            raise ValueError(f"{role} index must include a datetime level")
        values = index.get_level_values("datetime")
    else:
        values = index
    if not isinstance(values, pd.DatetimeIndex) and not pd.api.types.is_datetime64_any_dtype(
        values.dtype
    ):
        raise ValueError(f"{role} index must be a datetime index")
    try:
        datetimes = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{role} datetime index is invalid") from exc
    if datetimes.hasnans:
        raise ValueError(f"{role} datetime index must not contain missing values")
    if not datetimes.is_unique:
        raise ValueError(f"{role} datetime index must be unique")
    if not datetimes.is_monotonic_increasing:
        raise ValueError(f"{role} datetime index must be strictly increasing")
    return datetimes


def _require_ordered_non_overlapping_segments(
    left: _SegmentData,
    right: _SegmentData,
    *,
    left_name: str,
    right_name: str,
) -> None:
    left_dates = _validate_datetime_index(left.features.index, role=left_name)
    right_dates = _validate_datetime_index(right.features.index, role=right_name)
    if left_dates[-1] >= right_dates[0]:
        raise ValueError(
            f"{left_name} and {right_name} must not overlap and {left_name} must precede {right_name}"
        )


def _prepare_learning_segment(dataset, segment: str, spec: PanelSequenceSpec) -> _SegmentData:
    from qlib.data.dataset.handler import DataHandlerLP

    frame = dataset.prepare(
        segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"segment {segment!r} is empty")
    labels_needed = (spec.return_label, spec.downside_label, spec.crash_label)
    if (
        isinstance(frame.columns, pd.MultiIndex)
        and "feature" in frame.columns.get_level_values(0)
    ):
        features = _coerce_feature_frame(frame["feature"])
        if "label" not in frame.columns.get_level_values(0):
            raise ValueError(f"segment {segment!r} does not include qlib labels")
        labels = frame["label"].copy()
    else:
        missing = [name for name in labels_needed if name not in frame.columns]
        if missing:
            raise ValueError(f"segment {segment!r} is missing labels: {missing}")
        labels = frame.loc[:, list(labels_needed)].copy()
        features = _coerce_feature_frame(frame.drop(columns=list(labels_needed)))
    labels.columns = [str(column) for column in labels.columns]
    missing = [name for name in labels_needed if name not in labels.columns]
    if missing:
        raise ValueError(f"segment {segment!r} is missing labels: {missing}")
    labels = labels.loc[:, list(labels_needed)].apply(pd.to_numeric, errors="coerce")
    quality = pd.Series(
        np.isfinite(features.to_numpy(dtype="float64")).all(axis=1), index=features.index
    )
    _validate_datetime_index(features.index, role=segment)
    return _SegmentData(features, labels.astype("float64"), quality)


def _prepare_inference_segment(
    dataset,
    segment: str,
    *,
    frozen_decision: _FrozenRouteDecision,
    event_trace: list[str],
) -> _SegmentData:
    from qlib.data.dataset.handler import DataHandlerLP

    if not isinstance(frozen_decision, _FrozenRouteDecision) or not frozen_decision.frozen:
        raise RuntimeError("test DK_I access requires a frozen route decision")
    if not event_trace or event_trace[-1] != "candidate_selection_frozen":
        raise RuntimeError("frozen decision must be recorded before test DK_I access")
    features = _coerce_feature_frame(
        dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
    )
    event_trace.append("test_dk_i_accessed")
    quality = pd.Series(
        np.isfinite(features.to_numpy(dtype="float64")).all(axis=1), index=features.index
    )
    _validate_datetime_index(features.index, role=segment)
    return _SegmentData(features, pd.DataFrame(index=features.index), quality)


def build_sequence_windows(
    features: pd.DataFrame,
    targets: pd.Series | Sequence[float] | None = None,
    *,
    window: int,
    endpoint_index: pd.Index | None = None,
) -> SequenceWindowBatch:
    """Build past-only windows ending at each included decision row."""

    if int(window) < 1:
        raise ValueError("window must be positive")
    numeric = _coerce_feature_frame(features)
    _validate_datetime_index(numeric.index, role="sequence window")
    values = numeric.to_numpy(dtype="float64")
    target_series = (
        None
        if targets is None
        else pd.Series(targets, index=numeric.index, dtype="float64")
    )
    allowed = None if endpoint_index is None else set(endpoint_index.tolist())
    sequences: list[np.ndarray] = []
    y_values: list[float] = []
    endpoints: list[Any] = []
    starts: list[Any] = []
    ends: list[Any] = []
    for end_position in range(int(window) - 1, len(numeric)):
        endpoint = numeric.index[end_position]
        if allowed is not None and endpoint not in allowed:
            continue
        start_position = end_position - int(window) + 1
        sequences.append(values[start_position : end_position + 1])
        endpoints.append(endpoint)
        starts.append(numeric.index[start_position])
        ends.append(endpoint)
        if target_series is not None:
            y_values.append(float(target_series.iloc[end_position]))
    shape = (0, int(window), numeric.shape[1])
    sequence_values = np.stack(sequences) if sequences else np.empty(shape, dtype="float64")
    try:
        endpoint_values = numeric.index.__class__.from_tuples(
            endpoints, names=numeric.index.names
        )
    except (AttributeError, TypeError):
        endpoint_values = pd.Index(endpoints, name=numeric.index.name)
    return SequenceWindowBatch(
        values=sequence_values,
        targets=None if target_series is None else np.asarray(y_values, dtype="float64"),
        endpoints=endpoint_values,
        source_starts=tuple(starts),
        source_endpoints=tuple(ends),
    )


def _fit_preprocessor(train: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    median = train.replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    filled = train.replace([np.inf, -np.inf], np.nan).fillna(median)
    mean = filled.mean()
    scale = filled.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return median, mean, scale


def _transform(
    frame: pd.DataFrame, median: pd.Series, mean: pd.Series, scale: pd.Series
) -> pd.DataFrame:
    aligned = (
        frame.reindex(columns=median.index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(median)
    )
    return ((aligned - mean) / scale).fillna(0.0).astype("float64")


@contextmanager
def _isolated_torch_process_state(seed: int | None = None):
    torch, _ = _torch_modules()
    numpy_state = np.random.get_state()
    thread_count = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.random.default_generator.manual_seed(int(seed))
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            yield
    finally:
        np.random.set_state(numpy_state)
        torch.set_num_threads(thread_count)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _fit_torch_candidate(
    train: SequenceWindowBatch,
    valid: SequenceWindowBatch,
    *,
    family: str,
    spec: PanelSequenceSpec,
) -> _FittedTorchSequenceEstimator:
    with _isolated_torch_process_state(spec.random_state):
        return _fit_torch_candidate_impl(train, valid, family=family, spec=spec)


def _fit_torch_candidate_impl(
    train: SequenceWindowBatch,
    valid: SequenceWindowBatch,
    *,
    family: str,
    spec: PanelSequenceSpec,
) -> _FittedTorchSequenceEstimator:
    torch, nn = _torch_modules()
    if (
        train.targets is None
        or valid.targets is None
        or not len(train.values)
        or not len(valid.values)
    ):
        raise ValueError("sequence candidate requires non-empty train and validation batches")
    train_finite = np.isfinite(train.values).all(axis=(1, 2)) & np.isfinite(train.targets)
    valid_finite = np.isfinite(valid.values).all(axis=(1, 2)) & np.isfinite(valid.targets)
    if not train_finite.all() or not valid_finite.all():
        raise ValueError("sequence train and validation evidence must be finite")
    model = _build_torch_model(
        family=family,
        input_width=train.values.shape[2],
        hidden_width=spec.hidden_width,
        temporal_layers=spec.temporal_layers,
        dropout=spec.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    loss_fn = nn.MSELoss()
    x_train = torch.as_tensor(train.values, dtype=torch.float32)
    y_train = torch.as_tensor(train.targets, dtype=torch.float32)
    x_valid = torch.as_tensor(valid.values, dtype=torch.float32)
    y_valid = torch.as_tensor(valid.targets, dtype=torch.float32)
    generator = torch.Generator().manual_seed(spec.random_state)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, np.ndarray] | None = None
    stale = 0
    for epoch in range(spec.max_epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), spec.batch_size):
            indices = order[start : start + spec.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[indices]), y_train[indices])
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(x_valid), y_valid).item())
        if not np.isfinite(validation_loss):
            raise ValueError("non-finite validation loss")
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().numpy().copy()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= spec.early_stopping_patience:
                break
    if best_state is None:
        raise ValueError("sequence candidate did not produce a finite checkpoint")
    return _FittedTorchSequenceEstimator(
        family=family,
        input_width=train.values.shape[2],
        hidden_width=spec.hidden_width,
        temporal_layers=spec.temporal_layers,
        dropout=spec.dropout,
        random_state=spec.random_state,
        state=best_state,
        best_epoch=best_epoch,
        validation_loss=best_loss,
    )


def _combine_quality(
    intrinsic: pd.Series, external: bool | pd.Series | Sequence[bool] | None
) -> pd.Series:
    if external is None:
        return intrinsic.astype(bool)
    if isinstance(external, pd.Series):
        supplied = external.reindex(intrinsic.index).fillna(False).astype(bool)
    elif np.isscalar(external):
        supplied = pd.Series(bool(external), index=intrinsic.index)
    else:
        values = np.asarray(external)
        if len(values) != len(intrinsic):
            raise ValueError("data quality input must align with its segment")
        supplied = pd.Series(values.astype(bool), index=intrinsic.index)
    return intrinsic.astype(bool) & supplied


def _quality_evidence(quality: pd.Series) -> dict[str, Any]:
    invalid = ~quality
    return {
        "invalid_count": int(invalid.sum()),
        "first_offending_date": (
            _index_value_to_string(quality.index[int(np.flatnonzero(invalid.to_numpy())[0])])
            if invalid.any()
            else None
        ),
        "reason_code": "invalid_required_data" if invalid.any() else "",
    }


def _label_completeness_evidence(
    labels: pd.DataFrame,
    *,
    required_labels: Sequence[str],
    purge_rows: int,
) -> dict[str, Any]:
    finite = pd.Series(
        np.isfinite(labels.loc[:, list(required_labels)].to_numpy(dtype="float64")).all(
            axis=1
        ),
        index=labels.index,
    )
    boundary_count = min(int(purge_rows), len(finite))
    eligible_count = len(finite) - boundary_count
    eligible = finite.iloc[:eligible_count]
    boundary = finite.iloc[eligible_count:]
    internal_invalid = int((~eligible).sum())
    return {
        "checked_observations": int(len(eligible)),
        "boundary_excluded_rows": boundary_count,
        "boundary_invalid_rows": int((~boundary).sum()),
        "internal_invalid_rows": internal_invalid,
        "labels_complete": internal_invalid == 0,
    }


def _index_value_to_string(value: Any) -> str:
    if isinstance(value, tuple):
        value = value[-1]
    try:
        return pd.Timestamp(value).isoformat()
    except (TypeError, ValueError):
        return str(value)


def _sequence_quality(
    history_quality: pd.Series, current_quality: pd.Series, *, window: int
) -> pd.Series:
    combined = pd.concat([history_quality, current_quality])
    rolling = (
        combined.astype("int8")
        .rolling(window, min_periods=window)
        .min()
        .fillna(0)
        .astype(bool)
    )
    return rolling.reindex(current_quality.index).fillna(False).astype(bool)


def _scores_to_predictions(
    raw_prediction: pd.Series,
    quality: pd.Series,
    *,
    mu_center: float,
    mu_scale: float,
    floor: float,
    cap: float,
) -> pd.DataFrame:
    mu_score = (raw_prediction - float(mu_center)) / float(mu_scale)
    tail_score = (-mu_score).clip(0.0, 1.0)
    crash_probability = 1.0 / (1.0 + np.exp(np.clip(mu_score, -40.0, 40.0)))
    target = soft_budget_position(
        mu_score,
        tail_score,
        crash_probability,
        data_quality_valid=quality,
        floor=floor,
        cap=cap,
    )
    return pd.DataFrame(
        {
            "mu_raw": raw_prediction,
            "mu_score": mu_score,
            "tail_score": tail_score,
            "crash_probability": crash_probability,
            "target_position": target,
        },
        index=raw_prediction.index,
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float64")


def _predict_with_context(
    estimator: _FittedTorchSequenceEstimator,
    transformed_history: pd.DataFrame,
    transformed_current: pd.DataFrame,
    history_quality: pd.Series,
    current_quality: pd.Series,
    *,
    window: int,
    mu_center: float,
    mu_scale: float,
    floor: float,
    cap: float,
) -> pd.DataFrame:
    combined = pd.concat([transformed_history.tail(window - 1), transformed_current])
    batch = build_sequence_windows(
        combined, window=window, endpoint_index=transformed_current.index
    )
    raw = pd.Series(0.0, index=transformed_current.index, name="mu_raw")
    if len(batch.values):
        values = estimator.predict(batch.values)
        if len(values) != len(batch.endpoints) or not np.isfinite(values).all():
            raise ValueError("sequence candidate produced invalid inference values")
        raw.loc[batch.endpoints] = values
    quality = _sequence_quality(
        history_quality.tail(window - 1), current_quality, window=window
    )
    return _scores_to_predictions(
        raw,
        quality,
        mu_center=mu_center,
        mu_scale=mu_scale,
        floor=floor,
        cap=cap,
    )


def _predict_policy_segment(
    policy: FittedPanelSequencePolicy,
    features: pd.DataFrame,
    quality: pd.Series,
    *,
    segment: str,
) -> pd.DataFrame:
    del segment
    history_raw = policy.history_tail.reindex(columns=list(policy.feature_columns))
    transformed_history = _transform(
        history_raw,
        policy.preprocessing_median,
        policy.preprocessing_mean,
        policy.preprocessing_scale,
    )
    transformed_current = _transform(
        features,
        policy.preprocessing_median,
        policy.preprocessing_mean,
        policy.preprocessing_scale,
    )
    history_quality = pd.Series(True, index=transformed_history.index)
    return _predict_with_context(
        policy.estimator,
        transformed_history,
        transformed_current,
        history_quality,
        quality,
        window=policy.window,
        mu_center=policy.mu_center,
        mu_scale=policy.mu_scale,
        floor=policy.floor,
        cap=policy.cap,
    )


def _validation_utility(realized: pd.Series, position: pd.Series) -> float:
    aligned = pd.concat(
        [realized.rename("return"), position.rename("position")], axis=1
    ).dropna()
    if aligned.empty:
        return float("-inf")
    strategy = aligned["return"].to_numpy(dtype="float64") * aligned["position"].to_numpy(
        dtype="float64"
    )
    downside = np.minimum(strategy, 0.0)
    return float(np.mean(strategy) / max(float(np.sqrt(np.mean(downside**2))), 1e-6))


def _failure_row(candidate: str, exc: Exception) -> dict[str, Any]:
    backend = isinstance(exc, (ImportError, ModuleNotFoundError, OSError))
    return {
        "candidate": candidate,
        "model": candidate,
        "family": candidate.rsplit("_w", 1)[0],
        "window": int(candidate.rsplit("_w", 1)[1]),
        "status": "unavailable" if backend else "failed",
        "validation_score": np.nan,
        "validation_loss": np.nan,
        "best_epoch": np.nan,
        "floor": np.nan,
        "cap": np.nan,
        "selected": False,
        "reason_code": "backend_unavailable" if backend else "fit_failed",
        "failure_type": type(exc).__name__,
        "detail": str(exc),
    }


def _zero_predictions(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        0.0,
        index=index,
        columns=("mu_raw", "mu_score", "tail_score", "crash_probability", "target_position"),
    )


def _pit_evidence(
    pit_panel: pd.DataFrame | None,
    *,
    qlib_indices: Sequence[pd.Index],
    qlib_lineage: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if pit_panel is None:
        evidence = {
            "status": "blocked_missing_pit_panel",
            "reason_code": "blocked_missing_pit_panel",
            "point_in_time_membership": False,
            "validation_status": "missing_pit_panel",
            "validation_errors": ("missing_panel",),
            "provenance": {
                "point_in_time_membership": False,
                "source": None,
                "as_of": None,
                "effective_date_field": None,
                "provider_uri": None,
                "manifest_hash": None,
            },
        }
        row = {
            "candidate": "pit_cross_sectional_panel",
            "model": "pit_cross_sectional_panel",
            "family": "constituent_cross_sectional",
            "window": np.nan,
            "status": "blocked",
            "validation_score": np.nan,
            "validation_loss": np.nan,
            "best_epoch": np.nan,
            "floor": np.nan,
            "cap": np.nan,
            "selected": False,
            "reason_code": "blocked_missing_pit_panel",
            "failure_type": "MissingPointInTimePanel",
            "detail": "point-in-time constituent membership and aligned features are required",
        }
        return evidence, row
    errors: list[str] = []
    if not isinstance(pit_panel, pd.DataFrame) or pit_panel.empty:
        errors.append("empty_panel")
    attrs = pit_panel.attrs if isinstance(pit_panel, pd.DataFrame) else {}
    provenance = {
        "point_in_time_membership": attrs.get("point_in_time_membership") is True,
        "source": attrs.get("source"),
        "as_of": attrs.get("as_of"),
        "effective_date_field": attrs.get("effective_date_field"),
        "provider_uri": attrs.get("provider_uri"),
        "manifest_hash": attrs.get("manifest_hash"),
    }
    if not provenance["point_in_time_membership"]:
        errors.append("missing_point_in_time_provenance")
    for field in (
        "source",
        "as_of",
        "effective_date_field",
        "provider_uri",
        "manifest_hash",
    ):
        if not isinstance(provenance[field], str) or not provenance[field].strip():
            errors.append(f"missing_provenance_{field}")
    lineage = dict(qlib_lineage or {})
    if not lineage:
        errors.append("missing_qlib_lineage")
    else:
        for field in ("provider_uri", "manifest_hash"):
            expected = lineage.get(field)
            if not isinstance(expected, str) or not expected.strip():
                errors.append(f"missing_qlib_lineage_{field}")
            elif provenance[field] != expected:
                errors.append(f"provenance_{field}_mismatch")
    index_valid = isinstance(pit_panel, pd.DataFrame) and isinstance(
        pit_panel.index, pd.MultiIndex
    )
    if not index_valid or not {"datetime", "instrument"}.issubset(
        set(pit_panel.index.names if index_valid else ())
    ):
        errors.append("missing_datetime_instrument_index")
    elif not pit_panel.index.is_unique:
        errors.append("duplicate_datetime_instrument_rows")
    if not isinstance(pit_panel, pd.DataFrame) or "pit_member" not in pit_panel.columns:
        errors.append("missing_membership_column")
        active_members = pd.Series(dtype=bool)
    else:
        membership = pit_panel["pit_member"]
        membership_valid = membership.notna() & membership.isin((True, False, 0, 1))
        if not bool(membership_valid.all()):
            errors.append("invalid_membership_values")
        active_members = membership.fillna(False).astype(bool)
        if not bool(active_members.any()):
            errors.append("no_active_membership")
    raw_feature_columns = (
        tuple(column for column in pit_panel.columns if str(column) != "pit_member")
        if isinstance(pit_panel, pd.DataFrame)
        else ()
    )
    feature_columns = tuple(str(column) for column in raw_feature_columns)
    if not raw_feature_columns:
        errors.append("missing_feature_columns")
    elif not np.isfinite(
        pit_panel.loc[:, list(raw_feature_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype="float64")
    ).all():
        errors.append("non_finite_feature_values")
    qlib_dates = set()
    for index in qlib_indices:
        if isinstance(index, pd.MultiIndex):
            level = "datetime" if "datetime" in index.names else index.names[-1]
            values = index.get_level_values(level)
        else:
            values = index
        qlib_dates.update(pd.DatetimeIndex(values).normalize().tolist())
    pit_dates: set[pd.Timestamp] = set()
    if index_valid and "datetime" in pit_panel.index.names:
        try:
            pit_dates = set(
                pd.DatetimeIndex(pit_panel.index.get_level_values("datetime"))
                .normalize()
                .tolist()
            )
        except (TypeError, ValueError):
            errors.append("invalid_datetime_values")
    if not pit_dates or not pit_dates.issubset(qlib_dates):
        errors.append("dates_not_aligned_to_qlib_segments")
    effective_date_field = provenance["effective_date_field"]
    if isinstance(effective_date_field, str) and effective_date_field.strip():
        if effective_date_field not in pit_panel.columns and effective_date_field not in (
            pit_panel.index.names if index_valid else ()
        ):
            errors.append("effective_date_field_not_found")
    as_of = pd.NaT
    if isinstance(provenance["as_of"], str) and provenance["as_of"].strip():
        try:
            as_of = pd.Timestamp(provenance["as_of"]).normalize()
        except (TypeError, ValueError):
            errors.append("invalid_provenance_as_of")
    if not pd.isna(as_of) and pit_dates and as_of < max(pit_dates):
        errors.append("provenance_as_of_precedes_panel")
    if not pd.isna(as_of) and qlib_dates and as_of > max(qlib_dates):
        errors.append("provenance_as_of_exceeds_qlib_data")
    valid = not errors
    status = "eligible" if valid else "blocked_missing_pit_panel"
    validation_status = (
        "valid_aligned_pit_panel" if valid else f"invalid_pit_panel_{errors[0]}"
    )
    evidence = {
        "status": status,
        "reason_code": "" if valid else "blocked_missing_pit_panel",
        "point_in_time_membership": bool(valid),
        "observations": int(len(pit_panel)) if isinstance(pit_panel, pd.DataFrame) else 0,
        "active_observations": int(active_members.sum()),
        "feature_columns": feature_columns,
        "aligned_dates": len(pit_dates & qlib_dates),
        "validation_status": validation_status,
        "validation_errors": tuple(errors),
        "provenance": provenance,
        "qlib_lineage": lineage,
    }
    row = {
        "candidate": "pit_cross_sectional_panel",
        "model": "pit_cross_sectional_panel",
        "family": "constituent_cross_sectional",
        "window": np.nan,
        "status": "eligible" if valid else "blocked",
        "validation_score": np.nan,
        "validation_loss": np.nan,
        "best_epoch": np.nan,
        "floor": np.nan,
        "cap": np.nan,
        "selected": False,
        "reason_code": "" if valid else "blocked_missing_pit_panel",
        "failure_type": "" if valid else "InvalidPointInTimePanel",
        "detail": (
            "PIT panel passed membership gate"
            if valid
            else "PIT panel failed membership gate"
        ),
    }
    return evidence, row


def _diagnostic_source_evidence(
    features_by_segment: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    required_prefixes = {
        "SOXL": ("soxl_",),
        "SOXX": ("soxx_",),
        "VIX": ("vix_",),
    }
    evidence: dict[str, dict[str, Any]] = {}
    for source, prefixes in required_prefixes.items():
        matches_by_segment: dict[str, tuple[str, ...]] = {}
        for segment, features in features_by_segment.items():
            matches_by_segment[segment] = tuple(
                column
                for column in features.columns
                if any(str(column).lower().startswith(prefix) for prefix in prefixes)
            )
        common_matches = tuple(
            sorted(set.intersection(*(set(values) for values in matches_by_segment.values())))
        )
        evidence[source] = {
            "available": all(bool(values) for values in matches_by_segment.values())
            and bool(common_matches),
            "matched_columns": common_matches,
            "matches_by_segment": matches_by_segment,
        }
    return evidence


def run_panel_sequence_models(
    dataset,
    *,
    spec: PanelSequenceSpec | None = None,
    data_quality_by_segment: Mapping[str, bool | pd.Series | Sequence[bool]] | None = None,
    pit_panel: pd.DataFrame | None = None,
    qlib_lineage: Mapping[str, Any] | None = None,
) -> PanelSequenceResult:
    """Fit compact sequences, select on validation, then access test inference data."""

    spec = spec or PanelSequenceSpec()
    external_quality = dict(data_quality_by_segment or {})
    train = _prepare_learning_segment(dataset, spec.train_segment, spec)
    valid = _prepare_learning_segment(dataset, spec.selection_segment, spec)
    _require_ordered_non_overlapping_segments(
        train,
        valid,
        left_name=spec.train_segment,
        right_name=spec.selection_segment,
    )
    if tuple(train.features.columns) != tuple(valid.features.columns):
        raise ValueError("train and validation feature columns must match")
    train_quality = _combine_quality(
        train.intrinsic_quality, external_quality.get(spec.train_segment)
    )
    valid_quality = _combine_quality(
        valid.intrinsic_quality, external_quality.get(spec.selection_segment)
    )
    source_evidence = _diagnostic_source_evidence(
        {spec.train_segment: train.features, spec.selection_segment: valid.features}
    )
    missing_required_source = not all(
        item["available"] for item in source_evidence.values()
    )
    purge_rows = spec.label_horizon + 1
    required_labels = (spec.return_label, spec.downside_label, spec.crash_label)
    train_label_evidence = _label_completeness_evidence(
        train.labels,
        required_labels=required_labels,
        purge_rows=purge_rows,
    )
    valid_label_evidence = _label_completeness_evidence(
        valid.labels,
        required_labels=required_labels,
        purge_rows=purge_rows,
    )
    label_completeness_evidence = {
        spec.train_segment: train_label_evidence,
        spec.selection_segment: valid_label_evidence,
    }
    median, mean, scale = _fit_preprocessor(train.features)
    train_x = _transform(train.features, median, mean, scale)
    valid_x = _transform(valid.features, median, mean, scale)
    train_endpoints = train.features.index[:-purge_rows]
    valid_endpoints = valid.features.index[:-purge_rows]
    rows: list[dict[str, Any]] = []
    fitted: dict[str, _FittedTorchSequenceEstimator] = {}
    candidate_predictions: dict[
        str, tuple[pd.Series, pd.Series, float, float, float, float]
    ] = {}
    train_target = train.labels[spec.return_label]
    valid_target = valid.labels[spec.return_label]

    if (
        missing_required_source
        or not train_quality.all()
        or not valid_quality.all()
        or not all(
            evidence["labels_complete"]
            for evidence in label_completeness_evidence.values()
        )
    ):
        backend_candidates: Sequence[tuple[str, int]] = ()
    else:
        backend_candidates = tuple(
            (family, window) for family in spec.model_families for window in spec.windows
        )
    for family, window in backend_candidates:
        candidate = f"{family}_w{window}"
        try:
            train_batch = build_sequence_windows(
                train_x,
                train_target,
                window=window,
                endpoint_index=train_endpoints,
            )
            valid_combined_x = pd.concat([train_x.tail(window - 1), valid_x])
            valid_combined_y = pd.concat([train_target.tail(window - 1), valid_target])
            valid_batch = build_sequence_windows(
                valid_combined_x,
                valid_combined_y,
                window=window,
                endpoint_index=valid_endpoints,
            )
            estimator = _fit_torch_candidate(
                train_batch, valid_batch, family=family, spec=spec
            )
            train_raw = pd.Series(0.0, index=train.features.index)
            train_values = estimator.predict(train_batch.values)
            if len(train_values) != len(train_batch.endpoints) or not np.isfinite(
                train_values
            ).all():
                raise ValueError("sequence candidate produced invalid train predictions")
            train_raw.loc[train_batch.endpoints] = train_values
            valid_all_batch = build_sequence_windows(
                valid_combined_x,
                window=window,
                endpoint_index=valid.features.index,
            )
            valid_raw = pd.Series(0.0, index=valid.features.index)
            valid_values = estimator.predict(valid_all_batch.values)
            if len(valid_values) != len(valid_all_batch.endpoints) or not np.isfinite(
                valid_values
            ).all():
                raise ValueError("sequence candidate produced invalid validation predictions")
            valid_raw.loc[valid_all_batch.endpoints] = valid_values
            fitted_train_values = train_raw.loc[train_batch.endpoints]
            mu_center = float(fitted_train_values.mean())
            mu_scale = max(float(fitted_train_values.std(ddof=0)), 1e-6)
            best_policy: tuple[float, float, float] | None = None
            for floor, cap in spec.floor_cap_candidates:
                validation_quality = _sequence_quality(
                    train_quality.tail(window - 1), valid_quality, window=window
                )
                predictions = _scores_to_predictions(
                    valid_raw,
                    validation_quality,
                    mu_center=mu_center,
                    mu_scale=mu_scale,
                    floor=float(floor),
                    cap=float(cap),
                )
                utility = _validation_utility(
                    valid_target.loc[valid_endpoints],
                    predictions.loc[valid_endpoints, "target_position"],
                )
                choice = (utility, -float(floor), -float(cap))
                if best_policy is None or choice > (
                    best_policy[0], -best_policy[1], -best_policy[2]
                ):
                    best_policy = (utility, float(floor), float(cap))
            assert best_policy is not None
            utility, floor, cap = best_policy
            candidate_predictions[candidate] = (
                train_raw,
                valid_raw,
                mu_center,
                mu_scale,
                floor,
                cap,
            )
            fitted[candidate] = estimator
            rows.append(
                {
                    "candidate": candidate,
                    "model": candidate,
                    "family": family,
                    "window": window,
                    "status": "ok",
                    "validation_score": utility,
                    "validation_loss": estimator.validation_loss,
                    "best_epoch": estimator.best_epoch,
                    "floor": floor,
                    "cap": cap,
                    "selected": False,
                    "reason_code": "",
                    "failure_type": "",
                    "detail": "",
                }
            )
        except Exception as exc:  # candidate-local failure is retained
            rows.append(_failure_row(candidate, exc))

    pre_quality_evidence = {
        spec.train_segment: _quality_evidence(train_quality),
        spec.selection_segment: _quality_evidence(valid_quality),
    }
    policy: FittedPanelSequencePolicy | None = None
    selected_row: dict[str, Any] | None = None
    estimator: _FittedTorchSequenceEstimator | None = None
    train_predictions = _zero_predictions(train.features.index)
    valid_predictions = _zero_predictions(valid.features.index)
    floor = float(spec.floor_cap_candidates[0][0])
    cap = float(spec.floor_cap_candidates[0][1])
    if fitted:
        successful_rows = [row for row in rows if row["status"] == "ok"]
        selected_row = max(
            successful_rows,
            key=lambda row: (
                float(row["validation_score"]),
                -int(row["window"]),
                row["family"],
            ),
        )
        selected_candidate = str(selected_row["candidate"])
        selected_row["selected"] = True
        estimator = fitted[selected_candidate]
        train_raw, valid_raw, mu_center, mu_scale, floor, cap = candidate_predictions[
            selected_candidate
        ]
        policy = FittedPanelSequencePolicy(
            feature_columns=tuple(train.features.columns),
            preprocessing_median=median,
            preprocessing_mean=mean,
            preprocessing_scale=scale,
            estimator=estimator,
            family=str(selected_row["family"]),
            window=int(selected_row["window"]),
            mu_center=mu_center,
            mu_scale=mu_scale,
            floor=floor,
            cap=cap,
            history_tail=pd.concat([train.features, valid.features]).tail(
                int(selected_row["window"]) - 1
            ),
        )
        train_predictions = _scores_to_predictions(
            train_raw,
            _sequence_quality(pd.Series(dtype=bool), train_quality, window=policy.window),
            mu_center=mu_center,
            mu_scale=mu_scale,
            floor=floor,
            cap=cap,
        )
        valid_predictions = _scores_to_predictions(
            valid_raw,
            _sequence_quality(
                train_quality.tail(policy.window - 1), valid_quality, window=policy.window
            ),
            mu_center=mu_center,
            mu_scale=mu_scale,
            floor=floor,
            cap=cap,
        )
        frozen_reason = selected_candidate
    elif missing_required_source:
        frozen_reason = "blocked_missing_required_source"
    elif any(item["invalid_count"] for item in pre_quality_evidence.values()) or not all(
        item["labels_complete"] for item in label_completeness_evidence.values()
    ):
        frozen_reason = "blocked_data_quality"
    elif rows and all(row["reason_code"] == "backend_unavailable" for row in rows):
        frozen_reason = "blocked_model_backend_unavailable"
    else:
        frozen_reason = "blocked_no_valid_sequence_candidate"
    event_trace: list[str] = []
    frozen_decision = _freeze_route_decision(
        frozen_reason,
        policy,
        event_trace=event_trace,
    )
    test = _prepare_inference_segment(
        dataset,
        spec.test_segment,
        frozen_decision=frozen_decision,
        event_trace=event_trace,
    )
    _require_ordered_non_overlapping_segments(
        valid,
        test,
        left_name=spec.selection_segment,
        right_name=spec.test_segment,
    )
    test_aligned = test.features.reindex(columns=train.features.columns)
    test_intrinsic_quality = pd.Series(
        np.isfinite(test_aligned.to_numpy(dtype="float64")).all(axis=1),
        index=test.features.index,
    )
    test_quality = _combine_quality(
        test_intrinsic_quality, external_quality.get(spec.test_segment)
    )
    source_evidence = _diagnostic_source_evidence(
        {
            spec.train_segment: train.features,
            spec.selection_segment: valid.features,
            spec.test_segment: test.features,
        }
    )
    missing_required_source = not all(
        item["available"] for item in source_evidence.values()
    )
    quality_evidence = {
        **pre_quality_evidence,
        spec.test_segment: _quality_evidence(test_quality),
    }
    pit_mode, pit_row = _pit_evidence(
        pit_panel,
        qlib_indices=(train.features.index, valid.features.index, test.features.index),
        qlib_lineage=qlib_lineage,
    )
    test_accessed_after_freeze = (
        event_trace.index("candidate_selection_frozen")
        < event_trace.index("test_dk_i_accessed")
    )
    if missing_required_source or policy is None:
        reason = (
            "blocked_missing_required_source"
            if missing_required_source
            else frozen_decision.selected_candidate
        )
        if not rows:
            for family in spec.model_families:
                for window in spec.windows:
                    row = _failure_row(
                        f"{family}_w{window}",
                        ValueError(reason),
                    )
                    row["reason_code"] = (
                        "blocked_missing_required_source"
                        if reason == "blocked_missing_required_source"
                        else "invalid_required_data"
                        if reason == "blocked_data_quality"
                        else row["reason_code"]
                    )
                    rows.append(row)
        if policy is not None:
            selected_row["selected"] = False
            rows.append(
                {
                    **_failure_row("diagnostic_source_gate_w0", ValueError(reason)),
                    "candidate": "diagnostic_source_gate",
                    "model": "diagnostic_source_gate",
                    "family": "source_gate",
                    "window": np.nan,
                    "reason_code": reason,
                }
            )
        predictions = {
            spec.train_segment: _zero_predictions(train.features.index),
            spec.selection_segment: _zero_predictions(valid.features.index),
            spec.test_segment: _zero_predictions(test.features.index),
        }
        failure_rows = pd.DataFrame([row for row in rows if row["status"] != "ok"])
        comparison = pd.DataFrame([*rows, pit_row])
        return PanelSequenceResult(
            model_comparison=comparison,
            selected_candidate=reason,
            predictions_by_segment=predictions,
            positions_by_segment={
                name: frame["target_position"].copy() for name, frame in predictions.items()
            },
            failure_rows=failure_rows,
            selection_evidence={
                "route_status": reason,
                "selection_segment": spec.selection_segment,
                "test_labels_used_for_selection": False,
                "test_accessed_after_freeze": test_accessed_after_freeze,
                "event_trace": tuple(event_trace),
                "label_horizon": spec.label_horizon,
                "purge_rows": purge_rows,
                "random_state": spec.random_state,
                "source_evidence": source_evidence,
                "data_quality_evidence": quality_evidence,
                "label_completeness_evidence": label_completeness_evidence,
            },
            mode_evidence={
                "diagnostic_multivariate": {
                    "status": reason,
                    "source_evidence": source_evidence,
                },
                "constituent_cross_sectional": pit_mode,
            },
            blocked_mode_positions={
                "constituent_cross_sectional": pd.Series(0.0, index=test.features.index)
            },
            floor=floor,
            cap=cap,
            fitted_policy=None,
        )

    test_prediction_failure: dict[str, Any] | None = None
    try:
        test_predictions = policy.predict(
            test.features, data_quality_valid=test_quality, segment=spec.test_segment
        )
    except Exception as exc:
        test_predictions = _zero_predictions(test.features.index)
        test_prediction_failure = {
            "reason_code": "prediction_failed",
            "first_offending_date": _index_value_to_string(test.features.index[0]),
            "failure_type": type(exc).__name__,
            "detail": str(exc),
        }
    data_quality_blocked = any(item["invalid_count"] for item in quality_evidence.values())
    data_quality_blocked = data_quality_blocked or not all(
        item["labels_complete"] for item in label_completeness_evidence.values()
    )
    failure_rows = pd.DataFrame([row for row in rows if row["status"] != "ok"])
    degraded = bool(failure_rows.shape[0])
    route_status = (
        "blocked_prediction_failed"
        if test_prediction_failure is not None
        else "blocked_data_quality"
        if data_quality_blocked
        else "ok_degraded_candidate_failure" if degraded else "ok"
    )
    comparison = pd.DataFrame([*rows, pit_row])
    predictions = {
        spec.train_segment: train_predictions,
        spec.selection_segment: valid_predictions,
        spec.test_segment: test_predictions,
    }
    assert selected_row is not None and estimator is not None
    frozen_policy = {
        "candidate": selected_candidate,
        "family": policy.family,
        "window": policy.window,
        "floor": policy.floor,
        "cap": policy.cap,
        "best_epoch": estimator.best_epoch,
        "validation_loss": estimator.validation_loss,
    }
    return PanelSequenceResult(
        model_comparison=comparison,
        selected_candidate=selected_candidate,
        predictions_by_segment=predictions,
        positions_by_segment={
            name: frame["target_position"].copy() for name, frame in predictions.items()
        },
        failure_rows=failure_rows,
        selection_evidence={
            "route_status": route_status,
            "selection_segment": spec.selection_segment,
            "test_labels_used_for_selection": False,
            "test_accessed_after_freeze": test_accessed_after_freeze,
            "event_trace": tuple(event_trace),
            "label_horizon": spec.label_horizon,
            "purge_rows": purge_rows,
            "random_state": spec.random_state,
            "selected_family": policy.family,
            "selected_window": policy.window,
            "selected_validation_utility": float(selected_row["validation_score"]),
            "early_stopping_segment": spec.selection_segment,
            "network_contract": {
                "hidden_width": spec.hidden_width,
                "temporal_layers": spec.temporal_layers,
                "dropout": spec.dropout,
            },
            "diagnostic_assets": tuple(
                source for source, item in source_evidence.items() if item["available"]
            ),
            "data_keys": {
                spec.train_segment: "DataHandlerLP.DK_L",
                spec.selection_segment: "DataHandlerLP.DK_L",
                spec.test_segment: "DataHandlerLP.DK_I",
            },
            "data_quality_evidence": quality_evidence,
            "label_completeness_evidence": label_completeness_evidence,
            "source_evidence": source_evidence,
            "test_prediction_failure": test_prediction_failure,
            "frozen_policy": frozen_policy,
        },
        mode_evidence={
            "diagnostic_multivariate": {
                "status": route_status,
                "assets": tuple(
                    source for source, item in source_evidence.items() if item["available"]
                ),
                "source_evidence": source_evidence,
            },
            "constituent_cross_sectional": pit_mode,
        },
        blocked_mode_positions={
            "constituent_cross_sectional": pd.Series(0.0, index=test.features.index)
        },
        floor=floor,
        cap=cap,
        fitted_policy=policy,
    )
