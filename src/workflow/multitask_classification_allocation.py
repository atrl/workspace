from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from torch import nn

from src.workflow.direct_continuous_soxx_allocation import (
    DEVELOPMENT_YEARS,
    DIRECT_FEATURES,
    OUTER_START,
    DirectContinuousAllocationSpec,
    _annual_metrics,
    _evaluate,
    _load_frame,
    _prepare,
    _regime_metrics,
)
from src.workflow.soxl_objective_feasibility_audit import _json_ready
from src.workflow.trainable_regime_policy_nested_wf import (
    _remove_qlib_code_snapshot_artifacts,
)


EXPERIMENT = "multitask_classification_allocation_v1"
BASE_FEATURES = tuple(DIRECT_FEATURES)
DELTA_LAGS = (1, 3, 5)
LABEL_COLUMNS = (
    "tail_1d_label",
    "tail_3d_label",
    "tail_5d_label",
    "relative_alpha_label",
    "recovery_label",
)
TARGET_WEIGHT_COLUMNS = (
    "target_soxl_weight",
    "target_soxx_weight",
    "target_cash_weight",
)
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/predictions.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/training_diagnostics.csv",
)


@dataclass(frozen=True)
class AllocationCandidate:
    name: str
    encoder: str
    feature_set: str
    multitask: bool
    sequence_length: int = 1


CANDIDATES = (
    AllocationCandidate(
        name="tabular_base_allocation_only",
        encoder="tabular",
        feature_set="base",
        multitask=False,
    ),
    AllocationCandidate(
        name="tabular_base_multitask",
        encoder="tabular",
        feature_set="base",
        multitask=True,
    ),
    AllocationCandidate(
        name="tabular_fast_multitask",
        encoder="tabular",
        feature_set="fast",
        multitask=True,
    ),
    AllocationCandidate(
        name="gru_fast_multitask",
        encoder="gru",
        feature_set="fast",
        multitask=True,
        sequence_length=20,
    ),
)


@dataclass(frozen=True)
class MultitaskClassificationAllocationSpec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    development_years: tuple[int, ...] = DEVELOPMENT_YEARS
    development_end: str = "2022-12-30"
    outer: tuple[str, str] = (OUTER_START, "2026-12-31")
    purge_rows: int = 20
    hidden_units: int = 24
    epochs: int = 350
    learning_rate: float = 0.003
    weight_decay: float = 0.03
    maximum_soxl_weight: float = 0.30
    costs_bps: tuple[int, ...] = (10, 25)
    seed: int = 29

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def daily_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"


class MultitaskAllocationNet(nn.Module):
    """Shared encoder with auxiliary classification and direct weight heads."""

    def __init__(
        self,
        *,
        feature_count: int,
        hidden_units: int,
        encoder: str,
    ):
        super().__init__()
        self.encoder_name = encoder
        if encoder == "tabular":
            self.encoder = nn.Sequential(
                nn.Linear(feature_count, hidden_units),
                nn.Tanh(),
                nn.Linear(hidden_units, hidden_units),
                nn.Tanh(),
            )
        elif encoder == "gru":
            self.encoder = nn.GRU(
                input_size=feature_count,
                hidden_size=hidden_units,
                batch_first=True,
            )
        else:
            raise ValueError(f"unsupported encoder: {encoder}")
        self.classification_heads = nn.ModuleDict(
            {label: nn.Linear(hidden_units, 1) for label in LABEL_COLUMNS}
        )
        self.allocation_head = nn.Linear(hidden_units, 2)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if self.encoder_name == "gru":
            encoded, _ = self.encoder(features)
            state = encoded[:, -1, :]
        else:
            state = self.encoder(features)
        logits = {
            label: head(state).squeeze(1)
            for label, head in self.classification_heads.items()
        }
        return logits, self.allocation_head(state)


def run_multitask_classification_allocation_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    return _run_card(
        MultitaskClassificationAllocationSpec(root=repo_root)
    )


def _future_path(
    returns: pd.Series, horizon: int
) -> tuple[pd.Series, pd.Series]:
    forward = pd.concat(
        [returns.shift(-offset) for offset in range(horizon)], axis=1
    )
    path = (1.0 + forward).cumprod(axis=1) - 1.0
    terminal = path.iloc[:, -1].copy()
    adverse = path.min(axis=1).copy()
    incomplete = forward.isna().any(axis=1)
    terminal.loc[incomplete] = np.nan
    adverse.loc[incomplete] = np.nan
    return terminal, adverse


def _attach_multitask_targets(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    result = frame.copy()
    path_columns: dict[str, pd.Series] = {}
    for asset in ("soxl", "soxx"):
        returns = pd.to_numeric(
            result[f"{asset}_forward_open_return"], errors="coerce"
        )
        for horizon in (1, 3, 5, 20):
            terminal, adverse = _future_path(returns, horizon)
            path_columns[f"{asset}_future_return_{horizon}d"] = terminal
            path_columns[f"{asset}_future_adverse_{horizon}d"] = adverse
    result = result.join(pd.DataFrame(path_columns, index=result.index))

    result["tail_1d_label"] = result[
        "soxl_future_adverse_1d"
    ].le(-0.08).astype(float)
    result["tail_3d_label"] = result[
        "soxx_future_adverse_3d"
    ].le(-0.04).astype(float)
    result["tail_5d_label"] = (
        result["soxl_future_adverse_5d"].le(-0.12)
        | result["soxx_future_adverse_5d"].le(-0.06)
    ).astype(float)
    utility_soxl = (
        0.60 * result["soxl_future_return_5d"]
        + 0.40 * result["soxl_future_return_20d"]
        + 0.45 * result["soxl_future_adverse_5d"]
        + 0.20 * result["soxl_future_adverse_20d"]
    )
    utility_soxx = (
        0.60 * result["soxx_future_return_5d"]
        + 0.40 * result["soxx_future_return_20d"]
        + 0.45 * result["soxx_future_adverse_5d"]
        + 0.20 * result["soxx_future_adverse_20d"]
    )
    result["relative_alpha_label"] = (
        utility_soxl.sub(utility_soxx).gt(0.02)
        & result["tail_5d_label"].eq(0.0)
    ).astype(float)
    result["stress_context"] = (
        pd.to_numeric(result["soxx_drawdown60"], errors="coerce").le(-0.10)
        | pd.to_numeric(result["soxx_vol20"], errors="coerce").ge(0.30)
        | pd.to_numeric(result["vix_z_60"], errors="coerce").ge(1.0)
    ).astype(float)
    result["recovery_label"] = (
        result["stress_context"].eq(1.0)
        & result["soxx_future_return_5d"].ge(0.04)
        & result["soxx_future_adverse_5d"].ge(-0.04)
    ).astype(float)

    adverse_severity = pd.concat(
        [
            (
                -result["soxl_future_adverse_5d"] - 0.04
            ).div(0.18),
            (
                -result["soxx_future_adverse_5d"] - 0.02
            ).div(0.10),
            (
                -result["soxx_future_adverse_20d"] - 0.05
            ).div(0.20),
        ],
        axis=1,
    ).max(axis=1)
    cash_target = adverse_severity.clip(lower=0.0, upper=0.85)
    cash_target = cash_target.mul(
        np.where(result["recovery_label"].eq(1.0), 0.25, 1.0)
    )
    alpha_strength = 1.0 / (
        1.0
        + np.exp(
            -utility_soxl.sub(utility_soxx).clip(-0.20, 0.20).div(0.03)
        )
    )
    soxl_target = (
        (1.0 - cash_target)
        * 0.30
        * pd.Series(alpha_strength, index=result.index)
    ).clip(lower=0.0, upper=0.30)
    soxx_target = (1.0 - cash_target - soxl_target).clip(lower=0.0)
    total = soxl_target + soxx_target + cash_target
    result["target_soxl_weight"] = soxl_target.div(total)
    result["target_soxx_weight"] = soxx_target.div(total)
    result["target_cash_weight"] = cash_target.div(total)

    delta_frame = pd.concat(
        {
            f"{feature}_delta_{lag}d": pd.to_numeric(
                result[feature], errors="coerce"
            ).sub(pd.to_numeric(result[feature], errors="coerce").shift(lag))
            for feature in BASE_FEATURES
            if feature in result
            for lag in DELTA_LAGS
        },
        axis=1,
    )
    result = result.join(delta_frame)
    base_features = [feature for feature in BASE_FEATURES if feature in result]
    fast_features = [
        *base_features,
        *delta_frame.columns.astype(str).tolist(),
    ]

    required_future = [
        "soxl_future_return_20d",
        "soxx_future_return_20d",
        "soxl_future_adverse_20d",
        "soxx_future_adverse_20d",
    ]
    incomplete = result.loc[:, required_future].isna().any(axis=1)
    result.loc[
        incomplete,
        [*LABEL_COLUMNS, *TARGET_WEIGHT_COLUMNS],
    ] = np.nan
    return result, base_features, fast_features


def _build_dataset(
    frame: pd.DataFrame,
    all_features: list[str],
    spec: MultitaskClassificationAllocationSpec,
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    labels = [
        "soxl_forward_open_return",
        "soxx_forward_open_return",
        "stress_context",
        *LABEL_COLUMNS,
        *TARGET_WEIGHT_COLUMNS,
    ]
    static = pd.concat(
        {
            "feature": frame.loc[:, all_features],
            "label": frame.loc[:, labels],
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [frame.index, np.full(len(frame), "SOXL_SOXX_CASH")],
        names=["datetime", "instrument"],
    )
    outer_end = min(
        spec.outer[1], frame.index.max().date().isoformat()
    )
    segments: dict[str, tuple[str, str]] = {
        "development": (
            f"{spec.development_years[0]}-01-01",
            spec.development_end,
        ),
        "outer": (spec.outer[0], outer_end),
    }
    for year in spec.development_years:
        segments[f"fold_{year}"] = (
            f"{year}-01-01",
            f"{year}-12-31",
        )
    handler = DataHandlerLP(
        instruments=None,
        start_time=frame.index.min(),
        end_time=frame.index.max(),
        data_loader=StaticDataLoader(config=static),
    )
    dataset = DatasetH(handler=handler, segments=segments)
    audit = {
        "api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"],
        "segments": {name: list(bounds) for name, bounds in segments.items()},
        "segment_rows": {
            name: int(len(dataset.prepare(name))) for name in segments
        },
        "purge_rows": spec.purge_rows,
        "labels": list(LABEL_COLUMNS),
        "continuous_target": list(TARGET_WEIGHT_COLUMNS),
        "selection_uses_outer_test": False,
    }
    return dataset, audit


def _factorized_weights(
    allocation_logits: torch.Tensor, maximum_soxl_weight: float
) -> torch.Tensor:
    soxl = maximum_soxl_weight * torch.sigmoid(
        allocation_logits[:, 0]
    )
    cash = (1.0 - soxl) * torch.sigmoid(allocation_logits[:, 1])
    soxx = 1.0 - soxl - cash
    return torch.stack([soxl, soxx, cash], dim=1)


def _standardizer(
    train: pd.DataFrame, features: list[str]
) -> dict[str, pd.Series]:
    median = train.loc[:, features].median()
    values = train.loc[:, features].fillna(median)
    mean = values.mean()
    scale = values.std(ddof=1).replace(0.0, 1.0).fillna(1.0)
    return {"median": median, "mean": mean, "scale": scale}


def _scale_frame(
    frame: pd.DataFrame,
    features: list[str],
    scaler: dict[str, pd.Series],
) -> pd.DataFrame:
    values = frame.loc[:, features].fillna(scaler["median"])
    return (
        (values - scaler["mean"]) / scaler["scale"]
    ).clip(-8.0, 8.0)


def _model_input(
    *,
    scaled_history: pd.DataFrame,
    target_index: pd.Index,
    candidate: AllocationCandidate,
) -> torch.Tensor:
    if candidate.encoder == "tabular":
        values = scaled_history.loc[target_index].to_numpy(dtype=np.float32)
        return torch.tensor(values, dtype=torch.float32)
    positions = scaled_history.index.get_indexer(target_index)
    sequences: list[np.ndarray] = []
    for position in positions:
        if position < 0:
            raise ValueError("blocked: sequence target missing from history")
        start = max(0, position - candidate.sequence_length + 1)
        values = scaled_history.iloc[start : position + 1].to_numpy(
            dtype=np.float32
        )
        if len(values) < candidate.sequence_length:
            padding = np.repeat(
                values[[0]],
                candidate.sequence_length - len(values),
                axis=0,
            )
            values = np.vstack([padding, values])
        sequences.append(values)
    return torch.tensor(np.stack(sequences), dtype=torch.float32)


def _balanced_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        logits = logits[mask]
        targets = targets[mask]
    if not len(targets):
        return logits.sum() * 0.0
    positive = targets.sum()
    negative = len(targets) - positive
    pos_weight = torch.clamp(
        negative / torch.clamp(positive, min=1.0),
        min=1.0,
        max=10.0,
    )
    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight
    )


def _training_loss(
    *,
    logits: dict[str, torch.Tensor],
    weights: torch.Tensor,
    labels: dict[str, torch.Tensor],
    target_weights: torch.Tensor,
    stress_context: torch.Tensor,
    candidate: AllocationCandidate,
) -> tuple[torch.Tensor, dict[str, float]]:
    allocation_loss = nn.functional.mse_loss(weights, target_weights)
    turnover_loss = 0.5 * torch.abs(
        weights[1:] - weights[:-1]
    ).sum(dim=1).mean()
    classification_loss = weights.sum() * 0.0
    component_losses: dict[str, torch.Tensor] = {}
    if candidate.multitask:
        for label in ("tail_1d_label", "tail_3d_label", "tail_5d_label"):
            component_losses[label] = _balanced_bce(
                logits[label], labels[label]
            )
        alpha_mask = labels["tail_5d_label"].eq(0.0)
        component_losses["relative_alpha_label"] = _balanced_bce(
            logits["relative_alpha_label"],
            labels["relative_alpha_label"],
            alpha_mask,
        )
        recovery_mask = stress_context.eq(1.0)
        component_losses["recovery_label"] = _balanced_bce(
            logits["recovery_label"],
            labels["recovery_label"],
            recovery_mask,
        )
        classification_loss = (
            0.35 * component_losses["tail_1d_label"]
            + 0.35 * component_losses["tail_3d_label"]
            + 0.35 * component_losses["tail_5d_label"]
            + 0.25 * component_losses["relative_alpha_label"]
            + 0.25 * component_losses["recovery_label"]
        )
    loss = 4.0 * allocation_loss + classification_loss + 0.10 * turnover_loss
    diagnostics = {
        "loss": float(loss.detach()),
        "allocation_loss": float(allocation_loss.detach()),
        "classification_loss": float(classification_loss.detach()),
        "turnover_loss": float(turnover_loss.detach()),
        **{
            f"{label}_loss": float(value.detach())
            for label, value in component_losses.items()
        },
    }
    return loss, diagnostics


def _fit_predict(
    *,
    full_frame: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    candidate: AllocationCandidate,
    spec: MultitaskClassificationAllocationSpec,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    required = [
        "soxl_forward_open_return",
        "soxx_forward_open_return",
        "stress_context",
        *LABEL_COLUMNS,
        *TARGET_WEIGHT_COLUMNS,
    ]
    train = train.dropna(subset=required).copy()
    test = test.dropna(subset=required).copy()
    if train.empty or test.empty:
        raise ValueError("blocked: empty multitask train or test data")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = _standardizer(train, features)
    history_end = max(train.index.max(), test.index.max())
    history = full_frame.loc[:history_end, features].copy()
    scaled_history = _scale_frame(history, features, scaler)
    train_x = _model_input(
        scaled_history=scaled_history,
        target_index=train.index,
        candidate=candidate,
    )
    test_x = _model_input(
        scaled_history=scaled_history,
        target_index=test.index,
        candidate=candidate,
    )
    labels = {
        label: torch.tensor(
            train[label].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        for label in LABEL_COLUMNS
    }
    target_weights = torch.tensor(
        train.loc[:, TARGET_WEIGHT_COLUMNS].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    stress_context = torch.tensor(
        train["stress_context"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    model = MultitaskAllocationNet(
        feature_count=len(features),
        hidden_units=spec.hidden_units,
        encoder=candidate.encoder,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    diagnostics: dict[str, float] = {}
    model.train()
    for _ in range(spec.epochs):
        optimizer.zero_grad()
        logits, allocation_logits = model(train_x)
        weights = _factorized_weights(
            allocation_logits, spec.maximum_soxl_weight
        )
        loss, diagnostics = _training_loss(
            logits=logits,
            weights=weights,
            labels=labels,
            target_weights=target_weights,
            stress_context=stress_context,
            candidate=candidate,
        )
        if not torch.isfinite(loss):
            raise ValueError(
                f"blocked: non-finite loss for {candidate.name}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        test_logits, allocation_logits = model(test_x)
        test_weights = _factorized_weights(
            allocation_logits, spec.maximum_soxl_weight
        ).cpu().numpy()
        probabilities = {
            label: torch.sigmoid(value).cpu().numpy()
            for label, value in test_logits.items()
        }
    output = pd.DataFrame(
        test_weights,
        index=test.index,
        columns=["soxl_weight", "soxx_weight", "cash_weight"],
    )
    for label, values in probabilities.items():
        output[f"p_{label.removesuffix('_label')}"] = values
        output[label] = test[label]
    output["stress_context"] = test["stress_context"]
    state = {
        "model_state": {
            key: value.detach().cpu().numpy()
            for key, value in model.state_dict().items()
        },
        "scaler": {
            name: values.to_dict() for name, values in scaler.items()
        },
        "features": features,
        "candidate": asdict(candidate),
        "spec": {
            "hidden_units": spec.hidden_units,
            "epochs": spec.epochs,
            "learning_rate": spec.learning_rate,
            "weight_decay": spec.weight_decay,
            "maximum_soxl_weight": spec.maximum_soxl_weight,
        },
    }
    return output, state, diagnostics


def _safe_classification_metrics(
    actual: pd.Series, probability: pd.Series
) -> dict[str, float]:
    usable = pd.concat([actual, probability], axis=1).dropna()
    y = usable.iloc[:, 0].astype(int)
    p = usable.iloc[:, 1].astype(float).clip(0.0, 1.0)
    if y.nunique() < 2:
        return {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "brier": float("nan"),
            "positive_rate": float(y.mean()) if len(y) else float("nan"),
        }
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "positive_rate": float(y.mean()),
    }


def _classification_rows(
    predictions: pd.DataFrame,
    *,
    candidate: str,
    segment: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in predictions.groupby(predictions.index.year):
        for label in LABEL_COLUMNS:
            mask = pd.Series(True, index=group.index)
            if label == "relative_alpha_label":
                mask = group["tail_5d_label"].eq(0.0)
            elif label == "recovery_label":
                mask = group["stress_context"].eq(1.0)
            metrics = _safe_classification_metrics(
                group.loc[mask, label],
                group.loc[
                    mask, f"p_{label.removesuffix('_label')}"
                ],
            )
            rows.append(
                {
                    "candidate": candidate,
                    "segment": segment,
                    "year": int(year),
                    "task": label,
                    "selection_uses_outer_test": False,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _walk_forward_candidate(
    *,
    dataset,
    frame: pd.DataFrame,
    base_features: list[str],
    fast_features: list[str],
    candidate: AllocationCandidate,
    spec: MultitaskClassificationAllocationSpec,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    features = (
        base_features if candidate.feature_set == "base" else fast_features
    )
    predictions: list[pd.DataFrame] = []
    states: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for year in spec.development_years:
        test = _prepare(dataset, f"fold_{year}")
        test = test.loc[test.index.year == year]
        start_position = frame.index.searchsorted(
            pd.Timestamp(f"{year}-01-01")
        )
        train_end_position = start_position - spec.purge_rows - 1
        if train_end_position < 0:
            raise ValueError(
                f"blocked: insufficient history for fold {year}"
            )
        train_end = frame.index[train_end_position]
        train = _prepare(dataset, (spec.train_start, train_end))
        output, state, fit_diagnostics = _fit_predict(
            full_frame=frame,
            train=train,
            test=test,
            features=features,
            candidate=candidate,
            spec=spec,
            seed=spec.seed + year,
        )
        output["candidate"] = candidate.name
        output["segment"] = "development_walk_forward"
        output["fold_year"] = year
        output["train_end"] = train_end
        predictions.append(output)
        states[f"fold_{year}"] = state
        diagnostics.append(
            {
                "candidate": candidate.name,
                "fold_year": year,
                "train_end": train_end.date().isoformat(),
                "train_rows": int(len(train)),
                "test_rows": int(len(output)),
                "feature_count": len(features),
                "selection_uses_outer_test": False,
                **fit_diagnostics,
            }
        )
    combined = pd.concat(predictions).sort_index()
    classification = _classification_rows(
        combined,
        candidate=candidate.name,
        segment="development_walk_forward",
    )
    return (
        combined,
        states,
        pd.DataFrame(diagnostics),
        classification,
    )


def _outer_candidate(
    *,
    dataset,
    frame: pd.DataFrame,
    base_features: list[str],
    fast_features: list[str],
    candidate: AllocationCandidate,
    spec: MultitaskClassificationAllocationSpec,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame]:
    features = (
        base_features if candidate.feature_set == "base" else fast_features
    )
    train = _prepare(dataset, (spec.train_start, spec.development_end))
    test = _prepare(dataset, "outer")
    output, state, diagnostics = _fit_predict(
        full_frame=frame,
        train=train,
        test=test,
        features=features,
        candidate=candidate,
        spec=spec,
        seed=spec.seed,
    )
    output["candidate"] = candidate.name
    output["segment"] = "outer_diagnostic"
    output["fold_year"] = output.index.year
    output["train_end"] = pd.Timestamp(spec.development_end)
    classification = _classification_rows(
        output,
        candidate=candidate.name,
        segment="outer_diagnostic",
    )
    return output, state, diagnostics, classification


def _classification_summary(
    rows: pd.DataFrame, segment: str
) -> pd.DataFrame:
    subset = rows.loc[rows["segment"].eq(segment)]
    summary = (
        subset.groupby(["candidate", "task"], as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_brier=("brier", "mean"),
            mean_positive_rate=("positive_rate", "mean"),
            years_auc_above_055=(
                "roc_auc",
                lambda values: int(pd.Series(values).gt(0.55).sum()),
            ),
        )
    )
    return summary


def _candidate_backtests(
    *,
    frame: pd.DataFrame,
    prediction_map: dict[str, pd.DataFrame],
    segment: str,
    spec: MultitaskClassificationAllocationSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons: list[dict[str, Any]] = []
    annual_frames: list[pd.DataFrame] = []
    stored_returns: list[pd.DataFrame] = []
    stored_positions: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        predictions = prediction_map[candidate.name]
        for cost_bps in spec.costs_bps:
            metrics, result, benchmark = _evaluate(
                market=frame,
                predictions=predictions,
                profile=candidate,
                cost_bps=cost_bps,
            )
            comparisons.append(
                {
                    "segment": segment,
                    "candidate": candidate.name,
                    "cost_bps": cost_bps,
                    "selection_uses_outer_test": False,
                    **metrics,
                }
            )
            annual_frames.append(
                _annual_metrics(
                    result,
                    benchmark,
                    {
                        "segment": segment,
                        "candidate": candidate.name,
                        "cost_bps": cost_bps,
                        "selection_uses_outer_test": False,
                    },
                )
            )
            stored_returns.append(
                result.assign(
                    segment=segment,
                    candidate=candidate.name,
                    cost_bps=cost_bps,
                )
            )
            stored_positions.append(
                result.loc[
                    :,
                    [
                        "signal_date",
                        "trade_date",
                        "exit_date",
                        "soxl_weight",
                        "soxx_weight",
                        "cash_weight",
                        "turnover",
                    ],
                ].assign(
                    segment=segment,
                    candidate=candidate.name,
                    cost_bps=cost_bps,
                )
            )
    return (
        pd.DataFrame(comparisons),
        pd.concat(annual_frames, ignore_index=True),
        pd.concat(stored_returns, ignore_index=True),
        pd.concat(stored_positions, ignore_index=True),
    )


def _development_selection(
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    classification_summary: pd.DataFrame,
) -> pd.DataFrame:
    selection = comparison.loc[comparison["cost_bps"].eq(10)].copy()
    stress = comparison.loc[
        comparison["cost_bps"].eq(25),
        ["candidate", "terminal_wealth_ratio_vs_soxx"],
    ].rename(
        columns={
            "terminal_wealth_ratio_vs_soxx": "ratio_25bps",
        }
    )
    selection = selection.rename(
        columns={
            "terminal_wealth_ratio_vs_soxx": "ratio_10bps",
        }
    ).merge(stress, on="candidate", how="left")
    annual10 = annual.loc[annual["cost_bps"].eq(10)]
    annual_summary = (
        annual10.groupby("candidate")["terminal_wealth_ratio_vs_soxx"]
        .agg(
            positive_relative_years=lambda values: int(values.gt(1.0).sum()),
            median_annual_ratio="median",
            minimum_annual_ratio="min",
        )
        .reset_index()
    )
    selection = selection.merge(annual_summary, on="candidate", how="left")
    pivot = classification_summary.pivot(
        index="candidate", columns="task", values="mean_roc_auc"
    ).add_suffix("_auc")
    selection = selection.merge(
        pivot.reset_index(), on="candidate", how="left"
    )
    selection["return_gate"] = selection["ratio_10bps"].gt(1.05)
    selection["cost_gate"] = selection["ratio_25bps"].gt(1.0)
    selection["year_gate"] = selection["positive_relative_years"].ge(5)
    selection["maxdd_gate"] = selection["max_drawdown_pct"].ge(-20.0)
    selection["worst_day_gate"] = selection["worst_day_pct"].ge(-12.0)
    selection["cvar_gate"] = selection["cvar5_pct"].ge(-8.0)
    selection["worst_5d_gate"] = selection[
        "worst_rolling_5d_pct"
    ].ge(-12.0)
    selection["cash_gate"] = (
        selection["avg_cash_weight_pct"].ge(15.0)
        & selection["downside_avg_cash_weight_pct"].gt(
            selection["avg_cash_weight_pct"]
        )
    )
    selection["activity_gate"] = selection[
        "avg_equity_position_pct"
    ].ge(15.0)
    selection["learnability_gate"] = (
        selection["tail_1d_label_auc"].ge(0.55)
        & selection["relative_alpha_label_auc"].ge(0.55)
        & selection["recovery_label_auc"].ge(0.60)
    )
    gates = [
        "return_gate",
        "cost_gate",
        "year_gate",
        "maxdd_gate",
        "worst_day_gate",
        "cvar_gate",
        "worst_5d_gate",
        "cash_gate",
        "activity_gate",
        "learnability_gate",
    ]
    selection["gate_count"] = selection[gates].sum(axis=1)
    selection["all_gate"] = selection[gates].all(axis=1)
    selection["selection_score"] = (
        np.log(selection["ratio_10bps"].clip(lower=1e-12))
        + 0.20 * np.log(selection["ratio_25bps"].clip(lower=1e-12))
        + 0.03
        * (
            selection["max_drawdown_pct"]
            - selection["soxx_max_drawdown_pct"]
        )
        + 0.25
        * (
            selection["tail_1d_label_auc"].fillna(0.5)
            + selection["relative_alpha_label_auc"].fillna(0.5)
            - 1.0
        )
    )
    passing = selection.loc[selection["all_gate"]]
    pool = passing if not passing.empty else selection
    selected_index = pool.sort_values(
        ["gate_count", "selection_score", "candidate"],
        ascending=[False, False, True],
    ).index[0]
    selection["diagnostic_selected"] = selection.index == selected_index
    selection["selection_policy"] = np.where(
        selection["all_gate"],
        "all_required_development_gates_then_score",
        "diagnostic_only_maximum_gate_count_then_score",
    )
    selection["selection_uses_outer_test"] = False
    return selection


def _objective_audit(
    selected: pd.Series, outer: pd.Series
) -> pd.DataFrame:
    rows = [
        ("no_base_or_prior", True, True, "none"),
        ("no_candidate_weight_grid", True, True, "none"),
        (
            "outer_not_used_for_selection",
            True,
            not bool(selected["selection_uses_outer_test"]),
            "development_2016_2022_only",
        ),
    ]
    for gate in (
        "return_gate",
        "cost_gate",
        "year_gate",
        "maxdd_gate",
        "worst_day_gate",
        "cvar_gate",
        "worst_5d_gate",
        "cash_gate",
        "activity_gate",
        "learnability_gate",
    ):
        rows.append(
            (
                f"development_{gate}",
                True,
                bool(selected[gate]),
                selected.get(gate),
            )
        )
    rows.extend(
        [
            (
                "outer_ratio_above_soxx_report_only",
                False,
                bool(outer["terminal_wealth_ratio_vs_soxx"] > 1.0),
                outer["terminal_wealth_ratio_vs_soxx"],
            ),
            (
                "outer_maxdd_minus_20_report_only",
                False,
                bool(outer["max_drawdown_pct"] >= -20.0),
                outer["max_drawdown_pct"],
            ),
        ]
    )
    return pd.DataFrame(
        rows, columns=["gate", "required", "passed", "evidence"]
    )


def _run_card(
    spec: MultitaskClassificationAllocationSpec,
) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    source_frame, source_audit, _ = _load_frame(
        DirectContinuousAllocationSpec(root=spec.root)
    )
    frame, base_features, fast_features = _attach_multitask_targets(
        source_frame
    )
    all_features = list(dict.fromkeys([*base_features, *fast_features]))
    dataset, dataset_audit = _build_dataset(
        frame, all_features, spec
    )

    development_predictions: dict[str, pd.DataFrame] = {}
    outer_predictions: dict[str, pd.DataFrame] = {}
    model_states: dict[str, Any] = {}
    diagnostics: list[pd.DataFrame] = []
    classification_rows: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        (
            development,
            states,
            fit_diagnostics,
            classification,
        ) = _walk_forward_candidate(
            dataset=dataset,
            frame=frame,
            base_features=base_features,
            fast_features=fast_features,
            candidate=candidate,
            spec=spec,
        )
        outer, outer_state, outer_fit, outer_classification = (
            _outer_candidate(
                dataset=dataset,
                frame=frame,
                base_features=base_features,
                fast_features=fast_features,
                candidate=candidate,
                spec=spec,
            )
        )
        development_predictions[candidate.name] = development
        outer_predictions[candidate.name] = outer
        states["outer_refit"] = outer_state
        model_states[candidate.name] = states
        diagnostics.append(fit_diagnostics)
        diagnostics.append(
            pd.DataFrame(
                [
                    {
                        "candidate": candidate.name,
                        "fold_year": "outer_refit",
                        "train_end": spec.development_end,
                        "train_rows": int(
                            len(
                                _prepare(
                                    dataset,
                                    (
                                        spec.train_start,
                                        spec.development_end,
                                    ),
                                )
                            )
                        ),
                        "test_rows": int(len(outer)),
                        "feature_count": len(
                            base_features
                            if candidate.feature_set == "base"
                            else fast_features
                        ),
                        "selection_uses_outer_test": False,
                        **outer_fit,
                    }
                ]
            )
        )
        classification_rows.extend(
            [classification, outer_classification]
        )
    classification = pd.concat(classification_rows, ignore_index=True)
    development_classification = _classification_summary(
        classification, "development_walk_forward"
    )
    outer_classification = _classification_summary(
        classification, "outer_diagnostic"
    )
    (
        development_comparison,
        development_annual,
        development_returns,
        development_positions,
    ) = _candidate_backtests(
        frame=frame,
        prediction_map=development_predictions,
        segment="development_walk_forward",
        spec=spec,
    )
    outer_comparison, outer_annual, outer_returns, outer_positions = (
        _candidate_backtests(
            frame=frame,
            prediction_map=outer_predictions,
            segment="outer_diagnostic",
            spec=spec,
        )
    )
    selection = _development_selection(
        development_comparison,
        development_annual,
        development_classification,
    )
    selected = selection.loc[
        selection["diagnostic_selected"]
    ].iloc[0]
    selected_name = str(selected["candidate"])
    outer_selected = outer_comparison.loc[
        outer_comparison["candidate"].eq(selected_name)
        & outer_comparison["cost_bps"].eq(10)
    ].iloc[0]
    objective_audit = _objective_audit(selected, outer_selected)
    objective_pass = bool(
        objective_audit.loc[
            objective_audit["required"], "passed"
        ].all()
    )
    decision = (
        "validated_multitask_direct_allocation"
        if objective_pass
        else "not_promoted_multitask_classification_or_risk_gates_failed"
    )
    selected_outer_result = outer_returns.loc[
        outer_returns["candidate"].eq(selected_name)
        & outer_returns["cost_bps"].eq(10)
    ].copy()
    regimes = _regime_metrics(frame, selected_outer_result)
    comparison = pd.concat(
        [development_comparison, outer_comparison], ignore_index=True
    )
    annual = pd.concat(
        [development_annual, outer_annual], ignore_index=True
    )
    all_predictions = pd.concat(
        [
            *development_predictions.values(),
            *outer_predictions.values(),
        ]
    ).sort_index()
    strategy_returns = pd.concat(
        [development_returns, outer_returns], ignore_index=True
    )
    positions = pd.concat(
        [development_positions, outer_positions], ignore_index=True
    )
    training_diagnostics = pd.concat(diagnostics, ignore_index=True)

    from qlib.workflow import R

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(
        prefix=f"{spec.experiment}_"
    ) as temporary:
        temp = Path(temporary)
        all_predictions.reset_index(names="signal_date").to_csv(
            temp / "predictions.csv", index=False
        )
        positions.to_csv(temp / "positions.csv", index=False)
        strategy_returns.to_csv(
            temp / "strategy_returns.csv", index=False
        )
        training_diagnostics.to_csv(
            temp / "training_diagnostics.csv", index=False
        )
        models_dir = temp / "models"
        models_dir.mkdir()
        for candidate_name, states in model_states.items():
            with (
                models_dir / f"{candidate_name}.pkl"
            ).open("wb") as handle:
                pickle.dump(states, handle)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                benchmark="SOXX",
                assets="SOXL,SOXX,cash",
                candidate_count=len(CANDIDATES),
                allocation_output="direct_continuous_factorized",
                auxiliary_tasks="tail_1d,tail_3d,tail_5d,relative_alpha,recovery",
                base_constrained_ref_role="none",
                candidate_weight_grid="false",
                selection_uses_outer_test="false",
            )
            R.log_metrics(
                development_ratio_10bps=float(
                    selected["ratio_10bps"]
                ),
                development_max_drawdown_pct=float(
                    selected["max_drawdown_pct"]
                ),
                outer_ratio_10bps=float(
                    outer_selected["terminal_wealth_ratio_vs_soxx"]
                ),
                outer_max_drawdown_pct=float(
                    outer_selected["max_drawdown_pct"]
                ),
                objective_pass=float(objective_pass),
            )
            for artifact in temp.glob("*.csv"):
                recorder.log_artifact(
                    artifact.as_posix(), artifact_path=spec.experiment
                )
            for artifact in models_dir.glob("*.pkl"):
                recorder.log_artifact(
                    artifact.as_posix(),
                    artifact_path=f"{spec.experiment}/models",
                )
            recorder_id = recorder.id
            recorder_local_dir = Path(
                recorder.get_local_dir()
            ).resolve()
            summary = _summary(
                spec=spec,
                source_audit=source_audit,
                dataset_audit=dataset_audit,
                base_features=base_features,
                fast_features=fast_features,
                selection=selection,
                selected=selected,
                comparison=comparison,
                annual=annual,
                classification=classification,
                development_classification=development_classification,
                outer_classification=outer_classification,
                regimes=regimes,
                objective_audit=objective_audit,
                objective_pass=objective_pass,
                decision=decision,
                recorder_id=recorder_id,
                recorder_uri=recorder_uri,
                recorder_local_dir=recorder_local_dir,
            )
            _write_card(
                card=card,
                summary=summary,
                selection=selection,
                comparison=comparison,
                annual=annual,
                classification=classification,
                development_classification=development_classification,
                outer_classification=outer_classification,
                regimes=regimes,
                objective_audit=objective_audit,
                diagnostics=training_diagnostics,
                base_features=base_features,
                fast_features=fast_features,
            )
            for filename in summary["sidecars"].values():
                path = card / filename
                if path.exists():
                    recorder.log_artifact(
                        path.as_posix(), artifact_path=spec.experiment
                    )
            _remove_qlib_code_snapshot_artifacts(recorder_local_dir)
    return _json_ready(summary)


def _summary(
    *,
    spec: MultitaskClassificationAllocationSpec,
    source_audit: dict[str, Any],
    dataset_audit: dict[str, Any],
    base_features: list[str],
    fast_features: list[str],
    selection: pd.DataFrame,
    selected: pd.Series,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    classification: pd.DataFrame,
    development_classification: pd.DataFrame,
    outer_classification: pd.DataFrame,
    regimes: pd.DataFrame,
    objective_audit: pd.DataFrame,
    objective_pass: bool,
    decision: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
) -> dict[str, Any]:
    selected_name = str(selected["candidate"])
    outer10 = comparison.loc[
        comparison["segment"].eq("outer_diagnostic")
        & comparison["candidate"].eq(selected_name)
        & comparison["cost_bps"].eq(10)
    ].iloc[0]
    sidecars = {
        "selected_model": "selected_model.json",
        "model_selection": "model_selection.csv",
        "model_comparison": "model_comparison.csv",
        "classification_metrics": "classification_metrics.csv",
        "classification_summary": "classification_summary.csv",
        "backtest_benchmark_comparison": (
            "backtest_benchmark_comparison.csv"
        ),
        "backtest_walk_forward_metrics": (
            "backtest_walk_forward_metrics.csv"
        ),
        "risk_analysis": "risk_analysis.csv",
        "regime_metrics": "regime_metrics.csv",
        "training_summary": "training_summary.csv",
        "feature_manifest": "feature_manifest.csv",
        "objective_gate_audit": "objective_gate_audit.csv",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
        "conclusion": "conclusion.md",
        "workflow_config": "workflow.yaml",
    }
    return {
        "experiment": spec.experiment,
        "system_of_record": "qlib",
        "provider_uri": spec.daily_provider_uri.resolve().as_posix(),
        "market": "US",
        "region": "us",
        "as_of": source_audit.get("execution_provider", {}).get("as_of"),
        "data_range": [
            source_audit.get("calendar_start"),
            source_audit.get("calendar_end"),
        ],
        "benchmark": "SOXX",
        "assets": ["SOXL", "SOXX", "cash"],
        "architecture": {
            "base_position": None,
            "explicit_prior": None,
            "candidate_weight_grid": False,
            "allocation_output": (
                "SOXL=cap*sigmoid(a); cash=(1-SOXL)*sigmoid(b); "
                "SOXX=residual"
            ),
            "candidates": [asdict(candidate) for candidate in CANDIDATES],
            "auxiliary_tasks": list(LABEL_COLUMNS),
            "final_weights_are_direct_model_outputs": True,
        },
        "feature_sets": {
            "base": base_features,
            "fast": fast_features,
            "base_count": len(base_features),
            "fast_count": len(fast_features),
        },
        "labels": {
            "tail_1d": "future SOXL adverse 1d <= -8%",
            "tail_3d": "future SOXX adverse 3d <= -4%",
            "tail_5d": (
                "future SOXL adverse 5d <= -12% or "
                "SOXX adverse 5d <= -6%"
            ),
            "relative_alpha": (
                "risk-adjusted 5d/20d SOXL utility exceeds SOXX by 2%"
            ),
            "recovery": (
                "known stress context and future SOXX 5d >= 4% "
                "with adverse path >= -4%"
            ),
        },
        "data_audit": {**source_audit, "dataset": dataset_audit},
        "development": {
            "years": list(spec.development_years),
            "selection_uses_outer_test": False,
            "diagnostic_selected": _json_ready(selected.to_dict()),
        },
        "outer_diagnostic": {
            "start": OUTER_START,
            "selection_uses_outer_test": False,
            "selected_candidate_metrics_10bps": _json_ready(
                outer10.to_dict()
            ),
            "all_candidate_metrics": comparison.loc[
                comparison["segment"].eq("outer_diagnostic")
            ].to_dict(orient="records"),
        },
        "objective_pass": objective_pass,
        "promotion_decision": decision,
        "objective_gate_audit": objective_audit.to_dict(
            orient="records"
        ),
        "model_selection": selection.to_dict(orient="records"),
        "annual_metrics": annual.to_dict(orient="records"),
        "classification_metrics": classification.to_dict(
            orient="records"
        ),
        "development_classification_summary": (
            development_classification.to_dict(orient="records")
        ),
        "outer_classification_summary": outer_classification.to_dict(
            orient="records"
        ),
        "regime_metrics": regimes.to_dict(orient="records"),
        "qlib_docs": [
            "https://qlib.readthedocs.io/en/latest/component/data.html",
            "https://qlib.readthedocs.io/en/latest/component/model.html",
            "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            "https://qlib.readthedocs.io/en/latest/component/recorder.html",
        ],
        "recorder_id": recorder_id,
        "recorder_uri": recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": list(RECORDER_ONLY_ARTIFACTS),
        },
    }


def _write_card(
    *,
    card: Path,
    summary: dict[str, Any],
    selection: pd.DataFrame,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    classification: pd.DataFrame,
    development_classification: pd.DataFrame,
    outer_classification: pd.DataFrame,
    regimes: pd.DataFrame,
    objective_audit: pd.DataFrame,
    diagnostics: pd.DataFrame,
    base_features: list[str],
    fast_features: list[str],
) -> None:
    selected = selection.loc[
        selection["diagnostic_selected"]
    ].iloc[0]
    selected_name = str(selected["candidate"])
    outer10 = comparison.loc[
        comparison["segment"].eq("outer_diagnostic")
        & comparison["candidate"].eq(selected_name)
        & comparison["cost_bps"].eq(10)
    ].iloc[0]
    (card / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8"
    )
    (card / "selected_model.json").write_text(
        json.dumps(_json_ready(selected.to_dict()), indent=2),
        encoding="utf-8",
    )
    selection.to_csv(card / "model_selection.csv", index=False)
    comparison.to_csv(card / "model_comparison.csv", index=False)
    comparison.to_csv(
        card / "backtest_benchmark_comparison.csv", index=False
    )
    annual.to_csv(card / "backtest_walk_forward_metrics.csv", index=False)
    classification.to_csv(
        card / "classification_metrics.csv", index=False
    )
    pd.concat(
        [
            development_classification.assign(
                summary_segment="development_walk_forward"
            ),
            outer_classification.assign(
                summary_segment="outer_diagnostic"
            ),
        ],
        ignore_index=True,
    ).to_csv(card / "classification_summary.csv", index=False)
    comparison.loc[
        :,
        [
            "segment",
            "candidate",
            "cost_bps",
            "max_drawdown_pct",
            "worst_day_pct",
            "cvar5_pct",
            "worst_rolling_5d_pct",
            "soxx_max_drawdown_pct",
        ],
    ].to_csv(card / "risk_analysis.csv", index=False)
    regimes.to_csv(card / "regime_metrics.csv", index=False)
    diagnostics.to_csv(card / "training_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "feature": feature,
                "feature_set": "base",
                "point_in_time": True,
                "base_or_prior_feature": False,
            }
            for feature in base_features
        ]
        + [
            {
                "feature": feature,
                "feature_set": "fast_delta",
                "point_in_time": True,
                "base_or_prior_feature": False,
            }
            for feature in fast_features
            if feature not in base_features
        ]
    ).to_csv(card / "feature_manifest.csv", index=False)
    objective_audit.to_csv(card / "objective_gate_audit.csv", index=False)
    analysis = {
        "objective_pass": summary["objective_pass"],
        "promotion_decision": summary["promotion_decision"],
        "diagnostic_selected": selected_name,
        "development_ratio_10bps": selected["ratio_10bps"],
        "development_max_drawdown_pct": selected["max_drawdown_pct"],
        "outer_ratio_10bps": outer10[
            "terminal_wealth_ratio_vs_soxx"
        ],
        "outer_max_drawdown_pct": outer10["max_drawdown_pct"],
        "classification_learnability_pass": bool(
            selected["learnability_gate"]
        ),
        "selection_uses_outer_test": False,
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(_json_ready(analysis), indent=2), encoding="utf-8"
    )
    failed = objective_audit.loc[
        objective_audit["required"] & ~objective_audit["passed"]
    ]
    (card / "failure_modes.md").write_text(
        "# Failure modes\n\n"
        + (
            "\n".join(
                f"- {row.gate}: {row.evidence}"
                for row in failed.itertuples()
            )
            if not failed.empty
            else "- none"
        )
        + "\n",
        encoding="utf-8",
    )
    conclusion = (
        "# 多任务分类直接连续配置结论\n\n"
        "- 跑了什么：比较 allocation-only、tabular 多任务、快速变化"
        "特征多任务和20日GRU多任务；所有候选直接输出连续"
        "SOXL/SOXX/cash权重。\n"
        f"- 开发期诊断优胜：{selected_name}；10bp财富/SOXX="
        f"{selected['ratio_10bps']:.4f}，25bp="
        f"{selected['ratio_25bps']:.4f}，MaxDD="
        f"{selected['max_drawdown_pct']:.2f}%。\n"
        f"- 分类：tail1 AUC={selected['tail_1d_label_auc']:.4f}，"
        f"relative alpha AUC="
        f"{selected['relative_alpha_label_auc']:.4f}，"
        f"recovery AUC={selected['recovery_label_auc']:.4f}。\n"
        f"- Outer仅诊断：10bp财富/SOXX="
        f"{outer10['terminal_wealth_ratio_vs_soxx']:.4f}，MaxDD="
        f"{outer10['max_drawdown_pct']:.2f}%。\n"
        f"- 晋升决定：{summary['promotion_decision']}。\n"
        "- 风险：分类标签若跨年份不可学习，继续更换网络没有意义；"
        "outer不得用于选择标签、特征或模型。\n"
        "- 下一步：只根据开发期分类与风险门决定是继续优化现有"
        "特征，还是转向新的PIT盘前/期权/成分股实时信息。\n"
    )
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    workflow = (
        f"experiment: {EXPERIMENT}\n"
        "system_of_record: qlib\n"
        f"provider_uri: {summary['provider_uri']}\n"
        "benchmark: SOXX\n"
        "assets: [SOXL, SOXX, cash]\n"
        "allocation_output: direct_continuous_factorized\n"
        "auxiliary_tasks: [tail_1d, tail_3d, tail_5d, relative_alpha, recovery]\n"
        "candidates: [tabular_base_allocation_only, "
        "tabular_base_multitask, tabular_fast_multitask, "
        "gru_fast_multitask]\n"
        "base_position: none\n"
        "explicit_prior: none\n"
        "candidate_weight_grid: false\n"
        "development: [2016, 2022]\n"
        f"outer_diagnostic: [2023, {summary['data_range'][1]}]\n"
        "selection_uses_outer_test: false\n"
        "artifact_policy: slim_card_recorder_first\n"
    )
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
    _write_notebook(card, summary)


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Multitask classification direct allocation\n\n"
            "同时比较标签、快速变化特征、tabular/GRU编码器；"
            "最终权重始终由模型直接连续输出。"
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "import qlib\n"
            "from qlib.data import D\n"
            "from qlib.workflow import R\n"
            "repo_root = Path.cwd().parents[1]\n"
            "if str(repo_root) not in sys.path:\n"
            "    sys.path.insert(0, str(repo_root))\n"
            "qlib.init(provider_uri=str(repo_root / 'data/qlib/soxl'), "
            "region='us', kernels=1, skip_if_reg=False)\n"
            "calendar = D.calendar(start_time='2011-01-03', "
            "end_time='2026-12-31', freq='day')\n"
            "from src.workflow.multitask_classification_allocation "
            "import run_multitask_classification_allocation_card\n"
            "summary = run_multitask_classification_allocation_card("
            "root=repo_root)\n"
            "summary['promotion_decision']"
        ),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "card = Path.cwd()\n"
            "pd.read_csv(card / 'model_selection.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'classification_summary.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'backtest_benchmark_comparison.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'objective_gate_audit.csv')"
        ),
        nbformat.v4.new_markdown_cell(
            (card / "conclusion.md").read_text(encoding="utf-8")
        ),
    ]
    nbformat.write(
        notebook, card / f"{summary['experiment']}.ipynb"
    )


if __name__ == "__main__":
    print(
        json.dumps(
            run_multitask_classification_allocation_card(), indent=2
        )
    )
