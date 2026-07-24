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
from torch import nn

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.factor import (
    CONSTITUENT_MODEL_FEATURES,
    actual_2x_calibration_metrics,
    build_constituent_2x_tail_training_frame,
    load_soxx_pit_leveraged_training_frame,
)
from src.workflow.direct_continuous_soxx_allocation import (
    DEVELOPMENT_YEARS,
    DirectContinuousAllocationSpec,
    DirectLossProfile,
    _annual_metrics,
    _evaluate,
    _load_frame,
    _prepare,
)
from src.workflow.soxl_objective_feasibility_audit import _json_ready
from src.workflow.trainable_regime_policy_nested_wf import (
    _remove_qlib_code_snapshot_artifacts,
)


EXPERIMENT = "pit_constituent_auxiliary_training_allocation_v1"
DRAWDOWN_EXPERIMENT = "drawdown_targeted_pit_allocation_v1"
PRIMARY_INSTRUMENT = "SOXX"
COMMON_FEATURES = tuple(
    feature
    for feature in CONSTITUENT_MODEL_FEATURES
    if not feature.endswith("_rank")
)
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/predictions.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/training_diagnostics.csv",
)


def _recorder_only_artifacts(experiment: str) -> tuple[str, ...]:
    return (
        f"artifacts/{experiment}/models/",
        f"artifacts/{experiment}/predictions.csv",
        f"artifacts/{experiment}/positions.csv",
        f"artifacts/{experiment}/strategy_returns.csv",
        f"artifacts/{experiment}/training_diagnostics.csv",
    )


@dataclass(frozen=True)
class AuxiliaryTrainingCandidate:
    name: str
    auxiliary_weight: float
    include_union_backfill: bool
    method: str = "relative_return_baseline"
    absolute_drawdown_penalty: float = 0.0
    conditional_drawdown_penalty: float = 0.0
    rolling_5d_penalty: float = 0.0
    cash_target_penalty: float = 0.0
    tail_event_weight_multiplier: float = 1.0
    stress_cash_ceiling: float = 0.55
    smoothing_span: int | None = None
    rebalance_threshold: float | None = None
    drawdown_floor: float | None = None
    drawdown_budget_multiplier: float = 0.0
    drawdown_budget_metric: str = "risky_weight"
    drawdown_budget_buffer: float = 0.0


CANDIDATES = (
    AuxiliaryTrainingCandidate(
        name="primary_only",
        auxiliary_weight=0.0,
        include_union_backfill=False,
    ),
    AuxiliaryTrainingCandidate(
        name="pit_only_aux_005",
        auxiliary_weight=0.05,
        include_union_backfill=False,
    ),
    AuxiliaryTrainingCandidate(
        name="pit_plus_union_aux_003",
        auxiliary_weight=0.03,
        include_union_backfill=True,
    ),
    AuxiliaryTrainingCandidate(
        name="pit_plus_union_aux_005",
        auxiliary_weight=0.05,
        include_union_backfill=True,
    ),
    AuxiliaryTrainingCandidate(
        name="pit_plus_union_aux_008",
        auxiliary_weight=0.08,
        include_union_backfill=True,
    ),
)

DRAWDOWN_CANDIDATES = (
    AuxiliaryTrainingCandidate(
        name="primary_only",
        auxiliary_weight=0.0,
        include_union_backfill=False,
        method="primary_relative_return_baseline",
    ),
    AuxiliaryTrainingCandidate(
        name="pit_only_aux_005",
        auxiliary_weight=0.05,
        include_union_backfill=False,
        method="relative_return_baseline",
    ),
    AuxiliaryTrainingCandidate(
        name="pit_mdd_003",
        auxiliary_weight=0.05,
        include_union_backfill=False,
        method="deepdow_absolute_max_drawdown",
        absolute_drawdown_penalty=0.03,
    ),
    AuxiliaryTrainingCandidate(
        name="pit_mdd_ced_008",
        auxiliary_weight=0.05,
        include_union_backfill=False,
        method="maximum_drawdown_plus_conditional_expected_drawdown",
        absolute_drawdown_penalty=0.08,
        conditional_drawdown_penalty=0.02,
        rolling_5d_penalty=0.05,
    ),
    AuxiliaryTrainingCandidate(
        name="pit_cdar_cash_005",
        auxiliary_weight=0.05,
        include_union_backfill=False,
        method="conditional_drawdown_plus_direct_cash_supervision",
        absolute_drawdown_penalty=0.05,
        conditional_drawdown_penalty=0.05,
        rolling_5d_penalty=0.10,
        cash_target_penalty=0.05,
        tail_event_weight_multiplier=4.0,
        stress_cash_ceiling=0.75,
    ),
    AuxiliaryTrainingCandidate(
        name="pit_cdar_cash_fast_010",
        auxiliary_weight=0.05,
        include_union_backfill=False,
        method="conditional_drawdown_cash_supervision_fast_transition",
        absolute_drawdown_penalty=0.10,
        conditional_drawdown_penalty=0.08,
        rolling_5d_penalty=0.20,
        cash_target_penalty=0.10,
        tail_event_weight_multiplier=8.0,
        stress_cash_ceiling=0.85,
        smoothing_span=1,
        rebalance_threshold=0.0,
    ),
    AuxiliaryTrainingCandidate(
        name="pit_budget_risky_250",
        auxiliary_weight=0.05,
        include_union_backfill=False,
        method="state_dependent_drawdown_budget_risky_weight",
        drawdown_floor=0.70,
        drawdown_budget_multiplier=2.50,
        drawdown_budget_metric="risky_weight",
    ),
    AuxiliaryTrainingCandidate(
        name="pit_budget_beta_400",
        auxiliary_weight=0.05,
        include_union_backfill=False,
        method="state_dependent_drawdown_budget_effective_beta",
        drawdown_floor=0.70,
        drawdown_budget_multiplier=4.00,
        drawdown_budget_metric="effective_beta",
    ),
    AuxiliaryTrainingCandidate(
        name="pit_budget_risky_200",
        auxiliary_weight=0.05,
        include_union_backfill=False,
        method="state_dependent_drawdown_budget_conservative",
        drawdown_floor=0.70,
        drawdown_budget_multiplier=2.00,
        drawdown_budget_metric="risky_weight",
    ),
)


@dataclass(frozen=True)
class PitAuxiliaryAllocationSpec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    development_years: tuple[int, ...] = DEVELOPMENT_YEARS
    development_end: str = "2022-12-30"
    outer: tuple[str, str] = ("2023-01-03", "2026-07-02")
    purge_rows: int = 20
    hidden_units: int = 16
    tail_hidden_units: int = 8
    epochs: int = 300
    learning_rate: float = 0.002
    weight_decay: float = 0.08
    maximum_soxl_weight: float = 0.30
    minimum_cash_weight: float = 0.15
    primary_tail_loss_weight: float = 0.02
    turnover_loss_weight: float = 0.006
    training_cost_bps: int = 10
    costs_bps: tuple[int, ...] = (10, 25)
    auxiliary_batch_size: int = 2048
    smoothing_span: int = 3
    rebalance_threshold: float = 0.02
    research_max_drawdown_target_pct: float | None = None
    research_worst_5d_target_pct: float = -15.0
    research_minimum_ratio_10bps: float = 1.0
    seed: int = 71

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def daily_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def auxiliary_provider_uri(self) -> Path:
        return (
            self.root
            / "data"
            / "qlib"
            / "soxx_pit_leveraged_training_v1"
        )


class PitAuxiliaryAllocationNet(nn.Module):
    """Direct allocator with a constituent-trained, shared tail encoder."""

    def __init__(
        self,
        *,
        primary_feature_count: int,
        common_feature_count: int,
        hidden_units: int,
        tail_hidden_units: int,
    ):
        super().__init__()
        self.tail_encoder = nn.Sequential(
            nn.Linear(common_feature_count, tail_hidden_units),
            nn.Tanh(),
        )
        self.tail_head = nn.Linear(tail_hidden_units, 1)
        self.alpha_head = nn.Linear(tail_hidden_units, 1)
        self.allocation_head = nn.Sequential(
            nn.Linear(
                primary_feature_count + tail_hidden_units,
                hidden_units,
            ),
            nn.Tanh(),
            nn.Linear(hidden_units, 3),
        )

    def forward(
        self,
        primary_features: torch.Tensor,
        common_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tail_state = self.tail_encoder(common_features)
        allocation_logits = self.allocation_head(
            torch.cat([primary_features, tail_state], dim=1)
        )
        tail_logit = self.tail_head(tail_state).squeeze(1)
        alpha_logit = self.alpha_head(tail_state).squeeze(1)
        return allocation_logits, tail_logit, alpha_logit


def run_pit_constituent_auxiliary_training_allocation_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    return _run_card(
        PitAuxiliaryAllocationSpec(root=repo_root),
        candidates=CANDIDATES,
    )


def run_drawdown_targeted_pit_allocation_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    return _run_card(
        PitAuxiliaryAllocationSpec(
            root=repo_root,
            experiment=DRAWDOWN_EXPERIMENT,
            research_max_drawdown_target_pct=-30.0,
            research_worst_5d_target_pct=-15.0,
            research_minimum_ratio_10bps=0.50,
        ),
        candidates=DRAWDOWN_CANDIDATES,
    )


def _project_legal_weights(
    logits: torch.Tensor,
    *,
    maximum_soxl_weight: float,
    minimum_cash_weight: float,
) -> torch.Tensor:
    if not 0.0 <= minimum_cash_weight < 1.0:
        raise ValueError("minimum cash weight must be in [0, 1)")
    probabilities = torch.softmax(torch.clamp(logits, -8.0, 8.0), dim=1)
    weights = probabilities * (1.0 - minimum_cash_weight)
    weights = weights.clone()
    weights[:, 2] = weights[:, 2] + minimum_cash_weight
    excess = torch.relu(weights[:, 0] - maximum_soxl_weight)
    denominator = weights[:, 1] + weights[:, 2] + 1e-8
    projected = torch.stack(
        [
            weights[:, 0] - excess,
            weights[:, 1] + excess * weights[:, 1] / denominator,
            weights[:, 2] + excess * weights[:, 2] / denominator,
        ],
        dim=1,
    )
    return projected / projected.sum(dim=1, keepdim=True)


def _soxx_common_training_frame(
    spec: PitAuxiliaryAllocationSpec,
    *,
    end: str,
) -> pd.DataFrame:
    raw = load_qlib_feature_frame(
        QlibProviderConfig(provider_uri=spec.daily_provider_uri),
        instruments=[PRIMARY_INSTRUMENT],
        fields=["$open", "$high", "$low", "$close", "$volume"],
        start="2010-03-11",
        end=end,
    ).reset_index()
    raw["$adj_close"] = raw["$close"]
    raw["$pit_weight_pct"] = 100.0
    raw["$pit_available"] = 0.0
    raw["$pit_member"] = 1.0
    raw["$instrument_type"] = 0.0
    raw["$leverage_multiple"] = 1.0
    raw["$is_actual_2x"] = 0.0
    result = build_constituent_2x_tail_training_frame(raw)
    result = result.set_index("datetime").sort_index()
    return result


def _load_auxiliary_frame(
    spec: PitAuxiliaryAllocationSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path = spec.auxiliary_provider_uri / "qlib_store_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qlib_frame = load_soxx_pit_leveraged_training_frame(
        QlibProviderConfig(provider_uri=spec.auxiliary_provider_uri),
        instruments=list(manifest["instruments"]),
        start=manifest["calendar_start"],
        end=min(spec.outer[1], manifest["calendar_end"]),
    )
    auxiliary = build_constituent_2x_tail_training_frame(qlib_frame)
    auxiliary = auxiliary.loc[
        auxiliary["training_eligible"]
        & auxiliary["synthetic_2x_crash_label"].notna()
    ].copy()
    calibration = actual_2x_calibration_metrics(qlib_frame)
    audit = {
        "provider_uri": spec.auxiliary_provider_uri.resolve().as_posix(),
        "source_api": "qlib.data.D.features",
        "row_count": int(len(auxiliary)),
        "instrument_count": int(auxiliary["instrument"].nunique()),
        "pit_member_rows": int(
            auxiliary["training_scope"].eq("pit_member").sum()
        ),
        "union_backfill_rows": int(
            auxiliary["training_scope"]
            .eq("union_backfill_pretrain")
            .sum()
        ),
        "synthetic_2x_crash_rows": int(
            auxiliary["synthetic_2x_crash_label"].eq(1.0).sum()
        ),
        "actual_2x_role": (
            "calibrates synthetic daily-reset labels; short histories are "
            "not pooled into allocation rows"
        ),
        "actual_2x_calibration_products": calibration[
            "product"
        ].tolist(),
        "pre_pit_limitation": (
            "union backfill is survivorship-biased and receives 0.25 "
            "training weight; it is a separately selectable pretrain source"
        ),
    }
    return auxiliary, calibration, audit


def _build_auxiliary_dataset(
    auxiliary: pd.DataFrame,
    spec: PitAuxiliaryAllocationSpec,
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    labels = [
        "synthetic_2x_crash_label",
        "synthetic_2x_upside_label",
        "synthetic_2x_tail_severity",
        "training_weight",
        "$pit_available",
    ]
    static = pd.concat(
        {
            "feature": auxiliary.loc[:, list(COMMON_FEATURES)],
            "label": auxiliary.loc[:, labels],
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(auxiliary["datetime"]).dt.normalize(),
            auxiliary["instrument"].astype(str).str.upper(),
        ],
        names=["datetime", "instrument"],
    )
    dataset = DatasetH(
        handler=DataHandlerLP(
            instruments=None,
            start_time=static.index.get_level_values("datetime").min(),
            end_time=static.index.get_level_values("datetime").max(),
            data_loader=StaticDataLoader(config=static),
            infer_processors=[],
            learn_processors=[],
        ),
        segments={
            "all": (
                static.index.get_level_values("datetime")
                .min()
                .date()
                .isoformat(),
                static.index.get_level_values("datetime")
                .max()
                .date()
                .isoformat(),
            )
        },
    )
    return dataset, {
        "dataset_class": "qlib.data.dataset.DatasetH",
        "handler_class": "qlib.data.dataset.handler.DataHandlerLP",
        "loader_class": "qlib.data.dataset.loader.StaticDataLoader",
        "rows": int(len(dataset.prepare("all"))),
        "feature_count": len(COMMON_FEATURES),
    }


def _build_primary_dataset(
    frame: pd.DataFrame,
    features: list[str],
    spec: PitAuxiliaryAllocationSpec,
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    labels = [
        "soxl_forward_open_return",
        "soxx_forward_open_return",
        "tail_event_5d",
        "tail_cash_target_5d",
        "relative_alpha_5d",
    ]
    static = pd.concat(
        {
            "feature": frame.loc[:, features],
            "label": frame.loc[:, labels],
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [frame.index, np.full(len(frame), "SOXL_SOXX_CASH")],
        names=["datetime", "instrument"],
    )
    segments: dict[str, tuple[str, str]] = {
        "development": (
            f"{spec.development_years[0]}-01-01",
            spec.development_end,
        ),
        "outer": (
            spec.outer[0],
            min(spec.outer[1], frame.index.max().date().isoformat()),
        ),
    }
    for year in spec.development_years:
        segments[f"fold_{year}"] = (
            f"{year}-01-01",
            f"{year}-12-31",
        )
    dataset = DatasetH(
        handler=DataHandlerLP(
            instruments=None,
            start_time=frame.index.min(),
            end_time=frame.index.max(),
            data_loader=StaticDataLoader(config=static),
            infer_processors=[],
            learn_processors=[],
        ),
        segments=segments,
    )
    return dataset, {
        "dataset_class": "qlib.data.dataset.DatasetH",
        "handler_class": "qlib.data.dataset.handler.DataHandlerLP",
        "loader_class": "qlib.data.dataset.loader.StaticDataLoader",
        "segments": {name: list(bounds) for name, bounds in segments.items()},
        "segment_rows": {
            name: int(len(dataset.prepare(name))) for name in segments
        },
        "labels": labels,
        "purge_rows": spec.purge_rows,
    }


def _prepare_auxiliary(dataset, segment) -> pd.DataFrame:
    result = dataset.prepare(segment, col_set=["feature", "label"])
    result.columns = [str(column[-1]) for column in result.columns]
    result = result.reset_index()
    result["datetime"] = pd.to_datetime(result["datetime"]).dt.normalize()
    return result.sort_values(["datetime", "instrument"]).reset_index(
        drop=True
    )


def _standardize_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] | tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, pd.Series]]:
    columns = list(features)
    median = train.loc[:, columns].median()
    train_values = train.loc[:, columns].fillna(median)
    test_values = test.loc[:, columns].fillna(median)
    mean = train_values.mean()
    scale = (
        train_values.std(ddof=1)
        .replace(0.0, 1.0)
        .fillna(1.0)
    )

    def transform(values: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(
            ((values - mean) / scale)
            .clip(-8.0, 8.0)
            .to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

    return (
        transform(train_values),
        transform(test_values),
        {"median": median, "mean": mean, "scale": scale},
    )


def _transform_with_scaler(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    scaler: dict[str, pd.Series],
) -> torch.Tensor:
    values = frame.loc[:, list(features)].fillna(scaler["median"])
    scaled = (
        (values - scaler["mean"]) / scaler["scale"]
    ).clip(-8.0, 8.0)
    return torch.tensor(
        scaled.to_numpy(dtype=np.float32), dtype=torch.float32
    )


def _allocation_loss(
    *,
    weights: torch.Tensor,
    returns: torch.Tensor,
    years: torch.Tensor,
    cash_target: torch.Tensor,
    tail_event: torch.Tensor,
    candidate: AuxiliaryTrainingCandidate,
    drawdown_target_pct: float,
    worst_5d_target_pct: float,
    minimum_cash_weight: float,
    cost_bps: int,
    turnover_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    initial = torch.tensor(
        [0.0, 0.0, 1.0],
        dtype=weights.dtype,
        device=weights.device,
    )
    turnover = torch.cat(
        [
            (0.5 * torch.abs(weights[0] - initial).sum()).reshape(1),
            0.5 * torch.abs(weights[1:] - weights[:-1]).sum(dim=1),
        ]
    )
    portfolio_return = (
        weights[:, :2] * returns
    ).sum(dim=1) - turnover * cost_bps / 10_000.0
    portfolio_log = torch.log1p(torch.clamp(portfolio_return, min=-0.95))
    relative_log = portfolio_log - torch.log1p(
        torch.clamp(returns[:, 1], min=-0.95)
    )
    annual = torch.stack(
        [
            relative_log[years.eq(year)].mean()
            for year in torch.unique(years)
        ]
    )
    loss = (
        -annual.mean()
        + 0.50 * annual.std(unbiased=False)
        + turnover_weight * turnover.mean()
    )
    max_drawdown_limit = -float(np.log1p(drawdown_target_pct / 100.0))
    conditional_drawdown_limit = -float(
        np.log1p((drawdown_target_pct * 2.0 / 3.0) / 100.0)
    )
    max_drawdown_losses: list[torch.Tensor] = []
    conditional_drawdown_losses: list[torch.Tensor] = []
    for mask in [
        torch.ones_like(years, dtype=torch.bool),
        *(years.eq(year) for year in torch.unique(years)),
    ]:
        path = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=portfolio_log.dtype,
                    device=portfolio_log.device,
                ),
                torch.cumsum(portfolio_log[mask], dim=0),
            ]
        )
        drawdown = torch.cummax(path, dim=0).values - path
        max_drawdown_losses.append(
            torch.relu(drawdown.max() - max_drawdown_limit).square()
        )
        tail_count = max(1, int(np.ceil(0.10 * len(drawdown))))
        conditional_drawdown = torch.topk(
            drawdown, tail_count
        ).values.mean()
        conditional_drawdown_losses.append(
            torch.relu(
                conditional_drawdown - conditional_drawdown_limit
            ).square()
        )
    max_drawdown_loss = torch.stack(max_drawdown_losses).mean()
    conditional_drawdown_loss = torch.stack(
        conditional_drawdown_losses
    ).mean()

    rolling_5d_log = portfolio_log.unfold(0, 5, 1).sum(dim=1)
    rolling_5d_limit = float(np.log1p(worst_5d_target_pct / 100.0))
    rolling_5d_shortfall = torch.relu(
        rolling_5d_limit - rolling_5d_log
    )
    rolling_5d_loss = rolling_5d_shortfall.square().mean()

    normalized_cash_target = torch.clamp(cash_target / 0.55, 0.0, 1.0)
    supervised_cash_target = minimum_cash_weight + (
        candidate.stress_cash_ceiling - minimum_cash_weight
    ) * normalized_cash_target
    cash_sample_weight = 1.0 + (
        candidate.tail_event_weight_multiplier - 1.0
    ) * tail_event
    cash_target_loss = (
        torch.nn.functional.smooth_l1_loss(
            weights[:, 2],
            supervised_cash_target,
            reduction="none",
        )
        * cash_sample_weight
    ).mean()

    loss = (
        loss
        + candidate.absolute_drawdown_penalty * max_drawdown_loss
        + candidate.conditional_drawdown_penalty
        * conditional_drawdown_loss
        + candidate.rolling_5d_penalty * rolling_5d_loss
        + candidate.cash_target_penalty * cash_target_loss
    )
    components = {
        "max_drawdown_loss": max_drawdown_loss,
        "conditional_drawdown_loss": conditional_drawdown_loss,
        "rolling_5d_loss": rolling_5d_loss,
        "cash_target_loss": cash_target_loss,
    }
    return loss, turnover, annual, components


def _fit_predict(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    soxx_common: pd.DataFrame,
    auxiliary: pd.DataFrame,
    primary_features: list[str],
    candidate: AuxiliaryTrainingCandidate,
    spec: PitAuxiliaryAllocationSpec,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    required = [
        "soxl_forward_open_return",
        "soxx_forward_open_return",
        "tail_event_5d",
        "tail_cash_target_5d",
        "relative_alpha_5d",
    ]
    train = train.dropna(subset=required).copy()
    test = test.dropna(subset=required).copy()
    common_train = soxx_common.reindex(train.index)
    common_test = soxx_common.reindex(test.index)
    if train.empty or test.empty:
        raise ValueError("blocked: empty PIT auxiliary train or test")

    primary_x, primary_test_x, primary_scaler = _standardize_pair(
        train, test, primary_features
    )
    common_x, common_test_x, common_scaler = _standardize_pair(
        common_train, common_test, COMMON_FEATURES
    )
    auxiliary = auxiliary.dropna(
        subset=[
            *COMMON_FEATURES,
            "synthetic_2x_crash_label",
            "synthetic_2x_upside_label",
            "training_weight",
        ]
    ).copy()
    if not candidate.include_union_backfill:
        auxiliary = auxiliary.loc[
            pd.to_numeric(
                auxiliary["$pit_available"], errors="coerce"
            ).gt(0.5)
        ]
    auxiliary_x = _transform_with_scaler(
        auxiliary, COMMON_FEATURES, common_scaler
    )
    auxiliary_target = torch.tensor(
        auxiliary["synthetic_2x_crash_label"].to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32,
    )
    auxiliary_upside_target = torch.tensor(
        auxiliary["synthetic_2x_upside_label"].to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32,
    )
    auxiliary_weight = torch.tensor(
        auxiliary["training_weight"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    if len(auxiliary_weight):
        auxiliary_weight = auxiliary_weight / auxiliary_weight.mean()

    returns = torch.tensor(
        train.loc[
            :, ["soxl_forward_open_return", "soxx_forward_open_return"]
        ].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    years = torch.tensor(train.index.year.to_numpy(), dtype=torch.int64)
    primary_tail = torch.tensor(
        train["tail_event_5d"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    primary_cash_target = torch.tensor(
        train["tail_cash_target_5d"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    primary_upside = torch.tensor(
        train["relative_alpha_5d"].gt(0.0).to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PitAuxiliaryAllocationNet(
        primary_feature_count=len(primary_features),
        common_feature_count=len(COMMON_FEATURES),
        hidden_units=spec.hidden_units,
        tail_hidden_units=spec.tail_hidden_units,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    diagnostics: dict[str, float] = {}
    for epoch in range(spec.epochs):
        optimizer.zero_grad()
        (
            logits,
            primary_tail_logit,
            primary_alpha_logit,
        ) = model(primary_x, common_x)
        weights = _project_legal_weights(
            logits,
            maximum_soxl_weight=spec.maximum_soxl_weight,
            minimum_cash_weight=spec.minimum_cash_weight,
        )
        allocation_loss, turnover, annual, risk_components = _allocation_loss(
            weights=weights,
            returns=returns,
            years=years,
            cash_target=primary_cash_target,
            tail_event=primary_tail,
            candidate=candidate,
            drawdown_target_pct=(
                spec.research_max_drawdown_target_pct
                if spec.research_max_drawdown_target_pct is not None
                else -30.0
            ),
            worst_5d_target_pct=spec.research_worst_5d_target_pct,
            minimum_cash_weight=spec.minimum_cash_weight,
            cost_bps=spec.training_cost_bps,
            turnover_weight=spec.turnover_loss_weight,
        )
        primary_tail_loss = nn.functional.binary_cross_entropy_with_logits(
            primary_tail_logit, primary_tail
        )
        primary_alpha_loss = nn.functional.binary_cross_entropy_with_logits(
            primary_alpha_logit, primary_upside
        )
        if candidate.auxiliary_weight > 0.0 and not auxiliary.empty:
            generator = torch.Generator().manual_seed(seed * 1000 + epoch)
            sample = torch.randint(
                len(auxiliary_x),
                (
                    min(
                        spec.auxiliary_batch_size,
                        len(auxiliary_x),
                    ),
                ),
                generator=generator,
            )
            auxiliary_logit = model.tail_head(
                model.tail_encoder(auxiliary_x[sample])
            ).squeeze(1)
            auxiliary_alpha_logit = model.alpha_head(
                model.tail_encoder(auxiliary_x[sample])
            ).squeeze(1)
            per_row = nn.functional.binary_cross_entropy_with_logits(
                auxiliary_logit,
                auxiliary_target[sample],
                reduction="none",
            )
            auxiliary_loss = (
                per_row * auxiliary_weight[sample]
            ).mean()
            auxiliary_alpha_loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    auxiliary_alpha_logit,
                    auxiliary_upside_target[sample],
                    reduction="none",
                )
                * auxiliary_weight[sample]
            ).mean()
        else:
            auxiliary_loss = torch.zeros(
                (), dtype=allocation_loss.dtype
            )
            auxiliary_alpha_loss = torch.zeros(
                (), dtype=allocation_loss.dtype
            )
        loss = (
            allocation_loss
            + spec.primary_tail_loss_weight
            * (primary_tail_loss + primary_alpha_loss)
            + candidate.auxiliary_weight
            * (auxiliary_loss + auxiliary_alpha_loss)
        )
        if not torch.isfinite(loss):
            raise ValueError(
                f"blocked: non-finite PIT auxiliary loss {candidate.name}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        value = float(loss.detach())
        diagnostics = {
            "loss": value,
            "allocation_loss": float(allocation_loss.detach()),
            "primary_tail_loss": float(primary_tail_loss.detach()),
            "primary_alpha_loss": float(primary_alpha_loss.detach()),
            "auxiliary_tail_loss": float(auxiliary_loss.detach()),
            "auxiliary_alpha_loss": float(
                auxiliary_alpha_loss.detach()
            ),
            "mean_turnover_loss": float(turnover.mean().detach()),
            "mean_annual_relative_log": float(annual.mean().detach()),
            **{
                name: float(component.detach())
                for name, component in risk_components.items()
            },
        }
        if value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().clone()
                for key, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise ValueError("blocked: no PIT auxiliary model state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits, test_tail_logit, test_alpha_logit = model(
            primary_test_x, common_test_x
        )
        prediction = _project_legal_weights(
            test_logits,
            maximum_soxl_weight=spec.maximum_soxl_weight,
            minimum_cash_weight=spec.minimum_cash_weight,
        ).cpu().numpy()
        tail_probability = torch.sigmoid(test_tail_logit).cpu().numpy()
        alpha_probability = torch.sigmoid(
            test_alpha_logit
        ).cpu().numpy()
    output = pd.DataFrame(
        prediction,
        index=test.index,
        columns=["soxl_weight", "soxx_weight", "cash_weight"],
    )
    output["tail_probability"] = tail_probability
    output["alpha_probability"] = alpha_probability
    output = _smooth_weights(output, spec=spec, candidate=candidate)
    state = {
        "model_state": {
            key: value.detach().cpu().numpy()
            for key, value in model.state_dict().items()
        },
        "primary_scaler": {
            name: values.to_dict()
            for name, values in primary_scaler.items()
        },
        "common_scaler": {
            name: values.to_dict()
            for name, values in common_scaler.items()
        },
        "candidate": asdict(candidate),
        "primary_features": primary_features,
        "common_features": list(COMMON_FEATURES),
    }
    diagnostics.update(
        {
            "best_loss": best_loss,
            "primary_train_rows": float(len(train)),
            "auxiliary_train_rows": float(len(auxiliary)),
            "auxiliary_crash_rows": float(
                auxiliary["synthetic_2x_crash_label"].eq(1.0).sum()
            ),
            "test_rows": float(len(output)),
        }
    )
    return output, state, diagnostics


def _smooth_weights(
    prediction: pd.DataFrame,
    *,
    spec: PitAuxiliaryAllocationSpec,
    candidate: AuxiliaryTrainingCandidate | None = None,
) -> pd.DataFrame:
    result = prediction.copy()
    columns = ["soxl_weight", "soxx_weight", "cash_weight"]
    smoothing_span = (
        candidate.smoothing_span
        if candidate is not None and candidate.smoothing_span is not None
        else spec.smoothing_span
    )
    rebalance_threshold = (
        candidate.rebalance_threshold
        if candidate is not None
        and candidate.rebalance_threshold is not None
        else spec.rebalance_threshold
    )
    smoothed = result.loc[:, columns].ewm(
        span=smoothing_span, adjust=False
    ).mean()
    accepted = np.empty_like(smoothed.to_numpy(dtype=float))
    previous = np.array([0.0, 0.0, 1.0], dtype=float)
    for index, values in enumerate(smoothed.to_numpy(dtype=float)):
        if 0.5 * np.abs(values - previous).sum() >= (
            rebalance_threshold
        ):
            previous = values
        accepted[index] = previous
    accepted = np.clip(accepted, 0.0, 1.0)
    accepted = accepted / accepted.sum(axis=1, keepdims=True)
    for column_index, column in enumerate(columns):
        result[column] = accepted[:, column_index].astype("float64")
    return result


def _training_end(
    market: pd.DataFrame,
    year: int,
    purge_rows: int,
) -> pd.Timestamp:
    start_position = market.index.searchsorted(
        pd.Timestamp(f"{year}-01-01")
    )
    end_position = start_position - purge_rows - 1
    if end_position < 0:
        raise ValueError(f"blocked: insufficient purge history for {year}")
    return market.index[end_position]


def _candidate_predictions(
    *,
    primary_dataset,
    auxiliary_dataset,
    market: pd.DataFrame,
    soxx_common: pd.DataFrame,
    primary_features: list[str],
    candidate: AuxiliaryTrainingCandidate,
    spec: PitAuxiliaryAllocationSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    development: list[pd.DataFrame] = []
    states: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for year in spec.development_years:
        train_end = _training_end(market, year, spec.purge_rows)
        train = _prepare(
            primary_dataset, (spec.train_start, train_end)
        )
        test = _prepare(primary_dataset, f"fold_{year}")
        test = test.loc[test.index.year == year]
        auxiliary = _prepare_auxiliary(
            auxiliary_dataset,
            (
                spec.train_start,
                train_end.date().isoformat(),
            ),
        )
        output, state, fit = _fit_predict(
            train=train,
            test=test,
            soxx_common=soxx_common,
            auxiliary=auxiliary,
            primary_features=primary_features,
            candidate=candidate,
            spec=spec,
            seed=spec.seed + year,
        )
        output["candidate"] = candidate.name
        output["segment"] = "development_walk_forward"
        output["fold_year"] = year
        output["train_end"] = train_end
        development.append(output)
        states[f"fold_{year}"] = state
        diagnostics.append(
            {
                "candidate": candidate.name,
                "segment": "development_walk_forward",
                "fold_year": year,
                "train_end": train_end.date().isoformat(),
                "selection_uses_outer_test": False,
                **fit,
            }
        )

    train = _prepare(
        primary_dataset, (spec.train_start, spec.development_end)
    )
    test = _prepare(primary_dataset, "outer")
    auxiliary = _prepare_auxiliary(
        auxiliary_dataset, (spec.train_start, spec.development_end)
    )
    outer, outer_state, outer_fit = _fit_predict(
        train=train,
        test=test,
        soxx_common=soxx_common,
        auxiliary=auxiliary,
        primary_features=primary_features,
        candidate=candidate,
        spec=spec,
        seed=spec.seed,
    )
    outer["candidate"] = candidate.name
    outer["segment"] = "outer_diagnostic"
    outer["fold_year"] = outer.index.year
    outer["train_end"] = pd.Timestamp(spec.development_end)
    states["outer_refit"] = outer_state
    diagnostics.append(
        {
            "candidate": candidate.name,
            "segment": "outer_diagnostic",
            "fold_year": "outer_refit",
            "train_end": spec.development_end,
            "selection_uses_outer_test": False,
            **outer_fit,
        }
    )
    return (
        pd.concat(development).sort_index(),
        outer.sort_index(),
        states,
        pd.DataFrame(diagnostics),
    )


def _profile(candidate: AuxiliaryTrainingCandidate) -> DirectLossProfile:
    return DirectLossProfile(
        name=candidate.name,
        maximum_soxl_weight=0.30,
        negative_year_penalty=0.0,
        year_dispersion_penalty=0.0,
        relative_drawdown_penalty=0.0,
        turnover_penalty=0.0,
    )


def _apply_drawdown_budget(
    *,
    market: pd.DataFrame,
    prediction: pd.DataFrame,
    candidate: AuxiliaryTrainingCandidate,
    cost_bps: int,
) -> pd.DataFrame:
    if candidate.drawdown_floor is None:
        return prediction.copy()
    if not 0.0 < candidate.drawdown_floor < 1.0:
        raise ValueError("drawdown floor must be in (0, 1)")
    if candidate.drawdown_budget_metric not in {
        "risky_weight",
        "effective_beta",
    }:
        raise ValueError("unsupported drawdown budget metric")

    aligned = market.reindex(prediction.index)
    required = ["soxl_forward_open_return", "soxx_forward_open_return"]
    if aligned.loc[:, required].isna().any().any():
        raise ValueError("blocked: drawdown budget market returns missing")
    result = prediction.copy()
    raw = result.loc[
        :, ["soxl_weight", "soxx_weight", "cash_weight"]
    ].to_numpy(dtype=float)
    accepted = np.empty_like(raw)
    previous = np.array([0.0, 0.0, 1.0], dtype=float)
    equity = 1.0
    peak = 1.0
    protected_floor = (
        candidate.drawdown_floor + candidate.drawdown_budget_buffer
    )
    for row_index, values in enumerate(raw):
        cushion = max(
            0.0,
            (equity - protected_floor * peak) / max(equity, 1e-12),
        )
        exposure_cap = candidate.drawdown_budget_multiplier * cushion
        if candidate.drawdown_budget_metric == "effective_beta":
            proposed_exposure = 3.0 * values[0] + values[1]
        else:
            proposed_exposure = values[0] + values[1]
        scale = min(
            1.0,
            exposure_cap / max(proposed_exposure, 1e-12),
        )
        weights = np.array(
            [
                values[0] * scale,
                values[1] * scale,
                1.0 - (values[0] + values[1]) * scale,
            ],
            dtype=float,
        )
        turnover = 0.5 * np.abs(weights - previous).sum()
        daily_return = (
            weights[0]
            * float(aligned.iloc[row_index]["soxl_forward_open_return"])
            + weights[1]
            * float(aligned.iloc[row_index]["soxx_forward_open_return"])
            - turnover * cost_bps / 10_000.0
        )
        equity *= 1.0 + daily_return
        peak = max(peak, equity)
        previous = weights
        accepted[row_index] = weights
    for column_index, column in enumerate(
        ["soxl_weight", "soxx_weight", "cash_weight"]
    ):
        result[column] = accepted[:, column_index]
    return result


def _evaluate_candidates(
    *,
    market: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    segment: str,
    spec: PitAuxiliaryAllocationSpec,
    candidates: tuple[AuxiliaryTrainingCandidate, ...] = CANDIDATES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    annual: list[pd.DataFrame] = []
    for candidate in candidates:
        if candidate.name not in predictions:
            continue
        constrained_prediction = _apply_drawdown_budget(
            market=market,
            prediction=predictions[candidate.name],
            candidate=candidate,
            cost_bps=spec.training_cost_bps,
        )
        for cost_bps in spec.costs_bps:
            metrics, result, benchmark = _evaluate(
                market=market,
                predictions=constrained_prediction,
                profile=_profile(candidate),
                cost_bps=cost_bps,
            )
            metadata = {
                "segment": segment,
                "candidate": candidate.name,
                "cost_bps": cost_bps,
                "selection_uses_outer_test": False,
            }
            rows.append({**metadata, **metrics})
            annual.append(
                _annual_metrics(result, benchmark, metadata)
            )
    return pd.DataFrame(rows), pd.concat(annual, ignore_index=True)


def _selection_table(
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    *,
    candidates: tuple[AuxiliaryTrainingCandidate, ...] = CANDIDATES,
    research_max_drawdown_target_pct: float | None = None,
    research_worst_5d_target_pct: float = -15.0,
    research_minimum_ratio_10bps: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = comparison.loc[
        comparison["candidate"].eq("primary_only")
        & comparison["cost_bps"].eq(10)
    ].iloc[0]
    baseline25 = comparison.loc[
        comparison["candidate"].eq("primary_only")
        & comparison["cost_bps"].eq(25)
    ].iloc[0]
    for candidate in candidates:
        cost10 = comparison.loc[
            comparison["candidate"].eq(candidate.name)
            & comparison["cost_bps"].eq(10)
        ].iloc[0]
        cost25 = comparison.loc[
            comparison["candidate"].eq(candidate.name)
            & comparison["cost_bps"].eq(25)
        ].iloc[0]
        annual10 = annual.loc[
            annual["candidate"].eq(candidate.name)
            & annual["cost_bps"].eq(10)
        ]
        rows.append(
            {
                **cost10.to_dict(),
                "auxiliary_weight": candidate.auxiliary_weight,
                "include_union_backfill": (
                    candidate.include_union_backfill
                ),
                "ratio_10bps": float(
                    cost10["terminal_wealth_ratio_vs_soxx"]
                ),
                "ratio_25bps": float(
                    cost25["terminal_wealth_ratio_vs_soxx"]
                ),
                "positive_relative_years": int(
                    annual10[
                        "terminal_wealth_ratio_vs_soxx"
                    ].gt(1.0).sum()
                ),
                "incremental_ratio_10bps_vs_primary": float(
                    cost10["terminal_wealth_ratio_vs_soxx"]
                    - baseline["terminal_wealth_ratio_vs_soxx"]
                ),
                "incremental_ratio_25bps_vs_primary": float(
                    cost25["terminal_wealth_ratio_vs_soxx"]
                    - baseline25["terminal_wealth_ratio_vs_soxx"]
                ),
            }
        )
    selection = pd.DataFrame(rows)
    selection["return_20pct_gate"] = selection["ratio_10bps"].ge(1.20)
    selection["cost_stress_gate"] = selection["ratio_25bps"].gt(1.05)
    selection["year_gate"] = selection[
        "positive_relative_years"
    ].ge(5)
    selection["matched_increment_gate"] = (
        selection["candidate"].eq("primary_only")
        | (
            selection["incremental_ratio_10bps_vs_primary"].gt(0.0)
            & selection["incremental_ratio_25bps_vs_primary"].gt(0.0)
        )
    )
    selection["maxdd_gate"] = selection[
        "max_drawdown_pct"
    ].ge(-20.0)
    selection["worst_day_gate"] = selection["worst_day_pct"].ge(-12.0)
    selection["cvar_gate"] = selection["cvar5_pct"].ge(-8.0)
    selection["worst_5d_gate"] = selection[
        "worst_rolling_5d_pct"
    ].ge(-12.0)
    selection["cash_gate"] = selection[
        "avg_cash_weight_pct"
    ].ge(15.0)
    selection["capture_gate"] = selection[
        "upside_capture_pct"
    ].ge(60.0)
    relative_gates = [
        "return_20pct_gate",
        "cost_stress_gate",
        "year_gate",
        "matched_increment_gate",
    ]
    risk_gates = [
        "maxdd_gate",
        "worst_day_gate",
        "cvar_gate",
        "worst_5d_gate",
        "cash_gate",
        "capture_gate",
    ]
    selection["relative_gate_count"] = selection[
        relative_gates
    ].sum(axis=1)
    selection["risk_gate_count"] = selection[risk_gates].sum(axis=1)
    selection["relative_objective_pass"] = selection[
        relative_gates
    ].all(axis=1)
    selection["all_gate"] = (
        selection["relative_objective_pass"]
        & selection[risk_gates].all(axis=1)
    )
    selection["research_maxdd_gate"] = (
        True
        if research_max_drawdown_target_pct is None
        else selection["max_drawdown_pct"].ge(
            research_max_drawdown_target_pct
        )
    )
    selection["research_worst_5d_gate"] = selection[
        "worst_rolling_5d_pct"
    ].ge(research_worst_5d_target_pct)
    research_minimum_ratio_25bps = max(
        0.0, research_minimum_ratio_10bps - 0.05
    )
    selection["research_relative_preservation_gate"] = (
        selection["ratio_10bps"].ge(research_minimum_ratio_10bps)
        & selection["ratio_25bps"].ge(research_minimum_ratio_25bps)
    )
    research_gates = [
        "research_maxdd_gate",
        "research_worst_5d_gate",
        "research_relative_preservation_gate",
    ]
    selection["research_gate_count"] = selection[research_gates].sum(axis=1)
    selection["research_target_pass"] = selection[research_gates].all(axis=1)
    selection["selection_score"] = (
        np.log(selection["ratio_10bps"].clip(lower=1e-12))
        + 0.25
        * np.log(selection["ratio_25bps"].clip(lower=1e-12))
        + 0.04 * selection["positive_relative_years"]
        + 0.10
        * selection["incremental_ratio_10bps_vs_primary"]
    )
    if research_max_drawdown_target_pct is not None:
        passing = selection.loc[selection["research_target_pass"]]
        maxdd_passing = selection.loc[selection["research_maxdd_gate"]]
        pool = (
            passing
            if not passing.empty
            else maxdd_passing
            if not maxdd_passing.empty
            else selection
        )
        if not passing.empty:
            selected_index = passing.sort_values(
                [
                    "ratio_10bps",
                    "ratio_25bps",
                    "max_drawdown_pct",
                    "candidate",
                ],
                ascending=[False, False, False, True],
            ).index[0]
        else:
            selected_index = pool.sort_values(
                [
                    "research_gate_count",
                    "max_drawdown_pct",
                    "ratio_10bps",
                    "risk_gate_count",
                    "candidate",
                ],
                ascending=[False, False, False, False, True],
            ).index[0]
    else:
        passing = selection.loc[selection["relative_objective_pass"]]
        pool = passing if not passing.empty else selection
        selected_index = pool.sort_values(
            [
                "relative_gate_count",
                "selection_score",
                "risk_gate_count",
                "candidate",
            ],
            ascending=[False, False, False, True],
        ).index[0]
    selection["selected"] = selection.index == selected_index
    if research_max_drawdown_target_pct is not None:
        selection["selection_policy"] = np.where(
            selection["research_target_pass"],
            "development_drawdown_target_then_relative_preservation",
            "diagnostic_fallback_drawdown_closeness_then_relative_return",
        )
    else:
        selection["selection_policy"] = np.where(
            selection["relative_objective_pass"],
            "development_relative_gates_then_score",
            "diagnostic_fallback_relative_gate_count_then_score",
        )
    return selection


def _selected_candidate(
    selection: pd.DataFrame,
    *,
    candidates: tuple[AuxiliaryTrainingCandidate, ...] = CANDIDATES,
) -> AuxiliaryTrainingCandidate:
    name = str(selection.loc[selection["selected"], "candidate"].iloc[0])
    return next(candidate for candidate in candidates if candidate.name == name)


def _run_card(
    spec: PitAuxiliaryAllocationSpec,
    *,
    candidates: tuple[AuxiliaryTrainingCandidate, ...] = CANDIDATES,
) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    market, source_audit, primary_features = _load_frame(
        DirectContinuousAllocationSpec(
            root=spec.root,
            train_start=spec.train_start,
            development_years=spec.development_years,
            development_end=spec.development_end,
            outer=spec.outer,
            purge_rows=spec.purge_rows,
        )
    )
    primary_dataset, primary_dataset_audit = _build_primary_dataset(
        market, primary_features, spec
    )
    soxx_common = _soxx_common_training_frame(
        spec, end=market.index.max().date().isoformat()
    ).reindex(market.index)
    auxiliary, calibration, auxiliary_audit = _load_auxiliary_frame(spec)
    auxiliary_dataset, auxiliary_dataset_audit = _build_auxiliary_dataset(
        auxiliary, spec
    )

    development_predictions: dict[str, pd.DataFrame] = {}
    outer_predictions: dict[str, pd.DataFrame] = {}
    model_states: dict[str, Any] = {}
    diagnostics: list[pd.DataFrame] = []
    for candidate in candidates:
        development, outer, states, fit = _candidate_predictions(
            primary_dataset=primary_dataset,
            auxiliary_dataset=auxiliary_dataset,
            market=market,
            soxx_common=soxx_common,
            primary_features=primary_features,
            candidate=candidate,
            spec=spec,
        )
        development_predictions[candidate.name] = development
        outer_predictions[candidate.name] = outer
        model_states[candidate.name] = states
        diagnostics.append(fit)

    development_comparison, development_annual = _evaluate_candidates(
        market=market,
        predictions=development_predictions,
        segment="development_walk_forward",
        spec=spec,
        candidates=candidates,
    )
    selection = _selection_table(
        development_comparison,
        development_annual,
        candidates=candidates,
        research_max_drawdown_target_pct=(
            spec.research_max_drawdown_target_pct
        ),
        research_worst_5d_target_pct=spec.research_worst_5d_target_pct,
        research_minimum_ratio_10bps=(
            spec.research_minimum_ratio_10bps
        ),
    )
    selected_candidate = _selected_candidate(
        selection, candidates=candidates
    )
    raw_selected_development = development_predictions[
        selected_candidate.name
    ]
    raw_selected_outer = outer_predictions[selected_candidate.name]
    outer_comparison, outer_annual = _evaluate_candidates(
        market=market,
        predictions={selected_candidate.name: raw_selected_outer},
        segment="outer_diagnostic",
        spec=spec,
        candidates=(selected_candidate,),
    )
    selected_development = _apply_drawdown_budget(
        market=market,
        prediction=raw_selected_development,
        candidate=selected_candidate,
        cost_bps=spec.training_cost_bps,
    )
    selected_outer = _apply_drawdown_budget(
        market=market,
        prediction=raw_selected_outer,
        candidate=selected_candidate,
        cost_bps=spec.training_cost_bps,
    )
    comparison = pd.concat(
        [development_comparison, outer_comparison], ignore_index=True
    )
    annual = pd.concat(
        [development_annual, outer_annual], ignore_index=True
    )
    selected_row = selection.loc[selection["selected"]].iloc[0]
    outer10 = outer_comparison.loc[
        outer_comparison["cost_bps"].eq(10)
    ].iloc[0]
    outer25 = outer_comparison.loc[
        outer_comparison["cost_bps"].eq(25)
    ].iloc[0]
    outer_passing_years = int(
        outer_annual.loc[
            outer_annual["cost_bps"].eq(10),
            "terminal_wealth_ratio_vs_soxx",
        ].gt(1.0).sum()
    )
    objective_audit = pd.DataFrame(
        [
            {
                "gate": "development_20pct_excess_10bps",
                "required": True,
                "passed": bool(selected_row["ratio_10bps"] >= 1.20),
                "evidence": float(selected_row["ratio_10bps"]),
            },
            {
                "gate": "development_25bps_cost_stress",
                "required": True,
                "passed": bool(selected_row["ratio_25bps"] > 1.05),
                "evidence": float(selected_row["ratio_25bps"]),
            },
            {
                "gate": "development_positive_relative_years",
                "required": True,
                "passed": bool(
                    selected_row["positive_relative_years"] >= 5
                ),
                "evidence": int(selected_row["positive_relative_years"]),
            },
            {
                "gate": "outer_20pct_excess_10bps",
                "required": True,
                "passed": bool(
                    outer10["terminal_wealth_ratio_vs_soxx"] >= 1.20
                ),
                "evidence": float(
                    outer10["terminal_wealth_ratio_vs_soxx"]
                ),
            },
            {
                "gate": "outer_25bps_cost_stress",
                "required": True,
                "passed": bool(
                    outer25["terminal_wealth_ratio_vs_soxx"] > 1.05
                ),
                "evidence": float(
                    outer25["terminal_wealth_ratio_vs_soxx"]
                ),
            },
            {
                "gate": "outer_three_of_four_positive_years",
                "required": True,
                "passed": outer_passing_years >= 3,
                "evidence": outer_passing_years,
            },
            {
                "gate": "max_drawdown_minus_20pct",
                "required": True,
                "passed": bool(outer10["max_drawdown_pct"] >= -20.0),
                "evidence": float(outer10["max_drawdown_pct"]),
            },
            {
                "gate": "worst_day_minus_12pct",
                "required": True,
                "passed": bool(outer10["worst_day_pct"] >= -12.0),
                "evidence": float(outer10["worst_day_pct"]),
            },
            {
                "gate": "cvar5_minus_8pct",
                "required": True,
                "passed": bool(outer10["cvar5_pct"] >= -8.0),
                "evidence": float(outer10["cvar5_pct"]),
            },
            {
                "gate": "worst_rolling_5d_minus_12pct",
                "required": True,
                "passed": bool(
                    outer10["worst_rolling_5d_pct"] >= -12.0
                ),
                "evidence": float(outer10["worst_rolling_5d_pct"]),
            },
            {
                "gate": "average_cash_15pct",
                "required": True,
                "passed": bool(
                    outer10["avg_cash_weight_pct"] >= 15.0
                ),
                "evidence": float(outer10["avg_cash_weight_pct"]),
            },
            {
                "gate": "upside_capture_60pct",
                "required": True,
                "passed": bool(
                    outer10["upside_capture_pct"] >= 60.0
                ),
                "evidence": float(outer10["upside_capture_pct"]),
            },
            {
                "gate": "outer_not_used_for_selection",
                "required": True,
                "passed": True,
                "evidence": "development_2016_2022_only",
            },
            {
                "gate": "development_research_maxdrawdown_target",
                "required": False,
                "passed": bool(selected_row["research_maxdd_gate"]),
                "evidence": float(selected_row["max_drawdown_pct"]),
            },
            {
                "gate": "outer_research_maxdrawdown_target",
                "required": False,
                "passed": bool(
                    spec.research_max_drawdown_target_pct is None
                    or outer10["max_drawdown_pct"]
                    >= spec.research_max_drawdown_target_pct
                ),
                "evidence": float(outer10["max_drawdown_pct"]),
            },
            {
                "gate": "outer_research_worst_5d_target",
                "required": False,
                "passed": bool(
                    outer10["worst_rolling_5d_pct"]
                    >= spec.research_worst_5d_target_pct
                ),
                "evidence": float(outer10["worst_rolling_5d_pct"]),
            },
            {
                "gate": "outer_research_relative_preservation",
                "required": False,
                "passed": bool(
                    outer10["terminal_wealth_ratio_vs_soxx"]
                    >= spec.research_minimum_ratio_10bps
                ),
                "evidence": float(
                    outer10["terminal_wealth_ratio_vs_soxx"]
                ),
            },
        ]
    )
    objective_pass = bool(
        objective_audit.loc[
            objective_audit["required"], "passed"
        ].all()
    )
    relative_objective_pass = bool(
        objective_audit.loc[
            objective_audit["gate"].isin(
                [
                    "development_20pct_excess_10bps",
                    "development_25bps_cost_stress",
                    "development_positive_relative_years",
                    "outer_20pct_excess_10bps",
                    "outer_25bps_cost_stress",
                    "outer_three_of_four_positive_years",
                ]
            ),
            "passed",
        ].all()
    )
    research_target_pass = bool(
        selected_row["research_target_pass"]
        and (
            spec.research_max_drawdown_target_pct is None
            or outer10["max_drawdown_pct"]
            >= spec.research_max_drawdown_target_pct
        )
        and outer10["worst_rolling_5d_pct"]
        >= spec.research_worst_5d_target_pct
        and outer10["terminal_wealth_ratio_vs_soxx"]
        >= spec.research_minimum_ratio_10bps
    )
    decision = (
        "validated_pit_auxiliary_training_allocation"
        if objective_pass
        else "research_target_30pct_met_not_promoted"
        if research_target_pass
        else "not_promoted_pit_auxiliary_training_gates_failed"
    )

    predictions = pd.concat(
        [selected_development, selected_outer]
    ).sort_index()
    stored_returns: list[pd.DataFrame] = []
    stored_positions: list[pd.DataFrame] = []
    for segment, prediction in [
        ("development_walk_forward", selected_development),
        ("outer_diagnostic", selected_outer),
    ]:
        for cost_bps in spec.costs_bps:
            _, result, _ = _evaluate(
                market=market,
                predictions=prediction,
                profile=_profile(selected_candidate),
                cost_bps=cost_bps,
            )
            stored_returns.append(
                result.assign(
                    segment=segment,
                    candidate=selected_candidate.name,
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
                    candidate=selected_candidate.name,
                    cost_bps=cost_bps,
                )
            )
    strategy_returns = pd.concat(stored_returns, ignore_index=True)
    positions = pd.concat(stored_positions, ignore_index=True)
    diagnostics_frame = pd.concat(diagnostics, ignore_index=True)

    from qlib.workflow import R

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(
        prefix=f"{spec.experiment}_"
    ) as temporary:
        temp = Path(temporary)
        predictions.reset_index(names="signal_date").to_csv(
            temp / "predictions.csv", index=False
        )
        positions.to_csv(temp / "positions.csv", index=False)
        strategy_returns.to_csv(
            temp / "strategy_returns.csv", index=False
        )
        diagnostics_frame.to_csv(
            temp / "training_diagnostics.csv", index=False
        )
        models_dir = temp / "models"
        models_dir.mkdir()
        for candidate_name, states in model_states.items():
            with (models_dir / f"{candidate_name}.pkl").open("wb") as handle:
                pickle.dump(states, handle)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                benchmark="SOXX",
                assets="SOXL,SOXX,cash",
                allocation_output="direct_legal_weights",
                auxiliary_assets_role="training_rows_only",
                selected_candidate=selected_candidate.name,
                selection_uses_outer_test="false",
            )
            R.log_metrics(
                development_ratio_10bps=float(
                    selected_row["ratio_10bps"]
                ),
                development_ratio_25bps=float(
                    selected_row["ratio_25bps"]
                ),
                outer_ratio_10bps=float(
                    outer10["terminal_wealth_ratio_vs_soxx"]
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
                primary_dataset_audit=primary_dataset_audit,
                auxiliary_audit=auxiliary_audit,
                auxiliary_dataset_audit=auxiliary_dataset_audit,
                primary_features=primary_features,
                calibration=calibration,
                selection=selection,
                selected=selected_row,
                comparison=comparison,
                annual=annual,
                objective_audit=objective_audit,
                objective_pass=objective_pass,
                relative_objective_pass=relative_objective_pass,
                research_target_pass=research_target_pass,
                decision=decision,
                candidates=candidates,
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
                calibration=calibration,
                diagnostics=diagnostics_frame,
                objective_audit=objective_audit,
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
    spec: PitAuxiliaryAllocationSpec,
    source_audit: dict[str, Any],
    primary_dataset_audit: dict[str, Any],
    auxiliary_audit: dict[str, Any],
    auxiliary_dataset_audit: dict[str, Any],
    primary_features: list[str],
    calibration: pd.DataFrame,
    selection: pd.DataFrame,
    selected: pd.Series,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    objective_audit: pd.DataFrame,
    objective_pass: bool,
    relative_objective_pass: bool,
    research_target_pass: bool,
    decision: str,
    candidates: tuple[AuxiliaryTrainingCandidate, ...],
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
) -> dict[str, Any]:
    outer10 = comparison.loc[
        comparison["segment"].eq("outer_diagnostic")
        & comparison["cost_bps"].eq(10)
    ].iloc[0]
    sidecars = {
        "selected_model": "selected_model.json",
        "model_selection": "model_selection.csv",
        "backtest_benchmark_comparison": (
            "backtest_benchmark_comparison.csv"
        ),
        "backtest_walk_forward_metrics": (
            "backtest_walk_forward_metrics.csv"
        ),
        "risk_analysis": "risk_analysis.csv",
        "actual_2x_calibration": "actual_2x_calibration.csv",
        "training_summary": "training_summary.csv",
        "objective_gate_audit": "objective_gate_audit.csv",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
        "conclusion": "conclusion.md",
        "workflow_config": "workflow.yaml",
    }
    return {
        "experiment": spec.experiment,
        "card_type": "qlib_pit_auxiliary_training_research_card",
        "primary_notebook": f"{spec.experiment}.ipynb",
        "system_of_record": "qlib",
        "provider_uri": spec.daily_provider_uri.resolve().as_posix(),
        "auxiliary_provider_uri": (
            spec.auxiliary_provider_uri.resolve().as_posix()
        ),
        "benchmark": "SOXX",
        "assets": ["SOXL", "SOXX", "cash"],
        "architecture": {
            "allocation": (
                "one direct network emits legal SOXL/SOXX/cash weights"
            ),
            "path_risk_control": (
                "the selected candidate may causally scale the model's "
                "full risky allocation from prior realized equity and "
                "running peak; cash remains the residual model state"
            ),
            "tail_encoder": (
                "shared SOXX-style encoder; constituent synthetic-2x "
                "labels enter training loss only"
            ),
            "inference_inputs": (
                "original SOXL/SOXX features plus SOXX OHLCV common "
                "features; no constituent or 2x input at inference"
            ),
            "base_position": None,
            "prior_position": None,
            "hard_constraints": {
                "minimum_cash_weight": spec.minimum_cash_weight,
                "maximum_soxl_weight": spec.maximum_soxl_weight,
                "long_only": True,
                "weights_sum_to_one": True,
                "selected_drawdown_budget": _json_ready(
                    {
                        key: value
                        for key, value in asdict(
                            next(
                                candidate
                                for candidate in candidates
                                if candidate.name
                                == str(selected["candidate"])
                            )
                        ).items()
                        if key.startswith("drawdown_")
                    }
                ),
            },
        },
        "primary_feature_count": len(primary_features),
        "primary_features": primary_features,
        "common_feature_count": len(COMMON_FEATURES),
        "common_features": list(COMMON_FEATURES),
        "training_data_candidates": [
            asdict(candidate) for candidate in candidates
        ],
        "data_audit": {
            "primary": source_audit,
            "primary_dataset": primary_dataset_audit,
            "auxiliary": auxiliary_audit,
            "auxiliary_dataset": auxiliary_dataset_audit,
        },
        "actual_2x_calibration": calibration.to_dict(orient="records"),
        "development": {
            "years": list(spec.development_years),
            "selection_uses_outer_test": False,
            "selected": _json_ready(selected.to_dict()),
        },
        "outer_diagnostic": {
            "range": list(spec.outer),
            "selection_uses_outer_test": False,
            "metrics_10bps": _json_ready(outer10.to_dict()),
        },
        "relative_objective_pass": relative_objective_pass,
        "research_target_pass": research_target_pass,
        "objective_pass": objective_pass,
        "promotion_decision": decision,
        "objective_gate_audit": objective_audit.to_dict(
            orient="records"
        ),
        "model_selection": selection.to_dict(orient="records"),
        "annual_metrics": {
            "sidecar": sidecars["backtest_walk_forward_metrics"],
            "row_count": int(len(annual)),
        },
        "qlib_docs": [
            "https://qlib.readthedocs.io/en/latest/component/data.html",
            "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            "https://qlib.readthedocs.io/en/latest/component/strategy.html",
        ],
        "external_research": [
            {
                "source": (
                    "Goldberg and Mahmoud - Drawdown: From Practice "
                    "to Theory and Back Again"
                ),
                "url": "https://arxiv.org/abs/1404.7493",
                "use": (
                    "conditional expected drawdown captures serially "
                    "clustered path risk"
                ),
            },
            {
                "source": (
                    "Dorador - Constrained Max Drawdown: a Fast and "
                    "Robust Portfolio Optimization Approach"
                ),
                "url": "https://arxiv.org/abs/2401.02601",
                "use": "explicit maximum-drawdown constraint target",
            },
            {
                "source": "DeepDow MaximumDrawdown loss",
                "url": (
                    "https://deepdow.readthedocs.io/en/latest/"
                    "source/losses.html"
                ),
                "use": (
                    "end-to-end differentiable portfolio path loss "
                    "on direct allocation weights"
                ),
            },
            {
                "source": "Kaggle Hull Tactical solution patterns",
                "url": (
                    "https://www.kaggle.com/competitions/"
                    "hull-tactical-market-prediction"
                ),
                "use": (
                    "separate return objective from volatility-risk "
                    "control under time-series validation"
                ),
            },
            {
                "source": "Gu, Kelly, Xiu - Empirical Asset Pricing via Machine Learning",
                "url": "https://www.nber.org/papers/w25398",
                "use": "nonlinear momentum, liquidity and volatility interactions",
            },
            {
                "source": "Kritzman et al. - Principal Components as a Measure of Systemic Risk",
                "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1582687",
                "use": "absorption-ratio systemic stress feature",
            },
            {
                "source": "Kelly and Jiang - Tail Risk and Asset Prices",
                "url": "https://www.nber.org/papers/w19375",
                "use": "cross-sectional firm crashes as common tail-risk evidence",
            },
        ],
        "recorder_id": recorder_id,
        "recorder_uri": recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": list(
                _recorder_only_artifacts(spec.experiment)
            ),
        },
    }


def _write_card(
    *,
    card: Path,
    summary: dict[str, Any],
    selection: pd.DataFrame,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    calibration: pd.DataFrame,
    diagnostics: pd.DataFrame,
    objective_audit: pd.DataFrame,
) -> None:
    selected = selection.loc[selection["selected"]].iloc[0]
    selected_name = str(selected["candidate"])
    outer10 = comparison.loc[
        comparison["segment"].eq("outer_diagnostic")
        & comparison["cost_bps"].eq(10)
    ].iloc[0]
    (card / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8"
    )
    selection.to_csv(card / "model_selection.csv", index=False)
    comparison.to_csv(
        card / "backtest_benchmark_comparison.csv", index=False
    )
    annual.to_csv(
        card / "backtest_walk_forward_metrics.csv", index=False
    )
    calibration.to_csv(card / "actual_2x_calibration.csv", index=False)
    diagnostics.to_csv(card / "training_summary.csv", index=False)
    objective_audit.to_csv(
        card / "objective_gate_audit.csv", index=False
    )
    comparison.loc[
        comparison["candidate"].eq(selected_name)
    ].to_csv(card / "risk_analysis.csv", index=False)
    selected_payload = {
        "candidate": selected_name,
        "selection_uses_outer_test": False,
        "development_metrics": _json_ready(selected.to_dict()),
        "architecture": summary["architecture"],
    }
    (card / "selected_model.json").write_text(
        json.dumps(selected_payload, indent=2), encoding="utf-8"
    )
    analysis = {
        "selected_candidate": selected_name,
        "development_ratio_10bps": float(selected["ratio_10bps"]),
        "development_ratio_25bps": float(selected["ratio_25bps"]),
        "development_positive_relative_years": int(
            selected["positive_relative_years"]
        ),
        "incremental_ratio_10bps_vs_primary": float(
            selected["incremental_ratio_10bps_vs_primary"]
        ),
        "outer_ratio_10bps": float(
            outer10["terminal_wealth_ratio_vs_soxx"]
        ),
        "relative_objective_pass": summary["relative_objective_pass"],
        "research_target_pass": summary["research_target_pass"],
        "objective_pass": summary["objective_pass"],
        "promotion_decision": summary["promotion_decision"],
        "selection_uses_outer_test": False,
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    failed = objective_audit.loc[
        ~objective_audit["passed"], "gate"
    ].tolist()
    (card / "failure_modes.md").write_text(
        "# Failure modes\n\n"
        + (
            "\n".join(f"- {gate}" for gate in failed)
            if failed
            else "- none"
        )
        + "\n",
        encoding="utf-8",
    )
    conclusion = (
        "# 结论\n\n"
        "- what_ran: 成分股与合成2x标签只进入训练辅助头；开发期和"
        "outer推理均只使用原有SOXL/SOXX数据，网络直接输出完整仓位，"
        "路径预算候选仅根据此前已实现净值与峰值因果缩放风险仓位。\n"
        f"- key_metrics: 开发期选择 `{selected_name}`，10bp相对SOXX="
        f"{float(selected['ratio_10bps']):.4f}，25bp="
        f"{float(selected['ratio_25bps']):.4f}，正超额年份="
        f"{int(selected['positive_relative_years'])}/7；outer 10bp="
        f"{float(outer10['terminal_wealth_ratio_vs_soxx']):.4f}，"
        f"MaxDD={float(outer10['max_drawdown_pct']):.2f}%。\n"
        "- regime_window: SEC filing_date 后使用PIT成员；更早历史并非PIT，"
        "只在单独候选中以0.25权重作为预训练样本。\n"
        f"- promotion_decision: `{summary['promotion_decision']}`；"
        f"相对目标通过={summary['relative_objective_pass']}，完整风险门="
        f"{summary['objective_pass']}，30%研究目标="
        f"{summary['research_target_pass']}。\n"
        f"- risks: 失败门为 {', '.join(failed) if failed else '无'}；"
        "实际单股2x历史太短，只用于校准合成标签，不伪造早期可交易历史。\n"
        "- next_action: 30%路径风险目标与跑赢SOXX目标分开判定；"
        "只有开发期先达到目标后才解释outer诊断，不用outer反选。\n"
        f"- recorder: `{summary['recorder_id']}`。\n"
    )
    (card / "conclusion.md").write_text(
        conclusion, encoding="utf-8"
    )
    workflow = (
        f"experiment: {summary['experiment']}\n"
        f"provider_uri: {summary['provider_uri']}\n"
        f"auxiliary_provider_uri: {summary['auxiliary_provider_uri']}\n"
        f"recorder_uri: {summary['recorder_uri']}\n"
        "market: US\n"
        "region: us\n"
        "assets: [SOXL, SOXX, cash]\n"
        "auxiliary_assets_role: training_rows_only\n"
        "allocation_output: direct_full_weights\n"
        "selection_uses_outer_test: false\n"
        "artifact_policy:\n"
        "  policy: slim_card_recorder_first\n"
        "  workflow_config: workflow.yaml\n"
        "  recorder_only_artifacts:\n"
        + "\n".join(
            f"    - {artifact}"
            for artifact in summary["artifact_policy"][
                "recorder_only_artifacts"
            ]
        )
        + "\n"
    )
    (card / "workflow.yaml").write_text(
        workflow, encoding="utf-8"
    )
    _write_notebook(card, summary)


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    drawdown_experiment = summary["experiment"] == DRAWDOWN_EXPERIMENT
    runner = (
        "run_drawdown_targeted_pit_allocation_card"
        if drawdown_experiment
        else "run_pit_constituent_auxiliary_training_allocation_card"
    )
    title = (
        "# Drawdown-targeted PIT allocation"
        if drawdown_experiment
        else "# PIT constituent auxiliary training allocation"
    )
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            f"{title}\n\n"
            "成分股与2x标签只参与训练；推理仍只依赖SOXL/SOXX。"
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "import qlib\n"
            "from qlib.data import D\n"
            "from qlib.data.dataset import DatasetH\n"
            "from qlib.workflow import R\n"
            "repo_root = Path.cwd().parents[1]\n"
            "if str(repo_root) not in sys.path:\n"
            "    sys.path.insert(0, str(repo_root))\n"
            "qlib.init(provider_uri=str(repo_root / 'data/qlib/soxl'), "
            "region='us', kernels=1, skip_if_reg=False)\n"
            "calendar = D.calendar(start_time='2011-01-03', "
            "end_time='2026-07-02', freq='day')\n"
            "from src.workflow import "
            f"{runner}\n"
            "summary = "
            f"{runner}("
            "root=repo_root)\n"
            "summary['promotion_decision']"
        ),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "card = Path.cwd()\n"
            "pd.read_csv(card / 'model_selection.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'actual_2x_calibration.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'backtest_walk_forward_metrics.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'objective_gate_audit.csv')"
        ),
        nbformat.v4.new_markdown_cell(
            (card / "conclusion.md").read_text(encoding="utf-8")
        ),
    ]
    nbformat.write(notebook, card / f"{summary['experiment']}.ipynb")
