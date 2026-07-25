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

from src.workflow.cash_floor_benchmark_relative_wf import _result_metrics
from src.workflow.cash_floor_relative_feature_reintegration import (
    CashFloorRelativeFeatureSpec,
    _load_reintegrated_frame,
)
from src.workflow.hierarchical_soxx_excess_walk_forward import (
    ALPHA_FEATURES,
    CASH_FEATURES,
    HAZARD_FEATURES,
    _drawdown_duration,
)
from src.workflow.soxl_objective_feasibility_audit import _json_ready, _simulate
from src.workflow.trainable_regime_policy_nested_wf import (
    _remove_qlib_code_snapshot_artifacts,
)


EXPERIMENT = "direct_continuous_soxx_allocation_v1"
DEVELOPMENT_YEARS = tuple(range(2016, 2023))
OUTER_START = "2023-01-03"
DIRECT_FEATURES = tuple(
    dict.fromkeys([*ALPHA_FEATURES, *HAZARD_FEATURES, *CASH_FEATURES])
)
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/predictions.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/training_diagnostics.csv",
)


@dataclass(frozen=True)
class DirectLossProfile:
    name: str
    maximum_soxl_weight: float
    negative_year_penalty: float
    year_dispersion_penalty: float
    relative_drawdown_penalty: float
    turnover_penalty: float
    cash_supervision_penalty: float = 0.0
    tail_soxl_penalty: float = 0.0
    cvar_penalty: float = 0.0


PROFILES = (
    DirectLossProfile(
        "return_stability",
        maximum_soxl_weight=0.30,
        negative_year_penalty=1.0,
        year_dispersion_penalty=0.50,
        relative_drawdown_penalty=0.002,
        turnover_penalty=0.002,
    ),
    DirectLossProfile(
        "turnover_stability",
        maximum_soxl_weight=0.30,
        negative_year_penalty=1.0,
        year_dispersion_penalty=0.50,
        relative_drawdown_penalty=0.004,
        turnover_penalty=0.006,
    ),
    DirectLossProfile(
        "tail_supervised",
        maximum_soxl_weight=0.30,
        negative_year_penalty=1.0,
        year_dispersion_penalty=0.50,
        relative_drawdown_penalty=0.004,
        turnover_penalty=0.006,
        cash_supervision_penalty=0.020,
        tail_soxl_penalty=0.020,
        cvar_penalty=0.002,
    ),
    DirectLossProfile(
        "low_leverage",
        maximum_soxl_weight=0.20,
        negative_year_penalty=0.50,
        year_dispersion_penalty=0.20,
        relative_drawdown_penalty=0.001,
        turnover_penalty=0.001,
    ),
)


@dataclass(frozen=True)
class DirectContinuousAllocationSpec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    development_years: tuple[int, ...] = DEVELOPMENT_YEARS
    development_end: str = "2022-12-30"
    outer: tuple[str, str] = (OUTER_START, "2026-12-31")
    purge_rows: int = 20
    hidden_units: int = 16
    epochs: int = 700
    learning_rate: float = 0.002
    weight_decay: float = 0.08
    training_cost_bps: int = 10
    costs_bps: tuple[int, ...] = (10, 25)
    seed: int = 17

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def daily_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"


class DirectContinuousAllocationNet(nn.Module):
    """Small supervised network whose primary output is the full allocation."""

    def __init__(self, feature_count: int, hidden_units: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def run_direct_continuous_soxx_allocation_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    return _run_card(DirectContinuousAllocationSpec(root=repo_root))


def _attach_direct_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    soxx = pd.to_numeric(
        result["soxx_forward_open_return"], errors="coerce"
    )
    forward = pd.concat([soxx.shift(-offset) for offset in range(5)], axis=1)
    path = (1.0 + forward).cumprod(axis=1) - 1.0
    worst_path = path.min(axis=1)
    result["tail_event_5d"] = worst_path.le(-0.06).astype(float)
    result["tail_cash_target_5d"] = 0.55 / (
        1.0 + np.exp((worst_path + 0.04) / 0.015)
    )
    result.loc[forward.isna().any(axis=1), "tail_event_5d"] = np.nan
    result.loc[
        forward.isna().any(axis=1), "tail_cash_target_5d"
    ] = np.nan
    return result


def _load_frame(
    spec: DirectContinuousAllocationSpec,
    *,
    include_incomplete_latest: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    source_spec = CashFloorRelativeFeatureSpec(root=spec.root)
    frame, source_audit, _, source_manifest = _load_reintegrated_frame(
        source_spec,
        include_incomplete_latest=include_incomplete_latest,
    )
    frame = _attach_direct_targets(frame)
    features = [column for column in DIRECT_FEATURES if column in frame]
    if not features:
        raise ValueError("blocked: no direct allocation features available")
    forbidden = {
        "base_constrained_ref",
        "prior_share",
        "prior_weight",
        "cash_rule",
    }
    if forbidden.intersection(features):
        raise ValueError("blocked: prior or rule feature entered direct model")
    manifest = source_manifest.loc[
        source_manifest["feature"].isin(features)
    ].copy()
    audit = {
        **source_audit,
        "feature_count": len(features),
        "feature_families": sorted(manifest["family"].unique().tolist()),
        "base_constrained_ref_in_features": False,
        "base_constrained_ref_role": "none",
        "candidate_weight_grid": False,
        "allocation_output": "direct_continuous_softmax_soxl_soxx_cash",
        "maximum_soxl_weight_is_risk_constraint": True,
        "selection_uses_outer_test": False,
        "include_incomplete_latest": bool(include_incomplete_latest),
    }
    return frame, audit, features


def _build_dataset(
    frame: pd.DataFrame,
    features: list[str],
    spec: DirectContinuousAllocationSpec,
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    labels = [
        "soxl_forward_open_return",
        "soxx_forward_open_return",
        "tail_event_5d",
        "tail_cash_target_5d",
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
        "target_timing": (
            "signal T close; allocation earns T+1 open to T+2 open; "
            "tail auxiliary target uses next five open-to-open returns"
        ),
    }
    return dataset, audit


def _prepare(dataset, segment) -> pd.DataFrame:
    result = dataset.prepare(segment, col_set=["feature", "label"])
    result = result.droplevel("instrument").sort_index()
    result.columns = [str(column[-1]) for column in result.columns]
    return result


def _project_weights_tensor(
    logits: torch.Tensor, maximum_soxl_weight: float
) -> torch.Tensor:
    probabilities = torch.softmax(torch.clamp(logits, -8.0, 8.0), dim=1)
    excess = torch.relu(probabilities[:, 0] - maximum_soxl_weight)
    soxl = probabilities[:, 0] - excess
    denominator = probabilities[:, 1] + probabilities[:, 2] + 1e-8
    soxx = probabilities[:, 1] + excess * probabilities[:, 1] / denominator
    cash = probabilities[:, 2] + excess * probabilities[:, 2] / denominator
    weights = torch.stack([soxl, soxx, cash], dim=1)
    return weights / weights.sum(dim=1, keepdim=True)


def _standardize(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, pd.Series]]:
    median = train.loc[:, features].median()
    train_values = train.loc[:, features].fillna(median)
    test_values = test.loc[:, features].fillna(median)
    mean = train_values.mean()
    scale = train_values.std(ddof=1).replace(0.0, 1.0).fillna(1.0)
    train_scaled = ((train_values - mean) / scale).clip(-8.0, 8.0)
    test_scaled = ((test_values - mean) / scale).clip(-8.0, 8.0)
    return (
        torch.tensor(train_scaled.to_numpy(), dtype=torch.float32),
        torch.tensor(test_scaled.to_numpy(), dtype=torch.float32),
        {"median": median, "mean": mean, "scale": scale},
    )


def _training_loss(
    *,
    weights: torch.Tensor,
    returns: torch.Tensor,
    years: torch.Tensor,
    cash_target: torch.Tensor,
    tail_event: torch.Tensor,
    profile: DirectLossProfile,
    cost_bps: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    initial = torch.tensor(
        [0.0, 0.0, 1.0], dtype=weights.dtype, device=weights.device
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
    benchmark_log = torch.log1p(torch.clamp(returns[:, 1], min=-0.95))
    relative_log = portfolio_log - benchmark_log

    annual_relative: list[torch.Tensor] = []
    relative_drawdown: list[torch.Tensor] = []
    for year in torch.unique(years):
        mask = years.eq(year)
        annual_relative.append(relative_log[mask].mean())
        portfolio_path = torch.cumsum(portfolio_log[mask], dim=0)
        benchmark_path = torch.cumsum(benchmark_log[mask], dim=0)
        portfolio_dd = torch.cummax(
            portfolio_path, dim=0
        ).values - portfolio_path
        benchmark_dd = torch.cummax(
            benchmark_path, dim=0
        ).values - benchmark_path
        relative_drawdown.append(
            torch.relu(portfolio_dd - benchmark_dd).square().mean()
        )
    annual = torch.stack(annual_relative)
    drawdown_loss = torch.stack(relative_drawdown).mean()
    worst_count = max(1, int(0.05 * len(portfolio_return)))
    cvar_loss = torch.topk(-portfolio_return, worst_count).values.mean()
    cash_loss = (weights[:, 2] - cash_target).square().mean()
    tail_soxl_loss = (
        tail_event * torch.relu(weights[:, 0] - 0.10).square()
    ).mean()
    loss = (
        -annual.mean()
        + profile.negative_year_penalty * torch.relu(-annual).mean()
        + profile.year_dispersion_penalty * annual.std(unbiased=False)
        + profile.relative_drawdown_penalty * drawdown_loss
        + profile.turnover_penalty * turnover.mean()
        + profile.cash_supervision_penalty * cash_loss
        + profile.tail_soxl_penalty * tail_soxl_loss
        + profile.cvar_penalty * cvar_loss
    )
    diagnostics = {
        "loss": float(loss.detach()),
        "mean_annual_relative_log": float(annual.mean().detach()),
        "negative_year_loss": float(torch.relu(-annual).mean().detach()),
        "year_dispersion": float(annual.std(unbiased=False).detach()),
        "relative_drawdown_loss": float(drawdown_loss.detach()),
        "mean_turnover": float(turnover.mean().detach()),
        "cash_supervision_loss": float(cash_loss.detach()),
        "tail_soxl_loss": float(tail_soxl_loss.detach()),
        "cvar_loss": float(cvar_loss.detach()),
    }
    return loss, diagnostics


def _fit_predict(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    profile: DirectLossProfile,
    spec: DirectContinuousAllocationSpec,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    required = [
        "soxl_forward_open_return",
        "soxx_forward_open_return",
        "tail_event_5d",
        "tail_cash_target_5d",
    ]
    train = train.dropna(subset=required).copy()
    test = test.dropna(subset=required).copy()
    if train.empty or test.empty:
        raise ValueError("blocked: empty direct allocation train or test data")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, test_x, scaler = _standardize(train, test, features)
    returns = torch.tensor(
        train.loc[
            :, ["soxl_forward_open_return", "soxx_forward_open_return"]
        ].to_numpy(),
        dtype=torch.float32,
    )
    years = torch.tensor(train.index.year.to_numpy(), dtype=torch.int64)
    cash_target = torch.tensor(
        train["tail_cash_target_5d"].to_numpy(), dtype=torch.float32
    )
    tail_event = torch.tensor(
        train["tail_event_5d"].to_numpy(), dtype=torch.float32
    )
    model = DirectContinuousAllocationNet(
        feature_count=len(features), hidden_units=spec.hidden_units
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
        weights = _project_weights_tensor(
            model(train_x), profile.maximum_soxl_weight
        )
        loss, diagnostics = _training_loss(
            weights=weights,
            returns=returns,
            years=years,
            cash_target=cash_target,
            tail_event=tail_event,
            profile=profile,
            cost_bps=spec.training_cost_bps,
        )
        if not torch.isfinite(loss):
            raise ValueError(
                f"blocked: non-finite direct allocation loss for {profile.name}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        predicted = _project_weights_tensor(
            model(test_x), profile.maximum_soxl_weight
        ).cpu().numpy()
    predicted = np.clip(predicted, 0.0, 1.0)
    predicted = predicted / predicted.sum(axis=1, keepdims=True)
    state = {
        "model_state": {
            key: value.detach().cpu().numpy()
            for key, value in model.state_dict().items()
        },
        "scaler": {
            name: series.to_dict() for name, series in scaler.items()
        },
        "features": features,
        "profile": asdict(profile),
        "spec": {
            "hidden_units": spec.hidden_units,
            "epochs": spec.epochs,
            "learning_rate": spec.learning_rate,
            "weight_decay": spec.weight_decay,
            "training_cost_bps": spec.training_cost_bps,
        },
    }
    return predicted, state, diagnostics


def _profile_walk_forward(
    *,
    dataset,
    market: pd.DataFrame,
    features: list[str],
    profile: DirectLossProfile,
    spec: DirectContinuousAllocationSpec,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    states: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for year in spec.development_years:
        test = _prepare(dataset, f"fold_{year}")
        test = test.loc[test.index.year == year]
        start_position = market.index.searchsorted(
            pd.Timestamp(f"{year}-01-01")
        )
        train_end_position = start_position - spec.purge_rows - 1
        if train_end_position < 0:
            raise ValueError(
                f"blocked: insufficient history for direct fold {year}"
            )
        train_end = market.index[train_end_position]
        train = _prepare(dataset, (spec.train_start, train_end))
        weights, state, fit_diagnostics = _fit_predict(
            train=train,
            test=test,
            features=features,
            profile=profile,
            spec=spec,
            seed=spec.seed + year,
        )
        output = pd.DataFrame(
            weights,
            index=test.dropna(
                subset=[
                    "soxl_forward_open_return",
                    "soxx_forward_open_return",
                    "tail_event_5d",
                    "tail_cash_target_5d",
                ]
            ).index,
            columns=["soxl_weight", "soxx_weight", "cash_weight"],
        )
        output["segment"] = "development_walk_forward"
        output["fold_year"] = year
        output["profile"] = profile.name
        output["train_end"] = train_end
        predictions.append(output)
        states[f"fold_{year}"] = state
        diagnostics.append(
            {
                "profile": profile.name,
                "fold_year": year,
                "train_end": train_end.date().isoformat(),
                "train_rows": int(len(train)),
                "test_rows": int(len(output)),
                "selection_uses_outer_test": False,
                **fit_diagnostics,
            }
        )
    return (
        pd.concat(predictions).sort_index(),
        states,
        pd.DataFrame(diagnostics),
    )


def _outer_predict(
    *,
    dataset,
    features: list[str],
    profile: DirectLossProfile,
    spec: DirectContinuousAllocationSpec,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    train = _prepare(dataset, (spec.train_start, spec.development_end))
    test = _prepare(dataset, "outer")
    weights, state, diagnostics = _fit_predict(
        train=train,
        test=test,
        features=features,
        profile=profile,
        spec=spec,
        seed=spec.seed,
    )
    usable = test.dropna(
        subset=[
            "soxl_forward_open_return",
            "soxx_forward_open_return",
            "tail_event_5d",
            "tail_cash_target_5d",
        ]
    )
    output = pd.DataFrame(
        weights,
        index=usable.index,
        columns=["soxl_weight", "soxx_weight", "cash_weight"],
    )
    output["segment"] = "outer_diagnostic"
    output["fold_year"] = output.index.year
    output["profile"] = profile.name
    output["train_end"] = pd.Timestamp(spec.development_end)
    return output, state, diagnostics


def _evaluate(
    *,
    market: pd.DataFrame,
    predictions: pd.DataFrame,
    profile: DirectLossProfile,
    cost_bps: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    aligned = market.loc[predictions.index]
    direct = predictions.loc[
        :, ["soxl_weight", "soxx_weight", "cash_weight"]
    ].to_numpy(dtype=float)
    weights = np.column_stack(
        [direct[:, 0], direct[:, 1], np.zeros(len(direct)), direct[:, 2]]
    )
    result = _simulate(
        aligned,
        weights,
        cost_bps=cost_bps,
        name=profile.name,
    )
    benchmark = _simulate(
        aligned,
        np.tile([0.0, 1.0, 0.0, 0.0], (len(aligned), 1)),
        cost_bps=cost_bps,
        name="soxx_buyhold",
    )
    metrics = _result_metrics(result, benchmark)
    benchmark_metrics = _result_metrics(benchmark, benchmark)
    duration = _drawdown_duration(result["strategy_return"])
    benchmark_duration = _drawdown_duration(benchmark["strategy_return"])
    downside = pd.to_numeric(
        aligned["soxx_trend60"], errors="coerce"
    ).lt(0.0).to_numpy()
    metrics.update(
        {
            **duration,
            "soxx_max_drawdown_pct": benchmark_metrics[
                "max_drawdown_pct"
            ],
            "soxx_longest_underwater_days": benchmark_duration[
                "longest_underwater_days"
            ],
            "soxx_max_drawdown_recovery_days": benchmark_duration[
                "max_drawdown_recovery_days"
            ],
            "downside_avg_cash_weight_pct": 100.0
            * float(result.loc[downside, "cash_weight"].mean()),
            "downside_min_cash_weight_pct": 100.0
            * float(result.loc[downside, "cash_weight"].min()),
            "turnover": float(result["turnover"].sum()),
        }
    )
    return metrics, result, benchmark


def _annual_metrics(
    result: pd.DataFrame,
    benchmark: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    years = pd.to_datetime(result["signal_date"]).dt.year
    for year in sorted(years.unique()):
        mask = years.eq(year)
        rows.append(
            {
                "fold_year": int(year),
                **metadata,
                **_result_metrics(
                    result.loc[mask].reset_index(drop=True),
                    benchmark.loc[mask].reset_index(drop=True),
                ),
            }
        )
    return pd.DataFrame(rows)


def _development_selection(
    *,
    market: pd.DataFrame,
    profile_predictions: dict[str, pd.DataFrame],
    spec: DirectContinuousAllocationSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    annual_frames: list[pd.DataFrame] = []
    for profile in PROFILES:
        predictions = profile_predictions[profile.name]
        cost_metrics: dict[int, dict[str, Any]] = {}
        for cost_bps in spec.costs_bps:
            metrics, result, benchmark = _evaluate(
                market=market,
                predictions=predictions,
                profile=profile,
                cost_bps=cost_bps,
            )
            cost_metrics[cost_bps] = metrics
            if cost_bps == 10:
                annual_frames.append(
                    _annual_metrics(
                        result,
                        benchmark,
                        {
                            "segment": "development_walk_forward",
                            "profile": profile.name,
                            "cost_bps": cost_bps,
                            "selection_uses_outer_test": False,
                        },
                    )
                )
        base = {
            "profile": profile.name,
            "selection_uses_outer_test": False,
            **cost_metrics[10],
            "ratio_10bps": cost_metrics[10][
                "terminal_wealth_ratio_vs_soxx"
            ],
            "ratio_25bps": cost_metrics[25][
                "terminal_wealth_ratio_vs_soxx"
            ],
            "max_drawdown_25bps_pct": cost_metrics[25][
                "max_drawdown_pct"
            ],
        }
        rows.append(base)
    selection = pd.DataFrame(rows)
    annual = pd.concat(annual_frames, ignore_index=True)
    annual_summary = (
        annual.groupby("profile")["terminal_wealth_ratio_vs_soxx"]
        .agg(
            positive_relative_years=lambda values: int(values.gt(1.0).sum()),
            median_annual_ratio="median",
            minimum_annual_ratio="min",
        )
        .reset_index()
    )
    selection = selection.merge(annual_summary, on="profile", how="left")
    selection["return_gate"] = selection["ratio_10bps"].ge(1.20)
    selection["cost_stress_gate"] = selection["ratio_25bps"].gt(1.05)
    selection["year_gate"] = selection["positive_relative_years"].ge(5)
    selection["drawdown_gate"] = selection["max_drawdown_pct"].gt(
        selection["soxx_max_drawdown_pct"]
    )
    selection["recovery_gate"] = (
        selection["longest_underwater_days"].lt(
            selection["soxx_longest_underwater_days"]
        )
        & selection["max_drawdown_recovery_days"].le(
            selection["soxx_max_drawdown_recovery_days"]
        )
    )
    selection["cash_gate"] = selection[
        "downside_avg_cash_weight_pct"
    ].gt(selection["avg_cash_weight_pct"])
    gate_columns = [
        "return_gate",
        "cost_stress_gate",
        "year_gate",
        "drawdown_gate",
        "recovery_gate",
        "cash_gate",
    ]
    selection["gate_count"] = selection[gate_columns].sum(axis=1)
    selection["all_gate"] = selection[gate_columns].all(axis=1)
    selection["selection_score"] = (
        np.log(selection["ratio_10bps"].clip(lower=1e-12))
        + 0.25 * np.log(selection["ratio_25bps"].clip(lower=1e-12))
        + 0.05 * selection["positive_relative_years"]
        + 0.02
        * (
            selection["max_drawdown_pct"]
            - selection["soxx_max_drawdown_pct"]
        )
    )
    passing = selection.loc[selection["all_gate"]]
    pool = passing if not passing.empty else selection
    selected_index = pool.sort_values(
        ["gate_count", "selection_score", "profile"],
        ascending=[False, False, True],
    ).index[0]
    selection["selected"] = selection.index == selected_index
    selection["selection_policy"] = np.where(
        selection["all_gate"],
        "all_required_development_gates_then_score",
        "diagnostic_fallback_maximum_gate_count_then_score",
    )
    return selection, annual


def _selected_profile(selection: pd.DataFrame) -> DirectLossProfile:
    name = str(selection.loc[selection["selected"], "profile"].iloc[0])
    return next(profile for profile in PROFILES if profile.name == name)


def _regime_metrics(
    market: pd.DataFrame, result: pd.DataFrame
) -> pd.DataFrame:
    aligned = market.loc[pd.to_datetime(result["signal_date"])].copy()
    regimes = {
        "all": np.ones(len(result), dtype=bool),
        "uptrend": pd.to_numeric(
            aligned["soxx_trend60"], errors="coerce"
        ).ge(0.0).to_numpy(),
        "downtrend": pd.to_numeric(
            aligned["soxx_trend60"], errors="coerce"
        ).lt(0.0).to_numpy(),
        "high_volatility": pd.to_numeric(
            aligned["soxx_vol20"], errors="coerce"
        )
        .ge(
            pd.to_numeric(
                aligned["soxx_vol20"], errors="coerce"
            ).median()
        )
        .to_numpy(),
    }
    rows: list[dict[str, Any]] = []
    for regime, mask in regimes.items():
        group = result.loc[mask]
        strategy_wealth = float((1.0 + group["strategy_return"]).prod())
        benchmark_wealth = float((1.0 + group["soxx_return"]).prod())
        rows.append(
            {
                "regime": regime,
                "rows": int(mask.sum()),
                "terminal_wealth_ratio_vs_soxx": strategy_wealth
                / benchmark_wealth,
                "avg_soxl_weight_pct": 100.0
                * float(group["soxl_weight"].mean()),
                "avg_soxx_weight_pct": 100.0
                * float(group["soxx_weight"].mean()),
                "avg_cash_weight_pct": 100.0
                * float(group["cash_weight"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _objective_audit(
    *,
    selected: pd.Series,
    outer: pd.Series,
    features: list[str],
) -> pd.DataFrame:
    rows = [
        (
            "no_base_or_prior",
            True,
            "base_constrained_ref" not in features,
            "base_constrained_ref_role=none",
        ),
        (
            "no_candidate_weight_grid",
            True,
            True,
            "network emits three continuous logits and softmax weights",
        ),
        (
            "outer_not_used_for_selection",
            True,
            not bool(selected["selection_uses_outer_test"]),
            "2016_2022_development_only",
        ),
        (
            "development_20pct_excess_10bps",
            True,
            bool(selected["ratio_10bps"] >= 1.20),
            selected["ratio_10bps"],
        ),
        (
            "development_25bps_stress",
            True,
            bool(selected["ratio_25bps"] > 1.05),
            selected["ratio_25bps"],
        ),
        (
            "development_year_stability",
            True,
            bool(selected["positive_relative_years"] >= 5),
            selected["positive_relative_years"],
        ),
        (
            "development_maxdd_better_than_soxx",
            True,
            bool(
                selected["max_drawdown_pct"]
                > selected["soxx_max_drawdown_pct"]
            ),
            (
                f"{selected['max_drawdown_pct']:.6f} vs "
                f"{selected['soxx_max_drawdown_pct']:.6f}"
            ),
        ),
        (
            "development_recovery_better_than_soxx",
            True,
            bool(selected["recovery_gate"]),
            (
                f"underwater {selected['longest_underwater_days']} vs "
                f"{selected['soxx_longest_underwater_days']}; "
                f"recovery {selected['max_drawdown_recovery_days']} vs "
                f"{selected['soxx_max_drawdown_recovery_days']}"
            ),
        ),
        (
            "development_model_uses_cash_in_downside",
            True,
            bool(
                selected["downside_avg_cash_weight_pct"]
                > selected["avg_cash_weight_pct"]
            ),
            (
                f"{selected['downside_avg_cash_weight_pct']:.6f} vs "
                f"{selected['avg_cash_weight_pct']:.6f}"
            ),
        ),
        (
            "outer_20pct_excess_10bps",
            True,
            bool(outer["terminal_wealth_ratio_vs_soxx"] >= 1.20),
            outer["terminal_wealth_ratio_vs_soxx"],
        ),
        (
            "outer_maxdd_better_than_soxx",
            True,
            bool(
                outer["max_drawdown_pct"]
                > outer["soxx_max_drawdown_pct"]
            ),
            (
                f"{outer['max_drawdown_pct']:.6f} vs "
                f"{outer['soxx_max_drawdown_pct']:.6f}"
            ),
        ),
        (
            "outer_recovery_better_than_soxx",
            True,
            bool(
                outer["longest_underwater_days"]
                < outer["soxx_longest_underwater_days"]
                and outer["max_drawdown_recovery_days"]
                <= outer["soxx_max_drawdown_recovery_days"]
            ),
            (
                f"underwater {outer['longest_underwater_days']} vs "
                f"{outer['soxx_longest_underwater_days']}; "
                f"recovery {outer['max_drawdown_recovery_days']} vs "
                f"{outer['soxx_max_drawdown_recovery_days']}"
            ),
        ),
        (
            "outer_model_uses_cash_in_downside",
            True,
            bool(
                outer["downside_avg_cash_weight_pct"]
                > outer["avg_cash_weight_pct"]
            ),
            (
                f"{outer['downside_avg_cash_weight_pct']:.6f} vs "
                f"{outer['avg_cash_weight_pct']:.6f}"
            ),
        ),
        (
            "legacy_absolute_worst_5d_minus_12pct",
            False,
            bool(outer["worst_rolling_5d_pct"] >= -12.0),
            outer["worst_rolling_5d_pct"],
        ),
    ]
    return pd.DataFrame(
        rows, columns=["gate", "required", "passed", "evidence"]
    )


def _run_card(spec: DirectContinuousAllocationSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    frame, source_audit, features = _load_frame(spec)
    dataset, dataset_audit = _build_dataset(frame, features, spec)

    profile_predictions: dict[str, pd.DataFrame] = {}
    model_states: dict[str, Any] = {}
    training_diagnostics: list[pd.DataFrame] = []
    for profile in PROFILES:
        predictions, states, diagnostics = _profile_walk_forward(
            dataset=dataset,
            market=frame,
            features=features,
            profile=profile,
            spec=spec,
        )
        profile_predictions[profile.name] = predictions
        model_states[profile.name] = states
        training_diagnostics.append(diagnostics)

    selection, development_annual = _development_selection(
        market=frame,
        profile_predictions=profile_predictions,
        spec=spec,
    )
    selected_profile = _selected_profile(selection)
    selected_row = selection.loc[selection["selected"]].iloc[0]
    development_predictions = profile_predictions[selected_profile.name]
    outer_predictions, outer_state, outer_fit = _outer_predict(
        dataset=dataset,
        features=features,
        profile=selected_profile,
        spec=spec,
    )
    model_states[selected_profile.name]["outer_refit"] = outer_state
    training_diagnostics.append(
        pd.DataFrame(
            [
                {
                    "profile": selected_profile.name,
                    "fold_year": "outer_refit",
                    "train_end": spec.development_end,
                    "train_rows": int(
                        len(
                            _prepare(
                                dataset,
                                (spec.train_start, spec.development_end),
                            )
                        )
                    ),
                    "test_rows": int(len(outer_predictions)),
                    "selection_uses_outer_test": False,
                    **outer_fit,
                }
            ]
        )
    )
    all_predictions = pd.concat(
        [development_predictions, outer_predictions]
    ).sort_index()

    comparison_rows: list[dict[str, Any]] = []
    annual_frames: list[pd.DataFrame] = []
    stored_returns: list[pd.DataFrame] = []
    stored_positions: list[pd.DataFrame] = []
    segment_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for segment, predictions in (
        ("development_walk_forward", development_predictions),
        ("outer_diagnostic", outer_predictions),
    ):
        for cost_bps in spec.costs_bps:
            metrics, result, benchmark = _evaluate(
                market=frame,
                predictions=predictions,
                profile=selected_profile,
                cost_bps=cost_bps,
            )
            comparison_rows.append(
                {
                    "segment": segment,
                    "profile": selected_profile.name,
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
                        "profile": selected_profile.name,
                        "cost_bps": cost_bps,
                        "selection_uses_outer_test": False,
                    },
                )
            )
            stored_returns.append(
                result.assign(
                    segment=segment,
                    profile=selected_profile.name,
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
                    profile=selected_profile.name,
                    cost_bps=cost_bps,
                )
            )
            if cost_bps == 10:
                segment_results[segment] = (result, benchmark)
    comparison = pd.DataFrame(comparison_rows)
    annual = pd.concat(annual_frames, ignore_index=True)
    strategy_returns = pd.concat(stored_returns, ignore_index=True)
    positions = pd.concat(stored_positions, ignore_index=True)
    diagnostics = pd.concat(training_diagnostics, ignore_index=True)
    outer10 = comparison.loc[
        comparison["segment"].eq("outer_diagnostic")
        & comparison["cost_bps"].eq(10)
    ].iloc[0]
    objective_audit = _objective_audit(
        selected=selected_row, outer=outer10, features=features
    )
    required = objective_audit.loc[objective_audit["required"]]
    objective_pass = bool(required["passed"].all())
    decision = (
        "validated_research_direct_continuous_allocation"
        if objective_pass
        else "needs_iteration_direct_continuous_allocation"
    )
    regimes = _regime_metrics(
        frame, segment_results["outer_diagnostic"][0]
    )

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
        strategy_returns.to_csv(temp / "strategy_returns.csv", index=False)
        diagnostics.to_csv(temp / "training_diagnostics.csv", index=False)
        models_dir = temp / "models"
        models_dir.mkdir()
        for profile_name, states in model_states.items():
            with (models_dir / f"{profile_name}.pkl").open("wb") as handle:
                pickle.dump(states, handle)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                benchmark="SOXX",
                assets="SOXL,SOXX,cash",
                allocation_output="direct_continuous_softmax",
                candidate_weight_grid="false",
                base_constrained_ref_role="none",
                selected_profile=selected_profile.name,
                selection_uses_outer_test="false",
            )
            R.log_metrics(
                development_ratio_10bps=float(selected_row["ratio_10bps"]),
                development_ratio_25bps=float(selected_row["ratio_25bps"]),
                development_max_drawdown_pct=float(
                    selected_row["max_drawdown_pct"]
                ),
                outer_ratio_10bps=float(
                    outer10["terminal_wealth_ratio_vs_soxx"]
                ),
                outer_max_drawdown_pct=float(
                    outer10["max_drawdown_pct"]
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
                features=features,
                selection=selection,
                selected=selected_row,
                comparison=comparison,
                annual=annual,
                regimes=regimes,
                objective_audit=objective_audit,
                decision=decision,
                objective_pass=objective_pass,
                recorder_id=recorder_id,
                recorder_uri=recorder_uri,
                recorder_local_dir=recorder_local_dir,
            )
            _write_card(
                card=card,
                summary=summary,
                selection=selection,
                annual=annual,
                comparison=comparison,
                regimes=regimes,
                objective_audit=objective_audit,
                diagnostics=diagnostics,
                features=features,
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
    spec: DirectContinuousAllocationSpec,
    source_audit: dict[str, Any],
    dataset_audit: dict[str, Any],
    features: list[str],
    selection: pd.DataFrame,
    selected: pd.Series,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    regimes: pd.DataFrame,
    objective_audit: pd.DataFrame,
    decision: str,
    objective_pass: bool,
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
            "model": "16-unit tanh MLP",
            "output": "three continuous allocation logits plus softmax",
            "base_position": None,
            "explicit_prior": None,
            "candidate_weight_grid": False,
            "hard_constraints": [
                "long_only",
                "weights_sum_to_one",
                "profile maximum SOXL weight",
            ],
            "cash_logic": (
                "cash is a direct model output; tail cash is training "
                "supervision only and never an inference threshold"
            ),
        },
        "feature_count": len(features),
        "features": features,
        "data_audit": {**source_audit, "dataset": dataset_audit},
        "development": {
            "years": list(spec.development_years),
            "selection_uses_outer_test": False,
            "selected": _json_ready(selected.to_dict()),
        },
        "outer_diagnostic": {
            "start": OUTER_START,
            "selection_uses_outer_test": False,
            "metrics_10bps": _json_ready(outer10.to_dict()),
        },
        "objective_pass": objective_pass,
        "promotion_decision": decision,
        "objective_gate_audit": objective_audit.to_dict(
            orient="records"
        ),
        "model_selection": selection.to_dict(orient="records"),
        "annual_metrics": annual.to_dict(orient="records"),
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
    annual: pd.DataFrame,
    comparison: pd.DataFrame,
    regimes: pd.DataFrame,
    objective_audit: pd.DataFrame,
    diagnostics: pd.DataFrame,
    features: list[str],
) -> None:
    selected = selection.loc[selection["selected"]].iloc[0]
    outer10 = comparison.loc[
        comparison["segment"].eq("outer_diagnostic")
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
    annual.to_csv(card / "backtest_walk_forward_metrics.csv", index=False)
    comparison.to_csv(
        card / "backtest_benchmark_comparison.csv", index=False
    )
    comparison.loc[
        :,
        [
            "segment",
            "profile",
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
        {
            "feature": features,
            "point_in_time": True,
            "base_or_prior_feature": False,
        }
    ).to_csv(card / "feature_manifest.csv", index=False)
    objective_audit.to_csv(card / "objective_gate_audit.csv", index=False)
    analysis = {
        "objective_pass": summary["objective_pass"],
        "promotion_decision": summary["promotion_decision"],
        "selected_profile": selected["profile"],
        "development_ratio_10bps": selected["ratio_10bps"],
        "development_ratio_25bps": selected["ratio_25bps"],
        "development_max_drawdown_pct": selected["max_drawdown_pct"],
        "outer_ratio_10bps": outer10[
            "terminal_wealth_ratio_vs_soxx"
        ],
        "outer_max_drawdown_pct": outer10["max_drawdown_pct"],
        "no_base_or_prior": True,
        "no_candidate_weight_grid": True,
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(_json_ready(analysis), indent=2), encoding="utf-8"
    )
    failed = objective_audit.loc[
        objective_audit["required"] & ~objective_audit["passed"]
    ]
    failure_text = (
        "# Failure modes\n\n"
        + (
            "\n".join(
                f"- {row.gate}: {row.evidence}"
                for row in failed.itertuples()
            )
            if not failed.empty
            else "- none"
        )
        + "\n"
    )
    (card / "failure_modes.md").write_text(
        failure_text, encoding="utf-8"
    )
    conclusion = (
        "# 纯模型连续 SOXL/SOXX/cash 配置结论\n\n"
        "- 跑了什么：模型直接输出三资产连续权重；没有SOXX底仓、"
        "规则先验或合法仓位表。\n"
        f"- 开发期：10bp财富/SOXX={selected['ratio_10bps']:.4f}，"
        f"25bp={selected['ratio_25bps']:.4f}，MaxDD="
        f"{selected['max_drawdown_pct']:.2f}% vs SOXX "
        f"{selected['soxx_max_drawdown_pct']:.2f}%。\n"
        f"- 年份稳定性：{int(selected['positive_relative_years'])}/7 "
        "个开发年份相对收益为正。\n"
        f"- Outer仅诊断：10bp财富/SOXX="
        f"{outer10['terminal_wealth_ratio_vs_soxx']:.4f}，MaxDD="
        f"{outer10['max_drawdown_pct']:.2f}% vs SOXX "
        f"{outer10['soxx_max_drawdown_pct']:.2f}%。\n"
        f"- 现金行为：开发期总体平均 {selected['avg_cash_weight_pct']:.2f}%，"
        f"下跌期 {selected['downside_avg_cash_weight_pct']:.2f}%；"
        f"outer总体 {outer10['avg_cash_weight_pct']:.2f}%，"
        f"下跌期 {outer10['downside_avg_cash_weight_pct']:.2f}%。\n"
        f"- 晋升决定：{summary['promotion_decision']}。\n"
        "- 风险：直接日频权重模型可能以高换手换取样本内相对收益；"
        "MaxDD、恢复期和25bp结果必须同时通过，不能只看终值。\n"
        "- 下一步：若失败，只调整模型的时序状态和训练目标；"
        "不恢复底仓、先验或外层参数选择。\n"
    )
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    workflow = (
        f"experiment: {EXPERIMENT}\n"
        "system_of_record: qlib\n"
        f"provider_uri: {summary['provider_uri']}\n"
        "benchmark: SOXX\n"
        "assets: [SOXL, SOXX, cash]\n"
        "allocation_output: direct_continuous_softmax\n"
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
            "# Direct continuous SOXL/SOXX/cash allocation\n\n"
            "模型直接输出完整连续权重；没有底仓、规则先验或候选仓位表。"
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
            "from src.workflow.direct_continuous_soxx_allocation "
            "import run_direct_continuous_soxx_allocation_card\n"
            "summary = run_direct_continuous_soxx_allocation_card("
            "root=repo_root)\n"
            "summary['promotion_decision']"
        ),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "card = Path.cwd()\n"
            "pd.read_csv(card / 'model_selection.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'backtest_benchmark_comparison.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'objective_gate_audit.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'regime_metrics.csv')"
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
            run_direct_continuous_soxx_allocation_card(), indent=2
        )
    )
