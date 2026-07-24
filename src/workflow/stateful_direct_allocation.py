from __future__ import annotations

import json
import math
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

from src.workflow.cost_aware_target_tracking import (
    CostAwareTargetTrackingSpec,
    apply_split_fixed_decision_cost_stress,
    simulate_cost_aware_policy,
)
from src.workflow.direct_continuous_soxx_allocation import (
    DirectContinuousAllocationSpec,
    _load_frame,
)
from src.workflow.hierarchical_four_asset_policy import (
    _benchmark_efficiency,
)
from src.workflow.intraday_recovery_direct_allocation import (
    _annual_metrics,
    _build_teacher_training_frame,
    _development_stability,
    _load_source_frame,
    _passes_risk,
    _risk_metrics,
)
from src.workflow.pit_constituent_auxiliary_training_allocation import (
    PitAuxiliaryAllocationSpec,
    _project_legal_weights,
    _soxx_common_training_frame,
)
from src.workflow.soxl_objective_feasibility_audit import _json_ready


EXPERIMENT = "stateful_direct_allocation_v1"
BASELINE_EXPERIMENT = "intraday_on_policy_staged_validation_v1"
CARD_TYPE = "stateful_direct_three_asset_allocation_card"
STATE_FEATURES = (
    "previous_soxl_weight",
    "previous_soxx_weight",
    "previous_cash_weight",
    "portfolio_drawdown",
    "portfolio_drawdown_squared",
    "portfolio_return_1d",
    "portfolio_return_3d",
    "portfolio_return_5d",
    "underwater_age_252",
)
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/predictions.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/teacher_training_frame.csv",
    f"artifacts/{EXPERIMENT}/model_selection_full.csv",
    f"artifacts/{EXPERIMENT}/dataset_audit.json",
)


@dataclass(frozen=True)
class StatefulCandidate:
    name: str
    recurrent_units: int
    teacher_loss_weight: float
    risk_loss_weight: float
    turnover_loss_weight: float
    trade_loss_weight: float
    trade_threshold: float
    epochs: int = 260
    on_policy_epochs: int = 80


CANDIDATES = (
    StatefulCandidate(
        name="gru8_imitation_gate",
        recurrent_units=8,
        teacher_loss_weight=20.00,
        risk_loss_weight=5.00,
        turnover_loss_weight=0.02,
        trade_loss_weight=1.00,
        trade_threshold=0.60,
    ),
    StatefulCandidate(
        name="gru16_imitation_gate",
        recurrent_units=16,
        teacher_loss_weight=20.00,
        risk_loss_weight=10.00,
        turnover_loss_weight=0.03,
        trade_loss_weight=1.00,
        trade_threshold=0.60,
    ),
    StatefulCandidate(
        name="gru16_tail_gate",
        recurrent_units=16,
        teacher_loss_weight=10.00,
        risk_loss_weight=20.00,
        turnover_loss_weight=0.04,
        trade_loss_weight=1.50,
        trade_threshold=0.70,
    ),
)


@dataclass(frozen=True)
class StatefulDirectAllocationSpec(CostAwareTargetTrackingSpec):
    experiment: str = EXPERIMENT
    baseline_experiment: str = BASELINE_EXPERIMENT
    selection_years: tuple[int, ...] = (2019, 2020, 2021, 2022)
    purge_rows: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.03
    training_open_cost_bps: float = 25.0
    minimum_open_turnover_reduction_pct: float = 30.0
    seed: int = 71


class StatefulDirectAllocationNet(nn.Module):
    """One model that emits the final legal SOXL/SOXX/cash allocation."""

    def __init__(
        self,
        *,
        primary_feature_count: int,
        common_feature_count: int,
        source_tail_units: int,
        source_hidden_units: int,
        state_feature_count: int,
        recurrent_units: int,
    ):
        super().__init__()
        self.tail_encoder = nn.Sequential(
            nn.Linear(common_feature_count, source_tail_units),
            nn.Tanh(),
        )
        self.base_hidden = nn.Linear(
            primary_feature_count + source_tail_units,
            source_hidden_units,
        )
        self.state_encoder = nn.GRU(
            input_size=state_feature_count,
            hidden_size=recurrent_units,
            batch_first=True,
        )
        self.state_adapter = nn.Linear(
            recurrent_units,
            source_hidden_units,
            bias=False,
        )
        self.allocation_output = nn.Linear(source_hidden_units, 3)
        self.state_allocation_adapter = nn.Linear(
            recurrent_units,
            3,
            bias=False,
        )
        self.trade_output = nn.Linear(source_hidden_units, 1)

    def forward_sequence(
        self,
        primary_features: torch.Tensor,
        common_features: torch.Tensor,
        state_features: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tail_state = self.tail_encoder(common_features)
        base_state = self.base_hidden(
            torch.cat([primary_features, tail_state], dim=1)
        )
        recurrent, next_hidden = self.state_encoder(
            state_features.unsqueeze(0),
            hidden,
        )
        combined = torch.tanh(
            base_state + self.state_adapter(recurrent.squeeze(0))
        )
        return (
            (
                self.allocation_output(combined)
                + self.state_allocation_adapter(
                    recurrent.squeeze(0)
                )
            ),
            self.trade_output(combined).squeeze(1),
            next_hidden,
        )


def run_stateful_direct_allocation_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    return _run_card(
        StatefulDirectAllocationSpec(root=_resolve_root(root))
    )


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def select_stateful_development_candidate(
    model_selection: pd.DataFrame,
) -> pd.Series:
    required = {
        "segment",
        "selection_uses_outer_test",
        "development_gate_pass",
        "risk_window_pass_count",
        "positive_relative_window_count",
        "relative_ratio_p25",
        "terminal_wealth",
        "open_turnover",
    }
    missing = required.difference(model_selection.columns)
    if missing:
        raise ValueError(
            f"model selection is missing columns: {sorted(missing)}"
        )
    development = model_selection[
        model_selection["segment"].eq("development_walk_forward")
        & ~model_selection["selection_uses_outer_test"].astype(bool)
    ].copy()
    if development.empty:
        raise ValueError("no development-only stateful candidates")
    eligible = development[
        development["development_gate_pass"].astype(bool)
    ].copy()
    if not eligible.empty:
        return eligible.sort_values(
            [
                "terminal_wealth",
                "relative_ratio_p25",
                "open_turnover",
            ],
            ascending=[False, False, True],
        ).iloc[0]
    return development.sort_values(
        [
            "risk_window_pass_count",
            "positive_relative_window_count",
            "relative_ratio_p25",
            "terminal_wealth",
            "open_turnover",
        ],
        ascending=[False, False, False, False, True],
    ).iloc[0]


def _portfolio_state(
    *,
    previous: np.ndarray,
    equity: float,
    peak: float,
    history: list[float],
    underwater_age: int,
) -> dict[str, float]:
    drawdown = max(0.0, 1.0 - equity / peak)
    return {
        "previous_soxl_weight": float(previous[0]),
        "previous_soxx_weight": float(previous[1]),
        "previous_cash_weight": float(previous[2]),
        "portfolio_drawdown": drawdown,
        "portfolio_drawdown_squared": drawdown * drawdown,
        "portfolio_return_1d": history[-1] if history else 0.0,
        "portfolio_return_3d": (
            float(np.prod(1.0 + np.asarray(history[-3:])) - 1.0)
            if history
            else 0.0
        ),
        "portfolio_return_5d": (
            float(np.prod(1.0 + np.asarray(history[-5:])) - 1.0)
            if history
            else 0.0
        ),
        "underwater_age_252": min(underwater_age, 252) / 252.0,
    }


def _teacher_state_frame(replay: pd.DataFrame) -> pd.DataFrame:
    equity = peak = 1.0
    previous = np.array([0.0, 0.0, 1.0], dtype="float64")
    history: list[float] = []
    underwater_age = 0
    rows: list[dict[str, Any]] = []
    for row in replay.sort_values("signal_date").itertuples():
        drawdown = max(0.0, 1.0 - equity / peak)
        underwater_age = underwater_age + 1 if drawdown > 1e-12 else 0
        state = _portfolio_state(
            previous=previous,
            equity=equity,
            peak=peak,
            history=history,
            underwater_age=underwater_age,
        )
        rows.append(
            {
                "signal_date": row.signal_date,
                **state,
                "teacher_soxl_weight": row.open_soxl_weight,
                "teacher_soxx_weight": row.open_soxx_weight,
                "teacher_cash_weight": row.open_cash_weight,
                "teacher_open_turnover": row.open_turnover,
            }
        )
        strategy_return = float(row.strategy_return)
        history.append(strategy_return)
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        previous = np.array(
            [row.soxl_weight, row.soxx_weight, row.cash_weight],
            dtype="float64",
        )
    return pd.DataFrame(rows).set_index("signal_date")


def _baseline_replays(
    source: pd.DataFrame,
    *,
    spec: StatefulDirectAllocationSpec,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    baseline = json.loads(
        (spec.baseline_card_dir / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    if baseline.get("selection_uses_outer_test") is not False:
        raise ValueError("baseline selection must exclude outer")
    selected = baseline["selected_model"]
    if selected.get("stage2_enabled") is not False:
        raise ValueError("stateful comparison requires frozen Stage1")
    teacher = _build_teacher_training_frame(
        source,
        spec=spec,
        teacher_beta=float(selected["teacher_recovery_beta"]),
    )
    params = {
        "c_value": float(selected["logistic_c"]),
        "threshold": float(selected["probability_threshold"]),
        "action_beta": float(selected["action_beta"]),
    }
    model_cache: dict[tuple[int, float], Any] = {}
    replays: dict[str, pd.DataFrame] = {}
    scenarios = (
        ("development_10_10", "development_walk_forward", 10.0, 10.0),
        ("development_25_10", "development_walk_forward", 25.0, 10.0),
        ("outer_10_10", "outer_diagnostic", 10.0, 10.0),
        ("outer_25_10", "outer_diagnostic", 25.0, 10.0),
    )
    for name, segment, open_cost, intraday_cost in scenarios:
        frame = source[source["segment"].eq(segment)].copy()
        replays[name] = simulate_cost_aware_policy(
            frame,
            spec=spec,
            teacher_frame=teacher,
            first_action_year=(
                spec.first_intraday_action_year
                if segment == "development_walk_forward"
                else min(spec.outer_years)
            ),
            model_cache=model_cache,
            risk_up_fraction=1.0,
            open_cost_bps=open_cost,
            intraday_cost_bps=intraday_cost,
            aim_momentum_multiplier=0.0,
            **params,
        )
    return replays, teacher


def _load_model_states(
    source_audit: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    path = Path(source_audit["source_model_path"])
    states = pickle.loads(path.read_bytes())
    if "outer_refit" not in states:
        raise ValueError("blocked: source model has no outer_refit state")
    return states


def _assemble_model_frame(
    *,
    source: pd.DataFrame,
    development_teacher: pd.DataFrame,
    spec: StatefulDirectAllocationSpec,
    source_states: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    direct_frame, direct_audit, _ = _load_frame(
        DirectContinuousAllocationSpec(root=spec.root)
    )
    common = _soxx_common_training_frame(
        PitAuxiliaryAllocationSpec(root=spec.root),
        end=source["signal_date"].max().date().isoformat(),
    )
    primary_features = list(
        source_states["outer_refit"]["primary_features"]
    )
    common_features = list(
        source_states["outer_refit"]["common_features"]
    )
    indexed_source = source.set_index("signal_date").sort_index()
    index = indexed_source.index
    frame = pd.concat(
        [
            direct_frame.reindex(index).loc[:, primary_features],
            common.reindex(index).loc[:, common_features],
        ],
        axis=1,
    )
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    frame = frame.join(
        indexed_source[
            [
                "trade_date",
                "exit_date",
                "segment",
                "soxl_return",
                "soxx_return",
            ]
        ]
    )
    frame = frame.join(development_teacher, how="left")
    for column in STATE_FEATURES:
        if column not in frame:
            frame[column] = np.nan
    for column in (
        "teacher_soxl_weight",
        "teacher_soxx_weight",
        "teacher_cash_weight",
        "teacher_open_turnover",
    ):
        if column not in frame:
            frame[column] = np.nan
    dataset, dataset_audit = _build_qlib_dataset(
        frame=frame,
        primary_features=primary_features,
        common_features=common_features,
        spec=spec,
    )
    audit = {
        "direct_feature_audit": direct_audit,
        "primary_feature_count": len(primary_features),
        "common_feature_count": len(common_features),
        "state_feature_count": len(STATE_FEATURES),
        "state_features": list(STATE_FEATURES),
        "dataset": dataset_audit,
    }
    return frame, dataset, audit


def _build_qlib_dataset(
    *,
    frame: pd.DataFrame,
    primary_features: list[str],
    common_features: list[str],
    spec: StatefulDirectAllocationSpec,
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    feature_columns = [
        *primary_features,
        *common_features,
        *STATE_FEATURES,
    ]
    label_columns = [
        "soxl_return",
        "soxx_return",
        "teacher_soxl_weight",
        "teacher_soxx_weight",
        "teacher_cash_weight",
        "teacher_open_turnover",
    ]
    static = pd.concat(
        {
            "feature": frame.loc[:, feature_columns],
            "label": frame.loc[:, label_columns],
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [
            frame.index,
            np.full(len(frame), "SOXL_SOXX_CASH"),
        ],
        names=["datetime", "instrument"],
    )
    outer_end = frame.index.max().date().isoformat()
    segments: dict[str, tuple[str, str]] = {
        "development": ("2016-01-01", "2022-12-31"),
        "outer": ("2023-01-01", outer_end),
    }
    for year in spec.selection_years:
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
    audit = {
        "dataset_class": "qlib.data.dataset.DatasetH",
        "handler_class": "qlib.data.dataset.handler.DataHandlerLP",
        "loader_class": "qlib.data.dataset.loader.StaticDataLoader",
        "segments": {
            name: list(bounds) for name, bounds in segments.items()
        },
        "segment_rows": {
            name: int(len(dataset.prepare(name))) for name in segments
        },
        "feature_count": len(feature_columns),
        "label_count": len(label_columns),
        "purge_rows": spec.purge_rows,
        "outer_labels_used_for_selection": False,
    }
    return dataset, audit


def _prepare_dataset(dataset, segment) -> pd.DataFrame:
    result = dataset.prepare(
        segment,
        col_set=["feature", "label"],
    )
    result = result.droplevel("instrument").sort_index()
    result.columns = [str(column[-1]) for column in result.columns]
    return result


def _series_scaler(
    packed: dict[str, dict[str, float]],
) -> dict[str, pd.Series]:
    return {
        name: pd.Series(values, dtype="float64")
        for name, values in packed.items()
    }


def _state_scaler(
    frame: pd.DataFrame,
) -> dict[str, pd.Series]:
    columns = list(STATE_FEATURES)
    median = frame.loc[:, columns].median()
    values = frame.loc[:, columns].fillna(median)
    mean = values.mean()
    scale = (
        values.std(ddof=1).replace(0.0, 1.0).fillna(1.0)
    )
    return {"median": median, "mean": mean, "scale": scale}


def _transform_frame(
    frame: pd.DataFrame,
    *,
    source_state: dict[str, Any],
    state_scaler: dict[str, pd.Series],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    primary_features = list(source_state["primary_features"])
    common_features = list(source_state["common_features"])
    primary_scaler = _series_scaler(source_state["primary_scaler"])
    common_scaler = _series_scaler(source_state["common_scaler"])

    def transform(
        columns: list[str],
        scaler: dict[str, pd.Series],
    ) -> torch.Tensor:
        values = frame.loc[:, columns].fillna(scaler["median"])
        values = (
            (values - scaler["mean"]) / scaler["scale"]
        ).clip(-8.0, 8.0)
        return torch.tensor(
            values.to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

    return (
        transform(primary_features, primary_scaler),
        transform(common_features, common_scaler),
        transform(list(STATE_FEATURES), state_scaler),
    )


def _initialize_model(
    *,
    source_state: dict[str, Any],
    candidate: StatefulCandidate,
) -> StatefulDirectAllocationNet:
    source_weights = source_state["model_state"]
    tail_weight = np.asarray(
        source_weights["tail_encoder.0.weight"]
    )
    hidden_weight = np.asarray(
        source_weights["allocation_head.0.weight"]
    )
    output_weight = np.asarray(
        source_weights["allocation_head.2.weight"]
    )
    model = StatefulDirectAllocationNet(
        primary_feature_count=len(source_state["primary_features"]),
        common_feature_count=len(source_state["common_features"]),
        source_tail_units=tail_weight.shape[0],
        source_hidden_units=hidden_weight.shape[0],
        state_feature_count=len(STATE_FEATURES),
        recurrent_units=candidate.recurrent_units,
    )
    model.trade_threshold = candidate.trade_threshold
    with torch.no_grad():
        model.tail_encoder[0].weight.copy_(
            torch.tensor(tail_weight, dtype=torch.float32)
        )
        model.tail_encoder[0].bias.copy_(
            torch.tensor(
                source_weights["tail_encoder.0.bias"],
                dtype=torch.float32,
            )
        )
        model.base_hidden.weight.copy_(
            torch.tensor(hidden_weight, dtype=torch.float32)
        )
        model.base_hidden.bias.copy_(
            torch.tensor(
                source_weights["allocation_head.0.bias"],
                dtype=torch.float32,
            )
        )
        model.allocation_output.weight.copy_(
            torch.tensor(output_weight, dtype=torch.float32)
        )
        model.allocation_output.bias.copy_(
            torch.tensor(
                source_weights["allocation_head.2.bias"],
                dtype=torch.float32,
            )
        )
        nn.init.zeros_(model.state_adapter.weight)
        nn.init.zeros_(model.state_allocation_adapter.weight)
        nn.init.zeros_(model.trade_output.weight)
        nn.init.constant_(model.trade_output.bias, 2.0)
    for parameter in (
        *model.tail_encoder.parameters(),
        *model.base_hidden.parameters(),
        *model.allocation_output.parameters(),
    ):
        parameter.requires_grad = False
    return model


def _path_loss(
    *,
    weights: torch.Tensor,
    returns: torch.Tensor,
    years: torch.Tensor,
    teacher: torch.Tensor,
    teacher_trade: torch.Tensor,
    material_teacher: torch.Tensor,
    candidate: StatefulCandidate,
    cost_bps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    initial = torch.tensor(
        [0.0, 0.0, 1.0],
        dtype=weights.dtype,
        device=weights.device,
    )
    turnover = torch.cat(
        [
            (
                0.5 * torch.abs(weights[0] - initial).sum()
            ).reshape(1),
            0.5 * torch.abs(weights[1:] - weights[:-1]).sum(dim=1),
        ]
    )
    portfolio_return = (
        weights[:, :2] * returns
    ).sum(dim=1) - turnover * cost_bps / 10_000.0
    portfolio_log = torch.log1p(
        torch.clamp(portfolio_return, min=-0.95)
    )
    benchmark_log = torch.log1p(
        torch.clamp(returns[:, 1], min=-0.95)
    )
    relative_log = portfolio_log - benchmark_log
    annual = torch.stack(
        [
            relative_log[years.eq(year)].mean()
            for year in torch.unique(years)
        ]
    )
    relative_loss = (
        -annual.mean()
        + 0.50 * annual.std(unbiased=False)
        + torch.relu(-annual).mean()
    )
    max_drawdown_limit = -math.log1p(-0.30)
    max_drawdown_losses: list[torch.Tensor] = []
    for mask in (
        torch.ones_like(years, dtype=torch.bool),
        *(years.eq(year) for year in torch.unique(years)),
    ):
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
            torch.relu(
                drawdown.max() - max_drawdown_limit
            ).square()
        )
    max_drawdown_loss = torch.stack(max_drawdown_losses).mean()
    rolling_5d_losses: list[torch.Tensor] = []
    worst_day_losses: list[torch.Tensor] = []
    for mask in (
        torch.ones_like(years, dtype=torch.bool),
        *(years.eq(year) for year in torch.unique(years)),
    ):
        section = portfolio_log[mask]
        rolling_5d = section.unfold(0, 5, 1).sum(dim=1)
        rolling_count = max(1, int(math.ceil(0.05 * len(rolling_5d))))
        rolling_tail = torch.topk(
            -rolling_5d,
            rolling_count,
        ).values
        rolling_5d_losses.append(
            torch.relu(
                rolling_tail + math.log1p(-0.12)
            ).square().mean()
        )
        worst_count = max(1, int(math.ceil(0.01 * len(section))))
        worst_tail = torch.topk(-section, worst_count).values
        worst_day_losses.append(
            torch.relu(
                worst_tail + math.log1p(-0.12)
            ).square().mean()
        )
    rolling_5d_loss = torch.stack(rolling_5d_losses).mean()
    worst_day_loss = torch.stack(worst_day_losses).mean()
    worst_count = max(1, int(math.ceil(0.05 * len(portfolio_return))))
    cvar = torch.topk(-portfolio_return, worst_count).values.mean()
    cvar_loss = torch.relu(cvar - 0.08).square()
    teacher_row_loss = nn.functional.smooth_l1_loss(
        weights,
        teacher,
        reduction="none",
    ).mean(dim=1)
    teacher_loss = (
        teacher_row_loss * (1.0 + 19.0 * material_teacher)
    ).mean()
    trade_loss = nn.functional.binary_cross_entropy_with_logits(
        teacher_trade[:, 0],
        teacher_trade[:, 1],
    )
    risk_loss = (
        max_drawdown_loss
        + 5.0 * rolling_5d_loss
        + 2.0 * worst_day_loss
        + cvar_loss
    )
    loss = (
        relative_loss
        + candidate.teacher_loss_weight * teacher_loss
        + candidate.risk_loss_weight * risk_loss
        + candidate.turnover_loss_weight * turnover.mean()
        + candidate.trade_loss_weight * trade_loss
    )
    return loss, {
        "loss": float(loss.detach()),
        "relative_loss": float(relative_loss.detach()),
        "teacher_loss": float(teacher_loss.detach()),
        "risk_loss": float(risk_loss.detach()),
        "trade_loss": float(trade_loss.detach()),
        "mean_turnover_loss": float(turnover.mean().detach()),
        "mean_annual_relative_log": float(annual.mean().detach()),
    }


def _fit_model(
    *,
    train: pd.DataFrame,
    source_state: dict[str, Any],
    candidate: StatefulCandidate,
    spec: StatefulDirectAllocationSpec,
    seed: int,
    epochs: int,
    initial_model_state: dict[str, torch.Tensor] | None = None,
    scaler: dict[str, pd.Series] | None = None,
) -> tuple[
    StatefulDirectAllocationNet,
    dict[str, pd.Series],
    dict[str, float],
]:
    required = [
        "soxl_return",
        "soxx_return",
        "teacher_soxl_weight",
        "teacher_soxx_weight",
        "teacher_cash_weight",
        "teacher_open_turnover",
        *STATE_FEATURES,
    ]
    train = train.dropna(subset=required).copy()
    if train.empty:
        raise ValueError("blocked: empty stateful training frame")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = scaler or _state_scaler(train)
    primary, common, state = _transform_frame(
        train,
        source_state=source_state,
        state_scaler=scaler,
    )
    returns = torch.tensor(
        train.loc[:, ["soxl_return", "soxx_return"]].to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32,
    )
    years = torch.tensor(
        train.index.year.to_numpy(),
        dtype=torch.int64,
    )
    teacher = torch.tensor(
        train.loc[
            :,
            [
                "teacher_soxl_weight",
                "teacher_soxx_weight",
                "teacher_cash_weight",
            ],
        ].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    teacher_trade_target = torch.tensor(
        train["teacher_open_turnover"]
        .ge(0.02)
        .to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    material_teacher = torch.tensor(
        (
            train["teacher_open_turnover"].ge(0.15)
            | train["portfolio_drawdown"].ge(0.08)
        ).to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    model = _initialize_model(
        source_state=source_state,
        candidate=candidate,
    )
    if initial_model_state is not None:
        model.load_state_dict(initial_model_state)
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    diagnostics: dict[str, float] = {}
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, trade_logit, _ = model.forward_sequence(
            primary,
            common,
            state,
        )
        raw_weights = _project_legal_weights(
            logits,
            maximum_soxl_weight=spec.maximum_soxl_weight,
            minimum_cash_weight=spec.minimum_cash_weight,
        )
        previous_weights = torch.tensor(
            train.loc[
                :,
                [
                    "previous_soxl_weight",
                    "previous_soxx_weight",
                    "previous_cash_weight",
                ],
            ].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        soft_trade = torch.sigmoid(trade_logit)
        hard_trade = soft_trade.ge(
            candidate.trade_threshold
        ).to(
            dtype=soft_trade.dtype
        )
        trade_gate = (
            hard_trade.detach()
            - soft_trade.detach()
            + soft_trade
        ).unsqueeze(1)
        weights = previous_weights + trade_gate * (
            raw_weights - previous_weights
        )
        loss, diagnostics = _path_loss(
            weights=weights,
            returns=returns,
            years=years,
            teacher=teacher,
            teacher_trade=torch.stack(
                [trade_logit, teacher_trade_target],
                dim=1,
            ),
            material_teacher=material_teacher,
            candidate=candidate,
            cost_bps=spec.training_open_cost_bps,
        )
        if not torch.isfinite(loss):
            raise ValueError(
                f"blocked: non-finite loss for {candidate.name}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            max_norm=2.0,
        )
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().clone()
                for key, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise ValueError("blocked: no stateful model state")
    model.load_state_dict(best_state)
    model.eval()
    diagnostics["best_loss"] = best_loss
    diagnostics["train_rows"] = float(len(train))
    return model, scaler, diagnostics


def _simulate_model(
    *,
    model: StatefulDirectAllocationNet,
    scaler: dict[str, pd.Series],
    source_state: dict[str, Any],
    frame: pd.DataFrame,
    spec: StatefulDirectAllocationSpec,
    cost_bps: float,
    segment: str,
) -> pd.DataFrame:
    ordered = frame.sort_index().copy()
    placeholder = ordered.copy()
    placeholder.loc[:, list(STATE_FEATURES)] = 0.0
    primary, common, _ = _transform_frame(
        placeholder,
        source_state=source_state,
        state_scaler=scaler,
    )
    equity = peak = 1.0
    previous = np.array([0.0, 0.0, 1.0], dtype="float64")
    history: list[float] = []
    underwater_age = 0
    hidden: torch.Tensor | None = None
    rows: list[dict[str, Any]] = []
    for position, (signal_date, row) in enumerate(ordered.iterrows()):
        drawdown = max(0.0, 1.0 - equity / peak)
        underwater_age = (
            underwater_age + 1 if drawdown > 1e-12 else 0
        )
        causal_state = _portfolio_state(
            previous=previous,
            equity=equity,
            peak=peak,
            history=history,
            underwater_age=underwater_age,
        )
        state_frame = pd.DataFrame(
            [causal_state],
            index=[signal_date],
            dtype="float64",
        )
        state_values = (
            (
                state_frame.loc[:, list(STATE_FEATURES)].fillna(
                    scaler["median"]
                )
                - scaler["mean"]
            )
            / scaler["scale"]
        ).clip(-8.0, 8.0)
        state_tensor = torch.tensor(
            state_values.to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits, trade_logit, hidden = model.forward_sequence(
                primary[position : position + 1],
                common[position : position + 1],
                state_tensor,
                hidden,
            )
            raw_weights = _project_legal_weights(
                logits,
                maximum_soxl_weight=spec.maximum_soxl_weight,
                minimum_cash_weight=spec.minimum_cash_weight,
            ).cpu().numpy()[0]
            trade_probability = float(
                torch.sigmoid(trade_logit).cpu().item()
            )
        trade_action = trade_probability >= float(
            getattr(model, "trade_threshold", 0.5)
        )
        weights = raw_weights if trade_action else previous.copy()
        open_turnover = float(
            np.abs(weights[:2] - previous[:2]).sum()
        )
        gross_return = float(
            weights[0] * row["soxl_return"]
            + weights[1] * row["soxx_return"]
        )
        strategy_return = (
            gross_return - open_turnover * cost_bps / 10_000.0
        )
        history.append(strategy_return)
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        rows.append(
            {
                "signal_date": signal_date,
                "trade_date": row.get("trade_date", pd.NaT),
                "exit_date": row.get("exit_date", pd.NaT),
                "segment": segment,
                "soxl_return": float(row["soxl_return"]),
                "soxx_return": float(row["soxx_return"]),
                "soxl_weight": float(weights[0]),
                "soxx_weight": float(weights[1]),
                "cash_weight": float(weights[2]),
                **causal_state,
                "gross_return": gross_return,
                "open_turnover": open_turnover,
                "trade_probability": trade_probability,
                "trade_action": trade_action,
                "intraday_turnover": 0.0,
                "turnover": open_turnover,
                "strategy_return": strategy_return,
                "recovery_action": False,
                "cost_bps": float(cost_bps),
            }
        )
        previous = weights
    return pd.DataFrame(rows)


def _fit_with_on_policy_refresh(
    *,
    train: pd.DataFrame,
    source_state: dict[str, Any],
    candidate: StatefulCandidate,
    spec: StatefulDirectAllocationSpec,
    seed: int,
) -> tuple[
    StatefulDirectAllocationNet,
    dict[str, pd.Series],
    dict[str, float],
]:
    model, scaler, first = _fit_model(
        train=train,
        source_state=source_state,
        candidate=candidate,
        spec=spec,
        seed=seed,
        epochs=candidate.epochs,
    )
    on_policy = _simulate_model(
        model=model,
        scaler=scaler,
        source_state=source_state,
        frame=train,
        spec=spec,
        cost_bps=spec.training_open_cost_bps,
        segment="training_on_policy_refresh",
    ).set_index("signal_date")
    refreshed = train.copy()
    refreshed.loc[
        on_policy.index,
        list(STATE_FEATURES),
    ] = on_policy.loc[:, list(STATE_FEATURES)]
    model, scaler, second = _fit_model(
        train=refreshed,
        source_state=source_state,
        candidate=candidate,
        spec=spec,
        seed=seed,
        epochs=candidate.on_policy_epochs,
        initial_model_state={
            key: tensor.detach().clone()
            for key, tensor in model.state_dict().items()
        },
        scaler=scaler,
    )
    return model, scaler, {
        **{f"teacher_forced_{key}": value for key, value in first.items()},
        **{f"on_policy_{key}": value for key, value in second.items()},
    }


def _development_candidate(
    *,
    dataset,
    frame: pd.DataFrame,
    source_states: dict[str, dict[str, Any]],
    candidate: StatefulCandidate,
    spec: StatefulDirectAllocationSpec,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    fitted_states: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for year in spec.selection_years:
        test = _prepare_dataset(dataset, f"fold_{year}")
        test = test.loc[test.index.year == year].copy()
        start_position = frame.index.searchsorted(
            pd.Timestamp(f"{year}-01-01")
        )
        train_end_position = start_position - spec.purge_rows - 1
        if train_end_position < 0:
            raise ValueError(
                f"blocked: insufficient stateful history for {year}"
            )
        train_end = frame.index[train_end_position]
        train = _prepare_dataset(
            dataset,
            ("2016-01-01", train_end),
        )
        source_state = source_states[f"fold_{year}"]
        model, scaler, fit_diagnostics = _fit_with_on_policy_refresh(
            train=train,
            source_state=source_state,
            candidate=candidate,
            spec=spec,
            seed=spec.seed + year,
        )
        replay = _simulate_model(
            model=model,
            scaler=scaler,
            source_state=source_state,
            frame=test,
            spec=spec,
            cost_bps=spec.training_open_cost_bps,
            segment="development_walk_forward",
        )
        parts.append(replay)
        fitted_states[f"fold_{year}"] = _pack_model(
            model=model,
            scaler=scaler,
            candidate=candidate,
            source_state=source_state,
        )
        diagnostics.append(
            {
                "candidate": candidate.name,
                "fold": f"fold_{year}",
                **fit_diagnostics,
            }
        )
    return (
        pd.concat(parts, ignore_index=True),
        fitted_states,
        pd.DataFrame(diagnostics),
    )


def _pack_model(
    *,
    model: StatefulDirectAllocationNet,
    scaler: dict[str, pd.Series],
    candidate: StatefulCandidate,
    source_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": {
            key: tensor.detach().cpu().numpy()
            for key, tensor in model.state_dict().items()
        },
        "state_scaler": {
            name: values.to_dict()
            for name, values in scaler.items()
        },
        "candidate": asdict(candidate),
        "primary_features": list(source_state["primary_features"]),
        "common_features": list(source_state["common_features"]),
        "state_features": list(STATE_FEATURES),
    }


def _candidate_metrics(
    replay: pd.DataFrame,
    *,
    candidate: StatefulCandidate,
) -> dict[str, Any]:
    metrics = _risk_metrics(replay["strategy_return"].to_numpy())
    stability = _development_stability(replay)
    gate = bool(
        _passes_risk(metrics)
        and stability["risk_window_pass_count"] == 7
        and stability["positive_relative_window_count"] >= 5
        and stability["relative_ratio_p25"] >= 0.95
    )
    return {
        "candidate": candidate.name,
        "segment": "development_walk_forward",
        "selection_uses_outer_test": False,
        **asdict(candidate),
        **metrics,
        **stability,
        "open_turnover": float(replay["open_turnover"].sum()),
        "open_active_day_count": int(
            replay["open_turnover"].gt(1e-12).sum()
        ),
        "open_material_day_count": int(
            replay["open_turnover"].ge(0.02).sum()
        ),
        "avg_cash_weight_pct": float(
            100.0 * replay["cash_weight"].mean()
        ),
        "development_gate_pass": gate,
    }


def _fixed_cost_metrics(
    replay: pd.DataFrame,
    *,
    cost_bps: float,
) -> dict[str, float]:
    returns = apply_split_fixed_decision_cost_stress(
        replay["gross_return"],
        replay["open_turnover"],
        replay["intraday_turnover"],
        open_cost_bps=cost_bps,
        intraday_cost_bps=cost_bps,
    )
    return _risk_metrics(returns)


def _run_card(spec: StatefulDirectAllocationSpec) -> dict[str, Any]:
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    _bootstrap_source_pin(spec)
    source, source_audit = _load_source_frame(spec)
    if source_audit.get("source_model_frozen") is not True:
        raise ValueError("stateful model requires a frozen source model")
    source_states = _load_model_states(source_audit)
    baseline_replays, teacher_training = _baseline_replays(
        source,
        spec=spec,
    )
    development_teacher = _teacher_state_frame(
        baseline_replays["development_10_10"]
    )
    frame, dataset, dataset_audit = _assemble_model_frame(
        source=source,
        development_teacher=development_teacher,
        spec=spec,
        source_states=source_states,
    )

    candidate_rows: list[dict[str, Any]] = []
    development_replays: dict[str, pd.DataFrame] = {}
    development_states: dict[str, dict[str, Any]] = {}
    all_diagnostics: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        replay, states, diagnostics = _development_candidate(
            dataset=dataset,
            frame=frame,
            source_states=source_states,
            candidate=candidate,
            spec=spec,
        )
        development_replays[candidate.name] = replay
        development_states[candidate.name] = states
        all_diagnostics.append(diagnostics)
        candidate_rows.append(
            _candidate_metrics(replay, candidate=candidate)
        )
    model_selection = pd.DataFrame(candidate_rows)
    selected = select_stateful_development_candidate(model_selection)
    selected_name = str(selected["candidate"])
    selected_candidate = next(
        candidate
        for candidate in CANDIDATES
        if candidate.name == selected_name
    )

    development = _prepare_dataset(dataset, "development")
    outer = _prepare_dataset(dataset, "outer")
    outer_model, outer_scaler, outer_diagnostics = (
        _fit_with_on_policy_refresh(
            train=development,
            source_state=source_states["outer_refit"],
            candidate=selected_candidate,
            spec=spec,
            seed=spec.seed + 17_022,
        )
    )
    outer_10 = _simulate_model(
        model=outer_model,
        scaler=outer_scaler,
        source_state=source_states["outer_refit"],
        frame=outer,
        spec=spec,
        cost_bps=10.0,
        segment="outer_diagnostic",
    )
    outer_25 = _simulate_model(
        model=outer_model,
        scaler=outer_scaler,
        source_state=source_states["outer_refit"],
        frame=outer,
        spec=spec,
        cost_bps=25.0,
        segment="outer_diagnostic_dynamic_25bp",
    )
    selected_development = development_replays[selected_name]
    development_metrics = _risk_metrics(
        selected_development["strategy_return"].to_numpy()
    )
    outer_metrics = _risk_metrics(
        outer_10["strategy_return"].to_numpy()
    )
    outer_dynamic_25_metrics = _risk_metrics(
        outer_25["strategy_return"].to_numpy()
    )
    fixed_25_metrics = _fixed_cost_metrics(
        outer_10,
        cost_bps=25.0,
    )
    baseline_outer = baseline_replays["outer_10_10"]
    baseline_outer_25 = baseline_replays["outer_25_10"]
    baseline_metrics = _risk_metrics(
        baseline_outer["strategy_return"].to_numpy()
    )
    baseline_dynamic_25_metrics = _risk_metrics(
        baseline_outer_25["strategy_return"].to_numpy()
    )
    soxx_wealth = float(
        np.prod(1.0 + outer_10["soxx_return"].to_numpy())
    )
    ratio_vs_soxx = outer_metrics["terminal_wealth"] / soxx_wealth
    annual = _annual_metrics(outer_10)
    annual_relative_pass_count = int(
        annual["terminal_wealth_ratio_vs_soxx"].gt(1.0).sum()
    )
    annual_risk_pass_count = int(
        annual.apply(
            lambda row: _passes_risk(row.to_dict()),
            axis=1,
        ).sum()
    )
    model_open_turnover = float(outer_10["open_turnover"].sum())
    baseline_open_turnover = float(
        baseline_outer["open_turnover"].sum()
    )
    turnover_reduction_pct = 100.0 * (
        1.0 - model_open_turnover / baseline_open_turnover
    )
    model_active_days = int(
        outer_10["open_turnover"].gt(1e-12).sum()
    )
    baseline_active_days = int(
        baseline_outer["open_turnover"].gt(1e-12).sum()
    )
    benchmark_efficiency = _benchmark_efficiency(
        outer_10.assign(soxs_weight=0.0),
        cost_bps=10,
    )
    development_gate = bool(selected["development_gate_pass"])
    replacement_gate = bool(
        development_gate
        and _passes_risk(outer_metrics)
        and _passes_risk(outer_dynamic_25_metrics)
        and ratio_vs_soxx > 1.0
        and annual_relative_pass_count >= 3
        and annual_risk_pass_count >= 3
        and outer_metrics["return_pct"] >= baseline_metrics["return_pct"]
        and outer_dynamic_25_metrics["return_pct"]
        >= baseline_dynamic_25_metrics["return_pct"]
        and turnover_reduction_pct
        >= spec.minimum_open_turnover_reduction_pct
        and model_active_days <= baseline_active_days
        and benchmark_efficiency["return_maxdd"]
        > benchmark_efficiency["matched_beta_return_maxdd"]
    )
    decision = (
        "promoted_research_candidate_stateful_direct_allocation"
        if replacement_gate
        else "retain_current_stage1_stateful_direct_model_not_promoted"
    )

    selected_pack = {
        **development_states[selected_name],
        "outer_refit": _pack_model(
            model=outer_model,
            scaler=outer_scaler,
            candidate=selected_candidate,
            source_state=source_states["outer_refit"],
        ),
    }
    diagnostics = pd.concat(
        [
            *all_diagnostics,
            pd.DataFrame(
                [
                    {
                        "candidate": selected_name,
                        "fold": "outer_refit",
                        **outer_diagnostics,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    recorder = _record_artifacts(
        spec=spec,
        source_audit=source_audit,
        selected_candidate=selected_candidate,
        selected_pack=selected_pack,
        model_selection=model_selection,
        outer_replay=outer_10,
        teacher_frame=development_teacher,
        diagnostics=diagnostics,
        dataset_audit=dataset_audit,
        development_metrics=development_metrics,
        outer_metrics=outer_metrics,
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "decision": decision,
        "selection_uses_outer_test": False,
        "outer_status": "reused_diagnostic_not_fresh_unseen_test",
        "promotion_scope": (
            "replace external cap_wait_ramp with one stateful direct "
            "SOXL_SOXX_cash allocation model"
        ),
        "assets": ["SOXL", "SOXX", "cash"],
        "system_of_record": "qlib",
        "method": {
            "allocation_output": "direct_final_legal_three_asset_weights",
            "external_cap_wait_ramp": False,
            "base_plus_overlay": False,
            "source_feature_encoder": (
                "initialized from each frozen qlib walk-forward source fold"
            ),
            "state_model": (
                "causal GRU over previous weights, portfolio drawdown, "
                "recent returns, and underwater age"
            ),
            "training": (
                "teacher-forced fit followed by one on-policy state refresh"
            ),
            "selection": (
                "2019-2022 development walk-forward at 25bp open cost; "
                "outer excluded"
            ),
        },
        "selected_model": {
            **asdict(selected_candidate),
            "selected_from_segment": "development_walk_forward",
            "selection_uses_outer_test": False,
        },
        "development_selection": _json_ready(selected.to_dict()),
        "development_gate_pass": development_gate,
        "development_metrics_25bps": development_metrics,
        "outer_diagnostic_metrics_10bps": outer_metrics,
        "outer_dynamic_metrics_25bps": outer_dynamic_25_metrics,
        "outer_fixed_decision_metrics_25bps": fixed_25_metrics,
        "baseline_outer_metrics_10bps": baseline_metrics,
        "baseline_outer_dynamic_metrics_25_open_10_intraday": (
            baseline_dynamic_25_metrics
        ),
        "benchmark_comparison_10bps": {
            "terminal_wealth_ratio_vs_soxx": ratio_vs_soxx,
            **benchmark_efficiency,
        },
        "outer_annual_gate_summary": {
            "relative_pass_count": annual_relative_pass_count,
            "risk_pass_count": annual_risk_pass_count,
            "required_count": 3,
        },
        "turnover_effect": {
            "model_open_turnover": model_open_turnover,
            "baseline_open_turnover": baseline_open_turnover,
            "open_turnover_reduction_pct": turnover_reduction_pct,
            "minimum_reduction_pct": (
                spec.minimum_open_turnover_reduction_pct
            ),
            "model_open_active_day_count": model_active_days,
            "baseline_open_active_day_count": baseline_active_days,
            "model_avg_cash_weight_pct": float(
                100.0 * outer_10["cash_weight"].mean()
            ),
            "baseline_avg_cash_weight_pct": float(
                100.0 * baseline_outer["cash_weight"].mean()
            ),
        },
        "replacement_gate_pass": replacement_gate,
        "source_audit": source_audit,
        "dataset_audit": dataset_audit,
        "qlib_docs": {
            "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
            "workflow": (
                "https://qlib.readthedocs.io/en/latest/component/workflow.html"
            ),
            "strategy": (
                "https://qlib.readthedocs.io/en/latest/component/strategy.html"
            ),
        },
        "risk_gates": {
            "max_drawdown_floor_pct": -30.0,
            "worst_day_floor_pct": -12.0,
            "cvar5_floor_pct": -8.0,
            "worst_rolling_5d_floor_pct": -12.0,
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": list(
                RECORDER_ONLY_ARTIFACTS
            ),
        },
        **recorder,
    }
    _write_card(
        spec=spec,
        summary=summary,
        model_selection=model_selection,
        annual=annual,
        outer_replay=outer_10,
        baseline_outer=baseline_outer,
    )
    return summary


def _bootstrap_source_pin(spec: StatefulDirectAllocationSpec) -> None:
    workflow_path = spec.card_dir / "workflow.yaml"
    if workflow_path.exists():
        return
    baseline = json.loads(
        (spec.baseline_card_dir / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    source_recorder_id = baseline["source_audit"][
        "source_recorder_id"
    ]
    workflow_path.write_text(
        f"experiment: {spec.experiment}\n"
        f"source_experiment: {spec.source_experiment}\n"
        f"source_recorder_id: {source_recorder_id}\n"
        "selection_uses_outer_test: false\n",
        encoding="utf-8",
    )


def _record_artifacts(
    *,
    spec: StatefulDirectAllocationSpec,
    source_audit: dict[str, Any],
    selected_candidate: StatefulCandidate,
    selected_pack: dict[str, Any],
    model_selection: pd.DataFrame,
    outer_replay: pd.DataFrame,
    teacher_frame: pd.DataFrame,
    diagnostics: pd.DataFrame,
    dataset_audit: dict[str, Any],
    development_metrics: dict[str, float],
    outer_metrics: dict[str, float],
) -> dict[str, Any]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(
        prefix=f"{spec.experiment}_"
    ) as tmp:
        tmp_dir = Path(tmp)
        model_dir = tmp_dir / "models"
        model_dir.mkdir(parents=True)
        (model_dir / f"{selected_candidate.name}.pkl").write_bytes(
            pickle.dumps(selected_pack)
        )
        outer_replay[
            [
                "signal_date",
                "trade_date",
                "soxl_weight",
                "soxx_weight",
                "cash_weight",
                "open_turnover",
            ]
        ].to_csv(tmp_dir / "positions.csv", index=False)
        outer_replay.to_csv(
            tmp_dir / "strategy_returns.csv",
            index=False,
        )
        outer_replay[
            [
                "signal_date",
                "soxl_weight",
                "soxx_weight",
                "cash_weight",
            ]
        ].to_csv(tmp_dir / "predictions.csv", index=False)
        teacher_frame.reset_index().to_csv(
            tmp_dir / "teacher_training_frame.csv",
            index=False,
        )
        model_selection.to_csv(
            tmp_dir / "model_selection_full.csv",
            index=False,
        )
        diagnostics.to_csv(
            tmp_dir / "training_diagnostics.csv",
            index=False,
        )
        (tmp_dir / "dataset_audit.json").write_text(
            json.dumps(_json_ready(dataset_audit), indent=2),
            encoding="utf-8",
        )
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_experiment=spec.source_experiment,
                source_recorder_id=source_audit["source_recorder_id"],
                selected_candidate=selected_candidate.name,
                allocation_output="direct_final_legal_weights",
                external_cap_wait_ramp="false",
                selection_uses_outer_test="false",
                training_open_cost_bps=str(
                    spec.training_open_cost_bps
                ),
            )
            R.log_metrics(
                development_return_pct=development_metrics["return_pct"],
                development_max_drawdown_pct=development_metrics[
                    "max_drawdown_pct"
                ],
                outer_return_pct=outer_metrics["return_pct"],
                outer_max_drawdown_pct=outer_metrics[
                    "max_drawdown_pct"
                ],
            )
            recorder.log_artifact(
                model_dir.as_posix(),
                artifact_path=f"{spec.experiment}/models",
            )
            for filename in (
                "positions.csv",
                "strategy_returns.csv",
                "predictions.csv",
                "teacher_training_frame.csv",
                "model_selection_full.csv",
                "training_diagnostics.csv",
                "dataset_audit.json",
            ):
                recorder.log_artifact(
                    (tmp_dir / filename).as_posix(),
                    artifact_path=spec.experiment,
                )
            info = {
                "recorder_id": recorder.id,
                "recorder_uri": str(recorder_uri.resolve()),
                "recorder_local_dir": str(
                    Path(recorder.get_local_dir()).resolve()
                ),
            }
    recorder_artifacts = Path(info["recorder_local_dir"]) / "artifacts"
    for filename in (
        "code_cached.txt",
        "code_diff.txt",
        "code_status.txt",
    ):
        (recorder_artifacts / filename).unlink(missing_ok=True)
    return info


def _write_card(
    *,
    spec: StatefulDirectAllocationSpec,
    summary: dict[str, Any],
    model_selection: pd.DataFrame,
    annual: pd.DataFrame,
    outer_replay: pd.DataFrame,
    baseline_outer: pd.DataFrame,
) -> None:
    card = spec.card_dir
    (card / "summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    selected_payload = {
        **summary["selected_model"],
        "development_gate_pass": summary["development_gate_pass"],
        "replacement_gate_pass": summary["replacement_gate_pass"],
        "development_metrics_25bps": summary[
            "development_metrics_25bps"
        ],
    }
    (card / "selected_model.json").write_text(
        json.dumps(
            _json_ready(selected_payload),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    model_selection.to_csv(card / "model_selection.csv", index=False)
    annual.to_csv(card / "annual_metrics.csv", index=False)
    risk_rows = []
    for name, metrics in (
        ("stateful_outer_10bps", summary["outer_diagnostic_metrics_10bps"]),
        ("stateful_dynamic_25bps", summary["outer_dynamic_metrics_25bps"]),
        (
            "stateful_fixed_25bps",
            summary["outer_fixed_decision_metrics_25bps"],
        ),
        ("stage1_outer_10bps", summary["baseline_outer_metrics_10bps"]),
        (
            "stage1_dynamic_25_open_10_intraday",
            summary[
                "baseline_outer_dynamic_metrics_25_open_10_intraday"
            ],
        ),
    ):
        risk_rows.append({"scenario": name, **metrics})
    pd.DataFrame(risk_rows).to_csv(
        card / "risk_analysis.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "model": "stateful_direct",
                "open_turnover": outer_replay["open_turnover"].sum(),
                "active_days": outer_replay["open_turnover"].gt(1e-12).sum(),
                "material_days": outer_replay["open_turnover"].ge(0.02).sum(),
                "avg_cash_weight_pct": 100.0
                * outer_replay["cash_weight"].mean(),
            },
            {
                "model": "current_stage1",
                "open_turnover": baseline_outer["open_turnover"].sum(),
                "active_days": baseline_outer["open_turnover"].gt(1e-12).sum(),
                "material_days": baseline_outer["open_turnover"].ge(0.02).sum(),
                "avg_cash_weight_pct": 100.0
                * baseline_outer["cash_weight"].mean(),
            },
        ]
    ).to_csv(card / "turnover_analysis.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "stateful_direct",
                **summary["benchmark_comparison_10bps"],
            }
        ]
    ).to_csv(
        card / "backtest_benchmark_comparison.csv",
        index=False,
    )
    workflow = f"""experiment: {spec.experiment}
source_experiment: {spec.source_experiment}
source_recorder_id: {summary['source_audit']['source_recorder_id']}
baseline_experiment: {spec.baseline_experiment}
assets: [SOXL, SOXX, cash]
allocation_output: direct_final_legal_three_asset_weights
external_cap_wait_ramp: false
base_plus_overlay: false
training_open_cost_bps: {spec.training_open_cost_bps}
selection_years: [2019, 2020, 2021, 2022]
selection_uses_outer_test: false
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  recorder_only_artifacts:
"""
    workflow += "\n".join(
        f"    - {artifact}" for artifact in RECORDER_ONLY_ARTIFACTS
    )
    (card / "workflow.yaml").write_text(
        workflow + "\n",
        encoding="utf-8",
    )
    outer = summary["outer_diagnostic_metrics_10bps"]
    development = summary["development_metrics_25bps"]
    turnover = summary["turnover_effect"]
    conclusion = f"""# 结论

- what_ran: 保留冻结 qlib walk-forward 特征编码作为初始化，在同一个最终仓位头中加入因果 GRU 账户状态；模型直接输出 SOXL/SOXX/cash，已删除推理时 cap/wait/ramp 和底仓加 overlay。
- key_metrics: development 25bp return={development['return_pct']:.4f}%，MaxDD={development['max_drawdown_pct']:.4f}%，worst 5d={development['worst_rolling_5d_pct']:.4f}%；outer 10bp return={outer['return_pct']:.4f}%，MaxDD={outer['max_drawdown_pct']:.4f}%，worst 5d={outer['worst_rolling_5d_pct']:.4f}%。
- regime_window: 候选只由 2019-2022 年度与两年 development 窗口选择；2023-至今 outer 只在选择冻结后诊断。
- turnover: outer 开盘换手由 {turnover['baseline_open_turnover']:.4f} 降至 {turnover['model_open_turnover']:.4f}，变化 {turnover['open_turnover_reduction_pct']:.2f}%；活跃日 {turnover['model_open_active_day_count']} 对 {turnover['baseline_open_active_day_count']}。
- promotion_decision: `{summary['decision']}`；development gate={summary['development_gate_pass']}，完整替换门={summary['replacement_gate_pass']}。
- risks: outer 已被此前多轮研究重复观察，不是全新未见测试；教师来自当前 Stage1，只用于 development 行为压缩，outer 标签不参与训练或选择；尚未验证真实 bid/ask、延迟和排队。
- next_action: 若 development 风险门失败，停止扩大网络或使用 outer 调参；只保留该结构证据并继续使用当前 Stage1。
- recorder: `{summary['recorder_id']}`。
- 不构成买卖建议。
"""
    (card / "conclusion.md").write_text(
        conclusion,
        encoding="utf-8",
    )
    _write_notebook(spec=spec, summary=summary)


def _write_notebook(
    *,
    spec: StatefulDirectAllocationSpec,
    summary: dict[str, Any],
) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Stateful direct SOXL/SOXX/cash allocation\n\n"
            "单个模型直接输出最终合法仓位；outer 不参与选择。"
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
            "calendar = D.calendar(start_time='2016-01-01', "
            "end_time='2026-12-31', freq='day')\n"
            "from src.workflow.stateful_direct_allocation import "
            "run_stateful_direct_allocation_card\n"
            "summary = run_stateful_direct_allocation_card(root=repo_root)\n"
            "summary['decision']"
        ),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "card = Path.cwd()\n"
            "pd.read_csv(card / 'model_selection.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'risk_analysis.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'turnover_analysis.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'annual_metrics.csv')"
        ),
        nbformat.v4.new_markdown_cell(
            (spec.card_dir / "conclusion.md").read_text(
                encoding="utf-8"
            )
        ),
    ]
    nbformat.write(
        notebook,
        spec.card_dir / f"{summary['experiment']}.ipynb",
    )
