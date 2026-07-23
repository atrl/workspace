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
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from torch import nn

from src.factor import (
    EMPIRICAL_TAIL_FEATURES,
    IMPLIED_TAIL_FEATURES,
    STRUCTURAL_FRAGILITY_FEATURES,
)
from src.workflow.direct_continuous_soxx_allocation import (
    DEVELOPMENT_YEARS,
    OUTER_START,
    DirectContinuousAllocationSpec,
    DirectLossProfile,
    _annual_metrics,
    _evaluate,
    _load_frame,
    _prepare,
    _regime_metrics,
)
from src.workflow.hierarchical_soxx_excess_walk_forward import (
    ALPHA_FEATURES,
    CASH_FEATURES,
    HAZARD_FEATURES,
)
from src.workflow.multitask_classification_allocation import (
    TARGET_WEIGHT_COLUMNS,
    _attach_multitask_targets,
    _build_dataset,
)
from src.workflow.soxl_objective_feasibility_audit import _json_ready
from src.workflow.taleb_fragility_allocation import (
    TalebFragilityAllocationSpec,
    _load_taleb_features,
)
from src.workflow.trainable_regime_policy_nested_wf import (
    _remove_qlib_code_snapshot_artifacts,
)


EXPERIMENT = "role_separated_direct_allocation_v1"
HEAD_START_YEAR = 2014
HEAD_COLUMNS = ("head_hazard", "head_alpha", "head_recovery")
ROLE_TARGETS = (
    "future_downside_risk_20d",
    "future_relative_utility_20d",
    "recovery_label",
)
FAST_LAGS = (1, 3, 5)
SENSITIVITY_SEEDS = (17, 29, 53, 71, 89)
FAST_SOURCE_FEATURES = tuple(
    dict.fromkeys(
        [
            *IMPLIED_TAIL_FEATURES,
            "soxx_exceedance_ewm",
            "soxx_downside_semivariance_share_20",
            "soxx_rebalance_drag_pressure_20",
            "soxl_levered_decay_acceleration_5",
            *CASH_FEATURES,
        ]
    )
)
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/head_predictions.csv",
    f"artifacts/{EXPERIMENT}/predictions.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/training_diagnostics.csv",
)


@dataclass(frozen=True)
class AllocatorCandidate:
    name: str
    router: bool
    objective: str
    use_hazard: bool = True


CANDIDATES = (
    AllocatorCandidate(
        name="crossfit_heads_direct_mse",
        router=False,
        objective="oracle_weight_mse",
    ),
    AllocatorCandidate(
        name="crossfit_alpha_recovery_direct_mse",
        router=False,
        objective="oracle_weight_mse",
        use_hazard=False,
    ),
    AllocatorCandidate(
        name="crossfit_heads_router_mse",
        router=True,
        objective="oracle_weight_mse",
    ),
    AllocatorCandidate(
        name="crossfit_alpha_recovery_router_mse",
        router=True,
        objective="oracle_weight_mse",
        use_hazard=False,
    ),
    AllocatorCandidate(
        name="crossfit_alpha_recovery_router_utility",
        router=True,
        objective="relative_wealth_cvar",
        use_hazard=False,
    ),
)


@dataclass(frozen=True)
class RoleSeparatedAllocationSpec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    development_years: tuple[int, ...] = DEVELOPMENT_YEARS
    development_end: str = "2022-12-30"
    outer: tuple[str, str] = (OUTER_START, "2026-07-02")
    purge_rows: int = 20
    hidden_units: int = 12
    router_experts: int = 4
    epochs: int = 500
    learning_rate: float = 0.004
    weight_decay: float = 0.04
    maximum_soxl_weight: float = 0.30
    costs_bps: tuple[int, ...] = (10, 25)
    training_cost_bps: int = 10
    seed: int = 53
    as_of: str = "2026-07-02"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def daily_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def taleb_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "taleb_fragility_v1"


class DirectHeadAllocator(nn.Module):
    """Small direct allocator that only sees cross-fitted head outputs."""

    def __init__(self, hidden_units: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(HEAD_COLUMNS), hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 2),
        )

    def forward(
        self, features: torch.Tensor, maximum_soxl_weight: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.network(features)
        return _factorized_weights(logits, maximum_soxl_weight), logits


class RouterHeadAllocator(nn.Module):
    """Learned mixture of full-allocation experts with no fixed base sleeve."""

    def __init__(self, hidden_units: int, expert_count: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(len(HEAD_COLUMNS), hidden_units),
            nn.Tanh(),
        )
        self.router = nn.Linear(hidden_units, expert_count)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_units, 2) for _ in range(expert_count)]
        )

    def forward(
        self, features: torch.Tensor, maximum_soxl_weight: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encoder(features)
        router_probability = torch.softmax(self.router(state), dim=1)
        expert_weights = torch.stack(
            [
                _factorized_weights(expert(state), maximum_soxl_weight)
                for expert in self.experts
            ],
            dim=1,
        )
        weights = (router_probability.unsqueeze(2) * expert_weights).sum(
            dim=1
        )
        return weights, router_probability


def run_role_separated_direct_allocation_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    return _run_card(RoleSeparatedAllocationSpec(root=repo_root))


def _factorized_weights(
    logits: torch.Tensor, maximum_soxl_weight: float
) -> torch.Tensor:
    soxl = maximum_soxl_weight * torch.sigmoid(logits[:, 0])
    cash = (1.0 - soxl) * torch.sigmoid(logits[:, 1])
    soxx = 1.0 - soxl - cash
    return torch.stack([soxl, soxx, cash], dim=1)


def _future_path(
    returns: pd.Series, horizon: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    forward = pd.concat(
        [returns.shift(-offset) for offset in range(horizon)], axis=1
    )
    path = (1.0 + forward).cumprod(axis=1) - 1.0
    incomplete = forward.isna().any(axis=1)
    terminal = path.iloc[:, -1].mask(incomplete)
    adverse = path.min(axis=1).mask(incomplete)
    return forward, terminal, adverse


def _attach_role_targets_and_deltas(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    soxx = pd.to_numeric(
        result["soxx_forward_open_return"], errors="coerce"
    )
    soxl = pd.to_numeric(
        result["soxl_forward_open_return"], errors="coerce"
    )
    soxx_forward, soxx_terminal, soxx_adverse = _future_path(soxx, 20)
    _, soxl_terminal, soxl_adverse = _future_path(soxl, 20)
    downside = soxx_forward.clip(upper=0.0).pow(2).mean(axis=1).pow(0.5)
    downside = downside.mul(np.sqrt(252.0))
    result["future_downside_risk_20d"] = (
        downside + 1.5 * soxx_adverse.abs()
    ).clip(lower=0.0, upper=1.5)
    result["future_relative_utility_20d"] = (
        soxl_terminal.sub(soxx_terminal)
        + 0.60 * soxl_adverse.sub(soxx_adverse)
    ).clip(lower=-0.75, upper=0.75)
    incomplete = soxx_forward.isna().any(axis=1)
    result.loc[
        incomplete,
        ["future_downside_risk_20d", "future_relative_utility_20d"],
    ] = np.nan

    deltas: dict[str, pd.Series] = {}
    for feature in FAST_SOURCE_FEATURES:
        if feature not in result:
            continue
        values = pd.to_numeric(result[feature], errors="coerce")
        for lag in FAST_LAGS:
            name = f"{feature}_delta_{lag}d"
            if name not in result:
                deltas[name] = values.sub(values.shift(lag))
    if deltas:
        result = result.join(pd.DataFrame(deltas, index=result.index))
    return result


def _feature_roles(frame: pd.DataFrame) -> dict[str, list[str]]:
    hazard = list(
        dict.fromkeys(
            [
                *HAZARD_FEATURES,
                *CASH_FEATURES,
                *EMPIRICAL_TAIL_FEATURES,
                *STRUCTURAL_FRAGILITY_FEATURES,
            ]
        )
    )
    alpha = list(dict.fromkeys(ALPHA_FEATURES))
    recovery = list(
        dict.fromkeys(
            [
                *CASH_FEATURES,
                *[
                    f"{feature}_delta_{lag}d"
                    for feature in FAST_SOURCE_FEATURES
                    for lag in FAST_LAGS
                ],
            ]
        )
    )
    roles = {
        "hazard": [feature for feature in hazard if feature in frame],
        "alpha": [feature for feature in alpha if feature in frame],
        "recovery": [
            feature for feature in recovery if feature in frame
        ],
    }
    if any(not features for features in roles.values()):
        raise ValueError(f"blocked: empty role feature group {roles}")
    return roles


def _head_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "future_downside_risk_20d",
        "future_relative_utility_20d",
        "recovery_label",
        "stress_context",
    ]
    return frame.dropna(subset=required).copy()


def _fill_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    medians = train.loc[:, features].median()
    return (
        train.loc[:, features].fillna(medians),
        test.loc[:, features].fillna(medians),
        medians,
    )


def _fit_predict_heads(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    roles: dict[str, list[str]],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    train = _head_training_frame(train)
    test = _head_training_frame(test)
    if train.empty or test.empty:
        raise ValueError("blocked: empty role-head train or test data")
    predictions = pd.DataFrame(index=test.index)
    state: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []

    for offset, (role, target) in enumerate(
        (
            ("hazard", "future_downside_risk_20d"),
            ("alpha", "future_relative_utility_20d"),
        )
    ):
        features = roles[role]
        train_x, test_x, medians = _fill_features(
            train, test, features
        )
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.04,
            max_iter=160,
            max_leaf_nodes=11,
            min_samples_leaf=35,
            l2_regularization=2.0,
            random_state=seed + offset,
        )
        model.fit(train_x, train[target])
        raw = pd.Series(model.predict(test_x), index=test.index)
        center = float(train[target].median())
        scale = float(
            train[target].quantile(0.75)
            - train[target].quantile(0.25)
        )
        scale = max(scale, 1e-6)
        predictions[f"head_{role}"] = raw.sub(center).div(scale).clip(
            -5.0, 5.0
        )
        actual = test[target].astype(float)
        baseline = np.full(len(actual), center)
        model_mae = mean_absolute_error(actual, raw)
        baseline_mae = mean_absolute_error(actual, baseline)
        diagnostics.append(
            {
                "role": role,
                "target": target,
                "rows": int(len(actual)),
                "feature_count": len(features),
                "mae": float(model_mae),
                "baseline_mae": float(baseline_mae),
                "mae_skill": (
                    1.0 - model_mae / baseline_mae
                    if baseline_mae > 0.0
                    else float("nan")
                ),
                "mse": float(mean_squared_error(actual, raw)),
                "spearman": float(actual.corr(raw, method="spearman")),
            }
        )
        state[role] = {
            "model": model,
            "medians": medians.to_dict(),
            "features": features,
            "center": center,
            "scale": scale,
        }

    recovery_features = roles["recovery"]
    train_x, test_x, medians = _fill_features(
        train, test, recovery_features
    )
    train_mask = train["stress_context"].eq(1.0)
    test_mask = test["stress_context"].eq(1.0)
    y = train.loc[train_mask, "recovery_label"].astype(int)
    if y.nunique() < 2:
        raise ValueError("blocked: recovery head has one training class")
    positive = max(float(y.mean()), 1e-6)
    sample_weight = np.where(
        y.eq(1), 0.5 / positive, 0.5 / max(1.0 - positive, 1e-6)
    )
    recovery = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=160,
        max_leaf_nodes=9,
        min_samples_leaf=25,
        l2_regularization=2.0,
        random_state=seed + 2,
    )
    recovery.fit(
        train_x.loc[train_mask],
        y,
        sample_weight=sample_weight,
    )
    recovery_probability = pd.Series(
        recovery.predict_proba(test_x)[:, 1], index=test.index
    )
    predictions["head_recovery"] = recovery_probability
    if test_mask.any() and test.loc[
        test_mask, "recovery_label"
    ].nunique() > 1:
        recovery_auc = float(
            roc_auc_score(
                test.loc[test_mask, "recovery_label"],
                recovery_probability.loc[test_mask],
            )
        )
    else:
        recovery_auc = float("nan")
    diagnostics.append(
        {
            "role": "recovery",
            "target": "recovery_label",
            "rows": int(test_mask.sum()),
            "feature_count": len(recovery_features),
            "roc_auc": recovery_auc,
            "positive_rate": float(
                test.loc[test_mask, "recovery_label"].mean()
            ),
        }
    )
    state["recovery"] = {
        "model": recovery,
        "medians": medians.to_dict(),
        "features": recovery_features,
    }
    predictions = predictions.join(
        test.loc[
            :,
            [
                "future_downside_risk_20d",
                "future_relative_utility_20d",
                "recovery_label",
                "stress_context",
                "soxl_forward_open_return",
                "soxx_forward_open_return",
                *TARGET_WEIGHT_COLUMNS,
            ],
        ]
    )
    return predictions, state, diagnostics


def _cross_fitted_heads(
    *,
    dataset,
    frame: pd.DataFrame,
    roles: dict[str, list[str]],
    spec: RoleSeparatedAllocationSpec,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    states: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for year in range(HEAD_START_YEAR, spec.development_years[-1] + 1):
        test = _prepare(dataset, (f"{year}-01-01", f"{year}-12-31"))
        test = test.loc[test.index.year == year]
        start_position = frame.index.searchsorted(
            pd.Timestamp(f"{year}-01-01")
        )
        train_end = frame.index[
            start_position - spec.purge_rows - 1
        ]
        train = _prepare(dataset, (spec.train_start, train_end))
        output, state, fold_diagnostics = _fit_predict_heads(
            train=train,
            test=test,
            roles=roles,
            seed=spec.seed + year,
        )
        output["head_fold_year"] = year
        output["head_train_end"] = train_end
        outputs.append(output)
        states[f"fold_{year}"] = state
        diagnostics.extend(
            [
                {
                    "fold_year": year,
                    "train_end": train_end.date().isoformat(),
                    "selection_uses_outer_test": False,
                    **row,
                }
                for row in fold_diagnostics
            ]
        )
    return (
        pd.concat(outputs).sort_index(),
        states,
        pd.DataFrame(diagnostics),
    )


def _outer_heads(
    *,
    dataset,
    roles: dict[str, list[str]],
    spec: RoleSeparatedAllocationSpec,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    train = _prepare(dataset, (spec.train_start, spec.development_end))
    test = _prepare(dataset, "outer")
    output, state, diagnostics = _fit_predict_heads(
        train=train,
        test=test,
        roles=roles,
        seed=spec.seed,
    )
    output["head_fold_year"] = output.index.year
    output["head_train_end"] = pd.Timestamp(spec.development_end)
    return (
        output,
        state,
        pd.DataFrame(
            [
                {
                    "fold_year": "outer_refit",
                    "train_end": spec.development_end,
                    "selection_uses_outer_test": False,
                    **row,
                }
                for row in diagnostics
            ]
        ),
    )


def _allocator_model(
    candidate: AllocatorCandidate,
    spec: RoleSeparatedAllocationSpec,
) -> nn.Module:
    if candidate.router:
        return RouterHeadAllocator(
            hidden_units=spec.hidden_units,
            expert_count=spec.router_experts,
        )
    return DirectHeadAllocator(hidden_units=spec.hidden_units)


def _head_scaler(train: pd.DataFrame) -> dict[str, pd.Series]:
    mean = train.loc[:, HEAD_COLUMNS].mean()
    scale = (
        train.loc[:, HEAD_COLUMNS]
        .std(ddof=1)
        .replace(0.0, 1.0)
        .fillna(1.0)
    )
    return {"mean": mean, "scale": scale}


def _head_tensor(
    frame: pd.DataFrame,
    scaler: dict[str, pd.Series],
    candidate: AllocatorCandidate,
) -> torch.Tensor:
    values = (
        (frame.loc[:, HEAD_COLUMNS] - scaler["mean"])
        / scaler["scale"]
    ).clip(-6.0, 6.0)
    if not candidate.use_hazard:
        values = values.copy()
        values["head_hazard"] = 0.0
    return torch.tensor(
        values.to_numpy(dtype=np.float32), dtype=torch.float32
    )


def _directional_consistency_loss(
    *,
    model: nn.Module,
    features: torch.Tensor,
    maximum_soxl_weight: float,
) -> torch.Tensor:
    base, _ = model(features, maximum_soxl_weight)
    hazard = features.clone()
    alpha = features.clone()
    recovery = features.clone()
    hazard[:, 0] = hazard[:, 0] + 0.25
    alpha[:, 1] = alpha[:, 1] + 0.25
    recovery[:, 2] = recovery[:, 2] + 0.25
    hazard_weights, _ = model(hazard, maximum_soxl_weight)
    alpha_weights, _ = model(alpha, maximum_soxl_weight)
    recovery_weights, _ = model(recovery, maximum_soxl_weight)
    return (
        torch.relu(base[:, 2] - hazard_weights[:, 2]).mean()
        + torch.relu(hazard_weights[:, 0] - base[:, 0]).mean()
        + torch.relu(base[:, 0] - alpha_weights[:, 0]).mean()
        + torch.relu(recovery_weights[:, 2] - base[:, 2]).mean()
    )


def _allocator_loss(
    *,
    model: nn.Module,
    features: torch.Tensor,
    weights: torch.Tensor,
    train: pd.DataFrame,
    candidate: AllocatorCandidate,
    spec: RoleSeparatedAllocationSpec,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_weights = torch.tensor(
        train.loc[:, TARGET_WEIGHT_COLUMNS].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    allocation_loss = nn.functional.mse_loss(weights, target_weights)
    turnover = 0.5 * torch.abs(weights[1:] - weights[:-1]).sum(dim=1)
    turnover_loss = turnover.mean()
    consistency_loss = _directional_consistency_loss(
        model=model,
        features=features,
        maximum_soxl_weight=spec.maximum_soxl_weight,
    )
    utility_loss = weights.sum() * 0.0
    drawdown_loss = weights.sum() * 0.0
    cvar_loss = weights.sum() * 0.0
    annual_loss = weights.sum() * 0.0
    cash_floor_loss = torch.relu(0.15 - weights[:, 2].mean()).pow(2)

    if candidate.objective == "relative_wealth_cvar":
        soxl_return = torch.tensor(
            train["soxl_forward_open_return"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        soxx_return = torch.tensor(
            train["soxx_forward_open_return"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        portfolio = (
            weights[:, 0] * soxl_return + weights[:, 1] * soxx_return
        )
        cost = torch.cat(
            [
                torch.zeros(1),
                turnover * (spec.training_cost_bps / 10_000.0),
            ]
        )
        net = portfolio - cost
        relative_log = torch.log1p(net.clamp(min=-0.95)) - torch.log1p(
            soxx_return.clamp(min=-0.95)
        )
        utility_loss = -252.0 * relative_log.mean()
        log_wealth = torch.cumsum(
            torch.log1p(net.clamp(min=-0.95)), dim=0
        )
        peak = torch.cummax(log_wealth, dim=0).values
        drawdown = torch.exp(log_wealth - peak) - 1.0
        drawdown_loss = torch.relu(-drawdown - 0.20).pow(2).mean()
        tail_count = max(1, int(round(len(net) * 0.05)))
        cvar = torch.topk(-net, k=tail_count).values.mean()
        cvar_loss = torch.relu(cvar - 0.08).pow(2)
        year_values = train.index.year
        annual_penalties: list[torch.Tensor] = []
        for year in sorted(np.unique(year_values)):
            mask = torch.tensor(year_values == year)
            annual_relative = 252.0 * relative_log[mask].mean()
            annual_penalties.append(torch.relu(-annual_relative).pow(2))
        annual_loss = torch.stack(annual_penalties).mean()
        loss = (
            utility_loss
            + 3.0 * annual_loss
            + 35.0 * drawdown_loss
            + 10.0 * cvar_loss
            + 0.03 * turnover_loss
            + 0.20 * allocation_loss
            + 0.75 * consistency_loss
            + 4.0 * cash_floor_loss
        )
    elif candidate.objective == "oracle_weight_mse":
        loss = (
            4.0 * allocation_loss
            + 0.10 * turnover_loss
            + 0.75 * consistency_loss
            + 2.0 * cash_floor_loss
        )
    else:
        raise ValueError(f"blocked: unknown objective {candidate.objective}")

    diagnostics = {
        "loss": float(loss.detach()),
        "allocation_loss": float(allocation_loss.detach()),
        "utility_loss": float(utility_loss.detach()),
        "drawdown_loss": float(drawdown_loss.detach()),
        "cvar_loss": float(cvar_loss.detach()),
        "annual_loss": float(annual_loss.detach()),
        "turnover_loss": float(turnover_loss.detach()),
        "consistency_loss": float(consistency_loss.detach()),
        "cash_floor_loss": float(cash_floor_loss.detach()),
    }
    return loss, diagnostics


def _fit_predict_allocator(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate: AllocatorCandidate,
    spec: RoleSeparatedAllocationSpec,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    required = [
        *HEAD_COLUMNS,
        "soxl_forward_open_return",
        "soxx_forward_open_return",
        *TARGET_WEIGHT_COLUMNS,
    ]
    train = train.dropna(subset=required).sort_index().copy()
    test = test.dropna(subset=required).sort_index().copy()
    if train.empty or test.empty:
        raise ValueError("blocked: empty allocator train or test frame")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = _head_scaler(train)
    train_x = _head_tensor(train, scaler, candidate)
    test_x = _head_tensor(test, scaler, candidate)
    model = _allocator_model(candidate, spec)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    diagnostics: dict[str, float] = {}
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    model.train()
    for _ in range(spec.epochs):
        optimizer.zero_grad()
        weights, _ = model(train_x, spec.maximum_soxl_weight)
        loss, diagnostics = _allocator_loss(
            model=model,
            features=train_x,
            weights=weights,
            train=train,
            candidate=candidate,
            spec=spec,
        )
        if not torch.isfinite(loss):
            raise ValueError(
                f"blocked: non-finite allocator loss for {candidate.name}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().clone()
                for key, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise ValueError("blocked: allocator failed to retain state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        weights, auxiliary = model(
            test_x, spec.maximum_soxl_weight
        )
        test_weights = weights.cpu().numpy()
        auxiliary_values = auxiliary.cpu().numpy()
    output = pd.DataFrame(
        test_weights,
        index=test.index,
        columns=["soxl_weight", "soxx_weight", "cash_weight"],
    )
    output = output.join(test.loc[:, HEAD_COLUMNS])
    if candidate.router:
        for expert in range(spec.router_experts):
            output[f"router_probability_{expert}"] = auxiliary_values[
                :, expert
            ]
        diagnostics["router_entropy"] = float(
            (
                -auxiliary_values
                * np.log(np.clip(auxiliary_values, 1e-12, 1.0))
            )
            .sum(axis=1)
            .mean()
        )
    state = {
        "model_state": {
            key: value.detach().cpu().numpy()
            for key, value in model.state_dict().items()
        },
        "scaler": {
            name: values.to_dict() for name, values in scaler.items()
        },
        "candidate": asdict(candidate),
        "spec": {
            "hidden_units": spec.hidden_units,
            "router_experts": spec.router_experts,
            "maximum_soxl_weight": spec.maximum_soxl_weight,
            "training_cost_bps": spec.training_cost_bps,
        },
    }
    diagnostics["best_loss"] = best_loss
    return output, state, diagnostics


def _development_allocator(
    *,
    heads: pd.DataFrame,
    candidate: AllocatorCandidate,
    spec: RoleSeparatedAllocationSpec,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    states: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for year in spec.development_years:
        train = heads.loc[heads.index.year < year].copy()
        test = heads.loc[heads.index.year == year].copy()
        output, state, fit_diagnostics = _fit_predict_allocator(
            train=train,
            test=test,
            candidate=candidate,
            spec=spec,
            seed=spec.seed + year,
        )
        output["candidate"] = candidate.name
        output["segment"] = "development_walk_forward"
        output["fold_year"] = year
        predictions.append(output)
        states[f"fold_{year}"] = state
        diagnostics.append(
            {
                "candidate": candidate.name,
                "fold_year": year,
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


def _outer_allocator(
    *,
    development_heads: pd.DataFrame,
    outer_heads: pd.DataFrame,
    candidate: AllocatorCandidate,
    spec: RoleSeparatedAllocationSpec,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    output, state, diagnostics = _fit_predict_allocator(
        train=development_heads,
        test=outer_heads,
        candidate=candidate,
        spec=spec,
        seed=spec.seed,
    )
    output["candidate"] = candidate.name
    output["segment"] = "outer_diagnostic"
    output["fold_year"] = output.index.year
    return output, state, diagnostics


def _profile(candidate: AllocatorCandidate) -> DirectLossProfile:
    return DirectLossProfile(
        name=candidate.name,
        maximum_soxl_weight=0.30,
        negative_year_penalty=0.0,
        year_dispersion_penalty=0.0,
        relative_drawdown_penalty=0.0,
        turnover_penalty=0.0,
    )


def _candidate_backtests(
    *,
    frame: pd.DataFrame,
    prediction_map: dict[str, pd.DataFrame],
    segment: str,
    spec: RoleSeparatedAllocationSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons: list[dict[str, Any]] = []
    annual_frames: list[pd.DataFrame] = []
    returns: list[pd.DataFrame] = []
    positions: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        profile = _profile(candidate)
        predictions = prediction_map[candidate.name]
        for cost_bps in spec.costs_bps:
            metrics, result, benchmark = _evaluate(
                market=frame,
                predictions=predictions,
                profile=profile,
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
            returns.append(
                result.assign(
                    segment=segment,
                    candidate=candidate.name,
                    cost_bps=cost_bps,
                )
            )
            positions.append(
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
        pd.concat(returns, ignore_index=True),
        pd.concat(positions, ignore_index=True),
    )


def _head_summary(head_diagnostics: pd.DataFrame) -> pd.DataFrame:
    fold_year = pd.to_numeric(
        head_diagnostics["fold_year"], errors="coerce"
    )
    development = head_diagnostics.loc[
        fold_year.between(
            DEVELOPMENT_YEARS[0], DEVELOPMENT_YEARS[-1]
        )
    ].copy()
    rows: list[dict[str, Any]] = []
    for role, group in development.groupby("role"):
        row: dict[str, Any] = {
            "role": role,
            "folds": int(len(group)),
        }
        if role in {"hazard", "alpha"}:
            row.update(
                {
                    "mean_mae_skill": float(group["mae_skill"].mean()),
                    "positive_skill_years": int(
                        group["mae_skill"].gt(0.0).sum()
                    ),
                    "mean_spearman": float(group["spearman"].mean()),
                    "same_sign_spearman_years": int(
                        group["spearman"]
                        .mul(group["spearman"].mean())
                        .gt(0.0)
                        .sum()
                    ),
                }
            )
        else:
            row.update(
                {
                    "mean_roc_auc": float(group["roc_auc"].mean()),
                    "auc_above_060_years": int(
                        group["roc_auc"].gt(0.60).sum()
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _development_selection(
    *,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    head_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        cost10 = comparison.loc[
            comparison["segment"].eq("development_walk_forward")
            & comparison["candidate"].eq(candidate.name)
            & comparison["cost_bps"].eq(10)
        ].iloc[0]
        cost25 = comparison.loc[
            comparison["segment"].eq("development_walk_forward")
            & comparison["candidate"].eq(candidate.name)
            & comparison["cost_bps"].eq(25)
        ].iloc[0]
        annual10 = annual.loc[
            annual["segment"].eq("development_walk_forward")
            & annual["candidate"].eq(candidate.name)
            & annual["cost_bps"].eq(10)
        ]
        rows.append(
            {
                "candidate": candidate.name,
                "router": candidate.router,
                "objective": candidate.objective,
                **cost10.to_dict(),
                "ratio_10bps": cost10[
                    "terminal_wealth_ratio_vs_soxx"
                ],
                "ratio_25bps": cost25[
                    "terminal_wealth_ratio_vs_soxx"
                ],
                "positive_relative_years": int(
                    annual10["terminal_wealth_ratio_vs_soxx"].gt(1.0).sum()
                ),
                "median_annual_ratio": float(
                    annual10["terminal_wealth_ratio_vs_soxx"].median()
                ),
                "minimum_annual_ratio": float(
                    annual10["terminal_wealth_ratio_vs_soxx"].min()
                ),
            }
        )
    selection = pd.DataFrame(rows)
    hazard = head_summary.loc[head_summary["role"].eq("hazard")].iloc[0]
    alpha = head_summary.loc[head_summary["role"].eq("alpha")].iloc[0]
    recovery = head_summary.loc[
        head_summary["role"].eq("recovery")
    ].iloc[0]
    selection["return_gate"] = selection["ratio_10bps"].gt(1.0)
    selection["cost_gate"] = selection["ratio_25bps"].gt(1.0)
    selection["year_gate"] = selection["positive_relative_years"].ge(4)
    selection["maxdd_gate"] = selection["max_drawdown_pct"].ge(-20.0)
    selection["worst_day_gate"] = selection["worst_day_pct"].ge(-12.0)
    selection["cvar_gate"] = selection["cvar5_pct"].ge(-8.0)
    selection["worst_5d_gate"] = selection[
        "worst_rolling_5d_pct"
    ].ge(-12.0)
    selection["cash_gate"] = selection["avg_cash_weight_pct"].ge(15.0)
    selection["capture_gate"] = selection["upside_capture_pct"].ge(60.0)
    selection["hazard_gate"] = (
        int(hazard["positive_skill_years"]) >= 5
        and float(hazard["mean_spearman"]) > 0.0
    )
    selection["alpha_gate"] = (
        int(alpha["same_sign_spearman_years"]) >= 5
        and float(alpha["mean_spearman"]) > 0.0
    )
    selection["recovery_gate"] = (
        int(recovery["auc_above_060_years"]) >= 5
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
        "capture_gate",
        "hazard_gate",
        "alpha_gate",
        "recovery_gate",
    ]
    selection["gate_count"] = selection[gates].sum(axis=1)
    selection["all_gate"] = selection[gates].all(axis=1)
    selection["selection_score"] = (
        np.log(selection["ratio_10bps"].clip(lower=1e-12))
        + 0.25 * np.log(selection["ratio_25bps"].clip(lower=1e-12))
        + 0.05 * selection["positive_relative_years"]
        + 0.02 * (selection["max_drawdown_pct"] + 20.0)
    )
    pool = selection.loc[selection["all_gate"]]
    if pool.empty:
        pool = selection
    selected_index = pool.sort_values(
        ["gate_count", "selection_score", "candidate"],
        ascending=[False, False, True],
    ).index[0]
    selection["diagnostic_selected"] = selection.index == selected_index
    selection["selection_policy"] = (
        "development_only_all_gates_else_gate_count_then_score"
    )
    return selection


def _seed_sensitivity(
    *,
    heads: pd.DataFrame,
    frame: pd.DataFrame,
    spec: RoleSeparatedAllocationSpec,
) -> pd.DataFrame:
    candidate = next(
        candidate
        for candidate in CANDIDATES
        if candidate.name == "crossfit_alpha_recovery_direct_mse"
    )
    rows: list[dict[str, Any]] = []
    for seed in SENSITIVITY_SEEDS:
        seed_spec = RoleSeparatedAllocationSpec(
            **{**asdict(spec), "seed": seed}
        )
        predictions, _, _ = _development_allocator(
            heads=heads,
            candidate=candidate,
            spec=seed_spec,
        )
        metrics10, result, benchmark = _evaluate(
            market=frame,
            predictions=predictions,
            profile=_profile(candidate),
            cost_bps=10,
        )
        metrics25, _, _ = _evaluate(
            market=frame,
            predictions=predictions,
            profile=_profile(candidate),
            cost_bps=25,
        )
        annual = _annual_metrics(result, benchmark, {})
        rows.append(
            {
                "candidate": candidate.name,
                "seed": seed,
                "ratio_10bps": metrics10[
                    "terminal_wealth_ratio_vs_soxx"
                ],
                "ratio_25bps": metrics25[
                    "terminal_wealth_ratio_vs_soxx"
                ],
                "max_drawdown_pct": metrics10["max_drawdown_pct"],
                "positive_relative_years": int(
                    annual["terminal_wealth_ratio_vs_soxx"].gt(1.0).sum()
                ),
                "turnover": metrics10["turnover"],
                "selection_uses_outer_test": False,
            }
        )
    return pd.DataFrame(rows)


def _window_metrics(result: pd.DataFrame) -> pd.DataFrame:
    signal_date = pd.to_datetime(result["signal_date"])
    windows = {
        "2020_pre_crash": ("2020-01-02", "2020-02-19"),
        "2020_crash": ("2020-02-20", "2020-03-23"),
        "2020_recovery_early": ("2020-03-24", "2020-05-29"),
        "2020_recovery_late": ("2020-06-01", "2020-08-31"),
        "2022_bear": ("2022-01-03", "2022-12-30"),
    }
    rows: list[dict[str, Any]] = []
    for name, (start, end) in windows.items():
        mask = signal_date.between(start, end)
        group = result.loc[mask]
        if group.empty:
            continue
        strategy_wealth = float((1.0 + group["strategy_return"]).prod())
        soxx_wealth = float((1.0 + group["soxx_return"]).prod())
        rows.append(
            {
                "window": name,
                "start": start,
                "end": end,
                "rows": int(len(group)),
                "terminal_wealth_ratio_vs_soxx": (
                    strategy_wealth / soxx_wealth
                ),
                "avg_cash_weight_pct": 100.0
                * float(group["cash_weight"].mean()),
                "avg_soxl_weight_pct": 100.0
                * float(group["soxl_weight"].mean()),
                "turnover": float(group["turnover"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _objective_gate_audit(
    *,
    selected: pd.Series,
    outer_selected: pd.Series,
    outer_annual: pd.DataFrame,
    head_summary: pd.DataFrame,
    seed_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    outer_years_passing = int(
        (
            outer_annual["terminal_wealth_ratio_vs_soxx"].gt(1.0)
            & outer_annual["max_drawdown_pct"].ge(-20.0)
            & outer_annual["worst_day_pct"].ge(-12.0)
        ).sum()
    )
    rows = [
        (
            "no_base_or_prior",
            True,
            True,
            "all candidates emit full SOXL/SOXX/cash weights",
        ),
        (
            "cross_fitted_heads",
            True,
            True,
            "allocator training only receives expanding-window OOF heads",
        ),
        (
            "outer_not_used_for_selection",
            True,
            not bool(selected["selection_uses_outer_test"]),
            "2023-2026 is selected-candidate validation only",
        ),
        (
            "development_all_gates",
            True,
            bool(selected["all_gate"]),
            f"gate_count={int(selected['gate_count'])}/12",
        ),
        (
            "outer_year_robustness",
            True,
            outer_years_passing >= 3,
            f"passing_outer_years={outer_years_passing}/4",
        ),
        (
            "outer_drawdown",
            True,
            float(outer_selected["max_drawdown_pct"]) >= -20.0,
            f"outer_maxdd={float(outer_selected['max_drawdown_pct']):.2f}%",
        ),
        (
            "seed_stability_10bps",
            True,
            bool(seed_sensitivity["ratio_10bps"].gt(1.0).all()),
            (
                f"seed_ratio_range="
                f"{float(seed_sensitivity['ratio_10bps'].min()):.4f}-"
                f"{float(seed_sensitivity['ratio_10bps'].max()):.4f}"
            ),
        ),
    ]
    for _, role in head_summary.iterrows():
        if role["role"] in {"hazard", "alpha"}:
            passed = (
                (
                    int(role["positive_skill_years"]) >= 5
                    if role["role"] == "hazard"
                    else int(role["same_sign_spearman_years"]) >= 5
                )
                and float(role["mean_spearman"]) > 0.0
            )
            detail = (
                f"positive_skill_years={int(role['positive_skill_years'])}, "
                f"same_sign_spearman_years="
                f"{int(role['same_sign_spearman_years'])}, "
                f"mean_spearman={float(role['mean_spearman']):.4f}"
            )
        else:
            passed = int(role["auc_above_060_years"]) >= 5
            detail = (
                f"auc_above_060_years="
                f"{int(role['auc_above_060_years'])}"
            )
        rows.append(
            (
                f"{role['role']}_learnability",
                True,
                bool(passed),
                detail,
            )
        )
    return pd.DataFrame(
        rows, columns=["gate", "required", "passed", "detail"]
    )


def _summary(
    *,
    spec: RoleSeparatedAllocationSpec,
    source_audit: dict[str, Any],
    taleb_manifest: dict[str, Any],
    dataset_audit: dict[str, Any],
    roles: dict[str, list[str]],
    head_summary: pd.DataFrame,
    seed_sensitivity: pd.DataFrame,
    selection: pd.DataFrame,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    regimes: pd.DataFrame,
    windows: pd.DataFrame,
    objective_audit: pd.DataFrame,
    objective_pass: bool,
    decision: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
) -> dict[str, Any]:
    selected = selection.loc[selection["diagnostic_selected"]].iloc[0]
    selected_name = str(selected["candidate"])
    outer10 = comparison.loc[
        comparison["segment"].eq("outer_diagnostic")
        & comparison["candidate"].eq(selected_name)
        & comparison["cost_bps"].eq(10)
    ].iloc[0]
    return {
        "experiment": spec.experiment,
        "system_of_record": "qlib",
        "provider_uri": spec.daily_provider_uri.resolve().as_posix(),
        "taleb_provider_uri": spec.taleb_provider_uri.resolve().as_posix(),
        "market": "US",
        "region": "us",
        "as_of": spec.as_of,
        "benchmark": "SOXX",
        "assets": ["SOXL", "SOXX", "cash"],
        "hypothesis": (
            "Role-separated expanding-window hazard, alpha, and recovery "
            "heads improve direct allocation by preventing slow tail-state "
            "features from contaminating short-horizon alpha."
        ),
        "architecture": {
            "base_position": None,
            "explicit_prior": None,
            "candidate_weight_grid": False,
            "head_training": "expanding_window_cross_fitted",
            "head_columns": list(HEAD_COLUMNS),
            "allocator_output": (
                "full SOXL/SOXX/cash weights; SOXL capped at 30%; "
                "no fixed sleeve"
            ),
            "router": "learned mixture of four full-allocation experts",
            "candidates": [asdict(candidate) for candidate in CANDIDATES],
            "final_weights_are_direct_model_outputs": True,
        },
        "feature_roles": roles,
        "targets": {
            "hazard": (
                "future 20d downside semivariance plus 1.5x worst path"
            ),
            "alpha": (
                "future 20d SOXL-SOXX return plus downside utility gap"
            ),
            "recovery": (
                "stress context with future SOXX 5d >=4% and adverse >=-4%"
            ),
        },
        "data_audit": {
            "base": source_audit,
            "taleb_provider": taleb_manifest,
            "dataset": dataset_audit,
        },
        "development": {
            "years": list(spec.development_years),
            "selection_uses_outer_test": False,
            "diagnostic_selected": _json_ready(selected.to_dict()),
            "head_summary": head_summary.to_dict(orient="records"),
            "seed_sensitivity": seed_sensitivity.to_dict(orient="records"),
        },
        "outer_diagnostic": {
            "start": OUTER_START,
            "selection_uses_outer_test": False,
            "selected_candidate_metrics_10bps": _json_ready(
                outer10.to_dict()
            ),
        },
        "objective_pass": objective_pass,
        "promotion_decision": decision,
        "objective_gate_audit": objective_audit.to_dict(orient="records"),
        "model_selection": selection.to_dict(orient="records"),
        "model_comparison": comparison.to_dict(orient="records"),
        "annual_metrics": annual.to_dict(orient="records"),
        "regime_metrics": regimes.to_dict(orient="records"),
        "window_metrics": windows.to_dict(orient="records"),
        "qlib_docs": [
            "https://qlib.readthedocs.io/en/latest/component/data.html",
            "https://qlib.readthedocs.io/en/latest/component/model.html",
            "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            "https://qlib.readthedocs.io/en/latest/component/recorder.html",
        ],
        "recorder_id": recorder_id,
        "recorder_uri": recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "sidecars": {
            "selected_model": "selected_model.json",
            "model_selection": "model_selection.csv",
            "model_comparison": "model_comparison.csv",
            "head_metrics": "head_metrics.csv",
            "head_summary": "head_summary.csv",
            "seed_sensitivity": "seed_sensitivity.csv",
            "backtest_benchmark_comparison": (
                "backtest_benchmark_comparison.csv"
            ),
            "backtest_walk_forward_metrics": (
                "backtest_walk_forward_metrics.csv"
            ),
            "risk_analysis": "risk_analysis.csv",
            "regime_metrics": "regime_metrics.csv",
            "window_metrics": "window_metrics.csv",
            "feature_roles": "feature_roles.csv",
            "objective_gate_audit": "objective_gate_audit.csv",
            "analysis_summary": "analysis_summary.json",
            "failure_modes": "failure_modes.md",
            "conclusion": "conclusion.md",
            "workflow_config": "workflow.yaml",
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": list(RECORDER_ONLY_ARTIFACTS),
        },
    }


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Role-separated direct allocation\n\n"
            "慢速hazard、相对alpha和快速recovery分别cross-fit，"
            "再由直接仓位模型输出SOXL/SOXX/cash。"
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
            "end_time='2026-07-02', freq='day')\n"
            "from src.workflow.role_separated_direct_allocation "
            "import run_role_separated_direct_allocation_card\n"
            "summary = run_role_separated_direct_allocation_card("
            "root=repo_root)\n"
            "summary['promotion_decision']"
        ),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "card = Path.cwd()\n"
            "pd.read_csv(card / 'head_summary.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'model_selection.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'window_metrics.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'objective_gate_audit.csv')"
        ),
        nbformat.v4.new_markdown_cell(
            (card / "conclusion.md").read_text(encoding="utf-8")
        ),
    ]
    nbformat.write(notebook, card / f"{summary['experiment']}.ipynb")


def _write_card(
    *,
    card: Path,
    summary: dict[str, Any],
    roles: dict[str, list[str]],
    head_metrics: pd.DataFrame,
    head_summary: pd.DataFrame,
    seed_sensitivity: pd.DataFrame,
    selection: pd.DataFrame,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    regimes: pd.DataFrame,
    windows: pd.DataFrame,
    objective_audit: pd.DataFrame,
) -> None:
    selected = selection.loc[selection["diagnostic_selected"]].iloc[0]
    selected_name = str(selected["candidate"])
    outer10 = comparison.loc[
        comparison["segment"].eq("outer_diagnostic")
        & comparison["candidate"].eq(selected_name)
        & comparison["cost_bps"].eq(10)
    ].iloc[0]
    (card / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8"
    )
    selection.to_csv(card / "model_selection.csv", index=False)
    comparison.to_csv(card / "model_comparison.csv", index=False)
    comparison.to_csv(
        card / "backtest_benchmark_comparison.csv", index=False
    )
    annual.to_csv(
        card / "backtest_walk_forward_metrics.csv", index=False
    )
    head_metrics.to_csv(card / "head_metrics.csv", index=False)
    head_summary.to_csv(card / "head_summary.csv", index=False)
    seed_sensitivity.to_csv(card / "seed_sensitivity.csv", index=False)
    regimes.to_csv(card / "regime_metrics.csv", index=False)
    windows.to_csv(card / "window_metrics.csv", index=False)
    objective_audit.to_csv(
        card / "objective_gate_audit.csv", index=False
    )
    comparison.loc[
        comparison["candidate"].eq(selected_name)
    ].to_csv(card / "risk_analysis.csv", index=False)
    pd.DataFrame(
        [
            {
                "role": role,
                "feature": feature,
                "shared_between_roles": sum(
                    feature in values for values in roles.values()
                )
                > 1,
            }
            for role, features in roles.items()
            for feature in features
        ]
    ).to_csv(card / "feature_roles.csv", index=False)
    selected_payload = {
        "candidate": selected_name,
        "selection_uses_outer_test": False,
        "diagnostic_only": not bool(summary["objective_pass"]),
        "architecture": summary["architecture"],
        "development_metrics": _json_ready(selected.to_dict()),
    }
    (card / "selected_model.json").write_text(
        json.dumps(selected_payload, indent=2), encoding="utf-8"
    )
    analysis = {
        "objective_pass": summary["objective_pass"],
        "promotion_decision": summary["promotion_decision"],
        "selected_candidate": selected_name,
        "development_ratio_10bps": float(selected["ratio_10bps"]),
        "development_ratio_25bps": float(selected["ratio_25bps"]),
        "development_max_drawdown_pct": float(
            selected["max_drawdown_pct"]
        ),
        "outer_ratio_10bps": float(
            outer10["terminal_wealth_ratio_vs_soxx"]
        ),
        "outer_max_drawdown_pct": float(outer10["max_drawdown_pct"]),
        "selection_uses_outer_test": False,
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    failed = objective_audit.loc[~objective_audit["passed"], "gate"].tolist()
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
        f"- 已执行三头cross-fit与直接仓位分配；选择完全不使用outer。\n"
        f"- 开发期诊断候选：`{selected_name}`；10bp相对SOXX="
        f"{float(selected['ratio_10bps']):.4f}，25bp="
        f"{float(selected['ratio_25bps']):.4f}，MaxDD="
        f"{float(selected['max_drawdown_pct']):.2f}%。\n"
        f"- Outer仅报告：相对SOXX="
        f"{float(outer10['terminal_wealth_ratio_vs_soxx']):.4f}，"
        f"MaxDD={float(outer10['max_drawdown_pct']):.2f}%。\n"
        f"- 晋升：`{summary['promotion_decision']}`。\n"
        f"- 主要失败门：{', '.join(failed) if failed else '无'}。\n"
        f"- 风险：头部可学性不足时，router不能创造新信息；"
        f"outer不得反向影响模型选择。\n"
        f"- 下一步：根据head_summary判断继续改目标，还是停止网络搜索并"
        f"补充长期PIT期权/高频数据。\n"
        f"- Qlib recorder：`{summary['recorder_id']}`。\n"
    )
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    workflow = (
        f"experiment: {summary['experiment']}\n"
        f"provider_uri: {summary['provider_uri']}\n"
        f"taleb_provider_uri: {summary['taleb_provider_uri']}\n"
        f"recorder_uri: {summary['recorder_uri']}\n"
        "market: US\n"
        "region: us\n"
        f"as_of: {summary['as_of']}\n"
        "assets: [SOXL, SOXX, cash]\n"
        "head_training: expanding_window_cross_fitted\n"
        "allocation_output: direct_full_weights\n"
        "base_position: null\n"
        "candidate_weight_grid: false\n"
        "selection_uses_outer_test: false\n"
        "artifact_policy:\n"
        "  policy: slim_card_recorder_first\n"
    )
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
    _write_notebook(card, summary)


def _run_card(spec: RoleSeparatedAllocationSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    source_frame, source_audit, _ = _load_frame(
        DirectContinuousAllocationSpec(
            root=spec.root,
            development_end=spec.development_end,
            outer=spec.outer,
        )
    )
    taleb_features, taleb_manifest = _load_taleb_features(
        TalebFragilityAllocationSpec(
            root=spec.root,
            development_end=spec.development_end,
            outer=spec.outer,
            as_of=spec.as_of,
        )
    )
    frame = source_frame.join(taleb_features, how="left")
    frame, _, _ = _attach_multitask_targets(frame)
    frame = _attach_role_targets_and_deltas(frame)
    roles = _feature_roles(frame)
    all_features = list(
        dict.fromkeys(
            feature
            for features in roles.values()
            for feature in features
        )
    )
    dataset, dataset_audit = _build_dataset(
        frame,
        all_features,
        spec,
        extra_labels=(
            "future_downside_risk_20d",
            "future_relative_utility_20d",
        ),
    )
    development_heads, head_states, development_head_metrics = (
        _cross_fitted_heads(
            dataset=dataset,
            frame=frame,
            roles=roles,
            spec=spec,
        )
    )
    outer_heads, outer_head_state, outer_head_metrics = _outer_heads(
        dataset=dataset,
        roles=roles,
        spec=spec,
    )
    head_states["outer_refit"] = outer_head_state
    head_metrics = pd.concat(
        [development_head_metrics, outer_head_metrics],
        ignore_index=True,
    )
    head_summary = _head_summary(development_head_metrics)

    development_predictions: dict[str, pd.DataFrame] = {}
    outer_predictions: dict[str, pd.DataFrame] = {}
    model_states: dict[str, Any] = {}
    allocator_diagnostics: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        development, states, diagnostics = _development_allocator(
            heads=development_heads,
            candidate=candidate,
            spec=spec,
        )
        outer, outer_state, outer_diagnostics = _outer_allocator(
            development_heads=development_heads,
            outer_heads=outer_heads,
            candidate=candidate,
            spec=spec,
        )
        states["outer_refit"] = outer_state
        model_states[candidate.name] = states
        development_predictions[candidate.name] = development
        outer_predictions[candidate.name] = outer
        allocator_diagnostics.append(diagnostics)
        allocator_diagnostics.append(
            pd.DataFrame(
                [
                    {
                        "candidate": candidate.name,
                        "fold_year": "outer_refit",
                        "train_rows": int(len(development_heads)),
                        "test_rows": int(len(outer)),
                        "selection_uses_outer_test": False,
                        **outer_diagnostics,
                    }
                ]
            )
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
    (
        outer_comparison,
        outer_annual,
        outer_returns,
        outer_positions,
    ) = _candidate_backtests(
        frame=frame,
        prediction_map=outer_predictions,
        segment="outer_diagnostic",
        spec=spec,
    )
    comparison = pd.concat(
        [development_comparison, outer_comparison], ignore_index=True
    )
    annual = pd.concat(
        [development_annual, outer_annual], ignore_index=True
    )
    selection = _development_selection(
        comparison=development_comparison,
        annual=development_annual,
        head_summary=head_summary,
    )
    selected = selection.loc[selection["diagnostic_selected"]].iloc[0]
    selected_name = str(selected["candidate"])
    seed_sensitivity = _seed_sensitivity(
        heads=development_heads,
        frame=frame,
        spec=spec,
    )
    outer_selected = outer_comparison.loc[
        outer_comparison["candidate"].eq(selected_name)
        & outer_comparison["cost_bps"].eq(10)
    ].iloc[0]
    selected_outer_annual = outer_annual.loc[
        outer_annual["candidate"].eq(selected_name)
        & outer_annual["cost_bps"].eq(10)
    ]
    objective_audit = _objective_gate_audit(
        selected=selected,
        outer_selected=outer_selected,
        outer_annual=selected_outer_annual,
        head_summary=head_summary,
        seed_sensitivity=seed_sensitivity,
    )
    objective_pass = bool(
        objective_audit.loc[
            objective_audit["required"], "passed"
        ].all()
    )
    decision = (
        "validated_role_separated_direct_allocation"
        if objective_pass
        else "not_promoted_role_or_risk_gates_failed"
    )
    selected_outer_result = outer_returns.loc[
        outer_returns["candidate"].eq(selected_name)
        & outer_returns["cost_bps"].eq(10)
    ].copy()
    selected_development_result = development_returns.loc[
        development_returns["candidate"].eq(selected_name)
        & development_returns["cost_bps"].eq(10)
    ].copy()
    regimes = _regime_metrics(frame, selected_outer_result)
    windows = _window_metrics(selected_development_result)

    all_head_predictions = pd.concat(
        [
            development_heads.assign(segment="development_walk_forward"),
            outer_heads.assign(segment="outer_diagnostic"),
        ]
    ).sort_index()
    all_predictions = pd.concat(
        [
            *development_predictions.values(),
            *outer_predictions.values(),
        ]
    ).sort_index()
    positions = pd.concat(
        [development_positions, outer_positions], ignore_index=True
    )
    strategy_returns = pd.concat(
        [development_returns, outer_returns], ignore_index=True
    )
    training_diagnostics = pd.concat(
        allocator_diagnostics, ignore_index=True
    )

    from qlib.workflow import R

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(
        prefix=f"{spec.experiment}_"
    ) as temporary:
        temp = Path(temporary)
        all_head_predictions.reset_index(names="signal_date").to_csv(
            temp / "head_predictions.csv", index=False
        )
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
        with (models_dir / "cross_fitted_heads.pkl").open("wb") as handle:
            pickle.dump(head_states, handle)
        for candidate_name, states in model_states.items():
            with (models_dir / f"{candidate_name}.pkl").open(
                "wb"
            ) as handle:
                pickle.dump(states, handle)

        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                benchmark="SOXX",
                assets="SOXL,SOXX,cash",
                candidate_count=len(CANDIDATES),
                head_training="expanding_window_cross_fitted",
                allocation_output="direct_full_weights",
                router="learned_four_expert_mixture",
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
                taleb_manifest=taleb_manifest,
                dataset_audit=dataset_audit,
                roles=roles,
                head_summary=head_summary,
                seed_sensitivity=seed_sensitivity,
                selection=selection,
                comparison=comparison,
                annual=annual,
                regimes=regimes,
                windows=windows,
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
                roles=roles,
                head_metrics=head_metrics,
                head_summary=head_summary,
                seed_sensitivity=seed_sensitivity,
                selection=selection,
                comparison=comparison,
                annual=annual,
                regimes=regimes,
                windows=windows,
                objective_audit=objective_audit,
            )
            for artifact_name in summary["sidecars"].values():
                recorder.log_artifact(
                    (card / artifact_name).as_posix(),
                    artifact_path=spec.experiment,
                )
            recorder.log_artifact(
                (card / f"{spec.experiment}.ipynb").as_posix(),
                artifact_path=spec.experiment,
            )
            _remove_qlib_code_snapshot_artifacts(recorder_local_dir)
    return _json_ready(summary)


if __name__ == "__main__":
    print(
        json.dumps(
            run_role_separated_direct_allocation_card(), indent=2
        )
    )
