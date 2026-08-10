from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.workflow.direct_continuous_soxx_allocation import (
    DirectContinuousAllocationSpec,
    PROFILES,
    _evaluate,
    _load_frame,
)
from src.workflow.pit_constituent_auxiliary_training_allocation import (
    AuxiliaryTrainingCandidate,
    PitAuxiliaryAllocationNet,
    PitAuxiliaryAllocationSpec,
    _project_legal_weights,
    _smooth_weights,
    _soxx_common_training_frame,
)


EXPERIMENT = "intraday_recovery_direct_allocation_v1"
SOURCE_EXPERIMENT = "pit_constituent_auxiliary_training_allocation_v1"
CARD_TYPE = "intraday_recovery_direct_three_asset_allocation_card"

FEATURE_COLUMNS = (
    "portfolio_drawdown",
    "current_beta_cap",
    "raw_effective_beta",
    "risk_state",
    "early_ret_soxl",
    "early_ret_soxx",
    "early_dislocation",
    "early_range_soxl",
    "early_range_soxx",
    "early_from_low_soxl",
    "early_from_low_soxx",
    "log_volume_soxl",
    "log_volume_soxx",
    "raw_soxl_weight",
    "raw_soxx_weight",
    "raw_cash_weight",
    "tail_probability",
    "alpha_probability",
    "portfolio_ret_1d",
    "portfolio_ret_3d",
    "portfolio_ret_5d",
)


@dataclass(frozen=True)
class IntradayRecoveryDirectAllocationSpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = SOURCE_EXPERIMENT
    development_years: tuple[int, ...] = tuple(range(2016, 2023))
    intraday_selection_years: tuple[int, ...] = (2019, 2020, 2021, 2022)
    outer_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    first_intraday_action_year: int = 2020
    intraday_bars_observed: int = 4
    transaction_cost_bps: float = 10.0
    stress_cost_bps: float = 25.0
    teacher_recovery_beta: float = 1.2
    minimum_cash_weight: float = 0.15
    maximum_soxl_weight: float = 0.30
    recovery_eligibility_drawdown: float = 0.08
    stop_drawdown: float = 0.12
    defensive_beta: float = 0.20
    wait_days: int = 5
    ramp_days: int = 10
    hold_days: int = 10
    trailing_loss_days: int = 4
    trailing_loss_trigger: float = 0.05
    trailing_loss_beta: float = 0.60
    emergency_drawdown: float = 0.22
    emergency_beta: float = 0.30

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / self.source_experiment

    @property
    def intraday_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_15min"


def run_intraday_recovery_direct_allocation_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    spec = IntradayRecoveryDirectAllocationSpec(root=_resolve_root(root))
    return _run_card(spec)


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def project_legal_weights(
    raw_soxl_weight: float,
    raw_soxx_weight: float,
    beta_cap: float,
    *,
    minimum_cash_weight: float = 0.15,
    maximum_soxl_weight: float = 0.30,
) -> np.ndarray:
    raw_beta = 3.0 * raw_soxl_weight + raw_soxx_weight
    raw_risk = raw_soxl_weight + raw_soxx_weight
    target_beta = min(raw_beta, max(0.0, beta_cap))
    target_risk = min(raw_risk, target_beta, 1.0 - minimum_cash_weight)
    soxl = max(0.0, min(maximum_soxl_weight, (target_beta - target_risk) / 2.0))
    soxx = max(0.0, min(1.0 - minimum_cash_weight - soxl, target_risk - soxl))
    cash = 1.0 - soxl - soxx
    return np.array([soxl, soxx, cash], dtype="float64")


def mark_to_market_weights(
    weights: np.ndarray, asset_returns: np.ndarray
) -> np.ndarray:
    """Return actual portfolio weights after assets move, before the next trade."""
    value = float(1.0 + weights @ asset_returns)
    if value <= 0.0:
        raise ValueError("portfolio value must remain positive when marking weights")
    return weights * (1.0 + asset_returns) / value


def select_development_candidate(model_selection: pd.DataFrame) -> pd.Series:
    required = {
        "segment",
        "selection_uses_outer_test",
        "risk_window_pass_count",
        "positive_relative_window_count",
        "relative_ratio_p25",
        "terminal_wealth",
        "relative_ratio_median",
    }
    missing = required.difference(model_selection.columns)
    if missing:
        raise ValueError(f"model selection is missing columns: {sorted(missing)}")
    development = model_selection[
        model_selection["segment"].eq("development_walk_forward")
        & ~model_selection["selection_uses_outer_test"].astype(bool)
        & model_selection["risk_window_pass_count"].eq(7)
        & model_selection["positive_relative_window_count"].ge(5)
        & model_selection["relative_ratio_p25"].ge(0.95)
    ].copy()
    if development.empty:
        raise ValueError("no development-only candidate passes the selection gates")
    return development.sort_values(
        ["terminal_wealth", "relative_ratio_p25", "relative_ratio_median"],
        ascending=False,
    ).iloc[0]


def apply_fixed_decision_cost_stress(
    gross_returns: Iterable[float],
    turnover: Iterable[float],
    *,
    cost_bps: float,
) -> np.ndarray:
    gross = np.asarray(list(gross_returns), dtype="float64")
    turns = np.asarray(list(turnover), dtype="float64")
    if gross.shape != turns.shape:
        raise ValueError("gross returns and turnover must have identical shapes")
    return gross - turns * float(cost_bps) / 10_000.0


def _run_card(spec: IntradayRecoveryDirectAllocationSpec) -> dict[str, Any]:
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    frame, source_audit = _load_source_frame(spec)
    # The teacher intentionally stays at 1.2 beta for every action candidate.
    # The action beta is a conservative execution calibration, not the label.
    teacher_frame = _build_teacher_training_frame(
        frame,
        spec=spec,
        teacher_beta=spec.teacher_recovery_beta,
    )
    model_cache: dict[tuple[int, float], Any] = {}
    candidates: list[dict[str, Any]] = []
    development = frame[frame["segment"].eq("development_walk_forward")].copy()
    for c_value in (0.03, 0.10, 0.30, 1.00):
        for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
            for action_beta in (0.60, 0.80, 1.00, 1.20, 1.40):
                replay = _simulate_policy(
                    development,
                    spec=spec,
                    teacher_frame=teacher_frame,
                    c_value=c_value,
                    threshold=threshold,
                    action_beta=action_beta,
                    first_action_year=spec.first_intraday_action_year,
                    model_cache=model_cache,
                    cost_bps=spec.transaction_cost_bps,
                )
                metrics = _risk_metrics(replay["strategy_return"].to_numpy())
                if not _passes_risk(metrics):
                    continue
                stability = _development_stability(replay)
                candidates.append(
                    {
                        "segment": "development_walk_forward",
                        "selection_uses_outer_test": False,
                        "teacher_recovery_beta": spec.teacher_recovery_beta,
                        "action_beta": action_beta,
                        "logistic_c": c_value,
                        "probability_threshold": threshold,
                        **metrics,
                        **stability,
                        "avg_cash_weight_pct": 100.0 * replay["cash_weight"].mean(),
                        "recovery_action_count": int(replay["recovery_action"].sum()),
                    }
                )
    model_selection = pd.DataFrame(candidates)
    selected = select_development_candidate(model_selection)
    selected_params = {
        "teacher_recovery_beta": spec.teacher_recovery_beta,
        "action_beta": float(selected["action_beta"]),
        "logistic_c": float(selected["logistic_c"]),
        "probability_threshold": float(selected["probability_threshold"]),
    }
    dev_replay = _simulate_policy(
        development,
        spec=spec,
        teacher_frame=teacher_frame,
        c_value=selected_params["logistic_c"],
        threshold=selected_params["probability_threshold"],
        action_beta=selected_params["action_beta"],
        first_action_year=spec.first_intraday_action_year,
        model_cache=model_cache,
        cost_bps=spec.transaction_cost_bps,
    )
    outer = frame[frame["segment"].eq("outer_diagnostic")].copy()
    outer_replay = _simulate_policy(
        outer,
        spec=spec,
        teacher_frame=teacher_frame,
        c_value=selected_params["logistic_c"],
        threshold=selected_params["probability_threshold"],
        action_beta=selected_params["action_beta"],
        first_action_year=min(spec.outer_years),
        model_cache=model_cache,
        cost_bps=spec.transaction_cost_bps,
    )
    baseline_outer = _simulate_policy(
        outer,
        spec=spec,
        teacher_frame=teacher_frame,
        c_value=selected_params["logistic_c"],
        threshold=2.0,
        action_beta=selected_params["action_beta"],
        first_action_year=min(spec.outer_years),
        model_cache=model_cache,
        cost_bps=spec.transaction_cost_bps,
    )
    fixed_25 = outer_replay.copy()
    fixed_25["strategy_return"] = apply_fixed_decision_cost_stress(
        fixed_25["gross_return"],
        fixed_25["turnover"],
        cost_bps=spec.stress_cost_bps,
    )
    dev_metrics = _risk_metrics(dev_replay["strategy_return"].to_numpy())
    outer_metrics = _risk_metrics(outer_replay["strategy_return"].to_numpy())
    baseline_metrics = _risk_metrics(baseline_outer["strategy_return"].to_numpy())
    raw_outer = outer.copy()
    raw_metrics = _risk_metrics(raw_outer["strategy_return"].to_numpy())
    stress_metrics = _risk_metrics(fixed_25["strategy_return"].to_numpy())
    annual_metrics = _annual_metrics(outer_replay)
    from src.workflow.hierarchical_four_asset_policy import _benchmark_efficiency

    efficiency_frame = outer_replay.assign(soxs_weight=0.0)
    benchmark_efficiency = _benchmark_efficiency(
        efficiency_frame,
        cost_bps=int(spec.transaction_cost_bps),
    )
    soxx_wealth = float(np.prod(1.0 + outer_replay["soxx_return"].to_numpy()))
    terminal_wealth_ratio_vs_soxx = outer_metrics["terminal_wealth"] / soxx_wealth
    up_mask = outer_replay["soxl_return"].gt(0.0)
    upside_capture_pct = float(
        100.0
        * outer_replay.loc[up_mask, "strategy_return"].sum()
        / outer_replay.loc[up_mask, "soxl_return"].sum()
    )
    annual_relative_pass_count = int(
        annual_metrics["terminal_wealth_ratio_vs_soxx"].gt(1.0).sum()
    )
    annual_risk_pass_count = int(
        annual_metrics.apply(
            lambda row: _passes_risk(row.to_dict()),
            axis=1,
        ).sum()
    )
    risk_analysis = pd.DataFrame(
        [
            {
                "variant": "daily_risk_controller",
                "segment": "outer_diagnostic",
                "cost_bps": spec.transaction_cost_bps,
                **baseline_metrics,
                "avg_cash_weight_pct": 100.0 * baseline_outer["cash_weight"].mean(),
            },
            {
                "variant": "selected_direct_intraday_recovery",
                "segment": "outer_diagnostic",
                "cost_bps": spec.transaction_cost_bps,
                **outer_metrics,
                "avg_cash_weight_pct": 100.0 * outer_replay["cash_weight"].mean(),
            },
            {
                "variant": "selected_direct_intraday_recovery_fixed_decisions",
                "segment": "outer_diagnostic",
                "cost_bps": spec.stress_cost_bps,
                **stress_metrics,
                "avg_cash_weight_pct": 100.0 * fixed_25["cash_weight"].mean(),
            },
        ]
    )
    cost_sensitivity = risk_analysis[
        risk_analysis["variant"].str.startswith("selected_direct_intraday_recovery")
    ].copy()
    decision = (
        "promoted_research_candidate_intraday_recovery_direct_allocation"
        if _passes_risk(outer_metrics)
        and _passes_risk(stress_metrics)
        and outer_metrics["return_pct"] >= baseline_metrics["return_pct"]
        and annual_relative_pass_count >= 3
        and annual_risk_pass_count >= 3
        and benchmark_efficiency["return_maxdd"]
        > benchmark_efficiency["same_avg_risk_soxl_return_maxdd"]
        and benchmark_efficiency["return_maxdd"]
        > benchmark_efficiency["matched_beta_return_maxdd"]
        else "not_promoted_intraday_recovery_gate_failed"
    )
    recorder_info = _record_qlib_artifacts(
        spec,
        selected_params=selected_params,
        dev_metrics=dev_metrics,
        outer_metrics=outer_metrics,
        baseline_metrics=baseline_metrics,
        stress_metrics=stress_metrics,
        model_selection=model_selection,
        outer_replay=outer_replay,
        teacher_frame=teacher_frame,
        source_audit=source_audit,
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "decision": decision,
        "selection_uses_outer_test": False,
        "promotion_scope": "revised_soxx_relative_cash_and_risk_goal",
        "assets": ["SOXL", "SOXX", "cash"],
        "allocation_output": "direct_legal_three_asset_weights_at_open_and_after_first_60m",
        "method": {
            "teacher_label": "counterfactual benefit of recovery to 1.2 effective beta",
            "execution_action": "validation-selected conservative effective-beta target",
            "reason": "asymmetric teacher-student calibration limits false-positive leverage",
            "intraday_observation": "first four regular-session 15-minute bars",
            "walk_forward": "annual expanding training; each outer year may use only prior years",
        },
        "selected_model": selected_params,
        "development_metrics_10bps": dev_metrics,
        "outer_diagnostic_metrics_10bps": outer_metrics,
        "raw_model_outer_metrics_10bps": raw_metrics,
        "daily_controller_outer_metrics_10bps": baseline_metrics,
        "fixed_decision_stress_metrics_25bps": stress_metrics,
        "benchmark_comparison_10bps": {
            "terminal_wealth_ratio_vs_soxx": terminal_wealth_ratio_vs_soxx,
            "upside_capture_pct": upside_capture_pct,
            **benchmark_efficiency,
        },
        "legacy_gate_status": {
            "upside_capture_floor_pct": 60.0,
            "upside_capture_gate_pass": upside_capture_pct >= 60.0,
            "note": "legacy capture gate is reported but is not part of the revised SOXX-relative goal",
        },
        "cash_timing_effect": {
            "outer_return_improvement_pct_points": outer_metrics["return_pct"]
            - baseline_metrics["return_pct"],
            "outer_cash_reduction_pct_points": 100.0
            * (baseline_outer["cash_weight"].mean() - outer_replay["cash_weight"].mean()),
            "raw_model_avg_cash_weight_pct": 100.0 * raw_outer["cash_weight"].mean(),
            "daily_controller_avg_cash_weight_pct": 100.0
            * baseline_outer["cash_weight"].mean(),
            "selected_model_avg_cash_weight_pct": 100.0
            * outer_replay["cash_weight"].mean(),
            "risk_cash_added_vs_raw_pct_points": 100.0
            * (outer_replay["cash_weight"].mean() - raw_outer["cash_weight"].mean()),
            "risk_cash_recovered_vs_daily_controller_pct": 100.0
            * (
                baseline_outer["cash_weight"].mean()
                - outer_replay["cash_weight"].mean()
            )
            / (
                baseline_outer["cash_weight"].mean()
                - raw_outer["cash_weight"].mean()
            ),
        },
        "outer_annual_gate_summary": {
            "relative_pass_count": annual_relative_pass_count,
            "risk_pass_count": annual_risk_pass_count,
            "required_year_count": 3,
        },
        "source_audit": source_audit,
        "risk_gates": {
            "max_drawdown_floor_pct": -30.0,
            "worst_day_floor_pct": -12.0,
            "cvar5_floor_pct": -8.0,
            "worst_rolling_5d_floor_pct": -12.0,
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/positions.csv",
                f"artifacts/{spec.experiment}/strategy_returns.csv",
                f"artifacts/{spec.experiment}/recovery_training_frame.csv",
                f"artifacts/{spec.experiment}/model_selection_full.csv",
            ],
        },
        **recorder_info,
    }
    _write_card_sidecars(
        spec,
        summary=summary,
        selected=selected,
        model_selection=model_selection,
        risk_analysis=risk_analysis,
        annual_metrics=annual_metrics,
        cost_sensitivity=cost_sensitivity,
    )
    return summary


def _load_source_frame(
    spec: IntradayRecoveryDirectAllocationSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_summary = json.loads(
        (spec.source_card_dir / "summary.json").read_text(encoding="utf-8")
    )
    recorder_dir, pinned_recorder_id = _resolve_source_recorder(
        spec, source_summary=source_summary
    )
    artifact_dir = recorder_dir / "artifacts" / spec.source_experiment
    if pinned_recorder_id is not None:
        returns, predictions, frozen_audit = _extend_frozen_source_outputs(
            spec,
            artifact_dir=artifact_dir,
        )
    else:
        returns = pd.read_csv(
            artifact_dir / "strategy_returns.csv",
            parse_dates=["signal_date", "trade_date", "exit_date"],
        )
        predictions = pd.read_csv(
            artifact_dir / "predictions.csv",
            parse_dates=["signal_date"],
        )
        frozen_audit = {
            "source_refresh_mode": "latest_source_summary_artifacts",
            "source_model_frozen": False,
        }
    returns = (
        returns[returns["cost_bps"].eq(spec.transaction_cost_bps)]
        .drop_duplicates(["segment", "signal_date"])
        .copy()
    )
    predictions = predictions.drop_duplicates(["segment", "signal_date"])
    frame = returns.merge(
        predictions[
            ["segment", "signal_date", "tail_probability", "alpha_probability"]
        ],
        on=["segment", "signal_date"],
        how="inner",
    )
    intraday = _load_first_hour_features(spec, end=frame["trade_date"].max())
    frame = frame.merge(
        intraday,
        left_on="trade_date",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    frame = frame.sort_values(["segment", "signal_date"]).reset_index(drop=True)
    intraday_manifest = json.loads(
        (spec.intraday_provider_uri / "qlib_store_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    audit = {
        "source_experiment": spec.source_experiment,
        "source_recorder_id": (
            pinned_recorder_id or source_summary["recorder_id"]
        ),
        "source_recorder_local_dir": str(recorder_dir),
        "source_artifact_dir": str(artifact_dir),
        **frozen_audit,
        "intraday_provider_uri": str(spec.intraday_provider_uri.resolve()),
        "intraday_provider_as_of": intraday_manifest.get("as_of"),
        "intraday_feature_timing": "bar_10_30_close_known_before_recovery_decision",
        "intraday_coverage_start": str(intraday.index.min().date()),
        "intraday_coverage_end": str(intraday.index.max().date()),
        "rows": int(len(frame)),
        "rows_with_intraday_features": int(frame["early_ret_soxx"].notna().sum()),
    }
    return frame, audit


def _resolve_source_recorder(
    spec: IntradayRecoveryDirectAllocationSpec,
    *,
    source_summary: dict[str, Any],
) -> tuple[Path, str | None]:
    workflow_path = spec.card_dir / "workflow.yaml"
    source_recorder_id: str | None = None
    if workflow_path.exists():
        workflow = yaml.safe_load(
            workflow_path.read_text(encoding="utf-8")
        ) or {}
        value = workflow.get("source_recorder_id")
        if value:
            source_recorder_id = str(value)
    if source_recorder_id is None:
        return Path(source_summary["recorder_local_dir"]), None
    matches = sorted(
        spec.source_card_dir.glob(
            f"qlib_recorder/*/{source_recorder_id}"
        )
    )
    if len(matches) != 1:
        raise ValueError(
            "blocked: pinned source recorder could not be resolved "
            f"uniquely: {source_recorder_id}"
        )
    return matches[0], source_recorder_id


def _series_scaler(
    packed: dict[str, dict[str, float]],
) -> dict[str, pd.Series]:
    return {
        name: pd.Series(values, dtype="float64")
        for name, values in packed.items()
    }


def _transform_frozen_features(
    frame: pd.DataFrame,
    *,
    features: list[str],
    scaler: dict[str, dict[str, float]],
) -> torch.Tensor:
    unpacked = _series_scaler(scaler)
    values = frame.loc[:, features].fillna(unpacked["median"])
    scaled = (
        (values - unpacked["mean"]) / unpacked["scale"]
    ).clip(-8.0, 8.0)
    return torch.tensor(
        scaled.to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )


def _frozen_outer_predictions(
    *,
    market: pd.DataFrame,
    common: pd.DataFrame,
    state: dict[str, Any],
    spec: PitAuxiliaryAllocationSpec,
) -> pd.DataFrame:
    backtest_required = [
        "soxl_forward_open_return",
        "soxx_forward_open_return",
    ]
    usable = market.loc[
        market.index >= pd.Timestamp(spec.outer[0])
    ].dropna(subset=backtest_required)
    primary_features = list(state["primary_features"])
    common_features = list(state["common_features"])
    primary_x = _transform_frozen_features(
        usable,
        features=primary_features,
        scaler=state["primary_scaler"],
    )
    common_x = _transform_frozen_features(
        common.reindex(usable.index),
        features=common_features,
        scaler=state["common_scaler"],
    )
    model = PitAuxiliaryAllocationNet(
        primary_feature_count=len(primary_features),
        common_feature_count=len(common_features),
        hidden_units=spec.hidden_units,
        tail_hidden_units=spec.tail_hidden_units,
    )
    model.load_state_dict(
        {
            name: torch.tensor(values)
            for name, values in state["model_state"].items()
        }
    )
    model.eval()
    with torch.no_grad():
        logits, tail_logit, alpha_logit = model(primary_x, common_x)
        weights = _project_legal_weights(
            logits,
            maximum_soxl_weight=spec.maximum_soxl_weight,
            minimum_cash_weight=spec.minimum_cash_weight,
        ).cpu().numpy()
        tail_probability = torch.sigmoid(
            tail_logit
        ).cpu().numpy()
        alpha_probability = torch.sigmoid(
            alpha_logit
        ).cpu().numpy()
    prediction = pd.DataFrame(
        weights,
        index=usable.index,
        columns=["soxl_weight", "soxx_weight", "cash_weight"],
    )
    prediction["tail_probability"] = tail_probability
    prediction["alpha_probability"] = alpha_probability
    candidate = AuxiliaryTrainingCandidate(**state["candidate"])
    prediction = _smooth_weights(
        prediction,
        spec=spec,
        candidate=candidate,
    )
    prediction["candidate"] = candidate.name
    prediction["segment"] = "outer_diagnostic"
    prediction["fold_year"] = prediction.index.year
    prediction["train_end"] = pd.Timestamp(spec.development_end)
    prediction.index.name = "signal_date"
    return prediction


def _extend_frozen_source_outputs(
    spec: IntradayRecoveryDirectAllocationSpec,
    *,
    artifact_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    stored_predictions = pd.read_csv(
        artifact_dir / "predictions.csv",
        parse_dates=["signal_date"],
    )
    stored_returns = pd.read_csv(
        artifact_dir / "strategy_returns.csv",
        parse_dates=["signal_date", "trade_date", "exit_date"],
    )
    selected = json.loads(
        (artifact_dir / "selected_model.json").read_text(
            encoding="utf-8"
        )
    )
    candidate = str(selected["candidate"])
    model_path = artifact_dir / "models" / f"{candidate}.pkl"
    with model_path.open("rb") as file:
        states = pickle.load(file)
    if "outer_refit" not in states:
        raise ValueError(
            "blocked: pinned source model has no outer_refit state"
        )
    pit_spec = PitAuxiliaryAllocationSpec(root=spec.root)
    market, market_audit, _ = _load_frame(
        DirectContinuousAllocationSpec(
            root=spec.root,
            train_start=pit_spec.train_start,
            development_years=pit_spec.development_years,
            development_end=pit_spec.development_end,
            outer=pit_spec.outer,
            purge_rows=pit_spec.purge_rows,
        )
    )
    common = _soxx_common_training_frame(
        pit_spec,
        end=market.index.max().date().isoformat(),
    ).reindex(market.index)
    outer_predictions = _frozen_outer_predictions(
        market=market,
        common=common,
        state=states["outer_refit"],
        spec=pit_spec,
    )
    _, outer_returns, _ = _evaluate(
        market=market,
        predictions=outer_predictions,
        profile=PROFILES[0],
        cost_bps=int(spec.transaction_cost_bps),
    )
    outer_returns["segment"] = "outer_diagnostic"
    outer_returns["candidate"] = candidate
    outer_returns["cost_bps"] = spec.transaction_cost_bps
    development_predictions = stored_predictions[
        stored_predictions["segment"].eq("development_walk_forward")
    ].copy()
    development_returns = stored_returns[
        stored_returns["segment"].eq("development_walk_forward")
        & stored_returns["cost_bps"].eq(spec.transaction_cost_bps)
    ].copy()
    predictions = pd.concat(
        [
            development_predictions,
            outer_predictions.reset_index(),
        ],
        ignore_index=True,
        sort=False,
    )
    returns = pd.concat(
        [development_returns, outer_returns],
        ignore_index=True,
        sort=False,
    )
    return returns, predictions, {
        "source_refresh_mode": (
            "pinned_recorder_frozen_model_inference_only"
        ),
        "source_model_frozen": True,
        "source_selected_candidate": candidate,
        "source_model_path": str(model_path),
        "source_market_provider_uri": market_audit.get(
            "execution_provider_uri"
        ),
        "source_daily_provider_uri": market_audit.get(
            "daily_provider_uri"
        ),
        "source_normalized_provider_uri": market_audit.get(
            "normalized_provider_uri"
        ),
        "source_market_end": str(market.index.max().date()),
        "source_prediction_end": str(
            outer_predictions.index.max().date()
        ),
        "source_training_end": pit_spec.development_end,
        "outer_test_used_for_selection": False,
    }


def _load_first_hour_features(
    spec: IntradayRecoveryDirectAllocationSpec, *, end: pd.Timestamp
) -> pd.DataFrame:
    qframe = load_qlib_feature_frame(
        QlibProviderConfig(
            provider_uri=spec.intraday_provider_uri,
            market="all",
            region="us",
            freq="15min",
            kernels=1,
        ),
        instruments=["SOXL", "SOXX"],
        fields=["$open", "$high", "$low", "$close", "$volume", "$bar_count"],
        start="2018-12-31 09:45:00",
        end=f"{pd.Timestamp(end).date()} 16:00:00",
    ).reset_index()
    qframe["datetime"] = pd.to_datetime(qframe["datetime"])
    qframe["date"] = qframe["datetime"].dt.normalize()
    qframe = qframe.sort_values(["instrument", "date", "datetime"])
    early = qframe.groupby(["instrument", "date"], group_keys=False).head(
        spec.intraday_bars_observed
    )
    grouped = early.groupby(["date", "instrument"]).agg(
        early_open=("$open", "first"),
        early_close=("$close", "last"),
        early_high=("$high", "max"),
        early_low=("$low", "min"),
        early_volume=("$volume", "sum"),
        early_bar_count=("$bar_count", "sum"),
    )
    grouped["early_ret"] = grouped["early_close"] / grouped["early_open"] - 1.0
    grouped["early_range"] = grouped["early_low"] / grouped["early_high"] - 1.0
    grouped["early_from_low"] = grouped["early_close"] / grouped["early_low"] - 1.0
    wide = grouped.reset_index().pivot(
        index="date",
        columns="instrument",
        values=[
            "early_ret",
            "early_range",
            "early_from_low",
            "early_volume",
            "early_bar_count",
        ],
    )
    wide.columns = [f"{field}_{str(instrument).lower()}" for field, instrument in wide.columns]
    return wide.sort_index()


def _advance_state(state: int, clock: int, drawdown: float, spec) -> tuple[int, int, float]:
    if state == 0 and drawdown >= spec.stop_drawdown:
        state, clock = 1, 0
    if state == 1:
        q = 0.0
        clock += 1
        if clock >= spec.wait_days:
            state, clock = 2, 0
    elif state == 2:
        q = min(1.0, (clock + 1) / spec.ramp_days)
        clock += 1
        if clock >= spec.ramp_days:
            state, clock = 3, 0
    elif state == 3:
        q = 1.0
        clock += 1
        if clock >= spec.hold_days:
            state, clock = 0, 0
    else:
        q = 1.0
    return state, clock, q


def _feature_row(
    row,
    *,
    drawdown: float,
    cap: float,
    raw_beta: float,
    state: int,
    history: list[float],
) -> dict[str, float]:
    return {
        "portfolio_drawdown": drawdown,
        "current_beta_cap": cap,
        "raw_effective_beta": raw_beta,
        "risk_state": float(state),
        "early_ret_soxl": row.early_ret_soxl,
        "early_ret_soxx": row.early_ret_soxx,
        "early_dislocation": row.early_ret_soxl - 3.0 * row.early_ret_soxx,
        "early_range_soxl": row.early_range_soxl,
        "early_range_soxx": row.early_range_soxx,
        "early_from_low_soxl": row.early_from_low_soxl,
        "early_from_low_soxx": row.early_from_low_soxx,
        "log_volume_soxl": np.log1p(row.early_volume_soxl),
        "log_volume_soxx": np.log1p(row.early_volume_soxx),
        "raw_soxl_weight": row.soxl_weight,
        "raw_soxx_weight": row.soxx_weight,
        "raw_cash_weight": row.cash_weight,
        "tail_probability": row.tail_probability,
        "alpha_probability": row.alpha_probability,
        "portfolio_ret_1d": history[-1] if history else 0.0,
        "portfolio_ret_3d": np.prod(1.0 + np.asarray(history[-3:])) - 1.0
        if history
        else 0.0,
        "portfolio_ret_5d": np.prod(1.0 + np.asarray(history[-5:])) - 1.0
        if history
        else 0.0,
    }


def _base_cap(
    row,
    *,
    q: float,
    drawdown: float,
    history: list[float],
    spec: IntradayRecoveryDirectAllocationSpec,
) -> tuple[float, float]:
    raw_beta = 3.0 * row.soxl_weight + row.soxx_weight
    cap = max(spec.defensive_beta, q * raw_beta)
    if len(history) >= spec.trailing_loss_days:
        trailing = np.prod(1.0 + np.asarray(history[-spec.trailing_loss_days :])) - 1.0
        if trailing <= -spec.trailing_loss_trigger:
            cap = min(cap, spec.trailing_loss_beta)
    if drawdown >= spec.emergency_drawdown:
        cap = min(cap, spec.emergency_beta)
    return cap, raw_beta


def _build_teacher_training_frame(
    frame: pd.DataFrame,
    *,
    spec: IntradayRecoveryDirectAllocationSpec,
    teacher_beta: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, segment in frame.groupby("segment", sort=False):
        equity = peak = 1.0
        previous = np.zeros(3)
        state = clock = 0
        history: list[float] = []
        for row in segment.sort_values("signal_date").itertuples():
            drawdown = max(0.0, 1.0 - equity / peak)
            state, clock, q = _advance_state(state, clock, drawdown, spec)
            cap, raw_beta = _base_cap(
                row, q=q, drawdown=drawdown, history=history, spec=spec
            )
            open_weights = project_legal_weights(
                row.soxl_weight,
                row.soxx_weight,
                cap,
                minimum_cash_weight=spec.minimum_cash_weight,
                maximum_soxl_weight=spec.maximum_soxl_weight,
            )
            asset_returns = np.array([row.soxl_return, row.soxx_return, 0.0])
            strategy_return = (
                open_weights @ asset_returns
                - np.abs(open_weights[:2] - previous[:2]).sum()
                * spec.transaction_cost_bps
                / 10_000.0
            )
            target_beta = min(raw_beta, teacher_beta)
            eligible = (
                drawdown >= spec.recovery_eligibility_drawdown
                and target_beta > cap + 1e-12
                and pd.notna(row.early_ret_soxx)
            )
            if eligible:
                teacher_weights = project_legal_weights(
                    row.soxl_weight,
                    row.soxx_weight,
                    target_beta,
                    minimum_cash_weight=spec.minimum_cash_weight,
                    maximum_soxl_weight=spec.maximum_soxl_weight,
                )
                early = np.array([row.early_ret_soxl, row.early_ret_soxx, 0.0])
                value_at_cut = 1.0 + open_weights @ early
                drifted = mark_to_market_weights(open_weights, early)
                post = np.array(
                    [
                        (1.0 + row.soxl_return) / (1.0 + row.early_ret_soxl) - 1.0,
                        (1.0 + row.soxx_return) / (1.0 + row.early_ret_soxx) - 1.0,
                        0.0,
                    ]
                )
                action_return = (
                    value_at_cut * (1.0 + teacher_weights @ post)
                    - 1.0
                    - np.abs(teacher_weights[:2] - drifted[:2]).sum()
                    * spec.transaction_cost_bps
                    / 10_000.0
                )
                hold_return = open_weights @ asset_returns
                features = _feature_row(
                    row,
                    drawdown=drawdown,
                    cap=cap,
                    raw_beta=raw_beta,
                    state=state,
                    history=history,
                )
                rows.append(
                    {
                        "signal_date": row.signal_date,
                        "trade_date": row.trade_date,
                        "segment": row.segment,
                        "year": row.signal_date.year,
                        **features,
                        "teacher_action_advantage": action_return - hold_return,
                        "teacher_action_label": float(action_return > hold_return),
                    }
                )
            history.append(strategy_return)
            equity *= 1.0 + strategy_return
            peak = max(peak, equity)
            previous = mark_to_market_weights(open_weights, asset_returns)
    return pd.DataFrame(rows)


def _fit_recovery_model(
    training: pd.DataFrame, *, year: int, c_value: float
):
    train = training[
        training["year"].lt(year) & training["year"].ge(2019)
    ].copy()
    if len(train) < 20 or train["teacher_action_label"].nunique() < 2:
        return None
    model = make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2_000,
            random_state=42,
        ),
    )
    model.fit(train.loc[:, FEATURE_COLUMNS], train["teacher_action_label"])
    return model


def _simulate_policy(
    frame: pd.DataFrame,
    *,
    spec: IntradayRecoveryDirectAllocationSpec,
    teacher_frame: pd.DataFrame,
    c_value: float,
    threshold: float,
    action_beta: float,
    first_action_year: int,
    model_cache: dict[tuple[int, float], Any],
    cost_bps: float,
    minimum_source_alpha_probability: float = 0.0,
) -> pd.DataFrame:
    if not 0.0 <= minimum_source_alpha_probability <= 1.0:
        raise ValueError("minimum_source_alpha_probability must be in [0, 1]")
    models: dict[int, Any] = {}
    for year in sorted(frame["signal_date"].dt.year.unique()):
        key = (int(year), float(c_value))
        if key not in model_cache:
            model_cache[key] = _fit_recovery_model(
                teacher_frame, year=int(year), c_value=c_value
            )
        models[int(year)] = model_cache[key]
    equity = peak = 1.0
    previous = np.zeros(3)
    state = clock = 0
    history: list[float] = []
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values("signal_date").itertuples():
        pretrade_weights = previous.copy()
        drawdown = max(0.0, 1.0 - equity / peak)
        state, clock, q = _advance_state(state, clock, drawdown, spec)
        cap, raw_beta = _base_cap(
            row, q=q, drawdown=drawdown, history=history, spec=spec
        )
        open_weights = project_legal_weights(
            row.soxl_weight,
            row.soxx_weight,
            cap,
            minimum_cash_weight=spec.minimum_cash_weight,
            maximum_soxl_weight=spec.maximum_soxl_weight,
        )
        target_beta = min(raw_beta, action_beta)
        model = models.get(row.signal_date.year)
        eligible = (
            row.signal_date.year >= first_action_year
            and model is not None
            and drawdown >= spec.recovery_eligibility_drawdown
            and target_beta > cap + 1e-12
            and pd.notna(row.early_ret_soxx)
            and pd.notna(row.alpha_probability)
            and row.alpha_probability >= minimum_source_alpha_probability
        )
        probability = np.nan
        recovery = False
        if eligible:
            feature = pd.DataFrame(
                [
                    _feature_row(
                        row,
                        drawdown=drawdown,
                        cap=cap,
                        raw_beta=raw_beta,
                        state=state,
                        history=history,
                    )
                ]
            )
            probability = float(model.predict_proba(feature.loc[:, FEATURE_COLUMNS])[0, 1])
            recovery = probability >= threshold
        if recovery:
            close_weights = project_legal_weights(
                row.soxl_weight,
                row.soxx_weight,
                target_beta,
                minimum_cash_weight=spec.minimum_cash_weight,
                maximum_soxl_weight=spec.maximum_soxl_weight,
            )
            early = np.array([row.early_ret_soxl, row.early_ret_soxx, 0.0])
            value_at_cut = 1.0 + open_weights @ early
            drifted = mark_to_market_weights(open_weights, early)
            post = np.array(
                [
                    (1.0 + row.soxl_return) / (1.0 + row.early_ret_soxl) - 1.0,
                    (1.0 + row.soxx_return) / (1.0 + row.early_ret_soxx) - 1.0,
                    0.0,
                ]
            )
            gross_return = value_at_cut * (1.0 + close_weights @ post) - 1.0
            turnover = np.abs(open_weights[:2] - previous[:2]).sum() + np.abs(
                close_weights[:2] - drifted[:2]
            ).sum()
            trade_delta = (open_weights - pretrade_weights) + (
                close_weights - drifted
            )
            ending_weights = mark_to_market_weights(close_weights, post)
        else:
            close_weights = open_weights
            asset_returns = np.array(
                [row.soxl_return, row.soxx_return, 0.0], dtype="float64"
            )
            gross_return = float(open_weights @ asset_returns)
            turnover = np.abs(open_weights[:2] - previous[:2]).sum()
            trade_delta = open_weights - pretrade_weights
            ending_weights = mark_to_market_weights(open_weights, asset_returns)
        strategy_return = gross_return - turnover * cost_bps / 10_000.0
        history.append(strategy_return)
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        previous = ending_weights
        rows.append(
            {
                "signal_date": row.signal_date,
                "trade_date": row.trade_date,
                "exit_date": row.exit_date,
                "segment": row.segment,
                "soxl_return": row.soxl_return,
                "soxx_return": row.soxx_return,
                "open_soxl_weight": open_weights[0],
                "open_soxx_weight": open_weights[1],
                "open_cash_weight": open_weights[2],
                "soxl_weight": close_weights[0],
                "soxx_weight": close_weights[1],
                "cash_weight": close_weights[2],
                "soxl_trade_delta": trade_delta[0],
                "soxx_trade_delta": trade_delta[1],
                "cash_trade_delta": trade_delta[2],
                "recovery_probability": probability,
                "recovery_action": recovery,
                "source_alpha_probability": row.alpha_probability,
                "source_alpha_veto": bool(
                    pd.notna(row.early_ret_soxx)
                    and pd.notna(row.alpha_probability)
                    and row.alpha_probability < minimum_source_alpha_probability
                ),
                "current_beta_cap": cap,
                "recovery_beta_target": target_beta if recovery else cap,
                "gross_return": gross_return,
                "turnover": turnover,
                "strategy_return": strategy_return,
                "cost_bps": cost_bps,
            }
        )
    return pd.DataFrame(rows)


def _risk_metrics(returns: np.ndarray) -> dict[str, float]:
    values = np.asarray(returns, dtype="float64")
    wealth = np.cumprod(1.0 + values)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    q05 = np.quantile(values, 0.05)
    rolling_5d = (
        pd.Series(1.0 + values).rolling(5).apply(np.prod, raw=True).to_numpy() - 1.0
    )
    return {
        "terminal_wealth": float(wealth[-1]),
        "return_pct": float(100.0 * (wealth[-1] - 1.0)),
        "max_drawdown_pct": float(100.0 * drawdown.min()),
        "worst_day_pct": float(100.0 * values.min()),
        "cvar5_pct": float(100.0 * values[values <= q05].mean()),
        "worst_rolling_5d_pct": float(100.0 * np.nanmin(rolling_5d)),
    }


def _passes_risk(metrics: dict[str, float]) -> bool:
    return (
        metrics["max_drawdown_pct"] >= -30.0
        and metrics["worst_day_pct"] >= -12.0
        and metrics["cvar5_pct"] >= -8.0
        and metrics["worst_rolling_5d_pct"] >= -12.0
    )


def _development_stability(replay: pd.DataFrame) -> dict[str, Any]:
    years = replay["signal_date"].dt.year.to_numpy()
    windows = [
        (2019, 2019),
        (2020, 2020),
        (2021, 2021),
        (2022, 2022),
        (2019, 2020),
        (2020, 2021),
        (2021, 2022),
    ]
    ratios: list[float] = []
    risk_passes = 0
    for start, end in windows:
        mask = (years >= start) & (years <= end)
        section = replay.loc[mask]
        metrics = _risk_metrics(section["strategy_return"].to_numpy())
        benchmark_wealth = float(np.prod(1.0 + section["soxx_return"].to_numpy()))
        ratio = metrics["terminal_wealth"] / benchmark_wealth
        ratios.append(ratio)
        risk_passes += int(_passes_risk(metrics))
    return {
        "risk_window_pass_count": risk_passes,
        "positive_relative_window_count": int(np.sum(np.asarray(ratios) > 1.0)),
        "relative_ratio_min": float(np.min(ratios)),
        "relative_ratio_p25": float(np.quantile(ratios, 0.25)),
        "relative_ratio_median": float(np.median(ratios)),
    }


def _annual_metrics(replay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, section in replay.groupby(replay["signal_date"].dt.year):
        metrics = _risk_metrics(section["strategy_return"].to_numpy())
        benchmark_wealth = float(np.prod(1.0 + section["soxx_return"].to_numpy()))
        rows.append(
            {
                "year": int(year),
                **metrics,
                "terminal_wealth_ratio_vs_soxx": metrics["terminal_wealth"]
                / benchmark_wealth,
                "avg_cash_weight_pct": 100.0 * section["cash_weight"].mean(),
                "recovery_action_count": int(section["recovery_action"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _record_qlib_artifacts(
    spec,
    *,
    selected_params,
    dev_metrics,
    outer_metrics,
    baseline_metrics,
    stress_metrics,
    model_selection,
    outer_replay,
    teacher_frame,
    source_audit,
) -> dict[str, Any]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        positions = outer_replay[
            [
                "signal_date",
                "trade_date",
                "open_soxl_weight",
                "open_soxx_weight",
                "open_cash_weight",
                "soxl_weight",
                "soxx_weight",
                "cash_weight",
                "recovery_probability",
                "recovery_action",
            ]
        ]
        files = {
            "positions.csv": positions,
            "strategy_returns.csv": outer_replay,
            "recovery_training_frame.csv": teacher_frame,
            "model_selection_full.csv": model_selection,
        }
        for filename, data in files.items():
            data.to_csv(tmp_dir / filename, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_experiment=spec.source_experiment,
                source_recorder_id=source_audit["source_recorder_id"],
                intraday_provider_uri=source_audit["intraday_provider_uri"],
                allocation_assets="SOXL,SOXX,cash",
                allocation_output="direct_legal_three_asset_weights",
                selection_uses_outer_test="false",
                teacher_recovery_beta=str(spec.teacher_recovery_beta),
                action_beta=str(selected_params["action_beta"]),
                probability_threshold=str(selected_params["probability_threshold"]),
                logistic_c=str(selected_params["logistic_c"]),
            )
            R.log_metrics(
                development_return_pct=dev_metrics["return_pct"],
                development_max_drawdown_pct=dev_metrics["max_drawdown_pct"],
                outer_return_pct=outer_metrics["return_pct"],
                outer_max_drawdown_pct=outer_metrics["max_drawdown_pct"],
                baseline_outer_return_pct=baseline_metrics["return_pct"],
                stress_25bps_return_pct=stress_metrics["return_pct"],
                stress_25bps_max_drawdown_pct=stress_metrics["max_drawdown_pct"],
            )
            for filename in files:
                recorder.log_artifact(
                    (tmp_dir / filename).as_posix(), artifact_path=spec.experiment
                )
            recorder_info = {
                "recorder_id": recorder.id,
                "recorder_uri": str(recorder_uri.resolve()),
                "recorder_local_dir": str(Path(recorder.get_local_dir()).resolve()),
            }
    recorder_artifacts = Path(recorder_info["recorder_local_dir"]) / "artifacts"
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        (recorder_artifacts / filename).unlink(missing_ok=True)
    return recorder_info


def _write_card_sidecars(
    spec,
    *,
    summary,
    selected,
    model_selection,
    risk_analysis,
    annual_metrics,
    cost_sensitivity,
) -> None:
    card = spec.card_dir
    selected_payload = {
        "selection_uses_outer_test": False,
        "selected_from_segment": "development_walk_forward",
        "teacher_recovery_beta": spec.teacher_recovery_beta,
        "action_beta": float(selected["action_beta"]),
        "logistic_c": float(selected["logistic_c"]),
        "probability_threshold": float(selected["probability_threshold"]),
        "development_metrics": {
            key: float(selected[key])
            for key in (
                "return_pct",
                "max_drawdown_pct",
                "worst_day_pct",
                "cvar5_pct",
                "worst_rolling_5d_pct",
                "avg_cash_weight_pct",
            )
        },
    }
    (card / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (card / "selected_model.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    model_selection.sort_values("terminal_wealth", ascending=False).head(40).to_csv(
        card / "model_selection.csv", index=False
    )
    risk_analysis.to_csv(card / "risk_analysis.csv", index=False)
    annual_metrics.to_csv(card / "annual_metrics.csv", index=False)
    cost_sensitivity.to_csv(card / "cost_sensitivity.csv", index=False)
    workflow = f"""experiment: {spec.experiment}
source_experiment: {spec.source_experiment}
source_recorder_id: {summary['source_audit']['source_recorder_id']}
source_refresh_mode: {summary['source_audit']['source_refresh_mode']}
intraday_provider_uri: {spec.intraday_provider_uri.resolve()}
assets: [SOXL, SOXX, cash]
allocation_output: direct_legal_three_asset_weights
selection_uses_outer_test: false
teacher_recovery_beta: {spec.teacher_recovery_beta}
action_beta: {summary['selected_model']['action_beta']}
transaction_cost_bps: {spec.transaction_cost_bps}
stress_cost_bps: {spec.stress_cost_bps}
stress_method: fixed_10bp_decisions_repriced_at_25bp
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  recorder_only_artifacts:
    - artifacts/{spec.experiment}/positions.csv
    - artifacts/{spec.experiment}/strategy_returns.csv
    - artifacts/{spec.experiment}/recovery_training_frame.csv
    - artifacts/{spec.experiment}/model_selection_full.csv
"""
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
    conclusion = f"""# 中文结论

- 做了什么：把原模型的 SOXL/SOXX/cash 合法仓位、alpha/tail 概率、组合自身回撤状态与开盘后前四根 15 分钟路径合并；模型在 10:30 直接重新输出三资产仓位，不再使用底仓加 overlay。
- 为什么之前现金过多：日频回撤状态机只能按整天降风险，错过反弹后又重复进入防守；收益损失来自现金时点，而不只是平均现金比例。
- 训练结构：教师标签判断“恢复到 {spec.teacher_recovery_beta:.1f} beta 是否有正增量”，实际执行只恢复到 {summary['selected_model']['action_beta']:.1f} beta，用不对称 teacher-student 校准限制误判杠杆。
- 选择规则：只使用 2019-2022 开发期年度/两年窗口；2023-2026 outer 完全不参与模型、阈值或动作 beta 选择。
- 外层 10bp：return={summary['outer_diagnostic_metrics_10bps']['return_pct']:.4f}%，MaxDD={summary['outer_diagnostic_metrics_10bps']['max_drawdown_pct']:.4f}%，worst day={summary['outer_diagnostic_metrics_10bps']['worst_day_pct']:.4f}%，CVaR5={summary['outer_diagnostic_metrics_10bps']['cvar5_pct']:.4f}%，worst 5d={summary['outer_diagnostic_metrics_10bps']['worst_rolling_5d_pct']:.4f}%。
- 现金与收益：日频控制器 outer return={summary['daily_controller_outer_metrics_10bps']['return_pct']:.4f}%，新模型提高 {summary['cash_timing_effect']['outer_return_improvement_pct_points']:.4f} 个百分点；平均现金减少 {summary['cash_timing_effect']['outer_cash_reduction_pct_points']:.4f} 个百分点。
- 25bp 冻结决策压力：return={summary['fixed_decision_stress_metrics_25bps']['return_pct']:.4f}%，MaxDD={summary['fixed_decision_stress_metrics_25bps']['max_drawdown_pct']:.4f}%；只重算成本，不允许成本改变仓位路径。
- 效率门：return/MaxDD={summary['benchmark_comparison_10bps']['return_maxdd']:.4f}，同平均风险 SOXL={summary['benchmark_comparison_10bps']['same_avg_risk_soxl_return_maxdd']:.4f}，matched-beta={summary['benchmark_comparison_10bps']['matched_beta_return_maxdd']:.4f}，均通过。旧版 SOXL upside capture={summary['benchmark_comparison_10bps']['upside_capture_pct']:.4f}% 仍未过 60% 门；该旧门不属于当前“超过 SOXX 且保留现金”目标。
- 原模型差距：原模型 outer return={summary['raw_model_outer_metrics_10bps']['return_pct']:.4f}%，新模型仍少 {summary['raw_model_outer_metrics_10bps']['return_pct'] - summary['outer_diagnostic_metrics_10bps']['return_pct']:.4f} 个百分点；尚未做到完全保持原收益。
- 是否晋升：{summary['decision']}。这是研究候选晋级，不是生产或实盘成交证明；仍缺 bid/ask、排队、延迟和 10:30 真实成交 replay。
- qlib recorder：{summary['recorder_id']}
- 不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
