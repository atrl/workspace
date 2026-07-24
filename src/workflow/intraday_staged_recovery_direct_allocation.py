from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.workflow.intraday_recovery_direct_allocation import (
    FEATURE_COLUMNS,
    IntradayRecoveryDirectAllocationSpec,
    _advance_state,
    _annual_metrics,
    _base_cap,
    _build_teacher_training_frame,
    _development_stability,
    _feature_row,
    _fit_recovery_model,
    _load_source_frame,
    _passes_risk,
    _risk_metrics,
    _simulate_policy,
    apply_fixed_decision_cost_stress,
    project_legal_weights,
    select_development_candidate,
)


EXPERIMENT = "intraday_staged_recovery_direct_allocation_v1"
BASE_EXPERIMENT = "intraday_recovery_direct_allocation_v1"
CARD_TYPE = "intraday_staged_recovery_direct_three_asset_allocation_card"

STAGE2_FEATURE_COLUMNS = (
    "portfolio_drawdown",
    "current_beta_cap",
    "raw_effective_beta",
    "risk_state",
    "cut_ret_soxl",
    "cut_ret_soxx",
    "cut_dislocation",
    "cut_range_soxl",
    "cut_range_soxx",
    "cut_from_low_soxl",
    "cut_from_low_soxx",
    "log_cut_volume_soxl",
    "log_cut_volume_soxx",
    "second_leg_ret_soxl",
    "second_leg_ret_soxx",
    "second_leg_dislocation",
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
class IntradayStagedRecoveryDirectAllocationSpec(
    IntradayRecoveryDirectAllocationSpec
):
    experiment: str = EXPERIMENT
    base_experiment: str = BASE_EXPERIMENT
    second_cut_bar_candidates: tuple[int, ...] = (6, 8)
    stage2_teacher_beta: float = 1.2

    @property
    def base_card_dir(self) -> Path:
        return self.root / "research" / self.base_experiment


def run_intraday_staged_recovery_direct_allocation_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    if root is None:
        current = Path.cwd().resolve()
        while current != current.parent:
            if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
                root = current
                break
            current = current.parent
    if root is None:
        raise ValueError("could not resolve repository root")
    return _run_card(
        IntradayStagedRecoveryDirectAllocationSpec(root=Path(root).resolve())
    )


def _load_cut_features(
    spec: IntradayStagedRecoveryDirectAllocationSpec,
    *,
    bars: int,
    end: pd.Timestamp,
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
    cut = qframe.groupby(["instrument", "date"], group_keys=False).head(bars)
    grouped = cut.groupby(["date", "instrument"]).agg(
        cut_open=("$open", "first"),
        cut_close=("$close", "last"),
        cut_high=("$high", "max"),
        cut_low=("$low", "min"),
        cut_volume=("$volume", "sum"),
        cut_bar_count=("$bar_count", "sum"),
    )
    grouped["cut_ret"] = grouped["cut_close"] / grouped["cut_open"] - 1.0
    grouped["cut_range"] = grouped["cut_low"] / grouped["cut_high"] - 1.0
    grouped["cut_from_low"] = grouped["cut_close"] / grouped["cut_low"] - 1.0
    wide = grouped.reset_index().pivot(
        index="date",
        columns="instrument",
        values=[
            "cut_ret",
            "cut_range",
            "cut_from_low",
            "cut_volume",
            "cut_bar_count",
        ],
    )
    wide.columns = [
        f"{field}_{str(instrument).lower()}" for field, instrument in wide.columns
    ]
    full = qframe.groupby(["date", "instrument"]).agg(
        day_open=("$open", "first"),
        day_close=("$close", "last"),
    )
    full["day_ret"] = full["day_close"] / full["day_open"] - 1.0
    full_wide = full.reset_index().pivot(
        index="date",
        columns="instrument",
        values=["day_ret"],
    )
    full_wide.columns = [
        f"{field}_{str(instrument).lower()}"
        for field, instrument in full_wide.columns
    ]
    return wide.join(full_wide).sort_index()


def _add_cut_features(frame: pd.DataFrame, cut: pd.DataFrame) -> pd.DataFrame:
    expanded = frame.merge(
        cut,
        left_on="trade_date",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    for asset in ("soxl", "soxx"):
        expanded[f"second_leg_ret_{asset}"] = (
            (1.0 + expanded[f"cut_ret_{asset}"])
            / (1.0 + expanded[f"early_ret_{asset}"])
            - 1.0
        )
    return expanded


def _stage2_feature_row(
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
        "cut_ret_soxl": row.cut_ret_soxl,
        "cut_ret_soxx": row.cut_ret_soxx,
        "cut_dislocation": row.cut_ret_soxl - 3.0 * row.cut_ret_soxx,
        "cut_range_soxl": row.cut_range_soxl,
        "cut_range_soxx": row.cut_range_soxx,
        "cut_from_low_soxl": row.cut_from_low_soxl,
        "cut_from_low_soxx": row.cut_from_low_soxx,
        "log_cut_volume_soxl": np.log1p(row.cut_volume_soxl),
        "log_cut_volume_soxx": np.log1p(row.cut_volume_soxx),
        "second_leg_ret_soxl": row.second_leg_ret_soxl,
        "second_leg_ret_soxx": row.second_leg_ret_soxx,
        "second_leg_dislocation": row.second_leg_ret_soxl
        - 3.0 * row.second_leg_ret_soxx,
        "raw_soxl_weight": row.soxl_weight,
        "raw_soxx_weight": row.soxx_weight,
        "raw_cash_weight": row.cash_weight,
        "tail_probability": row.tail_probability,
        "alpha_probability": row.alpha_probability,
        "portfolio_ret_1d": history[-1] if history else 0.0,
        "portfolio_ret_3d": (
            np.prod(1.0 + np.asarray(history[-3:])) - 1.0 if history else 0.0
        ),
        "portfolio_ret_5d": (
            np.prod(1.0 + np.asarray(history[-5:])) - 1.0 if history else 0.0
        ),
    }


def _build_stage2_teacher_training_frame(
    frame: pd.DataFrame,
    *,
    spec: IntradayStagedRecoveryDirectAllocationSpec,
    current_beta: float,
    teacher_beta: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
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
            daily_return = (
                open_weights @ asset_returns
                - np.abs(open_weights[:2] - previous[:2]).sum()
                * spec.transaction_cost_bps
                / 10_000.0
            )
            current_target = min(raw_beta, current_beta)
            teacher_target = min(raw_beta, teacher_beta)
            eligible = (
                drawdown >= spec.recovery_eligibility_drawdown
                and current_target > cap + 1e-12
                and teacher_target > current_target + 1e-12
                and pd.notna(row.cut_ret_soxx)
                and pd.notna(row.day_ret_soxx)
            )
            if eligible:
                current_weights = project_legal_weights(
                    row.soxl_weight,
                    row.soxx_weight,
                    current_target,
                    minimum_cash_weight=spec.minimum_cash_weight,
                    maximum_soxl_weight=spec.maximum_soxl_weight,
                )
                teacher_weights = project_legal_weights(
                    row.soxl_weight,
                    row.soxx_weight,
                    teacher_target,
                    minimum_cash_weight=spec.minimum_cash_weight,
                    maximum_soxl_weight=spec.maximum_soxl_weight,
                )
                first_leg = np.array(
                    [row.early_ret_soxl, row.early_ret_soxx, 0.0]
                )
                value_at_stage1 = 1.0 + open_weights @ first_leg
                second_leg = np.array(
                    [
                        row.second_leg_ret_soxl,
                        row.second_leg_ret_soxx,
                        0.0,
                    ]
                )
                second_leg_growth = 1.0 + current_weights @ second_leg
                value_at_cut = value_at_stage1 * second_leg_growth
                drifted_at_cut = (
                    current_weights
                    * (1.0 + second_leg)
                    / second_leg_growth
                )
                cut_to_close = np.array(
                    [
                        (1.0 + row.day_ret_soxl)
                        / (1.0 + row.cut_ret_soxl)
                        - 1.0,
                        (1.0 + row.day_ret_soxx)
                        / (1.0 + row.cut_ret_soxx)
                        - 1.0,
                        0.0,
                    ]
                )
                overnight = np.array(
                    [
                        (1.0 + row.soxl_return)
                        / (1.0 + row.day_ret_soxl)
                        - 1.0,
                        (1.0 + row.soxx_return)
                        / (1.0 + row.day_ret_soxx)
                        - 1.0,
                        0.0,
                    ]
                )
                action_gross_return, action_turnover = (
                    time_limited_stage2_path(
                        value_at_cut=value_at_cut,
                        drifted_at_cut=drifted_at_cut,
                        action_weights=teacher_weights,
                        restore_weights=current_weights,
                        cut_to_close_returns=cut_to_close,
                        overnight_returns=overnight,
                    )
                )
                action_return = (
                    action_gross_return
                    - action_turnover
                    * spec.transaction_cost_bps
                    / 10_000.0
                )
                post_cut = np.array(
                    [
                        (1.0 + row.soxl_return)
                        / (1.0 + row.cut_ret_soxl)
                        - 1.0,
                        (1.0 + row.soxx_return)
                        / (1.0 + row.cut_ret_soxx)
                        - 1.0,
                        0.0,
                    ]
                )
                hold_return = (
                    value_at_cut * (1.0 + current_weights @ post_cut) - 1.0
                )
                records.append(
                    {
                        "signal_date": row.signal_date,
                        "trade_date": row.trade_date,
                        "segment": row.segment,
                        "year": row.signal_date.year,
                        **_stage2_feature_row(
                            row,
                            drawdown=drawdown,
                            cap=cap,
                            raw_beta=raw_beta,
                            state=state,
                            history=history,
                        ),
                        "teacher_action_advantage": action_return
                        - hold_return,
                        "teacher_action_label": float(
                            action_return > hold_return
                        ),
                    }
                )
            history.append(daily_return)
            equity *= 1.0 + daily_return
            peak = max(peak, equity)
            previous = open_weights
    return pd.DataFrame(records)


def _fit_stage2_model(
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
    model.fit(
        train.loc[:, STAGE2_FEATURE_COLUMNS],
        train["teacher_action_label"],
    )
    return model


def time_limited_stage2_path(
    *,
    value_at_cut: float,
    drifted_at_cut: np.ndarray,
    action_weights: np.ndarray,
    restore_weights: np.ndarray,
    cut_to_close_returns: np.ndarray,
    overnight_returns: np.ndarray,
) -> tuple[float, float]:
    close_growth = 1.0 + action_weights @ cut_to_close_returns
    value_at_close = value_at_cut * close_growth
    drifted_at_close = (
        action_weights * (1.0 + cut_to_close_returns) / close_growth
    )
    gross_return = (
        value_at_close * (1.0 + restore_weights @ overnight_returns) - 1.0
    )
    additional_turnover = (
        np.abs(action_weights[:2] - drifted_at_cut[:2]).sum()
        + np.abs(restore_weights[:2] - drifted_at_close[:2]).sum()
    )
    return float(gross_return), float(additional_turnover)


def simulate_staged_policy(
    frame: pd.DataFrame,
    *,
    spec: IntradayStagedRecoveryDirectAllocationSpec,
    stage1_teacher_frame: pd.DataFrame,
    stage1_c_value: float,
    stage1_threshold: float,
    stage1_action_beta: float,
    stage2_teacher_frame: pd.DataFrame,
    stage2_c_value: float,
    stage2_threshold: float,
    stage2_action_beta: float,
    first_action_year: int,
    stage1_model_cache: dict[tuple[int, float], Any],
    stage2_model_cache: dict[tuple[int, float], Any],
    cost_bps: float,
) -> pd.DataFrame:
    stage1_models: dict[int, Any] = {}
    stage2_models: dict[int, Any] = {}
    for year in sorted(frame["signal_date"].dt.year.unique()):
        year = int(year)
        key1 = (year, float(stage1_c_value))
        if key1 not in stage1_model_cache:
            stage1_model_cache[key1] = _fit_recovery_model(
                stage1_teacher_frame,
                year=year,
                c_value=stage1_c_value,
            )
        stage1_models[year] = stage1_model_cache[key1]
        key2 = (year, float(stage2_c_value))
        if key2 not in stage2_model_cache:
            stage2_model_cache[key2] = _fit_stage2_model(
                stage2_teacher_frame,
                year=year,
                c_value=stage2_c_value,
            )
        stage2_models[year] = stage2_model_cache[key2]

    equity = peak = 1.0
    previous = np.zeros(3)
    state = clock = 0
    history: list[float] = []
    records: list[dict[str, Any]] = []
    for row in frame.sort_values("signal_date").itertuples():
        drawdown = max(0.0, 1.0 - equity / peak)
        state, clock, q = _advance_state(state, clock, drawdown, spec)
        cap, raw_beta = _base_cap(
            row,
            q=q,
            drawdown=drawdown,
            history=history,
            spec=spec,
        )
        open_weights = project_legal_weights(
            row.soxl_weight,
            row.soxx_weight,
            cap,
            minimum_cash_weight=spec.minimum_cash_weight,
            maximum_soxl_weight=spec.maximum_soxl_weight,
        )
        stage1_target = min(raw_beta, stage1_action_beta)
        stage1_model = stage1_models.get(row.signal_date.year)
        stage1_eligible = (
            row.signal_date.year >= first_action_year
            and stage1_model is not None
            and drawdown >= spec.recovery_eligibility_drawdown
            and stage1_target > cap + 1e-12
            and pd.notna(row.early_ret_soxx)
        )
        stage1_probability = np.nan
        stage1_action = False
        if stage1_eligible:
            stage1_features = pd.DataFrame(
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
            stage1_probability = float(
                stage1_model.predict_proba(
                    stage1_features.loc[:, FEATURE_COLUMNS]
                )[0, 1]
            )
            stage1_action = stage1_probability >= stage1_threshold

        open_turnover = np.abs(open_weights[:2] - previous[:2]).sum()
        stage2_probability = np.nan
        stage2_action = False
        stage2_target = stage1_target
        if not stage1_action:
            final_weights = open_weights
            stage2_weights = open_weights
            gross_return = (
                open_weights[0] * row.soxl_return
                + open_weights[1] * row.soxx_return
            )
            turnover = open_turnover
        else:
            stage1_weights = project_legal_weights(
                row.soxl_weight,
                row.soxx_weight,
                stage1_target,
                minimum_cash_weight=spec.minimum_cash_weight,
                maximum_soxl_weight=spec.maximum_soxl_weight,
            )
            early = np.array(
                [row.early_ret_soxl, row.early_ret_soxx, 0.0]
            )
            value_at_stage1 = 1.0 + open_weights @ early
            drifted_at_stage1 = (
                open_weights * (1.0 + early) / value_at_stage1
            )
            second_leg = np.array(
                [row.second_leg_ret_soxl, row.second_leg_ret_soxx, 0.0]
            )
            second_leg_growth = 1.0 + stage1_weights @ second_leg
            value_at_stage2 = value_at_stage1 * second_leg_growth
            drifted_at_stage2 = (
                stage1_weights * (1.0 + second_leg) / second_leg_growth
            )
            stage2_target = min(raw_beta, stage2_action_beta)
            stage2_model = stage2_models.get(row.signal_date.year)
            stage2_eligible = (
                stage2_model is not None
                and stage2_target > stage1_target + 1e-12
                and pd.notna(row.cut_ret_soxx)
                and pd.notna(row.day_ret_soxx)
            )
            if stage2_eligible:
                stage2_features = pd.DataFrame(
                    [
                        _stage2_feature_row(
                            row,
                            drawdown=drawdown,
                            cap=cap,
                            raw_beta=raw_beta,
                            state=state,
                            history=history,
                        )
                    ]
                )
                stage2_probability = float(
                    stage2_model.predict_proba(
                        stage2_features.loc[:, STAGE2_FEATURE_COLUMNS]
                    )[0, 1]
                )
                stage2_action = stage2_probability >= stage2_threshold
            if stage2_action:
                stage2_weights = project_legal_weights(
                    row.soxl_weight,
                    row.soxx_weight,
                    stage2_target,
                    minimum_cash_weight=spec.minimum_cash_weight,
                    maximum_soxl_weight=spec.maximum_soxl_weight,
                )
                cut_to_close = np.array(
                    [
                        (1.0 + row.day_ret_soxl)
                        / (1.0 + row.cut_ret_soxl)
                        - 1.0,
                        (1.0 + row.day_ret_soxx)
                        / (1.0 + row.cut_ret_soxx)
                        - 1.0,
                        0.0,
                    ]
                )
                overnight = np.array(
                    [
                        (1.0 + row.soxl_return)
                        / (1.0 + row.day_ret_soxl)
                        - 1.0,
                        (1.0 + row.soxx_return)
                        / (1.0 + row.day_ret_soxx)
                        - 1.0,
                        0.0,
                    ]
                )
                final_weights = stage1_weights
                gross_return, stage2_turnover = time_limited_stage2_path(
                    value_at_cut=value_at_stage2,
                    drifted_at_cut=drifted_at_stage2,
                    action_weights=stage2_weights,
                    restore_weights=final_weights,
                    cut_to_close_returns=cut_to_close,
                    overnight_returns=overnight,
                )
                turnover = (
                    open_turnover
                    + np.abs(
                        stage1_weights[:2] - drifted_at_stage1[:2]
                    ).sum()
                    + stage2_turnover
                )
            else:
                stage2_weights = stage1_weights
                final_weights = stage1_weights
                post_stage1 = np.array(
                    [
                        (1.0 + row.soxl_return)
                        / (1.0 + row.early_ret_soxl)
                        - 1.0,
                        (1.0 + row.soxx_return)
                        / (1.0 + row.early_ret_soxx)
                        - 1.0,
                        0.0,
                    ]
                )
                gross_return = (
                    value_at_stage1
                    * (1.0 + final_weights @ post_stage1)
                    - 1.0
                )
                turnover = open_turnover + np.abs(
                    stage1_weights[:2] - drifted_at_stage1[:2]
                ).sum()
        strategy_return = gross_return - turnover * cost_bps / 10_000.0
        history.append(strategy_return)
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        previous = final_weights
        records.append(
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
                "stage2_soxl_weight": stage2_weights[0],
                "stage2_soxx_weight": stage2_weights[1],
                "stage2_cash_weight": stage2_weights[2],
                "soxl_weight": final_weights[0],
                "soxx_weight": final_weights[1],
                "cash_weight": final_weights[2],
                "recovery_probability": stage1_probability,
                "recovery_action": stage1_action,
                "stage2_probability": stage2_probability,
                "stage2_action": stage2_action,
                "current_beta_cap": cap,
                "stage1_beta_target": stage1_target
                if stage1_action
                else cap,
                "stage2_beta_target": stage2_target
                if stage2_action
                else stage1_target,
                "gross_return": gross_return,
                "turnover": turnover,
                "strategy_return": strategy_return,
                "cost_bps": cost_bps,
            }
        )
    return pd.DataFrame(records)


def _run_card(
    spec: IntradayStagedRecoveryDirectAllocationSpec,
) -> dict[str, Any]:
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    frame, source_audit = _load_source_frame(spec)
    base_summary = json.loads(
        (spec.base_card_dir / "summary.json").read_text(encoding="utf-8")
    )
    if base_summary.get("selection_uses_outer_test") is not False:
        raise ValueError("base intraday model must be selected without outer test")
    stage1_params = base_summary["selected_model"]
    stage1_teacher = _build_teacher_training_frame(
        frame,
        spec=spec,
        teacher_beta=float(stage1_params["teacher_recovery_beta"]),
    )
    development_frames: dict[int, pd.DataFrame] = {}
    stage2_teachers: dict[int, pd.DataFrame] = {}
    candidates: list[dict[str, Any]] = []
    stage1_model_cache: dict[tuple[int, float], Any] = {}
    stage2_model_caches: dict[int, dict[tuple[int, float], Any]] = {}
    for bars in spec.second_cut_bar_candidates:
        cut = _load_cut_features(
            spec,
            bars=bars,
            end=frame["trade_date"].max(),
        )
        expanded = _add_cut_features(frame, cut)
        development_frames[bars] = expanded
        stage2_teacher = _build_stage2_teacher_training_frame(
            expanded,
            spec=spec,
            current_beta=float(stage1_params["action_beta"]),
            teacher_beta=spec.stage2_teacher_beta,
        )
        stage2_teachers[bars] = stage2_teacher
        stage2_model_cache = {}
        stage2_model_caches[bars] = stage2_model_cache
        development = expanded[
            expanded["segment"].eq("development_walk_forward")
        ].copy()
        for c_value in (0.01, 0.03, 0.10, 0.30):
            for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
                for action_beta in (0.80, 1.00, 1.20):
                    replay = simulate_staged_policy(
                        development,
                        spec=spec,
                        stage1_teacher_frame=stage1_teacher,
                        stage1_c_value=float(stage1_params["logistic_c"]),
                        stage1_threshold=float(
                            stage1_params["probability_threshold"]
                        ),
                        stage1_action_beta=float(stage1_params["action_beta"]),
                        stage2_teacher_frame=stage2_teacher,
                        stage2_c_value=c_value,
                        stage2_threshold=threshold,
                        stage2_action_beta=action_beta,
                        first_action_year=spec.first_intraday_action_year,
                        stage1_model_cache=stage1_model_cache,
                        stage2_model_cache=stage2_model_cache,
                        cost_bps=spec.transaction_cost_bps,
                    )
                    metrics = _risk_metrics(
                        replay["strategy_return"].to_numpy()
                    )
                    if not _passes_risk(metrics):
                        continue
                    stability = _development_stability(replay)
                    candidates.append(
                        {
                            "segment": "development_walk_forward",
                            "selection_uses_outer_test": False,
                            "second_cut_bars": bars,
                            "second_cut_time": "11:00"
                            if bars == 6
                            else "11:30",
                            "stage2_teacher_beta": spec.stage2_teacher_beta,
                            "stage2_action_beta": action_beta,
                            "stage2_logistic_c": c_value,
                            "stage2_probability_threshold": threshold,
                            **metrics,
                            **stability,
                            "avg_cash_weight_pct": 100.0
                            * replay["cash_weight"].mean(),
                            "stage1_action_count": int(
                                replay["recovery_action"].sum()
                            ),
                            "stage2_action_count": int(
                                replay["stage2_action"].sum()
                            ),
                        }
                    )
    model_selection = pd.DataFrame(candidates)
    selected = select_development_candidate(model_selection)
    selected_params = {
        "second_cut_bars": int(selected["second_cut_bars"]),
        "second_cut_time": str(selected["second_cut_time"]),
        "stage2_teacher_beta": spec.stage2_teacher_beta,
        "stage2_action_beta": float(selected["stage2_action_beta"]),
        "stage2_logistic_c": float(selected["stage2_logistic_c"]),
        "stage2_probability_threshold": float(
            selected["stage2_probability_threshold"]
        ),
    }
    selected_bars = selected_params["second_cut_bars"]
    selected_frame = development_frames[selected_bars]
    selected_teacher = stage2_teachers[selected_bars]
    selected_stage2_cache = stage2_model_caches[selected_bars]
    dev_replay = simulate_staged_policy(
        selected_frame[
            selected_frame["segment"].eq("development_walk_forward")
        ].copy(),
        spec=spec,
        stage1_teacher_frame=stage1_teacher,
        stage1_c_value=float(stage1_params["logistic_c"]),
        stage1_threshold=float(stage1_params["probability_threshold"]),
        stage1_action_beta=float(stage1_params["action_beta"]),
        stage2_teacher_frame=selected_teacher,
        stage2_c_value=selected_params["stage2_logistic_c"],
        stage2_threshold=selected_params["stage2_probability_threshold"],
        stage2_action_beta=selected_params["stage2_action_beta"],
        first_action_year=spec.first_intraday_action_year,
        stage1_model_cache=stage1_model_cache,
        stage2_model_cache=selected_stage2_cache,
        cost_bps=spec.transaction_cost_bps,
    )
    outer = selected_frame[
        selected_frame["segment"].eq("outer_diagnostic")
    ].copy()
    outer_replay = simulate_staged_policy(
        outer,
        spec=spec,
        stage1_teacher_frame=stage1_teacher,
        stage1_c_value=float(stage1_params["logistic_c"]),
        stage1_threshold=float(stage1_params["probability_threshold"]),
        stage1_action_beta=float(stage1_params["action_beta"]),
        stage2_teacher_frame=selected_teacher,
        stage2_c_value=selected_params["stage2_logistic_c"],
        stage2_threshold=selected_params["stage2_probability_threshold"],
        stage2_action_beta=selected_params["stage2_action_beta"],
        first_action_year=min(spec.outer_years),
        stage1_model_cache=stage1_model_cache,
        stage2_model_cache=selected_stage2_cache,
        cost_bps=spec.transaction_cost_bps,
    )
    stage1_outer = _simulate_policy(
        outer,
        spec=spec,
        teacher_frame=stage1_teacher,
        c_value=float(stage1_params["logistic_c"]),
        threshold=float(stage1_params["probability_threshold"]),
        action_beta=float(stage1_params["action_beta"]),
        first_action_year=min(spec.outer_years),
        model_cache=stage1_model_cache,
        cost_bps=spec.transaction_cost_bps,
    )
    daily_outer = _simulate_policy(
        outer,
        spec=spec,
        teacher_frame=stage1_teacher,
        c_value=float(stage1_params["logistic_c"]),
        threshold=2.0,
        action_beta=float(stage1_params["action_beta"]),
        first_action_year=min(spec.outer_years),
        model_cache=stage1_model_cache,
        cost_bps=spec.transaction_cost_bps,
    )
    fixed_25 = outer_replay.copy()
    fixed_25["strategy_return"] = apply_fixed_decision_cost_stress(
        fixed_25["gross_return"],
        fixed_25["turnover"],
        cost_bps=spec.stress_cost_bps,
    )
    dev_metrics = _risk_metrics(dev_replay["strategy_return"].to_numpy())
    outer_metrics = _risk_metrics(
        outer_replay["strategy_return"].to_numpy()
    )
    stage1_metrics = _risk_metrics(
        stage1_outer["strategy_return"].to_numpy()
    )
    daily_metrics = _risk_metrics(
        daily_outer["strategy_return"].to_numpy()
    )
    raw_metrics = _risk_metrics(outer["strategy_return"].to_numpy())
    stress_metrics = _risk_metrics(fixed_25["strategy_return"].to_numpy())
    annual_metrics = _annual_metrics(outer_replay)
    annual_metrics["stage2_action_count"] = [
        int(
            outer_replay.loc[
                outer_replay["signal_date"].dt.year.eq(year),
                "stage2_action",
            ].sum()
        )
        for year in annual_metrics["year"]
    ]
    annual_relative_pass_count = int(
        annual_metrics["terminal_wealth_ratio_vs_soxx"].gt(1.0).sum()
    )
    annual_risk_pass_count = int(
        annual_metrics.apply(
            lambda row: _passes_risk(row.to_dict()),
            axis=1,
        ).sum()
    )
    from src.workflow.hierarchical_four_asset_policy import (
        _benchmark_efficiency,
    )

    benchmark_efficiency = _benchmark_efficiency(
        outer_replay.assign(soxs_weight=0.0),
        cost_bps=int(spec.transaction_cost_bps),
    )
    soxx_wealth = float(np.prod(1.0 + outer_replay["soxx_return"]))
    terminal_wealth_ratio_vs_soxx = (
        outer_metrics["terminal_wealth"] / soxx_wealth
    )
    up_mask = outer_replay["soxl_return"].gt(0.0)
    upside_capture_pct = float(
        100.0
        * outer_replay.loc[up_mask, "strategy_return"].sum()
        / outer_replay.loc[up_mask, "soxl_return"].sum()
    )
    stage1_by_signal = stage1_outer.set_index("signal_date")
    action_attribution = outer_replay.loc[
        outer_replay["stage2_action"],
        [
            "signal_date",
            "trade_date",
            "soxx_return",
            "recovery_probability",
            "stage2_probability",
            "strategy_return",
            "turnover",
        ],
    ].copy()
    action_attribution["stage1_only_return"] = action_attribution[
        "signal_date"
    ].map(stage1_by_signal["strategy_return"])
    action_attribution["incremental_return"] = (
        action_attribution["strategy_return"]
        - action_attribution["stage1_only_return"]
    )
    risk_analysis = pd.DataFrame(
        [
            {
                "variant": "daily_risk_controller",
                "segment": "outer_diagnostic_reused",
                "cost_bps": spec.transaction_cost_bps,
                **daily_metrics,
                "avg_cash_weight_pct": 100.0
                * daily_outer["cash_weight"].mean(),
            },
            {
                "variant": "stage1_10_30_recovery",
                "segment": "outer_diagnostic_reused",
                "cost_bps": spec.transaction_cost_bps,
                **stage1_metrics,
                "avg_cash_weight_pct": 100.0
                * stage1_outer["cash_weight"].mean(),
            },
            {
                "variant": "selected_staged_intraday_recovery",
                "segment": "outer_diagnostic_reused",
                "cost_bps": spec.transaction_cost_bps,
                **outer_metrics,
                "avg_cash_weight_pct": 100.0
                * outer_replay["cash_weight"].mean(),
            },
            {
                "variant": "selected_staged_intraday_recovery_fixed_decisions",
                "segment": "outer_diagnostic_reused",
                "cost_bps": spec.stress_cost_bps,
                **stress_metrics,
                "avg_cash_weight_pct": 100.0
                * fixed_25["cash_weight"].mean(),
            },
        ]
    )
    cost_sensitivity = risk_analysis[
        risk_analysis["variant"].str.startswith(
            "selected_staged_intraday_recovery"
        )
    ].copy()
    decision = (
        "promoted_research_candidate_staged_intraday_recovery"
        if _passes_risk(outer_metrics)
        and _passes_risk(stress_metrics)
        and outer_metrics["return_pct"] >= stage1_metrics["return_pct"]
        and annual_relative_pass_count >= 3
        and annual_risk_pass_count >= 3
        and benchmark_efficiency["return_maxdd"]
        > benchmark_efficiency["same_avg_risk_soxl_return_maxdd"]
        and benchmark_efficiency["return_maxdd"]
        > benchmark_efficiency["matched_beta_return_maxdd"]
        else "not_promoted_staged_intraday_recovery_gate_failed"
    )
    recorder_info = _record_qlib_artifacts(
        spec,
        stage1_params=stage1_params,
        selected_params=selected_params,
        dev_metrics=dev_metrics,
        outer_metrics=outer_metrics,
        stage1_metrics=stage1_metrics,
        stress_metrics=stress_metrics,
        model_selection=model_selection,
        outer_replay=outer_replay,
        stage1_teacher=stage1_teacher,
        stage2_teacher=selected_teacher,
        source_audit=source_audit,
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "decision": decision,
        "selection_uses_outer_test": False,
        "outer_status": "reused_diagnostic_not_fresh_unseen_test",
        "promotion_scope": "revised_soxx_relative_cash_and_risk_goal",
        "assets": ["SOXL", "SOXX", "cash"],
        "allocation_output": (
            "direct_legal_three_asset_weights_at_open_10_30_11_00_and_close"
        ),
        "method": {
            "stage1": (
                "10:30 learned recovery to validation-selected 0.6 beta"
            ),
            "stage2_teacher_label": (
                "counterfactual benefit of 11:00/11:30 recovery to 1.2 beta "
                "held only until regular-session close"
            ),
            "stage2_execution": (
                "second confirmation may raise beta intraday; close "
                "deterministically restores stage1 weights before overnight"
            ),
            "walk_forward": (
                "annual expanding training; each year uses only prior years"
            ),
            "selection": (
                "2019-2022 development windows only; outer years excluded "
                "from cut, model, threshold, and beta selection"
            ),
        },
        "base_experiment": {
            "experiment": spec.base_experiment,
            "recorder_id": base_summary["recorder_id"],
            "selected_model": stage1_params,
        },
        "selected_model": {
            **stage1_params,
            **selected_params,
        },
        "development_metrics_10bps": dev_metrics,
        "outer_diagnostic_metrics_10bps": outer_metrics,
        "raw_model_outer_metrics_10bps": raw_metrics,
        "stage1_outer_metrics_10bps": stage1_metrics,
        "daily_controller_outer_metrics_10bps": daily_metrics,
        "fixed_decision_stress_metrics_25bps": stress_metrics,
        "benchmark_comparison_10bps": {
            "terminal_wealth_ratio_vs_soxx": terminal_wealth_ratio_vs_soxx,
            "upside_capture_pct": upside_capture_pct,
            "weight_basis_note": (
                "same-risk and matched-beta comparators use close/overnight "
                "target weights; seven temporary stage2 intraday exposures "
                "are separately reported"
            ),
            **benchmark_efficiency,
        },
        "legacy_gate_status": {
            "upside_capture_floor_pct": 60.0,
            "upside_capture_gate_pass": upside_capture_pct >= 60.0,
            "note": (
                "legacy capture gate is reported but is not part of the "
                "revised SOXX-relative goal"
            ),
        },
        "stage2_effect": {
            "outer_return_improvement_pct_points": outer_metrics["return_pct"]
            - stage1_metrics["return_pct"],
            "terminal_wealth_ratio_vs_stage1": outer_metrics[
                "terminal_wealth"
            ]
            / stage1_metrics["terminal_wealth"],
            "outer_stage2_action_count": int(
                outer_replay["stage2_action"].sum()
            ),
            "outer_stage2_positive_increment_count": int(
                action_attribution["incremental_return"].gt(0.0).sum()
            ),
            "outer_stage2_negative_increment_count": int(
                action_attribution["incremental_return"].lt(0.0).sum()
            ),
            "outer_cash_reduction_pct_points": 100.0
            * (
                stage1_outer["cash_weight"].mean()
                - outer_replay["cash_weight"].mean()
            ),
            "stage1_avg_cash_weight_pct": 100.0
            * stage1_outer["cash_weight"].mean(),
            "selected_avg_close_overnight_cash_weight_pct": 100.0
            * outer_replay["cash_weight"].mean(),
            "selected_avg_stage2_intraday_cash_weight_pct": 100.0
            * outer_replay["stage2_cash_weight"].mean(),
        },
        "cash_timing_effect": {
            "raw_model_avg_cash_weight_pct": 100.0
            * outer["cash_weight"].mean(),
            "daily_controller_avg_cash_weight_pct": 100.0
            * daily_outer["cash_weight"].mean(),
            "selected_model_avg_cash_weight_pct": 100.0
            * outer_replay["cash_weight"].mean(),
            "risk_cash_added_vs_raw_pct_points": 100.0
            * (
                outer_replay["cash_weight"].mean()
                - outer["cash_weight"].mean()
            ),
        },
        "outer_annual_gate_summary": {
            "relative_pass_count": annual_relative_pass_count,
            "risk_pass_count": annual_risk_pass_count,
            "required_year_count": 3,
        },
        "source_audit": {
            **source_audit,
            "second_cut_timing": (
                f"bar_{selected_params['second_cut_time'].replace(':', '_')}"
                "_close_known_before_stage2_decision"
            ),
            "close_reversion_timing": (
                "regular_session_close; no stage2 leverage carried overnight"
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
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/positions.csv",
                f"artifacts/{spec.experiment}/strategy_returns.csv",
                f"artifacts/{spec.experiment}/stage1_recovery_training_frame.csv",
                f"artifacts/{spec.experiment}/stage2_recovery_training_frame.csv",
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
        action_attribution=action_attribution,
    )
    return summary


def _record_qlib_artifacts(
    spec,
    *,
    stage1_params,
    selected_params,
    dev_metrics,
    outer_metrics,
    stage1_metrics,
    stress_metrics,
    model_selection,
    outer_replay,
    stage1_teacher,
    stage2_teacher,
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
                "stage2_soxl_weight",
                "stage2_soxx_weight",
                "stage2_cash_weight",
                "soxl_weight",
                "soxx_weight",
                "cash_weight",
                "recovery_probability",
                "recovery_action",
                "stage2_probability",
                "stage2_action",
            ]
        ]
        files = {
            "positions.csv": positions,
            "strategy_returns.csv": outer_replay,
            "stage1_recovery_training_frame.csv": stage1_teacher,
            "stage2_recovery_training_frame.csv": stage2_teacher,
            "model_selection_full.csv": model_selection,
        }
        for filename, data in files.items():
            data.to_csv(tmp_dir / filename, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_experiment=spec.source_experiment,
                source_recorder_id=source_audit["source_recorder_id"],
                base_experiment=spec.base_experiment,
                allocation_assets="SOXL,SOXX,cash",
                allocation_output="direct_staged_legal_three_asset_weights",
                selection_uses_outer_test="false",
                stage1_action_beta=str(stage1_params["action_beta"]),
                second_cut_time=selected_params["second_cut_time"],
                stage2_teacher_beta=str(
                    selected_params["stage2_teacher_beta"]
                ),
                stage2_action_beta=str(
                    selected_params["stage2_action_beta"]
                ),
                stage2_probability_threshold=str(
                    selected_params["stage2_probability_threshold"]
                ),
                stage2_logistic_c=str(
                    selected_params["stage2_logistic_c"]
                ),
                stage2_holding_period="second_cut_to_regular_session_close",
            )
            R.log_metrics(
                development_return_pct=dev_metrics["return_pct"],
                development_max_drawdown_pct=dev_metrics[
                    "max_drawdown_pct"
                ],
                outer_return_pct=outer_metrics["return_pct"],
                outer_max_drawdown_pct=outer_metrics["max_drawdown_pct"],
                stage1_outer_return_pct=stage1_metrics["return_pct"],
                stress_25bps_return_pct=stress_metrics["return_pct"],
                stress_25bps_max_drawdown_pct=stress_metrics[
                    "max_drawdown_pct"
                ],
            )
            for filename in files:
                recorder.log_artifact(
                    (tmp_dir / filename).as_posix(),
                    artifact_path=spec.experiment,
                )
            recorder_info = {
                "recorder_id": recorder.id,
                "recorder_uri": str(recorder_uri.resolve()),
                "recorder_local_dir": str(
                    Path(recorder.get_local_dir()).resolve()
                ),
            }
    recorder_artifacts = (
        Path(recorder_info["recorder_local_dir"]) / "artifacts"
    )
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
    action_attribution,
) -> None:
    card = spec.card_dir
    (card / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_payload = {
        "selection_uses_outer_test": False,
        "selected_from_segment": "development_walk_forward",
        "outer_status": "reused_diagnostic_not_fresh_unseen_test",
        "selected_model": summary["selected_model"],
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
    (card / "selected_model.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    model_selection.sort_values(
        "terminal_wealth", ascending=False
    ).head(40).to_csv(card / "model_selection.csv", index=False)
    risk_analysis.to_csv(card / "risk_analysis.csv", index=False)
    annual_metrics.to_csv(card / "annual_metrics.csv", index=False)
    cost_sensitivity.to_csv(card / "cost_sensitivity.csv", index=False)
    action_attribution.to_csv(
        card / "stage2_action_attribution.csv", index=False
    )
    workflow = f"""experiment: {spec.experiment}
base_experiment: {spec.base_experiment}
source_experiment: {spec.source_experiment}
source_recorder_id: {summary['source_audit']['source_recorder_id']}
intraday_provider_uri: {spec.intraday_provider_uri.resolve()}
assets: [SOXL, SOXX, cash]
allocation_output: direct_staged_legal_three_asset_weights
selection_uses_outer_test: false
outer_status: reused_diagnostic_not_fresh_unseen_test
stage1_action_beta: {summary['selected_model']['action_beta']}
second_cut_time: {summary['selected_model']['second_cut_time']}
stage2_teacher_beta: {summary['selected_model']['stage2_teacher_beta']}
stage2_action_beta: {summary['selected_model']['stage2_action_beta']}
stage2_holding_period: second_cut_to_regular_session_close
close_action: restore_stage1_weights_before_overnight
transaction_cost_bps: {spec.transaction_cost_bps}
stress_cost_bps: {spec.stress_cost_bps}
stress_method: fixed_10bp_decisions_repriced_at_25bp
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  recorder_only_artifacts:
    - artifacts/{spec.experiment}/positions.csv
    - artifacts/{spec.experiment}/strategy_returns.csv
    - artifacts/{spec.experiment}/stage1_recovery_training_frame.csv
    - artifacts/{spec.experiment}/stage2_recovery_training_frame.csv
    - artifacts/{spec.experiment}/model_selection_full.csv
"""
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
    conclusion = f"""# 中文结论

- 做了什么：保留上一张 10:30 第一段恢复模型；只有第一段已经触发时，第二个模型才使用新增的 10:30-{summary['selected_model']['second_cut_time']} 路径确认是否把有效 beta 临时提高到 {summary['selected_model']['stage2_action_beta']:.1f}。
- 风险结构：第二段仓位只持有到当日常规收盘，收盘确定性退回第一段 {summary['selected_model']['action_beta']:.1f} beta，第二段杠杆不隔夜。每次 10:30、第二切点和收盘退回都计算换手成本；未触发时不允许隐含免费再平衡。
- 选择规则：第二切点、Logistic C、阈值和 beta 只使用 2019-2022 开发期年度/两年窗口；2023-2026 不参与选择。由于这些 outer 年份已被前序实验反复诊断，本卡只把它们标记为 reused diagnostic，不能声称是全新未见测试。
- 开发期 10bp：return={summary['development_metrics_10bps']['return_pct']:.4f}%，MaxDD={summary['development_metrics_10bps']['max_drawdown_pct']:.4f}%，worst 5d={summary['development_metrics_10bps']['worst_rolling_5d_pct']:.4f}%。
- 外层诊断 10bp：return={summary['outer_diagnostic_metrics_10bps']['return_pct']:.4f}%，MaxDD={summary['outer_diagnostic_metrics_10bps']['max_drawdown_pct']:.4f}%，worst day={summary['outer_diagnostic_metrics_10bps']['worst_day_pct']:.4f}%，CVaR5={summary['outer_diagnostic_metrics_10bps']['cvar5_pct']:.4f}%，worst 5d={summary['outer_diagnostic_metrics_10bps']['worst_rolling_5d_pct']:.4f}%。
- 相对第一段：第一段 return={summary['stage1_outer_metrics_10bps']['return_pct']:.4f}%，第二段提高 {summary['stage2_effect']['outer_return_improvement_pct_points']:.4f} 个百分点；终值倍率={summary['stage2_effect']['terminal_wealth_ratio_vs_stage1']:.4f}。outer 只有 {summary['stage2_effect']['outer_stage2_action_count']} 次二段动作，其中正增量 {summary['stage2_effect']['outer_stage2_positive_increment_count']} 次、负增量 {summary['stage2_effect']['outer_stage2_negative_increment_count']} 次。
- 现金解释：收盘/隔夜平均现金仅由 {summary['stage2_effect']['stage1_avg_cash_weight_pct']:.4f}% 降到 {summary['stage2_effect']['selected_avg_close_overnight_cash_weight_pct']:.4f}%；收益改善来自少数反弹日的日内现金时点，不是长期取消现金。
- 25bp 冻结决策压力：return={summary['fixed_decision_stress_metrics_25bps']['return_pct']:.4f}%，MaxDD={summary['fixed_decision_stress_metrics_25bps']['max_drawdown_pct']:.4f}%，worst 5d={summary['fixed_decision_stress_metrics_25bps']['worst_rolling_5d_pct']:.4f}%。
- 原模型差距：原模型 return={summary['raw_model_outer_metrics_10bps']['return_pct']:.4f}%，新模型仍少 {summary['raw_model_outer_metrics_10bps']['return_pct'] - summary['outer_diagnostic_metrics_10bps']['return_pct']:.4f} 个百分点，但原模型 MaxDD={summary['raw_model_outer_metrics_10bps']['max_drawdown_pct']:.4f}%。
- 是否晋升：{summary['decision']}。这是研究候选，不是生产或实盘成交证明；仍缺 bid/ask、排队、延迟、11:00 与收盘成交 replay。
- qlib recorder：{summary['recorder_id']}
- 不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
