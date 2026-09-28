from __future__ import annotations

import json
import hashlib
import pickle
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
from src.backtest.position_accounting import settle_target_session

from src.workflow.intraday_recovery_direct_allocation import (
    FEATURE_COLUMNS,
    IntradayRecoveryDirectAllocationSpec,
    _advance_state,
    _base_cap,
    _build_teacher_training_frame,
    _feature_row,
    _fit_recovery_model,
    _load_source_frame,
    _risk_metrics,
    project_legal_weights,
)
from src.workflow.stage1_confidence_veto_recovery import _simulate
from src.workflow.trainable_regime_policy_nested_wf import (
    _remove_qlib_code_snapshot_artifacts,
)


EXPERIMENT = "stage1_open_action_value_v1"
SOURCE_EXPERIMENT = "stage1_confidence_veto_recovery_v1"
OPEN_KNOWN_BASE_FEATURES = (
    "portfolio_drawdown",
    "current_beta_cap",
    "raw_effective_beta",
    "risk_state",
    "raw_soxl_weight",
    "raw_soxx_weight",
    "raw_cash_weight",
    "tail_probability",
    "alpha_probability",
    "portfolio_ret_1d",
    "portfolio_ret_3d",
    "portfolio_ret_5d",
)
ACTION_FEATURES = (
    *OPEN_KNOWN_BASE_FEATURES,
    "target_soxl_weight",
    "target_soxx_weight",
    "target_cash_weight",
    "pretrade_soxl_weight",
    "pretrade_soxx_weight",
    "pretrade_cash_weight",
    "soxl_weight_delta",
    "soxx_weight_delta",
    "effective_beta_delta",
    "open_turnover",
)


@dataclass(frozen=True)
class Stage1OpenActionValueSpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = SOURCE_EXPERIMENT
    transaction_cost_bps: float = 10.0
    stress_cost_bps: float = 25.0
    logistic_c: float = 0.10
    execute_probability_threshold: float = 0.50
    maximum_skippable_turnover: float = 0.05
    mandatory_derisk_beta_delta: float = 0.0
    minimum_training_rows: int = 200
    maximum_open_trade_day_ratio: float = 0.92
    minimum_annual_wealth_ratio: float = 0.985

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / self.source_experiment


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _source_contract(spec: Stage1OpenActionValueSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads(
        (spec.source_card_dir / "summary.json").read_text(encoding="utf-8")
    )
    selected = json.loads(
        (spec.source_card_dir / "selected_model.json").read_text(
            encoding="utf-8"
        )
    )
    if summary.get("selection_uses_outer_test") is not False:
        raise ValueError("blocked: confidence-veto source used outer for selection")
    if summary.get("development_gate_pass") is not True:
        raise ValueError("blocked: confidence-veto source did not pass development gates")
    if selected.get("production_enabled") is not False:
        raise ValueError("blocked: research source unexpectedly marked production enabled")
    return summary, selected


def _baseline_replays(
    frame: pd.DataFrame,
    *,
    source_spec: IntradayRecoveryDirectAllocationSpec,
    teacher: pd.DataFrame,
    selected: dict[str, Any],
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    kwargs = {
        "base_spec": source_spec,
        "teacher": teacher,
        "c_value": float(selected["logistic_c"]),
        "threshold": float(selected["probability_threshold"]),
        "action_beta": float(selected["action_beta"]),
        "alpha_floor": float(selected["minimum_source_alpha_probability"]),
    }
    development = _simulate(
        frame[frame["segment"].eq("development_walk_forward")],
        **kwargs,
        first_action_year=source_spec.first_intraday_action_year,
        model_cache={},
    )
    first_outer_year = min(source_spec.outer_years)
    frozen_recovery = _fit_recovery_model(
        teacher, year=first_outer_year, c_value=float(selected["logistic_c"])
    )
    outer = _simulate(
        frame[frame["segment"].eq("outer_diagnostic")],
        **kwargs,
        first_action_year=first_outer_year,
        model_cache={
            (year, float(selected["logistic_c"])): frozen_recovery
            for year in source_spec.outer_years
        },
    )
    if not np.isclose(cost_bps, source_spec.transaction_cost_bps):
        raise ValueError("blocked: source baseline cost contract drifted")
    return development, outer, frozen_recovery


def _branch_outcomes(
    row: pd.Series,
    *,
    pretrade: np.ndarray,
    cost_bps: float,
) -> dict[str, Any]:
    target = row[["open_soxl_weight", "open_soxx_weight", "open_cash_weight"]].to_numpy(dtype="float64")
    recovery = bool(row["recovery_action"])
    recovery_target = row[["soxl_weight", "soxx_weight", "cash_weight"]].to_numpy(dtype="float64") if recovery else None
    kwargs = {
        "asset_returns": [row["soxl_return"], row["soxx_return"], 0.0],
        "cost_bps": cost_bps,
        "recovery_target_weights": recovery_target,
        "early_returns": [row["early_ret_soxl"], row["early_ret_soxx"], 0.0] if recovery else None,
    }
    executed = settle_target_session(pretrade, open_target_weights=target, **kwargs)
    held = settle_target_session(pretrade, open_target_weights=None, **kwargs)
    return {
        "execute_net": executed["net_return"], "skip_net": held["net_return"],
        "execute_turnover": executed["turnover"], "skip_turnover": held["turnover"],
        "execute_end": executed["ending_weights"], "skip_end": held["ending_weights"],
    }


def _action_features(
    row: pd.Series,
    *,
    pretrade: np.ndarray,
    drawdown: float,
    state: int,
    history: list[float],
) -> dict[str, float]:
    target = row[
        ["open_soxl_weight", "open_soxx_weight", "open_cash_weight"]
    ].to_numpy(dtype="float64")
    feature_row = row.copy()
    feature_row["soxl_weight"] = row["soxl_weight_source"]
    feature_row["soxx_weight"] = row["soxx_weight_source"]
    feature_row["cash_weight"] = row["cash_weight_source"]
    return build_open_action_features(
        feature_row,
        target=target,
        pretrade=pretrade,
        drawdown=drawdown,
        cap=float(row["current_beta_cap"]),
        state=state,
        history=history,
    )


def build_open_action_features(
    row: pd.Series,
    *,
    target: np.ndarray,
    pretrade: np.ndarray,
    drawdown: float,
    cap: float,
    state: int,
    history: list[float],
) -> dict[str, float]:
    """Build the exact training/runtime contract without reading intraday data."""
    delta = target - pretrade
    return {
        "portfolio_drawdown": drawdown,
        "current_beta_cap": cap,
        "raw_effective_beta": float(3.0 * row["soxl_weight"] + row["soxx_weight"]),
        "risk_state": float(state),
        "raw_soxl_weight": float(row["soxl_weight"]),
        "raw_soxx_weight": float(row["soxx_weight"]),
        "raw_cash_weight": float(row["cash_weight"]),
        "tail_probability": float(row["tail_probability"]),
        "alpha_probability": float(row["alpha_probability"]),
        "portfolio_ret_1d": history[-1] if history else 0.0,
        "portfolio_ret_3d": float(np.prod(1.0 + np.asarray(history[-3:])) - 1.0) if history else 0.0,
        "portfolio_ret_5d": float(np.prod(1.0 + np.asarray(history[-5:])) - 1.0) if history else 0.0,
        "target_soxl_weight": float(target[0]),
        "target_soxx_weight": float(target[1]),
        "target_cash_weight": float(target[2]),
        "pretrade_soxl_weight": float(pretrade[0]),
        "pretrade_soxx_weight": float(pretrade[1]),
        "pretrade_cash_weight": float(pretrade[2]),
        "soxl_weight_delta": float(delta[0]),
        "soxx_weight_delta": float(delta[1]),
        "effective_beta_delta": float(3.0 * delta[0] + delta[1]),
        "open_turnover": float(np.abs(delta[:2]).sum()),
    }


def _join_market_and_replay(
    market: pd.DataFrame, replay: pd.DataFrame
) -> pd.DataFrame:
    replay_columns = [
        "signal_date",
        "open_soxl_weight",
        "open_soxx_weight",
        "open_cash_weight",
        "soxl_weight",
        "soxx_weight",
        "cash_weight",
        "recovery_action",
        "current_beta_cap",
        "recovery_beta_target",
        "gross_return",
        "turnover",
        "strategy_return",
    ]
    source = market.rename(
        columns={
            "soxl_weight": "soxl_weight_source",
            "soxx_weight": "soxx_weight_source",
            "cash_weight": "cash_weight_source",
        }
    )
    source = source.drop(
        columns=[
            column
            for column in replay_columns
            if column != "signal_date" and column in source.columns
        ]
    )
    return (
        source.merge(
            replay.loc[:, replay_columns],
            on="signal_date",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("signal_date")
        .reset_index(drop=True)
    )


def _session_timestamp(dates: pd.Series, *, minute: int) -> pd.Series:
    """Timestamp a US session event without assuming calendar-day spacing."""
    return (
        pd.to_datetime(dates).dt.normalize()
        + pd.Timedelta(hours=9, minutes=minute)
    ).dt.tz_localize("America/New_York").dt.tz_convert("UTC")


def _build_training_frame(
    market: pd.DataFrame,
    replay: pd.DataFrame,
    *,
    source_spec: IntradayRecoveryDirectAllocationSpec,
    label_cost_bps: float,
) -> pd.DataFrame:
    joined = _join_market_and_replay(market, replay)
    equity = peak = 1.0
    state = clock = 0
    history: list[float] = []
    pretrade = np.array([0.0, 0.0, 1.0])
    rows: list[dict[str, Any]] = []
    branch_cache: list[dict[str, Any]] = []
    for row in joined.to_dict(orient="records"):
        series = pd.Series(row)
        drawdown = max(0.0, 1.0 - equity / peak)
        state, clock, _ = _advance_state(state, clock, drawdown, source_spec)
        features = _action_features(
            series,
            pretrade=pretrade,
            drawdown=drawdown,
            state=state,
            history=history,
        )
        branches = _branch_outcomes(series, pretrade=pretrade, cost_bps=label_cost_bps)
        baseline_branches = _branch_outcomes(
            series, pretrade=pretrade, cost_bps=source_spec.transaction_cost_bps
        )
        rows.append(
            {
                "signal_date": series["signal_date"],
                "year": int(series["signal_date"].year),
                **features,
            }
        )
        branch_cache.append(branches)
        strategy_return = float(series["strategy_return"])
        history.append(strategy_return)
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        pretrade = baseline_branches["execute_end"]
    training = pd.DataFrame(rows)
    advantages: list[float] = []
    for index, branches in enumerate(branch_cache):
        execute_turnover = skip_turnover = 0.0
        if index + 1 < len(training):
            next_target = training.iloc[index + 1][
                [
                    "target_soxl_weight",
                    "target_soxx_weight",
                    "target_cash_weight",
                ]
            ].to_numpy(dtype="float64")
            execute_turnover += float(
                np.abs(next_target[:2] - branches["execute_end"][:2]).sum()
            ) * (1.0 + branches["execute_net"])
            skip_turnover += float(
                np.abs(next_target[:2] - branches["skip_end"][:2]).sum()
            ) * (1.0 + branches["skip_net"])
        advantage = (
            branches["execute_net"]
            - branches["skip_net"]
            - (execute_turnover - skip_turnover)
            * label_cost_bps
            / 10_000.0
        )
        advantages.append(float(advantage))
    training["execute_advantage"] = advantages
    training.loc[training.index[-1], "execute_advantage"] = np.nan
    training["decision_at"] = _session_timestamp(joined["trade_date"], minute=31)
    exit_at = _session_timestamp(joined["exit_date"], minute=30)
    next_target_at = training["decision_at"].shift(-1)
    # The return needs the exit open; the reset-cost term also needs the next
    # target, which includes the shadow controller's state at its actual open.
    training["label_available_at"] = pd.concat(
        [exit_at, next_target_at], axis=1
    ).max(axis=1).where(next_target_at.notna())
    return training


def _prepare_mature_training(dataset, segment: str, *, decision_at: str) -> pd.DataFrame:
    prepared = dataset.prepare(segment, col_set=["feature", "label", "metadata"])
    prepared.columns = [str(column[-1]) for column in prepared.columns]
    available_at = pd.to_datetime(prepared["label_available_at"], utc=True)
    return prepared.loc[
        available_at.notna() & available_at.lt(pd.Timestamp(decision_at))
    ].dropna(subset=["execute_advantage"])


def _build_qlib_dataset(
    training: pd.DataFrame,
    spec: Stage1OpenActionValueSpec,
    *,
    evaluation_frame: pd.DataFrame,
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    static = pd.concat(
        {
            "feature": training.loc[:, list(ACTION_FEATURES)].astype("float64"),
            "label": training.loc[:, ["execute_advantage"]].astype("float64"),
            "metadata": training.loc[:, ["label_available_at"]],
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(training["signal_date"]).dt.normalize(),
            np.full(len(training), "STAGE1_OPEN_ACTION"),
        ],
        names=["datetime", "instrument"],
    )
    start = pd.to_datetime(training["signal_date"]).min().date().isoformat()
    years = sorted(int(year) for year in training["year"].unique())
    segments = {
        f"train_{year}": (start, f"{year - 1}-12-31") for year in years[1:]
    }
    segments["train_outer"] = (start, "2022-12-31")
    evaluation_dates = pd.to_datetime(evaluation_frame["signal_date"])
    evaluation_opens = _session_timestamp(evaluation_frame["trade_date"], minute=31)
    training_cutoffs: dict[str, str] = {}
    for name in segments:
        test_year = 2023 if name == "train_outer" else int(name.removeprefix("train_"))
        first_test_open = evaluation_opens.loc[evaluation_dates.dt.year.eq(test_year)].min()
        if pd.isna(first_test_open):
            raise ValueError(f"blocked: missing actual test execution date for {name}")
        training_cutoffs[name] = first_test_open.isoformat()
    dataset = DatasetH(
        handler=DataHandlerLP(
            instruments=None,
            start_time=static.index.get_level_values("datetime").min(),
            end_time=static.index.get_level_values("datetime").max(),
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
        "features": list(ACTION_FEATURES),
        "label": "execute_advantage_after_25bps_current_and_next_reset_cost",
        "label_available_at_rule": "max(exit_date_09:30_ET, next_target_trade_date_09:31_ET)",
        "training_cutoffs": training_cutoffs,
        "label_maturity_filter": "label_available_at strictly before first test execution",
        "segments": {name: list(bounds) for name, bounds in segments.items()},
        "segment_rows": {
            name: int(len(_prepare_mature_training(dataset, name, decision_at=training_cutoffs[name])))
            for name in segments
        },
        "excluded_unavailable_label_rows": {
            name: int(
                len(dataset.prepare(name))
                - len(_prepare_mature_training(dataset, name, decision_at=training_cutoffs[name]))
            )
            for name in segments
        },
    }
    return dataset, audit


def _fit_models(
    dataset, spec: Stage1OpenActionValueSpec, *, training_cutoffs: dict[str, str]
):
    models: dict[int, Any] = {}
    rows: list[dict[str, Any]] = []
    for year in range(2017, 2023):
        segment = f"train_{year}"
        if segment not in dataset.segments:
            continue
        prepared = _prepare_mature_training(
            dataset, segment, decision_at=training_cutoffs[segment]
        )
        if len(prepared) < spec.minimum_training_rows:
            continue
        model = make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            LogisticRegression(
                C=spec.logistic_c,
                class_weight="balanced",
                max_iter=2_000,
                random_state=42,
            ),
        )
        model.fit(
            prepared.loc[:, list(ACTION_FEATURES)],
            prepared["execute_advantage"].gt(0.0),
        )
        models[year] = model
        probabilities = model.predict_proba(
            prepared.loc[:, list(ACTION_FEATURES)]
        )[:, 1]
        rows.append(
            {
                "model_year": year,
                "training_rows": int(len(prepared)),
                "training_end": f"{year - 1}-12-31",
                "label_available_at_max": prepared["label_available_at"].max().isoformat(),
                "test_decision_at": training_cutoffs[segment],
                "positive_label_ratio": float(
                    prepared["execute_advantage"].gt(0.0).mean()
                ),
                "training_accuracy": float(
                    np.mean(
                        (probabilities >= spec.execute_probability_threshold)
                        == prepared["execute_advantage"].gt(0.0).to_numpy()
                    )
                ),
            }
        )
    outer = _prepare_mature_training(
        dataset, "train_outer", decision_at=training_cutoffs["train_outer"]
    )
    if len(outer) < spec.minimum_training_rows:
        raise ValueError("blocked: insufficient development rows for outer model")
    frozen = make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        LogisticRegression(
            C=spec.logistic_c,
            class_weight="balanced",
            max_iter=2_000,
            random_state=42,
        ),
    )
    frozen.fit(
        outer.loc[:, list(ACTION_FEATURES)],
        outer["execute_advantage"].gt(0.0),
    )
    for year in (2023, 2024, 2025, 2026):
        models[year] = frozen
    rows.append(
        {
            "model_year": "outer_frozen_2023_2026",
            "training_rows": int(len(outer)),
            "training_end": "2022-12-31",
            "label_available_at_max": outer["label_available_at"].max().isoformat(),
            "test_decision_at": training_cutoffs["train_outer"],
            "positive_label_ratio": float(
                outer["execute_advantage"].gt(0.0).mean()
            ),
            "training_accuracy": float(
                np.mean(
                    (
                        frozen.predict_proba(
                            outer.loc[:, list(ACTION_FEATURES)]
                        )[:, 1]
                        >= spec.execute_probability_threshold
                    )
                    == outer["execute_advantage"].gt(0.0).to_numpy()
                )
            ),
        }
    )
    return models, pd.DataFrame(rows)


def decide_open_execution(
    features: dict[str, Any] | pd.Series,
    model: Any | None,
    *,
    execute_probability_threshold: float,
    maximum_skippable_turnover: float,
    mandatory_derisk_beta_delta: float,
) -> dict[str, Any]:
    """Route an open target using only information known at that decision."""
    probability = np.nan
    if model is not None:
        probability = float(
            model.predict_proba(pd.DataFrame([features]).loc[:, list(ACTION_FEATURES)])[0, 1]
        )
    mandatory_derisk = (
        float(features["effective_beta_delta"]) < -mandatory_derisk_beta_delta
    )
    large_transition = float(features["open_turnover"]) > maximum_skippable_turnover
    execute = bool(
        mandatory_derisk
        or large_transition
        or not np.isfinite(probability)
        or probability >= execute_probability_threshold
    )
    return {
        "execute_open_target": execute,
        "execute_probability": probability,
        "mandatory_derisk": mandatory_derisk,
        "large_transition": large_transition,
    }


def prepare_on_policy_open(
    row: pd.Series,
    *,
    pretrade_weights: np.ndarray,
    equity: float,
    peak: float,
    risk_state: int,
    risk_clock: int,
    history: list[float],
    recovery_spec: IntradayRecoveryDirectAllocationSpec,
    model: Any | None,
    action_spec: Stage1OpenActionValueSpec,
) -> dict[str, Any]:
    """Prepare one open decision from the strategy's own settled ledger state."""
    drawdown = max(0.0, 1.0 - equity / peak)
    state, clock, q = _advance_state(risk_state, risk_clock, drawdown, recovery_spec)
    cap, raw_beta = _base_cap(
        row, q=q, drawdown=drawdown, history=history, spec=recovery_spec
    )
    target = project_legal_weights(
        float(row["soxl_weight"]), float(row["soxx_weight"]), cap,
        minimum_cash_weight=recovery_spec.minimum_cash_weight,
        maximum_soxl_weight=recovery_spec.maximum_soxl_weight,
    )
    pretrade = np.asarray(pretrade_weights, dtype="float64")
    features = build_open_action_features(
        row, target=target, pretrade=pretrade, drawdown=drawdown,
        cap=cap, state=state, history=history,
    )
    decision = decide_open_execution(
        features, model,
        execute_probability_threshold=action_spec.execute_probability_threshold,
        maximum_skippable_turnover=action_spec.maximum_skippable_turnover,
        mandatory_derisk_beta_delta=action_spec.mandatory_derisk_beta_delta,
    )
    legal_holdings = bool(
        np.isfinite(pretrade).all()
        and pretrade.min() >= -1e-12
        and pretrade[0] <= recovery_spec.maximum_soxl_weight + 1e-12
        and pretrade[2] >= recovery_spec.minimum_cash_weight - 1e-12
        and 3.0 * pretrade[0] + pretrade[1] <= cap + 1e-12
    )
    decision["mandatory_position_repair"] = not legal_holdings
    decision["execute_open_target"] |= not legal_holdings
    return {
        **decision,
        "desired_open_weights": target,
        "open_weights": target if decision["execute_open_target"] else pretrade.copy(),
        "drawdown": drawdown,
        "current_beta_cap": cap,
        "raw_beta": raw_beta,
        "risk_state": state,
        "risk_clock": clock,
        "history": list(history),
        "features": features,
    }


def prepare_on_policy_recovery(
    row: pd.Series,
    *,
    open_state: dict[str, Any],
    recovery_spec: IntradayRecoveryDirectAllocationSpec,
    model: Any | None,
    selected_model: dict[str, Any],
) -> dict[str, Any]:
    """Use the observed first hour and the same day's saved open controller."""
    target_beta = min(open_state["raw_beta"], float(selected_model["action_beta"]))
    eligible = bool(
        model is not None
        and open_state["drawdown"] >= recovery_spec.recovery_eligibility_drawdown
        and target_beta > open_state["current_beta_cap"] + 1e-12
        and pd.notna(row.get("early_ret_soxx"))
        and pd.notna(row.get("alpha_probability"))
        and row["alpha_probability"] >= selected_model["minimum_source_alpha_probability"]
    )
    probability = np.nan
    if eligible:
        features = _feature_row(
            row, drawdown=open_state["drawdown"], cap=open_state["current_beta_cap"],
            raw_beta=open_state["raw_beta"], state=open_state["risk_state"],
            history=open_state["history"],
        )
        probability = float(model.predict_proba(pd.DataFrame([features]).loc[:, list(FEATURE_COLUMNS)])[0, 1])
    recovery = bool(eligible and probability >= selected_model["probability_threshold"])
    target = project_legal_weights(
        float(row["soxl_weight"]), float(row["soxx_weight"]), target_beta,
        minimum_cash_weight=recovery_spec.minimum_cash_weight,
        maximum_soxl_weight=recovery_spec.maximum_soxl_weight,
    ) if recovery else None
    return {"recovery_action": recovery, "recovery_probability": probability, "target_weights": target}


def _simulate_on_policy_router(
    frame: pd.DataFrame,
    *,
    models: dict[int, Any],
    recovery_models: dict[int, Any],
    source_spec: IntradayRecoveryDirectAllocationSpec,
    spec: Stage1OpenActionValueSpec,
    selected_model: dict[str, Any],
    segment: str,
) -> pd.DataFrame:
    equity = peak = 1.0
    state = clock = 0
    history: list[float] = []
    previous = np.array([0.0, 0.0, 1.0])
    rows = []
    for _, row in frame.sort_values("signal_date").iterrows():
        year = int(row["signal_date"].year)
        opened = prepare_on_policy_open(
            row, pretrade_weights=previous, equity=equity, peak=peak,
            risk_state=state, risk_clock=clock, history=history,
            recovery_spec=source_spec, model=models.get(year), action_spec=spec,
        )
        state, clock = opened["risk_state"], opened["risk_clock"]
        recovered = prepare_on_policy_recovery(
            row, open_state=opened, recovery_spec=source_spec,
            model=recovery_models.get(year), selected_model=selected_model,
        )
        open_weights = opened["open_weights"]
        open_turnover = float(np.abs(open_weights[:2] - previous[:2]).sum())
        close_weights = recovered["target_weights"] if recovered["recovery_action"] else open_weights
        settled = settle_target_session(
            previous,
            open_target_weights=opened["desired_open_weights"] if opened["execute_open_target"] else None,
            asset_returns=[row["soxl_return"], row["soxx_return"], 0.0],
            cost_bps=spec.transaction_cost_bps,
            recovery_target_weights=recovered["target_weights"],
            early_returns=[row["early_ret_soxl"], row["early_ret_soxx"], 0.0] if recovered["recovery_action"] else None,
        )
        open_turnover = settled["open_turnover"]
        gross, turnover = settled["gross_return"], settled["turnover"]
        strategy_return = settled["net_return"]
        previous = settled["ending_weights"]
        history.append(strategy_return)
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        rows.append({
            "signal_date": row["signal_date"], "trade_date": row["trade_date"], "exit_date": row["exit_date"],
            "segment": segment, "execute_probability": opened["execute_probability"],
            "execute_open_target": opened["execute_open_target"],
            "execute_open_trade": open_turnover > 1e-8,
            "skipped_open_target": not opened["execute_open_target"],
            "mandatory_derisk": opened["mandatory_derisk"],
            "mandatory_position_repair": opened["mandatory_position_repair"],
            "large_transition": opened["large_transition"],
            "recovery_action": recovered["recovery_action"],
            "open_soxl_weight": float(open_weights[0]), "open_soxx_weight": float(open_weights[1]),
            "open_cash_weight": float(open_weights[2]),
            "soxl_return": float(row["soxl_return"]), "soxx_return": float(row["soxx_return"]),
            "early_ret_soxl": row["early_ret_soxl"], "early_ret_soxx": row["early_ret_soxx"],
            "soxl_weight": float(close_weights[0]), "soxx_weight": float(close_weights[1]), "cash_weight": float(close_weights[2]),
            "portfolio_drawdown": opened["drawdown"], "current_beta_cap": opened["current_beta_cap"],
            "risk_state": state, "risk_clock": clock, "equity": equity,
            "gross_return": gross, "turnover": turnover, "strategy_return": strategy_return,
            "ending_soxl_weight": previous[0], "ending_soxx_weight": previous[1], "ending_cash_weight": previous[2],
            "fees": settled["fees"],
            "cost_bps": spec.transaction_cost_bps,
        })
    return pd.DataFrame(rows)


def _simulate_router(
    market: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    models: dict[int, Any],
    source_spec: IntradayRecoveryDirectAllocationSpec,
    spec: Stage1OpenActionValueSpec,
    segment: str,
) -> pd.DataFrame:
    joined = _join_market_and_replay(market, baseline)
    shadow_training = _build_training_frame(
        market,
        baseline,
        source_spec=source_spec,
        label_cost_bps=spec.stress_cost_bps,
    ).set_index("signal_date")
    pretrade = np.array([0.0, 0.0, 1.0])
    rows: list[dict[str, Any]] = []
    for row in joined.to_dict(orient="records"):
        series = pd.Series(row)
        shadow = shadow_training.loc[series["signal_date"]].copy()
        target = series[
            ["open_soxl_weight", "open_soxx_weight", "open_cash_weight"]
        ].to_numpy(dtype="float64")
        delta = target - pretrade
        shadow["pretrade_soxl_weight"] = pretrade[0]
        shadow["pretrade_soxx_weight"] = pretrade[1]
        shadow["pretrade_cash_weight"] = pretrade[2]
        shadow["soxl_weight_delta"] = delta[0]
        shadow["soxx_weight_delta"] = delta[1]
        shadow["effective_beta_delta"] = 3.0 * delta[0] + delta[1]
        shadow["open_turnover"] = np.abs(delta[:2]).sum()
        model = models.get(int(series["signal_date"].year))
        decision = decide_open_execution(
            shadow,
            model,
            execute_probability_threshold=spec.execute_probability_threshold,
            maximum_skippable_turnover=spec.maximum_skippable_turnover,
            mandatory_derisk_beta_delta=spec.mandatory_derisk_beta_delta,
        )
        probability = decision["execute_probability"]
        mandatory_derisk = decision["mandatory_derisk"]
        large_transition = decision["large_transition"]
        execute = decision["execute_open_target"]
        open_weights = target if execute else pretrade
        open_turnover = float(
            np.abs(open_weights[:2] - pretrade[:2]).sum()
        )
        close_weights = series[
            ["soxl_weight", "soxx_weight", "cash_weight"]
        ].to_numpy(dtype="float64")
        # The recovery signal is observed at 10:31, after the open decision.
        # It may change the recovery trade, never the earlier execute/hold route.
        recovery_day = bool(series["recovery_action"])
        settled = settle_target_session(
            pretrade, open_target_weights=target if execute else None,
            asset_returns=[series["soxl_return"], series["soxx_return"], 0.0],
            cost_bps=spec.transaction_cost_bps,
            recovery_target_weights=close_weights if recovery_day else None,
            early_returns=[series["early_ret_soxl"], series["early_ret_soxx"], 0.0] if recovery_day else None,
        )
        if not recovery_day:
            close_weights = open_weights
        gross_return, turnover = settled["gross_return"], settled["turnover"]
        ending = settled["ending_weights"]
        strategy_return = settled["net_return"]
        rows.append(
            {
                "signal_date": series["signal_date"],
                "trade_date": series["trade_date"],
                "exit_date": series["exit_date"],
                "segment": segment,
                "execute_probability": probability,
                "execute_open_trade": bool(open_turnover > 1e-8),
                "skipped_open_target": bool(not execute),
                "mandatory_derisk": mandatory_derisk,
                "large_transition": large_transition,
                "recovery_action": recovery_day,
                "soxl_weight": float(close_weights[0]),
                "soxx_weight": float(close_weights[1]),
                "cash_weight": float(close_weights[2]),
                "gross_return": gross_return,
                "turnover": turnover,
                "strategy_return": strategy_return,
                "cost_bps": spec.transaction_cost_bps,
            }
        )
        pretrade = ending
    return pd.DataFrame(rows)


def _stress(replay: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    """Reprice the fixed target/HOLD/recovery path with actual net cash at each fee."""
    result = replay.copy()
    previous = np.array([0.0, 0.0, 1.0])
    equity = 1.0
    if "equity" in result:
        result["decision_source_equity"] = result["equity"]
    for index, row in result.iterrows():
        open_target = row[["open_soxl_weight", "open_soxx_weight", "open_cash_weight"]].to_numpy(dtype="float64") if bool(row["execute_open_target"]) else None
        recovery = bool(row["recovery_action"])
        recovery_target = row[["soxl_weight", "soxx_weight", "cash_weight"]].to_numpy(dtype="float64") if recovery else None
        settled = settle_target_session(
            previous, open_target_weights=open_target,
            asset_returns=[row["soxl_return"], row["soxx_return"], 0.0],
            cost_bps=cost_bps, recovery_target_weights=recovery_target,
            early_returns=[row["early_ret_soxl"], row["early_ret_soxx"], 0.0] if recovery else None,
        )
        result.loc[index, "strategy_return"] = settled["net_return"]
        result.loc[index, "gross_return"] = settled["gross_return"]
        result.loc[index, "turnover"] = settled["turnover"]
        result.loc[index, "fees"] = settled["fees"]
        previous = settled["ending_weights"]
        equity *= 1.0 + settled["net_return"]
        result.loc[index, "equity"] = equity
        for asset, weight in zip(("soxl", "soxx", "cash"), previous, strict=True):
            result.loc[index, f"ending_{asset}_weight"] = weight
    result["cost_bps"] = float(cost_bps)
    return result


def _metrics(
    replay: pd.DataFrame,
    *,
    variant: str,
    segment: str,
    cost_bps: float,
) -> dict[str, Any]:
    open_trade_column = (
        "execute_open_trade" if "execute_open_trade" in replay else None
    )
    return {
        "variant": variant,
        "segment": segment,
        "cost_bps": float(cost_bps),
        **_risk_metrics(replay["strategy_return"].to_numpy()),
        "total_turnover": float(replay["turnover"].sum()),
        "open_trade_days": int(
            replay[open_trade_column].sum()
            if open_trade_column
            else (replay["turnover"] > 1e-12).sum()
        ),
        "recovery_action_count": int(replay["recovery_action"].sum()),
        "avg_cash_weight_pct": float(100.0 * replay["cash_weight"].mean()),
    }


def _annual_metrics(
    baseline: pd.DataFrame, candidate: pd.DataFrame, *, segment: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in sorted(baseline["signal_date"].dt.year.unique()):
        for variant, replay in (
            ("confidence_veto_baseline", baseline),
            ("open_action_value_candidate", candidate),
        ):
            section = replay[replay["signal_date"].dt.year.eq(year)]
            rows.append(
                {
                    "segment": segment,
                    "year": int(year),
                    "variant": variant,
                    **_risk_metrics(section["strategy_return"].to_numpy()),
                    "total_turnover": float(section["turnover"].sum()),
                    "open_trade_days": int(
                        section["execute_open_trade"].sum()
                        if "execute_open_trade" in section
                        else (section["turnover"] > 1e-12).sum()
                    ),
                }
            )
    result = pd.DataFrame(rows)
    baseline_wealth = result[
        result["variant"].eq("confidence_veto_baseline")
    ].set_index("year")["terminal_wealth"]
    result["wealth_ratio_vs_baseline"] = result.apply(
        lambda row: row["terminal_wealth"] / baseline_wealth.loc[row["year"]],
        axis=1,
    )
    return result


def _promotion_gates(
    *,
    baseline_10: dict[str, Any],
    baseline_25: dict[str, Any],
    candidate_10: dict[str, Any],
    candidate_25: dict[str, Any],
    annual: pd.DataFrame,
    fallback: pd.DataFrame,
    baseline_replay: pd.DataFrame,
    spec: Stage1OpenActionValueSpec,
) -> dict[str, bool]:
    candidate_annual = annual[
        annual["variant"].eq("open_action_value_candidate")
        & annual["segment"].eq("development_walk_forward")
        & annual["year"].ge(2017)
    ]
    return {
        "development_return_above_baseline": bool(
            candidate_10["return_pct"] > baseline_10["return_pct"]
        ),
        "stress_25bps_return_above_baseline": bool(
            candidate_25["return_pct"] > baseline_25["return_pct"]
        ),
        "max_drawdown_not_worse": bool(
            candidate_10["max_drawdown_pct"]
            >= baseline_10["max_drawdown_pct"] - 1e-10
        ),
        "worst_day_not_worse": bool(
            candidate_10["worst_day_pct"] >= baseline_10["worst_day_pct"] - 1e-10
        ),
        "cvar5_not_worse": bool(
            candidate_10["cvar5_pct"] >= baseline_10["cvar5_pct"] - 1e-10
        ),
        "worst_5d_not_worse": bool(
            candidate_10["worst_rolling_5d_pct"]
            >= baseline_10["worst_rolling_5d_pct"] - 1e-10
        ),
        "open_trade_days_reduced_8pct": bool(
            candidate_10["open_trade_days"]
            <= baseline_10["open_trade_days"]
            * spec.maximum_open_trade_day_ratio
        ),
        "turnover_not_increased": bool(
            candidate_10["total_turnover"] <= baseline_10["total_turnover"]
        ),
        "annual_wealth_floor_pass": bool(
            candidate_annual["wealth_ratio_vs_baseline"].min()
            >= spec.minimum_annual_wealth_ratio
        ),
        "baseline_fallback_identity": bool(
            np.allclose(
                fallback["strategy_return"].to_numpy(),
                baseline_replay["strategy_return"].to_numpy(),
                atol=1e-12,
            )
            and np.allclose(
                fallback["turnover"].to_numpy(),
                baseline_replay["turnover"].to_numpy(),
                atol=1e-12,
            )
        ),
        "outer_not_used_for_selection": True,
    }


def _record(
    spec: Stage1OpenActionValueSpec,
    *,
    training: pd.DataFrame,
    diagnostics: pd.DataFrame,
    comparison: pd.DataFrame,
    qlib_risk: pd.DataFrame,
    candidate_dev: pd.DataFrame,
    candidate_outer: pd.DataFrame,
    shadow_dev: pd.DataFrame,
    shadow_outer: pd.DataFrame,
    models: dict[int, Any],
    frozen_recovery: Any,
    dataset_audit: dict[str, Any],
    gates: dict[str, bool],
) -> dict[str, Any]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        temp = Path(tmp)
        training.to_csv(temp / "action_value_training_frame.csv", index=False)
        diagnostics.to_csv(temp / "training_diagnostics.csv", index=False)
        comparison.to_csv(temp / "model_comparison.csv", index=False)
        qlib_risk.to_csv(temp / "qlib_risk_analysis.csv", index=False)
        pd.concat([candidate_dev, candidate_outer], ignore_index=True).to_csv(
            temp / "portfolio_report.csv", index=False
        )
        pd.concat([shadow_dev, shadow_outer], ignore_index=True).to_csv(
            temp / "shadow_overlay_report.csv", index=False
        )
        (temp / "dataset_audit.json").write_text(
            json.dumps(dataset_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (temp / "open_action_value_models.pkl").open("wb") as handle:
            pickle.dump(models, handle)
        runtime_models = {}
        for filename, model in (
            ("frozen_open_action_value_model.pkl", models[2023]),
            ("frozen_recovery_model.pkl", frozen_recovery),
        ):
            payload = pickle.dumps(model)
            (temp / filename).write_bytes(payload)
            runtime_models[filename.removesuffix(".pkl")] = {
                "artifact": f"artifacts/{spec.experiment}/{filename}",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "training_signal_end": "2022-12-31",
                "available": model is not None,
            }
        (temp / "runtime_models.json").write_text(
            json.dumps(runtime_models, indent=2) + "\n", encoding="utf-8"
        )
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_experiment=spec.source_experiment,
                model="logistic_open_execution_action_value",
                objective="execute_vs_hold_after_25bps_current_and_next_reset_cost",
                selection_uses_outer_test="false",
                execute_probability_threshold=str(
                    spec.execute_probability_threshold
                ),
                maximum_skippable_turnover=str(spec.maximum_skippable_turnover),
                mandatory_derisk_beta_delta=str(
                    spec.mandatory_derisk_beta_delta
                ),
            )
            development = comparison[
                comparison["variant"].eq("open_action_value_candidate")
                & comparison["segment"].eq("development_walk_forward")
                & comparison["cost_bps"].eq(spec.transaction_cost_bps)
            ].iloc[0]
            R.log_metrics(
                development_return_pct=float(development["return_pct"]),
                development_open_trade_days=float(
                    development["open_trade_days"]
                ),
                development_total_turnover=float(development["total_turnover"]),
                development_all_gates_pass=float(all(gates.values())),
            )
            for path in temp.iterdir():
                recorder.log_artifact(
                    path.as_posix(), artifact_path=spec.experiment
                )
            R.save_objects(
                **{
                    "pred.pkl": pd.concat(
                        [candidate_dev, candidate_outer], ignore_index=True
                    ).set_index("signal_date")["execute_probability"],
                    "label.pkl": training.set_index("signal_date")[
                        "execute_advantage"
                    ],
                }
            )
            info = {
                "recorder_id": recorder.id,
                "recorder_uri": str(recorder_uri.resolve()),
                "recorder_local_dir": str(
                    Path(recorder.get_local_dir()).resolve()
                ),
                "runtime_models": runtime_models,
            }
    _remove_qlib_code_snapshot_artifacts(Path(info["recorder_local_dir"]))
    return info


def run_stage1_open_action_value_card(
    *,
    root: Path | str | None = None,
    experiment: str = EXPERIMENT,
    source_spec: IntradayRecoveryDirectAllocationSpec | None = None,
    execute_probability_threshold: float = 0.50,
) -> dict[str, Any]:
    spec = Stage1OpenActionValueSpec(
        root=_resolve_root(root), experiment=experiment,
        execute_probability_threshold=execute_probability_threshold,
    )
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    source_summary, source_selected = _source_contract(spec)
    source_spec = source_spec or IntradayRecoveryDirectAllocationSpec(root=spec.root)
    if source_spec.root.resolve() != spec.root:
        raise ValueError("blocked: action-value source and card workspace roots differ")
    frame, source_audit = _load_source_frame(source_spec)
    teacher = _build_teacher_training_frame(
        frame, spec=source_spec, teacher_beta=source_spec.teacher_recovery_beta
    )
    baseline_dev, baseline_outer, frozen_recovery = _baseline_replays(
        frame,
        source_spec=source_spec,
        teacher=teacher,
        selected=source_selected,
        cost_bps=spec.transaction_cost_bps,
    )
    development_market = frame[
        frame["segment"].eq("development_walk_forward")
    ].copy()
    outer_market = frame[frame["segment"].eq("outer_diagnostic")].copy()
    training = _build_training_frame(
        development_market,
        baseline_dev,
        source_spec=source_spec,
        label_cost_bps=spec.stress_cost_bps,
    )
    dataset, dataset_audit = _build_qlib_dataset(training, spec, evaluation_frame=frame)
    models, diagnostics = _fit_models(
        dataset, spec, training_cutoffs=dataset_audit["training_cutoffs"]
    )
    shadow_dev = _simulate_router(
        development_market,
        baseline_dev,
        models=models,
        source_spec=source_spec,
        spec=spec,
        segment="development_walk_forward",
    )
    shadow_outer = _simulate_router(
        outer_market,
        baseline_outer,
        models=models,
        source_spec=source_spec,
        spec=spec,
        segment="outer_reused_diagnostic",
    )
    recovery_models_dev = {
        int(year): _fit_recovery_model(
            teacher, year=int(year), c_value=float(source_selected["logistic_c"])
        )
        for year in development_market["signal_date"].dt.year.unique()
    }
    candidate_dev = _simulate_on_policy_router(
        development_market,
        models=models,
        recovery_models=recovery_models_dev,
        source_spec=source_spec,
        spec=spec,
        selected_model=source_selected,
        segment="development_walk_forward",
    )
    candidate_outer = _simulate_on_policy_router(
        outer_market,
        models=models,
        recovery_models={year: frozen_recovery for year in source_spec.outer_years},
        source_spec=source_spec,
        spec=spec,
        selected_model=source_selected,
        segment="outer_reused_diagnostic",
    )
    fallback_dev = _simulate_on_policy_router(
        development_market,
        models={},
        recovery_models=recovery_models_dev,
        source_spec=source_spec,
        spec=spec,
        selected_model=source_selected,
        segment="development_walk_forward",
    )
    comparison_rows: list[dict[str, Any]] = []
    for segment, baseline, candidate in (
        ("development_walk_forward", baseline_dev, candidate_dev),
        ("outer_reused_diagnostic", baseline_outer, candidate_outer),
    ):
        for cost_bps in (spec.transaction_cost_bps, spec.stress_cost_bps):
            for variant, replay in (
                ("confidence_veto_baseline", baseline),
                ("open_action_value_candidate", candidate),
            ):
                priced = (
                    replay
                    if cost_bps == spec.transaction_cost_bps
                    else _stress(replay, cost_bps=cost_bps)
                )
                comparison_rows.append(
                    _metrics(
                        priced,
                        variant=variant,
                        segment=segment,
                        cost_bps=cost_bps,
                    )
                )
    comparison = pd.DataFrame(comparison_rows)
    from qlib.contrib.evaluate import risk_analysis

    qlib_risk_rows: list[pd.DataFrame] = []
    for segment, baseline, candidate in (
        ("development_walk_forward", baseline_dev, candidate_dev),
        ("outer_reused_diagnostic", baseline_outer, candidate_outer),
    ):
        for variant, replay in (
            ("confidence_veto_baseline", baseline),
            ("open_action_value_candidate", candidate),
        ):
            for cost_bps in (spec.transaction_cost_bps, spec.stress_cost_bps):
                returns = (
                    replay["strategy_return"] if cost_bps == spec.transaction_cost_bps
                    else _stress(replay, cost_bps=cost_bps)["strategy_return"]
                )
                table = risk_analysis(
                    pd.Series(
                        returns.to_numpy(),
                        index=pd.to_datetime(replay["trade_date"]),
                    ),
                    freq="day",
                ).reset_index()
                table["segment"] = segment
                table["variant"] = variant
                table["cost_bps"] = float(cost_bps)
                qlib_risk_rows.append(table)
    qlib_risk = pd.concat(qlib_risk_rows, ignore_index=True)
    annual = pd.concat(
        [
            _annual_metrics(
                baseline_dev,
                candidate_dev,
                segment="development_walk_forward",
            ),
            _annual_metrics(
                baseline_outer,
                candidate_outer,
                segment="outer_reused_diagnostic",
            ),
        ],
        ignore_index=True,
    )
    lookup = comparison.set_index(["segment", "variant", "cost_bps"])
    baseline_10 = lookup.loc[
        ("development_walk_forward", "confidence_veto_baseline", 10.0)
    ].to_dict()
    baseline_25 = lookup.loc[
        ("development_walk_forward", "confidence_veto_baseline", 25.0)
    ].to_dict()
    candidate_10 = lookup.loc[
        ("development_walk_forward", "open_action_value_candidate", 10.0)
    ].to_dict()
    candidate_25 = lookup.loc[
        ("development_walk_forward", "open_action_value_candidate", 25.0)
    ].to_dict()
    gates = _promotion_gates(
        baseline_10=baseline_10,
        baseline_25=baseline_25,
        candidate_10=candidate_10,
        candidate_25=candidate_25,
        annual=annual,
        fallback=fallback_dev,
        baseline_replay=baseline_dev,
        spec=spec,
    )
    development_pass = bool(all(gates.values()))
    decision = (
        "validated_research_candidate_for_clean_forward_shadow"
        if development_pass
        else "not_promoted_open_action_value_development_gates_failed"
    )
    recorder = _record(
        spec,
        training=training,
        diagnostics=diagnostics,
        comparison=comparison,
        qlib_risk=qlib_risk,
        candidate_dev=candidate_dev,
        candidate_outer=candidate_outer,
        shadow_dev=shadow_dev,
        shadow_outer=shadow_outer,
        models=models,
        frozen_recovery=frozen_recovery,
        dataset_audit=dataset_audit,
        gates=gates,
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": "stage1_open_execution_action_value_research_card",
        "decision": decision,
        "production_change": "blocked_pending_clean_forward_shadow",
        "selection_uses_outer_test": False,
        "outer_role": "reused_diagnostic_only_not_model_selection",
        "development_gate_pass": development_pass,
        "accounting_contract": "fees_debited_from_cash_at_each_execution_then_net_nav_weights",
        "cost_stress_contract": "fixed_target_hold_recovery_events_repriced_at_each_stage_net_nav",
        "risk_comparison_roundoff_tolerance_pct_points": 1e-10,
        "source_stage1": {
            "experiment": spec.source_experiment,
            "recorder_id": source_summary["recorder_id"],
            "recorder_role": "fixed_parameter_selection_provenance_only",
            "selected_model": source_selected,
            "allocation_source_experiment": source_spec.source_experiment,
            "allocation_source_recorder_id": source_audit["source_recorder_id"],
            "training_label_state": "confidence_veto_baseline_counterfactual_branches",
            "validation_controller_state": "own_on_policy_equity_peak_history_and_risk_clock",
            "execution_holdings_state": "actual_marked_holdings_after_router_decisions",
            "outer_recovery_training": "frozen_at_first_2023_decision_no_outer_labels",
        },
        "hypothesis": (
            "A point-in-time action-value router can skip only small non-derisk "
            "open trades whose execute-versus-hold value is predicted negative"
        ),
        "selected_model": {
            "model": "balanced_logistic_regression",
            "logistic_c": spec.logistic_c,
            "execute_probability_threshold": spec.execute_probability_threshold,
            "maximum_skippable_turnover": spec.maximum_skippable_turnover,
            "mandatory_derisk_beta_delta": spec.mandatory_derisk_beta_delta,
            "label_cost_bps": spec.stress_cost_bps,
            "candidate_count": 1,
            "selection_note": (
                "single structurally fixed candidate after exploratory development diagnostics; "
                "requires clean forward shadow before production"
            ),
        },
        "development_baseline": baseline_10,
        "development_candidate": candidate_10,
        "development_candidate_25bps": candidate_25,
        "development_gates": gates,
        "outer_diagnostic": comparison[
            comparison["segment"].eq("outer_reused_diagnostic")
        ].to_dict(orient="records"),
        "frequency_change": {
            "open_trade_day_reduction_pct": float(
                100.0
                * (
                    1.0
                    - candidate_10["open_trade_days"]
                    / baseline_10["open_trade_days"]
                )
            ),
            "turnover_reduction_pct": float(
                100.0
                * (
                    1.0
                    - candidate_10["total_turnover"]
                    / baseline_10["total_turnover"]
                )
            ),
            "development_skipped_open_targets": int(
                candidate_dev["skipped_open_target"].sum()
            ),
            "outer_skipped_open_targets": int(
                candidate_outer["skipped_open_target"].sum()
            ),
        },
        "dataset_audit": dataset_audit,
        "recovery_training_cutoffs": teacher.attrs.get("recovery_training_cutoffs", {}),
        "source_audit": source_audit,
        "qlib_api": [
            "DatasetH",
            "DataHandlerLP",
            "StaticDataLoader",
            "qlib.workflow.R",
            "R.save_objects",
        ],
        "qlib_docs_reference": [
            "https://qlib.readthedocs.io/en/latest/component/data.html",
            "https://qlib.readthedocs.io/en/stable/component/workflow.html",
            "https://qlib.readthedocs.io/en/stable/component/recorder.html",
        ],
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                "artifacts/pred.pkl",
                "artifacts/label.pkl",
                f"artifacts/{spec.experiment}/action_value_training_frame.csv",
                f"artifacts/{spec.experiment}/training_diagnostics.csv",
                f"artifacts/{spec.experiment}/model_comparison.csv",
                f"artifacts/{spec.experiment}/portfolio_report.csv",
                f"artifacts/{spec.experiment}/dataset_audit.json",
                f"artifacts/{spec.experiment}/open_action_value_models.pkl",
                f"artifacts/{spec.experiment}/shadow_overlay_report.csv",
                f"artifacts/{spec.experiment}/frozen_open_action_value_model.pkl",
                f"artifacts/{spec.experiment}/frozen_recovery_model.pkl",
                f"artifacts/{spec.experiment}/runtime_models.json",
            ],
        },
        **recorder,
    }
    _write_card(
        spec,
        summary=summary,
        comparison=comparison,
        annual=annual,
        diagnostics=diagnostics,
        qlib_risk=qlib_risk,
    )
    return summary


def _write_card(
    spec: Stage1OpenActionValueSpec,
    *,
    summary: dict[str, Any],
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    diagnostics: pd.DataFrame,
    qlib_risk: pd.DataFrame,
) -> None:
    card = spec.card_dir
    (card / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (card / "selected_model.json").write_text(
        json.dumps(summary["selected_model"], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    comparison.to_csv(card / "model_selection.csv", index=False)
    comparison.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    qlib_risk.to_csv(card / "risk_analysis.csv", index=False)
    comparison[
        [
            "segment",
            "variant",
            "cost_bps",
            "return_pct",
            "total_turnover",
            "open_trade_days",
            "recovery_action_count",
        ]
    ].to_csv(card / "execution_metrics.csv", index=False)
    annual.to_csv(card / "annual_metrics.csv", index=False)
    diagnostics.to_csv(card / "training_summary.csv", index=False)
    (card / "analysis_summary.json").write_text(
        json.dumps(
            {
                "decision": summary["decision"],
                "development_gate_pass": summary["development_gate_pass"],
                "development_gates": summary["development_gates"],
                "frequency_change": summary["frequency_change"],
                "outer_role": summary["outer_role"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    failed = [
        name for name, passed in summary["development_gates"].items() if not passed
    ]
    (card / "failure_modes.md").write_text(
        "# Failure modes\n\n"
        + (
            "- 开发门全部通过。\n"
            if not failed
            else "".join(f"- 未通过：{name}\n" for name in failed)
        )
        + "- 该候选来自已反复使用的开发窗口，尚无 clean unseen holdout。\n"
        + "- 激活前必须核对回放与运行时使用同一自身账本状态和开盘/恢复决策函数。\n"
        + "- outer 只能作为 reused diagnostic，不能作为晋升依据。\n",
        encoding="utf-8",
    )
    workflow = f"""experiment: {spec.experiment}
source_experiment: {spec.source_experiment}
allocation_source_experiment: {summary['source_stage1']['allocation_source_experiment']}
source_recorder_id: {summary['source_stage1']['allocation_source_recorder_id']}
system_of_record: qlib
provider_uri: {summary['source_audit']['source_market_provider_uri']}
intraday_provider_uri: {summary['source_audit']['intraday_provider_uri']}
selection_segment: 2016-01-01_to_2022-12-31
selection_uses_outer_test: false
outer_role: reused_diagnostic_only_not_model_selection
validation_controller_state: own_on_policy_equity_peak_history_and_risk_clock
execution_holdings_state: actual_marked_holdings
transaction_cost_bps: {spec.transaction_cost_bps}
stress_cost_bps: {spec.stress_cost_bps}
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  recorder_only_artifacts:
""" + "".join(
        f"    - {artifact}\n"
        for artifact in summary["artifact_policy"]["recorder_only_artifacts"]
    )
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
    baseline = summary["development_baseline"]
    candidate = summary["development_candidate"]
    candidate_25 = summary["development_candidate_25bps"]
    baseline_25 = comparison[
        comparison["segment"].eq("development_walk_forward")
        & comparison["variant"].eq("confidence_veto_baseline")
        & comparison["cost_bps"].eq(spec.stress_cost_bps)
    ].iloc[0]
    outer = pd.DataFrame(summary["outer_diagnostic"])
    outer_10 = outer[outer["cost_bps"].eq(spec.transaction_cost_bps)].set_index(
        "variant"
    )
    conclusion = f"""# 中文结论

- 做了什么：在固定 Stage1 confidence-veto 上新增开盘 action-value 执行路由。训练标签来自baseline分支；正式验证每天用候选自身净值、峰值、历史收益和风险状态重新计算目标。执行层只对不超过5% L1 turnover、非去风险且当前持仓合法的交易判断 execute/hold；09:31不读取未来恢复信号，10:31复用当日开盘控制器快照后按首小时信号独立执行。
- 标签：用当日执行与继续持有的实现收益差，扣除25bp当日及下一日复位成本；Logistic 按年度扩展训练，2023-2026模型固定使用2022年末及以前的信号行。标签记录收益结算与次日目标均已可知的时间，只允许在实际测试开盘前已成熟的标签进入训练。
- 开发期10bp：baseline return={baseline['return_pct']:.4f}%、candidate={candidate['return_pct']:.4f}%；MaxDD为 {baseline['max_drawdown_pct']:.4f}% / {candidate['max_drawdown_pct']:.4f}%，candidate worst5={candidate['worst_rolling_5d_pct']:.4f}%。
- 成本压力：candidate 25bp return={candidate_25['return_pct']:.4f}%，baseline 25bp={baseline_25['return_pct']:.4f}%。正式对比见 backtest_benchmark_comparison.csv。
- 频率：开盘交易日减少 {summary['frequency_change']['open_trade_day_reduction_pct']:.2f}%，turnover减少 {summary['frequency_change']['turnover_reduction_pct']:.2f}%，10:31恢复语义没有被固定带宽误删。
- 年度稳定性：开发期逐年财富下限门={summary['development_gates']['annual_wealth_floor_pass']}；完整年度比值见 annual_metrics.csv。
- outer复用诊断：baseline={outer_10.loc['confidence_veto_baseline', 'return_pct']:.4f}%、candidate={outer_10.loc['open_action_value_candidate', 'return_pct']:.4f}%；不参与选择，也不是clean holdout。
- 决策：{summary['decision']}。开发门全部通过={summary['development_gate_pass']}；生产晋升仍 blocked，通过开发门后还须新的连续前向模拟验证。
- qlib：DatasetH/DataHandlerLP 承载年度扩展训练，预测、标签、训练帧、模型和portfolio replay均进入 recorder {summary['recorder_id']}。
- 不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")


__all__ = [
    "ACTION_FEATURES",
    "build_open_action_features",
    "decide_open_execution",
    "prepare_on_policy_open",
    "prepare_on_policy_recovery",
    "Stage1OpenActionValueSpec",
    "run_stage1_open_action_value_card",
]
