from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from soxl_backtest.reference import replay_soxlbacktest_reference
from src.workflow.direct_continuous_soxx_allocation import (
    DirectContinuousAllocationSpec,
    PROFILES,
    _evaluate,
    _load_frame,
)
from src.workflow.intraday_recovery_direct_allocation import (
    FEATURE_COLUMNS,
    IntradayRecoveryDirectAllocationSpec,
    _advance_state,
    _base_cap,
    _build_teacher_training_frame,
    _feature_row,
    _fit_recovery_model,
    _load_first_hour_features,
    _load_source_frame,
    _passes_risk,
    _resolve_source_recorder,
    _risk_metrics,
    _simulate_policy,
    _transform_frozen_features,
    mark_to_market_weights,
    project_legal_weights,
)
from src.workflow.pit_constituent_auxiliary_training_allocation import (
    AuxiliaryTrainingCandidate,
    PitAuxiliaryAllocationNet,
    PitAuxiliaryAllocationSpec,
    _project_legal_weights,
    _smooth_weights,
    _soxx_common_training_frame,
)


EXPERIMENT = "stage1_futu_sim_v1"
SOURCE_EXPERIMENT = "intraday_on_policy_staged_validation_v1"
SOURCE_RECORDER_ID = "f6b09330175e41719886d58ee2d11bec"
DEFAULT_NOTIONAL = 100_000.0


@dataclass(frozen=True)
class Stage1FutuSimSpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = SOURCE_EXPERIMENT
    source_recorder_id: str = SOURCE_RECORDER_ID
    account_notional: float = DEFAULT_NOTIONAL
    transaction_cost_bps: float = 10.0

    @property
    def card_dir(self) -> Path:
        return self.root / "production" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / self.source_experiment


@dataclass
class RuntimeState:
    equity: float = 1.0
    peak: float = 1.0
    previous: np.ndarray | None = None
    risk_state: int = 0
    risk_clock: int = 0
    history: list[float] | None = None

    def __post_init__(self) -> None:
        if self.previous is None:
            self.previous = np.array([0.0, 0.0, 1.0], dtype="float64")
        if self.history is None:
            self.history = []


@dataclass(frozen=True)
class PreparedTarget:
    signal_date: str
    open_weights: np.ndarray
    close_weights: np.ndarray
    recovery_probability: float | None
    recovery_action: bool
    drawdown: float
    current_beta_cap: float
    raw_effective_beta: float


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _selected_stage1(spec: Stage1FutuSimSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads((spec.source_card_dir / "summary.json").read_text(encoding="utf-8"))
    selected = json.loads((spec.source_card_dir / "selected_model.json").read_text(encoding="utf-8"))
    if summary.get("recorder_id") != spec.source_recorder_id:
        raise ValueError("blocked: production source recorder no longer matches the pinned Stage1 recorder")
    if not str(summary.get("decision", "")).startswith("promoted_research_candidate_stage1"):
        raise ValueError("blocked: pinned Stage1 source is not promoted")
    if bool(summary["selected_model"].get("stage2_enabled")):
        raise ValueError("blocked: production v1 only permits the validated Stage1 path")
    return summary, selected


def _source_artifact_dir(
    spec: Stage1FutuSimSpec,
) -> tuple[Path, dict[str, Any]]:
    stage1_summary, _ = _selected_stage1(spec)
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=spec.root)
    source_summary = json.loads(
        (recovery_spec.source_card_dir / "summary.json").read_text(encoding="utf-8")
    )
    recorder_dir, recorder_id = _resolve_source_recorder(
        recovery_spec,
        source_summary=source_summary,
    )
    if recorder_id != stage1_summary["source_audit"]["source_recorder_id"]:
        raise ValueError("blocked: Stage1 frozen source recorder drifted")
    return recorder_dir / "artifacts" / recovery_spec.source_experiment, stage1_summary


def load_frozen_inference_frame(
    *, root: Path | str | None = None,
    current_prices: dict[str, float] | None = None,
    execution_date: str | None = None,
    current_prices_as_of: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    repo_root = _resolve_root(root)
    production_spec = Stage1FutuSimSpec(root=repo_root)
    artifact_dir, stage1_summary = _source_artifact_dir(production_spec)
    return infer_frozen_allocation(
        root=repo_root, artifact_dir=artifact_dir,
        current_prices=current_prices, execution_date=execution_date,
        current_prices_as_of=current_prices_as_of,
        source_identity={
            "source_experiment": stage1_summary["source_audit"]["source_experiment"],
            "source_recorder_id": stage1_summary["source_audit"]["source_recorder_id"],
            "stage1_recorder_id": stage1_summary["recorder_id"],
        },
    )


def infer_frozen_allocation(
    *, root: Path, artifact_dir: Path, source_identity: dict[str, Any],
    current_prices: dict[str, float] | None = None,
    execution_date: str | None = None,
    current_prices_as_of: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply explicitly identified retained weights without fitting a new model."""
    repo_root = root
    selected = json.loads((artifact_dir / "selected_model.json").read_text(encoding="utf-8"))
    candidate_name = str(selected["candidate"])
    with (artifact_dir / "models" / f"{candidate_name}.pkl").open("rb") as file:
        states = pickle.load(file)
    state = states.get("outer_refit")
    if state is None:
        raise ValueError("blocked: frozen source model has no outer_refit state")

    pit_spec = PitAuxiliaryAllocationSpec(root=repo_root)
    market, market_audit, _ = _load_frame(
        DirectContinuousAllocationSpec(
            root=repo_root,
            train_start=pit_spec.train_start,
            development_years=pit_spec.development_years,
            development_end=pit_spec.development_end,
            outer=pit_spec.outer,
            purge_rows=pit_spec.purge_rows,
        ),
        include_incomplete_latest=True,
    )
    market = market.loc[market.index >= pd.Timestamp(pit_spec.outer[0])].copy()
    common = _soxx_common_training_frame(
        pit_spec,
        end=market.index.max().date().isoformat(),
    ).reindex(market.index)
    primary_features = list(state["primary_features"])
    common_features = list(state["common_features"])
    primary_x = _transform_frozen_features(
        market,
        features=primary_features,
        scaler=state["primary_scaler"],
    )
    common_x = _transform_frozen_features(
        common,
        features=common_features,
        scaler=state["common_scaler"],
    )
    model = PitAuxiliaryAllocationNet(
        primary_feature_count=len(primary_features),
        common_feature_count=len(common_features),
        hidden_units=pit_spec.hidden_units,
        tail_hidden_units=pit_spec.tail_hidden_units,
    )
    model.load_state_dict(
        {name: torch.tensor(values) for name, values in state["model_state"].items()}
    )
    model.eval()
    with torch.no_grad():
        logits, tail_logit, alpha_logit = model(primary_x, common_x)
        weights = _project_legal_weights(
            logits,
            maximum_soxl_weight=pit_spec.maximum_soxl_weight,
            minimum_cash_weight=pit_spec.minimum_cash_weight,
        ).cpu().numpy()
        tail_probability = torch.sigmoid(tail_logit).cpu().numpy()
        alpha_probability = torch.sigmoid(alpha_logit).cpu().numpy()
    predictions = pd.DataFrame(
        weights,
        index=market.index,
        columns=["soxl_weight", "soxx_weight", "cash_weight"],
    )
    predictions["tail_probability"] = tail_probability
    predictions["alpha_probability"] = alpha_probability
    candidate = AuxiliaryTrainingCandidate(**state["candidate"])
    predictions, market, smoothing_audit = _smooth_frozen_allocation(
        predictions, market=market, spec=pit_spec, candidate=candidate,
        execution_provider_uri=Path(market_audit["execution_provider_uri"]),
        current_prices=current_prices, execution_date=execution_date,
        current_prices_as_of=current_prices_as_of,
    )
    frame = market.join(predictions, how="inner")
    frame["soxl_return"] = frame["soxl_forward_open_return"]
    frame["soxx_return"] = frame["soxx_forward_open_return"]
    frame["segment"] = "outer_diagnostic"
    frame.index.name = "signal_date"
    audit = {
        **source_identity,
        "source_model_path": str(artifact_dir / "models" / f"{candidate_name}.pkl"),
        "source_model_frozen": True,
        "source_training_end": pit_spec.development_end,
        "latest_signal_date": frame.index.max().date().isoformat(),
        "latest_complete_signal_date": frame.dropna(
            subset=["soxl_forward_open_return", "soxx_forward_open_return"]
        ).index.max().date().isoformat(),
        "market_audit": market_audit,
        "primary_features": primary_features,
        "common_features": common_features,
        "smoothing": smoothing_audit,
    }
    return frame, audit


def _smooth_frozen_allocation(
    predictions: pd.DataFrame,
    *, market: pd.DataFrame, spec: PitAuxiliaryAllocationSpec,
    candidate: AuxiliaryTrainingCandidate, execution_provider_uri: Path,
    current_prices: dict[str, float] | None = None,
    execution_date: str | None = None,
    current_prices_as_of: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Replay marked holdings using observed returns; preserve unknown tail labels."""
    columns = ["soxl_forward_open_return", "soxx_forward_open_return"]
    market = market.copy()
    returns = market.loc[predictions.index, columns].astype(float).copy()
    missing_before_last = returns.iloc[:-1].isna().any(axis=1)
    missing_dates = list(missing_before_last.index[missing_before_last])
    audit: dict[str, Any] = {"status": "ready", "current_execution_marks": [], "pending_signal_dates": []}
    if missing_dates and current_prices is not None:
        if len(missing_dates) != 1 or missing_dates[0] != predictions.index[-2]:
            raise ValueError("blocked: frozen smoothing has an unresolved historical return gap")
        if execution_date is None or current_prices_as_of is None:
            raise ValueError("blocked: current execution prices require an observed same-session timestamp")
        observed = pd.Timestamp(current_prices_as_of)
        if observed.tzinfo is None:
            raise ValueError("blocked: current execution price timestamp must include a timezone")
        observed = observed.tz_convert("America/New_York")
        observed_minutes = observed.hour * 60 + observed.minute
        if observed.date().isoformat() != execution_date or not 570 <= observed_minutes < 960:
            raise ValueError("blocked: current execution price mark is stale or outside the regular session")
        if set(current_prices) != {"SOXL", "SOXX"} or any(
            not np.isfinite(float(value)) or float(value) <= 0 for value in current_prices.values()
        ):
            raise ValueError("blocked: invalid current execution price mark")
        signal_date = missing_dates[0]
        entry_date = pd.Timestamp(market.loc[signal_date, "trade_date"])
        if pd.isna(entry_date) or entry_date != predictions.index[-1] or pd.Timestamp(execution_date) <= entry_date:
            raise ValueError("blocked: execution mark does not settle the preceding allocation interval")
        recorded_exit = pd.Timestamp(market.loc[signal_date, "exit_date"])
        if pd.notna(recorded_exit) and recorded_exit != pd.Timestamp(execution_date):
            raise ValueError("blocked: execution price mark date differs from the pending interval exit")
        entry = _qlib_open_prices(provider_uri=execution_provider_uri, session_date=entry_date.date().isoformat())
        for symbol, column in zip(("SOXL", "SOXX"), columns):
            realized = float(current_prices[symbol]) / float(entry[symbol]) - 1.0
            if pd.notna(returns.loc[signal_date, column]) and not np.isclose(
                returns.loc[signal_date, column], realized, atol=1e-8
            ):
                raise ValueError("blocked: partial historical return conflicts with the execution price mark")
            returns.loc[signal_date, column] = realized
            market.loc[signal_date, column] = realized
        market.loc[signal_date, "exit_date"] = pd.Timestamp(execution_date)
        audit["current_execution_marks"].append({
            "signal_date": pd.Timestamp(signal_date).date().isoformat(),
            "entry_date": entry_date.date().isoformat(), "exit_date": execution_date,
            "price_observed_at": observed.isoformat(), "prices": current_prices,
            "basis": "observed execution quote; not an assumed official opening print",
        })
        missing_dates = []
    elif len(missing_dates) > 1 or (missing_dates and missing_dates[0] != predictions.index[-2]):
        raise ValueError("blocked: frozen smoothing has an unresolved historical return gap")

    usable_count = len(predictions) - (1 if missing_dates else 0)
    if missing_dates:
        audit["status"] = "needs_execution_mark"
        audit["pending_signal_dates"] = [predictions.index[-1].date().isoformat()]
    usable_returns = returns.iloc[:usable_count].copy()
    unobserved_final = usable_returns.iloc[-1].isna()
    audit["unobserved_final_return_signal_date"] = (
        usable_returns.index[-1].date().isoformat() if unobserved_final.any() else None
    )
    audit["unobserved_final_return_consumed"] = False
    # _smooth_weights marks its last row only after producing the final target.
    # An internal placeholder cannot affect any target and is never written to
    # market/return evidence. Unknown final realized returns remain NaN there.
    usable_returns.iloc[-1] = usable_returns.iloc[-1].fillna(0.0)
    smoothed = _smooth_weights(
        predictions.iloc[:usable_count], spec=spec, candidate=candidate,
        asset_returns=usable_returns,
    )
    result = predictions.copy()
    weight_columns = ["soxl_weight", "soxx_weight", "cash_weight"]
    result[weight_columns] = np.nan
    result.loc[smoothed.index, weight_columns] = smoothed[weight_columns]
    return result, market, audit


@lru_cache(maxsize=2)
def _cached_stage1_simulation_inputs(
    root_text: str,
    stage1_recorder_id: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    dict[str, Any],
    IntradayRecoveryDirectAllocationSpec,
    dict[tuple[int, float], Any],
]:
    repo_root = Path(root_text).resolve()
    production_spec = Stage1FutuSimSpec(root=repo_root)
    source_summary, _ = _selected_stage1(production_spec)
    selected_model = dict(source_summary["selected_model"])
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=repo_root)
    inference, inference_audit = load_frozen_inference_frame(root=repo_root)

    complete = _with_intraday(inference, recovery_spec=recovery_spec).dropna(
        subset=["soxl_forward_open_return", "soxx_forward_open_return"]
    ).copy()
    complete["trade_date"] = pd.to_datetime(complete["trade_date"])
    complete["exit_date"] = pd.to_datetime(complete["exit_date"])
    source_frame, _ = _load_source_frame(recovery_spec)
    teacher_frame = _build_teacher_training_frame(
        source_frame,
        spec=recovery_spec,
        teacher_beta=recovery_spec.teacher_recovery_beta,
    )
    return (
        complete,
        inference,
        inference_audit,
        teacher_frame,
        selected_model,
        recovery_spec,
        {},
    )


@lru_cache(maxsize=16)
def _cached_stage1_simulation_from_start(
    root_text: str,
    stage1_recorder_id: str,
    requested_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Replay Stage1 from cash at the first trade session on/after start."""
    (
        complete,
        inference,
        inference_audit,
        teacher_frame,
        selected_model,
        recovery_spec,
        model_cache,
    ) = _cached_stage1_simulation_inputs(root_text, stage1_recorder_id)
    requested_ts = pd.Timestamp(requested_start).normalize()
    available_start = complete["trade_date"].min().normalize()
    available_end = complete["trade_date"].max().normalize()
    if requested_ts > available_end:
        raise ValueError("simulation start date is after the latest completed trade session")

    simulation_input = complete.loc[complete["trade_date"] >= requested_ts].copy()
    if simulation_input.empty:
        raise ValueError("simulation start date contains no completed Stage1 trade session")

    report = _simulate_policy(
        simulation_input.reset_index(),
        spec=recovery_spec,
        teacher_frame=teacher_frame,
        c_value=float(selected_model["logistic_c"]),
        threshold=float(selected_model["probability_threshold"]),
        action_beta=float(selected_model["action_beta"]),
        first_action_year=min(recovery_spec.outer_years),
        model_cache=model_cache,
        cost_bps=recovery_spec.transaction_cost_bps,
    )
    if report.empty:
        raise ValueError("simulation replay produced no completed rows")
    report["initial_trade"] = False
    report.loc[report.index[0], "initial_trade"] = True
    report["simulation_start_requested"] = requested_ts
    resolved_start = pd.Timestamp(report.iloc[0]["trade_date"]).normalize()
    initial_turnover = float(report.iloc[0]["turnover"])
    initial_invested_weight = float(
        report.iloc[0]["open_soxl_weight"] + report.iloc[0]["open_soxx_weight"]
    )
    if not np.isclose(initial_turnover, initial_invested_weight, atol=1e-10):
        raise ValueError("blocked: initial Stage1 build was not replayed from cash")

    audit = {
        **inference_audit,
        "production_recorder_id": stage1_recorder_id,
        "simulation_mode": "fresh_cash_replay_from_selected_trade_date",
        "requested_start": requested_ts.date().isoformat(),
        "resolved_start_trade_date": resolved_start.date().isoformat(),
        "available_trade_date_start": available_start.date().isoformat(),
        "available_trade_date_end": available_end.date().isoformat(),
        "initial_state": {"SOXL": 0.0, "SOXX": 0.0, "cash": 1.0},
        "initial_trade_recorded": True,
        "initial_turnover": initial_turnover,
    }
    return report, inference, audit


def replay_stage1_simulation_from_start(
    *,
    root: Path | str | None = None,
    start: str,
    stage1_recorder_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build an independent historical paper replay from a selected trade date.

    The account is reset to 100% cash.  The first target therefore creates an
    explicit initial-build transaction instead of inheriting pre-start holdings.
    """
    repo_root = _resolve_root(root)
    report, inference, audit = _cached_stage1_simulation_from_start(
        repo_root.as_posix(),
        str(stage1_recorder_id),
        pd.Timestamp(start).date().isoformat(),
    )
    return report.copy(), inference.copy(), dict(audit)


def _with_intraday(
    frame: pd.DataFrame,
    *,
    recovery_spec: IntradayRecoveryDirectAllocationSpec,
) -> pd.DataFrame:
    known_trade_dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
    if known_trade_dates.empty:
        return frame.copy()
    intraday = _load_first_hour_features(recovery_spec, end=known_trade_dates.max())
    result = frame.reset_index().merge(
        intraday,
        left_on="trade_date",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    return result.set_index("signal_date").sort_index()


def _prepare_target(
    row: pd.Series,
    *,
    state: RuntimeState,
    model: Any,
    recovery_spec: IntradayRecoveryDirectAllocationSpec,
    action_beta: float,
    threshold: float,
) -> PreparedTarget:
    assert state.history is not None
    drawdown = max(0.0, 1.0 - state.equity / state.peak)
    state.risk_state, state.risk_clock, q = _advance_state(
        state.risk_state,
        state.risk_clock,
        drawdown,
        recovery_spec,
    )
    cap, raw_beta = _base_cap(
        row,
        q=q,
        drawdown=drawdown,
        history=state.history,
        spec=recovery_spec,
    )
    open_weights = project_legal_weights(
        float(row["soxl_weight"]),
        float(row["soxx_weight"]),
        cap,
        minimum_cash_weight=recovery_spec.minimum_cash_weight,
        maximum_soxl_weight=recovery_spec.maximum_soxl_weight,
    )
    target_beta = min(raw_beta, action_beta)
    eligible = (
        model is not None
        and drawdown >= recovery_spec.recovery_eligibility_drawdown
        and target_beta > cap + 1e-12
        and pd.notna(row.get("early_ret_soxx"))
    )
    probability: float | None = None
    recovery = False
    if eligible:
        feature = pd.DataFrame(
            [
                _feature_row(
                    row,
                    drawdown=drawdown,
                    cap=cap,
                    raw_beta=raw_beta,
                    state=state.risk_state,
                    history=state.history,
                )
            ]
        )
        probability = float(model.predict_proba(feature.loc[:, FEATURE_COLUMNS])[0, 1])
        recovery = probability >= threshold
    close_weights = (
        project_legal_weights(
            float(row["soxl_weight"]),
            float(row["soxx_weight"]),
            target_beta,
            minimum_cash_weight=recovery_spec.minimum_cash_weight,
            maximum_soxl_weight=recovery_spec.maximum_soxl_weight,
        )
        if recovery
        else open_weights.copy()
    )
    return PreparedTarget(
        signal_date=pd.Timestamp(row.name).date().isoformat(),
        open_weights=open_weights,
        close_weights=close_weights,
        recovery_probability=probability,
        recovery_action=recovery,
        drawdown=drawdown,
        current_beta_cap=cap,
        raw_effective_beta=raw_beta,
    )


def _settle_target(
    row: pd.Series,
    prepared: PreparedTarget,
    *,
    state: RuntimeState,
    soxl_return: float,
    soxx_return: float,
    cost_bps: float,
) -> float:
    assert state.previous is not None
    assert state.history is not None
    asset_returns = np.array([soxl_return, soxx_return, 0.0], dtype="float64")
    from src.backtest.position_accounting import settle_target_session
    settlement = settle_target_session(
        state.previous, open_target_weights=prepared.open_weights,
        asset_returns=asset_returns, cost_bps=cost_bps,
        recovery_target_weights=prepared.close_weights if prepared.recovery_action else None,
        early_returns=(
            np.array([float(row["early_ret_soxl"]), float(row["early_ret_soxx"]), 0.0])
            if prepared.recovery_action else None
        ),
    )
    strategy_return = float(settlement["net_return"])
    state.history.append(strategy_return)
    state.equity *= 1.0 + strategy_return
    state.peak = max(state.peak, state.equity)
    state.previous = settlement["ending_weights"]
    return strategy_return


def replay_runtime_state(
    frame: pd.DataFrame,
    *,
    recovery_spec: IntradayRecoveryDirectAllocationSpec,
    teacher_frame: pd.DataFrame,
    selected_model: dict[str, Any],
) -> tuple[RuntimeState, pd.DataFrame]:
    state = RuntimeState()
    model_cache: dict[int, Any] = {}
    rows: list[dict[str, Any]] = []
    for signal_date, row in frame.sort_index().iterrows():
        year = int(pd.Timestamp(signal_date).year)
        if year not in model_cache:
            model_cache[year] = _fit_recovery_model(
                teacher_frame,
                year=year,
                c_value=float(selected_model["logistic_c"]),
            )
        prepared = _prepare_target(
            row,
            state=state,
            model=model_cache[year],
            recovery_spec=recovery_spec,
            action_beta=float(selected_model["action_beta"]),
            threshold=float(selected_model["probability_threshold"]),
        )
        strategy_return = _settle_target(
            row,
            prepared,
            state=state,
            soxl_return=float(row["soxl_forward_open_return"]),
            soxx_return=float(row["soxx_forward_open_return"]),
            cost_bps=recovery_spec.transaction_cost_bps,
        )
        rows.append(
            {
                "signal_date": signal_date,
                "open_soxl_weight": prepared.open_weights[0],
                "open_soxx_weight": prepared.open_weights[1],
                "open_cash_weight": prepared.open_weights[2],
                "soxl_weight": prepared.close_weights[0],
                "soxx_weight": prepared.close_weights[1],
                "cash_weight": prepared.close_weights[2],
                "recovery_probability": prepared.recovery_probability,
                "recovery_action": prepared.recovery_action,
                "strategy_return": strategy_return,
            }
        )
    return state, pd.DataFrame(rows)


def runtime_state_from_paper_ledger(
    ledger: dict[str, Any],
    *,
    recovery_spec: IntradayRecoveryDirectAllocationSpec,
) -> RuntimeState:
    """Build the live controller state from the current paper sleeve.

    Historical model replay remains useful as a parity diagnostic, but it is
    not the account state of a paper cycle that started later.  The live risk
    controller therefore uses confirmed broker fills and qlib settled marks
    from the active cycle only.
    """

    initial_equity = float(ledger.get("initial_equity", 0.0))
    equity = float(ledger.get("equity", 0.0))
    peak_equity = float(ledger.get("peak_equity", 0.0))
    if initial_equity <= 0.0 or equity <= 0.0 or peak_equity <= 0.0:
        raise ValueError("blocked: paper ledger has invalid live controller equity")

    positions = dict(ledger.get("positions", {}))
    prices = dict(ledger.get("prices", {}))
    normalized_equity = equity / initial_equity
    previous = np.array(
        [
            float(positions.get("SOXL", 0)) * float(prices.get("SOXL", 0.0)) / equity,
            float(positions.get("SOXX", 0)) * float(prices.get("SOXX", 0.0)) / equity,
            0.0,
        ],
        dtype="float64",
    )
    previous[2] = max(0.0, 1.0 - float(previous[:2].sum()))

    marks = sorted(
        [dict(row) for row in ledger.get("settled_marks", [])],
        key=lambda row: str(row.get("date", "")),
    )
    history: list[float] = []
    prior_equity = initial_equity
    for mark in marks:
        mark_equity = float(mark["equity"])
        if prior_equity > 0.0:
            history.append(mark_equity / prior_equity - 1.0)
        prior_equity = mark_equity

    risk_state = 0
    risk_clock = 0
    # The latest settled drawdown is consumed by _prepare_target below.  Older
    # marks reconstruct the controller's prior discrete state exactly once.
    for mark in marks[:-1]:
        drawdown = max(0.0, -float(mark.get("drawdown", 0.0)))
        risk_state, risk_clock, _ = _advance_state(
            risk_state,
            risk_clock,
            drawdown,
            recovery_spec,
        )

    return RuntimeState(
        equity=normalized_equity,
        peak=peak_equity / initial_equity,
        previous=previous,
        risk_state=risk_state,
        risk_clock=risk_clock,
        history=history,
    )


def build_runtime_target(
    *,
    root: Path | str | None = None,
    current_prices: dict[str, float] | None = None,
    include_current_intraday: bool = False,
    execution_date: str | None = None,
    live_controller_state: RuntimeState | None = None,
    current_prices_as_of: str | pd.Timestamp | None = None,
    open_decision_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = _resolve_root(root)
    release_path = repo_root / "production" / EXPERIMENT / "runtime_release.json"
    if release_path.exists():
        from src.workflow.stage1_runtime_release import build_release_target
        return build_release_target(
            root=repo_root, release_path=release_path, current_prices=current_prices,
            include_current_intraday=include_current_intraday,
            execution_date=execution_date, live_controller_state=live_controller_state,
            current_prices_as_of=current_prices_as_of, open_decision_context=open_decision_context,
        )
    spec = Stage1FutuSimSpec(root=repo_root)
    source_summary, _ = _selected_stage1(spec)
    selected_model = dict(source_summary["selected_model"])
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=repo_root)
    inference, audit = load_frozen_inference_frame(root=repo_root)
    if include_current_intraday:
        if execution_date is None:
            raise ValueError(
                "blocked: recovery target requires the current execution_date"
            )
        latest_signal = inference.index.max()
        inference.loc[latest_signal, "trade_date"] = pd.Timestamp(execution_date)
    inference = _with_intraday(inference, recovery_spec=recovery_spec)
    complete = inference.dropna(
        subset=["soxl_forward_open_return", "soxx_forward_open_return"]
    ).copy()
    source_frame, _ = _load_source_frame(recovery_spec)
    teacher_frame = _build_teacher_training_frame(
        source_frame,
        spec=recovery_spec,
        teacher_beta=recovery_spec.teacher_recovery_beta,
    )
    state, runtime_replay = replay_runtime_state(
        complete,
        recovery_spec=recovery_spec,
        teacher_frame=teacher_frame,
        selected_model=selected_model,
    )
    selected_replay = _simulate_policy(
        complete.reset_index(),
        spec=recovery_spec,
        teacher_frame=teacher_frame,
        c_value=float(selected_model["logistic_c"]),
        threshold=float(selected_model["probability_threshold"]),
        action_beta=float(selected_model["action_beta"]),
        first_action_year=min(recovery_spec.outer_years),
        model_cache={},
        cost_bps=recovery_spec.transaction_cost_bps,
    )
    parity_error = float(
        np.max(
            np.abs(
                runtime_replay["strategy_return"].to_numpy()
                - selected_replay["strategy_return"].to_numpy()
            )
        )
    )
    if parity_error > 1e-10:
        raise ValueError(f"blocked: runtime Stage1 replay parity error {parity_error}")

    latest_complete = complete.index.max()
    latest_signal = inference.index.max()
    pending = inference.loc[(inference.index > latest_complete) & (inference.index < latest_signal)]
    pending_settled: list[str] = []
    if live_controller_state is not None:
        state = live_controller_state
        pending_settled = ["paper_ledger_cycle"] if not pending.empty else []
    elif not pending.empty and current_prices is not None:
        if len(pending) != 1:
            raise ValueError("blocked: more than one unsettled Stage1 row requires historical open marks")
        signal_date, row = next(iter(pending.iterrows()))
        entry_date = pd.Timestamp(row["trade_date"]).date().isoformat()
        entry_open = _qlib_open_prices(
            provider_uri=Path(audit["market_audit"]["execution_provider_uri"]),
            session_date=entry_date,
        )
        year = int(pd.Timestamp(signal_date).year)
        model = _fit_recovery_model(
            teacher_frame,
            year=year,
            c_value=float(selected_model["logistic_c"]),
        )
        prepared = _prepare_target(
            row,
            state=state,
            model=model,
            recovery_spec=recovery_spec,
            action_beta=float(selected_model["action_beta"]),
            threshold=float(selected_model["probability_threshold"]),
        )
        _settle_target(
            row,
            prepared,
            state=state,
            soxl_return=float(current_prices["SOXL"]) / entry_open["SOXL"] - 1.0,
            soxx_return=float(current_prices["SOXX"]) / entry_open["SOXX"] - 1.0,
            cost_bps=recovery_spec.transaction_cost_bps,
        )
        pending_settled.append(pd.Timestamp(signal_date).date().isoformat())

    latest = inference.loc[latest_signal].copy()
    if not include_current_intraday:
        for column in latest.index:
            if str(column).startswith("early_") or str(column).startswith("log_volume_"):
                latest[column] = np.nan
    current_year_model = _fit_recovery_model(
        teacher_frame,
        year=int(latest_signal.year),
        c_value=float(selected_model["logistic_c"]),
    )
    prepared = _prepare_target(
        latest,
        state=state,
        model=current_year_model,
        recovery_spec=recovery_spec,
        action_beta=float(selected_model["action_beta"]),
        threshold=float(selected_model["probability_threshold"]),
    )
    status = "ready" if pending.empty or pending_settled else "pending_next_open_mark"
    weights = prepared.close_weights if include_current_intraday else prepared.open_weights
    return {
        "status": status,
        "slot": "recovery" if include_current_intraday else "open",
        "signal_date": latest_signal.date().isoformat(),
        "latest_complete_signal_date": latest_complete.date().isoformat(),
        "pending_signal_dates": [pd.Timestamp(value).date().isoformat() for value in pending.index],
        "pending_settled": pending_settled,
        "weights": {
            "SOXL": float(weights[0]),
            "SOXX": float(weights[1]),
            "cash": float(weights[2]),
        },
        "open_weights": {
            "SOXL": float(prepared.open_weights[0]),
            "SOXX": float(prepared.open_weights[1]),
            "cash": float(prepared.open_weights[2]),
        },
        "recovery_weights": {
            "SOXL": float(prepared.close_weights[0]),
            "SOXX": float(prepared.close_weights[1]),
            "cash": float(prepared.close_weights[2]),
        },
        "recovery_probability": prepared.recovery_probability,
        "recovery_action": prepared.recovery_action,
        "portfolio_drawdown": prepared.drawdown,
        "controller_state_source": (
            "paper_ledger_cycle" if live_controller_state is not None else "historical_model_replay"
        ),
        "current_beta_cap": prepared.current_beta_cap,
        "raw_effective_beta": prepared.raw_effective_beta,
        "runtime_replay_parity_max_abs_error": parity_error,
        "source_audit": audit,
    }


def _qlib_open_prices(*, provider_uri: Path, session_date: str) -> dict[str, float]:
    import qlib
    from qlib.constant import REG_US
    from qlib.data import D

    qlib.init(provider_uri=provider_uri.resolve().as_posix(), region=REG_US)
    frame = D.features(
        ["SOXL", "SOXX"],
        ["$open"],
        start_time=session_date,
        end_time=session_date,
        freq="day",
    )
    if frame.empty:
        raise ValueError(f"blocked: qlib pending entry open is missing for {session_date}")
    values = frame.reset_index().set_index("instrument")["$open"]
    result = {symbol: float(values.loc[symbol]) for symbol in ("SOXL", "SOXX")}
    if any(value <= 0 for value in result.values()):
        raise ValueError(f"blocked: qlib pending entry open is invalid for {session_date}")
    return result


def _legacy_cost_normalized_metrics(daily: pd.DataFrame, *, cost_bps: float) -> dict[str, float]:
    capital = pd.to_numeric(daily["capital"], errors="coerce")
    position = pd.to_numeric(daily["position"], errors="coerce").fillna(0.0)
    returns = capital.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    return _risk_metrics((returns - turnover * cost_bps / 10_000.0).to_numpy())


def build_matched_comparison(
    *, root: Path | str | None = None
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    repo_root = _resolve_root(root)
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=repo_root)
    source_summary = json.loads(
        (repo_root / "research" / SOURCE_EXPERIMENT / "summary.json").read_text(encoding="utf-8")
    )
    selected = dict(source_summary["selected_model"])
    inference, _ = load_frozen_inference_frame(root=repo_root)
    complete = _with_intraday(inference, recovery_spec=recovery_spec).dropna(
        subset=["soxl_forward_open_return", "soxx_forward_open_return"]
    )
    source_frame, _ = _load_source_frame(recovery_spec)
    teacher = _build_teacher_training_frame(
        source_frame,
        spec=recovery_spec,
        teacher_beta=recovery_spec.teacher_recovery_beta,
    )
    stage1 = _simulate_policy(
        complete.reset_index(),
        spec=recovery_spec,
        teacher_frame=teacher,
        c_value=float(selected["logistic_c"]),
        threshold=float(selected["probability_threshold"]),
        action_beta=float(selected["action_beta"]),
        first_action_year=min(recovery_spec.outer_years),
        model_cache={},
        cost_bps=recovery_spec.transaction_cost_bps,
    )
    start = pd.Timestamp(stage1["trade_date"].min()).date().isoformat()
    end = pd.Timestamp(stage1["exit_date"].max()).date().isoformat()
    legacy = replay_soxlbacktest_reference(start=start, end=end)
    stage1_metrics = _risk_metrics(stage1["strategy_return"].to_numpy())
    legacy_native = dict(legacy.metrics)
    legacy_10bp = _legacy_cost_normalized_metrics(
        legacy.daily,
        cost_bps=recovery_spec.transaction_cost_bps,
    )
    soxx_wealth = float(np.prod(1.0 + stage1["soxx_return"].to_numpy()))
    comparison = {
        "window": {"start": start, "end": end, "stage1_rows": int(len(stage1))},
        "stage1_10bps": stage1_metrics,
        "soxx": {
            "terminal_wealth": soxx_wealth,
            "return_pct": 100.0 * (soxx_wealth - 1.0),
        },
        "soxlbacktest_native_frictionless": legacy_native,
        "soxlbacktest_cost_normalized_10bps": legacy_10bp,
        "stage1_minus_soxlbacktest_10bps_return_pct_points": (
            stage1_metrics["return_pct"] - legacy_10bp["return_pct"]
        ),
        "methodology_warning": (
            "Stage1 uses prior-close signal with open/open and 10:30 execution; "
            "soxlbacktest is a legacy close-price rule engine. Date and cost are matched, "
            "but execution semantics are not identical."
        ),
    }
    return comparison, stage1, legacy.daily


def _record_artifacts(
    spec: Stage1FutuSimSpec,
    *,
    target: dict[str, Any],
    comparison: dict[str, Any],
    stage1: pd.DataFrame,
    legacy_daily: pd.DataFrame,
) -> dict[str, str]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        stage1.to_csv(tmp_dir / "portfolio_report.csv", index=False)
        legacy_daily.to_csv(tmp_dir / "soxlbacktest_daily.csv", index=False)
        (tmp_dir / "latest_target.json").write_text(
            json.dumps(target, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_experiment=spec.source_experiment,
                source_recorder_id=spec.source_recorder_id,
                execution_environment="futu_simulate_only",
                account_notional=str(spec.account_notional),
                real_broker_orders="false",
                live_controller_state_source="active_paper_cycle_ledger",
            )
            R.log_metrics(
                stage1_return_pct=comparison["stage1_10bps"]["return_pct"],
                stage1_max_drawdown_pct=comparison["stage1_10bps"]["max_drawdown_pct"],
                soxlbacktest_10bps_return_pct=comparison["soxlbacktest_cost_normalized_10bps"][
                    "return_pct"
                ],
            )
            for filename in ("portfolio_report.csv", "soxlbacktest_daily.csv", "latest_target.json"):
                recorder.log_artifact((tmp_dir / filename).as_posix(), artifact_path=spec.experiment)
            result = {
                "recorder_id": recorder.id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": Path(recorder.get_local_dir()).resolve().as_posix(),
            }
    recorder_artifacts = Path(result["recorder_local_dir"]) / "artifacts"
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        (recorder_artifacts / filename).unlink(missing_ok=True)
    return result


def run_stage1_futu_sim_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = _resolve_root(root)
    spec = Stage1FutuSimSpec(root=repo_root)
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    source_summary, _ = _selected_stage1(spec)
    target = build_runtime_target(root=repo_root)
    comparison, stage1, legacy_daily = build_matched_comparison(root=repo_root)
    risk_gate_pass = _passes_risk(comparison["stage1_10bps"])
    recorder = _record_artifacts(
        spec,
        target=target,
        comparison=comparison,
        stage1=stage1,
        legacy_daily=legacy_daily,
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": "stage1_futu_sim_production_card",
        "decision": (
            "paper_sim_enabled_waiting_next_us_session"
            if risk_gate_pass
            else "paper_sim_blocked_failed_position_risk_gates"
        ),
        "source_experiment": spec.source_experiment,
        "source_recorder_id": spec.source_recorder_id,
        "source_decision": source_summary["decision"],
        "assets": ["SOXL", "SOXX", "cash"],
        "execution": {
            "broker": "futu",
            "trading_environment": "SIMULATE",
            "real_trading_supported": False,
            "account_notional": spec.account_notional,
            "slots": ["09:31 America/New_York", "10:31 America/New_York"],
            "manual_start": {
                "supported": True,
                "entrypoint": "production/run_stage1_futu_sim.sh --slot start-now",
                "regular_session": "one-time idempotent rebalance from live quote and active paper ledger",
                "after_1030_et": "include observed first-hour recovery state",
                "outside_regular_session": "armed without order until the next regular session",
                "same_session_repeat": "no_action",
            },
            "postclose_refresh": "20:15 America/New_York",
            "order_type": "MARKET",
            "cash_is_residual": True,
            "sizing_basis": "reconciled_live_sleeve_equity",
            "cycle_manifest": "runtime/soxl/stage1_futu_sim/paper_cycle.json",
            "ledger": "active cycle ledger selected by paper_cycle.json",
            "controller_state_source": "confirmed cycle fills and qlib settled marks",
            "recovery_execution": "rebalance only when recovery_action is true; otherwise no_action",
            "backtest_position_accounting": "mark actual holdings to market before each target rebalance",
            "risk_failure_mode": "blocked",
            "historical_target_is_executable": False,
            "paper_trial": {
                "authorization": "explicit_user_requested_restart",
                "production_promotion_gate_bypassed": True,
                "scope": "Futu SIMULATE only",
                "requires_fresh_data": True,
                "requires_live_cycle_controller": True,
            },
            "notifications": {
                "event_bark": "disabled; execution and sync checks run silently",
                "daily_heartbeat": "10:05 Asia/Shanghai every calendar day with latest trade sync",
                "max_pushes_per_day": 1,
                "credentials": "runtime environment only",
            },
        },
        "latest_target": target,
        "backtest_comparison": comparison,
        "risk_gate_pass": risk_gate_pass,
        "dashboard": {
            "entrypoint": "production/run_stage1_dashboard.sh",
            "url": "http://127.0.0.1:8765/",
            "cloudflare_url": "https://soxl.atrl.me/",
            "cloudflare_status": "needs_zero_trust_free_plan_activation",
            "cloudflare_access": "required_self_hosted_application",
            "launchagents": [
                "com.soxl.stage1.dashboard",
                "com.soxl.stage1.cloudflare",
            ],
            "source": "qlib_recorder",
            "features": [
                "simulation_start_date_replay_from_cash",
                "initial_build_trade_record",
                "trade_history",
                "factor_change_trade_markers",
                "soxl_soxx_benchmark_comparison",
            ],
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/portfolio_report.csv",
                f"artifacts/{spec.experiment}/soxlbacktest_daily.csv",
                f"artifacts/{spec.experiment}/latest_target.json",
            ],
        },
        **recorder,
    }
    (spec.card_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (spec.card_dir / "selected_model.json").write_text(
        json.dumps(
            {
                "source_experiment": spec.source_experiment,
                "source_recorder_id": spec.source_recorder_id,
                "selection_uses_outer_test": False,
                "selected_model": source_summary["selected_model"],
                "stage2_enabled": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "strategy": "Stage1",
                "cost_bps": 10.0,
                **comparison["stage1_10bps"],
            },
            {
                "strategy": "soxlbacktest",
                "cost_bps": 10.0,
                **comparison["soxlbacktest_cost_normalized_10bps"],
            },
        ]
    ).to_csv(spec.card_dir / "backtest_benchmark_comparison.csv", index=False)
    (spec.card_dir / "workflow.yaml").write_text(
        "\n".join(
            [
                f"experiment: {spec.experiment}",
                f"source_experiment: {spec.source_experiment}",
                f"source_recorder_id: {spec.source_recorder_id}",
                f"decision: {summary['decision']}",
                f"risk_gate_pass: {str(risk_gate_pass).lower()}",
                "assets: [SOXL, SOXX, cash]",
                "broker:",
                "  name: futu",
                "  environment: SIMULATE",
                "  real_trading_supported: false",
                f"  account_notional: {spec.account_notional}",
                "  sizing_basis: reconciled_live_sleeve_equity",
                "  cycle_manifest: runtime/soxl/stage1_futu_sim/paper_cycle.json",
                "  ledger: active cycle ledger selected by paper_cycle.json",
                "  controller_state_source: confirmed cycle fills and qlib settled marks",
                "  recovery_execution: rebalance only when recovery_action is true; otherwise no_action",
                "  backtest_position_accounting: mark actual holdings to market before each target rebalance",
                "  risk_failure_mode: blocked",
                "  historical_target_is_executable: false",
                "paper_trial:",
                "  authorization: explicit_user_requested_restart",
                "  production_promotion_gate_bypassed: true",
                "  scope: Futu SIMULATE only",
                "  requires_fresh_data: true",
                "  requires_live_cycle_controller: true",
                "schedule:",
                "  open: 09:31 America/New_York",
                "  recovery: 10:31 America/New_York",
                "  manual_start: production/run_stage1_futu_sim.sh --slot start-now",
                "  manual_start_regular_session: one-time idempotent rebalance from live quote and active paper ledger",
                "  manual_start_after_1030_et: include observed first-hour recovery state",
                "  manual_start_outside_regular_session: armed without order until the next regular session",
                "  postclose_refresh: 20:15 America/New_York",
                "  bark_event_push: disabled; execution and sync checks run silently",
                "  bark_daily_heartbeat: 10:05 Asia/Shanghai every calendar day with latest trade sync",
                "  bark_max_pushes_per_day: 1",
                "  bark_credentials: runtime environment only",
                "dashboard:",
                "  entrypoint: production/run_stage1_dashboard.sh",
                "  url: http://127.0.0.1:8765/",
                "  cloudflare_url: https://soxl.atrl.me/",
                "  cloudflare_status: needs_zero_trust_free_plan_activation",
                "  cloudflare_access: required_self_hosted_application",
                "  launchagents: [com.soxl.stage1.dashboard, com.soxl.stage1.cloudflare]",
                "  source: qlib_recorder",
                "  simulation_start_date: fresh 100% cash replay from the selected trade session",
                "  initial_build_trade_recorded: true",
                "  factor_trade_markers: true",
                "  benchmarks: [SOXL, SOXX]",
                "artifact_policy:",
                "  policy: slim_card_recorder_first",
                "  workflow_config: workflow.yaml",
                "  recorder_only_artifacts:",
                f"    - artifacts/{spec.experiment}/portfolio_report.csv",
                f"    - artifacts/{spec.experiment}/soxlbacktest_daily.csv",
                f"    - artifacts/{spec.experiment}/latest_target.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (spec.card_dir / "conclusion.md").write_text(
        "\n".join(
            [
                "# 中文结论",
                "",
                "- 已把晋级的 Stage1 固定为生产模拟源；Stage2 仍关闭，失败的 stateful direct 模型没有进入任务。",
                f"- 同口径区间：{comparison['window']['start']} 至 {comparison['window']['end']}；Stage1 10bp return={comparison['stage1_10bps']['return_pct']:.4f}%，MaxDD={comparison['stage1_10bps']['max_drawdown_pct']:.4f}%。",
                f"- 当前生产晋升决策={summary['decision']}；risk_gate_pass={str(risk_gate_pass).lower()}。晋升门仍未通过；本次仅按用户明确要求运行独立的 Futu SIMULATE 观察周期，不代表生产晋升。",
                f"- legacy soxlbacktest 成本归一到 10bp 后 return={comparison['soxlbacktest_cost_normalized_10bps']['return_pct']:.4f}%，MaxDD={comparison['soxlbacktest_cost_normalized_10bps']['max_drawdown_pct']:.4f}%。",
                f"- card 内当前最新信号日={target['signal_date']}；目标计算状态={target['status']}；controller_state_source={target['controller_state_source']}。该完整历史目标只用于 qlib 回放一致性诊断，不可直接执行。",
                "- 实际模拟目标必须读取 active paper cycle：旧周期先平仓归档，新周期按平仓后实际现金净值重新基准化；控制器只使用本周期确认成交与 qlib 结算标记。",
                "- 数据过期、控制器来源不属于 active cycle、风险状态无法重建时一律 blocked，不再允许错误目标以 close-only 名义卖出。",
                "- 后续目标股数按已对账的实时策略净值计算，不重复使用固定 100,000 美元。现金是 SOXL/SOXX 目标后的剩余权重，不下现金订单。",
                "- 10:31 仅在 recovery_action=true 时换仓；若恢复信号未触发则明确 no_action，不因价格、净值或整股取整变化重复对齐同一目标。",
                "- 支持通过 `production/run_stage1_futu_sim.sh --slot start-now` 在任意时刻请求启动：美股常规交易时段内按实时价格和 active paper ledger 幂等接管；10:30 ET 后纳入已观察的首小时恢复状态；闭市时仅 armed，不提交订单。",
                "- 回测换手按每次决策前经价格变动漂移后的实际持仓计算，不再用上一次目标权重代替实际权重；因此交易成本与模拟执行口径一致。",
                "- 富途接口被限制为 SIMULATE；本任务没有真实账户分支。",
                "- 模拟盘的 open/recovery/postclose 执行与同步检查继续运行但不再逐次推送；每日上海时间 10:05 仅发一次 Bark，合并最近任务、模拟委托与持仓同步摘要。Bark 凭据仅从运行环境读取，不写入仓库。",
                "- Stage1 专用 dashboard 可通过 `production/run_stage1_dashboard.sh` 启动；只选择一个模拟开始日，账户从该交易日的 100% 现金状态独立回放到最新可用日。",
                "- 初始建仓必须记录为首笔交易并计入换手与成本，后续交易、因子变化诊断及 SOXL/SOXX 同窗对比均来自该独立模拟，不继承开始日前仓位。",
                "- dashboard 不突出展示 active paper cycle、账本或执行同步；本地源仅监听 127.0.0.1。`https://soxl.atrl.me/` 是预留入口，在 Zero Trust Free 方案与 Access 应用启用、验证前保持 tunnel 停止。",
                "- 因子变化只用于交易点附近的标准化诊断，属于非因果归因，不是精确 SHAP。",
                f"- qlib recorder：{recorder['recorder_id']}",
                "- 不构成买卖建议。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary
