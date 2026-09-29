"""Causal annual supervision from an economic router's own portfolio path."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.backtest.position_accounting import settle_target_session
from src.workflow.stage1_economic_action import (
    ECONOMIC_FEATURES, fee_affordable_target, prepare_economic_open, prepare_economic_recovery,
)


LABEL_COST_BPS = 25.0
RIDGE_ALPHA = 10.0
MINIMUM_TRAINING_ROWS = 200


@dataclass
class _Portfolio:
    equity: float
    peak: float
    weights: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    risk_state: int = 0
    risk_clock: int = 0
    history: list[float] = field(default_factory=list)
    days_since_trade: int = 0


def _event_at(value, *, minute: int) -> pd.Timestamp:
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError("blocked: economic training requires actual execution dates")
    if date.tzinfo is not None:
        date = date.tz_convert("America/New_York").tz_localize(None)
    return pd.Timestamp(datetime.combine(date.date(), time(9, minute), tzinfo=ZoneInfo("America/New_York"))).tz_convert("UTC")


def _ordered_frame(frame: pd.DataFrame, *, allow_terminal_close: bool = False) -> pd.DataFrame:
    required = {"signal_date", "trade_date", "exit_date", "soxl_return", "soxx_return",
                "soxl_weight", "soxx_weight", "cash_weight", "tail_probability", "alpha_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"blocked: economic input frame missing {sorted(missing)}")
    result = frame.copy()
    for column in ("signal_date", "trade_date", "exit_date"):
        result[column] = pd.to_datetime(result[column])
        if result[column].isna().any():
            raise ValueError(f"blocked: incomplete economic input {column}")
    if result.signal_date.duplicated().any() or result.empty:
        raise ValueError("blocked: economic input has duplicate dates or no rows")
    result = result.sort_values("signal_date").reset_index(drop=True)
    valid_exit = result.trade_date.lt(result.exit_date)
    if allow_terminal_close and result.iloc[-1].get("exit_basis") == "terminal_close":
        valid_exit.iloc[-1] = result.iloc[-1].trade_date == result.iloc[-1].exit_date
    if not (result.signal_date.lt(result.trade_date) & valid_exit).all():
        raise ValueError("blocked: invalid economic signal/entry/exit ordering")
    if "segment" not in result:
        result["segment"] = "diagnostic"
    return result


def _model_at(models: dict[int, Any], year: int):
    eligible = [int(key) for key in models if int(key) <= year]
    return models[max(eligible)] if eligible else None


def _open(row, state: _Portfolio, *, model, source_spec, policy: str):
    return prepare_economic_open(
        row, pretrade_weights=state.weights.copy(), equity=state.equity, peak=state.peak,
        risk_state=state.risk_state, risk_clock=state.risk_clock, history=list(state.history),
        recovery_spec=source_spec, model=model, policy=policy, days_since_trade=state.days_since_trade,
        cost_bps=float(source_spec.transaction_cost_bps),
    )


def _settle_branch(row: pd.Series, *, pretrade: np.ndarray, opened: dict,
                   execute: bool, cost_bps: float, recovery_model, source_spec, selected_model) -> dict:
    source_spec = replace(source_spec, transaction_cost_bps=cost_bps)
    # A counterfactual EXECUTE still trades the candidate when the observed
    # router chose HOLD; its recorded execution_target_weights may be holdings.
    target = fee_affordable_target(opened["desired_open_weights"], pretrade_weights=pretrade, cost_bps=cost_bps) if execute else None
    open_only = settle_target_session(pretrade, open_target_weights=target,
                                     asset_returns=[0.0, 0.0, 0.0], cost_bps=cost_bps)
    branch_context = dict(opened)
    branch_context["history"] = list(opened["history"])
    branch_context["open_weights"] = open_only["open_net_weights"].copy()
    branch_context["branch_open_net_equity"] = open_only["open_net_equity"]
    early = np.array([row.get("early_ret_soxl", np.nan), row.get("early_ret_soxx", np.nan), 0.0], dtype=float)
    marked = open_only["open_net_weights"] * (1 + early)
    branch_context["pretrade_weights"] = marked / marked.sum() if np.isfinite(marked).all() and marked.sum() > 0 else None
    # Recompute each branch's recovery instead of copying the observed action.
    # The 09:31 controller snapshot remains fixed, matching the runtime contract;
    # branch-specific holdings and cash fees are preserved in accounting.
    recovered = prepare_economic_recovery(
        row, open_state=branch_context, recovery_spec=source_spec,
        model=recovery_model, selected_model=selected_model,
        pretrade_weights=branch_context["pretrade_weights"], cost_bps=cost_bps,
    )
    recovery_target = recovered["target_weights"] if recovered["recovery_action"] else None
    settled = settle_target_session(
        pretrade, open_target_weights=target,
        asset_returns=[row["soxl_return"], row["soxx_return"], 0.0], cost_bps=cost_bps,
        recovery_target_weights=recovery_target, early_returns=early if recovery_target is not None else None,
    )
    return {"settled": settled, "recovery": recovered,
            "display_weights": recovered.get("net_target_weights", recovery_target) if recovery_target is not None else (opened["desired_open_weights"] if execute else pretrade)}


def _finalize_label(sample: dict, branches: dict, *, next_row: pd.Series, next_target: np.ndarray) -> None:
    if pd.Timestamp(sample["exit_date"]).normalize() != pd.Timestamp(next_row["trade_date"]).normalize():
        sample["label_status"] = "unavailable_noncontiguous_next_reset"
        return
    reset_fees = {}
    for name, branch in branches.items():
        settled = branch["settled"]
        reset_target = fee_affordable_target(
            next_target, pretrade_weights=settled["ending_weights"], cost_bps=LABEL_COST_BPS,
        )
        reset = settle_target_session(
            settled["ending_weights"], open_target_weights=reset_target,
            asset_returns=[0.0, 0.0, 0.0], cost_bps=LABEL_COST_BPS,
        )
        reset_fees[name] = reset["fees"] * settled["net_equity"]
    execute_net = branches["execute"]["settled"]["net_return"]
    hold_net = branches["hold"]["settled"]["net_return"]
    sample.update(
        execute_advantage_bps=10_000.0 * (execute_net - hold_net - reset_fees["execute"] + reset_fees["hold"]),
        execute_reset_fee=reset_fees["execute"], hold_reset_fee=reset_fees["hold"],
        label_available_at=max(_event_at(sample["exit_date"], minute=30), _event_at(next_row["trade_date"], minute=31)),
        label_status="complete_one_session_common_next_target_reset_proxy",
    )


def build_economic_dataset(training_rows: pd.DataFrame):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    if training_rows.empty:
        raise ValueError("blocked: no own-state rows available for the economic DatasetH")
    ordered = training_rows.sort_values("signal_date").copy()
    static = pd.concat({
        "feature": ordered.loc[:, list(ECONOMIC_FEATURES)].astype(float),
        "label": ordered.loc[:, ["execute_advantage_bps"]].astype(float),
        "metadata": ordered.loc[:, ["label_available_at"]],
    }, axis=1)
    dates = pd.to_datetime(ordered.signal_date)
    static.index = pd.MultiIndex.from_arrays([dates, np.full(len(ordered), "ECONOMIC_OPEN")], names=["datetime", "instrument"])
    return DatasetH(
        handler=DataHandlerLP(instruments=None, start_time=dates.min(), end_time=dates.max(),
            data_loader=StaticDataLoader(config=static), infer_processors=[], learn_processors=[]),
        segments={"all": (dates.min(), dates.max())},
    )


def mature_economic_rows(dataset, *, model_year: int, first_decision_at: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp]:
    prepared = dataset.prepare("all", col_set=["feature", "label", "metadata"])
    prepared.columns = [str(column[-1]) for column in prepared.columns]
    year_boundary = pd.Timestamp(f"{model_year}-01-01", tz="America/New_York").tz_convert("UTC")
    decision = pd.Timestamp(first_decision_at)
    if decision.tzinfo is None:
        raise ValueError("blocked: model decision timestamp must be timezone-aware")
    cutoff = min(year_boundary, decision.tz_convert("UTC"))
    available = pd.to_datetime(prepared.label_available_at, utc=True)
    dates = pd.to_datetime(prepared.index.get_level_values("datetime"))
    valid = available.notna() & available.lt(cutoff) & (dates.year < model_year)
    result = prepared.loc[valid].dropna(subset=["execute_advantage_bps"])
    if not np.isfinite(result.loc[:, [*ECONOMIC_FEATURES, "execute_advantage_bps"]].to_numpy(dtype=float)).all():
        raise ValueError("blocked: nonfinite economic model training data")
    return result, cutoff


def _fit(training_rows: list[dict], *, model_year: int, decision_at: pd.Timestamp) -> tuple[Any, dict]:
    dataset = build_economic_dataset(pd.DataFrame(training_rows))
    prepared, cutoff = mature_economic_rows(dataset, model_year=model_year, first_decision_at=decision_at)
    model = None
    if len(prepared) >= MINIMUM_TRAINING_ROWS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        model.fit(prepared.loc[:, list(ECONOMIC_FEATURES)], prepared.execute_advantage_bps)
    return model, {
        "model_year": model_year, "training_rows": len(prepared), "model_available": model is not None,
        "training_signal_end": (prepared.index.get_level_values("datetime").max().date().isoformat() if len(prepared) else None),
        "label_available_at_max": (pd.to_datetime(prepared.label_available_at, utc=True).max().isoformat() if len(prepared) else None),
        "information_cutoff_exclusive": cutoff.isoformat(), "first_test_decision_at": decision_at.isoformat(),
        "ridge_alpha": RIDGE_ALPHA, "label_cost_bps": LABEL_COST_BPS,
        "minimum_training_rows": MINIMUM_TRAINING_ROWS, "training_state_source": "causal_own_policy_path",
        "dataset_class": "qlib.data.dataset.DatasetH", "selection_uses_future_year": False,
    }


def _record_day(row, *, opened: dict, branch: dict, state: _Portfolio, cost_bps: float,
                decision_days_since_trade: int, model_year: int | None) -> dict:
    settled, recovered = branch["settled"], branch["recovery"]
    weights = np.asarray(branch["display_weights"], dtype=float)
    return {
        "signal_date": row.signal_date, "trade_date": row.trade_date, "exit_date": row.exit_date, "segment": row.segment,
        "exit_basis": row.get("exit_basis", "next_open"),
        "model_year": model_year, "decision_year": int(row.signal_date.year),
        "strategy_return": settled["net_return"], "gross_return": settled["gross_return"],
        "turnover": settled["turnover"], "open_turnover": settled["open_turnover"], "fees": settled["fees"], "cost_bps": cost_bps,
        "execute_open_trade": settled["open_turnover"] > 1e-8, "execute_open_target": bool(opened["execute_open_target"]),
        "skipped_open_target": not bool(opened["execute_open_target"]), "recovery_action": bool(recovered["recovery_action"]),
        "mandatory_position_repair": bool(opened.get("hard_risk_violation", False)),
        "hard_risk_violation": bool(opened.get("hard_risk_violation", False)), "mandatory_derisk": bool(opened.get("mandatory_derisk", False)),
        "large_transition": bool(opened.get("large_transition", False)), "decision_reason": opened.get("decision_reason"),
        "predicted_net_advantage_bps": opened.get("predicted_net_advantage_bps", np.nan),
        "execute_probability": opened.get("execute_probability", np.nan),
        "risk_state": state.risk_state, "risk_clock": state.risk_clock, "days_since_trade": decision_days_since_trade,
        "ending_days_since_trade": state.days_since_trade, "equity": state.equity,
        "portfolio_drawdown": opened["drawdown"], "current_beta_cap": opened["current_beta_cap"], "hard_beta_cap": opened["hard_beta_cap"],
        "open_soxl_weight": float(opened["open_weights"][0]), "open_soxx_weight": float(opened["open_weights"][1]), "open_cash_weight": float(opened["open_weights"][2]),
        "soxl_weight": weights[0], "soxx_weight": weights[1], "cash_weight": weights[2],
        "soxl_return": row.soxl_return, "soxx_return": row.soxx_return,
        "ending_soxl_weight": state.weights[0], "ending_soxx_weight": state.weights[1], "ending_cash_weight": state.weights[2],
    }


def _run(frame, *, models, recovery_models, source_spec, selected_model, policy,
         cost_bps, startcash, reset_on_segment, update_through_year):
    frame = _ordered_frame(frame, allow_terminal_close=update_through_year is None)
    if startcash <= 0 or policy not in {"net_value", "rules_only"}:
        raise ValueError("blocked: invalid economic simulation contract")
    source_spec = replace(source_spec, transaction_cost_bps=float(cost_bps))
    state = _Portfolio(float(startcash), float(startcash))
    rows, training_rows, diagnostics = [], [], []
    models = dict(models or {})
    fitted_years: dict[int, int | None] = {}
    pending = None
    previous_segment = None
    last_year = None
    last_fit_year = None
    last_model = None
    for _, row in frame.iterrows():
        if reset_on_segment and previous_segment is not None and row.segment != previous_segment:
            if pending is not None:
                pending[0]["label_status"] = "unavailable_segment_reset_boundary"
            pending = None
            state = _Portfolio(float(startcash), float(startcash))
        year = int(row.signal_date.year)
        if update_through_year is not None and year != last_year:
            if year >= 2017 and year <= update_through_year + 1 and training_rows:
                last_model, diagnostic = _fit(training_rows, model_year=year, decision_at=_event_at(row.trade_date, minute=31))
                diagnostics.append(diagnostic)
                last_fit_year = year
            models[year] = last_model
            fitted_years[year] = last_fit_year
        model = _model_at(models, year)
        model_fit_year = fitted_years.get(year)
        if update_through_year is None and model is not None:
            model_fit_year = min(int(key) for key, value in models.items() if value is model)
        decision_age = state.days_since_trade
        opened = _open(row, state, model=model, source_spec=source_spec, policy=policy)
        label_opened = (
            _open(row, state, model=model, source_spec=replace(source_spec, transaction_cost_bps=LABEL_COST_BPS), policy=policy)
            if update_through_year is not None else None
        )
        if pending is not None:
            _finalize_label(pending[0], pending[1], next_row=row, next_target=label_opened["desired_open_weights"])
        recovery_model = _model_at(recovery_models, year)
        if update_through_year is not None:
            branches = {
                name: _settle_branch(row, pretrade=state.weights.copy(), opened=label_opened, execute=execute,
                    cost_bps=LABEL_COST_BPS, recovery_model=recovery_model, source_spec=source_spec, selected_model=selected_model)
                for name, execute in (("execute", True), ("hold", False))
            }
            sample = {
                "signal_date": row.signal_date, "trade_date": row.trade_date, "exit_date": row.exit_date, "year": year,
                "segment": row.segment, **opened["features"], "decision_at": _event_at(row.trade_date, minute=31),
                "execute_advantage_bps": np.nan, "label_available_at": pd.NaT, "label_status": "pending_next_reset_target",
                "execute_net_return": branches["execute"]["settled"]["net_return"],
                "hold_net_return": branches["hold"]["settled"]["net_return"],
                "execute_recovery_action": bool(branches["execute"]["recovery"]["recovery_action"]),
                "hold_recovery_action": bool(branches["hold"]["recovery"]["recovery_action"]),
                "label_target_cost_bps": LABEL_COST_BPS,
                "label_target_soxl_weight": float(label_opened["desired_open_weights"][0]),
                "label_target_soxx_weight": float(label_opened["desired_open_weights"][1]),
                "label_target_cash_weight": float(label_opened["desired_open_weights"][2]),
            }
            training_rows.append(sample)
            pending = sample, branches
        branch = _settle_branch(row, pretrade=state.weights.copy(), opened=opened, execute=bool(opened["execute_open_target"]),
            cost_bps=cost_bps, recovery_model=recovery_model, source_spec=source_spec, selected_model=selected_model)
        settled = branch["settled"]
        state.risk_state, state.risk_clock = opened["risk_state"], opened["risk_clock"]
        state.weights = settled["ending_weights"].copy()
        state.history.append(float(settled["net_return"]))
        state.equity *= settled["net_equity"]
        state.peak = max(state.peak, state.equity)
        state.days_since_trade = 0 if settled["turnover"] > 1e-8 else state.days_since_trade + 1
        rows.append(_record_day(row, opened=opened, branch=branch, state=state, cost_bps=cost_bps,
                               decision_days_since_trade=decision_age, model_year=model_fit_year))
        previous_segment, last_year = row.segment, year
    return {"replay": pd.DataFrame(rows), "models": models, "training_rows": pd.DataFrame(training_rows),
            "diagnostics": pd.DataFrame(diagnostics), "final_model": last_model, "final_model_year": last_fit_year,
            "dataset_audit": {"dataset_class": "qlib.data.dataset.DatasetH", "handler_class": "qlib.data.dataset.handler.DataHandlerLP",
                "loader_class": "qlib.data.dataset.loader.StaticDataLoader", "features": list(ECONOMIC_FEATURES),
                "training_state": "causal_own_policy_path", "evaluation_segment_reset": reset_on_segment,
                "training_history_reset_on_segment": False, "label_cost_bps": LABEL_COST_BPS, "ridge_alpha": RIDGE_ALPHA,
                "label_proxy": "one session execute-versus-hold net value plus common next own-policy desired-target reset cost; not a multiday policy rollout",
                "maturity": "label_available_at strictly before both January 1 New York of fit year and first fit-year decision",
                "recovery_branches": "separately recomputed from each branch's marked holdings; same frozen open controller snapshot",
                "maximum_training_signal_year": update_through_year, "fit_year_by_decision_year": fitted_years}}


def train_causal_economic_models(
    frame: pd.DataFrame, *, recovery_models: dict[int, Any], source_spec, selected_model: dict,
    update_through_year: int = 2022, cost_bps: float = 10.0, reset_on_segment: bool = True, startcash: float = 1.0,
) -> dict[str, Any]:
    if update_through_year not in {2022, 2025}:
        raise ValueError("blocked: only preregistered 2022 or 2025 information windows are permitted")
    return _run(frame, models=None, recovery_models=recovery_models, source_spec=source_spec, selected_model=selected_model,
                policy="net_value", cost_bps=cost_bps, startcash=startcash, reset_on_segment=reset_on_segment,
                update_through_year=update_through_year)


def simulate_economic_policy(
    frame: pd.DataFrame, *, models: dict[int, Any], recovery_models: dict[int, Any], source_spec,
    selected_model: dict, policy: str = "net_value", cost_bps: float = 10.0, startcash: float = 1.0,
    reset_on_segment: bool = True,
) -> pd.DataFrame:
    return _run(frame, models=models, recovery_models=recovery_models, source_spec=source_spec, selected_model=selected_model,
                policy=policy, cost_bps=cost_bps, startcash=startcash, reset_on_segment=reset_on_segment,
                update_through_year=None)["replay"]
