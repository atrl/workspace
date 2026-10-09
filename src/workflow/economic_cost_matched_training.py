"""Matched-cost annual economic training thin wrapper.

The sealed ``stage1_economic_training`` module bakes a single global label-reset
fee (``LABEL_COST_BPS = 25``) into its label construction while the evaluation
book can run at a different cost. This wrapper lets a caller pass the label-reset
fee and the evaluation fee explicitly, so a 10bp book can be supervised with a
10bp label and a 25bp book with a 25bp label.

It reuses the sealed *pure* helpers (feature prep, settlement, Dataset/fit) and
never edits that module or monkeypatches its global. With
``label_cost_bps = 25`` and the evaluation cost unchanged it reproduces the
sealed control exactly; that parity is asserted by the tests.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.position_accounting import settle_target_session
from src.workflow.stage1_economic_action import (
    ECONOMIC_FEATURES,
    fee_affordable_target,
)
from src.workflow import stage1_economic_training as E

LABEL_PROXY = ("one session execute-versus-hold net value plus common next own-policy "
               "desired-target reset cost at the declared label fee; not a multiday rollout")


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _finalize_label_cost(sample: dict, branches: dict, *, next_row: pd.Series,
                         next_target: np.ndarray, label_cost_bps: float) -> None:
    """Settle the shared next-target reset at the declared label fee."""
    if pd.Timestamp(sample["exit_date"]).normalize() != pd.Timestamp(next_row["trade_date"]).normalize():
        sample["label_status"] = "unavailable_noncontiguous_next_reset"
        return
    reset_fees = {}
    for name, branch in branches.items():
        settled = branch["settled"]
        reset_target = fee_affordable_target(
            next_target, pretrade_weights=settled["ending_weights"], cost_bps=label_cost_bps)
        reset = settle_target_session(
            settled["ending_weights"], open_target_weights=reset_target,
            asset_returns=[0.0, 0.0, 0.0], cost_bps=label_cost_bps)
        reset_fees[name] = reset["fees"] * settled["net_equity"]
    execute_net = branches["execute"]["settled"]["net_return"]
    hold_net = branches["hold"]["settled"]["net_return"]
    sample.update(
        execute_advantage_bps=10_000.0 * (execute_net - hold_net - reset_fees["execute"] + reset_fees["hold"]),
        execute_reset_fee=reset_fees["execute"], hold_reset_fee=reset_fees["hold"],
        label_available_at=max(E._event_at(sample["exit_date"], minute=30),
                               E._event_at(next_row["trade_date"], minute=31)),
        label_status="complete_one_session_common_next_target_reset_proxy",
    )


def _run_cost_book(frame: pd.DataFrame, *, recovery_models, source_spec, selected_model, policy: str,
                   evaluation_cost_bps: float, label_cost_bps: float, startcash: float,
                   reset_on_segment: bool, update_through_year: int | None) -> dict[str, Any]:
    """Replicate the sealed annual loop with an explicit label fee.

    Structure mirrors ``stage1_economic_training._run``; only the label-fee
    constant becomes a parameter. The evaluation book uses ``evaluation_cost_bps``.
    """
    _require(isinstance(label_cost_bps, (int, float)) and not isinstance(label_cost_bps, bool)
             and np.isfinite(label_cost_bps) and label_cost_bps >= 0,
             "label_cost_bps must be a finite nonnegative number")
    _require(isinstance(evaluation_cost_bps, (int, float)) and not isinstance(evaluation_cost_bps, bool)
             and np.isfinite(evaluation_cost_bps) and evaluation_cost_bps >= 0,
             "evaluation_cost_bps must be a finite nonnegative number")
    frame = E._ordered_frame(frame, allow_terminal_close=update_through_year is None)
    if startcash <= 0 or policy not in {"net_value", "rules_only"}:
        raise ValueError("blocked: invalid economic simulation contract")
    source_spec = replace(source_spec, transaction_cost_bps=float(evaluation_cost_bps))
    state = E._Portfolio(float(startcash), float(startcash))
    rows, training_rows, diagnostics = [], [], []
    models = {}
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
            state = E._Portfolio(float(startcash), float(startcash))
        year = int(row.signal_date.year)
        if update_through_year is not None and year != last_year:
            if year >= 2017 and year <= update_through_year + 1 and training_rows:
                last_model, diagnostic = E._fit(training_rows, model_year=year,
                                                decision_at=E._event_at(row.trade_date, minute=31))
                diagnostic["label_cost_bps"] = float(label_cost_bps)
                diagnostics.append(diagnostic)
                last_fit_year = year
            models[year] = last_model
            fitted_years[year] = last_fit_year
        model = E._model_at(models, year)
        model_fit_year = fitted_years.get(year)
        decision_age = state.days_since_trade
        opened = E._open(row, state, model=model, source_spec=source_spec, policy=policy)
        label_opened = (
            E._open(row, state, model=model,
                    source_spec=replace(source_spec, transaction_cost_bps=label_cost_bps), policy=policy)
            if update_through_year is not None else None
        )
        if pending is not None:
            _finalize_label_cost(pending[0], pending[1], next_row=row,
                                 next_target=label_opened["desired_open_weights"],
                                 label_cost_bps=label_cost_bps)
        recovery_model = E._model_at(recovery_models, year)
        if update_through_year is not None:
            branches = {
                name: E._settle_branch(row, pretrade=state.weights.copy(), opened=label_opened, execute=execute,
                                       cost_bps=label_cost_bps, recovery_model=recovery_model,
                                       source_spec=source_spec, selected_model=selected_model)
                for name, execute in (("execute", True), ("hold", False))
            }
            sample = {
                "signal_date": row.signal_date, "trade_date": row.trade_date, "exit_date": row.exit_date, "year": year,
                "segment": row.segment, **opened["features"], "decision_at": E._event_at(row.trade_date, minute=31),
                "execute_advantage_bps": np.nan, "label_available_at": pd.NaT,
                "label_status": "pending_next_reset_target",
                "execute_net_return": branches["execute"]["settled"]["net_return"],
                "hold_net_return": branches["hold"]["settled"]["net_return"],
                "label_target_cost_bps": float(label_cost_bps),
                "label_target_soxl_weight": float(label_opened["desired_open_weights"][0]),
                "label_target_soxx_weight": float(label_opened["desired_open_weights"][1]),
                "label_target_cash_weight": float(label_opened["desired_open_weights"][2]),
            }
            training_rows.append(sample)
            pending = sample, branches
        branch = E._settle_branch(row, pretrade=state.weights.copy(), opened=opened,
                                  execute=bool(opened["execute_open_target"]),
                                  cost_bps=evaluation_cost_bps, recovery_model=recovery_model,
                                  source_spec=source_spec, selected_model=selected_model)
        settled = branch["settled"]
        state.risk_state, state.risk_clock = opened["risk_state"], opened["risk_clock"]
        state.weights = settled["ending_weights"].copy()
        state.history.append(float(settled["net_return"]))
        state.equity *= settled["net_equity"]
        state.peak = max(state.peak, state.equity)
        state.days_since_trade = 0 if settled["turnover"] > 1e-8 else state.days_since_trade + 1
        rows.append(E._record_day(row, opened=opened, branch=branch, state=state,
                                  cost_bps=evaluation_cost_bps, decision_days_since_trade=decision_age,
                                  model_year=model_fit_year))
        previous_segment, last_year = row.segment, year
    return {
        "replay": pd.DataFrame(rows), "models": models,
        "training_rows": pd.DataFrame(training_rows), "diagnostics": pd.DataFrame(diagnostics),
        "final_model": last_model, "final_model_year": last_fit_year,
        "dataset_audit": {
            "dataset_class": "qlib.data.dataset.DatasetH",
            "handler_class": "qlib.data.dataset.handler.DataHandlerLP",
            "loader_class": "qlib.data.dataset.loader.StaticDataLoader",
            "features": list(ECONOMIC_FEATURES),
            "training_state": "causal_own_policy_path",
            "evaluation_cost_bps": float(evaluation_cost_bps),
            "label_cost_bps": float(label_cost_bps),
            "ridge_alpha": E.RIDGE_ALPHA, "minimum_training_rows": E.MINIMUM_TRAINING_ROWS,
            "label_proxy": LABEL_PROXY,
            "maturity": "label_available_at strictly before both January 1 New York of fit year and first fit-year decision",
            "maximum_training_signal_year": update_through_year,
            "fit_year_by_decision_year": fitted_years,
        },
    }


def train_cost_matched_book(
    frame: pd.DataFrame, *, recovery_models: dict[int, Any], source_spec, selected_model: dict,
    label_cost_bps: float, evaluation_cost_bps: float | None = None,
    update_through_year: int = 2022, reset_on_segment: bool = True, startcash: float = 1.0,
) -> dict[str, Any]:
    """Train one annual economic book whose label fee matches its cost book.

    ``evaluation_cost_bps`` defaults to ``label_cost_bps`` so a single book is
    internally cost-consistent; it may be set separately only to stress an
    unchanged decision path.
    """
    if update_through_year not in {2022, 2025}:
        raise ValueError("blocked: only preregistered 2022 or 2025 information windows are permitted")
    evaluation = float(label_cost_bps if evaluation_cost_bps is None else evaluation_cost_bps)
    return _run_cost_book(
        frame, recovery_models=recovery_models, source_spec=source_spec, selected_model=selected_model,
        policy="net_value", evaluation_cost_bps=evaluation, label_cost_bps=float(label_cost_bps),
        startcash=startcash, reset_on_segment=reset_on_segment, update_through_year=update_through_year)
