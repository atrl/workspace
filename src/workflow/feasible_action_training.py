"""Feasible-action training domain: thin wrapper over the sealed economic trainer.

The sealed module ``stage1_economic_training`` (E) is NOT modified. This wrapper
re-reads E's collected causal training rows and re-fits the annual Ridge models on
a decision-time legality domain:

- ``domain="all"``      : every matured row; must reproduce the frozen control.
- ``domain="feasible"`` : only rows whose decision-time holdings are legal
  (``routing_eligible = not hard_risk_violation``), so HOLD is a real choice.

The mask is a known time-point condition on decision-time holdings. It reads only
``pretrade_*_weight`` and ``hard_beta_cap``, never the label or any future return.
Frozen hyper-parameters (Ridge alpha, minimum rows, positive-net-value execute
rule, hard-repair rule) are unchanged; no new parameter search.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.workflow import stage1_economic_training as E
from src.workflow.stage1_economic_training import (
    ECONOMIC_FEATURES,
    MINIMUM_TRAINING_ROWS,
    RIDGE_ALPHA,
    build_economic_dataset,
    mature_economic_rows,
    simulate_economic_policy,
    train_causal_economic_models,
)

DOMAINS = ("all", "feasible")
FEASIBILITY_TOLERANCE = 1e-9
PRETRADE_COLUMNS = ("pretrade_soxl_weight", "pretrade_soxx_weight", "pretrade_cash_weight")


def feasibility_mask(rows: pd.DataFrame, *, maximum_soxl_weight: float, minimum_cash_weight: float,
                     tolerance: float = FEASIBILITY_TOLERANCE) -> tuple[np.ndarray, dict]:
    """Decision-time legality mask; independent of labels and future returns."""
    pretrade = rows.loc[:, list(PRETRADE_COLUMNS)].to_numpy(float)
    if not np.isfinite(pretrade).all():
        raise ValueError("blocked: nonfinite pretrade holdings in training rows")
    beta = 3.0 * pretrade[:, 0] + pretrade[:, 1]
    hard_beta_cap = rows["hard_beta_cap"].to_numpy(float)
    if not np.isfinite(hard_beta_cap).all():
        raise ValueError("blocked: nonfinite hard_beta_cap in training rows")
    reasons = {
        "long_only": pretrade.min(axis=1) < -tolerance,
        "maximum_soxl_weight": pretrade[:, 0] > maximum_soxl_weight + tolerance,
        "minimum_cash_weight": pretrade[:, 2] < minimum_cash_weight - tolerance,
        "hard_beta_cap": beta > hard_beta_cap + tolerance,
    }
    violations = np.zeros(len(rows), dtype=bool)
    for value in reasons.values():
        violations |= value
    return ~violations, reasons


def _first_decision_by_year(frame: pd.DataFrame) -> dict[int, pd.Timestamp]:
    dates = pd.to_datetime(frame["trade_date"])
    decisions = {}
    for year, group in frame.assign(_td=dates).groupby(dates.dt.year):
        first = pd.Timestamp(group["_td"].iloc[0])
        decisions[int(year)] = E._event_at(first, minute=31)
    return decisions


def _annual_models(rows: pd.DataFrame, *, domain: str, update_through_year: int,
                   first_decision: dict[int, pd.Timestamp]) -> tuple[dict, list[dict]]:
    dataset = build_economic_dataset(rows)
    eligible = rows.set_index("signal_date")["routing_eligible"]
    if eligible.index.duplicated().any():
        raise ValueError("blocked: duplicate training signal dates")
    years = sorted(first_decision)
    models: dict[int, Any] = {}
    diagnostics = []
    last_model = None
    for year in years:
        if 2017 <= year <= update_through_year + 1 and year in first_decision:
            prepared, cutoff = mature_economic_rows(dataset, model_year=year, first_decision_at=first_decision[year])
            signal_dates = pd.DatetimeIndex(prepared.index.get_level_values("datetime"))
            mask = eligible.reindex(signal_dates).to_numpy()
            if domain == "all":
                mask = np.ones(len(prepared), dtype=bool)
            elif np.isnan(mask.astype(float)).any():
                raise ValueError(f"blocked: missing feasibility for a matured row in {year}")
            selected = prepared.loc[mask]
            model = None
            if len(selected) >= MINIMUM_TRAINING_ROWS:
                model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
                model.fit(selected.loc[:, list(ECONOMIC_FEATURES)], selected["execute_advantage_bps"])
            last_model = model
            diagnostics.append({"model_year": year, "domain": domain, "matured_rows": int(len(prepared)),
                                "domain_rows": int(len(selected)), "model_available": model is not None,
                                "fallback": model is None,
                                "information_cutoff_exclusive": cutoff.isoformat(),
                                "minimum_training_rows": MINIMUM_TRAINING_ROWS, "ridge_alpha": RIDGE_ALPHA})
        models[year] = last_model
    return models, diagnostics


def train_feasible_action_book(frame: pd.DataFrame, *, domain: str, update_through_year: int = 2022,
                               cost_bps: float = 10.0, recovery_models: dict, source_spec,
                               selected_model: dict) -> dict:
    """Train the annual economic models on a decision-time domain and replay them."""
    if domain not in DOMAINS:
        raise ValueError(f"blocked: unknown training domain {domain!r}")
    base = train_causal_economic_models(frame, recovery_models=recovery_models, source_spec=source_spec,
                                        selected_model=selected_model, update_through_year=update_through_year,
                                        cost_bps=cost_bps)
    rows = base["training_rows"].copy()
    feasible, reasons = feasibility_mask(rows, maximum_soxl_weight=source_spec.maximum_soxl_weight,
                                         minimum_cash_weight=source_spec.minimum_cash_weight)
    rows["routing_eligible"] = feasible
    first_decision = _first_decision_by_year(frame)
    models, diagnostics = _annual_models(rows, domain=domain, update_through_year=update_through_year,
                                         first_decision=first_decision)
    replay = simulate_economic_policy(frame, models=models, recovery_models=recovery_models,
                                      source_spec=source_spec, selected_model=selected_model,
                                      policy="net_value", cost_bps=cost_bps)
    return {"domain": domain, "models": models, "training_rows": rows, "diagnostics": diagnostics,
            "feasibility_reasons": {name: int(value.sum()) for name, value in reasons.items()},
            "feasible_rows": int(feasible.sum()), "infeasible_rows": int((~feasible).sum()),
            "replay": replay, "base_models": base["models"], "base_replay": base["replay"],
            "base_training_rows": base["training_rows"], "base_diagnostics": base["diagnostics"]}
