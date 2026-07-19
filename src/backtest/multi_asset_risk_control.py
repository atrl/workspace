"""Stateful SOXL/SOXX/cash controller for asymmetric high-chop risk control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskControlSpec:
    target_vol: float = 0.35
    whipsaw_probability_threshold: float = 0.50
    breakdown_probability_threshold: float = 0.65
    recovery_probability_threshold: float = 0.40
    breakdown_soxl_cap: float = 0.15
    high_chop_soxl_cap: float = 0.25
    hard_stop_soxx_cap: float = 0.10
    drawdown_hard_stop: float = -0.08
    drawdown_soft_stop: float = -0.04
    soft_stop_risky_cap: float = 0.25
    cooldown_days: int = 5
    rebalance_threshold: float = 0.10
    risk_up_step: float = 0.20
    min_state_days: int = 3


_REQUIRED_COLUMNS = {
    "datetime",
    "soxl_return",
    "soxx_return",
    "realized_vol_20",
    "p_benign_chop",
    "p_breakout",
    "p_breakdown",
    "p_whipsaw",
    "price_break",
    "breadth_break",
    "high_chop_intensity",
}


def _desired_weights(row: pd.Series, spec: RiskControlSpec) -> tuple[str, float, float]:
    p_breakdown = float(row["p_breakdown"])
    vol_scale = float(np.clip(spec.target_vol / max(float(row["realized_vol_20"]), 1e-6), 0.15, 1.0))
    chop_scale = 1.0 - 0.50 * float(np.clip(row["high_chop_intensity"], 0.0, 1.0))
    downside_scale = (1.0 - float(np.clip(p_breakdown, 0.0, 1.0))) ** 2
    soxl = float(np.clip(vol_scale * chop_scale * downside_scale, 0.0, 1.0))

    price_break = bool(row["price_break"])
    breadth_break = bool(row["breadth_break"])
    replacement_allowed = (
        float(row["p_breakout"]) > p_breakdown and not price_break and not breadth_break
    )
    soxx = min(0.50, 0.50 * (1.0 - soxl)) if replacement_allowed else 0.0

    probability_confirmed = p_breakdown >= spec.breakdown_probability_threshold
    if probability_confirmed and price_break and breadth_break:
        return "hard_downside_gate", 0.0, min(soxx, spec.hard_stop_soxx_cap)
    if probability_confirmed and (price_break or breadth_break):
        return "breakdown", min(soxl, spec.breakdown_soxl_cap), min(soxx, spec.high_chop_soxl_cap)

    high_chop = (
        float(row["p_whipsaw"]) >= spec.whipsaw_probability_threshold
        or float(row["high_chop_intensity"]) >= spec.breakdown_probability_threshold
    )
    if high_chop:
        return "high_chop", min(soxl, spec.high_chop_soxl_cap), soxx
    return "normal", soxl, soxx


def _cap_risk_increase(
    soxl: float,
    soxx: float,
    *,
    previous_soxl: float,
    previous_soxx: float,
    max_increase: float,
) -> tuple[float, float]:
    desired_risk = soxl + soxx
    prior_risk = previous_soxl + previous_soxx
    allowed = min(desired_risk, prior_risk + max_increase)
    if desired_risk <= 0.0 or allowed >= desired_risk:
        return soxl, soxx
    scale = allowed / desired_risk
    return soxl * scale, soxx * scale


def apply_multi_asset_risk_control(
    signals: pd.DataFrame,
    *,
    spec: RiskControlSpec | None = None,
) -> pd.DataFrame:
    """Translate point-in-time path probabilities into next-open target weights."""

    policy = spec or RiskControlSpec()
    missing = sorted(_REQUIRED_COLUMNS.difference(signals.columns))
    if missing:
        raise ValueError(f"signals missing required columns: {missing}")
    if not 0.0 < policy.risk_up_step <= 1.0:
        raise ValueError("risk_up_step must be in (0, 1]")

    out = signals.copy().sort_values("datetime").reset_index(drop=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    if out["datetime"].duplicated().any():
        raise ValueError("signals contain duplicate datetimes")

    previous_soxl = 0.0
    previous_soxx = 0.0
    equity = 1.0
    equity_peak = 1.0
    last_risk_event = -10_000
    prior_state = "initial"
    state_age = 0
    drawdown_budget_active = False
    drawdown_budget_started = -10_000
    rows: list[dict[str, object]] = []

    for index, row in out.iterrows():
        # The position held before this decision earns the current session return.
        realized = previous_soxl * float(row["soxl_return"]) + previous_soxx * float(row["soxx_return"])
        equity *= 1.0 + realized
        equity_peak = max(equity_peak, equity)
        drawdown = equity / equity_peak - 1.0

        if not drawdown_budget_active and drawdown <= policy.drawdown_hard_stop:
            drawdown_budget_active = True
            drawdown_budget_started = index
        if (
            drawdown_budget_active
            and drawdown > policy.drawdown_soft_stop
            and index - drawdown_budget_started >= policy.cooldown_days
        ):
            drawdown_budget_active = False

        state, desired_soxl, desired_soxx = _desired_weights(row, policy)
        if drawdown_budget_active:
            desired_risk = min(desired_soxl + desired_soxx, policy.soft_stop_risky_cap)
            # Recovery budget uses the lower-volatility sleeve before restoring leverage.
            desired_soxx = min(desired_risk, policy.soft_stop_risky_cap)
            desired_soxl = max(0.0, desired_risk - desired_soxx)
            state = "drawdown_budget"

        de_risk_states = {"drawdown_budget", "hard_downside_gate", "breakdown", "high_chop"}
        if prior_state in de_risk_states and state not in de_risk_states and state_age < policy.min_state_days:
            state, desired_soxl, desired_soxx = prior_state, previous_soxl, previous_soxx
        is_risk_event = state in de_risk_states
        if is_risk_event:
            last_risk_event = index
        days_since_event = index - last_risk_event

        desired_risk = desired_soxl + desired_soxx
        previous_risk = previous_soxl + previous_soxx
        if desired_risk > previous_risk:
            if days_since_event <= policy.cooldown_days:
                desired_soxl, desired_soxx = _cap_risk_increase(
                    desired_soxl,
                    desired_soxx,
                    previous_soxl=previous_soxl,
                    previous_soxx=previous_soxx,
                    max_increase=0.0,
                )
                if state == "normal":
                    state = "cooldown"
            else:
                desired_soxl, desired_soxx = _cap_risk_increase(
                    desired_soxl,
                    desired_soxx,
                    previous_soxl=previous_soxl,
                    previous_soxx=previous_soxx,
                    max_increase=policy.risk_up_step,
                )
                if desired_soxl + desired_soxx < desired_risk - 1e-12 and state == "normal":
                    state = "recovery"

        state_changed = state != prior_state
        de_risking = desired_soxl + desired_soxx < previous_risk - 1e-12
        if (
            index > 0
            and not state_changed
            and not de_risking
            and abs(desired_soxl - previous_soxl) < policy.rebalance_threshold
            and abs(desired_soxx - previous_soxx) < policy.rebalance_threshold
        ):
            desired_soxl, desired_soxx = previous_soxl, previous_soxx

        desired_soxl = float(np.clip(desired_soxl, 0.0, 1.0))
        desired_soxx = float(np.clip(desired_soxx, 0.0, 1.0 - desired_soxl))
        cash = float(1.0 - desired_soxl - desired_soxx)
        state_age = state_age + 1 if state == prior_state else 1
        rows.append(
            {
                "strategy_equity_before_target": equity,
                "strategy_drawdown_before_target": drawdown,
                "realized_return_before_target": realized,
                "controller_state": state,
                "controller_state_age": state_age,
                "soxl_weight": desired_soxl,
                "soxx_weight": desired_soxx,
                "cash_weight": cash,
                "risky_weight": desired_soxl + desired_soxx,
            }
        )
        previous_soxl, previous_soxx = desired_soxl, desired_soxx
        prior_state = state

    return pd.concat([out, pd.DataFrame(rows)], axis=1)


def evaluate_multi_asset_risk_control(
    controlled: pd.DataFrame,
    *,
    cost_per_turnover: float = 0.0003,
    annualization: int = 252,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate targets as next-session holdings with two-asset one-way turnover."""

    required = {
        "datetime",
        "soxl_return",
        "soxx_return",
        "soxl_weight",
        "soxx_weight",
        "cash_weight",
    }
    missing = sorted(required.difference(controlled.columns))
    if missing:
        raise ValueError(f"controlled frame missing required columns: {missing}")
    report = controlled.copy().sort_values("datetime").reset_index(drop=True)
    held = report[["soxl_weight", "soxx_weight", "cash_weight"]].shift(1).fillna(
        {"soxl_weight": 0.0, "soxx_weight": 0.0, "cash_weight": 1.0}
    )
    prior_held = held.shift(1).fillna({"soxl_weight": 0.0, "soxx_weight": 0.0, "cash_weight": 1.0})
    report["held_soxl_weight"] = held["soxl_weight"]
    report["held_soxx_weight"] = held["soxx_weight"]
    report["held_cash_weight"] = held["cash_weight"]
    report["gross_return"] = (
        held["soxl_weight"] * report["soxl_return"].astype(float)
        + held["soxx_weight"] * report["soxx_return"].astype(float)
    )
    report["turnover"] = 0.5 * held.sub(prior_held).abs().sum(axis=1)
    report["transaction_cost"] = report["turnover"] * float(cost_per_turnover)
    report["strategy_return"] = report["gross_return"] - report["transaction_cost"]
    report["equity"] = (1.0 + report["strategy_return"]).cumprod()
    report["drawdown"] = report["equity"].div(report["equity"].cummax()).sub(1.0)

    returns = report["strategy_return"]
    periods = len(returns)
    total_return = float(report["equity"].iloc[-1] - 1.0) if periods else 0.0
    annualized_return = float((1.0 + total_return) ** (annualization / periods) - 1.0) if periods else 0.0
    annualized_volatility = float(returns.std(ddof=0) * np.sqrt(annualization)) if periods else 0.0
    sharpe = annualized_return / annualized_volatility if annualized_volatility > 0.0 else 0.0
    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": float(report["drawdown"].min()) if periods else 0.0,
        "sharpe": float(sharpe),
        "total_turnover": float(report["turnover"].sum()),
        "total_transaction_cost": float(report["transaction_cost"].sum()),
    }
    return report, metrics
