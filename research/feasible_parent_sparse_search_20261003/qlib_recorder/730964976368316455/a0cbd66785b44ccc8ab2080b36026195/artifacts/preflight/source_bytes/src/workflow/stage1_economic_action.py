"""Research-only action routing with separate soft targets and hard risk limits.

This module does not replace the frozen production policy. Desired allocations
are legal net weights; a fee-affordable pre-fee target is passed to the shared
settlement helper. A required repair follows the existing desired allocation,
not a newly optimized nearest-feasible portfolio.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.workflow.intraday_recovery_direct_allocation import (
    IntradayRecoveryDirectAllocationSpec,
    _advance_state,
    _base_cap,
    project_legal_weights,
)
from src.workflow.stage1_open_action_value import (
    ACTION_FEATURES,
    build_open_action_features,
    prepare_on_policy_recovery,
)


ECONOMIC_FEATURES = (*ACTION_FEATURES, "risk_clock", "days_since_trade", "hard_beta_cap")
RULES_EXECUTE_PROBABILITY = 0.50
RULES_LARGE_TURNOVER = 0.05
RISK_TOLERANCE = 1e-12


def fee_affordable_target(
    net_weights: np.ndarray,
    *,
    pretrade_weights: np.ndarray,
    cost_bps: float,
) -> np.ndarray:
    """Convert a legal net allocation into a cash-funded pre-fee order target.

    Solve ``v = 1 - fee_rate * sum(abs(v * z_risky - current_risky))``.
    Risky target values are ``v*z`` and the cash component includes the fee
    reserve, so the returned vector sums to one for ``settle_target_session``.
    After the fee is debited, normalized holdings equal the requested ``z``.
    """
    desired = np.asarray(net_weights, dtype="float64")
    current = np.asarray(pretrade_weights, dtype="float64")
    if (desired.shape != (3,) or current.shape != (3,)
            or not np.isfinite(desired).all() or not np.isfinite(current).all()
            or not np.isclose(desired.sum(), 1.0, rtol=0, atol=1e-10)
            or not np.isclose(current.sum(), 1.0, rtol=0, atol=1e-10)
            or desired.min() < -RISK_TOLERANCE):
        raise ValueError("fee projection requires normalized finite weights and a long-only target")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("invalid proportional transaction cost")
    rate = float(cost_bps) / 10_000.0
    if rate == 0 or np.array_equal(desired, current):
        return desired.copy()
    if rate * np.abs(current[:2]).sum() >= 1:
        raise ValueError("insufficient equity to fund rebalance fees")
    lower, upper = 0.0, 1.0
    for _ in range(64):
        value = (lower + upper) / 2.0
        residual = value + rate * np.abs(value * desired[:2] - current[:2]).sum() - 1.0
        if residual > 0:
            upper = value
        else:
            lower = value
    net_value = (lower + upper) / 2.0
    execution = np.r_[net_value * desired[:2], 1.0 - net_value * desired[:2].sum()]
    fee = rate * np.abs(execution[:2] - current[:2]).sum()
    if execution[2] < fee - RISK_TOLERANCE:
        raise ValueError("fee projection did not leave enough cash")
    realized = (execution - np.array([0.0, 0.0, fee])) / (1.0 - fee)
    if not np.allclose(realized, desired, rtol=0, atol=5e-13):
        raise ValueError("fee projection failed the post-fee allocation identity")
    return execution


def _risk_envelope(
    *, drawdown: float, history: list[float], recovery_spec: IntradayRecoveryDirectAllocationSpec
) -> dict[str, float | bool]:
    """Tail ceilings do not depend on the allocator's current desired beta."""
    maximum_soxl = float(recovery_spec.maximum_soxl_weight)
    minimum_cash = float(recovery_spec.minimum_cash_weight)
    if not 0 <= maximum_soxl <= 1 - minimum_cash <= 1:
        raise ValueError("invalid structural portfolio limits")
    structural = 3.0 * maximum_soxl + 1.0 - minimum_cash - maximum_soxl
    trailing = (
        len(history) >= recovery_spec.trailing_loss_days
        and np.prod(1.0 + np.asarray(history[-recovery_spec.trailing_loss_days:])) - 1.0
        <= -recovery_spec.trailing_loss_trigger
    )
    emergency = drawdown >= recovery_spec.emergency_drawdown
    tail_cap = structural
    if trailing:
        tail_cap = min(tail_cap, recovery_spec.trailing_loss_beta)
    if emergency:
        tail_cap = min(tail_cap, recovery_spec.emergency_beta)
    return {
        "structural_beta_cap": float(structural),
        "tail_beta_cap": float(tail_cap),
        "trailing_loss_limit_active": bool(trailing),
        "emergency_limit_active": bool(emergency),
    }


def prepare_economic_open(
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
    policy: str = "rules_only",
    days_since_trade: int = 0,
    cost_bps: float = 10.0,
) -> dict[str, Any]:
    """Prepare a causal open decision; the caller settles fees and updates state.

    ``rules_only`` uses the frozen 22-feature classifier and retains the 5%
    large-transition override. ``net_value`` uses all 25 features and executes
    only positive predicted execute-versus-HOLD net bps, subject to hard repairs
    and the missing-model fallback. Neither policy treats an ordinary decrease
    in desired beta as a mandatory risk cut.
    """
    if policy not in {"rules_only", "net_value"}:
        raise ValueError("unknown economic action policy")
    pretrade = np.asarray(pretrade_weights, dtype="float64")
    if pretrade.shape != (3,) or not np.isfinite(pretrade).all() or not np.isclose(pretrade.sum(), 1.0):
        raise ValueError("economic action requires finite normalized holdings")
    if not np.isfinite([equity, peak, *history]).all() or min(equity, peak) <= 0:
        raise ValueError("economic action requires valid settled controller state")
    if risk_state not in (0, 1, 2, 3) or risk_clock < 0:
        raise ValueError("invalid controller state or clock")
    if not isinstance(days_since_trade, (int, np.integer)) or days_since_trade < 0:
        raise ValueError("days_since_trade must be a nonnegative completed-session count")
    drawdown = max(0.0, 1.0 - equity / peak)
    state, clock, fraction = _advance_state(risk_state, risk_clock, drawdown, recovery_spec)
    soft_cap, raw_beta = _base_cap(
        row, q=fraction, drawdown=drawdown, history=history, spec=recovery_spec
    )
    envelope = _risk_envelope(drawdown=drawdown, history=history, recovery_spec=recovery_spec)
    hard_cap = min(
        max(recovery_spec.defensive_beta, fraction * envelope["structural_beta_cap"]),
        envelope["tail_beta_cap"],
    )
    target_cap = min(soft_cap, hard_cap)
    target = project_legal_weights(
        float(row["soxl_weight"]), float(row["soxx_weight"]), target_cap,
        minimum_cash_weight=recovery_spec.minimum_cash_weight,
        maximum_soxl_weight=recovery_spec.maximum_soxl_weight,
    )
    if not np.isfinite(target).all():
        raise ValueError("allocator produced a nonfinite desired target")
    if 3.0 * target[0] + target[1] > hard_cap + RISK_TOLERANCE:
        raise ValueError("desired target exceeds the independent hard beta cap")
    execution_target = fee_affordable_target(target, pretrade_weights=pretrade, cost_bps=cost_bps)
    features = build_open_action_features(
        row, target=target, pretrade=pretrade, drawdown=drawdown,
        cap=soft_cap, state=state, history=history,
    )
    features.update(risk_clock=float(clock), days_since_trade=float(days_since_trade), hard_beta_cap=float(hard_cap))
    holding_beta = float(3.0 * pretrade[0] + pretrade[1])
    violations = []
    if pretrade.min() < -RISK_TOLERANCE:
        violations.append("long_only")
    if pretrade[0] > recovery_spec.maximum_soxl_weight + RISK_TOLERANCE:
        violations.append("maximum_soxl_weight")
    if pretrade[2] < recovery_spec.minimum_cash_weight - RISK_TOLERANCE:
        violations.append("minimum_cash_weight")
    if holding_beta > hard_cap + RISK_TOLERANCE:
        violations.append("hard_beta_cap")
    hard_repair = bool(violations)
    large_transition = features["open_turnover"] > RULES_LARGE_TURNOVER
    probability = predicted_bps = np.nan
    if model is not None:
        if policy == "rules_only":
            probability = float(model.predict_proba(pd.DataFrame([features]).loc[:, list(ACTION_FEATURES)])[0, 1])
        else:
            predicted_bps = float(model.predict(pd.DataFrame([features]).loc[:, list(ECONOMIC_FEATURES)])[0])
    score = probability if policy == "rules_only" else predicted_bps
    if hard_repair:
        execute, reason = True, "hard_risk_repair"
    elif model is None:
        execute, reason = True, "model_unavailable_execute_target"
    elif not np.isfinite(score):
        execute, reason = True, "nonfinite_prediction_execute_target"
    elif policy == "rules_only" and large_transition:
        execute, reason = True, "large_transition_execute_target"
    elif policy == "rules_only":
        execute = probability >= RULES_EXECUTE_PROBABILITY
        reason = "classifier_execute" if execute else "classifier_hold"
    else:
        execute = predicted_bps > 0.0
        reason = "positive_net_advantage" if execute else "nonpositive_net_advantage"
    return {
        "policy": policy,
        "execute_open_target": bool(execute),
        "execute_probability": probability,
        "predicted_net_advantage_bps": predicted_bps,
        "decision_reason": reason,
        "hard_risk_violation": hard_repair,
        "hard_risk_violation_reasons": violations,
        "mandatory_position_repair": hard_repair,
        "mandatory_derisk": bool(hard_repair and features["effective_beta_delta"] < 0),
        "large_transition": bool(large_transition),
        "large_transition_override": bool(policy == "rules_only" and large_transition),
        "routing_eligible": not hard_repair,
        "desired_open_weights": target.copy(),
        "execution_target_weights": execution_target.copy() if execute else pretrade.copy(),
        "candidate_execution_target_weights": execution_target.copy(),
        "open_weights": target.copy() if execute else pretrade.copy(),
        "pretrade_weights": pretrade.copy(),
        "execution_turnover": float(np.abs(execution_target[:2] - pretrade[:2]).sum()) if execute else 0.0,
        "cost_bps": float(cost_bps),
        "features": features,
        "drawdown": drawdown,
        "current_beta_cap": float(soft_cap),  # Frozen recovery feature contract.
        "soft_beta_cap": float(soft_cap),
        "hard_beta_cap": float(hard_cap),
        "desired_target_beta_cap": float(target_cap),
        "soft_cap_clipped_by_hard_limit": bool(soft_cap > hard_cap + RISK_TOLERANCE),
        "holding_beta": holding_beta,
        "raw_beta": float(raw_beta),
        "risk_state": state,
        "risk_clock": clock,
        "history": list(history),
        "days_since_trade": int(days_since_trade),
        **envelope,
    }


def prepare_economic_recovery(
    row: pd.Series,
    *,
    open_state: dict[str, Any],
    recovery_spec: IntradayRecoveryDirectAllocationSpec,
    model: Any | None,
    selected_model: dict[str, Any],
    pretrade_weights: np.ndarray | None = None,
    cost_bps: float = 10.0,
) -> dict[str, Any]:
    """Preserve conditional recovery while enforcing independent tail ceilings.

    The stop-clock open cap is deliberately not an intraday absolute ceiling:
    the original first-hour model may permit temporary recovery during waiting.
    Trailing-loss and emergency ceilings still apply, using the saved same-day
    controller context. The clock is never advanced here.
    """
    result = prepare_on_policy_recovery(
        row, open_state=open_state, recovery_spec=recovery_spec,
        model=model, selected_model=selected_model,
    )
    envelope = _risk_envelope(
        drawdown=float(open_state["drawdown"]), history=open_state["history"],
        recovery_spec=recovery_spec,
    )
    result.update(
        tail_beta_cap=envelope["tail_beta_cap"],
        soft_beta_cap=float(open_state["current_beta_cap"]),
        recovery_clipped_by_tail_cap=False,
        requested_recovery_beta=None,
        effective_recovery_beta=None,
        holding_beta_at_recovery=None,
        desired_recovery_weights=None,
        net_target_weights=None,
        decision_reason="baseline_recovery_not_selected",
    )
    if not result["recovery_action"]:
        return result
    requested = float(3 * result["target_weights"][0] + result["target_weights"][1])
    clipped_beta = min(requested, envelope["tail_beta_cap"])
    result["requested_recovery_beta"] = requested
    result["recovery_clipped_by_tail_cap"] = clipped_beta < requested - RISK_TOLERANCE
    if clipped_beta <= float(open_state["current_beta_cap"]) + RISK_TOLERANCE:
        result.update(recovery_action=False, target_weights=None,
                      decision_reason="tail_cap_leaves_no_recovery_room")
        return result
    target = project_legal_weights(
        float(row["soxl_weight"]), float(row["soxx_weight"]), clipped_beta,
        minimum_cash_weight=recovery_spec.minimum_cash_weight,
        maximum_soxl_weight=recovery_spec.maximum_soxl_weight,
    )
    if pretrade_weights is None:
        raise ValueError("selected recovery requires actual post-fee first-hour holdings")
    current = np.asarray(pretrade_weights, dtype="float64")
    if current.shape != (3,) or not np.isfinite(current).all() or not np.isclose(current.sum(), 1.0):
        raise ValueError("recovery requires finite normalized first-hour holdings")
    current_beta = float(3.0 * current[0] + current[1])
    result["holding_beta_at_recovery"] = current_beta
    desired_beta = float(3.0 * target[0] + target[1])
    if desired_beta <= current_beta + RISK_TOLERANCE:
        result.update(recovery_action=False, target_weights=None,
                      decision_reason="no_incremental_recovery")
        return result
    execution = fee_affordable_target(target, pretrade_weights=pretrade_weights, cost_bps=cost_bps)
    result.update(target_weights=execution, desired_recovery_weights=target, net_target_weights=target.copy(),
                  effective_recovery_beta=float(3 * target[0] + target[1]),
                  decision_reason="tail_capped_recovery" if result["recovery_clipped_by_tail_cap"] else "original_conditional_recovery")
    return result


__all__ = ["ECONOMIC_FEATURES", "fee_affordable_target", "prepare_economic_open", "prepare_economic_recovery"]
