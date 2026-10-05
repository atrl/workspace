"""Causal legacy order intents and an independent reconciliation ledger.

This module is not the research portfolio authority.  The workflow must feed
its orders to Qlib and reconcile the native Account/report against this ledger.
Legacy economic thresholds, indicator formulas, integer sizing and LIFO lots
are retained.  The compatibility reference keeps same-close, zero-cost
semantics; only its first-row buy lookahead guard is repaired in place.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from soxl_backtest.config.backtest_config import BacktestConfig
from soxl_backtest.indicators.indicator_calculator import IndicatorCalculator
from soxl_backtest.position.position_manager import PositionManager
from soxl_backtest.strategy.buy_strategy import BuyStrategy
from soxl_backtest.strategy.sell_strategy import SellStrategy


REPLAY_VERSION = "legacy_soxl_causal_v1"


class ExactPositionManager(PositionManager):
    """Daily exact marked weight, retaining legacy lot/cost-basis behavior."""

    def update_position(self, current_price):
        total_value = self.capital + self.shares * current_price
        self.position = self.shares * current_price / total_value if total_value > 0 else 0.0
        return total_value


@dataclass(frozen=True)
class LegacyOrderIntent:
    signal_date: pd.Timestamp
    side: str
    kind: str
    requested_shares: int
    signal_price: float
    signal_account: float
    signal_exposure: float
    conditions: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LegacyReplayResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    decisions: pd.DataFrame
    features: pd.DataFrame
    config: dict[str, Any]


def prepare_legacy_features(frame: pd.DataFrame, config: BacktestConfig | None = None) -> pd.DataFrame:
    """Calculate unchanged trailing indicators before slicing a replay window.

    ``vix`` must already reflect its availability cutoff.  No interpolation,
    forward/back fill, calendar construction, or data-source fallback occurs.
    A 180-observation readiness gate prevents trading on synthetic warmup
    values (legacy HV fills zero and RSI begins at zero).
    """
    required = {"date", "open", "close", "vix"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"blocked: missing legacy input columns: {sorted(missing)}")
    result = frame.copy(deep=True).reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"])
    if result.empty or result["date"].isna().any():
        raise ValueError("blocked: empty or invalid legacy calendar")
    if result["date"].duplicated().any() or not result["date"].is_monotonic_increasing:
        raise ValueError("blocked: legacy input calendar must be unique and increasing")
    for field in ("open", "close"):
        values = pd.to_numeric(result[field], errors="coerce")
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"blocked: invalid legacy {field}")
        result[field] = values.astype(float)
    result["vix"] = pd.to_numeric(result["vix"], errors="coerce")
    result["price"] = result["close"]
    effective = config or BacktestConfig()
    result = IndicatorCalculator(effective).process_data(result)
    # The old zero-volatility override disables volatility scaling.  Avoid
    # leaking 0/0 into the readiness gate while leaving the rule unchanged.
    if effective["VOLATILITY_MEDIAN"] == 0:
        result["VOL_COEF"] = 1.0
    fields = ["RSI6", "MA5", "MA20", "HV_20", "HV_30", "VOL_COEF", "panic_score", "vix"]
    result["signal_ready"] = np.isfinite(result[fields]).all(axis=1) & (np.arange(len(result)) >= 179)
    return result


def _buy_state(strategy: BuyStrategy) -> dict[str, Any]:
    return deepcopy({key: value for key, value in vars(strategy).items() if key not in {"config", "position_manager"}})


def _restore_buy_state(strategy: BuyStrategy, state: dict[str, Any]) -> None:
    for key in tuple(vars(strategy)):
        if key not in {"config", "position_manager"}:
            delattr(strategy, key)
    for key, value in deepcopy(state).items():
        setattr(strategy, key, value)


class LegacyDecisionEngine:
    """One close decision, one frozen intent, and at most one next-open fill."""

    def __init__(self, *, initial_cash: float = 100_000.0, cost_bps: float = 10.0,
                 config_overrides: dict[str, Any] | None = None):
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        if not math.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
            raise ValueError("cost_bps must lie in [0, 10000)")
        overrides = dict(config_overrides or {})
        overrides["INITIAL_CAPITAL"] = float(initial_cash)
        self.config = BacktestConfig(overrides)
        self.cost_rate = float(cost_bps) / 10_000.0
        self.position_manager = ExactPositionManager(self.config)
        self.buy_strategy = BuyStrategy(self.config, self.position_manager)
        self.sell_strategy = SellStrategy(self.config, self.position_manager)
        self.pending: LegacyOrderIntent | None = None
        self._pending_buy_state: dict[str, Any] | None = None
        self._last_decision_date: pd.Timestamp | None = None

    def _sell_quantity(self, current: pd.Series, kind: str, account: float) -> int:
        pm = self.position_manager
        price = float(current["price"])
        if kind == "take_profit":
            quantity = int(pm.shares * self.config["TAKE_PROFIT_POSITION"])
        elif kind == "stop_loss":
            quantity = int(pm.shares * self.config["STOP_LOSS_POSITION"])
        else:
            last_buy = pm.get_last_position()
            last_profit = (price / last_buy["price"] - 1.0) if last_buy else 0.0
            if (last_buy and current["panic_score"] >= self.config["SELL_SCORE_THRESHOLD"]
                    and last_profit >= self.config["LAST_BUY_PROFIT_TARGET"]):
                quantity = int(last_buy["shares"])
                remaining_weight = (pm.shares - quantity) * price / account
                if remaining_weight < self.config["MIN_POSITION"]:
                    keep_shares = int(account * self.config["MIN_POSITION"] / price)
                    quantity = int(pm.shares - keep_shares)
            else:
                amount = min(self.config["SELL_AMOUNT"], pm.position - self.config["MIN_POSITION"])
                quantity = int(account * amount / price)
        return max(0, min(int(pm.shares), quantity))

    def decide(self, features: pd.DataFrame, index: int) -> tuple[LegacyOrderIntent | None, dict[str, Any]]:
        """Evaluate only data through ``index``; reporting reuses this decision."""
        current = features.iloc[index]
        date = pd.Timestamp(current["date"])
        if self._last_decision_date is not None and date <= self._last_decision_date:
            raise ValueError("blocked: decision already evaluated or out of order")
        if self.pending is not None:
            raise ValueError("blocked: previous intent must settle before next close")
        self._last_decision_date = date
        pm = self.position_manager
        account = pm.update_position(float(current["price"]))
        if pm.shares > 0 and pm.position >= self.config["MIN_POSITION"]:
            pm.initial_position_built = True
        if self.buy_strategy.cumulative_buy_cooldown > 0:
            self.buy_strategy.cumulative_buy_cooldown -= 1
        audit: dict[str, Any] = {
            "signal_date": date, "signal_ready": bool(current["signal_ready"]),
            "should_buy": False, "sell_type": None, "requested_shares": 0,
            "signal_account": account, "signal_exposure": pm.position,
            "no_trade_reason": "indicator warmup", "conditions": [],
        }
        if not current["signal_ready"]:
            return None, audit
        prefix = features.iloc[:index + 1]
        before = _buy_state(self.buy_strategy)
        should_buy = bool(self.buy_strategy.should_buy(current, prefix, index))
        after = _buy_state(self.buy_strategy)
        _restore_buy_state(self.buy_strategy, before)
        audit["should_buy"] = should_buy
        sell_type = None
        if should_buy:
            step = self.config["BUY_STEP_NORMAL"] if pm.initial_position_built else self.config["BUY_STEP_INITIAL"]
            if current["price"] > current["MA20"]:
                step *= 0.4
            increment = max(0.0, min(pm.position + step, self.config["MAX_POSITION"]) - pm.position)
            quantity = int(account * increment / float(current["price"]))
            conditions = after["last_buy_conditions"]
            side, kind = "buy", "buy"
        else:
            sell_type = self.sell_strategy.should_sell(current)
            audit["sell_type"] = sell_type or None
            quantity = self._sell_quantity(current, sell_type, account) if sell_type else 0
            conditions = self.sell_strategy.get_last_sell_conditions() if sell_type else []
            side, kind = "sell", str(sell_type)
        audit["conditions"] = deepcopy(conditions)
        audit["requested_shares"] = quantity
        audit["no_trade_reason"] = (
            None if quantity > 0 else "zero requested shares" if should_buy or sell_type else
            "; ".join(str(x) for x in (after.get("no_buy_reason"), self.sell_strategy.get_no_sell_reason()) if x)
        )
        if quantity <= 0:
            return None, audit
        self.pending = LegacyOrderIntent(date, side, kind, quantity, float(current["price"]), account,
                                         pm.position, tuple(deepcopy(conditions)))
        self._pending_buy_state = after if should_buy else None
        return self.pending, audit

    def settle(self, *, date: pd.Timestamp, open_price: float) -> dict[str, Any] | None:
        """Execute frozen requested shares at the next supplied session open.

        Only affordability and actual holdings constrain the request.  Signals,
        thresholds, sizing inputs and indicator state are never recomputed here.
        Rejected/unaffordable buys do not consume buy cooldown state.
        """
        if self.pending is None:
            return None
        intent = self.pending
        date = pd.Timestamp(date)
        price = float(open_price)
        if date <= intent.signal_date or not math.isfinite(price) or price <= 0:
            raise ValueError("blocked: invalid next-open execution")
        pm = self.position_manager
        if intent.side == "buy":
            affordable = math.floor(pm.capital / (price * (1.0 + self.cost_rate)))
            quantity = min(intent.requested_shares, affordable)
            # Guard a floor boundary rounded upward by floating division.
            # Never fund a share by silently clamping a negative cash balance.
            while quantity > 0 and quantity * price + quantity * price * self.cost_rate > pm.capital:
                quantity -= 1
        else:
            quantity = min(intent.requested_shares, int(pm.shares))
        self.pending = None
        pending_state, self._pending_buy_state = self._pending_buy_state, None
        if quantity <= 0:
            return None
        notional = quantity * price
        fee = notional * self.cost_rate
        if intent.side == "buy":
            pm.capital -= notional + fee
            if pm.capital < 0:
                raise AssertionError("negative funded cash")
            pm.add_position(price, quantity)
            if pending_state is not None:
                _restore_buy_state(self.buy_strategy, pending_state)
            signed_quantity = quantity
        else:
            # Legacy implementation consumes the last lot first.  Its FIFO
            # comment was incorrect; changing that would change future signals.
            pm.reduce_position(quantity)
            pm.capital += notional - fee
            signed_quantity = -quantity
        account = pm.update_position(price)
        if sum(lot["shares"] for lot in pm.position_stack) != pm.shares or pm.shares < 0:
            raise AssertionError("lot/share mismatch")
        return {
            "date": date, "signal_date": intent.signal_date, "symbol": "SOXL",
            "amount": signed_quantity, "requested_shares": intent.requested_shares,
            "price": price, "fee": fee, "side": intent.side, "kind": intent.kind,
            "notional": notional, "cash": pm.capital, "shares": pm.shares,
            "account_at_open": account, "conditions": list(intent.conditions),
        }


def run_legacy_repaired(frame: pd.DataFrame, *, cost_bps: float = 10.0,
                        start: str | None = None, end: str | None = None,
                        initial_cash: float = 100_000.0,
                        config_overrides: dict[str, Any] | None = None) -> LegacyReplayResult:
    """Build causal orders/reconciliation evidence from a Qlib-loaded frame.

    With an explicit ``start``, reset to cash once at the immediately preceding
    canonical close and evaluate that close for a possible first-session open
    fill.  Earlier prehistory is indicator-only; no pre-window trades occur.
    With no explicit start, the full input starts in cash and warms up naturally.
    A final-close intent is retained as unfilled, with no fabricated execution.
    """
    engine = LegacyDecisionEngine(initial_cash=initial_cash, cost_bps=cost_bps,
                                  config_overrides=config_overrides)
    features = prepare_legacy_features(frame, engine.config)
    active = pd.Series(True, index=features.index)
    if start is not None:
        active &= features["date"] >= pd.Timestamp(start)
    if end is not None:
        active &= features["date"] <= pd.Timestamp(end)
    indices = features.index[active].tolist()
    if not indices:
        raise ValueError("blocked: empty legacy replay window")
    if not np.isfinite(features.loc[indices, "vix"]).all():
        raise ValueError("blocked: missing availability-aligned VIX in replay window")
    daily, trades, decisions = [], [], []
    if start is not None:
        prior_index = indices[0] - 1
        if prior_index < 0:
            raise ValueError("blocked: explicit replay start requires a previous canonical close")
        if not features.iloc[prior_index]["signal_ready"]:
            raise ValueError("blocked: previous-close indicators or VIX not ready at replay start")
        initial_intent, initial_decision = engine.decide(features, prior_index)
        initial_decision["planned_execution_date"] = features.iloc[indices[0]]["date"]
        initial_decision["status"] = "pending" if initial_intent else "no_order"
        initial_decision["initial_signal_only"] = True
        decisions.append(initial_decision)
    previous_account = float(initial_cash)
    for offset, index in enumerate(indices):
        row = features.iloc[index]
        trade = engine.settle(date=row["date"], open_price=float(row["open"]))
        if trade is not None:
            trades.append(trade)
        if decisions and decisions[-1]["status"] == "pending":
            decisions[-1]["status"] = "filled" if trade else "unfilled_unaffordable"
        intent, decision = engine.decide(features, index)
        decision["planned_execution_date"] = features.iloc[indices[offset + 1]]["date"] if offset + 1 < len(indices) else pd.NaT
        decision["status"] = "pending" if intent else "no_order"
        decision["initial_signal_only"] = False
        decisions.append(decision)
        pm = engine.position_manager
        account = pm.capital + pm.shares * float(row["close"])
        daily.append({
            "date": row["date"], "signal_date": trade["signal_date"] if trade else pd.NaT,
            "open": float(row["open"]), "close": float(row["close"]),
            "account": account, "cash": pm.capital, "shares": pm.shares,
            "exposure": pm.position, "position": pm.position,
            "fee": trade["fee"] if trade else 0.0,
            "turnover": trade["notional"] / previous_account if trade else 0.0,
            "net_return": account / previous_account - 1.0,
        })
        previous_account = account
    trade_columns = ["date", "signal_date", "symbol", "amount", "requested_shares", "price", "fee",
                     "side", "kind", "notional", "cash", "shares", "account_at_open", "conditions"]
    return LegacyReplayResult(pd.DataFrame(daily), pd.DataFrame(trades, columns=trade_columns),
                              pd.DataFrame(decisions), features, dict(engine.config.params))


__all__ = ["REPLAY_VERSION", "ExactPositionManager", "LegacyDecisionEngine",
           "LegacyOrderIntent", "LegacyReplayResult", "prepare_legacy_features", "run_legacy_repaired"]
