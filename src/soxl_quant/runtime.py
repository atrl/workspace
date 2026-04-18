from __future__ import annotations

"""Runtime helpers shared by paper replay and future live execution.

The training path owns model creation and validation. This module owns
everything that happens after a model has already produced a position target:

- convert model output into an auditable signal intent
- run execution-side risk gates before any order is sent
- delegate actual order placement to a broker adapter
- persist enough state to reconcile target vs realized exposure later

Keeping this boundary explicit lets paper replay and live execution share the
same control flow instead of drifting into separate ad-hoc scripts.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .data import build_dataset, init_qlib
from .policy.environment import TrancheLedger
from .policy.observation import make_observation
from .policy.policy_builder import actor_action, build_policy
from .policy.positioning import action_to_position
from .settings import ACCOUNT, MODEL_DIR, PositionCapConfig, QLIB_DATA_PATH, load_profile


@dataclass(frozen=True)
class SignalIntent:
    date: str
    symbol: str
    profile: str
    model_path: str
    seed: Optional[int]
    action: int
    target_position: float
    current_position: float
    price: float
    regime_risk: float
    note: str


@dataclass(frozen=True)
class RiskCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class RiskGateDecision:
    approved: bool
    capped_target_position: float
    checks: Tuple[RiskCheck, ...]
    blocking_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class ExecutionCommand:
    date: str
    symbol: str
    side: str
    quantity: int
    price: float
    target_position: float
    reason: str


@dataclass(frozen=True)
class BrokerOrderStatus:
    date: str
    symbol: str
    side: str
    quantity: int
    avg_price: float
    status: str
    broker_order_id: str = ""
    message: str = ""


@dataclass(frozen=True)
class BrokerExecutionResult:
    submitted: bool
    broker_name: str
    status: str
    target_position: float
    resulting_position: float
    broker_order_id: str = ""
    filled_qty: int = 0
    avg_price: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class AccountSnapshot:
    date: str
    cash: float
    shares: int
    market_value: float
    account_value: float
    position: float


@dataclass(frozen=True)
class ReconciliationRecord:
    date: str
    expected_position: float
    actual_position: float
    position_gap: float
    action_required: str
    note: str


@dataclass(frozen=True)
class ExperimentRecord:
    profile: str
    model_path: str
    start: str
    end: str
    commands: int
    ending_account_value: float
    approved_signals: int
    blocked_signals: int


class BrokerAdapter(ABC):
    """Execution boundary for paper and live brokers.

    The training/runtime code should only care about target positions and
    resulting account state. Broker-specific session management lives behind
    this interface.
    """

    broker_name = "broker"

    @abstractmethod
    def healthcheck(self) -> Tuple[bool, str]:
        """Return whether the adapter is healthy enough to accept an order."""

    @abstractmethod
    def get_account_snapshot(self, date: str, price: float) -> AccountSnapshot:
        """Return the account state used by risk gating and audit logs."""

    @abstractmethod
    def get_position(self, symbol: str, price: float) -> float:
        """Return current exposure as a 0-1 portfolio weight."""

    @abstractmethod
    def get_open_orders(self, symbol: str) -> Sequence[BrokerOrderStatus]:
        """Return working orders that would make another rebalance unsafe."""

    @abstractmethod
    def place_target_position(
        self,
        date: str,
        symbol: str,
        target_position: float,
        price: float,
        reason: str,
    ) -> Tuple[List[ExecutionCommand], List[BrokerOrderStatus], BrokerExecutionResult]:
        """Translate a target weight into broker-specific orders."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Best-effort cancel. Paper mode always returns False for unknown ids."""


def latest_model_path(profile_name: str) -> Path:
    path = MODEL_DIR / f"soxl_{profile_name}_actor.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing model artifact: {path}")
    return path


def load_actor_bundle(profile_name: str, model_path: Optional[Path] = None):
    profile = load_profile(profile_name)
    bundle_path = model_path or latest_model_path(profile_name)
    bundle = torch.load(bundle_path, map_location="cpu")
    obs_dim = int(bundle["obs_dim"])
    _, actor = build_policy(obs_dim, profile)
    actor.load_state_dict(bundle["state_dict"])
    actor.eval()
    return actor, bundle_path, bundle


def bundle_position_cap_config(bundle: Dict[str, object], profile_name: str) -> PositionCapConfig:
    payload = bundle.get("position_cap")
    if isinstance(payload, dict):
        return PositionCapConfig(**payload)
    return load_profile(profile_name).position_cap


def summary_path_for_model(model_path: Path) -> Path:
    return model_path.with_name(model_path.name.replace("_actor.pt", "_summary.json"))


def promotion_gate_status(model_path: Path) -> Tuple[bool, str]:
    """Return whether the saved model has passed the promotion gate."""

    summary_path = summary_path_for_model(model_path)
    if not summary_path.exists():
        return False, f"missing summary sidecar: {summary_path.name}"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    promotion = payload.get("promotion")
    if not isinstance(promotion, dict):
        return False, "promotion gate missing from summary"
    if not promotion.get("walk_forward_evaluated", False):
        return False, "walk-forward promotion gate not evaluated"
    if not promotion.get("eligible", False):
        return False, "model failed promotion gate"
    return True, "model passed promotion gate"


class PaperBroker(BrokerAdapter):
    """Broker adapter used for deterministic replay and contract testing."""

    broker_name = "paper"

    def __init__(self, initial_cash: float = ACCOUNT, fee_rate: float = 0.001, lot_size: int = 100):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.shares = 0
        self.fee_rate = float(fee_rate)
        self.lot_size = int(lot_size)

    def account_value(self, price: float) -> float:
        return float(self.cash + self.shares * price)

    def get_position(self, symbol: str, price: float) -> float:
        del symbol
        account_value = self.account_value(price)
        if account_value <= 0:
            return 0.0
        return float(self.shares * price / account_value)

    def healthcheck(self) -> Tuple[bool, str]:
        return True, "paper broker ready"

    def get_open_orders(self, symbol: str) -> Sequence[BrokerOrderStatus]:
        del symbol
        return ()

    # Backward-compatible helper used by the existing unit test and older
    # scripts. New code should call place_target_position() through the adapter.
    def rebalance(self, date: str, symbol: str, target_position: float, price: float, reason: str) -> List[ExecutionCommand]:
        commands, _, _ = self.place_target_position(date, symbol, target_position, price, reason)
        return commands

    def place_target_position(
        self,
        date: str,
        symbol: str,
        target_position: float,
        price: float,
        reason: str,
    ) -> Tuple[List[ExecutionCommand], List[BrokerOrderStatus], BrokerExecutionResult]:
        account_value = self.account_value(price)
        target_shares = int(account_value * target_position / max(price, 1e-6) / self.lot_size) * self.lot_size
        delta = target_shares - self.shares
        if abs(delta) < self.lot_size:
            current_position = self.get_position(symbol, price)
            return (
                [],
                [],
                BrokerExecutionResult(
                    submitted=False,
                    broker_name=self.broker_name,
                    status="skipped",
                    target_position=float(target_position),
                    resulting_position=current_position,
                    error="below minimum lot threshold",
                ),
            )

        commands: List[ExecutionCommand] = []
        statuses: List[BrokerOrderStatus] = []
        if delta > 0:
            gross = delta * price
            fee = gross * self.fee_rate
            if gross + fee > self.cash:
                affordable = int(self.cash / (price * (1.0 + self.fee_rate)) / self.lot_size) * self.lot_size
                delta = max(affordable, 0)
                gross = delta * price
                fee = gross * self.fee_rate
            if delta >= self.lot_size:
                self.cash -= gross + fee
                self.shares += delta
                commands.append(ExecutionCommand(date, symbol, "BUY", int(delta), float(price), float(target_position), reason))
        else:
            sell_qty = min(abs(delta), self.shares)
            if sell_qty >= self.lot_size:
                gross = sell_qty * price
                fee = gross * self.fee_rate
                self.cash += gross - fee
                self.shares -= sell_qty
                commands.append(ExecutionCommand(date, symbol, "SELL", int(sell_qty), float(price), float(target_position), reason))

        for index, command in enumerate(commands, start=1):
            statuses.append(
                BrokerOrderStatus(
                    date=command.date,
                    symbol=command.symbol,
                    side=command.side,
                    quantity=command.quantity,
                    avg_price=command.price,
                    status="filled",
                    broker_order_id=f"paper-{date}-{index}",
                )
            )

        result = BrokerExecutionResult(
            submitted=bool(commands),
            broker_name=self.broker_name,
            status="filled" if commands else "rejected",
            target_position=float(target_position),
            resulting_position=self.get_position(symbol, price),
            broker_order_id=statuses[-1].broker_order_id if statuses else "",
            filled_qty=sum(item.quantity for item in commands),
            avg_price=float(price if commands else 0.0),
            error="" if commands else "insufficient cash or lot threshold",
        )
        return commands, statuses, result

    def cancel_order(self, order_id: str) -> bool:
        del order_id
        return False

    def get_account_snapshot(self, date: str, price: float) -> AccountSnapshot:
        market_value = self.shares * price
        account_value = self.cash + market_value
        position = market_value / account_value if account_value > 0 else 0.0
        return AccountSnapshot(date, float(self.cash), int(self.shares), float(market_value), float(account_value), float(position))


def risk_gate(
    broker: BrokerAdapter,
    signal: SignalIntent,
    max_position_change: float = 0.35,
    max_target_position: float = 1.0,
    model_ready: Tuple[bool, str] = (True, "model passed promotion gate"),
) -> RiskGateDecision:
    """Execution-side guard rails that must pass before any live order.

    These checks are intentionally simple and deterministic. Strategy logic
    chooses the target; runtime logic decides whether the broker should be
    allowed to pursue it today.
    """

    model_ok, model_detail = model_ready
    approved, detail = broker.healthcheck()
    target = min(signal.target_position, max_target_position)
    checks = [
        RiskCheck("model_promotion", model_ok, model_detail),
        RiskCheck("broker_health", approved, detail),
    ]
    blocking_reasons: List[str] = []
    if not model_ok:
        blocking_reasons.append(model_detail)

    open_orders = list(broker.get_open_orders(signal.symbol))
    open_orders_ok = len(open_orders) == 0
    checks.append(
        RiskCheck(
            "open_orders",
            open_orders_ok,
            "no working orders" if open_orders_ok else f"{len(open_orders)} open order(s) present",
        )
    )
    if not open_orders_ok:
        blocking_reasons.append("open orders must be reconciled before another rebalance")

    change = abs(target - signal.current_position)
    within_step_limit = change <= max_position_change + 1e-9
    checks.append(
        RiskCheck(
            "position_step_limit",
            within_step_limit,
            f"change={change:.3f} limit={max_position_change:.3f}",
        )
    )
    if not within_step_limit:
        blocking_reasons.append(f"target step {change:.3f} exceeds limit {max_position_change:.3f}")

    target_ok = target <= max_target_position + 1e-9
    checks.append(
        RiskCheck(
            "target_cap",
            target_ok,
            f"target={signal.target_position:.3f} capped={target:.3f} cap={max_target_position:.3f}",
        )
    )
    if not approved:
        blocking_reasons.append(detail)

    return RiskGateDecision(
        approved=model_ok and approved and open_orders_ok and within_step_limit and target_ok,
        capped_target_position=float(target),
        checks=tuple(checks),
        blocking_reasons=tuple(blocking_reasons),
    )


def reconcile_positions(date: str, expected_position: float, actual_position: float, tolerance: float = 0.03) -> ReconciliationRecord:
    gap = float(actual_position - expected_position)
    if abs(gap) <= tolerance:
        action_required = "none"
        note = "position aligned"
    else:
        action_required = "manual_review"
        note = f"position gap {gap:+.3f} exceeds tolerance {tolerance:.3f}"
    return ReconciliationRecord(
        date=date,
        expected_position=float(expected_position),
        actual_position=float(actual_position),
        position_gap=gap,
        action_required=action_required,
        note=note,
    )


def _runtime_frame() -> Tuple[pd.DataFrame, pd.Series]:
    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset()
    full_df = dataset.handler.feature_frame.droplevel("instrument").sort_index()
    price_series = dataset.handler.price_frame.droplevel("instrument")["price"].sort_index()
    return full_df, price_series


def _build_signal_intent(
    actor,
    profile_name: str,
    model_path: Path,
    bundle: Dict[str, object],
    feature_df: pd.DataFrame,
    price_path: np.ndarray,
    idx: int,
    broker: BrokerAdapter,
    ledger: TrancheLedger,
    peak_value: float,
    symbol: str = "SOXL",
) -> SignalIntent:
    date = feature_df.index[idx]
    price = float(price_path[idx])
    account = broker.get_account_snapshot(str(pd.Timestamp(date).date()), price)
    current_position = broker.get_position(symbol, price)
    obs = make_observation(feature_df, idx, current_position, account.account_value, peak_value, ledger, price_path)
    action = actor_action(actor, obs)
    target_position = action_to_position(action, obs, config=bundle_position_cap_config(bundle, profile_name))
    regime_risk = float(feature_df.iloc[idx]["regime_risk"])
    return SignalIntent(
        date=str(pd.Timestamp(date).date()),
        symbol=symbol,
        profile=profile_name,
        model_path=str(model_path),
        seed=int(bundle.get("seed")) if bundle.get("seed") is not None else None,
        action=int(action),
        target_position=float(target_position),
        current_position=float(current_position),
        price=price,
        regime_risk=regime_risk,
        note=f"regime={regime_risk:.2f}",
    )


def replay_profile(
    profile_name: str,
    start: str,
    end: str,
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    broker: Optional[BrokerAdapter] = None,
) -> Dict[str, object]:
    """Replay the policy on daily bars using the same broker interface as live."""

    full_df, price_series = _runtime_frame()
    feature_df = full_df.loc[start:end].copy()
    price_slice = price_series.loc[start:end].copy()
    if len(feature_df) <= 80:
        raise ValueError(f"insufficient replay bars between {start} and {end}")

    actor, bundle_path, bundle = load_actor_bundle(profile_name, model_path=model_path)
    broker = broker or PaperBroker()
    ledger = TrancheLedger()
    commands: List[ExecutionCommand] = []
    statuses: List[BrokerOrderStatus] = []
    executions: List[BrokerExecutionResult] = []
    accounts: List[AccountSnapshot] = []
    signals: List[SignalIntent] = []
    decisions: List[RiskGateDecision] = []
    reconciliations: List[ReconciliationRecord] = []
    price_path = price_slice.reindex(feature_df.index).ffill().bfill().to_numpy(dtype=np.float32)
    approved_signals = 0
    blocked_signals = 0
    peak_value = ACCOUNT

    for i in range(60, len(feature_df)):
        signal = _build_signal_intent(actor, profile_name, bundle_path, bundle, feature_df, price_path, i, broker, ledger, peak_value)
        decision = risk_gate(broker, signal)
        signals.append(signal)
        decisions.append(decision)
        if decision.approved:
            approved_signals += 1
            new_commands, new_statuses, execution = broker.place_target_position(
                signal.date,
                signal.symbol,
                decision.capped_target_position,
                signal.price,
                reason=signal.note,
            )
            commands.extend(new_commands)
            statuses.extend(new_statuses)
            executions.append(execution)
            if execution.submitted:
                ledger.update(signal.current_position, execution.resulting_position, signal.price)
        else:
            blocked_signals += 1
            executions.append(
                BrokerExecutionResult(
                    submitted=False,
                    broker_name=broker.broker_name,
                    status="blocked",
                    target_position=decision.capped_target_position,
                    resulting_position=signal.current_position,
                    error="; ".join(decision.blocking_reasons),
                )
            )
            if signal.current_position > 1e-6:
                ledger.update(signal.current_position, signal.current_position, signal.price)

        account = broker.get_account_snapshot(signal.date, signal.price)
        peak_value = max(peak_value, account.account_value)
        accounts.append(account)
        reconciliations.append(
            reconcile_positions(
                signal.date,
                expected_position=decision.capped_target_position if decision.approved else signal.current_position,
                actual_position=account.position,
            )
        )

    record = ExperimentRecord(
        profile=profile_name,
        model_path=str(bundle_path),
        start=start,
        end=end,
        commands=len(commands),
        ending_account_value=accounts[-1].account_value if accounts else ACCOUNT,
        approved_signals=approved_signals,
        blocked_signals=blocked_signals,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "signals.jsonl", signals)
        _write_jsonl(output_dir / "risk_gates.jsonl", decisions)
        _write_jsonl(output_dir / "commands.jsonl", commands)
        _write_jsonl(output_dir / "orders.jsonl", statuses)
        _write_jsonl(output_dir / "executions.jsonl", executions)
        _write_jsonl(output_dir / "accounts.jsonl", accounts)
        _write_jsonl(output_dir / "reconcile.jsonl", reconciliations)
        (output_dir / "summary.json").write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "record": record,
        "signals": signals,
        "risk_gates": decisions,
        "commands": commands,
        "orders": statuses,
        "executions": executions,
        "accounts": accounts,
        "reconcile": reconciliations,
    }


def plan_manual_session(
    profile_name: str,
    as_of: str,
    broker: BrokerAdapter,
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Generate a single-day semi-automatic execution plan.

    This does not auto-submit anything by itself. The caller can inspect the
    signal, risk gate, and reconciliation payloads before deciding to place a
    live order through a concrete broker adapter.
    """

    full_df, price_series = _runtime_frame()
    feature_df = full_df.loc[:as_of].copy()
    price_slice = price_series.loc[:as_of].copy()
    if len(feature_df) <= 80:
        raise ValueError(f"insufficient bars available up to {as_of}")

    actor, bundle_path, bundle = load_actor_bundle(profile_name, model_path=model_path)
    ledger = TrancheLedger()
    price_path = price_slice.reindex(feature_df.index).ffill().bfill().to_numpy(dtype=np.float32)
    idx = len(feature_df) - 1
    current_price = float(price_path[idx])
    account = broker.get_account_snapshot(str(pd.Timestamp(feature_df.index[idx]).date()), current_price)
    signal = _build_signal_intent(
        actor,
        profile_name,
        bundle_path,
        bundle,
        feature_df,
        price_path,
        idx,
        broker,
        ledger,
        peak_value=max(account.account_value, ACCOUNT),
    )
    decision = risk_gate(broker, signal, model_ready=promotion_gate_status(bundle_path))
    reconcile = reconcile_positions(signal.date, decision.capped_target_position if decision.approved else signal.current_position, account.position)

    payload = {
        "signal": signal,
        "risk_gate": decision,
        "account": account,
        "reconcile": reconcile,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, value in payload.items():
            (output_dir / f"{name}.json").write_text(json.dumps(asdict(value), ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _write_jsonl(path: Path, items: Sequence[object]) -> None:
    path.write_text("\n".join(json.dumps(asdict(item), ensure_ascii=False) for item in items), encoding="utf-8")
