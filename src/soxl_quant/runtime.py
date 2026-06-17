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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .backtest import (
    apply_selective_release_overlay,
    selective_release_execution_guard_ready,
    selective_release_gate_ready,
    selective_release_post_release_de_risk_ready,
    selective_release_post_release_target_cap,
)
from .config import StrategyConfig
from .policy.execution_overlays import overlay_config as named_execution_overlay_config
from .settings import ACCOUNT, MODEL_DIR, PositionCapConfig, ROOT, load_profile
from .policy.two_speed_budget import (
    BullParticipationGuardConfig,
    TwoSpeedBudgetConfig,
    bull_participation_guard_snapshot,
    compose_budget_usage_position,
    infer_mode_from_slow_position,
    maybe_rebalance_position,
)

LEGACY_ACTOR_RUNTIME_RETIRED = (
    "Legacy actor runtime replay has been retired. Use the simplified supervised "
    "Alpha158/cadence artifacts under runtime/model_selection/card/"
    "qlib_alpha158_supervised_cadence_v1."
)


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
    target_before_overlay: float = 0.0
    execution_overlay: str = ""
    overlay_gate_ready: bool = False
    overlay_execution_ready: bool = False
    post_release_state_days: int = 0
    post_release_drawdown: float = 0.0


@dataclass(frozen=True)
class RuntimeExecutionOverlayState:
    state_days: int = 0
    peak_account_value: float = ACCOUNT


@dataclass(frozen=True)
class RuntimeExecutionOverlayDecision:
    name: str
    enabled: bool
    target_before_overlay: float
    target_after_overlay: float
    target_after_post_release_cap: float
    gate_ready: bool
    execution_ready: bool
    post_release_de_risk_ready: bool
    post_release_state_days_before: int
    post_release_drawdown: float


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
    path = MODEL_DIR / profile_name / f"soxl_{profile_name}_actor.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing model artifact: {path}")
    return path


def load_actor_bundle(profile_name: str, model_path: Optional[Path] = None):
    del profile_name, model_path
    raise RuntimeError(LEGACY_ACTOR_RUNTIME_RETIRED)


def bundle_position_cap_config(bundle: Dict[str, object], profile_name: str) -> PositionCapConfig:
    payload = bundle.get("position_cap")
    if isinstance(payload, dict):
        return PositionCapConfig(**payload)
    return load_profile(profile_name).position_cap


def runtime_feature_packs(profile: StrategyConfig) -> Tuple[str, ...]:
    packs = list(profile.feature.packs)
    if profile.execution_overlay.enabled and profile.execution_overlay.name.startswith("selective_release"):
        packs.append("release_signals_v1")
    return tuple(dict.fromkeys(packs))


def execution_overlay_state_path(profile: StrategyConfig) -> Path | None:
    if not profile.execution_overlay.state_path:
        return None
    path = Path(profile.execution_overlay.state_path)
    return path if path.is_absolute() else ROOT / path


def load_runtime_execution_overlay_state(
    profile: StrategyConfig,
    *,
    account_value: float = ACCOUNT,
) -> RuntimeExecutionOverlayState:
    path = execution_overlay_state_path(profile)
    if path is None or not path.exists():
        return RuntimeExecutionOverlayState(peak_account_value=float(account_value))
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RuntimeExecutionOverlayState(
        state_days=int(payload.get("state_days", 0)),
        peak_account_value=float(payload.get("peak_account_value", account_value)),
    )


def save_runtime_execution_overlay_state(
    profile: StrategyConfig,
    state: RuntimeExecutionOverlayState,
    *,
    date: str,
) -> None:
    path = execution_overlay_state_path(profile)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": date,
        "profile": profile.name,
        "overlay": profile.execution_overlay.name,
        "state_days": int(state.state_days),
        "peak_account_value": float(state.peak_account_value),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def next_runtime_execution_overlay_state(
    profile: StrategyConfig,
    *,
    previous_state: RuntimeExecutionOverlayState,
    signal: SignalIntent,
    account_value: float,
    order_submitted: bool,
) -> RuntimeExecutionOverlayState:
    if not profile.execution_overlay.enabled or not profile.execution_overlay.name:
        return previous_state

    if (
        order_submitted
        and signal.execution_overlay == profile.execution_overlay.name
        and signal.overlay_execution_ready
        and signal.target_position > signal.current_position + 1e-6
    ):
        overlay = named_execution_overlay_config(profile.execution_overlay.name)
        return RuntimeExecutionOverlayState(
            state_days=max(int(overlay.get("post_release_state_days", 0)), 0),
            peak_account_value=max(float(account_value), 1e-8),
        )

    if previous_state.state_days > 0:
        next_days = max(int(previous_state.state_days) - 1, 0)
        return RuntimeExecutionOverlayState(
            state_days=next_days,
            peak_account_value=(
                max(float(previous_state.peak_account_value), float(account_value))
                if next_days > 0
                else float(account_value)
            ),
        )
    return RuntimeExecutionOverlayState(peak_account_value=float(account_value))


def apply_runtime_execution_overlay(
    profile: StrategyConfig,
    *,
    target_position: float,
    row: pd.Series,
    regime_risk: float,
    current_position: float,
    current_drawdown: float,
    account_value: float,
    state: RuntimeExecutionOverlayState | None = None,
) -> Tuple[float, RuntimeExecutionOverlayDecision]:
    if not profile.execution_overlay.enabled or not profile.execution_overlay.name:
        target = round(float(target_position), 10)
        return target, RuntimeExecutionOverlayDecision(
            name="",
            enabled=False,
            target_before_overlay=target,
            target_after_overlay=target,
            target_after_post_release_cap=target,
            gate_ready=False,
            execution_ready=False,
            post_release_de_risk_ready=False,
            post_release_state_days_before=0,
            post_release_drawdown=0.0,
        )

    overlay = named_execution_overlay_config(profile.execution_overlay.name)
    state = state or RuntimeExecutionOverlayState(peak_account_value=float(account_value))
    state_days = int(state.state_days)
    release_peak = (
        max(float(state.peak_account_value), float(account_value))
        if state_days > 0
        else float(account_value)
    )
    release_drawdown = (
        max(1.0 - float(account_value) / max(release_peak, 1e-8), 0.0)
        if state_days > 0
        else 0.0
    )

    target_before = float(target_position)
    gate_ready = selective_release_gate_ready(row, regime_risk, overlay)
    target_after_overlay = apply_selective_release_overlay(
        target_before,
        row,
        regime_risk,
        overlay,
        current_pos=current_position,
        current_drawdown=current_drawdown,
    )
    execution_ready = selective_release_execution_guard_ready(
        row,
        regime_risk,
        overlay,
        current_pos=current_position,
        target_pos=target_after_overlay,
        current_drawdown=current_drawdown,
    )
    de_risk_ready = selective_release_post_release_de_risk_ready(
        overlay,
        release_state_days=state_days,
        release_drawdown=release_drawdown,
    )
    target_after_cap = selective_release_post_release_target_cap(
        target_after_overlay,
        overlay,
        release_state_days=state_days,
        release_drawdown=release_drawdown,
    )
    target = round(float(target_after_cap), 10)
    decision = RuntimeExecutionOverlayDecision(
        name=profile.execution_overlay.name,
        enabled=True,
        target_before_overlay=round(target_before, 10),
        target_after_overlay=round(float(target_after_overlay), 10),
        target_after_post_release_cap=target,
        gate_ready=bool(gate_ready),
        execution_ready=bool(execution_ready),
        post_release_de_risk_ready=bool(de_risk_ready),
        post_release_state_days_before=state_days,
        post_release_drawdown=round(float(release_drawdown), 10),
    )
    return target, decision


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


def profile_name_for_model_path(model_path: Path, family: str) -> str:
    stem = model_path.name.replace("_actor.pt", "")
    prefix = "soxl_"
    return stem[len(prefix):] if stem.startswith(prefix) else family


def runtime_auto_select_allowed(profile_name: str) -> Tuple[bool, str]:
    try:
        profile = load_profile(profile_name)
    except FileNotFoundError:
        return False, f"profile {profile_name} is not registered for runtime auto-selection"
    overlay = profile.execution_overlay
    if overlay.enabled and not overlay.auto_select_allowed:
        return False, f"profile overlay {overlay.name} is not approved for runtime auto-selection"
    return True, "profile approved for runtime auto-selection"


WALK_FORWARD_SELECTION_WINDOWS = ("wf_2020", "wf_2022", "wf_2023_2024")


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def walk_forward_runtime_selection_score(
    model_path: Path,
    summary: dict[str, Any],
    profile_name: str,
) -> Tuple[float, str] | None:
    promotion = summary.get("promotion") or {}
    if not promotion.get("walk_forward_evaluated", False):
        return None
    report_path = model_path.parent / f"{profile_name}_report.json"
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        profile = load_profile(profile_name)
    except (OSError, json.JSONDecodeError, FileNotFoundError):
        return None

    scores: list[float] = []
    for window in WALK_FORWARD_SELECTION_WINDOWS:
        row = report.get(window)
        if not isinstance(row, dict):
            return None
        metrics = row.get("main_backtest")
        if not isinstance(metrics, dict):
            return None
        return_value = row.get("relative_outperformance", metrics.get("relative_return", metrics.get("return")))
        score = (
            _float_or_default(return_value, float("-inf")) * profile.selection_score.return_weight
            + _float_or_default(metrics.get("sharpe")) * profile.selection_score.sharpe_weight
            - _float_or_default(metrics.get("max_dd")) * profile.selection_score.max_dd_penalty
        )
        scores.append(score)
    return sum(scores) / len(scores), "walk_forward_mean"


def runtime_selection_score(
    model_path: Path,
    summary: dict[str, Any],
    profile_name: str,
) -> Tuple[float, str]:
    walk_forward_score = walk_forward_runtime_selection_score(model_path, summary, profile_name)
    if walk_forward_score is not None:
        return walk_forward_score
    return float(summary.get("best_score", float("-inf"))), "best_score"


def best_model_for_family(family: str = "balanced") -> Path:
    """自动选择通过 promotion gate 且 runtime selection score 最高的模型。

    扫描 ``runtime/model/card/*/soxl_{family}*_actor.pt``，过滤通过 promotion gate
    的候选。若存在完整 walk-forward report，使用 walk-forward 均值分数；否则
    回退到训练 summary 的 ``best_score``。

    若无任何候选通过 gate，拒绝自动选择，避免定时任务静默使用
    后续 risk gate 会拒绝的未晋升 actor。
    """
    candidates = sorted(MODEL_DIR.rglob(f"soxl_{family}*_actor.pt"))
    if not candidates:
        raise FileNotFoundError(f"no model artifacts found for family={family}")

    scored: List[Tuple[Path, float]] = []
    for path in candidates:
        ok, reason = promotion_gate_status(path)
        if not ok:
            continue
        profile_name = profile_name_for_model_path(path, family)
        ok, reason = runtime_auto_select_allowed(profile_name)
        if not ok:
            continue
        summary_path = summary_path_for_model(path)
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            score, source = runtime_selection_score(path, payload, profile_name)
        except Exception:
            score = float("-inf")
            source = "unavailable"
        scored.append((path, score, source))

    if scored:
        best_path, best_score, best_source = max(scored, key=lambda x: x[1])
        print(
            f"[auto_select] family={family} model={best_path.name} "
            f"score={best_score:.2f} source={best_source}"
        )
        return best_path

    raise RuntimeError(
        f"no promoted model for family={family}; pass --profile explicitly after reviewing "
        "promotion evidence or update the model summary sidecar"
    )


def auto_select_profile(family: str = "balanced") -> Tuple[str, Path]:
    """返回选中的 profile 名和模型路径。

    Profile 名从模型文件名解析：``soxl_{profile}_actor.pt``。
    """
    model_path = best_model_for_family(family)
    return profile_name_for_model_path(model_path, family), model_path


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
    clip_position_change: bool = False,
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

    raw_change = target - signal.current_position
    change = abs(raw_change)
    clipped_by_step = False
    if change > max_position_change + 1e-9 and clip_position_change:
        direction = 1.0 if raw_change > 0 else -1.0
        target = signal.current_position + direction * max_position_change
        target = max(min(target, max_target_position), 0.0)
        clipped_by_step = True
        change = abs(target - signal.current_position)
    within_step_limit = change <= max_position_change + 1e-9
    checks.append(
        RiskCheck(
            "position_step_limit",
            within_step_limit,
            (
                f"change={abs(raw_change):.3f} clipped_target={target:.3f} "
                f"limit={max_position_change:.3f}"
                if clipped_by_step
                else f"change={change:.3f} limit={max_position_change:.3f}"
            ),
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


def _combined_promotion_gate_status(model_paths: Sequence[Path]) -> Tuple[bool, str]:
    results = [(path, promotion_gate_status(path)) for path in model_paths]
    failed = [f"{path.name}: {reason}" for path, (ok, reason) in results if not ok]
    if failed:
        return False, "; ".join(failed)
    labels = ", ".join(path.name for path, _status in results)
    return True, f"all composed model gates passed: {labels}"


def composer_promotion_gate_status(
    composer_profile: str,
    model_paths: Sequence[Path],
) -> Tuple[bool, str]:
    """Return model-gate status for a governed multi-actor composer."""

    review_path = MODEL_DIR / composer_profile / "runtime_promotion_review.json"
    if not review_path.exists():
        return _combined_promotion_gate_status(model_paths)

    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"composer promotion review unreadable: {exc}"

    if review.get("profile") != composer_profile:
        return False, f"composer promotion review profile mismatch: {review.get('profile')}"
    if review.get("decision") != "approved" or not review.get("runtime_model_gate_allowed", False):
        reasons = review.get("blocking_reasons") or ["composer promotion review is not approved"]
        return False, "; ".join(str(reason) for reason in reasons)

    components = review.get("components") or {}
    expected = [
        str(components.get("slow_model", "")),
        str(components.get("fast_model", "")),
    ]
    actual = [path.name for path in model_paths]
    if expected and expected != actual:
        return False, f"composer component model mismatch: expected={expected} actual={actual}"
    try:
        label = review_path.relative_to(ROOT)
    except ValueError:
        label = review_path
    return True, f"composer promotion review approved: {label}"


def compose_budget_usage_runtime_signal(
    slow_signal: SignalIntent,
    fast_signal: SignalIntent,
    price_series: pd.Series,
    *,
    composer_profile: str,
    config: TwoSpeedBudgetConfig | None = None,
    bull_guard: BullParticipationGuardConfig | None = None,
) -> tuple[SignalIntent, dict[str, object]]:
    """Build a single runtime signal from slow risk mode and fast budget usage."""

    if slow_signal.date != fast_signal.date:
        raise ValueError(f"slow/fast signal dates differ: {slow_signal.date} != {fast_signal.date}")
    if slow_signal.symbol != fast_signal.symbol:
        raise ValueError(f"slow/fast symbols differ: {slow_signal.symbol} != {fast_signal.symbol}")

    cfg = config or TwoSpeedBudgetConfig()
    guard = bull_guard or BullParticipationGuardConfig()
    slow_position = max(0.0, min(float(slow_signal.target_position), 1.0))
    fast_usage = max(0.0, min(float(fast_signal.target_position), 1.0))
    mode = infer_mode_from_slow_position(slow_position)
    base_target = compose_budget_usage_position(fast_usage, mode, cfg)
    guard_snapshot = (
        bull_participation_guard_snapshot(price_series, guard, as_of=slow_signal.date)
        if guard.enabled
        else {
            "enabled": False,
            "ready": False,
            "price": float(slow_signal.price),
            "moving_average": float("nan"),
            "momentum": float("nan"),
            "drawdown": float("nan"),
        }
    )
    raw_target = max(base_target, fast_usage) if bool(guard_snapshot["ready"]) else base_target
    target = maybe_rebalance_position(slow_signal.current_position, raw_target, cfg.trade_lock_days, cfg)
    action = 0 if abs(target - slow_signal.current_position) < 0.01 else (1 if target > slow_signal.current_position else 2)
    model_path = f"slow={slow_signal.model_path};fast={fast_signal.model_path}"
    note = (
        f"composer=budget_usage slow={slow_signal.profile}:{slow_position:.2%} "
        f"fast={fast_signal.profile}:{fast_usage:.2%} mode={mode} "
        f"base={base_target:.2%} raw={raw_target:.2%} "
        f"bull_guard={bool(guard_snapshot['ready'])}"
    )
    signal = SignalIntent(
        date=slow_signal.date,
        symbol=slow_signal.symbol,
        profile=composer_profile,
        model_path=model_path,
        seed=None,
        action=action,
        target_position=float(target),
        current_position=float(slow_signal.current_position),
        price=float(slow_signal.price),
        regime_risk=float(slow_signal.regime_risk),
        note=note,
        target_before_overlay=float(base_target),
    )
    composition = {
        "name": "budget_usage",
        "composer_profile": composer_profile,
        "slow_profile": slow_signal.profile,
        "fast_profile": fast_signal.profile,
        "slow_position": slow_position,
        "fast_usage": fast_usage,
        "mode": mode,
        "base_two_speed_position": float(base_target),
        "raw_two_speed_position": float(raw_target),
        "target_position": float(target),
        "bull_participation_guard": bool(guard_snapshot["ready"]),
        "bull_guard": guard_snapshot,
    }
    return signal, composition


def replay_profile(
    profile_name: str,
    start: str,
    end: str,
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    broker: Optional[BrokerAdapter] = None,
) -> Dict[str, object]:
    del profile_name, start, end, model_path, output_dir, broker
    raise RuntimeError(LEGACY_ACTOR_RUNTIME_RETIRED)


def plan_manual_session(
    profile_name: str,
    as_of: str,
    broker: BrokerAdapter,
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    clip_position_change: bool = False,
) -> Dict[str, object]:
    del profile_name, as_of, broker, model_path, output_dir, clip_position_change
    raise RuntimeError(LEGACY_ACTOR_RUNTIME_RETIRED)


def plan_budget_usage_composed_session(
    *,
    slow_profile: str,
    fast_profile: str,
    as_of: str,
    broker: BrokerAdapter,
    composer_profile: str,
    slow_model_path: Optional[Path] = None,
    fast_model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    config: TwoSpeedBudgetConfig | None = None,
    bull_guard: BullParticipationGuardConfig | None = None,
    clip_position_change: bool = False,
) -> Dict[str, object]:
    del (
        slow_profile,
        fast_profile,
        as_of,
        broker,
        composer_profile,
        slow_model_path,
        fast_model_path,
        output_dir,
        config,
        bull_guard,
        clip_position_change,
    )
    raise RuntimeError(LEGACY_ACTOR_RUNTIME_RETIRED)


def _write_jsonl(path: Path, items: Sequence[object]) -> None:
    path.write_text("\n".join(json.dumps(asdict(item), ensure_ascii=False) for item in items), encoding="utf-8")
