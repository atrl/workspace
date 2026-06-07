from __future__ import annotations

"""Public paper-trading review helpers.

Internal production summaries keep account and reconciliation evidence for
audits. Public dashboard/status surfaces should only expose the order-entry
view needed to review the signal and execution decision.
"""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PUBLIC_PAPER_FORBIDDEN_KEYS = frozenset(
    {
        "account",
        "account_value",
        "cash",
        "shares",
        "market_value",
        "reconcile",
        "reconcile_action",
        "actual_position",
        "position_gap",
        "current_position",
        "resulting_position",
        "quantity",
        "filled_qty",
        "broker_order_id",
    }
)


def _mapping(value: object) -> Mapping[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _float_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _clean(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if value is not None and value != ""}


def _execution_status(payload: Mapping[str, Any], risk_gate: Mapping[str, Any]) -> str:
    if risk_gate.get("approved") is False:
        return "blocked"
    order_entry = _mapping(payload.get("order_entry"))
    if order_entry.get("entry_status"):
        return str(order_entry["entry_status"])
    execution = _mapping(payload.get("execution_result"))
    if execution.get("status"):
        return str(execution["status"])
    if payload.get("dry_run") is True:
        return "dry_run"
    if payload.get("executed") is True:
        return "executed"
    return "planned"


def _entry_target(
    payload: Mapping[str, Any],
    signal: Mapping[str, Any],
    risk_gate: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> float | None:
    order_entry = _mapping(payload.get("order_entry"))
    for value in (
        order_entry.get("entry_target_position"),
        risk_gate.get("capped_target_position"),
        execution.get("target_position"),
        signal.get("target_position"),
    ):
        number = _float_or_none(value)
        if number is not None:
            return number
    return None


def _entry_side(
    *,
    status: str,
    target_position: float | None,
    signal: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str:
    if status == "blocked":
        return "BLOCKED"
    order_entry = _mapping(payload.get("order_entry"))
    side = _text_or_none(order_entry.get("entry_side") or order_entry.get("side"))
    if side:
        return side.upper()
    current = _float_or_none(signal.get("current_position"))
    if target_position is None or current is None:
        return "UNKNOWN"
    eps = 1e-9
    if target_position > current + eps:
        return "BUY"
    if target_position < current - eps:
        return "SELL"
    return "HOLD"


def public_order_entry_from_execution(
    *,
    signal: object,
    risk_gate: object,
    commands: Sequence[object] = (),
    execution_result: object | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Build a public order-entry view without account or holding details."""
    signal_map = _mapping(signal)
    risk_map = _mapping(risk_gate)
    execution = _mapping(execution_result)
    first_command = _mapping(next(iter(_sequence(commands)), {}))
    status = _execution_status(
        {
            "dry_run": dry_run,
            "execution_result": execution,
            "order_entry": {"entry_status": first_command.get("status")},
        },
        risk_map,
    )
    target = _entry_target({}, signal_map, risk_map, execution)
    side = _text_or_none(first_command.get("side"))
    if side is None:
        side = _entry_side(status=status, target_position=target, signal=signal_map, payload={})
    return _clean(
        {
            "entry_status": status,
            "entry_side": side.upper() if side else None,
            "entry_symbol": _text_or_none(first_command.get("symbol") or signal_map.get("symbol")),
            "entry_price": _float_or_none(first_command.get("price") or signal_map.get("price")),
            "entry_target_position": target,
            "submitted": execution.get("submitted") if isinstance(execution.get("submitted"), bool) else None,
        }
    )


def public_paper_review_from_summary(
    payload: Mapping[str, Any],
    *,
    summary_path: Path | str | None = None,
) -> dict[str, object]:
    """Return the safe review payload for public dashboards and notifications."""
    signal = _mapping(payload.get("signal"))
    risk_gate = _mapping(payload.get("risk_gate"))
    execution = _mapping(payload.get("execution_result"))
    status = _execution_status(payload, risk_gate)
    target = _entry_target(payload, signal, risk_gate, execution)
    signal_target = _float_or_none(signal.get("target_position"))
    side = _entry_side(status=status, target_position=target, signal=signal, payload=payload)
    blocking_reasons = risk_gate.get("blocking_reasons")
    if isinstance(blocking_reasons, tuple):
        blocking_reasons = list(blocking_reasons)
    if not isinstance(blocking_reasons, list) or risk_gate.get("approved") is not False:
        blocking_reasons = None

    return _clean(
        {
            "date": _text_or_none(payload.get("date") or signal.get("date")),
            "profile": _text_or_none(payload.get("profile") or signal.get("profile")),
            "broker": _text_or_none(payload.get("broker")),
            "dry_run": payload.get("dry_run") is True,
            "entry_status": status,
            "entry_side": side,
            "entry_symbol": _text_or_none(signal.get("symbol")),
            "entry_price": _float_or_none(signal.get("price")),
            "entry_target_position": target,
            "entry_signal_target_position": signal_target,
            "risk_gate_approved": risk_gate.get("approved") if isinstance(risk_gate.get("approved"), bool) else None,
            "submitted": execution.get("submitted") if isinstance(execution.get("submitted"), bool) else None,
            "blocking_reasons": blocking_reasons,
            "summary_path": str(summary_path) if summary_path is not None else None,
        }
    )


def collect_sensitive_public_keys(payload: object, *, prefix: str = "") -> list[str]:
    """Return sensitive key paths found in a public payload."""
    findings: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in PUBLIC_PAPER_FORBIDDEN_KEYS or any(
                key_text.endswith(f"_{forbidden}") for forbidden in PUBLIC_PAPER_FORBIDDEN_KEYS
            ):
                findings.append(path)
            findings.extend(collect_sensitive_public_keys(value, prefix=path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(collect_sensitive_public_keys(value, prefix=f"{prefix}[{index}]"))
    return findings


__all__ = [
    "PUBLIC_PAPER_FORBIDDEN_KEYS",
    "collect_sensitive_public_keys",
    "public_order_entry_from_execution",
    "public_paper_review_from_summary",
]
