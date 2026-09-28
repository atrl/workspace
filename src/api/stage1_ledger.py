from __future__ import annotations

import json
import hashlib
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from .futu_sim import terminal_fill_quantity


TRACKED_CODES = {"SOXL": "US.SOXL", "SOXX": "US.SOXX"}
REVERSE_CODES = {code: symbol for symbol, code in TRACKED_CODES.items()}


def reconcile_stage1_ledger(
    *,
    path: Path,
    initial_equity: float,
    broker_reserve_cash: float,
    inception_date: str,
    snapshot: dict[str, Any],
    orders: Iterable[dict[str, Any]],
    prices: dict[str, float],
    as_of: str,
    settled_marks: Iterable[dict[str, Any]] = (),
    remark_prefix: str = "stage1_",
    income_events: Iterable[dict[str, Any]] = (),
    cash_adjustment_confirmation: dict[str, Any] | None = None,
    cycle_id: str | None = None,
) -> dict[str, Any]:
    """Rebuild the Stage1 sleeve from confirmed Futu SIMULATE fills.

    The Futu simulated stock account is larger than this strategy sleeve.  The
    configured reserve belongs to the host account and is never available for
    Stage1 sizing.  Rebuilding from broker fills makes the ledger idempotent and
    prevents a model replay from pretending that an unexecuted trade occurred.
    """

    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive")
    if broker_reserve_cash < 0:
        raise ValueError("broker_reserve_cash cannot be negative")
    if set(prices) != set(TRACKED_CODES) or any(float(value) <= 0 for value in prices.values()):
        raise ValueError("ledger prices must contain positive SOXL and SOXX values")

    previous = _load_previous(path)
    normalized = _normalize_filled_orders(
        orders,
        inception_date=inception_date,
        remark_prefix=remark_prefix,
    )
    calculated_positions = {symbol: 0 for symbol in TRACKED_CODES}
    net_trade_cash = 0.0
    for order in normalized:
        direction = 1 if order["side"] == "BUY" else -1
        calculated_positions[order["symbol"]] += direction * int(order["quantity"])
        net_trade_cash -= direction * float(order["quantity"]) * float(order["price"])

    broker_positions = {
        symbol: int(dict(snapshot.get("positions", {})).get(symbol, 0))
        for symbol in TRACKED_CODES
    }
    if calculated_positions != broker_positions:
        raise RuntimeError(
            "blocked: Stage1 ledger position reconciliation failed: "
            f"fills={calculated_positions}, broker={broker_positions}"
        )

    sleeve_cash = float(snapshot.get("cash", 0.0)) - float(broker_reserve_cash)
    if sleeve_cash < -1e-6:
        raise RuntimeError("blocked: broker reserve cash exceeds Futu simulated cash")
    income_rows = _normalize_income_events(
        income_events, orders=normalized, inception_date=inception_date, as_of=as_of,
        previous_events=(previous or {}).get("income_events", []),
    )
    recognized_income = sum(row["recognized_cash"] for row in income_rows)
    confirmation = cash_adjustment_confirmation or (previous or {}).get("cash_adjustment_confirmation")
    net_adjustment = None
    if confirmation:
        if recognized_income:
            raise RuntimeError("blocked: confirmed income and cycle cash adjustment cannot be counted together")
        net_adjustment = _confirmed_cycle_cash_adjustment(
            confirmation, cycle_id=cycle_id, initial_equity=initial_equity,
            broker_reserve_cash=broker_reserve_cash, inception_date=inception_date,
            remark_prefix=remark_prefix, orders=normalized, sleeve_cash=sleeve_cash,
            net_trade_cash=net_trade_cash, previous=previous, as_of=as_of,
        )
    cash_before_fees = float(initial_equity) + net_trade_cash + recognized_income
    if net_adjustment:
        cash_before_fees += net_adjustment["anchor_net_cash_adjustment"]
    inferred_fees = cash_before_fees - sleeve_cash
    if inferred_fees < -0.02:
        raise RuntimeError(
            "blocked: Stage1 ledger inferred negative fees; broker account has unowned cash flows"
        )
    inferred_fees = max(0.0, inferred_fees)

    previous_income_ids = {row["event_id"] for row in (previous or {}).get("income_events", [])
                           if row.get("status") == "credited_confirmed"}
    for row in income_rows:
        if row["status"] == "credited_confirmed" and row["event_id"] not in previous_income_ids:
            if any(str(mark["date"]) >= row["credit_evidence"]["credited_at"][:10]
                   for mark in (previous or {}).get("settled_marks", [])):
                raise RuntimeError("blocked: late income evidence requires explicit settled-mark repair")

    mark_rows = _rebuild_settled_marks(
        initial_equity=float(initial_equity),
        orders=normalized,
        total_fees=0.0 if net_adjustment else inferred_fees,
        settled_marks=settled_marks,
        inception_date=inception_date,
        previous_marks=(previous or {}).get("settled_marks", []),
        income_events=income_rows,
        cash_adjustment_anchor=net_adjustment,
    )
    equity = sleeve_cash + sum(
        broker_positions[symbol] * float(prices[symbol]) for symbol in TRACKED_CODES
    )
    previous_peak = float(previous.get("peak_equity", initial_equity)) if previous else float(initial_equity)
    settled_peak = max([float(initial_equity), *[float(row["equity"]) for row in mark_rows]])
    peak_equity = max(previous_peak, settled_peak, equity)
    drawdown = equity / peak_equity - 1.0
    state = {
        "version": 1,
        "ledger_type": "stage1_futu_sim_sleeve",
        "execution_environment": "SIMULATE",
        "real_broker_orders": False,
        "initial_equity": float(initial_equity),
        "broker_reserve_cash": float(broker_reserve_cash),
        "inception_date": inception_date,
        "remark_prefix": remark_prefix,
        "cash": sleeve_cash,
        "positions": broker_positions,
        "prices": {symbol: float(prices[symbol]) for symbol in TRACKED_CODES},
        "equity": equity,
        "peak_equity": peak_equity,
        "drawdown": drawdown,
        "total_inferred_fees": None if net_adjustment else inferred_fees,
        "inferred_fees_basis": ("unknown_unallocated_net_investment_cash_flow" if net_adjustment else
                                "aggregate_cash_residual_after_confirmed_income_not_verified_per_order_fees"),
        "income_events": income_rows,
        "total_expected_gross_income": sum(row["gross_entitlement"] for row in income_rows),
        "total_confirmed_cash_income": recognized_income,
        "fills": normalized,
        "settled_marks": mark_rows,
        "reconciliation": {
            "status": "matched",
            "broker_positions": broker_positions,
            "fill_positions": calculated_positions,
            "broker_cash": float(snapshot.get("cash", 0.0)),
            "sleeve_cash": sleeve_cash,
            "confirmed_cash_income": recognized_income,
            "unexplained_positive_cash": max(0.0, sleeve_cash - cash_before_fees),
            "as_of": as_of,
        },
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if net_adjustment:
        state.update(cycle_id=cycle_id, cash_adjustment_confirmation=confirmation,
                     net_investment_cash_adjustment=net_adjustment)
        state["reconciliation"]["cash_basis"] = "broker_snapshot_with_user_confirmed_no_external_flows_through_anchor"
    _write_json(path, state)
    return state


def _fill_fingerprint(row: dict[str, Any]) -> tuple:
    return tuple(row[k] for k in ("order_id", "symbol", "side", "quantity", "price", "create_time", "remark"))


def _confirmed_cycle_cash_adjustment(
    confirmation: dict[str, Any], *, cycle_id: str | None, initial_equity: float,
    broker_reserve_cash: float, inception_date: str, remark_prefix: str,
    orders: list[dict[str, Any]], sleeve_cash: float, net_trade_cash: float,
    previous: dict[str, Any] | None, as_of: str,
) -> dict[str, Any]:
    """Recognize a fixed net investment cash residual, never imaginary dividends.

    User confirmation excludes external flows only through the original broker
    snapshot. The immutable anchor cannot grow to explain future cash credits.
    Later trading cash deductions remain inferred and are not total paid fees.
    """
    required_identity = {"cycle_id": cycle_id, "initial_equity": float(initial_equity),
                         "broker_reserve_cash": float(broker_reserve_cash),
                         "inception_date": inception_date, "remark_prefix": remark_prefix}
    if (not cycle_id or confirmation.get("schema") != "stage1_cycle_cash_adjustment_confirmation_v1"
            or confirmation.get("classification") != "unallocated_net_investment_cash_flow"
            or any(confirmation.get(key) != value for key, value in required_identity.items())):
        raise RuntimeError("blocked: cash adjustment confirmation does not match this cycle")
    user = confirmation.get("user_confirmation", {})
    if (not str(user.get("text", "")).strip() or user.get("external_cash_flows") is not False
            or user.get("account_reset") is not False or user.get("scope_start") != inception_date
            or user.get("scope_end_snapshot") != confirmation.get("snapshot_observed_at")):
        raise RuntimeError("blocked: cash adjustment lacks scoped user confirmation")
    old_confirmation = (previous or {}).get("cash_adjustment_confirmation")
    if old_confirmation and old_confirmation != confirmation:
        raise RuntimeError("blocked: immutable cash adjustment confirmation changed")
    try:
        anchor_time = datetime.fromisoformat(confirmation["snapshot_observed_at"])
        now = datetime.fromisoformat(as_of)
        if anchor_time.tzinfo is None or now.tzinfo is None or anchor_time > now:
            raise ValueError("invalid anchor time")
        sources = confirmation["sources"]
        decoded = {name: json.loads(_verify_income_source(str(meta["source_ref"]), str(meta["source_sha256"])))
                   for name, meta in sources.items()}
        balance, positions = decoded["balance"][0], decoded["positions"]
        anchor_orders = _normalize_filled_orders(decoded["orders"], inception_date=inception_date,
                                                 remark_prefix=remark_prefix)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("blocked: invalid cash adjustment snapshot evidence") from exc
    anchor_fills = {_fill_fingerprint(row) for row in anchor_orders}
    current_fills = {_fill_fingerprint(row) for row in orders}
    if not anchor_fills or not anchor_fills.issubset(current_fills):
        raise RuntimeError("blocked: original fills changed after cash adjustment confirmation")
    anchor_market_time = anchor_time.astimezone(ZoneInfo("America/New_York")).replace(tzinfo=None)
    for row in orders:
        if _fill_fingerprint(row) not in anchor_fills and datetime.fromisoformat(row["create_time"]) <= anchor_market_time:
            raise RuntimeError("blocked: unrecorded historical fill invalidates cash adjustment anchor")
    anchor_positions = {symbol: 0 for symbol in TRACKED_CODES}
    anchor_trade_cash = 0.0
    for row in anchor_orders:
        direction = 1 if row["side"] == "BUY" else -1
        anchor_positions[row["symbol"]] += direction * int(row["quantity"])
        anchor_trade_cash -= direction * row["trade_value"]
    raw_positions = {REVERSE_CODES[row["code"]]: int(row["qty"]) for row in positions if row["code"] in REVERSE_CODES}
    if anchor_positions != {symbol: raw_positions.get(symbol, 0) for symbol in TRACKED_CODES}:
        raise RuntimeError("blocked: original snapshot positions do not match original fills")
    anchor_cash = float(balance["cash"]) - broker_reserve_cash
    fixed_adjustment = anchor_cash - (initial_equity + anchor_trade_cash)
    if abs(fixed_adjustment - float(confirmation.get("anchor_net_cash_adjustment", float("nan")))) > 1e-6 or not math.isfinite(float(confirmation.get("anchor_net_cash_adjustment", float("nan")))):
        raise RuntimeError("blocked: cash adjustment amount does not match original snapshot")
    current_residual = sleeve_cash - (initial_equity + net_trade_cash)
    if current_residual > fixed_adjustment + 0.02:
        raise RuntimeError("blocked: new unexplained cash credit after confirmed snapshot")
    previous_adjustment = (previous or {}).get("net_investment_cash_adjustment")
    if previous_adjustment:
        prior_residual = float(previous_adjustment["current_net_cash_adjustment"])
        if current_residual > prior_residual + 0.02:
            raise RuntimeError("blocked: new unexplained cash credit after last reconciliation")
        if len(orders) == len((previous or {}).get("fills", [])) and abs(current_residual - prior_residual) > 0.02:
            raise RuntimeError("blocked: cash changed without new confirmed fills")
    elif current_fills == anchor_fills and abs(current_residual - fixed_adjustment) > 0.02:
        raise RuntimeError("blocked: cash changed since confirmed snapshot without new fills")
    return {"classification": "unallocated_net_investment_cash_flow",
            "anchor_net_cash_adjustment": fixed_adjustment,
            "anchor_observed_at": confirmation["snapshot_observed_at"],
            "anchor_market_date": anchor_market_time.date().isoformat(),
            "current_net_cash_adjustment": current_residual,
            "post_confirmation_inferred_cash_deductions": max(0.0, fixed_adjustment - current_residual),
            "total_fees": None,
            "historical_allocation": "none_before_anchor_observation; later_marks_include_fixed_cash_anchor_only"}


def _verify_income_source(ref: str, digest: str) -> bytes:
    path = Path(ref)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise RuntimeError("blocked: income source requires SHA-256")
    if not path.is_file():
        raise RuntimeError("blocked: income source evidence is missing")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest.lower():
        raise RuntimeError("blocked: income source evidence hash changed")
    return payload


def _normalize_income_events(
    events: Iterable[dict[str, Any]], *, orders: list[dict[str, Any]],
    inception_date: str, as_of: str, previous_events: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Entitlement is not payment: only separately sourced account credits count.

    Statement extraction is an upstream responsibility. This contract requires
    the immutable statement/cash-flow artifact plus its transaction identity,
    amount, date and category; quote adjustment records alone never credit cash.
    """
    result: dict[str, dict[str, Any]] = {}
    evidence_transactions: dict[str, str] = {}
    for event in [*previous_events, *events]:
        symbol = str(event.get("symbol", ""))
        ex_date = str(event.get("ex_div_date", ""))
        try:
            date.fromisoformat(ex_date)
            per_share = float(event["per_share_cash"])
        except (ValueError, TypeError, KeyError) as exc:
            raise RuntimeError("blocked: malformed income entitlement") from exc
        if (symbol not in TRACKED_CODES or event.get("currency") != "USD"
                or not inception_date <= ex_date <= as_of[:10]
                or not math.isfinite(per_share) or per_share <= 0
                or event.get("source_api") != "futu.get_rehab"):
            raise RuntimeError("blocked: unsupported income entitlement")
        raw_source = _verify_income_source(str(event.get("source_ref", "")), str(event.get("source_sha256", "")))
        source_rows = json.loads(raw_source)
        if not isinstance(source_rows, list) or not any(
            row.get("ex_div_date") == ex_date and row.get("per_cash_div") == per_share
            for row in source_rows if isinstance(row, dict)
        ):
            raise RuntimeError("blocked: entitlement does not match corporate action source")
        quantity = sum((1 if order["side"] == "BUY" else -1) * int(order["quantity"])
                       for order in orders if order["symbol"] == symbol and order["create_time"][:10] < ex_date)
        if quantity < 0:
            raise RuntimeError("blocked: unsupported negative dividend entitlement")
        row = {"event_id": f"corporate_cash_dividend:{symbol}:{ex_date}:USD", "symbol": symbol,
               "ex_div_date": ex_date, "per_share_cash": per_share, "currency": "USD",
               "eligible_quantity": quantity, "gross_entitlement": quantity * per_share,
               "source_api": "futu.get_rehab", "source_ref": str(event["source_ref"]),
               "source_sha256": str(event["source_sha256"]), "status": "expected_only",
               "recognized_cash": 0.0}
        credit = event.get("credit_evidence")
        if credit:
            try:
                credited_at = str(credit["credited_at"])
                date.fromisoformat(credited_at[:10])
                amount = float(credit["credited_amount"])
                if len(credited_at) > 10:
                    credit_time = datetime.fromisoformat(credited_at)
                    snapshot_time = datetime.fromisoformat(as_of)
                    if credit_time.tzinfo is None:
                        credit_time = credit_time.replace(tzinfo=snapshot_time.tzinfo)
                    if len(as_of) > 10 and credit_time > snapshot_time:
                        raise ValueError("future credit timestamp")
            except (ValueError, TypeError, KeyError) as exc:
                raise RuntimeError("blocked: malformed cash credit evidence") from exc
            if (credit.get("kind") not in {"broker_cash_flow", "broker_statement"}
                    or credit.get("category") != "DIVIDEND" or credit.get("currency") != "USD"
                    or not str(credit.get("transaction_id", "")).strip()
                    or not ex_date <= credited_at[:10] <= as_of[:10]
                    or not math.isfinite(amount) or amount <= 0
                    or quantity <= 0
                    or amount > row["gross_entitlement"] + 0.02):
                raise RuntimeError("blocked: unsupported cash credit evidence")
            if credit.get("source_sha256") == event.get("source_sha256"):
                raise RuntimeError("blocked: quote entitlement is not account credit evidence")
            _verify_income_source(str(credit.get("source_ref", "")), str(credit.get("source_sha256", "")))
            transaction_key = str(credit["transaction_id"])
            if transaction_key in evidence_transactions and evidence_transactions[transaction_key] != row["event_id"]:
                raise RuntimeError("blocked: cash credit transaction reused for multiple entitlements")
            evidence_transactions[transaction_key] = row["event_id"]
            row.update(status="credited_confirmed", recognized_cash=amount, credit_evidence=dict(credit))
        prior = result.get(row["event_id"])
        if prior:
            if any(prior[k] != row[k] for k in ("symbol", "ex_div_date", "per_share_cash", "eligible_quantity", "currency")):
                raise RuntimeError("blocked: conflicting corporate income event")
            if prior["status"] == "credited_confirmed":
                if row["status"] == "credited_confirmed" and prior["credit_evidence"] != row["credit_evidence"]:
                    raise RuntimeError("blocked: conflicting cash credit evidence")
                continue  # A recurring quote-only record must not remove confirmed income.
        result[row["event_id"]] = row
    return sorted(result.values(), key=lambda row: (row["ex_div_date"], row["symbol"]))


def _normalize_filled_orders(
    orders: Iterable[dict[str, Any]], *, inception_date: str, remark_prefix: str
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    pending: list[str] = []
    for raw in orders:
        code = str(raw.get("code", ""))
        remark = str(raw.get("remark", ""))
        created_at = str(raw.get("create_time", ""))
        if code not in REVERSE_CODES or not remark.startswith(remark_prefix):
            continue
        if created_at and created_at[:10] < inception_date:
            continue
        order_id = str(raw.get("order_id", ""))
        status = str(raw.get("order_status", "")).upper()
        quantity = terminal_fill_quantity(raw)
        if quantity is None:
            pending.append(f"{order_id}:{status or 'UNKNOWN'}")
            continue
        if quantity == 0:
            continue
        price = float(raw.get("dealt_avg_price", 0.0))
        side = str(raw.get("trd_side", "")).upper()
        if quantity <= 0 or price <= 0 or side not in {"BUY", "SELL"}:
            raise RuntimeError(f"blocked: invalid confirmed Stage1 fill {order_id}")
        normalized.append(
            {
                "order_id": order_id,
                "symbol": REVERSE_CODES[code],
                "side": side,
                "quantity": quantity,
                "price": price,
                "trade_value": quantity * price,
                "create_time": created_at,
                "updated_time": str(raw.get("updated_time", "")),
                "remark": remark,
                "status": status,
            }
        )
    if pending:
        raise RuntimeError(
            "blocked: Stage1 ledger has orders awaiting final fill reconciliation: "
            + ", ".join(sorted(pending))
        )
    return sorted(normalized, key=lambda row: (row["create_time"], row["order_id"]))


def _rebuild_settled_marks(
    *,
    initial_equity: float,
    orders: list[dict[str, Any]],
    total_fees: float,
    settled_marks: Iterable[dict[str, Any]],
    inception_date: str,
    previous_marks: Iterable[dict[str, Any]] = (),
    income_events: Iterable[dict[str, Any]] = (),
    cash_adjustment_anchor: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    total_trade_value = sum(float(order["trade_value"]) for order in orders)
    rows: list[dict[str, Any]] = []
    peak = float(initial_equity)
    frozen = {str(row["date"]): dict(row) for row in previous_marks}
    for mark in sorted(settled_marks, key=lambda row: str(row["date"])):
        mark_date = str(mark["date"])
        if mark_date < inception_date:
            continue
        if mark_date in frozen:
            row = frozen[mark_date]
            peak = max(peak, float(row["peak_equity"]), float(row["equity"]))
            rows.append(row)
            continue
        eligible = [order for order in orders if str(order["create_time"])[:10] <= mark_date]
        positions = {symbol: 0 for symbol in TRACKED_CODES}
        cash = float(initial_equity)
        eligible_value = 0.0
        for order in eligible:
            direction = 1 if order["side"] == "BUY" else -1
            positions[order["symbol"]] += direction * int(order["quantity"])
            cash -= direction * float(order["trade_value"])
            eligible_value += float(order["trade_value"])
        if total_trade_value > 0:
            cash -= total_fees * eligible_value / total_trade_value
        credited = [event for event in income_events if event["status"] == "credited_confirmed"
                    and event["credit_evidence"]["credited_at"][:10] <= mark_date]
        cash += sum(event["recognized_cash"] for event in credited)
        if cash_adjustment_anchor and mark_date >= cash_adjustment_anchor["anchor_market_date"]:
            cash += cash_adjustment_anchor["anchor_net_cash_adjustment"]
        mark_prices = {symbol: float(mark[symbol]) for symbol in TRACKED_CODES}
        equity = cash + sum(positions[symbol] * mark_prices[symbol] for symbol in TRACKED_CODES)
        peak = max(peak, equity)
        rows.append(
            {
                "date": mark_date,
                "cash": cash,
                "positions": positions,
                "prices": mark_prices,
                "equity": equity,
                "peak_equity": peak,
                "drawdown": equity / peak - 1.0,
                "source": "qlib.data.D.features",
                "confirmed_income_event_ids": [event["event_id"] for event in credited],
                "fee_basis": "proportional_allocation_of_aggregate_inferred_residual_not_observed_fees",
            }
        )
        if cash_adjustment_anchor:
            rows[-1]["fee_basis"] = "unknown_cash_adjustment_timing_not_allocated"
            rows[-1]["cash_adjustment_basis"] = cash_adjustment_anchor["historical_allocation"]
    return rows


def _load_previous(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("ledger_type") != "stage1_futu_sim_sleeve":
        raise ValueError("blocked: invalid Stage1 simulated ledger")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
