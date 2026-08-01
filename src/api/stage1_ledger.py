from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


TRACKED_CODES = {"SOXL": "US.SOXL", "SOXX": "US.SOXX"}
REVERSE_CODES = {code: symbol for symbol, code in TRACKED_CODES.items()}
FILLED_STATUSES = {"FILLED_ALL"}


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
    cash_before_fees = float(initial_equity) + net_trade_cash
    inferred_fees = cash_before_fees - sleeve_cash
    if inferred_fees < -0.02:
        raise RuntimeError(
            "blocked: Stage1 ledger inferred negative fees; broker account has unowned cash flows"
        )
    inferred_fees = max(0.0, inferred_fees)

    mark_rows = _rebuild_settled_marks(
        initial_equity=float(initial_equity),
        orders=normalized,
        total_fees=inferred_fees,
        settled_marks=settled_marks,
        inception_date=inception_date,
        previous_marks=(previous or {}).get("settled_marks", []),
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
        "total_inferred_fees": inferred_fees,
        "fills": normalized,
        "settled_marks": mark_rows,
        "reconciliation": {
            "status": "matched",
            "broker_positions": broker_positions,
            "fill_positions": calculated_positions,
            "broker_cash": float(snapshot.get("cash", 0.0)),
            "sleeve_cash": sleeve_cash,
            "as_of": as_of,
        },
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_json(path, state)
    return state


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
        if status not in FILLED_STATUSES:
            pending.append(f"{order_id}:{status or 'UNKNOWN'}")
            continue
        quantity = int(float(raw.get("dealt_qty", 0)))
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
            }
        )
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
