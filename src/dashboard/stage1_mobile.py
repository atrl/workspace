"""Small read-only view of the active Stage1 simulated account.

Account capital growth uses inception capital and reconciled current equity.
The separate comparison uses common settled Qlib marks, excluding the first
session's return. Neither is certified time-weighted or net benchmark alpha.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _positive_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _account_performance(ledger: dict[str, Any]) -> dict[str, Any]:
    initial = _positive_number(ledger.get("initial_equity"))
    equity = _positive_number(ledger.get("equity"))
    reconciled = ledger.get("reconciliation", {}).get("status") == "matched"
    ready = initial is not None and equity is not None and reconciled
    return {
        "status": "ready" if ready else "needs_data",
        "label": "成立以来本金净值变动",
        "start": ledger.get("inception_date"),
        "as_of": ledger.get("updated_at"),
        "initial_equity": initial,
        "equity": equity,
        "capital_change": equity - initial if ready else None,
        "capital_growth_pct": 100 * (equity / initial - 1) if ready else None,
        "nav": equity / initial if ready else None,
        "basis": "reconciled_equity_divided_by_initial_capital",
        "twr_pct": None,
        "twr_status": "needs_data",
        "inception_excess_vs_soxx_pp": None,
        "inception_comparison_status": "needs_data",
        "note": "本金净值变动包含首日，未作外部资金流调整；完整资金流核证前不称TWR。"
                "成立时点同刻基准及费用/分红证据不全，不计算成立以来费后超额。",
    }


def build_stage1_mobile_summary(root: Path) -> dict[str, Any]:
    runtime = root / "runtime/soxl/stage1_futu_sim"
    cycle_path = runtime / "paper_cycle.json"
    if not cycle_path.is_file():
        return {"status": "needs_data", "reason": "paper_cycle_unavailable"}
    cycle = json.loads(cycle_path.read_text(encoding="utf-8"))
    ledger_name = cycle.get("ledger_file")
    if not isinstance(ledger_name, str) or Path(ledger_name).name != ledger_name:
        return {"status": "needs_data", "reason": "invalid_ledger_reference"}
    ledger_path = runtime / ledger_name
    if not ledger_path.is_file():
        return {"status": "needs_data", "reason": "paper_ledger_unavailable"}
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("cycle_id") != cycle.get("cycle_id") or ledger.get("execution_environment") != "SIMULATE":
        return {"status": "needs_data", "reason": "paper_cycle_identity_mismatch"}

    account = _account_performance(ledger)
    marks = sorted((row for row in ledger.get("settled_marks", [])
                    if isinstance(row, dict) and row.get("source") == "qlib.data.D.features"),
                   key=lambda row: str(row.get("date", "")))
    paired = [row for row in marks if all(float(row.get("prices", {}).get(symbol, 0)) > 0
                                            for symbol in ("SOXX", "SOXL"))
              and float(row.get("equity", 0)) > 0]
    performance: dict[str, Any] | None = None
    if len(paired) >= 2:
        first, last = paired[0], paired[-1]
        if first.get("date") != last.get("date"):
            first_equity = float(first["equity"])
            first_prices = {symbol: float(first["prices"][symbol]) for symbol in ("SOXX", "SOXL")}
            returns = {
                "Stage1": 100 * (float(last["equity"]) / first_equity - 1),
                **{symbol: 100 * (float(last["prices"][symbol]) /
                                  first_prices[symbol] - 1)
                   for symbol in ("SOXX", "SOXL")},
            }
            series = [
                {
                    "date": row["date"],
                    "stage1_nav": float(row["equity"]) / first_equity,
                    "soxx_nav": float(row["prices"]["SOXX"]) / first_prices["SOXX"],
                    "soxl_nav": float(row["prices"]["SOXL"]) / first_prices["SOXL"],
                }
                for row in paired
            ]
            performance = {
                "label": "首个共同结算收盘以来（不含首日）",
                "start": first["date"], "end": last["date"],
                "start_equity": first_equity,
                "end_equity": float(last["equity"]),
                "valuation_clock": "US_regular_session_close",
                "source": "paired_settled_qlib_marks",
                "returns_pct": returns,
                "series": series,
                "series_basis": "first_paired_settled_mark_equals_one",
                "difference_vs_soxx_pp": returns["Stage1"] - returns["SOXX"],
                "difference_vs_soxl_pp": returns["Stage1"] - returns["SOXL"],
                "relative_wealth_vs_soxx_pct": 100 * (
                    (1 + returns["Stage1"] / 100) / (1 + returns["SOXX"] / 100) - 1),
                "comparison_status": "diagnostic_only",
                "benchmark_basis": "price_return_without_costs_or_verified_distributions",
                "account_fee_basis": last.get("fee_basis", "unknown"),
                "note": "首个共同收盘归一为1，排除首日收益；仅连接已有Qlib结算标记。"
                        "历史费用时点未完整核证，SOXX/SOXL为价格收益，未计交易成本和已核证分红；"
                        "收益差单位为百分点，不是费后alpha；含停机被动持仓，非连续策略前向证据。",
            }

    # Do not silently replace a historical close with a live broker snapshot.
    # Their cash, marks, positions and observation times can all differ.
    reconciliation = None
    if paired:
        last = paired[-1]
        current = _positive_number(ledger.get("equity"))
        reconciliation = {
            "settled_date": last["date"], "account_as_of": ledger.get("updated_at"),
            "settled_equity": float(last["equity"]), "account_equity": current,
            "settled_minus_account_equity": float(last["equity"]) - current if current else None,
            "note": "结算曲线与账户快照分别保留；差额不自动认作手续费或回填历史。",
        }

    fills = [row for row in ledger.get("fills", []) if isinstance(row, dict)
             and row.get("side") in {"BUY", "SELL"} and row.get("symbol") in {"SOXL", "SOXX"}
             and float(row.get("quantity", 0)) > 0]
    fills.sort(key=lambda row: str(row.get("updated_time") or row.get("create_time") or ""))
    recent = [{"time": row.get("updated_time") or row.get("create_time"),
               "side": row["side"], "symbol": row["symbol"],
               "quantity": int(float(row["quantity"])), "price": float(row.get("price", 0))}
              for row in fills[-6:]]
    latest_by_side = []
    for side in ("BUY", "SELL"):
        match = next((row for row in reversed(fills) if row["side"] == side), None)
        if match is not None:
            latest_by_side.append({
                "time": match.get("updated_time") or match.get("create_time"),
                "side": side, "symbol": match["symbol"],
                "quantity": int(float(match["quantity"])),
                "price": float(match.get("price", 0)),
            })
    latest_by_side.sort(key=lambda row: str(row["time"]))
    return {
        "schema_version": 2,
        "status": "ready", "mode": "SIMULATE", "cycle_id": cycle["cycle_id"],
        "as_of": ledger.get("updated_at"), "inception_date": ledger.get("inception_date"),
        "equity": ledger.get("equity"), "drawdown_pct": 100 * float(ledger.get("drawdown", 0)),
        "positions": {symbol: int(ledger.get("positions", {}).get(symbol, 0))
                      for symbol in ("SOXL", "SOXX")},
        "account_performance": account,
        "valuation_reconciliation": reconciliation,
        "performance": performance, "recent_fills": recent,
        "latest_by_side": latest_by_side,
        "real_broker_orders": False,
    }
