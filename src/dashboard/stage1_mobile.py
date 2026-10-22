"""Small read-only view of the active Stage1 simulated account.

Returns are aligned to the first and last common settled Qlib mark. They are
operational attribution, never a forward promotion decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
            returns = {
                "Stage1": 100 * (float(last["equity"]) / float(first["equity"]) - 1),
                **{symbol: 100 * (float(last["prices"][symbol]) /
                                  float(first["prices"][symbol]) - 1)
                   for symbol in ("SOXX", "SOXL")},
            }
            performance = {
                "start": first["date"], "end": last["date"],
                "source": "paired_settled_qlib_marks",
                "returns_pct": returns,
                "difference_vs_soxx_pp": returns["Stage1"] - returns["SOXX"],
                "difference_vs_soxl_pp": returns["Stage1"] - returns["SOXL"],
                "comparison_status": "diagnostic_only",
                "note": "同窗收盘标记；SOXX/SOXL 为持有收益，未计基线交易成本；非晋升证据",
            }

    fills = [row for row in ledger.get("fills", []) if isinstance(row, dict)
             and row.get("side") in {"BUY", "SELL"} and row.get("symbol") in {"SOXL", "SOXX"}
             and float(row.get("quantity", 0)) > 0]
    fills.sort(key=lambda row: str(row.get("updated_time") or row.get("create_time") or ""))
    recent = [{"time": row.get("updated_time") or row.get("create_time"),
               "side": row["side"], "symbol": row["symbol"],
               "quantity": int(float(row["quantity"])), "price": float(row.get("price", 0))}
              for row in fills[-6:]]
    return {
        "status": "ready", "mode": "SIMULATE", "cycle_id": cycle["cycle_id"],
        "as_of": ledger.get("updated_at"), "inception_date": ledger.get("inception_date"),
        "equity": ledger.get("equity"), "drawdown_pct": 100 * float(ledger.get("drawdown", 0)),
        "positions": {symbol: int(ledger.get("positions", {}).get(symbol, 0))
                      for symbol in ("SOXL", "SOXX")},
        "performance": performance, "recent_fills": recent,
        "real_broker_orders": False,
    }
