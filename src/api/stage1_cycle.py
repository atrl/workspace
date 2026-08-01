from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


VALID_STATUSES = {"restart_armed", "flatten_pending", "active", "blocked"}


def load_stage1_cycle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError("blocked: active Stage1 paper cycle manifest is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("blocked: unsupported Stage1 paper cycle version")
    if payload.get("status") not in VALID_STATUSES:
        raise ValueError("blocked: invalid Stage1 paper cycle status")
    if not str(payload.get("cycle_id", "")).strip():
        raise ValueError("blocked: Stage1 paper cycle id is missing")
    if not str(payload.get("remark_prefix", "")).startswith("stage1_"):
        raise ValueError("blocked: Stage1 paper cycle remark prefix is invalid")
    if float(payload.get("broker_reserve_cash", -1.0)) < 0.0:
        raise ValueError("blocked: Stage1 paper cycle reserve cash is invalid")
    for key in ("ledger_file", "execution_state_file"):
        value = str(payload.get(key, ""))
        if not value or Path(value).name != value:
            raise ValueError(f"blocked: Stage1 paper cycle {key} is invalid")
    return payload


def write_stage1_cycle(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    validated = dict(payload)
    validated["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return load_stage1_cycle(path)


def activate_flat_stage1_cycle(
    cycle: dict[str, Any],
    *,
    snapshot: dict[str, Any],
    session_date: str,
) -> dict[str, Any]:
    positions = {
        symbol: int(dict(snapshot.get("positions", {})).get(symbol, 0))
        for symbol in ("SOXL", "SOXX")
    }
    if any(positions.values()):
        raise RuntimeError(
            f"blocked: restart flatten is not complete; broker positions={positions}"
        )
    reserve = float(cycle["broker_reserve_cash"])
    initial_equity = float(snapshot.get("cash", 0.0)) - reserve
    if initial_equity <= 0.0:
        raise RuntimeError("blocked: restarted Stage1 sleeve has no positive cash")
    activated = dict(cycle)
    activated.update(
        {
            "status": "active",
            "inception_date": session_date,
            "initial_equity": initial_equity,
            "opening_positions": positions,
            "activated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    return activated
