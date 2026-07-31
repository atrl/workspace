from __future__ import annotations

import argparse
import json
from datetime import datetime, time
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from src.api import FutuSimulatedBroker, opend_reachable
from src.data import build_qlib_store_from_frame
from src.data.intraday import (
    fetch_futu_intraday_bars,
    merge_intraday_bars,
    read_intraday_cache,
    resample_intraday_bars,
    write_intraday_cache,
)
from src.data.market_update import _default_end_date, update_market_data
from src.data.multi_3x_etf_update import update_multi_3x_etf_data
from src.data.soxs_defensive import build_soxs_defensive_provider
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf import (
    DynamicSoxxSoxlCashAllocationWfSpec,
    _attach_dynamic_allocation_features,
    _attach_intraday_risk_state,
    _attach_soxx_constituent_model_features,
    _build_allocation_provider,
    _build_intraday_risk_state_daily,
    _load_multi_3x_raw_panel,
    _load_soxx_constituent_factor_frame,
)
from src.workflow.stage1_futu_sim import (
    DEFAULT_NOTIONAL,
    build_runtime_target,
    run_stage1_futu_sim_card,
)
from src.schedule.notifications import BarkNotifier, BarkPushConfig


NEW_YORK_TZ = ZoneInfo("America/New_York")
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
RUNTIME_RELATIVE = Path("runtime/soxl/stage1_futu_sim")


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    return Path(__file__).resolve().parents[2]


def refresh_intraday_store(
    *,
    root: Path,
    session_date: str,
    host: str = "127.0.0.1",
    port: int = 11111,
) -> dict[str, Any]:
    cache_path = root / "data" / "processed" / "intraday" / "soxl_soxx_1m.parquet"
    existing = read_intraday_cache(cache_path)
    fetched = [
        fetch_futu_intraday_bars(
            symbol,
            start=session_date,
            end=session_date,
            host=host,
            port=port,
        )
        for symbol in ("SOXL", "SOXX")
    ]
    minute = merge_intraday_bars(existing, pd.concat(fetched, ignore_index=True))
    write_intraday_cache(minute, cache_path)
    manifests: dict[str, Any] = {}
    fields = {
        f"${name}": name
        for name in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
            "bar_index",
            "bar_count",
        )
    }
    for freq in ("5min", "15min"):
        bars = resample_intraday_bars(minute, freq=freq, symbols=["SOXL", "SOXX"])
        bars["datetime"] = pd.to_datetime(bars["datetime"]).dt.tz_localize(None)
        manifests[freq] = build_qlib_store_from_frame(
            bars,
            provider_uri=root / "data" / "qlib" / f"soxl_intraday_{freq}",
            market="all",
            region="us",
            freq=freq,
            instrument_col="instrument",
            datetime_col="datetime",
            field_map=fields,
            as_of=session_date,
            source="data/processed/intraday/soxl_soxx_1m.parquet",
        )
    _update_intraday_registry(root, minute)
    return {
        "session_date": session_date,
        "cache_rows": int(len(minute)),
        "cache_end": str(pd.to_datetime(minute["timestamp"]).max()),
        "manifests": manifests,
    }


def _update_intraday_registry(root: Path, minute: pd.DataFrame) -> None:
    import yaml

    registry_path = root / "data" / "datasets.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    timestamps = pd.to_datetime(minute["timestamp"])
    registry.setdefault("intraday", {})["soxl_soxx_1m"] = {
        "exists": True,
        "path": "data/processed/intraday/soxl_soxx_1m.parquet",
        "start": timestamps.min().isoformat(),
        "end": timestamps.max().isoformat(),
        "rows": int(len(minute)),
        "intervals": ["1m"],
        "providers": ["futu"],
        "symbols": ["SOXL", "SOXX"],
    }
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )


def rebuild_normalized_provider(*, root: Path) -> dict[str, Any]:
    spec = DynamicSoxxSoxlCashAllocationWfSpec(root=root)
    raw, _ = _load_multi_3x_raw_panel(spec)
    frame = _attach_dynamic_allocation_features(raw)
    common_end_ts = pd.to_datetime(frame.groupby("instrument")["datetime"].max()).min()
    common_end = common_end_ts.date().isoformat()
    frame = frame.loc[pd.to_datetime(frame["datetime"]).le(common_end_ts)].copy()
    constituent, _ = _load_soxx_constituent_factor_frame(
        spec,
        start=spec.train_start,
        end=common_end,
    )
    frame, _ = _attach_soxx_constituent_model_features(frame, constituent)
    intraday, _ = _build_intraday_risk_state_daily(
        spec,
        start=spec.train_start,
        end=common_end,
    )
    frame = _attach_intraday_risk_state(frame, intraday)
    return _build_allocation_provider(spec, frame, as_of=common_end)


def refresh_stage1_data(
    *,
    root: Path | str | None = None,
    end: str | None = None,
    host: str = "127.0.0.1",
    port: int = 11111,
) -> dict[str, Any]:
    repo_root = _resolve_root(root)
    if not opend_reachable(host, port):
        raise RuntimeError(
            f"blocked: Futu OpenD is not reachable at {host}:{port}; "
            "market-data refresh was not started"
        )
    settled = end or _default_end_date()
    market = update_market_data(
        provider="futu",
        end=settled,
        refresh_overlap_days=5,
        replace_overlap=True,
        futu_host=host,
        futu_port=port,
    )
    multi = update_multi_3x_etf_data(
        end=settled,
        refresh_overlap_days=5,
        replace_overlap=True,
    )
    soxs = build_soxs_defensive_provider(root=repo_root, end=settled)
    intraday = refresh_intraday_store(
        root=repo_root,
        session_date=settled,
        host=host,
        port=port,
    )
    normalized = rebuild_normalized_provider(root=repo_root)
    return {
        "status": "ok",
        "settled_date": settled,
        "market": market,
        "multi_3x": multi,
        "soxs": soxs,
        "intraday": intraday,
        "normalized_provider": normalized,
    }


def detect_slot(now: datetime | None = None) -> str:
    current = now or datetime.now(tz=NEW_YORK_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=NEW_YORK_TZ)
    current = current.astimezone(NEW_YORK_TZ)
    if current.weekday() < 5 and time(9, 30) <= current.time() < time(9, 46):
        return "open"
    if current.weekday() < 5 and time(10, 30) <= current.time() < time(10, 46):
        return "recovery"
    if current.weekday() < 5 and current.time() >= time(20, 15):
        return "postclose"
    return "not_due"


def _production_execution_gate(root: Path) -> tuple[bool, str | None]:
    summary_path = root / "production" / "stage1_futu_sim_v1" / "summary.json"
    if not summary_path.exists():
        return False, "missing_production_summary"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("risk_gate_pass") is not True:
        return False, "failed_position_risk_gates"
    if summary.get("decision") != "paper_sim_enabled_waiting_next_us_session":
        return False, "production_decision_not_enabled"
    return True, None


def run_stage1_futu_task(
    *,
    root: Path | str | None = None,
    slot: str = "auto",
    dry_run: bool = False,
    skip_refresh: bool = False,
    notional: float = DEFAULT_NOTIONAL,
    host: str = "127.0.0.1",
    port: int = 11111,
    now: datetime | None = None,
) -> dict[str, Any]:
    repo_root = _resolve_root(root)
    selected_slot = detect_slot(now) if slot == "auto" else slot
    runtime_dir = repo_root / RUNTIME_RELATIVE
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if selected_slot == "not_due":
        result = {"status": "not_due", "slot": selected_slot}
        _record_run(runtime_dir, result, now=now, actionable=False)
        return result
    if selected_slot == "heartbeat":
        latest = _read_json(runtime_dir / "latest_actionable_run.json")
        if latest is None:
            latest = {
                "status": "needs_data",
                "slot": "unknown",
                "reason": "missing_latest_actionable_run",
            }
        result = {
            "status": "heartbeat",
            "slot": "daily_heartbeat",
            "latest_actionable_run": latest,
            "real_broker_order": False,
        }
        _record_run(runtime_dir, result, now=now, actionable=False)
        return result
    if selected_slot == "postclose":
        refresh = None if skip_refresh else refresh_stage1_data(
            root=repo_root,
            host=host,
            port=port,
        )
        card = run_stage1_futu_sim_card(root=repo_root)
        result = {
            "status": "ok",
            "slot": "postclose",
            "refresh": refresh,
            "card": {
                "decision": card["decision"],
                "recorder_id": card["recorder_id"],
                "latest_target_status": card["latest_target"]["status"],
                "latest_signal_date": card["latest_target"]["signal_date"],
            },
            "real_broker_order": False,
        }
        _record_run(runtime_dir, result, now=now)
        return result
    if selected_slot not in {"open", "recovery"}:
        raise ValueError(f"unsupported Stage1 Futu task slot: {selected_slot}")

    execution_allowed, gate_reason = _production_execution_gate(repo_root)
    if not execution_allowed:
        result = {
            "status": "blocked",
            "reason": gate_reason,
            "slot": selected_slot,
            "real_broker_order": False,
        }
        _record_run(runtime_dir, result, now=now)
        return result

    with FutuSimulatedBroker(host=host, port=port) as broker:
        prices = broker.quotes()
        session_date = (now or datetime.now(tz=NEW_YORK_TZ)).astimezone(
            NEW_YORK_TZ
        ).date().isoformat()
        settlement_prices = prices
        if selected_slot == "recovery":
            settlement_prices = _latest_open_settlement_prices(
                runtime_dir / "execution_state.json",
                session_date=session_date,
            )
            if settlement_prices is None:
                result = {
                    "status": "blocked",
                    "reason": "missing_same_day_open_mark",
                    "slot": selected_slot,
                    "real_broker_order": False,
                }
                _record_run(runtime_dir, result, now=now)
                return result
            refresh_intraday_store(
                root=repo_root,
                session_date=session_date,
                host=host,
                port=port,
            )
        target = build_runtime_target(
            root=repo_root,
            current_prices=settlement_prices,
            include_current_intraday=selected_slot == "recovery",
            execution_date=session_date,
        )
        if target["status"] != "ready":
            result = {
                "status": "blocked",
                "reason": target["status"],
                "slot": selected_slot,
                "target": target,
                "real_broker_order": False,
            }
        else:
            execution = broker.rebalance(
                weights=target["weights"],
                prices=prices,
                notional=notional,
                signal_date=target["signal_date"],
                slot=selected_slot,
                state_path=runtime_dir / "execution_state.json",
                dry_run=dry_run,
                now=now,
            )
            result = {
                "status": execution["status"],
                "slot": selected_slot,
                "target": target,
                "execution": execution,
                "real_broker_order": False,
            }
    _record_run(runtime_dir, result, now=now)
    return result


def _latest_open_settlement_prices(
    state_path: Path,
    *,
    session_date: str,
) -> dict[str, float] | None:
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        payload.get("environment") != "SIMULATE"
        or payload.get("real_trading_supported") is not False
    ):
        raise ValueError("blocked: invalid Futu simulated execution state")
    candidates = [
        value
        for key, value in payload.get("completed_slots", {}).items()
        if (
            str(key).endswith(":open")
            and isinstance(value, dict)
            and value.get("execution_session_date") == session_date
        )
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda value: str(value.get("recorded_at", "")))
    prices = latest.get("prices")
    if not isinstance(prices, dict) or set(prices) != {"SOXL", "SOXX"}:
        return None
    result = {symbol: float(prices[symbol]) for symbol in ("SOXL", "SOXX")}
    if any(value <= 0 for value in result.values()):
        return None
    return result


def _write_latest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _record_run(
    runtime_dir: Path,
    result: dict[str, Any],
    *,
    now: datetime | None = None,
    actionable: bool = True,
) -> dict[str, Any]:
    observed = now or datetime.now().astimezone()
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=SHANGHAI_TZ)
    recorded = dict(result)
    recorded["recorded_at"] = observed.astimezone(SHANGHAI_TZ).isoformat(timespec="seconds")
    _write_latest(runtime_dir / "latest_run.json", recorded)
    if actionable:
        _write_latest(runtime_dir / "latest_actionable_run.json", recorded)
    return recorded


def build_stage1_bark_message(
    result: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, str]:
    observed = now or datetime.now().astimezone()
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=SHANGHAI_TZ)
    shanghai = observed.astimezone(SHANGHAI_TZ)
    new_york = observed.astimezone(NEW_YORK_TZ)
    status = str(result.get("status", "unknown"))
    slot = str(result.get("slot", "unknown"))

    if status == "heartbeat":
        title = "SOXL 模拟盘每日心跳"
        latest = result.get("latest_actionable_run")
        if not isinstance(latest, dict):
            latest = {
                "status": "needs_data",
                "slot": "unknown",
                "reason": "missing_latest_actionable_run",
            }
        latest_slot = str(latest.get("slot", "unknown"))
        latest_status = str(latest.get("status", "unknown"))
        lines = [
            "调度心跳正常",
            f"最近任务 {latest_slot} / {latest_status}",
        ]
        latest_reason = latest.get("reason")
        if latest_reason:
            lines.append(f"原因 {latest_reason}")
        latest_at = latest.get("recorded_at")
        if latest_at:
            lines.append(f"最近记录 {latest_at}")
    else:
        title = f"SOXL 模拟盘 {status}"
        lines = [f"任务 {slot} / {status}"]
        reason = result.get("reason")
        if reason:
            lines.append(f"原因 {reason}")
        execution = result.get("execution")
        if isinstance(execution, dict):
            environment = str(execution.get("environment", "SIMULATE"))
            lines.append(f"环境 {environment}")
            orders = execution.get("orders", [])
            if isinstance(orders, list) and orders:
                order_text = "; ".join(
                    f"{order.get('side', '?')} {int(order.get('quantity', 0))} {order.get('symbol', '?')}"
                    for order in orders
                    if isinstance(order, dict)
                )
                lines.append(f"模拟委托 {order_text}")
            else:
                lines.append("模拟委托 无")
        target = result.get("target")
        if isinstance(target, dict):
            weights = target.get("weights")
            if isinstance(weights, dict) and {"SOXL", "SOXX", "cash"}.issubset(weights):
                lines.append(
                    "目标 "
                    f"SOXL {float(weights['SOXL']):.1%} / "
                    f"SOXX {float(weights['SOXX']):.1%} / "
                    f"现金 {float(weights['cash']):.1%}"
                )
            if target.get("signal_date"):
                lines.append(f"信号日 {target['signal_date']}")

    lines.extend(
        [
            f"上海 {shanghai.strftime('%Y-%m-%d %H:%M')} / 纽约 {new_york.strftime('%Y-%m-%d %H:%M')}",
            "真实下单=false",
        ]
    )
    return {
        "title": title,
        "body": "\n".join(lines),
        "group": "SOXL Stage1 模拟盘",
    }


def deliver_stage1_bark(
    result: dict[str, Any],
    *,
    dry_run: bool = False,
    notifier: BarkNotifier | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    active_notifier = notifier or BarkNotifier(
        BarkPushConfig.from_env(dry_run=dry_run)
    )
    message = build_stage1_bark_message(result, now=now)
    return active_notifier.send_message(**message)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Stage1 strategy in Futu paper trading.")
    parser.add_argument(
        "--slot",
        choices=["auto", "open", "recovery", "postclose", "heartbeat"],
        default="auto",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-dry-run", action="store_true")
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--notional", type=float, default=DEFAULT_NOTIONAL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11111)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_stage1_futu_task(
            slot=args.slot,
            dry_run=args.dry_run,
            skip_refresh=args.skip_refresh,
            notional=args.notional,
            host=args.host,
            port=args.port,
        )
    except Exception as error:
        result = {
            "status": "failed",
            "slot": args.slot,
            "reason": str(error),
            "real_broker_order": False,
        }
        runtime_dir = _resolve_root(None) / RUNTIME_RELATIVE
        _record_run(runtime_dir, result, actionable=True)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        if args.notify:
            try:
                deliver_stage1_bark(result, dry_run=args.notify_dry_run)
            except Exception as notify_error:
                print(f"Bark notification failed: {notify_error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.notify and result.get("status") != "not_due":
        try:
            push_result = deliver_stage1_bark(result, dry_run=args.notify_dry_run)
            print(json.dumps({"bark": push_result}, sort_keys=True, default=str))
        except Exception as error:
            print(f"Bark notification failed: {error}")
            return 3
    return 0 if result["status"] not in {"blocked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
