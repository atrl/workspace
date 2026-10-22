from __future__ import annotations

import argparse
import fcntl
import json
from datetime import datetime, time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from src.api import (
    FutuSimulatedBroker,
    activate_flat_stage1_cycle,
    load_stage1_cycle,
    opend_reachable,
    reconcile_stage1_ledger,
    write_stage1_cycle,
)
from src.api.futu_sim import reconcile_execution_state, record_hold_decision
from src.data import build_qlib_store_from_frame
from src.data.intraday import (
    fetch_futu_intraday_bars,
    merge_intraday_bars,
    read_intraday_cache,
    resample_intraday_bars,
    write_intraday_cache,
)
from src.data.market_update import (
    DEFAULT_PROVIDER_SEQUENCE,
    _default_end_date,
    _max_settled_us_daily_date,
    update_market_data,
)
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
    runtime_state_from_paper_ledger,
    run_stage1_futu_sim_card,
)
from src.workflow.intraday_recovery_direct_allocation import (
    IntradayRecoveryDirectAllocationSpec,
)
from src.schedule.notifications import BarkNotifier, BarkPushConfig


NEW_YORK_TZ = ZoneInfo("America/New_York")
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
RUNTIME_RELATIVE = Path("runtime/soxl/stage1_futu_sim")
DEFAULT_BROKER_RESERVE_CASH = 900_000.0
CYCLE_MANIFEST = "paper_cycle.json"


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    return Path(__file__).resolve().parents[2]


def refresh_intraday_store(
    *,
    root: Path,
    session_date: str,
    start_session_date: str | None = None,
    host: str = "127.0.0.1",
    port: int = 11111,
) -> dict[str, Any]:
    if ((root / "data/nas_sources.json").exists()
            or (root / "data/processed/intraday/soxl_soxx_none_1m.parquet").exists()):
        from src.data.minute_refresh import refresh_published_core

        result = refresh_published_core(root=root, session_date=session_date,
            start_session_date=start_session_date, host=host, port=port)
        _update_intraday_registry(root, read_intraday_cache(
            root / "data/processed/intraday/soxl_soxx_1m.parquet"))
        return result
    cache_path = root / "data" / "processed" / "intraday" / "soxl_soxx_1m.parquet"
    existing = read_intraday_cache(cache_path)
    starts = [session_date]
    if start_session_date is not None:
        starts.append(start_session_date)
    if not existing.empty:
        for symbol in ("SOXL", "SOXX"):
            dates = existing.loc[existing["symbol"].eq(symbol), "date"]
            if not dates.empty:
                starts.append(str(dates.max())[:10])
    fetch_start = min(starts)
    fetched = [
        fetch_futu_intraday_bars(
            symbol,
            start=fetch_start,
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
        "fetch_start": fetch_start,
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
    if "adjustment_type" in minute and minute.adjustment_type.eq("PUBLISHED_FUTU_AFFINE").all():
        registry["intraday"]["soxl_soxx_1m"].update(
            raw_path="data/processed/intraday/soxl_soxx_none_1m.parquet",
            adjustment="published_futu_affine_v1",
        )
    if (root / "data/nas_sources.json").exists():
        registry["intraday"]["soxl_soxx_1m"].update(
            authority="yanxuan_nas", source_index="sync.current/source_updates/us_semis_intraday",
            local_cache_disposable=True,
        )
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
    opend_ready = opend_reachable(host, port)
    settled = end or _default_end_date()
    market = update_market_data(
        provider="auto",
        provider_sequence=DEFAULT_PROVIDER_SEQUENCE,
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
    intraday = (
        refresh_intraday_store(
            root=repo_root,
            session_date=settled,
            start_session_date=market.get("old_last_date"),
            host=host,
            port=port,
        )
        if opend_ready
        else {"status": "needs_data", "reason": "opend_unreachable", "session_date": settled}
    )
    normalized = rebuild_normalized_provider(root=repo_root)
    cycle = _read_json(repo_root / RUNTIME_RELATIVE / CYCLE_MANIFEST)
    if cycle and cycle.get("marks_provider_uri"):
        from src.data.futu_account_marks import refresh_account_marks
        refresh_account_marks(root=repo_root, start=str(cycle["inception_date"]),
                              end=settled, host=host, port=port)
    freshness = _refresh_freshness(
        settled=settled,
        market=market,
        soxs=soxs,
        normalized=normalized,
        intraday=intraday,
    )
    return {
        "status": "ok" if freshness["fresh"] else "needs_data",
        "settled_date": settled,
        "opend_reachable": opend_ready,
        "market": market,
        "multi_3x": multi,
        "soxs": soxs,
        "intraday": intraday,
        "normalized_provider": normalized,
        "freshness": freshness,
    }


def _manifest_end(payload: dict[str, Any]) -> str | None:
    manifest = payload.get("manifest", payload)
    value = manifest.get("calendar_end") if isinstance(manifest, dict) else None
    return str(value)[:10] if value else None


def _refresh_freshness(
    *,
    settled: str,
    market: dict[str, Any],
    soxs: dict[str, Any],
    normalized: dict[str, Any],
    intraday: dict[str, Any],
) -> dict[str, Any]:
    market_end = str(market.get("new_last_date") or "")[:10]
    soxs_manifest = soxs.get("qlib_manifest", soxs)
    soxs_end = _manifest_end(soxs_manifest) if isinstance(soxs_manifest, dict) else None
    normalized_end = _manifest_end(normalized)
    intraday_end = None
    manifests = intraday.get("manifests") if isinstance(intraday, dict) else None
    if isinstance(manifests, dict) and isinstance(manifests.get("15min"), dict):
        intraday_end = _manifest_end(manifests["15min"])
    checks = {
        "market": market_end == settled,
        "soxs": soxs_end == settled,
        "normalized": normalized_end == settled,
        "intraday": intraday_end == settled,
    }
    return {
        "fresh": all(checks.values()),
        "required_settled_date": settled,
        "market_end": market_end or None,
        "soxs_end": soxs_end,
        "normalized_end": normalized_end,
        "intraday_end": intraday_end,
        "checks": checks,
    }


def detect_slot(now: datetime | None = None) -> str:
    current = now or datetime.now(tz=NEW_YORK_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=NEW_YORK_TZ)
    current = current.astimezone(NEW_YORK_TZ)
    if current.weekday() < 5 and time(8, 0) <= current.time() < time(9, 25):
        return "postclose"
    if current.weekday() < 5 and time(9, 31) <= current.time() < time(9, 46):
        return "open"
    if current.weekday() < 5 and time(10, 31) <= current.time() < time(10, 46):
        return "recovery"
    if current.weekday() < 5 and current.time() >= time(20, 15):
        return "postclose"
    return "not_due"


def manual_start_context(now: datetime | None = None) -> dict[str, Any]:
    """Resolve a one-time manual paper start without pretending it is an open fill."""
    current = now or datetime.now(tz=NEW_YORK_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=NEW_YORK_TZ)
    current = current.astimezone(NEW_YORK_TZ)
    regular_session = (
        current.weekday() < 5 and time(9, 30) <= current.time() < time(16, 0)
    )
    return {
        "requested_at": current.isoformat(timespec="seconds"),
        "session_date": current.date().isoformat(),
        "regular_session": regular_session,
        "include_current_intraday": regular_session and current.time() >= time(10, 30),
        "price_mark": "live_start_quote" if regular_session else None,
    }


def _same_session_execution(
    state_path: Path,
    *,
    session_date: str,
) -> dict[str, Any] | None:
    payload = _read_json(state_path)
    if payload is None:
        return None
    if (
        payload.get("environment") != "SIMULATE"
        or payload.get("real_trading_supported") is not False
    ):
        raise ValueError("blocked: invalid Futu simulated execution state")
    candidates: list[dict[str, Any]] = []
    for bucket_name in ("completed_slots", "pending_slots"):
        bucket = payload.get(bucket_name, {})
        if not isinstance(bucket, dict):
            raise ValueError("blocked: invalid Futu simulated execution slots")
        for value in bucket.values():
            if not isinstance(value, dict):
                continue
            if (
                value.get("execution_session_date") == session_date
                and value.get("slot") in {"open", "recovery", "start_now"}
            ):
                candidates.append(value)
    if not candidates:
        return None
    return max(candidates, key=lambda value: str(value.get("recorded_at", "")))


def _production_execution_gate(
    root: Path,
    *,
    target: dict[str, Any] | None = None,
    required_signal_date: str | None = None,
    paper_trial_authorized: bool = False,
) -> tuple[str, str | None]:
    summary_path = root / "production" / "stage1_futu_sim_v1" / "summary.json"
    if not summary_path.exists():
        return "blocked", "missing_production_summary"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    active_target = target or summary.get("latest_target")
    if not isinstance(active_target, dict) or active_target.get("status") != "ready":
        return "blocked", "missing_ready_target"
    source_audit = active_target.get("source_audit", {})
    market_audit = source_audit.get("market_audit", {}) if isinstance(source_audit, dict) else {}
    feature_end = str(market_audit.get("common_feature_end") or "")[:10]
    signal_date = str(active_target.get("signal_date") or "")[:10]
    fresh = (
        required_signal_date is None
        or (feature_end >= required_signal_date and signal_date >= required_signal_date)
    )
    if not fresh:
        return "blocked", "stale_qlib_target"
    if paper_trial_authorized:
        if active_target.get("controller_state_source") != "paper_ledger_cycle":
            return "blocked", "paper_trial_requires_live_cycle_controller"
        return "normal", "user_authorized_restarted_paper_trial"
    if summary.get("risk_gate_pass") is not True:
        return "blocked", "failed_position_risk_gates"
    if summary.get("decision") != "paper_sim_enabled_waiting_next_us_session":
        return "blocked", "production_decision_not_enabled"
    return "normal", None


def _cycle_runtime_paths(
    runtime_dir: Path,
    cycle: dict[str, Any],
) -> tuple[Path, Path]:
    return (
        runtime_dir / str(cycle["ledger_file"]),
        runtime_dir / str(cycle["execution_state_file"]),
    )


def _cycle_ledger(
    *,
    repo_root: Path,
    runtime_dir: Path,
    cycle: dict[str, Any],
    snapshot: dict[str, Any],
    prices: dict[str, float],
    history: list[dict[str, Any]],
    now: datetime | None,
) -> dict[str, Any]:
    ledger_path, execution_state_path = _cycle_runtime_paths(runtime_dir, cycle)
    reconcile_execution_state(execution_state_path, history)
    return reconcile_stage1_ledger(
        path=ledger_path,
        initial_equity=float(cycle["initial_equity"]),
        broker_reserve_cash=float(cycle["broker_reserve_cash"]),
        inception_date=str(cycle["inception_date"]),
        snapshot=snapshot,
        orders=history,
        prices=prices,
        as_of=(now or datetime.now(tz=NEW_YORK_TZ)).isoformat(),
        settled_marks=_load_qlib_settled_marks(
            repo_root,
            start=str(cycle["inception_date"]),
            end=settled_date(now),
            provider_uri=(repo_root / str(cycle["marks_provider_uri"])) if cycle.get("marks_provider_uri") else None,
        ),
        remark_prefix=str(cycle["remark_prefix"]) + "_",
        cycle_id=str(cycle["cycle_id"]),
        cash_adjustment_confirmation=(
            _read_json(runtime_dir / str(cycle["cash_adjustment_confirmation_file"]))
            if cycle.get("cash_adjustment_confirmation_file") else None
        ),
    )


def _restart_cycle_if_needed(
    *,
    runtime_dir: Path,
    cycle: dict[str, Any],
    broker: FutuSimulatedBroker,
    snapshot: dict[str, Any],
    prices: dict[str, float],
    history: list[dict[str, Any]],
    session_date: str,
    slot: str,
    dry_run: bool,
    now: datetime | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if cycle["status"] == "active":
        return cycle, None
    if cycle["status"] == "blocked":
        return cycle, {
            "status": "blocked",
            "slot": slot,
            "reason": "paper_cycle_blocked",
            "cycle_id": cycle["cycle_id"],
            "cycle_status": cycle["status"],
            "real_broker_order": False,
        }
    if session_date < str(cycle["start_date"]):
        return cycle, {
            "status": "restart_armed",
            "slot": slot,
            "reason": f"waiting_for_{cycle['start_date']}_us_session",
            "cycle_id": cycle["cycle_id"],
            "cycle_status": cycle["status"],
            "real_broker_order": False,
        }

    positions = {
        symbol: int(snapshot["positions"].get(symbol, 0))
        for symbol in ("SOXL", "SOXX")
    }
    execution_state_path = runtime_dir / str(cycle["execution_state_file"])
    reconcile_execution_state(execution_state_path, history)
    if not any(positions.values()):
        activated = activate_flat_stage1_cycle(
            cycle,
            snapshot=snapshot,
            session_date=session_date,
        )
        activated = write_stage1_cycle(runtime_dir / CYCLE_MANIFEST, activated)
        return activated, None

    if cycle["status"] == "flatten_pending":
        return cycle, {
            "status": "restart_waiting_fill",
            "slot": slot,
            "reason": "tracked_positions_not_flat_yet",
            "cycle_id": cycle["cycle_id"],
            "cycle_status": cycle["status"],
            "positions": positions,
            "real_broker_order": False,
        }

    sleeve_equity = (
        float(snapshot["cash"])
        - float(cycle["broker_reserve_cash"])
        + sum(positions[symbol] * float(prices[symbol]) for symbol in positions)
    )
    execution = broker.rebalance(
        weights={"SOXL": 0.0, "SOXX": 0.0, "cash": 1.0},
        prices=prices,
        portfolio_equity=sleeve_equity,
        signal_date=str(cycle["start_date"]),
        slot="restart_flatten",
        state_path=execution_state_path,
        dry_run=dry_run,
        execution_mode="normal",
        current_quantities=positions,
        now=now,
        remark_prefix=str(cycle.get("restart_remark_prefix", "stage1_restart")),
    )
    if not dry_run:
        cycle = write_stage1_cycle(
            runtime_dir / CYCLE_MANIFEST,
            dict(cycle) | {"status": "flatten_pending"},
        )
    return cycle, {
        "status": execution["status"],
        "slot": "restart_flatten",
        "reason": "new_paper_cycle_flattening_legacy_positions",
        "cycle_id": cycle["cycle_id"],
        "cycle_status": cycle["status"],
        "execution": execution,
        "real_broker_order": False,
    }


def run_stage1_futu_task(
    *,
    root: Path | str | None = None,
    slot: str = "auto",
    dry_run: bool = False,
    skip_refresh: bool = False,
    notional: float = DEFAULT_NOTIONAL,
    broker_reserve_cash: float = DEFAULT_BROKER_RESERVE_CASH,
    host: str = "127.0.0.1",
    port: int = 11111,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Serialize reconciliation and execution; retry due work after a missed tick."""
    repo_root = _resolve_root(root)
    runtime_dir = repo_root / RUNTIME_RELATIVE
    runtime_dir.mkdir(parents=True, exist_ok=True)
    selected_slot = detect_slot(now) if slot == "auto" else slot
    if selected_slot == "not_due":
        return {"status": "not_due", "slot": selected_slot}
    with (runtime_dir / "task.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "status": "blocked", "slot": selected_slot,
                "reason": "stage1_task_already_running", "real_broker_order": False,
            }
        completed_path = runtime_dir / "latest_postclose_run.json"
        if slot == "auto" and selected_slot == "postclose":
            completed = _read_json(completed_path)
            if completed and completed.get("settled_date") == settled_date(now):
                return {
                    "status": "no_action", "slot": selected_slot,
                    "reason": "settled_data_already_refreshed", "real_broker_order": False,
                }
        result = _run_stage1_futu_task(
            root=repo_root, slot=selected_slot, dry_run=dry_run,
            skip_refresh=skip_refresh, notional=notional,
            broker_reserve_cash=broker_reserve_cash, host=host, port=port, now=now,
        )
        if selected_slot == "postclose" and result.get("status") == "ok" and not skip_refresh:
            _write_latest(completed_path, {**result, "settled_date": settled_date(now)})
        return result


def _run_stage1_futu_task(
    *,
    root: Path | str | None = None,
    slot: str = "auto",
    dry_run: bool = False,
    skip_refresh: bool = False,
    notional: float = DEFAULT_NOTIONAL,
    broker_reserve_cash: float = DEFAULT_BROKER_RESERVE_CASH,
    host: str = "127.0.0.1",
    port: int = 11111,
    now: datetime | None = None,
) -> dict[str, Any]:
    repo_root = _resolve_root(root)
    selected_slot = detect_slot(now) if slot == "auto" else slot
    if selected_slot in {"start-now", "start_now"}:
        selected_slot = "start_now"
    runtime_dir = repo_root / RUNTIME_RELATIVE
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cycle_path = runtime_dir / CYCLE_MANIFEST
    cycle = load_stage1_cycle(cycle_path) if cycle_path.exists() else None
    if cycle is not None and cycle.get("status") != "active":
        cycle = dict(cycle)
        cycle.setdefault("target_initial_equity", float(notional))
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
        staleness = detect_execution_outage(runtime_dir=runtime_dir, now=now)
        result = {
            "status": "heartbeat",
            "slot": "daily_heartbeat",
            "latest_actionable_run": latest,
            "staleness": staleness,
            "cycle_id": cycle.get("cycle_id") if cycle else None,
            "cycle_status": cycle.get("status") if cycle else "missing",
            "real_broker_order": False,
        }
        _record_run(runtime_dir, result, now=now, actionable=False)
        return result
    if selected_slot == "postclose":
        refresh_stamp = runtime_dir / "latest_data_refresh.json"
        previous_refresh = _read_json(refresh_stamp)
        if skip_refresh:
            refresh = None
        elif previous_refresh and previous_refresh.get("settled_date") == settled_date(now):
            refresh = {"status": "ok", "settled_date": settled_date(now), "reused": True}
        else:
            refresh = refresh_stage1_data(root=repo_root, host=host, port=port)
            if refresh.get("status") == "ok":
                _write_latest(refresh_stamp, {"settled_date": settled_date(now), "status": "ok"})
        card_error = None
        try:
            release_path = repo_root / "production/stage1_futu_sim_v1/runtime_release.json"
            if release_path.exists():
                from src.workflow.stage1_runtime_release import load_runtime_release
                released = load_runtime_release(repo_root, release_path)
                card = {
                    "decision": (
                        "paper_trial_not_promoted_waiting_next_open"
                        if released.get("paper_trial_authorization")
                        else "validated_paper_release_waiting_next_open"
                    ),
                    "recorder_id": released["action_summary"]["recorder_id"],
                    "latest_target": {"status": "pending_next_open_mark", "signal_date": settled_date(now)},
                }
            else:
                card = run_stage1_futu_sim_card(root=repo_root)
        except (FileNotFoundError, ValueError) as error:
            card_error = str(error)
            card = {"decision": "blocked", "recorder_id": None,
                    "latest_target": {"status": "blocked", "signal_date": None}}
        ledger = None
        if opend_reachable(host, port) and cycle and cycle.get("status") == "active":
            with FutuSimulatedBroker(host=host, port=port) as broker:
                prices = broker.quotes()
                snapshot = broker.account_snapshot()
                history = broker.order_history(
                    start=str(cycle["inception_date"]),
                    end=settled_date(now),
                )
                ledger = _cycle_ledger(
                    repo_root=repo_root,
                    runtime_dir=runtime_dir,
                    cycle=cycle,
                    snapshot=snapshot,
                    prices=prices,
                    history=history,
                    now=now,
                )
        result = {
            "status": "blocked" if card_error else ("ok" if refresh is None or refresh.get("status") == "ok" else "needs_data"),
            "reason": card_error,
            "slot": "postclose",
            "refresh": refresh,
            "card": {
                "decision": card["decision"],
                "recorder_id": card["recorder_id"],
                "latest_target_status": card["latest_target"]["status"],
                "latest_signal_date": card["latest_target"]["signal_date"],
            },
            "ledger": _ledger_summary(ledger) if ledger else None,
            "cycle_id": cycle.get("cycle_id") if cycle else None,
            "cycle_status": cycle.get("status") if cycle else "missing",
            "real_broker_order": False,
        }
        _record_run(runtime_dir, result, now=now)
        return result
    if selected_slot not in {"open", "recovery", "start_now"}:
        raise ValueError(f"unsupported Stage1 Futu task slot: {selected_slot}")
    if cycle is None:
        result = {
            "status": "blocked",
            "slot": selected_slot,
            "reason": "missing_paper_cycle_manifest",
            "real_broker_order": False,
        }
        _record_run(runtime_dir, result, now=now)
        return result

    manual_context = manual_start_context(now) if selected_slot == "start_now" else None
    if manual_context is not None and not manual_context["regular_session"]:
        result = {
            "status": "armed",
            "slot": selected_slot,
            "reason": "waiting_for_us_regular_session",
            "manual_start": manual_context,
            "next_action": "scheduled_open_or_run_start_now_during_regular_session",
            "cycle_id": cycle["cycle_id"],
            "cycle_status": cycle["status"],
            "real_broker_order": False,
        }
        _record_run(runtime_dir, result, now=now, actionable=False)
        return result

    if manual_context is not None and cycle.get("status") == "active":
        _, state_path = _cycle_runtime_paths(runtime_dir, cycle)
        existing = _same_session_execution(
            state_path,
            session_date=str(manual_context["session_date"]),
        )
        if existing is not None:
            result = {
                "status": "no_action",
                "slot": selected_slot,
                "reason": "paper_cycle_already_started_this_session",
                "manual_start": manual_context,
                "existing_execution": {
                    key: existing.get(key)
                    for key in (
                        "slot",
                        "signal_date",
                        "status",
                        "recorded_at",
                        "target_quantities",
                    )
                },
                "cycle_id": cycle["cycle_id"],
                "cycle_status": cycle["status"],
                "real_broker_order": False,
            }
            _record_run(runtime_dir, result, now=now, actionable=False)
            return result

    with FutuSimulatedBroker(host=host, port=port) as broker:
        if manual_context is not None and not dry_run:
            market_state = broker.us_market_state()
            if market_state not in {"MORNING", "AFTERNOON"}:
                result = {
                    "status": "armed",
                    "slot": selected_slot,
                    "reason": "futu_us_regular_session_not_open",
                    "manual_start": manual_context | {"futu_market_state": market_state},
                    "next_action": "scheduled_open_or_run_start_now_during_regular_session",
                    "cycle_id": cycle["cycle_id"],
                    "cycle_status": cycle["status"],
                    "real_broker_order": False,
                }
                _record_run(runtime_dir, result, now=now, actionable=False)
                return result
        prices = broker.quotes()
        snapshot = broker.account_snapshot()
        if (repo_root / "production/stage1_futu_sim_v1/runtime_release.json").exists():
            quote_error = _execution_quote_error(getattr(broker, "last_quote_as_of", None), now=now)
            if quote_error:
                result = {"status": "needs_data", "slot": selected_slot,
                          "reason": quote_error, "real_broker_order": False,
                          "quote_as_of": getattr(broker, "last_quote_as_of", None)}
                _record_run(runtime_dir, result, now=now, actionable=not dry_run)
                return result
        session_date = (now or datetime.now(tz=NEW_YORK_TZ)).astimezone(
            NEW_YORK_TZ
        ).date().isoformat()
        history = (
            []
            if session_date < str(cycle["start_date"])
            else broker.order_history(start=str(cycle["start_date"]), end=session_date)
        )
        cycle, restart_result = _restart_cycle_if_needed(
            runtime_dir=runtime_dir,
            cycle=cycle,
            broker=broker,
            snapshot=snapshot,
            prices=prices,
            history=history,
            session_date=session_date,
            slot=selected_slot,
            dry_run=dry_run,
            now=now,
        )
        if restart_result is not None:
            _record_run(runtime_dir, restart_result, now=now, actionable=not dry_run)
            return restart_result

        history = broker.order_history(start=str(cycle["inception_date"]), end=session_date)
        ledger = _cycle_ledger(
            repo_root=repo_root,
            runtime_dir=runtime_dir,
            cycle=cycle,
            snapshot=snapshot,
            prices=prices,
            history=history,
            now=now,
        )
        settlement_prices = prices
        settlement_as_of = getattr(broker, "last_quote_as_of", None)
        open_decision_context = None
        include_current_intraday = selected_slot == "recovery" or bool(
            manual_context and manual_context["include_current_intraday"]
        )
        if selected_slot == "recovery":
            _, cycle_execution_state_path = _cycle_runtime_paths(runtime_dir, cycle)
            open_record = _latest_open_record(cycle_execution_state_path, session_date=session_date)
            if open_record:
                open_decision_context = open_record.get("decision_context")
                settlement_as_of = (open_decision_context or {}).get("quote_as_of") or open_record.get("recorded_at")
            settlement_prices = _latest_open_settlement_prices(
                cycle_execution_state_path,
                session_date=session_date,
            )
            if settlement_prices is None:
                if (
                    str(cycle.get("inception_date")) == session_date
                    and not any(int(value) for value in ledger["positions"].values())
                ):
                    settlement_prices = prices
                else:
                    result = {
                        "status": "blocked",
                        "reason": "missing_same_day_open_mark",
                        "slot": selected_slot,
                        "ledger": _ledger_summary(ledger),
                        "cycle_id": cycle["cycle_id"],
                        "real_broker_order": False,
                    }
                    _record_run(runtime_dir, result, now=now)
                    return result
        if include_current_intraday:
            refresh_intraday_store(
                root=repo_root,
                session_date=session_date,
                host=host,
                port=port,
            )
        try:
            target = build_runtime_target(
                root=repo_root,
                current_prices=settlement_prices,
                current_prices_as_of=settlement_as_of,
                open_decision_context=open_decision_context,
                include_current_intraday=include_current_intraday,
                execution_date=session_date,
                live_controller_state=runtime_state_from_paper_ledger(
                    ledger,
                    recovery_spec=IntradayRecoveryDirectAllocationSpec(root=repo_root),
                ),
            )
        except Exception as error:
            target = {
                "status": "blocked",
                "reason": str(error),
                "controller_state_source": "paper_ledger_cycle_failed",
            }
        required_signal_date = settled_date(now)
        execution_mode, gate_reason = _production_execution_gate(
            repo_root,
            target=target,
            required_signal_date=required_signal_date,
            paper_trial_authorized=bool(cycle.get("user_authorized_paper_trial")),
        )
        if target["status"] != "ready":
            result = {
                "status": "blocked",
                "reason": target.get("reason", target["status"]),
                "slot": selected_slot,
                "target": target,
                "ledger": _ledger_summary(ledger),
                "cycle_id": cycle["cycle_id"],
                "cycle_status": cycle["status"],
                "real_broker_order": False,
            }
        elif execution_mode == "blocked":
            result = {
                "status": "blocked",
                "reason": gate_reason,
                "slot": selected_slot,
                "target": target,
                "ledger": _ledger_summary(ledger),
                "cycle_id": cycle["cycle_id"],
                "cycle_status": cycle["status"],
                "real_broker_order": False,
            }
        elif selected_slot == "recovery" and not bool(
            target.get("recovery_action")
        ):
            result = {
                "status": "no_action",
                "slot": selected_slot,
                "target": target,
                "execution_mode": execution_mode,
                "reason": "recovery_signal_not_triggered",
                "ledger": _ledger_summary(ledger),
                "cycle_id": cycle["cycle_id"],
                "cycle_status": cycle["status"],
                "real_broker_order": False,
            }
        elif target.get("decision") == "hold":
            if not dry_run:
                record_hold_decision(
                    state_path=_cycle_runtime_paths(runtime_dir, cycle)[1],
                    signal_date=target["signal_date"], slot=selected_slot,
                    prices=prices, quantities=ledger["positions"],
                    decision_context=target["decision_context"], now=now,
                )
            result = {
                "status": "no_action", "slot": selected_slot,
                "reason": "hold_model_rejected_uneconomic_rebalance",
                "target": target, "ledger": _ledger_summary(ledger),
                "cycle_id": cycle["cycle_id"], "cycle_status": cycle["status"],
                "real_broker_order": False,
            }
        else:
            execution = broker.rebalance(
                weights=target["weights"],
                prices=prices,
                portfolio_equity=float(ledger["equity"]),
                signal_date=target["signal_date"],
                slot=selected_slot,
                state_path=_cycle_runtime_paths(runtime_dir, cycle)[1],
                dry_run=dry_run,
                execution_mode=execution_mode,
                current_quantities=ledger["positions"],
                now=now,
                remark_prefix=str(cycle["remark_prefix"]),
                decision_context=target.get("decision_context"),
            )
            result = {
                "status": execution["status"],
                "slot": selected_slot,
                "manual_start": manual_context,
                "target": target,
                "execution": execution,
                "execution_mode": execution_mode,
                "reason": gate_reason,
                "ledger": _ledger_summary(ledger),
                "cycle_id": cycle["cycle_id"],
                "cycle_status": cycle["status"],
                "real_broker_order": False,
            }
    _record_run(runtime_dir, result, now=now, actionable=not dry_run)
    return result


def settled_date(now: datetime | None = None) -> str:
    return _max_settled_us_daily_date(now=now).isoformat()


def _execution_quote_error(as_of: str | None, *, now: datetime | None = None) -> str | None:
    if not as_of:
        return "missing_execution_quote_timestamp"
    try:
        quote_time = datetime.fromisoformat(as_of)
    except (ValueError, TypeError):
        return "invalid_execution_quote_timestamp"
    if quote_time.tzinfo is None:
        return "undated_execution_quote_timezone"
    current = now or datetime.now(tz=NEW_YORK_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=NEW_YORK_TZ)
    age = (current - quote_time).total_seconds()
    return "stale_or_future_execution_quote" if age > 120 or age < -5 else None


def _stored_target(root: Path) -> dict[str, Any]:
    summary = _read_json(root / "production" / "stage1_futu_sim_v1" / "summary.json") or {}
    target = summary.get("latest_target")
    if not isinstance(target, dict):
        return {"status": "needs_data"}
    return target


def _load_qlib_settled_marks(root: Path, *, start: str, end: str, provider_uri: Path | None = None) -> list[dict[str, Any]]:
    import qlib
    from qlib.constant import REG_US
    from qlib.data import D

    if start > end:
        return []
    provider_uri = provider_uri or root / "data" / "qlib" / "soxl"
    qlib.init(provider_uri=provider_uri.resolve().as_posix(), region=REG_US, kernels=1, joblib_backend="threading")
    frame = D.features(
        ["SOXL", "SOXX"],
        ["$close"],
        start_time=start,
        end_time=end,
        freq="day",
    )
    if frame.empty:
        raise RuntimeError("blocked: qlib settled marks are unavailable")
    values = frame.reset_index().pivot(index="datetime", columns="instrument", values="$close")
    values = values.dropna(subset=["SOXL", "SOXX"])
    if values.empty or values.index.max().date().isoformat() != end:
        raise RuntimeError("blocked: qlib account marks do not cover the latest settled session")
    return [
        {"date": index.date().isoformat(), "SOXL": row["SOXL"], "SOXX": row["SOXX"]}
        for index, row in values.iterrows()
    ]


def _ledger_summary(ledger: dict[str, Any] | None) -> dict[str, Any] | None:
    if ledger is None:
        return None
    return {
        key: ledger[key]
        for key in ("cash", "positions", "equity", "peak_equity", "drawdown", "total_inferred_fees")
    }


def _latest_open_settlement_prices(
    state_path: Path,
    *,
    session_date: str,
) -> dict[str, float] | None:
    latest = _latest_open_record(state_path, session_date=session_date)
    if latest is None:
        return None
    prices = latest.get("prices")
    if not isinstance(prices, dict) or set(prices) != {"SOXL", "SOXX"}:
        return None
    result = {symbol: float(prices[symbol]) for symbol in ("SOXL", "SOXX")}
    return result if all(value > 0 for value in result.values()) else None


def _latest_open_record(state_path: Path, *, session_date: str) -> dict[str, Any] | None:
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
    return max(candidates, key=lambda value: str(value.get("recorded_at", "")))


def detect_execution_outage(
    *,
    runtime_dir: Path,
    now: datetime | None = None,
    max_idle_sessions: int = 3,
) -> dict[str, Any]:
    """Report how long the simulated cycle has gone without a confirmed fill.

    A service outage is silent in the ledger: the account keeps marking to
    market while no rebalance runs, so the observed equity is not strategy
    evidence. This helper turns that silent gap into an explicit, auditable
    staleness record so the heartbeat can alert instead of reporting healthy.
    """
    observed = now or datetime.now().astimezone()
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=SHANGHAI_TZ)
    try:
        cycle = load_stage1_cycle(runtime_dir / CYCLE_MANIFEST)
    except (RuntimeError, ValueError):
        cycle = None
    if cycle is None:
        raw_cycle = _read_json(runtime_dir / CYCLE_MANIFEST)
        cycle = raw_cycle if isinstance(raw_cycle, dict) else {}
    ledger_path = runtime_dir / (cycle or {}).get("ledger_file", "paper_ledger_v2.json")
    ledger = _read_json(ledger_path)
    result: dict[str, Any] = {
        "cycle_id": (cycle or {}).get("cycle_id"),
        "ledger_file": ledger_path.name,
        "max_idle_sessions": int(max_idle_sessions),
        "status": "unknown",
    }
    if not isinstance(ledger, dict) or not ledger.get("fills"):
        result.update(status="no_fills", reason="ledger_missing_or_empty")
        return result
    fills = sorted(ledger["fills"], key=lambda value: str(value.get("create_time", "")))
    last_fill = fills[-1]
    last_time = str(last_fill.get("create_time", ""))
    try:
        last_dt = pd.Timestamp(last_time)
    except (TypeError, ValueError):
        result.update(status="unknown", reason="unparseable_last_fill", last_fill_at=last_time)
        return result
    if last_dt.tzinfo is None:
        last_dt = last_dt.tz_localize(SHANGHAI_TZ)
    idle_days = float((observed - last_dt).total_seconds() / 86400.0)
    # Business sessions idle, ignoring weekends, is a closer proxy for missed runs.
    idle_business = int(pd.bdate_range(last_dt.date(), observed.date()).size) - 1
    stale = idle_business > int(max_idle_sessions)
    result.update({
        "status": "stale" if stale else "fresh",
        "last_fill_at": last_time,
        "last_fill_symbol": last_fill.get("symbol"),
        "idle_calendar_days": round(idle_days, 2),
        "idle_business_days": idle_business,
        "alert": stale,
    })
    if stale:
        result["reason"] = (
            f"no_confirmed_fill_for_{idle_business}_business_days"
        )
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
    mobile_summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    observed = now or datetime.now().astimezone()
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=SHANGHAI_TZ)
    shanghai = observed.astimezone(SHANGHAI_TZ)
    new_york = observed.astimezone(NEW_YORK_TZ)
    status = str(result.get("status", "unknown"))
    slot = str(result.get("slot", "unknown"))

    if status == "heartbeat":
        title = "SOXL 模拟盘每日同步"
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
        _append_stage1_bark_run_details(lines, latest)
        staleness = result.get("staleness")
        if isinstance(staleness, dict):
            if staleness.get("alert"):
                title = "⚠ SOXL 模拟盘调度中断"
                lines.insert(0, "警报：超过阈值无确认成交")
                lines.append(
                    f"距上次成交 {staleness.get('idle_business_days')} 个交易日"
                )
                if staleness.get("last_fill_at"):
                    lines.append(f"上次成交 {staleness['last_fill_at']}")
            elif staleness.get("status") == "no_fills":
                lines.append("警告：尚无确认成交记录")
    else:
        title = f"SOXL 模拟盘 {status}"
        lines = [f"任务 {slot} / {status}"]
        _append_stage1_bark_run_details(lines, result)

    if result.get("cycle_id"):
        lines.append(
            f"模拟周期 {result['cycle_id']} / {result.get('cycle_status', 'unknown')}"
        )

    if mobile_summary and mobile_summary.get("status") == "ready":
        lines.append(f"账本截至 {mobile_summary.get('as_of') or '未知'}")
        lines.append(
            f"模拟净值 ${_format_money(mobile_summary['equity'])} / "
            f"回撤 {float(mobile_summary['drawdown_pct']):.2f}%"
        )
        performance = mobile_summary.get("performance")
        if performance:
            returns = performance["returns_pct"]
            lines.append(
                f"同窗 {performance['start']}~{performance['end']} "
                f"Stage1 {returns['Stage1']:+.2f}% / SOXX {returns['SOXX']:+.2f}% / SOXL {returns['SOXL']:+.2f}%"
            )
            lines.append(f"相对 SOXX {performance['difference_vs_soxx_pp']:+.2f}pp（诊断值）")
        for fill in mobile_summary.get("latest_by_side") or []:
            lines.append(f"最近成交 {fill['time']} 纽约 {fill['side']} {fill['quantity']} {fill['symbol']}")

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


def _append_stage1_bark_run_details(
    lines: list[str],
    result: dict[str, Any],
) -> None:
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
    execution_mode = result.get("execution_mode")
    if execution_mode:
        lines.append(f"执行模式 {execution_mode}")
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
        if target.get("status"):
            lines.append(f"推理状态 {target['status']}")
    ledger = result.get("ledger")
    if isinstance(ledger, dict):
        positions = ledger.get("positions", {})
        if isinstance(positions, dict):
            lines.append(
                f"持仓 SOXL {int(positions.get('SOXL', 0))} / "
                f"SOXX {int(positions.get('SOXX', 0))}"
            )
        if ledger.get("equity") is not None and ledger.get("drawdown") is not None:
            lines.append(
                f"策略净值 ${_format_money(ledger['equity'])} / "
                f"真实回撤 {float(ledger['drawdown']):.2%}"
            )


def _format_money(value: Any) -> str:
    rounded = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{rounded:,.2f}"


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
    from src.dashboard.stage1_mobile import build_stage1_mobile_summary

    message = build_stage1_bark_message(
        result, now=now, mobile_summary=build_stage1_mobile_summary(_resolve_root(None))
    )
    return active_notifier.send_message(**message)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Stage1 strategy in Futu paper trading.")
    parser.add_argument(
        "--slot",
        choices=["auto", "start-now", "open", "recovery", "postclose", "heartbeat"],
        default="auto",
        help=(
            "Use start-now for a one-time idempotent SIMULATE start at any wall-clock "
            "time; outside US regular hours it arms the next scheduled open."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--notify-dry-run", action="store_true")
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--notional", type=float, default=DEFAULT_NOTIONAL)
    parser.add_argument(
        "--broker-reserve-cash",
        type=float,
        default=DEFAULT_BROKER_RESERVE_CASH,
        help="Cash in the host Futu SIMULATE account that is outside the Stage1 sleeve.",
    )
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
            broker_reserve_cash=args.broker_reserve_cash,
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
