"""Collect one read-only economic-policy observation per US opening session."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def in_open_window(now: datetime) -> bool:
    if now.tzinfo is None:
        raise ValueError("timezone-aware observation clock required")
    local = now.astimezone(NY)
    return local.weekday() < 5 and 571 <= local.hour * 60 + local.minute < 577


def read_open_quotes(*, host: str, port: int):
    from futu import OpenQuoteContext, RET_OK
    context = OpenQuoteContext(host=host, port=port)
    try:
        status, state = context.get_global_state()
        if status != RET_OK or str(state.get("market_us", "")).upper() != "MORNING":
            raise ValueError("needs_data: US regular morning session is not open")
        status, frame = context.get_market_snapshot(["US.SOXL", "US.SOXX"])
        if status != RET_OK:
            raise ValueError("needs_data: quote snapshot failed")
        prices, stamps = {}, []
        for row in frame.to_dict("records"):
            if row["code"] not in ("US.SOXL", "US.SOXX"):
                continue
            prices[row["code"].removeprefix("US.")] = float(row["last_price"])
            stamp = datetime.fromisoformat(str(row["update_time"]))
            stamps.append(stamp.replace(tzinfo=NY) if stamp.tzinfo is None else stamp)
        if len(prices) != 2 or len(stamps) != 2 or any(p <= 0 for p in prices.values()):
            raise ValueError("needs_data: incomplete price/time snapshot")
        return prices, min(stamps).isoformat()
    finally:
        context.close()


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def run_shadow_schedule(*, root, config_path, now=None, quote_reader=read_open_quotes):
    root = Path(root).resolve()
    live_clock = now is None
    now = now or datetime.now(NY)
    runtime = root / "runtime/soxl/stage1_economic_shadow"
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / "scheduler.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "already_running", "orders_submitted": 0}
        result = {"observed_at": now.isoformat(), "orders_submitted": 0, "execution_mode": "READ_ONLY_SHADOW"}
        if not in_open_window(now):
            result.update(status="waiting", reason="outside_09_31_to_09_36_ET")
        else:
            config = json.loads(Path(config_path).read_text())
            session = now.astimezone(NY).date().isoformat()
            candidates = config["candidates"]
            completed = set()
            for name in candidates:
                path = runtime / "forward" / session / f"{name}.json"
                if not path.exists():
                    continue
                prior = json.loads(path.read_text())
                if prior.get("candidate_identity", {}).get("manifest_sha256") != config["expected_manifest_sha256"]:
                    raise ValueError("blocked: shadow candidate revision changed within the same session")
                if prior.get("status") == "ready":
                    completed.add(name)
            if completed == set(candidates):
                result.update(status="already_observed", session=session)
            else:
                try:
                    prices, quote_as_of = quote_reader(host=config.get("host", "127.0.0.1"), port=int(config.get("port", 11111)))
                    from src.workflow.stage1_economic_shadow import run_once
                    observations = []
                    for candidate in candidates:
                        if candidate in completed:
                            continue
                        observation = run_once(root=root, card_path=config["card_path"], candidate=candidate,
                            expected_manifest_sha256=config["expected_manifest_sha256"],
                            ledger_path=root / "runtime/soxl/stage1_futu_sim/paper_ledger_v2.json",
                            execution_state_path=root / "runtime/soxl/stage1_futu_sim/execution_state_v2.json",
                            prices=prices, quote_as_of=quote_as_of, execution_date=session, mode="forward",
                            now=datetime.now(NY) if live_clock else now)
                        observations.append({"candidate": candidate, "status": observation.get("status"), "decision": observation.get("decision"), "decision_reason": observation.get("decision_reason")})
                    result.update(status="observed", session=session, observations=observations)
                except (ValueError, RuntimeError, OSError, KeyError) as error:
                    result.update(status="needs_data", reason=str(error))
        _write(runtime / "latest_status.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_shadow_schedule(root=args.root, config_path=args.config), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
