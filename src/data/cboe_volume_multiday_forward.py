"""Bounded source-only Cboe volume vintages after the frozen Oct 21 probe."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.data.cboe_volume_forward_capture import _rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("forward receipt already exists")
    path.write_text(json.dumps(value, indent=2) + "\n")


def _window(target: str, phase: dict) -> tuple[dt.datetime, dt.datetime]:
    day = dt.date.fromisoformat(target) + dt.timedelta(days=1)
    start = dt.datetime.fromisoformat(day.isoformat() + "T" + phase["next_calendar_day_UTC_time"] + ":00+00:00")
    end = dt.datetime.fromisoformat(day.isoformat() + "T" + phase["latest_UTC_time"] + ":00+00:00")
    if end <= start:
        raise ValueError("invalid frozen observation window")
    return start, end


def bind_third(root: str | Path) -> dict:
    """Pin the completed Oct 21 check before any distinct-date observation."""
    root = Path(root).resolve()
    card = root / "research/cboe_total_options_activity_source_20261022"
    plan_path = card / "multiday_forward_plan.json"
    plan = json.loads(plan_path.read_text())
    addendum = json.loads((card / "forward_transport_addendum.json").read_text())
    for name, expected in [("forward_publication_plan.json", plan["source_forward_plan_sha256"]),
                           ("forward_transport_addendum.json", plan["source_transport_addendum_sha256"]),
                           (plan["source_transport_identity_clarification"],
                            plan["source_transport_identity_clarification_sha256"])]:
        if _sha(card / name) != expected:
            raise ValueError("frozen source identity changed: " + name)
    for relative, digest in addendum["prior_receipts_sha256"].items():
        if _sha(card / relative) != digest:
            raise ValueError("prior Oct21 observation differs")
    receipts = sorted((card / "forward_observations").glob("*.json"))
    prior = {card / relative for relative in addendum["prior_receipts_sha256"]}
    third_paths = [path for path in receipts if path not in prior]
    if len(receipts) != 3 or len(third_paths) != 1:
        raise ValueError("third Oct21 observation is missing or duplicated")
    second_path = max(prior)
    second = json.loads(second_path.read_text())
    third_path = third_paths[0]
    third = json.loads(third_path.read_text())
    if third.get("status") != "observed" or third.get("index") != 3:
        raise ValueError("third Oct21 observation is unfinished")
    second_by_symbol = {row["symbol"]: row for row in second["records"]}
    third_by_symbol = {row["symbol"]: row for row in third["records"]}
    if set(second_by_symbol) != set(plan["symbols"]) or set(third_by_symbol) != set(plan["symbols"]):
        raise ValueError("Oct21 symbol inventory differs")
    not_before = dt.datetime.fromisoformat(addendum["third_observation_not_before_utc"])
    values = {}
    for symbol in plan["symbols"]:
        current, earlier = third_by_symbol[symbol], second_by_symbol[symbol]
        requested = dt.datetime.fromisoformat(current["requested_at_utc"])
        if (requested < not_before or not current.get("target_date_present")
                or not current.get("earlier_date_unchanged")
                or current.get("sha256") != earlier.get("sha256")
                or current.get("daily_contracts") != earlier.get("daily_contracts")):
            raise ValueError("third Oct21 report changed or missed the frozen gate")
        values[symbol] = current["daily_contracts"]["2026-10-21"]
    binding = {"schema_version": 1, "frozen": True, "multiday_plan_sha256": _sha(plan_path),
               "observer_module_sha256": _sha(root / plan["observer_module"]),
               "third_receipt_path": str(third_path.relative_to(card)),
               "third_receipt_sha256": _sha(third_path), "third_values": values,
               "models": 0, "accounts": 0, "market_outcome_reads": 0}
    _write_json(card / plan["third_same_date_binding"], binding)
    return binding


def _third_binding(card: Path, plan: dict) -> dict:
    binding_path = card / plan["third_same_date_binding"]
    if not binding_path.is_file():
        raise ValueError("third Oct21 observation binding is missing")
    binding = json.loads(binding_path.read_text())
    if not binding.get("frozen") or binding.get("multiday_plan_sha256") != _sha(card / "multiday_forward_plan.json"):
        raise ValueError("third observation binding differs from frozen plan")
    observer = card.parents[1] / plan["observer_module"]
    if _sha(observer) != binding.get("observer_module_sha256"):
        raise ValueError("forward observer implementation differs")
    third_path = card / binding["third_receipt_path"]
    if not third_path.is_file() or _sha(third_path) != binding["third_receipt_sha256"]:
        raise ValueError("third Oct21 observation receipt differs")
    third = json.loads(third_path.read_text())
    if third.get("status") != "observed" or third.get("index") != 3 or len(third.get("records", [])) != 2:
        raise ValueError("third Oct21 observation is unfinished")
    by_symbol = {row["symbol"]: row for row in third["records"]}
    if set(by_symbol) != set(plan["symbols"]):
        raise ValueError("third Oct21 symbols differ")
    addendum = json.loads((card / "forward_transport_addendum.json").read_text())
    not_before = dt.datetime.fromisoformat(addendum["third_observation_not_before_utc"])
    for symbol, row in by_symbol.items():
        requested = dt.datetime.fromisoformat(row["requested_at_utc"])
        if (requested < not_before or not row.get("target_date_present")
                or row.get("daily_contracts", {}).get("2026-10-21") != binding["third_values"][symbol]):
            raise ValueError("third Oct21 source value differs")
    return binding


def _predecessor_values(card: Path, plan: dict, target: str, predecessor: str, binding: dict) -> dict:
    if target == plan["target_trade_dates"][0]:
        return binding["third_values"]
    path = card / "multiday_forward_observations" / predecessor / "later.json"
    if not path.exists():
        return {}
    previous = json.loads(path.read_text())
    if previous.get("status") != "observed":
        return {}
    return {row["symbol"]: row["daily_contracts"][predecessor]
            for row in previous["records"] if row.get("target_date_present")
            and predecessor in row.get("daily_contracts", {})}


def _capture(root: Path, target: str, phase_name: str, *, now: dt.datetime, fetch=urlopen, clock=None) -> dict:
    root = Path(root).resolve()
    card = root / "research/cboe_total_options_activity_source_20261022"
    plan_path = card / "multiday_forward_plan.json"
    plan = json.loads(plan_path.read_text())
    if not plan["frozen"] or target not in plan["target_trade_dates"] or phase_name not in plan["phases"]:
        raise ValueError("target or phase is outside frozen source plan")
    for name, expected in [("forward_publication_plan.json", plan["source_forward_plan_sha256"]),
                           ("forward_transport_addendum.json", plan["source_transport_addendum_sha256"]),
                           (plan["source_transport_identity_clarification"],
                            plan["source_transport_identity_clarification_sha256"]),
                           (plan["parser_module"], plan["parser_module_sha256"])]:
        path = root / name if name.startswith("src/") else card / name
        if _sha(path) != expected:
            raise ValueError("frozen source identity changed: " + name)
    binding = _third_binding(card, plan)
    if now.tzinfo is None:
        raise ValueError("UTC observation time must be timezone-aware")
    now = now.astimezone(dt.timezone.utc)
    first, last = _window(target, plan["phases"][phase_name])
    if now < first:
        raise ValueError("observation before frozen window")
    receipt_path = card / "multiday_forward_observations" / target / (phase_name + ".json")
    if receipt_path.exists():
        raise ValueError("forward observation already attempted")
    common = {"schema_version": 1, "target_trade_date": target, "phase": phase_name,
              "plan_sha256": _sha(plan_path), "binding_sha256": _sha(card / plan["third_same_date_binding"]),
              "observed_at_utc": now.isoformat(), "models": 0, "accounts": 0,
              "market_outcome_reads": 0, "PIT_qualified": False}
    if now > last:
        receipt = {**common, "status": "missed_window", "records": [], "http_GETs": 0}
        _write_json(receipt_path, receipt)
        return receipt
    observed = list((card / "multiday_forward_observations").glob("*/*.json"))
    consumed = sum(json.loads(path.read_text()).get("http_GETs", 0) for path in observed)
    if consumed + len(plan["symbols"]) > plan["budget"]["max_http_GETs_total"]:
        raise ValueError("forward HTTP budget exhausted")
    position = plan["target_trade_dates"].index(target)
    predecessor = plan["predecessor_trade_dates"][position]
    expected = _predecessor_values(card, plan, target, predecessor, binding)
    raw_dir = root / "data/snapshots/cboe_total_options_activity_source_20261022/multiday_forward" / target / phase_name
    if raw_dir.exists():
        raise ValueError("forward source raw path already exists")
    raw_dir.mkdir(parents=True)
    records = []
    attempted_GETs = 0
    if clock is None:
        clock = lambda: dt.datetime.now(dt.timezone.utc)
    for symbol in plan["symbols"]:
        query = {**{k: v for k, v in plan["form_parameters"].items() if k != "exchanges"},
                 "symbol": symbol, "startDate": predecessor, "endDate": target}
        url = plan["source_endpoint"] + "?" + urlencode(query) + "&" + urlencode(
            [("exchanges", venue) for venue in plan["form_parameters"]["exchanges"]])
        requested = clock()
        if requested.tzinfo is None:
            raise ValueError("request clock must be timezone-aware")
        requested = requested.astimezone(dt.timezone.utc)
        row = {"symbol": symbol, "url": url, "requested_at_utc": requested.isoformat()}
        try:
            if not first <= requested <= last:
                raise ValueError("request outside frozen window")
            attempted_GETs += 1
            with fetch(Request(url, headers={"User-Agent": "Mozilla/5.0 research source qualification",
                                             "Accept": "text/csv,*/*"}), timeout=30) as response:
                body = response.read(plan["max_response_bytes"] + 1)
                row.update(http_status=response.status, content_type=response.headers.get("Content-Type"),
                           response_date=response.headers.get("Date"), response_age=response.headers.get("Age"),
                           cache_status=response.headers.get("CF-Cache-Status"),
                           cache_control=response.headers.get("Cache-Control"),
                           last_modified=response.headers.get("Last-Modified"))
            if len(body) > plan["max_response_bytes"]:
                raise ValueError("response exceeds frozen byte budget")
            path = raw_dir / (symbol.lower() + ".csv")
            path.write_bytes(body)
            row.update(relative_path=str(path.relative_to(root)), bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            if row["http_status"] != 200 or not str(row["content_type"]).startswith("text/csv"):
                raise ValueError("official source is not CSV HTTP 200")
            days = _rows(body, symbol, set(plan["form_parameters"]["exchanges"]))
            daily = {day: sum(venue.values()) for day, venue in sorted(days.items())}
            earlier = daily.get(predecessor)
            row.update(target_date_present=target in daily, predecessor_date=predecessor,
                       predecessor_date_present=earlier is not None,
                       predecessor_unchanged=None if symbol not in expected or earlier is None
                       else earlier == expected[symbol], daily_contracts=daily,
                       target_exchange_rows=len(days.get(target, {})))
        except Exception as error:
            row["error"] = type(error).__name__ + ": " + str(error)
        records.append(row)
    receipt = {**common, "status": "observed" if all("error" not in row for row in records) else "partial_failure",
               "records": records, "http_GETs": attempted_GETs, "predecessor_trade_date": predecessor}
    _write_json(receipt_path, receipt)
    return receipt


def capture(root: str | Path, target: str, phase: str) -> dict:
    """Attempt exactly one frozen source observation at the actual current UTC time."""
    return _capture(Path(root), target, phase, now=dt.datetime.now(dt.timezone.utc))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m src.data.cboe_volume_multiday_forward TARGET PHASE")
    print(json.dumps(capture(Path.cwd(), sys.argv[1], sys.argv[2]), indent=2))
