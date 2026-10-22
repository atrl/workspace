"""One bounded, immutable Cboe publication observation for the SOXX campaign."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(body: bytes, symbol: str, exchanges: set[str]) -> dict[str, dict[str, int]]:
    reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
    if reader.fieldnames != ["Trade Date", "Underlying", "Product Type", "Exchange", "Volume"]:
        raise ValueError("Cboe volume schema changed")
    days: dict[str, dict[str, int]] = {}
    for row in reader:
        if row["Underlying"] != symbol or row["Exchange"] not in exchanges:
            raise ValueError("Cboe symbol or exchange scope changed")
        stamp = dt.datetime.strptime(row["Trade Date"], "%Y/%m/%d").date().isoformat()
        amount = int(row["Volume"])
        if amount < 0 or row["Exchange"] in days.setdefault(stamp, {}):
            raise ValueError("negative or duplicate exchange-day volume")
        days[stamp][row["Exchange"]] = amount
    return days


def capture(root: str | Path) -> dict:
    root = Path(root).resolve()
    card = root / "research/cboe_total_options_activity_source_20261022"
    plan_path = card / "forward_publication_plan.json"
    plan = json.loads(plan_path.read_text())
    if not plan["frozen"]:
        raise ValueError("forward observation plan is not frozen")
    for file, key in [("plan.json", "source_plan_sha256"),
                      ("source_coverage_audit.json", "source_coverage_sha256"),
                      ("availability_observation.json", "baseline_availability_sha256")]:
        if _sha(card / file) != plan[key]:
            raise ValueError(f"source identity changed: {file}")
    now = dt.datetime.now(dt.timezone.utc)
    if now < dt.datetime.fromisoformat(plan["first_followup_not_before_utc"]):
        raise ValueError("followup before frozen observation time")
    receipts_dir = card / "forward_observations"
    previous = sorted(receipts_dir.glob("*.json")) if receipts_dir.exists() else []
    if len(previous) >= plan["max_followup_observations"]:
        raise ValueError("forward observation budget exhausted")
    if len(previous) == 2:
        addendum = json.loads((card / "forward_transport_addendum.json").read_text())
        if not addendum["frozen"] or _sha(plan_path) != addendum["original_forward_plan_sha256"]:
            raise ValueError("third observation protocol changed")
        for path in previous:
            relative = str(path.relative_to(card))
            if _sha(path) != addendum["prior_receipts_sha256"][relative]:
                raise ValueError("prior observation receipt changed")
        if now < dt.datetime.fromisoformat(addendum["third_observation_not_before_utc"]):
            raise ValueError("third observation before frozen preopen check")
    observation_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    raw_dir = root / "data/snapshots/cboe_total_options_activity_source_20261022/forward_observations" / observation_id
    if raw_dir.exists() or (receipts_dir / (observation_id + ".json")).exists():
        raise ValueError("observation path already exists")
    baseline = json.loads((card / "retrieval_receipts.json").read_text())
    baseline_by_symbol = {item["symbol"]: item for item in baseline["queries"]}
    exchanges = set(plan["form_parameters"]["exchanges"])
    raw_dir.mkdir(parents=True)
    records = []
    for symbol in plan["symbols"]:
        args = {**{k: v for k, v in plan["form_parameters"].items() if k != "exchanges"},
                "symbol": symbol, "startDate": plan["query_start_date"],
                "endDate": plan["query_end_date"]}
        url = plan["source_endpoint"] + "?" + urlencode(args) + "&" + urlencode(
            [("exchanges", x) for x in plan["form_parameters"]["exchanges"]])
        requested = dt.datetime.now(dt.timezone.utc).isoformat()
        record = {"symbol": symbol, "requested_at_utc": requested, "url": url}
        try:
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0 research source qualification",
                                               "Accept": "text/csv,*/*"}), timeout=30) as response:
                body = response.read(plan["max_response_bytes"] + 1)
                record.update(http_status=response.status,
                              content_type=response.headers.get("Content-Type"),
                              content_disposition=response.headers.get("Content-Disposition"),
                              response_date=response.headers.get("Date"),
                              response_age=response.headers.get("Age"),
                              cache_status=response.headers.get("CF-Cache-Status"),
                              cache_control=response.headers.get("Cache-Control"),
                              last_modified=response.headers.get("Last-Modified"))
            if len(body) > plan["max_response_bytes"]:
                raise ValueError("response exceeds frozen budget")
            raw_path = raw_dir / (symbol.lower() + ".csv")
            raw_path.write_bytes(body)
            record.update(relative_path=str(raw_path.relative_to(root)), bytes=len(body),
                          sha256=hashlib.sha256(body).hexdigest())
            if record["http_status"] != 200 or not record["content_type"].startswith("text/csv"):
                raise ValueError("official response is not CSV HTTP 200")
            days = _rows(body, symbol, exchanges)
            parent = baseline_by_symbol[symbol]
            if _sha(root / parent["relative_path"]) != parent["sha256"]:
                raise ValueError("original baseline snapshot changed")
            baseline_days = _rows((root / parent["relative_path"]).read_bytes(), symbol, exchanges)
            earlier = plan["query_start_date"]
            record.update(target_date_present=plan["target_trade_date"] in days,
                          latest_date=max(days) if days else None,
                          earlier_date=earlier,
                          earlier_date_unchanged=days.get(earlier) == baseline_days.get(earlier),
                          daily_contracts={day: sum(venues.values()) for day, venues in sorted(days.items())},
                          baseline_sha256=parent["sha256"])
        except Exception as error:
            record["error"] = type(error).__name__ + ": " + str(error)
        records.append(record)
    receipt = {"schema_version": 1, "status": "observed" if all("error" not in x for x in records) else "partial_failure",
               "observation_id": observation_id, "plan_sha256": _sha(plan_path),
               "index": len(previous) + 1, "records": records,
               "models": 0, "accounts": 0, "market_outcome_reads": 0,
               "PIT_qualified": False}
    receipts_dir.mkdir(parents=True, exist_ok=True)
    (receipts_dir / (observation_id + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    import sys
    workspace = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print(json.dumps(capture(workspace), indent=2))
