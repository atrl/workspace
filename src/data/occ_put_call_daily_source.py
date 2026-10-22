"""Bounded, resumable OCC daily put/call source acquisition; no market labels."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_daily(body: bytes, symbol: str, date: str) -> dict:
    lines = list(csv.reader(io.StringIO(body.decode("utf-8-sig"))))
    if not lines or lines[0] != ["quantity", "underlying", "symbol", "actype", "porc", "exchange", "actdate"]:
        raise ValueError("OCC daily header differs")
    if len(lines) == 1:
        raise ValueError("OCC daily response has no source rows")
    target = dt.date.fromisoformat(date).strftime("%m/%d/%Y")
    accounts, roots, options, venues = set(), set(), set(), set()
    call, put = 0, 0
    for row in lines[1:]:
        if len(row) != 8 or row[7] != "" or row[1] != symbol or row[6] != target:
            raise ValueError("OCC daily row identity differs")
        if row[3] not in ("C", "F", "M") or row[4] not in ("C", "P"):
            raise ValueError("OCC account or call/put class differs")
        amount = int(row[0])
        if amount < 0:
            raise ValueError("negative OCC quantity")
        if row[4] == "C":
            call += amount
        else:
            put += amount
        accounts.add(row[3]); roots.add(row[2]); options.add(row[4]); venues.add(row[5])
    return {"rows": len(lines) - 1, "call_quantity": call, "put_quantity": put,
            "accounts": sorted(accounts), "roots": sorted(roots), "venues": sorted(venues),
            "option_types_present": sorted(options)}


def acquire(root: str | Path) -> dict:
    root = Path(root).resolve()
    card = root / "research/occ_put_call_daily_source_20261022"
    plan_path = card / "plan.json"
    manifest_path = card / "request_manifest.json"
    plan = json.loads(plan_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if not plan["frozen"] or not plan["execution_authorized"] or manifest["plan_sha256"] != digest(plan_path):
        raise ValueError("frozen OCC source identities differ")
    if manifest["planned_http_gets"] > plan["request_policy"]["max_total_http_gets"]:
        raise ValueError("HTTP budget differs")
    requests = [x for x in manifest["items"] if "reuse_sha256" not in x]
    raw_root = root / "data/snapshots/occ_put_call_daily_source_20261022/raw"
    receipts_path = card / "fetch_receipts.jsonl"
    prior = []
    if receipts_path.exists():
        prior = [json.loads(line) for line in receipts_path.read_text().splitlines() if line]
    if len(prior) > len(requests):
        raise ValueError("receipt count exceeds frozen request manifest")
    for index, item in enumerate(prior):
        expected = requests[index]
        if (item["symbol"], item["date"], item["url"]) != (
            expected["symbol"], expected["date"], expected["url"]):
            raise ValueError("resume receipt is not exact request prefix")
        path = root / item["relative_path"]
        if digest(path) != item["sha256"] or item.get("status") != "ok":
            raise ValueError("resume snapshot identity or status differs")
    state_path = card / "fetch_run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["manifest_sha256"] != digest(manifest_path):
            raise ValueError("resume manifest changed")
        if state["status"] == "completed":
            return state
    else:
        state = {"status": "running", "plan_sha256": digest(plan_path),
                 "manifest_sha256": digest(manifest_path), "completed_http_gets": len(prior),
                 "planned_http_gets": len(requests)}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    last_start = None
    for index in range(len(prior), len(requests)):
        item = requests[index]
        if last_start is not None:
            wait = plan["request_policy"]["minimum_seconds_between_gets"] - (time.monotonic() - last_start)
            if wait > 0:
                time.sleep(wait)
        last_start = time.monotonic()
        requested_at = dt.datetime.now(dt.timezone.utc).isoformat()
        record = {"index": index, "symbol": item["symbol"], "date": item["date"],
                  "url": item["url"], "requested_at_utc": requested_at}
        try:
            try:
                with urlopen(Request(item["url"], headers={"User-Agent": "Mozilla/5.0 research source qualification",
                                                            "Accept": "text/csv,*/*"}), timeout=30) as response:
                    body = response.read(plan["request_policy"]["max_bytes_per_response"] + 1)
                    record.update(http_status=response.status,
                                  content_type=response.headers.get("Content-Type"),
                                  response_date=response.headers.get("Date"),
                                  age=response.headers.get("Age"),
                                  cache_control=response.headers.get("Cache-Control"))
            except HTTPError as error:
                body = error.read(plan["request_policy"]["max_bytes_per_response"] + 1)
                record.update(http_status=error.code, content_type=error.headers.get("Content-Type"),
                              response_date=error.headers.get("Date"))
            if len(body) > plan["request_policy"]["max_bytes_per_response"]:
                raise ValueError("response exceeds frozen source byte budget")
            path = raw_root / item["date"] / (item["symbol"].lower() + ".bin")
            if path.exists():
                raise ValueError("unreceipted raw source file already exists")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            record.update(relative_path=str(path.relative_to(root)), bytes=len(body),
                          sha256=hashlib.sha256(body).hexdigest())
            if record["http_status"] != 200:
                raise ValueError("HTTP status differs")
            record.update(parse_daily(body, item["symbol"], item["date"]))
            record["status"] = "ok"
        except Exception as error:
            record["status"] = "failed"
            record["error"] = type(error).__name__ + ": " + str(error)
        with receipts_path.open("a") as output:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()
            os.fsync(output.fileno())
        if record["status"] != "ok":
            state.update(status="failed", completed_http_gets=index,
                         failure_receipt_index=index, failure=record["error"])
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            raise ValueError(f"OCC source request {index} failed: {record['error']}")
        if (index + 1) % 20 == 0 or index + 1 == len(requests):
            state["completed_http_gets"] = index + 1
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"OCC acquired {index + 1}/{len(requests)}", flush=True)
    state.update(status="completed", completed_http_gets=len(requests))
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return state


if __name__ == "__main__":
    import sys
    print(json.dumps(acquire(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())))
