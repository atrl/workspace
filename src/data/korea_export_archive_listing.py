"""Bounded read-only collection of original Korean export release listing pages."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def listing_rows(body: bytes) -> list[dict[str, str]]:
    soup = BeautifulSoup(body, "html.parser")
    result = []
    for li in soup.select(".board-list li"):
        title, date, anchor = li.select_one(".title"), li.select_one(".date"), li.select_one("a")
        if title is None or date is None or anchor is None:
            continue
        stamp = date.get_text(" ", strip=True).replace("date", "").strip()
        if not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", stamp):
            raise ValueError("official archive registration date differs")
        ref = anchor.get("href") or ""
        if not re.fullmatch(r"javascript:article\.view\('\d+', '2'\);", ref):
            raise ValueError("official archive article reference differs")
        result.append({"title": title.get_text(" ", strip=True),
                       "registration_date": stamp, "article_ref": ref})
    if not result:
        raise ValueError("official archive listing rows absent")
    return result


def collect(root: Path) -> dict:
    root = root.resolve()
    card = root / "research/korea_semiconductor_exports_source_20261022"
    plan_path = card / "archive_listing_harvest_plan.json"
    plan = json.loads(plan_path.read_text())
    if not plan["frozen"] or not plan["execution_authorized"]:
        raise ValueError("archive listing plan is not frozen")
    if sha(card / "summary.json") != plan["parent_qualification_summary_sha256"]:
        raise ValueError("parent source qualification summary changed")
    if sha(card / "archive_depth_result.json") != plan["archive_depth_result_sha256"]:
        raise ValueError("archive depth source changed")
    for item in plan["reused_exact_pages"].values():
        if sha(root / item["path"]) != item["sha256"]:
            raise ValueError("reused archive page source differs")
    indices = [i for i in range(plan["page_index_first"], plan["page_index_last"] + 1)
               if str(i) not in plan["reused_exact_pages"]]
    if len(indices) != plan["request_policy"]["max_new_HTTP_gets"]:
        raise ValueError("archive page count differs")
    receipts_path = card / "archive_listing_receipts.jsonl"
    receipts = [json.loads(x) for x in receipts_path.read_text().splitlines() if x] if receipts_path.exists() else []
    if len(receipts) > len(indices):
        raise ValueError("archive receipt count exceeds plan")
    for idx, rec in enumerate(receipts):
        if rec["page_index"] != indices[idx] or rec["status"] != "ok":
            raise ValueError("archive resume prefix differs")
        if sha(root / rec["relative_path"]) != rec["sha256"]:
            raise ValueError("archive resume source differs")
    state_path = card / "archive_listing_run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["plan_sha256"] != sha(plan_path):
            raise ValueError("archive run-state plan differs")
        if state["status"] == "completed":
            return state
    else:
        state = {"status": "running", "plan_sha256": sha(plan_path),
                 "completed_pages": len(receipts), "planned_pages": len(indices)}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    raw_root = root / "data/snapshots/korea_semiconductor_exports_source_20261022/archive_full"
    last_start = None
    for pos in range(len(receipts), len(indices)):
        i = indices[pos]
        if last_start is not None:
            pause = plan["request_policy"]["minimum_seconds_between_gets"] - (time.monotonic() - last_start)
            if pause > 0:
                time.sleep(pause)
        url = plan["base_url"] + "?pageIndex=" + str(i)
        record = {"page_index": i, "url": url,
                  "requested_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        body = None
        for attempt in range(1 + plan["request_policy"]["retry_count_for_transient_error"]):
            last_start = time.monotonic()
            record["wire_attempts"] = attempt + 1
            try:
                with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                   "Accept": "text/html,*/*"}), timeout=30) as response:
                    body = response.read(plan["max_response_bytes"] + 1)
                    record.update(http_status=response.status,
                                  content_type=response.headers.get("Content-Type"),
                                  response_date=response.headers.get("Date"))
                break
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt:
                    record.update(http_status=error.code, error="HTTPError: " + str(error))
                    break
            except (URLError, TimeoutError, socket.timeout) as error:
                if attempt:
                    record["error"] = type(error).__name__ + ": " + str(error)
                    break
            time.sleep(2)
        try:
            if body is None or record.get("http_status") != 200:
                raise ValueError(record.get("error", "missing official HTTP200 response"))
            if len(body) > plan["max_response_bytes"]:
                raise ValueError("archive response byte budget exceeded")
            path = raw_root / (str(i) + ".html")
            if path.exists():
                raise ValueError("unreceipted archive page already exists")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            record.update(relative_path=str(path.relative_to(root)), bytes=len(body),
                          sha256=hashlib.sha256(body).hexdigest())
            rows = listing_rows(body)
            record.update(status="ok", article_rows=len(rows),
                          newest_registration_date=max(x["registration_date"] for x in rows),
                          oldest_registration_date=min(x["registration_date"] for x in rows))
        except Exception as error:
            record.update(status="failed", error=type(error).__name__ + ": " + str(error))
        with receipts_path.open("a") as out:
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            out.flush()
        if record["status"] != "ok":
            state.update(status="failed", completed_pages=pos,
                         failed_page=i, error=record["error"])
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            raise ValueError(f"archive page {i} failed: {record['error']}")
        if (pos + 1) % 20 == 0 or pos + 1 == len(indices):
            state["completed_pages"] = pos + 1
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"official archive pages {pos + 1}/{len(indices)}", flush=True)
    state.update(status="completed", completed_pages=len(indices))
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return state


if __name__ == "__main__":
    import sys
    print(json.dumps(collect(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())))
