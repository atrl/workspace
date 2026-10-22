"""Bounded official Korean export press-release listing capture."""
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
    rows = []
    for tr in soup.select("tr"):
        anchor = tr.find("a", href=re.compile(r"/kor/article/ATCL3f49a5a8c/\d+/view"))
        if anchor is None:
            continue
        title = anchor.get_text(" ", strip=True)
        dates = re.findall(r"20\d{2}-\d{2}-\d{2}", tr.get_text(" ", strip=True))
        ref = re.search(r"/ATCL3f49a5a8c/(\d+)/view", anchor["href"])
        if not title or len(dates) != 1 or ref is None:
            raise ValueError("official Korean list row identity differs")
        rows.append({"article_id": ref.group(1), "title": title,
                     "registration_date": dates[0], "href": anchor["href"]})
    if len(rows) < 5:
        raise ValueError("official Korean filtered archive has too few rows")
    return rows


def collect(root: Path) -> dict:
    root = root.resolve()
    card = root / "research/korea_semiconductor_exports_korean_vintage_20261022"
    plan_path = card / "listing_harvest_plan.json"
    plan = json.loads(plan_path.read_text())
    if not plan["frozen"] or not plan["execution_authorized"]:
        raise ValueError("Korean listing plan not frozen")
    base = plan["base_search_url"]
    if base.count("pageIndex=1") != 1:
        raise ValueError("Korean search URL page index differs")
    original = json.loads((card / "search_probe_receipt.json").read_text())
    if sha(card / "search_probe_receipt.json") != plan["page1_receipt_sha256"]:
        raise ValueError("page1 receipt changed")
    if sha(root / original["relative_path"]) != plan["page1_raw_sha256"]:
        raise ValueError("page1 source changed")
    listing_rows((root / original["relative_path"]).read_bytes())
    indices = list(range(plan["page_index_first"] + 1, plan["page_index_last"] + 1))
    if len(indices) != plan["request_policy"]["new_distinct_HTTP_gets"]:
        raise ValueError("Korean listing request count differs")
    receipt_path = card / "listing_receipts.jsonl"
    prior = [json.loads(x) for x in receipt_path.read_text().splitlines() if x] if receipt_path.exists() else []
    if len(prior) > len(indices):
        raise ValueError("Korean listing receipts exceed plan")
    for index, rec in enumerate(prior):
        if rec["page_index"] != indices[index] or rec["status"] != "ok":
            raise ValueError("Korean listing resume prefix differs")
        if sha(root / rec["relative_path"]) != rec["sha256"]:
            raise ValueError("Korean listing resume source changed")
    state_path = card / "listing_run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["plan_sha256"] != sha(plan_path):
            raise ValueError("Korean listing state plan differs")
        if state["status"] == "completed":
            return state
    else:
        state = {"status": "running", "plan_sha256": sha(plan_path),
                 "completed_new_pages": len(prior), "planned_new_pages": len(indices)}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    raw_root = root / "data/snapshots/korea_semiconductor_exports_korean_vintage_20261022/listings"
    for index in range(len(prior), len(indices)):
        page = indices[index]
        url = base.replace("pageIndex=1", "pageIndex=" + str(page))
        rec = {"page_index": page, "url": url,
               "requested_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        body = None
        for attempt in range(2 if plan["request_policy"]["one_retry_on_transient"] else 1):
            rec["wire_attempts"] = attempt + 1
            try:
                with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                   "Accept": "text/html,*/*"}), timeout=30) as response:
                    body = response.read(plan["request_policy"]["max_response_bytes"] + 1)
                    rec.update(http_status=response.status,
                               content_type=response.headers.get("Content-Type"),
                               response_date=response.headers.get("Date"))
                break
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt:
                    rec.update(http_status=error.code, error="HTTPError: " + str(error))
                    break
            except (URLError, TimeoutError, socket.timeout) as error:
                if attempt:
                    rec["error"] = type(error).__name__ + ": " + str(error)
                    break
            time.sleep(2)
        try:
            if body is None or rec.get("http_status") != 200:
                raise ValueError(rec.get("error", "missing Korean archive HTTP200"))
            if len(body) > plan["request_policy"]["max_response_bytes"]:
                raise ValueError("Korean listing response byte budget exceeded")
            path = raw_root / (str(page) + ".html")
            if path.exists():
                raise ValueError("unreceipted Korean listing page exists")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            rec.update(relative_path=str(path.relative_to(root)), bytes=len(body),
                       sha256=hashlib.sha256(body).hexdigest())
            rows = listing_rows(body)
            rec.update(status="ok", rows=len(rows),
                       newest_date=max(x["registration_date"] for x in rows),
                       oldest_date=min(x["registration_date"] for x in rows))
        except Exception as error:
            rec.update(status="failed", error=type(error).__name__ + ": " + str(error))
        with receipt_path.open("a") as out:
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            out.flush()
        if rec["status"] != "ok":
            state.update(status="failed", completed_new_pages=index,
                         failed_page=page, error=rec["error"])
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            raise ValueError(f"Korean listing page {page} failed: {rec['error']}")
        state["completed_new_pages"] = index + 1
        if (index + 1) % 5 == 0 or index + 1 == len(indices):
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"Korean filtered listing {index + 1}/{len(indices)}", flush=True)
        time.sleep(plan["request_policy"]["minimum_seconds_between_gets"])
    state["status"] = "completed"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return state


if __name__ == "__main__":
    import sys
    print(json.dumps(collect(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())))
