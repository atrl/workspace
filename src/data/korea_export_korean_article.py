"""Exact monthly Korean national export press-release HTML capture."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(body: bytes) -> tuple[str, str]:
    soup = BeautifulSoup(body, "html.parser")
    title = soup.select_one(".detail-tit span")
    dates = [li.find("span") for li in soup.find_all("li") if li.find("em")
             and li.find("em").get_text(" ", strip=True) == "등록일"]
    if title is None or len(dates) != 1 or dates[0] is None:
        raise ValueError("Korean article title or registration date absent")
    return title.get_text(" ", strip=True), dates[0].get_text(" ", strip=True)


def acquire(root: Path) -> dict:
    root = root.resolve()
    card = root / "research/korea_semiconductor_exports_korean_vintage_20261022"
    plan_path = card / "korean_article_html_plan.json"
    plan = json.loads(plan_path.read_text())
    source = json.loads((card / "korean_national_article_manifest.json").read_text())
    if not plan["frozen"] or not plan["execution_authorized"]:
        raise ValueError("Korean article plan not frozen")
    if sha(card / "korean_national_article_manifest.json") != plan["article_manifest_sha256"]:
        raise ValueError("Korean article manifest changed")
    months = sorted(source["articles_by_month"])
    if len(months) != plan["target_months"] or any(len(source["articles_by_month"][m]) != 1 for m in months):
        raise ValueError("Korean national month count differs")
    for month, item in plan["reused_exact_months"].items():
        p = root / item["path"]
        if sha(p) != item["sha256"]:
            raise ValueError("reused Korean original HTML changed")
        expected = source["articles_by_month"][month][0]
        if identity(p.read_bytes()) != (expected["title"], expected["registration_date"]):
            raise ValueError("reused Korean original article identity differs")
    requested = [m for m in months if m not in plan["reused_exact_months"]]
    if len(requested) != plan["new_HTML_requests"]:
        raise ValueError("Korean article request count differs")
    receipts_path = card / "korean_article_html_receipts.jsonl"
    prior = [json.loads(x) for x in receipts_path.read_text().splitlines() if x] if receipts_path.exists() else []
    if len(prior) > len(requested):
        raise ValueError("Korean article receipts exceed plan")
    for i, rec in enumerate(prior):
        expected = source["articles_by_month"][requested[i]][0]
        if rec["month"] != requested[i] or rec["article_id"] != expected["article_id"] or rec["status"] != "ok":
            raise ValueError("Korean article resume prefix differs")
        if sha(root / rec["relative_path"]) != rec["sha256"]:
            raise ValueError("Korean article resume bytes differ")
    state_path = card / "korean_article_html_run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["plan_sha256"] != sha(plan_path):
            raise ValueError("Korean article state plan differs")
        if state["status"] == "completed":
            return state
    else:
        state = {"status": "running", "plan_sha256": sha(plan_path),
                 "completed_new_articles": len(prior), "planned_new_articles": len(requested)}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    raw_root = root / "data/snapshots/korea_semiconductor_exports_korean_vintage_20261022/article_html"
    for i in range(len(prior), len(requested)):
        month = requested[i]
        expected = source["articles_by_month"][month][0]
        url = urljoin("https://www.motir.go.kr", expected["href"].split("?", 1)[0])
        rec = {"month": month, "article_id": expected["article_id"], "url": url,
               "requested_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        body = None
        for attempt in range(2 if plan["request_policy"]["one_retry_on_transient"] else 1):
            rec["wire_attempts"] = attempt + 1
            try:
                with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                   "Accept": "text/html,*/*"}), timeout=30) as response:
                    body = response.read(plan["request_policy"]["max_bytes_per_response"] + 1)
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
                raise ValueError(rec.get("error", "missing Korean article HTTP200"))
            if len(body) > plan["request_policy"]["max_bytes_per_response"]:
                raise ValueError("Korean article response byte budget exceeded")
            path = raw_root / (month + ".html")
            if path.exists():
                raise ValueError("unreceipted Korean article file exists")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            rec.update(relative_path=str(path.relative_to(root)), bytes=len(body),
                       sha256=hashlib.sha256(body).hexdigest())
            if identity(body) != (expected["title"], expected["registration_date"]):
                raise ValueError("Korean article title/date differs from listing")
            rec["status"] = "ok"
        except Exception as error:
            rec.update(status="failed", error=type(error).__name__ + ": " + str(error))
        with receipts_path.open("a") as out:
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            out.flush()
        if rec["status"] != "ok":
            state.update(status="failed", completed_new_articles=i,
                         failed_month=month, error=rec["error"])
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            raise ValueError(f"Korean article {month} failed: {rec['error']}")
        state["completed_new_articles"] = i + 1
        if (i + 1) % 10 == 0 or i + 1 == len(requested):
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"Korean national articles {i + 1}/{len(requested)}", flush=True)
        time.sleep(plan["request_policy"]["minimum_seconds_between_gets"])
    state["status"] = "completed"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return state


if __name__ == "__main__":
    import sys
    print(json.dumps(acquire(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())))
