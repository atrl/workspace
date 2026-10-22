"""Immutable, resumable source capture of selected Korean national export releases."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(body: bytes) -> tuple[str, str]:
    soup = BeautifulSoup(body, "html.parser")
    title = soup.select_one(".detail-tit span")
    dates = [li.find("span") for li in soup.find_all("li") if li.find("em")
             and li.find("em").get_text(" ", strip=True) == "Registration date"]
    if title is None or len(dates) != 1 or dates[0] is None:
        raise ValueError("official article title or registration date absent")
    return title.get_text(" ", strip=True), dates[0].get_text(" ", strip=True)


def acquire(root: Path) -> dict:
    root = root.resolve()
    card = root / "research/korea_semiconductor_exports_source_20261022"
    plan_path = card / "article_html_acquisition_plan.json"
    plan = json.loads(plan_path.read_text())
    candidates = json.loads((card / "national_release_candidate_manifest_v3.json").read_text())
    if not plan["frozen"] or not plan["execution_authorized"]:
        raise ValueError("article acquisition plan not frozen")
    if sha(card / "national_release_candidate_manifest_v3.json") != plan["candidate_manifest_sha256"]:
        raise ValueError("frozen article candidate manifest changed")
    months = sorted(candidates["articles_by_month"])
    if len(months) != plan["target_months"] or any(len(candidates["articles_by_month"][m]) != 1 for m in months):
        raise ValueError("national monthly article selection differs")
    for month, item in plan["reused_exact_months"].items():
        path = root / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError("reused article bytes differ")
        expected = candidates["articles_by_month"][month][0]
        if identity(path.read_bytes()) != (expected["title"], expected["registration_date"]):
            raise ValueError("reused article semantic identity differs")
    request_months = [m for m in months if m not in plan["reused_exact_months"]]
    if len(request_months) != plan["new_HTML_requests"]:
        raise ValueError("new article request count differs")
    receipts_path = card / "article_html_receipts.jsonl"
    prior = [json.loads(line) for line in receipts_path.read_text().splitlines() if line] if receipts_path.exists() else []
    if len(prior) > len(request_months):
        raise ValueError("article receipts exceed frozen budget")
    for index, rec in enumerate(prior):
        month = request_months[index]
        expected = candidates["articles_by_month"][month][0]
        if rec["month"] != month or rec["url"] != expected["url"] or rec["status"] != "ok":
            raise ValueError("article resume prefix differs")
        if sha(root / rec["relative_path"]) != rec["sha256"]:
            raise ValueError("article resume source hash differs")
    state_path = card / "article_html_run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["plan_sha256"] != sha(plan_path):
            raise ValueError("article run-state plan differs")
        if state["status"] == "completed":
            return state
    else:
        state = {"status": "running", "plan_sha256": sha(plan_path),
                 "completed_new_articles": len(prior), "planned_new_articles": len(request_months)}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    raw_root = root / "data/snapshots/korea_semiconductor_exports_source_20261022/article_html"
    last_start = None
    for index in range(len(prior), len(request_months)):
        month = request_months[index]
        expected = candidates["articles_by_month"][month][0]
        if last_start is not None:
            pause = plan["request_policy"]["minimum_seconds_between_gets"] - (time.monotonic() - last_start)
            if pause > 0:
                time.sleep(pause)
        rec = {"month": month, "article_id": expected["article_id"], "url": expected["url"],
               "requested_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        body = None
        for attempt in range(2 if plan["request_policy"]["one_retry_on_transient"] else 1):
            last_start = time.monotonic()
            rec["wire_attempts"] = attempt + 1
            try:
                with urlopen(Request(expected["url"], headers={"User-Agent": "Mozilla/5.0",
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
                raise ValueError(rec.get("error", "missing official HTTP200 article"))
            if len(body) > plan["request_policy"]["max_bytes_per_response"]:
                raise ValueError("article response byte budget exceeded")
            path = raw_root / (month + ".html")
            if path.exists():
                raise ValueError("unreceipted article file exists")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            rec.update(relative_path=str(path.relative_to(root)), bytes=len(body),
                       sha256=hashlib.sha256(body).hexdigest())
            found = identity(body)
            if found != (expected["title"], expected["registration_date"]):
                raise ValueError("article title/date differs from official listing")
            rec.update(status="ok", registration_date=found[1])
        except Exception as error:
            rec.update(status="failed", error=type(error).__name__ + ": " + str(error))
        with receipts_path.open("a") as out:
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            out.flush()
        if rec["status"] != "ok":
            state.update(status="failed", completed_new_articles=index,
                         failed_month=month, error=rec["error"])
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            raise ValueError(f"article {month} failed: {rec['error']}")
        if (index + 1) % 10 == 0 or index + 1 == len(request_months):
            state["completed_new_articles"] = index + 1
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"official articles {index + 1}/{len(request_months)}", flush=True)
    state.update(status="completed", completed_new_articles=len(request_months))
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return state


if __name__ == "__main__":
    import sys
    print(json.dumps(acquire(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())))
