"""Immutable source capture of selected official Korean export PDF attachments."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire(root: Path) -> dict:
    root = root.resolve()
    card = root / "research/korea_semiconductor_exports_source_20261022"
    plan_path = card / "english_pdf_acquisition_plan.json"
    plan = json.loads(plan_path.read_text())
    audit = json.loads((card / "article_html_audit.json").read_text())
    if not plan["frozen"] or not plan["execution_authorized"]:
        raise ValueError("PDF acquisition plan not frozen")
    if sha(card / "article_html_audit.json") != plan["article_HTML_audit_sha256"]:
        raise ValueError("article source audit changed")
    months = sorted(audit["month_details"])
    if len(months) != plan["month_count"] or any(len(audit["month_details"][m]["PDF_links"]) != 1 for m in months):
        raise ValueError("exact one-PDF-per-month inventory differs")
    for item in plan["reused_exact_months"].values():
        if sha(root / item["path"]) != item["sha256"]:
            raise ValueError("reused original PDF bytes differ")
    request_months = [m for m in months if m not in plan["reused_exact_months"]]
    if len(request_months) != plan["new_PDF_requests"]:
        raise ValueError("PDF request budget differs")
    receipts_path = card / "english_pdf_receipts.jsonl"
    prior = [json.loads(line) for line in receipts_path.read_text().splitlines() if line] if receipts_path.exists() else []
    if len(prior) > len(request_months):
        raise ValueError("PDF receipts exceed frozen budget")
    for index, rec in enumerate(prior):
        month = request_months[index]
        expected = audit["month_details"][month]["PDF_links"][0]["url"]
        if rec["month"] != month or rec["url"] != expected or rec["status"] != "ok":
            raise ValueError("PDF resume prefix differs")
        if sha(root / rec["relative_path"]) != rec["sha256"]:
            raise ValueError("PDF resume source hash differs")
    state_path = card / "english_pdf_run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["plan_sha256"] != sha(plan_path):
            raise ValueError("PDF run state plan differs")
        if state["status"] == "completed":
            return state
    else:
        state = {"status": "running", "plan_sha256": sha(plan_path),
                 "completed_new_PDFs": len(prior), "planned_new_PDFs": len(request_months)}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    raw_root = root / "data/snapshots/korea_semiconductor_exports_source_20261022/article_pdf"
    last_start = None
    for index in range(len(prior), len(request_months)):
        month = request_months[index]
        url = audit["month_details"][month]["PDF_links"][0]["url"]
        if last_start is not None:
            pause = plan["request_policy"]["minimum_seconds_between_gets"] - (time.monotonic() - last_start)
            if pause > 0:
                time.sleep(pause)
        rec = {"month": month, "url": url,
               "requested_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        body = None
        for attempt in range(2 if plan["request_policy"]["one_retry_on_transient"] else 1):
            last_start = time.monotonic()
            rec["wire_attempts"] = attempt + 1
            try:
                with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                   "Accept": "application/pdf,*/*"}), timeout=45) as response:
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
                raise ValueError(rec.get("error", "missing official PDF HTTP200"))
            if len(body) > plan["request_policy"]["max_bytes_per_response"]:
                raise ValueError("PDF response byte budget exceeded")
            path = raw_root / (month + ".pdf")
            if path.exists():
                raise ValueError("unreceipted PDF file exists")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            rec.update(relative_path=str(path.relative_to(root)), bytes=len(body),
                       sha256=hashlib.sha256(body).hexdigest(), pdf_magic=body.startswith(b"%PDF-"))
            if not rec["pdf_magic"]:
                raise ValueError("official attachment does not contain PDF bytes")
            rec["status"] = "ok"
        except Exception as error:
            rec.update(status="failed", error=type(error).__name__ + ": " + str(error))
        with receipts_path.open("a") as out:
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            out.flush()
        if rec["status"] != "ok":
            state.update(status="failed", completed_new_PDFs=index,
                         failed_month=month, error=rec["error"])
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            raise ValueError(f"PDF {month} failed: {rec['error']}")
        if (index + 1) % 10 == 0 or index + 1 == len(request_months):
            state["completed_new_PDFs"] = index + 1
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"official PDF files {index + 1}/{len(request_months)}", flush=True)
    state.update(status="completed", completed_new_PDFs=len(request_months))
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return state


if __name__ == "__main__":
    import sys
    print(json.dumps(acquire(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())))
