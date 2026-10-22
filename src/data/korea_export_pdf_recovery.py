"""One bounded tail recovery after the first PDF source process terminated."""
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


def recover(root: Path) -> dict:
    root = root.resolve()
    card = root / "research/korea_semiconductor_exports_source_20261022"
    plan_path = card / "english_pdf_recovery_plan.json"
    plan = json.loads(plan_path.read_text())
    if not plan["frozen"] or sha(card / "english_pdf_acquisition_plan.json") != plan["original_plan_sha256"]:
        raise ValueError("PDF recovery original plan differs")
    if sha(card / "english_pdf_receipts.jsonl") != plan["original_receipts_sha256"]:
        raise ValueError("PDF failure receipt history differs")
    if sha(card / "english_pdf_run_state.json") != plan["original_failed_state_sha256"]:
        raise ValueError("PDF failed state differs")
    if sha(card / "article_html_audit.json") != plan["article_audit_sha256"]:
        raise ValueError("PDF link inventory differs")
    audit = json.loads((card / "article_html_audit.json").read_text())
    original = [json.loads(line) for line in (card / "english_pdf_receipts.jsonl").read_text().splitlines() if line]
    if len(original) != plan["successful_prefix_new_PDFs"] + 1:
        raise ValueError("original PDF prefix length differs")
    if original[-1]["status"] != "failed" or original[-1]["month"] != plan["failed_month"]:
        raise ValueError("terminal PDF failure differs")
    for item in original[:-1]:
        if item["status"] != "ok" or sha(root / item["relative_path"]) != item["sha256"]:
            raise ValueError("successful PDF prefix bytes differ")
    months = plan["remaining_fixed_months"]
    if months[0] != plan["failed_month"] or len(months) != plan["request_policy"]["max_new_distinct_HTTP_gets"]:
        raise ValueError("recovery month list differs")
    receipts_path = card / "english_pdf_recovery_receipts.jsonl"
    if receipts_path.exists():
        raise ValueError("recovery already attempted; inspect its receipt before any further operation")
    raw_root = root / "data/snapshots/korea_semiconductor_exports_source_20261022/article_pdf"
    state = {"status": "running", "plan_sha256": sha(plan_path), "completed_recovery_PDFs": 0,
             "planned_recovery_PDFs": len(months)}
    state_path = card / "english_pdf_recovery_run_state.json"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    for index, month in enumerate(months):
        url = audit["month_details"][month]["PDF_links"][0]["url"]
        rec = {"month": month, "url": url,
               "requested_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        body = None
        for attempt in range(2 if plan["request_policy"]["one_retry_per_month"] else 1):
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
                raise ValueError("recovery PDF byte budget exceeded")
            path = raw_root / (month + ".pdf")
            if path.exists():
                raise ValueError("recovery PDF would overwrite existing source")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            rec.update(relative_path=str(path.relative_to(root)), bytes=len(body),
                       sha256=hashlib.sha256(body).hexdigest(), pdf_magic=body.startswith(b"%PDF-"))
            if not rec["pdf_magic"]:
                raise ValueError("recovery response not PDF")
            rec["status"] = "ok"
        except Exception as error:
            rec.update(status="failed", error=type(error).__name__ + ": " + str(error))
        with receipts_path.open("a") as out:
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            out.flush()
        if rec["status"] != "ok":
            state.update(status="failed", completed_recovery_PDFs=index,
                         failed_month=month, error=rec["error"])
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            raise ValueError(f"recovery PDF {month} failed: {rec['error']}")
        state["completed_recovery_PDFs"] = index + 1
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(f"recovered PDF {index + 1}/{len(months)}", flush=True)
    state["status"] = "completed"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return state


if __name__ == "__main__":
    import sys
    print(json.dumps(recover(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())))
