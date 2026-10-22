"""Immutable capture of every selected Korean national export PDF version."""
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
    card = root / "research/korea_semiconductor_exports_korean_vintage_20261022"
    plan_path = card / "korean_PDF_acquisition_plan.json"
    plan = json.loads(plan_path.read_text())
    audit = json.loads((card / "korean_article_link_audit.json").read_text())
    if not plan["frozen"] or not plan["execution_authorized"]:
        raise ValueError("Korean PDF plan not frozen")
    if sha(card / "korean_article_link_audit.json") != plan["article_link_audit_sha256"]:
        raise ValueError("Korean PDF link audit changed")
    items = [(month, index, link) for month in sorted(audit["articles_by_month"])
             for index, link in enumerate(audit["articles_by_month"][month]["PDF_links"], start=1)]
    if len(audit["articles_by_month"]) != plan["months"] or len(items) != plan["total_attachment_links"]:
        raise ValueError("Korean PDF attachment count differs")
    for month, saved in plan["reused_exact_samples"].items():
        if len(audit["articles_by_month"][month]["PDF_links"]) != 1:
            raise ValueError("reused Korean PDF has multiple link versions")
        link = audit["articles_by_month"][month]["PDF_links"][0]
        if link["url"] != saved["original_url"] or sha(root / saved["path"]) != saved["sha256"]:
            raise ValueError("reused Korean PDF source changed")
    requested = [(m, i, link) for m, i, link in items if m not in plan["reused_exact_samples"]]
    if len(requested) != plan["new_PDF_gets"]:
        raise ValueError("Korean PDF new request count differs")
    receipt_path = card / "korean_PDF_receipts.jsonl"
    prior = [json.loads(line) for line in receipt_path.read_text().splitlines() if line] if receipt_path.exists() else []
    if len(prior) > len(requested):
        raise ValueError("Korean PDF receipts exceed plan")
    for n, rec in enumerate(prior):
        month, index, link = requested[n]
        if (rec["month"], rec["version_index"], rec["url"], rec["status"]) != (
            month, index, link["url"], "ok"):
            raise ValueError("Korean PDF resume prefix differs")
        if sha(root / rec["relative_path"]) != rec["sha256"]:
            raise ValueError("Korean PDF resume bytes differ")
    state_path = card / "korean_PDF_run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["plan_sha256"] != sha(plan_path):
            raise ValueError("Korean PDF run state plan differs")
        if state["status"] == "completed":
            return state
    else:
        state = {"status": "running", "plan_sha256": sha(plan_path),
                 "completed_new_PDFs": len(prior), "planned_new_PDFs": len(requested)}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    raw_root = root / "data/snapshots/korea_semiconductor_exports_korean_vintage_20261022/article_pdf"
    for n in range(len(prior), len(requested)):
        month, index, link = requested[n]
        rec = {"month": month, "version_index": index, "filename_label": link["label"],
               "url": link["url"], "requested_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        body = None
        for attempt in range(2 if plan["request_policy"]["one_retry_on_transient"] else 1):
            rec["wire_attempts"] = attempt + 1
            try:
                with urlopen(Request(link["url"], headers={"User-Agent": "Mozilla/5.0",
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
                raise ValueError(rec.get("error", "missing official Korean PDF HTTP200"))
            if len(body) > plan["request_policy"]["max_bytes_per_response"]:
                raise ValueError("Korean PDF response byte budget exceeded")
            path = raw_root / (month + "_" + str(index) + ".pdf")
            if path.exists():
                raise ValueError("unreceipted Korean PDF file exists")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            rec.update(relative_path=str(path.relative_to(root)), bytes=len(body),
                       sha256=hashlib.sha256(body).hexdigest(), pdf_magic=body.startswith(b"%PDF-"))
            if not rec["pdf_magic"]:
                raise ValueError("official Korean attachment is not PDF")
            rec["status"] = "ok"
        except Exception as error:
            rec.update(status="failed", error=type(error).__name__ + ": " + str(error))
        with receipt_path.open("a") as out:
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            out.flush()
        if rec["status"] != "ok":
            state.update(status="failed", completed_new_PDFs=n,
                         failed_month=month, failed_version_index=index, error=rec["error"])
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            raise ValueError(f"Korean PDF {month} version {index} failed: {rec['error']}")
        state["completed_new_PDFs"] = n + 1
        if (n + 1) % 10 == 0 or n + 1 == len(requested):
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"Korean PDF files {n + 1}/{len(requested)}", flush=True)
        time.sleep(plan["request_policy"]["minimum_seconds_between_gets"])
    state["status"] = "completed"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return state


if __name__ == "__main__":
    import sys
    print(json.dumps(acquire(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())))
