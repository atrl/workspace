"""Bounded first-observed TSMC monthly press release source capture."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("TSMC forward phase already attempted")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _title(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _press_links(body: bytes, target: str) -> list[str]:
    soup = BeautifulSoup(body, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        label, expected = _title(anchor.get_text(" ", strip=True)), _title(target)
        if label == expected or re.fullmatch(r"\d{4}/\d{2}/\d{2} " + re.escape(expected), label):
            url = urljoin("https://pr.tsmc.com", anchor["href"])
            if urlparse(url).hostname == "pr.tsmc.com" and re.fullmatch(r"/english/news/\d+", urlparse(url).path):
                links.append(url)
    return list(dict.fromkeys(links))


def _pdf_links(body: bytes) -> list[str]:
    soup = BeautifulSoup(body, "html.parser")
    urls = [urljoin("https://pr.tsmc.com", a["href"]) for a in soup.find_all("a", href=True)
            if a["href"].startswith("/system/files/newspdf/") and a["href"].lower().endswith(".pdf")]
    return list(dict.fromkeys(urls))


def _capture(root: Path, phase: str, *, now: dt.datetime, fetch=urlopen, clock=None) -> dict:
    root = Path(root).resolve()
    card = root / "research/tsmc_monthly_revenue_release_source_20261022"
    plan_path = card / "future_2026_10_press_capture_plan.json"
    plan = json.loads(plan_path.read_text())
    if not plan["frozen"] or phase not in plan["observations"]:
        raise ValueError("TSMC phase outside frozen plan")
    for name, digest in [("future_2026_10_revenue_calendar_probe.json", plan["calendar_probe_sha256"]),
                         ("alternate_issuer_press_review.json", plan["alternate_issuer_source_review_sha256"])]:
        if _sha(card / name) != digest:
            raise ValueError("TSMC source identity differs: " + name)
    binding_path = card / plan["implementation_binding"]
    if not binding_path.is_file():
        raise ValueError("TSMC implementation binding missing")
    binding = json.loads(binding_path.read_text())
    if (not binding.get("frozen") or binding.get("plan_sha256") != _sha(plan_path)
            or binding.get("implementation_sha256") != _sha(root / plan["implementation_module"])):
        raise ValueError("TSMC capture implementation differs")
    start = dt.datetime.fromisoformat(plan["observations"][phase]["not_before_utc"])
    end = dt.datetime.fromisoformat(plan["observations"][phase]["not_after_utc"])
    if now.tzinfo is None:
        raise ValueError("UTC observation clock must be timezone-aware")
    now = now.astimezone(dt.timezone.utc)
    if now < start:
        raise ValueError("TSMC observation before frozen window")
    receipt_path = card / "future_2026_10_press" / (phase + ".json")
    if receipt_path.exists():
        raise ValueError("TSMC forward phase already attempted")
    common = {"schema_version": 1, "phase": phase, "target_reference_month": plan["target_reference_month"],
              "plan_sha256": _sha(plan_path), "binding_sha256": _sha(binding_path),
              "observed_at_utc": now.isoformat(), "market_outcome_reads": 0,
              "model_fits": 0, "accounts": 0, "strategy_forward_day": False}
    if now > end:
        receipt = {**common, "status": "missed_window", "records": [], "http_GETs": 0}
        _save(receipt_path, receipt)
        return receipt
    previous = list((card / "future_2026_10_press").glob("*.json"))
    consumed = sum(json.loads(path.read_text()).get("http_GETs", 0) for path in previous)
    if consumed + 4 > plan["budget"]["max_HTTP_GETs_total"]:
        raise ValueError("TSMC forward HTTP budget exhausted")
    raw_dir = root / "data/snapshots/tsmc_monthly_revenue_release_source_20261022/forward_oct_2026" / phase
    if raw_dir.exists():
        raise ValueError("unreceipted TSMC forward raw bytes exist")
    raw_dir.mkdir(parents=True)
    if clock is None:
        clock = lambda: dt.datetime.now(dt.timezone.utc)
    records = []
    attempts = 0

    def get(url: str, label: str, limit: int, accept: str, host: str) -> tuple[bytes | None, dict]:
        nonlocal attempts
        requested = clock()
        if requested.tzinfo is None:
            raise ValueError("request clock must be timezone-aware")
        requested = requested.astimezone(dt.timezone.utc)
        item = {"kind": label, "url": url, "requested_at_utc": requested.isoformat()}
        try:
            if not start <= requested <= end:
                raise ValueError("request outside frozen TSMC window")
            attempts += 1
            with fetch(Request(url, headers={"User-Agent": "Mozilla/5.0 research source qualification",
                                             "Accept": accept}), timeout=30) as response:
                body = response.read(limit + 1)
                item.update(http_status=response.status, content_type=response.headers.get("Content-Type"),
                            response_date=response.headers.get("Date"), response_age=response.headers.get("Age"),
                            cache_control=response.headers.get("Cache-Control"),
                            last_modified=response.headers.get("Last-Modified"), final_url=response.url)
            if item["http_status"] != 200 or len(body) > limit or urlparse(item["final_url"]).hostname != host:
                raise ValueError("source HTTP/size/host differs")
            path = raw_dir / (label + (".pdf" if label == "report" else ".html"))
            path.write_bytes(body)
            item.update(relative_path=str(path.relative_to(root)), bytes=len(body),
                        sha256=hashlib.sha256(body).hexdigest())
            return body, item
        except Exception as error:
            item["error"] = type(error).__name__ + ": " + str(error)
            return None, item

    calendar, calendar_row = get(plan["official_calendar_url"], "calendar",
                                 plan["budget"]["max_HTML_bytes"], "text/html,*/*", "investor.tsmc.com")
    records.append(calendar_row)
    confirmed = None
    if calendar is not None:
        text = BeautifulSoup(calendar, "html.parser").get_text(" ", strip=True)
        confirmed = (plan["target_title"] in text and "2026-11-10 13:30:00" in text)
        calendar_row["scheduled_time_still_listed"] = confirmed
    listing, listing_row = get(plan["official_press_listing_url"], "listing",
                               plan["budget"]["max_HTML_bytes"], "text/html,*/*", "pr.tsmc.com")
    records.append(listing_row)
    links = _press_links(listing, plan["target_title"]) if listing is not None else []
    listing_row["matching_article_urls"] = links
    if len(links) != 1:
        receipt = {**common, "status": "target_absent" if not links and "error" not in listing_row
                   else "ambiguous_or_failed_listing", "calendar_schedule_confirmed": confirmed,
                   "records": records, "http_GETs": attempts}
        _save(receipt_path, receipt)
        return receipt
    article, article_row = get(links[0], "article", plan["budget"]["max_HTML_bytes"],
                                "text/html,*/*", "pr.tsmc.com")
    records.append(article_row)
    pdf_links = []
    if article is not None:
        text = BeautifulSoup(article, "html.parser").get_text(" ", strip=True)
        article_row["title_present"] = plan["target_title"] in text
        issued = re.search(r"Issued on:\s*(20\d{2}/\d{2}/\d{2})", text)
        article_row["issuer_date"] = issued.group(1) if issued else None
        pdf_links = _pdf_links(article)
        if not article_row["title_present"] or issued is None or len(pdf_links) != 1:
            article_row["error"] = "issuer article title/date/PDF inventory differs"
            pdf_links = []
    if pdf_links:
        report, report_row = get(pdf_links[0], "report", plan["budget"]["max_PDF_bytes"],
                                  "application/pdf,*/*", "pr.tsmc.com")
        records.append(report_row)
        if report is not None and not report.startswith(b"%PDF-"):
            report_row["error"] = "issuer report PDF magic differs"
    source_ok = bool(pdf_links) and all("error" not in row for row in records[1:])
    status = ("source_observed_calendar_unverified" if source_ok and calendar_row.get("error")
              else "source_observed_schedule_changed" if source_ok and confirmed is False
              else "observed" if source_ok and confirmed is True else "partial_failure")
    receipt = {**common, "status": status, "calendar_schedule_confirmed": confirmed,
               "matching_article_urls": links, "records": records, "http_GETs": attempts}
    _save(receipt_path, receipt)
    return receipt


def capture(root: str | Path, phase: str) -> dict:
    """Capture one source phase using actual current UTC."""
    return _capture(Path(root), phase, now=dt.datetime.now(dt.timezone.utc))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.data.tsmc_press_forward_capture PHASE")
    print(json.dumps(capture(Path.cwd(), sys.argv[1]), ensure_ascii=False, indent=2))
