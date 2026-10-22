"""Prospective, source-only exact-byte Korean monthly export release capture."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from src.data.korea_export_korean_article import identity
from src.data.korea_export_korean_listing import listing_rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_receipt(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("prospective Korean release phase already attempted")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _title_matches(title: str) -> bool:
    normalized = re.sub(r"\s+", "", title)
    return ("2026년10월" in normalized and "수출입동향" in normalized
            and "ICT" not in normalized.upper() and "정보통신" not in normalized)


def _attachment_links(body: bytes, allowed: set[str]) -> list[dict]:
    soup = BeautifulSoup(body, "html.parser")
    links = []
    seen = set()
    for anchor in soup.find_all("a", href=re.compile(r"^/attach/down/")):
        label = anchor.get_text(" ", strip=True)
        match = re.search(r"\.([A-Za-z0-9]+)(?:\s|\[|$)", label)
        kind = match.group(1).lower() if match else None
        if kind not in allowed:
            continue
        url = urljoin("https://www.motir.go.kr", anchor["href"])
        if url not in seen:
            links.append({"url": url, "label": label, "kind": kind})
            seen.add(url)
    return links


def _capture(root: Path, phase: str, *, now: dt.datetime, fetch=urlopen, clock=None) -> dict:
    root = Path(root).resolve()
    card = root / "research/korea_semiconductor_exports_korean_vintage_20261022"
    plan_path = card / "forward_oct_2026_publication_plan.json"
    plan = json.loads(plan_path.read_text())
    if not plan["frozen"] or phase not in plan["observations"]:
        raise ValueError("phase is outside frozen Korean source plan")
    for name, digest in [(plan["source_listing_parser"], plan["source_listing_parser_sha256"]),
                         (plan["source_article_parser"], plan["source_article_parser_sha256"])]:
        if _sha(root / name) != digest:
            raise ValueError("Korean source parser identity changed: " + name)
    if _sha(card / "korean_national_article_manifest.json") != plan["source_announcement_index_sha256"]:
        raise ValueError("prior Korean announcement index changed")
    binding_path = card / plan["implementation_binding"]
    if not binding_path.is_file():
        raise ValueError("forward Korean implementation binding is missing")
    binding = json.loads(binding_path.read_text())
    if (not binding.get("frozen") or binding.get("plan_sha256") != _sha(plan_path)
            or binding.get("implementation_sha256") != _sha(root / plan["implementation_module"])):
        raise ValueError("forward Korean implementation identity changed")
    first = dt.datetime.fromisoformat(plan["observations"][phase]["not_before_utc"])
    last = dt.datetime.fromisoformat(plan["observations"][phase]["not_after_utc"])
    if now.tzinfo is None:
        raise ValueError("UTC observation time must be timezone-aware")
    now = now.astimezone(dt.timezone.utc)
    if now < first:
        raise ValueError("observation before frozen Korean source window")
    receipt_path = card / "forward_oct_2026" / (phase + ".json")
    if receipt_path.exists():
        raise ValueError("prospective Korean release phase already attempted")
    common = {"schema_version": 1, "phase": phase, "target_reference_month": plan["target_reference_month"],
              "plan_sha256": _sha(plan_path), "binding_sha256": _sha(binding_path),
              "observed_at_utc": now.isoformat(), "market_outcome_reads": 0,
              "model_fits": 0, "accounts": 0, "PIT_qualified": False}
    if now > last:
        result = {**common, "status": "missed_window", "http_GETs": 0, "records": []}
        _save_receipt(receipt_path, result)
        return result
    previous = list((card / "forward_oct_2026").glob("*.json"))
    consumed = sum(json.loads(path.read_text()).get("http_GETs", 0) for path in previous)
    if consumed + 7 > plan["budget"]["max_HTTP_GETs_total"]:
        raise ValueError("prospective Korean source HTTP budget exhausted")
    raw_dir = root / "data/snapshots/korea_semiconductor_exports_korean_vintage_20261022/forward_oct_2026" / phase
    if raw_dir.exists():
        raise ValueError("unreceipted prospective Korean source bytes exist")
    raw_dir.mkdir(parents=True)
    if clock is None:
        clock = lambda: dt.datetime.now(dt.timezone.utc)
    records = []
    attempts = 0

    def get(url: str, name: str, limit: int, accept: str) -> tuple[bytes | None, dict]:
        nonlocal attempts
        requested = clock()
        if requested.tzinfo is None:
            raise ValueError("request clock must be timezone-aware")
        requested = requested.astimezone(dt.timezone.utc)
        row = {"kind": name, "url": url, "requested_at_utc": requested.isoformat()}
        try:
            if not first <= requested <= last:
                raise ValueError("request outside frozen Korean source window")
            attempts += 1
            with fetch(Request(url, headers={"User-Agent": "Mozilla/5.0 research source qualification",
                                             "Accept": accept}), timeout=30) as response:
                body = response.read(limit + 1)
                row.update(http_status=response.status, content_type=response.headers.get("Content-Type"),
                           response_date=response.headers.get("Date"), response_age=response.headers.get("Age"),
                           cache_control=response.headers.get("Cache-Control"),
                           last_modified=response.headers.get("Last-Modified"))
            if row["http_status"] != 200 or len(body) > limit:
                raise ValueError("official Korean source HTTP/byte budget differs")
            suffix = "pdf" if name.endswith(".pdf") else "bin" if name.startswith("attachment_") else "html"
            path = raw_dir / (name + "." + suffix if "." not in name else name)
            path.write_bytes(body)
            row.update(relative_path=str(path.relative_to(root)), bytes=len(body),
                       sha256=hashlib.sha256(body).hexdigest())
            return body, row
        except Exception as error:
            row["error"] = type(error).__name__ + ": " + str(error)
            return None, row

    matched = {}
    for index, url in enumerate(plan["official_listing_urls"], 1):
        body, record = get(url, f"listing_{index}", plan["max_response_bytes"]["listing"], "text/html,*/*")
        records.append(record)
        if body is None:
            continue
        try:
            rows = listing_rows(body)
            record["listing_rows"] = len(rows)
            for item in rows:
                if _title_matches(item["title"]):
                    matched[item["article_id"]] = item
        except Exception as error:
            record["error"] = type(error).__name__ + ": " + str(error)
    if len(matched) != 1:
        result = {**common, "status": "target_absent" if not matched and all("error" not in r for r in records)
                  else "ambiguous_or_failed_listing", "http_GETs": attempts,
                  "matching_article_ids": sorted(matched), "records": records}
        _save_receipt(receipt_path, result)
        return result
    article = next(iter(matched.values()))
    url = urljoin("https://www.motir.go.kr", article["href"].split("?", 1)[0])
    body, record = get(url, "article", plan["max_response_bytes"]["article"], "text/html,*/*")
    records.append(record)
    links = []
    if body is not None:
        try:
            title, registration = identity(body)
            if not _title_matches(title) or registration != article["registration_date"]:
                raise ValueError("article identity differs from official listing")
            record.update(article_id=article["article_id"], title=title, registration_date=registration)
            links = _attachment_links(body, set(plan["allowed_attachment_types"]))
            if not links or len(links) > plan["max_attachment_links_per_observation"]:
                raise ValueError("attachment inventory differs from frozen budget")
            record["attachment_links"] = links
        except Exception as error:
            record["error"] = type(error).__name__ + ": " + str(error)
            links = []
    for index, link in enumerate(links, 1):
        name = f"attachment_{index}.{link['kind']}"
        body, item = get(link["url"], name, plan["max_response_bytes"]["attachment"],
                         "application/pdf,application/octet-stream,*/*")
        item.update(label=link["label"], attachment_type=link["kind"])
        if body is not None:
            magic = (body.startswith(b"%PDF-") if link["kind"] == "pdf"
                     else body.startswith(b"PK") if link["kind"] == "hwpx"
                     else body.startswith(b"\xd0\xcf\x11\xe0"))
            item["file_magic_valid"] = magic
            if not magic:
                item["error"] = "attachment magic differs"
        records.append(item)
    result = {**common, "status": "observed" if links and all("error" not in r for r in records)
              else "partial_failure", "http_GETs": attempts,
              "matching_article_ids": sorted(matched), "records": records}
    _save_receipt(receipt_path, result)
    return result


def capture(root: str | Path, phase: str) -> dict:
    """Capture one frozen phase using the actual current UTC clock."""
    return _capture(Path(root), phase, now=dt.datetime.now(dt.timezone.utc))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.data.korea_export_forward_capture PHASE")
    print(json.dumps(capture(Path.cwd(), sys.argv[1]), ensure_ascii=False, indent=2))
