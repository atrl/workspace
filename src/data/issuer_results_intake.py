"""Bounded original issuer release collection, including explicit preliminary releases."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup
import pandas as pd

from src.data.issuer_result_tables import parse_quarterly_results


def collect_issuer_results(directory, plan):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    def fetch(url):
        if urlparse(url).hostname not in {"ir.amd.com", "www.intc.com", "nvidianews.nvidia.com", "investor.nvidia.com", "www.nvidia.com"}:
            raise ValueError("issuer source hostname outside fixed scope")
        path = directory / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".html")
        receipt = path.with_suffix(".receipt.json")
        if not (path.exists() and receipt.exists()):
            partial = path.with_suffix(".partial")
            result = subprocess.run(["curl", "--http1.1", "--compressed", "--location", "--fail", "--silent", "--show-error", "--max-time", "30", url, "-o", str(partial)], capture_output=True)
            if result.returncode:
                partial.unlink(missing_ok=True)
                raise ValueError("issuer fetch failed: " + result.stderr.decode()[:140])
            partial.replace(path)
            receipt.write_text(json.dumps({"url": url, "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
        raw = path.read_bytes(); metadata = json.loads(receipt.read_text())
        if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
            raise ValueError("issuer cached bytes differ")
        return raw, metadata
    def index(item):
        symbol, year = item
        if symbol == "NVDA":
            origin = "https://nvidianews.nvidia.com"
            raw, receipt = fetch(origin + f"/news?q=financial%20results&page={year}")
            soup = BeautifulSoup(raw, "html.parser"); rows = []
            for node in soup.select(".index-item-text"):
                anchor = node.select_one("h3 a[href]"); date = node.select_one(".index-item-text-info-date")
                if anchor is None or date is None:
                    continue
                title = anchor.get_text(" ", strip=True)
                if not re.search(r"NVIDIA.*(?:financial results|updates financial guidance)", title, re.I):
                    continue
                rows.append({"symbol": symbol, "report_date": str(pd.Timestamp(date.get_text(strip=True)).date()),
                    "title": title, "url": urljoin(origin, anchor["href"]), "published_clock_raw": None,
                    "published_clock_display": date.get_text(strip=True),
                    "event_role": "preliminary" if re.search("preliminary|updates financial guidance", title, re.I) else "quarterly"})
            return receipt, rows
        origin = "https://ir.amd.com" if symbol == "AMD" else "https://www.intc.com"
        raw, receipt = fetch(origin + f"/news-events/press-releases?category=financial&year={year}")
        soup = BeautifulSoup(raw, "html.parser"); candidates = {}
        for anchor in soup.select('a[href*="/press-releases/detail/"]'):
            title = anchor.get_text(" ", strip=True)
            if not re.search(r"(?:reports|preliminary).*financial results", title, re.I):
                continue
            ancestor = anchor.parent
            for _ in range(6):
                time = ancestor.find("time", attrs={"datetime": True})
                if time is not None:
                    break
                ancestor = ancestor.parent
            if time is None:
                raise ValueError("issuer index release date absent")
            date = str(pd.Timestamp(time["datetime"]).date())
            candidates[anchor["href"]] = {"symbol": symbol, "report_date": date, "title": title, "url": anchor["href"],
                "published_clock_raw": time["datetime"], "published_clock_display": time.get_text(" ", strip=True),
                "event_role": "preliminary" if "preliminary" in title.lower() else "quarterly"}
        return receipt, list(candidates.values())
    index_items = [(symbol, year) for symbol in ["AMD", "INTC"] for year in plan["index_years"]]
    index_items += [("NVDA", page) for page in range(1, plan["nvidia_index_pages"] + 1)]
    with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
        indexes = list(pool.map(index, index_items))
    catalog = [entry for _, rows in indexes for entry in rows]
    catalog.extend(plan.get("additional_quarterly_sources", []))
    selected, failures = [], []
    for event in plan["events"]:
        matches = [row for row in catalog if row["symbol"] == event["symbol"] and row["report_date"] == event["report_date"] and row["event_role"] == "quarterly"]
        overrides = [row for row in matches if row.get("replace_current_web_page")]
        if overrides:
            matches = overrides
        if len(matches) == 1:
            selected.append({**event, **matches[0]})
        else:
            failures.append({**event, "error": "expected exactly one issuer release on calendar date", "matching_rows": len(matches)})
    selected.extend(row for row in catalog if row["event_role"] == "preliminary" and plan["start"] <= row["report_date"] <= plan["as_of"])
    selected.extend(plan["additional_preliminary_sources"])
    selected = list({(row["symbol"], row["report_date"], row["event_role"]): row for row in selected}.values())
    if len(selected) + len(indexes) > plan["maximum_documents"]:
        raise ValueError("fixed issuer acquisition budget exceeded")
    def article(item):
        try:
            raw, receipt = fetch(item["url"])
            parsed = parse_quarterly_results(raw) if item["event_role"] == "quarterly" else None
            if parsed is not None:
                for basis, value in parsed["basis"].items():
                    match = re.fullmatch(r"Q([1-4])\s*(?:FY)?\s*(\d{2,4})", value["quarter_label"], re.I)
                    quarter, year = int(match[1]), int(match[2])
                    year = year + 2000 if year < 100 else year
                    if (quarter, year) != (item["expected_quarter"], item["expected_year"]):
                        raise ValueError(f"{basis} summary period differs from frozen fiscal period")
            return {**item, **receipt, "parsed": parsed, "original_publication_bytes_certified": False}, None
        except Exception as error:
            return None, {**item, "error": str(error)}
    with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
        results = list(pool.map(article, selected))
    manifest = {"indexes": [receipt for receipt, _ in indexes], "catalog": catalog,
        "entries": [entry for entry, _ in results if entry], "failures": failures + [failure for _, failure in results if failure],
        "explicit_preliminary_results_only": True, "all_unscheduled_disclosures_complete": False}
    (directory / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
