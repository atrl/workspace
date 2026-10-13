"""Bounded SIA article inventory with separate publication and modification clocks."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from bs4 import BeautifulSoup
import pandas as pd


def article_metadata(raw):
    soup = BeautifulSoup(raw, "html.parser")
    main = soup.select_one("main#main")
    if main is None or main.find("h1") is None:
        raise ValueError("article body absent")
    clocks = {}
    for field in ["published", "modified"]:
        node = soup.find("meta", attrs={"property": f"article:{field}_time"})
        if node is None:
            if field == "modified":
                clocks[field + "_at_utc"] = None
                continue
            raise ValueError(f"article {field} timestamp absent")
        value = pd.Timestamp(node.get("content"))
        if value.tzinfo is None:
            raise ValueError("article timestamp has no timezone")
        clocks[field + "_at_utc"] = value.tz_convert("UTC").isoformat()
    published = pd.Timestamp(clocks["published_at_utc"])
    modified = pd.Timestamp(clocks["modified_at_utc"]) if clocks["modified_at_utc"] else pd.NaT
    if pd.notna(modified) and modified < published:
        raise ValueError("modification precedes publication")
    paragraphs = [p.get_text(" ", strip=True) for p in main.find_all("p")]
    # Preserve candidate passages for a later strict parser, not machine-invented values.
    sales = [p for p in paragraphs if re.search(r"(?:global|worldwide).*sales", p, re.I) and "$" in p]
    return {**clocks, "title": main.find("h1").get_text(" ", strip=True),
        "modification_lag_days": (modified - published).total_seconds() / 86400 if pd.notna(modified) else None,
        "modification_clock_known": bool(pd.notna(modified)),
        "monthly_three_month_average_explicit": any(re.search(r"three.month (?:moving )?average", p, re.I) for p in paragraphs),
        "candidate_sales_paragraphs": sales,
        "downloaded_body_is_original_published_vintage": False,
        "historical_modification_content_diff_available": False}


def collect_sia_sources(directory, plan):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    def fetch(url, name):
        path = directory / name; receipt = directory / (name + ".receipt.json")
        if path.exists() and receipt.exists():
            metadata = json.loads(receipt.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
                raise ValueError("frozen SIA cache digest differs")
            return path.read_bytes(), metadata
        partial = path.with_suffix(".partial")
        result = subprocess.run(["curl", "--http1.1", "--compressed", "--location", "--fail", "--silent", "--show-error",
            "--max-time", "30", url, "-o", str(partial)], capture_output=True)
        if result.returncode:
            partial.unlink(missing_ok=True)
            raise ValueError(f"source fetch failed: {url}")
        data = partial.read_bytes(); partial.replace(path)
        metadata = {"url": url, "path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
        receipt.write_text(json.dumps(metadata, indent=2) + "\n")
        return data, metadata
    def index(number):
        base = "https://www.semiconductors.org/policies/tax/market-data/"
        raw, receipt = fetch(base if number == 1 else base + f"page/{number}/", f"index_{number:02}.html")
        soup = BeautifulSoup(raw, "html.parser"); rows = []
        for marker in soup.select(".resource-item-meta"):
            parent = marker.parent; heading = parent.find("h3")
            stamp = re.search(r"(\d{2}/\d{2}/\d{2})", marker.get_text(" ", strip=True))
            if heading is None or stamp is None:
                continue
            title = heading.get_text(" ", strip=True)
            date = pd.to_datetime(stamp[1], format="%m/%d/%y")
            if re.search(r"semiconductor.*sales|sales.*semiconductor", title, re.I) and pd.Timestamp(plan["start"]) <= date <= pd.Timestamp(plan["as_of"]):
                rows.append({"url": heading.parent["href"], "index_date": str(date.date()), "title": title})
        return receipt, rows
    with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
        indexes = list(pool.map(index, range(1, plan["maximum_index_pages"] + 1)))
    inventory = {row["url"]: row for _, rows in indexes for row in rows}
    if not inventory or len(inventory) > plan["maximum_articles"]:
        raise ValueError("SIA inventory empty or exceeds fixed acquisition budget")
    def article(item):
        try:
            name = hashlib.sha256(item["url"].encode()).hexdigest()[:20] + ".html"
            raw, receipt = fetch(item["url"], name)
            metadata = article_metadata(raw)
            public_day = pd.Timestamp(metadata["published_at_utc"]).tz_convert("America/New_York").date()
            if str(public_day) != item["index_date"]:
                raise ValueError("index and publication dates disagree")
            return {**item, **receipt, **metadata}, None
        except (ValueError, OSError) as error:
            return None, {**item, "error": str(error)}
    with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
        results = list(pool.map(article, sorted(inventory.values(), key=lambda row: (row["index_date"], row["url"]))))
    manifest = {"indexes": [receipt for receipt, _ in indexes], "candidate_articles": len(inventory),
        "entries": [entry for entry, _ in results if entry], "failures": [failure for _, failure in results if failure]}
    (directory / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
