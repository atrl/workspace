"""Strict monthly GSR PDF table extraction with revision and timestamp boundaries."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from pypdf import PdfReader


def parse_sales_pdf(raw):
    reader = PdfReader(io.BytesIO(raw))
    text = re.sub(r"\s+", " ", reader.pages[0].extract_text()).strip()
    month = re.match(r"([A-Za-z]+)\s+(20\d{2})\s+(Billions|000's omitted)", text)
    if not month:
        raise ValueError("monthly heading or billions unit absent")
    period = pd.Timestamp(f"{month[1]} 1 {month[2]}")
    scale = 1 if month[3] == "Billions" else 1_000_000
    headings = ["Month-to-Month Sales", "Year-to-Year Sales", "Three-Month-Moving Average Sales"]
    tables = {}
    number = r"(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    names = ["Americas", "Europe", "Japan", "China", "Asia Pacific/All Other", "Total"]
    for index, heading in enumerate(headings):
        if text.count(heading) != 1:
            raise ValueError("monthly table heading missing or repeated")
        section = text.split(heading)[1]
        if index < 2:
            section = section.split(headings[index + 1])[0]
        rows = {}
        for name in names:
            matches = re.findall(re.escape(name) + r"\s+" + number + r"\s+" + number + r"\s+" + number + r"\s*%", section)
            if len(matches) != 1:
                raise ValueError(f"expected one {name} row in {heading}")
            tokens = [token.replace(",", "") for token in matches[0]]
            previous, current, percentage = map(float, tokens)
            precision = [.5 * 10 ** (-len(token.split(".")[1]) if "." in token else 0) for token in tokens]
            previous, current = previous / scale, current / scale
            p_half, c_half, pct_half = precision[0] / scale, precision[1] / scale, precision[2]
            if min(previous, current) <= 0:
                raise ValueError("nonpositive billings")
            lower = ((current - c_half) / (previous + p_half) - 1) * 100
            upper = ((current + c_half) / (previous - p_half) - 1) * 100
            if percentage + pct_half + 1e-6 < lower or percentage - pct_half - 1e-6 > upper:
                raise ValueError("reported percentage inconsistent with rounded amounts")
            rows[name] = {"previous": previous, "current": current, "percentage": percentage, "current_rounding_half": c_half}
        if abs(sum(rows[name]["current"] for name in names[:-1]) - rows["Total"]["current"]) > sum(row["current_rounding_half"] for row in rows.values()) + 1e-8:
            raise ValueError("regional amounts fail rounding-aware total reconciliation")
        tables[heading] = rows
    if len({table["Total"]["current"] for table in tables.values()}) != 1:
        raise ValueError("current total differs across tables")
    metadata = reader.metadata
    dates = {}
    for field, value in [("created_at", metadata.creation_date), ("modified_at", metadata.modification_date)]:
        if value is None or value.tzinfo is None:
            dates[field] = None
        else:
            dates[field] = pd.Timestamp(value).tz_convert("UTC").isoformat()
    return {"observation_month": str(period.date()), "tables": tables, "reported_units": month[3], **dates,
        "sales_billion_usd": tables[headings[0]]["Total"]["current"],
        "sales_mom_pct": tables[headings[0]]["Total"]["percentage"],
        "sales_yoy_pct": tables[headings[1]]["Total"]["percentage"],
        "regional_yoy_dispersion": float(np.std([tables[headings[1]][name]["percentage"] for name in names[:-1]])),
        "pdf_pages": len(reader.pages), "file_metadata_is_publication_proof": False}


def collect_sales_pdfs(*, html_manifest, html_documents, directory, maximum_documents=110, workers=3):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    inventory = []
    for article in html_manifest["entries"]:
        raw = html_documents[article["path"]]
        if hashlib.sha256(raw).hexdigest() != article["sha256"]:
            raise ValueError("frozen HTML source digest differs")
        soup = BeautifulSoup(raw, "html.parser")
        links = sorted({urljoin(article["url"], a["href"]) for a in soup.select("main#main a[href]")
            if ".pdf" in a["href"].lower() and any(word in a["href"].lower() for word in ["graph", "chart", "gsr", "table"])
            and re.search(r"(?:Jan\w*|Feb\w*|Mar\w*|Apr\w*|May|Jun\w*|Jul\w*|Aug\w*|Sep\w*|Oct\w*|Nov\w*|Dec\w*)[-_ ]+20\d{2}", a["href"], re.I)})
        if len(links) > 1:
            raise ValueError("ambiguous source attachment")
        if links:
            inventory.append({"url": links[0], "article_url": article["url"], "published_at_utc": article["published_at_utc"],
                "article_modified_at_utc": article["modified_at_utc"], "article_three_month_average_explicit": article["monthly_three_month_average_explicit"]})
    if len(inventory) > maximum_documents or not inventory:
        raise ValueError("fixed PDF acquisition budget violated")
    def acquire(item):
        name = hashlib.sha256(item["url"].encode()).hexdigest()[:20] + ".pdf"
        path = directory / name; receipt_path = directory / (name + ".receipt.json")
        try:
            if not (path.exists() and receipt_path.exists()):
                partial = path.with_suffix(".partial")
                result = subprocess.run(["curl", "--http1.1", "--compressed", "--location", "--fail", "--silent", "--show-error",
                    "--max-time", "30", item["url"], "-o", str(partial)], capture_output=True)
                if result.returncode:
                    partial.unlink(missing_ok=True)
                    raise ValueError("PDF fetch failed")
                partial.replace(path)
                receipt_path.write_text(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
            raw = path.read_bytes(); receipt = json.loads(receipt_path.read_text())
            if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
                raise ValueError("PDF source digest differs")
            parsed = parse_sales_pdf(raw)
            if pd.Timestamp(parsed["observation_month"]).tz_localize("UTC") >= pd.Timestamp(item["published_at_utc"]):
                raise ValueError("observation month not before release")
            return {**item, **receipt, "path": name, "bytes": len(raw), **parsed}, None
        except Exception as error:
            return None, {**item, "path": name, "error": str(error)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(acquire, inventory))
    manifest = {"expected_documents": len(inventory), "entries": [entry for entry, _ in results if entry],
        "failures": [failure for _, failure in results if failure]}
    (directory / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
