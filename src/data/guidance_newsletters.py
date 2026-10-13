"""Source-only next-session earnings narrative inventory; no inferred consensus values."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

import pandas as pd
from pypdf import PdfReader

from src.data.refinitiv_newsletter import COMPANY_NAMES


def guidance_paragraphs(raw, symbol):
    pdf = PdfReader(io.BytesIO(raw))
    created, modified = pdf.metadata.creation_date, pdf.metadata.modification_date
    clocks = {"created_at": str(created) if created else None, "modified_at": str(modified) if modified else None,
        "clock_timezone_known": all(value is not None and value.tzinfo is not None for value in [created, modified])}
    text = re.sub(r"\s+", " ", "\n".join(page.extract_text() for page in pdf.pages))
    start = text.upper().find("STOCKS TO WATCH")
    if start < 0:
        return {"paragraphs": [], "scope_found": False, **clocks}
    section = text[start:]
    boundary = re.search(r"ANALYSTS?['’]?\s+RECOMMENDATIONS?|ECONOMIC EVENTS|COMPANIES REPORTING RESULTS", section, re.I)
    if boundary:
        section = section[:boundary.start()]
    blocks = re.split(r"[•]", section)
    prefix = r"\s*" + COMPANY_NAMES[symbol] + r"(?:\s*\(" + re.escape(symbol) + r"(?:\.O)?\))?\s*[.:]"
    company = [block.strip() for block in blocks if re.match(prefix, block, re.I)]
    return {"paragraphs": company, "scope_found": True, **clocks, "page_count": len(pdf.pages),
        "guidance_word_present": any(re.search(r"\bforecast|\bguidance|\boutlook|\bexpects?\b", block, re.I) for block in company),
        "quarter_word_present": any(re.search(r"\bquarter", block, re.I) for block in company),
        "estimate_word_present": any(re.search(r"\bestimat|\bconsensus|\bexpectations", block, re.I) for block in company),
        "numeric_pair_qualified": False, "target_quarter_qualified": False}


def collect_guidance_newsletters(directory, plan):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    if len(plan["events"]) > plan["maximum_documents"]:
        raise ValueError("guidance source acquisition budget exceeded")
    def collect(event):
        date = pd.Timestamp(event["newsletter_date"]).strftime("%m%d%Y")
        url = "https://share.refinitiv.com/assets/newsletters/Morning_News_Call/MNCGeneric_US_" + date + ".pdf"
        path = directory / (date + ".pdf"); receipt = path.with_suffix(".receipt.json")
        try:
            if not (path.exists() and receipt.exists()):
                partial = path.with_suffix(".partial")
                command = ["curl", "--http1.1", "--compressed", "--fail", "--silent", "--show-error", "--max-time", "30", url, "-o", str(partial)]
                result = subprocess.run(command, capture_output=True); verified = result.returncode == 0
                if result.returncode == 60 and b"certificate has expired" in result.stderr and plan["allow_expired_public_certificate_probe"]:
                    result = subprocess.run([command[0], "--insecure", *command[1:]], capture_output=True)
                if result.returncode:
                    partial.unlink(missing_ok=True)
                    raise ValueError(result.stderr.decode()[:180])
                partial.replace(path)
                receipt.write_text(json.dumps({"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "tls_certificate_verified": verified}, indent=2) + "\n")
            raw = path.read_bytes(); meta = json.loads(receipt.read_text())
            if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
                raise ValueError("frozen guidance newsletter hash differs")
            parsed = guidance_paragraphs(raw, event["symbol"])
            return {**event, **meta, "path": path.name, **parsed}, None
        except Exception as error:
            return None, {**event, "url": url, "path": path.name, "error": str(error)}
    with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
        results = list(pool.map(collect, plan["events"]))
    manifest = {"entries": [entry for entry, _ in results if entry], "failures": [failure for _, failure in results if failure]}
    (directory / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
