"""Read a bounded public newsletter sample; keep estimate columns and clocks distinct."""
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


COMPANY_NAMES = {"NVDA": r"NVIDIA(?: Corp(?:oration)?\.?)?", "AMD": r"Advanced Micro Devices(?:,? Inc\.?)?", "INTC": r"Intel(?: Corp(?:oration)?\.?)?"}
PROVIDER_FOOTERS = (
    "I/B/E/S EPS and revenue estimates, and StarMine Smart Estimates, provided by Refinitiv.",
    "EPS and Revenue estimates are according to Thomson Reuters I/B/E/S. Smart Estimates are according to Thomson Reuters StarMine.",
    "All analysts' estimates are according to LSEG IBES data.",
)


def parse_newsletter(raw, symbol, report_date, family="day_ahead"):
    reader = PdfReader(io.BytesIO(raw))
    pages = [page.extract_text() for page in reader.pages]
    whole = " ".join(pages)
    if family == "morning_call":
        if not any(footer.rstrip(".") in whole for footer in ["All analysts' estimates are according to I/B/E/S Refinitiv data", *PROVIDER_FOOTERS]):
            raise ValueError("morning estimate provider unknown")
        candidates = []
        for number, text in enumerate(pages):
            if "COMPANIES REPORTING RESULTS" not in text:
                continue
            for match in re.findall(r"(" + COMPANY_NAMES[symbol] + r"): Expected (Q[1-4]) (earnings|loss) of (\$[\d.]+|[\d.]+ cents) per share", text, re.I):
                candidates.append((number, match))
        if len(candidates) != 1:
            raise ValueError("expected one morning company EPS row")
        number, (company, quarter, role, token) = candidates[0]
        consensus = float(token[1:]) if token.startswith("$") else float(token.split()[0]) / 100
        if role.lower() == "loss":
            consensus = -abs(consensus)
        created, modified = reader.metadata.creation_date, reader.metadata.modification_date
        known = all(value is not None and value.tzinfo is not None for value in [created, modified])
        latest = max(pd.Timestamp(created), pd.Timestamp(modified)).tz_convert("America/New_York") if known else pd.NaT
        return {"symbol": symbol, "report_date": report_date, "company_name": company, "quarter": quarter,
            "expected_report_time": "unknown", "smart_eps": None, "consensus_eps": consensus, "prior_year_eps": None,
            "consensus_revenue_million_usd": None, "source_page_zero_based": number,
            "pdf_created_at": str(created) if created else None, "pdf_modified_at": str(modified) if modified else None,
            "document_clock_before_report_day": bool(pd.notna(latest) and latest.tz_localize(None) < pd.Timestamp(report_date)),
            "historical_publication_clock_verified": False, "estimate_basis": "I/B/E/S morning EPS snapshot; other estimate fields unavailable",
            "not_final_preannouncement_consensus": True, "raw_split_basis_not_restated": True, "newsletter_family": family}
    if family != "day_ahead":
        raise ValueError("unsupported newsletter family")
    if "KEY RESULTS" not in whole or not any(footer in whole for footer in PROVIDER_FOOTERS):
        raise ValueError("I/B/E/S KEY RESULTS scope absent")
    candidates = []
    for number, text in enumerate(pages):
        if "EPS Estimates" not in text:
            continue
        for required in ["Smart Estimates", "EPS Estimates", "Year Ago", "Rev Estimates (mln)"]:
            if required not in text:
                raise ValueError("newsletter table schema changed")
        matches = re.findall(r"^\s*(" + COMPANY_NAMES[symbol] + r")\s+(Q[1-4])\s+(AMC|BMO|NTS|\d{2}:\d{2}|--)\s+([^\n]+)", text, re.M | re.I)
        candidates.extend((number, match) for match in matches)
    if len(candidates) != 1:
        raise ValueError("expected one company row in I/B/E/S KEY RESULTS table")
    number, (company, quarter, clock, tail) = candidates[0]
    tokens = tail.split()
    if len(tokens) != 4:
        raise ValueError("expected SmartEstimate, consensus EPS, year-ago EPS, revenue estimate")
    def numeric(token):
        if token in {"--", "N/A"}:
            return None
        if not re.fullmatch(r"-?\$\d[\d,]*(?:\.\d+)?", token):
            raise ValueError("unexpected estimate token")
        return float(token.replace("$", "").replace(",", ""))
    smart, consensus, year_ago, revenue = map(numeric, tokens)
    created, modified = reader.metadata.creation_date, reader.metadata.modification_date
    clocks_known = all(value is not None and value.tzinfo is not None for value in [created, modified])
    latest = max(pd.Timestamp(created), pd.Timestamp(modified)).tz_convert("America/New_York") if clocks_known else pd.NaT
    prior = bool(pd.notna(latest) and latest.tz_localize(None) < pd.Timestamp(report_date))
    return {"symbol": symbol, "report_date": report_date, "company_name": company, "quarter": quarter,
        "expected_report_time": clock, "smart_eps": smart, "consensus_eps": consensus,
        "prior_year_eps": year_ago, "consensus_revenue_million_usd": revenue, "source_page_zero_based": number,
        "pdf_created_at": str(created) if created else None, "pdf_modified_at": str(modified) if modified else None,
        "document_clock_before_report_day": prior, "historical_publication_clock_verified": False,
        "estimate_basis": "I/B/E/S table snapshot; ordinary consensus and SmartEstimate retained separately",
        "not_final_preannouncement_consensus": True, "raw_split_basis_not_restated": True, "newsletter_family": family}


def collect_newsletters(directory, plan):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    if len(plan["events"]) > plan["maximum_documents"]:
        raise ValueError("fixed newsletter budget exceeded")
    def acquire(event):
        day = pd.Timestamp(event["report_date"]).strftime("%m%d%Y")
        family = event.get("newsletter_family", "day_ahead")
        prefix = "The_Day_Ahead/TDAGeneric_NAM_" if family == "day_ahead" else "Morning_News_Call/MNCGeneric_US_"
        url = "https://share.refinitiv.com/assets/newsletters/" + prefix + day + ".pdf"
        name = day if family == "day_ahead" else "morning_" + day
        path = directory / (name + ".pdf"); receipt = directory / (name + ".receipt.json")
        try:
            if not (path.exists() and receipt.exists()):
                partial = path.with_suffix(".partial")
                # Scoped public, credential-free request only. Never change global TLS settings.
                command = ["curl", "--http1.1", "--compressed", "--fail", "--silent", "--show-error",
                    "--max-time", "30", url, "-o", str(partial)]
                result = subprocess.run(command, capture_output=True)
                verified = result.returncode == 0
                if result.returncode == 60 and b"certificate has expired" in result.stderr and plan["allow_expired_public_certificate_probe"]:
                    result = subprocess.run([command[0], "--insecure", *command[1:]], capture_output=True)
                if result.returncode:
                    partial.unlink(missing_ok=True)
                    raise ValueError("public source fetch failed: " + result.stderr.decode()[:180])
                partial.replace(path)
                receipt.write_text(json.dumps({"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "tls_certificate_verified": verified,
                    "transport_limitation": None if verified else "normal request failed curl60 expired certificate; explicitly scoped public credential-free retrieval"}, indent=2) + "\n")
            metadata = json.loads(receipt.read_text()); raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
                raise ValueError("frozen newsletter digest differs")
            return {**metadata, "path": path.name, **parse_newsletter(raw, event["symbol"], event["report_date"], family)}, None
        except Exception as error:
            return None, {**event, "url": url, "path": path.name, "error": str(error)}
    with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
        results = list(pool.map(acquire, plan["events"]))
    manifest = {"expected_events": plan["events"], "entries": [entry for entry, _ in results if entry],
        "failures": [failure for _, failure in results if failure]}
    (directory / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
