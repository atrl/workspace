"""Validate source-grounded, assistant-reviewed prospective revenue comparisons."""
import hashlib
import re

import numpy as np
import pandas as pd


def validate_guidance_reviews(manifest, reviewed, issuer_guidance):
    entries = manifest["entries"]
    if len(reviewed) != len(entries) or [row["source_index"] for row in reviewed] != list(range(len(entries))):
        raise ValueError("review must cover every downloaded document, including unknowns")
    nvda = issuer_guidance.loc[issuer_guidance.next_quarter_numeric].copy()
    nvda["report_key"] = nvda.release_date.dt.strftime("%Y-%m-%d")
    nvda = nvda.set_index("report_key")
    if not nvda.index.is_unique:
        raise ValueError("issuer guidance date ambiguous")
    output = []
    for review, entry in zip(reviewed, entries):
        text = "\n".join(entry["paragraphs"])
        if hashlib.sha256(text.encode()).hexdigest() != review["paragraph_sha256"]:
            raise ValueError("reviewed source paragraph changed")
        money = [float(number.replace(",", "")) * (1 if unit.lower() == "billion" else .001)
            for number, unit in re.findall(r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million)", text, re.I)]
        def grounded(value):
            return any(abs(value - candidate) < 1e-8 for candidate in money)
        guide, estimate = review["guidance_billion"], review["consensus_billion"]
        bounds = review.get("guidance_range_billion")
        if guide is not None:
            if bounds:
                if len(bounds) != 2 or not all(grounded(value) for value in bounds) or abs(sum(bounds) / 2 - guide) > 1e-8:
                    raise ValueError("guidance midpoint not supported by explicit source range")
            elif not grounded(guide):
                raise ValueError(f"guide amount absent from source: {entry['symbol']} {entry['report_date']}")
        if estimate is not None and not grounded(estimate):
            raise ValueError("consensus amount absent from source")
        numeric = guide is not None and estimate is not None
        if numeric and (estimate <= 0 or np.sign(guide - estimate) != review["direction"]):
            raise ValueError("numeric midpoint comparison and reviewed direction disagree")
        anchor = False
        issuer_mid = None
        if entry["symbol"] == "NVDA" and (guide is not None or review["direction"] is not None):
            issuer = nvda.loc[entry["report_date"]]
            if issuer.target_quarter != entry["target_fiscal_quarter"]:
                raise ValueError("review target differs from original issuer target")
            if issuer.guidance_unit not in {"million", "USD_million"}:
                raise ValueError("unsupported issuer guidance unit")
            issuer_mid = float(issuer.guidance_mid) / 1000
            if guide is not None and abs(issuer_mid - guide) > 1e-8:
                raise ValueError("reported guidance differs from frozen issuer midpoint")
            anchor = True
        known = review["direction"] is not None
        output.append({"symbol": entry["symbol"], "report_date": entry["report_date"],
            "newsletter_date": entry["newsletter_date"], "target_fiscal_quarter": entry["target_fiscal_quarter"],
            "source_pdf_sha256": entry["sha256"], "source_url": entry["url"], **review,
            "numeric_pair_present": numeric, "comparison_direction_known": known,
            "reported_midpoint_gap": guide / estimate - 1 if numeric else np.nan,
            "issuer_target_anchor_checked": anchor, "issuer_midpoint_billion": issuer_mid,
            "status": "scope_caveat" if review["scope_caveat"] else "numeric_pair" if numeric else "direction_only" if known else "unknown",
            "normalized_economic_scope_certified": False, "first_public_timestamp_certified": False})
    for missing in manifest["failures"]:
        output.append({"symbol": missing["symbol"], "report_date": missing["report_date"],
            "newsletter_date": missing["newsletter_date"], "target_fiscal_quarter": missing["target_fiscal_quarter"],
            "source_url": missing["url"], "status": "source_missing", "direction": None,
            "numeric_pair_present": False, "comparison_direction_known": False,
            "normalized_economic_scope_certified": False, "first_public_timestamp_certified": False})
    result = pd.DataFrame(output).sort_values(["report_date", "symbol"]).reset_index(drop=True)
    if result.duplicated(["symbol", "report_date"]).any():
        raise ValueError("duplicate reviewed event")
    return result
