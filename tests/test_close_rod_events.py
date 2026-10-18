from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.data.close_rod_events import REHAB_FIELDS, audit_event_coverage


def test_event_mask_retains_zero_unknown_and_unqualified_tail():
    cash = pd.DataFrame([
        {"instrument": "SOXL", "event_id": "l", "ex_date": "2026-09-22", "original_total_decimal": "0.1"},
        {"instrument": "SOXX", "event_id": "x", "ex_date": "2026-09-23", "original_total_decimal": "0"},
    ])
    split = pd.DataFrame(columns=["instrument", "event_id", "effective_trading_date", "new_shares_per_old_share"])
    vendor = {s: pd.DataFrame(columns=REHAB_FIELDS) for s in ("SOXL", "SOXX")}
    row = dict.fromkeys(REHAB_FIELDS, np.nan)
    row.update(ex_div_date="2026-09-22", per_cash_div=0.1)
    vendor["SOXL"] = pd.DataFrame([row])
    source = {"market_as_of": "2026-09-25", "sources": {
        s: {"expected_distribution_rows": 1, "expected_ex_date_min": d, "expected_ex_date_max": d}
        for s, d in [("SOXL", "2026-09-22"), ("SOXX", "2026-09-23")]
    }}
    manifest = {"schema_version": "semis_minute_source_v1", "complete": True,
                "adjustment": "NONE", "session": "RTH", "start": "2018-10-12", "as_of": "2026-10-09",
                "rehab": [{"symbol": s, "path": s + "_rehab.parquet", "sha256": "a" * 64,
                           "rows": len(vendor[s]), "fetched_at": "2026-10-12T07:00:00Z"} for s in vendor]}
    sessions = pd.DatetimeIndex(["2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25", "2026-09-28", "2026-10-09"])
    kwargs = dict(sessions=sessions, source_contract=source, rehab_source_manifest=manifest)
    mask, audit, differences = audit_event_coverage(cash, split, vendor, **kwargs)
    assert len(mask) == len(sessions) and not mask.iloc[:2].event_source_ready.any()
    assert mask.iloc[2:].event_source_ready.all() and audit["event_tail_supported"]
    assert not audit["historical_PIT_qualified"]
    assert differences.iloc[0].status == "issuer_explicit_zero_retained_no_vendor_cash_required"

    unknown = dict.fromkeys(REHAB_FIELDS, np.nan)
    unknown.update(ex_div_date="2026-09-28", spin_off_base=1, spin_off_ert=1)
    changed = deepcopy(vendor)
    changed["SOXX"] = pd.DataFrame([unknown])
    changed_manifest = deepcopy(manifest)
    changed_manifest["rehab"][1]["rows"] = 1
    unknown_mask, unknown_audit, _ = audit_event_coverage(
        cash, split, changed, **{**kwargs, "rehab_source_manifest": changed_manifest})
    assert len(unknown_mask) == len(mask) and not unknown_audit["event_tail_supported"]
    assert unknown_audit["qualified_through"] == "2026-09-25"
    assert unknown_mask.loc["2026-09-28":].shared_no_increment.all()
    assert unknown_mask.loc["2026-09-28", "unknown_event"]

    stale = deepcopy(manifest)
    stale["as_of"] = "2026-09-25"
    stale_mask, stale_audit, _ = audit_event_coverage(cash, split, vendor, **{**kwargs, "rehab_source_manifest": stale})
    assert not stale_audit["event_tail_supported"] and stale_mask.loc["2026-09-28":].shared_no_increment.all()
    with pytest.raises(ValueError, match="duplicate issuer"):
        audit_event_coverage(pd.concat([cash, cash.iloc[[0]]]), split, vendor, **kwargs)
