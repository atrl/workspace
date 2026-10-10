"""Focused contracts for the TSMC monthly-revenue source plane."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.data.tsmc_monthly_revenue_store as T

W_EXPERIMENT = "tsmc_monthly_revenue_pit_qualification_20261010"

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / T.RAW_DIR / "docs"


def _doc(accn, name):
    p = DOCS / f"{accn}_{name}"
    assert p.is_file(), f"missing cached doc {p}"
    return p.read_bytes()


# --- parser / units ---
def test_mom_is_not_yoy_and_unit_is_million():
    parsed = T.parse_monthly_revenue(_doc("0001046179-25-000026", "tsm-revenue20250310.htm"))
    assert parsed["format"] == "main_table" and parsed["unit"] == "NT$ million" and parsed["period"] == "2025-02"
    assert abs(parsed["reported_yoy"] - 43.1) < 0.01
    assert abs(parsed["reported_mom"] - (-11.3)) < 0.01
    assert T.table_numeric_gate(parsed)


def test_unit_conversion_binds_to_record_unit():
    assert T.to_ntd_million(260009, "NT$ million") == 260009
    assert T.to_ntd_million(58.35, "NT$ billion") == 58350.0
    assert T.to_ntd_million(260008796, "NT$ thousand") == pytest.approx(260008.796)
    assert T.to_ntd_million(1.0, "shares") is None


def test_gzip_and_identity_both_parse():
    import gzip
    raw = _doc("0001046179-25-000026", "tsm-revenue20250310.htm")
    assert T.parse_monthly_revenue(raw)["reported_yoy"] == 43.1
    assert T.parse_monthly_revenue(gzip.compress(raw))["reported_yoy"] == 43.1


def test_old_paragraph_and_new_table_formats():
    old = T.parse_monthly_revenue(_doc("0001193125-16-499090", "d134551d6k.htm"))
    assert old["format"] == "paragraph" and old["period"] == "2016-02"
    assert abs(old["reported_yoy"] - (-4.9)) < 0.01 and abs(old["reported_mom"] - (-16.0)) < 0.01
    new = T.parse_monthly_revenue(_doc("0001046179-24-000014", "tsm-revenue20240207x6k.htm"))
    assert new["format"] == "main_table" and new["period"] == "2024-01"


NON_MONTHLY_FIXTURES = {
    "quarterly": b"<html>TSMC today announced consolidated revenue of NT$221.81 billion and diluted EPS for the second quarter ended June 30, 2016.</html>",
    "board": b"<html>TSMC board of directors resolved the ex-dividend date and cash dividend distribution.</html>",
    "earthquake": b"<html>TSMC announces the impact of the earthquake on operations and wafer shipments.</html>",
}


@pytest.mark.parametrize("kind", list(NON_MONTHLY_FIXTURES))
def test_non_monthly_fixtures_are_rejected(kind):
    assert T.parse_monthly_revenue(NON_MONTHLY_FIXTURES[kind]) is None


# --- clock ---
def test_next_us_open_is_strictly_next_session():
    cal = pd.DatetimeIndex(pd.to_datetime(["2025-01-10", "2025-01-13", "2025-01-14"]))
    assert T.next_us_open(pd.Timestamp("2025-01-10T14:00:00Z"), cal).strftime("%Y-%m-%d") == "2025-01-13"
    assert T.next_us_open(pd.Timestamp("2025-01-13T15:00:00Z"), cal).strftime("%Y-%m-%d") == "2025-01-14"
    assert T.next_us_open(pd.Timestamp("2025-01-14T15:00:00Z"), cal) is None
    with pytest.raises(ValueError):
        T.next_us_open(pd.Timestamp("2025-01-10"), cal)


def test_holiday_next_session_is_not_sunday():
    cal, _ = T.us_session_calendar(ROOT)
    # Memorial Day Mon 2025-05-26 is a holiday: Friday after-open -> Tuesday, never a weekend.
    nxt = T.next_us_open(pd.Timestamp("2025-05-23T20:00:00Z"), cal)
    assert nxt.strftime("%Y-%m-%d") == "2025-05-27"
    assert nxt.dayofweek < 5
    # July 4 2025 (Friday holiday) -> Monday 2025-07-07.
    nxt2 = T.next_us_open(pd.Timestamp("2025-07-03T20:00:00Z"), cal)
    assert nxt2.strftime("%Y-%m-%d") == "2025-07-07" and nxt2.dayofweek == 0


# --- availability / late prior ---
def _paragraph_record(accn, period, accept, yoy, mom=1.0):
    return {"accessionNumber": accn, "filingDate": accept[:10], "acceptanceDateTime": accept,
            "url": f"u/{accn}", "raw_sha256": accn, "period": period,
            "parsed": {"format": "paragraph", "unit": "NT$ billion", "reported_mom": mom,
                       "reported_yoy": yoy, "current_approx": 100.0, "scale": "billion"}}


def test_late_prior_does_not_regress_and_factor_waits():
    cal = pd.DatetimeIndex(pd.to_datetime([f"2025-03-{d:02d}" for d in range(10, 21)]))
    cal = cal[cal.dayofweek < 5]
    records = [_paragraph_record("FEB", "2025-02", "2025-03-10T15:39:23Z", 43.1, -11.3),
               _paragraph_record("JAN", "2025-01", "2025-03-18T15:00:00Z", 35.9, 5.4)]
    frame = T.build_events(records, cal)
    state = T.reference_asof_state(frame, cal).set_index("datetime")
    before = pd.Timestamp("2025-03-12")
    after = pd.Timestamp("2025-03-19")
    # current value (Feb) never regresses to the late old Jan
    assert state.loc[before, "reported_yoy_pct"] == pytest.approx(43.1)
    assert state.loc[after, "reported_yoy_pct"] == pytest.approx(43.1)
    # factor unknown until the prior report is available, then delta
    assert np.isnan(state.loc[before, "acceleration_pct_points"])
    assert state.loc[after, "acceleration_pct_points"] == pytest.approx(43.1 - 35.9)


def test_first_version_kept_and_revisions_separate():
    cal = pd.DatetimeIndex(pd.to_datetime(["2025-01-10", "2025-01-11", "2025-01-13"]))
    base = _paragraph_record("A1", "2024-11", "2024-12-10T13:00:00Z", 2.0)
    later = _paragraph_record("A2", "2024-11", "2025-01-11T13:00:00Z", 9.9)
    frame = T.build_events([base, later], cal)
    assert len(frame) == 1 and frame.iloc[0]["accn"] == "A1" and frame.iloc[0]["revisions"] == 1


def test_future_revision_is_rejected_before_selection():
    good = _paragraph_record("A1", "2026-07", "2026-08-10T13:00:00Z", 1.0)
    future = _paragraph_record("A2", "2026-07", "2026-12-01T13:00:00Z", 2.0)  # beyond source_as_of
    with pytest.raises(ValueError, match="beyond"):
        T.assert_source_asof_records([good, future])


# --- coverage / scope ---
def test_coverage_expected_130_and_extra_not_admitted():
    frame = pd.read_parquet(ROOT / T.RAW_DIR / "events.parquet")
    cov = T.coverage(frame)
    assert cov["expected"] == 130 and cov["present"] == 130 and cov["missing"] == []
    assert "2015-10" in cov["not_admitted_to_scope"]


def test_admitted_to_scope_flag_marks_130():
    frame = pd.read_parquet(ROOT / T.RAW_DIR / "events.parquet")
    assert int(frame["admitted_to_scope"].sum()) == 130
    assert not bool(frame[frame["period"] == "2015-10"]["admitted_to_scope"].iloc[0])


# --- workflow gate (monkeypatched, no real build) ---
def test_workflow_blocks_on_dirty_source(monkeypatch, tmp_path):
    from src.workflow import tsmc_monthly_revenue_qualification as W
    frame = pd.read_parquet(ROOT / T.RAW_DIR / "events.parquet")
    trimmed = frame[frame["period"] != "2020-06"].reset_index(drop=True)
    called = {"build": False}
    monkeypatch.setattr(T, "build_events_from_raw", lambda root, cal: (trimmed, [], []))
    monkeypatch.setattr(T, "build_qlib_store", lambda *a, **k: called.__setitem__("build", True))
    summary = W.run_tsmc_monthly_revenue_qualification(root=ROOT, card_dir=tmp_path)
    assert summary["source_gate"] == "blocked" and summary["decision"] == "needs_data"
    assert called["build"] is False  # store never built on a dirty source


def test_docs_manifest_sha_preflight():
    import json
    from src.workflow import tsmc_monthly_revenue_qualification as W
    man = json.loads((ROOT / T.RAW_DIR / "docs_manifest.json").read_text())
    assert len(man) >= 130
    result = W._verify_docs_manifest(ROOT, man)
    assert result["ok"] and result["checked"] == len(man)


def test_store_readback_has_130_events():
    rb = T.readback(ROOT)
    assert int((rb["$event_available"] == 1).sum()) == 130
    assert rb["datetime"].max() <= pd.Timestamp(T.USABLE_PRICE_CUTOFF)
    first = rb[rb["$event_available"] == 1].iloc[0]
    assert np.isnan(first["$acceleration_pct_points"])


ORACLE_12 = {
    "2015-12": (-16.1, -8.0), "2016-01": (-18.7, 21.4), "2016-02": (-4.9, -16.0),
    "2019-12": (15.0, -4.2), "2020-01": (32.8, 0.4), "2020-02": (53.4, -9.9),
    "2023-12": (-8.4, -14.4), "2024-01": (7.9, 22.4), "2024-02": (11.3, -15.8),
    "2024-12": (57.8, 0.8), "2025-01": (35.9, 5.4), "2025-02": (43.1, -11.3),
}


@pytest.mark.parametrize("period,yoy,mom", [(p, *v) for p, v in ORACLE_12.items()])
def test_parser_oracle_12_from_raw_docs(period, yoy, mom):
    import json
    manifest = {m["period"]: m for m in json.loads((ROOT / T.RAW_DIR / "docs_manifest.json").read_text())}
    assert period in manifest, f"period {period} not in docs manifest"
    entry = manifest[period]
    path = DOCS / f"{entry['accn']}_{Path(entry['url']).name}"
    parsed = T.parse_monthly_revenue(path.read_bytes())
    assert parsed["period"] == period
    assert abs(parsed["reported_yoy"] - yoy) < 0.05
    assert abs(parsed["reported_mom"] - mom) < 0.05


def test_verify_parser_oracle_on_rebuilt_events():
    frame = pd.read_parquet(ROOT / T.RAW_DIR / "events.parquet")
    result = T.verify_parser_oracle(frame)
    assert result["all_pass"] and result["passed"] == 12


def test_workflow_records_parser_oracle_check():
    import json
    summary = json.loads((ROOT / "research" / W_EXPERIMENT / "summary.json").read_text())
    assert summary["checks"]["parser_oracle"]["all_pass"]
    assert "parser_oracle_12" in summary["causal_contracts"]
    assert "source_as_of=2026-10-10" in summary["source_as_of_note"]
