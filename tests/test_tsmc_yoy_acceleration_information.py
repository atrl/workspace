"""Focused contracts for the fixed TSMC YoY-acceleration 5-session information screen."""
from pathlib import Path

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import tsmc_yoy_acceleration_information as A

ROOT = Path(__file__).resolve().parents[1]


PLAN = json.loads((ROOT / A.FREEZE_MANIFEST).read_text())
REAL_PLAN = {"frozen_sources": {"source_info_provider": {"manifest_sha256": PLAN["source_provider_manifest_sha256"]},
                               "price_provider": {"manifest_sha256": PLAN["price_provider_manifest_sha256"]}}}


@pytest.fixture(scope="module")
def _built():
    return A.build_observations(ROOT, REAL_PLAN)


def test_full_calendar_and_scope_flags(_built):
    obs = _built["observations"]
    assert len(obs) >= 130
    assert int(obs["admitted"].sum()) == 130
    assert (obs["period"] == "2015-10").any() and not bool(obs[obs["period"] == "2015-10"]["admitted"].iloc[0])


def test_price_calendar_not_compressed(_built):
    cal, opens = A.load_price_calendar_and_opens(ROOT)
    assert len(opens) == len(cal)  # reindexed to full calendar; missing opens are NaN rows, not dropped


def test_exit_is_five_real_us_sessions_and_aware_maturity(_built):
    obs = _built["observations"]
    cal = _built["calendar"]
    pos = {d: i for i, d in enumerate(cal)}
    n = 0
    for _, r in obs[np.isfinite(obs["spread_5d"])].iterrows():
        assert pos[r["exit_session"]] - pos[r["entry_session"]] == A.HORIZON
        assert r["exit_session"].weekday() < 5
        stamp = pd.Timestamp(r["label_available_at"])
        assert stamp.tz is not None and str(stamp.tz) == "UTC"  # aware UTC, not naive
        expected = (r["exit_session"].tz_localize(A.NY) + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")
        assert stamp == expected
        n += 1
    assert n > 0


def test_factor_clock_before_decision_per_row(_built):
    obs = _built["observations"]
    scored = obs[np.isfinite(obs["score"])]
    assert len(scored) > 0
    for _, r in scored.iterrows():
        assert r["source_factor_clock_utc"] is not None and r["decision_at"] is not None
        assert pd.Timestamp(r["source_factor_clock_utc"]) <= pd.Timestamp(r["decision_at"])
        assert bool(r["factor_clock_ok"])


def test_unknown_retained_not_filled(_built):
    obs = _built["observations"]
    unknown = obs[~obs["stat_valid"]]
    assert len(unknown) > 0
    assert unknown[unknown["unknown_reason"].str.contains("missing_mark|no_exit_session|out_of_price_scope")]["spread_5d"].isna().all()


def test_mask_excludes_invalid_but_finite_rows():
    # a row with finite score/spread but stat_valid False must NOT enter the statistic
    section = pd.DataFrame({
        "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0],
        "spread_5d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -100.0],
        "stat_valid": [True, True, True, True, True, True, False]})
    cleaned = A.masked_spearman(section)
    full = A.masked_spearman(section.assign(stat_valid=True))
    assert cleaned is not None and cleaned > 0.99
    assert full is not None and full < cleaned  # the excluded outlier would change the rank IC


def test_bootstrap_resamples_with_mask(_built):
    obs = _built["observations"]
    section = obs[(obs["entry_session"] >= pd.Timestamp("2016-01-04")) & (obs["entry_session"] <= pd.Timestamp("2022-12-31"))].reset_index(drop=True)
    result = A.block_bootstrap(section)
    assert result["status"] == "ok" and result["valid_replicates"] > 0
    assert result["ci_low"] < result["ci_high"]
    assert result["blocks"] == len(section) - A.BLOCK + 1


def test_frozen_hash_mismatch_blocks():
    good_source = PLAN["source_provider_manifest_sha256"]
    good_price = PLAN["price_provider_manifest_sha256"]
    good_events = PLAN["parent_events_object_sha256"]
    # matching everything -> ok
    assert A.verify_frozen(ROOT, REAL_PLAN, events_sha=good_events, source_sha=good_source, price_sha=good_price)["verified"]
    # tampered parent events object -> blocked
    with pytest.raises(ValueError, match="identity changed"):
        A.verify_frozen(ROOT, REAL_PLAN, events_sha="0" * 64, source_sha=good_source, price_sha=good_price)
    # tampered source provider -> blocked
    with pytest.raises(ValueError, match="identity changed"):
        A.verify_frozen(ROOT, REAL_PLAN, events_sha=good_events, source_sha="0" * 64, price_sha=good_price)


def test_score_alignment_rejects_wrong_period():
    assert A.score_is_aligned(43.1, 43.1)
    assert not A.score_is_aligned(43.1, -11.3)   # a superseded/other-period score must not be valid
    assert not A.score_is_aligned(float("nan"), 43.1)


def test_calendar_injection_missing_midsession_keeps_real_sessions():
    cal = pd.DatetimeIndex(pd.to_datetime(["2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06",
                                           "2025-03-07", "2025-03-10", "2025-03-11"]))
    opens = pd.DataFrame({"SOXL": [10.0, 10.1, np.nan, 10.3, 10.4, 10.5, 10.6],
                          "SOXX": [20.0, 20.1, np.nan, 20.3, 20.4, 20.5, 20.6]}, index=cal)
    entry = cal[0]
    exit_day = A.select_exit(cal, entry)
    # exit is the real session at position +5, NOT shifted past the missing mid-session
    assert exit_day == cal[5]  # real session +5, NOT shifted past the missing mid-session
    # a missing MID-session does not shift the exit and the open-to-open compound only needs entry/exit
    assert np.isfinite(A.mark_returns(opens, entry, exit_day)).all()
    # a missing ENTRY or EXIT mark -> unknown, never a shifted/other price
    gaps = opens.copy(); gaps.loc[entry, "SOXL"] = np.nan
    assert all(np.isnan(A.mark_returns(gaps, entry, exit_day)))
    gaps2 = opens.copy(); gaps2.loc[exit_day, "SOXX"] = np.nan
    assert all(np.isnan(A.mark_returns(gaps2, entry, exit_day)))
    # out-of-range entry -> no exit session
    assert A.select_exit(cal, cal[6]) is None


def test_secondary_ic_computed_and_outer_not_gate(_built):
    result = A.evaluate(ROOT, REAL_PLAN)["results"]
    dev = result["per_window"]["development"]
    assert dev["secondary_soxl_ic"] is not None and dev["secondary_soxx_ic"] is not None
    assert result["primary_gate"]["scope"] == "development"
    assert result["primary_gate"]["valid_n"] == dev["valid_n"]
    assert "insufficient_evidence" in " ".join(result["caveats"])


def test_primary_is_time_series_not_cross_sectional(_built):
    valid = _built["observations"][_built["observations"]["stat_valid"]]
    assert A.masked_spearman(valid.assign(stat_valid=True)) is not None
