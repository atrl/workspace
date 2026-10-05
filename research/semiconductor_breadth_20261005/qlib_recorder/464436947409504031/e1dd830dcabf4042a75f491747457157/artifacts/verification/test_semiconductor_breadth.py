import numpy as np
import pandas as pd
import pytest
import json
import hashlib

from src.data.semiconductor_breadth import compute_complete_basket_features, qualify_recovered_wolf, FEATURE_COLUMNS


def fixture():
    calendar = pd.bdate_range("2020-01-01", periods=85)
    x = np.arange(len(calendar))
    close = pd.DataFrame({"A": 100 * np.exp(.001 * x + .02 * np.sin(x)),
                          "B": 50 * np.exp(.002 * x + .015 * np.cos(x)),
                          "C": 80 * np.exp(.0015 * x + .01 * np.sin(x / 2))}, index=calendar)
    holdings = pd.DataFrame({"filing_date": [str(calendar[0].date())] * 2,
                             "report_date": ["2019-12-31"] * 2,
                             "accession": ["a"] * 2, "ticker": ["A", "B"],
                             "weight_pct": [70., 30.]})
    return calendar, close, holdings


def test_filing_membership_is_lagged_one_session_and_current_members_use_full_history():
    calendar, close, holdings = fixture()
    later = pd.DataFrame({"filing_date": [str(calendar[70].date())] * 2,
                          "report_date": ["2020-03-31"] * 2,
                          "accession": ["b"] * 2, "ticker": ["A", "C"],
                          "weight_pct": [40., 60.]})
    both = pd.concat([holdings, later], ignore_index=True)
    features, eligibility = compute_complete_basket_features(close, both, calendar)
    original, _ = compute_complete_basket_features(close, holdings, calendar)
    assert features.index.min() == calendar[60]
    pd.testing.assert_series_equal(features.loc[calendar[70]], original.loc[calendar[70]])
    assert eligibility.loc[calendar[71], "filing_date"] == str(calendar[70].date())
    changed = close.copy()
    changed.loc[calendar[20], "C"] = np.nan
    invalid, audit = compute_complete_basket_features(changed, both, calendar)
    assert calendar[71] not in invalid.index  # C existed before entry, but has incomplete trailing history.
    assert audit.loc[calendar[71], "status"] == "incomplete_61close_history"


def test_no_member_drop_price_fill_or_pairwise_correlation_omission():
    calendar, close, holdings = fixture()
    missing, audit = compute_complete_basket_features(close.drop(columns="B"), holdings, calendar)
    assert missing.empty
    assert audit.loc[calendar[70], "missing_members"] == "B"
    close.loc[calendar[10], "B"] = np.nan
    features, audit = compute_complete_basket_features(close, holdings, calendar)
    assert calendar[70] not in features.index
    assert audit.loc[calendar[70], "status"] == "incomplete_61close_history"
    assert calendar[71] in features.index


def test_fixed_feature_formula_and_future_invariance():
    calendar, close, holdings = fixture()
    features, audit = compute_complete_basket_features(close, holdings, calendar)
    day = calendar[65]
    block = close.loc[calendar[5]:day, ["A", "B"]]
    returns = block.to_numpy()[1:] / block.to_numpy()[:-1] - 1
    last = returns[-1]
    expected = [(last > 0).mean(), last.mean() - last @ np.array([.7, .3]),
                last.std(ddof=0), abs(np.corrcoef(returns, rowvar=False)[0, 1])]
    np.testing.assert_allclose(features.loc[day, list(FEATURE_COLUMNS)], expected)
    mutated = close.copy()
    mutated.loc[calendar[66]:] *= 12
    after, _ = compute_complete_basket_features(mutated, holdings, calendar)
    pd.testing.assert_frame_equal(features.loc[:day], after.loc[:day])
    assert audit.loc[day, "expected_members"] == 2


def test_unordered_calendar_and_duplicate_holdings_fail_closed():
    calendar, close, holdings = fixture()
    with pytest.raises(ValueError, match="ordered"):
        compute_complete_basket_features(close, holdings, calendar[::-1])
    with pytest.raises(ValueError, match="duplicate"):
        compute_complete_basket_features(close, pd.concat([holdings, holdings]), calendar)


def test_recovery_hash_and_cancelled_security_boundary_fail_before_store_access(tmp_path):
    raw = tmp_path / "observed.csv"
    raw.write_text("original source")
    manifest = {"hashes": {raw.name: hashlib.sha256(raw.read_bytes()).hexdigest()},
                "autype": "NONE", "code": "US.WOLF", "requested_start": "2019-06-03",
                "requested_end": "2025-09-29"}
    (tmp_path / "recovery_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="2025 cancellation"):
        qualify_recovered_wolf(tmp_path, tmp_path / "not_a_provider")
    raw.write_text("tampered source")
    with pytest.raises(ValueError, match="hash changed"):
        qualify_recovered_wolf(tmp_path, tmp_path / "not_a_provider")
