"""Focused synthetic contracts for the generic horizon + clocks + audit (no fit, no replay)."""
import numpy as np
import pandas as pd
import pytest

from src.data import generic_horizon as G
from src.data.convex_allocation import HORIZON_SESSIONS as LEGACY_HORIZON
from src.analysis import h5_source_compatibility_audit as AU


def _calendar(n=60):
    return pd.bdate_range("2020-01-01", periods=n)


def _prices(calendar):
    steps = np.arange(len(calendar), dtype=float)
    return pd.DataFrame({"open_SOXL": 100.0 + steps, "open_SOXX": 200.0 + 0.5 * steps}, index=calendar)


def _frame(calendar, signals):
    return pd.DataFrame(index=pd.DatetimeIndex(calendar)[signals])


def test_label_t_plus_21_from_explicit_calendar():
    cal = _calendar()
    prices = _prices(cal)
    frame = _frame(cal, list(range(0, 10)))
    labels, audit = G.build_horizon_labels(frame, prices, horizon_sessions=20,
                                           availability_hour=9, availability_minute=31)
    for i, sig in enumerate(range(0, 10)):
        entry, exit_day = cal[sig + 1], cal[sig + 21]
        expected = prices.loc[exit_day, "open_SOXL"] / prices.loc[entry, "open_SOXL"] - 1
        assert labels["soxl_return_h20"].iloc[i] == pytest.approx(expected)
        assert pd.Timestamp(labels["exit_date_h20"].iloc[i]) == exit_day
    assert audit["mean_unit"] == "twenty_session_simple_return"
    assert audit["formula"] == "open[t+21]/open[t+1]-1"


def test_terminal_unknown_fields_and_exact_tail_count():
    cal = _calendar(30)
    prices = _prices(cal)
    frame = _frame(cal, list(range(0, 30)))
    labels, audit = G.build_horizon_labels(frame, prices, horizon_sessions=20)
    assert len(labels) == 30  # no date deletion / dropna
    for column in ("soxl_return_h20", "soxx_return_h20", "label_h20_available_at", "exit_date_h20"):
        assert column in labels.columns  # required fields, not a vacuous if/else
    unknown = labels["soxl_return_h20"].isna()
    # signal i is known iff i+21 <= 29 -> i<=8, so exactly 21 terminal unknowns
    assert int(unknown.sum()) == 21
    assert unknown.iloc[:9].sum() == 0 and unknown.iloc[9:].all()
    assert labels["label_h20_available_at"].isna().equals(unknown)
    assert audit["terminal_unknown_labels_retained"] == 21


def test_strict_horizon_rejects_bool_fractional_zero():
    for bad in (True, False, 0, -3, 20.7, np.float64(1.5), np.nan):
        with pytest.raises(ValueError):
            G.strict_positive_int(bad)
    assert G.strict_positive_int(20) == 20
    assert G.strict_positive_int(20.0) == 20
    with pytest.raises(ValueError):
        G.build_horizon_labels(_frame(_calendar(), [0]), _prices(_calendar()), horizon_sessions=True)
    with pytest.raises(ValueError):
        G.build_horizon_labels(_frame(_calendar(), [0]), _prices(_calendar()), horizon_sessions=20.7)


def test_signal_index_validation():
    cal = _calendar()
    prices = _prices(cal)
    # NaT
    with pytest.raises(ValueError, match="NaT"):
        G.build_horizon_labels(pd.DataFrame(index=pd.DatetimeIndex(["2020-01-01", None])), prices)
    # non-midnight
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-01 16:15")])
    with pytest.raises(ValueError, match="midnight"):
        G.build_horizon_labels(pd.DataFrame(index=idx), prices)
    # disallowed timezone
    idx = pd.DatetimeIndex(["2020-01-01"]).tz_localize("Europe/London")
    with pytest.raises(ValueError, match="timezone"):
        G.build_horizon_labels(pd.DataFrame(index=idx), prices)
    # unsorted / duplicate
    with pytest.raises(ValueError, match="unique and increasing"):
        G.build_horizon_labels(pd.DataFrame(index=pd.DatetimeIndex([cal[2], cal[1]])), prices)


def test_price_calendar_missing_or_extra_day_blocked():
    cal = _calendar()
    prices = _prices(cal)
    dropped = prices.drop(cal[5])  # skipping a real day must NOT silently compress t+21
    with pytest.raises(ValueError, match="1:1 with the actual"):
        G.build_horizon_labels(_frame(cal, [0, 1]), dropped, calendar=cal)
    extra = pd.concat([prices, prices.iloc[[-1]].set_index(pd.DatetimeIndex([cal[-1] + pd.Timedelta(days=1)]))])
    with pytest.raises(ValueError, match="1:1 with the actual"):
        G.build_horizon_labels(_frame(cal, [0, 1]), extra, calendar=cal)


def test_from_qlib_uses_actual_calendar_and_blocks_missing(monkeypatch):
    cal = _calendar(30)
    raw = pd.DataFrame({"SOXL": cal.dayofyear.astype(float), "SOXX": cal.dayofyear.astype(float) + 1.0}, index=cal)
    monkeypatch.setattr(G, "actual_calendar_prices", lambda *a, **k: (cal, raw))
    frame = _frame(cal, list(range(0, 5)))
    labels, audit = G.build_horizon_labels_from_qlib(frame, provider_uri="x", start_time="a", end_time="b", horizon_sessions=20)
    assert audit["calendar_source"] == "actual D.calendar"
    assert labels["soxl_return_h20"].iloc[0] == pytest.approx(raw["SOXL"].iloc[21] / raw["SOXL"].iloc[1] - 1)
    monkeypatch.setattr(G, "actual_calendar_prices", lambda *a, **k: (cal, raw.drop(cal[3])))
    with pytest.raises(ValueError, match="1:1 with the actual"):
        G.build_horizon_labels_from_qlib(frame, provider_uri="x", start_time="a", end_time="b", horizon_sessions=20)


def test_mature_mask_preserves_index_and_requires_timezone():
    available = pd.Series(["2026-09-24 13:31:00+00:00", None], index=["rowA", "rowB"])
    mask = G.mature_mask(available, "2026-09-25 21:00:00+00:00")
    assert list(mask.index) == ["rowA", "rowB"]           # original index preserved
    assert bool(mask.loc["rowA"]) is True and bool(mask.loc["rowB"]) is False
    with pytest.raises(ValueError, match="naive cutoff requires an explicit timezone"):
        G.mature_mask(available, "2026-09-25 21:00:00")
    aware = G.mature_mask(available, "2026-09-25 21:00:00", naive_timezone="America/New_York")
    assert bool(aware.loc["rowA"]) is True


def test_h5_default_regression_compatibility():
    assert G.DEFAULT_HORIZON_SESSIONS == LEGACY_HORIZON == 5
    cal = _calendar()
    prices = _prices(cal)
    frame = _frame(cal, list(range(0, 10)))
    labels, audit = G.build_horizon_labels(frame, prices)  # default 5
    legacy = (prices["open_SOXL"].shift(-6) / prices["open_SOXL"].shift(-1) - 1).reindex(frame.index)
    assert np.allclose(labels["soxl_return_h5"].to_numpy(), legacy.to_numpy())
    assert audit["horizon_sessions"] == 5


def test_mu_is_not_squared():
    assert G.mean_unit(20) == "twenty_session_simple_return"
    assert G.mean_variance_unit(20) == "twenty_session_simple_return_squared"
    assert G.DAILY_RISK_UNIT == "one_session_simple_return_squared"
    assert "squared" not in G.mean_unit(20)
    assert G.mean_unit(5) == "five_session_simple_return"


def test_nextopen_pretrade_cannot_be_relabeled_and_nat_rejected():
    signals = _calendar(4)
    trades = signals + pd.Timedelta(days=1)
    prediction = G.predictor_clock(signals)
    allocation = G.allocation_clock(trades)
    assert bool((allocation > prediction).all())
    with pytest.raises(ValueError, match="cannot be claimed known at prior signal"):
        G.assert_pretrade_not_relabeled(allocation, prediction)
    G.assert_pretrade_not_relabeled(allocation, allocation)
    with pytest.raises(ValueError, match="NaT"):
        G.predictor_clock(pd.DatetimeIndex(["2020-01-01", None]))


def test_clocks_are_separated():
    day = pd.Timestamp("2020-01-02")
    pred = G.predictor_clock([day])[0]
    alloc = G.allocation_clock([day + pd.Timedelta(days=1)])[0]
    assert pred.tz_convert(G.NY).strftime("%H:%M") == "16:15"
    assert alloc.tz_convert(G.NY).strftime("%H:%M") == "09:31"
    assert alloc > pred


def test_dataset_horizon_scope_guard():
    class _DS:
        pass
    ds = _DS()
    G.attach_horizon_scope(ds, 20)
    G.require_horizon_scope(ds, 20)
    with pytest.raises(ValueError, match="does not match the required horizon"):
        G.require_horizon_scope(ds, 5)
    with pytest.raises(ValueError):
        G.attach_horizon_scope(_DS(), 20.7)


def test_h5_source_compatibility_audit_scoped_not_whole_causal():
    bad = AU.audit_h5_source_compatibility(held_is_current_opening_state=True)
    assert bad["clock_contradiction_detected"] is True
    assert bad["legacy_clock_claim_verdict"] == AU.CLAIM_INVALID
    assert bad["whole_model_causal_certified"] is False
    assert bad["actual_future_dependency"] == "unknown"       # no asserted lookahead advantage
    assert bad["financial_target_impact"] == "unknown"
    assert bad["check_scope"] == "single_clock_contradiction_only"
    assert bad["old_promotion_status_modified"] is False and bad["parent_modified"] is False
    good = AU.audit_h5_source_compatibility(held_is_current_opening_state=False)
    assert good["legacy_clock_claim_verdict"] == AU.NO_CONTRADICTION
    assert good["whole_model_causal_certified"] is False       # NOT a whole-model causal pass
    from_frames = AU.audit_legacy_clock_from_frames(
        signal_dates=_calendar(3), trade_dates=_calendar(3) + pd.Timedelta(days=1))
    assert from_frames["allocation_after_prediction"] is True
    assert from_frames["legacy_clock_claim_verdict"] == AU.CLAIM_INVALID
    with pytest.raises(ValueError, match="NaT"):
        AU.audit_legacy_clock_from_frames(signal_dates=pd.DatetimeIndex(["2020-01-01", None]),
                                          trade_dates=_calendar(2))


def test_clock_session_date_contract_dst_and_zones():
    pred = G.predictor_clock(["2020-03-06", "2020-03-09", "2020-11-02"])  # around DST + standard time
    ny = pred.tz_convert(G.NY)
    assert [d.strftime("%Y-%m-%d %H:%M") for d in ny] == ["2020-03-06 16:15", "2020-03-09 16:15", "2020-11-02 16:15"]
    aware = pd.DatetimeIndex(["2020-03-09", "2020-11-02"]).tz_localize(G.NY)
    out = G.predictor_clock(aware)
    assert [d.strftime("%Y-%m-%d") for d in out.tz_convert(G.NY)] == ["2020-03-09", "2020-11-02"]
    with pytest.raises(ValueError, match="America/New_York"):
        G.predictor_clock(pd.DatetimeIndex(["2020-03-09"]).tz_localize("UTC"))  # UTC would shift the date
    with pytest.raises(ValueError, match="midnight"):
        G.predictor_clock(pd.DatetimeIndex([pd.Timestamp("2020-12-25 09:30")]))


def test_full_calendar_labels_retain_terminal_tail_mock():
    cal = _calendar(40)
    prices = _prices(cal)
    frame = pd.DataFrame(index=cal)  # every session is a signal -> real terminal tail
    for horizon, expected in ((5, 6), (20, 21)):
        labels, audit = G.build_horizon_labels(frame, prices, horizon_sessions=horizon)
        col = f"soxl_return_h{horizon}"
        assert len(labels) == len(cal) and int(labels[col].isna().sum()) == expected
        # label exit NaT is allowed and the availability NaT must be preserved
        assert labels[f"label_h{horizon}_available_at"].isna().equals(labels[col].isna())
        assert audit["terminal_unknown_labels_retained"] == expected


def test_decision_clocks_reject_nat_but_label_clock_allows():
    cal = _calendar(30)
    with pytest.raises(ValueError, match="NaT"):
        G.predictor_clock(pd.DatetimeIndex([cal[0], None]))       # valid decision clock forbids NaT
    with pytest.raises(ValueError, match="NaT"):
        G.allocation_clock(pd.DatetimeIndex([cal[0], None]))
    prices = _prices(cal)
    labels, _ = G.build_horizon_labels(pd.DataFrame(index=cal), prices, horizon_sessions=20)
    assert labels["label_h20_available_at"].isna().sum() == 21   # terminal label NaT is allowed


def test_full_calendar_labels_retain_terminal_tail_real_provider():
    from pathlib import Path
    provider = Path(__file__).resolve().parents[1] / "data/qlib/published_futu_affine_20260925_v1"
    if not provider.exists():
        pytest.skip("published provider not present")
    cal, _ = G.actual_calendar_prices(provider, start_time="2016-01-04", end_time="2026-09-25",
                                      symbols=("SOXL", "SOXX"))
    frame = pd.DataFrame(index=cal)
    for horizon, expected in ((5, 6), (20, 21)):
        labels, audit = G.build_horizon_labels_from_qlib(frame, provider_uri=provider, start_time="2016-01-04",
                                                         end_time="2026-09-25", symbols=("SOXL", "SOXX"),
                                                         horizon_sessions=horizon)
        col = f"soxl_return_h{horizon}"
        assert len(labels) == 2698 and int(labels[col].isna().sum()) == expected
        assert labels[f"label_h{horizon}_available_at"].isna().equals(labels[col].isna())
        assert audit["terminal_unknown_labels_retained"] == expected
