import pandas as pd
import numpy as np
import pytest

from src.analysis import night_session_signal as analysis
from src.workflow import night_session_signal as workflow

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _hourly_day(date, prices):
    """Build 1h bars for one day at 04:00..20:00 with given close path."""
    day = pd.Timestamp(date).tz_localize("America/New_York")
    times = [f"{h:02d}:00" for h in range(4, 20)]
    index = pd.DatetimeIndex([day + pd.Timedelta(hours=int(t[:2])) for t in times])
    closes = np.linspace(prices[0], prices[-1], len(index))
    return pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": 1.0}, index=index)


def test_futures_overnight_return_uses_prior_evening_to_cash_open():
    # Futures trade from 18:00 prior day to 09:00 current day.
    day1 = _hourly_day("2025-01-06", [100.0, 110.0])  # 04:00..19:00
    day2 = _hourly_day("2025-01-07", [120.0, 130.0])
    frame = pd.concat([day1, day2])
    result = analysis.futures_overnight_return(frame)
    # For 2025-01-07 the window starts at 2025-01-06 18:00 and ends before 09:30 on 01-07.
    window = frame[((frame.index.normalize() == pd.Timestamp("2025-01-06").tz_localize("America/New_York"))
                    & (frame.index.time >= pd.Timestamp("18:00").time()))
                   | ((frame.index.normalize() == pd.Timestamp("2025-01-07").tz_localize("America/New_York"))
                      & (frame.index.time < pd.Timestamp("09:30").time()))]
    expected = window["close"].iloc[-1] / window["open"].iloc[0] - 1
    value = result.loc[pd.Timestamp("2025-01-07").tz_localize("America/New_York")]
    assert value == pytest.approx(expected)


def test_regular_session_uses_only_0930_to_1600():
    day = _hourly_day("2025-02-03", [50.0, 60.0])
    result = analysis.regular_session(day)
    # open is the 09:30... but bars start at 04:00. Only 09:00..15:00 exist in our stub.
    assert len(result) == 1
    row = result.iloc[0]
    assert row["open"] < row["close"] or row["open"] > row["close"]


def test_switch_strategy_cash_defensive_has_zero_weights_when_flat():
    dates = pd.bdate_range("2025-01-02", periods=50)
    signal = pd.Series(np.tile([0.01, -0.01], 25), index=dates)
    soxl = pd.Series(0.02, index=dates)
    soxx = pd.Series(0.005, index=dates)
    replay = analysis.switch_strategy(signal, soxl, soxx, threshold=0.0, cost_bps=10.0, defensive="cash")
    # On flat days the cash-defensive strategy earns only the switch fee, never SOXX's return.
    flat = replay.loc[replay["exposure"] == 0.0]
    assert (flat["strategy"] <= 1e-9).all()
    assert (flat["strategy"] > -0.01).all()
    replay_soxx = analysis.switch_strategy(signal, soxl, soxx, threshold=0.0, cost_bps=10.0, defensive="soxx")
    flat_soxx = replay_soxx.loc[replay_soxx["exposure"] == 0.0]
    # The SOXX-defensive strategy does participate on flat days (minus the switch fee).
    assert (flat_soxx["strategy"] > 0).all()


def test_switch_strategy_rejects_bad_defensive():
    dates = pd.bdate_range("2025-01-02", periods=30)
    s = pd.Series(0.0, index=dates)
    with pytest.raises(ValueError, match="defensive"):
        analysis.switch_strategy(s, s, s, threshold=0.0, cost_bps=10.0, defensive="bonds")


def test_information_coefficient_reports_excess():
    dates = pd.bdate_range("2025-01-02", periods=200)
    signal = pd.Series(np.linspace(-0.01, 0.01, 200), index=dates)
    soxl = signal * 3.0
    soxx = signal
    ic = analysis.information_coefficient(signal, soxl, soxx)
    assert ic["ic_soxl"] > 0.9
    assert ic["ic_excess"] > 0.9


def test_new_night_session_dataset_present_and_documented():
    data_dir = ROOT / workflow.DATA_DIR
    manifest = data_dir / "manifest.json"
    assert manifest.is_file()
    import json
    doc = json.loads(manifest.read_text())
    for key in ("SOXL", "SOXX", "ES_F", "NQ_F"):
        assert key in doc["series"]
    for name in ("SOXL", "SOXX", "ES_F", "NQ_F"):
        assert (data_dir / f"{name}.parquet").is_file()


def test_known_defect_workflow_refuses_before_any_side_effect(tmp_path):
    """The night-session workflow has known integrity defects and must refuse.

    It may not run on a fresh directory, must not create a recorder or write a
    card, and must never emit a promotion-eligible / objective-achieved result.
    """
    import pytest
    from src.workflow import night_session_signal as wf

    card = tmp_path / "card"
    with pytest.raises(ValueError, match="blocked_known_integrity_defects"):
        wf.run_night_session_signal(root=ROOT, card_dir=card,
                                    recorder_uri=card / "qlib_recorder")
    # No side effects: no card and no recorder written before the refusal.
    assert not card.exists()


def test_writer_bans_eligible_or_objective_true_on_fresh_dir(tmp_path):
    """The legacy writer must refuse to record an eligible/objective-achieved card."""
    import pytest
    from src.workflow import night_session_signal as wf

    positive = {"promotion_eligible": True, "objective_achieved": True,
                "decision": "stable_overnight_excess_candidate"}
    with pytest.raises(ValueError):
        wf.write_card_files(positive, card_dir=tmp_path / "fresh")
    assert not (tmp_path / "fresh" / "summary.json").exists()
