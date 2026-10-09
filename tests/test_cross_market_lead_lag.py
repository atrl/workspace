import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import cross_market_lead_lag as analysis
from src.workflow import cross_market_lead_lag as workflow

ROOT = Path(__file__).resolve().parents[1]


def _series(periods=800, seed=4):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=periods)
    return pd.Series(rng.normal(0.0003, 0.01, periods), index=dates)


def test_information_coefficient_lag_is_strict():
    dates = pd.bdate_range("2015-01-02", periods=400)
    rng = np.random.default_rng(7)
    predictor = pd.Series(rng.normal(0, 1, 400), index=dates)
    target = predictor.copy()
    same = analysis.information_coefficient(predictor, target, lag=0)
    lagged = analysis.information_coefficient(predictor, target, lag=1)
    assert same["pearson_ic"] == pytest.approx(1.0)
    # A strictly lagged alignment of an i.i.d. series cannot be near-perfect.
    assert abs(lagged["pearson_ic"]) < 0.3


def test_information_coefficient_rejects_short_overlap():
    dates = pd.bdate_range("2015-01-02", periods=30)
    a = pd.Series(np.arange(30.0), index=dates)
    with pytest.raises(ValueError, match="not enough"):
        analysis.information_coefficient(a, a, lag=0)


def test_switch_strategy_uses_lagged_signal_and_charges_turnover():
    dates = pd.bdate_range("2015-01-02", periods=300)
    signal = pd.Series(np.tile([1.0, -1.0], 150), index=dates)
    soxl = pd.Series(0.01, index=dates)
    soxx = pd.Series(0.0, index=dates)
    replay = analysis.switch_strategy(signal, soxl, soxx, lag=1, cost_bps=10.0)
    assert set(replay) == {"strategy", "benchmark", "exposure", "turnover", "sessions"}
    # Exposure at t equals signal>0 at t-1.
    assert replay["exposure"].iloc[1] == pytest.approx(float(signal.iloc[0] > 0))
    assert (replay["turnover"] >= 0).all()
    assert (replay["strategy"] <= soxl.loc[replay["strategy"].index] + 1e-12).all()


def test_evaluate_strategy_flags_only_full_dominance():
    dates = pd.bdate_range("2015-01-02", periods=800)
    rng = np.random.default_rng(9)
    noise = pd.Series(rng.normal(0, 0.002, len(dates)), index=dates)
    bench = 0.0004 + noise
    win = bench + 0.0003
    replay = {"strategy": win, "benchmark": bench,
              "exposure": pd.Series(1.0, index=dates),
              "turnover": pd.Series(0.0, index=dates), "sessions": len(dates)}
    result = analysis.evaluate_strategy(replay, minimum_positive_years=2)
    assert result["stable_excess"] is True
    lose = {"strategy": bench * 0.5, "benchmark": bench,
            "exposure": pd.Series(0.0, index=dates),
            "turnover": pd.Series(0.0, index=dates), "sessions": len(dates)}
    assert analysis.evaluate_strategy(lose, minimum_positive_years=2)["stable_excess"] is False


def test_manifest_loader_flattens_series():
    manifest = {"series": {"twii": {"symbol": "^TWII", "timezone": "Asia/Taipei",
                                    "rows": 10, "start": "2020-01-01", "end": "2020-01-20"}}}
    frame = analysis.load_manifest(manifest)
    assert list(frame.columns) == ["series", "symbol", "timezone", "rows", "start", "end"]
    assert frame.iloc[0]["symbol"] == "^TWII"


def test_new_external_dataset_and_manifest_exist():
    data = ROOT / workflow.DATA_DIR / "daily_closes_localdate.csv"
    manifest = ROOT / workflow.DATA_DIR / "manifest.json"
    assert data.is_file() and manifest.is_file()
    frame = pd.read_csv(data, index_col=0, parse_dates=True)
    # All required series and the target must be present with a long history.
    for column in list(workflow.ASIAN_MARKETS) + list(workflow.US_REFERENCE) + [workflow.TARGET]:
        assert column in frame.columns
    assert len(frame) > 3000
    doc = json.loads(manifest.read_text())
    assert "causality_note" in doc and len(doc["series"]) >= 10
