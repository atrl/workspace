import numpy as np
import pandas as pd
import pytest

from src.analysis import stable_excess_frontier as analysis
from src.workflow import stable_excess_frontier as workflow


def _prices(periods=700, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=periods)
    soxx_ret = rng.normal(0.0004, 0.011, periods)
    soxl_ret = 3.0 * soxx_ret + rng.normal(0.0, 0.006, periods)
    soxx = 100.0 * np.cumprod(1.0 + soxx_ret)
    soxl = 50.0 * np.cumprod(1.0 + soxl_ret)
    return pd.DataFrame({"signal_date": dates, "soxl_close": soxl, "soxx_close": soxx})


def test_build_state_uses_only_past_information():
    prices = _prices()
    state = analysis.build_state(prices, vol_window=20, trend_window=50)
    # First vol_window rows have no realized volatility.
    assert state["realized_vol"].iloc[:19].isna().all()
    assert np.isfinite(state["realized_vol"].iloc[20])
    # The realized return at row t equals close_t/close_{t-1}-1.
    expected = prices["soxx_close"].pct_change().iloc[5]
    assert state["soxx_return_realized"].iloc[5] == pytest.approx(expected)


def test_target_weights_respect_funded_simplex_for_every_policy():
    prices = _prices()
    state = analysis.build_state(prices, vol_window=20, trend_window=50)
    row = state.iloc[100]
    for policy in workflow.registered_candidates():
        weights = analysis.target_weights(row, policy)
        assert weights.shape == (3,)
        assert (weights >= -1e-12).all()
        assert weights.sum() == pytest.approx(1.0)
        # No candidate may short or exceed full allocation.
    buy_hold = analysis.target_weights(row, analysis.ExposurePolicy("b", "soxx_buy_hold"))
    assert buy_hold.tolist() == [0.0, 1.0, 0.0]


def test_vol_target_never_exceeds_target_and_never_holds_soxl():
    prices = _prices()
    state = analysis.build_state(prices, vol_window=20, trend_window=50)
    policy = analysis.ExposurePolicy("vt", "vol_target_soxx", target_vol=0.20)
    for _, row in state.dropna(subset=["realized_vol"]).iterrows():
        weights = analysis.target_weights(row, policy)
        assert weights[0] == 0.0
        exposure = weights[1]
        realized = float(row["realized_vol"])
        expected = min(1.0, 0.20 / realized)
        assert exposure == pytest.approx(expected)


def test_simulate_policy_applies_next_session_return_and_charges_turnover():
    prices = _prices(periods=120)
    state = analysis.build_state(prices, vol_window=10, trend_window=20).dropna(
        subset=["realized_vol", "above_trend"]).reset_index(drop=True)
    policy = analysis.ExposurePolicy("bh", "soxx_buy_hold")
    replay = analysis.simulate_policy(state, policy, cost_bps=25.0)
    # Buy-and-hold SOXX should match the realized SOXX return minus the fee only
    # on the first rebalance (all turnover is at entry).
    first = replay.iloc[1]
    assert first["strategy_return"] == pytest.approx(first["soxx_return"])
    # Terminal row (no next session) must be dropped.
    assert len(replay) == len(state) - 1


def test_risk_metrics_match_manual_terminal_and_drawdown():
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
    metrics = analysis.risk_metrics(returns)
    growth = np.cumprod(1.0 + returns)
    assert metrics["terminal_wealth"] == pytest.approx(float(growth[-1]))
    peak = np.maximum.accumulate(growth)
    assert metrics["max_drawdown_pct"] == pytest.approx(100.0 * float((growth / peak - 1.0).min()))


def test_excess_verdict_requires_return_drawdown_and_positive_years():
    excess = {"beats_return": True, "beats_drawdown": True, "beats_sharpe": True}
    annual = pd.DataFrame({"wealth_ratio_vs_soxx": [1.2, 1.1, 0.9, 1.3]})
    verdict = analysis.stable_excess_verdict(excess, annual, minimum_positive_years=3)
    assert verdict["stable_excess"] is True
    assert verdict["positive_year_count"] == 3
    tight = analysis.stable_excess_verdict(excess, annual, minimum_positive_years=4)
    assert tight["stable_excess"] is False


def test_forecast_ic_flags_degenerate_constant_forecast():
    realized = np.array([0.01, -0.02, 0.03, 0.0, 0.01])
    constant = np.zeros_like(realized)
    result = analysis.forecast_information_coefficient(constant, realized)
    assert result["degenerate"] is True
    assert result["pearson_ic"] is None
    informative = analysis.forecast_information_coefficient(realized, realized)
    assert informative["pearson_ic"] == pytest.approx(1.0)


def test_early_path_correlation_reports_conditional_means():
    early = np.array([0.01, -0.01, 0.02, -0.02, 0.005])
    later = np.array([0.002, -0.001, 0.003, -0.002, 0.001])
    result = analysis.intraday_early_predictive_correlation(early, later)
    assert result["observations"] == 5
    assert result["later_mean_when_early_positive_pct"] > result["later_mean_when_early_negative_pct"]


def test_registered_candidate_family_is_bounded_and_includes_benchmark():
    candidates = workflow.registered_candidates()
    names = {policy.name for policy in candidates}
    assert "soxx_buy_hold" in names
    assert len(candidates) <= 10
    assert len(names) == len(candidates)
    for policy in candidates:
        policy.validate()


def test_development_selection_never_picks_benchmark():
    dev = {
        "soxx_buy_hold": {"verdict": {"stable_excess": True, "positive_year_count": 7},
                          "excess": {"sharpe_difference": 9.0, "max_drawdown_improvement_pct": 9.0}},
        "candidate": {"verdict": {"stable_excess": False, "positive_year_count": 3},
                      "excess": {"sharpe_difference": 0.1, "max_drawdown_improvement_pct": 0.1}},
    }
    assert workflow.select_development_candidate(dev) == "candidate"


def test_select_development_candidate_requires_non_benchmark():
    dev = {"soxx_buy_hold": {"verdict": {"stable_excess": True, "positive_year_count": 7},
                             "excess": {"sharpe_difference": 0.0, "max_drawdown_improvement_pct": 0.0}}}
    with pytest.raises(ValueError, match="no non-benchmark"):
        workflow.select_development_candidate(dev)


def test_segment_filters_to_registered_window():
    state = analysis.build_state(_prices(), vol_window=20, trend_window=50)
    window = ("2018-06-01", "2018-12-31")
    segment = workflow._segment(state, window)
    assert segment["signal_date"].min() >= pd.Timestamp(window[0])
    assert segment["signal_date"].max() <= pd.Timestamp(window[1])
    assert len(segment) > 0


def _build_synthetic_provider(tmp_path, periods=3000, seed=11):
    from src.data.qlib_native import build_qlib_store_from_frame

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-10-08", periods=periods)
    soxx_ret = rng.normal(0.0004, 0.010, periods)
    soxl_ret = 3.0 * soxx_ret + rng.normal(0.0, 0.005, periods)
    soxx = 100.0 * np.cumprod(1.0 + soxx_ret)
    soxl = 50.0 * np.cumprod(1.0 + soxl_ret)
    rows = []
    for date, sxx, sxl in zip(dates, soxx, soxl):
        rows.append({"instrument": "SOXX", "datetime": date, "close": sxx, "open": sxx * 0.999})
        rows.append({"instrument": "SOXL", "datetime": date, "close": sxl, "open": sxl * 0.999})
    frame = pd.DataFrame(rows)
    provider_uri = tmp_path / "data" / "qlib" / "soxl"
    build_qlib_store_from_frame(
        frame,
        provider_uri=provider_uri,
        as_of=str(dates[-1].date()),
        source="test_synthetic",
    )
    return provider_uri


def test_workflow_end_to_end_records_and_writes_slim_card(tmp_path):
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    provider_uri = _build_synthetic_provider(tmp_path)
    card_dir = tmp_path / "card"
    summary = workflow.run_stable_excess_frontier(
        root=tmp_path,
        card_dir=card_dir,
        recorder_uri=card_dir / "qlib_recorder",
    )
    assert summary["provider_uri"].endswith("data/qlib/soxl")
    assert summary["recorder_id"]
    assert summary["candidate_count"] == len(workflow.registered_candidates())
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["return_forecast_used"] is False
    assert summary["model_fits"] == 0
    workflow.write_card_files(summary, card_dir=card_dir)
    assert (card_dir / "summary.json").exists()
    assert (card_dir / "workflow.yaml").exists()
    assert (card_dir / "conclusion.md").exists()
    assert (card_dir / "policy_comparison.csv").exists()
    # No recorder-only large CSV may live in the card root.
    for forbidden in ("predictions.csv", "signals.csv", "all_model_predictions.csv",
                      "portfolio_report.csv", "portfolio_report.csv"):
        assert not (card_dir / forbidden).exists()


# ---------------------------------------------------------------------------
# Intraday early-momentum card
# ---------------------------------------------------------------------------

from src.analysis import intraday_early_momentum as momentum
from src.workflow import intraday_early_momentum as momentum_workflow


def _intraday_day(date, open_price, decision_price, close_price):
    day = pd.Timestamp(date).tz_localize("America/New_York")
    index = pd.DatetimeIndex([
        day + pd.Timedelta(hours=9, minutes=45),
        day + pd.Timedelta(hours=10, minutes=30),
        day + pd.Timedelta(hours=16),
    ])
    # 09:45 bar: open == session open, close == 10:00 close
    # 10:30 bar: close == decision price
    # 16:00 bar: close == session close
    return pd.DataFrame({"open": [open_price, decision_price, close_price],
                         "close": [decision_price, decision_price, close_price]}, index=index)


def test_build_early_session_frame_uses_open_to_decision_and_decision_to_close():
    days = pd.bdate_range("2021-01-04", periods=5)
    soxl = pd.concat([_intraday_day(d, 100.0, 102.0, 105.0) for d in days])
    soxx = pd.concat([_intraday_day(d, 100.0, 101.0, 103.0) for d in days])
    frame = momentum.build_early_session_frame(soxl, soxx).frame
    assert len(frame) == 5
    assert frame["early_return"].iloc[0] == pytest.approx(0.02)
    assert frame["rest_soxl"].iloc[0] == pytest.approx(105.0 / 102.0 - 1.0)
    assert frame["rest_soxx"].iloc[0] == pytest.approx(103.0 / 101.0 - 1.0)


def test_momentum_target_weights_thresholds_and_simplex():
    policy = momentum.MomentumPolicy("p", long_threshold=0.02, short_threshold=0.03)
    assert momentum.target_weights(0.05, policy).tolist() == [1.0, 0.0, 0.0]
    assert momentum.target_weights(0.0, policy).tolist() == [0.0, 1.0, 0.0]
    assert momentum.target_weights(-0.05, policy).tolist() == [0.0, 0.0, 1.0]
    for value in (0.05, 0.0, -0.05):
        weights = momentum.target_weights(value, policy)
        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= 0).all()


def test_momentum_simulation_charges_turnover_and_has_no_overnight():
    days = pd.bdate_range("2021-01-04", periods=4)
    rows = []
    for i, d in enumerate(days):
        rows.append({
            "signal_date": d,
            "soxl_early": 0.05 if i % 2 == 0 else -0.05,
            "soxx_early": 0.05 if i % 2 == 0 else -0.05,
            "rest_soxl": 0.01, "rest_soxx": 0.002,
        })
    frame = pd.DataFrame(rows)
    policy = momentum.MomentumPolicy("p", long_threshold=0.02, short_threshold=0.02)
    replay = momentum.simulate_policy(frame, policy, cost_bps=10.0)
    assert len(replay) == 4
    # A full flip each session means turnover 2 each time after the first.
    assert replay["turnover"].iloc[1] == pytest.approx(2.0)
    # Strategy return must equal gross minus turnover fee.
    row = replay.iloc[1]
    assert row["strategy_return"] == pytest.approx(row["gross_return"] - 2.0 * 10.0 / 1e4)


def test_momentum_benchmark_is_soxx_rest_of_day():
    days = pd.bdate_range("2021-01-04", periods=3)
    frame = pd.DataFrame({"signal_date": days, "early_return": 0.0,
                          "rest_soxl": 0.02, "rest_soxx": 0.004})
    bench = momentum.benchmark_replay(frame, cost_bps=10.0)
    assert (bench["soxx_weight"] == 1.0).all()
    assert bench["strategy_return"].iloc[0] == pytest.approx(0.004)


def test_momentum_registered_family_is_bounded_and_includes_off_control():
    candidates = momentum_workflow.registered_candidates()
    names = {policy.name for policy in candidates}
    assert "early_momentum_off" in names
    assert len(candidates) <= 8
    for policy in candidates:
        policy.validate()


def test_momentum_selection_never_picks_off_control():
    results = {
        "early_momentum_off": {"verdict": {"stable_excess": True, "positive_year_count": 9},
                               "excess": {"sharpe_difference": 9.0, "max_drawdown_improvement_pct": 9.0}},
        "early_momentum_l2": {"verdict": {"stable_excess": False, "positive_year_count": 2},
                              "excess": {"sharpe_difference": 0.2, "max_drawdown_improvement_pct": 0.1}},
    }
    assert momentum_workflow.select_development_candidate(results) == "early_momentum_l2"


# ---------------------------------------------------------------------------
# Cross-source signal search
# ---------------------------------------------------------------------------

from src.analysis import cross_source_signal_search as search_analysis
from src.workflow import cross_source_signal_search as search_workflow


def _signal_series(periods=800, seed=3, drift=0.0003):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=periods)
    ret = rng.normal(drift, 0.012, periods)
    return pd.Series(ret, index=dates)


def test_build_exposure_is_causal_expanding_quantile():
    rng = np.random.default_rng(1)
    signal = pd.Series(rng.normal(0, 1, 300), index=pd.bdate_range("2016-01-04", periods=300))
    exposure = search_analysis.build_exposure(signal, direction=1, threshold=0.5)
    # First 59 values cannot be classified (min_periods=60).
    assert exposure.iloc[:59].isna().all()
    assert exposure.iloc[60:].isin([0.0, 1.0]).all() or exposure.iloc[60:].isna().all()


def test_evaluate_signal_applies_next_day_exposure_and_charges_turnover():
    dates = pd.bdate_range("2016-01-04", periods=200)
    signal = pd.Series(np.tile([1.0, -1.0], 100), index=dates)
    risky = pd.Series(0.01, index=dates)
    safe = pd.Series(0.0, index=dates)
    replay = search_analysis.evaluate_signal(signal, risky, safe, direction=1, threshold=0.5, cost_bps=10.0)
    assert replay is not None
    assert set(replay) == {"strategy", "benchmark", "exposure", "turnover", "sessions"}
    # First applied exposure is 0 (no history).
    assert replay["exposure"].iloc[0] == 0.0
    # Turnover is nonnegative and the strategy never exceeds the risky return.
    assert (replay["turnover"] >= 0).all()
    assert (replay["strategy"] <= risky.loc[replay["strategy"].index] + 1e-12).all()


def test_evaluate_signal_returns_none_when_history_too_short():
    dates = pd.bdate_range("2016-01-04", periods=30)
    signal = pd.Series(np.arange(30.0), index=dates)
    assert search_analysis.evaluate_signal(signal, signal * 0.0, signal * 0.0,
                                           direction=1, threshold=0.5, cost_bps=10.0) is None


def test_summarize_search_flags_stable_excess_only_when_all_dimensions_pass():
    dates = pd.bdate_range("2016-01-04", periods=800)
    rng = np.random.default_rng(5)
    noise = pd.Series(rng.normal(0.0, 0.001, len(dates)), index=dates)
    # A strategy that strictly dominates the benchmark in level with identical noise.
    bench = 0.0005 + noise
    good = bench + 0.0004
    results = {
        "dominates": {"strategy": good, "benchmark": bench,
                      "exposure": pd.Series(1.0, index=dates), "turnover": pd.Series(0.0, index=dates),
                      "sessions": len(dates)},
        "loses": {"strategy": bench * 0.5, "benchmark": bench,
                  "exposure": pd.Series(0.0, index=dates), "turnover": pd.Series(0.0, index=dates),
                  "sessions": len(dates)},
    }
    table = search_analysis.summarize_search(results, minimum_positive_years=2)
    row = table.set_index("signal").loc["dominates"]
    assert bool(row["beats_return"]) is True
    assert bool(row["beats_sharpe"]) is True
    assert bool(row["stable_excess"]) is True
    assert bool(table.set_index("signal").loc["loses", "stable_excess"]) is False


def test_registered_signals_are_bounded_and_causal():
    dates = pd.bdate_range("2015-01-02", periods=400)
    rng = np.random.default_rng(2)
    series = {name: pd.Series(rng.normal(0, 0.01, 400), index=dates)
              for name in ("soxl_return", "soxx_return", "soxl_reversal", "soxx_reversal",
                           "leader_capmkt_reversal", "leader_breadth_reversal",
                           "leader_dispersion", "realized_vol", "trend",
                           "intraday_first_hour_return", "intraday_close_location",
                           "intraday_realized_vol_1m", "macro_vix_level", "macro_vix_term_slope",
                           "macro_credit_spread", "macro_dollar_index", "macro_ten_year_yield",
                           "macro_semiconductor_index")}
    candidates = search_workflow.registered_signals(series)
    assert 12 <= len(candidates) <= 80
    for spec in candidates:
        spec.validate()
        assert spec.direction in (1, -1)
