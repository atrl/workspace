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
