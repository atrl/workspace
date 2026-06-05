from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import src.soxl_quant.data as data_mod
from scripts.evaluate_signed_te_reversion import (
    compute_time_series_ic,
    run_long_flat_backtest,
)
from scripts.evaluate_signed_te_stress_gate import build_gated_signal
from src.soxl_quant.app.factor_diagnostics import (
    build_factor_gate_summary,
    render_factor_gate_summary,
)
from src.soxl_quant.data import engineer_feature_frame, feature_columns_for_packs


def _frame_from_panel(df: pd.DataFrame, prefix: str, *, include_vix: bool = False) -> pd.DataFrame:
    payload = {
        "$close": df[f"{prefix}_close"],
        "$open": df[f"{prefix}_open"],
        "$high": df[f"{prefix}_high"],
        "$low": df[f"{prefix}_low"],
        "$volume": df[f"{prefix}_volume"],
        "Mean($close, 5)": df[f"{prefix}_close"].rolling(5).mean(),
        "Mean($close, 21)": df[f"{prefix}_close"].rolling(21).mean(),
        "Mean($close, 63)": df[f"{prefix}_close"].rolling(63).mean(),
        "Mean($close, 180)": df[f"{prefix}_close"].rolling(180).mean(),
        "Std($close / Ref($close, 1) - 1, 10)": df[f"{prefix}_close"].pct_change().rolling(10).std(),
        "Std($close / Ref($close, 1) - 1, 21)": df[f"{prefix}_close"].pct_change().rolling(21).std(),
        "Std($close / Ref($close, 1) - 1, 63)": df[f"{prefix}_close"].pct_change().rolling(63).std(),
        "Mean($volume, 21)": df[f"{prefix}_volume"].rolling(21).mean(),
    }
    if include_vix:
        payload["$vix"] = df["vix"]
    return pd.DataFrame(payload, index=df.index)


def test_time_series_ic_scores_positive_forward_alignment():
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    feature = pd.Series([1, -1, 1, -1, 1, -1, 1, -1], index=index, dtype=float)
    returns = pd.Series([0.0, 0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.02], index=index)

    result = compute_time_series_ic(feature, returns, horizon=1)

    assert result.n_days == 7
    assert result.ic_mean > 0.0
    assert result.ic_positive_ratio > 0.70


def test_long_flat_backtest_uses_signal_after_previous_close():
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    signal = pd.Series([0.0, 2.0, 2.0, -1.0, -1.0, 2.0], index=index)
    returns = pd.Series([0.0, 0.10, 0.10, -0.10, -0.10, 0.10], index=index)

    result = run_long_flat_backtest(signal, returns, threshold=0.0)

    expected_daily = pd.Series([0.0, 0.0, 0.10, -0.10, 0.0, 0.0], index=index)
    expected_equity = (1.0 + expected_daily).cumprod()
    assert np.allclose(result.equity.to_numpy(), expected_equity.to_numpy())
    assert result.num_signals == 3
    assert result.trades == 2


def test_build_gated_signal_keeps_only_release_days_above_thresholds():
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    signed_te = pd.Series([0.2, 0.8, 1.2, 1.4, 0.9, 1.8], index=index)
    vix_divergence = pd.Series([0.0, 0.2, 0.6, 0.1, 0.7, 0.8], index=index)
    regime_risk = pd.Series([0.8, 0.75, 0.70, 0.72, 0.65, 0.60], index=index)

    signal = build_gated_signal(
        signed_te=signed_te,
        vix_divergence=vix_divergence,
        regime_risk=regime_risk,
        te_threshold=1.0,
        vix_divergence_threshold=0.5,
        regime_risk_threshold=0.7,
        regime_risk_drop_5_threshold=0.1,
    )

    expected = pd.Series([0.0, 0.0, 1.2, 0.0, 0.0, 1.8], index=index)
    assert np.allclose(signal.to_numpy(), expected.to_numpy())


def test_factor_gate_summary_separates_pass_redundant_and_weak_features():
    ic_df = pd.DataFrame(
        {
            "feature_name": ["strong", "redundant", "weak"],
            "horizon": [20, 20, 20],
            "n_days": [100, 100, 100],
            "ic_mean": [0.041, 0.052, 0.010],
            "ic_std": [0.10, 0.12, 0.11],
            "ic_t": [4.1, 4.3, 0.9],
            "ic_positive_ratio": [0.62, 0.64, 0.49],
            "pearson_corr": [0.10, 0.12, 0.01],
        }
    )
    backtest_df = pd.DataFrame(
        {
            "feature_name": ["strong", "redundant", "weak"],
            "direction": ["long", "long", "long"],
            "threshold": [0.5, 0.5, 0.5],
            "total_return": [0.20, 0.30, 0.02],
            "sharpe": [0.80, 0.90, 0.10],
            "max_drawdown": [-0.20, -0.25, -0.15],
            "profit_factor": [1.35, 1.40, 1.05],
            "signal_accuracy": [0.58, 0.59, 0.51],
            "num_signals": [40, 45, 20],
            "trades": [8, 9, 4],
        }
    )
    corr_df = pd.DataFrame(
        {
            "feature_name": ["strong", "redundant", "weak"],
            "max_abs_core_correlation": [0.30, 0.91, 0.20],
            "max_abs_core_correlation_feature": ["mom_5", "mom_5", "vol_21"],
        }
    )

    summary = build_factor_gate_summary(ic_df, backtest_df, corr_df).set_index("feature_name")

    assert bool(summary.loc["strong", "full_gate"]) is True
    assert summary.loc["strong", "primary_failure"] == "pass"
    assert bool(summary.loc["redundant", "full_gate"]) is False
    assert summary.loc["redundant", "primary_failure"] == "correlation_redundant"
    assert bool(summary.loc["weak", "full_gate"]) is False
    assert summary.loc["weak", "primary_failure"] == "weak_direction_and_execution"

    review = render_factor_gate_summary(summary.reset_index())
    assert "Factor Gate Summary" in review
    assert "correlation_redundant" in review


def test_intraday_microstructure_pack_is_optional_when_cache_is_missing():
    with TemporaryDirectory() as tmp:
        old_external = data_mod.EXTERNAL_DATA_DIR
        data_mod.EXTERNAL_DATA_DIR = Path(tmp)
        try:
            panel = pd.read_csv(data_mod.ROOT / "data/processed/panels/soxl_soxx_vix_full.csv", parse_dates=["date"])
            panel = panel.set_index("date").tail(360)
            soxl = _frame_from_panel(panel, "soxl", include_vix=True)
            soxx = _frame_from_panel(panel, "soxx")

            feature_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=("intraday_microstructure_v1",))
            flat = feature_df.droplevel("instrument")
        finally:
            data_mod.EXTERNAL_DATA_DIR = old_external

    expected = set(feature_columns_for_packs(("intraday_microstructure_v1",))) - set(feature_columns_for_packs(()))
    assert expected.issubset(flat.columns)
    assert (flat[list(expected)].abs().sum(axis=1) == 0.0).all()


def test_intraday_microstructure_pack_uses_lagged_daily_cache():
    with TemporaryDirectory() as tmp:
        old_external = data_mod.EXTERNAL_DATA_DIR
        data_mod.EXTERNAL_DATA_DIR = Path(tmp)
        try:
            panel = pd.read_csv(data_mod.ROOT / "data/processed/panels/soxl_soxx_vix_full.csv", parse_dates=["date"])
            panel = panel.set_index("date").tail(360)
            opening_range = pd.Series(np.linspace(0.01, 0.05, len(panel)), index=panel.index)
            close_location = pd.Series(np.linspace(0.20, 0.90, len(panel)), index=panel.index)
            cached = pd.DataFrame(
                {
                    "date": panel.index.strftime("%Y-%m-%d"),
                    "opening_range_30": opening_range.to_numpy(),
                    "close_location_value": close_location.to_numpy(),
                    "realized_intraday_vol": np.linspace(0.001, 0.020, len(panel)),
                    "volume_concentration_open_30": np.linspace(0.10, 0.40, len(panel)),
                    "soxl_soxx_intraday_relative_volume": np.linspace(0.80, 1.20, len(panel)),
                    "intraday_tracking_error": np.linspace(-0.015, 0.015, len(panel)),
                }
            )
            cached.to_csv(Path(tmp) / "intraday_features_soxl_1m.csv", index=False)

            soxl = _frame_from_panel(panel, "soxl", include_vix=True)
            soxx = _frame_from_panel(panel, "soxx")
            feature_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=("intraday_microstructure_v1",))
            flat = feature_df.droplevel("instrument")
        finally:
            data_mod.EXTERNAL_DATA_DIR = old_external

    shifted_close_location = close_location.shift(1).fillna(0.0).reindex(flat.index)
    actual_close_location = flat["INTRADAY_CLOSE_LOCATION_VALUE_LAG1"].reindex(flat.index)
    assert np.allclose(actual_close_location.tail(50).to_numpy(), shifted_close_location.tail(50).to_numpy(), atol=1e-8)

    shifted_opening = opening_range.shift(1)
    expected_z = ((shifted_opening - shifted_opening.rolling(21).mean()) / (shifted_opening.rolling(21).std() + 1e-8)).fillna(0.0)
    actual_z = flat["INTRADAY_OPENING_RANGE_30_ZSCORE_21"].reindex(flat.index)
    assert np.allclose(actual_z.tail(50).to_numpy(), expected_z.reindex(flat.index).tail(50).to_numpy(), atol=1e-8)
