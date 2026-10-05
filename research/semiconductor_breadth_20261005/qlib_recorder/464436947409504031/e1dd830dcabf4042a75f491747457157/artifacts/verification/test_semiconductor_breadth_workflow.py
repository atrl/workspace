"""Contract tests use synthetic observations, never the study's financial outcomes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow import semiconductor_breadth as w
from src.workflow import soxs_cash5_revalidation as old


def prices():
    dates = pd.bdate_range("2021-01-01", "2026-09-25")
    t = np.arange(len(dates))
    return pd.DataFrame({"soxl_close": 20 * np.exp(.0005 * t + .1 * np.sin(t / 17)),
                         "soxx_close": 40 * np.exp(.0002 * t + .06 * np.sin(t / 23)),
                         "soxx_open": 40 * np.exp(.0002 * t + .06 * np.sin((t - .3) / 23))}, index=dates)


def market_frame():
    frame = w.build_base_features_and_labels(prices()).loc[w.TRAIN_START:w.FEATURE_END].copy()
    t = np.arange(len(frame))
    for i, name in enumerate(w.SECTOR_FEATURES):
        frame[name] = np.sin(t / (11 + i)) + i
    return frame


def evaluation_frame(n=180):
    dates = pd.bdate_range("2024-01-01", periods=n)
    t = np.arange(n)
    y = .04 * np.sin(t / 13) + .01 * np.cos(t / 5)
    return pd.DataFrame({w.TARGET: y, "base": .015 * np.sin(t / 11),
                         "base_plus_breadth": .03 * np.sin(t / 13), "prior_mean": np.zeros(n),
                         "trade_date": dates + pd.Timedelta(days=1),
                         "label_mature_date": dates + pd.Timedelta(days=8)}, index=dates)


def test_exact_base5_parity_and_full_calendar_h5_before_sector_window():
    source = prices()
    old_source = source.copy()
    old_source["soxs_close"] = source.soxx_close
    actual = w.build_base_features_and_labels(source)
    reference = old.build_features_and_labels(old_source)
    pd.testing.assert_frame_equal(actual[list(w.BASE_FEATURES)], reference[list(w.BASE_FEATURES)])
    selected = actual.loc[w.TRAIN_START:w.FEATURE_END]
    day = pd.Timestamp(w.FEATURE_END)
    position = source.index.get_loc(day)
    assert selected.loc[day, w.TARGET] == pytest.approx(-np.log(source.soxx_open.iloc[position + 6] / source.soxx_open.iloc[position + 1]))
    assert selected.loc[day, "label_mature_date"] > day
    assert selected.loc[day, "feature_available_at"].hour == 17
    revised = source.copy()
    revised.loc[revised.index > day] *= 3
    causal = w.build_base_features_and_labels(revised)
    pd.testing.assert_frame_equal(actual.loc[:day, list(w.BASE_FEATURES)], causal.loc[:day, list(w.BASE_FEATURES)])


def test_native_dataset_and_paired_annual_training_purge_maturity():
    from qlib.data.dataset import DatasetH
    market = market_frame()
    dataset, audit = w.build_dataset(market)
    assert isinstance(dataset, DatasetH)
    prepared = w.prepare_dataset(dataset)
    pd.testing.assert_frame_equal(prepared, market[[*w.ALL_FEATURES, w.TARGET]], check_names=False, check_freq=False)
    assert audit["feature_count"] == 9
    for year in w.FIT_YEARS:
        train, fit = w.training_rows(prepared, market, year)
        prior = prepared.loc[prepared.index < f"{year}-01-01"]
        assert train.index.max() == prior.index[-21]
        assert fit["purged_signal_rows"] == 20
        assert fit["maximum_label_available_at"] < pd.Timestamp(f"{year}-01-01", tz="America/New_York")
        assert len(train) >= 252
    poisoned = market.copy()
    poisoned.loc[market.index[0], "label_available_at"] = pd.Timestamp("2030-01-01", tz="America/New_York")
    train, fit = w.training_rows(prepared, poisoned, 2024)
    assert market.index[0] not in train.index
    assert fit["maturity_excluded_rows"] == 1


def test_missing_rows_fail_closed_and_fit_budget_is_exactly_six():
    market = market_frame()
    broken = market.copy()
    broken.loc[market.index[10], w.SECTOR_FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="interior missing"):
        w.build_dataset(broken)
    short = market.loc["2023-10-01":]
    dataset, _ = w.build_dataset(short)
    with pytest.raises(ValueError, match="252"):
        w.fit_predict(dataset, short, deadline="2030-01-01T00:00:00+00:00")
    dataset, _ = w.build_dataset(market)
    predicted, audit, models = w.fit_predict(dataset, market, deadline="2030-01-01T00:00:00+00:00")
    assert len(models) == len(audit) == 6
    assert set(audit.fit_year) == {2024, 2025, 2026}
    assert set(audit.feature_count) == {5, 9}
    assert predicted.index.equals(market.loc["2024-01-01":].index)
    for model in models.values():
        assert model["ridge"].alpha == 10


def test_paired_bootstrap_preserves_identical_draws_and_undefined_bounds():
    sample = evaluation_frame()
    stats, draws = w.paired_bootstrap(sample, paths=71, seed=20261005)
    repeated, again = w.paired_bootstrap(sample, paths=71, seed=20261005)
    assert stats == repeated
    pd.testing.assert_frame_equal(draws, again)
    assert stats["paired_same_indices"]
    np.testing.assert_allclose(draws.delta_mse_full_minus_base, -draws.mse_reduction_base_minus_full)
    identical = sample.copy()
    identical["base_plus_breadth"] = identical["base"]
    _, same = w.paired_bootstrap(identical, paths=31)
    assert same.delta_ic_full_minus_base.eq(0).all()
    assert same.mse_reduction_base_minus_full.eq(0).all()
    sample["base_plus_breadth"] = 0
    undefined, _ = w.paired_bootstrap(sample, paths=31)
    assert not undefined["base_plus_breadth_ic_official_interval_defined"]
    assert np.isnan(undefined["base_plus_breadth_ic_q0.025"])
    assert np.isnan(undefined["full_conditional_net_event_mean_bp25_q0.025"])


def test_three_registered_gates_require_annual_and_economic_evidence():
    points = pd.DataFrame({"slice": [f"year_{y}" for y in w.FIT_YEARS],
                           "base_plus_breadth_ic": [.1, .2, -.1],
                           "delta_ic_full_minus_base": [.1, .2, -.1],
                           "mse_reduction_base_minus_full": [.1, -.1, .2]})
    rows = []
    for block in (20, 60):
        row = {"slice": "pooled_2024_2026_partial", "block_length": block}
        for metric in ("base_plus_breadth_ic", "delta_ic_full_minus_base", "mse_reduction_base_minus_full",
                       "full_conditional_net_event_mean_bp10", "full_conditional_net_event_mean_bp25"):
            row[metric + "_q0.025"] = .01
        rows.append(row)
    bootstrap = pd.DataFrame(rows)
    assert w.research_gates(points, bootstrap)["all_incremental_information_gates_pass"]
    # Registration requires each annual count >= 2, not the intersection of counts.
    assert w.research_gates(points, bootstrap)["joint_positive_increment_years"] == 1
    bootstrap.loc[1, "full_conditional_net_event_mean_bp25_q0.025"] = np.nan
    gate = w.research_gates(points, bootstrap)
    assert not gate["economic_pass"] and not gate["all_incremental_information_gates_pass"]
    assert gate["promotion_decision"] == "not_promoted"
    points.loc[1, "base_plus_breadth_ic"] = -.1
    assert not w.research_gates(points, bootstrap)["information_pass"]


def test_no_financial_execution_without_root_go(tmp_path):
    root = Path(__file__).resolve().parents[1]
    card = tmp_path / "research" / w.EXPERIMENT
    card.mkdir(parents=True)
    plan = json.loads((root / "research" / w.EXPERIMENT / "plan.json").read_text())
    (card / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(PermissionError, match="root GO"):
        w.run_experiment(root=tmp_path)


def test_native_card_is_notebook_first_and_keeps_large_evidence_recorder_only():
    root = Path(__file__).resolve().parents[1]
    card = root / "research" / w.EXPERIMENT
    notebook = json.loads((card / f"{w.EXPERIMENT}.ipynb").read_text())
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for token in ("qlib.init", "D.calendar", "DatasetH", "R.get_recorder", "run_experiment"):
        assert token in text
    assert notebook["cells"][-1]["cell_type"] == "markdown"
    for filename in ("predictions.csv", "signals.csv", "portfolio_report.csv", "all_model_predictions.csv"):
        assert not (card / filename).exists()
