"""Synthetic-only tests for the registered earnings information diagnostic."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow import earnings_enhancement as w
from src.workflow.semiconductor_breadth import build_base_features_and_labels
from src.analysis.frozen_cash_information import correlation


def prices():
    dates = pd.bdate_range("2010-01-01", "2026-09-25")
    t = np.arange(len(dates))
    return pd.DataFrame({"soxl_close": 20*np.exp(.0002*t+.06*np.sin(t/19)),
                         "soxx_close": 40*np.exp(.0001*t+.02*np.sin(t/31)),
                         "soxl_open": 20*np.exp(.0002*t+.06*np.sin((t-.3)/19)),
                         "soxx_open": 40*np.exp(.0001*t+.02*np.sin((t-.3)/31))}, index=dates)


def market_frame():
    frame = w.build_features_and_labels(prices()).loc[w.TRAIN_START:w.FEATURE_END].copy()
    t = np.arange(len(frame))
    for i, name in enumerate(w.EARNINGS_FEATURES):
        frame[name] = np.sin(np.floor(t/60)/(3+i))
    return frame


def predictions(n=180):
    dates = pd.bdate_range("2016-01-01", periods=n)
    t = np.arange(n)
    result = pd.DataFrame({"trade_date": dates+pd.Timedelta(days=1),
                           "label_mature_date": dates+pd.Timedelta(days=8)}, index=dates)
    for i, head in enumerate(w.TARGETS):
        y = .03+.04*np.sin(t/(11+i))
        result[head] = y
        result[f"{head}_base"] = .02+.01*np.sin(t/(14+i))
        result[f"{head}_base_plus_earnings"] = .03+.02*np.sin(t/(11+i))
        result[f"{head}_prior_mean"] = .005
    return result


def test_exact_base5_dual_open_labels_and_cross_year_maturity():
    p = prices()
    actual = w.build_features_and_labels(p)
    reference = build_base_features_and_labels(p)
    pd.testing.assert_frame_equal(actual[list(w.BASE_FEATURES)], reference[list(w.BASE_FEATURES)])
    day = pd.Timestamp("2022-12-30")
    j = p.index.get_loc(day)
    soxx = p.soxx_open.iloc[j+6]/p.soxx_open.iloc[j+1]
    soxl = p.soxl_open.iloc[j+6]/p.soxl_open.iloc[j+1]
    assert actual.loc[day, "cash5"] == pytest.approx(-np.log(soxx))
    assert actual.loc[day, "leverage5"] == pytest.approx(np.log(soxl/soxx))
    assert actual.loc[day, "label_available_at"] > pd.Timestamp("2023-01-01", tz="America/New_York")
    assert actual[list(w.TARGETS)].isna().any(axis=1).sum() == 6
    changed = p.copy()
    changed.loc[changed.index > day] *= 2
    revised = w.build_features_and_labels(changed)
    pd.testing.assert_frame_equal(actual.loc[:day, list(w.BASE_FEATURES)], revised.loc[:day, list(w.BASE_FEATURES)])


def test_native_dataset_maturity_and_exact_fourteen_multioutput_fits():
    from qlib.data.dataset import DatasetH
    market = market_frame()
    dataset, audit = w.build_dataset(market)
    assert isinstance(dataset, DatasetH) and audit["feature_count"] == 8
    prepared = w.prepare_dataset(dataset)
    for year in w.FIT_YEARS:
        train, fit = w.training_rows(prepared, market, year)
        prior = prepared.loc[prepared.index < f"{year}-01-01"]
        assert train.index[-1] == prior.index[-21]
        assert fit["maximum_label_available_at"] < pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    poison = market.copy()
    poison.loc[market.index[0], "label_available_at"] = pd.Timestamp("2030-01-01", tz="America/New_York")
    train, audit = w.training_rows(prepared, poison, 2016)
    assert market.index[0] not in train.index and audit["maturity_excluded_rows"] == 1
    result, fits, models = w.fit_predict(dataset, market, deadline="2030-01-01T00:00:00Z")
    assert len(fits) == len(models) == 14
    assert set(fits.feature_count) == {5, 8}
    assert set(result.index.year) == set(range(2016, 2023))
    assert result.index[-1] == pd.Timestamp(w.FEATURE_END)
    assert all(model["ridge"].coef_.shape[0] == 2 and model["ridge"].alpha == 10 for model in models.values())


def test_missing_feature_fails_and_post2022_fit_is_forbidden():
    market = market_frame()
    bad = market.copy()
    bad.loc[market.index[10], w.EARNINGS_FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="interior missing"):
        w.build_dataset(bad)
    future = market.iloc[-1:].copy()
    future.index = pd.DatetimeIndex(["2023-01-03"])
    extended = pd.concat([market, future])
    dataset, _ = w.build_dataset(extended)
    with pytest.raises(ValueError, match="post2022"):
        w.fit_predict(dataset, extended, deadline="2030-01-01T00:00:00Z")


def test_cash_and_leverage_hurdles_equal_explicit_stock_transaction_legs():
    cost = 25
    c = cost/10000
    soxx_gross, soxl_gross = 1.04, 1.11
    # Begin with one unit of SOXX value. Buying spends cash inclusive of fee.
    cash_after_soxx_sale = 1*(1-c)
    cash_final_soxx_value = cash_after_soxx_sale/(1+c)
    soxl_initial_value = cash_after_soxx_sale/(1+c)
    cash_after_soxl_sale = soxl_initial_value*soxl_gross*(1-c)
    leverage_final_soxx_value = cash_after_soxl_sale/(1+c)
    assert np.log(cash_final_soxx_value/soxx_gross) == pytest.approx(-np.log(soxx_gross)-w.event_hurdle("cash5", cost))
    assert np.log(leverage_final_soxx_value/soxx_gross) == pytest.approx(np.log(soxl_gross/soxx_gross)-w.event_hurdle("leverage5", cost))
    hurdle = w.event_hurdle("leverage5", cost)
    metric = w.event_metrics([.1, .2], [hurdle, hurdle+1e-5], head="leverage5", cost_bps=cost)
    assert metric["event_count"] == 1
    assert metric["mean_event_net_log_bp"] == pytest.approx((.2-hurdle)*10000)


def test_shared_stationary_indices_direct_draw_and_degenerate_fail_closed():
    data = predictions()
    summary, draws, indices = w.paired_bootstrap(data, paths=61, block_length=20)
    assert summary.indices_sha256.nunique() == 1
    assert summary.shared_heads_same_indices.all()
    idx = indices[0]
    for head in w.TARGETS:
        sample = w.head_sample(data, head).iloc[idx]
        base, _ = correlation(sample.base, sample.target, rank=True)
        full, _ = correlation(sample.base_plus_earnings, sample.target, rank=True)
        assert draws[head].delta_ic_full_minus_base.iloc[0] == pytest.approx(full-base)
        exact = np.mean((sample.base-sample.target)**2)-np.mean((sample.base_plus_earnings-sample.target)**2)
        assert draws[head].mse_reduction_base_minus_full.iloc[0] == pytest.approx(exact)
    for head in w.TARGETS:
        data[f"{head}_base_plus_earnings"] = 0.
    summary, _, _ = w.paired_bootstrap(data, paths=31)
    assert not summary.full_ic_official_interval_defined.any()
    assert summary["full_ic_q0.0125"].isna().all()
    assert summary["full_conditional_net_event_mean_bp25_q0.0125"].isna().all()


def passing_gate_inputs():
    annual = [{"slice": f"year_{year}", "head": head, "full_ic": .1,
               "delta_ic_full_minus_base": .05, "mse_reduction_base_minus_full": .01}
              for head in w.TARGETS for year in w.FIT_YEARS]
    boot = []
    for head in w.TARGETS:
        for block in (20, 60):
            row = {"slice": w.POOLED, "head": head, "block_length": block, "paths": 2000}
            for metric in ("full_ic", "delta_ic_full_minus_base", "mse_reduction_base_minus_full", "full_conditional_net_event_mean_bp10", "full_conditional_net_event_mean_bp25"):
                row[metric+"_q0.0125"] = .01
                row[metric+"_official_interval_defined"] = True
            boot.append(row)
    return pd.DataFrame(annual), pd.DataFrame(boot)


def test_two_head_gates_cannot_select_winner_or_override_source_gate():
    points, bootstrap = passing_gate_inputs()
    result = w.research_gates(points, bootstrap)
    assert result["both_heads_information_and_economic_pass"]
    assert not result["source_deployment_gate"]
    assert result["promotion_decision"] == "not_promoted"
    bootstrap.loc[0, "full_ic_q0.0125"] = -.001
    bootstrap.loc[0, "full_ic_q0.025"] = .001
    assert not w.research_gates(points, bootstrap)["both_heads_information_and_economic_pass"]
    points, bootstrap = passing_gate_inputs()
    points.loc[(points["head"] == "leverage5") & points["slice"].isin(["year_2016", "year_2017", "year_2018"]), "full_ic"] = -.1
    assert not w.research_gates(points, bootstrap)["heads"]["leverage5"]["information_pass"]
    points, bootstrap = passing_gate_inputs()
    bootstrap.loc[3, "full_conditional_net_event_mean_bp25_official_interval_defined"] = False
    assert not w.research_gates(points, bootstrap)["heads"]["leverage5"]["economic_pass"]


def test_no_root_go_no_financial_execution_and_notebook_recorder_contract(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root/"research"/w.EXPERIMENT
    plan = json.loads((source/"plan.json").read_text())
    w.verify_registered_plan(plan)
    card = tmp_path/"research"/w.EXPERIMENT
    card.mkdir(parents=True)
    (card/"plan.json").write_text(json.dumps(plan))
    with pytest.raises(PermissionError, match="root GO"):
        w.run_experiment(root=tmp_path)
    notebook = json.loads((source/f"{w.EXPERIMENT}.ipynb").read_text())
    content = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for token in ("qlib.init", "D.calendar", "DatasetH", "R.get_recorder", "run_experiment"):
        assert token in content
    assert notebook["cells"][-1]["cell_type"] == "markdown"
    for filename in ("predictions.csv", "signals.csv", "all_model_predictions.csv", "portfolio_report.csv"):
        assert not (source/filename).exists()
    assert plan["artifact_policy"]["policy"] == "slim_card_recorder_first"
