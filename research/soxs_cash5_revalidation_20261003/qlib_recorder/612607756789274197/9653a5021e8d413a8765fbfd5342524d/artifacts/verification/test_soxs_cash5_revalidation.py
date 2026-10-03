from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from src.workflow import soxs_cash5_revalidation as w


def market_frame(start="2010-01-01", end="2026-10-01"):
    dates = pd.bdate_range(start, end)
    t = np.arange(len(dates))
    return pd.DataFrame({
        "soxl_close": 15 * np.exp(0.0004*t + 0.1*np.sin(t/37)),
        "soxx_close": 40 * np.exp(0.0002*t + 0.05*np.sin(t/41)),
        "soxs_close": 80 * np.exp(-0.0003*t + 0.07*np.cos(t/31)),
        "soxx_open": 40 * np.exp(0.0002*t + 0.05*np.sin((t-.2)/41)),
    }, index=dates)


def prepared_frame(frame):
    return frame.loc["2011-01-03":, [*w.ALL_FEATURES, w.TARGET]]


def evaluation_frame(n=180):
    dates = pd.bdate_range("2020-01-01", periods=n)
    t = np.arange(n)
    y = .04*np.sin(t/13) + .01*np.cos(t/5)
    return pd.DataFrame({w.TARGET:y, "base":.015*np.sin(t/11), "base_plus_soxs":.03*np.sin(t/13), "prior_mean":np.zeros(n), "trade_date":dates+pd.Timedelta(days=1), "label_mature_date":dates+pd.Timedelta(days=8)},index=dates)


def test_features_are_causal_and_exact_frozen_recipe():
    prices = market_frame()
    frame = w.build_features_and_labels(prices)
    point = 500
    revised = prices.copy()
    revised.iloc[point+1:] *= 3
    shifted = w.build_features_and_labels(revised)
    pd.testing.assert_frame_equal(frame.iloc[:point+1][list(w.ALL_FEATURES)], shifted.iloc[:point+1][list(w.ALL_FEATURES)])
    assert frame.soxx_trend20.iloc[point] == pytest.approx(prices.soxx_close.iloc[point]/prices.soxx_close.iloc[point-19:point+1].mean()-1)
    assert frame.inverse_tracking_gap1.iloc[point] == pytest.approx(prices.soxs_close.pct_change(fill_method=None).iloc[point]+3*prices.soxx_close.pct_change(fill_method=None).iloc[point])


def test_h5_label_uses_unfiltered_calendar_and_sixth_future_open():
    prices = market_frame("2020-12-20", "2021-02-01").drop(pd.Timestamp("2021-01-01"))
    frame = w.build_features_and_labels(prices)
    t=3
    assert frame[w.TARGET].iloc[t] == pytest.approx(-np.log(prices.soxx_open.iloc[t+6]/prices.soxx_open.iloc[t+1]))
    assert frame.trade_date.iloc[t] == prices.index[t+1]
    assert frame.label_mature_date.iloc[t] == prices.index[t+6]
    assert frame[w.TARGET].tail(6).isna().all()
    assert frame.label_available_at.iloc[t].hour == 9
    assert frame.label_available_at.iloc[t].minute == 30
    assert frame.feature_available_at.iloc[t].hour == 16


def test_purge_and_maturity_apply_to_every_fit_including_frozen_2023():
    market = w.build_features_and_labels(market_frame())
    prepared = prepared_frame(market)
    for year in w.FIT_YEARS:
        train, audit = w.training_rows(prepared, market, year)
        prior = prepared.loc[prepared.index < f"{year}-01-01"]
        assert train.index.max() == prior.index[-21]
        assert audit["purged_signal_rows"] == 20
        assert audit["maximum_label_available_at"] < pd.Timestamp(f"{year}-01-01", tz="America/New_York")
        assert len(train) >= 252
    poisoned = market.copy()
    poisoned.loc[prepared.index[10], "label_available_at"] = pd.Timestamp("2030-01-01",tz="America/New_York")
    train,_ = w.training_rows(prepared, poisoned, 2023)
    assert prepared.index[10] not in train.index


def test_invalid_interior_labels_and_source_prices_fail_closed():
    market = w.build_features_and_labels(market_frame())
    prepared = prepared_frame(market)
    prepared.loc["2012-03-01",w.TARGET]=np.nan
    with pytest.raises(ValueError,match="interior training"):
        w.training_rows(prepared,market,2023)
    bad_evaluation = market.copy()
    bad_evaluation.loc["2016-03-01",w.TARGET] = np.nan
    dataset,_ = w.build_dataset(bad_evaluation)
    with pytest.raises(ValueError,match="interior evaluation"):
        w.fit_predict(dataset,bad_evaluation,deadline="2030-01-01T00:00:00+00:00")
    prices = market_frame();prices.iloc[20,0] = -1
    with pytest.raises(ValueError,match="strictly positive"):
        w.build_features_and_labels(prices)
    market.loc["2012-03-01",w.ALL_FEATURES[0]]=np.nan
    with pytest.raises(ValueError,match="interior missing feature"):
        w.build_dataset(market)


def test_native_dataset_prepare_preserves_features_and_labels():
    market = w.build_features_and_labels(market_frame())
    dataset, audit = w.build_dataset(market)
    from qlib.data.dataset import DatasetH
    assert isinstance(dataset,DatasetH)
    actual=w.prepare_dataset(dataset)
    pd.testing.assert_frame_equal(actual,prepared_frame(market),check_names=False,check_freq=False)
    assert audit["api"] == ["DatasetH","DataHandlerLP","StaticDataLoader"]


def test_paired_bootstrap_same_indices_and_mse_sign():
    sample=evaluation_frame()
    stats,draws=w.paired_bootstrap(sample,paths=101,block_length=20)
    repeated,again=w.paired_bootstrap(sample,paths=101,block_length=20)
    pd.testing.assert_frame_equal(draws,again)
    assert stats==repeated
    assert stats["paired_same_indices"] is True
    np.testing.assert_allclose(draws.delta_mse_full_minus_base,-draws.mse_reduction_base_minus_full)
    same=sample.copy();same["base_plus_soxs"]=same["base"]
    _,same_draws=w.paired_bootstrap(same,paths=50)
    assert (same_draws.delta_ic_full_minus_base==0).all()
    assert (same_draws.delta_mse_full_minus_base==0).all()


def test_undefined_bootstrap_interval_is_not_silently_dropped():
    sample=evaluation_frame();sample["base_plus_soxs"]=0.
    stats,draws=w.paired_bootstrap(sample,paths=21)
    assert stats["base_plus_soxs_ic_undefined_draws"]==21
    assert not stats["base_plus_soxs_ic_official_interval_defined"]
    assert np.isnan(stats["base_plus_soxs_ic_q0.025"])
    assert np.isnan(stats["full_conditional_net_event_mean_bp10_q0.025"])


def test_cost_funding_equation_and_user_window_are_exact():
    sample=evaluation_frame()
    event=w.cost_event_diagnostics(sample,25)[1]
    hurdle=np.log((1+.0025)/(1-.0025))
    mask=sample.base_plus_soxs>hurdle
    assert event["log_hurdle"]==pytest.approx(hurdle)
    assert event["mean_event_net_log_bp"]==pytest.approx(10000*(sample.loc[mask,w.TARGET]-hurdle).mean())
    dates=pd.bdate_range("2026-07-01","2026-09-25")
    prices=market_frame("2026-07-01","2026-09-25")
    frame=w.build_features_and_labels(prices);frame["segment"]="reused_2023_plus"
    window=w.evaluation_slices(frame)["user_window"]
    assert window.trade_date.min()>=pd.Timestamp("2026-08-03")
    assert window.label_mature_date.max()<=pd.Timestamp("2026-09-25")
    assert window.label_mature_date.notna().all()
    assert len(window)<len(dates[dates>=pd.Timestamp("2026-08-03")])


def test_gate_false_when_any_official_bound_undefined_and_requires_approval(tmp_path):
    points=pd.DataFrame({"slice":[f"year_{y}" for y in range(2016,2023)],"base_plus_soxs_ic":[.1]*7})
    rows=[]
    for name in ("development","reused_2023_plus"):
        for block in (20,60):
            row={"slice":name,"block_length":block,"base_plus_soxs_ic_q0.05":.01}
            for metric in ("base_plus_soxs_ic","delta_ic_full_minus_base","mse_reduction_base_minus_full","full_conditional_net_event_mean_bp10","full_conditional_net_event_mean_bp25"):
                row[metric+"_q0.025"]=.01
            rows.append(row)
    bootstrap=pd.DataFrame(rows)
    assert w.research_gates(points,bootstrap)["all_research_gates_pass"]
    bootstrap.loc[2,"delta_ic_full_minus_base_q0.025"]=np.nan
    assert not w.research_gates(points,bootstrap)["incremental_pass"]
    root=Path(__file__).resolve().parents[1]
    card=tmp_path/"research"/w.EXPERIMENT;card.mkdir(parents=True)
    plan=json.loads((root/"research"/w.EXPERIMENT/"plan.json").read_text())
    (card/"plan.json").write_text(json.dumps(plan))
    with pytest.raises(PermissionError,match="root GO"):
        w.run_revalidation(root=tmp_path)
