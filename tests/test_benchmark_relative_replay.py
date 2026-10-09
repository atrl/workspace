"""Independent Qlib/provider, causal-clock and funded-execution regressions."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.analysis.convex_qlib import native_open_reconciliation
from src.backtest.benchmark_relative_replay import (
    replay_fixed_target, replay_online_allocator, validate_market,
)
from src.data.benchmark_relative_data import build_relative_dataset
from src.data.qlib_native import build_qlib_store_from_frame
from src.model.online_expert_allocation import FixedShareAllocator, FixedShareConfig


ASSETS = ("soxl", "soxx", "cash")
OPEN = [f"open_{asset}_weight" for asset in ASSETS]
PRETRADE = [f"pretrade_{asset}_weight" for asset in ASSETS]
ENDING = [f"ending_{asset}_weight" for asset in ASSETS]


def _config():
    return FixedShareConfig(
        expert_names=("soxx", "leveraged", "defensive"),
        expert_weights=((0., 1., 0.), (.2, .8, 0.), (0., .8, .2)),
        anchor_weights=(0., 1., 0.), prior=(1 / 3,) * 3,
        active_fraction=.5, eta=1., share=1 / 252,
    )


@pytest.fixture(scope="module")
def synthetic_source(tmp_path_factory):
    # Cross the US DST boundary: local decision/open clocks must stay fixed.
    dates = pd.bdate_range("2024-03-04", periods=16)
    rows = []
    values = {
        "SOXL": [100, 105, 125, 92, 119, 95, 126, 115, 97, 123, 140, 109, 119, 135, 124, 148],
        "SOXX": [100, 102, 107, 99, 108, 101, 111, 109, 103, 111, 119, 109, 114, 121, 117, 126],
    }
    for symbol, prices in values.items():
        for day, price in zip(dates, prices):
            rows.append({"instrument": symbol, "datetime": day, "open": float(price),
                         "high": price * 1.01, "low": price * .99, "close": float(price),
                         "volume": 1_000_000.})
    raw = pd.DataFrame(rows)
    provider = tmp_path_factory.mktemp("relative_provider") / "provider"
    build_qlib_store_from_frame(raw, provider_uri=provider, as_of=str(dates[-1].date()),
                               source="synthetic benchmark-relative contract test")
    return provider, dates, raw


@pytest.fixture
def native_market(synthetic_source):
    provider, dates, raw = synthetic_source
    dataset, frame, calendar = build_relative_dataset(
        provider_uri=provider, signal_start=str(dates[0].date()), as_of=str(dates[-1].date()))
    return provider, dataset, frame, calendar


def _experts(frame, *, cost=25., rebalance=3):
    config = _config()
    return {name: replay_fixed_target(frame, target=target, cost_bps=cost,
                rebalance_sessions=rebalance, buy_and_hold=name == "soxx")
            for name, target in zip(config.expert_names, config.expert_weights)}


def _online(frame, *, cost=25., rebalance=3, experts=None, config=None):
    return replay_online_allocator(frame, _experts(frame, cost=cost, rebalance=rebalance)
                                   if experts is None else experts,
        allocator=FixedShareAllocator(_config() if config is None else config),
        cost_bps=cost, rebalance_sessions=rebalance)


def test_native_dataset_roundtrip_uses_complete_declared_calendar_and_dst_clock(native_market):
    from qlib.data import D
    from qlib.data.dataset import DatasetH

    provider, dataset, frame, calendar = native_market
    assert isinstance(dataset, DatasetH)
    pd.testing.assert_index_equal(calendar, pd.DatetimeIndex(D.calendar(freq="day", future=False)))
    assert len(frame) == len(calendar) - 2
    pd.testing.assert_index_equal(pd.DatetimeIndex(frame.signal_date), calendar[:-2], check_names=False)
    pd.testing.assert_index_equal(pd.DatetimeIndex(frame.trade_date), calendar[1:-1], check_names=False)
    pd.testing.assert_index_equal(pd.DatetimeIndex(frame.exit_date), calendar[2:], check_names=False)
    opens = D.features(["SOXL", "SOXX"], ["$open"], start_time=calendar[0], end_time=calendar[-1])["$open"].unstack("instrument")
    expected = opens.iloc[2:].to_numpy(float) / opens.iloc[1:-1].to_numpy(float) - 1
    np.testing.assert_array_equal(frame[["soxl_return", "soxx_return"]].to_numpy(), expected)
    labels = dataset.prepare("all", col_set="label")
    np.testing.assert_array_equal(labels.to_numpy(), expected)
    assert set(frame.decision_at.dt.strftime("%H:%M")) == {"16:15"}
    assert set(frame.return_available_at.dt.strftime("%H:%M")) == {"09:31"}
    assert frame.decision_at.iloc[0].utcoffset() != frame.decision_at.iloc[-1].utcoffset()
    assert (frame.decision_at < frame.return_available_at).all()


@pytest.mark.parametrize("case", ["early_asof", "late_asof", "missing_start", "empty_start"])
def test_declared_date_contract_rejects_truncation_and_empty_labels(synthetic_source, case):
    provider, dates, _ = synthetic_source
    start, as_of = str(dates[0].date()), str(dates[-1].date())
    if case == "early_asof":
        as_of = str(dates[-2].date())
    elif case == "late_asof":
        as_of = str((dates[-1] + pd.Timedelta(days=1)).date())
    elif case == "missing_start":
        start = "2024-03-09"  # Not a session in the Qlib provider.
    else:
        start = str(dates[-1].date())
    with pytest.raises(ValueError, match="blocked:"):
        build_relative_dataset(provider_uri=provider, signal_start=start, as_of=as_of)


@pytest.mark.parametrize("case", ["nan", "zero", "negative", "missing_row", "wrong_universe"])
def test_native_provider_missing_or_invalid_prices_never_silently_drop_rows(synthetic_source, tmp_path, case):
    _, dates, source = synthetic_source
    raw = source.copy()
    row = raw.index[(raw.instrument == "SOXL") & (raw.datetime == dates[5])][0]
    if case == "missing_row":
        raw = raw.drop(index=row)
    elif case == "wrong_universe":
        raw.loc[raw.instrument.eq("SOXL"), "instrument"] = "SPXL"
    else:
        raw.loc[row, "open"] = {"nan": np.nan, "zero": 0., "negative": -1.}[case]
    provider = tmp_path / "provider"
    build_qlib_store_from_frame(raw, provider_uri=provider, as_of=str(dates[-1].date()))
    with pytest.raises(ValueError, match="blocked:"):
        build_relative_dataset(provider_uri=provider, signal_start=str(dates[0].date()),
                               as_of=str(dates[-1].date()))


@pytest.mark.parametrize("cost", [10., 25.])
def test_funded_expert_and_learner_paths_reconcile_with_native_qlib_positions(native_market, cost):
    provider, _, all_rows, _ = native_market
    # Leave a provider session after the final exit: Qlib's executor requests
    # a successor calendar entry even though that session is not backtested.
    frame = all_rows.iloc[:-1].copy()
    experts = _experts(frame, cost=cost, rebalance=3)
    learned, _ = _online(frame, cost=cost, rebalance=3, experts=experts)
    for name, replay in {**experts, "learner": learned}.items():
        original = replay.copy(deep=True)
        result = native_open_reconciliation(frame, replay, provider_uri=provider, cost_bps=cost)
        assert result["audit"]["status"] == "passed", name
        assert result["audit"]["maximum_session_return_error"] < 1e-10, name
        assert len(result["positions"]) == len(frame)
        np.testing.assert_allclose(result["reconciliation"].native_next_open_return,
                                   replay.strategy_return, rtol=0, atol=1e-10)
        pd.testing.assert_frame_equal(replay, original, check_exact=True)


def test_nonrebalance_is_true_hold_with_no_cost_and_drifting_weights(native_market):
    _, _, frame, _ = native_market
    fixed = replay_fixed_target(frame, target=[.2, .8, 0.], cost_bps=25., rebalance_sessions=3)
    online, audit = _online(frame, rebalance=3)
    for path in (fixed, online):
        hold = ~path.execute_open_target
        np.testing.assert_array_equal(path.loc[hold, OPEN].to_numpy(), path.loc[hold, PRETRADE].to_numpy())
        np.testing.assert_array_equal(path.loc[hold, ["fees", "turnover"]].to_numpy(), 0.)
        np.testing.assert_array_equal(path[PRETRADE].iloc[1:].to_numpy(), path[ENDING].iloc[:-1].to_numpy())
        assert not np.array_equal(path.loc[hold, OPEN].to_numpy(), path.loc[hold, ENDING].to_numpy())
    np.testing.assert_array_equal(audit.execute_open_target, np.arange(len(frame)) % 3 == 0)
    benchmark = replay_fixed_target(frame, target=[0., 1., 0.], cost_bps=25.,
                                   rebalance_sessions=1, buy_and_hold=True)
    assert benchmark.execute_open_target.sum() == 1
    assert benchmark.fees.iloc[1:].sum() == 0.


def test_learner_pays_for_its_own_funded_trades_not_the_expert_path_average(native_market):
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target

    _, _, frame, _ = native_market
    cost = 125.
    experts = _experts(frame, cost=cost, rebalance=1)
    learned, audit = _online(frame, cost=cost, rebalance=1, experts=experts)
    held = np.array([0., 0., 1.])
    for i, market in enumerate(frame.itertuples()):
        target = learned.loc[i, OPEN].to_numpy(float)
        executable = fee_affordable_target(target, pretrade_weights=held, cost_bps=cost)
        account = settle_target_session(held, open_target_weights=executable,
            asset_returns=np.array([market.soxl_return, market.soxx_return, 0.]), cost_bps=cost)
        np.testing.assert_allclose([learned.strategy_return.iloc[i], learned.fees.iloc[i], learned.turnover.iloc[i]],
            [account["net_return"], account["fees"], account["turnover"]], atol=1e-12, rtol=0)
        held = account["ending_weights"]
    expert_fees = np.column_stack([experts[name].fees for name in _config().expert_names])
    posterior = audit[[f"posterior_{name}" for name in _config().expert_names]].to_numpy()
    incorrectly_averaged_fee = .5 * experts["soxx"].fees.to_numpy() + .5 * (posterior * expert_fees).sum(axis=1)
    assert np.max(np.abs(learned.fees - incorrectly_averaged_fee)) > 1e-5


def test_feedback_is_strictly_delayed_and_future_returns_cannot_change_earlier_targets(native_market):
    _, _, frame, _ = native_market
    original, audit = _online(frame, rebalance=1)
    altered_frame = frame.copy()
    altered_frame.loc[altered_frame.index[5:], ["soxl_return", "soxx_return"]] = [1.5, -.4]
    altered, changed_audit = _online(altered_frame, rebalance=1)
    np.testing.assert_array_equal(audit.feedback_count, np.maximum(np.arange(len(frame)) - 1, 0))
    mature = audit.feedback_count > 0
    assert (audit.loc[mature, "last_feedback_available_at"] < audit.loc[mature, "decision_at"]).all()
    # Outcome 5 matures at session 7 open, after session 6's target decision.
    pd.testing.assert_frame_equal(original.loc[:6, OPEN], altered.loc[:6, OPEN], check_exact=True)
    pd.testing.assert_frame_equal(audit.loc[:6], changed_audit.loc[:6], check_exact=True)
    assert not np.array_equal(original.loc[7:, OPEN].to_numpy(), altered.loc[7:, OPEN].to_numpy())


@pytest.mark.parametrize("case", ["missing_expert", "extra_expert", "wrong_cost", "missing_cost", "wrong_dates", "no_soxx"])
def test_expert_inventory_cost_and_calendar_must_match_exactly(native_market, case):
    _, _, frame, _ = native_market
    experts = _experts(frame)
    config = _config()
    if case == "missing_expert":
        experts.pop("defensive")
    elif case == "extra_expert":
        experts["extra"] = experts["defensive"].copy()
    elif case == "wrong_cost":
        experts["defensive"] = replay_fixed_target(frame, target=[0., .8, .2], cost_bps=10., rebalance_sessions=3)
    elif case == "missing_cost":
        experts["defensive"] = experts["defensive"].drop(columns="cost_bps")
    elif case == "wrong_dates":
        experts["defensive"] = experts["defensive"].iloc[1:].copy()
    else:
        experts["baseline"] = experts.pop("soxx")
        config = replace(config, expert_names=("baseline", "leveraged", "defensive"))
    with pytest.raises(ValueError, match="blocked:"):
        _online(frame, experts=experts, config=config)


@pytest.mark.parametrize("case", ["naive_decision", "naive_available", "future_decision",
    "early_availability", "missing_session", "invalid_return"])
def test_market_chronology_and_complete_outcomes_fail_closed(native_market, case):
    _, _, original, _ = native_market
    frame = original.copy()
    if case == "naive_decision":
        frame["decision_at"] = frame.decision_at.dt.tz_localize(None)
    elif case == "naive_available":
        frame["return_available_at"] = frame.return_available_at.dt.tz_localize(None)
    elif case == "future_decision":
        # Valid ordering versus outcome, but illegally later than target fill.
        frame["decision_at"] = pd.DatetimeIndex(frame.exit_date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
    elif case == "early_availability":
        frame["return_available_at"] = pd.DatetimeIndex(frame.exit_date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=29)
    elif case == "missing_session":
        frame = frame.drop(index=frame.index[4])
    else:
        frame.loc[frame.index[3], "soxl_return"] = np.nan
    with pytest.raises(ValueError, match="blocked:"):
        validate_market(frame)
