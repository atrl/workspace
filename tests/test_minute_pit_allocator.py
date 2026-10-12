from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_completed_minute_paths_respect_half_days_future_changes_and_invalid_source(tmp_path):
    from src.data import QlibProviderConfig, build_qlib_store_from_frame, load_qlib_feature_frame
    from src.data.minute_allocation_features import completed_session_features

    plan = json.loads((Path(__file__).resolve().parents[1] / "research/minute_information_pit_allocator_20261012/plan.json").read_text())
    days = pd.to_datetime(["2025-11-28", "2025-12-01", "2025-12-02"])
    schedule = pd.DataFrame({"open": pd.to_datetime([str(d.date()) + " 14:30Z" for d in days]),
        "close": pd.to_datetime(["2025-11-28 18:00Z", "2025-12-01 21:00Z", "2025-12-02 21:00Z"])}, index=days)
    frames = {}
    for name, symbols, freq, step in [("core", ["SOXL", "SOXX"], "1min", 1),
                                       ("peer", ["QQQ", "SMH", "NVDA", "AMD"], "5min", 5)]:
        rows = []
        for day in days:
            count = (210 if day == days[0] else 390) // step
            for j, symbol in enumerate(symbols):
                r = .0001 + .00006 * np.sin(np.arange(count) * .3 + j)
                p = 100 * np.exp(np.r_[0., np.cumsum(r)])
                for i, stamp in enumerate(pd.date_range(day + pd.Timedelta(hours=9, minutes=30 + step), periods=count, freq=freq)):
                    rows.append(dict(instrument=symbol, datetime=stamp, open=p[i], close=p[i + 1],
                        high=max(p[i], p[i + 1]) + .01, low=min(p[i], p[i + 1]) - .01,
                        volume=100 + i, price_valid=1.))
        frame = pd.DataFrame(rows)
        provider = tmp_path / name
        build_qlib_store_from_frame(frame, provider_uri=provider, freq=freq, as_of="2025-12-02",
            field_map={"$" + n: n for n in ["open", "high", "low", "close", "volume", "price_valid"]})
        frames[name] = load_qlib_feature_frame(QlibProviderConfig(provider, freq=freq), instruments=symbols,
            fields=["$" + n for n in ["open", "high", "low", "close", "volume", "price_valid"]],
            start="2025-11-28 09:31", end="2025-12-02 16:00")
    arguments = dict(schedule=schedule, core_names=plan["feature_groups"]["core"],
                     peer_names=plan["feature_groups"]["peer"], peer_symbols=["QQQ", "SMH", "NVDA", "AMD"])
    original, coverage = completed_session_features(frames["core"], frames["peer"], **arguments)
    assert coverage.expected_1m.tolist() == [210, 390, 390]
    assert original.loc[days[0], ["minute_core_ready", "minute_peer_ready"]].eq(1).all()
    changed = frames["core"].copy()
    clock = changed.index.get_level_values("datetime")
    future = clock >= days[2] + pd.Timedelta(hours=12)
    changed.loc[future, ["$open", "$high", "$low", "$close"]] *= 1.2
    after, _ = completed_session_features(changed, frames["peer"], **arguments)
    pd.testing.assert_frame_equal(original.loc[:days[1]], after.loc[:days[1]])
    bad = ("SOXX", days[1] + pd.Timedelta(hours=9, minutes=31))
    changed.loc[bad, "$price_valid"] = 0.
    blocked, _ = completed_session_features(changed, frames["peer"], **arguments)
    assert len(blocked) == len(days) and blocked.loc[days[1], "minute_core_ready"] == 0
    assert blocked.loc[days[1], plan["feature_groups"]["core"]].isna().all()


def test_native_pit_model_ignores_unmatured_labels_and_neutrally_encodes_missing_minutes(tmp_path):
    from src.workflow.minute_pit_allocator import _dataset, _auxiliary_dataset
    from src.model.minute_pit_allocator import MinutePitAllocator, TARGETS
    from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES

    days = pd.bdate_range("2018-01-02", periods=600)
    features = pd.DataFrame({"past_state": np.sin(np.arange(600) * .1), "minute_core_ready": np.r_[np.zeros(200), np.ones(400)],
                            "minute_peer_ready": np.r_[np.zeros(200), np.ones(400)],
                            "minute_risk": np.r_[np.full(200, np.nan), np.linspace(.1, .9, 400)]}, index=days)
    labels = pd.DataFrame({TARGETS[0]: .001 * np.sin(np.arange(600)), TARGETS[1]: .0003 * np.cos(np.arange(600)),
                          TARGETS[2]: (np.arange(600) % 5 == 0).astype(float), TARGETS[3]: .2,
                          TARGETS[4]: .01 * np.sin(np.arange(600))}, index=days)
    common = pd.DataFrame({n: np.sin(np.arange(600) * .02 + j) for j, n in enumerate(COMMON_FEATURES)}, index=days)
    metadata = pd.DataFrame({"label_available_at": (days + pd.offsets.BDay(6)).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)}, index=days)
    auxiliary = common.iloc[:50].reset_index(drop=True)
    auxiliary["datetime"] = days[:50]; auxiliary["instrument"] = "TEST"
    auxiliary["$pit_available"] = 1.; auxiliary["training_eligible"] = True
    auxiliary["synthetic_2x_crash_label"] = 0.; auxiliary["synthetic_2x_upside_label"] = 1.; auxiliary["training_weight"] = 1.
    aux_dataset = _auxiliary_dataset(auxiliary, "2019-12-02")
    args = dict(base_features=["past_state"], extra_features=["minute_core_ready", "minute_peer_ready", "minute_risk"],
        feature_masks={"minute_risk": "minute_core_ready"}, spec={"root": str(tmp_path), "epochs": 2}, seed=71,
        cutoff="2020-01-01T00:00:00-05:00", train_end="2019-12-02", minimum_observations=20)
    dataset = _dataset(features, labels, common, metadata, {"train": ("2018-01-02", "2020-04-01"), "predict": ("2018-01-02", "2018-01-31")})
    first = MinutePitAllocator(**args).fit(dataset, auxiliary_dataset=aux_dataset)
    corrupted = labels.copy()
    corrupted.loc[metadata.label_available_at >= pd.Timestamp(args["cutoff"])] = 1e6
    altered = _dataset(features, corrupted, common, metadata, dataset.segments)
    second = MinutePitAllocator(**args).fit(altered, auxiliary_dataset=aux_dataset)
    pd.testing.assert_frame_equal(first.predict(dataset), second.predict(dataset))
    early = dataset.prepare("predict", col_set="feature")
    assert first._extra_tensor(early).eq(0).all()
    assert pd.Timestamp(first.audit["last_mature_label_at"]) < pd.Timestamp(args["cutoff"])


def test_native_allocator_accounts_preserve_hold_and_separate_cost_state(tmp_path):
    from src.data import build_qlib_store_from_frame
    from src.backtest.minute_allocator import native_allocator_book

    dates = pd.bdate_range("2025-04-01", periods=9)
    prices = {"SOXL": 100 * np.cumprod([1., 1.01, 1.005, 1., .998, 1.002, 1.001, 1., 1.]),
              "SOXX": 200 * np.cumprod([1., 1.002, 1.001, 1., .999, 1.001, 1., 1., 1.])}
    rows = [dict(instrument=s, datetime=d, open=prices[s][i], close=prices[s][i] * 1.001)
            for s in prices for i, d in enumerate(dates)]
    provider = tmp_path / "native"
    build_qlib_store_from_frame(pd.DataFrame(rows), provider_uri=provider, as_of=str(dates[-1].date()),
                               field_map={"$open": "open", "$close": "close"})
    # Returns use Qlib's float32-stored marks, matching the account authority.
    from src.data import QlibProviderConfig, load_qlib_feature_frame
    native = load_qlib_feature_frame(QlibProviderConfig(provider), instruments=list(prices), fields=["$open"],
        start=str(dates[0].date()), end=str(dates[-1].date()))["$open"].unstack("instrument").astype(float)
    market = pd.DataFrame({"signal_date": dates[:-2], "trade_date": dates[1:-1], "exit_date": dates[2:]})
    market["decision_at"] = pd.DatetimeIndex(market.trade_date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=29)
    market["return_available_at"] = pd.DatetimeIndex(market.exit_date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
    for s in prices:
        market[s.lower() + "_return"] = native[s].reindex(dates[2:]).to_numpy() / native[s].reindex(dates[1:-1]).to_numpy() - 1
    targets = pd.DataFrame({"soxl_weight": .2, "soxx_weight": .5, "cash_weight": .3}, index=dates[:-2])
    books = [native_allocator_book(market, targets, provider_uri=provider, cost_bps=fee,
        smoothing_span=3, rebalance_threshold=.02) for fee in [10, 25]]
    for book in books:
        assert book["native"]["audit"]["status"] == "passed"
        held = ~book["authority"].execute_open_target
        assert held.any() and book["authority"].loc[held, "fees"].eq(0).all()
    assert books[0]["authority"].fees.sum() < books[1]["authority"].fees.sum()
