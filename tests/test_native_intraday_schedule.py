from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.native_intraday_schedule import native_intraday_reconciliation
from src.backtest.position_accounting import settle_target_session
from src.data.native_execution_marks import build_coherent_daily_view, build_execution_mark_view, build_uniform_intraday_view, session_price_marks, validate_complete_minute_sessions
from src.data.qlib_native import build_qlib_store_from_frame
from src.workflow.stage1_economic_action import fee_affordable_target


@pytest.fixture
def observed_prices(tmp_path):
    days = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    rows = []
    for symbol, starts in [("SOXL", [100., 103., 102.]), ("SOXX", [200., 201., 199.])]:
        for day, opening in zip(days, starts):
            previous = opening
            for j, stamp in enumerate(pd.date_range(day + pd.Timedelta(hours=9, minutes=45),
                                                    day + pd.Timedelta(hours=16), freq="15min")):
                closing = opening * (1 + .0005 * (j + 1))
                rows.append({"instrument": symbol, "datetime": stamp, "open": previous,
                             "high": max(previous, closing), "low": min(previous, closing),
                             "close": closing, "volume": 100_000.})
                previous = closing
    source, view = tmp_path / "source", tmp_path / "view"
    fields = {key: key for key in ["open", "high", "low", "close", "volume"]}
    build_qlib_store_from_frame(pd.DataFrame(rows), provider_uri=source, field_map=fields,
                               freq="15min", as_of="2020-01-06", source="deterministic test quotes")
    manifest = build_execution_mark_view(source_provider=source, provider_uri=view,
                                        start="2020-01-02", end="2020-01-06 16:00:00", as_of="2020-01-06")
    sessions = pd.DataFrame({"signal_date": pd.to_datetime(["2019-12-31", "2020-01-02"]),
                             "trade_date": days[:2], "exit_date": days[1:]})
    marks = session_price_marks(provider_uri=view, sessions=sessions)
    return source, view, manifest, sessions, marks


def _schedule(sessions, marks, *, cost_bps, inherited=False, target_convention="post_fee_desired"):
    held = np.array([.1, .4, .5]) if inherited else np.array([0., 0., 1.])
    rows = []
    for i, row in sessions.iterrows():
        desired_open = np.array([.3, .5, .2]) if i == 0 else held.copy()
        recovery = np.array([.4, .4, .2]) if i == 0 else None
        early = marks.loc[i, ["early_ret_soxl", "early_ret_soxx"]].to_numpy(float)
        daily = marks.loc[i, ["soxl_return", "soxx_return"]].to_numpy(float)
        opening = (fee_affordable_target(desired_open, pretrade_weights=held, cost_bps=cost_bps)
                   if target_convention == "post_fee_desired" else desired_open) if i == 0 else None
        if recovery is not None:
            opened = settle_target_session(held, open_target_weights=opening, asset_returns=[0., 0., 0.], cost_bps=cost_bps)
            cut_values = opened["open_net_weights"] * (1 + np.r_[early, 0.])
            at_cut = cut_values / cut_values.sum()
            recovered = (fee_affordable_target(recovery, pretrade_weights=at_cut, cost_bps=cost_bps)
                         if target_convention == "post_fee_desired" else recovery)
        else:
            recovered = None
        settled = settle_target_session(held, open_target_weights=opening, asset_returns=np.r_[daily, 0.],
                                        recovery_target_weights=recovered,
                                        early_returns=np.r_[early, 0.] if recovery is not None else None,
                                        cost_bps=cost_bps)
        end = settled["ending_weights"]
        displayed = recovery if recovery is not None else held
        rows.append({**row.to_dict(), "execute_open_target": i == 0, "recovery_action": i == 0,
                     "open_soxl_weight": desired_open[0], "open_soxx_weight": desired_open[1], "open_cash_weight": desired_open[2],
                     "soxl_weight": displayed[0], "soxx_weight": displayed[1], "cash_weight": displayed[2],
                     "strategy_return": settled["net_return"], "ending_soxl_weight": end[0],
                     "ending_soxx_weight": end[1], "ending_cash_weight": end[2]})
        held = end
    return pd.DataFrame(rows)


def test_full_day_native_book_keeps_unknown_sessions_and_exact_hold(observed_prices):
    from src.backtest.full_day_minute_policy import run_full_day_native_book

    source, _, _, sessions, _ = observed_prices
    targets = sessions[["trade_date"]].assign(soxl_weight=.3, soxx_weight=.5, cash_weight=.2)
    scores = pd.DataFrame({"entry_at": [pd.Timestamp("2020-01-02 09:45"), pd.Timestamp("2020-01-03 10:45")],
                          "pred_soxl": [.04, 0.], "pred_soxx": [.01, 0.]})
    result = run_full_day_native_book(provider_uri=source, parent_targets=targets, predictions=scores,
        cost_bps=25., start="2020-01-02", end="2020-01-03", reference="model")
    assert len(result["daily_net"]) == 2
    assert result["report"].cost.sum() > 0
    assert len(result["events"]) == 1
    assert len(result["positions"]) == len(result["report"])


@pytest.mark.parametrize("cost_bps", [10., 25.])
@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("target_convention", ["post_fee_desired", "pre_fee_account"])
def test_native_two_slot_fees_hold_and_quantities_reconcile(observed_prices, cost_bps, inherited, target_convention):
    _, view, manifest, sessions, marks = observed_prices
    assert manifest["source_ohlcv_unchanged"] is True
    replay = _schedule(sessions, marks, cost_bps=cost_bps, inherited=inherited, target_convention=target_convention)
    market = sessions.join(marks[["soxl_return", "soxx_return", "early_ret_soxl", "early_ret_soxx"]])
    result = native_intraday_reconciliation(market, replay, provider_uri=view, cost_bps=cost_bps,
                                            initial_weights=np.array([.1, .4, .5]) if inherited else None,
                                            target_convention=target_convention)
    assert result["audit"]["status"] == "passed"
    assert result["audit"]["maximum_session_return_error"] < 1e-7
    assert result["audit"]["maximum_ending_weight_error"] < 1e-7
    assert result["events"].slot.tolist() == ["open", "recovery"]
    assert result["events"].calendar_at.dt.strftime("%H:%M").tolist() == ["09:45", "10:30"]
    assert result["audit"]["live_fill_validation"] is False
    assert result["audit"]["target_convention"] == target_convention


def test_different_price_source_is_rejected_before_backtest(observed_prices):
    _, view, _, sessions, marks = observed_prices
    replay = _schedule(sessions, marks, cost_bps=10.)
    market = sessions.join(marks[["soxl_return", "soxx_return", "early_ret_soxl", "early_ret_soxx"]])
    market.loc[0, "soxl_return"] += .001
    with pytest.raises(ValueError, match="differs from native price marks"):
        native_intraday_reconciliation(market, replay, provider_uri=view, cost_bps=10.)


def test_source_view_cannot_overwrite_source(observed_prices):
    source, _, _, _, _ = observed_prices
    with pytest.raises(ValueError, match="overwrite its source"):
        build_execution_mark_view(source_provider=source, provider_uri=source,
                                  start="2020-01-02", end="2020-01-06", as_of="2020-01-06")


def test_coherent_daily_prefix_and_intraday_transition_share_actual_open(observed_prices, tmp_path):
    source, _, _, _, _ = observed_prices
    rows = []
    for symbol, value in [("SOXL", 90.), ("SOXX", 190.)]:
        for day in pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-03", "2020-01-06"]):
            rows.append({"instrument": symbol, "datetime": day, "open": value,
                         "high": value + 1, "low": value - 1, "close": value, "volume": 1_000.})
    daily = tmp_path / "daily"
    build_qlib_store_from_frame(pd.DataFrame(rows), provider_uri=daily, as_of="2020-01-06",
                               field_map={k: k for k in ["open", "high", "low", "close", "volume"]})
    manifest = build_coherent_daily_view(daily_provider=daily, intraday_provider=source,
                                         provider_uri=tmp_path / "coherent", start="2019-12-31",
                                         end="2020-01-06", as_of="2020-01-06")
    assert manifest["daily_prefix_rows"] == 2
    assert manifest["intraday_price_source_from"] == "2020-01-02"
    from qlib.data import D
    prices = D.features(["SOXL"], ["$open"], start_time="2019-12-31", end_time="2020-01-06", freq="day")
    assert prices.loc[("SOXL", pd.Timestamp("2019-12-31")), "$open"] == 90.
    assert prices.loc[("SOXL", pd.Timestamp("2020-01-02")), "$open"] == 100.


@pytest.mark.parametrize("kind,close", [("WHOLE", "16:00"), ("MORNING", "13:00")])
def test_partial_and_duplicate_minutes_cannot_masquerade_as_full_sessions(kind, close):
    stamps = pd.date_range("2020-01-02 09:31", f"2020-01-02 {close}", freq="1min", tz="America/New_York")
    rows = [{"symbol": symbol, "date": "2020-01-02", "timestamp": stamp,
             "open": 10., "high": 11., "low": 9., "close": 10.5, "volume": 1.}
            for symbol in ["SOXL", "SOXX"] for stamp in stamps]
    full = pd.DataFrame(rows)
    validate_complete_minute_sessions(full, {"2020-01-02": kind})
    with pytest.raises(ValueError, match="partial or duplicate"):
        validate_complete_minute_sessions(full.iloc[1:], {"2020-01-02": kind})
    with pytest.raises(ValueError, match="partial or duplicate"):
        validate_complete_minute_sessions(pd.concat([full, full.iloc[[0]]]), {"2020-01-02": kind})


def test_uniform_vintage_validates_actual_bar_grid_despite_correct_declared_hash(observed_prices, tmp_path):
    import hashlib
    import json
    from qlib.data import D
    import qlib

    source, _, _, _, _ = observed_prices
    qlib.init(provider_uri=str(source), region="us", kernels=1)
    bars = D.features(["SOXL", "SOXX"], ["$open", "$high", "$low", "$close", "$volume"],
                      start_time="2020-01-02", end_time="2020-01-06 16:00:00", freq="15min").reset_index()
    folder = tmp_path / "uniform_receipt"
    folder.mkdir()
    entries = []
    for symbol, block in bars.groupby("instrument"):
        raw = block.rename(columns={f"${key}": key for key in ["open", "high", "low", "close", "volume"]})
        raw["code"] = f"US.{symbol}"
        raw["time_key"] = raw.datetime.dt.strftime("%Y-%m-%d %H:%M:%S")
        path = folder / f"{symbol}.parquet"
        raw.to_parquet(path, index=False)
        entries.append({"symbol": symbol, "path": path.name, "ktype": "K_15M", "adjustment": "QFQ",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    receipt = folder / "source_manifest.json"
    receipt.write_text(json.dumps({"entries": entries}))
    calendar = folder / "calendar.json"
    calendar.write_text(json.dumps({"days": [{"time": day, "trade_date_type": "WHOLE"}
                                               for day in ["2020-01-02", "2020-01-03", "2020-01-06"]]}))
    result = build_uniform_intraday_view(source_manifest=receipt, trading_calendar=calendar,
                                         provider_uri=tmp_path / "uniform", start="2020-01-02", as_of="2020-01-06")
    assert result["single_source_vintage"] is True
    path = folder / entries[0]["path"]
    raw = pd.read_parquet(path).iloc[1:]
    raw.to_parquet(path, index=False)
    entries[0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt.write_text(json.dumps({"entries": entries}))
    with pytest.raises(ValueError, match="incomplete uniform 15min session"):
        build_uniform_intraday_view(source_manifest=receipt, trading_calendar=calendar,
                                     provider_uri=tmp_path / "invalid_uniform", start="2020-01-02", as_of="2020-01-06")


@pytest.mark.parametrize("cost_bps", [10., 25.])
def test_legacy_day_targets_pay_fees_from_residual_cash_not_from_stock_targets(observed_prices, tmp_path, cost_bps):
    _, _, _, sessions, _ = observed_prices
    rows = []
    for symbol, prices in [("SOXL", [100., 103., 102.]), ("SOXX", [200., 201., 199.])]:
        for day, price in zip(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]), prices):
            rows.append({"instrument": symbol, "datetime": day, "open": price, "high": price,
                         "low": price, "close": price, "volume": 100_000.})
    provider = tmp_path / "legacy_daily"
    build_qlib_store_from_frame(pd.DataFrame(rows), provider_uri=provider, as_of="2020-01-06",
                               field_map={key: key for key in ["open", "high", "low", "close", "volume"]})
    marks = session_price_marks(provider_uri=provider, sessions=sessions, freq="day")
    cash = 20_000. - 80_000. * cost_bps / 10_000.
    expected_navs = [300. * 103. + 250. * 201. + cash, 300. * 102. + 250. * 199. + cash]
    previous = 100_000.
    recorded = []
    for i, row in sessions.iterrows():
        nav = expected_navs[i]
        ending = np.array([300. * marks.iloc[i].soxl_exit_open, 250. * marks.iloc[i].soxx_exit_open, cash]) / nav
        recorded.append({**row.to_dict(), "execute_open_target": i == 0, "recovery_action": False,
                         "open_soxl_weight": .3, "open_soxx_weight": .5, "open_cash_weight": .2,
                         "soxl_weight": .3, "soxx_weight": .5, "cash_weight": .2,
                         "strategy_return": nav / previous - 1., "ending_soxl_weight": ending[0],
                         "ending_soxx_weight": ending[1], "ending_cash_weight": ending[2]})
        previous = nav
    market = sessions.join(marks[["soxl_return", "soxx_return"]])
    result = native_intraday_reconciliation(market, pd.DataFrame(recorded), provider_uri=provider,
                                            cost_bps=cost_bps, freq="day", target_convention="pre_fee_account")
    assert result["reconciliation"].native_nav.to_list() == pytest.approx(expected_navs)
    position = result["positions"][pd.Timestamp("2020-01-03")]
    assert position.get_stock_amount("SOXL") == pytest.approx(300.)
    assert position.get_stock_amount("SOXX") == pytest.approx(250.)
    assert position.get_cash() == pytest.approx(cash)
