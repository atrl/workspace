"""Meaningful independent native-accounting and causal-order gates."""
import numpy as np
import pandas as pd
import pytest

from src.data.qlib_native import build_qlib_store_from_frame
from src.workflow.legacy_soxl_repair import native_order_backtest, reconcile_engine, path_metrics, bootstrap_excess


@pytest.fixture
def provider(tmp_path):
    dates = pd.bdate_range("2025-01-01", periods=8)
    rows = []
    for symbol in ("SOXL", "SOXX"):
        for index, date in enumerate(dates):
            price = 100. + index
            rows.append({"datetime": date, "instrument": symbol, "open": price,
                         "close": price + 1, "high": price + 2, "low": price - 1,
                         "volume": 1e7, "factor": 1.})
    build_qlib_store_from_frame(pd.DataFrame(rows), provider_uri=tmp_path / "provider",
        as_of="2025-01-10", source="synthetic native accounting fixture")
    return dates


def test_native_full_ledger_matches_integer_cash_fees_and_close_marks(provider):
    dates = provider[1:5]
    trades = pd.DataFrame([
        {"date": dates[0], "signal_date": provider[0], "symbol": "SOXL", "amount": 4,
         "price": 101., "fee": .404},
        {"date": dates[2], "signal_date": dates[1], "symbol": "SOXL", "amount": -2,
         "price": 103., "fee": .206},
    ])
    native = native_order_backtest(dates=dates, cost_bps=10, initial_cash=1000, trades=trades)
    expected = []
    cash, shares, previous = 1000., 0, 1000.
    for index, date in enumerate(dates):
        amount, fee = (4, .404) if index == 0 else ((-2, .206) if index == 2 else (0, 0.))
        price = 101. + index
        cash -= amount * price + fee
        shares += amount
        nav = cash + shares * (price + 1)
        expected.append({"date": date, "account": nav, "cash": cash, "shares": shares,
                         "fee": fee, "net_return": nav / previous - 1})
        previous = nav
    assert reconcile_engine(pd.DataFrame(expected), native)["status"] == "passed"
    assert native["executed_orders"].amount.tolist() == [4, -2]
    wrong = pd.DataFrame(expected)
    wrong.loc[1, "cash"] += 1
    with pytest.raises(ValueError, match="native cash"):
        reconcile_engine(wrong, native)


def test_order_causality_and_native_clipping_fail_closed(provider):
    trades = pd.DataFrame([{"date": provider[1], "signal_date": provider[1], "symbol": "SOXL",
                           "amount": 1, "price": 101., "fee": .101}])
    with pytest.raises(ValueError, match="prior-signal"):
        native_order_backtest(dates=provider[1:3], cost_bps=10, initial_cash=1000, trades=trades)
    trades.loc[0, "signal_date"] = provider[0]
    trades.loc[0, "amount"] = 100
    with pytest.raises(ValueError, match="clipped"):
        native_order_backtest(dates=provider[1:3], cost_bps=10, initial_cash=1000, trades=trades)


def test_buyhold_initial_fee_and_no_rebalance(provider):
    native = native_order_backtest(dates=provider[1:5], cost_bps=25, initial_cash=1000, control="soxx_buy_hold")
    assert len(native["executed_orders"]) == 1
    assert native["executed_orders"].iloc[0].amount == 9
    assert native["daily"].fee.iloc[0] == pytest.approx(909 * .0025)
    assert native["daily"].fee.iloc[1:].sum() == 0
    assert native["daily"].soxx_shares.eq(9).all()


def test_initial_capital_anchored_drawdown():
    daily = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=3),
        "net_return": [-.1, 0., 0.], "exposure": [1., 1., 1.], "fee": [0., 0., 0.],
        "turnover": [1., 0., 0.], "account": [90., 90., 90.]})
    assert path_metrics(daily)["max_drawdown_pct"] == pytest.approx(-10)


def test_bootstrap_never_inner_joins_missing_dates():
    frame = pd.DataFrame({"net_return": np.zeros(150)}, index=pd.bdate_range("2025-01-01", periods=150))
    with pytest.raises(ValueError, match="common-date"):
        bootstrap_excess(frame, frame.iloc[1:], mean_block=20, resamples=200)


def test_cached_recorder_requires_finished_summary_and_artifact_bytes(tmp_path):
    import hashlib
    from src.workflow.legacy_soxl_repair import _verify_cached_recorder
    artifact = tmp_path / "artifacts" / "audit"
    artifact.mkdir(parents=True)
    path = artifact / "evidence.pkl"
    path.write_bytes(b"original recorded evidence")
    manifest = {"audit/evidence.pkl": hashlib.sha256(path.read_bytes()).hexdigest()}
    summary = {"recorder_artifact_sha256": manifest}
    class Recorder:
        status = "FINISHED"
        def load_object(self, name):
            return summary if name == "audit/summary.pkl" else manifest
        def get_local_dir(self):
            return str(tmp_path)
    recorder = Recorder()
    _verify_cached_recorder(summary, recorder)
    recorder.status = "FAILED"
    with pytest.raises(ValueError, match="FINISHED"):
        _verify_cached_recorder(summary, recorder)
    recorder.status = "FINISHED"
    path.write_bytes(b"changed recorded evidence")
    with pytest.raises(ValueError, match="artifact hash changed"):
        _verify_cached_recorder(summary, recorder)


def test_authorization_binds_plan_source_and_vix(tmp_path, monkeypatch):
    import hashlib, json
    import src.workflow.legacy_soxl_repair as workflow
    plan = {"deadline_utc": "2026-10-05T02:19:11Z"}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    hashes = {"engine.py": "fixed-source"}
    monkeypatch.setattr(workflow, "_source_hashes", lambda root: hashes)
    audit = {"vix": {"snapshot_manifest_sha256": "frozen-vix"}, "execution": {"execution_manifest_sha256": "frozen-execution"}}
    auth = {"authorized": True, "deadline_utc": plan["deadline_utc"], "source_sha256": hashes,
            "plan_file_sha256": hashlib.sha256((tmp_path / "plan.json").read_bytes()).hexdigest(),
            "vix_snapshot_manifest_sha256": "frozen-vix", "execution_manifest_sha256": "frozen-execution"}
    with pytest.raises(ValueError, match="authorization is absent"):
        workflow._authorization(tmp_path, tmp_path, plan, audit)
    (tmp_path / "run_authorization.json").write_text(json.dumps(auth))
    workflow._authorization(tmp_path, tmp_path, plan, audit)
    audit["vix"]["snapshot_manifest_sha256"] = "different-vix"
    with pytest.raises(ValueError, match="authorization seal differs"):
        workflow._authorization(tmp_path, tmp_path, plan, audit)


def test_future_calendar_boundary_keeps_last_observed_close(provider, tmp_path):
    import qlib
    future = provider.append(pd.DatetimeIndex([provider[-1] + pd.offsets.BDay()]))
    (tmp_path / "provider/calendars/day_future.txt").write_text("\n".join(future.strftime("%Y-%m-%d")) + "\n")
    qlib.init(provider_uri=str(tmp_path / "provider"), region="us", kernels=1)
    native = native_order_backtest(dates=provider[-2:], cost_bps=10, initial_cash=1000, control="soxx_buy_hold")
    assert native["daily"].index[-1] == provider[-1]
    assert len(native["daily"]) == 2
    assert len(native["executed_orders"]) == 1
