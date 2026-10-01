from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import stage1_regime_validation as validation


def _bars():
    rows = []
    for symbol in validation.SYMBOLS:
        for time in ("09:45", "10:00", "10:15", "10:30", "10:45"):
            rows.append({"instrument": symbol, "datetime": pd.Timestamp(f"2026-09-24 {time}"),
                         "$open": 100.0, "$high": 102.0, "$low": 98.0, "$close": 101.0,
                         "$volume": 1000.0, "$bar_count": 26.0})
    return pd.DataFrame(rows)


def _retained():
    row = {"signal_date": pd.Timestamp("2026-09-23"), "trade_date": pd.Timestamp("2026-09-24")}
    for symbol in validation.SYMBOLS:
        for name, value in zip(validation.EARLY_FIELDS, [101 / 100 - 1, 98 / 102 - 1, 101 / 98 - 1, 4000., 104.]):
            row[f"{name}_{symbol.lower()}"] = value
    return pd.DataFrame([row])


def _provider(tmp_path):
    (tmp_path / "qlib_store_manifest.json").write_text('{"as_of":"2026-09-25"}')
    return tmp_path


def test_first_hour_audit_preserves_original_and_ignores_later_bars(monkeypatch, tmp_path):
    bars, retained = _bars(), _retained()
    source = retained.copy(deep=True)
    monkeypatch.setattr(validation, "_read_first_hour_bars", lambda *args, **kwargs: bars.copy())
    result, audit, summary = validation.audit_first_hour_inputs(retained, provider_uri=_provider(tmp_path))
    assert summary["status"] == "passed"
    assert summary["valid_recovery_rows"] == 1
    assert summary["maximum_retained_feature_error"] == 0
    assert audit.iloc[0].feature_available_at == "2026-09-24T14:30:00+00:00"
    assert audit.iloc[0].recovery_decision_at == "2026-09-24T14:31:00+00:00"
    pd.testing.assert_frame_equal(source, result)
    bars.loc[bars.datetime.dt.strftime("%H:%M").eq("10:45"), "$close"] = 9000.
    later, _, _ = validation.audit_first_hour_inputs(retained, provider_uri=tmp_path)
    pd.testing.assert_frame_equal(result, later)
    pd.testing.assert_frame_equal(source, retained)


def test_missing_early_bar_cannot_be_replaced_with_late_bar(monkeypatch, tmp_path):
    bars = _bars().loc[lambda f: ~((f.instrument == "SOXL") & f.datetime.dt.strftime("%H:%M").eq("10:00"))]
    retained = _retained()
    monkeypatch.setattr(validation, "_read_first_hour_bars", lambda *args, **kwargs: bars.copy())
    result, audit, summary = validation.audit_first_hour_inputs(retained, provider_uri=_provider(tmp_path))
    assert summary["invalid_retained_recovery_rows"] == 1
    assert "09:45,10:15,10:30,10:45" in audit.iloc[0].reasons
    assert result.filter(like="early_").isna().all().all()
    assert retained.filter(like="early_").notna().all().all()


def test_retained_source_mismatch_is_disabled(monkeypatch, tmp_path):
    retained = _retained()
    retained.loc[0, "early_ret_soxl"] += .001
    monkeypatch.setattr(validation, "_read_first_hour_bars", lambda *args, **kwargs: _bars())
    result, audit, summary = validation.audit_first_hour_inputs(retained, provider_uri=_provider(tmp_path))
    assert summary["invalid_retained_recovery_rows"] == 1
    assert "retained_feature_mismatch" in audit.iloc[0].reasons
    assert result.filter(like="early_").isna().all().all()


def _native_inputs():
    import qlib
    from qlib.data import D
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target

    provider = Path(__file__).resolve().parents[1] / "data/qlib/soxl_soxx_soxs"
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    prices = D.features(list(validation.SYMBOLS), ["$open"], start_time="2016-01-04", end_time="2016-01-08", freq="day")
    prices = prices["$open"].unstack("instrument").astype(float)
    returns = prices.shift(-1) / prices - 1
    holdings = np.array([0., 0., 1.])
    rows, replay = [], []
    for signal, trade, exit_day, target, execute in [
        ("2016-01-04", "2016-01-05", "2016-01-06", [0.2, 0.5, 0.3], True),
        ("2016-01-05", "2016-01-06", "2016-01-07", [0., 0., 0.], False),
        ("2016-01-06", "2016-01-07", "2016-01-08", [0.1, 0.6, 0.3], True),
    ]:
        asset_returns = np.r_[returns.loc[trade, list(validation.SYMBOLS)].to_numpy(), 0.]
        desired = np.array(target) if execute else holdings.copy()
        execution = fee_affordable_target(desired, pretrade_weights=holdings, cost_bps=25.) if execute else None
        settled = settle_target_session(holdings, open_target_weights=execution, asset_returns=asset_returns, cost_bps=25.)
        row = {"signal_date": pd.Timestamp(signal), "trade_date": pd.Timestamp(trade), "exit_date": pd.Timestamp(exit_day),
               "soxl_return": asset_returns[0], "soxx_return": asset_returns[1]}
        rows.append(row)
        replay.append({**row, "recovery_action": False, "execute_open_target": execute,
                       "open_soxl_weight": desired[0], "open_soxx_weight": desired[1], "open_cash_weight": desired[2],
                       "strategy_return": settled["net_return"]})
        holdings = settled["ending_weights"]
    return provider, pd.DataFrame(rows), pd.DataFrame(replay)


def test_native_qlib_report_reconciles_open_positions_costs_and_hold():
    provider, frame, replay = _native_inputs()
    native = validation.native_open_reconciliation(frame, replay, provider_uri=provider, cost_bps=25.)
    assert native["audit"]["status"] == "passed"
    assert native["audit"]["maximum_session_return_error"] < 1e-12
    assert len(native["report"]) == len(native["positions"]) == 3
    assert native["report"].iloc[1]["cost"] == 0
    assert native["report"].iloc[0]["cost"] > 0
    assert not native["risk_analysis"].empty


def test_native_daily_reconciliation_rejects_intraday_and_source_mismatch():
    provider, frame, replay = _native_inputs()
    replay.loc[0, "recovery_action"] = True
    with pytest.raises(ValueError, match="cannot represent intraday"):
        validation.native_open_reconciliation(frame, replay, provider_uri=provider)
    replay.loc[0, "recovery_action"] = False
    frame.loc[0, "soxl_return"] += .01
    with pytest.raises(ValueError, match="returns disagree"):
        validation.native_open_reconciliation(frame, replay, provider_uri=provider)
