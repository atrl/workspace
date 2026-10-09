import numpy as np
import pandas as pd
import pytest

from src.analysis import tail_risk_control as control


def test_control_engages_after_loss_and_trims_only_risk():
    raw_beta = np.full(20, 1.0)
    # A sharp loss in sessions 5-9 should engage the control afterwards.
    rets = np.zeros(20)
    rets[5:10] = -0.03
    ctrl = control.TailControl("c", trigger=-0.08, release=-0.02, lookback=5, trim=0.5)
    caps = control.apply_control(raw_beta, rets, ctrl)
    # Before the trailing loss crosses the trigger the cap is unchanged.
    assert np.allclose(caps[:8], 1.0)
    # Once the trailing 5-session loss crosses -8% the cap is trimmed.
    assert caps[8] == pytest.approx(0.5)
    assert caps[12] == pytest.approx(0.5)
    # The control never raises beta.
    assert (caps <= raw_beta + 1e-12).all()


def test_control_releases_after_recovery():
    raw_beta = np.full(30, 1.0)
    rets = np.zeros(30)
    rets[5:10] = -0.03           # engage
    rets[10:20] = 0.02           # recover above release
    ctrl = control.TailControl("c", trigger=-0.08, release=-0.02, lookback=5, trim=0.5)
    caps = control.apply_control(raw_beta, rets, ctrl)
    assert caps[12] == pytest.approx(0.5)      # engaged
    assert caps[29] == pytest.approx(1.0)      # released


def test_trim_one_disables_the_control():
    raw_beta = np.full(10, 1.0)
    rets = np.full(10, -0.1)
    caps = control.apply_control(raw_beta, rets, control.TailControl("off", trim=1.0))
    assert np.allclose(caps, raw_beta)


def test_validate_rejects_bad_params():
    with pytest.raises(ValueError, match="lookback"):
        control.TailControl("c", lookback=1).validate()
    with pytest.raises(ValueError, match="trim"):
        control.TailControl("c", trim=1.5).validate()
    with pytest.raises(ValueError, match="trigger"):
        control.TailControl("c", trigger=0.0, release=-0.02).validate()


def test_gate_pass_reports_each_floor():
    metrics = {"max_drawdown_pct": -28.0, "worst_day_pct": -9.0,
               "cvar5_pct": -5.0, "worst_rolling_5d_pct": -12.11}
    result = control.gate_pass(metrics)
    assert result["pass"] is False
    assert result["checks"]["worst_rolling_5d_pct"] is False
    assert result["checks"]["max_drawdown_pct"] is True
    better = dict(metrics, worst_rolling_5d_pct=-11.9)
    assert control.gate_pass(better)["pass"] is True


def test_replay_returns_gate_columns():
    dates = pd.bdate_range("2024-01-02", periods=120)
    raw = pd.DataFrame({"soxl_weight": 0.2, "soxx_weight": 0.3}, index=dates)
    rng = np.random.default_rng(2)
    returns = pd.DataFrame({"soxl": rng.normal(0, 0.02, 120), "soxx": rng.normal(0, 0.01, 120)}, index=dates)
    table = control.replay(raw, returns, {"none": control.TailControl("none", trim=1.0),
                                          "trim": control.TailControl("trim", trim=0.5)}, cost_bps=10.0)
    assert set(table["control"]) == {"none", "trim"}
    for column in ("max_drawdown_pct", "worst_day_pct", "cvar5_pct", "worst_rolling_5d_pct"):
        assert column in table.columns
