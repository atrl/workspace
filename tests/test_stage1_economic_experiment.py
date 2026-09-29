import pandas as pd

from src.workflow.stage1_economic_experiment import _gate_rows


def test_higher_return_and_fewer_trades_cannot_override_tail_risk_failure():
    variants = ["no_hold_reference", "active_hold_baseline", "rules_only", "own_state_net_value_frozen_2022", "own_state_net_value_expanding_2025"]
    rows, annual = [], []
    for variant in variants:
        candidate = variant not in variants[:2]
        for cost in (10.0, 25.0):
            rows.append(dict(segment="development_walk_forward", variant=variant, cost_bps=cost,
                             return_pct=110 if candidate else 100, max_drawdown_pct=-20,
                             worst_day_pct=-5, cvar5_pct=-4 if candidate else -3,
                             worst_rolling_5d_pct=-10, open_trade_days=60 if candidate else 100,
                             total_turnover=60 if candidate else 100))
        annual.append(dict(variant=variant, year=2020, terminal_wealth=2.1 if candidate else 2.0))
    result = _gate_rows(pd.DataFrame(rows), pd.DataFrame(annual))
    assert len(result) == 6
    assert result.development_return_above_baseline.all()
    assert result.open_trade_days_reduced_8pct.all()
    assert not result.cvar5_not_worse.any()
    assert not result.all_pass.any()


def test_outer_diagnostic_return_never_enters_development_gate():
    variants = ["no_hold_reference", "active_hold_baseline", "rules_only", "own_state_net_value_frozen_2022", "own_state_net_value_expanding_2025"]
    rows, annual = [], []
    for variant in variants:
        for cost in (10.0, 25.0):
            common = dict(variant=variant, cost_bps=cost, max_drawdown_pct=-20,
                          worst_day_pct=-5, cvar5_pct=-3, worst_rolling_5d_pct=-10,
                          open_trade_days=100, total_turnover=100)
            rows.append(dict(common, segment="development_walk_forward", return_pct=100))
            rows.append(dict(common, segment="outer_reused_diagnostic", return_pct=100000))
        annual.append(dict(variant=variant, year=2020, terminal_wealth=2.0))
    result = _gate_rows(pd.DataFrame(rows), pd.DataFrame(annual))
    assert not result.development_return_above_baseline.any()
    assert not result.all_pass.any()
