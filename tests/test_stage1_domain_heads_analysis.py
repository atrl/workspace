import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

import src.analysis.stage1_domain_heads as analysis


@pytest.mark.parametrize("finra", [False, True])
def test_account_summary_preserves_all_windows_calendar_and_intraday_risk(monkeypatch, finra):
    plan = json.loads(Path("research/stage1_domain_heads_20261018/plan.json").read_text())
    calendar = pd.bdate_range("2022-12-30", "2026-02-02")
    trade = calendar[(calendar >= "2023-01-03") & (calendar < "2026-02-02")]
    dates = pd.DataFrame({"signal_date": calendar[calendar.get_indexer(trade) - 1],
                          "trade_date": trade, "exit_date": calendar[calendar.get_indexer(trade) + 1]})
    plan["accounts"].update(expected_sessions=len(trade), first_signal="2023-01-02",
                            first_trade="2023-01-03", terminal="2026-02-02T09:47:00-05:00")
    rates = {"A_active_research_replay": .0003, "B_shared": .0004, "C_domain": .001,
             "SOXX": .0002, "strong9097": .0005, "additional27baa": .002}
    candidate_name = "C_domain"
    analyzer = analysis.analyze_accounts
    if finra:
        from src.analysis.finra_allocation import analyze_accounts, CANDIDATE, REQUIRED, FAMILIES
        candidate_name, analyzer = CANDIDATE, analyze_accounts
        rates[candidate_name] = rates.pop("C_domain")
        rates["B_FINRAcontext"] = .0006
        plan["accounts"]["families"] = list(FAMILIES)
        plan["gates"]["required_comparators"] = list(REQUIRED)
        plan["gates"]["paired_uncertainty"]["alpha"] = .025 / 15
    books = {}
    for family, rate in rates.items():
        for cost in plan["accounts"]["slippage_bps"]:
            path = dates.copy()
            path["strategy_return"], path["turnover"], path["fees"] = rate, .01, 0.
            path["equity"] = np.cumprod(1 + path.strategy_return)
            path["ending_soxl_quantity"] = 0.
            path["ending_soxl_weight"], path["ending_soxx_weight"], path["ending_cash_weight"] = 0., .5, .5
            path["execute_open_target"], path["recovery_action"] = True, False
            cash = plan["accounts"]["initial_cash"]
            rows = []
            for number, day in enumerate(trade):
                nav = cash if number == 0 else cash * path.equity.iloc[number - 1]
                rows.append({"stamp": day + pd.Timedelta(hours=9, minutes=45), "nav": nav,
                             "cash": cash if number == 0 else nav, "receivable": 0., "q_soxl": 0., "q_soxx": 0.})
                closing = nav * .6 if family == candidate_name and number == 300 else nav
                rows.append({"stamp": day + pd.Timedelta(hours=16), "nav": closing,
                             "cash": closing, "receivable": 0., "q_soxl": 0., "q_soxx": 0.})
            books[(family, cost)] = {"authority": path, "points": pd.DataFrame(rows), "audit": {"status": "passed"}}
    calls = []

    def bootstrap(left, right, **kwargs):
        calls.append(kwargs)
        return {"lower_bound_positive": True, "daily_lower_bound": .0001, **kwargs}

    monkeypatch.setattr(analysis, "stationary_bootstrap_excess", bootstrap)
    result = analyzer(books, pd.DataFrame({"trade_date": calendar}), plan)
    assert len(result["metrics"]) == (21 if finra else 18)
    assert len(result["paired_comparison"]) == (18 if finra else 15)
    assert len(result["rolling"]) == (21 if finra else 18)
    assert all(len(frame) == len(trade) - 125 for frame in result["rolling"].values())
    annual = result["annual_metrics"]
    assert set(annual.loc[annual.complete_period, "period"]) == {"2024", "2025"}
    # 2023 is missing a source-calendar session; 2026 matches only a source
    # prefix.  Neither can be admitted by simply matching the account slice.
    assert not annual.loc[annual.period.isin(["2023", "2026"]), "complete_period"].any()
    assert result["gates"]["minimum_complete_years"] is True
    assert result["gates"]["net_relative_wealth_all_costs"] is True  # 27baa is additional only.
    assert result["gates"]["sampled_drawdown_30pct"] is False
    candidate = result["metrics"].loc[result["metrics"].family.eq(candidate_name)]
    assert candidate.max_drawdown_pct.eq(0).all()
    assert candidate.close_max_drawdown_pct.le(-39).all()
    assert not result["gates"]["fresh_fullminute_risk_qualified"]
    assert not result["gates"]["qualified_independent_forward"]
    assert len(calls) == (15 if finra else 12)
    assert all(call == {"mean_block": 63, "resamples": 10000, "alpha": .025 / (15 if finra else 12), "seed": 20261014}
               for call in calls)
    if finra:
        # A missing newly required control must not silently fall back to the
        # old four-comparator analysis or discard the unmatched cost account.
        incomplete = {key: value for key, value in books.items() if key != ("B_FINRAcontext", 5)}
        with pytest.raises(ValueError, match="all three cost accounts"):
            analyzer(incomplete, pd.DataFrame({"trade_date": calendar}), plan)


def test_information_uses_mature_primary_labels_and_zero_alpha_is_negative():
    dates = pd.bdate_range("2026-10-01", periods=5)
    cutoff = pd.Timestamp("2026-10-09T09:47:00-04:00")
    labels = pd.DataFrame({"tail_event_5d": [0., 1., 0., 1., 1.],
                           "relative_alpha_5d": [-.01, .01, 0., .5, .5]}, index=dates)
    meta = pd.DataFrame({"label_available_at": [cutoff - pd.Timedelta(days=2), cutoff - pd.Timedelta(days=1),
                                               cutoff, cutoff + pd.Timedelta(minutes=1), pd.NaT]}, index=dates)
    data = pd.concat({"label": labels, "meta": meta}, axis=1)
    data.index = pd.MultiIndex.from_arrays([dates, ["SOXL_SOXX_CASH"] * len(dates)], names=["datetime", "instrument"])
    dataset = DatasetH(handler=DataHandlerLP.from_df(data), segments={"predict": (dates[0], dates[-1])})
    pred = pd.DataFrame({"tail_probability": [.1, .9, .2, .99, .99],
                         "alpha_probability": [.2, .8, .3, .99, .99]}, index=dates)
    result = analysis.information_metrics({("B_shared", 71): pred}, {"B_shared": pred}, dataset)
    assert len(result) == 4
    assert result.observations.eq(3).all()
    assert result.positive_events.eq(1).all()  # relative_alpha == 0 is not a positive event.
    assert result.excluded_unmatured_or_unknown_maturity.eq(2).all()
    assert result.auc.eq(1).all() and not result.account_execution_gate.any()
    alpha = result.query("task == 'alpha'").iloc[0]
    assert alpha.brier == pytest.approx((.2**2 + .2**2 + .3**2) / 3)
    assert alpha.bce == pytest.approx(-(np.log(.8) + np.log(.8) + np.log(.7)) / 3)
    changed = pred.copy()
    changed.iloc[3:] = 0.
    repeated = analysis.information_metrics({("B_shared", 71): changed}, {"B_shared": changed}, dataset)
    pd.testing.assert_frame_equal(result, repeated)
