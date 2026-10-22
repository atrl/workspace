import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.risk_projection_reachability import source_target_path


def test_source_target_account_tracks_own_shares_cash_and_terminal():
    days = pd.DatetimeIndex(["2023-01-04", "2023-01-05"])
    terminal = pd.Timestamp("2023-01-06")
    stamps = [day + pd.Timedelta(clock) for day in days
              for clock in ("9h45min", "9h47min", "10h32min", "16h")]
    stamps += [terminal + pd.Timedelta("9h45min"), terminal + pd.Timedelta("9h47min")]
    prices = pd.DataFrame({"SOXL": 100.0, "SOXX": 100.0}, index=pd.DatetimeIndex(stamps))
    trace = pd.DataFrame({"trade_date": days,
                          "source_target_soxl_weight": .31,
                          "source_target_soxx_weight": .49,
                          "source_target_cash_weight": .2})
    reference = pd.DataFrame({"signal_date": days - pd.Timedelta(days=1),
                              "trade_date": days,
                              "exit_date": [days[1], terminal]})
    tariff = json.loads(Path("research/path_area_action_bridge_20261018/P3_plan.json").read_text())[
        "costs_and_budgets"]["tariff"]
    path, grid = source_target_path(trace, reference, prices, {},
                                    tariff=tariff, cost=2, initial_cash=100000)
    assert len(grid) == len(path["points"]) == 10
    assert len(path["replay"]) == len(days) == 2
    assert path["replay"].exit_date.iloc[-1] == terminal
    assert (path["targets"].cash_weight >= .15).all()
    assert (path["targets"].soxl_weight > .30).all()
    assert (path["points"].cash >= -1e-7).all()
    assert np.isfinite(path["replay"].strategy_return).all()
    assert path["replay"].equity.iloc[-1] < 1.0
