"""Iterate independent after-open checkpoints without replaying every prefix."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import TieredFees
from src.backtest.stage1_event_checkpoint import _State, _open, _finish


def iter_after_open_checkpoints(frame, *, controller_by_date, recovery_spec,
                                action_spec, selection, plan, slippage_bps=0):
    initial_cash = float(plan["initial_cash"])
    state = _State(np.zeros(2), initial_cash, initial_cash, 0, 0, [], None,
                   TieredFees(plan["tariff"], slippage_bps=slippage_bps))
    for index in range(len(frame)):
        row = frame.iloc[index]
        active = controller_by_date(pd.Timestamp(row.trade_date))
        opened = _open(row, state, controller=active,
                       recovery_spec=recovery_spec, action_spec=action_spec)
        opened.index = index
        yield deepcopy(opened)
        _finish(row, opened, controller=active, recovery_spec=recovery_spec,
                selection=selection, initial_cash=initial_cash, event=None)
