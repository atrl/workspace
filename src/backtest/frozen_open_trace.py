"""Diagnostic target/HOLD trace; deliberately not a deployable risk policy."""
import numpy as np


class FrozenOpenTrace:
    def __init__(self,targets,ledger):
        for frame in [targets,ledger]:
            if frame.signal_date.duplicated().any():raise ValueError('duplicate frozen signal dates')
        if not targets.signal_date.equals(ledger.signal_date) or not targets.trade_date.equals(ledger.trade_date):
            raise ValueError('frozen trace calendars disagree')
        if ledger.recovery_action.any():raise ValueError('recovery trace unsupported; cannot silently omit trades')
        self.targets=targets.set_index('signal_date').copy()
        self.execute=ledger.set_index('signal_date').execute_open_target.copy()
        values=self.targets[['soxl_weight','soxx_weight','cash_weight']].to_numpy(float)
        if not np.isfinite(values).all() or (values<0).any() or not np.allclose(values.sum(axis=1),1,rtol=0,atol=1e-9):
            raise ValueError('invalid frozen target weights')

    def target(self,row,quantities,cash,known,peak):
        if row.signal_date not in self.targets.index:raise ValueError('missing frozen signal')
        target=self.targets.loc[row.signal_date]
        if target.trade_date!=row.trade_date:raise ValueError('frozen execution date changed')
        return target[['soxl_weight','soxx_weight','cash_weight']].to_numpy(float) if self.execute.loc[row.signal_date] else None
