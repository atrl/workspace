from pathlib import Path
import numpy as np
import pandas as pd
from src.analysis.cash_layers import projected_layers
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec


def test_sequential_binding_is_not_double_counted():
    spec=IntradayRecoveryDirectAllocationSpec(root=Path('.'))
    row=pd.Series(dict(soxl_weight=.3,soxx_weight=.4,cash_weight=.3))
    parts,target,cap,flags=projected_layers(row,state=3,clock=0,drawdown=.23,history=[-.02]*4,spec=spec)
    np.testing.assert_allclose([parts['raw_cash'],parts['legality_increment'],parts['cycle_increment'],parts['trailing_increment'],parts['emergency_increment']],[.3,0,0,.1,.3],atol=1e-12)
    np.testing.assert_allclose(target,[0,.3,.7],atol=1e-12)
    assert flags['trailing_beta_binding'] and flags['emergency_beta_binding'] and not flags['cycle_beta_binding']
    parts,target,cap,flags=projected_layers(row,state=1,clock=0,drawdown=.23,history=[-.02]*4,spec=spec)
    assert cap==.2 and not flags['trailing_beta_binding'] and not flags['emergency_beta_binding']
    np.testing.assert_allclose(sum(parts.values()),target[2],atol=1e-12)
