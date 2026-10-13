import numpy as np
from src.backtest.cushion_target import CushionTarget


def test_cushion_direct_target_spends_causal_budget_with_minimal_leveraged_weight():
    policy=CushionTarget(floor_fraction=.7,multiplier=4,minimum_cash=.15,maximum_soxl=.3)
    peak=100000.
    at_peak=policy.target(None,[0.,0.],100000.,[10.,100.],peak)
    np.testing.assert_allclose(at_peak,[.175,.675,.15],atol=1e-12)
    drawdown=policy.target(None,[0.,0.],80000.,[10.,100.],peak)
    np.testing.assert_allclose(drawdown,[0.,.5,.5],atol=1e-12)
    exhausted=policy.target(None,[0.,0.],69000.,[10.,100.],peak)
    np.testing.assert_array_equal(exhausted,[0.,0.,1.])
    constant=CushionTarget(floor_fraction=.7,multiplier=4,minimum_cash=.15,maximum_soxl=.3,constant=True)
    np.testing.assert_allclose(constant.target(None,[0.,0.],80000.,[10.,100.],peak),at_peak)
