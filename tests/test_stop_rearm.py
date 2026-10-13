from pathlib import Path
from src.backtest.stop_rearm import advance_rearmed_state
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec, _advance_state


def test_single_cycle_stays_armed_only_after_recovery():
    spec=IntradayRecoveryDirectAllocationSpec(root=Path('.'))
    state=clock=0
    for _ in range(spec.wait_days+spec.ramp_days+spec.hold_days-1):
        expected=_advance_state(state,clock,.13,spec)
        actual=advance_rearmed_state(state,clock,.13,spec)
        assert actual==expected
        state,clock,_=actual
    assert _advance_state(state,clock,.13,spec)==(0,0,1.)
    state,clock,q=advance_rearmed_state(state,clock,.13,spec)
    for _ in range(100):
        assert advance_rearmed_state(state,clock,.13,spec)==(state,clock,1.)
    assert advance_rearmed_state(state,clock,.119,spec)==(0,0,1.)
    assert advance_rearmed_state(0,0,.121,spec)==(1,1,0.)
