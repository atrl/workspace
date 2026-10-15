from copy import deepcopy
from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import src.backtest.diagnostic_forward_state as module
from src.backtest.diagnostic_forward_state import ForwardPaperState
from src.data.diagnostic_forward_inputs import require_phase,sanitize_global_state,validate_quote,digest,validate_bars
from src.schedule.diagnostic_forward_study import persist_checkpoint,load_checkpoint,registered_slots,assert_fixed_time


def test_slot_atomic_durable_clock_state_contract(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1];card=root/'research/diagnostic_forward_study_20261016'
    with pytest.raises(PermissionError):require_phase(card,'probe')
    assert sanitize_global_state({'market_us':'AFTERNOON','user_id':'SECRET','token':'SECRET'})=={'market_us':'AFTERNOON'}
    tariff=json.loads((root/'research/frozen_policy_raw_cashbook_20261015/plan.json').read_text())['tariff'];state=ForwardPaperState(tariff)
    origin=pd.Timestamp('2026-10-19 09:00',tz='America/New_York')
    def clock(time):
        stamp=pd.Timestamp('2026-10-19 '+time,tz='America/New_York');return {'utc':stamp.isoformat(),'monotonic_ns':int((stamp-origin).total_seconds()*1e9)+100}
    def quote(time):
        start=pd.Timestamp('2026-10-19 '+time,tz='America/New_York');received=start+pd.Timedelta(seconds=2)
        return {'request_started':clock(time),'received':clock(received.strftime('%H:%M:%S')),'payload_sha256':'a'*64,'source_identity':'synthetic_fixture','quotes':{s:{'price':p,'quote_at':(start+pd.Timedelta(seconds=1)).isoformat()} for s,p in [('SOXL',100.),('SOXX',50.)]}}
    from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights
    dates=pd.DatetimeIndex(['2026-10-16','2026-10-19']);proposals=pd.DataFrame([[.2,.5,.3],[.201,.499,.3]],index=dates,columns=['soxl_weight','soxx_weight','cash_weight'])
    expected=_smooth_weights(proposals,spec=SimpleNamespace(smoothing_span=3,rebalance_threshold=.02),asset_returns=np.array([[.01,.005],[0.,0.]]))
    smooth=ForwardPaperState(tariff)
    for day,ret in enumerate([[-.2,.08],[.01,.005]]):
        out=smooth.smooth({family:proposals.iloc[day].to_numpy() for family in smooth.ema},ret)
        for value in out.values():np.testing.assert_allclose(value,expected.iloc[day].to_numpy(),rtol=0,atol=1e-15)
    slots=registered_slots('2026-10-19');assert slots['previous_session']=='2026-10-16' and slots['terminal_session']=='2026-10-20'
    with pytest.raises(ValueError):assert_fixed_time('2026-10-19 09:48-04:00',slots['open_fill'])
    with pytest.raises(ValueError):assert_fixed_time('2026-10-21 09:47-04:00',slots['terminal_fill'])
    early={'request_started':clock('09:44:59'),'received':clock('09:45:10'),'payload_sha256':'a'*64,'source_identity':'fixture','sdk_sha256':'sdk','endpoint':'history','query_mode':'RTH'}
    evidence={'passed':True,'edge':'start','scope':'fixture','sdk_sha256':'sdk','endpoint':'history','query_mode':'RTH'}
    with pytest.raises(ValueError,match='request started'):
        validate_bars(pd.DataFrame(),early,session='2026-10-19',cutoff=slots['known_cutoff'],seal=clock('09:46:00'),mapping_evidence=evidence,synthetic=True)
    targets={k:[.2,.5,.3] for k in state.books};intent=state.seal('open',targets,clock('09:46:00'),slot_at='2026-10-19 09:47-04:00',bindings={'fixture':True})
    persist_checkpoint(tmp_path,1,{'intent':intent},{'state':state});restored=load_checkpoint(tmp_path)['state'];assert restored.state_digest()==state.state_digest()
    receipt=quote('09:47:00');bad=deepcopy(receipt);bad['request_started']=clock('09:45:59')
    with pytest.raises(ValueError):validate_quote(bad,seal=intent['clock'],slot_at=intent['slot_at'])
    alternative=deepcopy(state);otherquote=deepcopy(receipt);otherquote['quotes']['SOXL']['price']=120
    alternative.execute_slot('open',otherquote)
    assert alternative.intents['open']['targets']==intent['targets']
    opened=state.execute_slot('open',receipt);assert len(opened['reports'])==12
    before_duplicate=state.state_digest();assert state.execute_slot('open',receipt)==opened;assert state.state_digest()==before_duplicate
    changed=deepcopy(receipt);changed['quotes']['SOXL']['price']=101
    with pytest.raises(ValueError,match='conflicting'):state.execute_slot('open',changed)
    assert state.state_digest()==before_duplicate
    peak=[b.peak for b in state.books.values()];state.mark([120.,60.]);assert [b.peak for b in state.books.values()]==peak
    clone=deepcopy(state);h=clone.state_digest();clone.ema['index_candidate']['accepted'][0]=.1;assert clone.state_digest()!=h
    clone=deepcopy(state);h=clone.state_digest();next(iter(clone.books.values())).open_snapshot={'current_beta_cap':.3};assert clone.state_digest()!=h
    # Failed seventh settlement cannot publish the first six clones or undo open.
    state.seal('recovery',targets,clock('10:31:00'),slot_at='2026-10-19 10:32-04:00',bindings={'fixture':True});financial=deepcopy(state.snapshot());original=module.settle_raw;counter=[0]
    def fail_seventh(*args,**kwargs):
        counter[0]+=1
        if counter[0]==7:raise ValueError('synthetic seventh-book failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'settle_raw',fail_seventh)
    with pytest.raises(ValueError,match='seventh'):state.execute_slot('recovery',quote('10:32:00'))
    assert state.snapshot()==financial and 'open' in state.committed and state.trading_disabled
    assert len(state.events)==1
    # Nonfinite quote is rejected before any legacy settlement function call.
    other=deepcopy(restored);bad=quote('09:47:00');bad['quotes']['SOXL']['price']=np.nan;counter[0]=0
    with pytest.raises(ValueError):other.execute_slot('open',bad)
    assert counter[0]==0 and all(np.array_equal(b.q,[0,0]) for b in other.books.values())
