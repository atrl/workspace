"""Source, calendar, fee and native-stage contracts for frozen layer attribution."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from src.workflow import uniform_layer_attribution as U

ROOT = Path(__file__).resolve().parents[1]


def test_raw_source_bound_complete_and_original_hash(monkeypatch):
    raw, audit = U.load_raw_targets(ROOT)
    assert len(raw) == audit['rows'] == 2690
    assert audit['original_rows'] == 2686
    assert audit['f696_crosscheck_maxerr'] < 1e-12
    assert audit['extension_dates'] == ['2026-09-18','2026-09-21','2026-09-22','2026-09-23']
    assert audit['extension_training_cutoff'] == '2022-12-30'
    monkeypatch.setattr(U, 'EXPECTED_RAW_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='identity changed'):
        U.load_raw_targets(ROOT)


@pytest.fixture
def small_inputs(monkeypatch):
    monkeypatch.setattr(U, 'EXPECTED_ROWS', {'development_walk_forward': 2, 'outer_diagnostic': 2})
    dates = pd.to_datetime(['2020-01-02','2020-01-03','2023-01-03','2023-01-04'])
    panel = pd.DataFrame({'signal_date': dates, 'trade_date': dates + pd.Timedelta(days=1),
        'exit_date': dates + pd.Timedelta(days=2),
        'segment': ['development_walk_forward'] * 2 + ['outer_diagnostic'] * 2,
        'soxl_return': [.01, -.03, .02, -.01], 'soxx_return': [.02, -.01, .03, -.02]})
    raw = panel[['signal_date','segment']].assign(raw_soxl=.1, raw_soxx=.7, raw_cash=.2,
                                               train_end=pd.Timestamp('2019-01-01'))
    return panel, raw


@pytest.mark.parametrize('tamper', ['missing','extra','duplicate','weight','late_cutoff'])
def test_invalid_raw_targets_block_before_settlement(small_inputs, tamper):
    panel, raw = small_inputs
    if tamper == 'missing': raw = raw.iloc[1:]
    elif tamper == 'extra': raw = pd.concat([raw, raw.iloc[[0]].assign(signal_date=pd.Timestamp('2020-01-01'))])
    elif tamper == 'duplicate': raw.loc[1,'signal_date'] = raw.loc[0,'signal_date']
    elif tamper == 'weight': raw.loc[0,'raw_cash'] = 5
    elif tamper == 'late_cutoff': raw.loc[0,'train_end'] = raw.loc[0,'signal_date']
    with pytest.raises(ValueError, match='blocked'):
        U.raw_coverage(raw, panel)


def native_fixture(panel):
    objects = {}
    for cost in U.COSTS:
        bench = U.X.soxx_buy_hold_path(panel, cost_bps=cost)
        for span in U.SEGMENTS.values():
            b = bench[bench.segment.eq(span)]
            wealth = b.net_equity.to_numpy(float)
            r = np.r_[wealth[0]-1, wealth[1:]/wealth[:-1]-1]
            for layer in U.LAYERS:
                objects[f'{layer}::{span}::{int(cost)}::suffix_reconciliation'] = pd.DataFrame(
                    {'trade_date': b.trade_date.to_numpy(), 'native_return': r})
    return objects


def test_native_risk_benchmark_exact_execution_dates(small_inputs):
    panel, _ = small_inputs
    rows = U.risk_report(native_fixture(panel), panel)
    assert len(rows) == 20
    assert all(r['sessions'] == 2 and abs(r['relative_wealth_vs_soxx'] - 1) < 1e-12
               and abs(r['beta_vs_soxx'] - 1) < 1e-12 for r in rows)
    assert all(abs(a['relative_wealth_vs_soxx'] - 1) < 1e-12 for r in rows for a in r['annual'])


@pytest.mark.parametrize('tamper', ['missing','duplicate','shifted'])
def test_native_analysis_never_silently_shortens_calendar(small_inputs, tamper):
    panel, _ = small_inputs
    objects = native_fixture(panel)
    key = next(iter(objects));frame = objects[key].copy()
    if tamper == 'missing': frame = frame.iloc[1:]
    elif tamper == 'duplicate': frame = pd.concat([frame, frame.iloc[[0]]])
    elif tamper == 'shifted': frame['trade_date'] -= pd.Timedelta(days=1)
    objects[key] = frame
    with pytest.raises(ValueError, match='calendar mismatch'):
        U.risk_report(objects, panel)


def test_native_prefix_failure_cannot_hide_behind_passed_suffix():
    entry = {'prefix': {'status':'failed','sessions':2}, 'suffix': {'status':'passed','sessions':3}}
    assert not U.native_case_passed(entry, prefix_sessions=2, suffix_sessions=3)
    entry['prefix']['status'] = 'passed'
    assert U.native_case_passed(entry, prefix_sessions=2, suffix_sessions=3)
    assert not U.native_case_passed(entry, prefix_sessions=1, suffix_sessions=3)
    assert not U.native_case_passed({'suffix':entry['suffix']}, prefix_sessions=2, suffix_sessions=3)


def test_fee_ledger_resettles_base_and_blocks_tampered_net_return():
    from src.backtest.position_accounting import settle_target_session
    old = np.array([0.,0.,1.]);target=np.array([.2,.5,.3])
    settled=settle_target_session(old, open_target_weights=target,asset_returns=[.04,-.01,0],cost_bps=10)
    frame=pd.DataFrame([{'execute':True,'recovery_action':False,'desired_soxl':.2,'desired_soxx':.5,
        'desired_cash':.3,'recovery_soxl':np.nan,'recovery_soxx':np.nan,'recovery_cash':np.nan,
        'soxl_return':.04,'soxx_return':-.01,'early_ret_soxl':0.,'early_ret_soxx':0.,
        'strategy_return':settled['net_return'],'fees':settled['fees'],
        'ending_soxl_weight':settled['ending_weights'][0], 'ending_soxx_weight':settled['ending_weights'][1],
        'ending_cash_weight':settled['ending_weights'][2], 'funding_convention':U.LEGACY_CONVENTION}])
    cases={f'{layer}::{span}::10':frame.copy() for layer in U.LAYERS for span in U.SEGMENTS.values()}
    ledger=U.fee_ledger(cases)
    assert len(ledger)==20 and all(r['base_10bp_day_parity'] for r in ledger)
    cases[next(iter(cases))].loc[0,'strategy_return']+=.001
    with pytest.raises(ValueError,match='10bp day parity failed'):
        U.fee_ledger(cases)


@pytest.fixture(scope='module')
def integrated(tmp_path_factory):
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    def no_fit(*args, **kwargs): raise AssertionError('zero-fit contract violated')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Pipeline,'fit',no_fit);patch.setattr(Ridge,'fit',no_fit)
        return U.run_layer_attribution(root=ROOT,card_dir=tmp_path_factory.mktemp('layer-attribution'))


def test_full_frozen_workflow_native_and_card(integrated):
    assert integrated['models_trained']==0 and integrated['orders_submitted']==0
    assert all(integrated['full_parity_vs_e4_active_hold'].values())
    assert integrated['native']['coverage']=={'cases':20,'native_passed':20}
    assert all(r['all_stages_passed'] for r in integrated['native']['results'].values())
    assert len(integrated['metrics'])==len(integrated['risk_report'])==len(integrated['risk_mechanism'])==20
    assert all(m['status']=='ok' for m in integrated['risk_mechanism'])
    assert integrated['artifact_policy']['policy']=='slim_card_recorder_first'
    json.dumps(integrated,allow_nan=False)
    card=Path(integrated['recorder_uri']).parent
    assert (card/'workflow.yaml').is_file() and (card/'conclusion.md').is_file()


def test_no_recovery_layer_and_recorder_analysis_contract(integrated):
    from qlib.workflow import R
    with R.uri_context(integrated['recorder_uri']):
        rec=R.get_recorder(recorder_id=integrated['recorder_id'],experiment_name=integrated['experiment'])
        replay=rec.load_object('replay::open_risk_projection_nohold_no_recovery::development_walk_forward::10')
        assert not replay.recovery_action.any()
        assert replay.execute_open_target.dtype==replay.recovery_action.dtype==bool
        assert set(replay.funding_convention)=={U.LEGACY_CONVENTION}
        assert len(rec.load_object('risk_report'))==20
        assert len(rec.load_object('annual_metrics'))==len(integrated['annual_metrics'])


def test_recovery_toggle_uses_exact_independent_account_settlement(monkeypatch):
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.backtest.position_accounting import settle_target_session
    dates=pd.to_datetime(['2020-01-02'])
    market=pd.DataFrame({'signal_date':dates, 'trade_date':dates+pd.Timedelta(days=1),
        'exit_date':dates+pd.Timedelta(days=2),'segment':'development_walk_forward',
        'soxl_weight':.2,'soxx_weight':.5,'cash_weight':.3,
        'soxl_return':.04,'soxx_return':-.01,'early_ret_soxl':.02,'early_ret_soxx':-.005})
    target=np.array([.2,.5,.3]);recovery=np.array([.1,.7,.2])
    calls=[]
    def opened(*args, **kwargs):
        return {'execute_open_target':True,'desired_open_weights':target,'risk_state':0,
                'risk_clock':0,'current_beta_cap':1.1,'decision_reason':'fixture'}
    def recovered(*args, **kwargs):
        calls.append(True);return {'recovery_action':True,'target_weights':recovery}
    monkeypatch.setattr(U,'prepare_on_policy_open',opened)
    monkeypatch.setattr(U,'prepare_on_policy_recovery',recovered)
    common=dict(spec=U.Stage1OpenActionValueSpec(root=ROOT),
        recovery_spec=IntradayRecoveryDirectAllocationSpec(root=ROOT),models={},recovery_models={},
        selected_model={},cost=10.,use_hold=False,funding_convention=U.LEGACY_CONVENTION)
    nr=U._layer_loop(market,use_recovery=False,**common)
    assert not calls and not nr.recovery_action.any()
    with_rec=U._layer_loop(market,use_recovery=True,**common)
    expected=settle_target_session(np.array([0.,0.,1.]),open_target_weights=target,
        asset_returns=[.04,-.01,0.],cost_bps=10.,recovery_target_weights=recovery,
        early_returns=[.02,-.005,0.])
    assert calls==[True] and bool(with_rec.recovery_action.iloc[0])
    assert abs(with_rec.strategy_return.iloc[0]-expected['net_return'])<1e-12
    np.testing.assert_allclose(with_rec[['ending_soxl_weight','ending_soxx_weight','ending_cash_weight']].iloc[0],
                               expected['ending_weights'],rtol=0,atol=1e-12)
