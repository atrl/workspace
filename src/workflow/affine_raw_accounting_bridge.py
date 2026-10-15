"""Eight existing Qlib marks: finite quantity/exposure identities, never a portfolio replay."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.published_price_basis import _events
from src.workflow.minute_action_value import _source
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.data import D
    from qlib.workflow import R
    import yaml
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest();assert digest=='1e03bf5f8b8f813a46e2cb26909c8d2d1cec4759693bcb2e7d160db935bf75c6'
    for name,sha in plan['immutable_code_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
    source_bytes=(root/plan['raw_source_manifest']['path']).read_bytes();assert hashlib.sha256(source_bytes).hexdigest()==plan['raw_source_manifest']['sha256'];manifest=json.loads(source_bytes)
    provider_bytes=(root/plan['provider_uri']/'qlib_store_manifest.json').read_bytes();assert hashlib.sha256(provider_bytes).hexdigest()==plan['provider_manifest_sha256']
    reference,ra=_source(root,plan['reference'],plan['reference_artifacts']);readiness,rda=_source(root,plan['readiness'],plan['readiness_artifacts'])
    events={};event_bytes={};candidates=[]
    for source in plan['rehab_sources']:
        raw=(root/source['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==source['sha256'];event_bytes[source['symbol']]=raw
        df=_events(pd.read_parquet(root/source['path']));events[source['symbol']]=df
        for row in df.loc[df.event_date.between('2023-01-04','2026-10-09')].to_dict('records'):
            split=any(pd.notna(row.get(f'{prefix}_base')) for prefix in ['split','join'])
            cash=pd.notna(row.get('per_cash_div')) and row['per_cash_div']>0
            if split and not cash:candidates.append(('split',row['event_date'],source['symbol']))
            if cash and not split:candidates.append(('cash',row['event_date'],source['symbol']))
    assert sorted(x[1:] for x in candidates if x[0]=='split')[0]==(pd.Timestamp('2024-03-07'),'SOXX')
    assert sorted(x[1:] for x in candidates if x[0]=='cash')[0]==(pd.Timestamp('2023-03-21'),'SOXL')
    b_dates={s:df.loc[df.forward_adj_factorB.ne(0),'event_date'].dt.strftime('%Y-%m-%d').tolist() for s,df in events.items()}
    assert b_dates=={'SOXL':['2010-09-22'],'SOXX':[]}
    calendar=pd.DatetimeIndex(D.calendar(freq='1min'));sessions=pd.DatetimeIndex(calendar.normalize().unique()).sort_values()
    frame=reference['inputs/frame'];window=sessions[(sessions>=pd.Timestamp('2023-01-04'))&(sessions<=pd.Timestamp('2026-10-09'))]
    assert list(window[:2])==[pd.Timestamp('2023-01-04'),pd.Timestamp('2023-01-05')]
    marks=[];cash_rows=[];cases=[];feature_calls=0
    for case in plan['cases']:
        symbol=case['symbol'];ev=events[symbol];dates=pd.DatetimeIndex(case['anchors'])
        if 'event_date' in case:
            event_day=pd.Timestamp(case['event_date']);pos=sessions.get_loc(event_day)
            assert list(dates.normalize())==list(sessions[pos-1:pos+2])
            crossed=ev.loc[(ev.event_date>dates[0].normalize())&(ev.event_date<=dates[-1].normalize())]
            assert len(crossed)==1 and crossed.iloc[0].event_date==event_day
            event=crossed.iloc[0].to_dict()
        else:
            assert ev.loc[(ev.event_date>dates[0].normalize())&(ev.event_date<=dates[-1].normalize())].empty
            event=None
        local=[]
        for stamp in dates:
            read=D.features([symbol],['$execution_price','$raw_execution_price'],start_time=stamp,end_time=stamp,freq='1min');feature_calls+=1
            if len(read)!=1:raise ValueError('missing/duplicate anchor')
            price=float(read.iloc[0]['$execution_price']);raw_price=float(read.iloc[0]['$raw_execution_price'])
            if not np.isfinite([price,raw_price]).all() or min(price,raw_price)<=0:raise ValueError('invalid anchor')
            original=frame.loc[pd.to_datetime(frame.trade_date).eq(stamp.normalize())]
            assert len(original)==1
            np.testing.assert_allclose([price,raw_price],[original.iloc[0][symbol.lower()+'_open'],original.iloc[0]['raw_'+symbol.lower()+'_open']],atol=1e-5,rtol=5e-6)
            a,b=1.,0.
            for item in ev.loc[(ev.event_date>stamp.normalize())&(ev.event_date<=pd.Timestamp(plan['raw_source_manifest']['as_of']))].itertuples():
                a*=item.forward_adj_factorA;b=b*item.forward_adj_factorA+item.forward_adj_factorB
            formula=a*raw_price+b
            np.testing.assert_allclose(price,formula,atol=1e-5,rtol=5e-6)
            split_factor=1.
            for item in ev.loc[(ev.event_date>dates[0].normalize())&(ev.event_date<=stamp.normalize())].itertuples():split_factor/=item.exact_split_price_factor
            market_receipt=next(e for e in manifest['entries'] if e['symbol']==symbol)
            local.append({'case':case['id'],'symbol':symbol,'mark_at':stamp,'raw_bar_right_edge':stamp+pd.Timedelta(minutes=1),'raw_bar_start':stamp,'qlib_adjusted_price':price,'qlib_raw_price':raw_price,'published_A':a,'published_B':b,'formula_adjusted_price':formula,'provider_price_roundoff':price-formula,'physical_split_share_factor':split_factor,'physical_shares':100*split_factor,'historical_market_payload_received_at':market_receipt['fetched_at'],'receipt_role':'historicalretrieval_not_decisiontime_firstreceipt'})
        base=local[0];q_actual=100*base['qlib_raw_price']/base['qlib_adjusted_price'];q_formula=100*base['qlib_raw_price']/base['formula_adjusted_price']
        for row in local:
            row.update(q_adjusted_source_conversion=q_actual,q_adjusted_formula_control=q_formula,
                implied_raw_shares=q_formula*row['published_A'],affine_cash_offset=q_formula*row['published_B'],
                physical_raw_asset_leg=row['physical_shares']*row['qlib_raw_price'],proxy_formula_asset_value=q_formula*row['formula_adjusted_price'],
                proxy_qlib_asset_value=q_actual*row['qlib_adjusted_price'])
            row['implied_minus_physical_shares']=row['implied_raw_shares']-row['physical_shares']
            row['asset_leg_gap_excluding_unknown_cash']=row['proxy_formula_asset_value']-row['physical_raw_asset_leg']
            row['source_provider_roundoff_value']=row['proxy_qlib_asset_value']-row['proxy_formula_asset_value']
            decomposition=row['implied_raw_shares']*row['qlib_raw_price']+row['affine_cash_offset']
            row['exposure_decomposition_residual']=row['proxy_formula_asset_value']-decomposition
            np.testing.assert_allclose(row['proxy_formula_asset_value'],decomposition,atol=1e-10,rtol=1e-10)
            if case['id']=='stable_operator_control':
                assert row['published_A']==base['published_A'] and row['published_B']==base['published_B']
                expected=100*base['published_B']*(base['qlib_raw_price']-row['qlib_raw_price'])/base['formula_adjusted_price']
                row['stable_general_identity_rhs']=expected;row['stable_identity_residual']=row['asset_leg_gap_excluding_unknown_cash']-expected
                np.testing.assert_allclose(row['asset_leg_gap_excluding_unknown_cash'],expected,atol=1e-10,rtol=1e-10)
        share_match=all(np.isclose(x['implied_raw_shares'],x['physical_shares'],atol=1e-5,rtol=5e-6) for x in local)
        if case['id']=='split_declared_no_cash':
            assert event['forward_adj_factorB']==0
            ratio=float(event['exact_split_price_factor']);np.testing.assert_allclose(event['forward_adj_factorA'],ratio,atol=1e-5,rtol=5e-6)
        case_gate={'case':case['id'],'source_identity':'passed_pinned_source_and_anchors','unit_identity':'passed_algebra_and_provider_roundoff','event_accounting':'not_applicable_no_event' if event is None else 'split_quantity_bridge_within_declared_precision' if case['id']=='split_declared_no_cash' and share_match else 'needs_data_currency_entitlement_paydate_cash_ledger',
            'PIT_timing':'not_qualified_historical_receipt_only','implied_raw_share_identity_within_tolerance':bool(share_match),'final_implied_minus_physical_shares':local[-1]['implied_minus_physical_shares'],'final_asset_leg_gap_excluding_unknown_cash':local[-1]['asset_leg_gap_excluding_unknown_cash'],'whole_strategy_bias_estimated':False}
        cases.append(case_gate);marks.extend(local)
        if event is not None:
            receipt=next(e for e in manifest['rehab'] if e['symbol']==symbol)
            amount=event.get('per_cash_div')
            cash_rows.append({'case':case['id'],'symbol':symbol,'event_date':event['event_date'],'split_base':event.get('split_base'),'split_ert':event.get('split_ert'),'per_cash_div_original':amount,'published_event_A':event['forward_adj_factorA'],'published_event_B':event['forward_adj_factorB'],'conditional_lot_cash_in_source_units':100*amount if pd.notna(amount) else None,'cash_currency':None,'announcement_at':None,'record_date':None,'pay_date':None,'cash_available_at':None,'cash_or_receivable_USD_credited':None,'event_payload_received_at':receipt['fetched_at'],'event_receipt_role':'historicalretrieval_not_eventpublication','qualified_economic_cash_bridge':False,'cash_not_inferred_from_B':True})
    assert feature_calls==8
    mark_table=pd.DataFrame(marks);case_table=pd.DataFrame(cases);cash_table=pd.DataFrame(cash_rows)
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='fixed_unit_bridge',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_replays=0,new_accounts=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audits={'reference':ra,'readiness':rda},event_bytes=event_bytes,raw_source_manifest_bytes=source_bytes,provider_manifest_bytes=provider_bytes,selected_events=cash_table,old_clock_example=readiness['inputs/old_linked_audit'])
        rec.save_objects(artifact_path='sources',execution_workflow=Path(__file__).read_text(),**{f.replace('/','__'):(root/f).read_text() for f in plan['immutable_code_sha256']})
        policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*']}
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'finite_unit_bridge_only_dividend_cash_and_PIT_need_data','objective_achieved':False,'numeric_unit_cases':3,'Qlib_price_anchor_rows':len(marks),'Qlib_features_calls':feature_calls,'Qlib_calendar_calls':1,'symbolic_nonzero_B_identity':plan['algebra']['stable_general_identity'],'nonzero_B_scope':b_dates,'nonzero_B_empirical_case_available':False,'case_gates':cases,'new_fits':0,'new_policy_replays':0,'new_native_accounts':0,'new_market_or_CA_fetches':0,'orders':0,'observer_launches':0,'strategy_NAV_or_risk_recomputations':0,'cash_USD_NAV_qualified':False,'PIT_timing_qualified':False,'whole_strategy_bias_estimated':False,'next_action':'Qualify explicitissuer/vendorcashamountcurrency/entitlement/paydate andcausalreceipt ledger forfixedcashcase beforephysicalcashbridge; do not inferDfromA/B ormodify27baa. Futurebasisimplementationwouldrequireseparateidentity/plan.','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(root/plan['provider_uri']),'artifact_policy':policy}
        rec.save_objects(artifact_path='analysis',summary=summary,marks=mark_table,cases=case_table,cash_fields=cash_table)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),unit_cases=3)
    for name,table in [('unit_marks',mark_table),('case_gates',case_table),('cash_field_qualification',cash_table)]:table.to_csv(card/(name+'.csv'),index=False)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['reference']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary
