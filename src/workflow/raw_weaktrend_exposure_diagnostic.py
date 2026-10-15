"""Saved raw books: complete exposure/win-loss description, no model calls."""
import hashlib,json,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from src.workflow.minute_action_value import _source
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def summarize(group):
    fields=['proposal_beta','proposal_cash','requested_beta','requested_cash','actual_open_beta','actual_spendable_cash_weight','actual_receivable_weight','cap_reduction','execution_beta_deviation']
    return {'sessions':len(group),'relative_log_sum':float(group.relative_log.sum()),'candidate_minus_strong_log_sum':float(group.candidate_minus_strong_log.sum()),**{f'mean_{f}':float(group[f].mean()) if len(group) else None for f in fields}}


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest();assert digest=='7dffc8c8fde3650f034b13b70cc1f84e23eb49fd427fab0c76210fb1e357b33a'
    assert hashlib.sha256((root/'src/workflow/stage1_open_action_value.py').read_bytes()).hexdigest()==plan['recovery_sidecheck']['source_code_sha256']
    raw,ra=_source(root,plan['raw_source'],plan['raw_artifacts']);cand,ca=_source(root,plan['candidate_proposals'],plan['candidate_artifacts']);strong,sa=_source(root,plan['strong_proposals'],plan['strong_artifacts']);ctl,cta=_source(root,plan['controller_source'],plan['controller_artifacts']);action,aa=_source(root,plan['action_source'],plan['action_artifacts'])
    frame=raw['inputs/frame'].reset_index(drop=True);assert len(frame)==944 and np.isfinite(frame.soxx_trend20).all()
    selection=action['controller/source_recovery_selection'];recovery_spec=action['inputs/plan']['recovery_spec'];present=ctl['inputs/controller'].get('frozen_recovery_model') is not None
    families=['raw_candidate27baa','raw_strong9097'];costs=[0,2,5];daily={};funnel=[];conditional=[];periods=[];rolling={};winloss=[];eligibility_parameters={'model_present':present,'drawdown_floor':recovery_spec['recovery_eligibility_drawdown'],'action_beta':selection['action_beta'],'minimum_source_alpha_probability':selection['minimum_source_alpha_probability'],'probability_threshold':selection['probability_threshold']}
    for family in families:
        proposal=cand['analysis/targets']['candidate'].reindex(pd.DatetimeIndex(frame.signal_date)) if family=='raw_candidate27baa' else strong['inputs/frame'].set_index('signal_date').reindex(pd.DatetimeIndex(frame.signal_date))
        assert proposal[['soxl_weight','soxx_weight','cash_weight','alpha_probability']].notna().all().all()
        for cost in costs:
            book=raw[f'native/{family}_{cost}'];path=book['authority'].reset_index(drop=True);bench=raw[f'native/raw_SOXX_{cost}']['authority'].reset_index(drop=True)
            for x in [path,bench]:pd.testing.assert_frame_equal(x[['signal_date','trade_date','exit_date']],frame[['signal_date','trade_date','exit_date']])
            other_family='raw_strong9097' if family=='raw_candidate27baa' else 'raw_candidate27baa'
            other=raw[f'native/{other_family}_{cost}']['authority'].reset_index(drop=True)
            trace=raw['traces/'+family];trace=trace.loc[trace.slippage_bps.eq(cost)].set_index('trade_date').reindex(pd.DatetimeIndex(frame.trade_date))
            assert trace.common_cap.notna().all();np.testing.assert_array_equal(path.current_beta_cap,trace.common_cap)
            stamps=pd.DatetimeIndex(frame.trade_date)+pd.Timedelta('9h47min');point=book['points'].set_index('stamp').reindex(stamps);prices=raw['inputs/raw_points'].reindex(stamps)
            assert point.nav.notna().all() and prices.notna().all().all()
            requested=book['targets'].reset_index(drop=True);pd.testing.assert_series_equal(requested.trade_date,frame.trade_date,check_names=False)
            p=proposal.reset_index(drop=True);d=frame[['signal_date','trade_date','exit_date','soxx_trend20','early_ret_soxx']].copy()
            d['prior_trend']=np.where(d.soxx_trend20.gt(0),'positive','nonpositive');d['risk_state']=path.risk_state;d['risk_clock']=path.risk_clock
            d['proposal_beta']=3*p.soxl_weight+p.soxx_weight;d['proposal_cash']=p.cash_weight;d['source_alpha']=p.alpha_probability
            d['requested_beta']=3*requested.soxl_weight+requested.soxx_weight;d['requested_cash']=requested.cash_weight
            d['actual_open_beta']=(3*point.q_soxl.to_numpy()*prices.SOXL.to_numpy()+point.q_soxx.to_numpy()*prices.SOXX.to_numpy())/point.nav.to_numpy()
            d['actual_spendable_cash_weight']=point.cash.to_numpy()/point.nav.to_numpy();d['actual_receivable_weight']=point.receivable.to_numpy()/point.nav.to_numpy()
            d['cap_reduction']=d.proposal_beta-d.requested_beta;d['execution_beta_deviation']=d.requested_beta-d.actual_open_beta
            d['common_cap']=trace.common_cap.to_numpy();d['own_opening_drawdown']=path.portfolio_drawdown
            d['relative_log']=np.log1p(path.strategy_return)-np.log1p(bench.strategy_return)
            pair=np.log1p(path.strategy_return)-np.log1p(other.strategy_return)
            d['candidate_minus_strong_log']=pair if family=='raw_candidate27baa' else -pair
            d['benchmark_interval_up']=bench.strategy_return.gt(0).to_numpy();d['excess_sign']=np.select([d.relative_log.gt(0),d.relative_log.lt(0)],['positive','negative'],default='zero')
            d['saved_probability']=trace.recovery_probability.to_numpy();d['saved_recovery_action']=path.recovery_action.astype(bool)
            d['finite_early_diagnostic']=np.isfinite(d.early_ret_soxx);d['target_beta']=np.minimum(d.proposal_beta,float(selection['action_beta']))
            predicates={'model_present':np.full(len(d),present),'drawdown_floor':d.own_opening_drawdown.ge(float(recovery_spec['recovery_eligibility_drawdown'])),'target_above_COMMONcap':d.target_beta.gt(d.common_cap+1e-12),'early_nonnull_runtime':d.early_ret_soxx.notna(),'alpha_nonnull_runtime':d.source_alpha.notna(),'alpha_floor':d.source_alpha.ge(float(selection['minimum_source_alpha_probability']))}
            cumulative=np.ones(len(d),dtype=bool)
            for name,mask in predicates.items():
                mask=np.asarray(mask,bool);d['predicate_'+name]=mask;cumulative &= mask
                funnel.append({'family':family,'slippage_bps':cost,'stage':name,'marginal_pass':int(mask.sum()),'ordered_remaining':int(cumulative.sum()),'total':len(d)})
            np.testing.assert_array_equal(cumulative,d.saved_probability.notna().to_numpy())
            approval=cumulative & d.saved_probability.ge(float(selection['probability_threshold'])).to_numpy();d['eligible']=cumulative;d['local_approval']=approval
            funnel.append({'family':family,'slippage_bps':cost,'stage':'classifier_threshold','marginal_pass':int(d.saved_probability.ge(float(selection['probability_threshold'])).sum()),'ordered_remaining':int(approval.sum()),'total':len(d),'eligible_probability_median':float(d.saved_probability.median()),'eligible_probability_max':float(d.saved_probability.max())})
            for trend,up,sign in itertools.product(['positive','nonpositive'],[True,False],['positive','negative','zero']):
                g=d.loc[d.prior_trend.eq(trend)&d.benchmark_interval_up.eq(up)&d.excess_sign.eq(sign)]
                conditional.append({'family':family,'slippage_bps':cost,'prior_trend':trend,'benchmark_interval_up':up,'excess_sign':sign,'labels_are_expost_not_trade_signals':True,**summarize(g)})
            for kind,labels in [('year',d.trade_date.dt.to_period('Y')),('quarter',d.trade_date.dt.to_period('Q'))]:
                for period in labels.unique():periods.append({'family':family,'slippage_bps':cost,'kind':kind,'period':str(period),'complete_period':bool(period not in [labels.iloc[0],labels.iloc[-1]]),**summarize(d.loc[labels.eq(period)])})
            windows=pd.DataFrame({'first_trade':d.trade_date.iloc[:-125].to_numpy(),'last_exit':d.exit_date.iloc[125:].to_numpy(),'relative_log':d.relative_log.rolling(126).sum().iloc[125:].to_numpy(),'candidate_minus_strong_log':d.candidate_minus_strong_log.rolling(126).sum().iloc[125:].to_numpy()})
            for col in ['proposal_beta','proposal_cash','requested_beta','requested_cash','actual_open_beta','actual_spendable_cash_weight','actual_receivable_weight','cap_reduction','execution_beta_deviation']:windows['mean_'+col]=d[col].rolling(126).mean().iloc[125:].to_numpy()
            windows['fraction_nonpositive_trend']=d.prior_trend.eq('nonpositive').rolling(126).mean().iloc[125:].to_numpy()
            for state in [0,1,2,3]:windows['fraction_risk_state_'+str(state)]=d.risk_state.eq(state).rolling(126).mean().iloc[125:].to_numpy()
            assert len(windows)==819
            windows['sign']=np.select([windows.relative_log.gt(0),windows.relative_log.lt(0)],['positive','negative'],default='zero')
            for sign in ['positive','negative','zero']:
                g=windows.loc[windows.sign.eq(sign)];winloss.append({'family':family,'slippage_bps':cost,'window_sign':sign,'windows':len(g),'overlapping_not_independent':True,**{col:float(g[col].mean()) if len(g) else None for col in windows if col.startswith('mean_') or col.startswith('fraction_')}})
            daily[(family,cost)]=d;rolling[(family,cost)]=windows
        consensus=np.logical_and.reduce([daily[(family,cost)].local_approval.to_numpy() for cost in costs])
        for cost in costs:np.testing.assert_array_equal(consensus,daily[(family,cost)].saved_recovery_action.to_numpy())
    exposure=[]
    for (family,cost),d in daily.items():
        for name,labels in [('prior_trend',d.prior_trend),('risk_state',d.risk_state)]:
            for value in labels.unique():exposure.append({'family':family,'slippage_bps':cost,'stratum':name,'value':str(value),'noncontiguous_not_causal':True,**summarize(d.loc[labels.eq(value)])})
    outputs={'exposure_summary':pd.DataFrame(exposure),'conditional_win_loss':pd.DataFrame(conditional),'period_summary':pd.DataFrame(periods),'rolling_win_loss':pd.DataFrame(winloss),'recovery_funnel':pd.DataFrame(funnel)}
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='saved_raw_exposure_only',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_inference=0,new_fits=0,new_replays=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audits={'raw':ra,'candidate':ca,'strong':sa,'controller':cta,'action':aa},recovery_parameters=eligibility_parameters)
        rec.save_objects(artifact_path='sources',execution_workflow=Path(__file__).read_text(),closed_route_notes={f:(root/f).read_text() for f in plan['dedup_sources']})
        for (family,cost),d in daily.items():rec.save_objects(artifact_path='analysis/daily',**{f'{family}_{cost}':d});rec.save_objects(artifact_path='analysis/rolling',**{f'{family}_{cost}':rolling[(family,cost)]})
        for name,value in outputs.items():value.to_csv(card/(name+'.csv'),index=False);rec.save_objects(artifact_path='analysis',**{name:value})
        policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/daily/*','analysis/rolling/*']}
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'descriptive_raw_exposure_diagnosis_no_policy_claim','objective_achieved':False,'sessions_per_family_cost':944,'windows_per_family_cost':819,'costs':costs,'funnel_reconciled_all_six_own_paths':True,'ALL3_recovery_consensus_reconciled':True,'new_inference_calls':0,'new_fits':0,'new_replays':0,'new_data_or_price_queries':0,'recovery_parameters':eligibility_parameters,'closed_recovery_score_route_not_reopened':True,'next_action':'Use completeexposure/winloss findings tofreezeone genuinelydistinctmodelmechanism, notoldscorethreshold/refit rescue. No nextfit authorized bydiagnosis.','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(root/plan['provider_uri']),'artifact_policy':policy}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),diagnosed_own_paths=6)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['raw_source']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json');(card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False));summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary
