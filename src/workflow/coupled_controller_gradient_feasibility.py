"""Fixed zero-optimizer mechanical qualification; never a performance study."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.workflow.minute_action_value import _source
from src.workflow.beta_preserving_composition import verify_marks
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.backtest.causal_cost_scenario_controller import replay_cost_scenarios
from src.model.coupled_controller_gradient_probe import frozen_logits,raw_weights,reference_targets,smooth_probe,tensor_probe,reference_objective
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('mechanical plan changed')
        return old
    sources=plan['sources'];data,a1=_source(root,sources['training_frame'],['inputs/full_reference_frame'])
    models,a2=_source(root,sources['tangent'],plan['frozen_models'])
    inputs,a3=_source(root,sources['training_features'],['inputs/features','inputs/common','inputs/daily_frame'])
    policy,a4=_source(root,sources['coupled_reference'],['inputs/plan'])
    controller,a5=_source(root,sources['controller'],['inputs/controller'])
    action,a6=_source(root,sources['action'],['inputs/plan','controller/source_recovery_selection'])
    benchmark,a7=_source(root,sources['benchmark'],['analysis/native/soxx_2'])
    full=data['inputs/full_reference_frame'];mature=pd.to_datetime(full.label_available_at).dt.tz_localize('America/New_York')<pd.Timestamp('2023-01-01',tz='America/New_York')
    frame=full.loc[mature].tail(189).copy().reset_index(drop=True)
    if len(frame)!=189 or [str(frame.trade_date.iloc[i].date()) for i in [0,62,188]]!=['2022-03-31','2022-06-30','2022-12-29']:raise ValueError('fixed maturity window differs')
    book=benchmark['analysis/native/soxx_2'];report=book['native_report'];report=report.loc[(report.index>=frame.trade_date.iloc[0]+pd.Timedelta(hours=9,minutes=47))&(report.index<=frame.trade_date.iloc[-1]+pd.Timedelta(hours=16))]
    mark_audit=verify_marks(frame,{**book,'native_report':report})
    frozen=[models[name] for name in plan['frozen_models']];original=frozen[0]
    dates=pd.DatetimeIndex(frame.signal_date)
    daily=inputs['inputs/daily_frame'];daily=daily.set_index('signal_date') if 'signal_date' in daily else daily
    returns=daily.reindex(dates)[['soxl_forward_open_return','soxx_forward_open_return']].to_numpy(float)
    if not np.isfinite(returns).all():raise ValueError('original smoothing H1 returns unavailable')
    logits=frozen_logits(frozen,inputs['inputs/features'],inputs['inputs/common'],dates)
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**action['inputs/plan']['recovery_spec']);action_spec=Stage1OpenActionValueSpec(root=root,**action['inputs/plan']['action_spec']);selection=action['controller/source_recovery_selection']
    # Final2022 controller in earlier prices is explicitly in-sample mechanics.
    clock=pd.DatetimeIndex(book['native_report'].index);last=pd.Series(clock,index=clock.normalize()).groupby(level=0).last();endpoints=[]
    for i in [62,188]:
        pos=book['native_positions'][last.loc[frame.trade_date.iloc[i]]];q=np.array([pos.get_stock_amount(s) if pos.check_stock(s) else 0. for s in ['SOXL','SOXX']]);cash=pos.get_cash();close=frame.iloc[i][['close_soxl','close_soxx']].to_numpy(float)
        endpoints.append({'trade_date':str(frame.trade_date.iloc[i].date()),'close_stamp':str(last.loc[frame.trade_date.iloc[i]]),'close_nav':float(q@close+cash)})
    B=float(np.log(endpoints[1]['close_nav']/endpoints[0]['close_nav']))
    progress_path=card/'run_state.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'plan_sha256':digest,'reference_attempts':[],'reference_completed':[],'tensor_attempted':False,'tensor_completed':False}
    if progress['plan_sha256']!=digest:raise ValueError('attempt identity differs')
    tracking=card/'qlib_recorder';kwargs={'experiment_name':plan['experiment'],'uri':str(tracking)}
    if progress.get('recorder_id'):kwargs.update(recorder_id=progress['recorder_id'],resume=True)
    def save():atomic_json(progress,progress_path)
    with R.start(**kwargs):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));save()
        if not progress.get('source_saved'):
            rec.save_objects(artifact_path='inputs',plan=plan,frame=frame,source_audits=[a1,a2,a3,a4,a5,a6,a7],mark_audit=mark_audit,benchmark_endpoints=endpoints,B=B,original_smoothing_returns=returns)
            rec.save_objects(artifact_path='sources',execution_workflow=Path(__file__).read_text(),probe=(root/'src/model/coupled_controller_gradient_probe.py').read_text(),reference=(root/'src/backtest/causal_cost_scenario_controller.py').read_text(),tensor_fee=(root/'src/model/full_path_tiered.py').read_text(),actual_fee=(root/'src/backtest/ibkr_tiered.py').read_text(),smoothing=(root/'src/workflow/pit_constituent_auxiliary_training_allocation.py').read_text())
            progress['source_saved']=True;save()
        def reference(name,theta):
            if name in progress['reference_completed']:return rec.load_object('analysis/reference/'+name)
            if name in progress['reference_attempts'] or len(progress['reference_attempts'])>=9:raise ValueError('reference attempt budget exhausted; no rerun')
            progress['reference_attempts'].append(name);save()
            raw,targets=reference_targets(logits,theta,original.spec,original.candidate,dates,returns)
            prepared=frame.copy()
            for column in targets:prepared[column]=targets[column].to_numpy()
            paths,trace=replay_cost_scenarios(prepared,plan=policy['inputs/plan'],controller=controller['inputs/controller'],recovery_spec=recovery,action_spec=action_spec,selection=selection)
            J,close=reference_objective(paths,frame,B)
            value={'raw':raw,'targets':targets,'paths':paths,'trace':trace,'J':J,'primary_close_nav':close}
            rec.save_objects(artifact_path='analysis/reference',**{name:value});progress['reference_completed'].append(name);save();print('completed fixed reference',name,flush=True)
            return value
        failure=None;fd_rows=[];parity={};coverage={};gradient=None
        try:
            baseline=reference('baseline',[0.,0.])
            if progress['tensor_attempted']:raise ValueError('tensor attempt already used; no rerun')
            progress['tensor_attempted']=True;save()
            theta=torch.zeros(2,dtype=torch.float32,requires_grad=True);raw=raw_weights(logits,theta,original.spec)
            weights,accepted=smooth_probe(raw,baseline['raw'],returns,original.spec,original.candidate)
            smoothing_error=float(np.max(abs(weights.detach().numpy()-baseline['targets'][['soxl_weight','soxx_weight','cash_weight']].to_numpy())))
            J,audit,coverage=tensor_probe(frame,weights,baseline['paths'],baseline['trace'],recovery,selection,policy['inputs/plan']['tariff'],B)
            gradient=torch.autograd.grad(J,theta)[0].detach().double().numpy();progress['tensor_completed']=True;save()
            parity={'smoothing_max_error':smoothing_error,'target_max_error':float(audit.target_max_error.max()),'local_cap_max_error':float(audit.local_cap_error.max()),'common_cap_max_error':float(audit.common_cap_error.max()),'known_nav_relative_error':float(audit.known_nav_relative_error.max()),'objective_absolute_error':abs(float(J.detach())-baseline['J']),
                'enforced_q_error':float(audit.q_max_error.max()),'enforced_cash_error':float(audit.cash_error.max()),'enforced_monthly_counter_error':float(audit.monthly_error.max()),'ledger_parity_role':'q/cash/monthly/orderfees inherited via exact-forward correction, NOT independent accounting validation'}
            parity['passed']=bool(max(parity[k] for k in ['smoothing_max_error','target_max_error','local_cap_max_error','common_cap_max_error'])<=1e-10 and parity['known_nav_relative_error']<=1e-7 and parity['objective_absolute_error']<=1e-8 and np.isfinite(gradient).all())
            rec.save_objects(artifact_path='analysis',tensor_audit=audit,parity=parity,gradient=gradient,branch_coverage=coverage)
            if not parity['passed']:raise ValueError('fixed surrogate forward/numerical gate failed')
            for direction in [0,1]:
                for epsilon in plan['finite_differences']['epsilons']:
                    pair={}
                    for sign,label in [(-1,'minus'),(1,'plus')]:
                        point=[0.,0.];point[direction]=sign*epsilon;value=reference(f'd{direction+1}_{label}_{epsilon}',point);pair[label]=value
                    slope=(pair['plus']['J']-pair['minus']['J'])/(2*epsilon);tolerance=max(1e-5,.10*abs(slope));agreement=abs(gradient[direction]-slope)<=tolerance and (abs(slope)<=1e-5 or np.sign(gradient[direction])==np.sign(slope))
                    changes={}
                    for label in ['minus','plus']:
                        t=pair[label]['trace'];b=baseline['trace'];changes[label+'_state_clock_changes']=int((t.risk_state.ne(b.risk_state)|t.risk_clock.ne(b.risk_clock)).sum());changes[label+'_recovery_vote_changes']=int(t.recovery_local_approval.ne(b.recovery_local_approval).sum());changes[label+'_cap_value_changes']=int((t.common_cap-b.common_cap).abs().gt(1e-10).sum())
                    fd_rows.append({'direction':direction+1,'epsilon':epsilon,'autograd':float(gradient[direction]),'finite_difference':float(slope),'absolute_error':abs(float(gradient[direction])-slope),'allowed_error':tolerance,'agreement':bool(agreement),**changes})
            fd=pd.DataFrame(fd_rows);consistency=[]
            for _,part in fd.groupby('direction'):
                a,b=part.finite_difference.to_numpy();consistency.append(abs(a-b)<=max(1e-5,.1*max(abs(a),abs(b))))
            informative=any(part.finite_difference.abs().gt(1e-5).all() for _,part in fd.groupby('direction'))
            passed=bool(fd.agreement.all() and all(consistency) and informative)
            if not passed:failure='fixed surrogate finite-difference/scale/informative gate failed; no rescue'
        except Exception as error:
            failure=str(error);passed=False
        fd=pd.DataFrame(fd_rows);policy_meta={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/reference/*','analysis/tensor_audit']}
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'limited_exercised_branch_mechanical_pass_no_training_authorization' if passed else 'fixed_surrogate_rejected_no_optimization','objective_achieved':False,'mechanical_gate_pass':passed,'failure':failure,'optimizer_steps':0,'economic_model_fits':0,'native_accounts':0,'attempted_reference_shadow_paths':3*len(progress['reference_attempts']),'completed_reference_shadow_paths':3*len(progress['reference_completed']),'attempted_tensor_paths':3*int(progress['tensor_attempted']),'completed_tensor_paths':3*int(progress['tensor_completed']),'parity':parity,'branch_coverage':coverage,'temporal_role':plan['temporal_role'],'provider_uri':str(root/plan['provider_uri']),'region':'us','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy_meta}
        rec.save_objects(artifact_path='analysis',finite_differences=fd,summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(x.is_file() for x in artifacts.rglob('*')),reference_paths=3*len(progress['reference_completed']))
    atomic_json(parity,card/'parity.json');atomic_json(coverage,card/'branch_coverage.json');atomic_json({'B':B,'source':plan['sources']['benchmark'],'artifact':'analysis/native/soxx_2','endpoints':endpoints},card/'benchmark_endpoints.json');fd.to_csv(card/'finite_differences.csv',index=False)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['sources']['coupled_reference']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy_meta},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary
