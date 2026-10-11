"""A fixed, notebook-invoked matched information experiment on native Qlib evidence."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.core.recorder_run import start_evidence_run,archive_evidence,publish_evidence_manifest
from src.data.information_increment import bind_matched_information_datasets
from src.data.convex_allocation import LABEL_COLUMNS


def _require(value,message):
    if not value:
        raise ValueError('blocked: '+message)


def verify_frozen_inputs(root,card,plan):
    from src.data.provider_content_hash import provider_content_hash
    _require(set(plan['providers'])=={'price','execution','information'},'complete provider inventory required')
    for name,expected in plan['frozen_files_sha256'].items():
        _require(hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()==expected,'frozen file differs: '+name)
    for identity in plan['providers'].values():
        _require(provider_content_hash(Path(root)/identity['provider_uri'])[0]==identity['content_sha256'],'provider differs')
    prereg_path=plan['preregister_path']
    _require(prereg_path in plan['frozen_files_sha256'],'preregistration must be hash bound')
    prereg=json.loads((Path(root)/prereg_path).read_text())
    for key,value in prereg.items():
        if key not in {'status','authorization'}:
            _require(plan.get(key)==value,'execution changes preregistered field: '+key)
    _require(plan['source_deployment_gate'] is False and plan['authorization'].get('orders') is False,
             'research may not sign source deployment or orders')
    _require(plan['authorization']['fit'] is True and plan['authorization']['replay'] is True,'fixed execution not authorized')
    _require(plan['horizon_sessions']==5 and plan['costs_bps']==[10.,25.] and plan['robust_radius']==1.,'fixed reference method differs')
    _require(plan['fixed_fit_budget']==len(plan['outer_years'])*2*5 and plan['fixed_native_cases']==12,'budget/inventory differs')
    _require(plan['minimum_training_rows']==252 and plan['minimum_validation_rows']==126 and plan['hac_lags']==5,'reference estimation contract differs')


def run_matched_information_experiment(*,root,card,execution_plan_path):
    import os
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
    import qlib
    from src.core.recorder_sources import load_verified_recorder_objects
    from src.model.generic_annual_ridge_selection import select_annual_horizon
    from src.backtest.generic_horizon_window import replay_horizon_window,native_verify_window
    from src.workflow.benchmark_evidence import evaluate_registered_paths
    from src.analysis.excess_resolution import apply_excess_resolution
    from src.analysis.forecast_information_increment import paired_forecast_information
    root=Path(root).resolve();card=Path(card).resolve()
    plan_bytes=Path(execution_plan_path).read_bytes()
    plan=json.loads(plan_bytes);verify_frozen_inputs(root,card,plan)
    attempt=card/'execution_attempt.json'
    _require(not attempt.exists(),'one-shot execution was already attempted; no refit')
    qlib.init(provider_uri=str(root/plan['providers']['price']['provider_uri']),region='us',kernels=1)
    prefit=plan['prefit']
    objects,_=load_verified_recorder_objects(card,expected_recorder_id=prefit['recorder_id'],
        artifacts=['prefit/inputs.pkl','prefit/qualification.pkl'],expected_artifact_sha256=prefit['object_sha256'])
    bundle=bind_matched_information_datasets(objects['prefit/inputs.pkl'],as_of=plan['as_of'])
    frame,calendar,covariance=bundle['frame'],bundle['calendar'],bundle['covariance']
    with attempt.open('x') as f:json.dump(dict(status='starting',fit_calls=0,native_cases=0),f)
    identity=start_evidence_run(experiment=plan['experiment'],uri=card/'execution_recorder',name='fixed_matched_information',
        objects={'inputs.pkl':bundle,'execution_plan.pkl':plan},params=dict(horizon_sessions=5,
            fixed_fit_budget=plan['fixed_fit_budget'],fixed_native_cases=12,information_field=plan['information_field']))
    fits={};selections={a:{} for a in ('control','candidate')};cases={};case_artifacts={};fit_calls=0;stage='fit'
    def checkpoint():
        attempt.write_text(json.dumps(dict(status='running',stage=stage,fit_calls=fit_calls,native_cases=len(case_artifacts),**identity),indent=2)+'\n')
        archive_evidence(identity,{'selections.pkl':selections,'completed_fits.pkl':fits,'case_artifacts.pkl':case_artifacts})
    try:
        for year in plan['outer_years']:
            for arm in ('control','candidate'):
                stage=f'fit_{year}_{arm}'
                def fitted(kind,name,model):
                    nonlocal fit_calls
                    fits[(arm,year,kind,name)]=model;fit_calls+=1
                result=select_annual_horizon(bundle[arm],frame,outer_year=year,horizon_sessions=5,
                    feature_columns=bundle[arm+'_features'],label_columns=LABEL_COLUMNS,
                    availability_column='h5_available_at',eligibility_column='source_known',
                    minimum_training_rows=plan['minimum_training_rows'],minimum_validation_rows=plan['minimum_validation_rows'],
                    hac_lags=plan['hac_lags'],train_start=plan['train_start'],fit_callback=fitted)
                selections[arm][year]=result
                print(f'{arm} {year}: {result["selected_name"]}; fitted {fit_calls}',flush=True)
            for key in ('inner_identity','outer_identity','validation_identity'):
                _require(selections['control'][year]['audit'][key]==selections['candidate'][year]['audit'][key],'fitted arm identities differ')
            checkpoint()
        _require(fit_calls==plan['fixed_fit_budget'],'actual fit budget differs')
        unsettled=pd.to_datetime(frame.exit_date).isna().to_numpy()
        _require(int(unsettled.sum())==2 and unsettled[-2:].all(),'only terminal2 unsettled rows allowed')
        windows={};paths={};forecasts={}
        for name,years in [('development',plan['development_years']),('reused_diagnostic',plan['diagnostic_years'])]:
            mask=frame.index.year.isin(years)&~unsettled
            windows[name]={'frame':frame.loc[mask].reset_index(drop=True),'covariance':covariance[mask]}
        for window,parts in windows.items():
            for cost in plan['costs_bps']:
                for arm in ('control','candidate','soxx'):
                    key=f'{window}::{arm}::{int(cost)}';stage=key
                    result=replay_horizon_window(parts['frame'],{} if arm=='soxx' else selections[arm],parts['covariance'],
                        calendar=calendar,cost_bps=cost,horizon_sessions=5,robust_radius=plan['robust_radius'],
                        initial_cash=plan['initial_cash'],benchmark_only=arm=='soxx',feature_columns=bundle.get(arm+'_features',()),
                        availability_column='h5_available_at',eligibility_column='source_known',fallback_years=tuple(plan['fallback_years']))
                    cases[key]=result
                    result=native_verify_window(parts['frame'],result,provider_uri=root/plan['providers']['execution']['provider_uri'],
                        cost_bps=cost,initial_cash=plan['initial_cash'])
                    _require(result['native']['actual_position_audit']['status']=='passed','native cash/quantity audit failed')
                    cases[key]=result;paths[(window,arm,cost)]=result['path']
                    if arm!='soxx':
                        if (window,arm) in forecasts:pd.testing.assert_frame_equal(forecasts[(window,arm)],result['forecast'])
                        forecasts[(window,arm)]=result['forecast']
                    leaf=key.replace('::','__')+'.pkl.gz'
                    case_artifacts[key]=archive_evidence(identity,{leaf:result},artifact_path='cases',compressed=True)[leaf]
                    print('native passed '+key,flush=True);checkpoint()
        expected={f'{w}::{a}::{int(c)}' for w in windows for a in ('control','candidate','soxx') for c in plan['costs_bps']}
        _require(set(cases)==set(case_artifacts)==expected,'native case inventory differs')
        stage='statistics'
        combined={(arm,c):pd.concat([paths[(w,arm,c)] for w in windows],ignore_index=True)
                  for arm in ('control','candidate','soxx') for c in plan['costs_bps']}
        registered=dict(development_years=plan['development_years'],costs_bps=plan['costs_bps'],as_of=plan['as_of'],
            variants=[{'name':'candidate'},{'name':'control'}],candidate_budget=1,gates=plan['gates'])
        tables,evidence=evaluate_registered_paths(combined,plan=registered,calendar=calendar,candidate_controls={'candidate':'control'})
        dev={c:paths[('development','candidate',c)] for c in plan['costs_bps']};bench={c:paths[('development','soxx',c)] for c in plan['costs_bps']}
        resolved=apply_excess_resolution(evidence['candidate'],dev,bench,log_excess_floor=plan['log_excess_floor'])
        tables['raw_gates.csv']=tables['gates.csv'];tables['gates.csv']=resolved['gates']
        tables['resolved_annual_metrics.csv']=resolved['annual'];tables['bootstrap_analysis.csv']=resolved['bootstrap'];tables['rolling_summary.csv']=resolved['rolling_summary']
        diagnostic=tables['backtest_benchmark_comparison.csv'].query("strategy=='candidate' and window=='reused_diagnostic'")
        diag_pass=bool((diagnostic.log_excess_sum>plan['log_excess_floor']).all() and
            (diagnostic.max_drawdown_pct>=diagnostic.benchmark_max_drawdown_pct-1e-10).all() and
            (diagnostic.cvar5_pct>=diagnostic.benchmark_cvar5_pct-1e-10).all())
        portfolio_increment=bool((tables['ablation_comparison.csv'].log_excess_sum>plan['log_excess_floor']).all())
        info,information_pass=paired_forecast_information(pd.concat([forecasts[(w,'candidate')] for w in windows]),
            pd.concat([forecasts[(w,'control')] for w in windows]),cutoff=pd.Timestamp(plan['as_of']+' 16:15',tz='America/New_York'),
            windows={'development':plan['development_years'],'reused_diagnostic':plan['diagnostic_years']},resolution=plan['log_excess_floor'])
        tables['forecast_information.csv']=info
        tables['model_selection.csv']=pd.DataFrame([dict(arm=a,year=y,selected=r['selected_name']) for a,years in selections.items() for y,r in years.items()])
        tables['native_audit.csv']=pd.DataFrame([dict(case=k,**r['native']['actual_position_audit']) for k,r in cases.items()])
        tables['risk_analysis.csv']=pd.DataFrame([dict(case=k,metric=str(m),value=float(row['risk'])) for k,r in cases.items() for m,row in r['native']['risk_analysis_next_open'].iterrows()])
        verify_frozen_inputs(root,card,plan)
        _require(Path(execution_plan_path).read_bytes()==plan_bytes,'execution plan changed during run')
        all_pass=bool(resolved['historical_gate_pass'] and diag_pass and portfolio_increment and information_pass)
        policy=dict(policy='slim_card_recorder_first',workflow_config='workflow.yaml',recorder_only_artifacts=[
            'inputs/inputs.pkl','inputs/execution_plan.pkl','aggregate/selections.pkl','aggregate/completed_fits.pkl',
            'aggregate/case_artifacts.pkl','cases/*.pkl.gz','aggregate/tables.pkl','aggregate/raw_evidence.pkl',
            'aggregate/resolved_evidence.pkl','aggregate/final_summary.pkl'])
        summary=dict(identity,status='complete_not_promoted',promotion_decision='not_promoted',promotion_eligible=False,
            fit_calls=fit_calls,native_cases=12,historical_gate_pass=bool(resolved['historical_gate_pass']),diagnostic_gate_pass=diag_pass,
            incremental_portfolio_pass=portfolio_increment,incremental_information_pass=information_pass,
            route_decision='historical_pass_requires_independent_source_and_forward_evidence' if all_pass else 'end_original_guidance_route_no_further_search',
            source=dict(providers=plan['providers'],provider_uri=plan['providers']['price']['provider_uri'],
                execution_provider_uri=plan['providers']['execution']['provider_uri'],
                information_provider_uri=plan['providers']['information']['provider_uri'],region='us',as_of=plan['as_of']),
            source_deployment_gate=plan['source_deployment_gate'],recorder_status='FINISHED',
            execution_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
            artifact_policy=policy,compressed_case_artifacts=case_artifacts,scientific_scope=plan['scientific_scope'],
            qlib_docs=['https://qlib.readthedocs.io/en/latest/component/data.html',
                'https://qlib.readthedocs.io/en/latest/component/recorder.html',
                'https://qlib.readthedocs.io/en/latest/component/strategy.html'])
        archive_evidence(identity,{'tables.pkl':tables,'raw_evidence.pkl':evidence,'resolved_evidence.pkl':resolved,'final_summary.pkl':summary})
        for name,table in tables.items():table.to_csv(card/name,index=False)
        publish_evidence_manifest(card,summary)
        (card/'workflow.yaml').write_text(yaml.safe_dump(dict(experiment=plan['experiment'],source=summary['source'],recorder_id=summary['recorder_id'],recorder_uri=summary['recorder_uri'],artifact_policy=policy),sort_keys=False))
        (card/'selected_model.json').write_text(json.dumps(dict(model='native_nested_annual_prior_ridge',selections=tables['model_selection.csv'].to_dict('records'),promotion_decision='not_promoted'),indent=2)+'\n')
        lines=['# 原始指引单信息增量：结果','',f'固定 {fit_calls} 次拟合、12个原生现金/数量/费用场景完成。控制25价格特征与候选26特征的严格成熟日期、标签身份相同。','']
        for row in tables['backtest_benchmark_comparison.csv'].query("strategy=='candidate'").itertuples():
            lines.append(f'- {row.window}/{row.cost_bps:g}bp：净收益 {row.return_pct:.2f}%，相对SOXX终值 {row.relative_wealth:.6f}，最大回撤 {row.max_drawdown_pct:.2f}%。')
        lines+=['',f'历史门 {summary["historical_gate_pass"]}；复用诊断门 {diag_pass}；对匹配控制组净组合增量 {portfolio_increment}；固定超额响应MSE增量 {information_pass}。',
            f'不晋升；路线：{summary["route_decision"]}。',
            '日期保守的原始季度指导量，不是最新公司预测。公司并购口径原样保留，不用未来公司实际或consensus。历史文本快照与秒级首次公开仍有限制；63个季度信息观测不等于3956个独立样本。',
            'IC只是描述，不替代固定MSE及净超额门。费用改变优化动作时不同费用可能产生不同路径，不能解释为同路径加费改善。',
            '失败即停止原始指引路线，不改符号/alpha/期限/阈值。即便历史门全过，仍需源身份与至少60日新独立前向证据。',f'Recorder {summary["recorder_id"]}；FINISHED。']
        (card/'conclusion.md').write_text('\n'.join(lines)+'\n')
        attempt.write_text(json.dumps(dict(status='completed',fit_calls=fit_calls,native_cases=12,**identity),indent=2)+'\n')
        return summary
    except Exception as error:
        failure=dict(status='blocked',stage=stage,error=repr(error),fit_calls=fit_calls,native_cases=len(case_artifacts),**identity)
        archive_evidence(identity,{'completed_fits.pkl':fits,'selections.pkl':selections,'failure.pkl':failure})
        if cases:archive_evidence(identity,{'partial_cases.pkl.gz':cases},artifact_path='failure',compressed=True)
        attempt.write_text(json.dumps(failure,indent=2)+'\n')
        raise
