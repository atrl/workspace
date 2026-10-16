"""Frozen monthly direct-utility study with independent raw-share cost accounts."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd

NAME='etf_quarterly_profit_allocation_20261016'
PLAN_SHA='8c6814ae9a5b112993bb7b9a53526b47fe5efbd5c8d6850753f70c284eddf537'
DATES=['signal_date','trade_date','exit_date']
CONTEXT=['alpha_probability','tail_probability','early_ret_soxl','early_ret_soxx','early_range_soxl','early_range_soxx','early_from_low_soxl','early_from_low_soxx','early_volume_soxl','early_volume_soxx']
MARKET=['soxx_logmom63','soxx_logvol63','soxx_drawdown126','relative_logvol21','known_profit_age_mean_days','holdings_age_days','pair_coverage_weight','current_profit_coverage_weight']
PROFIT=['profit_direction_diffusion','profit_bounded_growth','quarterly_loss_weight']
FAMILIES=['candidate','control','primary9097','additional27baa','SOXX']
COSTS=[0,2,5]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,default=str)+'\n')


def isolated_risk_context(raw_frame,market_frame):
    """Do not inherit any27baa probability or unlisted field from physical-price frame."""
    raw=raw_frame.reset_index(drop=True);market=market_frame.reset_index(drop=True)
    if not raw[DATES].equals(market[DATES]):raise ValueError('risk context calendar differs')
    if not set(CONTEXT)<=set(market):raise ValueError('missing frozen9097 risk fields')
    out=raw[DATES].copy()
    for column in CONTEXT:out[column]=market[column].to_numpy(copy=True)
    audit={column:hashlib.sha256(pd.util.hash_pandas_object(out[column],index=False).values.tobytes()).hexdigest() for column in CONTEXT}
    return out,{'date_rows':len(out),'context_columns':CONTEXT,'final_columns':list(out),'source':'9097b0d6baae44b792ba7f7ec3e48d0b','column_sha256':audit}


def read_profit_source(root,binding):
    import qlib
    from qlib.data import D
    from src.core.recorder_sources import load_verified_recorder_objects
    card=root/binding['source_card']
    if sha(card/'summary.json')!=binding['summary_sha256']:raise ValueError('quarter source changed')
    for name,digest in binding['provider_hashes'].items():
        if sha(root/name)!=digest:raise ValueError('quarter provider changed')
    qlib.init(provider_uri=str(root/binding['provider_uri']),region='us',kernels=1)
    objects,audit=load_verified_recorder_objects(card,expected_recorder_id=binding['recorder_id'],expected_manifest_sha256=binding['manifest_sha256'],artifacts=[binding['lineage_artifact']])
    lineage=objects[binding['lineage_artifact']].copy();lineage['signal_at']=pd.to_datetime(lineage.signal_at,utc=True).dt.tz_convert('America/New_York')
    lineage.index=pd.DatetimeIndex(lineage.signal_at).tz_localize(None).normalize()
    if lineage.index.has_duplicates or not lineage.signal_at.dt.strftime('%H:%M').eq('16:15').all():raise ValueError('invalid quarter source clock')
    values=D.features([binding['instrument']],list(binding['field_map'].values()),start_time=lineage.index.min(),end_time=lineage.index.max()).xs(binding['instrument'],level='instrument').rename(columns={v:k for k,v in binding['field_map'].items()})
    if not lineage.eligible.isin([True,False,0,1]).all():raise ValueError('unknown quarterly eligibility')
    result=values.copy();result['source_qualified']=lineage.eligible.astype(bool)
    result['profit_direction_diffusion']=result.profit_improve_weight-result.profit_deteriorate_weight
    if not np.isfinite(result.loc[result.source_qualified,MARKET[4:]+PROFIT].to_numpy()).all():raise ValueError('qualified quarterly source missing values')
    return result,audit


def prepare(root):
    """Freeze source/data clocks and counts only: no economic labels or model calls."""
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    from src.data.annual_valuation_information import read_market_inputs
    from src.workflow.minute_action_value import _source
    from src.core.recorder_sources import load_verified_recorder_objects
    from src.backtest.raw_corporate_policy import make_schedule
    root=Path(root).resolve();card=root/'research'/NAME;plan=json.loads((card/'plan.json').read_text())
    review=json.loads((card/'design_review.json').read_text())
    if sha(card/'plan.json')!=PLAN_SHA or review.get('plan_sha256')!=PLAN_SHA:raise ValueError('frozen allocation design changed')
    for name,digest in plan['source_code_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('inherited policy implementation changed '+name)
    binding=json.loads((card/'source_binding.json').read_text());identity=json.loads((card/'execution_source_identity.json').read_text());prefix=json.loads((card/'raw_training_source_binding.json').read_text());market_plan=json.loads((card/'market_source_binding.json').read_text())
    for name,digest in {**identity,**prefix['provider_hashes']}.items():
        if sha(root/name)!=digest:raise ValueError('source or implementation identity changed: '+name)
    if (card/'prepared_state.json').exists():
        state=json.loads((card/'prepared_state.json').read_text())
        if state['source_binding_sha256']!=sha(card/'source_binding.json') or state['code_identity_sha256']!=sha(card/'execution_source_identity.json'):raise ValueError('prepared identity changed')
        return state
    basket,ba=read_profit_source(root,binding);monthly,_,_=read_market_inputs(root,market_plan)
    monthly=monthly.join(basket[['source_qualified',*MARKET[4:],*PROFIT]],how='left');monthly['source_qualified']=monthly.source_qualified.fillna(False).astype(bool)
    monthly.loc[~monthly.source_qualified,PROFIT]=np.nan
    monthly['market_complete']=np.isfinite(monthly[MARKET[:4]].to_numpy()).all(axis=1)
    counts=[]
    for year in [2024,2025,2026]:
        test=monthly.execution_year.eq(year);cutoff=monthly.loc[test,'signal_at'].iloc[0]
        train=monthly.source_qualified & monthly.market_complete & monthly.exit_at.lt(cutoff)
        counts.append({'execution_year':year,'cutoff':str(cutoff),'mature_qualified_rows':int(train.sum()),'training_entry_years':int(monthly.loc[train,'execution_year'].nunique()),'test_months':int(test.sum()),'fallback_months':int((~monthly.loc[test,'source_qualified']).sum())})
    evaluation=monthly.loc[monthly.execution_year.ge(2024)]
    if len(evaluation)!=31 or not evaluation.market_complete.all():raise ValueError('fixed31month evaluation missing')
    raw,ra=_source(root,plan['sources']['raw_cashbook'],plan['sources']['raw_cashbook']['artifacts'])
    primary,pa=_source(root,plan['sources']['primary_strong_reference_9097'],['inputs/frame','inputs/plan'])
    additional,aa=_source(root,plan['sources']['additional_comparator_27baa'],['inputs/frame','inputs/plan','analysis/targets'])
    required_targets=['soxl_weight','soxx_weight','cash_weight','alpha_probability','tail_probability']
    frozen_targets=additional['analysis/targets']['candidate']
    if not set(required_targets)<=set(frozen_targets):raise ValueError('27baa target fields incomplete before labels')
    target_dates=pd.DatetimeIndex(raw['inputs/frame'].signal_date)
    if not target_dates.isin(frozen_targets.index).all() or not np.isfinite(frozen_targets.reindex(target_dates)[required_targets].to_numpy()).all():raise ValueError('27baa frozen target coverage incomplete before labels')
    controller,cta=_source(root,plan['sources']['controller'],['inputs/controller','inputs/plan'])
    action,aca=_source(root,plan['sources']['action'],['inputs/plan','controller/source_recovery_selection'])
    unit,ua=_source(root,plan['sources']['unit_qualification'],['analysis/summary'])
    if not unit['analysis/summary']['all_unit_invariants_passed']:raise ValueError('raw unit qualification failed')
    corp=plan['sources']['corporate'];corporate,ca=load_verified_recorder_objects(root/corp['card'],expected_recorder_id=corp['recorder_id'],expected_manifest_sha256=corp['inventory_sha256'],artifacts=['data/cash_events.pkl','data/split_events.pkl','inputs/source_contract.pkl'])
    daily,context_audit=isolated_risk_context(raw['inputs/frame'],primary['inputs/frame'])
    if plan['daily_risk_context_contract']['required_9097_columns']!=CONTEXT:raise ValueError('context whitelist changed')
    if not raw['inputs/frame'][DATES].reset_index(drop=True).equals(additional['inputs/frame'][DATES].reset_index(drop=True)):raise ValueError('27baa calendar differs')
    start=pd.Timestamp(plan['core_clocks']['common_account_start']).tz_localize(None);end=pd.Timestamp(plan['core_clocks']['common_terminal']).tz_localize(None)
    mask=daily.trade_date.ge(start.normalize())&daily.exit_date.le(end.normalize());daily=daily.loc[mask].reset_index(drop=True)
    if daily.trade_date.iloc[0]!=start.normalize() or daily.exit_date.iloc[-1]!=end.normalize():raise ValueError('fresh account boundary missing')
    allpoints=raw['inputs/raw_points'];points=allpoints.loc[start.normalize()+pd.Timedelta('9h45min'):end].copy()
    if len(points)!=4*len(daily)+2 or not np.isfinite(points).all().all() or not points.gt(0).all().all():raise ValueError('raw execution point grid incomplete')
    guard_at=end+pd.Timedelta('45min');guard=allpoints.loc[[guard_at]].copy()
    sessions=pd.DatetimeIndex(points.index.normalize().unique());schedule=make_schedule(corporate['data/cash_events.pkl'],corporate['data/split_events.pkl'],sessions,daily.trade_date.iloc[0],daily.exit_date.iloc[-1])
    if raw['inputs/plan']['tariff']!=plan['accounts']['tariff'] or raw['inputs/plan']['initial_cash']!=100000:raise ValueError('frozen cost contract differs')
    qlib.init(provider_uri=str(root/prefix['provider_uri']),region='us',kernels=1)
    times=pd.DatetimeIndex(pd.concat([monthly.entry_at,monthly.exit_at]).unique()).tz_localize(None).sort_values()
    raw_endpoints=D.features(['SOXL','SOXX'],['$raw_execution_price'],start_time=times.min(),end_time=times.max(),freq='1min')['$raw_execution_price'].unstack('instrument').reindex(times)[['SOXL','SOXX']]
    if not np.isfinite(raw_endpoints).all().all() or not raw_endpoints.gt(0).all().all():raise ValueError('physical monthly source prefix missing')
    receipt={'plan_sha256':PLAN_SHA,'folds':counts,'all_folds_eligible':all(x['mature_qualified_rows']>=24 and x['training_entry_years']>=2 for x in counts),'monthly_reference_count':31,'daily_account_sessions':len(daily),'raw_endpoint_rows':len(raw_endpoints),'labels_computed':False,'fits':0,'accounts':0,'source_binding_sha256':sha(card/'source_binding.json'),'raw_training_binding_sha256':sha(card/'raw_training_source_binding.json'),'code_identity_sha256':sha(card/'execution_source_identity.json')}
    write(card/'prelabel_count_receipt.json',receipt)
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=NAME,uri=str(tracking)):
        rec=R.get_recorder();rec.save_objects(artifact_path='inputs',plan=plan,monthly_features=monthly,raw_endpoints=raw_endpoints,daily_context=daily,raw_points=points,guard_points=guard,schedule=schedule,corporate=corporate,primary_frame=primary['inputs/frame'],additional_frame=additional['inputs/frame'],additional_targets=additional['analysis/targets']['candidate'],controller=controller['inputs/controller'],action_plan=action['inputs/plan'],recovery_selection=action['controller/source_recovery_selection'],context_audit=context_audit,count_receipt=receipt,source_binding=binding,raw_training_binding=prefix,source_audits={'quarter':ba,'raw':ra,'primary':pa,'additional':aa,'controller':cta,'action':aca,'unit':ua,'corporate':ca})
        rec.save_objects(artifact_path='sources',**{path.replace('/','__'):(root/path).read_text() for path in identity});rec.log_metrics(prelabel_months=len(monthly),completed_fits=0,completed_native_accounts=0)
    state={**receipt,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'count_receipt_sha256':sha(card/'prelabel_count_receipt.json')};write(card/'prepared_state.json',state);return state


def physical_month_labels(monthly,endpoints,corporate):
    """One physical ETF share bought after entry-day ex entitlement; no dividend reinvestment."""
    rows=[];cash=corporate['data/cash_events.pkl'];splits=corporate['data/split_events.pkl']
    for date,row in monthly.iterrows():
        entry=pd.Timestamp(row.entry_at).tz_localize(None);exit_at=pd.Timestamp(row.exit_at).tz_localize(None);values={}
        for symbol in ['SOXL','SOXX']:
            events=[]
            for e in cash.loc[cash.instrument.eq(symbol)].to_dict('records'):
                day=pd.Timestamp(e['ex_date'])
                if entry.normalize()<day<=exit_at.normalize():events.append((day,'cash',float(e['original_total_decimal'])))
            for e in splits.loc[splits.instrument.eq(symbol)].to_dict('records'):
                day=pd.Timestamp(e['effective_trading_date'])
                if entry.normalize()<day<=exit_at.normalize():events.append((day,'split',float(e['new_shares_per_old_share'])))
            if any({kind for day,kind,_ in events if day==d}=={'cash','split'} for d,_,_ in events):raise ValueError('ambiguous same-day corporate unit ordering')
            q=1.;income=0.
            for _,kind,value in sorted(events):
                if kind=='split':q*=value
                else:income+=q*value
            opening=float(endpoints.loc[entry,symbol]);closing=float(endpoints.loc[exit_at,symbol]);low=symbol.lower()
            values['capital_growth_'+low]=q*closing/opening;values['income_yield_'+low]=income/opening
        rows.append({'signal_date':date,**values})
    return monthly.join(pd.DataFrame(rows).set_index('signal_date'))


def make_dataset(frame,train,test):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from src.model.monthly_profit_utility import ECONOMIC_LABELS
    selected=frame.loc[train|test];labels=selected[ECONOMIC_LABELS].copy();labels['entry_year']=selected.execution_year;labels['episode_id']=-1
    periods=pd.DatetimeIndex(frame.loc[train,'entry_at']).tz_localize(None).to_period('M').asi8
    episodes=np.r_[0,np.cumsum(np.diff(periods)!=1)]
    labels.loc[frame.index[train],'episode_id']=episodes
    data=pd.concat({'feature':selected[MARKET+PROFIT],'label':labels},axis=1);data.index=pd.MultiIndex.from_arrays([selected.index,['SOXX_ALLOCATION']*len(selected)],names=['datetime','instrument'])
    dataset=DatasetH(handler=DataHandlerLP.from_df(data),segments={'train':(frame.index[train].min(),frame.index[train].max()),'test':(frame.index[test].min(),frame.index[test].max())})
    if len(dataset.prepare('train',col_set='feature'))!=int(train.sum()):raise ValueError('dataset changed paired cohort')
    return dataset


def assess(books,daily):
    from src.analysis.integrated_allocation import paired_metrics,stationary_bootstrap_excess
    metrics=[];periods=[];pairs=[];rolling={};rolls=[];intervals=[]
    if set(books)!={(family,cost) for family in FAMILIES for cost in COSTS}:raise ValueError('all15fixed accounts required')
    expected=daily[DATES].reset_index(drop=True)
    for book in books.values():
        pd.testing.assert_frame_equal(book['authority'][DATES].reset_index(drop=True),expected)
        first=book['points'].iloc[0]
        np.testing.assert_allclose(first[['cash','receivable','q_soxl','q_soxx']].to_numpy(float),[100000.,0.,0.,0.],rtol=0,atol=1e-7)
    for (family,cost),book in books.items():
        path=book['authority'];bench=books[('SOXX',cost)]['authority'];meta={'family':family,'slippage_bps':cost}
        nav=np.r_[100000.,book['points'].nav.to_numpy(float)];dd=float(100*(nav/np.maximum.accumulate(nav)-1).min())
        metrics.append({**meta,**paired_metrics(path,bench),'sampled_point_max_drawdown_pct':dd,'fresh_initial_cash':100000.,'mean_ending_soxl_weight':float(path.ending_soxl_weight.mean()),'zero_SOXL_quantity_fraction':float(path.ending_soxl_quantity.eq(0).mean())})
        for frequency,kind in [('Y','year'),('Q','quarter')]:
            labels=path.trade_date.dt.to_period(frequency)
            for period in labels.unique():
                mask=labels.eq(period);complete=(period.year in [2024,2025]) if kind=='year' else period!=labels.iloc[-1]
                periods.append({**meta,'kind':kind,'period':str(period),'complete_period':complete,**paired_metrics(path.loc[mask],bench.loc[mask])})
        excess=np.log1p(path.strategy_return)-np.log1p(bench.strategy_return)
        windows=pd.DataFrame({'first_trade':path.trade_date.iloc[:-125].to_numpy(),'last_exit':path.exit_date.iloc[125:].to_numpy(),'relative_log':excess.rolling(126).sum().iloc[125:].to_numpy()})
        if len(windows)!=len(path)-125 or windows.relative_log.isna().any():raise ValueError('rolling window omission')
        rolling[f'{family}_{cost}']=windows;rolls.append({**meta,'windows':len(windows),'positive_fraction':float(windows.relative_log.gt(0).mean())})
        if family=='candidate':
            for reference in ['control','primary9097','additional27baa','SOXX']:
                other=books[(reference,cost)]['authority'];pairs.append({**meta,'comparator':reference,**paired_metrics(path,other)})
                if reference!='additional27baa':intervals.append({**meta,'comparator':reference,**stationary_bootstrap_excess(path,other,mean_block=63,resamples=2000,seed=20261014,alpha=.025/3)})
    metrics=pd.DataFrame(metrics);periods=pd.DataFrame(periods);pairs=pd.DataFrame(pairs);rolls=pd.DataFrame(rolls);intervals=pd.DataFrame(intervals)
    candidate=metrics[metrics.family.eq('candidate')];primary=metrics[metrics.family.eq('primary9097')];annual=periods[periods.family.eq('candidate')&periods.kind.eq('year')&periods.complete_period]
    if set(annual.period)!={'2024','2025'} or len(annual)!=6:raise ValueError('complete annual comparison missing')
    gates={'net_soxx_excess_all_costs':bool(candidate.relative_wealth.gt(1+1e-8).all()),'beat_matched_control_all_costs':bool(pairs[pairs.comparator.eq('control')].relative_wealth.gt(1+1e-8).all()),'beat_predeclared9097_all_costs':bool(pairs[pairs.comparator.eq('primary9097')].relative_wealth.gt(1+1e-8).all()),'sampled_drawdown_30pct':bool(candidate.max_drawdown_pct.ge(-30).all() and candidate.sampled_point_max_drawdown_pct.ge(-30).all()),'primary_sampled_risk_30pct':bool(primary.max_drawdown_pct.ge(-30).all() and primary.sampled_point_max_drawdown_pct.ge(-30).all()),'cvar_no_worse_soxx':bool(candidate.cvar5_pct.ge(candidate.benchmark_cvar5_pct).all()),'complete_annual_70pct':bool(annual.groupby('slippage_bps').relative_wealth.apply(lambda x:x.gt(1).mean()).ge(.7).all()),'rolling126_70pct':bool(rolls[rolls.family.eq('candidate')].positive_fraction.ge(.7).all()),'paired_block_lower_positive':bool(intervals.lower_bound_positive.all()),'fresh_fullminute_raw_risk_qualified':False,'qualified_independent_forward':False}
    coarse=all(v for k,v in gates.items() if k not in ['fresh_fullminute_raw_risk_qualified','qualified_independent_forward'])
    return {'metrics':metrics,'annual_metrics':periods[periods.kind.eq('year')],'quarterly_metrics':periods[periods.kind.eq('quarter')],'paired_comparison':pairs,'rolling_summary':rolls,'rolling':rolling,'uncertainty':intervals,'gates':gates,'coarse_economic_gates_pass':coarse}


def run(root):
    """Labels, six fits and fifteen accounts require the separately bound approval."""
    import qlib,yaml,torch
    from qlib.workflow import R
    from src.model.monthly_profit_utility import LinearMonthlyBetaUtility,WEIGHTS
    from src.backtest.raw_corporate_policy import replay_raw_package
    from src.backtest.native_raw_corporate import native_raw_validate
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
    from src.data.qlib_native import build_qlib_store_from_frame
    from src.data.nas_market_data import archive_nas_recorder
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true');root=Path(root).resolve();card=root/'research'/NAME
    state=prepare(root)
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    approval=json.loads((card/'execution_approval.json').read_text())
    required={'plan_sha256':'plan.json','source_binding_sha256':'source_binding.json','raw_training_binding_sha256':'raw_training_source_binding.json','code_identity_sha256':'execution_source_identity.json','count_receipt_sha256':'prelabel_count_receipt.json'}
    for key,path in required.items():
        if approval.get(key)!=sha(card/path):raise ValueError('execution approval identity differs '+key)
    if approval.get('decision')!='approved_bounded_direct_allocation_execution' or approval.get('maximum_fits')!=6 or approval.get('maximum_native_accounts')!=15 or not state['all_folds_eligible']:raise ValueError('labels/fits/accounts not authorized')
    plan=json.loads((card/'plan.json').read_text());qlib.init(provider_uri=str(root/'data/qlib/full_path_tiered_allocation_20261013'),region='us',kernels=1)
    R.set_uri(state['recorder_uri']);rec=R.get_recorder(experiment_name=NAME,recorder_id=state['recorder_id'])
    frame=physical_month_labels(rec.load_object('inputs/monthly_features'),rec.load_object('inputs/raw_endpoints'),rec.load_object('inputs/corporate'))
    daily=rec.load_object('inputs/daily_context');points=rec.load_object('inputs/raw_points');guard=rec.load_object('inputs/guard_points');schedule=rec.load_object('inputs/schedule')
    controller=rec.load_object('inputs/controller');action=rec.load_object('inputs/action_plan');selection=rec.load_object('inputs/recovery_selection')
    primary=rec.load_object('inputs/primary_frame');additional=rec.load_object('inputs/additional_frame');additional_targets=rec.load_object('inputs/additional_targets')
    progress_path=card/'run_state.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'fit_attempts':[],'fits':[],'package_attempts':[],'packages':[],'native_attempts':[],'native':[]}
    predictions={'control':[],'candidate':[]};fit_audits=[]
    with R.start(experiment_name=NAME,uri=state['recorder_uri'],recorder_id=state['recorder_id'],resume=True):
        rec=R.get_recorder();rec.save_objects(artifact_path='inputs',labelled_monthly_frame=frame,execution_approval=approval)
        for year in [2024,2025,2026]:
            test=frame.execution_year.eq(year);cutoff=frame.loc[test,'signal_at'].iloc[0];train=frame.source_qualified & frame.market_complete & frame.exit_at.lt(cutoff)
            dataset=make_dataset(frame,train,test)
            fresh={'control':LinearMonthlyBetaUtility(MARKET),'candidate':LinearMonthlyBetaUtility(MARKET+PROFIT)}
            torch.testing.assert_close(fresh['control'].linear.weight,fresh['candidate'].linear.weight[:,:len(MARKET)],rtol=0,atol=0)
            torch.testing.assert_close(fresh['control'].linear.bias,fresh['candidate'].linear.bias,rtol=0,atol=0)
            if not fresh['candidate'].linear.weight[:,len(MARKET):].eq(0).all():raise ValueError('extra inputs not initialized silent')
            models={};outputs={}
            for arm in ['control','candidate']:
                key=f'{arm}_{year}'
                if key in progress['fits']:model=rec.load_object('models/'+key)
                else:
                    if key in progress['fit_attempts'] or len(progress['fit_attempts'])>=6:raise ValueError('frozen fit budget exhausted; no silent retry')
                    progress['fit_attempts'].append(key);write(progress_path,progress)
                    model=fresh[arm];model.fit(dataset);rec.save_objects(artifact_path='models',**{key:model});progress['fits'].append(key);write(progress_path,progress)
                models[arm]=model;outputs[arm]=model.predict(dataset).xs('SOXX_ALLOCATION',level='instrument').astype(float)
                # Cash is the funded simplex residual, avoiding float32 serialization sum drift.
                outputs[arm]['cash_weight']=1-outputs[arm].soxl_weight-outputs[arm].soxx_weight
                if not outputs[arm].index.equals(frame.index[test]):raise ValueError('prediction calendar mismatch')
                fit_audits.append({'year':year,'arm':arm,'cutoff':str(cutoff),'last_train_maturity':str(frame.loc[train,'exit_at'].max()),**model.audit})
                rec.log_metrics(completed_fits=len(progress['fits']));print('fit complete',key,model.audit,flush=True)
            for attribute in ['medians','means','scales','active']:
                pd.testing.assert_series_equal(getattr(models['control'],attribute),getattr(models['candidate'],attribute).loc[MARKET])
            fallback=~frame.loc[test,'source_qualified'];outputs['candidate'].loc[fallback,WEIGHTS]=outputs['control'].loc[fallback,WEIGHTS]
            for arm in outputs:predictions[arm].append(outputs[arm])
        targets={arm:pd.concat(values).sort_index() for arm,values in predictions.items()}
        for arm,target in targets.items():
            if len(target)!=31 or not np.isfinite(target).all().all() or target.min().min()<0:raise ValueError('invalid monthly target')
            np.testing.assert_allclose(target.sum(axis=1),1.,rtol=0,atol=1e-10)
        rec.save_objects(artifact_path='analysis',monthly_targets=targets,fit_audits=fit_audits)
    provider=root/'data/qlib'/NAME
    allpoints=pd.concat([points,guard]).sort_index();source_rows=[]
    for symbol in ['SOXL','SOXX']:
        for stamp,value in allpoints[symbol].items():source_rows.append({'datetime':stamp,'instrument':symbol,**{x:float(value) for x in ['open','high','low','close','execution_price','raw_execution_price']}})
    manifest=build_qlib_store_from_frame(pd.DataFrame(source_rows),provider_uri=provider,as_of='2026-08-03',freq='1min',field_map={x:x for x in ['open','high','low','close','execution_price','raw_execution_price']},source='Frozen same physical rawpoint quotes, own account window and excluded following-bound guard')
    rec.save_objects(artifact_path='inputs',native_provider_manifest=manifest)
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**action['recovery_spec']);action_spec=Stage1OpenActionValueSpec(root=root,**action['action_spec'])
    execution={'slippage_bps':COSTS,'initial_cash':100000.,'tariff':plan['accounts']['tariff']};books={};family_frames={}
    for family in FAMILIES:
        local=daily.copy()
        if family in ['candidate','control']:
            target=targets[family].copy();target.index=pd.DatetimeIndex(frame.loc[target.index,'entry_at']).tz_localize(None)
            trade_times=pd.DatetimeIndex(local.trade_date)+pd.Timedelta('9h47min');aligned=target.reindex(trade_times,method='ffill')
            if aligned.isna().any().any():raise ValueError('monthly reference not known at daily entry')
            for column in WEIGHTS:local[column]=aligned[column].to_numpy()
        elif family=='primary9097':
            original=primary.set_index('signal_date').reindex(pd.DatetimeIndex(local.signal_date))
            for column in CONTEXT+WEIGHTS:local[column]=original[column].to_numpy()
        elif family=='additional27baa':
            original=additional.set_index('signal_date').reindex(pd.DatetimeIndex(local.signal_date))
            for column in CONTEXT:local[column]=original[column].to_numpy()
            target=additional_targets.reindex(pd.DatetimeIndex(local.signal_date))
            if not set(WEIGHTS+['alpha_probability','tail_probability'])<=set(target):raise ValueError('frozen27baa own signal fields missing')
            for column in WEIGHTS+['alpha_probability','tail_probability']:local[column]=target[column].to_numpy()
        else:
            for column,value in zip(WEIGHTS,[0.,1.,0.]):local[column]=value
        if not np.isfinite(local[WEIGHTS].to_numpy()).all():raise ValueError('daily reference absent')
        if family in ['candidate','control']:
            pd.testing.assert_frame_equal(local[CONTEXT],daily[CONTEXT])
        family_frames[family]=local
        if family in progress['packages']:paths=rec.load_object('paths/'+family)
        else:
            if family in progress['package_attempts'] or len(progress['package_attempts'])>=5:raise ValueError('fixed raw package attempt budget exhausted')
            progress['package_attempts'].append(family);write(progress_path,progress)
            paths,trace=replay_raw_package(local,point_prices=points,schedule=schedule,plan=execution,controller=controller,recovery_spec=recovery,action_spec=action_spec,selection=selection,benchmark=family=='SOXX')
            rec.save_objects(artifact_path='paths',**{family:paths});rec.save_objects(artifact_path='traces',**{family:trace});progress['packages'].append(family);write(progress_path,progress);print('raw package complete',family,flush=True)
        for cost in COSTS:
            key=f'{family}_{cost}'
            if key in progress['native']:book=rec.load_object('native/'+key)
            else:
                if key in progress['native_attempts'] or len(progress['native_attempts'])>=15:raise ValueError('fixed native attempt budget exhausted')
                progress['native_attempts'].append(key);write(progress_path,progress)
                book=native_raw_validate(paths[cost],provider_uri=provider,point_prices=points,schedule=schedule,tariff=execution['tariff'],cost=cost,initial_cash=100000.)
                rec.save_objects(artifact_path='native',**{key:book});progress['native'].append(key);write(progress_path,progress);print('native account complete',key,flush=True)
            books[(family,cost)]=book;rec.log_metrics(completed_native_accounts=len(progress['native']))
    rec.save_objects(artifact_path='inputs',executed_daily_frames=family_frames)
    results=assess(books,daily);coarse=results['coarse_economic_gates_pass']
    for name,value in results.items():
        if isinstance(value,pd.DataFrame):value.to_csv(card/(name+'.csv'),index=False);rec.save_objects(artifact_path='analysis',**{name:value})
        elif name=='rolling':rec.save_objects(artifact_path='analysis/rolling',**value)
    decision='needs_prespecified_fullminute_raw_risk_qualification' if coarse else 'not_promoted_fixed_quarterly_profit_direct_allocation'
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','models/*','paths/*','traces/*','native/*','analysis/monthly_targets','analysis/rolling/*']}
    summary={'experiment':NAME,'decision':decision,'objective_achieved':False,'model_fits':len(progress['fits']),'policy_paths':3*len(progress['packages']),'native_accounts':len(progress['native']),'monthly_references':31,'daily_account_sessions':len(daily),'gates':results['gates'],'coarse_economic_gates_pass':coarse,'fullminute_followup_required':coarse,'no_more_fullminute_work_after_negative':not coarse,'quarterly_information_MSE_used_as_gate':False,'new_source_requests':0,'initial_cash':100000,'common_start':plan['core_clocks']['common_account_start'],'common_terminal':plan['core_clocks']['common_terminal'],'historical_PIT_qualified':False,'independent_forward_qualified':False,'provider_uri':str(provider),'region':'us','as_of':'2026-08-03','plan_sha256':PLAN_SHA,'source_binding_sha256':sha(card/'source_binding.json'),'code_identity_sha256':sha(card/'execution_source_identity.json'),'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':state['recorder_uri'],'recorder_local_dir':str(Path(state['recorder_uri'])/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'limitations':['Directmonthlyutility is a postcomposition surrogate, not a differentiable fullrisk controller','Monthly core references have dailylegacyrisk/rebalance fills','Fresh rawphysicalCA riskqualification cannotbe inherited from adjustedpriceoldaccount','All historyreuseddevelopment; no production or goalachievement claim']}
    rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    write(card/'verification_manifest.json',{'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]})
    binding=json.loads((root/plan['sources']['raw_cashbook']['card']/'nas_binding.json').read_text());binding.pop('evidence',None);write(card/'nas_binding.json',binding)
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));write(card/'summary.json',summary);write(card/'fit_audit.json',fit_audits)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':NAME,'module':'src.workflow.quarterly_profit_allocation','plan':'plan.json','artifact_policy':policy},sort_keys=False))
    rows=results['metrics'];candidate_rows=rows.loc[rows.family.eq('candidate')]
    lines=['# 季度利润直接配置：独立经济检验','',f'完成{len(progress["fits"])}次冻结直接utility拟合及{len(progress["native"])}个独立费用/原生现金账核对；控制、候选、9097主参考、27baa额外对照及SOXX均从相同10万美元空持仓/风险状态启动。月度核心参考31次，实际成交仍受每日固定风险/再平衡与日内恢复包驱动。','', '|滑点|候选累计净收益|SOXX累计净收益|候选收盘回撤|候选CVaR5|平均实际SOXL权重|','|---|---:|---:|---:|---:|---:|']
    for row in candidate_rows.itertuples():lines.append(f'|{row.slippage_bps}bp|{row.return_pct:.3f}%|{row.benchmark_return_pct:.3f}%|{row.max_drawdown_pct:.3f}%|{row.cvar5_pct:.3f}%|{100*row.mean_ending_soxl_weight:.3f}%|')
    lines+=['', '全部2024/2025完整交易年、2026部分年、季度及全部126交易日窗口均保留。母来源信息MSE没有作为执行门，未交易失败年度或季度Ridge预测。训练使用后T月度utility代理；真实整数股数、股息应收/可用现金、当前费表投影及0/2/5bp均由独立账户裁决。','', '决定：'+decision+'。', '仍需预登记的同状态原始完整分钟风险核验；旧账户风险资格不可转移，不能晋升。' if coarse else '粗经济门未通过，关闭这套固定直接配置版本，不改阈值/正则/信号符号救援，也不继续为失败版本作无关完整分钟审计。','', '失败门：'+', '.join(k for k,v in results['gates'].items() if not v), '', '稳定超额目标未达成；历史均为复用研究数据，没有独立前向资格。主9097必须获得新原始现金账/RTH风险资格，27baa仅为额外经济对照，不能事后替换较弱主参考。', '', 'Recorder：'+rec.id+'。大模型、目标、全部逐日账户、原生仓位和滚动结果留在Recorder/NAS。']
    (card/'conclusion.md').write_text('\n'.join(lines)+'\n')
    return summary
