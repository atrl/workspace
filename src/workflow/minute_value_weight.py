"""Frozen, native-Qlib temporal Value/Weight information screen (no portfolio claims)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.data.minute_allocation_features import _native_groups, _qualified, completed_session_features
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ValueWeight(nn.Module):
    """Independent value heads; optional causal GRU scores pooled over completed day."""
    def __init__(self, family, config, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.family = family
        count = 12 if family == 'handcrafted' else 10
        self.value = nn.Sequential(nn.Linear(count, config['value_hidden']), nn.Tanh(),
                                   nn.Linear(config['value_hidden'], config['heads']))
        if family == 'learned':
            self.weight = nn.GRU(10, config['gru_hidden'], batch_first=True)
            self.score = nn.Linear(config['gru_hidden'], config['heads'])

    def forward(self, x, mask, handmade):
        if self.family == 'handcrafted':
            return self.value(handmade)
        values = self.value(x)
        if self.family == 'learned':
            state, _ = self.weight(x)
            scores = self.score(state).masked_fill(~mask[..., None], -torch.inf)
            weights = scores.softmax(dim=1)
        else:
            weights = mask[..., None] / mask.sum(dim=1)[:, None, None]
        return (weights * values).sum(dim=1)


def prepare(root, plan):
    """Read verified native providers and expose all model arrays through DatasetH."""
    import exchange_calendars as xc
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    binding = json.loads((root/'research'/plan['experiment']/plan['nas_binding']).read_text())
    if binding['source_id'] != plan['source_id'] or not binding['verified']:
        raise ValueError('source identity mismatch')
    verified = {}
    for item in binding['objects']:
        name = item['local_path']
        if name == binding['source_manifest_path'] or any(Path(name).name == s+'_none.parquet' for s in ['SOXL','SOXX']):
            if _digest(root/name) != item['sha256']:
                raise ValueError('immutable raw source hash differs: '+name)
            verified[name] = item['sha256']
    if len(verified) != 3:
        raise ValueError('missing immutable core source receipts')
    for name, digest in plan['provider_file_sha256'].items():
        if _digest(root/name) != digest:
            raise ValueError('frozen Qlib file changed: '+name)
    fields=['$open','$high','$low','$close','$volume','$price_valid']
    core=load_qlib_feature_frame(QlibProviderConfig(root/plan['providers']['minute'],freq='1min'),
        instruments=['SOXL','SOXX'],fields=fields,start=plan['source_start'],end=plan['as_of']+' 16:00:00')
    for symbol in ['SOXL','SOXX']:
        raw_path=next(root/name for name in verified if Path(name).name==symbol+'_none.parquet')
        raw=pd.read_parquet(raw_path).set_index(pd.to_datetime(pd.read_parquet(raw_path,columns=['time_key']).time_key))
        native=core.xs(symbol,level='instrument')
        if not native.index.equals(pd.DatetimeIndex(raw.index)):
            raise ValueError('native/raw minute calendar differs')
        np.testing.assert_array_equal(native[['$open','$high','$low','$close','$volume']].to_numpy(np.float32),raw[['open','high','low','close','volume']].to_numpy(np.float32))
    empty=core.iloc[:0].copy();empty.attrs=core.attrs.copy()
    schedule=xc.get_calendar('XNYS',start=plan['source_start'],end=plan['as_of']).schedule
    summaries,coverage=completed_session_features(core,empty,schedule,core_names=plan['core_features'],peer_names=[],peer_symbols=['QQQ','SMH','NVDA','AMD'])
    groups=_native_groups(core);dates=[];paths=[];lengths=[]
    for day,session in schedule.iterrows():
        day=pd.Timestamp(day).tz_localize(None).normalize()
        opening=pd.Timestamp(session['open']).tz_convert('America/New_York').tz_localize(None)
        closing=pd.Timestamp(session['close']).tz_convert('America/New_York').tz_localize(None)
        expected=pd.date_range(opening+pd.Timedelta(minutes=1),closing,freq='min')
        parts=[groups.get((s,day)) for s in ['SOXL','SOXX']]
        if not all(_qualified(p,expected) for p in parts) or summaries.loc[day,'minute_core_ready'] != 1:
            continue
        channels=[]
        for part in parts:
            price=np.log(part[['open','high','low','close']].to_numpy()/float(part.open.iloc[0]))
            volume=part.volume.to_numpy(); mean=volume.mean()
            if mean<=0: raise ValueError('nonpositive session volume')
            channels.append(np.column_stack([price,np.log1p(volume/mean)]))
        path=np.zeros((390,10),dtype=np.float32);path[:len(expected)]=np.column_stack(channels)
        paths.append(path);dates.append(day);lengths.append(len(expected))
    prices=load_qlib_feature_frame(QlibProviderConfig(root/plan['providers']['daily']),
        instruments=['SOXL','SOXX'],fields=['$open','$close'],start=plan['source_start'],end=plan['as_of'])
    import pickle
    parent=plan['daily_label_parent']; parent_path=root/parent['native_prices_path']
    if _digest(parent_path)!=parent['sha256']:raise ValueError('parent native price artifact changed')
    original=pickle.load(parent_path.open('rb')).reindex(prices.index)
    np.testing.assert_array_equal(prices.to_numpy(np.float32),original.to_numpy(np.float32))
    opens=prices['$open'].unstack('instrument').sort_index()
    label=(opens.shift(-6)/opens.shift(-1)-1)
    label=label['SOXL']-label['SOXX']
    maturity=pd.Series(opens.index,index=opens.index).shift(-6)
    sequence=np.stack(paths);dates=pd.DatetimeIndex(dates)
    frame=pd.concat({'sequence':pd.DataFrame(sequence.reshape(len(dates),-1),index=dates),
        'handcrafted':summaries.reindex(dates)[plan['core_features']],
        'label':pd.DataFrame({'h5_excess':label.reindex(dates)}),
        'meta':pd.DataFrame({'length':lengths,'maturity_ns':maturity.reindex(dates).astype('int64')},index=dates)},axis=1)
    frame.index=pd.MultiIndex.from_arrays([dates,['SOXL_SOXX']*len(dates)],names=['datetime','instrument'])
    dataset=DatasetH(handler=DataHandlerLP.from_df(frame),segments={'all':(None,None)})
    # Qlib Serializable otherwise excludes the private loaded data on pickle.
    dataset.config(dump_all=True,recursive=True)
    observed=dataset.prepare('all',col_set=['sequence','handcrafted','label','meta'],data_key=DataHandlerLP.DK_I)
    if not observed.equals(frame): raise ValueError('DatasetH changed prepared data')
    return dataset,coverage,{'raw_source_hashes':verified,'provider_file_sha256':plan['provider_file_sha256'],
        'qlib_data_api':['qlib.data.D.features','qlib.data.D.calendar','DatasetH.prepare','DataHandlerLP.from_df'],
        'qualified_sessions':len(dates),'first':str(dates.min().date()),'last':str(dates.max().date()),
        'half_days':int(np.sum(np.array(lengths)<390)),
        'raw_to_native_all_ohlcv_float32_equal':True,
        'native_daily_to_parent_recorder_float32_equal':True,
        'label_shift_on_full_daily_calendar_before_filter':True}


def _fit_predict(data, family, fold, seed, config):
    dates=pd.DatetimeIndex(data.index.get_level_values('datetime'))
    y=data['label'].iloc[:,0].to_numpy(float)
    # NaT has int64 minimum, so require finite labels as well as mature timestamps.
    cutoff=pd.Timestamp(fold['train_end'])+pd.Timedelta(days=1)
    maturity=pd.to_datetime(data['meta']['maturity_ns'].to_numpy().astype('int64'))+pd.Timedelta(hours=9,minutes=31)
    train=(dates<=pd.Timestamp(fold['train_end'])) & (maturity<cutoff) & np.isfinite(y)
    test=(dates>=pd.Timestamp(fold['predict_start'])) & (dates<=pd.Timestamp(fold['predict_end'])) & np.isfinite(y)
    if train.sum()<252 or test.sum()<126: raise ValueError('insufficient complete observations')
    x=data['sequence'].to_numpy(np.float32).reshape(-1,390,10)
    lengths=data['meta']['length'].to_numpy(int);mask=np.arange(390)[None,:]<lengths[:,None]
    hand=data['handcrafted'].to_numpy(np.float32)
    mean=x[train][mask[train]].mean(axis=0);std=np.maximum(x[train][mask[train]].std(axis=0),1e-6)
    hmean=hand[train].mean(axis=0);hstd=np.maximum(hand[train].std(axis=0),1e-6)
    ym=y[train].mean();ys=max(y[train].std(),1e-6)
    xx=torch.tensor((x-mean)/std);mm=torch.tensor(mask);hh=torch.tensor((hand-hmean)/hstd)
    yy=torch.tensor((y-ym)/ys,dtype=torch.float32)
    model=ValueWeight(family,config,seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    generator=np.random.default_rng(seed);idx=np.flatnonzero(train)
    for epoch in range(config['epochs']):
        model.train();losses=[]
        for batch in np.array_split(generator.permutation(idx),int(np.ceil(len(idx)/config['batch_size']))):
            out=model(xx[batch],mm[batch],hh[batch]);z=out-out.mean(dim=0)
            z=z/torch.sqrt(z.square().mean(dim=0)+1e-6)
            corr=z.T@z/len(batch);off=corr-torch.diag(torch.diag(corr))
            loss=(out-yy[batch,None]).square().mean()+config['decorrelation_weight']*off.square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();losses.append(float(loss.detach()))
    model.eval();testidx=np.flatnonzero(test);outputs=[]
    with torch.no_grad():
        for batch in np.array_split(testidx,int(np.ceil(len(testidx)/config['batch_size']))):
            outputs.append(model(xx[batch],mm[batch],hh[batch]).numpy()*ys+ym)
    prediction=np.concatenate(outputs).mean(axis=1)
    result=pd.DataFrame({'date':dates[test],'fold':fold['name'],'family':family,'seed':seed,'prediction':prediction,'label':y[test]})
    audit={'family':family,'fold':fold['name'],'seed':seed,'training_rows':int(train.sum()),'test_rows':int(test.sum()),
           'last_training_signal':str(dates[train].max()),'last_maturity':str(maturity[train].max()),'cutoff_exclusive':str(cutoff),
           'final_loss':float(np.mean(losses)),'epochs':config['epochs'],'parameters':sum(p.numel() for p in model.parameters())}
    state={'state_dict':model.state_dict(),'sequence_mean':mean,'sequence_std':std,'hand_mean':hmean,'hand_std':hstd,'label_mean':ym,'label_std':ys,'audit':audit}
    return result,state


def _metrics(frame):
    return {'n':len(frame),'mse':float(np.mean((frame.prediction-frame.label)**2)),
        'ic':float(frame.prediction.corr(frame.label,method='spearman'))}


def analyze(predictions,plan):
    ensemble=predictions.groupby(['date','fold','family'],as_index=False)[['prediction','label']].mean()
    rows=[]
    for (fold,family),part in ensemble.groupby(['fold','family']):rows.append({'fold':fold,'family':family,**_metrics(part)})
    comparison=pd.DataFrame(rows); intervals=[]
    for family,part in ensemble.groupby('family'):
        part=part.sort_values('date').copy()
        for frequency in ['Y','Q']:
            for period,group in part.groupby(part.date.dt.to_period(frequency)):
                intervals.append({'family':family,'kind':frequency,'period':str(period),'partial':str(period).startswith('2026'),**_metrics(group)})
        for fold,group in part.groupby('fold'):
            group=group.sort_values('date')
            for end in range(126,len(group)+1):
                section=group.iloc[end-126:end]
                intervals.append({'family':family,'kind':'126_complete_observed_sessions','period':str(section.date.iloc[-1].date()),'fold':fold,'partial':False,**_metrics(section)})
    seed_rows=[];gates={}
    for fold in plan['folds']:
        f=fold['name'];sub=comparison.loc[comparison.fold.eq(f)].set_index('family')
        gates[f+'_positive_spearman_ic']=bool(sub.loc['learned','ic']>0)
        for control in ['handcrafted','uniform']:
            gates[f+'_mse_better_'+control]=bool(sub.loc['learned','mse']<sub.loc[control,'mse'])
            count=0
            for seed in plan['seeds']:
                scores={family:_metrics(part)['mse'] for family,part in predictions.loc[predictions.fold.eq(f)&predictions.seed.eq(seed)].groupby('family')}
                increment=scores[control]-scores['learned'];count+=increment>0
                seed_rows.append({'fold':f,'control':control,'seed':seed,'mse_improvement':increment})
            gates[f+'_two_seeds_'+control]=bool(count>=2)
    uncertainty=[]
    for control in ['handcrafted','uniform']:
        blocks=[]
        for fold,part in ensemble.groupby('fold'):
            wide=part.pivot(index='date',columns='family',values='prediction');labels=part.groupby('date').label.first()
            blocks.append(((wide[control]-labels)**2-(wide.learned-labels)**2).to_numpy())
        cfg=plan['bootstrap'];rng=np.random.default_rng(cfg['seed']);means=[]
        for _ in range(cfg['resamples']):
            sample=[]
            for values in blocks:
                n=len(values);indices=np.empty(n,dtype=int);indices[0]=rng.integers(n)
                for j in range(1,n):indices[j]=rng.integers(n) if rng.random()<1/cfg['mean_block'] else (indices[j-1]+1)%n
                sample.append(values[indices])
            means.append(float(np.concatenate(sample).mean()))
        low=float(np.quantile(means,cfg['alpha']));high=float(np.quantile(means,1-cfg['alpha']))
        uncertainty.append({'control':control,'mean_mse_improvement':float(np.concatenate(blocks).mean()),'lower':low,'upper':high,'lower_positive':low>0})
        gates['paired_bootstrap_'+control]=low>0
    return ensemble,comparison,pd.DataFrame(intervals),pd.DataFrame(seed_rows),pd.DataFrame(uncertainty),gates


def run(*,root,card,plan):
    from qlib.workflow import R
    from qlib.data.dataset.handler import DataHandlerLP
    import yaml
    digest=_digest(card/'plan.json')
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('frozen plan changed')
        return old
    torch.set_num_threads(2)
    dataset,coverage,audit=prepare(root,plan)
    data=dataset.prepare('all',col_set=['sequence','handcrafted','label','meta'],data_key=DataHandlerLP.DK_I)
    progress={'plan_sha256':digest,'fitted':[]};tracking=card/'qlib_recorder'
    if (card/'run_state.json').exists():progress=json.loads((card/'run_state.json').read_text())
    if progress['plan_sha256']!=digest:raise ValueError('run plan changed')
    kwargs={'experiment_name':plan['experiment'],'uri':str(tracking)}
    if progress.get('recorder_id'):kwargs.update(recorder_id=progress['recorder_id'],resume=True)
    predictions=[];trainrows=[]
    with R.start(**kwargs):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,card/'run_state.json')
        rec.log_params(plan_sha256=digest,source_id=plan['source_id'],maximum_model_fits=plan['maximum_model_fits'])
        rec.save_objects(artifact_path='inputs',plan=plan,dataset=dataset,audit=audit,coverage=coverage)
        rec.save_objects(artifact_path='sources',workflow=Path(__file__).read_text())
        for fold in plan['folds']:
            for seed in plan['seeds']:
                for family in plan['families']:
                    name=f"{fold['name']}_{seed}_{family}"
                    if name in progress['fitted']:
                        result=rec.load_object('predictions/'+name);state=rec.load_object('models/'+name)
                    else:
                        if len(progress['fitted'])>=plan['maximum_model_fits']:raise ValueError('fit budget exceeded')
                        result,state=_fit_predict(data,family,fold,seed,plan['model'])
                        rec.save_objects(artifact_path='models',**{name:state});rec.save_objects(artifact_path='predictions',**{name:result})
                        progress['fitted'].append(name);atomic_json(progress,card/'run_state.json');rec.log_metrics(completed_fits=len(progress['fitted']))
                        print('completed',name,state['audit'],flush=True)
                    predictions.append(result);trainrows.append(state['audit'])
        predictions=pd.concat(predictions,ignore_index=True)
        ensemble,comparison,intervals,seeds,uncertainty,gates=analyze(predictions,plan)
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'information_screen_pass_needs_native_accounts' if all(gates.values()) else 'not_promoted_information_gate_failed','objective_achieved':False,'gates':gates,'new_fits':len(progress['fitted']),'new_native_cases':0,'qualified_forward_days':0,'history_status':'reused_diagnostic_not_independent','as_of':plan['as_of'],'provider_uri':str(root/plan['providers']['minute']),'source_id':plan['source_id'],'data_audit':audit,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/dataset','models/*','predictions/*','analysis/ensemble','analysis/intervals']},'limitations':['Information screen only: no portfolio, costs, return or drawdown claims','Fixed H5 temporal formulation only; negative does not reject all raw-minute mechanisms','Historical source availability assumption, not live first-receipt qualification','Unknown prior search multiplicity; all market history reused']}
        rec.save_objects(artifact_path='analysis',summary=summary,ensemble=ensemble,comparison=comparison,intervals=intervals,seeds=seeds,uncertainty=uncertainty,training=pd.DataFrame(trainrows))
        for name,frame in [('information_metrics',comparison),('uncertainty',uncertainty),('seed_comparison',seeds),('training_summary',pd.DataFrame(trainrows)),('minute_coverage_summary',coverage)]:frame.to_csv(card/(name+'.csv'),index=False)
        intervals.loc[intervals.kind.isin(['Y','Q'])].to_csv(card/'regime_window_metrics.csv',index=False)
        rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in (Path(summary['recorder_local_dir'])/'artifacts').rglob('*')))
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'providers':plan['providers'],'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id))
    atomic_json(summary,card/'summary.json')
    return summary
