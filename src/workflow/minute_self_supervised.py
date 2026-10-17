"""Train-only masked/contrastive minute representations with frozen linear probes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import Ridge
from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder, materialize_nas_evidence
from src.workflow.minute_value_weight import _metrics as temporal_metrics


def _metrics(frame):
    if frame.prediction.nunique() <= 1:
        return {'n':len(frame),'mse':float(np.mean((frame.prediction-frame.label)**2)),'ic':float('nan')}
    return temporal_metrics(frame)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TemporalEncoder(nn.Module):
    def __init__(self, config, seed):
        super().__init__();torch.manual_seed(seed)
        self.layers=nn.ModuleList([nn.Conv1d(a,b,k,padding=d*(k-1)//2,dilation=d)
            for a,b,k,d in zip(config['channels'][:-1],config['channels'][1:],config['kernels'],config['dilations'])])

    def forward(self,x,valid):
        z=(x*valid[...,None]).transpose(1,2)
        for layer in self.layers:
            z=F.relu(layer(z))*valid[:,None,:]
        return z.transpose(1,2)

    def pooled(self,x,valid):
        z=self(x,valid);length=valid.sum(dim=1)
        clock=torch.arange(x.shape[1],device=x.device)[None,:]
        bins=[]
        for i in range(4):
            select=(clock*4>=length[:,None]*i)&(clock*4<length[:,None]*(i+1))&valid
            bins.append((z*select[...,None]).sum(dim=1)/select.sum(dim=1)[:,None].clamp(min=1))
        return torch.cat(bins,dim=1)


def fold_arrays(data,fold):
    dates=pd.DatetimeIndex(data.index.get_level_values('datetime'))
    y=data['label'].iloc[:,0].to_numpy(float)
    maturity=pd.to_datetime(data['meta']['maturity_ns'].to_numpy().astype('int64'))+pd.Timedelta(hours=9,minutes=31)
    cutoff=pd.Timestamp(fold['train_end'])+pd.Timedelta(days=1)
    train=(dates<=pd.Timestamp(fold['train_end']))&(maturity<cutoff)&np.isfinite(y)
    test=(dates>=pd.Timestamp(fold['predict_start']))&(dates<=pd.Timestamp(fold['predict_end']))&np.isfinite(y)
    x=data['sequence'].to_numpy(np.float32).reshape(-1,390,10)
    length=data['meta']['length'].to_numpy(int);valid=np.arange(390)[None,:]<length[:,None]
    mean=x[train][valid[train]].mean(axis=0);std=np.maximum(x[train][valid[train]].std(axis=0),1e-6)
    scaled=(x-mean)/std;scaled[~valid]=0
    return {'dates':dates,'y':y,'train':train,'test':test,'x':torch.tensor(scaled),'valid':torch.tensor(valid),
        'hand':data['handcrafted'].to_numpy(float),'scaler':{'mean':mean,'std':std},'audit':{'training_rows':int(train.sum()),'test_rows':int(test.sum()),'last_training_signal':str(dates[train].max()),'last_mature_label':str(maturity[train].max()),'cutoff_exclusive':str(cutoff)}}


def pretrain(arrays,family,seed,config):
    encoder=TemporalEncoder(config,seed)
    decoder=nn.Linear(config['channels'][-1],10) if family=='masked_reconstruction' else None
    params=list(encoder.parameters())+(list(decoder.parameters()) if decoder is not None else [])
    optimizer=torch.optim.Adam(params,lr=config['learning_rate'],weight_decay=config['weight_decay'])
    rng=np.random.default_rng(seed);generator=torch.Generator().manual_seed(seed+1000)
    x=arrays['x'][arrays['train']];valid=arrays['valid'][arrays['train']]
    history=[]
    for epoch in range(config['epochs']):
        losses=[]
        for batch in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/config['batch_size']))):
            xb=x[batch];vb=valid[batch]
            if family=='masked_reconstruction':
                masked=(torch.rand(vb.shape,generator=generator)<config['reconstruction_mask_probability'])&vb
                observed=xb.clone();observed[masked]=0
                out=decoder(encoder(observed,vb))
                loss=(out[masked]-xb[masked]).square().mean()
            else:
                views=[]
                for _ in range(2):
                    kept=(torch.rand(vb.shape,generator=generator)>=config['contrastive_mask_probability'])&vb
                    noise=torch.randn(xb.shape,generator=generator)*config['contrastive_gaussian_noise_std']
                    views.append(F.normalize(encoder.pooled((xb+noise)*kept[...,None],vb),dim=1))
                logits=views[0]@views[1].T/config['contrastive_temperature'];target=torch.arange(len(batch))
                loss=(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    encoder.eval()
    for p in encoder.parameters():p.requires_grad_(False)
    return encoder,history


def representation(encoder,arrays):
    encoder.eval();outputs=[]
    with torch.no_grad():
        for start in range(0,len(arrays['x']),128):
            outputs.append(encoder.pooled(arrays['x'][start:start+128],arrays['valid'][start:start+128]).numpy())
    return np.concatenate(outputs)


def probe(features,arrays,config):
    train,test=arrays['train'],arrays['test'];mean=features[train].mean(axis=0);std=np.maximum(features[train].std(axis=0),1e-6)
    model=Ridge(alpha=config['ridge_alpha'],fit_intercept=True).fit((features[train]-mean)/std,arrays['y'][train])
    pred=model.predict((features[test]-mean)/std)
    return pred,{'probe':model,'probe_mean':mean,'probe_std':std}


def analysis(predictions,plan):
    ensemble=predictions.groupby(['date','fold','family'],as_index=False)[['prediction','label']].mean()
    comparison=pd.DataFrame([{'fold':fold,'family':family,**_metrics(part)} for (fold,family),part in ensemble.groupby(['fold','family'])])
    intervals=[]
    for family,part in ensemble.groupby('family'):
        for frequency in ['Y','Q']:
            for period,group in part.groupby(part.date.dt.to_period(frequency)):
                intervals.append({'family':family,'kind':frequency,'period':str(period),'partial':str(period).startswith('2026'),**_metrics(group)})
        for fold,group in part.groupby('fold'):
            group=group.sort_values('date')
            for end in range(126,len(group)+1):
                section=group.iloc[end-126:end]
                intervals.append({'family':family,'kind':'126_complete_observed_sessions','period':str(section.date.iloc[-1].date()),'fold':fold,'partial':False,**_metrics(section)})
    gates={};seeds=[];uncertainty=[]
    for candidate in plan['candidates']:
        gates[candidate]={}
        for fold in plan['folds']:
            name=fold['name'];table=comparison.loc[comparison.fold.eq(name)].set_index('family')
            gates[candidate][name+'_positive_ic']=bool(table.loc[candidate,'ic']>0)
            for control in plan['controls']:
                gates[candidate][name+'_mse_better_'+control]=bool(table.loc[candidate,'mse']<table.loc[control,'mse'])
            count=0
            for seed in plan['seeds']:
                scores={family:_metrics(part)['mse'] for family,part in predictions.loc[predictions.fold.eq(name)&predictions.seed.eq(seed)].groupby('family')}
                increment=scores['random_frozen']-scores[candidate];count+=increment>0
                seeds.append({'candidate':candidate,'fold':name,'seed':seed,'mse_improvement_over_random':increment})
            gates[candidate][name+'_two_seeds_better_random']=bool(count>=2)
        for control in plan['controls']:
            blocks=[]
            for fold,part in ensemble.groupby('fold'):
                wide=part.pivot(index='date',columns='family',values='prediction');labels=part.groupby('date').label.first()
                blocks.append(((wide[control]-labels)**2-(wide[candidate]-labels)**2).to_numpy())
            cfg=plan['bootstrap'];rng=np.random.default_rng(cfg['seed']);means=[]
            for _ in range(cfg['resamples']):
                sample=[]
                for values in blocks:
                    n=len(values);indices=np.empty(n,dtype=int);indices[0]=rng.integers(n)
                    for j in range(1,n):indices[j]=rng.integers(n) if rng.random()<1/cfg['mean_block'] else (indices[j-1]+1)%n
                    sample.append(values[indices])
                means.append(float(np.concatenate(sample).mean()))
            low=float(np.quantile(means,cfg['alpha']));high=float(np.quantile(means,1-cfg['alpha']))
            uncertainty.append({'candidate':candidate,'control':control,'mean_mse_improvement':float(np.concatenate(blocks).mean()),'lower':low,'upper':high,'lower_positive':low>0})
            gates[candidate]['paired_bootstrap_'+control]=low>0
    return ensemble,comparison,pd.DataFrame(intervals),pd.DataFrame(seeds),pd.DataFrame(uncertainty),gates


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    torch.set_num_threads(2);plan_hash=digest(card/'plan.json')
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=plan_hash:raise ValueError('frozen plan changed')
        return old
    design=json.loads((card/'design_review.json').read_text())
    if design['plan_sha256']!=plan_hash or not design['approved']:raise ValueError('missing approved design')
    source=plan['source'];parent=root/source['card'];binding=json.loads((parent/'nas_binding.json').read_text())
    if binding['evidence']['sha256']!=source['nas_archive_sha256']:raise ValueError('parent NAS identity differs')
    parent_summary=json.loads((parent/'summary.json').read_text())
    if not Path(parent_summary['recorder_local_dir']).exists():materialize_nas_evidence(parent/'nas_binding.json',card=parent)
    objects,audit=load_verified_recorder_objects(parent,expected_recorder_id=source['recorder_id'],artifacts=list(source['artifact_sha256']),expected_artifact_sha256=source['artifact_sha256'],expected_manifest_sha256=source['verification_manifest_sha256'])
    dataset=objects['inputs/dataset'];data=dataset.prepare('all',col_set=['sequence','handcrafted','label','meta']);prior=objects['analysis/ensemble']
    if data.shape!=(2007,3915):raise ValueError('unexpected source Dataset domain')
    progress={'plan_sha256':plan_hash,'pretraining':[],'probes':[],'attempts':[]};state_path=card/'run_state.json'
    if state_path.exists():progress=json.loads(state_path.read_text())
    if progress['plan_sha256']!=plan_hash:raise ValueError('progress plan differs')
    tracking=card/'qlib_recorder';kwargs={'experiment_name':plan['experiment'],'uri':str(tracking)}
    if progress.get('recorder_id'):kwargs.update(recorder_id=progress['recorder_id'],resume=True)
    predictions=[];trainrows=[]
    with R.start(**kwargs):
        rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,state_path)
        rec.log_params(plan_sha256=plan_hash,max_pretraining=12,max_probes=20)
        rec.save_objects(artifact_path='inputs',plan=plan,dataset=dataset,source_audit=audit,source_inventory=json.loads((card/'source_inventory.json').read_text()))
        rec.save_objects(artifact_path='sources',workflow=Path(__file__).read_text())
        for fold in plan['folds']:
            arrays=fold_arrays(data,fold);name=fold['name'];dates=arrays['dates'][arrays['test']];labels=arrays['y'][arrays['test']]
            for family in ['parent_best_observed','training_mean','zero_excess']:
                if family=='parent_best_observed':
                    part=prior.loc[prior.fold.eq(name)&prior.family.eq(plan['parent_best_observed'][name])].set_index('date').sort_index()
                    if not pd.DatetimeIndex(part.index).equals(dates):raise ValueError('parent dates differ')
                    np.testing.assert_allclose(part.label,labels,rtol=0,atol=0);pred=part.prediction.to_numpy()
                else:pred=np.full(len(dates),arrays['y'][arrays['train']].mean() if family=='training_mean' else 0.)
                predictions.append(pd.DataFrame({'date':dates,'fold':name,'family':family,'seed':-1,'prediction':pred,'label':labels}))
            jobs=[('handcrafted_ridge',-1)]+[(family,seed) for seed in plan['seeds'] for family in ['random_frozen',*plan['candidates']]]
            for family,seed in jobs:
                key=f'{name}_{family}_{seed}'
                if key in progress['probes']:
                    result=rec.load_object('predictions/'+key);state=rec.load_object('models/'+key)
                else:
                    if key in progress['attempts']:raise ValueError('unfinished previous attempt: explicit audit required '+key)
                    if len(progress['probes'])>=plan['maximum_probe_fits']:raise ValueError('probe budget exhausted')
                    progress['attempts'].append(key);atomic_json(progress,state_path)
                    encoder=None;losses=[]
                    if family=='handcrafted_ridge':features=arrays['hand']
                    else:
                        if family=='random_frozen':encoder=TemporalEncoder(plan['model'],seed)
                        else:
                            if len(progress['pretraining'])>=plan['maximum_pretraining_fits']:raise ValueError('pretraining budget exhausted')
                            encoder,losses=pretrain(arrays,family,seed,plan['model']);progress['pretraining'].append(key);atomic_json(progress,state_path)
                        features=representation(encoder,arrays)
                    pred,state=probe(features,arrays,plan['model'])
                    state.update(encoder_state=encoder.state_dict() if encoder is not None else None,sequence_scaler=arrays['scaler'],pretraining_loss=losses,audit={'fold':name,'family':family,'seed':seed,**arrays['audit'],'pretraining_epochs':len(losses),'probe_alpha':plan['model']['ridge_alpha'],'final_pretraining_loss':losses[-1] if losses else None})
                    result=pd.DataFrame({'date':dates,'fold':name,'family':family,'seed':seed,'prediction':pred,'label':labels})
                    rec.save_objects(artifact_path='models',**{key:state});rec.save_objects(artifact_path='predictions',**{key:result});progress['probes'].append(key);atomic_json(progress,state_path)
                    rec.log_metrics(completed_pretraining=len(progress['pretraining']),completed_probes=len(progress['probes']));print('completed',key,flush=True)
                predictions.append(result);trainrows.append(state['audit'])
        all_predictions=pd.concat(predictions,ignore_index=True)
        ensemble,comparison,intervals,seeds,uncertainty,gates=analysis(all_predictions,plan)
        passed=[name for name,values in gates.items() if all(values.values())]
        summary={'experiment':plan['experiment'],'plan_sha256':plan_hash,'decision':'information_screen_pass_needs_native_accounts' if passed else 'not_promoted_information_gate_failed','passing_candidates':passed,'objective_achieved':False,'gates':gates,'new_pretraining_fits':len(progress['pretraining']),'new_probe_fits':len(progress['probes']),'new_native_accounts':0,'history_status':plan['history_status'],'as_of':plan['as_of'],'provider_uri':plan['provider_uri'],'region':'us','source_id':plan['source_id'],'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/dataset','models/*','predictions/*','analysis/ensemble','analysis/intervals']},'limitations':['Mechanism-inspired adaptations, not exact paper replication','No independent history or forward evidence','Information screen only; no portfolio/cost/risk claims','Bootstrap tail alpha.0025 with2000 draws has sparse tail resolution','Prior best per-fold reference retrospectively known and frozen, not tradable selection']}
        rec.save_objects(artifact_path='analysis',summary=summary,ensemble=ensemble,comparison=comparison,intervals=intervals,seeds=seeds,uncertainty=uncertainty,training=pd.DataFrame(trainrows),all_predictions=all_predictions)
        for title,table in [('information_metrics',comparison),('uncertainty',uncertainty),('seed_comparison',seeds),('training_summary',pd.DataFrame(trainrows))]:table.to_csv(card/(title+'.csv'),index=False)
        intervals.loc[intervals.kind.isin(['Y','Q'])].to_csv(card/'regime_window_metrics.csv',index=False)
        rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in (Path(summary['recorder_local_dir'])/'artifacts').rglob('*')))
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':plan['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary
