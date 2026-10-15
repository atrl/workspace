"""Only a zero-initialized AMD linear logit residual is trainable."""
from copy import deepcopy
from dataclasses import asdict
import hashlib,json
import numpy as np
import pandas as pd
import torch
from torch import nn
from qlib.model.base import Model
from src.model.minute_pit_allocator import _flat_index,_scaled,TARGETS
from src.data.amd_own_guidance_realization import FEATURE,FLAGS
from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES,_standardize_pair,_project_legal_weights,_allocation_loss


def base_fingerprint(base):
    digest=hashlib.sha256()
    for name,value in sorted(base.network.state_dict().items()):digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    metadata={k:getattr(base,k) for k in ['primary_scaler','common_scaler','extra_scaler','base_features','extra_features','feature_masks']}
    metadata.update(spec=asdict(base.spec),candidate=asdict(base.candidate))
    digest.update(json.dumps(metadata,sort_keys=True,default=str).encode())
    return digest.hexdigest()


class FrozenAllocationResidual(Model):
    def __init__(self,base,*,numeric):
        self.base=deepcopy(base);self.numeric=bool(numeric);self.seed=base.seed
        self.base.network.eval()
        for parameter in self.base.network.parameters():parameter.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):self.residual=nn.Linear(4 if numeric else 3,3,bias=False)
        nn.init.zeros_(self.residual.weight)
        self.initial_base_fingerprint=base_fingerprint(self.base)

    def prepare_training(self,dataset):
        source=_flat_index(dataset.prepare('train',col_set=['feature','label','common','meta']))
        maturity=pd.to_datetime(source[('meta','label_available_at')])
        eligible=maturity.notna()&maturity.lt(self.base.cutoff)&(source.index<=self.base.train_end)
        eligible &= np.isfinite(source['label'].loc[:,list(TARGETS)].to_numpy(float)).all(axis=1)
        train=source.loc[eligible].copy();features=train['feature']
        if len(train)!=3014:raise ValueError('residual cohort differs')
        _,_,scaler=_standardize_pair(features,features,FLAGS)
        self.state_scaler={k:v.to_dict() for k,v in scaler.items()}
        observed=features.loc[features[FLAGS[1]].eq(1)&np.isfinite(features[FEATURE]),FEATURE]
        active=len(observed)>=self.base.minimum_observations
        std=float(observed.std(ddof=1)) if len(observed)>1 else 0.
        self.numeric_scaler={'mean':float(observed.mean()) if active else 0.,'scale':std if active and std>0 else 1.,'active':active,'observations':len(observed)}
        self.training_metadata={'training_rows':len(train),'first_training_signal':str(train.index.min().date()),'last_training_signal':str(train.index.max().date()),'last_mature_label_at':str(maturity.loc[eligible].max())}
        self._training_frame=train
        return self.training_metadata

    def _base_logits(self,features,common):
        primary=_scaled(features,self.base.base_features,self.base.primary_scaler)
        primary=torch.cat([primary,self.base._extra_tensor(features)],dim=1)
        shared=_scaled(common,list(COMMON_FEATURES),self.base.common_scaler)
        with torch.no_grad():return tuple(v.detach() for v in self.base.network(primary,shared))

    def _amd_tensor(self,features):
        state=_scaled(features,FLAGS,self.state_scaler)
        if not self.numeric:return state
        scaler=self.numeric_scaler;values=np.zeros(len(features),dtype=np.float32)
        known=features[FLAGS[1]].eq(1)&np.isfinite(features[FEATURE])
        if scaler['active']:values[known.to_numpy()]=((features.loc[known,FEATURE]-scaler['mean'])/scaler['scale']).clip(-8,8).to_numpy(np.float32)
        return torch.cat([state,torch.tensor(values).unsqueeze(1)],dim=1)

    def fit(self,dataset,**kwargs):
        if not hasattr(self,'_training_frame'):self.prepare_training(dataset)
        frame=self._training_frame;features=frame['feature'];labels=frame['label']
        logits,tail,alpha=self._base_logits(features,frame['common']);x=self._amd_tensor(features)
        if logits.requires_grad or tail.requires_grad or alpha.requires_grad:raise ValueError('base logits not detached')
        expected=12 if self.numeric else 9
        if sum(p.numel() for p in self.residual.parameters())!=expected:raise ValueError('residual capacity differs')
        if not self.residual.weight.eq(0).all():raise ValueError('fit must start at zero residual')
        returns=torch.tensor(labels.loc[:,list(TARGETS[:2])].to_numpy(np.float32));years=torch.tensor(frame.index.year.to_numpy(),dtype=torch.int64)
        cash=torch.tensor(labels.tail_cash_target_5d.to_numpy(np.float32));events=torch.tensor(labels.tail_event_5d.to_numpy(np.float32))
        optimizer=torch.optim.AdamW(self.residual.parameters(),lr=.002,weight_decay=.08,betas=(.9,.999),eps=1e-8,amsgrad=False)
        for _ in range(300):
            optimizer.zero_grad()
            weights=_project_legal_weights(logits+self.residual(x),maximum_soxl_weight=self.base.spec.maximum_soxl_weight,minimum_cash_weight=self.base.spec.minimum_cash_weight)
            loss,_,_,_=_allocation_loss(weights=weights,returns=returns,years=years,cash_target=cash,tail_event=events,candidate=self.base.candidate,drawdown_target_pct=-30.,worst_5d_target_pct=self.base.spec.research_worst_5d_target_pct,minimum_cash_weight=self.base.spec.minimum_cash_weight,cost_bps=self.base.spec.training_cost_bps,turnover_weight=self.base.spec.turnover_loss_weight)
            if not torch.isfinite(loss):raise ValueError('nonfinite residual objective')
            loss.backward();nn.utils.clip_grad_norm_(self.residual.parameters(),max_norm=2.);optimizer.step()
        if any(p.grad is not None for p in self.base.network.parameters()):raise ValueError('base gradient exists')
        if base_fingerprint(self.base)!=self.initial_base_fingerprint:raise ValueError('frozen base changed')
        self.residual.eval()
        self.audit={**self.training_metadata,'epochs':300,'checkpoint_policy':'fixed_final_epoch','trainable_parameters':expected,'base_model_updates':0,'new_auxiliary_gradient_rows':0,'inherited_base_auxiliary_rows':self.base.audit['auxiliary_rows'],'base_fingerprint':self.initial_base_fingerprint,'base_unchanged':True,'state_scaler':self.state_scaler,'numeric_scaler':self.numeric_scaler,'allocation_loss_before_last_update':float(loss.detach())}
        del self._training_frame
        return self

    def predict(self,dataset,segment='predict',**kwargs):
        prepared=dataset.prepare(segment,col_set=['feature','common'])
        logits,tail,alpha=self._base_logits(prepared['feature'],prepared['common'])
        with torch.no_grad():
            adjusted=logits+self.residual(self._amd_tensor(prepared['feature']))
            weights=_project_legal_weights(adjusted,maximum_soxl_weight=self.base.spec.maximum_soxl_weight,minimum_cash_weight=self.base.spec.minimum_cash_weight).numpy().astype(float)
        weights/=weights.sum(axis=1,keepdims=True)
        result=pd.DataFrame(weights,index=prepared.index,columns=['soxl_weight','soxx_weight','cash_weight'])
        result['tail_probability']=torch.sigmoid(tail).numpy();result['alpha_probability']=torch.sigmoid(alpha).numpy()
        return result
