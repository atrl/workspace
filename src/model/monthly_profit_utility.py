"""Tiny decision-aligned monthly allocator; no forecasting head or parameter search."""
import hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from qlib.model.base import Model

WEIGHTS=['soxl_weight','soxx_weight','cash_weight']
ECONOMIC_LABELS=['capital_growth_soxl','capital_growth_soxx','income_yield_soxl','income_yield_soxx']


def beta_raw_weights(beta):
    beta=beta.reshape(-1)
    risky=torch.minimum(beta,torch.full_like(beta,.85))
    leveraged=torch.clamp((beta-risky)/2,min=0.)
    return torch.stack([leveraged,risky-leveraged,1-risky],dim=1)


def compose_weights(raw):
    cancellation=torch.minimum(raw[:,0],raw[:,2]/2)
    return raw+cancellation[:,None]*raw.new_tensor([-1.,3.,-2.])[None,:]


def monthly_turnover(weights,capital_growth,income_yield,episodes):
    """NAV dividend-to-cash drift; proxy fees do not define a physical cash ledger."""
    if weights.ndim!=2 or weights.shape[1]!=3 or capital_growth.shape!=(len(weights),2) or income_yield.shape!=capital_growth.shape:
        raise ValueError('monthly drift shape differs')
    if not torch.isfinite(capital_growth).all() or not (capital_growth>0).all() or not torch.isfinite(income_yield).all() or (income_yield<0).any():
        raise ValueError('invalid physical one-unit labels')
    initial=weights.new_tensor([0.,0.,1.])
    previous=torch.cat([initial[None,:],weights[:-1]],dim=0)
    previous_capital=torch.cat([capital_growth.new_ones((1,2)),capital_growth[:-1]],dim=0)
    previous_income=torch.cat([income_yield.new_zeros((1,2)),income_yield[:-1]],dim=0)
    risky=previous[:,:2]*previous_capital
    cash=previous[:,2]+(previous[:,:2]*previous_income).sum(1)
    marked=torch.cat([risky,cash[:,None]],dim=1);marked=marked/marked.sum(1,keepdim=True)
    reset=torch.ones(len(weights),dtype=torch.bool,device=weights.device)
    if len(weights)>1:reset[1:]=episodes[1:]!=episodes[:-1]
    marked=torch.where(reset[:,None],initial[None,:],marked)
    return torch.abs(weights[:,:2]-marked[:,:2]).sum(1),marked


def direct_monthly_loss(raw_weights,capital_growth,income_yield,years,episodes):
    weights=compose_weights(raw_weights)
    turnover,_=monthly_turnover(weights,capital_growth,income_yield,episodes)
    returns=capital_growth-1+income_yield
    if not (returns>-1).all():raise ValueError('invalid total-return labels')
    r=(weights[:,:2]*returns).sum(1)-.001*turnover
    relative=torch.log1p(torch.clamp(r,min=-.95))-torch.log1p(returns[:,1])
    annual=torch.stack([relative[years==year].mean() for year in torch.unique(years)])
    loss=-annual.mean()+.50*annual.std(unbiased=False)+.006*turnover.mean()
    return loss,{'annual_monthly_mean_log_relative':annual,'turnover':turnover,'clamped_rows':int((r<-.95).sum().detach())}


class LinearMonthlyBetaUtility(Model):
    def __init__(self,features):
        self.features=list(features)
        with torch.random.fork_rng(devices=[]):self.linear=nn.Linear(len(self.features),1,device='cpu',dtype=torch.float32)
        with torch.no_grad():
            self.linear.weight.zero_();self.linear.bias.fill_(float(np.log(1/.45)))
        self.initial_sha256=hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in self.linear.state_dict().values())).hexdigest()
        self.fitted=False

    def raw_weights(self,x):return beta_raw_weights(1.45*torch.sigmoid(self.linear(x)).reshape(-1))

    def tensor(self,frame):
        values=frame[self.features].replace([np.inf,-np.inf],np.nan).fillna(self.medians)
        values=((values-self.means)/self.scales).clip(-8,8)
        values.loc[:,~self.active]=0.
        if not np.isfinite(values.to_numpy()).all():raise ValueError('nonfinite fixed transformed inputs')
        return torch.tensor(values.to_numpy(np.float32),device='cpu',dtype=torch.float32)

    def fit(self,dataset,**kwargs):
        torch.set_num_threads(1)
        data=dataset.prepare('train',col_set=['feature','label'])
        x=data['feature'][self.features].replace([np.inf,-np.inf],np.nan);labels=data['label']
        dates=pd.DatetimeIndex(data.index.get_level_values('datetime'))
        if dates.has_duplicates or not dates.is_monotonic_increasing:raise ValueError('training months must be chronological and unique')
        self.active=x.notna().any();self.medians=x.median().fillna(0.);filled=x.fillna(self.medians)
        self.means=filled.mean();self.scales=filled.std(ddof=0).replace(0.,1.)
        xt=self.tensor(x)
        cap=torch.tensor(labels[ECONOMIC_LABELS[:2]].to_numpy(np.float32));income=torch.tensor(labels[ECONOMIC_LABELS[2:]].to_numpy(np.float32))
        years=torch.tensor(labels.entry_year.to_numpy(np.int64));episodes=torch.tensor(labels.episode_id.to_numpy(np.int64))
        if len(x)<24 or len(torch.unique(years))<2:raise ValueError('fixed training count guard')
        optimizer=torch.optim.AdamW(self.linear.parameters(),lr=.002,weight_decay=.08)
        self.loss_trace=[]
        for _ in range(300):
            optimizer.zero_grad();loss,audit=direct_monthly_loss(self.raw_weights(xt),cap,income,years,episodes)
            if not torch.isfinite(loss):raise ValueError('nonfinite monthly utility')
            loss.backward();torch.nn.utils.clip_grad_norm_(self.linear.parameters(),2.);optimizer.step();self.loss_trace.append(float(loss.detach()))
        with torch.no_grad():final_loss,audit=direct_monthly_loss(self.raw_weights(xt),cap,income,years,episodes)
        self.fitted=True
        self.audit={'training_rows':len(x),'training_years':len(torch.unique(years)),'episodes':len(torch.unique(episodes)),'epochs':300,'checkpoint':'final_epoch_only','initial_sha256':self.initial_sha256,'parameter_count':sum(p.numel() for p in self.linear.parameters()),'inactive_features':list(self.active.index[~self.active]),'clamped_training_rows_final':audit['clamped_rows'],'loss_final':float(final_loss),'loss_trace_semantics':'pre_optimizer_step; loss_final evaluated on saved final weights','device':'cpu','dtype':'float32'}

    def predict(self,dataset,segment='test'):
        if not self.fitted:raise ValueError('model not fitted')
        frame=dataset.prepare(segment,col_set='feature')
        with torch.no_grad():values=self.raw_weights(self.tensor(frame)).numpy()
        return pd.DataFrame(values,index=frame.index,columns=WEIGHTS)
