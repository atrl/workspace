"""Fixed family-weighted utility experiment; each ETF has its own holdings path."""
import hashlib
import numpy as np
import pandas as pd
import torch
from qlib.model.base import Model
from src.workflow.auxiliary_3x_training_data_allocation import AuxiliaryTrainingAllocator
from src.workflow.pit_constituent_auxiliary_training_allocation import _holdings_aware_turnover

WEIGHTS=['soxl_weight','soxx_weight','cash_weight']

class LegalAllocator(AuxiliaryTrainingAllocator):
    def forward(self,x):
        z=self.network(x);leveraged=.30*torch.sigmoid(z[:,0]);cash=.15+(1-leveraged-.15)*torch.sigmoid(z[:,1])
        return torch.stack([leveraged,1-leveraged-cash,cash],dim=1)

def pair_loss(weights,returns,years):
    if not torch.isfinite(returns).all() or not (returns>-1).all():raise ValueError('invalid return labels')
    turnover,_=_holdings_aware_turnover(weights,returns)
    r=(weights[:,:2]*returns).sum(1)-.001*turnover
    relative=torch.log1p(r.clamp(min=-.95))-torch.log1p(returns[:,1].clamp(min=-.95))
    annual=torch.stack([relative[years==y].mean() for y in torch.unique(years)])
    return -annual.mean()+.50*annual.std(unbiased=False)+.006*turnover.mean()

def weighted_loss(network,groups,family_weights):
    return sum(weight*pair_loss(network(groups[asset][0]),groups[asset][1],groups[asset][2]) for asset,weight in family_weights.items())

class FamilyUtilityModel(Model):
    def __init__(self,features,scaler,seed,family_weights):
        self.features=features;self.scaler=scaler;self.seed=seed;self.family_weights=family_weights
        torch.manual_seed(seed);self.network=LegalAllocator(len(features),16)
        self.initial_sha=hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in self.network.state_dict().values())).hexdigest()
    def tensor(self,data):
        x=data[self.features].replace([np.inf,-np.inf],np.nan).fillna(self.scaler['median'])
        x=((x-self.scaler['mean'])/self.scaler['scale']).clip(-8,8).to_numpy(np.float32)
        if not np.isfinite(x).all():raise ValueError('nonfinite transformed feature')
        return torch.tensor(x)
    def fit(self,dataset,**kwargs):
        frame=dataset.prepare('train',col_set=['feature','label']);groups={};counts={}
        for asset in self.family_weights:
            g=frame.xs(asset,level='instrument').sort_index();x=self.tensor(g['feature']);r=torch.tensor(g['label'][['leveraged_return','underlying_return']].to_numpy(np.float32));y=torch.tensor(g.index.year.to_numpy())
            groups[asset]=(x,r,y);counts[asset]={'rows':len(g),'first':str(g.index[0]),'last':str(g.index[-1])}
        optimizer=torch.optim.AdamW(self.network.parameters(),lr=.002,weight_decay=.08)
        self.loss_trace=[];self.network.train()
        for epoch in range(300):
            optimizer.zero_grad();loss=weighted_loss(self.network,groups,self.family_weights)
            if not torch.isfinite(loss):raise ValueError('nonfinite utility')
            loss.backward();torch.nn.utils.clip_grad_norm_(self.network.parameters(),2);optimizer.step();self.loss_trace.append(float(loss.detach()))
        self.network.eval();self.audit={'seed':self.seed,'family_weights':self.family_weights,'training':counts,'epochs':300,'checkpoint':'final_epoch_only','initial_sha256':self.initial_sha,'feature_count':len(self.features),'auxiliary_gradient_rows':0,'loss_final':self.loss_trace[-1]}
    def predict(self,dataset,segment='test'):
        frame=dataset.prepare(segment,col_set='feature');
        with torch.no_grad():v=self.network(self.tensor(frame)).numpy()
        return pd.DataFrame(v,index=frame.index,columns=WEIGHTS)
