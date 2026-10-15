from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from src.model.frozen_allocation_residual import FrozenAllocationResidual,base_fingerprint
from src.data.amd_own_guidance_realization import align_realizations,FEATURE,FLAGS
from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES

@dataclass
class Settings:
    maximum_soxl_weight:float=.3
    minimum_cash_weight:float=.15

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__();self.head=torch.nn.Linear(2,3)
    def forward(self,primary,common):
        logits=self.head(primary)
        return logits,logits[:,0],logits[:,1]

def no_extra(frame):return torch.zeros((len(frame),0))

def test_zero_residual_and_only_residual_update_leave_base_and_classifiers_exact():
    scaler={'median':{'a':0.,'b':0.},'mean':{'a':0.,'b':0.},'scale':{'a':1.,'b':1.}}
    common_scaler={k:{c:(1. if k=='scale' else 0.) for c in COMMON_FEATURES} for k in ['median','mean','scale']}
    base=SimpleNamespace(network=Network(),seed=71,primary_scaler=scaler,common_scaler=common_scaler,extra_scaler={},base_features=['a','b'],extra_features=[],feature_masks={},spec=Settings(),candidate=Settings(),_extra_tensor=no_extra)
    model=FrozenAllocationResidual(base,numeric=True);model.state_scaler={k:{c:(1. if k=='scale' else 0.) for c in FLAGS} for k in ['median','mean','scale']};model.numeric_scaler={'active':True,'mean':0.,'scale':1.}
    feature=pd.DataFrame({'a':[.1,.4],'b':[.3,.2],FLAGS[0]:[1.,1.],FLAGS[1]:[1.,0.],FLAGS[2]:[0.,0.],FEATURE:[.2,np.nan]});common=pd.DataFrame(0.,index=feature.index,columns=COMMON_FEATURES)
    z,tail,alpha=model._base_logits(feature,common);x=model._amd_tensor(feature)
    assert model.residual.weight.numel()==12 and x[-1,-1]==0
    torch.testing.assert_close(z+model.residual(x),z,rtol=0,atol=0)
    before=base_fingerprint(model.base);optimizer=torch.optim.AdamW(model.residual.parameters(),lr=.002,weight_decay=.08)
    (z+model.residual(x)).square().sum().backward();optimizer.step()
    assert model.residual.weight.ne(0).any() and all(p.grad is None for p in model.base.network.parameters())
    assert before==base_fingerprint(model.base)
    after=model._base_logits(feature,common)
    torch.testing.assert_close(after[1],tail,rtol=0,atol=0);torch.testing.assert_close(after[2],alpha,rtol=0,atol=0)

def test_mixed_original_amount_percent_and_unknown_timing():
    pairs=pd.DataFrame({'target_quarter':['FY2011Q3','FY2011Q4','FY2012Q1'],'actual_report_available_at':['2020-01-02 09:31-05:00','2020-01-06 09:31-05:00','2020-01-08 09:31-05:00'],'pair_available_at':['2020-01-02 09:31-05:00','2020-01-06 09:31-05:00','2020-01-08 09:31-05:00'],'measurement_known':[True,True,False],'guide_unit':['pct_sequential_revenue_growth','USD_million','qualitative'],'percentage_conversion_supported':[True,np.nan,False],'guide_mid':[10.,np.nan,np.nan],'guide_mid_million':[np.nan,100.,np.nan],'guide_base_actual_million':[100.,np.nan,np.nan],'actual_revenue_million':[121.,120.,110.],'freshness':['unknown_update_coverage']*3})
    decisions=pd.Series(pd.to_datetime(['2020-01-02 09:29-05:00','2020-01-03 09:29-05:00','2020-01-07 09:29-05:00','2020-01-09 09:29-05:00'],utc=True))
    x,_,_=align_realizations(pairs,decisions)
    np.testing.assert_allclose(x[FEATURE],[np.nan,.1,.2,np.nan],equal_nan=True)
