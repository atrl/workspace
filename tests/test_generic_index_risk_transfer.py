import pandas as pd
import torch
from src.model.generic_index_risk_transfer import external_tail_loss,maturity_known_before
from src.workflow.pit_constituent_auxiliary_training_allocation import PitAuxiliaryAllocationNet
from src.workflow.generic_index_risk_transfer import external_labels

def test_external_tail_route_and_entry_relative_path_label():
    torch.manual_seed(17);net=PitAuxiliaryAllocationNet(primary_feature_count=39,common_feature_count=16,hidden_units=16,tail_hidden_units=8)
    groups={name:(torch.randn(7,16),torch.tensor([0.,1.,0.,0.,1.,0.,1.])) for name in ['QQQ','SPY','XLK']}
    external_tail_loss(net,groups).backward()
    assert all(p.grad is None for p in net.allocation_head.parameters())
    assert all(p.grad is None for p in net.alpha_head.parameters())
    assert all(p.grad is not None and p.grad.abs().sum()>0 for m in [net.tail_encoder,net.tail_head] for p in m.parameters())
    # H5 terminal positive but an entry-relative open trough breaches -6%.
    opening=pd.Series([99.,100.,93.,97.,101.,102.,103.])
    assert external_labels(opening).iloc[0]==1.
    assert external_labels(opening).iloc[1:].isna().all()

    maturity=pd.Series(pd.to_datetime(["2022-12-30 09:31","2023-01-03 09:31"]))
    assert maturity_known_before(maturity,pd.Timestamp("2023-01-01",tz="America/New_York")).tolist()==[True,False]
