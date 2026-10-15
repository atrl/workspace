import torch
from src.model.family_balanced_direct_utility import pair_loss,weighted_loss

def test_family_loss_keeps_independent_initial_cash_and_daily_units():
    w=torch.tensor([[.2,.5,.3],[.25,.5,.25]],dtype=torch.float64,requires_grad=True)
    r=torch.tensor([[.01,.003],[-.02,-.007]],dtype=torch.float64);years=torch.tensor([2020,2020])
    identity=lambda x:x
    a=weighted_loss(identity,{'A':(w,r,years),'B':(w.flip(0),r,years)},{'A':.5,'B':.5})
    b=.5*pair_loss(w,r,years)+.5*pair_loss(w.flip(0),r,years)
    torch.testing.assert_close(a,b)
    assert not torch.isclose(a,pair_loss(torch.cat([w,w.flip(0)]),torch.cat([r,r]),torch.cat([years,years])))
    initial=.7;marked=w[0]*torch.tensor([1.01,1.003,1.],dtype=w.dtype);pre=marked/marked.sum();turn=torch.stack([torch.tensor(initial,dtype=w.dtype),(w[1,:2]-pre[:2]).abs().sum()])
    expected=-(torch.log1p((w[:,:2]*r).sum(1)-.001*turn)-torch.log1p(r[:,1])).mean()+.006*turn.mean()
    torch.testing.assert_close(pair_loss(w,r,years),expected)
    a.backward();assert torch.isfinite(w.grad).all()
