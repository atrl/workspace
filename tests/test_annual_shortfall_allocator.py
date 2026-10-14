import torch
from src.model.annual_shortfall_allocator import annual_shortfall_penalty


def test_shortfall_is_monotone_with_finite_gradient_at_zero():
    for values in [[0.,0.],[.01,.02],[-.03,.01],[-.02,-.01]]:
        x=torch.tensor(values,requires_grad=True)
        loss=-x.mean()+.5*annual_shortfall_penalty(x)
        loss.backward()
        assert torch.isfinite(x.grad).all()
        assert (x.grad<0).all()
        if min(values)>=0:assert annual_shortfall_penalty(x).item()==0
