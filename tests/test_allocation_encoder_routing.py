import torch
from torch import nn
from src.model.allocation_encoder_routing import DetachedInputLinear
from src.workflow.pit_constituent_auxiliary_training_allocation import PitAuxiliaryAllocationNet


def test_identical_forward_and_parameter_count_with_classification_only_gradient_stop():
    torch.manual_seed(71)
    base=PitAuxiliaryAllocationNet(primary_feature_count=3,common_feature_count=2,hidden_units=4,tail_hidden_units=3)
    import copy
    routed=copy.deepcopy(base)
    for name in ['tail_head','alpha_head']:
        old=getattr(routed,name);head=DetachedInputLinear(old.in_features,old.out_features);head.load_state_dict(old.state_dict());setattr(routed,name,head)
    x,c=torch.randn(5,3),torch.randn(5,2)
    for a,b in zip(base(x,c),routed(x,c)):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert sum(p.numel() for p in base.parameters())==sum(p.numel() for p in routed.parameters())
    for net in [base,routed]:
        _,tail,alpha=net(x,c);(tail.square().mean()+alpha.square().mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in base.tail_encoder.parameters())
    assert all(p.grad is None for p in routed.tail_encoder.parameters())
    assert all(p.grad is not None for p in routed.tail_head.parameters())
    routed.zero_grad(set_to_none=True)
    auxiliary=routed.tail_encoder(c);(routed.tail_head(auxiliary).square().mean()+routed.alpha_head(auxiliary).square().mean()).backward()
    assert all(p.grad is None for p in routed.tail_encoder.parameters())
    routed.zero_grad(set_to_none=True);base.zero_grad(set_to_none=True)
    for net in [base,routed]:net(x,c)[0].square().mean().backward()
    for a,b in zip(base.tail_encoder.parameters(),routed.tail_encoder.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
