"""Reduce allocation-head capacity around a fixed standardized-origin anchor."""
import torch
from torch import nn
from src.model.minute_pit_allocator import MinutePitAllocator


def head_anchor(network):
    first=network.allocation_head[0]
    common=first.weight.new_zeros((1,network.tail_encoder[0].in_features))
    primary=first.weight.new_zeros((1,first.in_features-network.tail_head.in_features))
    return torch.cat([primary,network.tail_encoder(common)],dim=1).detach()


def tangent_head(head,anchor):
    first,activation,last=head
    if not isinstance(activation,nn.Tanh):raise ValueError('expected fixed Tanh head')
    with torch.random.fork_rng(devices=[]):
        linear=nn.Linear(first.in_features,last.out_features).to(first.weight)
    with torch.no_grad():
        hidden=torch.tanh(first(anchor))
        jacobian=(last.weight*(1-hidden[0].square()).unsqueeze(0))@first.weight
        linear.weight.copy_(jacobian)
        linear.bias.copy_(last(hidden)[0]-jacobian@anchor[0])
    return linear


class TangentHeadAllocator(MinutePitAllocator):
    def _initialize(self):
        network=super()._initialize()
        network.allocation_head=tangent_head(network.allocation_head,head_anchor(network))
        return network


def audit_tangent_initialization(control,candidate):
    for key,value in control.state_dict().items():
        if not key.startswith('allocation_head.'):
            torch.testing.assert_close(value,candidate.state_dict()[key],rtol=0,atol=0)
    anchor=head_anchor(control)
    before=control.allocation_head(anchor);after=candidate.allocation_head(anchor)
    jacobian=torch.autograd.functional.jacobian(lambda x:control.allocation_head(x.unsqueeze(0))[0],anchor[0])
    torch.testing.assert_close(before,after,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(jacobian,candidate.allocation_head.weight,rtol=1e-6,atol=1e-6)
    old=sum(p.numel() for p in control.parameters());new=sum(p.numel() for p in candidate.parameters())
    if not new<old:raise ValueError('capacity was not reduced')
    return {'control_parameters':old,'candidate_parameters':new,'anchor_logit_max_error':float((before-after).abs().max().detach()),'anchor_jacobian_max_error':float((jacobian-candidate.allocation_head.weight).abs().max().detach()),'unchanged_non_head_state':True}


def audit_zero_extended_heads(control,candidate,*,shared_primary_count,extra_count):
    """Match existing coefficients and leave new numeric channels initially silent."""
    for key,value in control.state_dict().items():
        if not key.startswith('allocation_head.'):
            torch.testing.assert_close(value,candidate.state_dict()[key],rtol=0,atol=0)
    a,b=control.allocation_head,candidate.allocation_head
    if b.in_features!=a.in_features+extra_count:raise ValueError('wrong extra-channel dimension')
    torch.testing.assert_close(a.weight[:,:shared_primary_count],b.weight[:,:shared_primary_count],rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(a.weight[:,shared_primary_count:],b.weight[:,shared_primary_count+extra_count:],rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(a.bias,b.bias,rtol=1e-6,atol=1e-6)
    if not b.weight[:,shared_primary_count:shared_primary_count+extra_count].eq(0).all():raise ValueError('new numeric channels not zero initialized')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0);x=torch.randn(7,shared_primary_count);extra=torch.randn(7,extra_count);common=torch.randn(7,control.tail_encoder[0].in_features)
        left=control(x,common);right=candidate(torch.cat([x,extra],dim=1),common)
        for u,v in zip(left,right):torch.testing.assert_close(u,v,rtol=1e-6,atol=1e-6)
    return {'control_parameters':sum(p.numel() for p in control.parameters()),'candidate_parameters':sum(p.numel() for p in candidate.parameters()),'new_numeric_channels':extra_count,'initial_logit_max_difference':float((left[0]-right[0]).abs().max().detach()),'new_channels_zero_initialized':True}
