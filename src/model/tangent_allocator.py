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
