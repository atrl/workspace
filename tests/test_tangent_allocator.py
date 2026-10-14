import copy
import torch
from src.model.tangent_allocator import head_anchor,tangent_head,audit_tangent_initialization
from src.workflow.pit_constituent_auxiliary_training_allocation import PitAuxiliaryAllocationNet


def test_tangent_head_preserves_origin_value_and_derivative_but_reduces_capacity():
    torch.manual_seed(71)
    control=PitAuxiliaryAllocationNet(primary_feature_count=4,common_feature_count=3,hidden_units=8,tail_hidden_units=2)
    candidate=copy.deepcopy(control);anchor=head_anchor(control);rng=torch.get_rng_state().clone()
    candidate.allocation_head=tangent_head(control.allocation_head,anchor)
    assert torch.equal(torch.get_rng_state(),rng)
    audit=audit_tangent_initialization(control,candidate)
    assert audit['candidate_parameters']<audit['control_parameters']
    x=torch.zeros(1,4);c=torch.zeros(1,3)
    for a,b in zip(control(x,c),candidate(x,c)):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
    assert not torch.allclose(control.allocation_head(anchor+3),candidate.allocation_head(anchor+3))
    candidate(x,c)[0].sum().backward()
    assert all(p.grad is not None for p in candidate.tail_encoder.parameters())
