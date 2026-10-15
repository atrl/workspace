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


def test_composition_objective_transforms_allocation_gradient_without_changing_inference():
    import numpy as np
    from src.model.composition_objective_allocator import CompositionObjectiveAllocator,composed_weights
    from src.model.nominal_beta_composition import cancel_leverage_cash
    from src.model.tangent_allocator import TangentHeadAllocator
    from src.workflow.pit_constituent_auxiliary_training_allocation import _allocation_loss,AuxiliaryTrainingCandidate
    weights=torch.tensor([[.2,.5,.3],[.1,.5,.4],[.25,.55,.2],[.2,.4,.4],[.1,.6,.3],[.2,.6,.2]],dtype=torch.float64,requires_grad=True)
    mapped=composed_weights(weights)
    np.testing.assert_allclose(mapped.detach().numpy(),np.array([cancel_leverage_cash(x) for x in weights.detach().numpy()]),rtol=0,atol=1e-15)
    kwargs=dict(weights=weights,returns=torch.tensor([[.03,.01],[-.06,-.02],[.09,.03],[-.03,-.01],[.06,.02],[.03,.01]],dtype=torch.float64),years=torch.tensor([2020]*3+[2021]*3),cash_target=torch.zeros(6,dtype=torch.float64),tail_event=torch.zeros(6,dtype=torch.float64),candidate=AuxiliaryTrainingCandidate(name='fixed',auxiliary_weight=.05,include_union_backfill=False),drawdown_target_pct=-30.,worst_5d_target_pct=-15.,minimum_cash_weight=.15,cost_bps=10,turnover_weight=.006)
    candidate=object.__new__(CompositionObjectiveAllocator)
    actual=candidate.allocation_objective(**kwargs);expected=_allocation_loss(**{**kwargs,'weights':mapped})
    for left,right in zip(actual[:3],expected[:3]):torch.testing.assert_close(left,right,rtol=0,atol=0)
    gradient=torch.autograd.grad(actual[0],weights,retain_graph=True)[0]
    baseline_gradient=torch.autograd.grad(_allocation_loss(**kwargs)[0],weights)[0]
    assert torch.isfinite(gradient).all() and not torch.allclose(gradient,baseline_gradient)
    assert CompositionObjectiveAllocator.predict is TangentHeadAllocator.predict
    a=torch.tensor(.2,requires_grad=True);b=torch.tensor(.2,requires_grad=True);torch.minimum(a,b).backward()
    assert a.grad==.5 and b.grad==.5
