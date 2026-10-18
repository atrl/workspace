from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from src.model.minute_pit_allocator import MinutePitAllocator
from src.model.stage1_domain_heads import DomainHeadAllocator, initialization_contract
from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES, PitAuxiliaryAllocationSpec


def models():
    spec = asdict(PitAuxiliaryAllocationSpec(root=Path(".")))
    args = dict(base_features=["x", "y"], extra_features=[], feature_masks={}, spec=spec,
                seed=71, cutoff="2023-01-01T00:00:00-05:00", train_end="2022-12-30",
                minimum_observations=1)
    return MinutePitAllocator(**args), DomainHeadAllocator(**args)


def test_initial_forward_rng_and_independent_auxiliary_storage():
    control, candidate = models()
    audit = initialization_contract(control, candidate)
    assert audit["candidate_parameters"] - audit["original_parameters"] == 18
    net = candidate._initialize()
    before = net.tail_head.weight.clone()
    with torch.no_grad():
        net.aux_tail_head.weight.add_(1)
    assert torch.equal(net.tail_head.weight, before)


def test_auxiliary_gradient_reaches_encoder_but_not_primary_heads():
    control, candidate = models()
    common = torch.linspace(-1, 1, 5 * len(COMMON_FEATURES)).reshape(5, -1)
    target = torch.tensor([0., 1., 0., 1., 1.])
    for model in [control, candidate]:
        model.network = model._initialize()
        logits = model.auxiliary_logits(model.network.tail_encoder(common))
        sum(F.binary_cross_entropy_with_logits(value, target) for value in logits).backward()
        assert model.network.tail_encoder[0].weight.grad.abs().sum() > 0
    for name in ["tail_head", "alpha_head"]:
        assert getattr(control.network, name).weight.grad is not None
        assert getattr(candidate.network, name).weight.grad is None
        assert getattr(candidate.network, "aux_" + name).weight.grad.abs().sum() > 0
    candidate.network.zero_grad(set_to_none=True)
    _, tail, alpha = candidate.network(torch.zeros(5, 2), common)
    (F.binary_cross_entropy_with_logits(tail, target) + F.binary_cross_entropy_with_logits(alpha, target)).backward()
    assert candidate.network.tail_head.weight.grad.abs().sum() > 0
    assert candidate.network.aux_tail_head.weight.grad is None
