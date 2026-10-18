"""Separate auxiliary output maps while retaining the original shared encoder."""
from copy import deepcopy
import hashlib

import torch

from src.model.minute_pit_allocator import MinutePitAllocator


class DomainHeadAllocator(MinutePitAllocator):
    """Only auxiliary BCE uses the cloned heads; primary inference is unchanged.

    Cloning consumes no random numbers. The extra 18 parameters and resulting
    global clipping/optimizer dynamics are part of the intervention, not a
    capacity-matched proof of negative transfer.
    """

    def _initialize(self):
        network = super()._initialize()
        network.aux_tail_head = deepcopy(network.tail_head)
        network.aux_alpha_head = deepcopy(network.alpha_head)
        return network

    def auxiliary_logits(self, state):
        return (self.network.aux_tail_head(state).squeeze(1),
                self.network.aux_alpha_head(state).squeeze(1))


def initialization_contract(control, candidate):
    """No fitting: check all original tensors and both domains before training."""
    left = control._initialize()
    left_rng = torch.get_rng_state().clone()
    right = candidate._initialize()
    if not torch.equal(left_rng, torch.get_rng_state()):
        raise ValueError("domain heads changed initialization RNG consumption")
    for name, tensor in left.state_dict().items():
        if not torch.equal(tensor, right.state_dict()[name]):
            raise ValueError("original initial tensor differs: " + name)
    for main, aux in [("tail_head", "aux_tail_head"), ("alpha_head", "aux_alpha_head")]:
        for name, tensor in getattr(left, main).state_dict().items():
            if not torch.equal(tensor, getattr(right, aux).state_dict()[name]):
                raise ValueError("auxiliary head is not an exact clone")
    primary = torch.linspace(-1, 1, 3 * len(control.base_features)).reshape(3, -1)
    common = torch.linspace(-.5, .5, 3 * left.tail_encoder[0].in_features).reshape(3, -1)
    with torch.no_grad():
        for expected, actual in zip(left(primary, common), right(primary, common)):
            if not torch.equal(expected, actual):
                raise ValueError("primary initial forward differs")
        encoded = right.tail_encoder(common)
        for main, aux in [("tail_head", "aux_tail_head"), ("alpha_head", "aux_alpha_head")]:
            if not torch.equal(getattr(right, main)(encoded), getattr(right, aux)(encoded)):
                raise ValueError("auxiliary initial forward differs")
    return {
        "original_parameters": sum(p.numel() for p in left.parameters()),
        "candidate_parameters": sum(p.numel() for p in right.parameters()),
        "original_tensors_identical": True, "initial_primary_and_auxiliary_outputs_identical": True,
        "initial_rng_state_identical": True,
        "original_initial_sha256": hashlib.sha256(b"".join(
            name.encode() + value.numpy().tobytes() for name, value in sorted(left.state_dict().items())
        )).hexdigest(),
    }
