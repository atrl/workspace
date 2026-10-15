"""Train allocation utility on fixed composition, keeping raw-weight inference."""
import torch
from src.model.tangent_allocator import TangentHeadAllocator


def composed_weights(weights):
    delta=torch.minimum(weights[:,0],weights[:,2]/2)
    return torch.stack([weights[:,0]-delta,weights[:,1]+3*delta,weights[:,2]-2*delta],dim=1)


class CompositionObjectiveAllocator(TangentHeadAllocator):
    def allocation_objective(self, **kwargs):
        return super().allocation_objective(**{**kwargs,'weights':composed_weights(kwargs['weights'])})
