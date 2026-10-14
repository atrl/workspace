"""Fixed gradient-routing ablation; preserve classifier meaning and forward graph."""
import torch
from torch import nn
from src.model.minute_pit_allocator import MinutePitAllocator


class DetachedInputLinear(nn.Linear):
    """Train the classifier head without its loss updating its input encoder."""
    def forward(self, value):
        return super().forward(value.detach())


class AllocationEncoderAllocator(MinutePitAllocator):
    def _initialize(self):
        network = super()._initialize()
        for name in ('tail_head', 'alpha_head'):
            old = getattr(network, name)
            # Preserve RNG as well as parameters; no extra initialization effect.
            with torch.random.fork_rng(devices=[]):
                head = DetachedInputLinear(old.in_features, old.out_features, bias=old.bias is not None)
            head.load_state_dict(old.state_dict())
            setattr(network, name, head)
        return network
