"""Fixed one-sided annual benchmark shortfall objective for paired research."""
import torch
from src.model.minute_pit_allocator import MinutePitAllocator


def annual_shortfall_penalty(annual):
    # vector_norm has a finite zero subgradient when all years beat benchmark.
    return torch.linalg.vector_norm(torch.relu(-annual)) / annual.numel() ** 0.5


class AnnualShortfallAllocator(MinutePitAllocator):
    def allocation_objective(self, **kwargs):
        loss, turnover, annual, components = super().allocation_objective(**kwargs)
        loss = loss - 0.50 * annual.std(unbiased=False) + 0.50 * annual_shortfall_penalty(annual)
        return loss, turnover, annual, components
