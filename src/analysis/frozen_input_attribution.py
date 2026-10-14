"""Numerical input attribution of a frozen differentiable score, not causal alpha."""
import numpy as np
import torch


def integrated_inputs(score, inputs, baseline, *, nodes=64):
    if inputs.shape != baseline.shape or not torch.isfinite(inputs).all() or not torch.isfinite(baseline).all():
        raise ValueError('finite aligned inputs and baseline required')
    points, weights = np.polynomial.legendre.leggauss(nodes)
    delta = inputs - baseline
    gradients = torch.zeros_like(inputs)
    for point, weight in zip((points + 1) / 2, weights / 2):
        value = (baseline + float(point) * delta).detach().requires_grad_(True)
        gradient, = torch.autograd.grad(score(value).sum(), value)
        gradients += float(weight) * gradient.detach()
    contribution = delta * gradients
    with torch.no_grad():
        difference = score(inputs) - score(baseline)
    residual = contribution.sum(dim=1) - difference
    return contribution.detach(), residual.detach()
