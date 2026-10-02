"""Exact fee funding with an explicit first-order generalized gradient.

This version does not alter the sealed legacy allocation kernel.  The ending
normalized state is the smooth post-fee-target map, independent of funding.
Scalar funding is nonsmooth: zero actual trades use sign zero in the implicit
equation.  This convention is not every feasible one-sided fee derivative.
"""
from __future__ import annotations

import math
import torch

from src.model.benchmark_anchor_utility import NumericalFitFailure


class _FundedNAV(torch.autograd.Function):
    """First derivatives of n + c |n z_r - h_r|_1 = 1.

    Away from kinks this is ordinary implicit differentiation.  At zero legs
    the declared sign-zero selection is used consistently for the scalar
    funding identity.  No straight-through or smoothed forward rule is used.
    """
    @staticmethod
    def forward(ctx, target, held, rate):
        signs = target.new_tensor(((-1., -1.), (-1., 1.), (1., -1.), (1., 1.)))
        z, h = target[..., None, :2], held[..., None, :2]
        candidates = (1 + rate * (signs * h).sum(-1)) / (1 + rate * (signs * z).sum(-1))
        residual = (candidates + rate * (candidates[..., None] * z - h).abs().sum(-1) - 1).abs()
        selected = residual.argmin(-1, keepdim=True)
        n = candidates.gather(-1, selected).squeeze(-1)
        trade = n[..., None] * target[..., :2] - held[..., :2]
        error = (n + rate * trade.abs().sum(-1) - 1).abs()
        if not bool(torch.isfinite(n).all()) or bool((n <= 0).any()) or bool((error > 1e-12).any()):
            raise NumericalFitFailure("invalid exact post-fee funding root")
        ctx.save_for_backward(n, target, torch.sign(trade))
        ctx.rate = rate
        return n

    @staticmethod
    def backward(ctx, upstream):
        n, target, signs = ctx.saved_tensors
        rate = ctx.rate
        denominator = 1 + rate * (signs * target[..., :2]).sum(-1)
        factor = upstream * rate / denominator
        dz = torch.cat((-factor[..., None] * n[..., None] * signs,
                        torch.zeros_like(factor[..., None])), -1)
        dh = torch.cat((factor[..., None] * signs,
                        torch.zeros_like(factor[..., None])), -1)
        return dz, dh, None


def _validate_torch(held, target, returns, rate):
    if not math.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("blocked: invalid proportional fee rate")
    if (not isinstance(held, torch.Tensor) or held.shape[-1:] != (3,)
            or not held.is_floating_point() or target.shape != held.shape or returns.shape != held.shape
            or target.dtype != held.dtype or returns.dtype != held.dtype
            or target.device != held.device or returns.device != held.device):
        raise ValueError("blocked: settlement requires matching three-asset tensors")
    for value in (held, target):
        if (not bool(torch.isfinite(value).all()) or bool((value < -1e-12).any())
                or bool(((value.sum(-1) - 1).abs() > 1e-10).any())):
            raise NumericalFitFailure("invalid settlement simplex")
    if not bool(torch.isfinite(returns).all()) or bool((returns <= -1).any()):
        raise NumericalFitFailure("invalid physical settlement returns")


def settle_torch(held, target, returns, rate):
    """Vectorized funded accounting; shapes [..., 3], rate is a fixed float.

    The public keys match the legacy kernel.  ``gross_return`` retains its
    legacy cash-addback meaning; ``gross_position_multiplier`` is the actual
    target-weighted gross multiplier used for economic fee decomposition.
    """
    target = held if target is None else target
    rate = float(rate)
    _validate_torch(held, target, returns, rate)
    n = _FundedNAV.apply(target, held, rate)
    risky = n[..., None] * target[..., :2]
    turnover = (risky - held[..., :2]).abs().sum(-1)
    fees = rate * turnover
    executable = torch.cat((risky, (1 - risky.sum(-1))[..., None]), -1)
    opened = n[..., None] * target
    gross_values = target * (1 + returns)
    gross = gross_values.sum(-1)
    ending = n[..., None] * gross_values
    net = n * gross
    # These value checks never replace a gradient or alter an accepted trade.
    cash_error = (executable[..., 2] - fees - opened[..., 2]).abs()
    if (not bool(torch.isfinite(net).all()) or bool((net <= 0).any())
            or bool((opened < -1e-12).any()) or bool((cash_error > 1e-12).any())):
        raise NumericalFitFailure("invalid fee-funded NAV or cash debit")
    return {
        "net_equity": net, "net_return": net - 1,
        "gross_return": net + fees - 1, "gross_position_multiplier": gross,
        "ending_values": ending, "ending_weights": gross_values / gross[..., None],
        "turnover": turnover, "open_turnover": turnover, "fees": fees,
        "open_net_equity": n, "open_net_weights": target,
        "execution_target_weights": executable,
    }


def scalar_settle(h0, h1, h2, z0, z1, z2, r0, r1, rate):
    """NumPy-path scalar counterpart with the legacy return tuple contract."""
    values = (h0, h1, h2, z0, z1, z2, r0, r1, rate)
    if (not all(map(math.isfinite, values)) or not 0 <= rate < 1
            or min(h0, h1, h2, z0, z1, z2) < -1e-12
            or abs(h0 + h1 + h2 - 1) > 1e-10 or abs(z0 + z1 + z2 - 1) > 1e-10
            or min(r0, r1) <= -1):
        raise NumericalFitFailure("invalid scalar settlement inputs")
    if (z0 == h0 and z1 == h1 and z2 == h2) or rate == 0:
        n = 1.
    else:
        best, n = math.inf, math.nan
        for s0, s1 in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            candidate = (1 + rate * (s0 * h0 + s1 * h1)) / (1 + rate * (s0 * z0 + s1 * z1))
            error = abs(candidate + rate * (abs(candidate * z0 - h0) + abs(candidate * z1 - h1)) - 1)
            if error < best:
                best, n = error, candidate
    x0, x1, x2 = n * z0, n * z1, n * z2
    turnover = abs(x0 - h0) + abs(x1 - h1)
    fee = rate * turnover
    execution_cash = 1 - x0 - x1
    g0, g1 = z0 * (1 + r0), z1 * (1 + r1)
    gross = g0 + g1 + z2
    net = n * gross
    if (not math.isfinite(net) or net <= 0 or n <= 0 or x2 < -1e-12
            or abs(n + fee - 1) > 1e-12 or abs(execution_cash - fee - x2) > 1e-12):
        raise NumericalFitFailure("invalid scalar funding identity")
    return ((g0 / gross, g1 / gross, z2 / gross), net, fee, turnover,
            (x0, x1, execution_cash), (x0, x1, x2))


__all__ = ["settle_torch", "scalar_settle"]
