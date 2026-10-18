"""One synthetic vector contract; no market forward, autograd, fit or optimizer."""
import pytest
import torch

from src.model.aux_projection_allocator import (
    ProjectionNumericalError, assemble_projected_gradient, clip_once,
)


def test_combined_authority_projection_parity_and_one_global_clip(monkeypatch):
    rng = torch.get_rng_state().clone()
    gp, ga = torch.zeros(973), torch.zeros(973)
    gp[0], gp[154], ga[1] = 1., 4., 2.
    combined = gp + ga
    combined[1] = torch.nextafter(combined[1], torch.tensor(float('inf')))
    # Within the fixed parity allowance but not equal to split: retain authority.
    assert not torch.equal(combined, gp + ga)
    selected, audit = assemble_projected_gradient(gp, ga, combined)
    assert selected is combined and audit['no_conflict_original_combined_identity']
    zero = torch.zeros_like(gp)
    selected, audit = assemble_projected_gradient(zero, ga, ga)
    assert selected is ga and audit['zero_primary_norm']

    ga[0], ga[1] = -1., 3.
    combined = gp + ga
    snapshots = [x.clone() for x in (gp, ga, combined)]
    selected, audit = assemble_projected_gradient(gp, ga, combined)
    assert audit['strict_projection_triggered'] and audit['classification'] == 'resolvable'
    assert torch.equal(selected[:2], torch.tensor([1., 3.]))
    assert torch.equal(selected[154:], combined[154:])
    assert all(torch.equal(before, after) for before, after in zip(snapshots, (gp, ga, combined)))

    shapes = [(8, 16), (8,), (1, 8), (1,), (1, 8), (1,), (16, 47), (16,), (3, 16), (3,)]
    parameters, offset = [], 0
    for shape in shapes:
        parameter = torch.nn.Parameter(torch.zeros(shape))
        parameter.grad = selected[offset:offset + parameter.numel()].reshape(shape).clone()
        offset += parameter.numel()
        parameters.append(parameter)
    assert offset == 973
    original_clip = torch.nn.utils.clip_grad_norm_
    calls = []
    def counted_clip(params, max_norm):
        calls.append(max_norm)
        return original_clip(params, max_norm=max_norm)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', counted_clip)
    norm = clip_once(iter(parameters))
    clipped = torch.cat([p.grad.reshape(-1) for p in parameters])
    coefficient = torch.clamp(2. / (norm + 1e-6), max=1.)
    assert calls == [2.] and norm > 2 and coefficient < 1
    assert torch.equal(clipped, selected * coefficient)
    assert not torch.equal(clipped[154:], selected[154:])
    assert all(torch.count_nonzero(p) == 0 for p in parameters)

    # Strictly negative remains a projection even below the numerical screen.
    tiny = ga.clone(); tiny[0] = -1e-12
    _, audit = assemble_projected_gradient(gp, tiny, gp + tiny)
    assert audit['strict_projection_triggered'] and audit['classification'] == 'unresolved'
    # Large private coordinates must not conceal a shared-local parity failure.
    dominant = gp.clone(); dominant[154] = 1e8
    bad_combined = dominant + ga; bad_combined[0] += 1e-3
    with pytest.raises(ProjectionNumericalError) as error:
        assemble_projected_gradient(dominant, ga, bad_combined)
    assert error.value.audit['full_parity_pass'] and not error.value.audit['shared_parity_pass']
    assert torch.equal(rng, torch.get_rng_state())
