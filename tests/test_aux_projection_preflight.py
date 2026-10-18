"""One synthetic regression contract for the approved arithmetic boundaries."""
import torch

from src.workflow.aux_projection_preflight import asymmetric_projection, vector_audit


def test_projection_identity_private_and_shared_local_parity():
    p = torch.tensor([1., 0.], dtype=torch.float32)
    a = torch.tensor([-1., 2.], dtype=torch.float32)
    projected, _, triggered = asymmetric_projection(p, a)
    assert triggered and torch.equal(projected, torch.tensor([0., 2.]))
    for primary, auxiliary in [(p, p), (torch.zeros_like(p), a)]:
        result, _, triggered = asymmetric_projection(primary, auxiliary)
        assert not triggered and torch.equal(result, auxiliary)
    gp = torch.tensor([1., 0., 1e8], dtype=torch.float32)
    ga = torch.tensor([-1., 2., 0.], dtype=torch.float32)
    original = gp + ga
    original[0] += 1e-3
    audit, split, candidate = vector_audit(gp, ga, original, shared_count=2)
    assert audit['full_parity_pass'] and not audit['shared_parity_pass']
    assert audit['classification'] == 'needs_numerical_review'
    assert torch.equal(candidate[2:], split[2:])
    audit, split, candidate = vector_audit(gp, ga, gp + ga, shared_count=2)
    assert audit['classification'] == 'resolvable'
    assert audit['shared_parity_pass'] and audit['full_parity_pass']
    assert split.dtype == candidate.dtype == gp.dtype
