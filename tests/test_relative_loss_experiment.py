import copy
import hashlib
from pathlib import Path

import pytest

from src.workflow.relative_loss_experiment import verify_selection_inputs


def _plan(tmp_path, monkeypatch):
    import src.data.provider_content_hash as provider
    monkeypatch.setattr(provider, "provider_content_hash", lambda path: ("a" * 64, {}))
    marker = tmp_path / "source.json"
    marker.write_text("{}")
    return dict(frozen_code_sha256={}, frozen_evidence_files_sha256={marker.name: hashlib.sha256(marker.read_bytes()).hexdigest()},
                providers={"price": {"provider_uri": "data", "content_sha256": "a" * 64}},
                horizon_sessions=5, costs_bps=[10., 25.], robust_radius=1., candidate_names=["prior", "ridge_10", "ridge_100", "ridge_1000"],
                tie_priority=["prior", "ridge_1000", "ridge_100", "ridge_10"], outer_years=[2016, 2017],
                maximum_new_fits=2, fixed_new_inner_fits=0, authorization={"orders": False}, promotion_eligible=False)


def test_source_identity_fixed_budget_and_no_orders_gate(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    plan = _plan(tmp_path, monkeypatch)
    verify_selection_inputs(evidence_root=tmp_path, source_root=root, plan=plan)
    for key, value, match in [("costs_bps", [0.], "reference"), ("fixed_new_inner_fits", 4, "budget"),
                              ("maximum_new_fits", 3, "budget"), ("promotion_eligible", True, "promotion"),
                              ("authorization", {"orders": True}, "orders")]:
        bad = copy.deepcopy(plan)
        bad[key] = value
        with pytest.raises(ValueError, match=match):
            verify_selection_inputs(evidence_root=tmp_path, source_root=root, plan=bad)
    (tmp_path / "source.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="frozen evidence"):
        verify_selection_inputs(evidence_root=tmp_path, source_root=root, plan=plan)
