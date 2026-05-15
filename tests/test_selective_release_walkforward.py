from __future__ import annotations

import pandas as pd

from scripts.evaluate_selective_release_walkforward import (
    gate_eligible_mask,
    variant_subset,
    walk_forward_actor_specs,
)


def test_walk_forward_actor_specs_follow_profile_artifact_layout():
    specs = walk_forward_actor_specs("balanced_rule_residual_v22")

    assert [spec.window for spec in specs] == ["wf_2020", "wf_2022", "wf_2023_2024"]
    assert specs[0].actor_path.as_posix().endswith(
        "runtime/model/card/balanced_rule_residual_v22_wf_2020/"
        "soxl_balanced_rule_residual_v22_wf_2020_actor.pt"
    )
    assert specs[2].start == "2024-01-01"
    assert specs[2].end == "2024-12-31"


def test_retained_variant_subset_keeps_baseline_and_recovery_gate():
    variants = variant_subset(retained_only=True)

    assert [name for name, _ in variants] == ["baseline", "selective_release_recovery_v1"]


def test_gate_eligible_mask_counts_signal_confirmation_inside_regime_band():
    frame = pd.DataFrame(
        {
            "regime_risk": [0.50, 0.70, 0.50],
            "RB_COOLDOWN_PROXY": [0.30, 0.30, 0.10],
            "TRANS_VIX_DIVERGENCE": [0.01, 0.10, 0.10],
            "CICC_RETURN_VOLUME_SYNC": [0.72, 0.72, 0.72],
        }
    )
    config = {
        "max_regime_risk": 0.60,
        "cooldown_threshold": 0.20,
        "vix_threshold": 0.05,
        "volume_sync_threshold": 0.70,
    }

    assert gate_eligible_mask(frame, config).tolist() == [True, False, False]
