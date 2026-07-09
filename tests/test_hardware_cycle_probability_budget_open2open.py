from __future__ import annotations

from pathlib import Path

import pandas as pd

import src.workflow as workflow_exports
from src.workflow.hardware_cycle_probability_budget_open2open import (
    HARDWARE_EXPERIMENT,
    HardwareCycleProbabilityBudgetOpen2OpenCardSpec,
    _hardware_cycle_feature_groups,
    _hardware_summary_payload,
    _write_notebook,
)
from src.workflow.probability_budget_open2open import _position_spec_for_probability_budget


def test_hardware_cycle_feature_group_extends_probability_budget_without_old_architecture():
    frame = pd.DataFrame(
        {
            "alpha158_MA5": [1.0],
            "soxx_const_breadth_ret5_pos_lag1": [0.5],
            "hardware_ai_accelerator_ret5_lag1": [0.1],
            "hardware_cloud_gpu_h100_on_demand_available_lag1": [1.0],
        }
    )
    base_groups = {"base_cogalpha_soxx": ["alpha158_MA5", "soxx_const_breadth_ret5_pos_lag1"]}

    groups = _hardware_cycle_feature_groups(
        base_groups,
        hardware_feature_cols=[
            "hardware_ai_accelerator_ret5_lag1",
            "hardware_cloud_gpu_h100_on_demand_available_lag1",
        ],
        frame=frame,
    )
    spec = HardwareCycleProbabilityBudgetOpen2OpenCardSpec(root=Path.cwd())
    position_spec = _position_spec_for_probability_budget(
        groups["base_cogalpha_soxx_hardware"],
        spec=spec,
        test=("2026-01-02", "2026-07-02"),
    )

    assert groups["base_cogalpha_soxx_hardware"] == [
        "alpha158_MA5",
        "soxx_const_breadth_ret5_pos_lag1",
        "hardware_ai_accelerator_ret5_lag1",
        "hardware_cloud_gpu_h100_on_demand_available_lag1",
    ]
    assert all(item["position_mode"] == "probability" for item in position_spec.position_specs or [])
    assert all("base_position_col" not in item for item in position_spec.position_specs or [])
    assert all("risk_gate_col" not in item for item in position_spec.position_specs or [])


def test_hardware_summary_payload_records_external_provider_provenance():
    payload = _hardware_summary_payload(
        provider_manifest={
            "provider_uri": "/tmp/hardware_cycle",
            "calendar_start": "2019-01-02",
            "calendar_end": "2026-07-02",
            "raw_manifest": "/tmp/hardware_cycle_raw_manifest.json",
        },
        factor_attrs={
            "source": "qlib.data.D.features",
            "provider_uri": "/tmp/hardware_cycle",
            "feature_cols": ["hardware_ai_accelerator_ret5_lag1"],
            "lag_days": 1,
            "limitations": ["short cloud GPU history"],
        },
    )

    assert workflow_exports.HardwareCycleProbabilityBudgetOpen2OpenCardSpec is HardwareCycleProbabilityBudgetOpen2OpenCardSpec
    assert workflow_exports.run_hardware_cycle_probability_budget_open2open_card.__name__ == "run_hardware_cycle_probability_budget_open2open_card"
    assert payload["provider_uri"] == "/tmp/hardware_cycle"
    assert payload["feature_count"] == 1
    assert payload["lag_days"] == 1
    assert HARDWARE_EXPERIMENT == "hardware_cycle_probability_budget_open2open_v1"


def test_hardware_notebook_bootstraps_repo_root(tmp_path):
    summary = {
        "experiment": HARDWARE_EXPERIMENT,
        "position_model": {
            "selected_candidate": {
                "test": {
                    "return_pct": 2.0,
                    "max_drawdown_pct": -1.0,
                    "max_daily_loss_pct": -0.5,
                    "cvar_5pct_pct": -0.2,
                }
            },
            "walk_forward": {
                "fold_count": 4,
                "all_folds_pass_risk_gates": False,
                "all_folds_pass_return_gate": True,
                "min_test_return_pct": 1.0,
            },
            "comparison": {"selected_vs_base_constrained_ref_return_gap_pct": -10.0},
        },
        "latest_signal": {"as_of": "2026-07-02", "action": "hold", "target_percent": 1.5},
        "hardware_cycle": {
            "feature_count": 191,
            "provider_uri": "/tmp/hardware_cycle",
            "calendar_start": "2019-01-02",
            "calendar_end": "2026-07-02",
            "lag_days": 1,
        },
        "promotion_decision": "not_promoted_failed_walk_forward_risk_or_return_gates",
        "recorder_id": "recorder-test",
    }

    _write_notebook(tmp_path, summary)

    notebook_text = (tmp_path / f"{HARDWARE_EXPERIMENT}.ipynb").read_text(encoding="utf-8")
    assert "while not (ROOT / 'src').exists()" in notebook_text
    assert "sys.path.insert(0, ROOT.as_posix())" in notebook_text
    assert "run_hardware_cycle_probability_budget_open2open_card(root=ROOT)" in notebook_text
