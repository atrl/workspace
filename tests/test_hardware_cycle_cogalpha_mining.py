from __future__ import annotations

import numpy as np
import pandas as pd

import src.workflow as workflow_exports
from src.factor.cognitive_alpha import materialize_cognitive_alpha_features


def test_hardware_cycle_cogalpha_candidates_are_lagged_and_materialize():
    from src.factor.hardware_cycle_cogalpha import build_hardware_cycle_cogalpha_candidates

    dates = pd.date_range("2023-01-02", periods=120, freq="B")
    x = np.linspace(-1.0, 1.0, len(dates))
    frame = pd.DataFrame(
        {
            "datetime": dates,
            "hardware_ai_accelerator_ret20_lag1": x,
            "hardware_ai_accelerator_ret60_lag1": x * 0.8,
            "hardware_memory_storage_ret20_lag1": -x,
            "hardware_cloud_capex_ret60_lag1": np.sin(np.linspace(0.0, 6.0, len(dates))),
            "hardware_ai_server_ret20_lag1": np.cos(np.linspace(0.0, 3.0, len(dates))),
            "hardware_cpu_compute_ret20_lag1": x**2,
            "hardware_enterprise_storage_ret20_lag1": np.tanh(x),
            "hardware_cloud_gpu_modern_on_demand_ret21_lag1": -0.5 * x,
            "hardware_cloud_gpu_modern_on_demand_available_lag1": 1.0,
            "hardware_cloud_gpu_h100_on_demand_staleness_days_lag1": np.arange(len(dates)) % 31,
            "hardware_cloud_gpu_h100_on_demand_is_carried_lag1": (np.arange(len(dates)) % 5 != 0).astype(float),
        }
    ).set_index("datetime")

    candidates = build_hardware_cycle_cogalpha_candidates()
    features = materialize_cognitive_alpha_features(frame, candidates)

    names = {candidate.name for candidate in candidates}
    roles = {candidate.metadata.get("history_role") for candidate in candidates}
    assert {
        "hardware_compute_scarcity_accelerator_vs_gpu_price",
        "hardware_capex_confirmed_accelerator_momentum",
        "hardware_memory_bottleneck_spread",
        "hardware_server_supply_chain_confirmation",
        "hardware_cpu_compute_breadth",
        "hardware_storage_cycle_confirmation",
    }.issubset(names)
    assert {"proxy_promotable", "cloud_diagnostic_short_history"}.issubset(roles)
    assert all(candidate.declared_lookahead == 0 for candidate in candidates)
    assert all("lag1" in column for candidate in candidates for column in candidate.required_columns)
    assert set(features.columns) == names
    assert all(features[column].dropna().nunique() > 5 for column in features.columns)


def test_hardware_cycle_cogalpha_daily_semantics_payload_marks_carried_cloud_state():
    from src.workflow.hardware_cycle_cogalpha_mining import _daily_data_semantics_payload

    payload = _daily_data_semantics_payload(
        provider_uri="data/qlib/hardware_cycle_v1",
        raw_manifest="data/raw/external/hardware_cycle/hardware_cycle_raw_manifest.json",
    )

    assert payload["calendar_frequency"] == "day"
    assert payload["cloud_gpu_level_semantics"] == "last_observed_state_forward_filled_to_qlib_daily_calendar"
    assert payload["source_observed_flag"] == "<field>_source_observed"
    assert payload["carried_state_flag"] == "<field>_is_carried"
    assert payload["staleness_measure"] == "<field>_staleness_days"
    assert payload["model_use_rule"] == "use_lagged_state_features_only"


def test_hardware_cycle_cogalpha_workflow_exports_and_summary_payload(tmp_path):
    from src.workflow.hardware_cycle_cogalpha_mining import (
        HARDWARE_COGALPHA_EXPERIMENT,
        HardwareCycleCogAlphaMiningCardSpec,
        _hardware_cogalpha_summary_payload,
    )

    spec = HardwareCycleCogAlphaMiningCardSpec(root=tmp_path)
    summary = _hardware_cogalpha_summary_payload(
        spec=spec,
        provider_meta={"provider_uri": "/tmp/soxl_provider"},
        hardware_manifest={
            "provider_uri": "/tmp/hardware_cycle",
            "raw_manifest": "/tmp/hardware_cycle_raw_manifest.json",
            "calendar_start": "2019-01-02",
            "calendar_end": "2026-07-02",
            "market": "all",
            "region": "us",
        },
        hardware_feature_cols=[
            "hardware_ai_accelerator_ret20_lag1",
            "hardware_cloud_gpu_h100_on_demand_source_observed_lag1",
        ],
        factor_mining={
            "workflow": "cognitive_alpha_factor_mining",
            "factor_library": {"selected_count": 2, "selected_feature_names": ["hardware_memory_bottleneck_spread"]},
        },
        position_summary={
            "workflow": "cognitive_alpha_position_model",
            "promotion_decision": "not_promoted_failed_walk_forward_risk_or_return_gates",
            "selected_candidate": {"test": {"return_pct": 1.0, "max_drawdown_pct": -2.0}},
            "comparison": {"selected_return_pct": 1.0},
            "risk_gates": {"selected_passes_all": False},
            "walk_forward": {"fold_count": 4, "all_folds_pass_risk_gates": False},
            "model_families": ["lgb_xgb_consensus"],
        },
        latest_signal={"as_of": "2026-07-02", "action": "hold", "target_position": 0.0},
        sidecars={"summary": "summary.json"},
        recorder_uri=tmp_path / "qlib_recorder",
        recorder_id="recorder-test",
        recorder_local_dir=tmp_path / "qlib_recorder" / "experiment" / "recorder-test",
        as_of="2026-07-02",
    )

    assert workflow_exports.HardwareCycleCogAlphaMiningCardSpec is HardwareCycleCogAlphaMiningCardSpec
    assert workflow_exports.run_hardware_cycle_cogalpha_mining_card.__name__ == "run_hardware_cycle_cogalpha_mining_card"
    assert HARDWARE_COGALPHA_EXPERIMENT == "hardware_cycle_cogalpha_mining_v1"
    assert summary["daily_data_semantics"]["cloud_gpu_level_semantics"] == "last_observed_state_forward_filled_to_qlib_daily_calendar"
    assert summary["factor_mining"]["workflow"] == "cognitive_alpha_factor_mining"
    assert summary["hardware_cycle"]["feature_count"] == 2
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["recorder_only_artifacts"]["positions"].startswith("artifacts/")


def test_hardware_cycle_cogalpha_notebook_bootstraps_repo_root(tmp_path):
    from src.workflow.hardware_cycle_cogalpha_mining import HARDWARE_COGALPHA_EXPERIMENT, _write_notebook

    summary = {
        "experiment": HARDWARE_COGALPHA_EXPERIMENT,
        "promotion_decision": "not_promoted_failed_walk_forward_risk_or_return_gates",
        "as_of": "2026-07-02",
        "recorder_id": "recorder-test",
        "hardware_cycle": {"feature_count": 12, "provider_uri": "/tmp/hardware_cycle"},
        "factor_mining": {
            "factor_library": {
                "selected_count": 1,
                "selected_feature_names": ["hardware_memory_bottleneck_spread"],
            }
        },
        "position_model": {
            "selected_candidate": {"test": {"return_pct": 1.0, "max_drawdown_pct": -2.0}},
            "walk_forward": {"fold_count": 4, "all_folds_pass_risk_gates": False},
        },
        "latest_signal": {"as_of": "2026-07-02", "action": "hold", "target_position": 0.0},
    }

    _write_notebook(tmp_path, summary)

    notebook_text = (tmp_path / f"{HARDWARE_COGALPHA_EXPERIMENT}.ipynb").read_text(encoding="utf-8")
    assert "while not (ROOT / 'src').exists()" in notebook_text
    assert "sys.path.insert(0, ROOT.as_posix())" in notebook_text
    assert "run_hardware_cycle_cogalpha_mining_card(root=ROOT)" in notebook_text
