from __future__ import annotations

import csv
import json

import pytest

from src.soxl_quant.app.strict_retraining_candidate_queue import (
    derive_strict_retraining_candidate_queue,
    write_strict_retraining_candidate_queue,
)


def _strict_gate_summary() -> dict[str, object]:
    return {
        "passing_profile_count": 0,
        "reference_profile": "soxl_backtest_policy_reimplementation_v1",
        "reference_return_pct": 146.043798,
        "current_best_return_pct": 224.361539,
        "required_best_return_pct": 213.143462,
        "max_walkforward_drawdown_pct": 20.0,
        "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
        "closest_wf_drawdown_candidate": {
            "profile": "mode_conditioned_rebound_reward_train_v2",
            "worst_wf_max_drawdown_pct": 21.83172,
            "wf_drawdown_gap_pct": 1.83172,
            "missing_wf_window_count": 0,
            "ytd_return_pct": 177.573569,
            "best_return_gap_pct": 35.569893,
        },
        "closest_ytd_return_candidate": {
            "profile": "smoke",
            "ytd_return_pct": 224.361539,
            "best_return_gap_pct": 0.0,
            "missing_wf_window_count": 3,
        },
    }


def _retention_gate_rows() -> list[dict[str, object]]:
    return [
        {
            "profile": "mode_conditioned_rebound_reward_train_v2",
            "gate_pass": False,
            "retention_decision": "retain_as_evidence_only",
            "ytd_return_pct": 177.573569,
            "required_best_return_pct": 213.143462,
            "required_soxlbacktest_return_pct": 146.043798,
            "worst_wf_max_drawdown_pct": 21.83172,
            "missing_wf_window_count": 0,
            "wf_drawdown_gap_pct": 1.83172,
            "best_return_gap_pct": 35.569893,
            "soxlbacktest_return_gap_pct": 0.0,
            "failures": "wf_max_drawdown_ge_20:wf_2020;wf_max_drawdown_ge_20:wf_2023_2024;ytd_return_below_best_alignment",
            "source_path": "runtime/model/card/mode_conditioned_rebound_reward_train_v2/mode_conditioned_rebound_reward_train_v2_soxl_daily.csv",
        },
        {
            "profile": "smoke",
            "gate_pass": False,
            "retention_decision": "retain_as_evidence_only",
            "ytd_return_pct": 224.361539,
            "required_best_return_pct": 213.143462,
            "required_soxlbacktest_return_pct": 146.043798,
            "worst_wf_max_drawdown_pct": None,
            "missing_wf_window_count": 3,
            "wf_drawdown_gap_pct": None,
            "best_return_gap_pct": 0.0,
            "soxlbacktest_return_gap_pct": 0.0,
            "failures": "missing_wf_window:wf_2020;missing_wf_window:wf_2022;missing_wf_window:wf_2023_2024",
            "source_path": "runtime/model/card/smoke/smoke_soxl_daily.csv",
        },
        {
            "profile": "soxl_backtest_policy_reimplementation_v1",
            "gate_pass": True,
            "retention_decision": "keep_reference_baseline",
            "ytd_return_pct": 146.043798,
            "required_soxlbacktest_return_pct": 146.043798,
            "missing_wf_window_count": 0,
            "failures": "",
            "source_path": "runtime/production/card/soxl_backtest_policy_reimplementation_v1/legacy_comparison_metrics.csv",
        },
    ]


def test_queue_prioritizes_complete_walkforward_candidate_before_return_only_actor():
    result = derive_strict_retraining_candidate_queue(
        strict_gate_summary=_strict_gate_summary(),
        retention_gate_rows=_retention_gate_rows(),
        final_contract_summary={
            "decision": "final_walkforward_retention_contract_no_retained_models",
            "final_retained_count": 0,
            "strict_passing_profile_count": 0,
            "candidate_filter_actor_retention_allowed_count": 0,
            "reference_is_current": True,
            "latest_reference_return_pct": 146.04379795135492,
        },
        candidate_filter_summary={
            "decision": "strict_candidate_retention_filter_no_promotable_candidates",
            "actor_retention_allowed_count": 0,
            "next_mechanism": "train_bull_trend_participation_with_drawdown_drift_stop_no_promotion",
            "ppo_allowed_next": False,
        },
        soxlbacktest_summary={
            "decision": "soxlbacktest_reference_qlib_aligned_not_promotable",
            "selected_buy_sell_reference": "legacy_default",
            "selected_risk_floor": "wf_pass_ytd18_15_15_02",
            "promotion_allowed": False,
            "actor_retention_allowed": False,
        },
        retraining_handoff_summary={
            "decision": "short_foldlocal_drawdown_drift_training_screen_only_no_promotion",
            "next_action": "short_foldlocal_bc_or_ppo_training_screen_with_drawdown_drift_reward_and_official_wf_gate",
            "promotion_allowed": False,
            "actor_retention_allowed": False,
        },
    )

    summary = result["summary"]
    queue = result["candidate_queue"]

    assert summary["decision"] == "strict_retraining_candidate_queue_ready_no_promotion"
    assert summary["final_retained_count"] == 0
    assert summary["promotion_allowed_count"] == 0
    assert summary["top_candidate_id"] == "mode_conditioned_rebound_reward_train_v2"
    assert queue[0]["candidate_id"] == "mode_conditioned_rebound_reward_train_v2"
    assert queue[0]["queue_role"] == "train_drawdown_drift_and_return_alignment"
    assert queue[0]["actor_retention_allowed"] is False
    assert queue[0]["promotion_allowed"] is False
    assert queue[0]["missing_wf_window_count"] == 0
    assert queue[0]["wf_drawdown_gap_pct"] == pytest.approx(1.83172)
    assert queue[0]["return_gap_pct"] == pytest.approx(35.569893)
    assert "official_walkforward_gate" in queue[0]["required_evidence"]

    smoke = next(row for row in queue if row["candidate_id"] == "smoke")
    assert smoke["queue_role"] == "complete_official_walkforward_before_actor_retention"
    assert smoke["missing_wf_window_count"] == 3
    assert smoke["return_gap_pct"] == pytest.approx(0.0)


def test_queue_keeps_soxlbacktest_qlib_as_reference_baseline_only():
    result = derive_strict_retraining_candidate_queue(
        strict_gate_summary=_strict_gate_summary(),
        retention_gate_rows=_retention_gate_rows(),
        soxlbacktest_summary={
            "decision": "soxlbacktest_reference_qlib_aligned_not_promotable",
            "selected_buy_sell_reference": "legacy_default",
            "selected_risk_floor": "wf_pass_ytd18_15_15_02",
            "promotion_allowed": False,
            "actor_retention_allowed": False,
        },
    )

    reference = next(
        row
        for row in result["candidate_queue"]
        if row["candidate_id"] == "soxl_backtest_policy_reimplementation_v1"
    )
    assert reference["candidate_type"] == "reference_baseline"
    assert reference["queue_role"] == "buy_sell_reference_baseline"
    assert reference["runnable_actor"] is False
    assert reference["actor_retention_allowed"] is False
    assert reference["promotion_allowed"] is False
    assert reference["selected_buy_sell_reference"] == "legacy_default"
    assert reference["selected_risk_floor"] == "wf_pass_ytd18_15_15_02"
    assert "compare_buy_sell_features" in reference["required_evidence"]


def test_queue_adds_screen_only_handoff_as_non_promotable_training_entry():
    result = derive_strict_retraining_candidate_queue(
        strict_gate_summary=_strict_gate_summary(),
        retention_gate_rows=_retention_gate_rows(),
        retraining_handoff_summary={
            "decision": "short_foldlocal_drawdown_drift_training_screen_only_no_promotion",
            "next_action": "short_foldlocal_bc_or_ppo_training_screen_with_drawdown_drift_reward_and_official_wf_gate",
            "promotion_allowed": False,
            "actor_retention_allowed": False,
        },
    )

    screen = next(row for row in result["candidate_queue"] if row["candidate_type"] == "training_screen")
    assert screen["candidate_id"] == "short_foldlocal_drawdown_drift_training_screen"
    assert screen["queue_role"] == "screen_only_no_promotion"
    assert screen["training_allowed_next"] is True
    assert screen["promotion_allowed"] is False
    assert screen["actor_retention_allowed"] is False
    assert screen["next_action"] == (
        "short_foldlocal_bc_or_ppo_training_screen_with_drawdown_drift_reward_and_official_wf_gate"
    )


def test_writer_exports_direct_summary_and_ranked_queue_csv(tmp_path):
    result = derive_strict_retraining_candidate_queue(
        strict_gate_summary=_strict_gate_summary(),
        retention_gate_rows=_retention_gate_rows(),
        final_contract_summary={"final_retained_count": 0},
        soxlbacktest_summary={
            "decision": "soxlbacktest_reference_qlib_aligned_not_promotable",
            "selected_buy_sell_reference": "legacy_default",
            "selected_risk_floor": "wf_pass_ytd18_15_15_02",
        },
    )

    sidecars = write_strict_retraining_candidate_queue(result, output_dir=tmp_path)

    summary = json.loads(sidecars["summary_json"].read_text())
    assert summary["decision"] == "strict_retraining_candidate_queue_ready_no_promotion"
    assert summary["top_candidate_id"] == "mode_conditioned_rebound_reward_train_v2"
    assert summary["queue_count"] == len(result["candidate_queue"])
    assert "candidate_queue" not in summary

    with sidecars["candidate_queue_csv"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["queue_rank"] == "1"
    assert rows[0]["candidate_id"] == "mode_conditioned_rebound_reward_train_v2"
