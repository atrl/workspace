from __future__ import annotations

from src.soxl_quant.app.final_walkforward_retention_contract import (
    derive_final_walkforward_retention_contract,
)


def _strict_gate_payload() -> dict[str, object]:
    return {
        "summary": {
            "passing_profile_count": 1,
            "passing_profiles": ["full_gate_model"],
            "profiles_scanned": 2,
            "reference_profile": "soxl_backtest_policy_reimplementation_v1",
            "reference_return_pct": 146.0,
            "reference_max_drawdown_pct": 13.5,
            "current_best_return_pct": 220.0,
            "required_best_return_pct": 209.0,
            "max_walkforward_drawdown_pct": 20.0,
            "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
        },
        "rows": [
            {
                "profile": "soxl_backtest_policy_reimplementation_v1",
                "status": "reference",
                "source_path": "runtime/production/card/soxl_backtest_policy_reimplementation_v1/legacy_comparison_metrics.csv",
                "ytd_return_pct": 146.0,
                "ytd_max_drawdown_pct": 13.5,
                "required_soxlbacktest_return_pct": 146.0,
                "wf_windows_present": "",
                "worst_wf_max_drawdown_pct": None,
                "missing_wf_window_count": 0,
                "gate_pass": True,
                "retention_decision": "keep_reference_baseline",
                "failures": [],
            },
            {
                "profile": "full_gate_model",
                "status": "candidate",
                "source_path": "runtime/model/card/full_gate_model/full_gate_model_soxl_daily.csv",
                "ytd_return_pct": 220.0,
                "ytd_max_drawdown_pct": 12.0,
                "current_best_return_pct": 220.0,
                "required_best_return_pct": 209.0,
                "required_soxlbacktest_return_pct": 146.0,
                "wf_windows_present": "wf_2020,wf_2022,wf_2023_2024",
                "worst_wf_max_drawdown_pct": 18.0,
                "missing_wf_window_count": 0,
                "wf_drawdown_gap_pct": 0.0,
                "best_return_gap_pct": 0.0,
                "soxlbacktest_return_gap_pct": 0.0,
                "gate_pass": True,
                "retention_decision": "retain_promotable",
                "failures": [],
            },
        ],
    }


def _candidate_filter_payload(*, actor_retention_allowed: bool = True) -> dict[str, object]:
    return {
        "summary": {
            "decision": "strict_candidate_retention_filter_promotion_review_required"
            if actor_retention_allowed
            else "strict_candidate_retention_filter_no_promotable_candidates",
            "promotable_profile_count": 1 if actor_retention_allowed else 0,
            "actor_retention_allowed_count": 1 if actor_retention_allowed else 0,
            "reference_baseline": "soxl_backtest_policy_reimplementation_v1",
        },
        "rows": [
            {
                "candidate_id": "full_gate_model",
                "candidate_type": "profile",
                "promotion_allowed": actor_retention_allowed,
                "actor_retention_allowed": actor_retention_allowed,
                "primary_blocker": "" if actor_retention_allowed else "strict_walkforward_gate_failed",
            }
        ],
    }


def test_final_walkforward_contract_retains_only_strict_candidate_actor_allowed_profile():
    result = derive_final_walkforward_retention_contract(
        strict_gate_payload=_strict_gate_payload(),
        strict_candidate_filter_payload=_candidate_filter_payload(actor_retention_allowed=True),
        latest_reference_baseline={"return_pct": 146.0, "max_drawdown_pct": 13.5},
    )

    assert result["summary"]["decision"] == "final_walkforward_retention_contract_ready"
    assert result["summary"]["final_retained_count"] == 1
    by_profile = {row["profile"]: row for row in result["rows"]}
    assert by_profile["full_gate_model"]["final_model_retained"] is True
    assert by_profile["soxl_backtest_policy_reimplementation_v1"]["final_model_retained"] is False
    assert by_profile["soxl_backtest_policy_reimplementation_v1"]["blockers"] == [
        "SOXLBacktest reference baseline cannot be retained as actor"
    ]


def test_final_walkforward_contract_blocks_when_candidate_filter_does_not_allow_actor():
    result = derive_final_walkforward_retention_contract(
        strict_gate_payload=_strict_gate_payload(),
        strict_candidate_filter_payload=_candidate_filter_payload(actor_retention_allowed=False),
        latest_reference_baseline={"return_pct": 146.0, "max_drawdown_pct": 13.5},
    )

    row = {item["profile"]: item for item in result["rows"]}["full_gate_model"]
    assert result["summary"]["final_retained_count"] == 0
    assert row["final_model_retained"] is False
    assert row["blockers"] == ["strict_candidate_filter_actor_retention_false"]


def test_final_walkforward_contract_blocks_stale_reference_return():
    result = derive_final_walkforward_retention_contract(
        strict_gate_payload=_strict_gate_payload(),
        strict_candidate_filter_payload=_candidate_filter_payload(actor_retention_allowed=True),
        latest_reference_baseline={"return_pct": 147.0, "max_drawdown_pct": 13.5},
    )

    assert result["summary"]["final_retained_count"] == 0
    assert result["summary"]["global_blockers"] == ["stale_reference_return_pct"]
    assert result["rows"][1]["blockers"] == ["stale_reference_return_pct"]
