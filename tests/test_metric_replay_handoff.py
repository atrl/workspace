from __future__ import annotations

import json

from src.soxl_quant.app.metric_replay_handoff import build_metric_replay_handoff_audit


def _source_summary() -> dict[str, object]:
    return {
        "experiment": "cap10_wf2020_bull_participation_diagnostic_v1",
        "decision": "checkpoint_selector_mixed_metric_source_no_promotion",
        "mixed_metric_source_detected": True,
        "next_mechanism": (
            "Rerun checkpoint selection with backtest_profile bnh_return and "
            "relative_outperformance from the same qlib replay source before any new training."
        ),
        "reference_baseline": "soxlbacktest_qlib_buysell_reference",
        "target_max_dd_pct": 20.0,
        "target_relative_return_pp": 30.0,
    }


def _replay_summary() -> dict[str, object]:
    return {
        "experiment": "cap10_three_fold_checkpoint_metric_replay_v2",
        "decision": "three_fold_checkpoint_metric_replay_partial_pass_no_promotion",
        "next_mechanism": "inspect_failed_fold_drawdown_or_reward_drift_before_full_retention_training",
        "promotion_allowed": False,
        "required_fold_coverage": True,
        "reference_baseline": "soxlbacktest_qlib_buysell_reference",
        "reference_thresholds": {
            "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            "max_walkforward_drawdown_pct": 20.0,
        },
        "selected_by_fold": {
            "wf_2020": {
                "selected_candidate": "bc_initialization",
                "selected_candidate_gate": False,
                "selected_max_dd": 14.9847,
                "selected_relative_return": -292.753,
            },
            "wf_2022": {
                "selected_candidate": "bc_initialization",
                "selected_candidate_gate": True,
                "selected_max_dd": 18.8023,
                "selected_relative_return": 65.8648,
            },
            "wf_2023_2024": {
                "selected_candidate": "ppo_retained_iter_4",
                "selected_candidate_gate": True,
                "selected_max_dd": 8.9558,
                "selected_relative_return": 35.5662,
            },
        },
        "target_max_dd_pct": 20.0,
        "target_relative_return_pp": 30.0,
    }


def test_metric_replay_handoff_completes_source_diagnostic_and_preserves_fold_metrics():
    notebook_text = """
    bt = backtest_profile(actor, eval_df, price_series, test_seg=test_seg)
    for src, dst in [
        ('bnh_return', 'buy_and_hold_return'),
        ('relative_outperformance', 'relative_return'),
    ]:
        out[dst] = bt[src]
    """

    audit = build_metric_replay_handoff_audit(
        source_summary=_source_summary(),
        replay_summary=_replay_summary(),
        replay_notebook_text=notebook_text,
    )

    assert audit.summary["decision"] == "metric_replay_handoff_completed_no_promotion"
    assert audit.summary["source_experiment"] == "cap10_wf2020_bull_participation_diagnostic_v1"
    assert audit.summary["direct_comparison_baseline"] == "cap10_three_fold_checkpoint_metric_replay_v2"
    assert audit.summary["metric_replay_source_fields_present"] is True
    assert audit.summary["metric_replay_handoff_completed"] is True
    assert audit.summary["missing_folds"] == []
    assert audit.summary["failed_folds"] == ["wf_2020"]
    assert audit.summary["selected_by_fold"]["wf_2020"]["selected_relative_return"] == -292.753
    assert audit.summary["next_mechanism"] == (
        "inspect_failed_fold_drawdown_or_reward_drift_before_full_retention_training"
    )
    assert bool(audit.fold_metrics.set_index("fold").loc["wf_2022", "selected_candidate_gate"]) is True
    json.dumps(audit.summary)


def test_metric_replay_handoff_blocks_if_replay_notebook_does_not_use_official_relative_fields():
    audit = build_metric_replay_handoff_audit(
        source_summary=_source_summary(),
        replay_summary=_replay_summary(),
        replay_notebook_text="bt = backtest_profile(actor, eval_df, price_series)",
    )

    assert audit.summary["decision"] == "metric_replay_handoff_incomplete_no_promotion"
    assert audit.summary["metric_replay_source_fields_present"] is False
    assert audit.summary["metric_replay_handoff_completed"] is False
    assert audit.summary["next_mechanism"].startswith("Rerun checkpoint selection")
