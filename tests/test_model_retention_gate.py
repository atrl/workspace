from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_report(root: Path, profile: str, max_drawdowns: dict[str, float]) -> None:
    payload = {
        window: {
            "profile": profile,
            "window": window,
            "main_backtest": {
                "return": 50.0,
                "sharpe": 1.2,
                "max_dd": max_dd,
                "trades": 10,
            },
            "buy_and_hold": {"return": 10.0},
            "relative_outperformance": 40.0,
        }
        for window, max_dd in max_drawdowns.items()
    }
    _write_json(root / "runtime" / "model" / "card" / profile / f"{profile}_report.json", payload)


def test_retention_gate_requires_complete_walkforward_drawdown_and_return_alignment(tmp_path: Path):
    from src.soxl_quant.app.model_retention_gate import scan_model_retention_gates

    _write_json(
        tmp_path / "dashboard_data" / "model_replay_freshness.json",
        {
            "safe_backtest_end": "2026-06-11",
            "profiles": [
                {"profile": "passer", "status": "candidate", "output": "runtime/model/card/passer/passer_soxl_daily.csv", "result": "current"},
                {"profile": "high_dd", "status": "candidate", "output": "runtime/model/card/high_dd/high_dd_soxl_daily.csv", "result": "current"},
                {"profile": "missing_wf", "status": "candidate", "output": "runtime/model/card/missing_wf/missing_wf_soxl_daily.csv", "result": "current"},
                {"profile": "low_return", "status": "candidate", "output": "runtime/model/card/low_return/low_return_soxl_daily.csv", "result": "current"},
            ],
        },
    )
    _write_csv(
        tmp_path / "runtime" / "production" / "card" / "soxl_backtest_policy_reimplementation_v1" / "legacy_comparison_metrics.csv",
        [
            {
                "window": "ytd_2026",
                "new_return_pct": 100.0,
                "new_max_drawdown_pct": 12.0,
            }
        ],
    )
    comparison_rows = [
        {"window": "ytd_2026_common", "profile": "passer", "status": "candidate", "model_return_pct": 100.0, "current_model_max_drawdown_pct": 18.0},
        {"window": "ytd_2026_common", "profile": "high_dd", "status": "candidate", "model_return_pct": 100.0, "current_model_max_drawdown_pct": 18.0},
        {"window": "ytd_2026_common", "profile": "missing_wf", "status": "candidate", "model_return_pct": 100.0, "current_model_max_drawdown_pct": 18.0},
        {"window": "ytd_2026_common", "profile": "low_return", "status": "candidate", "model_return_pct": 94.0, "current_model_max_drawdown_pct": 18.0},
    ]
    _write_csv(
        tmp_path / "runtime" / "production" / "card" / "current_model_ytd_june_comparison_v1" / "all_model_comparison_metrics.csv",
        comparison_rows,
    )
    _write_report(tmp_path, "passer", {"wf_2020": 10.0, "wf_2022": 19.9, "wf_2023_2024": 18.0})
    _write_report(tmp_path, "high_dd", {"wf_2020": 10.0, "wf_2022": 20.0, "wf_2023_2024": 18.0})
    _write_report(tmp_path, "missing_wf", {"wf_2020": 10.0, "wf_2022": 19.0})
    _write_report(tmp_path, "low_return", {"wf_2020": 10.0, "wf_2022": 19.0, "wf_2023_2024": 18.0})

    result = scan_model_retention_gates(root=tmp_path, max_walkforward_drawdown_pct=20.0, best_return_ratio=0.95)
    by_profile = {row["profile"]: row for row in result["rows"]}

    assert by_profile["soxl_backtest_policy_reimplementation_v1"]["retention_decision"] == "keep_reference_baseline"
    assert by_profile["passer"]["gate_pass"] is True
    assert by_profile["passer"]["retention_decision"] == "retain_promotable"
    assert by_profile["high_dd"]["gate_pass"] is False
    assert "wf_max_drawdown_ge_20:wf_2022" in by_profile["high_dd"]["failures"]
    assert by_profile["missing_wf"]["gate_pass"] is False
    assert "missing_wf_window:wf_2023_2024" in by_profile["missing_wf"]["failures"]
    assert by_profile["low_return"]["gate_pass"] is False
    assert "ytd_return_below_best_alignment" in by_profile["low_return"]["failures"]
    assert result["summary"]["passing_profiles"] == ["passer"]


def test_registry_keep_actor_audit_blocks_non_passing_retained_models(tmp_path: Path):
    from src.soxl_quant.app.model_retention_gate import scan_model_retention_gates

    _write_json(
        tmp_path / "dashboard_data" / "model_replay_freshness.json",
        {
            "safe_backtest_end": "2026-06-11",
            "profiles": [
                {
                    "profile": "passer",
                    "status": "candidate",
                    "output": "runtime/model/card/passer/passer_soxl_daily.csv",
                    "result": "current",
                },
                {
                    "profile": "high_dd",
                    "status": "candidate",
                    "output": "runtime/model/card/high_dd/high_dd_soxl_daily.csv",
                    "result": "current",
                },
            ],
        },
    )
    _write_json(
        tmp_path / "runtime" / "harness" / "card" / "registry" / "profile.yaml",
        {
            "profiles": [
                {"name": "passer", "status": "candidate", "keep_actor": True},
                {"name": "high_dd", "status": "candidate", "keep_actor": True},
                {"name": "missing_replay", "status": "candidate", "keep_actor": True},
                {"name": "evidence_only", "status": "candidate", "keep_actor": False},
            ]
        },
    )
    _write_csv(
        tmp_path
        / "runtime"
        / "production"
        / "card"
        / "soxl_backtest_policy_reimplementation_v1"
        / "legacy_comparison_metrics.csv",
        [{"window": "ytd_2026", "new_return_pct": 100.0, "new_max_drawdown_pct": 12.0}],
    )
    _write_csv(
        tmp_path
        / "runtime"
        / "production"
        / "card"
        / "current_model_ytd_june_comparison_v1"
        / "all_model_comparison_metrics.csv",
        [
            {
                "window": "ytd_2026_common",
                "profile": "passer",
                "status": "candidate",
                "model_return_pct": 100.0,
                "current_model_max_drawdown_pct": 18.0,
            },
            {
                "window": "ytd_2026_common",
                "profile": "high_dd",
                "status": "candidate",
                "model_return_pct": 100.0,
                "current_model_max_drawdown_pct": 18.0,
            },
        ],
    )
    _write_report(tmp_path, "passer", {"wf_2020": 10.0, "wf_2022": 19.9, "wf_2023_2024": 18.0})
    _write_report(tmp_path, "high_dd", {"wf_2020": 10.0, "wf_2022": 20.0, "wf_2023_2024": 18.0})

    result = scan_model_retention_gates(root=tmp_path, max_walkforward_drawdown_pct=20.0, best_return_ratio=0.95)
    by_profile = {row["profile"]: row for row in result["registry_keep_actor_audit"]}

    assert by_profile["passer"]["strict_retention_allowed"] is True
    assert by_profile["passer"]["strict_retention_decision"] == "allow_promotable_retention"
    assert by_profile["high_dd"]["strict_retention_allowed"] is False
    assert by_profile["high_dd"]["strict_retention_decision"] == "block_current_gate_failed"
    assert "wf_max_drawdown_ge_20:wf_2022" in by_profile["high_dd"]["failures"]
    assert by_profile["missing_replay"]["strict_retention_allowed"] is False
    assert by_profile["missing_replay"]["strict_retention_decision"] == "block_missing_replay_evidence"
    assert by_profile["missing_replay"]["failures"] == ["missing_replay_evidence"]
    assert "evidence_only" not in by_profile
    assert result["summary"]["registry_keep_actor_count"] == 3
    assert result["summary"]["registry_keep_actor_strict_violation_count"] == 2


def test_write_retention_gate_outputs_includes_registry_keep_actor_audit(tmp_path: Path):
    from src.soxl_quant.app.model_retention_gate import write_retention_gate_outputs

    output = tmp_path / "card"
    write_retention_gate_outputs(
        {
            "summary": {
                "reference_profile": "soxl_backtest_policy_reimplementation_v1",
                "reference_return_pct": 132.0,
                "reference_max_drawdown_pct": 12.0,
                "current_best_return_pct": 224.0,
                "best_return_ratio": 0.95,
                "required_best_return_pct": 212.8,
                "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
                "max_walkforward_drawdown_pct": 20.0,
                "passing_profiles": [],
                "registry_keep_actor_count": 1,
                "registry_keep_actor_strict_violation_count": 1,
            },
            "rows": [],
            "registry_keep_actor_audit": [
                {
                    "profile": "high_dd",
                    "registry_status": "candidate",
                    "keep_actor": True,
                    "strict_retention_allowed": False,
                    "strict_retention_decision": "block_current_gate_failed",
                    "failures": ["wf_max_drawdown_ge_20:wf_2022"],
                }
            ],
        },
        output_dir=output,
    )

    audit = pd.read_csv(output / "registry_keep_actor_audit.csv")
    assert audit.iloc[0]["profile"] == "high_dd"
    assert audit.iloc[0]["strict_retention_decision"] == "block_current_gate_failed"
    assert "keep_actor strict violations: `1`" in (output / "conclusion.md").read_text(encoding="utf-8")
