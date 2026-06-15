from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd
import yaml


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


def _write_report_with_relative_returns(
    root: Path,
    profile: str,
    metrics: dict[str, tuple[float, float]],
) -> None:
    payload = {
        window: {
            "profile": profile,
            "window": window,
            "main_backtest": {
                "return": relative_return + 10.0,
                "sharpe": 1.2,
                "max_dd": max_dd,
                "trades": 10,
            },
            "buy_and_hold": {"return": 10.0},
            "relative_outperformance": relative_return,
        }
        for window, (max_dd, relative_return) in metrics.items()
    }
    _write_json(root / "runtime" / "model" / "card" / profile / f"{profile}_report.json", payload)


def test_retention_gate_requires_complete_walkforward_drawdown_and_return_alignment():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_gate_requires_complete_walkforward_drawdown_and_return_alignment(tmp_path)


def _run_retention_gate_requires_complete_walkforward_drawdown_and_return_alignment(tmp_path: Path) -> None:
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
    _write_report(tmp_path, "high_dd", {"wf_2020": 10.0, "wf_2022": 22.5, "wf_2023_2024": 18.0})
    _write_report(tmp_path, "missing_wf", {"wf_2020": 10.0, "wf_2022": 19.0})
    _write_report(tmp_path, "low_return", {"wf_2020": 10.0, "wf_2022": 19.0, "wf_2023_2024": 18.0})

    result = scan_model_retention_gates(root=tmp_path, max_walkforward_drawdown_pct=20.0, best_return_ratio=0.95)
    by_profile = {row["profile"]: row for row in result["rows"]}

    assert by_profile["soxl_backtest_policy_reimplementation_v1"]["retention_decision"] == "keep_reference_baseline"
    assert by_profile["passer"]["gate_pass"] is True
    assert by_profile["passer"]["retention_decision"] == "retain_promotable"
    assert by_profile["high_dd"]["gate_pass"] is False
    assert "wf_max_drawdown_ge_20:wf_2022" in by_profile["high_dd"]["failures"]
    assert by_profile["high_dd"]["wf_drawdown_gap_pct"] == 2.5
    assert by_profile["missing_wf"]["missing_wf_window_count"] == 1
    assert by_profile["missing_wf"]["gate_pass"] is False
    assert "missing_wf_window:wf_2023_2024" in by_profile["missing_wf"]["failures"]
    assert by_profile["low_return"]["gate_pass"] is False
    assert "ytd_return_below_best_alignment" in by_profile["low_return"]["failures"]
    assert by_profile["low_return"]["best_return_gap_pct"] == 1.0
    assert by_profile["low_return"]["soxlbacktest_return_gap_pct"] == 6.0
    assert result["summary"]["passing_profiles"] == ["passer"]
    assert result["summary"]["closest_wf_drawdown_candidate"]["profile"] == "passer"
    assert result["summary"]["closest_ytd_return_candidate"]["profile"] == "passer"


def test_retention_gate_requires_walkforward_relative_return_alignment():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_gate_requires_walkforward_relative_return_alignment(tmp_path)


def _run_retention_gate_requires_walkforward_relative_return_alignment(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_gate import scan_model_retention_gates

    _write_json(
        tmp_path / "dashboard_data" / "model_replay_freshness.json",
        {
            "safe_backtest_end": "2026-06-11",
            "profiles": [
                {
                    "profile": "low_wf_return",
                    "status": "candidate",
                    "output": "runtime/model/card/low_wf_return/low_wf_return_soxl_daily.csv",
                    "result": "current",
                },
            ],
        },
    )
    _write_csv(
        tmp_path / "runtime" / "production" / "card" / "soxl_backtest_policy_reimplementation_v1" / "legacy_comparison_metrics.csv",
        [{"window": "ytd_2026", "new_return_pct": 100.0, "new_max_drawdown_pct": 12.0}],
    )
    _write_csv(
        tmp_path / "runtime" / "production" / "card" / "current_model_ytd_june_comparison_v1" / "all_model_comparison_metrics.csv",
        [
            {
                "window": "ytd_2026_common",
                "profile": "low_wf_return",
                "status": "candidate",
                "model_return_pct": 130.0,
                "current_model_max_drawdown_pct": 18.0,
            },
        ],
    )
    _write_report_with_relative_returns(
        tmp_path,
        "low_wf_return",
        {
            "wf_2020": (18.0, 5.0),
            "wf_2022": (12.0, 45.0),
            "wf_2023_2024": (14.0, 35.0),
        },
    )

    result = scan_model_retention_gates(
        root=tmp_path,
        max_walkforward_drawdown_pct=20.0,
        best_return_ratio=0.95,
    )
    row = {item["profile"]: item for item in result["rows"]}["low_wf_return"]

    assert row["gate_pass"] is False
    assert "wf_relative_return_below_30:wf_2020" in row["failures"]
    assert row["min_wf_relative_return_pct"] == 5.0
    assert row["wf_relative_return_gap_pct"] == 25.0
    assert result["summary"]["min_walkforward_relative_return_pct"] == 30.0


def test_retention_gate_scans_comparison_models_when_replay_freshness_has_no_profiles():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_retention_gate_scans_comparison_models_when_replay_freshness_has_no_profiles(tmp_path)


def _run_retention_gate_scans_comparison_models_when_replay_freshness_has_no_profiles(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_gate import scan_model_retention_gates

    _write_json(
        tmp_path / "dashboard_data" / "model_replay_freshness.json",
        {"safe_backtest_end": "2026-06-11", "profiles": []},
    )
    _write_csv(
        tmp_path / "runtime" / "production" / "card" / "soxl_backtest_policy_reimplementation_v1" / "legacy_comparison_metrics.csv",
        [{"window": "ytd_2026", "new_return_pct": 100.0, "new_max_drawdown_pct": 12.0}],
    )
    _write_csv(
        tmp_path / "runtime" / "production" / "card" / "current_model_ytd_june_comparison_v1" / "all_model_comparison_metrics.csv",
        [
            {
                "window": "ytd_2026_common",
                "profile": "drawdown_safe",
                "status": "ablation",
                "source_path": "runtime/model/card/drawdown_safe/drawdown_safe_soxl_daily.csv",
                "current_model_return_pct": 99.0,
                "current_model_max_drawdown_pct": 18.0,
            },
            {
                "window": "ytd_2026_common",
                "profile": "return_reference",
                "status": "canonical",
                "source_path": "runtime/model/card/return_reference/return_reference_soxl_daily.csv",
                "current_model_return_pct": 130.0,
                "current_model_max_drawdown_pct": 30.0,
            },
        ],
    )
    _write_report(tmp_path, "drawdown_safe", {"wf_2020": 10.0, "wf_2022": 19.5, "wf_2023_2024": 18.0})

    result = scan_model_retention_gates(root=tmp_path, max_walkforward_drawdown_pct=20.0, best_return_ratio=0.95)
    by_profile = {row["profile"]: row for row in result["rows"]}

    assert result["summary"]["profiles_scanned"] == 2
    assert by_profile["drawdown_safe"]["source_path"] == "runtime/model/card/drawdown_safe/drawdown_safe_soxl_daily.csv"
    assert by_profile["drawdown_safe"]["status"] == "ablation"
    assert by_profile["drawdown_safe"]["gate_pass"] is False
    assert "ytd_return_below_best_alignment" in by_profile["drawdown_safe"]["failures"]
    assert by_profile["return_reference"]["status"] == "canonical"
    assert by_profile["return_reference"]["missing_wf_window_count"] == 3
    assert result["summary"]["closest_wf_drawdown_candidate"]["profile"] == "drawdown_safe"
    assert result["summary"]["closest_ytd_return_candidate"]["profile"] == "return_reference"


def test_registry_keep_actor_audit_blocks_non_passing_retained_models():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_registry_keep_actor_audit_blocks_non_passing_retained_models(tmp_path)


def _run_registry_keep_actor_audit_blocks_non_passing_retained_models(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_gate import scan_model_retention_gates

    _write_json(
        tmp_path / "dashboard_data" / "model_replay_freshness.json",
        {
            "safe_backtest_end": "2026-06-11",
            "profiles": [
                {"profile": "passer", "status": "candidate", "output": "runtime/model/card/passer/passer_soxl_daily.csv", "result": "current"},
                {"profile": "high_dd", "status": "candidate", "output": "runtime/model/card/high_dd/high_dd_soxl_daily.csv", "result": "current"},
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
        tmp_path / "runtime" / "production" / "card" / "soxl_backtest_policy_reimplementation_v1" / "legacy_comparison_metrics.csv",
        [{"window": "ytd_2026", "new_return_pct": 100.0, "new_max_drawdown_pct": 12.0}],
    )
    _write_csv(
        tmp_path / "runtime" / "production" / "card" / "current_model_ytd_june_comparison_v1" / "all_model_comparison_metrics.csv",
        [
            {"window": "ytd_2026_common", "profile": "passer", "status": "candidate", "model_return_pct": 100.0, "current_model_max_drawdown_pct": 18.0},
            {"window": "ytd_2026_common", "profile": "high_dd", "status": "candidate", "model_return_pct": 100.0, "current_model_max_drawdown_pct": 18.0},
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
    plan_by_profile = {row["profile"]: row for row in result["registry_retention_plan"]}
    assert plan_by_profile["passer"]["action"] == "keep_actor_allowed"
    assert plan_by_profile["high_dd"]["action"] == "disable_keep_actor"
    assert plan_by_profile["missing_replay"]["action"] == "disable_keep_actor"
    assert result["summary"]["registry_keep_actor_disable_count"] == 2
    assert result["summary"]["registry_keep_actor_protected_exception_count"] == 0


def test_write_retention_gate_outputs_includes_registry_keep_actor_audit():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_write_retention_gate_outputs_includes_registry_keep_actor_audit(tmp_path)


def _run_write_retention_gate_outputs_includes_registry_keep_actor_audit(tmp_path: Path) -> None:
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
                "registry_keep_actor_disable_count": 1,
                "registry_keep_actor_protected_exception_count": 0,
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
            "registry_retention_plan": [
                {
                    "profile": "high_dd",
                    "status": "candidate",
                    "action": "disable_keep_actor",
                    "failures": ["wf_max_drawdown_ge_20:wf_2022"],
                }
            ],
        },
        output_dir=output,
    )

    audit = pd.read_csv(output / "registry_keep_actor_audit.csv")
    plan = pd.read_csv(output / "registry_retention_plan.csv")
    assert audit.iloc[0]["profile"] == "high_dd"
    assert audit.iloc[0]["strict_retention_decision"] == "block_current_gate_failed"
    assert plan.iloc[0]["action"] == "disable_keep_actor"
    assert "keep_actor strict violations: `1`" in (output / "conclusion.md").read_text(encoding="utf-8")
    assert "Registry actor disable plan: `1`" in (output / "conclusion.md").read_text(encoding="utf-8")


def test_plan_registry_retention_updates_disables_all_failed_keep_actor_entries():
    from src.soxl_quant.app.model_retention_gate import plan_registry_retention_updates

    registry_payload = {
        "profiles": [
            {"name": "passing_candidate", "status": "candidate", "keep_actor": True},
            {"name": "failed_candidate", "status": "candidate", "keep_actor": True},
            {"name": "failed_ablation", "status": "ablation", "keep_actor": True},
            {"name": "canonical_runtime", "status": "canonical", "keep_actor": True},
            {"name": "archived_rollback", "status": "archived", "keep_actor": True},
            {"name": "summary_only", "status": "candidate", "keep_actor": False},
        ]
    }
    gate_result = {
        "registry_keep_actor_audit": [
            {
                "profile": "passing_candidate",
                "strict_retention_allowed": True,
                "strict_retention_decision": "allow_promotable_retention",
                "failures": [],
            },
            {
                "profile": "failed_candidate",
                "strict_retention_allowed": False,
                "strict_retention_decision": "block_current_gate_failed",
                "failures": ["wf_max_drawdown_ge_20:wf_2022"],
            },
            {
                "profile": "failed_ablation",
                "strict_retention_allowed": False,
                "strict_retention_decision": "block_current_gate_failed",
                "failures": ["ytd_return_below_best_alignment"],
            },
            {
                "profile": "canonical_runtime",
                "strict_retention_allowed": False,
                "strict_retention_decision": "block_current_gate_failed",
                "failures": ["wf_max_drawdown_ge_20:wf_2022"],
            },
            {
                "profile": "archived_rollback",
                "strict_retention_allowed": False,
                "strict_retention_decision": "block_current_gate_failed",
                "failures": ["missing_wf_window:wf_2023_2024"],
            },
        ]
    }

    plan = plan_registry_retention_updates(registry_payload, gate_result)
    by_profile = {row["profile"]: row for row in plan}

    assert by_profile["passing_candidate"]["action"] == "keep_actor_allowed"
    assert by_profile["failed_candidate"]["action"] == "disable_keep_actor"
    assert by_profile["failed_ablation"]["action"] == "disable_keep_actor"
    assert by_profile["canonical_runtime"]["action"] == "disable_keep_actor"
    assert by_profile["archived_rollback"]["action"] == "disable_keep_actor"
    assert "summary_only" not in by_profile


def test_apply_registry_retention_updates_marks_failed_candidates_summary_only():
    from src.soxl_quant.app.model_retention_gate import apply_registry_retention_updates

    registry_payload = {
        "profiles": [
            {"name": "failed_candidate", "status": "candidate", "keep_actor": True, "keep_summary": True},
            {"name": "canonical_runtime", "status": "canonical", "keep_actor": True, "keep_summary": True},
        ]
    }
    plan = [
        {
            "profile": "failed_candidate",
            "action": "disable_keep_actor",
            "failures": ["wf_max_drawdown_ge_20:wf_2022"],
        },
        {
            "profile": "canonical_runtime",
            "action": "disable_keep_actor",
            "failures": ["wf_max_drawdown_ge_20:wf_2022"],
        },
    ]

    updated, changed = apply_registry_retention_updates(registry_payload, plan)
    by_name = {row["name"]: row for row in updated["profiles"]}

    assert changed == ["failed_candidate", "canonical_runtime"]
    assert by_name["failed_candidate"]["keep_actor"] is False
    assert by_name["failed_candidate"]["keep_summary"] is True
    assert by_name["failed_candidate"]["wf_status"] == "blocked"
    assert "wf_max_drawdown_ge_20:wf_2022" in by_name["failed_candidate"]["wf_notes"]
    assert by_name["canonical_runtime"]["keep_actor"] is False
    assert by_name["canonical_runtime"]["keep_summary"] is True
    assert by_name["canonical_runtime"]["wf_status"] == "blocked"


def test_write_registry_retention_updates_persists_summary_only_registry_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_write_registry_retention_updates_persists_summary_only_registry_metadata(tmp_path)


def _run_write_registry_retention_updates_persists_summary_only_registry_metadata(tmp_path: Path) -> None:
    from src.soxl_quant.app.model_retention_gate import write_registry_retention_updates

    registry_path = tmp_path / "runtime" / "harness" / "card" / "registry" / "profile.yaml"
    registry_path.parent.mkdir(parents=True)
    actor_path = tmp_path / "runtime" / "model" / "card" / "failed_candidate" / "soxl_failed_candidate_actor.pt"
    actor_path.parent.mkdir(parents=True)
    actor_path.write_bytes(b"actor")
    registry_path.write_text(
        yaml.safe_dump(
            {
                "profiles": [
                    {"name": "failed_candidate", "status": "candidate", "keep_actor": True, "keep_summary": True},
                    {"name": "canonical_runtime", "status": "canonical", "keep_actor": True, "keep_summary": True},
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    gate_result = {
        "registry_retention_plan": [
            {
                "profile": "failed_candidate",
                "action": "disable_keep_actor",
                "failures": ["wf_max_drawdown_ge_20:wf_2022"],
        },
        {
            "profile": "canonical_runtime",
            "action": "disable_keep_actor",
            "failures": ["wf_max_drawdown_ge_20:wf_2022"],
        },
        ]
    }

    changed = write_registry_retention_updates(tmp_path, gate_result)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in payload["profiles"]}

    assert changed == ["failed_candidate", "canonical_runtime"]
    assert actor_path.exists()
    assert by_name["failed_candidate"]["keep_actor"] is False
    assert by_name["failed_candidate"]["keep_summary"] is True
    assert by_name["canonical_runtime"]["keep_actor"] is False
