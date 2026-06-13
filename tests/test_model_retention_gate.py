from __future__ import annotations

import json
from pathlib import Path
import tempfile

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
