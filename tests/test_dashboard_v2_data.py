from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.soxl_quant.app.dashboard_v2 import (
    build_feature_action_summary,
    build_comparison_payload,
    build_dashboard_v2_data,
    derive_model_events,
    normalize_model_rows,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_dashboard_v2_data_writes_manifest_market_model_and_comparison(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    model_csv = tmp_path / "runtime" / "model" / "card" / "smoke" / "smoke_soxl_daily.csv"
    model_csv.parent.mkdir(parents=True)
    _write_csv(
        model_csv,
        [
            {
                "datetime": "2026-01-02",
                "strategy_nav": 100.0,
                "buyhold_nav": 100.0,
                "position": 0.0,
                "action": "HOLD",
                "price": 10.0,
            },
            {
                "datetime": "2026-01-05",
                "strategy_nav": 105.0,
                "buyhold_nav": 110.0,
                "position": 0.7,
                "action": "BUY",
                "price": 11.0,
            },
            {
                "datetime": "2026-01-06",
                "strategy_nav": 103.0,
                "buyhold_nav": 108.0,
                "position": 0.2,
                "action": "SELL",
                "price": 10.8,
            },
        ],
    )
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T00:00:00+00:00",
                "latest_data_date": "2026-01-06",
                "safe_backtest_end": "2026-01-06",
                "profiles": [
                    {
                        "profile": "smoke",
                        "status": "canonical",
                        "actor": "runtime/model/card/smoke/soxl_smoke_actor.pt",
                        "profile_yaml": "runtime/harness/card/smoke/profile.yaml",
                        "output": "runtime/model/card/smoke/smoke_soxl_daily.csv",
                        "result": "current",
                    }
                ],
            }
        )
    )
    market_rows = {
        "2026-01-02": {
            "open": 9.5,
            "high": 10.2,
            "low": 9.4,
            "close": 10.0,
            "volume": 1000,
            "rsi6": 50.0,
            "ma180": None,
            "hv_20": None,
            "vol_coef": 1.0,
            "panic_score": 0.0,
            "market_state": False,
        },
        "2026-01-05": {
            "open": 10.2,
            "high": 11.5,
            "low": 10.1,
            "close": 11.0,
            "volume": 1200,
            "rsi6": 60.0,
            "ma180": None,
            "hv_20": None,
            "vol_coef": 1.1,
            "panic_score": -5.0,
            "market_state": False,
        },
        "2026-01-06": {
            "open": 11.0,
            "high": 11.1,
            "low": 10.5,
            "close": 10.8,
            "volume": 1300,
            "rsi6": 58.0,
            "ma180": None,
            "hv_20": None,
            "vol_coef": 1.1,
            "panic_score": -3.0,
            "market_state": False,
        },
    }
    feature_rows = {"2026-01-05": {"vix": 18.0, "rsi": 60.0, "spread": 0.4}}

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        market_rows=market_rows,
        feature_rows=feature_rows,
        selected_profiles=["smoke"],
        baseline_profile="smoke",
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    market = json.loads((output_dir / "market" / "SOXL.json").read_text())
    model = json.loads((output_dir / "models" / "smoke.json").read_text())
    comparison = json.loads((output_dir / "comparison" / "default.json").read_text())

    assert manifest["safe_backtest_end"] == "2026-01-06"
    assert manifest["default_baseline_profile"] == "smoke"
    assert manifest["auto_update"]["model_replay_freshness"]["profile_count"] == 1
    assert manifest["auto_update"]["model_replay_freshness"]["ready_count"] == 1
    assert market["rows"][1]["vix"] == 18.0
    assert model["rows"][1]["action"] == "BUY"
    assert model["metrics"]["full"]["buyhold_return"] == pytest.approx(0.08)
    assert model["metrics"]["full"]["excess_return"] == pytest.approx(-0.05)
    assert model["events"][0]["kind"] == "BUY"
    assert comparison["common_start"] == "2026-01-02"
    assert comparison["rankings"][0]["buyhold_return"] == pytest.approx(0.08)
    assert comparison["rows"][2]["profiles"]["smoke"]["buyhold_return"] == pytest.approx(0.08)
    assert comparison["rows"][2]["profiles"]["smoke"]["return_delta_vs_buyhold"] == pytest.approx(-0.05)
    assert comparison["rows"][1]["action_mismatch_count"] == 0


def test_build_dashboard_v2_data_keeps_model_specific_feature_snapshots(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    for profile, action, position in [
        ("alpha_model", "BUY", 0.7),
        ("beta_model", "SELL", 0.2),
    ]:
        model_csv = tmp_path / "runtime" / "model" / "card" / profile / f"{profile}_soxl_daily.csv"
        model_csv.parent.mkdir(parents=True)
        (model_csv.parent / f"soxl_{profile.split('_')[0]}_actor.pt").write_bytes(b"actor")
        _write_csv(
            model_csv,
            [
                {
                    "datetime": "2026-01-02",
                    "strategy_nav": 100.0,
                    "buyhold_nav": 100.0,
                    "position": 0.0,
                    "action": "HOLD",
                    "price": 10.0,
                },
                {
                    "datetime": "2026-01-05",
                    "strategy_nav": 105.0,
                    "buyhold_nav": 110.0,
                    "position": position,
                    "action": action,
                    "price": 11.0,
                },
            ],
        )
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T00:00:00+00:00",
                "latest_data_date": "2026-01-05",
                "safe_backtest_end": "2026-01-05",
                "profiles": [
                    {
                        "profile": "alpha_model",
                        "status": "candidate",
                        "actor": "runtime/model/card/alpha_model/soxl_alpha_actor.pt",
                        "output": "runtime/model/card/alpha_model/alpha_model_soxl_daily.csv",
                        "result": "current",
                    },
                    {
                        "profile": "beta_model",
                        "status": "candidate",
                        "actor": "runtime/model/card/beta_model/soxl_beta_actor.pt",
                        "output": "runtime/model/card/beta_model/beta_model_soxl_daily.csv",
                        "result": "current",
                    },
                ],
            }
        )
    )
    market_rows = {
        "2026-01-02": {"open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0, "volume": 1000},
        "2026-01-05": {"open": 10.5, "high": 11.5, "low": 10.0, "close": 11.0, "volume": 1200},
    }

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        market_rows=market_rows,
        selected_profiles=["alpha_model", "beta_model"],
        baseline_profile="alpha_model",
        model_feature_rows_by_profile={
            "alpha_model": {
                "2026-01-05": {"alpha_signal": 1.25, "shared_feature": 10.0},
            },
            "beta_model": {
                "2026-01-05": {"beta_signal": -2.5, "shared_feature": 20.0},
            },
        },
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    alpha = json.loads((output_dir / "models" / "alpha_model.json").read_text())
    beta = json.loads((output_dir / "models" / "beta_model.json").read_text())

    assert manifest["available_profiles"] == ["alpha_model", "beta_model"]
    assert manifest["default_chart_profile"] == "alpha_model"
    assert alpha["runnable"] is True
    assert beta["runnable"] is True
    assert alpha["snapshot_columns"] == ["alpha_signal", "shared_feature"]
    assert beta["snapshot_columns"] == ["beta_signal", "shared_feature"]
    assert alpha["rows"][1]["feature_snapshot"] == {"alpha_signal": 1.25, "shared_feature": 10.0}
    assert beta["rows"][1]["feature_snapshot"] == {"beta_signal": -2.5, "shared_feature": 20.0}
    assert alpha["events"][0]["feature_snapshot"]["alpha_signal"] == 1.25
    assert beta["events"][0]["feature_snapshot"]["beta_signal"] == -2.5
    assert alpha["feature_action_summary"][0]["feature"] == "alpha_signal"
    assert beta["feature_action_summary"][0]["feature"] == "beta_signal"


def test_build_dashboard_v2_data_disables_models_with_missing_actor_artifacts(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    model_csv = tmp_path / "runtime" / "model" / "card" / "missing_actor_model" / "missing_actor_model_soxl_daily.csv"
    model_csv.parent.mkdir(parents=True)
    _write_csv(
        model_csv,
        [
            {
                "datetime": "2026-01-02",
                "strategy_nav": 100.0,
                "buyhold_nav": 100.0,
                "position": 0.0,
                "action": "HOLD",
                "price": 10.0,
            },
            {
                "datetime": "2026-01-05",
                "strategy_nav": 105.0,
                "buyhold_nav": 110.0,
                "position": 0.7,
                "action": "BUY",
                "price": 11.0,
            },
        ],
    )
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T00:00:00+00:00",
                "latest_data_date": "2026-01-05",
                "safe_backtest_end": "2026-01-05",
                "profiles": [
                    {
                        "profile": "missing_actor_model",
                        "status": "candidate",
                        "actor": "runtime/model/card/missing_actor_model/soxl_missing_actor_model_actor.pt",
                        "output": "runtime/model/card/missing_actor_model/missing_actor_model_soxl_daily.csv",
                        "result": "current",
                    }
                ],
            }
        )
    )

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        market_rows={
            "2026-01-02": {"open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0, "volume": 1000},
            "2026-01-05": {"open": 10.5, "high": 11.5, "low": 10.0, "close": 11.0, "volume": 1200},
        },
        selected_profiles=["missing_actor_model"],
        baseline_profile="missing_actor_model",
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    model = json.loads((output_dir / "models" / "missing_actor_model.json").read_text())

    assert model["runnable"] is False
    assert model["runnable_reason"] == "missing_actor_artifact"
    assert manifest["default_chart_profile"] == "missing_actor_model"


def test_build_dashboard_v2_data_includes_retained_dashboard_exports_when_replay_manifest_empty(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T00:00:00+00:00",
                "latest_data_date": "2026-01-06",
                "safe_backtest_end": "2026-01-06",
                "counts": {"current": 0, "refreshed": 0, "failed": 0, "skipped": 0},
                "profiles": [],
            }
        )
    )
    retained_json = tmp_path / "dashboard_data" / "versioned" / "retained_model" / "SOXL.json"
    retained_json.parent.mkdir(parents=True)
    retained_json.write_text(
        json.dumps(
            {
                "summary": {"etf": "SOXL", "strategy_return": 0.03},
                "rows": [
                    {
                        "datetime": "2026-01-02",
                        "strategy_nav": 100.0,
                        "buyhold_nav": 100.0,
                        "position": 0.0,
                        "action": "HOLD",
                        "price": 10.0,
                    },
                    {
                        "datetime": "2026-01-05",
                        "strategy_nav": 105.0,
                        "buyhold_nav": 110.0,
                        "position": 0.7,
                        "action": "BUY",
                        "price": 11.0,
                    },
                    {
                        "datetime": "2026-01-06",
                        "strategy_nav": 103.0,
                        "buyhold_nav": 108.0,
                        "position": 0.2,
                        "action": "SELL",
                        "price": 10.8,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        market_rows={
            "2026-01-02": {"open": 9.5, "high": 10.2, "low": 9.4, "close": 10.0, "volume": 1000},
            "2026-01-05": {"open": 10.2, "high": 11.5, "low": 10.1, "close": 11.0, "volume": 1200},
            "2026-01-06": {"open": 11.0, "high": 11.1, "low": 10.5, "close": 10.8, "volume": 1300},
        },
        model_feature_rows_by_profile={
            "retained_model": {
                "2026-01-05": {"retained_buy_signal": 1.25, "shared_feature": 10.0},
                "2026-01-06": {"retained_sell_signal": -0.5, "shared_feature": 11.0},
            }
        },
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    model = json.loads((output_dir / "models" / "retained_model.json").read_text())
    comparison = json.loads((output_dir / "comparison" / "default.json").read_text())

    assert manifest["available_profiles"] == ["retained_model"]
    assert manifest["default_chart_profile"] == "retained_model"
    assert model["source_kind"] == "retained_dashboard_export"
    assert model["runnable"] is False
    assert model["runnable_reason"] == "evidence_only_retained_dashboard_export"
    assert model["source_path"] == "dashboard_data/versioned/retained_model/SOXL.json"
    assert model["rows"][1]["feature_snapshot"]["retained_buy_signal"] == 1.25
    assert model["events"][0]["feature_snapshot"]["retained_buy_signal"] == 1.25
    assert model["feature_action_summary"][0]["feature"] in {"retained_buy_signal", "shared_feature"}
    assert comparison["baseline_profile"] == "retained_model"


def test_build_feature_action_summary_compares_buy_and_sell_snapshots():
    rows = [
        {
            "date": "2026-01-02",
            "action": "HOLD",
            "feature_snapshot": {"alpha_signal": 0.0, "shared_feature": 10.0},
        },
        {
            "date": "2026-01-03",
            "action": "BUY",
            "feature_snapshot": {"alpha_signal": 1.0, "shared_feature": 12.0},
        },
        {
            "date": "2026-01-04",
            "action": "BUY",
            "feature_snapshot": {"alpha_signal": 3.0, "shared_feature": 14.0},
        },
        {
            "date": "2026-01-05",
            "action": "SELL",
            "feature_snapshot": {"alpha_signal": -1.0, "shared_feature": 8.0},
        },
    ]

    summary = build_feature_action_summary(rows, feature_columns=["alpha_signal", "shared_feature"])
    by_feature = {row["feature"]: row for row in summary}

    assert by_feature["alpha_signal"]["buy_count"] == 2
    assert by_feature["alpha_signal"]["sell_count"] == 1
    assert by_feature["alpha_signal"]["buy_mean"] == pytest.approx(2.0)
    assert by_feature["alpha_signal"]["sell_mean"] == pytest.approx(-1.0)
    assert by_feature["alpha_signal"]["buy_minus_sell"] == pytest.approx(3.0)
    assert by_feature["shared_feature"]["buy_minus_sell"] == pytest.approx(5.0)


def test_build_dashboard_v2_data_reports_dashboard_refresh_task(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    model_csv = tmp_path / "runtime" / "model" / "card" / "smoke" / "smoke_soxl_daily.csv"
    model_csv.parent.mkdir(parents=True)
    _write_csv(
        model_csv,
        [
            {
                "datetime": "2026-01-02",
                "strategy_nav": 100.0,
                "buyhold_nav": 100.0,
                "position": 0.0,
                "action": "HOLD",
                "price": 10.0,
            }
        ],
    )
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T00:00:00+00:00",
                "latest_data_date": "2026-01-02",
                "safe_backtest_end": "2026-01-02",
                "profiles": [
                    {
                        "profile": "smoke",
                        "status": "canonical",
                        "actor": "runtime/model/card/smoke/soxl_smoke_actor.pt",
                        "output": "runtime/model/card/smoke/smoke_soxl_daily.csv",
                        "result": "current",
                    }
                ],
            }
        )
    )
    production = tmp_path / "production"
    production.mkdir()
    (production / "run_dashboard_refresh.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (production / "com.soxl.dashboard.refresh.plist").write_text("<plist></plist>\n", encoding="utf-8")

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        market_rows={"2026-01-02": {"open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0, "volume": 1000}},
        selected_profiles=["smoke"],
        baseline_profile="smoke",
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    task = {item["name"]: item for item in manifest["auto_update"]["tasks"]}["dashboard_refresh"]
    assert manifest["auto_update"]["configured"] is True
    assert task["enabled"] is True
    assert task["schedule"] == "06:30 Tue-Sat Asia/Shanghai"
    assert task["command_path"] == "production/run_dashboard_refresh.sh"


def test_build_dashboard_v2_data_can_report_loaded_dashboard_refresh_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    model_csv = tmp_path / "runtime" / "model" / "card" / "smoke" / "smoke_soxl_daily.csv"
    model_csv.parent.mkdir(parents=True)
    _write_csv(
        model_csv,
        [
            {
                "datetime": "2026-01-02",
                "strategy_nav": 100.0,
                "buyhold_nav": 100.0,
                "position": 0.0,
                "action": "HOLD",
                "price": 10.0,
            }
        ],
    )
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T00:00:00+00:00",
                "latest_data_date": "2026-01-02",
                "safe_backtest_end": "2026-01-02",
                "profiles": [
                    {
                        "profile": "smoke",
                        "status": "canonical",
                        "actor": "runtime/model/card/smoke/soxl_smoke_actor.pt",
                        "output": "runtime/model/card/smoke/smoke_soxl_daily.csv",
                        "result": "current",
                    }
                ],
            }
        )
    )
    production = tmp_path / "production"
    production.mkdir()
    (production / "run_dashboard_refresh.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (production / "com.soxl.dashboard.refresh.plist").write_text("<plist></plist>\n", encoding="utf-8")
    launchagents_dir = tmp_path / "LaunchAgents"
    launchagents_dir.mkdir()
    (launchagents_dir / "com.soxl.dashboard.refresh.plist").write_text("<plist></plist>\n", encoding="utf-8")
    monkeypatch.setenv(
        "SOXL_DASHBOARD_V2_LAUNCHCTL_LIST",
        "-\t0\tcom.soxl.dashboard.refresh\n-\t0\tcom.soxl.daily.scheduler\n",
    )
    monkeypatch.setenv("SOXL_DASHBOARD_V2_LAUNCHAGENTS_DIR", str(launchagents_dir))

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        market_rows={"2026-01-02": {"open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0, "volume": 1000}},
        selected_profiles=["smoke"],
        baseline_profile="smoke",
        detect_launchd=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    task = {item["name"]: item for item in manifest["auto_update"]["tasks"]}["dashboard_refresh"]
    assert task["configured"] is True
    assert task["installed"] is True
    assert task["loaded"] is True
    assert task["enabled"] is True


def test_build_dashboard_v2_data_attaches_model_retention_gate_status(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    model_csv = tmp_path / "runtime" / "model" / "card" / "smoke" / "smoke_soxl_daily.csv"
    model_csv.parent.mkdir(parents=True)
    _write_csv(
        model_csv,
        [
            {
                "datetime": "2026-01-02",
                "strategy_nav": 100.0,
                "buyhold_nav": 100.0,
                "position": 0.0,
                "action": "HOLD",
                "price": 10.0,
            },
            {
                "datetime": "2026-01-05",
                "strategy_nav": 105.0,
                "buyhold_nav": 110.0,
                "position": 0.5,
                "action": "BUY",
                "price": 11.0,
            },
        ],
    )
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T00:00:00+00:00",
                "latest_data_date": "2026-01-05",
                "safe_backtest_end": "2026-01-05",
                "profiles": [
                    {
                        "profile": "smoke",
                        "status": "canonical",
                        "actor": "runtime/model/card/smoke/soxl_smoke_actor.pt",
                        "output": "runtime/model/card/smoke/smoke_soxl_daily.csv",
                        "result": "current",
                    }
                ],
            }
        )
    )
    retention_summary = tmp_path / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1" / "summary.json"
    retention_summary.parent.mkdir(parents=True)
    retention_summary.write_text(
        json.dumps(
            {
                "summary": {
                    "passing_profile_count": 0,
                    "passing_profiles": [],
                    "max_walkforward_drawdown_pct": 20.0,
                    "required_best_return_pct": 213.14,
                    "reference_profile": "soxl_backtest_policy_reimplementation_v1",
                    "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
                    "registry_keep_actor_count": 3,
                    "registry_keep_actor_strict_violation_count": 2,
                    "registry_keep_actor_strict_violations": ["high_dd", "missing_replay"],
                },
                "rows": [
                    {
                        "profile": "smoke",
                        "gate_pass": False,
                        "retention_decision": "retain_as_evidence_only",
                        "failures": ["wf_max_drawdown_ge_20:wf_2022", "ytd_return_below_best_alignment"],
                        "worst_wf_max_drawdown_pct": 24.2,
                        "required_best_return_pct": 213.14,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        market_rows={
            "2026-01-02": {"open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0, "volume": 1000},
            "2026-01-05": {"open": 10.5, "high": 11.5, "low": 10.0, "close": 11.0, "volume": 1200},
        },
        selected_profiles=["smoke"],
        baseline_profile="smoke",
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    model = json.loads((output_dir / "models" / "smoke.json").read_text())

    assert manifest["model_retention_gate"]["passing_profile_count"] == 0
    assert manifest["model_retention_gate"]["max_walkforward_drawdown_pct"] == 20.0
    assert manifest["model_retention_gate"]["registry_keep_actor_count"] == 3
    assert manifest["model_retention_gate"]["registry_keep_actor_strict_violation_count"] == 2
    assert manifest["model_retention_gate"]["registry_keep_actor_strict_violations"] == ["high_dd", "missing_replay"]
    assert manifest["model_retention_gate"]["source_path"] == "runtime/model_selection/card/model_retention_walkforward_gate_v1/summary.json"
    assert model["retention_gate_pass"] is False
    assert model["retention_decision"] == "retain_as_evidence_only"
    assert model["retention_failures"] == ["wf_max_drawdown_ge_20:wf_2022", "ytd_return_below_best_alignment"]
    assert model["worst_wf_max_drawdown_pct"] == 24.2


def test_build_dashboard_v2_data_loads_market_from_requested_root(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    model_csv = tmp_path / "runtime" / "model" / "card" / "smoke" / "smoke_soxl_daily.csv"
    model_csv.parent.mkdir(parents=True)
    _write_csv(
        model_csv,
        [
            {
                "datetime": "2026-02-03",
                "strategy_nav": 100.0,
                "buyhold_nav": 100.0,
                "position": 0.0,
                "action": "HOLD",
                "price": 20.5,
            }
        ],
    )
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T00:00:00+00:00",
                "latest_data_date": "2026-02-03",
                "safe_backtest_end": "2026-02-03",
                "profiles": [
                    {
                        "profile": "smoke",
                        "status": "canonical",
                        "output": "runtime/model/card/smoke/smoke_soxl_daily.csv",
                        "result": "current",
                    }
                ],
            }
        )
    )
    processed = tmp_path / "data" / "processed" / "panels" / "soxl_soxx_vix_full.csv"
    processed.parent.mkdir(parents=True)
    _write_csv(
        processed,
        [
            {
                "date": "2026-02-03",
                "soxl_open": 20.0,
                "soxl_high": 21.0,
                "soxl_low": 19.5,
                "soxl_close": 20.5,
                "soxl_volume": 2000,
                "vix": 17.0,
            }
        ],
    )

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        selected_profiles=["smoke"],
        baseline_profile="smoke",
    )

    market = json.loads((output_dir / "market" / "SOXL.json").read_text())
    assert len(market["rows"]) == 1
    row = market["rows"][0]
    assert row["date"] == "2026-02-03"
    assert row["open"] == 20.0
    assert row["close"] == 20.5
    assert row["volume"] == 2000
    assert row["vix"] == 17.0
    assert row["rsi6"] is not None


def test_build_dashboard_v2_data_adds_soxl_backtest_reference_baseline(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    model_csv = tmp_path / "runtime" / "model" / "card" / "smoke" / "smoke_soxl_daily.csv"
    model_csv.parent.mkdir(parents=True)
    _write_csv(
        model_csv,
        [
            {
                "datetime": "2026-01-02",
                "strategy_nav": 100.0,
                "buyhold_nav": 100.0,
                "position": 0.0,
                "action": "HOLD",
                "price": 47.24,
            },
            {
                "datetime": "2026-03-30",
                "strategy_nav": 110.0,
                "buyhold_nav": 90.0,
                "position": 0.5,
                "action": "BUY",
                "price": 40.62,
            },
            {
                "datetime": "2026-05-27",
                "strategy_nav": 120.0,
                "buyhold_nav": 130.0,
                "position": 0.1,
                "action": "SELL",
                "price": 217.98,
            },
        ],
    )
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T00:00:00+00:00",
                "latest_data_date": "2026-05-27",
                "safe_backtest_end": "2026-05-27",
                "profiles": [
                    {
                        "profile": "smoke",
                        "status": "canonical",
                        "output": "runtime/model/card/smoke/smoke_soxl_daily.csv",
                        "result": "current",
                    }
                ],
            }
        )
    )
    reference_csv = (
        tmp_path
        / "runtime"
        / "production"
        / "card"
        / "soxl_backtest_policy_reimplementation_v1"
        / "new_policy_daily.csv"
    )
    reference_csv.parent.mkdir(parents=True)
    _write_csv(
        reference_csv,
        [
            {
                "window": "june_2026",
                "date": "2026-05-27",
                "strategy_nav": 999.0,
                "buyhold_nav": 999.0,
                "position": 0.9,
                "action": "BUY",
                "price": 200.0,
            },
            {
                "window": "ytd_2026",
                "date": "2026-01-02",
                "strategy_nav": 100000.0,
                "buyhold_nav": 100000.0,
                "position": 0.0,
                "action": "HOLD",
                "price": 47.24,
            },
            {
                "window": "ytd_2026",
                "date": "2026-03-30",
                "strategy_nav": 100000.0,
                "buyhold_nav": 90000.0,
                "position": 0.2998,
                "action": "BUY",
                "price": 40.62,
            },
            {
                "window": "ytd_2026",
                "date": "2026-05-27",
                "strategy_nav": 230891.68,
                "buyhold_nav": 130000.0,
                "position": 0.0699,
                "action": "TAKE_PROFIT",
                "price": 217.98,
            },
        ],
    )
    market_rows = {
        "2026-01-02": {"open": 47.0, "high": 48.0, "low": 46.0, "close": 47.24, "volume": 1000},
        "2026-03-30": {"open": 40.0, "high": 42.0, "low": 39.0, "close": 40.62, "volume": 1000},
        "2026-05-27": {"open": 216.0, "high": 218.5, "low": 210.0, "close": 217.98, "volume": 1000},
    }

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        market_rows=market_rows,
    )

    reference_name = "soxl_backtest_policy_reimplementation_v1"
    manifest = json.loads((output_dir / "manifest.json").read_text())
    reference = json.loads((output_dir / "models" / f"{reference_name}.json").read_text())
    comparison = json.loads((output_dir / "comparison" / "default.json").read_text())

    assert manifest["default_baseline_profile"] == reference_name
    assert manifest["default_comparison_profiles"] == [reference_name, "smoke"]
    assert manifest["default_chart_profile"] == "smoke"
    assert reference["source_path"] == "runtime/production/card/soxl_backtest_policy_reimplementation_v1/new_policy_daily.csv"
    assert reference["rows"][-1]["action"] == "SELL"
    assert [event["kind"] for event in reference["events"] if event["kind"] in {"BUY", "SELL"}] == ["BUY", "SELL"]
    assert comparison["baseline_profile"] == reference_name
    assert comparison["common_start"] == "2026-01-02"
    assert comparison["common_end"] == "2026-05-27"
    assert comparison["rows"][-1]["profiles"][reference_name]["buyhold_return"] == pytest.approx(0.30)
    assert comparison["rows"][-1]["profiles"][reference_name]["return_delta_vs_buyhold"] == pytest.approx(1.0089168)


def test_normalize_model_rows_rejects_invalid_numeric_fields(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "datetime,strategy_nav,buyhold_nav,position,action,price\n"
        "2026-01-02,nope,100,0,HOLD,10\n"
    )

    with pytest.raises(ValueError, match="strategy_nav"):
        normalize_model_rows(csv_path, safe_backtest_end="2026-01-02")


def test_derive_model_events_uses_actions_and_position_changes():
    rows = [
        {"date": "2026-01-02", "action": "HOLD", "position": 0.0, "price": 10.0, "strategy_nav": 100.0},
        {"date": "2026-01-03", "action": "HOLD", "position": 0.4, "price": 10.5, "strategy_nav": 101.0},
        {"date": "2026-01-04", "action": "SELL", "position": 0.1, "price": 10.2, "strategy_nav": 100.5},
    ]

    events = derive_model_events(rows, material_position_delta=0.25)

    assert [event["kind"] for event in events] == ["POSITION_INCREASE", "SELL", "POSITION_DECREASE"]
    assert events[0]["position_delta"] == 0.4


def test_build_comparison_payload_reports_action_mismatches_against_baseline():
    baseline_rows = [
        {"date": "2026-01-02", "strategy_nav": 100.0, "position": 0.0, "action": "HOLD"},
        {"date": "2026-01-03", "strategy_nav": 101.0, "position": 0.3, "action": "BUY"},
    ]
    challenger_rows = [
        {"date": "2026-01-02", "strategy_nav": 100.0, "position": 0.0, "action": "HOLD"},
        {"date": "2026-01-03", "strategy_nav": 99.0, "position": 0.0, "action": "HOLD"},
    ]
    payload = build_comparison_payload(
        {
            "baseline": {"rows": baseline_rows},
            "challenger": {"rows": challenger_rows},
        },
        baseline_profile="baseline",
        compared_profiles=["baseline", "challenger"],
    )

    assert payload["common_start"] == "2026-01-02"
    assert payload["common_end"] == "2026-01-03"
    assert payload["rows"][1]["profiles"]["challenger"]["nav_delta_vs_baseline"] == -2.0
    assert payload["rows"][1]["profiles"]["challenger"]["buyhold_nav"] == 99.0
    assert payload["rows"][1]["profiles"]["challenger"]["return_delta_vs_buyhold"] == 0.0
    assert payload["rows"][1]["profiles"]["challenger"]["action_mismatch"] is True
    assert payload["rows"][1]["action_mismatch_count"] == 1
