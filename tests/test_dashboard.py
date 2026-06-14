from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from scripts.build_backtest_dashboard import (
    _load_soxlbacktest_reference_version,
    _load_post_train_review_versions,
    _load_retained_dashboard_versions,
    build_html,
    compact_payload_rows,
    load_research_backtests,
    load_run_profile_windows,
    normalize_post_train_review_rows,
    select_default_chart_version,
    select_best_version,
    select_latest_version,
)
import scripts.build_backtest_dashboard as dashboard_builder
from src.soxl_quant.app.dashboard import review_daily_csv_path, write_review_daily_csv


ROOT = Path(__file__).resolve().parents[1]


def _assert_no_sensitive_public_fields(payload: object) -> None:
    forbidden_keys = {
        "account_value",
        "cash",
        "shares",
        "market_value",
        "current_position",
        "resulting_position",
        "actual_position",
        "position_gap",
        "filled_qty",
        "quantity",
        "broker_order_id",
        "reconcile_action",
    }

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                assert str(key) not in forbidden_keys
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)


def test_dashboard_builder_imports_without_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy, sys; "
                "root = sys.argv[1]; "
                "sys.path = ['scripts'] + [p for p in sys.path if p not in ('', root)]; "
                "runpy.run_path('scripts/build_backtest_dashboard.py', run_name='dashboard_builder_import_test')"
            ),
            str(ROOT),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_review_daily_csv_path_uses_dashboard_convention():
    assert review_daily_csv_path("balanced_rule_residual_v1").name == "balanced_rule_residual_v1_soxl_daily.csv"


def test_write_review_daily_csv_keeps_action_column(tmp_path: Path):
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    portfolio = pd.DataFrame({"account": [1_000_000.0, 1_010_000.0, 1_020_000.0]}, index=idx)
    prices = pd.Series([10.0, 10.2, 10.4], index=idx)
    out = tmp_path / "demo.csv"

    write_review_daily_csv(
        out,
        portfolio,
        prices,
        [0.0, 1.0, 0.0],
        [
            (idx[1], 10.2, 1.0, "buy"),
            (idx[2], 10.4, 0.0, "sell"),
        ],
    )

    rows = out.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].endswith(",action")
    assert rows[2].endswith(",BUY")
    assert rows[3].endswith(",SELL")


def test_select_latest_version_prefers_most_recent_timestamp():
    versioned = {
        "older": {"_meta": {"updated_at": 10, "source_kind": "generalization_version"}},
        "newer": {"_meta": {"updated_at": 20, "source_kind": "generalization_version"}},
    }
    assert select_latest_version(versioned) == "newer"


def test_select_latest_version_does_not_prefer_post_train_review_over_static_snapshots():
    versioned = {
        "snapshot_newer": {"_meta": {"updated_at": 50, "source_kind": "generalization_version"}},
        "post_train": {"_meta": {"updated_at": 20, "source_kind": "post_train_review"}},
    }
    assert select_latest_version(versioned) == "snapshot_newer"


def test_select_latest_version_prefers_paper_trading_for_static_dashboard():
    versioned = {
        "snapshot_newer": {"_meta": {"updated_at": 300, "source_kind": "generalization_version"}},
        "post_train": {"_meta": {"updated_at": 200, "source_kind": "post_train_review"}},
        "paper_trading": {"_meta": {"updated_at": 100, "source_kind": "paper_trading"}},
    }
    assert select_latest_version(versioned) == "paper_trading"


def test_select_default_chart_version_prefers_readable_model_over_short_paper_view():
    versioned = {
        "paper_trading": {
            "_meta": {"updated_at": 300, "source_kind": "paper_trading"},
            "windows": {"SOXL": {"start": "2026-06-04", "end": "2026-06-08", "rows": 2}},
        },
        "balanced_rule_residual_v19_moe_v2": {
            "_meta": {"updated_at": 200, "source_kind": "retained_dashboard_export"},
            "windows": {"SOXL": {"start": "2020-01-02", "end": "2026-05-19", "rows": 1603}},
        },
        "soxl_backtest_policy_reimplementation_v1": {
            "_meta": {"updated_at": 250, "source_kind": "reference_baseline"},
            "windows": {"SOXL": {"start": "2026-01-02", "end": "2026-06-09", "rows": 109}},
        },
    }

    assert select_default_chart_version(versioned, best_version="balanced_rule_residual_v19_moe_v2") == (
        "balanced_rule_residual_v19_moe_v2"
    )


def test_select_best_version_maps_soxl_prefix():
    model_versions = [
        {"name": "soxl_balanced_rule_residual_v1", "test_sharpe": 0.96},
        {"name": "soxl_balanced", "test_sharpe": 0.85},
    ]
    versioned = {
        "balanced_rule_residual_v1": {"_meta": {"updated_at": 20}},
        "balanced": {"_meta": {"updated_at": 10}},
    }
    assert select_best_version(model_versions, versioned) == "balanced_rule_residual_v1"


def test_normalize_post_train_review_rows_keeps_explicit_action_column():
    rows = normalize_post_train_review_rows(
        [
            {
                "datetime": "2024-01-02",
                "strategy_nav": "1000000",
                "price": "10",
                "buyhold_nav": "1000000",
                "position": "0.3",
                "action": "HOLD",
            },
            {
                "datetime": "2024-01-03",
                "strategy_nav": "1010000",
                "price": "10.1",
                "buyhold_nav": "1005000",
                "position": "0.4",
                "action": "BUY",
            },
            {
                "datetime": "2024-01-04",
                "strategy_nav": "1005000",
                "price": "10.0",
                "buyhold_nav": "1001000",
                "position": "0.2",
                "action": "SELL",
            },
        ]
    )
    assert [r["action"] for r in rows] == ["HOLD", "BUY", "SELL"]


def test_normalize_post_train_review_rows_falls_back_to_position_change_inference():
    rows = normalize_post_train_review_rows(
        [
            {
                "datetime": "2024-01-02",
                "strategy_nav": "1000000",
                "price": "10",
                "buyhold_nav": "1000000",
                "position": "0.3",
                "action": "",
            },
            {
                "datetime": "2024-01-03",
                "strategy_nav": "1010000",
                "price": "10.1",
                "buyhold_nav": "1005000",
                "position": "0.1",
                "action": "",
            },
        ]
    )
    assert [r["action"] for r in rows] == ["BUY", "SELL"]


def test_normalize_post_train_review_rows_does_not_synthesize_terminal_sell():
    rows = normalize_post_train_review_rows(
        [
            {
                "datetime": "2026-05-21",
                "strategy_nav": "1016052.687283",
                "price": "178.389999",
                "buyhold_nav": "3723439.777901",
                "position": "0.772515",
                "action": "BUY",
            },
            {
                "datetime": "2026-05-22",
                "strategy_nav": "1069600.679227",
                "price": "190.559998",
                "buyhold_nav": "3977457.699220",
                "position": "0.783904",
                "action": "HOLD",
            },
        ]
    )

    assert [r["action"] for r in rows] == ["BUY", "HOLD"]


def test_normalize_post_train_review_rows_suppresses_impossible_zero_position_action():
    rows = normalize_post_train_review_rows(
        [
            {
                "datetime": "2026-03-31",
                "strategy_nav": "1000000",
                "price": "47.91",
                "buyhold_nav": "1000000",
                "position": "0.0",
                "action": "BUY",
            }
        ]
    )

    assert rows[0]["action"] == "HOLD"


def test_load_versions_keeps_short_paper_trading_history(tmp_path: Path, monkeypatch):
    paper_dir = tmp_path / "runtime" / "soxl" / "paper_trading"
    paper_dir.mkdir(parents=True)
    (paper_dir / "paper_trading_soxl_daily.csv").write_text(
        "\n".join(
            [
                "datetime,strategy_nav,price,buyhold_nav,position,excess,excess_pct,action",
                "2026-05-14,1000000,186.19,1000000,0,0,0,HOLD",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_builder, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard_builder, "load_ohlc_and_indicators", lambda _etf: {})

    versions = _load_post_train_review_versions(features=None)

    assert "paper_trading" in versions
    paper = versions["paper_trading"]
    assert paper["_meta"]["source_kind"] == "paper_trading"
    assert paper["windows"]["SOXL"] == {"start": "2026-05-14", "end": "2026-05-14", "rows": 1}


def test_load_versions_excludes_model_card_post_train_preview_daily_csv(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "runtime" / "model" / "card" / "bad_preview"
    model_dir.mkdir(parents=True)
    rows = [
        "datetime,strategy_nav,price,buyhold_nav,position,excess,excess_pct,action",
        *[
            f"2026-01-{day:02d},{1000 + day},10,{1000 + day},0,0,0,HOLD"
            for day in range(2, 12)
        ],
    ]
    (model_dir / "bad_preview_soxl_daily.csv").write_text("\n".join(rows), encoding="utf-8")
    monkeypatch.setattr(dashboard_builder, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard_builder, "load_ohlc_and_indicators", lambda _etf: {})

    versions = _load_post_train_review_versions(features=None)

    assert "bad_preview" not in versions


def test_load_versions_enriches_paper_trading_row_with_latest_non_dry_summary_metadata(
    tmp_path: Path,
    monkeypatch,
):
    paper_dir = tmp_path / "runtime" / "soxl" / "paper_trading"
    run_dir = paper_dir / "2026-05-20"
    duplicate_dir = run_dir / "duplicate_skip" / "v19reg_v22_budget_usage_bull_guard_v1"
    dry_run_dir = run_dir / "dry_run"
    duplicate_dir.mkdir(parents=True)
    dry_run_dir.mkdir(parents=True)
    (paper_dir / "paper_trading_soxl_daily.csv").write_text(
        "\n".join(
            [
                "datetime,strategy_nav,price,buyhold_nav,position,excess,excess_pct,action",
                "2026-05-20,1001806.104446,173.199997,3615111.612583,0.484086,-2613305.508136,-72.288377,SELL",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "date": "2026-05-20",
                "profile": "balanced_rule_residual_v19",
                "broker": "paper",
                "dry_run": False,
                "executed": True,
                "signal": {"target_position": 0.5},
                "risk_gate": {"approved": True, "capped_target_position": 0.5},
                "reconcile": {"action_required": "none"},
                "account": {
                    "cash": 10_000,
                    "shares": 100,
                    "market_value": 17_319.9997,
                    "account_value": 27_319.9997,
                    "position": 0.63,
                },
            }
        ),
        encoding="utf-8",
    )
    (duplicate_dir / "summary.json").write_text(
        json.dumps(
            {
                "date": "2026-05-20",
                "profile": "v19reg_v22_budget_usage_bull_guard_v1",
                "broker": "paper",
                "dry_run": False,
                "executed": False,
                "execution_result": {
                    "status": "duplicate_skipped",
                    "target_position": 0.484086,
                    "resulting_position": 0.484086,
                    "filled_qty": 100,
                },
                "signal": {
                    "target_position": 0.484086,
                    "current_position": 0.484086,
                },
                "reconcile": {"action_required": "none"},
            }
        ),
        encoding="utf-8",
    )
    (dry_run_dir / "summary.json").write_text(
        json.dumps(
            {
                "date": "2026-05-20",
                "profile": "dry_run_profile",
                "broker": "paper",
                "dry_run": True,
                "executed": False,
                "signal": {"target_position": 0.1},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_builder, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard_builder, "load_ohlc_and_indicators", lambda _etf: {})

    versions = _load_post_train_review_versions(features=None)

    detail = versions["paper_trading"]["detailed"]["SOXL"]
    row = detail["rows"][0]
    meta = detail["meta"]
    assert row["profile"] == "v19reg_v22_budget_usage_bull_guard_v1"
    assert row["execution_status"] == "duplicate_skipped"
    assert row["dry_run"] is False
    assert row["target_position"] == 0.484086
    assert row["entry_target_position"] == 0.484086
    assert row["entry_status"] == "duplicate_skipped"
    assert "reconcile_action" not in row
    assert row["summary_path"] == (
        "runtime/soxl/paper_trading/2026-05-20/duplicate_skip/"
        "v19reg_v22_budget_usage_bull_guard_v1/summary.json"
    )
    assert meta["source_kind"] == "paper_trading"
    assert meta["latest_data_date"] == "2026-05-20"
    assert meta["latest_profile"] == "v19reg_v22_budget_usage_bull_guard_v1"
    assert meta["latest_execution_status"] == "duplicate_skipped"
    assert meta["latest_target_position"] == 0.484086
    assert meta["latest_entry_status"] == "duplicate_skipped"
    assert meta["latest_entry_target_position"] == 0.484086
    assert meta["latest_signal_profile"] == "v19reg_v22_budget_usage_bull_guard_v1"
    assert meta["latest_signal_execution_status"] == "duplicate_skipped"
    assert meta["latest_public_review"]["entry_status"] == "duplicate_skipped"
    assert meta["latest_public_review"]["entry_target_position"] == 0.484086
    assert meta["latest_signal_summary_path"] == (
        "runtime/soxl/paper_trading/2026-05-20/duplicate_skip/"
        "v19reg_v22_budget_usage_bull_guard_v1/summary.json"
    )
    _assert_no_sensitive_public_fields(meta)


def test_load_versions_exports_latest_dry_run_signal_metadata_without_annotating_seed_row(
    tmp_path: Path,
    monkeypatch,
):
    paper_dir = tmp_path / "runtime" / "soxl" / "paper_trading"
    dry_run_dir = paper_dir / "2026-05-22" / "dry_run" / "futu"
    dry_run_dir.mkdir(parents=True)
    (paper_dir / "paper_trading_soxl_daily.csv").write_text(
        "\n".join(
            [
                "datetime,strategy_nav,price,buyhold_nav,position,excess,excess_pct,action",
                "2026-05-22,10000,190.56,10000,0,0,0,HOLD",
            ]
        ),
        encoding="utf-8",
    )
    (dry_run_dir / "summary.json").write_text(
        json.dumps(
            {
                "date": "2026-05-22",
                "profile": "v19reg_v22_budget_usage_bull_guard_v1",
                "broker": "futu",
                "dry_run": True,
                "executed": False,
                "signal": {
                    "target_position": 0.8500000003,
                    "current_position": 0.0,
                    "price": 190.56,
                    "symbol": "SOXL",
                },
                "risk_gate": {
                    "approved": True,
                    "capped_target_position": 0.35,
                },
                "composition": {
                    "slow_profile": "balanced_rule_residual_v19_reg",
                    "fast_profile": "balanced_rule_residual_v22",
                    "mode": "trend_hold",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_builder, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard_builder, "load_ohlc_and_indicators", lambda _etf: {})

    versions = _load_post_train_review_versions(features=None)

    detail = versions["paper_trading"]["detailed"]["SOXL"]
    row = detail["rows"][0]
    meta = detail["meta"]
    assert "profile" not in row
    assert "execution_status" not in row
    assert meta["latest_data_date"] == "2026-05-22"
    assert meta["latest_signal_profile"] == "v19reg_v22_budget_usage_bull_guard_v1"
    assert meta["latest_signal_execution_status"] == "dry_run"
    assert meta["latest_signal_target_position"] == 0.8500000003
    assert meta["latest_signal_capped_target_position"] == 0.35
    assert meta["latest_entry_status"] == "dry_run"
    assert meta["latest_entry_side"] == "BUY"
    assert meta["latest_entry_target_position"] == 0.35
    assert meta["latest_public_review"]["entry_signal_target_position"] == 0.8500000003
    assert meta["latest_public_review"]["entry_target_position"] == 0.35
    assert meta["latest_signal_risk_gate_approved"] is True
    assert meta["latest_signal_slow_profile"] == "balanced_rule_residual_v19_reg"
    assert meta["latest_signal_fast_profile"] == "balanced_rule_residual_v22"
    assert meta["latest_signal_summary_path"] == (
        "runtime/soxl/paper_trading/2026-05-22/dry_run/futu/summary.json"
    )
    assert "latest_signal_reconcile_action" not in meta
    _assert_no_sensitive_public_fields(meta)


def test_load_versions_promotes_newer_dry_run_signal_to_visible_paper_status(
    tmp_path: Path,
    monkeypatch,
):
    paper_dir = tmp_path / "runtime" / "soxl" / "paper_trading"
    dry_run_dir = paper_dir / "2026-06-10" / "dry_run"
    dry_run_dir.mkdir(parents=True)
    (paper_dir / "paper_trading_soxl_daily.csv").write_text(
        "\n".join(
            [
                "datetime,strategy_nav,price,buyhold_nav,position,excess,excess_pct,action",
                "2026-06-08,10000,211.44,10000,0,0,0,HOLD",
            ]
        ),
        encoding="utf-8",
    )
    (dry_run_dir / "summary.json").write_text(
        json.dumps(
            {
                "date": "2026-06-10",
                "profile": "v19reg_v22_budget_usage_bull_guard_v1",
                "broker": "paper",
                "dry_run": True,
                "executed": False,
                "signal": {"target_position": 0.205},
                "risk_gate": {"approved": True, "capped_target_position": 0.205},
                "composition": {
                    "slow_profile": "balanced_rule_residual_v19_reg",
                    "fast_profile": "balanced_rule_residual_v22",
                    "mode": "defensive_hold",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_builder, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard_builder, "load_ohlc_and_indicators", lambda _etf: {})

    versions = _load_post_train_review_versions(features=None)

    detail = versions["paper_trading"]["detailed"]["SOXL"]
    rows = detail["rows"]
    meta = detail["meta"]
    assert [row["datetime"] for row in rows] == ["2026-06-08"]
    assert "profile" not in rows[0]
    assert meta["latest_data_date"] == "2026-06-10"
    assert meta["latest_profile"] == "v19reg_v22_budget_usage_bull_guard_v1"
    assert meta["latest_execution_status"] == "dry_run"
    assert meta["latest_target_position"] == 0.205
    assert meta["latest_summary_path"] == "runtime/soxl/paper_trading/2026-06-10/dry_run/summary.json"
    assert meta["latest_signal_date"] == "2026-06-10"


def test_load_versions_drops_unannotated_paper_seed_row_and_rebases_buyhold(
    tmp_path: Path,
    monkeypatch,
):
    paper_dir = tmp_path / "runtime" / "soxl" / "paper_trading"
    run_dir = paper_dir / "2026-05-14"
    next_run_dir = paper_dir / "2026-05-15"
    run_dir.mkdir(parents=True)
    next_run_dir.mkdir(parents=True)
    (paper_dir / "paper_trading_soxl_daily.csv").write_text(
        "\n".join(
            [
                "datetime,strategy_nav,price,buyhold_nav,position,excess,excess_pct,action",
                "2026-03-31,1000000,47.91,1000000,0,0,0,BUY",
                "2026-05-14,999664.857996,186.190002,3886245.093747,0.335254,-2886580.235751,-74.276845,BUY",
                "2026-05-15,937490.753601,151.75,3167397.210598,0.485605,-2229906.456997,-70.401857,HOLD",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "date": "2026-05-14",
                "profile": "paper_profile",
                "broker": "paper",
                "dry_run": False,
                "executed": True,
                "signal": {"target_position": 0.335254},
            }
        ),
        encoding="utf-8",
    )
    (next_run_dir / "summary.json").write_text(
        json.dumps(
            {
                "date": "2026-05-15",
                "profile": "paper_profile",
                "broker": "paper",
                "dry_run": False,
                "executed": False,
                "execution_result": {"status": "skipped", "target_position": 0.485605},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_builder, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard_builder, "load_ohlc_and_indicators", lambda _etf: {})

    versions = _load_post_train_review_versions(features=None)

    rows = versions["paper_trading"]["detailed"]["SOXL"]["rows"]
    assert [r["datetime"] for r in rows] == ["2026-05-14", "2026-05-15"]
    assert rows[0]["strategy_nav"] == pytest.approx(1.0)
    assert rows[0]["buyhold_nav"] == pytest.approx(1.0)
    assert rows[1]["strategy_nav"] == pytest.approx(937490.753601 / 999664.857996)
    assert rows[1]["buyhold_nav"] == pytest.approx(3167397.210598 / 3886245.093747)
    assert rows[0]["excess"] == pytest.approx(0.0, abs=1e-6)
    assert max(float(row["strategy_nav"]) for row in rows) < 10.0


def test_load_versions_keeps_retained_dashboard_export_without_source_csv(tmp_path: Path, monkeypatch):
    retained_dir = tmp_path / "dashboard_data" / "versioned" / "retained_ablation"
    retained_dir.mkdir(parents=True)
    (retained_dir / "SOXL.json").write_text(
        """
{
  "summary": {"etf": "SOXL", "strategy_return": 0.1},
  "csv_path": "runtime/model/card/retained_ablation/retained_ablation_soxl_daily.csv",
  "rows": [
    {"datetime": "2026-05-19", "strategy_nav": 1000000, "price": 100, "buyhold_nav": 1000000, "position": 0.2, "excess": 0, "excess_pct": 0, "action": "BUY"}
  ],
  "regime_shapes": [],
  "trades": [],
  "kelly": {}
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_builder, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard_builder, "DASHBOARD_DATA_DIR", tmp_path / "dashboard_data")
    monkeypatch.setattr(
        dashboard_builder,
        "load_ohlc_and_indicators",
        lambda _etf: {
            "2026-05-19": {
                "open": 98.0,
                "high": 101.0,
                "low": 97.0,
                "close": 100.0,
                "volume": 12345,
            }
        },
    )

    versions = _load_retained_dashboard_versions()

    assert "retained_ablation" in versions
    retained = versions["retained_ablation"]
    assert retained["_meta"]["source_kind"] == "retained_dashboard_export"
    assert retained["windows"]["SOXL"] == {"start": "2026-05-19", "end": "2026-05-19", "rows": 1}
    assert retained["detailed"]["SOXL"]["rows"][0]["open"] == 98.0
    assert retained["detailed"]["SOXL"]["rows"][0]["close"] == 100.0


def test_load_versions_skips_retained_post_train_review_export(tmp_path: Path, monkeypatch):
    retained_dir = tmp_path / "dashboard_data" / "versioned" / "bad_preview"
    retained_dir.mkdir(parents=True)
    (retained_dir / "SOXL.json").write_text(
        json.dumps(
            {
                "summary": {"etf": "SOXL", "strategy_return": 0.1},
                "csv_path": "runtime/model/card/bad_preview/bad_preview_soxl_daily.csv",
                "meta": {"source_kind": "post_train_review"},
                "rows": [
                    {
                        "datetime": "2026-05-19",
                        "strategy_nav": 1000000,
                        "price": 100,
                        "buyhold_nav": 1000000,
                        "position": 0.2,
                        "excess": 0,
                        "excess_pct": 0,
                        "action": "BUY",
                    }
                ],
                "regime_shapes": [],
                "trades": [],
                "kelly": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_builder, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard_builder, "DASHBOARD_DATA_DIR", tmp_path / "dashboard_data")

    versions = _load_retained_dashboard_versions()

    assert "bad_preview" not in versions


def test_load_run_profile_windows_reads_replay_csv_start_and_manifest_end(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "dashboard_data"
    replay_csv = tmp_path / "runtime" / "model" / "card" / "smoke" / "smoke_soxl_daily.csv"
    actor_path = tmp_path / "runtime" / "model" / "card" / "smoke" / "soxl_smoke_actor.pt"
    replay_csv.parent.mkdir(parents=True)
    actor_path.write_bytes(b"actor")
    data_dir.mkdir()
    replay_csv.write_text(
        "\n".join(
            [
                "datetime,strategy_nav,price,buyhold_nav,position,excess,excess_pct,action",
                "2020-01-02,1000000,10,1000000,0,0,0,HOLD",
                "2026-06-10,1000000,20,2000000,0,0,0,HOLD",
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "model_replay_freshness.json").write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "profile": "smoke",
                        "output": "runtime/model/card/smoke/smoke_soxl_daily.csv",
                        "before_end": "2026-06-10",
                        "after_end": "2026-06-11",
                        "result": "current",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_builder, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard_builder, "DASHBOARD_DATA_DIR", data_dir)
    monkeypatch.setattr(dashboard_builder, "load_profile", lambda _profile: object())
    monkeypatch.setattr(dashboard_builder, "model_artifact_path", lambda _profile, _tag="": actor_path)

    windows = load_run_profile_windows()

    assert windows["smoke"] == {
        "start": "2020-01-02",
        "end": "2026-06-11",
        "rows": 2,
        "source": "runtime/model/card/smoke/smoke_soxl_daily.csv",
        "runnable": True,
        "model_path": "runtime/model/card/smoke/soxl_smoke_actor.pt",
    }


def test_load_run_profile_windows_marks_missing_actor_unrunnable(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "dashboard_data"
    replay_csv = tmp_path / "runtime" / "model" / "card" / "missing_actor" / "missing_actor_soxl_daily.csv"
    actor_path = tmp_path / "runtime" / "model" / "card" / "missing_actor" / "soxl_missing_actor_actor.pt"
    replay_csv.parent.mkdir(parents=True)
    data_dir.mkdir()
    replay_csv.write_text(
        "\n".join(
            [
                "datetime,strategy_nav,price,buyhold_nav,position,excess,excess_pct,action",
                "2026-06-10,1000000,20,1000000,0,0,0,HOLD",
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "model_replay_freshness.json").write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "profile": "missing_actor",
                        "output": "runtime/model/card/missing_actor/missing_actor_soxl_daily.csv",
                        "after_end": "2026-06-10",
                        "result": "current",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_builder, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard_builder, "DASHBOARD_DATA_DIR", data_dir)
    monkeypatch.setattr(dashboard_builder, "load_profile", lambda _profile: object())
    monkeypatch.setattr(dashboard_builder, "model_artifact_path", lambda _profile, _tag="": actor_path)

    windows = load_run_profile_windows()

    assert windows["missing_actor"]["runnable"] is False
    assert windows["missing_actor"]["runnable_reason"] == "missing_model_artifact"
    assert windows["missing_actor"]["model_path"] == (
        "runtime/model/card/missing_actor/soxl_missing_actor_actor.pt"
    )


def test_load_research_backtests_exports_recent_model_selection_cards(tmp_path: Path, monkeypatch):
    card_dir = tmp_path / "runtime" / "model_selection" / "card" / "stress_case"
    card_dir.mkdir(parents=True)
    (card_dir / "stress_case.ipynb").write_text("{}", encoding="utf-8")
    (card_dir / "selection.json").write_text(
        json.dumps(
            {
                "experiment": "stress_case",
                "decision": "near_miss_no_promotion",
                "promotion_allowed": False,
                "role": "diagnose",
                "changed_layer": ["model_selection", "stress_test"],
                "selected_candidate": "candidate_a",
                "created_at": "2026-06-01T03:02:41+00:00",
                "selected_base_score": {
                    "min_relative_return": 29.817,
                    "max_max_dd": 15.107,
                    "full_gate": False,
                    "near_miss": True,
                },
                "stress_pass_count": 0,
                "stress_scenario_count": 4,
            }
        ),
        encoding="utf-8",
    )
    (card_dir / "summary.json").write_text(
        json.dumps(
            {
                "conclusion": ["base full_gate=False", "stress pass 0/4"],
                "next_mechanism": "reward timing",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_builder, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard_builder, "MODEL_SELECTION_DIR", tmp_path / "runtime" / "model_selection" / "card")

    rows = load_research_backtests()

    assert rows[0]["experiment"] == "stress_case"
    assert rows[0]["decision"] == "near_miss_no_promotion"
    assert rows[0]["min_relative_return"] == pytest.approx(29.817)
    assert rows[0]["max_drawdown"] == pytest.approx(15.107)
    assert rows[0]["stress_pass_count"] == 0
    assert rows[0]["stress_scenario_count"] == 4
    assert rows[0]["notebook_path"] == "runtime/model_selection/card/stress_case/stress_case.ipynb"
    assert "base full_gate=False stress pass 0/4" in rows[0]["conclusion"]


def test_load_versions_excludes_duplicate_model_card_post_train_preview_tags(tmp_path: Path, monkeypatch):
    canonical = tmp_path / "runtime" / "model" / "card" / "smoke"
    stale_alias = tmp_path / "runtime" / "model" / "card" / "smoke_tide"
    canonical.mkdir(parents=True)
    stale_alias.mkdir(parents=True)
    (canonical / "smoke_soxl_daily.csv").write_text(
        "\n".join(
            [
                "datetime,strategy_nav,price,buyhold_nav,position,excess,excess_pct,action",
                "2026-05-20,1000000,100,1000000,0,0,0,HOLD",
                "2026-05-21,1000000,100,1000000,0,0,0,HOLD",
                "2026-05-22,1000000,100,1000000,0,0,0,HOLD",
                "2026-05-23,1000000,100,1000000,0,0,0,HOLD",
                "2026-05-24,1000000,100,1000000,0,0,0,HOLD",
                "2026-05-25,1000000,100,1000000,0,0,0,HOLD",
                "2026-05-26,1000000,100,1000000,0,0,0,HOLD",
                "2026-05-27,1000000,100,1000000,0,0,0,HOLD",
                "2026-05-28,1000000,100,1000000,0,0,0,HOLD",
                "2026-05-29,1000000,100,1000000,0,0,0,HOLD",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (stale_alias / "smoke_soxl_daily.csv").write_text(
        "\n".join(
            [
                "datetime,strategy_nav,price,buyhold_nav,position,excess,excess_pct,action",
                "2026-03-20,1000000,100,1000000,0,0,0,HOLD",
                "2026-03-21,1000000,100,1000000,0,0,0,HOLD",
                "2026-03-22,1000000,100,1000000,0,0,0,HOLD",
                "2026-03-23,1000000,100,1000000,0,0,0,HOLD",
                "2026-03-24,1000000,100,1000000,0,0,0,HOLD",
                "2026-03-25,1000000,100,1000000,0,0,0,HOLD",
                "2026-03-26,1000000,100,1000000,0,0,0,HOLD",
                "2026-03-27,1000000,100,1000000,0,0,0,HOLD",
                "2026-03-30,1000000,100,1000000,0,0,0,HOLD",
                "2026-03-31,1000000,100,1000000,0,0,0,HOLD",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_builder, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard_builder, "load_ohlc_and_indicators", lambda _etf: {})

    versions = _load_post_train_review_versions(features=None)

    assert "smoke" not in versions


def test_compact_payload_rows_preserves_market_ohlc_for_static_price_chart():
    rows = compact_payload_rows(
        [
            {
                "datetime": "2024-01-02",
                "strategy_nav": 1_000_000.0,
                "price": 10.0,
                "buyhold_nav": 1_000_000.0,
                "position": 0.4,
                "excess": 0.0,
                "excess_pct": 0.0,
                "action": "BUY",
                "open": 9.8,
                "high": 10.1,
                "low": 9.7,
                "close": 10.0,
                "volume": 12345,
                "ma180": 9.5,
                "market_state": True,
                "feature_snapshot": {"alpha_signal": 1.25},
                "_features": {"vix": 20.0},
            }
        ]
    )
    assert rows == [
        {
            "datetime": "2024-01-02",
            "strategy_nav": 1_000_000.0,
            "price": 10.0,
            "buyhold_nav": 1_000_000.0,
            "position": 0.4,
            "excess": 0.0,
            "excess_pct": 0.0,
            "action": "BUY",
            "open": 9.8,
            "high": 10.1,
            "low": 9.7,
            "close": 10.0,
            "volume": 12345,
            "ma180": 9.5,
            "market_state": True,
            "feature_snapshot": {"alpha_signal": 1.25},
        }
    ]


def test_build_html_removes_unreliable_training_curve_and_history_compare():
    html = build_html({"detailed": {}, "versioned": {}, "model_versions": []}, {"all": {}})

    assert "Training Curve" not in html
    assert "trainingCurve" not in html
    assert "renderTrainingCurve" not in html
    assert "Model Version Comparison" not in html
    assert "versionCompare" not in html
    assert "renderVersionTable" not in html


def test_build_html_loads_production_status_json():
    html = build_html({"detailed": {}, "versioned": {}, "model_versions": []}, {"all": {}})

    assert 'id="productionStatusCard"' in html
    assert "dashboard_data/production_status.json" in html
    assert "dashboard_data/production_goal_gates.json" in html
    assert "renderProductionStatus" in html
    assert 'id="productionGateOverall"' in html
    assert 'id="productionPaperModel"' in html
    assert 'id="productionPaperExecution"' in html
    assert 'id="productionPaperSource"' in html
    assert 'id="productionModelReplayStatus"' in html
    assert 'id="productionAutoRefreshStatus"' in html
    assert "renderPaperStatus" in html
    assert "renderModelReplayStatus" in html
    assert "renderAutoUpdateStatus" in html
    assert "dashboard_data/model_replay_freshness.json" in html
    assert "dashboard_data/v2/manifest.json" in html
    assert "embeddedPaperPayload" in html
    assert "chooseFreshestPaperPayload(paperPayload, embeddedPaperPayload())" in html
    assert "cache: 'no-store'" in html
    assert "paperMeta.latest_data_date" in html
    assert "paperMeta.latest_signal_date" in html
    assert ".metric .k {\\n    .metric .k" not in html


def test_build_html_run_backtest_uses_same_origin_api_and_dedicated_window_inputs():
    html = build_html({"detailed": {}, "versioned": {}, "model_versions": []}, {"all": {}})

    assert 'id="runStartDateInput"' in html
    assert 'id="runEndDateInput"' in html
    assert 'id="useChartWindowBtn"' in html
    assert 'aria-label="Inference start date"' in html
    assert 'aria-label="Inference end date"' in html
    assert "copyChartWindowToBacktestInputs" in html
    assert "const runProfileWindows = DATA.run_windows || {};" in html
    assert "const runWindow = runProfileWindows[profile];" in html
    assert "function isRunnableRunProfile(profile)" in html
    assert "if (!isRunnableRunProfile(name)) return;" in html
    assert "return { start: globalBounds.start || '', end: profileRecord.after_end || profileRecord.before_end };" not in html
    assert "runProfileSelect.addEventListener('change', () => syncBacktestInputs(false));" in html
    assert "runStartDateInput.dataset.manual" in html
    assert "runProfileStatus" in html
    assert "fetch('/api/run-backtest'" in html
    assert "http://localhost:8765/api/run-backtest" not in html


def test_build_html_surfaces_dashboard_server_status_for_live_inference():
    html = build_html({"detailed": {}, "versioned": {}, "model_versions": []}, {"all": {}})

    assert 'id="serverModeStatus"' in html
    assert "function updateDashboardServerStatus" in html
    assert "fetch('/api/health'" in html
    assert "Static view: chart/recompute only" in html
    assert "dashboard_server.py" in html
    assert "reference baselines are not runnable inference profiles" in html


def test_build_html_falls_back_to_dashboard_api_server_when_static_server_rejects_post():
    html = build_html({"detailed": {}, "versioned": {}, "model_versions": []}, {"all": {}})

    assert "const DASHBOARD_API_FALLBACK = 'http://127.0.0.1:8765/api/run-backtest'" in html
    assert "function shouldRetryOnDashboardApi" in html
    assert "response.status === 501" in html
    assert "fetch(DASHBOARD_API_FALLBACK" in html


def test_build_html_registers_backtest_result_as_selectable_version():
    html = build_html({"detailed": {}, "versioned": {}, "model_versions": []}, {"all": {}})

    assert "function registerBacktestResult" in html
    assert "function getBacktestResultWindow(data, fallbackStart, fallbackEnd)" in html
    assert "const resultWindow = getBacktestResultWindow(data, start, end);" in html
    assert "const resultVersion = `rerun:${profile}:${resultWindow.start}:${resultWindow.end}`;" in html
    assert "warmup_start_date: data.warmup_start_date || resultWindow.start" in html
    assert "replay_aligned: Boolean(data.replay_aligned)" in html
    assert "replay warmup" in html
    assert "DATA.versioned[resultVersion]" in html
    assert "state.version = resultVersion" in html
    assert "DATA.detailed[state.etf] =" not in html


def test_build_html_rerun_chart_draws_full_price_and_position_lines():
    html = build_html({"detailed": {}, "versioned": {}, "model_versions": []}, {"all": {}})

    price_rows_index = html.index("const priceRows = rows.filter(r => r.close != null || r.price != null);")
    ohlc_index = html.index("if (ohlcRows.length)")
    assert price_rows_index < ohlc_index
    assert "name: 'SOXL Close', type: 'scatter', mode: 'lines'" in html
    assert "name: 'Position %', type: 'scatter', mode: 'lines+markers'" in html


def test_build_html_uses_default_chart_version_and_model_feature_snapshots():
    html = build_html({"detailed": {}, "versioned": {}, "model_versions": []}, {"all": {}})

    assert "version: DATA.default_chart_version || DATA.latest_version || DATA.best_version || ''" in html
    assert "const detailRows = DATA.detailed && DATA.detailed[etf] && DATA.detailed[etf].rows;" in html
    assert "const f = row.feature_snapshot || row._features || {};" in html


def test_build_html_binds_plotly_interactions_once_per_chart_node():
    html = build_html({"detailed": {}, "versioned": {}, "model_versions": []}, {"all": {}})

    assert "function bindPlotlyChartInteractions(gd)" in html
    assert "gd.dataset.boundChartInteractions === '1'" in html
    assert "gd.dataset.boundChartInteractions = '1'" in html
    assert "gd.__dashboardRowsByDate = new Map" in html
    assert "gd.__dashboardRowsByDate.get(dateStr)" in html
    assert "bindPlotlyChartInteractions(gd)" in html
    assert "Plotly.react('plotlyChart', traces, layout" in html


def test_build_html_fetches_version_detail_when_only_index_stub_exists():
    html = build_html({"detailed": {}, "versioned": {}, "model_versions": []}, {"all": {}})

    assert "function hasVersionDetail(version, etf)" in html
    assert "if (state.version && !hasVersionDetail(state.version, state.etf))" in html
    assert "if (!version || hasVersionDetail(version, etf)) return;" in html
    assert "if (!hasVersionDetail(state.version, state.etf))" in html


def test_soxlbacktest_reference_version_exports_buy_sell_baseline():
    versioned = _load_soxlbacktest_reference_version(features=None)

    payload = versioned["soxl_backtest_policy_reimplementation_v1"]
    detail = payload["detailed"]["SOXL"]
    rows = detail["rows"]
    actions = {row["action"] for row in rows}

    assert payload["_meta"]["source_kind"] == "reference_baseline"
    assert payload["windows"]["SOXL"] == {"start": "2026-01-02", "end": "2026-06-09", "rows": 109}
    assert detail["summary"]["source_profile"] == "soxl_backtest_policy_reimplementation_v1"
    assert detail["summary"]["window"] == "ytd_2026"
    assert "BUY" in actions
    assert "SELL" in actions
    assert "TAKE_PROFIT" not in actions
    assert all(field in rows[0] for field in ("open", "high", "low", "close"))
