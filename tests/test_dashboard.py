from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.build_backtest_dashboard import (
    _load_post_train_review_versions,
    _load_retained_dashboard_versions,
    build_html,
    compact_payload_rows,
    normalize_post_train_review_rows,
    select_best_version,
    select_latest_version,
)
import scripts.build_backtest_dashboard as dashboard_builder
from src.soxl_quant.app.dashboard import review_daily_csv_path, write_review_daily_csv


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


def test_select_latest_version_prefers_post_train_review_over_older_snapshots():
    versioned = {
        "snapshot_newer": {"_meta": {"updated_at": 50, "source_kind": "generalization_version"}},
        "post_train": {"_meta": {"updated_at": 20, "source_kind": "post_train_review"}},
    }
    assert select_latest_version(versioned) == "post_train"


def test_select_latest_version_prefers_paper_trading_for_static_dashboard():
    versioned = {
        "snapshot_newer": {"_meta": {"updated_at": 300, "source_kind": "generalization_version"}},
        "post_train": {"_meta": {"updated_at": 200, "source_kind": "post_train_review"}},
        "paper_trading": {"_meta": {"updated_at": 100, "source_kind": "paper_trading"}},
    }
    assert select_latest_version(versioned) == "paper_trading"


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
                "reconcile": {"action_required": "none"},
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

    row = versions["paper_trading"]["detailed"]["SOXL"]["rows"][0]
    assert row["profile"] == "v19reg_v22_budget_usage_bull_guard_v1"
    assert row["execution_status"] == "duplicate_skipped"
    assert row["dry_run"] is False
    assert row["target_position"] == 0.484086
    assert row["reconcile_action"] == "none"
    assert row["summary_path"] == (
        "runtime/soxl/paper_trading/2026-05-20/duplicate_skip/"
        "v19reg_v22_budget_usage_bull_guard_v1/summary.json"
    )


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

    versions = _load_retained_dashboard_versions()

    assert "retained_ablation" in versions
    retained = versions["retained_ablation"]
    assert retained["_meta"]["source_kind"] == "retained_dashboard_export"
    assert retained["windows"]["SOXL"] == {"start": "2026-05-19", "end": "2026-05-19", "rows": 1}


def test_compact_payload_rows_drops_redundant_market_metadata():
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
        }
    ]


def test_build_html_loads_production_status_json():
    html = build_html({"detailed": {}, "versioned": {}, "model_versions": []}, {"all": {}})

    assert 'id="productionStatusCard"' in html
    assert "dashboard_data/production_status.json" in html
    assert "dashboard_data/production_goal_gates.json" in html
    assert "renderProductionStatus" in html
    assert 'id="productionGateOverall"' in html
