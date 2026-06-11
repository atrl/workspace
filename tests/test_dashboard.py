from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_backtest_dashboard import (
    _load_post_train_review_versions,
    _load_retained_dashboard_versions,
    build_html,
    compact_payload_rows,
    load_research_backtests,
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

    detail = versions["paper_trading"]["detailed"]["SOXL"]
    row = detail["rows"][0]
    meta = detail["meta"]
    assert row["profile"] == "v19reg_v22_budget_usage_bull_guard_v1"
    assert row["execution_status"] == "duplicate_skipped"
    assert row["dry_run"] is False
    assert row["target_position"] == 0.484086
    assert row["reconcile_action"] == "none"
    assert row["summary_path"] == (
        "runtime/soxl/paper_trading/2026-05-20/duplicate_skip/"
        "v19reg_v22_budget_usage_bull_guard_v1/summary.json"
    )
    assert meta["source_kind"] == "paper_trading"
    assert meta["latest_data_date"] == "2026-05-20"
    assert meta["latest_profile"] == "v19reg_v22_budget_usage_bull_guard_v1"
    assert meta["latest_execution_status"] == "duplicate_skipped"
    assert meta["latest_target_position"] == 0.484086
    assert meta["latest_signal_profile"] == "v19reg_v22_budget_usage_bull_guard_v1"
    assert meta["latest_signal_execution_status"] == "duplicate_skipped"
    assert meta["latest_signal_summary_path"] == (
        "runtime/soxl/paper_trading/2026-05-20/duplicate_skip/"
        "v19reg_v22_budget_usage_bull_guard_v1/summary.json"
    )


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
    assert meta["latest_signal_risk_gate_approved"] is True
    assert meta["latest_signal_slow_profile"] == "balanced_rule_residual_v19_reg"
    assert meta["latest_signal_fast_profile"] == "balanced_rule_residual_v22"
    assert meta["latest_signal_summary_path"] == (
        "runtime/soxl/paper_trading/2026-05-22/dry_run/futu/summary.json"
    )


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
    assert rows[0]["buyhold_nav"] == pytest.approx(rows[0]["strategy_nav"])
    assert rows[1]["buyhold_nav"] == pytest.approx(
        3167397.210598 * (999664.857996 / 3886245.093747)
    )
    assert rows[0]["excess"] == pytest.approx(0.0, abs=1e-6)


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


def test_load_versions_prefers_matching_profile_directory_for_duplicate_tag(tmp_path: Path, monkeypatch):
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

    assert versions["smoke"]["windows"]["SOXL"]["end"] == "2026-05-29"
    assert versions["smoke"]["_meta"]["source"] == "runtime/model/card/smoke/smoke_soxl_daily.csv"


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
    assert "renderPaperStatus" in html
    assert "renderModelReplayStatus" in html
    assert "dashboard_data/model_replay_freshness.json" in html
    assert "embeddedPaperPayload" in html
    assert "chooseFreshestPaperPayload(paperPayload, embeddedPaperPayload())" in html
    assert "cache: 'no-store'" in html
    assert "paperMeta.latest_data_date" in html
    assert "paperMeta.latest_signal_date" in html
    assert ".metric .k {\\n    .metric .k" not in html
