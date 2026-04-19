from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_backtest_dashboard import select_best_version, select_latest_version
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
