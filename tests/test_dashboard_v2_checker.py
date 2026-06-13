from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_dashboard_v2_data import check_dashboard_v2_data


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_check_dashboard_v2_data_passes_valid_contract(tmp_path: Path):
    root = tmp_path / "dashboard_data" / "v2"
    _write_json(
        root / "manifest.json",
        {
            "safe_backtest_end": "2026-01-06",
            "default_baseline_profile": "smoke",
            "default_comparison_profiles": ["smoke"],
            "available_profiles": ["smoke"],
        },
    )
    _write_json(
        root / "market" / "SOXL.json",
        {
            "symbol": "SOXL",
            "rows": [
                {
                    "date": "2026-01-06",
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10.5,
                    "volume": 1000,
                }
            ],
        },
    )
    _write_json(
        root / "models" / "smoke.json",
        {
            "profile": "smoke",
            "rows": [
                {
                    "date": "2026-01-06",
                    "strategy_nav": 100,
                    "buyhold_nav": 100,
                    "position": 0.5,
                    "action": "BUY",
                    "price": 10.5,
                }
            ],
            "events": [{"date": "2026-01-06", "kind": "BUY"}],
        },
    )
    _write_json(
        root / "comparison" / "default.json",
        {
            "baseline_profile": "smoke",
            "compared_profiles": ["smoke"],
            "common_start": "2026-01-06",
            "common_end": "2026-01-06",
            "rows": [
                {
                    "date": "2026-01-06",
                    "profiles": {"smoke": {"action": "BUY"}},
                    "action_mismatch_count": 0,
                }
            ],
        },
    )

    result = check_dashboard_v2_data(root)

    assert result == {"profiles": 1, "market_rows": 1, "comparison_rows": 1}


def test_check_dashboard_v2_data_fails_missing_model_position(tmp_path: Path):
    root = tmp_path / "dashboard_data" / "v2"
    _write_json(
        root / "manifest.json",
        {"default_comparison_profiles": ["smoke"], "available_profiles": ["smoke"]},
    )
    _write_json(
        root / "market" / "SOXL.json",
        {"rows": [{"date": "2026-01-06", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000}]},
    )
    _write_json(
        root / "models" / "smoke.json",
        {
            "profile": "smoke",
            "rows": [{"date": "2026-01-06", "strategy_nav": 100, "buyhold_nav": 100, "action": "BUY", "price": 10.5}],
            "events": [],
        },
    )
    _write_json(root / "comparison" / "default.json", {"rows": []})

    with pytest.raises(ValueError, match="position"):
        check_dashboard_v2_data(root)


def test_check_dashboard_v2_data_fails_when_comparison_references_unknown_profile(tmp_path: Path):
    root = tmp_path / "dashboard_data" / "v2"
    _write_json(
        root / "manifest.json",
        {"default_comparison_profiles": ["smoke"], "available_profiles": ["smoke"]},
    )
    _write_json(
        root / "market" / "SOXL.json",
        {"rows": [{"date": "2026-01-06", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000}]},
    )
    _write_json(
        root / "models" / "smoke.json",
        {
            "profile": "smoke",
            "rows": [{"date": "2026-01-06", "strategy_nav": 100, "buyhold_nav": 100, "position": 0.5, "action": "BUY", "price": 10.5}],
            "events": [],
        },
    )
    _write_json(
        root / "comparison" / "default.json",
        {
            "baseline_profile": "smoke",
            "compared_profiles": ["smoke"],
            "rows": [
                {
                    "date": "2026-01-06",
                    "profiles": {"missing": {"action": "SELL"}},
                    "action_mismatch_count": 1,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="unknown profile"):
        check_dashboard_v2_data(root)
