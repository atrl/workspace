from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_dashboard_v2_data import check_dashboard_v2_data


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _model_contract(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "profile": "smoke",
        "source_kind": "qlib_model",
        "display_name": "smoke",
        "runnable": True,
        "feature_columns": [],
        "feature_schema_signature": "empty",
        "snapshot_columns": [],
        "feature_action_summary": [],
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
    }
    payload.update(overrides)
    return payload


def test_check_dashboard_v2_data_passes_valid_contract(tmp_path: Path):
    root = tmp_path / "dashboard_data" / "v2"
    _write_json(
        root / "manifest.json",
        {
            "safe_backtest_end": "2026-01-06",
            "default_baseline_profile": "smoke",
            "default_comparison_profiles": ["smoke"],
            "available_profiles": ["smoke"],
            "unavailable_profiles": [
                {
                    "profile": "missing_actor_model",
                    "result": "skipped",
                    "reason": "missing_actor_artifact",
                }
            ],
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
        _model_contract(),
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


def test_check_dashboard_v2_data_fails_malformed_unavailable_profile(tmp_path: Path):
    root = tmp_path / "dashboard_data" / "v2"
    _write_json(
        root / "manifest.json",
        {
            "safe_backtest_end": "2026-01-06",
            "default_baseline_profile": "smoke",
            "default_comparison_profiles": ["smoke"],
            "available_profiles": ["smoke"],
            "unavailable_profiles": [{"profile": "missing_actor_model", "result": "skipped"}],
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
    _write_json(root / "models" / "smoke.json", _model_contract())
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

    with pytest.raises(ValueError, match="unavailable profile"):
        check_dashboard_v2_data(root)


def test_check_dashboard_v2_data_fails_malformed_model_runtime_status(tmp_path: Path):
    root = tmp_path / "dashboard_data" / "v2"
    _write_json(
        root / "manifest.json",
        {
            "safe_backtest_end": "2026-01-06",
            "default_baseline_profile": "smoke",
            "default_comparison_profiles": ["smoke"],
            "available_profiles": ["smoke"],
            "model_runtime_status": [
                {
                    "profile": "smoke",
                    "role": "maybe",
                    "runnable": False,
                    "source_kind": "qlib_model",
                    "feature_count": 0,
                    "snapshot_count": 0,
                    "row_count": 1,
                }
            ],
        },
    )
    _write_json(
        root / "market" / "SOXL.json",
        {"rows": [{"date": "2026-01-06", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000}]},
    )
    _write_json(root / "models" / "smoke.json", _model_contract())
    _write_json(
        root / "comparison" / "default.json",
        {
            "baseline_profile": "smoke",
            "compared_profiles": ["smoke"],
            "rows": [
                {
                    "date": "2026-01-06",
                    "profiles": {"smoke": {"action": "BUY"}},
                    "action_mismatch_count": 0,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="model_runtime_status"):
        check_dashboard_v2_data(root)


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
        _model_contract(
            rows=[{"date": "2026-01-06", "strategy_nav": 100, "buyhold_nav": 100, "action": "BUY", "price": 10.5}],
            events=[],
        ),
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
        _model_contract(events=[]),
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


def test_check_dashboard_v2_data_fails_post_train_preview_model(tmp_path: Path):
    root = tmp_path / "dashboard_data" / "v2"
    _write_json(
        root / "manifest.json",
        {
            "default_baseline_profile": "bad_preview",
            "default_comparison_profiles": ["bad_preview"],
            "available_profiles": ["bad_preview"],
        },
    )
    _write_json(
        root / "market" / "SOXL.json",
        {"rows": [{"date": "2026-01-06", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000}]},
    )
    _write_json(
        root / "models" / "bad_preview.json",
        _model_contract(profile="bad_preview", source_kind="post_train_review"),
    )
    _write_json(
        root / "comparison" / "default.json",
        {
            "baseline_profile": "bad_preview",
            "compared_profiles": ["bad_preview"],
            "rows": [
                {
                    "date": "2026-01-06",
                    "profiles": {"bad_preview": {"action": "BUY"}},
                    "action_mismatch_count": 0,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="post-train preview"):
        check_dashboard_v2_data(root)


def test_check_dashboard_v2_data_fails_feature_snapshot_schema_mismatch(tmp_path: Path):
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
        {"rows": [{"date": "2026-01-06", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000}]},
    )
    _write_json(
        root / "models" / "smoke.json",
        _model_contract(
            feature_columns=["alpha_signal"],
            snapshot_columns=["alpha_signal"],
            feature_action_summary=[
                {
                    "feature": "beta_signal",
                    "buy_count": 1,
                    "sell_count": 0,
                    "buy_mean": 2.0,
                    "sell_mean": None,
                    "buy_minus_sell": None,
                }
            ],
            rows=[
                {
                    "date": "2026-01-06",
                    "strategy_nav": 100,
                    "buyhold_nav": 100,
                    "position": 0.5,
                    "action": "BUY",
                    "price": 10.5,
                    "feature_snapshot": {"beta_signal": 2.0},
                }
            ],
            events=[
                {
                    "date": "2026-01-06",
                    "kind": "BUY",
                    "feature_snapshot": {"beta_signal": 2.0},
                }
            ],
        ),
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

    with pytest.raises(ValueError, match="feature.*not declared"):
        check_dashboard_v2_data(root)
