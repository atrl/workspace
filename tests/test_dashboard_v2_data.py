from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.soxl_quant.app.dashboard_v2 import (
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
    assert manifest["profile_summaries"]["smoke"]["status"] == "canonical"
    assert manifest["profile_summaries"]["smoke"]["replay_end"] == "2026-01-06"
    assert manifest["profile_summaries"]["smoke"]["row_count"] == 3
    assert market["rows"][1]["vix"] == 18.0
    assert model["rows"][1]["action"] == "BUY"
    assert model["events"][0]["kind"] == "BUY"
    assert comparison["common_start"] == "2026-01-02"
    assert comparison["rows"][1]["action_mismatch_count"] == 0


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
    assert reference["source_path"] == "runtime/production/card/soxl_backtest_policy_reimplementation_v1/new_policy_daily.csv"
    assert reference["rows"][-1]["action"] == "SELL"
    assert [event["kind"] for event in reference["events"] if event["kind"] in {"BUY", "SELL"}] == ["BUY", "SELL"]
    assert comparison["baseline_profile"] == reference_name
    assert comparison["common_start"] == "2026-01-02"
    assert comparison["common_end"] == "2026-05-27"


def test_build_dashboard_v2_data_prefers_retained_actor_before_smoke_default(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    for profile in ("smoke", "balanced_rule_residual_v22"):
        model_csv = tmp_path / "runtime" / "model" / "card" / profile / f"{profile}_soxl_daily.csv"
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
                    "datetime": "2026-01-05",
                    "strategy_nav": 101.0,
                    "buyhold_nav": 102.0,
                    "position": 0.2,
                    "action": "BUY",
                    "price": 49.0,
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
                        "output": "runtime/model/card/smoke/smoke_soxl_daily.csv",
                        "result": "current",
                    },
                    {
                        "profile": "balanced_rule_residual_v22",
                        "status": "canonical",
                        "output": (
                            "runtime/model/card/balanced_rule_residual_v22/"
                            "balanced_rule_residual_v22_soxl_daily.csv"
                        ),
                        "result": "current",
                    },
                ],
            }
        )
    )
    market_rows = {
        "2026-01-02": {"open": 47.0, "high": 48.0, "low": 46.0, "close": 47.24, "volume": 1000},
        "2026-01-05": {"open": 48.0, "high": 50.0, "low": 47.0, "close": 49.0, "volume": 1000},
    }

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        market_rows=market_rows,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["default_comparison_profiles"] == ["balanced_rule_residual_v22", "smoke"]


def test_build_dashboard_v2_data_omits_smoke_when_reference_and_retained_actor_exist(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    for profile in ("smoke", "balanced_rule_residual_v22"):
        model_csv = tmp_path / "runtime" / "model" / "card" / profile / f"{profile}_soxl_daily.csv"
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
                    "datetime": "2026-01-05",
                    "strategy_nav": 101.0,
                    "buyhold_nav": 102.0,
                    "position": 0.2,
                    "action": "BUY",
                    "price": 49.0,
                },
            ],
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
                "window": "ytd_2026",
                "date": "2026-01-02",
                "strategy_nav": 100.0,
                "buyhold_nav": 100.0,
                "position": 0.0,
                "action": "HOLD",
                "price": 47.24,
            },
            {
                "window": "ytd_2026",
                "date": "2026-01-05",
                "strategy_nav": 105.0,
                "buyhold_nav": 102.0,
                "position": 0.2,
                "action": "BUY",
                "price": 49.0,
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
                        "output": "runtime/model/card/smoke/smoke_soxl_daily.csv",
                        "result": "current",
                    },
                    {
                        "profile": "balanced_rule_residual_v22",
                        "status": "canonical",
                        "output": (
                            "runtime/model/card/balanced_rule_residual_v22/"
                            "balanced_rule_residual_v22_soxl_daily.csv"
                        ),
                        "result": "current",
                    },
                ],
            }
        )
    )
    market_rows = {
        "2026-01-02": {"open": 47.0, "high": 48.0, "low": 46.0, "close": 47.24, "volume": 1000},
        "2026-01-05": {"open": 48.0, "high": 50.0, "low": 47.0, "close": 49.0, "volume": 1000},
    }

    output_dir = tmp_path / "dashboard_data" / "v2"
    stale_smoke = output_dir / "models" / "smoke.json"
    stale_smoke.parent.mkdir(parents=True)
    stale_smoke.write_text('{"profile":"smoke","rows":[]}', encoding="utf-8")
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        market_rows=market_rows,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["default_comparison_profiles"] == [
        "soxl_backtest_policy_reimplementation_v1",
        "balanced_rule_residual_v22",
    ]
    assert manifest["available_profiles"] == [
        "soxl_backtest_policy_reimplementation_v1",
        "balanced_rule_residual_v22",
    ]
    assert not stale_smoke.exists()


def test_build_dashboard_v2_data_manifest_does_not_expose_unselected_old_profiles(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    for profile in ("smoke", "balanced_rule_residual_v22", "old_candidate"):
        model_csv = tmp_path / "runtime" / "model" / "card" / profile / f"{profile}_soxl_daily.csv"
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
                    "datetime": "2026-01-05",
                    "strategy_nav": 101.0,
                    "buyhold_nav": 102.0,
                    "position": 0.2,
                    "action": "BUY",
                    "price": 49.0,
                },
            ],
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
                "window": "ytd_2026",
                "date": "2026-01-02",
                "strategy_nav": 100.0,
                "buyhold_nav": 100.0,
                "position": 0.0,
                "action": "HOLD",
                "price": 47.24,
            },
            {
                "window": "ytd_2026",
                "date": "2026-01-05",
                "strategy_nav": 105.0,
                "buyhold_nav": 102.0,
                "position": 0.2,
                "action": "BUY",
                "price": 49.0,
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
                        "output": "runtime/model/card/smoke/smoke_soxl_daily.csv",
                        "result": "current",
                    },
                    {
                        "profile": "balanced_rule_residual_v22",
                        "status": "canonical",
                        "output": (
                            "runtime/model/card/balanced_rule_residual_v22/"
                            "balanced_rule_residual_v22_soxl_daily.csv"
                        ),
                        "result": "current",
                    },
                    {
                        "profile": "old_candidate",
                        "status": "archived",
                        "output": "runtime/model/card/old_candidate/old_candidate_soxl_daily.csv",
                        "result": "current",
                    },
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
            "2026-01-02": {"open": 47.0, "high": 48.0, "low": 46.0, "close": 47.24, "volume": 1000},
            "2026-01-05": {"open": 48.0, "high": 50.0, "low": 47.0, "close": 49.0, "volume": 1000},
        },
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["default_comparison_profiles"] == [
        "soxl_backtest_policy_reimplementation_v1",
        "balanced_rule_residual_v22",
    ]
    assert manifest["available_profiles"] == manifest["default_comparison_profiles"]
    assert manifest["filtered_profile_count"] == 2


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
    assert payload["rows"][1]["profiles"]["challenger"]["action_mismatch"] is True
    assert payload["rows"][1]["action_mismatch_count"] == 1
