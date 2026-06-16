from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.soxl_quant.app.dashboard_v2 import (
    build_comparison_payload,
    build_dashboard_v2_data,
    derive_model_events,
    load_ohlc_and_indicators_for_root,
    normalize_model_rows,
)
import src.soxl_quant.app.dashboard_v2 as dashboard_v2_module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_final_retention_summary(
    root: Path,
    profile: str,
    *,
    retained: bool = True,
    reason: str = "unit-test final retention fixture",
) -> Path:
    summary_path = root / "runtime" / "model" / "card" / profile / f"soxl_{profile}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "profile": profile,
                "final_model_retention_decision": {
                    "status": "passed" if retained else "blocked",
                    "final_model_retained": retained,
                    "reason": reason,
                },
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def _write_strict_retention_gate_summary(root: Path, *, passing: int = 0) -> Path:
    summary_path = (
        root
        / "runtime"
        / "model_selection"
        / "card"
        / "model_retention_walkforward_gate_v1"
        / "summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "summary": {
                    "profiles_scanned": 12,
                    "passing_profile_count": passing,
                    "passing_profiles": ["passer"] if passing else [],
                    "reference_profile": "soxl_backtest_policy_reimplementation_v1",
                    "reference_return_pct": 132.077978,
                    "current_best_return_pct": 224.361539,
                    "best_return_ratio": 0.95,
                    "required_best_return_pct": 213.143462,
                    "max_walkforward_drawdown_pct": 20.0,
                    "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
                    "closest_wf_drawdown_candidate": {
                        "profile": "mode_conditioned_rebound_reward_train_v2",
                        "worst_wf_max_drawdown_pct": 21.83172,
                        "wf_drawdown_gap_pct": 1.83172,
                        "missing_wf_window_count": 0,
                        "ytd_return_pct": 177.573569,
                        "best_return_gap_pct": 35.569893,
                    },
                    "closest_ytd_return_candidate": {
                        "profile": "smoke",
                        "ytd_return_pct": 224.361539,
                        "best_return_gap_pct": 0.0,
                        "missing_wf_window_count": 3,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def _write_strict_candidate_retention_filter_summary(root: Path) -> Path:
    summary_path = (
        root
        / "runtime"
        / "model_selection"
        / "card"
        / "strict_candidate_retention_filter_v1"
        / "summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "experiment": "strict_candidate_retention_filter_v1",
                "row_count": 15,
                "candidate_type_counts": {
                    "checkpoint_selector": 1,
                    "profile": 12,
                    "qlib_reference_screen": 1,
                    "reference_baseline": 1,
                },
                "summary": {
                    "decision": "strict_candidate_retention_filter_no_promotable_candidates",
                    "passing_profile_count": 0,
                    "promotable_profile_count": 0,
                    "actor_retention_allowed_count": 0,
                    "reference_visible_count": 2,
                    "ppo_allowed_next": False,
                    "next_mechanism": "train_bull_trend_participation_with_drawdown_drift_stop_no_promotion",
                    "reference_baseline": "soxl_backtest_policy_reimplementation_v1",
                },
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def _write_final_walkforward_retention_contract_summary(root: Path) -> Path:
    summary_path = (
        root
        / "runtime"
        / "model_selection"
        / "card"
        / "final_walkforward_retention_contract_v1"
        / "summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "summary": {
                    "decision": "final_walkforward_retention_contract_no_retained_models",
                    "reference_baseline": "soxl_backtest_policy_reimplementation_v1",
                    "reference_return_pct": 146.043798,
                    "latest_reference_return_pct": 146.043798,
                    "reference_is_current": True,
                    "strict_passing_profile_count": 0,
                    "candidate_filter_actor_retention_allowed_count": 0,
                    "final_retained_count": 0,
                    "final_retained_profiles": [],
                    "global_blockers": [],
                    "current_best_return_pct": 224.361539,
                    "required_best_return_pct": 213.143462,
                    "max_walkforward_drawdown_pct": 20.0,
                    "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
                }
            }
        ),
        encoding="utf-8",
    )
    return summary_path


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
    _write_final_retention_summary(tmp_path, "smoke")
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
    assert manifest["baseline_profile"] == "smoke"
    assert manifest["models"] == ["smoke"]
    assert manifest["comparison_profiles"] == ["smoke"]
    assert manifest["profile_summaries"]["smoke"]["status"] == "canonical"
    assert manifest["profile_summaries"]["smoke"]["replay_end"] == "2026-01-06"
    assert manifest["profile_summaries"]["smoke"]["row_count"] == 3
    assert market["rows"][1]["vix"] == 18.0
    assert model["rows"][1]["action"] == "BUY"
    assert model["events"][0]["kind"] == "BUY"
    assert comparison["common_start"] == "2026-01-02"
    assert comparison["rows"][1]["action_mismatch_count"] == 0


def test_build_dashboard_v2_data_exposes_strict_retention_gate_summary(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T00:00:00+00:00",
                "latest_data_date": "2026-01-05",
                "safe_backtest_end": "2026-01-05",
                "profiles": [],
            }
        )
    )
    _write_strict_retention_gate_summary(tmp_path, passing=0)
    _write_strict_candidate_retention_filter_summary(tmp_path)
    _write_final_walkforward_retention_contract_summary(tmp_path)
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
                "strategy_nav": 101.0,
                "buyhold_nav": 102.0,
                "position": 0.1,
                "action": "BUY",
                "price": 49.0,
            },
        ],
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
    assert manifest["strict_retention_gate"]["status"] == "blocked"
    assert manifest["strict_retention_gate"]["passing_profile_count"] == 0
    assert manifest["strict_retention_gate"]["profiles_scanned"] == 12
    assert manifest["strict_retention_gate"]["max_walkforward_drawdown_pct"] == 20.0
    assert manifest["strict_retention_gate"]["required_best_return_pct"] == 213.143462
    assert manifest["strict_retention_gate"]["closest_wf_drawdown_candidate"]["profile"] == "mode_conditioned_rebound_reward_train_v2"
    assert manifest["strict_candidate_retention_filter"]["status"] == "blocked"
    assert manifest["strict_candidate_retention_filter"]["decision"] == "strict_candidate_retention_filter_no_promotable_candidates"
    assert manifest["strict_candidate_retention_filter"]["promotable_profile_count"] == 0
    assert manifest["strict_candidate_retention_filter"]["actor_retention_allowed_count"] == 0
    assert (
        manifest["strict_candidate_retention_filter"]["next_mechanism"]
        == "train_bull_trend_participation_with_drawdown_drift_stop_no_promotion"
    )
    assert manifest["final_walkforward_retention_contract"]["status"] == "blocked"
    assert manifest["final_walkforward_retention_contract"]["decision"] == (
        "final_walkforward_retention_contract_no_retained_models"
    )
    assert manifest["final_walkforward_retention_contract"]["final_retained_count"] == 0
    assert manifest["final_walkforward_retention_contract"]["reference_is_current"] is True
    assert manifest["final_walkforward_retention_contract"]["latest_reference_return_pct"] == 146.043798


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
    _write_final_retention_summary(tmp_path, "smoke")
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


def test_load_ohlc_and_indicators_extends_price_rows_from_raw_symbol_when_vix_lags(tmp_path: Path):
    processed = tmp_path / "data" / "processed" / "panels" / "soxl_soxx_vix_full.csv"
    processed.parent.mkdir(parents=True)
    _write_csv(
        processed,
        [
            {
                "date": "2026-06-12",
                "soxl_open": 222.22,
                "soxl_high": 242.40,
                "soxl_low": 214.7406,
                "soxl_close": 234.68,
                "soxl_volume": 48_833_409,
                "soxx_open": 584.64,
                "soxx_high": 602.69,
                "soxx_low": 578.53,
                "soxx_close": 596.25,
                "soxx_volume": 9_057_945,
                "vix": 17.68,
            }
        ],
    )
    raw = tmp_path / "data" / "raw" / "symbols" / "SOXL.csv"
    raw.parent.mkdir(parents=True)
    _write_csv(
        raw,
        [
            {
                "date": "2026-06-12",
                "open": 222.22,
                "high": 242.40,
                "low": 214.7406,
                "close": 234.68,
                "volume": 48_833_409,
            },
            {
                "date": "2026-06-15",
                "open": 265.91,
                "high": 272.18,
                "low": 261.6019,
                "close": 271.44,
                "volume": 24_818_910,
            },
        ],
    )

    rows = load_ohlc_and_indicators_for_root(tmp_path, "SOXL")

    assert rows["2026-06-15"]["close"] == 271.44
    assert rows["2026-06-15"]["volume"] == 24_818_910
    assert rows["2026-06-15"]["vix"] == 17.68


def test_build_dashboard_v2_data_manifest_uses_extended_price_latest_date(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    model_csv = tmp_path / "runtime" / "model" / "card" / "smoke" / "smoke_soxl_daily.csv"
    model_csv.parent.mkdir(parents=True)
    _write_csv(
        model_csv,
        [
            {
                "datetime": "2026-06-12",
                "strategy_nav": 100.0,
                "buyhold_nav": 100.0,
                "position": 0.0,
                "action": "HOLD",
                "price": 234.68,
            }
        ],
    )
    _write_final_retention_summary(tmp_path, "smoke")
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-15T00:00:00+00:00",
                "latest_data_date": "2026-06-12",
                "safe_backtest_end": "2026-06-12",
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

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        selected_profiles=["smoke"],
        baseline_profile="smoke",
        market_rows={
            "2026-06-12": {"open": 222.22, "high": 242.40, "low": 214.7406, "close": 234.68, "volume": 48_833_409},
            "2026-06-15": {"open": 265.91, "high": 272.18, "low": 261.6019, "close": 271.44, "volume": 24_818_910},
        },
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    market = json.loads((output_dir / "market" / "SOXL.json").read_text())
    assert manifest["latest_market_date"] == "2026-06-15"
    assert manifest["canonical_data_date"] == "2026-06-12"
    assert manifest["safe_backtest_end"] == "2026-06-12"
    assert market["rows"][-1]["date"] == "2026-06-15"


def test_build_dashboard_v2_data_marks_latest_actor_signals_as_comparison_included(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    actor_csv = tmp_path / "runtime" / "model" / "card" / "smoke" / "smoke_soxl_daily.csv"
    actor_csv.parent.mkdir(parents=True)
    _write_csv(
        actor_csv,
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
                "datetime": "2026-01-06",
                "strategy_nav": 110.0,
                "buyhold_nav": 108.0,
                "position": 0.2,
                "action": "HOLD",
                "price": 50.0,
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
                "date": "2026-01-06",
                "strategy_nav": 103.0,
                "buyhold_nav": 108.0,
                "position": 0.1,
                "action": "HOLD",
                "price": 50.0,
            },
            {
                "window": "ytd_2026",
                "date": "2026-01-07",
                "strategy_nav": 104.0,
                "buyhold_nav": 109.0,
                "position": 0.1,
                "action": "HOLD",
                "price": 51.0,
            },
        ],
    )
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-01-07T00:00:00+00:00",
                "latest_data_date": "2026-01-07",
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
    _write_final_retention_summary(tmp_path, "smoke", retained=False)
    monkeypatch.setattr(dashboard_v2_module, "_latest_signal_context", lambda _root, _date: {"ok": True})
    monkeypatch.setattr(
        dashboard_v2_module,
        "_infer_latest_signal",
        lambda _root, profile_name, _source, rows, latest_date, _context: {
            "date": latest_date,
            "price": 51.0,
            "position": 0.55,
            "action": "BUY",
            "replay_asof": rows[-1]["date"],
            "source": "latest_signal_stateless",
            "raw_action": 3,
        }
        if profile_name == "smoke"
        else None,
    )

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        market_rows={
            "2026-01-02": {"open": 47.0, "high": 48.0, "low": 46.0, "close": 47.24, "volume": 1000},
            "2026-01-06": {"open": 49.0, "high": 51.0, "low": 48.0, "close": 50.0, "volume": 1000},
            "2026-01-07": {"open": 50.0, "high": 52.0, "low": 49.0, "close": 51.0, "volume": 1000},
        },
        selected_profiles=["soxl_backtest_policy_reimplementation_v1", "smoke"],
        baseline_profile="soxl_backtest_policy_reimplementation_v1",
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    actor = json.loads((output_dir / "models" / "smoke.json").read_text())
    comparison = json.loads((output_dir / "comparison" / "default.json").read_text())
    assert actor["latest_signal"]["date"] == "2026-01-07"
    assert manifest["profile_summaries"]["smoke"]["latest_signal_date"] == "2026-01-07"
    assert manifest["latest_signal"] == {
        "source": "latest_signal_stateless",
        "available_count": 1,
        "date": "2026-01-07",
        "comparison_included": True,
    }
    assert comparison["latest_signal_differences"] == [
        {
            "date": "2026-01-07",
            "profile": "smoke",
            "baseline_profile": "soxl_backtest_policy_reimplementation_v1",
            "model_action": "BUY",
            "baseline_action": "HOLD",
            "model_position": 0.55,
            "baseline_position": 0.1,
            "position_delta_vs_baseline": pytest.approx(0.45),
            "replay_asof": "2026-01-06",
            "source": "latest_signal_stateless",
        }
    ]


def test_build_dashboard_v2_data_exposes_latest_baseline_and_actor_safe_replay_dates(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    actor_csv = tmp_path / "runtime" / "model" / "card" / "smoke" / "smoke_soxl_daily.csv"
    actor_csv.parent.mkdir(parents=True)
    _write_csv(
        actor_csv,
        [
            {
                "datetime": "2026-06-12",
                "strategy_nav": 100.0,
                "buyhold_nav": 100.0,
                "position": 0.0,
                "action": "HOLD",
                "price": 234.68,
            }
        ],
    )
    _write_final_retention_summary(tmp_path, "smoke", retained=False)
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
                "date": "2026-06-12",
                "strategy_nav": 100000.0,
                "buyhold_nav": 100000.0,
                "position": 0.1591,
                "action": "HOLD",
                "price": 234.68,
            },
            {
                "window": "ytd_2026",
                "date": "2026-06-15",
                "strategy_nav": 246043.79795135497,
                "buyhold_nav": 120000.0,
                "position": 0.1591,
                "action": "HOLD",
                "price": 271.86,
            },
        ],
    )
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-15T00:00:00+00:00",
                "latest_data_date": "2026-06-15",
                "safe_backtest_end": "2026-06-12",
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

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        market_rows={
            "2026-06-12": {"open": 222.22, "high": 242.40, "low": 214.7406, "close": 234.68, "volume": 50_419_140},
            "2026-06-15": {"open": 265.91, "high": 272.66, "low": 261.6019, "close": 271.86, "volume": 29_256_770},
        },
        selected_profiles=["soxl_backtest_policy_reimplementation_v1", "smoke"],
        baseline_profile="soxl_backtest_policy_reimplementation_v1",
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    comparison = json.loads((output_dir / "comparison" / "default.json").read_text())
    assert manifest["profile_labels"]["soxl_backtest_policy_reimplementation_v1"] == "SOXLBacktest"
    assert manifest["profile_source_kinds"]["soxl_backtest_policy_reimplementation_v1"] == "reference_baseline"
    assert manifest["profile_source_kinds"]["smoke"] == "diagnostic_actor"
    assert manifest["model_replay_dates"]["soxl_backtest_policy_reimplementation_v1"]["replay_end"] == "2026-06-15"
    assert manifest["model_replay_dates"]["smoke"]["replay_end"] == "2026-06-12"
    assert manifest["baseline_latest"]["date"] == "2026-06-15"
    assert manifest["baseline_latest"]["return_pct"] == pytest.approx(146.04379795135497)
    assert manifest["baseline_latest"]["position"] == pytest.approx(0.1591)
    assert manifest["comparison_window"] == {
        "common_start": "2026-06-12",
        "common_end": "2026-06-12",
        "latest_market_date": "2026-06-15",
        "baseline_replay_end": "2026-06-15",
        "actor_replay_end": "2026-06-12",
        "safe_backtest_end": "2026-06-12",
    }
    assert comparison["latest_market_date"] == "2026-06-15"
    assert comparison["safe_backtest_end"] == "2026-06-12"
    assert comparison["baseline_replay_end"] == "2026-06-15"
    assert comparison["actor_replay_end"] == "2026-06-12"
    assert comparison["market_start"] == "2026-06-12"
    assert comparison["market_end"] == "2026-06-15"
    assert comparison["rows"][0]["market"] == {
        "open": 222.22,
        "high": 242.40,
        "low": 214.7406,
        "close": 234.68,
        "volume": 50_419_140,
    }


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
    _write_final_retention_summary(tmp_path, "smoke")
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
        _write_final_retention_summary(tmp_path, profile)
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
        _write_final_retention_summary(tmp_path, profile)
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


def test_build_dashboard_v2_data_blocks_actor_without_final_retention_decision(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    actor_name = "balanced_rule_residual_v22"
    actor_csv = tmp_path / "runtime" / "model" / "card" / actor_name / f"{actor_name}_soxl_daily.csv"
    actor_csv.parent.mkdir(parents=True)
    _write_csv(
        actor_csv,
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
    (actor_csv.parent / f"soxl_{actor_name}_summary.json").write_text(
        json.dumps({"profile": actor_name}),
        encoding="utf-8",
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
                        "profile": actor_name,
                        "status": "canonical",
                        "output": f"runtime/model/card/{actor_name}/{actor_name}_soxl_daily.csv",
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
        selected_profiles=["soxl_backtest_policy_reimplementation_v1", actor_name],
        baseline_profile="soxl_backtest_policy_reimplementation_v1",
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["default_comparison_profiles"] == [
        "soxl_backtest_policy_reimplementation_v1",
        actor_name,
    ]
    assert manifest["deployable_profiles"] == ["soxl_backtest_policy_reimplementation_v1"]
    assert manifest["diagnostic_profiles"] == [actor_name]
    assert manifest["retention_filter"]["policy"] == "final_model_retained_or_reference"
    assert manifest["blocked_profiles"][actor_name]["code"] == "missing_final_retention_decision"
    assert "final_model_retention_decision" in manifest["blocked_profiles"][actor_name]["reason"]
    actor = json.loads((output_dir / "models" / f"{actor_name}.json").read_text())
    assert actor["retention"]["final_model_retained"] is False


def test_build_dashboard_v2_data_defaults_to_key_strict_gate_diagnostic_models(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    profiles = [
        "balanced_rule_residual_v22",
        "mode_conditioned_rebound_reward_train_v2",
        "smoke",
        "balanced_rule_residual_v19_reg",
        "balanced_rule_residual_v19",
        "state_conditioned_drawdown_rebound_reward_train_v1",
    ]
    for profile in profiles:
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
                        "profile": profile,
                        "status": "candidate",
                        "output": f"runtime/model/card/{profile}/{profile}_soxl_daily.csv",
                        "result": "current",
                    }
                    for profile in profiles
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
        "mode_conditioned_rebound_reward_train_v2",
        "smoke",
        "balanced_rule_residual_v19_reg",
        "balanced_rule_residual_v19",
    ]
    assert manifest["filtered_profile_count"] == 1


def test_build_dashboard_v2_data_keeps_reference_baseline_through_latest_market_date(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T00:00:00+00:00",
                "latest_data_date": "2026-01-06",
                "safe_backtest_end": "2026-01-05",
                "profiles": [],
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
                "strategy_nav": 101.0,
                "buyhold_nav": 102.0,
                "position": 0.1,
                "action": "BUY",
                "price": 49.0,
            },
            {
                "window": "ytd_2026",
                "date": "2026-01-06",
                "strategy_nav": 102.0,
                "buyhold_nav": 103.0,
                "position": 0.1,
                "action": "HOLD",
                "price": 50.0,
            },
        ],
    )

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        market_rows={
            "2026-01-02": {"open": 47.0, "high": 48.0, "low": 46.0, "close": 47.24, "volume": 1000},
            "2026-01-05": {"open": 48.0, "high": 50.0, "low": 47.0, "close": 49.0, "volume": 1000},
            "2026-01-06": {"open": 49.0, "high": 51.0, "low": 48.0, "close": 50.0, "volume": 1000},
        },
    )

    reference = json.loads(
        (output_dir / "models" / "soxl_backtest_policy_reimplementation_v1.json").read_text()
    )
    comparison = json.loads((output_dir / "comparison" / "default.json").read_text())
    assert reference["replay_end"] == "2026-01-06"
    assert comparison["common_end"] == "2026-01-06"


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
    _write_final_retention_summary(tmp_path, "balanced_rule_residual_v22")
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
        "smoke",
    ]
    assert manifest["available_profiles"] == manifest["default_comparison_profiles"]
    assert manifest["filtered_profile_count"] == 1


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


def test_build_comparison_payload_rebases_nav_delta_to_common_window_start():
    baseline_rows = [
        {"date": "2026-01-02", "strategy_nav": 100.0, "position": 0.0, "action": "HOLD"},
        {"date": "2026-01-05", "strategy_nav": 105.0, "position": 0.2, "action": "BUY"},
    ]
    actor_rows = [
        {"date": "2020-01-02", "strategy_nav": 100000.0, "position": 0.0, "action": "HOLD"},
        {"date": "2026-01-02", "strategy_nav": 1000000.0, "position": 0.0, "action": "HOLD"},
        {"date": "2026-01-05", "strategy_nav": 1100000.0, "position": 0.7, "action": "BUY"},
    ]

    payload = build_comparison_payload(
        {
            "baseline": {"rows": baseline_rows},
            "actor": {"rows": actor_rows},
        },
        baseline_profile="baseline",
        compared_profiles=["baseline", "actor"],
    )

    actor = payload["rows"][1]["profiles"]["actor"]
    assert actor["strategy_nav_rebased"] == pytest.approx(110.0)
    assert actor["nav_delta_vs_baseline"] == pytest.approx(5.0)
    assert actor["position_delta_vs_baseline"] == pytest.approx(0.5)
