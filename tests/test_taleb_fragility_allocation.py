from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.factor.taleb_fragility import (
    EMPIRICAL_TAIL_FEATURES,
    IMPLIED_TAIL_FEATURES,
    STRUCTURAL_FRAGILITY_FEATURES,
    TALEB_FRAGILITY_FEATURES,
    build_taleb_fragility_feature_frame,
)
from src.workflow.taleb_fragility_allocation import CANDIDATES


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "taleb_fragility_allocation_v1"


def _synthetic_inputs(
    rows: int = 1000,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    dates = pd.date_range("2010-01-04", periods=rows, freq="B")
    phase = np.arange(rows, dtype=float)
    soxx_return = (
        0.0004
        + 0.011 * np.sin(phase / 17.0)
        + 0.006 * np.cos(phase / 7.0)
    )
    soxx_return[np.arange(50, rows, 53)] -= 0.09
    soxl_return = 3.0 * soxx_return - 3.0 * soxx_return**2
    daily = pd.DataFrame(
        {
            "datetime": dates,
            "soxx_close": 100.0 * np.cumprod(1.0 + soxx_return),
            "soxl_close": 20.0 * np.cumprod(1.0 + soxl_return),
            "vix": 20.0 + 4.0 * np.sin(phase / 29.0),
        }
    )
    external = {
        "vix9d": pd.DataFrame(
            {
                "DATE": dates,
                "CLOSE": 19.0 + 5.0 * np.sin(phase / 23.0),
            }
        ),
        "vix3m": pd.DataFrame(
            {
                "DATE": dates,
                "CLOSE": 22.0 + 3.0 * np.sin(phase / 37.0),
            }
        ),
        "vvix": pd.DataFrame(
            {
                "DATE": dates,
                "VVIX": 90.0 + 12.0 * np.cos(phase / 31.0),
            }
        ),
        "skew": pd.DataFrame(
            {
                "DATE": dates,
                "SKEW": 130.0 + 8.0 * np.sin(phase / 41.0),
            }
        ),
    }
    return daily, external


def _build(
    daily: pd.DataFrame, external: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    return build_taleb_fragility_feature_frame(
        daily=daily,
        vix9d=external["vix9d"],
        vix3m=external["vix3m"],
        vvix=external["vvix"],
        skew=external["skew"],
    )


def test_taleb_feature_families_are_small_and_non_overlapping() -> None:
    assert len(TALEB_FRAGILITY_FEATURES) == 16
    assert set(EMPIRICAL_TAIL_FEATURES).isdisjoint(IMPLIED_TAIL_FEATURES)
    assert set(EMPIRICAL_TAIL_FEATURES).isdisjoint(
        STRUCTURAL_FRAGILITY_FEATURES
    )
    assert set(IMPLIED_TAIL_FEATURES).isdisjoint(
        STRUCTURAL_FRAGILITY_FEATURES
    )


def test_taleb_features_are_finite_and_bounded_after_warmup() -> None:
    daily, external = _synthetic_inputs()
    features = _build(daily, external)
    tail = features.tail(100)
    assert set(TALEB_FRAGILITY_FEATURES).issubset(features.columns)
    assert tail.loc[:, TALEB_FRAGILITY_FEATURES].notna().all().all()
    assert (
        tail["soxx_downside_semivariance_share_20"].between(0.0, 1.0)
    ).all()
    assert tail["soxx_jump_ratio_20"].between(0.0, 1.0).all()
    assert tail["soxx_tail_loss_concentration_252"].between(
        0.0, 1.0
    ).all()


def test_taleb_features_do_not_change_when_future_rows_are_appended() -> None:
    daily, external = _synthetic_inputs()
    cutoff = 900
    short_daily = daily.iloc[:cutoff].copy()
    short_external = {
        name: frame.iloc[:cutoff].copy()
        for name, frame in external.items()
    }
    short = _build(short_daily, short_external).set_index("datetime")
    long = _build(daily, external).set_index("datetime")
    pd.testing.assert_frame_equal(
        short.loc[:, TALEB_FRAGILITY_FEATURES],
        long.loc[short.index, TALEB_FRAGILITY_FEATURES],
    )


def test_feature_ablation_keeps_model_family_fixed() -> None:
    assert {candidate.feature_set for candidate in CANDIDATES} == {
        "base",
        "base_empirical",
        "base_implied",
        "base_structural",
        "base_all_taleb",
    }
    assert {candidate.encoder for candidate in CANDIDATES} == {"tabular"}
    assert all(candidate.multitask for candidate in CANDIDATES)


def test_taleb_card_contract_after_execution() -> None:
    if not (CARD / "summary.json").exists():
        return
    summary = json.loads(
        (CARD / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["artifact_policy"]["policy"] == (
        "slim_card_recorder_first"
    )
    assert summary["development"]["selection_uses_outer_test"] is False
    assert summary["architecture"]["base_position"] is None
    excluded = summary["data_audit"]["taleb_provider"][
        "excluded_sources"
    ]
    assert "2024" in excluded["intraday_external_context"]
    assert "non_PIT" in excluded["soxx_constituents"]
    root_names = {path.name for path in CARD.iterdir() if path.is_file()}
    assert "predictions.csv" not in root_names
    assert "positions.csv" not in root_names
    assert "strategy_returns.csv" not in root_names


def test_taleb_provider_manifest_records_qlib_readback() -> None:
    manifest_path = (
        ROOT
        / "data"
        / "qlib"
        / "taleb_fragility_v1"
        / "qlib_store_manifest.json"
    )
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["point_in_time"] is True
    assert manifest["feature_file_count"] == len(
        TALEB_FRAGILITY_FEATURES
    )
    assert {
        "qlib.init",
        "qlib.data.D.calendar",
        "qlib.data.D.instruments",
        "qlib.data.D.features",
    }.issubset(manifest["source_api"])
