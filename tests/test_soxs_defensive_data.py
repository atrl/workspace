from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.soxs_defensive import (
    _merge_overlap_preserving_history,
    _write_symbol_history,
    merge_yahoo_adjustment_factors,
    normalize_nasdaq_history,
    stitch_yahoo_to_nasdaq,
)


def test_normalize_nasdaq_history_parses_currency_and_commas() -> None:
    frame = normalize_nasdaq_history(
        [
            {
                "date": "07/21/2026",
                "open": "$46.34",
                "high": "49.09",
                "low": "44.44",
                "close": "45.13",
                "volume": "66,515,290",
            }
        ]
    )
    assert frame.loc[0, "date"] == pd.Timestamp("2026-07-21")
    assert frame.loc[0, "open"] == pytest.approx(46.34)
    assert frame.loc[0, "close"] == pytest.approx(45.13)


def test_stitch_yahoo_to_nasdaq_scales_early_history_and_keeps_nasdaq_overlap() -> None:
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    yahoo = pd.DataFrame(
        {
            "date": dates.date.astype(str),
            "open": range(100, 160),
            "high": range(101, 161),
            "low": range(99, 159),
            "close": range(100, 160),
        }
    )
    nasdaq = yahoo.iloc[20:].copy()
    for column in ["open", "high", "low", "close"]:
        nasdaq[column] = nasdaq[column] / 10.0

    stitched, audit = stitch_yahoo_to_nasdaq(yahoo, nasdaq)

    assert len(stitched) == 60
    assert stitched.loc[0, "close"] == pytest.approx(10.0)
    assert stitched.loc[20, "close"] == pytest.approx(12.0)
    assert audit["stitch_scales"]["close"] == pytest.approx(0.1)
    assert audit["max_anchor_relative_error"] == pytest.approx(0.0)


def test_adjustment_factors_allow_short_unadjusted_trailing_gap() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-24", "2026-07-27", "2026-07-28"]),
            "close": [100.0, 101.0, 102.0],
        }
    )
    adjustments = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-24", "2026-07-27"]),
            "adjustment_factor": [1.0, 1.0],
        }
    )

    merged, audit = merge_yahoo_adjustment_factors(
        prices,
        adjustments,
        symbol="SOXL",
    )

    assert merged["adjustment_factor"].tolist() == [1.0, 1.0, 1.0]
    assert audit["adjustment_factor_source_end"] == "2026-07-27"
    assert audit["trailing_adjustment_fill_dates"] == ["2026-07-28"]


def test_adjustment_factors_reject_internal_or_adjusted_trailing_gaps() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-24", "2026-07-27", "2026-07-28"]),
            "close": [100.0, 101.0, 102.0],
        }
    )
    internal_gap = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-24", "2026-07-28"]),
            "adjustment_factor": [1.0, 1.0],
        }
    )
    adjusted_tail = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-24", "2026-07-27"]),
            "adjustment_factor": [0.99, 0.99],
        }
    )

    with pytest.raises(ValueError, match="do not cover"):
        merge_yahoo_adjustment_factors(prices, internal_gap, symbol="SOXL")
    with pytest.raises(ValueError, match="do not cover"):
        merge_yahoo_adjustment_factors(prices, adjusted_tail, symbol="SOXL")


def test_soxs_overlap_refresh_preserves_all_existing_rows_and_appends_new_dates() -> None:
    existing = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01", "2026-07-28"]),
            "open": [10.0, 20.0],
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.5, 20.5],
        }
    )
    update = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-28", "2026-07-29"]),
            "open": [200.0, 300.0],
            "high": [201.0, 301.0],
            "low": [199.0, 299.0],
            "close": [200.5, 300.5],
        }
    )

    merged = _merge_overlap_preserving_history(
        existing, update, refresh_start="2026-07-20"
    )

    assert merged["date"].dt.date.astype(str).tolist() == [
        "2026-07-01",
        "2026-07-28",
        "2026-07-29",
    ]
    assert merged.loc[0, "close"] == pytest.approx(10.5)
    assert merged.loc[1, "close"] == pytest.approx(20.5)
    assert merged.loc[2, "close"] == pytest.approx(300.5)


def test_soxs_overlap_refresh_requires_the_requested_overlap() -> None:
    existing = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-28"]),
            "open": [20.0],
            "high": [21.0],
            "low": [19.0],
            "close": [20.5],
        }
    )
    update = existing.assign(date=pd.to_datetime(["2026-07-29"]))

    with pytest.raises(ValueError, match="missing its validation overlap"):
        _merge_overlap_preserving_history(
            existing,
            update,
            refresh_start="2026-07-20",
        )


def test_soxs_incremental_writer_preserves_existing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "SOXL.csv"
    original = (
        "datetime,instrument,open,high,low,close\n"
        "2026-07-28,SOXL,1.0000000000000002,2.0,0.5,1.5\n"
    )
    path.write_text(original, encoding="utf-8")
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-07-28", "2026-07-29"]),
            "instrument": ["SOXL", "SOXL"],
            "open": [1.0000000000000002, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.0],
            "close": [1.5, 2.5],
        }
    )

    _write_symbol_history(path, frame, symbol="SOXL")

    updated = path.read_text(encoding="utf-8")
    assert updated.startswith(original)
    assert updated.endswith("2026-07-29,SOXL,2.0,3.0,1.0,2.5\n")


def test_committed_soxs_card_is_qlib_native_constrained_and_rejected() -> None:
    root = Path(__file__).resolve().parents[1]
    card = root / "research" / "soxs_defensive_hurdle_v1"
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    source_manifest = json.loads(
        (root / "data" / "raw" / "soxs_defensive_v1" / "source_manifest.json").read_text(encoding="utf-8")
    )
    provider_manifest = json.loads(
        (root / "data" / "qlib" / "soxl_soxx_soxs" / "qlib_store_manifest.json").read_text(encoding="utf-8")
    )
    notebook = json.loads((card / "soxs_defensive_hurdle_v1.ipynb").read_text(encoding="utf-8"))
    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert provider_manifest["status"] == "ok"
    assert set(provider_manifest["instruments"]) == {"SOXL", "SOXX", "SOXS"}
    assert (
        pd.Timestamp(provider_manifest["as_of"])
        >= pd.Timestamp(summary["data_audit"]["as_of"])
    )
    assert source_manifest["relationship_audit"]["soxs_soxx_close_return_correlation"] < -0.99
    assert summary["promotion_decision"] == "research_reject_soxs_ablation_failed_validation_gates"
    assert summary["selection_uses_outer_test"] is False
    assert summary["constraints"]["soxl_soxs_mutually_exclusive"] is True
    assert summary["constraints"]["max_consecutive_soxs_days"] == 5
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    for name in ("probabilities.csv", "positions.csv", "strategy_returns.csv"):
        assert not (card / name).exists()

    artifact_root = Path(summary["recorder_local_dir"]) / "artifacts" / summary["experiment"]
    positions = pd.read_csv(artifact_root / "positions.csv")
    assert not ((positions["soxl_weight"] > 0.0) & (positions["soxs_weight"] > 0.0)).any()
    assert positions["soxs_weight"].max() <= 0.20
    soxs_active = positions["soxs_weight"].gt(0.0)
    run_ids = soxs_active.ne(soxs_active.shift()).cumsum()
    assert soxs_active.groupby(run_ids).sum().max() <= 5
    assert (pd.to_datetime(positions["signal_date"]) < pd.to_datetime(positions["trade_date"])).all()
    assert len(list((artifact_root / "models").glob("*.pkl"))) == 3

    for token in ("qlib.init", "D.calendar", "DatasetH", "DataHandlerLP", "R.get_recorder"):
        assert token in notebook_source
    assert "最终中文结论" in notebook_source
    assert notebook["cells"][1]["execution_count"] is not None
    assert summary["recorder_id"] in json.dumps(notebook)
