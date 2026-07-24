from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.soxs_defensive import normalize_nasdaq_history, stitch_yahoo_to_nasdaq


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
