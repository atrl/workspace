from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _price_frame(start: str, periods: int = 4) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame(
        {
            "date": dates.date.astype(str),
            "open": np.arange(periods, dtype=float) + 10.0,
            "high": np.arange(periods, dtype=float) + 11.0,
            "low": np.arange(periods, dtype=float) + 9.0,
            "close": np.arange(periods, dtype=float) + 10.5,
            "adj_close": np.arange(periods, dtype=float) + 10.5,
            "volume": np.arange(periods, dtype=float) + 100.0,
        }
    )


def test_synthetic_daily_reset_return_applies_leverage_fee_and_floor() -> None:
    from src.data.soxx_leveraged_training import (
        synthetic_daily_reset_return,
    )

    result = synthetic_daily_reset_return(
        pd.Series([0.10, -0.60, 0.0]),
        leverage_multiple=2.0,
        annual_fee_bps=252.0,
    )

    assert result.iloc[0] == pytest.approx(0.1999)
    assert result.iloc[1] == -0.999
    assert result.iloc[2] == pytest.approx(-0.0001)


def test_fetch_actual_2x_history_keeps_short_histories_calibration_only(
    tmp_path: Path,
) -> None:
    from src.data.soxx_leveraged_training import (
        ACTUAL_2X_TRAINING_PRODUCTS,
        SoxxLeveragedTrainingPaths,
        fetch_and_store_actual_2x_training_history,
    )

    paths = SoxxLeveragedTrainingPaths(root=tmp_path / "actual2x")
    manifest = fetch_and_store_actual_2x_training_history(
        paths=paths,
        start="2024-01-01",
        end="2024-01-31",
        sleep_seconds=0.0,
        price_fetcher=lambda _: _price_frame("2024-01-02"),
    )

    assert set(manifest["usable_products"]) == set(
        ACTUAL_2X_TRAINING_PRODUCTS
    )
    assert manifest["products"]["NVDL"]["role"].endswith(
        "single_stock_2x_calibration"
    )
    assert (
        manifest["products"]["NVDL"]["two_x_effective_date"]
        == "2024-01-22"
    )
    assert paths.coverage_csv.exists()
    assert json.loads(paths.manifest_json.read_text())["status"] == "ok"


def test_build_pit_leveraged_panel_uses_filing_date_and_numeric_types(
    tmp_path: Path,
) -> None:
    from src.data.soxx_leveraged_training import (
        SoxxLeveragedTrainingPaths,
        build_soxx_pit_leveraged_training_panel,
    )
    from src.data.soxx_pit_holdings import (
        PIT_HOLDINGS_COLUMNS,
        SoxxPitHoldingsPaths,
    )

    pit_paths = SoxxPitHoldingsPaths(root=tmp_path / "pit")
    leveraged_paths = SoxxLeveragedTrainingPaths(
        root=tmp_path / "actual2x"
    )
    pit_paths.daily_dir.mkdir(parents=True)
    leveraged_paths.daily_dir.mkdir(parents=True)
    history = pd.DataFrame(
        [
            {
                "report_date": "2019-09-30",
                "filing_date": "2019-11-25",
                "accession": "a1",
                "series_id": "S000004354",
                "ticker": "AMD",
                "cusip": "007903107",
                "isin": "US0079031078",
                "name": "AMD",
                "country": "US",
                "weight_pct": 100.0,
                "market_value_usd": 100.0,
            }
        ],
        columns=PIT_HOLDINGS_COLUMNS,
    )
    history.to_csv(pit_paths.history_csv, index=False)
    _price_frame("2019-11-22").to_csv(
        pit_paths.daily_dir / "AMD.csv",
        index=False,
    )
    pd.DataFrame(
        [{"symbol": "AMD", "status": "ok"}]
    ).to_csv(pit_paths.price_coverage_csv, index=False)
    pit_paths.price_manifest_json.write_text("{}")
    _price_frame("2019-11-22").to_csv(
        leveraged_paths.daily_dir / "USD.csv",
        index=False,
    )
    pd.DataFrame(
        [{"symbol": "USD", "status": "ok"}]
    ).to_csv(leveraged_paths.coverage_csv, index=False)
    leveraged_paths.manifest_json.write_text("{}")

    panel, audit = build_soxx_pit_leveraged_training_panel(
        pit_paths=pit_paths,
        leveraged_paths=leveraged_paths,
        as_of="2019-11-27",
    )

    amd = panel.loc[panel["instrument"].eq("AMD")].set_index("datetime")
    assert amd.loc[pd.Timestamp("2019-11-22"), "pit_available"] == 0.0
    assert amd.loc[pd.Timestamp("2019-11-25"), "pit_member"] == 1.0
    assert amd.loc[pd.Timestamp("2019-11-25"), "weight_pct"] == 100.0
    assert set(
        panel.loc[
            panel["instrument"].eq("USD"), "leverage_multiple"
        ]
    ) == {2.0}
    assert audit["pit_first_public_date"] == "2019-11-25"
