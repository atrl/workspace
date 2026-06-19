from __future__ import annotations

import pandas as pd

from src.analysis import assign_regimes


def test_assign_regimes_covers_drawdown_volatility_and_market_state():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-02", periods=6, freq="D"),
            "close": [100.0, 106.0, 80.0, 82.0, 112.0, 108.0],
            "vix": [18.0, 19.0, 35.0, 31.0, 22.0, 24.0],
            "volume": [100, 110, 250, 260, 120, 180],
        }
    )

    regimes = assign_regimes(frame, price_col="close")

    assert {
        "vix_regime",
        "trend_regime",
        "volume_regime",
        "drawdown_regime",
        "volatility_regime",
        "market_regime",
    }.issubset(regimes.columns)
    assert regimes["drawdown_regime"].tolist() == [
        "no_drawdown",
        "no_drawdown",
        "deep_drawdown",
        "deep_drawdown",
        "no_drawdown",
        "no_drawdown",
    ]
    assert regimes["market_regime"].tolist() == ["bull", "bull", "bear", "bear", "bull", "chop"]
    assert regimes.loc[2, "volatility_regime"] == "high_volatility"
