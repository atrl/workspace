from __future__ import annotations

import pandas as pd


def promotion_scores(
    frame: pd.DataFrame,
    *,
    min_score: float,
    max_drawdown_floor: float,
) -> pd.DataFrame:
    required = {"model", "return_pct", "max_drawdown_pct", "turnover", "ic", "rank_ic", "recorder_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing qlib recorder metric columns: {missing}")

    result = frame.copy()
    result["promotion_score"] = (
        result["return_pct"].astype(float)
        + result["ic"].astype(float) * 100.0
        + result["rank_ic"].astype(float) * 50.0
        - result["max_drawdown_pct"].abs().astype(float)
        - result["turnover"].astype(float) * 5.0
    )
    result["source"] = "qlib_recorder_metrics"
    result["promote"] = [
        bool(score >= min_score and drawdown >= max_drawdown_floor)
        for score, drawdown in zip(result["promotion_score"], result["max_drawdown_pct"], strict=True)
    ]
    result["promote"] = result["promote"].astype(object)
    return result.sort_values(["promotion_score", "return_pct"], ascending=[False, False]).reset_index(drop=True)
