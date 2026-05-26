from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


_PATH_COLUMNS = {
    "datetime",
    "label",
    "account",
    "nav",
    "daily_return",
    "peak",
    "drawdown",
    "position",
}


def _portfolio_frame(result: dict[str, Any]) -> pd.DataFrame:
    portfolio = result.get("portfolio")
    if isinstance(portfolio, tuple):
        portfolio = portfolio[0]
    if not isinstance(portfolio, pd.DataFrame):
        raise ValueError("backtest result missing portfolio DataFrame")
    if "account" not in portfolio.columns:
        raise ValueError("portfolio frame missing `account` column")
    frame = portfolio.copy()
    if "datetime" in frame.columns:
        index = pd.to_datetime(frame["datetime"])
    else:
        index = pd.to_datetime(frame.index)
    frame.index = index
    frame = frame.sort_index()
    frame = frame[~frame.index.isna()]
    return frame


def _position_series(result: dict[str, Any], index: pd.DatetimeIndex) -> pd.Series:
    positions = np.asarray(result.get("position_history", []), dtype=float)
    if positions.size == 0:
        return pd.Series(0.0, index=index)
    if positions.size >= len(index):
        return pd.Series(positions[: len(index)], index=index).astype(float)
    pos_index = index[: positions.size]
    series = pd.Series(positions, index=pos_index)
    return series.reindex(index).ffill().bfill().fillna(0.0).astype(float)


def _normalise_feature_frame(feature_frame: pd.DataFrame | None) -> pd.DataFrame:
    if feature_frame is None or feature_frame.empty:
        return pd.DataFrame()
    features = feature_frame.copy()
    if "datetime" in features.columns:
        features.index = pd.to_datetime(features["datetime"])
        features = features.drop(columns=["datetime"])
    else:
        features.index = pd.to_datetime(features.index)
    features = features.sort_index()
    return features[~features.index.isna()]


def make_backtest_daily_frame(
    result: dict[str, Any],
    *,
    label: str,
    feature_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert a backtest result into an aligned daily path diagnostics frame."""

    portfolio = _portfolio_frame(result)
    account = pd.to_numeric(portfolio["account"], errors="coerce").dropna()
    if account.empty:
        raise ValueError("portfolio account series is empty")
    index = pd.DatetimeIndex(account.index)
    out = pd.DataFrame(index=index)
    out["datetime"] = index
    out["label"] = str(label)
    out["account"] = account.astype(float).to_numpy()
    out["position"] = _position_series(result, index).to_numpy(dtype=float)
    first_account = float(out["account"].iloc[0])
    out["nav"] = out["account"] / first_account if first_account else out["account"]
    out["daily_return"] = out["account"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["peak"] = out["nav"].cummax()
    out["drawdown"] = (1.0 - out["nav"] / out["peak"].replace(0.0, np.nan)).fillna(0.0)

    features = _normalise_feature_frame(feature_frame)
    if not features.empty:
        out = out.join(features.reindex(index))
    return out.reset_index(drop=True)


def _normalise_daily_path(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        raise ValueError("daily path is empty")
    frame = daily.copy()
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame.dropna(subset=["datetime"]).sort_values("datetime")
        frame.index = pd.DatetimeIndex(frame["datetime"])
    else:
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        frame = frame[~frame.index.isna()].sort_index()
        frame["datetime"] = frame.index
    if "account" not in frame.columns and "nav" not in frame.columns:
        raise ValueError("daily path missing `account` or `nav` column")
    if "account" not in frame.columns:
        frame["account"] = pd.to_numeric(frame["nav"], errors="coerce")
    else:
        frame["account"] = pd.to_numeric(frame["account"], errors="coerce")
    if "position" not in frame.columns:
        frame["position"] = 0.0
    frame["position"] = pd.to_numeric(frame["position"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["account"])
    if frame.empty:
        raise ValueError("daily path account series is empty")
    first_account = float(frame["account"].iloc[0])
    frame["nav"] = frame["account"] / first_account if first_account else frame["account"]
    frame["peak"] = frame["nav"].cummax()
    frame["drawdown"] = (1.0 - frame["nav"] / frame["peak"].replace(0.0, np.nan)).fillna(0.0)
    return frame


def _min_depth_pct(min_depth: float) -> float:
    value = float(min_depth)
    return value * 100.0 if 0.0 <= value <= 1.0 else value


def _append_drawdown_interval(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    drawdown: pd.Series,
    *,
    peak_pos: int,
    end_pos: int,
    min_depth_pct: float,
) -> None:
    if end_pos < peak_pos:
        return
    segment = frame.iloc[peak_pos : end_pos + 1]
    segment_dd = drawdown.iloc[peak_pos : end_pos + 1]
    if segment.empty or segment_dd.empty:
        return
    trough_label = segment_dd.idxmax()
    max_dd_pct = float(segment_dd.loc[trough_label]) * 100.0
    if max_dd_pct < min_depth_pct:
        return
    peak_label = segment["nav"].loc[:trough_label].idxmax()
    path = frame.loc[peak_label:trough_label]
    peak_nav = float(frame.loc[peak_label, "nav"])
    trough_nav = float(frame.loc[trough_label, "nav"])
    period_return = (trough_nav / peak_nav - 1.0) * 100.0 if peak_nav else 0.0
    rows.append(
        {
            "peak_date": str(pd.Timestamp(peak_label).date()),
            "trough_date": str(pd.Timestamp(trough_label).date()),
            "start": str(pd.Timestamp(peak_label).date()),
            "end": str(pd.Timestamp(trough_label).date()),
            "rows": int(len(path)),
            "max_dd": round(max_dd_pct, 4),
            "max_dd_fraction": round(max_dd_pct / 100.0, 8),
            "return": round(float(period_return), 4),
            "avg_pos": round(float(path["position"].mean()), 6),
            "peak_nav": round(peak_nav, 8),
            "trough_nav": round(trough_nav, 8),
        }
    )


def find_worst_drawdown_intervals(
    daily: pd.DataFrame,
    *,
    top_n: int = 3,
    min_depth: float = 0.05,
) -> pd.DataFrame:
    """Find peak-to-trough drawdown intervals from a strategy daily path."""

    frame = _normalise_daily_path(daily).reset_index(drop=True)
    frame.index = pd.DatetimeIndex(frame["datetime"])
    drawdown = frame["drawdown"].astype(float)
    rows: list[dict[str, Any]] = []
    current_peak_pos = 0
    segment_peak_pos = 0
    in_segment = False
    threshold_pct = _min_depth_pct(min_depth)

    for pos, dd_value in enumerate(drawdown.to_numpy(dtype=float)):
        nav = float(frame["nav"].iloc[pos])
        peak = float(frame["peak"].iloc[pos])
        at_peak = nav >= peak - 1e-12
        if at_peak:
            if in_segment:
                _append_drawdown_interval(
                    rows,
                    frame,
                    drawdown,
                    peak_pos=segment_peak_pos,
                    end_pos=pos - 1,
                    min_depth_pct=threshold_pct,
                )
                in_segment = False
            current_peak_pos = pos
            continue
        if dd_value > 0.0 and not in_segment:
            in_segment = True
            segment_peak_pos = current_peak_pos

    if in_segment:
        _append_drawdown_interval(
            rows,
            frame,
            drawdown,
            peak_pos=segment_peak_pos,
            end_pos=len(frame) - 1,
            min_depth_pct=threshold_pct,
        )

    columns = [
        "rank",
        "peak_date",
        "trough_date",
        "start",
        "end",
        "rows",
        "max_dd",
        "max_dd_fraction",
        "return",
        "avg_pos",
        "peak_nav",
        "trough_nav",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows).drop_duplicates(subset=["peak_date", "trough_date"])
    out = out.sort_values(["max_dd", "peak_date"], ascending=[False, True]).head(int(top_n)).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1, dtype=int))
    return out[columns]


def _interval_bounds(row: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = row.get("peak_date", row.get("start"))
    end = row.get("trough_date", row.get("end"))
    if pd.isna(start) or pd.isna(end):
        raise ValueError("interval row missing peak/trough or start/end dates")
    return pd.Timestamp(start), pd.Timestamp(end)


def _path_return(path: pd.DataFrame) -> float:
    if path.empty:
        return 0.0
    first = float(path["account"].iloc[0])
    last = float(path["account"].iloc[-1])
    return round((last / first - 1.0) * 100.0, 4) if first else 0.0


def _feature_columns(frame: pd.DataFrame, feature_cols: Sequence[str] | None) -> list[str]:
    if feature_cols is not None:
        return [str(col) for col in feature_cols if str(col) in frame.columns]
    numeric = frame.select_dtypes(include=[np.number]).columns
    return [str(col) for col in numeric if str(col) not in _PATH_COLUMNS]


def compare_interval_feature_deltas(
    *,
    candidate_daily: pd.DataFrame,
    intervals: pd.DataFrame,
    baseline_daily: pd.DataFrame | None = None,
    feature_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare feature states inside failed intervals against the rest of the path."""

    candidate = _normalise_daily_path(candidate_daily)
    baseline = _normalise_daily_path(baseline_daily) if baseline_daily is not None else None
    features = _feature_columns(candidate, feature_cols)
    rows: list[dict[str, Any]] = []
    if intervals.empty or not features:
        return pd.DataFrame()

    for _, interval in intervals.iterrows():
        start, end = _interval_bounds(interval)
        mask = (candidate.index >= start) & (candidate.index <= end)
        window = candidate.loc[mask]
        if window.empty:
            continue
        reference = candidate.loc[~mask]
        if reference.empty:
            reference = candidate
        base_window = pd.DataFrame()
        if baseline is not None:
            base_window = baseline.loc[(baseline.index >= start) & (baseline.index <= end)]
        candidate_avg_pos = round(float(window["position"].mean()), 6)
        baseline_avg_pos = round(float(base_window["position"].mean()), 6) if not base_window.empty else 0.0
        candidate_return = _path_return(window)
        baseline_return = _path_return(base_window) if not base_window.empty else 0.0
        for feature in features:
            interval_series = pd.to_numeric(window[feature], errors="coerce").dropna()
            reference_series = pd.to_numeric(reference[feature], errors="coerce").dropna()
            if interval_series.empty or reference_series.empty:
                continue
            reference_std = float(reference_series.std(ddof=0))
            if reference_std == 0.0 or np.isnan(reference_std):
                reference_std = float(candidate[feature].std(ddof=0))
            if reference_std == 0.0 or np.isnan(reference_std):
                continue
            interval_mean = float(interval_series.mean())
            reference_mean = float(reference_series.mean())
            delta = interval_mean - reference_mean
            rows.append(
                {
                    "interval_rank": int(interval.get("rank", len(rows) + 1)),
                    "interval_start": str(start.date()),
                    "interval_end": str(end.date()),
                    "feature": str(feature),
                    "interval_mean": round(interval_mean, 8),
                    "reference_mean": round(reference_mean, 8),
                    "delta": round(float(delta), 8),
                    "delta_z": round(float(delta / reference_std), 8),
                    "interval_coverage": round(float(len(interval_series) / max(len(window), 1)), 6),
                    "reference_coverage": round(float(len(reference_series) / max(len(reference), 1)), 6),
                    "candidate_avg_pos": candidate_avg_pos,
                    "baseline_avg_pos": baseline_avg_pos,
                    "position_delta": round(candidate_avg_pos - baseline_avg_pos, 6),
                    "candidate_return": candidate_return,
                    "baseline_return": baseline_return,
                    "return_delta": round(candidate_return - baseline_return, 4),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["abs_delta_z"] = out["delta_z"].abs()
    return out.sort_values(["interval_rank", "abs_delta_z", "feature"], ascending=[True, False, True]).reset_index(drop=True)


def feature_group_for_name(feature: str) -> str:
    """Map a feature name into a coarse mechanism group for interval diagnostics."""

    name = str(feature).lower()
    if any(token in name for token in ("soxx", "relative", "ratio", "spread", "tracking", "te_")):
        return "relative_strength_spread"
    if any(token in name for token in ("vix", "vol", "drawdown", "stress", "risk", "atr", "range")):
        return "volatility_or_stress"
    if any(token in name for token in ("mom", "momentum", "trend", "ma_", "ret", "return", "rsi")):
        return "trend_or_momentum"
    if any(token in name for token in ("volume", "turnover", "amount", "money", "flow")):
        return "liquidity_or_volume"
    if any(token in name for token in ("price", "close", "open", "high", "low")):
        return "price_level"
    return "other"


def summarize_interval_feature_groups(
    feature_deltas: pd.DataFrame,
    *,
    top_n: int = 10,
    top_features_per_group: int = 4,
) -> pd.DataFrame:
    """Summarize interval feature deltas into coarse mechanism groups."""

    columns = [
        "feature_group",
        "feature_count",
        "mean_abs_delta_z",
        "max_abs_delta_z",
        "mean_delta_z",
        "mean_position_delta",
        "mean_return_delta",
        "top_features",
    ]
    if feature_deltas.empty:
        return pd.DataFrame(columns=columns)
    required = {"feature", "delta_z"}
    missing = sorted(required - set(feature_deltas.columns))
    if missing:
        raise ValueError(f"feature_deltas missing columns: {missing}")

    frame = feature_deltas.copy()
    frame["delta_z"] = pd.to_numeric(frame["delta_z"], errors="coerce")
    frame = frame.dropna(subset=["delta_z"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["abs_delta_z"] = frame["delta_z"].abs()
    frame["feature_group"] = frame["feature"].map(feature_group_for_name)
    if "position_delta" not in frame.columns:
        frame["position_delta"] = 0.0
    if "return_delta" not in frame.columns:
        frame["return_delta"] = 0.0
    frame["position_delta"] = pd.to_numeric(frame["position_delta"], errors="coerce").fillna(0.0)
    frame["return_delta"] = pd.to_numeric(frame["return_delta"], errors="coerce").fillna(0.0)

    rows: list[dict[str, object]] = []
    for group, sub in frame.groupby("feature_group", sort=True):
        ranked = sub.sort_values(["abs_delta_z", "feature"], ascending=[False, True])
        rows.append(
            {
                "feature_group": str(group),
                "feature_count": int(sub["feature"].nunique()),
                "mean_abs_delta_z": round(float(sub["abs_delta_z"].mean()), 6),
                "max_abs_delta_z": round(float(sub["abs_delta_z"].max()), 6),
                "mean_delta_z": round(float(sub["delta_z"].mean()), 6),
                "mean_position_delta": round(float(sub["position_delta"].mean()), 6),
                "mean_return_delta": round(float(sub["return_delta"].mean()), 6),
                "top_features": ",".join(str(value) for value in ranked["feature"].head(int(top_features_per_group))),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["mean_abs_delta_z", "feature_count", "feature_group"],
        ascending=[False, False, True],
    ).head(int(top_n)).reset_index(drop=True)[columns]


__all__ = [
    "compare_interval_feature_deltas",
    "feature_group_for_name",
    "find_worst_drawdown_intervals",
    "make_backtest_daily_frame",
    "summarize_interval_feature_groups",
]
