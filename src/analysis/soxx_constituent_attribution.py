from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


AGGREGATE_SIGNAL_COLUMNS = ("weighted_raw", "weighted_residual", "equal_raw", "breadth_positive")


@dataclass(frozen=True)
class ConstituentAttributionPanel:
    constituents: pd.DataFrame
    daily: pd.DataFrame
    betas: dict[str, float]
    train: tuple[str, str]
    lookback: int
    lag_days: int
    horizon: int


def build_constituent_attribution_panel(
    constituent_frame: pd.DataFrame,
    index_frame: pd.DataFrame,
    *,
    train: tuple[str, str],
    lookback: int = 20,
    lag_days: int = 1,
    horizon: int = 5,
) -> ConstituentAttributionPanel:
    required_constituent = {"datetime", "instrument", "adj_close", "weight_pct"}
    required_index = {"datetime", "soxx_open", "soxl_open"}
    if missing := sorted(required_constituent - set(constituent_frame.columns)):
        raise ValueError(f"constituent attribution frame missing columns: {missing}")
    if missing := sorted(required_index - set(index_frame.columns)):
        raise ValueError(f"index attribution frame missing columns: {missing}")
    if lookback < 1 or lag_days < 1 or horizon < 1:
        raise ValueError("lookback, lag_days, and horizon must be positive")

    index_data = index_frame.copy()
    index_data["datetime"] = pd.to_datetime(index_data["datetime"]).dt.normalize()
    index_data = index_data.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    for column in ["soxx_open", "soxl_open"]:
        index_data[column] = pd.to_numeric(index_data[column], errors="coerce")
    index_data["soxx_signal"] = index_data["soxx_open"].pct_change(lookback, fill_method=None).shift(lag_days)
    index_data["soxx_5d"] = index_data["soxx_open"].shift(-(horizon + 1)) / index_data["soxx_open"].shift(-1) - 1.0
    index_data["soxl_5d"] = index_data["soxl_open"].shift(-(horizon + 1)) / index_data["soxl_open"].shift(-1) - 1.0
    soxx_one_day = index_data["soxx_open"].shift(-1) / index_data["soxx_open"] - 1.0
    synthetic = pd.Series(1.0, index=index_data.index, dtype="float64")
    for step in range(1, horizon + 1):
        synthetic = synthetic * (1.0 + 3.0 * soxx_one_day.shift(-step))
    index_data["synthetic_soxl_5d"] = synthetic - 1.0
    index_data["tracking_residual_5d"] = index_data["soxl_5d"] - index_data["synthetic_soxl_5d"]

    constituents = constituent_frame.copy()
    constituents["datetime"] = pd.to_datetime(constituents["datetime"]).dt.normalize()
    constituents["instrument"] = constituents["instrument"].astype(str).str.upper()
    constituents["adj_close"] = pd.to_numeric(constituents["adj_close"], errors="coerce")
    constituents["weight_pct"] = pd.to_numeric(constituents["weight_pct"], errors="coerce")
    constituents = constituents.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    constituents["raw_signal"] = constituents.groupby("instrument", sort=False)["adj_close"].pct_change(
        lookback,
        fill_method=None,
    ).groupby(constituents["instrument"], sort=False).shift(lag_days)
    constituents = constituents.merge(index_data[["datetime", "soxx_signal"]], on="datetime", how="left")

    train_start, train_end = pd.Timestamp(train[0]), pd.Timestamp(train[1])
    train_mask = constituents["datetime"].between(train_start, train_end)
    betas: dict[str, float] = {}
    residual_parts = []
    for instrument, group in constituents.groupby("instrument", sort=True):
        train_values = group.loc[train_mask.loc[group.index], ["raw_signal", "soxx_signal"]].dropna()
        variance = float(train_values["soxx_signal"].var()) if len(train_values) > 1 else 0.0
        beta = (
            float(train_values["raw_signal"].cov(train_values["soxx_signal"]) / variance)
            if variance > 1e-12
            else 0.0
        )
        betas[str(instrument)] = beta
        part = group.copy()
        part["residual_signal"] = part["raw_signal"] - beta * part["soxx_signal"]
        residual_parts.append(part)
    constituents = pd.concat(residual_parts, ignore_index=True).sort_values(["datetime", "instrument"])

    aggregates = (
        constituents.groupby("datetime", sort=True)
        .apply(_aggregate_daily_signals, include_groups=False)
        .reset_index()
    )
    daily = index_data.merge(aggregates, on="datetime", how="left")
    return ConstituentAttributionPanel(
        constituents=constituents.reset_index(drop=True),
        daily=daily,
        betas=betas,
        train=train,
        lookback=int(lookback),
        lag_days=int(lag_days),
        horizon=int(horizon),
    )


def _aggregate_daily_signals(group: pd.DataFrame) -> pd.Series:
    raw = pd.to_numeric(group["raw_signal"], errors="coerce")
    residual = pd.to_numeric(group["residual_signal"], errors="coerce")
    weights = pd.to_numeric(group["weight_pct"], errors="coerce").clip(lower=0.0)
    valid_raw = raw.notna() & weights.notna() & weights.gt(0.0)
    valid_residual = residual.notna() & weights.notna() & weights.gt(0.0)
    weighted_raw = _weighted_average(raw.loc[valid_raw], weights.loc[valid_raw])
    weighted_residual = _weighted_average(residual.loc[valid_residual], weights.loc[valid_residual])
    return pd.Series(
        {
            "weighted_raw": weighted_raw,
            "weighted_residual": weighted_residual,
            "equal_raw": float(raw.loc[valid_raw].mean()) if valid_raw.any() else np.nan,
            "breadth_positive": float(raw.loc[valid_raw].gt(0.0).mean()) if valid_raw.any() else np.nan,
            "available_count": int(valid_raw.sum()),
            "max_weight_pct": float(weights.loc[valid_raw].max()) if valid_raw.any() else np.nan,
        }
    )


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    if values.empty or float(weights.sum()) <= 0.0:
        return np.nan
    return float((values * (weights / weights.sum())).sum())


def evaluate_constituent_attribution(
    panel: ConstituentAttributionPanel,
    *,
    segments: dict[str, tuple[str, str]],
) -> dict[str, pd.DataFrame]:
    individual_rows: list[dict[str, Any]] = []
    for instrument, group in panel.constituents.groupby("instrument", sort=True):
        merged = group.merge(panel.daily[["datetime", "soxx_5d"]], on="datetime", how="left")
        for segment, window in segments.items():
            sample = _window(merged, window)
            individual_rows.append(
                {
                    "instrument": instrument,
                    "segment": segment,
                    "weight_pct": float(pd.to_numeric(group["weight_pct"], errors="coerce").dropna().iloc[-1]),
                    "beta_to_soxx": float(panel.betas[str(instrument)]),
                    "raw_rank_correlation": _rank_correlation(sample["raw_signal"], sample["soxx_5d"]),
                    "residual_rank_correlation": _rank_correlation(sample["residual_signal"], sample["soxx_5d"]),
                    "observations": int(sample[["raw_signal", "soxx_5d"]].dropna().shape[0]),
                }
            )

    aggregate_rows: list[dict[str, Any]] = []
    nonoverlap_rows: list[dict[str, Any]] = []
    train_window = segments["train"]
    for signal in AGGREGATE_SIGNAL_COLUMNS:
        for target in ["soxx_5d", "soxl_5d"]:
            train_sample = _window(panel.daily, train_window)
            train_corr = _rank_correlation(train_sample[signal], train_sample[target])
            direction = 1.0 if not np.isfinite(train_corr) or train_corr >= 0.0 else -1.0
            for segment, window in segments.items():
                sample = _window(panel.daily, window).copy()
                oriented = pd.to_numeric(sample[signal], errors="coerce") * direction
                aggregate_rows.append(
                    {
                        "signal": signal,
                        "target": target,
                        "segment": segment,
                        "train_direction": direction,
                        "rank_correlation": _rank_correlation(oriented, sample[target]),
                        "directional_auc": _directional_auc(oriented, sample[target]),
                        "top_bottom_spread_pct": _quartile_spread(oriented, sample[target]) * 100.0,
                        "observations": int(pd.concat([oriented, sample[target]], axis=1).dropna().shape[0]),
                    }
                )
                ordered = sample.sort_values("datetime").reset_index(drop=True)
                oriented_ordered = pd.to_numeric(ordered[signal], errors="coerce") * direction
                for offset in range(panel.horizon):
                    offset_signal = oriented_ordered.iloc[offset :: panel.horizon]
                    offset_target = ordered[target].iloc[offset :: panel.horizon]
                    nonoverlap_rows.append(
                        {
                            "signal": signal,
                            "target": target,
                            "segment": segment,
                            "offset": offset,
                            "train_direction": direction,
                            "rank_correlation": _rank_correlation(offset_signal, offset_target),
                            "directional_auc": _directional_auc(offset_signal, offset_target),
                            "top_bottom_spread_pct": _quartile_spread(offset_signal, offset_target) * 100.0,
                            "observations": int(
                                pd.concat([offset_signal, offset_target], axis=1).dropna().shape[0]
                            ),
                            "overlap_purged": True,
                        }
                    )

    leverage_rows = []
    for segment, window in segments.items():
        sample = _window(panel.daily, window)
        clean = sample[["synthetic_soxl_5d", "soxl_5d", "tracking_residual_5d"]].dropna()
        residual = clean["tracking_residual_5d"]
        leverage_rows.append(
            {
                "segment": segment,
                "observations": int(len(clean)),
                "synthetic_actual_rank_correlation": _rank_correlation(
                    clean["synthetic_soxl_5d"], clean["soxl_5d"]
                ),
                "tracking_residual_mean_pct": float(residual.mean() * 100.0) if len(clean) else np.nan,
                "tracking_rmse_pct": float(np.sqrt(np.mean(np.square(residual))) * 100.0) if len(clean) else np.nan,
                "tracking_residual_p05_pct": float(residual.quantile(0.05) * 100.0) if len(clean) else np.nan,
                "tracking_residual_p95_pct": float(residual.quantile(0.95) * 100.0) if len(clean) else np.nan,
            }
        )

    aggregate = pd.DataFrame(aggregate_rows)
    stability_rows = []
    for signal in AGGREGATE_SIGNAL_COLUMNS:
        rows = aggregate.loc[aggregate["signal"].eq(signal) & aggregate["target"].eq("soxx_5d")]
        correlations = rows.set_index("segment")["rank_correlation"].to_dict()
        signs = [np.sign(value) for value in correlations.values() if np.isfinite(value)]
        stability_rows.append(
            {
                "signal": signal,
                "train_rank_correlation": correlations.get("train"),
                "validation_rank_correlation": correlations.get("validation"),
                "test_rank_correlation": correlations.get("test"),
                "all_segments_same_sign": bool(len(signs) == len(segments) and len(set(signs)) == 1),
            }
        )
    return {
        "individual": pd.DataFrame(individual_rows),
        "aggregate": aggregate,
        "leverage": pd.DataFrame(leverage_rows),
        "stability": pd.DataFrame(stability_rows),
        "nonoverlap": pd.DataFrame(nonoverlap_rows),
    }


def summarize_execution_censoring(positions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    data = positions.copy()
    data["datetime"] = pd.to_datetime(data["datetime"]).dt.normalize()
    if "selected" in data:
        selected = data["selected"]
        mask = selected if selected.dtype == bool else selected.astype(str).str.lower().eq("true")
        data = data.loc[mask].copy()
    label_frame = labels.copy()
    label_frame["datetime"] = pd.to_datetime(label_frame["datetime"]).dt.normalize()
    data = data.merge(label_frame[["datetime", "soxl_5d"]], on="datetime", how="left")
    rows = []
    for segment, group in data.groupby("segment", sort=True):
        target = pd.to_numeric(group["target_position"], errors="coerce")
        raw = pd.to_numeric(group["raw_target"], errors="coerce")
        probability = pd.to_numeric(group["probability"], errors="coerce")
        floor = float(target.min()) if target.notna().any() else np.nan
        rows.append(
            {
                "segment": segment,
                "candidate": str(group["candidate"].iloc[0]),
                "observations": int(len(group)),
                "raw_zero_ratio": float(raw.abs().le(1e-12).mean()),
                "position_floor": floor,
                "position_floor_ratio": float(target.sub(floor).abs().le(1e-12).mean()),
                "position_distinct_count": int(target.nunique(dropna=True)),
                "trade_count": int(pd.to_numeric(group["turnover"], errors="coerce").gt(1e-12).sum()),
                "probability_rank_correlation": _rank_correlation(probability, group["soxl_5d"]),
                "probability_directional_auc": _directional_auc(probability, group["soxl_5d"]),
                "probability_top_bottom_spread_pct": _quartile_spread(probability, group["soxl_5d"]) * 100.0,
            }
        )
    return pd.DataFrame(rows)


def _window(frame: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
    datetimes = pd.to_datetime(frame["datetime"]).dt.normalize()
    return frame.loc[datetimes.between(pd.Timestamp(window[0]), pd.Timestamp(window[1]))].copy()


def _rank_correlation(signal: pd.Series, target: pd.Series) -> float:
    clean = pd.concat(
        [pd.to_numeric(signal, errors="coerce").rename("signal"), pd.to_numeric(target, errors="coerce").rename("target")],
        axis=1,
    ).dropna()
    if len(clean) < 3 or clean["signal"].nunique() < 2 or clean["target"].nunique() < 2:
        return np.nan
    return float(clean["signal"].corr(clean["target"], method="spearman"))


def _directional_auc(signal: pd.Series, target: pd.Series) -> float:
    clean = pd.concat(
        [pd.to_numeric(signal, errors="coerce").rename("signal"), pd.to_numeric(target, errors="coerce").rename("target")],
        axis=1,
    ).dropna()
    labels = clean["target"].gt(0.0).astype(int)
    if len(clean) < 3 or labels.nunique() < 2 or clean["signal"].nunique() < 2:
        return np.nan
    return float(roc_auc_score(labels, clean["signal"]))


def _quartile_spread(signal: pd.Series, target: pd.Series) -> float:
    clean = pd.concat(
        [pd.to_numeric(signal, errors="coerce").rename("signal"), pd.to_numeric(target, errors="coerce").rename("target")],
        axis=1,
    ).dropna()
    if len(clean) < 4 or clean["signal"].nunique() < 4:
        return np.nan
    quantiles = pd.qcut(clean["signal"], 4, labels=False, duplicates="drop")
    if quantiles.nunique() < 2:
        return np.nan
    grouped = clean.assign(quantile=quantiles).groupby("quantile")["target"].mean()
    return float(grouped.iloc[-1] - grouped.iloc[0])
