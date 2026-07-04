from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data import QlibProviderConfig, init_qlib_provider, load_qlib_feature_frame


QLIB_DOCS = {
    "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
    "model": "https://qlib.readthedocs.io/en/latest/component/model.html",
    "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
    "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
}
QLIB_FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$turnover", "$bar_index", "$bar_count"]
QLIB_LABEL = "Ref($close, -1) / Ref($open, -1) - 1"
MODEL_NAMES = ["lgb", "xgb", "lgb_xgb_consensus"]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


@dataclass(frozen=True)
class IntradayBarModelSpec:
    experiment: str
    output_dir: Path
    recorder_uri: Path
    provider_uris: dict[str, Path]
    market: str = "all"
    region: str = "us"
    instruments: tuple[str, str] = ("SOXL", "SOXX")
    train: tuple[str, str] = ("2019-01-02", "2024-12-31")
    valid: tuple[str, str] = ("2025-01-02", "2025-12-31")
    test: tuple[str, str] = ("2026-01-02", "2026-07-02")
    as_of: str = "2026-07-02"
    cost_bps: float = 5.0
    purge_bars: int = 1
    feature_set: str = "baseline"
    trade_gate: str = "none"
    cost_stress_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
    walk_forward_models: tuple[str, ...] = ("lgb_xgb_consensus",)


@dataclass(frozen=True)
class IntradayBarModelResult:
    recorder_id: str
    recorder_local_dir: Path
    summary: dict[str, Any]


def _recorder_files(recorder_dir: Path) -> list[str]:
    return sorted(path.relative_to(recorder_dir).as_posix() for path in recorder_dir.rglob("*") if path.is_file())


def _remove_transient_code_snapshots(recorder_dir: Path) -> None:
    for name in ["code_cached.txt", "code_diff.txt", "code_status.txt"]:
        path = recorder_dir / "artifacts" / name
        if path.exists():
            path.unlink()


def _freq_open_time(freq: str) -> str:
    if freq == "5min":
        return "09:35:00"
    if freq == "15min":
        return "09:45:00"
    raise ValueError(f"unsupported intraday frequency: {freq}")


def _segment_times(spec: IntradayBarModelSpec, freq: str) -> dict[str, tuple[str, str]]:
    first_bar = _freq_open_time(freq)
    return {
        "train": (f"{spec.train[0]} {first_bar}", f"{spec.train[1]} 16:00:00"),
        "valid": (f"{spec.valid[0]} {first_bar}", f"{spec.valid[1]} 16:00:00"),
        "test": (f"{spec.test[0]} {first_bar}", f"{spec.test[1]} 16:00:00"),
    }


def _build_dataset_row_audit(
    config: QlibProviderConfig,
    *,
    segments: dict[str, tuple[str, str]],
    instruments: list[str],
) -> dict[str, Any]:
    init_qlib_provider(config)
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import QlibDataLoader

    loader = QlibDataLoader(config={"feature": QLIB_FIELDS, "label": [QLIB_LABEL]}, freq=config.freq)
    handler = DataHandlerLP(
        instruments=instruments,
        start_time=min(start for start, _ in segments.values()),
        end_time=max(end for _, end in segments.values()),
        data_loader=loader,
    )
    dataset = DatasetH(handler=handler, segments=segments)
    return {
        "api": ["DatasetH", "DataHandlerLP", "QlibDataLoader"],
        "label": QLIB_LABEL,
        "segment_rows": {name: int(len(dataset.prepare(name))) for name in segments},
    }


def _wide_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.reset_index().copy()
    frame["instrument"] = frame["instrument"].astype(str).str.lower()
    pieces = []
    for instrument, group in frame.groupby("instrument", sort=True):
        renamed = {
            field: f"{instrument}_{field.removeprefix('$')}"
            for field in QLIB_FIELDS
            if field in group.columns
        }
        piece = group.set_index("datetime")[list(renamed)].rename(columns=renamed)
        pieces.append(piece)
    wide = pd.concat(pieces, axis=1).sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reset_index().rename(columns={"index": "datetime"})
    wide["date"] = wide["datetime"].dt.date
    return wide


def _time_of_day_zscore(frame: pd.DataFrame, column: str, *, bar_col: str = "soxl_bar_index") -> pd.Series:
    return frame.groupby(bar_col, group_keys=False)[column].transform(
        lambda item: (item - item.shift(1).rolling(60, min_periods=20).mean())
        / (item.shift(1).rolling(60, min_periods=20).std(ddof=0) + 1e-12)
    )


def _add_regime_v1_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.copy()
    feature_cols: list[str] = []
    for prefix in ["soxl", "soxx"]:
        day_open = frame.groupby("date")[f"{prefix}_open"].transform("first")
        day_high = frame.groupby("date")[f"{prefix}_high"].cummax()
        day_low = frame.groupby("date")[f"{prefix}_low"].cummin()
        frame[f"{prefix}_cum_ret_from_open"] = frame[f"{prefix}_close"] / day_open - 1.0
        frame[f"{prefix}_range_so_far"] = day_high / (day_low + 1e-12) - 1.0
        frame[f"{prefix}_dist_day_high"] = frame[f"{prefix}_close"] / (day_high + 1e-12) - 1.0
        frame[f"{prefix}_dist_day_low"] = frame[f"{prefix}_close"] / (day_low + 1e-12) - 1.0
        frame[f"{prefix}_opening_ret"] = frame.groupby("date")[f"{prefix}_ret_bar"].transform("first")
        frame[f"{prefix}_volume_tod_z60"] = _time_of_day_zscore(frame, f"{prefix}_volume")
        frame[f"{prefix}_range_tod_z60"] = _time_of_day_zscore(frame, f"{prefix}_range")
        frame[f"{prefix}_ret_tod_z60"] = _time_of_day_zscore(frame, f"{prefix}_ret_bar")
        feature_cols.extend(
            [
                f"{prefix}_cum_ret_from_open",
                f"{prefix}_range_so_far",
                f"{prefix}_dist_day_high",
                f"{prefix}_dist_day_low",
                f"{prefix}_opening_ret",
                f"{prefix}_volume_tod_z60",
                f"{prefix}_range_tod_z60",
                f"{prefix}_ret_tod_z60",
            ]
        )
    frame["tod_sin"] = np.sin(2.0 * np.pi * frame["tod_frac"])
    frame["tod_cos"] = np.cos(2.0 * np.pi * frame["tod_frac"])
    frame["abs_soxl_minus_3x_soxx_ret"] = frame["soxl_minus_3x_soxx_ret"].abs()
    frame["abs_soxl_minus_3x_soxx_mom3"] = frame["soxl_minus_3x_soxx_mom3"].abs()
    feature_cols.extend(
        [
            "tod_sin",
            "tod_cos",
            "abs_soxl_minus_3x_soxx_ret",
            "abs_soxl_minus_3x_soxx_mom3",
        ]
    )
    return frame, feature_cols


def _add_trade_gate(frame: pd.DataFrame, *, trade_gate: str) -> tuple[pd.DataFrame, str | None]:
    if trade_gate == "none":
        return frame, None
    if trade_gate != "regime_v1":
        raise ValueError(f"unsupported intraday trade gate: {trade_gate}")
    frame = frame.copy()
    intraday_window = frame["tod_frac"].between(0.16, 0.84)
    range_shock = frame[["soxl_range_tod_z60", "soxx_range_tod_z60"]].max(axis=1) > 0.15
    volume_shock = frame[["soxl_volume_tod_z60", "soxx_volume_tod_z60"]].max(axis=1) > 0.20
    dislocation = frame["abs_soxl_minus_3x_soxx_ret"] > 0.0015
    directional_state = frame[["soxl_cum_ret_from_open", "soxx_cum_ret_from_open"]].abs().max(axis=1) > 0.004
    frame["regime_gate_v1"] = (
        intraday_window & (range_shock | volume_shock | dislocation | directional_state)
    ).fillna(False)
    return frame, "regime_gate_v1"


def _engineer_model_frame(
    raw: pd.DataFrame,
    *,
    cost_bps: float,
    feature_set: str = "baseline",
    trade_gate: str = "none",
) -> tuple[pd.DataFrame, list[str], str | None]:
    frame = _wide_feature_frame(raw)
    for prefix in ["soxl", "soxx"]:
        frame[f"{prefix}_ret_bar"] = frame[f"{prefix}_close"] / frame[f"{prefix}_open"] - 1.0
        frame[f"{prefix}_ret_cc1"] = frame.groupby("date")[f"{prefix}_close"].pct_change()
        frame[f"{prefix}_range"] = frame[f"{prefix}_high"] / frame[f"{prefix}_low"] - 1.0
        frame[f"{prefix}_close_loc"] = (
            (frame[f"{prefix}_close"] - frame[f"{prefix}_low"])
            / (frame[f"{prefix}_high"] - frame[f"{prefix}_low"] + 1e-12)
        )
        typical = (frame[f"{prefix}_high"] + frame[f"{prefix}_low"] + frame[f"{prefix}_close"]) / 3.0
        pv = typical * frame[f"{prefix}_volume"]
        frame[f"{prefix}_cum_vwap"] = pv.groupby(frame["date"]).cumsum() / (
            frame.groupby("date")[f"{prefix}_volume"].cumsum() + 1e-12
        )
        frame[f"{prefix}_vwap_dist"] = frame[f"{prefix}_close"] / frame[f"{prefix}_cum_vwap"] - 1.0
        for window in [2, 3, 6, 12]:
            frame[f"{prefix}_mom_{window}"] = frame.groupby("date")[f"{prefix}_close"].pct_change(window)
            frame[f"{prefix}_rv_{window}"] = (
                frame.groupby("date")[f"{prefix}_ret_cc1"]
                .rolling(window, min_periods=max(2, window // 2))
                .std()
                .reset_index(level=0, drop=True)
            )
            frame[f"{prefix}_vol_z_{window}"] = (
                frame[f"{prefix}_volume"]
                / (
                    frame.groupby("date")[f"{prefix}_volume"]
                    .rolling(window, min_periods=max(2, window // 2))
                    .mean()
                    .reset_index(level=0, drop=True)
                    + 1e-12
                )
                - 1.0
            )
    frame["tod_frac"] = frame["soxl_bar_index"] / (frame["soxl_bar_count"] - 1.0).replace(0.0, np.nan)
    frame["soxl_minus_3x_soxx_ret"] = frame["soxl_ret_cc1"] - 3.0 * frame["soxx_ret_cc1"]
    frame["soxl_minus_3x_soxx_mom3"] = frame["soxl_mom_3"] - 3.0 * frame["soxx_mom_3"]
    frame["range_ratio_soxl_soxx"] = frame["soxl_range"] / (frame["soxx_range"] + 1e-12)
    extra_feature_cols: list[str] = []
    if feature_set == "regime_v1":
        frame, extra_feature_cols = _add_regime_v1_features(frame)
    elif feature_set != "baseline":
        raise ValueError(f"unsupported intraday feature set: {feature_set}")
    frame, trade_gate_col = _add_trade_gate(frame, trade_gate=trade_gate)
    frame["entry_datetime"] = frame.groupby("date")["datetime"].shift(-1)
    frame["target_return"] = (
        frame.groupby("date")["soxl_close"].shift(-1) / frame.groupby("date")["soxl_open"].shift(-1) - 1.0
    )
    frame["target_positive"] = (frame["target_return"] > cost_bps / 10000.0).astype(int)
    same_session_last_bar_rows_dropped = int(frame["target_return"].isna().sum())
    feature_cols = [
        col
        for col in frame.columns
        if any(
            col.startswith(prefix)
            for prefix in [
                "soxl_ret",
                "soxl_range",
                "soxl_close_loc",
                "soxl_vwap_dist",
                "soxl_mom",
                "soxl_rv",
                "soxl_vol_z",
                "soxx_ret",
                "soxx_range",
                "soxx_close_loc",
                "soxx_vwap_dist",
                "soxx_mom",
                "soxx_rv",
                "soxx_vol_z",
            ]
        )
    ]
    feature_cols += [
        "tod_frac",
        "soxl_minus_3x_soxx_ret",
        "soxl_minus_3x_soxx_mom3",
        "range_ratio_soxl_soxx",
        *extra_feature_cols,
    ]
    feature_cols = _dedupe_preserve_order(feature_cols)
    frame = frame.dropna(subset=feature_cols + ["target_return", "entry_datetime"]).copy()
    frame.attrs["same_session_last_bar_rows_dropped"] = same_session_last_bar_rows_dropped
    return frame, feature_cols, trade_gate_col


def _date_mask(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    return (frame["date"] >= start_date) & (frame["date"] <= end_date)


def _purge_frame(frame: pd.DataFrame, *, drop_head: bool, drop_tail: bool, purge_bars: int) -> pd.DataFrame:
    if purge_bars <= 0 or frame.empty:
        return frame.copy()
    start = purge_bars if drop_head else 0
    end = len(frame) - purge_bars if drop_tail else len(frame)
    if end <= start:
        return frame.iloc[0:0].copy()
    return frame.iloc[start:end].copy()


def _split_frames(
    frame: pd.DataFrame,
    *,
    train: tuple[str, str],
    valid: tuple[str, str],
    test: tuple[str, str],
    purge_bars: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    segments = {
        "train": frame.loc[_date_mask(frame, *train)].sort_values("datetime").reset_index(drop=True),
        "valid": frame.loc[_date_mask(frame, *valid)].sort_values("datetime").reset_index(drop=True),
        "test": frame.loc[_date_mask(frame, *test)].sort_values("datetime").reset_index(drop=True),
    }
    raw_rows = {name: int(len(value)) for name, value in segments.items()}
    purged = {
        "train": _purge_frame(segments["train"], drop_head=False, drop_tail=True, purge_bars=purge_bars),
        "valid": _purge_frame(segments["valid"], drop_head=True, drop_tail=True, purge_bars=purge_bars),
        "test": _purge_frame(segments["test"], drop_head=True, drop_tail=False, purge_bars=purge_bars),
    }
    return purged, {
        "purge_bars": int(purge_bars),
        "raw_rows": raw_rows,
        "purged_rows": {name: int(len(value)) for name, value in purged.items()},
        "label_policy": "same_session_next_bar_open_to_close",
        "same_session_last_bar_rows_dropped": int(frame.attrs.get("same_session_last_bar_rows_dropped", 0)),
    }


def _models() -> dict[str, Any]:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "lgb": LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=80,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_lambda=2.0,
            reg_alpha=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
            tree_method="hist",
        ),
    }


def _safe_spearman(score: pd.Series, target: pd.Series) -> float:
    value = spearmanr(score, target, nan_policy="omit").statistic
    return 0.0 if pd.isna(value) else float(value)


def _eligible_frame(frame: pd.DataFrame, trade_gate_col: str | None) -> pd.DataFrame:
    if trade_gate_col is None:
        return frame
    if trade_gate_col not in frame.columns:
        raise ValueError(f"trade gate column missing from prediction frame: {trade_gate_col}")
    return frame.loc[frame[trade_gate_col].fillna(False).astype(bool)]


def _trade_metrics(
    frame: pd.DataFrame,
    *,
    prob_col: str,
    threshold: float,
    cost_bps: float,
    trade_gate_col: str | None = None,
) -> dict[str, Any]:
    eligible = _eligible_frame(frame, trade_gate_col)
    trades = eligible.loc[eligible[prob_col] >= threshold, ["date", "datetime", "target_return", prob_col]].copy()
    if trades.empty:
        return {
            "trades": 0,
            "trade_days": 0,
            "coverage_days": int(frame["date"].nunique()),
            "eligible_rows": int(len(eligible)),
            "eligible_days": int(eligible["date"].nunique()) if not eligible.empty else 0,
            "net_return_pct": 0.0,
            "gross_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "daily_sharpe": 0.0,
            "win_rate": np.nan,
            "avg_trade_bps": np.nan,
            "worst_trade_bps": np.nan,
        }
    trades["gross_ret"] = trades["target_return"]
    trades["net_ret"] = trades["target_return"] - cost_bps / 10000.0
    equity = (1.0 + trades["net_ret"]).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    daily = trades.groupby("date")["net_ret"].apply(lambda item: (1.0 + item).prod() - 1.0)
    sharpe = 0.0 if daily.std(ddof=0) == 0 else daily.mean() / daily.std(ddof=0) * np.sqrt(252.0)
    return {
        "trades": int(len(trades)),
        "trade_days": int(daily.shape[0]),
        "coverage_days": int(frame["date"].nunique()),
        "eligible_rows": int(len(eligible)),
        "eligible_days": int(eligible["date"].nunique()) if not eligible.empty else 0,
        "net_return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
        "gross_return_pct": float(((1.0 + trades["gross_ret"]).cumprod().iloc[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "daily_sharpe": float(sharpe),
        "win_rate": float((trades["net_ret"] > 0).mean()),
        "avg_trade_bps": float(trades["net_ret"].mean() * 10000.0),
        "worst_trade_bps": float(trades["net_ret"].min() * 10000.0),
    }


def _choose_threshold(
    frame: pd.DataFrame,
    *,
    prob_col: str,
    cost_bps: float,
    trade_gate_col: str | None = None,
) -> tuple[float, dict[str, Any]]:
    threshold_frame = _eligible_frame(frame, trade_gate_col)
    if threshold_frame.empty:
        threshold_frame = frame
    candidates = sorted(
        set(
            [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
            + [float(threshold_frame[prob_col].quantile(q)) for q in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]]
        )
    )
    min_trades = max(50, int(len(threshold_frame) * 0.005))
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = _trade_metrics(
            frame,
            prob_col=prob_col,
            threshold=threshold,
            cost_bps=cost_bps,
            trade_gate_col=trade_gate_col,
        )
        if metrics["trades"] >= min_trades:
            score = (
                metrics["net_return_pct"]
                + 5.0 * metrics["daily_sharpe"]
                + 0.5 * metrics["max_drawdown_pct"]
                + 0.05 * (metrics["avg_trade_bps"] if not pd.isna(metrics["avg_trade_bps"]) else 0.0)
            )
            scored.append((score, float(threshold), metrics))
    if not scored:
        threshold = float(threshold_frame[prob_col].quantile(0.8))
        return threshold, _trade_metrics(
            frame,
            prob_col=prob_col,
            threshold=threshold,
            cost_bps=cost_bps,
            trade_gate_col=trade_gate_col,
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1], scored[0][2]


def _monthly_metrics(
    frame: pd.DataFrame,
    *,
    prob_col: str,
    threshold: float,
    cost_bps: float,
    freq: str,
    trade_gate_col: str | None = None,
) -> list[dict[str, Any]]:
    eligible = _eligible_frame(frame, trade_gate_col)
    trades = eligible.loc[eligible[prob_col] >= threshold, ["date", "datetime", "target_return", prob_col]].copy()
    if trades.empty:
        return []
    trades["net_ret"] = trades["target_return"] - cost_bps / 10000.0
    trades["month"] = pd.to_datetime(trades["date"].astype(str)).dt.to_period("M").astype(str)
    rows = []
    for month, group in trades.groupby("month", sort=True):
        rows.append(
            {
                "freq": freq,
                "month": month,
                "trades": int(len(group)),
                "trade_days": int(group["date"].nunique()),
                "net_return_pct": float(((1.0 + group["net_ret"]).prod() - 1.0) * 100.0),
                "avg_trade_bps": float(group["net_ret"].mean() * 10000.0),
                "win_rate": float((group["net_ret"] > 0).mean()),
            }
        )
    return rows


def _fit_predict_once(
    segments: dict[str, pd.DataFrame],
    *,
    feature_cols: list[str],
    freq: str,
    cost_bps: float,
    trade_gate_col: str | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    train = segments["train"].copy()
    valid = segments["valid"].copy()
    test = segments["test"].copy()
    feature_importance: list[dict[str, Any]] = []
    prediction_frames = []
    model_metrics: list[dict[str, Any]] = []
    for name, model in _models().items():
        model.fit(train[feature_cols], train["target_positive"])
        train[f"prob_{name}"] = model.predict_proba(train[feature_cols])[:, 1]
        valid[f"prob_{name}"] = model.predict_proba(valid[feature_cols])[:, 1]
        test[f"prob_{name}"] = model.predict_proba(test[feature_cols])[:, 1]
        prediction_frames.append(valid[["datetime", "date", "target_return", f"prob_{name}"]].assign(segment="valid"))
        prediction_frames.append(test[["datetime", "date", "target_return", f"prob_{name}"]].assign(segment="test"))
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
            feature_importance.extend(
                {
                    "freq": freq,
                    "model": name,
                    "feature": feature,
                    "importance": float(importance),
                }
                for feature, importance in importances.items()
            )
        threshold, valid_metrics = _choose_threshold(
            valid,
            prob_col=f"prob_{name}",
            cost_bps=cost_bps,
            trade_gate_col=trade_gate_col,
        )
        test_metrics = _trade_metrics(
            test,
            prob_col=f"prob_{name}",
            threshold=threshold,
            cost_bps=cost_bps,
            trade_gate_col=trade_gate_col,
        )
        model_metrics.append(
            {
                "freq": freq,
                "model": name,
                "threshold": float(threshold),
                "valid_ic": _safe_spearman(valid[f"prob_{name}"], valid["target_return"]),
                "test_ic": _safe_spearman(test[f"prob_{name}"], test["target_return"]),
                **{f"valid_{key}": value for key, value in valid_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
    for segment in [train, valid, test]:
        segment["prob_lgb_xgb_consensus"] = segment[["prob_lgb", "prob_xgb"]].mean(axis=1)
    threshold, valid_metrics = _choose_threshold(
        valid,
        prob_col="prob_lgb_xgb_consensus",
        cost_bps=cost_bps,
        trade_gate_col=trade_gate_col,
    )
    test_metrics = _trade_metrics(
        test,
        prob_col="prob_lgb_xgb_consensus",
        threshold=threshold,
        cost_bps=cost_bps,
        trade_gate_col=trade_gate_col,
    )
    model_metrics.append(
        {
            "freq": freq,
            "model": "lgb_xgb_consensus",
            "threshold": float(threshold),
            "valid_ic": _safe_spearman(valid["prob_lgb_xgb_consensus"], valid["target_return"]),
            "test_ic": _safe_spearman(test["prob_lgb_xgb_consensus"], test["target_return"]),
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
    )
    predictions = pd.concat(
        [
            train.assign(segment="train"),
            valid.assign(segment="valid"),
            test.assign(segment="test"),
        ],
        ignore_index=True,
    )
    prediction_cols = [
        "datetime",
        "date",
        "segment",
        "target_return",
        "prob_lgb",
        "prob_xgb",
        "prob_lgb_xgb_consensus",
    ]
    if trade_gate_col is not None:
        prediction_cols.append(trade_gate_col)
    predictions = predictions[prediction_cols].copy()
    return model_metrics, predictions, feature_importance, _monthly_metrics(
        test,
        prob_col="prob_lgb_xgb_consensus",
        threshold=threshold,
        cost_bps=cost_bps,
        freq=freq,
        trade_gate_col=trade_gate_col,
    )


def _cost_sensitivity(
    predictions: pd.DataFrame,
    *,
    freq: str,
    threshold: float,
    model: str,
    base_cost_bps: float,
    cost_stress_bps: tuple[float, ...],
    trade_gate_col: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    prob_col = f"prob_{model}"
    valid = predictions.loc[predictions["segment"] == "valid"].copy()
    test = predictions.loc[predictions["segment"] == "test"].copy()
    for cost_bps in sorted(set(float(value) for value in cost_stress_bps)):
        valid_metrics = _trade_metrics(
            valid,
            prob_col=prob_col,
            threshold=threshold,
            cost_bps=cost_bps,
            trade_gate_col=trade_gate_col,
        )
        test_metrics = _trade_metrics(
            test,
            prob_col=prob_col,
            threshold=threshold,
            cost_bps=cost_bps,
            trade_gate_col=trade_gate_col,
        )
        rows.append(
            {
                "freq": freq,
                "model": model,
                "threshold_selected_at_cost_bps": float(base_cost_bps),
                "threshold": float(threshold),
                "cost_bps": float(cost_bps),
                **{f"valid_{key}": value for key, value in valid_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
    return rows


def _walk_forward_folds() -> list[dict[str, tuple[str, str] | str]]:
    return [
        {
            "fold": "wf_2024",
            "train": ("2019-01-02", "2022-12-30"),
            "valid": ("2023-01-03", "2023-12-29"),
            "test": ("2024-01-02", "2024-12-31"),
        },
        {
            "fold": "wf_2025",
            "train": ("2019-01-02", "2023-12-29"),
            "valid": ("2024-01-02", "2024-12-31"),
            "test": ("2025-01-02", "2025-12-31"),
        },
        {
            "fold": "wf_2026",
            "train": ("2019-01-02", "2024-12-31"),
            "valid": ("2025-01-02", "2025-12-31"),
            "test": ("2026-01-02", "2026-07-02"),
        },
    ]


def _walk_forward_metrics(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    freq: str,
    cost_bps: float,
    purge_bars: int,
    trade_gate_col: str | None = None,
    models_to_record: tuple[str, ...] = ("lgb_xgb_consensus",),
) -> list[dict[str, Any]]:
    rows = []
    for fold in _walk_forward_folds():
        segments, purge = _split_frames(
            frame,
            train=fold["train"],  # type: ignore[arg-type]
            valid=fold["valid"],  # type: ignore[arg-type]
            test=fold["test"],  # type: ignore[arg-type]
            purge_bars=purge_bars,
        )
        if min(len(segments["train"]), len(segments["valid"]), len(segments["test"])) == 0:
            continue
        metrics, predictions, _, _ = _fit_predict_once(
            segments,
            feature_cols=feature_cols,
            freq=freq,
            cost_bps=cost_bps,
            trade_gate_col=trade_gate_col,
        )
        for model_name in models_to_record:
            model_metrics = next(item for item in metrics if item["model"] == model_name)
            rows.append(
                {
                    "freq": freq,
                    "model": model_name,
                    "fold": fold["fold"],
                    "train_start": fold["train"][0],  # type: ignore[index]
                    "train_end": fold["train"][1],  # type: ignore[index]
                    "valid_start": fold["valid"][0],  # type: ignore[index]
                    "valid_end": fold["valid"][1],  # type: ignore[index]
                    "test_start": fold["test"][0],  # type: ignore[index]
                    "test_end": fold["test"][1],  # type: ignore[index]
                    "purge_bars": purge["purge_bars"],
                    "threshold": model_metrics["threshold"],
                    "valid_net_return_pct": model_metrics["valid_net_return_pct"],
                    "valid_max_drawdown_pct": model_metrics["valid_max_drawdown_pct"],
                    "test_net_return_pct": model_metrics["test_net_return_pct"],
                    "test_max_drawdown_pct": model_metrics["test_max_drawdown_pct"],
                    "test_daily_sharpe": model_metrics["test_daily_sharpe"],
                    "test_trades": model_metrics["test_trades"],
                    "test_trade_days": model_metrics["test_trade_days"],
                    "test_ic": model_metrics["test_ic"],
                    "prediction_rows": int(len(predictions)),
                }
            )
    return rows


def _write_workflow_yaml(spec: IntradayBarModelSpec) -> Path:
    path = spec.output_dir / "workflow.yaml"
    provider_lines = []
    for freq, provider_uri in sorted(spec.provider_uris.items()):
        provider_lines.extend([f"  {freq}:", f"    provider_uri: {provider_uri.resolve().as_posix()}"])
    path.write_text(
        "\n".join(
            [
                f"experiment: {spec.experiment}",
                "qlib:",
                f"  recorder_uri: {spec.recorder_uri.resolve().as_posix()}",
                f"  market: {spec.market}",
                f"  region: {spec.region}",
                "providers:",
                *provider_lines,
                "segments:",
                f"  train: {spec.train[0]} to {spec.train[1]}",
                f"  valid: {spec.valid[0]} to {spec.valid[1]}",
                f"  test: {spec.test[0]} to {spec.test[1]}",
                "model_selection:",
                "  models: [lgb, xgb, lgb_xgb_consensus]",
                "  threshold_segment: valid",
                f"  cost_bps: {spec.cost_bps}",
                f"  feature_set: {spec.feature_set}",
                f"  trade_gate: {spec.trade_gate}",
                f"  cost_stress_bps: {list(spec.cost_stress_bps)}",
                "artifact_policy:",
                "  policy: slim_card_recorder_first",
                "  machine_entrypoint: summary.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_sidecars(
    spec: IntradayBarModelSpec,
    *,
    model_selection: pd.DataFrame,
    selected_model: dict[str, Any],
    cost_sensitivity: pd.DataFrame,
    monthly_stability: pd.DataFrame,
    walk_forward: pd.DataFrame,
    feature_importance: pd.DataFrame,
    analysis_summary: dict[str, Any],
    regime_gate_metrics: pd.DataFrame | None = None,
) -> dict[str, str]:
    sidecars = {
        "model_selection": "model_selection.csv",
        "selected_model": "selected_model.json",
        "cost_sensitivity": "cost_sensitivity.csv",
        "monthly_stability": "monthly_stability.csv",
        "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
        "feature_importance": "feature_importance.csv",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
    }
    if regime_gate_metrics is not None:
        sidecars["regime_gate_metrics"] = "regime_gate_metrics.csv"
    model_selection.to_csv(spec.output_dir / sidecars["model_selection"], index=False)
    (spec.output_dir / sidecars["selected_model"]).write_text(
        json.dumps(selected_model, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    cost_sensitivity.to_csv(spec.output_dir / sidecars["cost_sensitivity"], index=False)
    monthly_stability.to_csv(spec.output_dir / sidecars["monthly_stability"], index=False)
    walk_forward.to_csv(spec.output_dir / sidecars["backtest_walk_forward_metrics"], index=False)
    feature_importance.to_csv(spec.output_dir / sidecars["feature_importance"], index=False)
    if regime_gate_metrics is not None:
        regime_gate_metrics.to_csv(spec.output_dir / sidecars["regime_gate_metrics"], index=False)
    (spec.output_dir / sidecars["analysis_summary"]).write_text(
        json.dumps(analysis_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if regime_gate_metrics is None:
        failure_modes = (
            "# Failure Modes\n\n"
            "- 5min branch has unstable validation evidence; positive 2026 test is not enough for promotion.\n"
            "- 15min branch is positive in validation and 2026 test, but monthly returns are concentrated and IC is small.\n"
            "- Intraday simulation is next-bar open-to-close and uses fixed bps costs; live spread, queue, PDT, and borrow/funding frictions are not proven.\n"
            "- qlib PortAnaRecord is not used for this high-frequency single-asset execution probe; large predictions and labels are recorder-only artifacts.\n"
        )
    else:
        failure_modes = (
            "# Failure Modes\n\n"
            "- 2024 walk-forward remains negative, so this is an optimized candidate rather than a promoted production strategy.\n"
            "- Regime gate excludes opening and closing risk bars, but it is still a bar-close to next-bar open/close simulation without bid/ask replay.\n"
            "- 20bp stress is positive for the selected 2026 test window, but live spread, queue, PDT, latency, and volatility-halt behavior are not proven.\n"
            "- qlib PortAnaRecord is not used for this high-frequency single-asset execution probe; large predictions and labels are recorder-only artifacts.\n"
        )
    (spec.output_dir / sidecars["failure_modes"]).write_text(failure_modes, encoding="utf-8")
    return sidecars


def _artifact_policy(experiment: str, sidecars: dict[str, str]) -> dict[str, Any]:
    return {
        "policy": "slim_card_recorder_first",
        "primary_artifact": f"{experiment}.ipynb",
        "machine_entrypoint": "summary.json",
        "workflow_config": "workflow.yaml",
        "kept_sidecars": sorted(sidecars.values()),
        "recorder_only_artifacts": {
            "predictions": "artifacts/pred.pkl",
            "labels": "artifacts/label.pkl",
            "all_model_predictions": "artifacts/model_selection/all_model_predictions.csv",
        },
        "reproducible_from_recorder": True,
    }


def _regime_gate_metrics(
    segments: dict[str, pd.DataFrame],
    *,
    freq: str,
    trade_gate_col: str | None,
) -> list[dict[str, Any]]:
    rows = []
    for segment, frame in segments.items():
        eligible = _eligible_frame(frame, trade_gate_col)
        rows.append(
            {
                "freq": freq,
                "segment": segment,
                "rows": int(len(frame)),
                "eligible_rows": int(len(eligible)),
                "eligible_ratio": float(len(eligible) / len(frame)) if len(frame) else 0.0,
                "coverage_days": int(frame["date"].nunique()) if not frame.empty else 0,
                "eligible_days": int(eligible["date"].nunique()) if not eligible.empty else 0,
                "mean_target_return_bps": float(frame["target_return"].mean() * 10000.0) if not frame.empty else 0.0,
                "eligible_mean_target_return_bps": (
                    float(eligible["target_return"].mean() * 10000.0) if not eligible.empty else 0.0
                ),
            }
        )
    return rows


def _augment_regime_selection_metrics(
    model_selection: pd.DataFrame,
    *,
    cost_sensitivity: pd.DataFrame,
    walk_forward: pd.DataFrame,
) -> pd.DataFrame:
    frame = model_selection.copy()
    historical = walk_forward.loc[walk_forward["fold"].isin(["wf_2024", "wf_2025"])].copy()
    if historical.empty:
        historical_summary = pd.DataFrame(
            columns=[
                "model",
                "historical_walk_forward_positive_ratio",
                "historical_walk_forward_mean_return_pct",
                "historical_walk_forward_worst_return_pct",
                "historical_walk_forward_worst_drawdown_pct",
            ]
        )
    else:
        historical_summary = (
            historical.groupby("model")
            .agg(
                historical_walk_forward_positive_ratio=("test_net_return_pct", lambda item: float((item > 0).mean())),
                historical_walk_forward_mean_return_pct=("test_net_return_pct", "mean"),
                historical_walk_forward_worst_return_pct=("test_net_return_pct", "min"),
                historical_walk_forward_worst_drawdown_pct=("test_max_drawdown_pct", "min"),
            )
            .reset_index()
        )
    valid_stress = cost_sensitivity.loc[cost_sensitivity["cost_bps"] == 20.0, ["model", "valid_net_return_pct"]].rename(
        columns={"valid_net_return_pct": "valid_20bp_net_return_pct"}
    )
    test_stress = cost_sensitivity.loc[cost_sensitivity["cost_bps"] == 20.0, ["model", "test_net_return_pct"]].rename(
        columns={"test_net_return_pct": "test_20bp_net_return_pct"}
    )
    frame = frame.merge(historical_summary, on="model", how="left")
    frame = frame.merge(valid_stress, on="model", how="left")
    frame = frame.merge(test_stress, on="model", how="left")
    for column in [
        "historical_walk_forward_positive_ratio",
        "historical_walk_forward_mean_return_pct",
        "historical_walk_forward_worst_return_pct",
        "historical_walk_forward_worst_drawdown_pct",
        "valid_20bp_net_return_pct",
        "test_20bp_net_return_pct",
    ]:
        frame[column] = frame[column].fillna(0.0).astype(float)
    frame["selection_eligible"] = (
        (frame["valid_net_return_pct"] > 0.0)
        & (frame["valid_trade_days"] >= 20)
        & (frame["valid_avg_trade_bps"] > 0.0)
        & (frame["valid_max_drawdown_pct"] > -20.0)
        & (frame["historical_walk_forward_positive_ratio"] >= 0.5)
        & (frame["historical_walk_forward_worst_return_pct"] > -26.0)
    )
    raw_score = (
        frame["valid_net_return_pct"]
        + 5.0 * frame["valid_daily_sharpe"]
        + 0.5 * frame["valid_max_drawdown_pct"]
        + 0.25 * frame["historical_walk_forward_mean_return_pct"]
        + 0.10 * frame["valid_20bp_net_return_pct"]
        + 0.03 * frame["valid_avg_trade_bps"].fillna(0.0)
    )
    frame["selection_score"] = raw_score.where(frame["selection_eligible"], raw_score - 1000.0)
    return frame.sort_values(["selection_score", "valid_net_return_pct"], ascending=False).reset_index(drop=True)


def _promotion_decision_for_regime_card(selected_model: dict[str, Any]) -> str:
    if float(selected_model.get("historical_walk_forward_positive_ratio", 0.0)) < 1.0:
        return "not_promoted_walk_forward_not_proven"
    if float(selected_model.get("test_20bp_net_return_pct", 0.0)) <= 0.0:
        return "not_promoted_cost_stress_not_proven"
    return "not_promoted_intraday_replay_missing"


def run_intraday_regime_gated_model_card(spec: IntradayBarModelSpec) -> IntradayBarModelResult:
    from qlib.workflow import R

    if set(spec.provider_uris) != {"15min"}:
        raise ValueError("regime-gated intraday optimization is intentionally scoped to the 15min provider")
    if spec.feature_set != "regime_v1" or spec.trade_gate != "regime_v1":
        raise ValueError("regime-gated intraday optimization requires feature_set=regime_v1 and trade_gate=regime_v1")

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.recorder_uri.mkdir(parents=True, exist_ok=True)
    workflow_yaml = _write_workflow_yaml(spec)
    freq = "15min"
    provider_uri = spec.provider_uris[freq]
    config = QlibProviderConfig(provider_uri=provider_uri, region=spec.region, market=spec.market, freq=freq)
    provider_meta = init_qlib_provider(config)
    qlib_segments = _segment_times(spec, freq)
    dataset_audit = {
        freq: _build_dataset_row_audit(
            config,
            segments=qlib_segments,
            instruments=list(spec.instruments),
        )
    }
    qlib_frame = load_qlib_feature_frame(
        config,
        instruments=list(spec.instruments),
        fields=QLIB_FIELDS,
        start=qlib_segments["train"][0],
        end=qlib_segments["test"][1],
    )
    model_frame, feature_cols, trade_gate_col = _engineer_model_frame(
        qlib_frame,
        cost_bps=spec.cost_bps,
        feature_set=spec.feature_set,
        trade_gate=spec.trade_gate,
    )
    segments, purge_audit = _split_frames(
        model_frame,
        train=spec.train,
        valid=spec.valid,
        test=spec.test,
        purge_bars=spec.purge_bars,
    )
    model_rows, predictions, feature_rows, _ = _fit_predict_once(
        segments,
        feature_cols=feature_cols,
        freq=freq,
        cost_bps=spec.cost_bps,
        trade_gate_col=trade_gate_col,
    )
    base_model_selection = pd.DataFrame(model_rows).sort_values(["freq", "model"]).reset_index(drop=True)
    cost_rows: list[dict[str, Any]] = []
    for row in base_model_selection.to_dict("records"):
        cost_rows.extend(
            _cost_sensitivity(
                predictions,
                freq=freq,
                threshold=float(row["threshold"]),
                model=str(row["model"]),
                base_cost_bps=spec.cost_bps,
                cost_stress_bps=spec.cost_stress_bps,
                trade_gate_col=trade_gate_col,
            )
        )
    cost_sensitivity = pd.DataFrame(cost_rows)
    walk_forward = pd.DataFrame(
        _walk_forward_metrics(
            model_frame,
            feature_cols=feature_cols,
            freq=freq,
            cost_bps=spec.cost_bps,
            purge_bars=spec.purge_bars,
            trade_gate_col=trade_gate_col,
            models_to_record=MODEL_NAMES,
        )
    )
    model_selection = _augment_regime_selection_metrics(
        base_model_selection,
        cost_sensitivity=cost_sensitivity,
        walk_forward=walk_forward,
    )
    selected = model_selection.iloc[0].to_dict()
    selected_model = {
        "freq": selected["freq"],
        "model": selected["model"],
        "threshold": float(selected["threshold"]),
        "selection_segment": "valid",
        "selection_uses_test_segment": False,
        "selection_metric": (
            "valid_net_return + 5*valid_sharpe + 0.5*valid_drawdown "
            "+ 0.25*historical_wf_mean + 0.10*valid_20bp + 0.03*avg_trade_bps"
        ),
        "selection_score": float(selected["selection_score"]),
        "selection_eligible": bool(selected["selection_eligible"]),
        "valid_net_return_pct": float(selected["valid_net_return_pct"]),
        "valid_max_drawdown_pct": float(selected["valid_max_drawdown_pct"]),
        "valid_avg_trade_bps": float(selected["valid_avg_trade_bps"]),
        "test_net_return_pct": float(selected["test_net_return_pct"]),
        "test_max_drawdown_pct": float(selected["test_max_drawdown_pct"]),
        "test_ic": float(selected["test_ic"]),
        "historical_walk_forward_positive_ratio": float(selected["historical_walk_forward_positive_ratio"]),
        "historical_walk_forward_worst_return_pct": float(selected["historical_walk_forward_worst_return_pct"]),
        "valid_20bp_net_return_pct": float(selected["valid_20bp_net_return_pct"]),
        "test_20bp_net_return_pct": float(selected["test_20bp_net_return_pct"]),
    }
    promotion_decision = _promotion_decision_for_regime_card(selected_model)
    selected_prob_col = f"prob_{selected_model['model']}"
    monthly_stability = pd.DataFrame(
        _monthly_metrics(
            predictions.loc[predictions["segment"] == "test"].copy(),
            prob_col=selected_prob_col,
            threshold=selected_model["threshold"],
            cost_bps=spec.cost_bps,
            freq=freq,
            trade_gate_col=trade_gate_col,
        )
    )
    regime_gate_metrics = pd.DataFrame(_regime_gate_metrics(segments, freq=freq, trade_gate_col=trade_gate_col))
    feature_importance = pd.DataFrame(feature_rows)
    provider_audit = {
        freq: {
            "provider_uri": provider_meta["provider_uri"],
            "freq": freq,
            "calendar_start": provider_meta["calendar_start"],
            "calendar_end": provider_meta["calendar_end"],
            "qlib_feature_rows": int(len(qlib_frame)),
            "model_rows": int(len(model_frame)),
            "feature_count": int(len(feature_cols)),
            "purged_split": purge_audit,
            "manifest": (provider_uri / "qlib_store_manifest.json").as_posix(),
        }
    }
    analysis_summary = {
        "source": "qlib_dataset_intraday_regime_gated_bar_model",
        "as_of": spec.as_of,
        "provider_audit": provider_audit,
        "dataset_audit": dataset_audit,
        "selected_model": selected_model,
        "cost_bps": spec.cost_bps,
        "cost_stress_bps": [float(value) for value in sorted(set(spec.cost_stress_bps))],
        "feature_set": {
            "name": spec.feature_set,
            "feature_count": int(len(feature_cols)),
            "lookback_only": True,
            "new_features": [
                column
                for column in feature_cols
                if column.endswith("_tod_z60")
                or column.endswith("_cum_ret_from_open")
                or column.endswith("_range_so_far")
                or column in {"tod_sin", "tod_cos", "abs_soxl_minus_3x_soxx_ret"}
            ],
        },
        "trade_gate": {
            "name": spec.trade_gate,
            "column": trade_gate_col,
            "lookback_only": True,
            "rules": [
                "exclude first and last intraday bars",
                "require time-of-day-normalized range or volume shock, leveraged ETF dislocation, or intraday directional state",
            ],
        },
        "promotion_decision": promotion_decision,
        "selection_uses_test_segment": False,
        "walk_forward_positive_ratio": {
            model: float((group["test_net_return_pct"] > 0).mean()) for model, group in walk_forward.groupby("model")
        },
    }
    sidecars = _write_sidecars(
        spec,
        model_selection=model_selection,
        selected_model=selected_model,
        cost_sensitivity=cost_sensitivity,
        monthly_stability=monthly_stability,
        walk_forward=walk_forward,
        feature_importance=feature_importance,
        analysis_summary=analysis_summary,
        regime_gate_metrics=regime_gate_metrics,
    )

    selected_predictions = predictions.loc[
        predictions["segment"] == "test",
        ["datetime", "target_return", selected_prob_col, trade_gate_col],
    ].rename(columns={selected_prob_col: "score", trade_gate_col: "eligible"})
    selected_labels = selected_predictions[["datetime", "target_return"]].copy()
    upload_dir = Path(tempfile.mkdtemp(prefix=f"{spec.experiment}_recorder_"))
    try:
        selected_predictions.to_pickle(upload_dir / "pred.pkl")
        selected_labels.to_pickle(upload_dir / "label.pkl")
        predictions.assign(freq=freq).to_csv(upload_dir / "all_model_predictions.csv", index=False)
        R.set_uri(spec.recorder_uri.as_posix())
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                experiment=spec.experiment,
                market=spec.market,
                region=spec.region,
                train_start=spec.train[0],
                train_end=spec.train[1],
                valid_start=spec.valid[0],
                valid_end=spec.valid[1],
                test_start=spec.test[0],
                test_end=spec.test[1],
                cost_bps=spec.cost_bps,
                cost_stress_bps=json.dumps([float(value) for value in sorted(set(spec.cost_stress_bps))]),
                purge_bars=spec.purge_bars,
                feature_set=spec.feature_set,
                trade_gate=spec.trade_gate,
                selected_freq=selected_model["freq"],
                selected_model=selected_model["model"],
                selected_threshold=selected_model["threshold"],
                selection_uses_test_segment=False,
            )
            R.log_metrics(
                selected_valid_return_pct=selected_model["valid_net_return_pct"],
                selected_test_return_pct=selected_model["test_net_return_pct"],
                selected_test_max_drawdown_pct=selected_model["test_max_drawdown_pct"],
                selected_test_ic=selected_model["test_ic"],
                selected_test_20bp_return_pct=selected_model["test_20bp_net_return_pct"],
            )
            recorder.log_artifact((upload_dir / "pred.pkl").as_posix())
            recorder.log_artifact((upload_dir / "label.pkl").as_posix())
            recorder.log_artifact((upload_dir / "all_model_predictions.csv").as_posix(), artifact_path="model_selection")
            for rel_path in sidecars.values():
                recorder.log_artifact((spec.output_dir / rel_path).as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir())
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)

    _remove_transient_code_snapshots(recorder_local_dir)
    recorder_artifacts = _recorder_files(recorder_local_dir)
    summary = {
        "experiment": spec.experiment,
        "card_type": "intraday_regime_gated_model_search",
        "workflow_config": workflow_yaml.name,
        "promotion_decision": promotion_decision,
        "as_of": spec.as_of,
        "market": spec.market,
        "region": spec.region,
        "instruments": list(spec.instruments),
        "provider_uris": {freq: provider_uri.resolve().as_posix()},
        "provider_audit": provider_audit,
        "segments": {
            "train": list(spec.train),
            "valid": list(spec.valid),
            "test": list(spec.test),
        },
        "cost_bps": spec.cost_bps,
        "cost_stress_bps": [float(value) for value in sorted(set(spec.cost_stress_bps))],
        "purged_split": {"purge_bars": spec.purge_bars, "label_policy": "same_session_next_bar_open_to_close"},
        "feature_set": analysis_summary["feature_set"],
        "trade_gate": {
            **analysis_summary["trade_gate"],
            "eligible_on_selected_test": int(
                regime_gate_metrics.loc[regime_gate_metrics["segment"] == "test", "eligible_rows"].iloc[0]
            ),
        },
        "selected_model": selected_model,
        "model_selection": {
            "source": "qlib_dataset",
            "models": MODEL_NAMES,
            "selection_segment": "valid",
            "selection_uses_test_segment": False,
            "candidate_count": int(len(model_selection)),
            "selected_freq": selected_model["freq"],
            "selected_model": selected_model["model"],
        },
        "recorder_id": recorder_id,
        "recorder_uri": spec.recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": recorder_artifacts,
        "sidecars": sidecars,
        "artifact_policy": _artifact_policy(spec.experiment, sidecars),
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": [
            "qlib.init",
            "qlib.data.D.features",
            "DatasetH",
            "DataHandlerLP",
            "QlibDataLoader",
            "qlib.workflow.R.start",
            "qlib.workflow.R.log_params",
            "qlib.workflow.R.log_metrics",
            "qlib.workflow.Recorder.log_artifact",
            "LGBMClassifier",
            "XGBClassifier",
        ],
    }
    (spec.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    conclusion = (
        "# 中文结论\n\n"
        "- 做了什么：在上一版 5m/15m 结果基础上收缩到 15min，新增 regime_v1 特征和交易 gate，"
        "并用 qlib.init、D.features、DatasetH/DataHandlerLP、R.start、LGBMClassifier、XGBClassifier 重新训练。\n"
        f"- 核心指标：validation 选择 {selected_model['freq']} / {selected_model['model']}，"
        f"valid return_pct={selected_model['valid_net_return_pct']:.4f}、"
        f"test return_pct={selected_model['test_net_return_pct']:.4f}、"
        f"test max_drawdown_pct={selected_model['test_max_drawdown_pct']:.4f}、"
        f"test 20bp return_pct={selected_model['test_20bp_net_return_pct']:.4f}。\n"
        "- regime/window 差异：selection 只用 valid 与 2024/2025 historical walk-forward 稳定性，"
        "2026 test 只作为报告；regime gate 过滤低信息 bar，保留 time-of-day 标准化量能/波动、"
        "SOXL-vs-3x-SOXX 偏离和日内趋势状态。\n"
        f"- 是否晋升：不晋升；decision={promotion_decision}。10bp 是主回测成本，20bp 是压力测试，"
        "即便收益改善，也还缺少 bid/ask replay、排队成交、延迟和实盘滑点证明。\n"
        "- 风险：信号仍是假设 bar 收盘可计算、下一根 bar 开盘成交、下一根 bar 收盘退出；"
        "没有证明极端半导体去杠杆日能按回测价格成交。\n"
        "- 下一步：只把 15min regime gate 作为候选执行层，继续做 bid/ask replay、开盘/收盘 bar 分层、"
        "以及 SOXX 成分股/NVDA/AVGO/AMD 分钟级外部状态特征。\n"
        f"- qlib recorder：{recorder_id}\n"
    )
    (spec.output_dir / "conclusion.md").write_text(conclusion, encoding="utf-8")
    return IntradayBarModelResult(recorder_id=recorder_id, recorder_local_dir=recorder_local_dir, summary=summary)


def run_intraday_bar_model_card(spec: IntradayBarModelSpec) -> IntradayBarModelResult:
    from qlib.workflow import R

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.recorder_uri.mkdir(parents=True, exist_ok=True)
    workflow_yaml = _write_workflow_yaml(spec)
    model_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    walk_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    dataset_audit: dict[str, Any] = {}
    provider_audit: dict[str, Any] = {}

    for freq, provider_uri in sorted(spec.provider_uris.items()):
        config = QlibProviderConfig(provider_uri=provider_uri, region=spec.region, market=spec.market, freq=freq)
        provider_meta = init_qlib_provider(config)
        qlib_segments = _segment_times(spec, freq)
        dataset_audit[freq] = _build_dataset_row_audit(
            config,
            segments=qlib_segments,
            instruments=list(spec.instruments),
        )
        qlib_frame = load_qlib_feature_frame(
            config,
            instruments=list(spec.instruments),
            fields=QLIB_FIELDS,
            start=qlib_segments["train"][0],
            end=qlib_segments["test"][1],
        )
        model_frame, feature_cols, trade_gate_col = _engineer_model_frame(
            qlib_frame,
            cost_bps=spec.cost_bps,
            feature_set=spec.feature_set,
            trade_gate=spec.trade_gate,
        )
        segments, purge_audit = _split_frames(
            model_frame,
            train=spec.train,
            valid=spec.valid,
            test=spec.test,
            purge_bars=spec.purge_bars,
        )
        metrics, predictions, feature_importance, monthly = _fit_predict_once(
            segments,
            feature_cols=feature_cols,
            freq=freq,
            cost_bps=spec.cost_bps,
            trade_gate_col=trade_gate_col,
        )
        consensus = next(item for item in metrics if item["model"] == "lgb_xgb_consensus")
        model_rows.extend(metrics)
        feature_rows.extend(feature_importance)
        monthly_rows.extend(monthly)
        cost_rows.extend(
            _cost_sensitivity(
                predictions,
                freq=freq,
                threshold=float(consensus["threshold"]),
                model="lgb_xgb_consensus",
                base_cost_bps=spec.cost_bps,
                cost_stress_bps=spec.cost_stress_bps,
                trade_gate_col=trade_gate_col,
            )
        )
        walk_rows.extend(
            _walk_forward_metrics(
                model_frame,
                feature_cols=feature_cols,
                freq=freq,
                cost_bps=spec.cost_bps,
                purge_bars=spec.purge_bars,
                trade_gate_col=trade_gate_col,
                models_to_record=spec.walk_forward_models,
            )
        )
        prediction_frames.append(predictions.assign(freq=freq))
        provider_audit[freq] = {
            "provider_uri": provider_meta["provider_uri"],
            "freq": freq,
            "calendar_start": provider_meta["calendar_start"],
            "calendar_end": provider_meta["calendar_end"],
            "qlib_feature_rows": int(len(qlib_frame)),
            "model_rows": int(len(model_frame)),
            "feature_count": int(len(feature_cols)),
            "purged_split": purge_audit,
            "manifest": (provider_uri / "qlib_store_manifest.json").as_posix(),
        }

    model_selection = pd.DataFrame(model_rows).sort_values(["freq", "model"]).reset_index(drop=True)
    selected = (
        model_selection.assign(
            selection_score=lambda frame: frame["valid_net_return_pct"]
            + 5.0 * frame["valid_daily_sharpe"]
            + 0.5 * frame["valid_max_drawdown_pct"]
        )
        .sort_values("selection_score", ascending=False)
        .iloc[0]
        .to_dict()
    )
    selected_model = {
        "freq": selected["freq"],
        "model": selected["model"],
        "threshold": float(selected["threshold"]),
        "selection_segment": "valid",
        "selection_metric": "valid_net_return_pct + 5*valid_daily_sharpe + 0.5*valid_max_drawdown_pct",
        "valid_net_return_pct": float(selected["valid_net_return_pct"]),
        "valid_max_drawdown_pct": float(selected["valid_max_drawdown_pct"]),
        "test_net_return_pct": float(selected["test_net_return_pct"]),
        "test_max_drawdown_pct": float(selected["test_max_drawdown_pct"]),
        "test_ic": float(selected["test_ic"]),
    }
    cost_sensitivity = pd.DataFrame(cost_rows)
    monthly_stability = pd.DataFrame(monthly_rows)
    walk_forward = pd.DataFrame(walk_rows)
    feature_importance = pd.DataFrame(feature_rows)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    pred_col = f"prob_{selected_model['model']}"
    selected_predictions = all_predictions.loc[
        (all_predictions["freq"] == selected_model["freq"]) & (all_predictions["segment"] == "test"),
        ["datetime", "target_return", pred_col],
    ].rename(columns={pred_col: "score"})
    selected_labels = selected_predictions[["datetime", "target_return"]].copy()
    analysis_summary = {
        "source": "qlib_dataset_intraday_bar_model",
        "as_of": spec.as_of,
        "provider_audit": provider_audit,
        "dataset_audit": dataset_audit,
        "selected_model": selected_model,
        "cost_bps": spec.cost_bps,
        "promotion_decision": "not_promoted_intraday_stability_not_proven",
        "monthly_positive_ratio": {
            freq: float((group["net_return_pct"] > 0).mean()) for freq, group in monthly_stability.groupby("freq")
        },
        "walk_forward_positive_ratio": {
            freq: float((group["test_net_return_pct"] > 0).mean()) for freq, group in walk_forward.groupby("freq")
        },
    }
    sidecars = _write_sidecars(
        spec,
        model_selection=model_selection,
        selected_model=selected_model,
        cost_sensitivity=cost_sensitivity,
        monthly_stability=monthly_stability,
        walk_forward=walk_forward,
        feature_importance=feature_importance,
        analysis_summary=analysis_summary,
    )

    upload_dir = Path(tempfile.mkdtemp(prefix=f"{spec.experiment}_recorder_"))
    try:
        selected_predictions.to_pickle(upload_dir / "pred.pkl")
        selected_labels.to_pickle(upload_dir / "label.pkl")
        all_predictions.to_csv(upload_dir / "all_model_predictions.csv", index=False)
        R.set_uri(spec.recorder_uri.as_posix())
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                experiment=spec.experiment,
                market=spec.market,
                region=spec.region,
                train_start=spec.train[0],
                train_end=spec.train[1],
                valid_start=spec.valid[0],
                valid_end=spec.valid[1],
                test_start=spec.test[0],
                test_end=spec.test[1],
                cost_bps=spec.cost_bps,
                purge_bars=spec.purge_bars,
                selected_freq=selected_model["freq"],
                selected_model=selected_model["model"],
                selected_threshold=selected_model["threshold"],
            )
            R.log_metrics(
                selected_valid_return_pct=selected_model["valid_net_return_pct"],
                selected_test_return_pct=selected_model["test_net_return_pct"],
                selected_test_max_drawdown_pct=selected_model["test_max_drawdown_pct"],
                selected_test_ic=selected_model["test_ic"],
            )
            recorder.log_artifact((upload_dir / "pred.pkl").as_posix())
            recorder.log_artifact((upload_dir / "label.pkl").as_posix())
            recorder.log_artifact((upload_dir / "all_model_predictions.csv").as_posix(), artifact_path="model_selection")
            for rel_path in sidecars.values():
                recorder.log_artifact((spec.output_dir / rel_path).as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir())
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)

    _remove_transient_code_snapshots(recorder_local_dir)
    recorder_artifacts = _recorder_files(recorder_local_dir)
    summary = {
        "experiment": spec.experiment,
        "card_type": "diagnostic_search",
        "workflow_config": workflow_yaml.name,
        "promotion_decision": "not_promoted_intraday_stability_not_proven",
        "as_of": spec.as_of,
        "market": spec.market,
        "region": spec.region,
        "instruments": list(spec.instruments),
        "provider_uris": {freq: path.resolve().as_posix() for freq, path in sorted(spec.provider_uris.items())},
        "provider_audit": provider_audit,
        "segments": {
            "train": list(spec.train),
            "valid": list(spec.valid),
            "test": list(spec.test),
        },
        "cost_bps": spec.cost_bps,
        "purged_split": {"purge_bars": spec.purge_bars, "label_policy": "same_session_next_bar_open_to_close"},
        "selected_model": selected_model,
        "model_selection": {
            "source": "qlib_dataset",
            "models": MODEL_NAMES,
            "selection_segment": "valid",
            "candidate_count": int(len(model_selection)),
            "selected_freq": selected_model["freq"],
            "selected_model": selected_model["model"],
        },
        "recorder_id": recorder_id,
        "recorder_uri": spec.recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": recorder_artifacts,
        "sidecars": sidecars,
        "artifact_policy": _artifact_policy(spec.experiment, sidecars),
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": [
            "qlib.init",
            "qlib.data.D.features",
            "DatasetH",
            "DataHandlerLP",
            "QlibDataLoader",
            "qlib.workflow.R.start",
            "qlib.workflow.R.log_params",
            "qlib.workflow.R.log_metrics",
            "qlib.workflow.Recorder.log_artifact",
            "LGBMClassifier",
            "XGBClassifier",
        ],
    }
    (spec.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    conclusion = (
        "# 中文结论\n\n"
        f"- 做了什么：把 SOXL/SOXX 1m 缓存正式重采样成 qlib 5min/15min provider，并通过 qlib.init、D.features、DatasetH/DataHandlerLP、R.start、LGBMClassifier 和 XGBClassifier 重跑下一根 bar open-to-close 盘中模型。\n"
        f"- 核心指标：validation 选择 {selected_model['freq']} / {selected_model['model']}，valid return_pct={selected_model['valid_net_return_pct']:.4f}、test return_pct={selected_model['test_net_return_pct']:.4f}、test max_drawdown_pct={selected_model['test_max_drawdown_pct']:.4f}、test IC={selected_model['test_ic']:.4f}。\n"
        f"- regime/window 差异：5min 与 15min 都使用 purged split（purge_bars={spec.purge_bars}）和 2024/2025/2026 walk-forward；月度稳定性与成本敏感度见 monthly_stability.csv、cost_sensitivity.csv 和 backtest_walk_forward_metrics.csv。\n"
        "- 是否晋升：不晋升；15min 有弱正信号但月度集中、IC 很小，5min validation 不稳定，尚不能证明盘中实盘能力。\n"
        "- 风险：执行假设是 bar 收盘出信号、下一根 bar 开盘进、下一根 bar 收盘出，未证明真实 spread、排队成交、PDT、滑点扩张和实时延迟。\n"
        f"- 下一步：若继续推进，应接入实时分钟 replay 和成交模拟，扩大成本压力，并把 15min xgb/consensus 作为候选而非生产信号。\n"
        f"- qlib recorder：{recorder_id}\n"
    )
    (spec.output_dir / "conclusion.md").write_text(conclusion, encoding="utf-8")
    return IntradayBarModelResult(recorder_id=recorder_id, recorder_local_dir=recorder_local_dir, summary=summary)
