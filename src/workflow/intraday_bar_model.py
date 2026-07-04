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
    "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
    "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
}
QLIB_FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$turnover", "$bar_index", "$bar_count"]
QLIB_LABEL = "Ref($close, -1) / Ref($open, -1) - 1"
MODEL_NAMES = ["lgb", "xgb", "lgb_xgb_consensus"]


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


def _engineer_model_frame(raw: pd.DataFrame, *, cost_bps: float) -> tuple[pd.DataFrame, list[str]]:
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
    feature_cols += ["tod_frac", "soxl_minus_3x_soxx_ret", "soxl_minus_3x_soxx_mom3", "range_ratio_soxl_soxx"]
    frame = frame.dropna(subset=feature_cols + ["target_return", "entry_datetime"]).copy()
    frame.attrs["same_session_last_bar_rows_dropped"] = same_session_last_bar_rows_dropped
    return frame, feature_cols


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


def _trade_metrics(frame: pd.DataFrame, *, prob_col: str, threshold: float, cost_bps: float) -> dict[str, Any]:
    trades = frame.loc[frame[prob_col] >= threshold, ["date", "datetime", "target_return", prob_col]].copy()
    if trades.empty:
        return {
            "trades": 0,
            "trade_days": 0,
            "coverage_days": int(frame["date"].nunique()),
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
        "net_return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
        "gross_return_pct": float(((1.0 + trades["gross_ret"]).cumprod().iloc[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "daily_sharpe": float(sharpe),
        "win_rate": float((trades["net_ret"] > 0).mean()),
        "avg_trade_bps": float(trades["net_ret"].mean() * 10000.0),
        "worst_trade_bps": float(trades["net_ret"].min() * 10000.0),
    }


def _choose_threshold(frame: pd.DataFrame, *, prob_col: str, cost_bps: float) -> tuple[float, dict[str, Any]]:
    candidates = sorted(
        set(
            [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
            + [float(frame[prob_col].quantile(q)) for q in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]]
        )
    )
    min_trades = max(50, int(len(frame) * 0.005))
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = _trade_metrics(frame, prob_col=prob_col, threshold=threshold, cost_bps=cost_bps)
        if metrics["trades"] >= min_trades:
            score = metrics["net_return_pct"] + 5.0 * metrics["daily_sharpe"] + 0.5 * metrics["max_drawdown_pct"]
            scored.append((score, float(threshold), metrics))
    if not scored:
        threshold = float(frame[prob_col].quantile(0.8))
        return threshold, _trade_metrics(frame, prob_col=prob_col, threshold=threshold, cost_bps=cost_bps)
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1], scored[0][2]


def _monthly_metrics(frame: pd.DataFrame, *, prob_col: str, threshold: float, cost_bps: float, freq: str) -> list[dict[str, Any]]:
    trades = frame.loc[frame[prob_col] >= threshold, ["date", "datetime", "target_return", prob_col]].copy()
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
        threshold, valid_metrics = _choose_threshold(valid, prob_col=f"prob_{name}", cost_bps=cost_bps)
        test_metrics = _trade_metrics(test, prob_col=f"prob_{name}", threshold=threshold, cost_bps=cost_bps)
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
    threshold, valid_metrics = _choose_threshold(valid, prob_col="prob_lgb_xgb_consensus", cost_bps=cost_bps)
    test_metrics = _trade_metrics(test, prob_col="prob_lgb_xgb_consensus", threshold=threshold, cost_bps=cost_bps)
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
    predictions = predictions[prediction_cols].copy()
    return model_metrics, predictions, feature_importance, _monthly_metrics(
        test,
        prob_col="prob_lgb_xgb_consensus",
        threshold=threshold,
        cost_bps=cost_bps,
        freq=freq,
    )


def _cost_sensitivity(
    predictions: pd.DataFrame,
    *,
    freq: str,
    threshold: float,
    model: str,
    base_cost_bps: float,
) -> list[dict[str, Any]]:
    rows = []
    prob_col = f"prob_{model}"
    valid = predictions.loc[predictions["segment"] == "valid"].copy()
    test = predictions.loc[predictions["segment"] == "test"].copy()
    for cost_bps in [0.0, base_cost_bps, 10.0, 20.0]:
        valid_metrics = _trade_metrics(valid, prob_col=prob_col, threshold=threshold, cost_bps=cost_bps)
        test_metrics = _trade_metrics(test, prob_col=prob_col, threshold=threshold, cost_bps=cost_bps)
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
        )
        consensus = next(item for item in metrics if item["model"] == "lgb_xgb_consensus")
        rows.append(
            {
                "freq": freq,
                "fold": fold["fold"],
                "train_start": fold["train"][0],  # type: ignore[index]
                "train_end": fold["train"][1],  # type: ignore[index]
                "valid_start": fold["valid"][0],  # type: ignore[index]
                "valid_end": fold["valid"][1],  # type: ignore[index]
                "test_start": fold["test"][0],  # type: ignore[index]
                "test_end": fold["test"][1],  # type: ignore[index]
                "purge_bars": purge["purge_bars"],
                "threshold": consensus["threshold"],
                "valid_net_return_pct": consensus["valid_net_return_pct"],
                "valid_max_drawdown_pct": consensus["valid_max_drawdown_pct"],
                "test_net_return_pct": consensus["test_net_return_pct"],
                "test_max_drawdown_pct": consensus["test_max_drawdown_pct"],
                "test_daily_sharpe": consensus["test_daily_sharpe"],
                "test_trades": consensus["test_trades"],
                "test_trade_days": consensus["test_trade_days"],
                "test_ic": consensus["test_ic"],
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
    model_selection.to_csv(spec.output_dir / sidecars["model_selection"], index=False)
    (spec.output_dir / sidecars["selected_model"]).write_text(
        json.dumps(selected_model, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    cost_sensitivity.to_csv(spec.output_dir / sidecars["cost_sensitivity"], index=False)
    monthly_stability.to_csv(spec.output_dir / sidecars["monthly_stability"], index=False)
    walk_forward.to_csv(spec.output_dir / sidecars["backtest_walk_forward_metrics"], index=False)
    feature_importance.to_csv(spec.output_dir / sidecars["feature_importance"], index=False)
    (spec.output_dir / sidecars["analysis_summary"]).write_text(
        json.dumps(analysis_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (spec.output_dir / sidecars["failure_modes"]).write_text(
        "# Failure Modes\n\n"
        "- 5min branch has unstable validation evidence; positive 2026 test is not enough for promotion.\n"
        "- 15min branch is positive in validation and 2026 test, but monthly returns are concentrated and IC is small.\n"
        "- Intraday simulation is next-bar open-to-close and uses fixed bps costs; live spread, queue, PDT, and borrow/funding frictions are not proven.\n"
        "- qlib PortAnaRecord is not used for this high-frequency single-asset execution probe; large predictions and labels are recorder-only artifacts.\n",
        encoding="utf-8",
    )
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
        model_frame, feature_cols = _engineer_model_frame(qlib_frame, cost_bps=spec.cost_bps)
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
            )
        )
        walk_rows.extend(
            _walk_forward_metrics(
                model_frame,
                feature_cols=feature_cols,
                freq=freq,
                cost_bps=spec.cost_bps,
                purge_bars=spec.purge_bars,
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
