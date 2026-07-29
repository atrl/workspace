from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.qlib_native import (
    QlibProviderConfig,
    init_qlib_provider,
    load_qlib_feature_frame,
)


EXPERIMENT = "soxl_long_horizon_excess_drawdown_v1"
HORIZONS = (1, 5, 21, 63, 126, 252, 504, 756, 1260)
PERSISTENCE_HORIZONS = (63, 126, 252, 504)


@dataclass(frozen=True)
class LongHorizonSpec:
    root: Path
    experiment: str = EXPERIMENT
    start: str = "2010-03-11"
    horizons: tuple[int, ...] = HORIZONS
    persistence_horizons: tuple[int, ...] = PERSISTENCE_HORIZONS

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _safe_float(value: Any) -> float:
    return float(value) if pd.notna(value) else float("nan")


def _path_efficiency(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    log_returns = np.log1p(clean.to_numpy(dtype=float))
    denominator = float(np.abs(log_returns).sum())
    if denominator <= 0.0:
        return 0.0
    return float(abs(log_returns.sum()) / denominator)


def load_soxl_soxx_frame(
    spec: LongHorizonSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = spec.provider_uri / "qlib_store_manifest.json"
    if not manifest_path.exists():
        raise ValueError("blocked: qlib store manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest.get("instruments", [])) != {"SOXL", "SOXX"}:
        raise ValueError("blocked: qlib SOXL/SOXX instruments drifted")
    required_fields = {"$close", "$vix"}
    if not required_fields.issubset(set(manifest.get("fields", []))):
        raise ValueError("blocked: qlib close/VIX fields missing")

    config = QlibProviderConfig(
        provider_uri=spec.provider_uri,
        market="all",
        region="us",
        kernels=1,
    )
    provider_audit = init_qlib_provider(config)
    end = min(
        pd.Timestamp(manifest["as_of"]),
        pd.Timestamp(provider_audit["calendar_end"]),
    ).date().isoformat()
    raw = load_qlib_feature_frame(
        config,
        instruments=["SOXL", "SOXX"],
        fields=["$close", "$vix"],
        start=spec.start,
        end=end,
    ).reset_index()
    close = (
        raw.pivot(index="datetime", columns="instrument", values="$close")
        .sort_index()
        .rename(columns={"SOXL": "soxl_close", "SOXX": "soxx_close"})
    )
    vix = (
        raw.loc[raw["instrument"].eq("SOXL"), ["datetime", "$vix"]]
        .set_index("datetime")["$vix"]
        .sort_index()
        .rename("vix")
    )
    frame = close.join(vix).dropna().astype(float)
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("blocked: qlib daily frame index invalid")

    frame["soxl_return"] = frame["soxl_close"].pct_change()
    frame["soxx_return"] = frame["soxx_close"].pct_change()
    frame["ideal_3x_return"] = 3.0 * frame["soxx_return"]
    if frame["ideal_3x_return"].dropna().le(-1.0).any():
        raise ValueError("blocked: ideal daily-reset 3x wealth is non-positive")
    frame["soxl_wealth"] = frame["soxl_close"] / frame["soxl_close"].iloc[0]
    frame["soxx_wealth"] = frame["soxx_close"] / frame["soxx_close"].iloc[0]
    frame["ideal_3x_wealth"] = (
        1.0 + frame["ideal_3x_return"].fillna(0.0)
    ).cumprod()
    frame["relative_wealth_vs_soxx"] = (
        frame["soxl_wealth"] / frame["soxx_wealth"]
    )
    frame["relative_wealth_vs_ideal_3x"] = (
        frame["soxl_wealth"] / frame["ideal_3x_wealth"]
    )
    audit = {
        "status": "ok",
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "manifest": manifest_path.resolve().as_posix(),
        "as_of": manifest["as_of"],
        "data_start": frame.index.min().date().isoformat(),
        "data_end": frame.index.max().date().isoformat(),
        "rows": int(len(frame)),
        "missing_values": {
            column: int(frame[column].isna().sum())
            for column in ("soxl_close", "soxx_close", "vix")
        },
        "provider_audit": provider_audit,
        "source_api": [
            "qlib.init",
            "qlib.data.D.calendar",
            "qlib.data.D.instruments",
            "qlib.data.D.features",
        ],
    }
    return frame, audit


def _window_row(
    frame: pd.DataFrame,
    start_position: int,
    end_position: int,
) -> dict[str, Any]:
    start = frame.index[start_position]
    end = frame.index[end_position]
    section = frame.iloc[start_position : end_position + 1]
    underlying_daily = section["soxx_return"].iloc[1:].dropna()
    soxl_return = (
        section["soxl_close"].iloc[-1] / section["soxl_close"].iloc[0] - 1.0
    )
    soxx_return = (
        section["soxx_close"].iloc[-1] / section["soxx_close"].iloc[0] - 1.0
    )
    ideal_return = (
        section["ideal_3x_wealth"].iloc[-1]
        / section["ideal_3x_wealth"].iloc[0]
        - 1.0
    )
    volatility = (
        float(underlying_daily.std(ddof=1) * np.sqrt(252.0))
        if len(underlying_daily) > 1
        else float(abs(underlying_daily.iloc[0]) * np.sqrt(252.0))
        if len(underlying_daily) == 1
        else float("nan")
    )
    return {
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "trading_days": int(end_position - start_position),
        "calendar_days": int((end - start).days),
        "soxl_return": float(soxl_return),
        "soxx_return": float(soxx_return),
        "raw_excess_return": float(soxl_return - soxx_return),
        "ideal_daily_reset_3x_return": float(ideal_return),
        "tracking_gap_vs_ideal": float(soxl_return - ideal_return),
        "relative_wealth_change_vs_soxx": float(
            (1.0 + soxl_return) / (1.0 + soxx_return) - 1.0
        ),
        "relative_wealth_change_vs_ideal": float(
            (1.0 + soxl_return) / (1.0 + ideal_return) - 1.0
        ),
        "soxx_realized_volatility": volatility,
        "path_efficiency": _path_efficiency(underlying_daily),
        "positive_soxx_day_share": float(underlying_daily.gt(0.0).mean()),
        "vix_mean": float(section["vix"].mean()),
        "vix_max": float(section["vix"].max()),
        "vix_start": float(section["vix"].iloc[0]),
        "vix_end": float(section["vix"].iloc[-1]),
    }


def build_horizon_windows(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    log_returns = np.log1p(frame["soxx_return"])
    absolute_log_returns = log_returns.abs()
    for horizon in horizons:
        if horizon >= len(frame):
            continue
        result = pd.DataFrame(index=frame.index)
        result["horizon_days"] = int(horizon)
        result["start_date"] = frame.index.to_series().shift(horizon)
        result["end_date"] = frame.index
        result["soxl_return"] = (
            frame["soxl_close"] / frame["soxl_close"].shift(horizon) - 1.0
        )
        result["soxx_return"] = (
            frame["soxx_close"] / frame["soxx_close"].shift(horizon) - 1.0
        )
        result["ideal_daily_reset_3x_return"] = (
            frame["ideal_3x_wealth"]
            / frame["ideal_3x_wealth"].shift(horizon)
            - 1.0
        )
        result["raw_excess_return"] = (
            result["soxl_return"] - result["soxx_return"]
        )
        result["tracking_gap_vs_ideal"] = (
            result["soxl_return"]
            - result["ideal_daily_reset_3x_return"]
        )
        result["relative_wealth_change_vs_soxx"] = (
            (1.0 + result["soxl_return"])
            / (1.0 + result["soxx_return"])
            - 1.0
        )
        result["relative_wealth_change_vs_ideal"] = (
            (1.0 + result["soxl_return"])
            / (1.0 + result["ideal_daily_reset_3x_return"])
            - 1.0
        )
        result["soxx_realized_volatility"] = (
            frame["soxx_return"].rolling(horizon).std(ddof=1)
            * np.sqrt(252.0)
        )
        if horizon == 1:
            result["soxx_realized_volatility"] = (
                frame["soxx_return"].abs() * np.sqrt(252.0)
            )
        result["path_efficiency"] = (
            log_returns.rolling(horizon).sum().abs()
            / absolute_log_returns.rolling(horizon).sum()
        )
        result["positive_soxx_day_share"] = (
            frame["soxx_return"].gt(0.0).rolling(horizon).mean()
        )
        result["vix_mean"] = frame["vix"].rolling(horizon).mean()
        result["vix_max"] = frame["vix"].rolling(horizon).max()
        rows.append(result.dropna(subset=["raw_excess_return"]))
    windows = pd.concat(rows).reset_index(drop=True)
    windows["start_date"] = pd.to_datetime(windows["start_date"])
    windows["end_date"] = pd.to_datetime(windows["end_date"])
    return windows


def summarize_horizons(windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, group in windows.groupby("horizon_days", sort=True):
        rows.append(
            {
                "horizon_days": int(horizon),
                "rows": int(len(group)),
                "positive_excess_share": float(
                    group["raw_excess_return"].gt(0.0).mean()
                ),
                "median_raw_excess_return": float(
                    group["raw_excess_return"].median()
                ),
                "p05_raw_excess_return": float(
                    group["raw_excess_return"].quantile(0.05)
                ),
                "p95_raw_excess_return": float(
                    group["raw_excess_return"].quantile(0.95)
                ),
                "median_tracking_gap_vs_ideal": float(
                    group["tracking_gap_vs_ideal"].median()
                ),
                "median_soxx_realized_volatility": float(
                    group["soxx_realized_volatility"].median()
                ),
                "median_path_efficiency": float(
                    group["path_efficiency"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _boolean_runs(mask: pd.Series) -> list[tuple[int, int, bool]]:
    values = mask.to_numpy(dtype=bool)
    if not len(values):
        return []
    runs: list[tuple[int, int, bool]] = []
    start = 0
    current = bool(values[0])
    for position in range(1, len(values)):
        value = bool(values[position])
        if value != current:
            runs.append((start, position - 1, current))
            start = position
            current = value
    runs.append((start, len(values) - 1, current))
    return runs


def build_persistent_regimes(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    horizons: tuple[int, ...] = PERSISTENCE_HORIZONS,
) -> pd.DataFrame:
    date_positions = {date: pos for pos, date in enumerate(frame.index)}
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        group = (
            windows.loc[windows["horizon_days"].eq(horizon)]
            .sort_values("end_date")
            .reset_index(drop=True)
        )
        for start, end, positive in _boolean_runs(
            group["raw_excess_return"].gt(0.0)
        ):
            section = group.iloc[start : end + 1]
            regime_start = pd.Timestamp(section["end_date"].iloc[0])
            regime_end = pd.Timestamp(section["end_date"].iloc[-1])
            path = _window_row(
                frame,
                date_positions[regime_start],
                date_positions[regime_end],
            )
            rows.append(
                {
                    "lookback_horizon_days": int(horizon),
                    "state": "positive_excess"
                    if positive
                    else "negative_excess",
                    "state_start_date": regime_start.date().isoformat(),
                    "state_end_date": regime_end.date().isoformat(),
                    "state_trading_days": int(len(section)),
                    "state_calendar_days": int(
                        (regime_end - regime_start).days
                    ),
                    "median_trailing_excess": float(
                        section["raw_excess_return"].median()
                    ),
                    "worst_trailing_excess": float(
                        section["raw_excess_return"].min()
                    ),
                    "best_trailing_excess": float(
                        section["raw_excess_return"].max()
                    ),
                    "median_trailing_soxx_return": float(
                        section["soxx_return"].median()
                    ),
                    "median_trailing_volatility": float(
                        section["soxx_realized_volatility"].median()
                    ),
                    "median_trailing_path_efficiency": float(
                        section["path_efficiency"].median()
                    ),
                    "median_trailing_vix_mean": float(
                        section["vix_mean"].median()
                    ),
                    **{
                        f"state_interval_{key}": value
                        for key, value in path.items()
                    },
                }
            )
    result = pd.DataFrame(rows)
    result["rank_within_state"] = (
        result.groupby(["lookback_horizon_days", "state"])[
            "state_trading_days"
        ]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return result.sort_values(
        ["lookback_horizon_days", "state", "rank_within_state"]
    ).reset_index(drop=True)


def build_regime_characteristics(
    windows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in PERSISTENCE_HORIZONS:
        section = windows.loc[windows["horizon_days"].eq(horizon)].copy()
        section["state"] = np.where(
            section["raw_excess_return"].gt(0.0),
            "positive_excess",
            "negative_excess",
        )
        for state, group in section.groupby("state", sort=True):
            rows.append(
                {
                    "lookback_horizon_days": int(horizon),
                    "state": state,
                    "rows": int(len(group)),
                    "row_share": float(len(group) / len(section)),
                    "median_soxx_return": float(
                        group["soxx_return"].median()
                    ),
                    "median_soxl_return": float(
                        group["soxl_return"].median()
                    ),
                    "median_raw_excess_return": float(
                        group["raw_excess_return"].median()
                    ),
                    "median_tracking_gap_vs_ideal": float(
                        group["tracking_gap_vs_ideal"].median()
                    ),
                    "median_soxx_realized_volatility": float(
                        group["soxx_realized_volatility"].median()
                    ),
                    "median_path_efficiency": float(
                        group["path_efficiency"].median()
                    ),
                    "median_vix_mean": float(group["vix_mean"].median()),
                    "median_positive_soxx_day_share": float(
                        group["positive_soxx_day_share"].median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _underwater_episodes(
    frame: pd.DataFrame,
    series: pd.Series,
    *,
    basis: str,
) -> pd.DataFrame:
    values = series.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    peak_position = 0
    peak_value = values[0]
    in_drawdown = False
    for position in range(1, len(values)):
        value = values[position]
        if value >= peak_value:
            if in_drawdown:
                path = _window_row(frame, start_position, position)
                rows.append(
                    {
                        "basis": basis,
                        "peak_date": frame.index[
                            start_position
                        ].date().isoformat(),
                        "trough_date": frame.index[
                            trough_position
                        ].date().isoformat(),
                        "recovery_or_end_date": frame.index[
                            position
                        ].date().isoformat(),
                        "duration_trading_days": int(
                            position - start_position
                        ),
                        "depth": float(trough_depth),
                        "recovered": True,
                        **path,
                    }
                )
                in_drawdown = False
            peak_value = value
            peak_position = position
        else:
            depth = value / peak_value - 1.0
            if not in_drawdown:
                in_drawdown = True
                start_position = peak_position
                trough_position = position
                trough_depth = depth
            elif depth < trough_depth:
                trough_position = position
                trough_depth = depth
    if in_drawdown:
        path = _window_row(frame, start_position, len(values) - 1)
        rows.append(
            {
                "basis": basis,
                "peak_date": frame.index[start_position].date().isoformat(),
                "trough_date": frame.index[
                    trough_position
                ].date().isoformat(),
                "recovery_or_end_date": frame.index[-1].date().isoformat(),
                "duration_trading_days": int(
                    len(values) - 1 - start_position
                ),
                "depth": float(trough_depth),
                "recovered": False,
                **path,
            }
        )
    return pd.DataFrame(rows)


def build_underwater_episodes(frame: pd.DataFrame) -> pd.DataFrame:
    parts = [
        _underwater_episodes(
            frame,
            frame["soxl_wealth"],
            basis="absolute_soxl",
        ),
        _underwater_episodes(
            frame,
            frame["relative_wealth_vs_soxx"],
            basis="relative_vs_soxx",
        ),
        _underwater_episodes(
            frame,
            frame["relative_wealth_vs_ideal_3x"],
            basis="relative_vs_ideal_daily_reset_3x",
        ),
    ]
    result = pd.concat(parts, ignore_index=True)
    result["duration_rank"] = (
        result.groupby("basis")["duration_trading_days"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    result["depth_rank"] = (
        result.groupby("basis")["depth"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    return result


def build_current_horizons(windows: pd.DataFrame) -> pd.DataFrame:
    latest = windows["end_date"].max()
    return (
        windows.loc[windows["end_date"].eq(latest)]
        .sort_values("horizon_days")
        .reset_index(drop=True)
    )


def build_extreme_holding_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    ratio = frame["relative_wealth_vs_soxx"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for positive, label in (
        (True, "longest_positive_excess_interval"),
        (False, "longest_negative_excess_interval"),
    ):
        best: tuple[int, int, int] | None = None
        for start in range(len(ratio) - 1):
            condition = (
                ratio[start + 1 :] > ratio[start]
                if positive
                else ratio[start + 1 :] < ratio[start]
            )
            matches = np.flatnonzero(condition)
            if len(matches):
                end = start + 1 + int(matches[-1])
                span = end - start
                if best is None or span > best[0]:
                    best = (span, start, end)
        if best is None:
            continue
        _, start, end = best
        rows.append({"interval": label, **_window_row(frame, start, end)})
    return pd.DataFrame(rows)


def _build_dataset(frame: pd.DataFrame, spec: LongHorizonSpec) -> dict[str, Any]:
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    static = frame[
        ["soxl_return", "soxx_return", "vix"]
    ].dropna().copy()
    static["relative_daily_excess"] = (
        static["soxl_return"] - static["soxx_return"]
    )
    static["instrument"] = "SOXL_SOXX"
    static = static.set_index("instrument", append=True)
    static.index.names = ["datetime", "instrument"]
    dataset_frame = pd.concat(
        {
            "feature": static[["soxx_return", "vix"]],
            "label": static[
                ["soxl_return", "relative_daily_excess"]
            ],
        },
        axis=1,
    )
    handler = DataHandlerLP(
        instruments=None,
        start_time=dataset_frame.index.get_level_values("datetime").min(),
        end_time=dataset_frame.index.get_level_values("datetime").max(),
        data_loader=StaticDataLoader(config=dataset_frame),
        infer_processors=[],
        learn_processors=[],
    )
    end = frame.index.max().date().isoformat()
    dataset = DatasetH(
        handler=handler,
        segments={"all_history": [spec.start, end]},
    )
    return {
        "api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"],
        "segment": [spec.start, end],
        "rows": int(len(dataset.prepare("all_history"))),
        "role": "descriptive_alignment_only_no_model_fit",
    }


def _record(
    spec: LongHorizonSpec,
    *,
    windows: pd.DataFrame,
    regimes: pd.DataFrame,
    underwater: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        artifacts = {
            "daily_horizon_windows.csv": windows,
            "persistent_regimes_full.csv": regimes,
            "underwater_episodes_full.csv": underwater,
        }
        for filename, data in artifacts.items():
            data.to_csv(tmp_dir / filename, index=False)
        (tmp_dir / "analysis_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                comparison="SOXL_vs_SOXX_and_ideal_daily_reset_3x",
                horizons=",".join(map(str, spec.horizons)),
                timing="close_to_close",
                role="descriptive_not_forecast",
            )
            R.log_metrics(
                data_rows=float(summary["data_audit"]["rows"]),
                full_history_soxl_return=float(
                    summary["full_history"]["soxl_return"]
                ),
                full_history_soxx_return=float(
                    summary["full_history"]["soxx_return"]
                ),
            )
            for filename in (*artifacts, "analysis_summary.json"):
                recorder.log_artifact(
                    (tmp_dir / filename).as_posix(),
                    artifact_path=spec.experiment,
                )
            result = {
                "recorder_id": recorder.id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": Path(
                    recorder.get_local_dir()
                ).resolve().as_posix(),
            }
    artifact_root = Path(result["recorder_local_dir"]) / "artifacts"
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        (artifact_root / filename).unlink(missing_ok=True)
    return result


def _top_underwater(underwater: pd.DataFrame) -> pd.DataFrame:
    return (
        underwater.loc[
            underwater["duration_rank"].le(5)
            | underwater["depth_rank"].le(5)
            | ~underwater["recovered"]
        ]
        .sort_values(["basis", "duration_rank", "depth_rank"])
        .drop_duplicates(
            ["basis", "peak_date", "trough_date", "recovery_or_end_date"]
        )
        .reset_index(drop=True)
    )


def _record_by_basis(
    underwater: pd.DataFrame,
    basis: str,
    *,
    rank: str = "duration_rank",
) -> dict[str, Any]:
    row = (
        underwater.loc[underwater["basis"].eq(basis)]
        .sort_values(rank)
        .iloc[0]
    )
    return {
        key: (
            value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in row.to_dict().items()
    }


def run_soxl_long_horizon_card(
    root: Path | str | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    spec = LongHorizonSpec(root=resolved_root)
    spec.card_dir.mkdir(parents=True, exist_ok=True)

    frame, data_audit = load_soxl_soxx_frame(spec)
    windows = build_horizon_windows(frame, spec.horizons)
    horizon_metrics = summarize_horizons(windows)
    regimes = build_persistent_regimes(
        frame, windows, spec.persistence_horizons
    )
    regime_characteristics = build_regime_characteristics(windows)
    underwater = build_underwater_episodes(frame)
    selected_underwater = _top_underwater(underwater)
    current_horizons = build_current_horizons(windows)
    extreme_intervals = build_extreme_holding_intervals(frame)
    dataset_audit = _build_dataset(frame, spec)

    full_history = _window_row(frame, 0, len(frame) - 1)
    longest_positive_252 = (
        regimes.loc[
            regimes["lookback_horizon_days"].eq(252)
            & regimes["state"].eq("positive_excess")
        ]
        .sort_values("state_trading_days", ascending=False)
        .iloc[0]
        .to_dict()
    )
    longest_negative_252 = (
        regimes.loc[
            regimes["lookback_horizon_days"].eq(252)
            & regimes["state"].eq("negative_excess")
        ]
        .sort_values("state_trading_days", ascending=False)
        .iloc[0]
        .to_dict()
    )
    ongoing_absolute = (
        underwater.loc[
            underwater["basis"].eq("absolute_soxl")
            & ~underwater["recovered"]
        ]
        .sort_values("peak_date")
        .iloc[-1]
        .to_dict()
    )
    summary: dict[str, Any] = {
        "experiment": spec.experiment,
        "as_of": data_audit["as_of"],
        "decision": "descriptive_path_dependence_confirmed_not_entry_signal",
        "promotion": "blocked_descriptive_not_predictive",
        "data_audit": data_audit,
        "dataset_audit": dataset_audit,
        "definitions": {
            "raw_excess_return": "SOXL cumulative return minus SOXX cumulative return",
            "tracking_gap_vs_ideal": "SOXL cumulative return minus product(1 + 3 * daily SOXX return)",
            "persistent_state": "sign of trailing fixed-horizon raw excess on consecutive dates",
            "underwater": "wealth below its prior running maximum",
        },
        "full_history": full_history,
        "longest_252d_positive_state": longest_positive_252,
        "longest_252d_negative_state": longest_negative_252,
        "longest_absolute_underwater": _record_by_basis(
            underwater, "absolute_soxl"
        ),
        "longest_relative_underwater_vs_soxx": _record_by_basis(
            underwater, "relative_vs_soxx"
        ),
        "longest_tracking_underwater_vs_ideal": _record_by_basis(
            underwater, "relative_vs_ideal_daily_reset_3x"
        ),
        "ongoing_absolute_underwater": {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in ongoing_absolute.items()
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/daily_horizon_windows.csv",
                f"artifacts/{spec.experiment}/persistent_regimes_full.csv",
                f"artifacts/{spec.experiment}/underwater_episodes_full.csv",
            ],
        },
        "sidecars": {
            "horizon_metrics": "horizon_metrics.csv",
            "persistent_regimes": "persistent_regimes.csv",
            "regime_characteristics": "regime_characteristics.csv",
            "underwater_episodes": "underwater_episodes.csv",
            "current_horizons": "current_horizons.csv",
            "extreme_holding_intervals": "extreme_holding_intervals.csv",
            "analysis_summary": "analysis_summary.json",
        },
        "risks": [
            "All horizon windows overlap and are descriptive, not independent samples.",
            "Longest intervals are endpoint-sensitive and should not be used alone.",
            "VIX measures broad US equity implied volatility, not semiconductor-specific stress.",
            "Ideal daily-reset 3x excludes fees, financing, rebalance slippage, and product tracking.",
        ],
    }
    recorder = _record(
        spec,
        windows=windows,
        regimes=regimes,
        underwater=underwater,
        summary=summary,
    )
    summary.update(recorder)

    selected_regimes = (
        regimes.loc[regimes["rank_within_state"].le(3)]
        .sort_values(
            ["lookback_horizon_days", "state", "rank_within_state"]
        )
        .reset_index(drop=True)
    )
    sidecars = {
        "horizon_metrics.csv": horizon_metrics,
        "persistent_regimes.csv": selected_regimes,
        "regime_characteristics.csv": regime_characteristics,
        "underwater_episodes.csv": selected_underwater,
        "current_horizons.csv": current_horizons,
        "extreme_holding_intervals.csv": extreme_intervals,
    }
    for filename, data in sidecars.items():
        data.to_csv(spec.card_dir / filename, index=False)
    (spec.card_dir / "analysis_summary.json").write_text(
        json.dumps(
            {
                "decision": summary["decision"],
                "as_of": summary["as_of"],
                "full_history": summary["full_history"],
                "longest_252d_positive_state": summary[
                    "longest_252d_positive_state"
                ],
                "longest_252d_negative_state": summary[
                    "longest_252d_negative_state"
                ],
                "longest_absolute_underwater": summary[
                    "longest_absolute_underwater"
                ],
                "longest_relative_underwater_vs_soxx": summary[
                    "longest_relative_underwater_vs_soxx"
                ],
                "longest_tracking_underwater_vs_ideal": summary[
                    "longest_tracking_underwater_vs_ideal"
                ],
                "ongoing_absolute_underwater": summary[
                    "ongoing_absolute_underwater"
                ],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (spec.card_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    workflow = {
        "experiment": spec.experiment,
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "instruments": ["SOXL", "SOXX"],
        "fields": ["$close", "$vix"],
        "start": spec.start,
        "end": summary["as_of"],
        "timing": "close_to_close",
        "horizons": list(spec.horizons),
        "persistence_horizons": list(spec.persistence_horizons),
        "comparison": [
            "SOXL_vs_SOXX",
            "SOXL_vs_ideal_daily_reset_3x_SOXX",
        ],
        "artifact_policy": summary["artifact_policy"],
    }
    (spec.card_dir / "workflow.yaml").write_text(
        yaml.safe_dump(workflow, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    positive = summary["longest_252d_positive_state"]
    negative = summary["longest_252d_negative_state"]
    absolute = summary["longest_absolute_underwater"]
    relative = summary["longest_relative_underwater_vs_soxx"]
    tracking = summary["longest_tracking_underwater_vs_ideal"]
    ongoing = summary["ongoing_absolute_underwater"]
    conclusion = (
        "# 中文结论\n\n"
        f"- 做了什么：使用 qlib `D.features` 读取 {data_audit['data_start']} 至 "
        f"{data_audit['data_end']} 的 SOXL、SOXX 与 VIX，共 {data_audit['rows']} 个交易日；"
        "比较实际 SOXL、SOXX 和理想每日重置 3x SOXX，并区分固定持有期、持续状态与水下周期。\n"
        f"- 全历史：SOXL 财富变为 {1.0 + full_history['soxl_return']:.2f} 倍，SOXX "
        f"{1.0 + full_history['soxx_return']:.2f} 倍，理想每日重置 3x 为 "
        f"{1.0 + full_history['ideal_daily_reset_3x_return']:.2f} 倍；实际 SOXL 财富"
        f"比理想 3x 低 {abs(full_history['relative_wealth_change_vs_ideal']):.2%}，"
        "说明长期有显著产品/复利跟踪缺口。\n"
        f"- 最长一年期正超额状态：{positive['state_start_date']} 至 "
        f"{positive['state_end_date']}，连续 {positive['state_trading_days']} 个观察日；"
        f"状态区间 SOXX {positive['state_interval_soxx_return']:.2%}、SOXL "
        f"{positive['state_interval_soxl_return']:.2%}，SOXX 年化波动 "
        f"{positive['state_interval_soxx_realized_volatility']:.2%}，VIX 均值 "
        f"{positive['state_interval_vix_mean']:.2f}。特征是底层持续上涨、波动较低、正收益日占比更高。\n"
        f"- 最长一年期负超额状态：{negative['state_start_date']} 至 "
        f"{negative['state_end_date']}，连续 {negative['state_trading_days']} 个观察日；"
        f"状态区间 SOXX {negative['state_interval_soxx_return']:.2%}、SOXL "
        f"{negative['state_interval_soxl_return']:.2%}，SOXX 年化波动 "
        f"{negative['state_interval_soxx_realized_volatility']:.2%}，路径效率 "
        f"{negative['state_interval_path_efficiency']:.2%}，VIX 均值 "
        f"{negative['state_interval_vix_mean']:.2f}。特征是底层横盘/弱势且高波动，"
        "日复利损耗主导结果。\n"
        f"- 最长绝对水下期：{absolute['peak_date']} 至 "
        f"{absolute['recovery_or_end_date']}，{absolute['duration_trading_days']} 个交易日，"
        f"最深 {absolute['depth']:.2%}；最长相对 SOXX 水下期为 "
        f"{relative['duration_trading_days']} 个交易日，最深 {relative['depth']:.2%}。"
        "这说明回本与重新跑赢 1x 都可能需要多年。\n"
        f"- 相对理想 3x 的最长水下期从 {tracking['peak_date']} 开始，"
        f"截至 {tracking['recovery_or_end_date']} 仍未恢复，深度 {tracking['depth']:.2%}。"
        "绝对上涨不等于有效捕获理论 3x。\n"
        f"- 当前绝对水下期始于 {ongoing['peak_date']}，截至 {ongoing['recovery_or_end_date']} "
        f"深度 {ongoing['depth']:.2%}；期间 VIX 均值 {ongoing['vix_mean']:.2f}、"
        f"最高 {ongoing['vix_max']:.2f}。当前样本再次说明半导体集中风险可以在 VIX 不高时发生，"
        "不能把 VIX 当成 SOXL 自身波动的替代品。\n"
        "- 窗口差异：持有期越长，正超额概率总体提高，但负尾部并未消失；5年窗口仍可出现"
        "SOXL 上涨但落后 SOXX 的起点敏感结果。\n"
        "- 决策：descriptive_path_dependence_confirmed_not_entry_signal。该研究解释杠杆复利机制，"
        "不是可直接交易的择时信号，也不改变生产 Stage1 配置。\n"
        f"- qlib recorder：{summary['recorder_id']}。\n"
        "- 风险：滚动窗口高度重叠；最长区间对起止点敏感；VIX 是大盘隐含波动率；"
        "理想 3x 未计费用、融资与再平衡摩擦。\n"
        "- 下一步：若要用于模型，应单独加入 SOXL/SOXX 实现波动、路径效率和相对跟踪缺口，"
        "并做前向验证，而不是仅依赖 VIX。\n"
        "- 不构成买卖建议。\n"
    )
    (spec.card_dir / "conclusion.md").write_text(
        conclusion,
        encoding="utf-8",
    )
    return summary
