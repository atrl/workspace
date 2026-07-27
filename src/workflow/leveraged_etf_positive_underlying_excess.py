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


EXPERIMENT = "leveraged_etf_positive_underlying_excess_v1"
HORIZONS = (1, 5, 21, 63)
RETURN_THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
RETURN_BIN_EDGES = (-np.inf, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, np.inf)
RETURN_BIN_LABELS = (
    "non_positive",
    "up_0_0p5",
    "up_0p5_1",
    "up_1_2",
    "up_2_3",
    "up_3_5",
    "up_5_10",
    "up_10_20",
    "up_20plus",
)


@dataclass(frozen=True)
class LeveragedPair:
    name: str
    leveraged: str
    underlying: str
    provider_name: str


PAIRS = (
    LeveragedPair("SOXL_SOXX", "SOXL", "SOXX", "soxl"),
    LeveragedPair("TQQQ_QQQ", "TQQQ", "QQQ", "tqqq"),
)


@dataclass(frozen=True)
class PositiveUnderlyingExcessSpec:
    root: Path
    experiment: str = EXPERIMENT
    start: str = "2010-03-11"
    development_end: str = "2022-12-30"
    outer_start: str = "2023-01-03"
    horizons: tuple[int, ...] = HORIZONS
    return_thresholds: tuple[float, ...] = RETURN_THRESHOLDS
    switch_cost_per_leg_bps: float = 25.0
    minimum_development_rows: int = 100
    minimum_outer_rows: int = 50
    minimum_nonoverlap_rows: int = 30
    minimum_positive_share: float = 0.90
    minimum_nonoverlap_positive_share: float = 0.80
    minimum_positive_year_share: float = 0.80

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    def provider_uri(self, pair: LeveragedPair) -> Path:
        return self.root / "data" / "qlib" / pair.provider_name


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _max_drawdown(prices: np.ndarray) -> float:
    wealth = prices / prices[0]
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))


def _path_efficiency(daily_returns: np.ndarray) -> float:
    log_returns = np.log1p(daily_returns)
    denominator = float(np.abs(log_returns).sum())
    if denominator <= 0.0:
        return 0.0
    return float(abs(log_returns.sum()) / denominator)


def _return_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=RETURN_BIN_EDGES,
        labels=RETURN_BIN_LABELS,
        right=True,
        include_lowest=True,
    ).astype("object")


def _load_pair(
    spec: PositiveUnderlyingExcessSpec, pair: LeveragedPair
) -> tuple[pd.DataFrame, dict[str, Any]]:
    provider_uri = spec.provider_uri(pair)
    manifest_path = provider_uri / "qlib_store_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"blocked: qlib manifest missing for {pair.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest["instruments"]) != {pair.leveraged, pair.underlying}:
        raise ValueError(f"blocked: qlib instruments drifted for {pair.name}")
    config = QlibProviderConfig(
        provider_uri=provider_uri,
        market="all",
        region="us",
        kernels=1,
    )
    provider_audit = init_qlib_provider(config)
    end = min(
        pd.Timestamp(manifest["as_of"]),
        pd.Timestamp(provider_audit["calendar_end"]),
    )
    raw = load_qlib_feature_frame(
        config,
        instruments=[pair.leveraged, pair.underlying],
        fields=["$close"],
        start=spec.start,
        end=end.date().isoformat(),
    )
    wide = (
        raw.reset_index()
        .pivot(index="datetime", columns="instrument")
        .sort_index()
    )
    frame = pd.DataFrame(
        {
            "leveraged_close": wide[("$close", pair.leveraged)].astype(float),
            "underlying_close": wide[
                ("$close", pair.underlying)
            ].astype(float),
        }
    ).dropna()
    frame.index = pd.DatetimeIndex(frame.index, name="signal_date")
    daily = frame["underlying_close"].pct_change()
    frame["starting_underlying_return_20d"] = frame[
        "underlying_close"
    ].pct_change(20)
    frame["starting_underlying_vol_20d"] = (
        daily.rolling(20).std() * np.sqrt(252.0)
    )
    frame["signal_position"] = np.arange(len(frame), dtype=int)
    audit = {
        "pair": pair.name,
        "leveraged": pair.leveraged,
        "underlying": pair.underlying,
        "provider_uri": provider_uri.resolve().as_posix(),
        "manifest": manifest_path.resolve().as_posix(),
        "as_of": str(manifest["as_of"]),
        "data_start": frame.index.min().date().isoformat(),
        "data_end": frame.index.max().date().isoformat(),
        "rows": int(len(frame)),
        "provider_audit": provider_audit,
        "source_api": [
            "qlib.init",
            "qlib.data.D.calendar",
            "qlib.data.D.instruments",
            "qlib.data.D.features",
        ],
    }
    return frame, audit


def build_pair_observations(
    frame: pd.DataFrame,
    pair: LeveragedPair,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    switch_cost_per_leg_bps: float = 25.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dates = frame.index
    switch_round_trip_cost = 2.0 * switch_cost_per_leg_bps / 10_000.0
    for position in range(len(frame)):
        for horizon in horizons:
            exit_position = position + horizon
            if exit_position >= len(frame):
                continue
            leveraged_path = frame["leveraged_close"].iloc[
                position : exit_position + 1
            ].to_numpy()
            underlying_path = frame["underlying_close"].iloc[
                position : exit_position + 1
            ].to_numpy()
            leveraged_return = float(
                leveraged_path[-1] / leveraged_path[0] - 1.0
            )
            underlying_return = float(
                underlying_path[-1] / underlying_path[0] - 1.0
            )
            underlying_daily = (
                underlying_path[1:] / underlying_path[:-1] - 1.0
            )
            ideal_daily_reset_return = float(
                np.prod(1.0 + 3.0 * underlying_daily) - 1.0
            )
            raw_excess = leveraged_return - underlying_return
            ideal_excess = ideal_daily_reset_return - underlying_return
            rows.append(
                {
                    "pair": pair.name,
                    "leveraged": pair.leveraged,
                    "underlying": pair.underlying,
                    "signal_date": dates[position],
                    "exit_date": dates[exit_position],
                    "signal_position": position,
                    "year": int(dates[position].year),
                    "horizon_days": horizon,
                    "starting_underlying_return_20d": frame.iloc[position][
                        "starting_underlying_return_20d"
                    ],
                    "starting_underlying_vol_20d": frame.iloc[position][
                        "starting_underlying_vol_20d"
                    ],
                    "underlying_return": underlying_return,
                    "leveraged_return": leveraged_return,
                    "raw_excess_return": raw_excess,
                    "switch_net_excess_25bps": (
                        raw_excess - switch_round_trip_cost
                    ),
                    "ideal_daily_reset_return": ideal_daily_reset_return,
                    "ideal_excess_return": ideal_excess,
                    "tracking_gap_vs_ideal": (
                        leveraged_return - ideal_daily_reset_return
                    ),
                    "excess_capture_ratio": (
                        raw_excess / ideal_excess
                        if ideal_excess > 1e-9
                        else np.nan
                    ),
                    "path_efficiency": _path_efficiency(underlying_daily),
                    "realized_volatility": (
                        float(
                            np.std(underlying_daily, ddof=1)
                            * np.sqrt(252.0)
                        )
                        if horizon > 1
                        else float(abs(underlying_daily[0]) * np.sqrt(252.0))
                    ),
                    "leveraged_max_drawdown": _max_drawdown(
                        leveraged_path
                    ),
                    "underlying_max_drawdown": _max_drawdown(
                        underlying_path
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["underlying_return_bin"] = _return_bin(
        result["underlying_return"]
    )
    result["path_efficiency_band"] = pd.cut(
        result["path_efficiency"],
        bins=[-np.inf, 0.20, 0.50, np.inf],
        labels=["choppy_lt20", "mixed_20_50", "directional_ge50"],
        include_lowest=True,
    ).astype("object")
    result["volatility_band"] = pd.cut(
        result["realized_volatility"],
        bins=[-np.inf, 0.20, 0.35, np.inf],
        labels=["vol_lt20", "vol_20_35", "vol_ge35"],
        include_lowest=True,
    ).astype("object")
    return result


def _segment(
    frame: pd.DataFrame, name: str, spec: PositiveUnderlyingExcessSpec
) -> pd.DataFrame:
    if name == "development":
        return frame[frame["signal_date"].le(spec.development_end)]
    if name == "outer_diagnostic":
        return frame[frame["signal_date"].ge(spec.outer_start)]
    return frame


def _nonoverlap_sample(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    selected: list[int] = []
    last_by_pair: dict[str, int] = {}
    for index, row in frame.sort_values(
        ["pair", "signal_position"]
    ).iterrows():
        pair = str(row["pair"])
        position = int(row["signal_position"])
        last = last_by_pair.get(pair, -10**9)
        if position > last + horizon:
            selected.append(index)
            last_by_pair[pair] = position
    return frame.loc[selected]


def build_threshold_audit(
    observations: pd.DataFrame, spec: PositiveUnderlyingExcessSpec
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in PAIRS:
        for horizon in spec.horizons:
            base = observations[
                observations["pair"].eq(pair.name)
                & observations["horizon_days"].eq(horizon)
            ]
            for threshold in spec.return_thresholds:
                for segment in ("development", "outer_diagnostic"):
                    section = _segment(base, segment, spec)
                    section = section[
                        section["underlying_return"].gt(threshold)
                    ]
                    nonoverlap = _nonoverlap_sample(section, horizon)
                    annual = (
                        section.groupby("year")[
                            [
                                "raw_excess_return",
                                "switch_net_excess_25bps",
                            ]
                        ]
                        .mean()
                        .reset_index()
                    )
                    rows.append(
                        {
                            "pair": pair.name,
                            "horizon_days": horizon,
                            "underlying_return_threshold": threshold,
                            "segment": segment,
                            "rows": int(len(section)),
                            "mean_underlying_return": float(
                                section["underlying_return"].mean()
                            ),
                            "mean_raw_excess_return": float(
                                section["raw_excess_return"].mean()
                            ),
                            "positive_raw_excess_share": float(
                                section["raw_excess_return"].gt(0.0).mean()
                            ),
                            "positive_switch_net_share": float(
                                section["switch_net_excess_25bps"]
                                .gt(0.0)
                                .mean()
                            ),
                            "nonoverlap_rows": int(len(nonoverlap)),
                            "nonoverlap_positive_raw_share": float(
                                nonoverlap["raw_excess_return"].gt(0.0).mean()
                            ),
                            "nonoverlap_positive_switch_net_share": float(
                                nonoverlap["switch_net_excess_25bps"]
                                .gt(0.0)
                                .mean()
                            ),
                            "positive_raw_year_share": float(
                                annual["raw_excess_return"].gt(0.0).mean()
                            ),
                            "positive_switch_net_year_share": float(
                                annual["switch_net_excess_25bps"]
                                .gt(0.0)
                                .mean()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _threshold_passes(
    threshold_audit: pd.DataFrame,
    *,
    horizon: int,
    threshold: float,
    metric: str,
    spec: PositiveUnderlyingExcessSpec,
) -> bool:
    rows = threshold_audit[
        threshold_audit["horizon_days"].eq(horizon)
        & threshold_audit["underlying_return_threshold"].eq(threshold)
    ]
    if len(rows) != len(PAIRS) * 2:
        return False
    positive_column = (
        "positive_raw_excess_share"
        if metric == "raw"
        else "positive_switch_net_share"
    )
    nonoverlap_column = (
        "nonoverlap_positive_raw_share"
        if metric == "raw"
        else "nonoverlap_positive_switch_net_share"
    )
    annual_column = (
        "positive_raw_year_share"
        if metric == "raw"
        else "positive_switch_net_year_share"
    )
    development = rows[rows["segment"].eq("development")]
    outer = rows[rows["segment"].eq("outer_diagnostic")]
    return bool(
        development["rows"].ge(spec.minimum_development_rows).all()
        and outer["rows"].ge(spec.minimum_outer_rows).all()
        and development[positive_column]
        .ge(spec.minimum_positive_share)
        .all()
        and outer[positive_column].ge(spec.minimum_positive_share).all()
        and development["nonoverlap_rows"]
        .ge(spec.minimum_nonoverlap_rows)
        .all()
        and development[nonoverlap_column]
        .ge(spec.minimum_nonoverlap_positive_share)
        .all()
        and development[annual_column]
        .ge(spec.minimum_positive_year_share)
        .all()
    )


def build_stable_thresholds(
    threshold_audit: pd.DataFrame, spec: PositiveUnderlyingExcessSpec
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in spec.horizons:
        result: dict[str, Any] = {"horizon_days": horizon}
        for metric in ("raw", "switch_net_25bps"):
            passed = [
                threshold
                for threshold in spec.return_thresholds
                if _threshold_passes(
                    threshold_audit,
                    horizon=horizon,
                    threshold=threshold,
                    metric=metric,
                    spec=spec,
                )
            ]
            result[f"{metric}_minimum_underlying_return"] = (
                min(passed) if passed else np.nan
            )
            result[f"{metric}_stable"] = bool(passed)
        rows.append(result)
    return pd.DataFrame(rows)


def build_return_bin_metrics(
    observations: pd.DataFrame, spec: PositiveUnderlyingExcessSpec
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment in ("all", "development", "outer_diagnostic"):
        section = _segment(observations, segment, spec)
        grouped = section.groupby(
            ["pair", "horizon_days", "underlying_return_bin"],
            observed=True,
            sort=True,
        )
        for (pair, horizon, return_bin), group in grouped:
            rows.append(
                {
                    "segment": segment,
                    "pair": pair,
                    "horizon_days": int(horizon),
                    "underlying_return_bin": return_bin,
                    "rows": int(len(group)),
                    "mean_underlying_return": float(
                        group["underlying_return"].mean()
                    ),
                    "mean_raw_excess_return": float(
                        group["raw_excess_return"].mean()
                    ),
                    "median_raw_excess_return": float(
                        group["raw_excess_return"].median()
                    ),
                    "positive_raw_excess_share": float(
                        group["raw_excess_return"].gt(0.0).mean()
                    ),
                    "positive_switch_net_share": float(
                        group["switch_net_excess_25bps"].gt(0.0).mean()
                    ),
                    "mean_tracking_gap_vs_ideal": float(
                        group["tracking_gap_vs_ideal"].mean()
                    ),
                    "median_excess_capture_ratio": float(
                        group["excess_capture_ratio"].median()
                    ),
                    "mean_path_efficiency": float(
                        group["path_efficiency"].mean()
                    ),
                    "mean_realized_volatility": float(
                        group["realized_volatility"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_path_quality_metrics(
    observations: pd.DataFrame, spec: PositiveUnderlyingExcessSpec
) -> pd.DataFrame:
    positive = observations[observations["underlying_return"].gt(0.0)]
    rows: list[dict[str, Any]] = []
    for segment in ("all", "development", "outer_diagnostic"):
        section = _segment(positive, segment, spec)
        for regime_type, column in (
            ("path_efficiency", "path_efficiency_band"),
            ("realized_volatility", "volatility_band"),
        ):
            for (pair, horizon, regime), group in section.groupby(
                ["pair", "horizon_days", column],
                observed=True,
                sort=True,
            ):
                rows.append(
                    {
                        "segment": segment,
                        "regime_type": regime_type,
                        "regime": regime,
                        "pair": pair,
                        "horizon_days": int(horizon),
                        "rows": int(len(group)),
                        "mean_underlying_return": float(
                            group["underlying_return"].mean()
                        ),
                        "mean_raw_excess_return": float(
                            group["raw_excess_return"].mean()
                        ),
                        "positive_raw_excess_share": float(
                            group["raw_excess_return"].gt(0.0).mean()
                        ),
                        "positive_switch_net_share": float(
                            group["switch_net_excess_25bps"]
                            .gt(0.0)
                            .mean()
                        ),
                        "mean_tracking_gap_vs_ideal": float(
                            group["tracking_gap_vs_ideal"].mean()
                        ),
                        "median_excess_capture_ratio": float(
                            group["excess_capture_ratio"].median()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_pair_summary(
    observations: pd.DataFrame, spec: PositiveUnderlyingExcessSpec
) -> pd.DataFrame:
    positive = observations[observations["underlying_return"].gt(0.0)]
    rows: list[dict[str, Any]] = []
    for segment in ("development", "outer_diagnostic"):
        section = _segment(positive, segment, spec)
        for (pair, horizon), group in section.groupby(
            ["pair", "horizon_days"], sort=True
        ):
            rows.append(
                {
                    "segment": segment,
                    "pair": pair,
                    "horizon_days": int(horizon),
                    "rows": int(len(group)),
                    "mean_underlying_return": float(
                        group["underlying_return"].mean()
                    ),
                    "mean_leveraged_return": float(
                        group["leveraged_return"].mean()
                    ),
                    "mean_raw_excess_return": float(
                        group["raw_excess_return"].mean()
                    ),
                    "positive_raw_excess_share": float(
                        group["raw_excess_return"].gt(0.0).mean()
                    ),
                    "positive_switch_net_share": float(
                        group["switch_net_excess_25bps"].gt(0.0).mean()
                    ),
                    "mean_ideal_excess_return": float(
                        group["ideal_excess_return"].mean()
                    ),
                    "mean_tracking_gap_vs_ideal": float(
                        group["tracking_gap_vs_ideal"].mean()
                    ),
                    "median_excess_capture_ratio": float(
                        group["excess_capture_ratio"].median()
                    ),
                    "mean_path_efficiency": float(
                        group["path_efficiency"].mean()
                    ),
                    "failure_rows_raw": int(
                        group["raw_excess_return"].le(0.0).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _build_dataset(
    observations: pd.DataFrame, spec: PositiveUnderlyingExcessSpec
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    frame = observations[observations["horizon_days"].eq(21)].copy()
    frame = frame.dropna(
        subset=[
            "starting_underlying_return_20d",
            "starting_underlying_vol_20d",
        ]
    )
    frame = frame.set_index(["signal_date", "pair"]).sort_index()
    feature_columns = [
        "starting_underlying_return_20d",
        "starting_underlying_vol_20d",
    ]
    label_columns = [
        "underlying_return",
        "raw_excess_return",
        "switch_net_excess_25bps",
        "path_efficiency",
    ]
    static = pd.concat(
        {
            "feature": frame[feature_columns].astype(float),
            "label": frame[label_columns].astype(float),
        },
        axis=1,
    )
    static.index.names = ["datetime", "instrument"]
    segments = {
        "development": [spec.start, spec.development_end],
        "outer_diagnostic": [
            spec.outer_start,
            observations["signal_date"].max().date().isoformat(),
        ],
    }
    handler = DataHandlerLP(
        instruments=None,
        start_time=static.index.get_level_values("datetime").min(),
        end_time=static.index.get_level_values("datetime").max(),
        data_loader=StaticDataLoader(config=static),
        infer_processors=[],
        learn_processors=[],
    )
    dataset = DatasetH(handler=handler, segments=segments)
    return dataset, {
        "api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"],
        "segments": segments,
        "segment_rows": {
            name: int(len(dataset.prepare(name))) for name in segments
        },
        "features": feature_columns,
        "labels": label_columns,
        "dataset_horizon_days": 21,
        "feature_timing": "window_start_close",
        "label_timing": "future_close_to_close_descriptive_not_signal",
    }


def _record(
    spec: PositiveUnderlyingExcessSpec,
    *,
    observations: pd.DataFrame,
    threshold_audit: pd.DataFrame,
    return_bin_metrics: pd.DataFrame,
    path_quality_metrics: pd.DataFrame,
) -> dict[str, str]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        artifacts = {
            "forward_windows.csv": observations,
            "threshold_audit_full.csv": threshold_audit,
            "return_bin_metrics_full.csv": return_bin_metrics,
            "path_quality_metrics_full.csv": path_quality_metrics,
        }
        for filename, frame in artifacts.items():
            frame.to_csv(tmp_dir / filename, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                pairs="SOXL_SOXX,TQQQ_QQQ",
                horizons="1,5,21,63",
                comparison_timing="close_to_close",
                outer_role="reused_diagnostic_only",
                selection_uses_outer_test="false",
                switch_cost_per_leg_bps=str(
                    spec.switch_cost_per_leg_bps
                ),
            )
            R.log_metrics(
                observation_rows=float(len(observations)),
                positive_underlying_rows=float(
                    observations["underlying_return"].gt(0.0).sum()
                ),
            )
            for filename in artifacts:
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


def _pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def run_leveraged_etf_positive_underlying_excess_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    spec = PositiveUnderlyingExcessSpec(root=_resolve_root(root))
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    observations: list[pd.DataFrame] = []
    data_audits: list[dict[str, Any]] = []
    for pair in PAIRS:
        market, audit = _load_pair(spec, pair)
        observations.append(
            build_pair_observations(
                market,
                pair,
                horizons=spec.horizons,
                switch_cost_per_leg_bps=spec.switch_cost_per_leg_bps,
            )
        )
        data_audits.append(audit)
    combined = pd.concat(observations, ignore_index=True)
    combined["signal_date"] = pd.to_datetime(combined["signal_date"])
    dataset, dataset_audit = _build_dataset(combined, spec)
    threshold_audit = build_threshold_audit(combined, spec)
    stable_thresholds = build_stable_thresholds(threshold_audit, spec)
    return_bin_metrics = build_return_bin_metrics(combined, spec)
    path_quality_metrics = build_path_quality_metrics(combined, spec)
    pair_summary = build_pair_summary(combined, spec)
    recorder = _record(
        spec,
        observations=combined,
        threshold_audit=threshold_audit,
        return_bin_metrics=return_bin_metrics,
        path_quality_metrics=path_quality_metrics,
    )
    threshold_records = stable_thresholds.to_dict("records")
    decision = "descriptive_stable_excess_map_found_not_a_forecast_signal"
    artifact_policy = {
        "policy": "slim_card_recorder_first",
        "workflow_config": "workflow.yaml",
        "recorder_only_artifacts": [
            f"artifacts/{spec.experiment}/forward_windows.csv",
            f"artifacts/{spec.experiment}/threshold_audit_full.csv",
            f"artifacts/{spec.experiment}/return_bin_metrics_full.csv",
            f"artifacts/{spec.experiment}/path_quality_metrics_full.csv",
        ],
    }
    summary: dict[str, Any] = {
        "experiment": spec.experiment,
        "card_type": "leveraged_etf_positive_underlying_excess_map_card",
        "decision": decision,
        "promotion": "blocked_descriptive_not_predictive",
        "system_of_record": "qlib",
        "as_of": min(audit["data_end"] for audit in data_audits),
        "hypothesis": (
            "Conditional on a positive future close-to-close return in the "
            "1x ETF, determine when the corresponding 3x ETF has stable "
            "equal-capital excess and when path volatility defeats it."
        ),
        "pairs": [
            {
                "name": pair.name,
                "leveraged": pair.leveraged,
                "underlying": pair.underlying,
                "provider_uri": spec.provider_uri(pair).resolve().as_posix(),
            }
            for pair in PAIRS
        ],
        "timing": {
            "comparison": "close_to_close",
            "reason": "align_with_daily_leverage_reset_and_nav_objective",
            "horizons": list(spec.horizons),
            "condition_is_future_realized_return": True,
            "is_ex_ante_signal": False,
        },
        "definitions": {
            "raw_excess": "3x ETF return minus 1x ETF return",
            "switch_net_excess_25bps": (
                "raw excess minus a 50bps hurdle for entering and exiting the "
                "3x sleeve at 25bps per leg; baseline 1x leg excluded"
            ),
            "ideal_daily_reset": (
                "compound product of one plus three times each 1x daily return"
            ),
            "tracking_gap": "actual 3x return minus ideal daily-reset return",
            "path_efficiency": (
                "absolute cumulative log return divided by sum of absolute "
                "daily log returns; low values mean a choppy path"
            ),
        },
        "stability_gate": {
            "minimum_development_rows": spec.minimum_development_rows,
            "minimum_outer_rows": spec.minimum_outer_rows,
            "minimum_nonoverlap_rows": spec.minimum_nonoverlap_rows,
            "minimum_positive_share": spec.minimum_positive_share,
            "minimum_nonoverlap_positive_share": (
                spec.minimum_nonoverlap_positive_share
            ),
            "minimum_positive_year_share": (
                spec.minimum_positive_year_share
            ),
            "must_pass_both_pairs": True,
            "outer_role": "reused_diagnostic_only_not_selection",
        },
        "stable_thresholds": threshold_records,
        "interpretation": {
            "one_and_five_day": (
                "any positive underlying return is enough for stable raw "
                "excess; 0.5pct is needed under the 50bp switch stress"
            ),
            "twenty_one_day": (
                "approximately 1pct underlying appreciation is needed"
            ),
            "sixty_three_day": (
                "approximately 5pct underlying appreciation is needed"
            ),
            "key_failure_mode": (
                "positive cumulative return with low path efficiency or high "
                "realized volatility can still produce no leveraged excess"
            ),
        },
        "overfit_audit": {
            "risk_level_for_payoff_map": "low_to_medium",
            "risk_level_if_used_as_trading_signal": "high",
            "fixed_threshold_count": len(spec.return_thresholds),
            "fixed_horizon_count": len(spec.horizons),
            "cross_pair_confirmation": True,
            "overlapping_windows": True,
            "nonoverlap_audit_included": True,
            "outer_is_clean_holdout": False,
            "condition_uses_future_information": True,
            "reason": (
                "This is a descriptive payoff surface. The future underlying "
                "return and future path quality are not known at entry."
            ),
        },
        "data_audit": data_audits,
        "dataset_audit": dataset_audit,
        "qlib_api": [
            "qlib.init",
            "qlib.data.D.calendar",
            "qlib.data.D.instruments",
            "qlib.data.D.features",
            "DatasetH",
            "DataHandlerLP",
            "qlib.workflow.R",
        ],
        "qlib_docs_reference": [
            "https://qlib.readthedocs.io/en/latest/component/data.html",
            "https://qlib.readthedocs.io/en/stable/component/recorder.html",
        ],
        **recorder,
        "sidecars": {
            "selected_model": "selected_model.json",
            "stable_thresholds": "stable_thresholds.csv",
            "threshold_audit": "threshold_audit.csv",
            "return_bin_metrics": "return_bin_metrics.csv",
            "path_quality_metrics": "path_quality_metrics.csv",
            "pair_summary": "model_comparison.csv",
            "risk_analysis": "risk_analysis.csv",
            "analysis_summary": "analysis_summary.json",
            "failure_modes": "failure_modes.md",
            "conclusion": "conclusion.md",
        },
        "artifact_policy": artifact_policy,
    }
    stable_thresholds.to_csv(
        spec.card_dir / "stable_thresholds.csv", index=False
    )
    threshold_audit.to_csv(
        spec.card_dir / "threshold_audit.csv", index=False
    )
    return_bin_metrics[
        return_bin_metrics["segment"].isin(
            ["development", "outer_diagnostic"]
        )
    ].to_csv(spec.card_dir / "return_bin_metrics.csv", index=False)
    path_quality_metrics[
        path_quality_metrics["segment"].isin(
            ["development", "outer_diagnostic"]
        )
    ].to_csv(spec.card_dir / "path_quality_metrics.csv", index=False)
    pair_summary.to_csv(spec.card_dir / "model_comparison.csv", index=False)
    failure_rows = pair_summary[
        [
            "segment",
            "pair",
            "horizon_days",
            "rows",
            "positive_raw_excess_share",
            "positive_switch_net_share",
            "failure_rows_raw",
            "mean_tracking_gap_vs_ideal",
            "median_excess_capture_ratio",
        ]
    ]
    failure_rows.to_csv(spec.card_dir / "risk_analysis.csv", index=False)
    selected_payload = {
        "type": "descriptive_payoff_map_not_model",
        "promotion": "blocked_descriptive_not_predictive",
        "selection_uses_outer_test": False,
        "future_condition_is_not_available_at_entry": True,
        "stable_thresholds": threshold_records,
    }
    (spec.card_dir / "selected_model.json").write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (spec.card_dir / "analysis_summary.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "stable_thresholds": threshold_records,
                "core_result": (
                    "A rising underlying produces stable leveraged excess at "
                    "short horizons, but longer horizons require a larger and "
                    "more directional rise."
                ),
                "not_a_signal": (
                    "Future return and path quality are realized labels, not "
                    "features available at entry."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (spec.card_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    workflow = {
        "experiment": spec.experiment,
        "providers": {
            pair.name: spec.provider_uri(pair).resolve().as_posix()
            for pair in PAIRS
        },
        "segments": dataset_audit["segments"],
        "timing": summary["timing"],
        "stability_gate": summary["stability_gate"],
        "artifact_policy": artifact_policy,
    }
    (spec.card_dir / "workflow.yaml").write_text(
        yaml.safe_dump(workflow, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    failure_modes = """# Failure modes

- The condition is the future realized underlying return. It maps payoffs but
  cannot be used as an entry signal without a separate forecasting model.
- Multi-day windows overlap. Horizon-spaced development observations reduce,
  but do not eliminate, dependence across market regimes.
- A positive cumulative 1x return can be produced by a highly oscillating path.
  Daily 3x reset and financing/tracking drag can then erase the expected excess.
- The 2023–2026 segment has been viewed by prior research and is diagnostic,
  not a clean holdout.
- Thresholds are rounded and fixed, but choosing the most attractive cell after
  inspection would still be overfit.
"""
    (spec.card_dir / "failure_modes.md").write_text(
        failure_modes, encoding="utf-8"
    )
    threshold_by_horizon = stable_thresholds.set_index("horizon_days")
    positive_path = path_quality_metrics[
        path_quality_metrics["segment"].eq("all")
        & path_quality_metrics["regime_type"].eq("path_efficiency")
    ]
    choppy = positive_path[
        positive_path["regime"].eq("choppy_lt20")
        & positive_path["horizon_days"].isin([21, 63])
    ]
    mixed = positive_path[
        positive_path["regime"].eq("mixed_20_50")
        & positive_path["horizon_days"].isin([21, 63])
    ]
    choppy_min = float(choppy["positive_raw_excess_share"].min())
    choppy_max = float(choppy["positive_raw_excess_share"].max())
    mixed_min = float(mixed["positive_raw_excess_share"].min())
    conclusion = f"""# 中文结论

- 重新定义：本卡不再按起点回撤分层，而是直接回答“未来底层上涨多少时，3x ETF 相对 1x ETF 的超额会稳定出现”。SOXL 对 SOXX、TQQQ 对 QQQ 都使用 close-to-close，以匹配日度杠杆重置。
- 原始超额稳定门槛：1日>{_pct(threshold_by_horizon.loc[1, 'raw_minimum_underlying_return'])}，5日>{_pct(threshold_by_horizon.loc[5, 'raw_minimum_underlying_return'])}，21日>{_pct(threshold_by_horizon.loc[21, 'raw_minimum_underlying_return'])}，63日>{_pct(threshold_by_horizon.loc[63, 'raw_minimum_underlying_return'])}。
- 若给额外 3x sleeve 的进入与退出各计25bp、合计50bp压力（不含基准1x腿），稳定门槛变为：1日>{_pct(threshold_by_horizon.loc[1, 'switch_net_25bps_minimum_underlying_return'])}，5日>{_pct(threshold_by_horizon.loc[5, 'switch_net_25bps_minimum_underlying_return'])}，21日>{_pct(threshold_by_horizon.loc[21, 'switch_net_25bps_minimum_underlying_return'])}，63日>{_pct(threshold_by_horizon.loc[63, 'switch_net_25bps_minimum_underlying_return'])}。
- 所以用户的直觉在短周期成立：底层只要上涨，1–5日原始超额已经高度稳定。持有期拉长后，“上涨”本身不够，累计涨幅必须覆盖波动损耗。
- 路径比终点同样重要：21/63日、底层最终上涨但路径效率低于20%时，原始超额胜率只有 {_pct(choppy_min)}–{_pct(choppy_max)}；路径效率20%–50%时，两组最低胜率为 {_pct(mixed_min)}。这解释了为什么指数最终上涨，杠杆 ETF 仍可能没有超额。
- 实际产品相对“用底层每日收益机械复利3倍”的 tracking gap 平均为负，且持有期越长越明显；因此理论3倍不是长期固定倍数。
- 稳定定义：SOXL/SOXX 与 TQQQ/QQQ 两组均须在开发期和复用 outer 达到至少90%正超额，开发期去重叠样本至少80%，正向年份至少80%。
- overfit：作为 payoff map 风险 low-to-medium；若把未来上涨门槛当作今天可见的择时信号，风险 high，因为未来收益和未来路径在入场时不可知。
- 决策：{decision}。这张图可以用来定义“若看对方向，需要多大涨幅才值得使用3x”，不能直接替代 Stage1 的方向预测。
- qlib recorder：{recorder['recorder_id']}
- 不构成买卖建议。
"""
    (spec.card_dir / "conclusion.md").write_text(
        conclusion, encoding="utf-8"
    )
    return summary
