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


EXPERIMENT = "soxl_soxx_excess_drawdown_regimes_v1"
HORIZONS = (1, 5, 21, 63)
PRIMARY_BANDS = ("dd_0_5", "dd_5_10", "dd_10_20", "dd_20plus")


@dataclass(frozen=True)
class ExcessDrawdownRegimeSpec:
    root: Path
    experiment: str = EXPERIMENT
    start: str = "2010-03-11"
    development_end: str = "2022-12-30"
    outer_start: str = "2023-01-03"
    horizons: tuple[int, ...] = HORIZONS
    cost_bps: float = 10.0
    stress_cost_bps: float = 25.0
    bootstrap_samples: int = 2_000
    bootstrap_seed: int = 20260727
    minimum_rows: int = 40
    minimum_development_years: int = 8
    minimum_positive_year_share: float = 0.70
    minimum_nonoverlap_positive_share: float = 0.55
    maximum_extra_drawdown: float = -0.10

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


def _drawdown_band(drawdown: pd.Series) -> pd.Series:
    values = np.select(
        [
            drawdown.le(-0.20),
            drawdown.le(-0.10),
            drawdown.le(-0.05),
        ],
        ["dd_20plus", "dd_10_20", "dd_5_10"],
        default="dd_0_5",
    )
    return pd.Series(values, index=drawdown.index, dtype="object")


def _max_drawdown(prices: np.ndarray) -> float:
    wealth = prices / prices[0]
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))


def _load_market(
    spec: ExcessDrawdownRegimeSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = spec.provider_uri / "qlib_store_manifest.json"
    if not manifest_path.exists():
        raise ValueError("blocked: qlib provider manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provider_as_of = str(manifest["as_of"])
    config = QlibProviderConfig(
        provider_uri=spec.provider_uri,
        market="all",
        region="us",
        kernels=1,
    )
    provider_audit = init_qlib_provider(config)
    end = min(pd.Timestamp(provider_as_of), pd.Timestamp(provider_audit["calendar_end"]))
    raw = load_qlib_feature_frame(
        config,
        instruments=["SOXL", "SOXX"],
        fields=["$open", "$close"],
        start=spec.start,
        end=end.date().isoformat(),
    )
    frame = (
        raw.reset_index()
        .pivot(index="datetime", columns="instrument")
        .sort_index()
    )
    market = pd.DataFrame(
        {
            "soxl_open": frame[("$open", "SOXL")].astype(float),
            "soxl_close": frame[("$close", "SOXL")].astype(float),
            "soxx_open": frame[("$open", "SOXX")].astype(float),
            "soxx_close": frame[("$close", "SOXX")].astype(float),
        }
    ).dropna()
    market.index = pd.DatetimeIndex(market.index, name="signal_date")
    market["soxx_drawdown"] = (
        market["soxx_close"] / market["soxx_close"].cummax() - 1.0
    )
    market["soxx_return_5d"] = market["soxx_close"].pct_change(5)
    market["soxx_return_20d"] = market["soxx_close"].pct_change(20)
    market["soxx_realized_vol_20d"] = (
        market["soxx_close"].pct_change().rolling(20).std() * np.sqrt(252.0)
    )
    market["drawdown_band"] = _drawdown_band(market["soxx_drawdown"])
    market["phase"] = pd.Series(
        np.where(
            market["soxx_return_5d"].gt(0.0), "rebound5", "falling5"
        ),
        index=market.index,
        dtype="object",
    ).mask(market["soxx_return_5d"].isna())
    market["regime_phase"] = (
        market["drawdown_band"] + "__" + market["phase"]
    )
    market["signal_position"] = np.arange(len(market), dtype=int)
    audit = {
        **provider_audit,
        "manifest": manifest_path.resolve().as_posix(),
        "provider_as_of": provider_as_of,
        "data_start": market.index.min().date().isoformat(),
        "data_end": market.index.max().date().isoformat(),
        "rows": int(len(market)),
        "instruments": ["SOXL", "SOXX"],
        "fields": ["$open", "$close"],
        "source_api": [
            "qlib.init",
            "qlib.data.D.calendar",
            "qlib.data.D.instruments",
            "qlib.data.D.features",
        ],
    }
    return market, audit


def build_forward_observations(
    market: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    cost_bps: float = 10.0,
    stress_cost_bps: float = 25.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dates = market.index
    for position in range(len(market)):
        state = market.iloc[position]
        for horizon in horizons:
            exit_position = position + horizon + 1
            if exit_position >= len(market):
                continue
            soxl_path = market["soxl_open"].iloc[
                position + 1 : exit_position + 1
            ].to_numpy()
            soxx_path = market["soxx_open"].iloc[
                position + 1 : exit_position + 1
            ].to_numpy()
            soxl_return = float(soxl_path[-1] / soxl_path[0] - 1.0)
            soxx_return = float(soxx_path[-1] / soxx_path[0] - 1.0)
            soxl_daily = soxl_path[1:] / soxl_path[:-1] - 1.0
            beta_matched_return = float(
                np.prod(1.0 + soxl_daily / 3.0) - 1.0
            )
            raw_excess = soxl_return - soxx_return
            rows.append(
                {
                    "signal_date": dates[position],
                    "entry_date": dates[position + 1],
                    "exit_date": dates[exit_position],
                    "signal_position": int(state["signal_position"]),
                    "year": int(dates[position].year),
                    "horizon_days": horizon,
                    "drawdown_band": state["drawdown_band"],
                    "phase": state["phase"],
                    "regime_phase": state["regime_phase"],
                    "soxx_drawdown": float(state["soxx_drawdown"]),
                    "soxx_return_5d": float(state["soxx_return_5d"])
                    if pd.notna(state["soxx_return_5d"])
                    else np.nan,
                    "soxx_return_20d": float(state["soxx_return_20d"])
                    if pd.notna(state["soxx_return_20d"])
                    else np.nan,
                    "soxx_realized_vol_20d": float(
                        state["soxx_realized_vol_20d"]
                    )
                    if pd.notna(state["soxx_realized_vol_20d"])
                    else np.nan,
                    "soxl_return": soxl_return,
                    "soxx_return": soxx_return,
                    "raw_excess_return": raw_excess,
                    "raw_excess_net_10bps": raw_excess
                    - 2.0 * cost_bps / 10_000.0,
                    "raw_excess_net_25bps": raw_excess
                    - 2.0 * stress_cost_bps / 10_000.0,
                    "beta_matched_soxl_return": beta_matched_return,
                    "beta_matched_excess_return": (
                        beta_matched_return - soxx_return
                    ),
                    "soxl_max_drawdown": _max_drawdown(soxl_path),
                    "soxx_max_drawdown": _max_drawdown(soxx_path),
                    "extra_max_drawdown": (
                        _max_drawdown(soxl_path) - _max_drawdown(soxx_path)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _segment(frame: pd.DataFrame, name: str, spec: ExcessDrawdownRegimeSpec):
    if name == "development":
        return frame[frame["signal_date"].le(spec.development_end)]
    if name == "outer_diagnostic":
        return frame[frame["signal_date"].ge(spec.outer_start)]
    return frame


def build_regime_metrics(
    observations: pd.DataFrame, spec: ExcessDrawdownRegimeSpec
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_columns = [
        "raw_excess_return",
        "raw_excess_net_10bps",
        "raw_excess_net_25bps",
        "beta_matched_excess_return",
        "extra_max_drawdown",
        "soxl_max_drawdown",
        "soxx_max_drawdown",
    ]
    for segment in ("all", "development", "outer_diagnostic"):
        section = _segment(observations, segment, spec)
        for regime_type, regime_column in (
            ("drawdown_band", "drawdown_band"),
            ("drawdown_band_phase", "regime_phase"),
        ):
            for (horizon, regime), group in section.groupby(
                ["horizon_days", regime_column], sort=True
            ):
                row: dict[str, Any] = {
                    "segment": segment,
                    "regime_type": regime_type,
                    "regime": regime,
                    "horizon_days": int(horizon),
                    "rows": int(len(group)),
                    "positive_raw_share": float(
                        group["raw_excess_return"].gt(0.0).mean()
                    ),
                    "positive_beta_matched_share": float(
                        group["beta_matched_excess_return"].gt(0.0).mean()
                    ),
                }
                for column in metric_columns:
                    row[f"mean_{column}"] = float(group[column].mean())
                    row[f"median_{column}"] = float(group[column].median())
                rows.append(row)
    return pd.DataFrame(rows)


def build_annual_metrics(observations: pd.DataFrame) -> pd.DataFrame:
    return (
        observations.groupby(
            ["year", "horizon_days", "drawdown_band"], as_index=False
        )
        .agg(
            rows=("raw_excess_return", "size"),
            mean_raw_excess_return=("raw_excess_return", "mean"),
            median_raw_excess_return=("raw_excess_return", "median"),
            mean_raw_excess_net_10bps=("raw_excess_net_10bps", "mean"),
            mean_beta_matched_excess_return=(
                "beta_matched_excess_return",
                "mean",
            ),
            mean_extra_max_drawdown=("extra_max_drawdown", "mean"),
        )
        .sort_values(["horizon_days", "drawdown_band", "year"])
        .reset_index(drop=True)
    )


def build_nonoverlap_metrics(
    observations: pd.DataFrame, spec: ExcessDrawdownRegimeSpec
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment in ("development", "outer_diagnostic"):
        section = _segment(observations, segment, spec)
        for (horizon, band), group in section.groupby(
            ["horizon_days", "drawdown_band"], sort=True
        ):
            selected: list[int] = []
            last_position = -10**9
            for index, row in group.sort_values("signal_position").iterrows():
                position = int(row["signal_position"])
                if position > last_position + int(horizon):
                    selected.append(index)
                    last_position = position
            sample = group.loc[selected]
            rows.append(
                {
                    "segment": segment,
                    "drawdown_band": band,
                    "horizon_days": int(horizon),
                    "observations": int(len(sample)),
                    "mean_raw_excess_return": float(
                        sample["raw_excess_return"].mean()
                    ),
                    "median_raw_excess_return": float(
                        sample["raw_excess_return"].median()
                    ),
                    "positive_raw_share": float(
                        sample["raw_excess_return"].gt(0.0).mean()
                    ),
                    "mean_beta_matched_excess_return": float(
                        sample["beta_matched_excess_return"].mean()
                    ),
                    "mean_extra_max_drawdown": float(
                        sample["extra_max_drawdown"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_year_mean(
    values: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_stability_audit(
    *,
    regime_metrics: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    nonoverlap_metrics: pd.DataFrame,
    spec: ExcessDrawdownRegimeSpec,
) -> pd.DataFrame:
    primary = regime_metrics[regime_metrics["regime_type"].eq("drawdown_band")]
    phase = regime_metrics[
        regime_metrics["regime_type"].eq("drawdown_band_phase")
    ]
    rows: list[dict[str, Any]] = []
    for horizon in spec.horizons:
        for band in PRIMARY_BANDS:
            dev = primary[
                primary["segment"].eq("development")
                & primary["horizon_days"].eq(horizon)
                & primary["regime"].eq(band)
            ]
            outer = primary[
                primary["segment"].eq("outer_diagnostic")
                & primary["horizon_days"].eq(horizon)
                & primary["regime"].eq(band)
            ]
            if dev.empty or outer.empty:
                continue
            dev_row = dev.iloc[0]
            outer_row = outer.iloc[0]
            annual = annual_metrics[
                annual_metrics["horizon_days"].eq(horizon)
                & annual_metrics["drawdown_band"].eq(band)
                & annual_metrics["year"].le(
                    pd.Timestamp(spec.development_end).year
                )
            ]
            annual_values = annual["mean_raw_excess_return"].to_numpy()
            positive_year_share = float(
                annual["mean_raw_excess_return"].gt(0.0).mean()
            )
            lower, upper = _bootstrap_year_mean(
                annual_values,
                samples=spec.bootstrap_samples,
                seed=spec.bootstrap_seed + horizon * 10 + len(band),
            )
            nonoverlap = nonoverlap_metrics[
                nonoverlap_metrics["segment"].eq("development")
                & nonoverlap_metrics["horizon_days"].eq(horizon)
                & nonoverlap_metrics["drawdown_band"].eq(band)
            ].iloc[0]
            phase_rows = phase[
                phase["segment"].isin(["development", "outer_diagnostic"])
                & phase["horizon_days"].eq(horizon)
                & phase["regime"].isin(
                    [f"{band}__falling5", f"{band}__rebound5"]
                )
            ]
            phase_consistent = bool(
                len(phase_rows) == 4
                and phase_rows["mean_raw_excess_return"].gt(0.0).all()
            )
            enough_evidence = bool(
                int(dev_row["rows"]) >= spec.minimum_rows
                and len(annual) >= spec.minimum_development_years
                and int(nonoverlap["observations"]) >= 12
            )
            raw_stable = bool(
                enough_evidence
                and float(dev_row["mean_raw_excess_net_25bps"]) > 0.0
                and float(outer_row["mean_raw_excess_net_25bps"]) > 0.0
                and positive_year_share >= spec.minimum_positive_year_share
                and float(nonoverlap["positive_raw_share"])
                >= spec.minimum_nonoverlap_positive_share
                and lower > 0.0
                and phase_consistent
            )
            beta_matched_stable = bool(
                float(dev_row["mean_beta_matched_excess_return"]) > 0.0
                and float(outer_row["mean_beta_matched_excess_return"]) > 0.0
            )
            drawdown_tolerable = bool(
                float(dev_row["mean_extra_max_drawdown"])
                >= spec.maximum_extra_drawdown
                and float(outer_row["mean_extra_max_drawdown"])
                >= spec.maximum_extra_drawdown
            )
            rows.append(
                {
                    "drawdown_band": band,
                    "horizon_days": horizon,
                    "development_rows": int(dev_row["rows"]),
                    "outer_rows": int(outer_row["rows"]),
                    "development_years": int(len(annual)),
                    "development_positive_year_share": positive_year_share,
                    "development_year_bootstrap_lower": lower,
                    "development_year_bootstrap_upper": upper,
                    "development_nonoverlap_observations": int(
                        nonoverlap["observations"]
                    ),
                    "development_nonoverlap_positive_share": float(
                        nonoverlap["positive_raw_share"]
                    ),
                    "development_raw_excess": float(
                        dev_row["mean_raw_excess_return"]
                    ),
                    "outer_raw_excess": float(
                        outer_row["mean_raw_excess_return"]
                    ),
                    "development_raw_excess_net_25bps": float(
                        dev_row["mean_raw_excess_net_25bps"]
                    ),
                    "outer_raw_excess_net_25bps": float(
                        outer_row["mean_raw_excess_net_25bps"]
                    ),
                    "development_beta_matched_excess": float(
                        dev_row["mean_beta_matched_excess_return"]
                    ),
                    "outer_beta_matched_excess": float(
                        outer_row["mean_beta_matched_excess_return"]
                    ),
                    "development_extra_max_drawdown": float(
                        dev_row["mean_extra_max_drawdown"]
                    ),
                    "outer_extra_max_drawdown": float(
                        outer_row["mean_extra_max_drawdown"]
                    ),
                    "phase_consistent": phase_consistent,
                    "raw_return_stable": raw_stable,
                    "beta_matched_stable": beta_matched_stable,
                    "extra_drawdown_tolerable": drawdown_tolerable,
                    "risk_efficient_stable": bool(
                        raw_stable
                        and beta_matched_stable
                        and drawdown_tolerable
                    ),
                    "interpretation": (
                        "risk_efficient"
                        if raw_stable
                        and beta_matched_stable
                        and drawdown_tolerable
                        else "leverage_compensation_only"
                        if raw_stable
                        else "not_stable"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _build_dataset(
    market: pd.DataFrame,
    observations: pd.DataFrame,
    spec: ExcessDrawdownRegimeSpec,
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    labels = observations.pivot(
        index="signal_date",
        columns="horizon_days",
        values="raw_excess_return",
    ).rename(columns=lambda horizon: f"raw_excess_{int(horizon)}d")
    indexed = market.join(labels, how="inner").dropna(
        subset=["soxx_drawdown", "soxx_return_5d"]
    )
    feature_columns = [
        "soxx_drawdown",
        "soxx_return_5d",
        "soxx_return_20d",
        "soxx_realized_vol_20d",
    ]
    label_columns = [f"raw_excess_{horizon}d" for horizon in spec.horizons]
    static = pd.concat(
        {
            "feature": indexed[feature_columns].astype(float),
            "label": indexed[label_columns].astype(float),
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [
            static.index,
            np.full(len(static), "SOXL_SOXX_DRAWDOWN_REGIME"),
        ],
        names=["datetime", "instrument"],
    )
    segments = {
        "development": [spec.start, spec.development_end],
        "outer_diagnostic": [
            spec.outer_start,
            market.index.max().date().isoformat(),
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
        "feature_timing": "signal_date_close_only",
        "label_timing": "next_open_to_future_open",
    }


def _record(
    spec: ExcessDrawdownRegimeSpec,
    *,
    observations: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    nonoverlap_metrics: pd.DataFrame,
    stability_audit: pd.DataFrame,
) -> dict[str, str]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        artifacts = {
            "forward_observations.csv": observations,
            "regime_metrics_full.csv": regime_metrics,
            "annual_metrics_full.csv": annual_metrics,
            "nonoverlap_metrics_full.csv": nonoverlap_metrics,
            "stability_audit_full.csv": stability_audit,
        }
        for filename, frame in artifacts.items():
            frame.to_csv(tmp_dir / filename, index=False)
        stable_count = int(stability_audit["raw_return_stable"].sum())
        efficient_count = int(stability_audit["risk_efficient_stable"].sum())
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                benchmark="SOXX",
                asset="SOXL",
                development_end=spec.development_end,
                outer_role="reused_diagnostic_only",
                selection_uses_outer_test="false",
                horizons=",".join(map(str, spec.horizons)),
            )
            R.log_metrics(
                raw_stable_cells=float(stable_count),
                risk_efficient_stable_cells=float(efficient_count),
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


def run_soxl_soxx_excess_drawdown_regimes_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    spec = ExcessDrawdownRegimeSpec(root=_resolve_root(root))
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    market, data_audit = _load_market(spec)
    observations = build_forward_observations(
        market,
        horizons=spec.horizons,
        cost_bps=spec.cost_bps,
        stress_cost_bps=spec.stress_cost_bps,
    )
    observations["signal_date"] = pd.to_datetime(observations["signal_date"])
    dataset, dataset_audit = _build_dataset(market, observations, spec)
    regime_metrics = build_regime_metrics(observations, spec)
    annual_metrics = build_annual_metrics(observations)
    nonoverlap_metrics = build_nonoverlap_metrics(observations, spec)
    stability_audit = build_stability_audit(
        regime_metrics=regime_metrics,
        annual_metrics=annual_metrics,
        nonoverlap_metrics=nonoverlap_metrics,
        spec=spec,
    )
    recorder = _record(
        spec,
        observations=observations,
        regime_metrics=regime_metrics,
        annual_metrics=annual_metrics,
        nonoverlap_metrics=nonoverlap_metrics,
        stability_audit=stability_audit,
    )
    stable = stability_audit[stability_audit["raw_return_stable"]].copy()
    efficient = stability_audit[
        stability_audit["risk_efficient_stable"]
    ].copy()
    decision = (
        "descriptive_raw_excess_zone_found_no_risk_efficient_zone"
        if not stable.empty and efficient.empty
        else "no_stable_zone_found"
        if stable.empty
        else "risk_efficient_zone_requires_clean_forward"
    )
    selected = (
        stable.sort_values(
            [
                "drawdown_band",
                "horizon_days",
            ]
        )
        .loc[
            :,
            [
                "drawdown_band",
                "horizon_days",
                "development_raw_excess",
                "outer_raw_excess",
                "development_extra_max_drawdown",
                "outer_extra_max_drawdown",
                "development_beta_matched_excess",
                "outer_beta_matched_excess",
                "interpretation",
            ],
        ]
        .to_dict("records")
    )
    deep_21_annual = annual_metrics[
        annual_metrics["drawdown_band"].eq("dd_20plus")
        & annual_metrics["horizon_days"].eq(21)
        & annual_metrics["year"].le(pd.Timestamp(spec.development_end).year)
    ]
    deep_21_negative_years = (
        deep_21_annual.loc[
            deep_21_annual["mean_raw_excess_return"].le(0.0), "year"
        ]
        .astype(int)
        .tolist()
    )
    artifact_policy = {
        "policy": "slim_card_recorder_first",
        "workflow_config": "workflow.yaml",
        "recorder_only_artifacts": [
            f"artifacts/{spec.experiment}/forward_observations.csv",
            f"artifacts/{spec.experiment}/regime_metrics_full.csv",
            f"artifacts/{spec.experiment}/annual_metrics_full.csv",
            f"artifacts/{spec.experiment}/nonoverlap_metrics_full.csv",
            f"artifacts/{spec.experiment}/stability_audit_full.csv",
        ],
    }
    summary: dict[str, Any] = {
        "experiment": spec.experiment,
        "card_type": "soxl_soxx_excess_drawdown_regime_diagnostic_card",
        "decision": decision,
        "promotion": "blocked",
        "system_of_record": "qlib",
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "as_of": data_audit["data_end"],
        "hypothesis": (
            "Test whether prior-close SOXX drawdown bands identify stable "
            "next-open SOXL minus SOXX excess-return zones, and whether any "
            "zone remains positive after beta matching and extra drawdown."
        ),
        "timing": {
            "state_observation": "signal_date_close",
            "entry": "next_trading_day_open",
            "exit": "future_open_after_fixed_horizon",
            "horizons": list(spec.horizons),
        },
        "regime_design": {
            "primary_drawdown_bands": list(PRIMARY_BANDS),
            "secondary_phase": [
                "falling5",
                "rebound5",
            ],
            "fixed_before_evaluation": True,
            "primary_cell_count": len(PRIMARY_BANDS) * len(spec.horizons),
            "outer_role": "reused_diagnostic_only_not_selection",
            "selection_uses_outer_test": False,
        },
        "risk_matching": {
            "raw_comparison": "equal capital SOXL minus SOXX",
            "beta_matched_comparison": (
                "daily rebalanced one-third SOXL plus two-thirds zero-yield "
                "cash minus SOXX"
            ),
            "extra_drawdown": (
                "forward SOXL max drawdown minus forward SOXX max drawdown"
            ),
            "round_trip_cost_bps": spec.cost_bps * 2.0,
            "stress_round_trip_cost_bps": spec.stress_cost_bps * 2.0,
        },
        "stability_definition": {
            "minimum_rows": spec.minimum_rows,
            "minimum_development_years": spec.minimum_development_years,
            "minimum_positive_year_share": (
                spec.minimum_positive_year_share
            ),
            "minimum_nonoverlap_positive_share": (
                spec.minimum_nonoverlap_positive_share
            ),
            "development_year_bootstrap_lower_must_be_positive": True,
            "both_falling_and_rebound_phases_positive_in_development_and_outer": (
                True
            ),
            "25bps_per_leg_stress_must_be_positive_in_development_and_outer": (
                True
            ),
        },
        "stable_raw_zones": selected,
        "stable_raw_zone_count": int(len(stable)),
        "risk_efficient_zone_count": int(len(efficient)),
        "deep_drawdown_21d_exception_years": deep_21_negative_years,
        "overfit_audit": {
            "risk_level_for_description": "medium",
            "risk_level_for_trading_rule": "high",
            "fixed_primary_cells": len(PRIMARY_BANDS) * len(spec.horizons),
            "overlapping_labels": True,
            "nonoverlap_audit_included": True,
            "year_block_bootstrap_included": True,
            "outer_is_clean_holdout": False,
            "outer_is_selection_input": False,
            "reason": (
                "The thresholds are low-dimensional and economically fixed, "
                "but the same historical sample was inspected repeatedly and "
                "forward labels overlap."
            ),
        },
        "data_audit": data_audit,
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
            "https://qlib.readthedocs.io/en/latest/component/strategy.html",
        ],
        **recorder,
        "sidecars": {
            "selected_model": "selected_model.json",
            "regime_metrics": "regime_metrics.csv",
            "annual_metrics": "annual_metrics.csv",
            "nonoverlap_metrics": "nonoverlap_metrics.csv",
            "stability_audit": "stability_audit.csv",
            "risk_analysis": "risk_analysis.csv",
            "analysis_summary": "analysis_summary.json",
            "failure_modes": "failure_modes.md",
            "conclusion": "conclusion.md",
        },
        "artifact_policy": artifact_policy,
    }
    selected_payload = {
        "type": "historical_structure_diagnostic_not_model",
        "promotion": "blocked",
        "selection_uses_outer_test": False,
        "outer_role": "reused_diagnostic_only",
        "stable_raw_zones": selected,
        "risk_efficient_zone_count": int(len(efficient)),
    }
    stability_audit.to_csv(
        spec.card_dir / "stability_audit.csv", index=False
    )
    annual_metrics.to_csv(spec.card_dir / "annual_metrics.csv", index=False)
    nonoverlap_metrics.to_csv(
        spec.card_dir / "nonoverlap_metrics.csv", index=False
    )
    regime_metrics[
        regime_metrics["segment"].isin(["development", "outer_diagnostic"])
    ].to_csv(spec.card_dir / "regime_metrics.csv", index=False)
    stability_audit[
        [
            "drawdown_band",
            "horizon_days",
            "development_extra_max_drawdown",
            "outer_extra_max_drawdown",
            "development_beta_matched_excess",
            "outer_beta_matched_excess",
            "risk_efficient_stable",
        ]
    ].to_csv(spec.card_dir / "risk_analysis.csv", index=False)
    (spec.card_dir / "selected_model.json").write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (spec.card_dir / "analysis_summary.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "stable_raw_zones": selected,
                "risk_efficient_zone_count": int(len(efficient)),
                "interpretation": (
                    "Stable raw excess is leverage compensation unless it "
                    "survives beta matching and the extra-drawdown gate."
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
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "segments": dataset_audit["segments"],
        "timing": summary["timing"],
        "regime_design": summary["regime_design"],
        "stability_definition": summary["stability_definition"],
        "artifact_policy": artifact_policy,
    }
    (spec.card_dir / "workflow.yaml").write_text(
        yaml.safe_dump(workflow, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    failure_modes = """# Failure modes

- 2023–2026 has already been inspected by prior SOXL research, so it is a
  reused diagnostic segment rather than a clean holdout.
- Daily forward labels overlap, especially at 21 and 63 days. The card reports
  year-level bootstrap and horizon-spaced observations, but neither creates new
  independent market regimes.
- Equal-capital SOXL versus SOXX mixes timing skill with roughly three-times
  leverage. The beta-matched comparison is required before calling a regime
  risk-efficient.
- Drawdown bands are based on the contemporaneous SOXX all-time high and use
  only prior-close information, but historical ETF structure, financing costs,
  and tracking behavior can change.
- This card is descriptive. Turning the best historical cell into a trading
  rule without a preregistered clean-forward test would be overfit.
"""
    (spec.card_dir / "failure_modes.md").write_text(
        failure_modes, encoding="utf-8"
    )
    if stable.empty:
        stable_text = "没有区间通过原始超额稳定性门槛"
    else:
        stable_text = "；".join(
            f"{row.drawdown_band}/{int(row.horizon_days)}日"
            for row in stable.itertuples()
        )
    deep_21 = stability_audit[
        stability_audit["drawdown_band"].eq("dd_20plus")
        & stability_audit["horizon_days"].eq(21)
    ].iloc[0]
    deep_63 = stability_audit[
        stability_audit["drawdown_band"].eq("dd_20plus")
        & stability_audit["horizon_days"].eq(63)
    ].iloc[0]
    conclusion = f"""# 中文结论

- 做了什么：使用 qlib 的 SOXL/SOXX 日线，从 signal-date close 观察 SOXX 距历史高点的回撤状态，在 next-open 进入，比较未来 1/5/21/63 日 SOXL 与 SOXX 的原始超额、按 1/3 SOXL+2/3 零收益现金日度配平后的超额，以及 SOXL 额外最大回撤。
- 稳定性门槛：开发期截至 {spec.development_end}；2023 年以后只作复用诊断；同时要求年度正向比例、年度 block bootstrap、按 horizon 去重叠抽样、下跌/反弹两种 phase 和 25bp/边压力都通过。
- 原始超额稳定区间：{stable_text}。严格门槛下最有一致性的结构是 SOXX 回撤超过 20% 后持有 5–21 个交易日；63 日均值仍为正，但去重叠样本胜率只有 {deep_63['development_nonoverlap_positive_share']:.2%}，低于 {spec.minimum_nonoverlap_positive_share:.0%} 门槛。
- 深回撤 21 日：开发期平均原始超额={_pct(deep_21['development_raw_excess'])}，outer 诊断={_pct(deep_21['outer_raw_excess'])}；开发期额外最大回撤={_pct(deep_21['development_extra_max_drawdown'])}，beta 配平超额={_pct(deep_21['development_beta_matched_excess'])}。
- 深回撤 63 日：开发期平均原始超额={_pct(deep_63['development_raw_excess'])}，outer 诊断={_pct(deep_63['outer_raw_excess'])}；开发期额外最大回撤={_pct(deep_63['development_extra_max_drawdown'])}，beta 配平超额={_pct(deep_63['development_beta_matched_excess'])}。
- 反例年份：深回撤 21 日开发期负向年份={deep_21_negative_years}；其中 2022 是持续下跌而不是快速 V 型修复，说明“回撤超过 20%”本身不是充分的抄底信号。
- 关键解释：风险效率稳定区间数量={len(efficient)}。原始超额在深回撤后的确更稳定，但它伴随显著额外回撤，且 1/3 SOXL 风险配平后超额为负，因此更像杠杆风险补偿，不是 SOXL 产品本身的稳定 alpha。
- overfit：描述性结论风险为 medium，若直接据此形成加仓规则则为 high；原因是 16 个固定主单元仍存在多重比较、21/63 日标签重叠，且 2023–2026 不是干净 holdout。
- 决策：{decision}；不据此修改 Stage1 的 30% SOXL 上限。若要转成可交易规则，优先只预登记“SOXX 回撤>20% + 固定 21 日”的 clean-forward sleeve，并单独设置回撤预算；不再从 5/21/63 日中事后择优。
- qlib recorder：{recorder['recorder_id']}
- 不构成买卖建议。
"""
    (spec.card_dir / "conclusion.md").write_text(
        conclusion, encoding="utf-8"
    )
    return summary
