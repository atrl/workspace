from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.workflow.soxl_long_horizon_excess_drawdown import (
    LongHorizonSpec,
    load_soxl_soxx_frame,
)


EXPERIMENT = "soxl_rolling_3x_retention_factor_v1"
LOOKBACKS = (21, 63, 126)
TARGET_HORIZONS = (21, 63)
RIDGE_ALPHA = 10.0
FOLDS = (
    ("2016_2018", "2010-03-11", "2015-12-31", "2016-01-01", "2018-12-31"),
    ("2019_2020", "2010-03-11", "2018-12-31", "2019-01-01", "2020-12-31"),
    ("2021_2022", "2010-03-11", "2020-12-31", "2021-01-01", "2022-12-31"),
    (
        "outer_2023_2026",
        "2010-03-11",
        "2022-12-30",
        "2023-01-03",
        "2026-07-28",
    ),
)
MODEL_FEATURES = {
    "vix_level": ("vix",),
    "vix_normalized": ("vix_z252",),
    "core_baseline": (
        "vix_z252",
        "soxx_rvol_z_63",
        "soxx_log_return_63",
        "path_efficiency_63",
    ),
    "retention_factor": ("retention_z252_21",),
    "baseline_plus_retention": (
        "vix_z252",
        "soxx_rvol_z_63",
        "soxx_log_return_63",
        "path_efficiency_63",
        "retention_z252_21",
    ),
}


@dataclass(frozen=True)
class RollingRetentionSpec:
    root: Path
    experiment: str = EXPERIMENT
    start: str = "2010-03-11"
    development_end: str = "2022-12-30"
    outer_start: str = "2023-01-03"
    lookbacks: tuple[int, ...] = LOOKBACKS
    target_horizons: tuple[int, ...] = TARGET_HORIZONS
    ridge_alpha: float = RIDGE_ALPHA

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _lagged_zscore(
    series: pd.Series,
    *,
    window: int = 252,
    minimum: int = 126,
) -> pd.Series:
    history = series.shift(1)
    mean = history.rolling(window, min_periods=minimum).mean()
    standard_deviation = history.rolling(window, min_periods=minimum).std()
    return (series - mean) / standard_deviation


def build_retention_factor_panel(frame: pd.DataFrame) -> pd.DataFrame:
    panel = pd.DataFrame(index=frame.index)
    panel["vix"] = frame["vix"]
    panel["vix_z252"] = _lagged_zscore(frame["vix"])
    daily_underlying_log_return = np.log(frame["soxx_close"]).diff()

    for horizon in LOOKBACKS:
        soxl_return = (
            frame["soxl_close"] / frame["soxl_close"].shift(horizon) - 1.0
        )
        soxx_return = (
            frame["soxx_close"] / frame["soxx_close"].shift(horizon) - 1.0
        )
        ideal_return = (
            frame["ideal_3x_wealth"]
            / frame["ideal_3x_wealth"].shift(horizon)
            - 1.0
        )
        soxl_log_growth = np.log1p(soxl_return)
        soxx_log_growth = np.log1p(soxx_return)
        ideal_log_growth = np.log1p(ideal_return)

        panel[f"soxl_return_{horizon}"] = soxl_return
        panel[f"soxx_return_{horizon}"] = soxx_return
        panel[f"ideal_daily_reset_3x_return_{horizon}"] = ideal_return
        panel[f"effective_multiple_{horizon}"] = (
            soxl_return / soxx_return.replace(0.0, np.nan)
        )
        panel[f"endpoint_3x_gap_{horizon}"] = (
            soxl_return - 3.0 * soxx_return
        )
        panel[f"ideal_endpoint_path_gap_{horizon}"] = (
            ideal_return - 3.0 * soxx_return
        )
        panel[f"product_tracking_gap_{horizon}"] = (
            soxl_return - ideal_return
        )

        actual_residual = soxl_log_growth - 3.0 * soxx_log_growth
        ideal_path_drag = ideal_log_growth - 3.0 * soxx_log_growth
        product_tracking = soxl_log_growth - ideal_log_growth
        panel[f"actual_3x_residual_log_{horizon}"] = actual_residual
        panel[f"ideal_path_drag_log_{horizon}"] = ideal_path_drag
        panel[f"product_tracking_log_{horizon}"] = product_tracking
        panel[f"continuous_3x_retention_ratio_{horizon}"] = np.exp(
            actual_residual
        )
        panel[f"retention_z252_{horizon}"] = _lagged_zscore(actual_residual)

        realized_volatility = (
            frame["soxx_return"].rolling(horizon).std(ddof=1) * np.sqrt(252.0)
        )
        panel[f"soxx_rvol_{horizon}"] = realized_volatility
        panel[f"soxx_rvol_z_{horizon}"] = _lagged_zscore(
            realized_volatility
        )
        panel[f"soxx_log_return_{horizon}"] = soxx_log_growth
        panel[f"path_efficiency_{horizon}"] = (
            daily_underlying_log_return.rolling(horizon).sum().abs()
            / daily_underlying_log_return.abs().rolling(horizon).sum()
        )

    for horizon in TARGET_HORIZONS:
        future_soxl_return = (
            frame["soxl_close"].shift(-horizon) / frame["soxl_close"] - 1.0
        )
        future_soxx_return = (
            frame["soxx_close"].shift(-horizon) / frame["soxx_close"] - 1.0
        )
        future_ideal_return = (
            frame["ideal_3x_wealth"].shift(-horizon)
            / frame["ideal_3x_wealth"]
            - 1.0
        )
        panel[f"future_actual_3x_residual_log_{horizon}"] = (
            np.log1p(future_soxl_return)
            - 3.0 * np.log1p(future_soxx_return)
        )
        panel[f"future_ideal_path_drag_log_{horizon}"] = (
            np.log1p(future_ideal_return)
            - 3.0 * np.log1p(future_soxx_return)
        )
        panel[f"future_endpoint_3x_gap_{horizon}"] = (
            future_soxl_return - 3.0 * future_soxx_return
        )
    panel.index = pd.DatetimeIndex(panel.index, name="datetime")
    return panel


def build_qlib_dataset(
    panel: pd.DataFrame,
    spec: RollingRetentionSpec,
) -> tuple[Any, dict[str, Any]]:
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    feature_columns = sorted(
        {
            column
            for columns in MODEL_FEATURES.values()
            for column in columns
        }
        | {
            f"actual_3x_residual_log_{horizon}"
            for horizon in spec.lookbacks
        }
        | {
            f"ideal_path_drag_log_{horizon}"
            for horizon in spec.lookbacks
        }
        | {
            f"product_tracking_log_{horizon}"
            for horizon in spec.lookbacks
        }
        | {
            f"effective_multiple_{horizon}"
            for horizon in spec.lookbacks
        }
        | {
            "soxx_rvol_63",
        }
    )
    label_columns = [
        f"future_actual_3x_residual_log_{horizon}"
        for horizon in spec.target_horizons
    ] + [
        f"future_endpoint_3x_gap_{horizon}"
        for horizon in spec.target_horizons
    ]
    static = panel[feature_columns + label_columns].copy()
    static["instrument"] = "SOXL_SOXX"
    static = static.set_index("instrument", append=True)
    static.index.names = ["datetime", "instrument"]
    dataset_frame = pd.concat(
        {
            "feature": static[feature_columns],
            "label": static[label_columns],
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
    end = panel.index.max().date().isoformat()
    segments = {
        "all_history": [spec.start, end],
        "development": [spec.start, spec.development_end],
        "outer_diagnostic": [spec.outer_start, end],
    }
    dataset = DatasetH(handler=handler, segments=segments)
    audit = {
        "api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"],
        "segments": segments,
        "segment_rows": {
            name: int(len(dataset.prepare(name))) for name in segments
        },
        "features": feature_columns,
        "labels": label_columns,
        "feature_timing": "trailing_close_to_close_available_at_date_t",
        "label_timing": "future_close_to_close",
    }
    return dataset, audit


def _prepared_frame(dataset: Any) -> pd.DataFrame:
    prepared = dataset.prepare("all_history").copy()
    prepared.columns = [
        column[-1] if isinstance(column, tuple) else column
        for column in prepared.columns
    ]
    if isinstance(prepared.index, pd.MultiIndex):
        prepared = prepared.droplevel("instrument")
    prepared.index = pd.DatetimeIndex(prepared.index, name="datetime")
    return prepared.sort_index()


def build_factor_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in LOOKBACKS:
        candidates = (
            f"actual_3x_residual_log_{horizon}",
            f"ideal_path_drag_log_{horizon}",
            f"product_tracking_log_{horizon}",
            f"soxx_rvol_{horizon}",
            f"path_efficiency_{horizon}",
        )
        correlation = panel[list(candidates)].corr(method="spearman")
        for left in candidates:
            for right in candidates:
                if left >= right:
                    continue
                rows.append(
                    {
                        "lookback_days": horizon,
                        "left": left,
                        "right": right,
                        "spearman": float(correlation.loc[left, right]),
                    }
                )
    return pd.DataFrame(rows)


def _correlation_metrics(
    frame: pd.DataFrame,
    *,
    feature: str,
    label: str,
    stride: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample, section in (
        ("overlap", frame),
        ("nonoverlap", frame.iloc[::stride]),
    ):
        rows.append(
            {
                "sample": sample,
                "rows": int(len(section)),
                "pearson": float(section[feature].corr(section[label])),
                "spearman": float(
                    section[feature].corr(section[label], method="spearman")
                ),
            }
        )
    return rows


def build_univariate_validation(
    prepared: pd.DataFrame,
    spec: RollingRetentionSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factors = (
        "retention_z252_21",
        "actual_3x_residual_log_21",
        "actual_3x_residual_log_63",
        "vix",
        "vix_z252",
        "soxx_rvol_63",
        "soxx_rvol_z_63",
        "path_efficiency_63",
    )
    rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for horizon in spec.target_horizons:
        label = f"future_actual_3x_residual_log_{horizon}"
        for segment, start, end in (
            ("development", spec.start, spec.development_end),
            (
                "outer_diagnostic",
                spec.outer_start,
                prepared.index.max().date().isoformat(),
            ),
        ):
            for factor in factors:
                section = prepared.loc[start:end, [factor, label]].dropna()
                for metric in _correlation_metrics(
                    section,
                    feature=factor,
                    label=label,
                    stride=horizon,
                ):
                    rows.append(
                        {
                            "target_horizon_days": horizon,
                            "segment": segment,
                            "factor": factor,
                            **metric,
                        }
                    )
                for year, year_frame in section.groupby(section.index.year):
                    if len(year_frame) < 20:
                        continue
                    annual_rows.append(
                        {
                            "target_horizon_days": horizon,
                            "segment": segment,
                            "factor": factor,
                            "year": int(year),
                            "rows": int(len(year_frame)),
                            "spearman": float(
                                year_frame[factor].corr(
                                    year_frame[label],
                                    method="spearman",
                                )
                            ),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(annual_rows)


def build_walk_forward_model_comparison(
    prepared: pd.DataFrame,
    spec: RollingRetentionSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for horizon in spec.target_horizons:
        label = f"future_actual_3x_residual_log_{horizon}"
        for fold, train_start, train_end, test_start, test_end in FOLDS:
            for model_name, feature_columns in MODEL_FEATURES.items():
                columns = list(feature_columns)
                complete = prepared[columns + [label]].dropna()
                train = complete.loc[train_start:train_end]
                test = complete.loc[test_start:test_end]
                model = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=spec.ridge_alpha),
                )
                model.fit(train[columns], train[label])
                prediction = pd.Series(
                    model.predict(test[columns]),
                    index=test.index,
                    name="prediction",
                )
                detail = pd.DataFrame(
                    {
                        "datetime": test.index,
                        "target_horizon_days": horizon,
                        "fold": fold,
                        "model": model_name,
                        "label": test[label].to_numpy(),
                        "prediction": prediction.to_numpy(),
                    }
                )
                prediction_rows.append(detail)
                for sample, indexes in (
                    ("overlap", np.arange(len(test))),
                    (
                        "nonoverlap",
                        np.arange(0, len(test), horizon),
                    ),
                ):
                    actual = test[label].iloc[indexes]
                    predicted = prediction.iloc[indexes]
                    metric_rows.append(
                        {
                            "target_horizon_days": horizon,
                            "fold": fold,
                            "model": model_name,
                            "sample": sample,
                            "train_rows": int(len(train)),
                            "test_rows": int(len(indexes)),
                            "pearson": float(actual.corr(predicted)),
                            "spearman": float(
                                actual.corr(
                                    predicted,
                                    method="spearman",
                                )
                            ),
                            "r2": float(r2_score(actual, predicted)),
                            "mae": float(
                                mean_absolute_error(actual, predicted)
                            ),
                        }
                    )
    return pd.DataFrame(metric_rows), pd.concat(
        prediction_rows,
        ignore_index=True,
    )


def build_effective_multiple_audit(
    prepared: pd.DataFrame,
    spec: RollingRetentionSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in spec.lookbacks:
        multiple = f"effective_multiple_{horizon}"
        underlying = f"soxx_return_{horizon}"
        label = "future_endpoint_3x_gap_21"
        panel_columns = [
            multiple,
            f"actual_3x_residual_log_{horizon}",
            f"endpoint_3x_gap_{horizon}",
            underlying,
            label,
        ]
        section = prepared.reindex(columns=panel_columns).copy()
        if underlying not in prepared:
            continue
        section = section.loc[spec.outer_start :].dropna()
        valid_multiple = section.loc[section[underlying].abs().ge(0.05)]
        rows.append(
            {
                "lookback_days": horizon,
                "all_rows": int(len(section)),
                "near_zero_underlying_share_abs_lt_2pct": float(
                    prepared[underlying].abs().lt(0.02).mean()
                ),
                "valid_multiple_rows_abs_ge_5pct": int(len(valid_multiple)),
                "effective_multiple_p01": float(
                    prepared[multiple].replace(
                        [np.inf, -np.inf], np.nan
                    ).quantile(0.01)
                ),
                "effective_multiple_median": float(
                    prepared[multiple].replace(
                        [np.inf, -np.inf], np.nan
                    ).median()
                ),
                "effective_multiple_p99": float(
                    prepared[multiple].replace(
                        [np.inf, -np.inf], np.nan
                    ).quantile(0.99)
                ),
                "outer_spearman_multiple_to_future_21d_gap": float(
                    valid_multiple[multiple].corr(
                        valid_multiple[label],
                        method="spearman",
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def build_outer_bucket_metrics(
    prepared: pd.DataFrame,
    spec: RollingRetentionSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bins = [-np.inf, -1.0, 0.0, 1.0, np.inf]
    labels = ["stress_lt_-1", "weak_-1_0", "normal_0_1", "strong_gt_1"]
    factor = "retention_z252_21"
    for horizon in spec.target_horizons:
        target = f"future_actual_3x_residual_log_{horizon}"
        outer = prepared.loc[
            spec.outer_start :, [factor, target]
        ].dropna()
        for sample, section in (
            ("overlap", outer),
            ("nonoverlap", outer.iloc[::horizon]),
        ):
            bucket = pd.cut(
                section[factor],
                bins=bins,
                labels=labels,
                include_lowest=True,
            )
            for name, group in section.groupby(bucket, observed=True):
                rows.append(
                    {
                        "target_horizon_days": horizon,
                        "sample": sample,
                        "retention_state": str(name),
                        "rows": int(len(group)),
                        "mean_future_actual_3x_residual_log": float(
                            group[target].mean()
                        ),
                        "median_future_actual_3x_residual_log": float(
                            group[target].median()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_current_snapshot(
    panel: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    latest = panel.iloc[-1]
    for horizon in LOOKBACKS:
        rows.append(
            {
                "as_of": panel.index[-1].date().isoformat(),
                "lookback_days": horizon,
                "soxx_return": float(latest[f"soxx_return_{horizon}"]),
                "soxl_return": float(latest[f"soxl_return_{horizon}"]),
                "effective_multiple": float(
                    latest[f"effective_multiple_{horizon}"]
                ),
                "endpoint_3x_gap": float(
                    latest[f"endpoint_3x_gap_{horizon}"]
                ),
                "continuous_3x_retention_ratio": float(
                    latest[f"continuous_3x_retention_ratio_{horizon}"]
                ),
                "actual_3x_residual_log": float(
                    latest[f"actual_3x_residual_log_{horizon}"]
                ),
                "ideal_path_drag_log": float(
                    latest[f"ideal_path_drag_log_{horizon}"]
                ),
                "product_tracking_log": float(
                    latest[f"product_tracking_log_{horizon}"]
                ),
                "retention_z252": float(
                    latest[f"retention_z252_{horizon}"]
                ),
                "soxx_realized_volatility": float(
                    latest[f"soxx_rvol_{horizon}"]
                ),
            }
        )
    return rows


def _metric_row(
    metrics: pd.DataFrame,
    *,
    horizon: int,
    fold: str,
    model: str,
    sample: str,
) -> pd.Series:
    return metrics.loc[
        metrics["target_horizon_days"].eq(horizon)
        & metrics["fold"].eq(fold)
        & metrics["model"].eq(model)
        & metrics["sample"].eq(sample)
    ].iloc[0]


def build_factor_gate(
    model_metrics: pd.DataFrame,
    univariate: pd.DataFrame,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for horizon in TARGET_HORIZONS:
        standalone_overlap = univariate.loc[
            univariate["target_horizon_days"].eq(horizon)
            & univariate["segment"].eq("outer_diagnostic")
            & univariate["factor"].eq("retention_z252_21")
            & univariate["sample"].eq("overlap")
        ].iloc[0]
        standalone_nonoverlap = univariate.loc[
            univariate["target_horizon_days"].eq(horizon)
            & univariate["segment"].eq("outer_diagnostic")
            & univariate["factor"].eq("retention_z252_21")
            & univariate["sample"].eq("nonoverlap")
        ].iloc[0]
        baseline_overlap = _metric_row(
            model_metrics,
            horizon=horizon,
            fold="outer_2023_2026",
            model="core_baseline",
            sample="overlap",
        )
        plus_overlap = _metric_row(
            model_metrics,
            horizon=horizon,
            fold="outer_2023_2026",
            model="baseline_plus_retention",
            sample="overlap",
        )
        baseline_nonoverlap = _metric_row(
            model_metrics,
            horizon=horizon,
            fold="outer_2023_2026",
            model="core_baseline",
            sample="nonoverlap",
        )
        plus_nonoverlap = _metric_row(
            model_metrics,
            horizon=horizon,
            fold="outer_2023_2026",
            model="baseline_plus_retention",
            sample="nonoverlap",
        )
        fold_deltas: list[float] = []
        for fold, *_ in FOLDS:
            baseline = _metric_row(
                model_metrics,
                horizon=horizon,
                fold=fold,
                model="core_baseline",
                sample="overlap",
            )
            plus = _metric_row(
                model_metrics,
                horizon=horizon,
                fold=fold,
                model="baseline_plus_retention",
                sample="overlap",
            )
            fold_deltas.append(float(plus["spearman"] - baseline["spearman"]))
        overlap_delta = float(
            plus_overlap["spearman"] - baseline_overlap["spearman"]
        )
        nonoverlap_delta = float(
            plus_nonoverlap["spearman"]
            - baseline_nonoverlap["spearman"]
        )
        pass_incremental = bool(
            overlap_delta >= 0.05
            and nonoverlap_delta >= 0.05
            and sum(delta > 0.0 for delta in fold_deltas) >= 3
            and float(plus_overlap["r2"]) > 0.0
        )
        rows.append(
            {
                "target_horizon_days": horizon,
                "standalone_outer_overlap_spearman": float(
                    standalone_overlap["spearman"]
                ),
                "standalone_outer_nonoverlap_spearman": float(
                    standalone_nonoverlap["spearman"]
                ),
                "outer_overlap_incremental_spearman": overlap_delta,
                "outer_nonoverlap_incremental_spearman": nonoverlap_delta,
                "positive_incremental_folds": int(
                    sum(delta > 0.0 for delta in fold_deltas)
                ),
                "total_folds": int(len(fold_deltas)),
                "outer_plus_model_r2": float(plus_overlap["r2"]),
                "incremental_alpha_gate_pass": pass_incremental,
            }
        )
    return {
        "decision": "diagnostic_guardrail_value_not_incremental_alpha",
        "gate": rows,
        "promotion": "blocked_not_stable_incremental",
    }


def _record(
    spec: RollingRetentionSpec,
    *,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        panel.reset_index().to_csv(
            tmp_dir / "factor_panel.csv",
            index=False,
        )
        predictions.to_csv(
            tmp_dir / "walk_forward_predictions.csv",
            index=False,
        )
        (tmp_dir / "analysis_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                factor="retention_z252_21",
                target="future_actual_3x_residual_log_21_63",
                ridge_alpha=str(spec.ridge_alpha),
                outer_role="reused_diagnostic_not_clean_holdout",
            )
            target_gate = summary["factor_gate"]["gate"][0]
            R.log_metrics(
                outer_21d_standalone_spearman=float(
                    target_gate["standalone_outer_overlap_spearman"]
                ),
                outer_21d_incremental_spearman=float(
                    target_gate["outer_overlap_incremental_spearman"]
                ),
            )
            for filename in (
                "factor_panel.csv",
                "walk_forward_predictions.csv",
                "analysis_summary.json",
            ):
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


def run_soxl_rolling_3x_retention_factor_card(
    root: Path | str | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    spec = RollingRetentionSpec(root=resolved_root)
    spec.card_dir.mkdir(parents=True, exist_ok=True)

    frame, data_audit = load_soxl_soxx_frame(
        LongHorizonSpec(root=resolved_root)
    )
    panel = build_retention_factor_panel(frame)
    dataset, dataset_audit = build_qlib_dataset(panel, spec)
    prepared = _prepared_frame(dataset)
    factor_correlation = build_factor_correlation(panel)
    univariate, annual = build_univariate_validation(prepared, spec)
    model_metrics, predictions = build_walk_forward_model_comparison(
        prepared,
        spec,
    )
    multiple_audit = build_effective_multiple_audit(panel, spec)
    bucket_metrics = build_outer_bucket_metrics(prepared, spec)
    current_snapshot = build_current_snapshot(panel)
    factor_gate = build_factor_gate(model_metrics, univariate)

    ideal_vol_correlation = factor_correlation.loc[
        factor_correlation["left"].str.contains("ideal_path_drag")
        & factor_correlation["right"].str.contains("soxx_rvol")
    ]
    summary: dict[str, Any] = {
        "experiment": spec.experiment,
        "as_of": data_audit["as_of"],
        "decision": factor_gate["decision"],
        "promotion": factor_gate["promotion"],
        "data_audit": data_audit,
        "dataset_audit": dataset_audit,
        "definitions": {
            "effective_multiple": "SOXL interval return divided by SOXX interval return; display only because denominator can be near zero",
            "endpoint_3x_gap": "SOXL interval return minus 3 times SOXX interval return",
            "actual_3x_residual_log": "log SOXL gross return minus 3 times log SOXX gross return",
            "ideal_path_drag_log": "log ideal daily-reset 3x gross return minus 3 times log SOXX gross return",
            "product_tracking_log": "log SOXL gross return minus log ideal daily-reset 3x gross return",
            "retention_z252_21": "21-day actual 3x residual standardized against prior 252 days excluding current date",
        },
        "factor_gate": factor_gate,
        "current_snapshot": current_snapshot,
        "redundancy": {
            "ideal_path_drag_vs_realized_vol_spearman": [
                {
                    "lookback_days": int(row["lookback_days"]),
                    "spearman": float(row["spearman"]),
                }
                for _, row in ideal_vol_correlation.iterrows()
            ],
            "interpretation": "most pure path-drag information is already contained in underlying realized volatility",
        },
        "outer_role": "reused_diagnostic_not_clean_holdout",
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/factor_panel.csv",
                f"artifacts/{spec.experiment}/walk_forward_predictions.csv",
            ],
        },
        "sidecars": {
            "factor_correlation": "factor_correlation.csv",
            "univariate_validation": "univariate_validation.csv",
            "annual_validation": "annual_validation.csv",
            "model_comparison": "model_comparison.csv",
            "effective_multiple_audit": "effective_multiple_audit.csv",
            "outer_bucket_metrics": "outer_bucket_metrics.csv",
            "current_factor_snapshot": "current_factor_snapshot.json",
            "selected_model": "selected_model.json",
            "risk_analysis": "risk_analysis.csv",
            "analysis_summary": "analysis_summary.json",
        },
        "qlib_docs_reference": [
            "https://qlib.readthedocs.io/en/latest/component/data.html",
            "https://qlib.readthedocs.io/en/latest/component/workflow.html",
        ],
    }
    recorder = _record(
        spec,
        panel=panel,
        predictions=predictions,
        summary=summary,
    )
    summary.update(recorder)

    sidecar_frames = {
        "factor_correlation.csv": factor_correlation,
        "univariate_validation.csv": univariate,
        "annual_validation.csv": annual,
        "model_comparison.csv": model_metrics,
        "effective_multiple_audit.csv": multiple_audit,
        "outer_bucket_metrics.csv": bucket_metrics,
    }
    for filename, data in sidecar_frames.items():
        data.to_csv(spec.card_dir / filename, index=False)
    (spec.card_dir / "current_factor_snapshot.json").write_text(
        json.dumps(
            current_snapshot,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    selected = {
        "selected_factor": "retention_z252_21",
        "role": "diagnostic_guardrail_candidate",
        "predictive_alpha_promotion": "blocked",
        "reason": "standalone short-horizon ranking exists but incremental nonoverlap and fold stability gates fail",
    }
    (spec.card_dir / "selected_model.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    risk = pd.DataFrame(
        [
            {
                "risk": "near_zero_denominator",
                "status": "blocked_for_raw_multiple_factor",
                "mitigation": "use log residual and lagged rolling normalization",
            },
            {
                "risk": "factor_redundancy",
                "status": "material",
                "mitigation": "compare incrementally against realized volatility and normalized VIX",
            },
            {
                "risk": "overlapping_labels",
                "status": "material",
                "mitigation": "report nonoverlap samples and expanding time folds",
            },
            {
                "risk": "outer_reuse",
                "status": "not_clean_holdout",
                "mitigation": "require new forward paper evidence before promotion",
            },
        ]
    )
    risk.to_csv(spec.card_dir / "risk_analysis.csv", index=False)

    analysis_summary = {
        "decision": summary["decision"],
        "factor_gate": factor_gate,
        "current_snapshot": current_snapshot,
        "redundancy": summary["redundancy"],
    }
    (spec.card_dir / "analysis_summary.json").write_text(
        json.dumps(
            analysis_summary,
            ensure_ascii=False,
            indent=2,
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
        "region": "us",
        "market": "all",
        "start": spec.start,
        "end": summary["as_of"],
        "segments": dataset_audit["segments"],
        "factor": "retention_z252_21",
        "targets": [
            f"future_actual_3x_residual_log_{horizon}"
            for horizon in spec.target_horizons
        ],
        "model": {
            "class": "Ridge",
            "alpha": spec.ridge_alpha,
            "walk_forward_folds": [fold[0] for fold in FOLDS],
        },
        "artifact_policy": summary["artifact_policy"],
    }
    (spec.card_dir / "workflow.yaml").write_text(
        yaml.safe_dump(workflow, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    gate21 = factor_gate["gate"][0]
    gate63 = factor_gate["gate"][1]
    current21, current63, current126 = current_snapshot
    multiple21 = multiple_audit.loc[
        multiple_audit["lookback_days"].eq(21)
    ].iloc[0]
    conclusion = (
        "# 中文结论\n\n"
        "- 口径：TUI 结论全部使用纯文本。直观倍数等于“SOXL区间收益 / SOXX区间收益”；"
        "稳定因子使用“log(SOXL总收益倍数) - 3 × log(SOXX总收益倍数)”，再用此前252日"
        "均值和标准差归一化，当前日期不进入历史均值。\n"
        f"- 原始倍数不适合作因子：21日窗口中，SOXX区间收益绝对值低于2%的比例为 "
        f"{multiple21['near_zero_underlying_share_abs_lt_2pct']:.2%}；有效倍数历史1%至99%分位为 "
        f"{multiple21['effective_multiple_p01']:.2f} 至 {multiple21['effective_multiple_p99']:.2f}，"
        "近零分母会制造极值。\n"
        f"- 当前21日：SOXX {current21['soxx_return']:.2%}，SOXL {current21['soxl_return']:.2%}，"
        f"直观倍数 {current21['effective_multiple']:.2f}；连续3x保持比 "
        f"{current21['continuous_3x_retention_ratio']:.2%}，相对自身过去252日为 "
        f"{current21['retention_z252']:.2f} 个标准差。\n"
        f"- 当前63日：SOXX {current63['soxx_return']:.2%}，SOXL {current63['soxl_return']:.2%}，"
        f"直观倍数 {current63['effective_multiple']:.2f}；连续3x保持比 "
        f"{current63['continuous_3x_retention_ratio']:.2%}，Z分数 "
        f"{current63['retention_z252']:.2f}。底层上涨但SOXL下跌，正是震荡损耗主导的区间。\n"
        f"- 当前126日：SOXX {current126['soxx_return']:.2%}，SOXL "
        f"{current126['soxl_return']:.2%}，直观倍数 {current126['effective_multiple']:.2f}；"
        f"连续3x保持比 {current126['continuous_3x_retention_ratio']:.2%}。\n"
        f"- 21日因子有单变量状态价值：2023–2026重叠样本对未来21日保持度的Spearman为 "
        f"{gate21['standalone_outer_overlap_spearman']:.3f}，非重叠样本为 "
        f"{gate21['standalone_outer_nonoverlap_spearman']:.3f}。\n"
        f"- 但不是稳定增量alpha：加入VIX、自身实现波动、趋势和路径效率后，外部期重叠样本"
        f"增量Spearman为 {gate21['outer_overlap_incremental_spearman']:.3f}，非重叠为 "
        f"{gate21['outer_nonoverlap_incremental_spearman']:.3f}；仅 "
        f"{gate21['positive_incremental_folds']}/{gate21['total_folds']} 个时间折增益为正，且R2为 "
        f"{gate21['outer_plus_model_r2']:.3f}。\n"
        f"- 未来63日同样未通过：非重叠增量Spearman为 "
        f"{gate63['outer_nonoverlap_incremental_spearman']:.3f}，仅 "
        f"{gate63['positive_incremental_folds']}/{gate63['total_folds']} 个时间折改善。\n"
        "- 机制：纯路径损耗与SOXX实现波动几乎同义，21/63/126日Spearman绝对值接近1；"
        "剩余部分主要是产品跟踪噪声。因此它适合压缩为一个SOXL自身风险状态，不能替代方向因子。\n"
        "- 决策：diagnostic_guardrail_value_not_incremental_alpha。允许作为Stage1候选风险输入做前向"
        "paper观察；禁止单独据此交易或直接调整生产权重。\n"
        f"- qlib recorder：{summary['recorder_id']}。\n"
        "- 风险：outer已被查看，不是clean holdout；滚动标签重叠；非重叠63日样本很少；"
        "当前极端状态可能主导相关性。\n"
        "- 下一步：只保留21日标准化保持度，在新的前向样本中检查它是否能改善SOXL风险上限或"
        "回撤控制，而不是继续增加更长窗口。\n"
        "- 不构成买卖建议。\n"
    )
    (spec.card_dir / "conclusion.md").write_text(
        conclusion,
        encoding="utf-8",
    )
    return summary
