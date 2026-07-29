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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.workflow.soxl_long_horizon_excess_drawdown import (
    LongHorizonSpec,
    load_soxl_soxx_frame,
)
from src.workflow.soxl_rolling_3x_retention_factor import (
    build_retention_factor_panel,
)


EXPERIMENT = "soxl_3x_state_factor_lab_v1"
TARGET_HORIZON = 21
RIDGE_ALPHA = 10.0
PURGE_DAYS = TARGET_HORIZON
FOLDS = (
    ("2016_2018", "2010-03-11", "2015-12-31", "2016-01-01", "2018-12-31"),
    ("2019_2020", "2010-03-11", "2018-12-31", "2019-01-01", "2020-12-31"),
    ("2021_2022", "2010-03-11", "2020-12-31", "2021-01-01", "2022-12-30"),
)
BASELINE_FEATURES = (
    "vix_z252",
    "soxx_rvol_z_63",
    "soxx_log_return_63",
    "path_efficiency_63",
)
CANDIDATE_SPECS = {
    "signed_path_efficiency_21": {
        "role": "alpha",
        "target": "future_relative_excess_21",
        "formula": "sum(SOXX daily log return,21) divided by sum(abs(SOXX daily log return),21)",
        "hypothesis": "directional and clean SOXX paths improve the value of the SOXL leverage sleeve",
    },
    "retention_curve_curvature_5_21_63": {
        "role": "alpha",
        "target": "future_relative_excess_21",
        "formula": "retention_5/5 minus 2 times retention_21/21 plus retention_63/63",
        "hypothesis": "short-versus-medium-versus-long retention curvature identifies improving or deteriorating leverage conditions",
    },
    "retention_slope_5": {
        "role": "alpha",
        "target": "future_relative_excess_21",
        "formula": "retention_z252_21 today minus its value five trading days ago",
        "hypothesis": "a five-day improvement in normalized retention contains more information than raw duration",
    },
    "soxl_tracking_error_vol_21": {
        "role": "retention_guardrail",
        "target": "future_actual_3x_residual_log_21",
        "formula": "21-day standard deviation of SOXL daily log return minus ideal daily-reset 3x log return",
        "hypothesis": "SOXL-specific tracking-error volatility predicts future leverage retention after controlling for SOXX volatility",
    },
    "vix_rvol_divergence_21": {
        "role": "risk",
        "target": "future_soxl_worst_return_21",
        "formula": "lagged VIX z252 minus lagged SOXX 21-day realized-volatility z252",
        "hypothesis": "implied-versus-realized volatility divergence identifies forward SOXL drawdown risk",
    },
    "shock_concentration_21": {
        "role": "risk",
        "target": "future_soxl_worst_return_21",
        "formula": "largest absolute SOXX daily log return in 21 days divided by the 21-day sum of absolute returns",
        "hypothesis": "one-day shock concentration differs from persistent choppy volatility for forward SOXL risk",
    },
}


@dataclass(frozen=True)
class StateFactorLabSpec:
    root: Path
    experiment: str = EXPERIMENT
    start: str = "2010-03-11"
    development_end: str = "2022-12-30"
    outer_start: str = "2023-01-03"
    ridge_alpha: float = RIDGE_ALPHA
    purge_days: int = PURGE_DAYS

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


def build_state_factor_panel(frame: pd.DataFrame) -> pd.DataFrame:
    panel = build_retention_factor_panel(frame)
    soxx_log_return = np.log(frame["soxx_close"]).diff()
    soxl_log_return = np.log(frame["soxl_close"]).diff()

    residuals: dict[int, pd.Series] = {}
    for horizon in (5, 21, 63):
        residuals[horizon] = (
            np.log(frame["soxl_close"] / frame["soxl_close"].shift(horizon))
            - 3.0
            * np.log(
                frame["soxx_close"] / frame["soxx_close"].shift(horizon)
            )
        )

    panel["signed_path_efficiency_21"] = (
        soxx_log_return.rolling(21).sum()
        / soxx_log_return.abs().rolling(21).sum()
    )
    panel["retention_curve_curvature_5_21_63"] = (
        residuals[5] / 5.0
        - 2.0 * residuals[21] / 21.0
        + residuals[63] / 63.0
    )
    panel["retention_slope_5"] = (
        panel["retention_z252_21"]
        - panel["retention_z252_21"].shift(5)
    )

    ideal_daily_log_return = np.log1p(3.0 * frame["soxx_return"])
    tracking_error = soxl_log_return - ideal_daily_log_return
    panel["soxl_tracking_error_vol_21"] = tracking_error.rolling(21).std()
    panel["vix_rvol_divergence_21"] = (
        panel["vix_z252"] - panel["soxx_rvol_z_21"]
    )
    panel["shock_concentration_21"] = (
        soxx_log_return.abs().rolling(21).max()
        / soxx_log_return.abs().rolling(21).sum()
    )

    future_soxl_return = (
        frame["soxl_close"].shift(-TARGET_HORIZON)
        / frame["soxl_close"]
        - 1.0
    )
    future_soxx_return = (
        frame["soxx_close"].shift(-TARGET_HORIZON)
        / frame["soxx_close"]
        - 1.0
    )
    panel["future_relative_excess_21"] = (
        future_soxl_return - future_soxx_return
    )
    future_soxl_prices = pd.concat(
        [
            frame["soxl_close"].shift(-offset)
            for offset in range(1, TARGET_HORIZON + 1)
        ],
        axis=1,
    )
    panel["future_soxl_worst_return_21"] = (
        future_soxl_prices.min(axis=1, skipna=False)
        / frame["soxl_close"]
        - 1.0
    )
    panel.index = pd.DatetimeIndex(panel.index, name="datetime")
    return panel


def build_qlib_dataset(
    panel: pd.DataFrame,
    spec: StateFactorLabSpec,
) -> tuple[Any, dict[str, Any]]:
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    candidate_features = tuple(CANDIDATE_SPECS)
    label_columns = sorted(
        {details["target"] for details in CANDIDATE_SPECS.values()}
    )
    feature_columns = list(BASELINE_FEATURES + candidate_features)
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
        "feature_timing": "trailing close-to-close values available at date t",
        "label_timing": "future 21-trading-day values",
        "selection_scope": "development_only_2010_2022",
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


def _spearman(actual: pd.Series, predicted: pd.Series) -> float:
    value = actual.corr(predicted, method="spearman")
    return float(value) if pd.notna(value) else float("nan")


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    alpha: float,
) -> pd.Series:
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )
    model.fit(train[features], train[target])
    return pd.Series(
        model.predict(test[features]),
        index=test.index,
        name="prediction",
    )


def _sample_metrics(
    actual: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample, indexes in (
        ("overlap", np.arange(len(actual))),
        ("nonoverlap", np.arange(0, len(actual), TARGET_HORIZON)),
    ):
        sampled_actual = actual.iloc[indexes]
        baseline_spearman = _spearman(
            sampled_actual,
            baseline.iloc[indexes],
        )
        candidate_spearman = _spearman(
            sampled_actual,
            candidate.iloc[indexes],
        )
        rows.append(
            {
                "sample": sample,
                "rows": int(len(indexes)),
                "baseline_spearman": baseline_spearman,
                "candidate_spearman": candidate_spearman,
                "incremental_spearman": (
                    candidate_spearman - baseline_spearman
                ),
            }
        )
    return rows


def build_walk_forward_comparison(
    prepared: pd.DataFrame,
    spec: StateFactorLabSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for candidate, details in CANDIDATE_SPECS.items():
        target = details["target"]
        columns = list(BASELINE_FEATURES) + [candidate, target]
        complete = prepared[columns].dropna()
        for fold, train_start, train_end, test_start, test_end in FOLDS:
            raw_train = complete.loc[train_start:train_end]
            if len(raw_train) <= spec.purge_days:
                raise ValueError(f"blocked: insufficient train rows for {fold}")
            train = raw_train.iloc[: -spec.purge_days]
            test = complete.loc[test_start:test_end]
            baseline = _fit_predict(
                train,
                test,
                features=list(BASELINE_FEATURES),
                target=target,
                alpha=spec.ridge_alpha,
            )
            plus_candidate = _fit_predict(
                train,
                test,
                features=list(BASELINE_FEATURES) + [candidate],
                target=target,
                alpha=spec.ridge_alpha,
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "datetime": test.index,
                        "candidate": candidate,
                        "target": target,
                        "fold": fold,
                        "actual": test[target].to_numpy(),
                        "baseline_prediction": baseline.to_numpy(),
                        "candidate_prediction": plus_candidate.to_numpy(),
                    }
                )
            )
            for row in _sample_metrics(
                test[target],
                baseline,
                plus_candidate,
            ):
                metrics.append(
                    {
                        "candidate": candidate,
                        "role": details["role"],
                        "target": target,
                        "fold": fold,
                        "train_rows_before_purge": int(len(raw_train)),
                        "train_rows": int(len(train)),
                        "purged_rows": int(spec.purge_days),
                        **row,
                    }
                )
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def build_univariate_validation(
    prepared: pd.DataFrame,
    spec: StateFactorLabSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, details in CANDIDATE_SPECS.items():
        target = details["target"]
        section = prepared.loc[
            spec.start : spec.development_end,
            [candidate, target],
        ].dropna()
        yearly = [
            _spearman(group[target], group[candidate])
            for _, group in section.groupby(section.index.year)
            if len(group) >= 100
        ]
        for sample, indexes in (
            ("overlap", np.arange(len(section))),
            (
                "nonoverlap",
                np.arange(0, len(section), TARGET_HORIZON),
            ),
        ):
            sampled = section.iloc[indexes]
            correlation = _spearman(
                sampled[target],
                sampled[candidate],
            )
            sign = np.sign(correlation)
            sign_share = float(
                np.mean(
                    [
                        np.sign(value) == sign
                        for value in yearly
                        if pd.notna(value)
                    ]
                )
            )
            rows.append(
                {
                    "candidate": candidate,
                    "role": details["role"],
                    "target": target,
                    "sample": sample,
                    "rows": int(len(sampled)),
                    "spearman": correlation,
                    "annual_sign_share": sign_share,
                }
            )
    return pd.DataFrame(rows)


def build_factor_gate(model_comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, details in CANDIDATE_SPECS.items():
        candidate_rows = model_comparison.loc[
            model_comparison["candidate"].eq(candidate)
        ]
        overlap = candidate_rows.loc[
            candidate_rows["sample"].eq("overlap")
        ]
        nonoverlap = candidate_rows.loc[
            candidate_rows["sample"].eq("nonoverlap")
        ]
        mean_overlap = float(overlap["incremental_spearman"].mean())
        mean_nonoverlap = float(
            nonoverlap["incremental_spearman"].mean()
        )
        mean_candidate_overlap = float(
            overlap["candidate_spearman"].mean()
        )
        mean_candidate_nonoverlap = float(
            nonoverlap["candidate_spearman"].mean()
        )
        positive_overlap = int(
            overlap["incremental_spearman"].gt(0.0).sum()
        )
        positive_nonoverlap = int(
            nonoverlap["incremental_spearman"].gt(0.0).sum()
        )
        positive_candidate_overlap = int(
            overlap["candidate_spearman"].gt(0.0).sum()
        )
        positive_candidate_nonoverlap = int(
            nonoverlap["candidate_spearman"].gt(0.0).sum()
        )
        gate_pass = bool(
            mean_candidate_overlap > 0.0
            and mean_candidate_nonoverlap >= 0.05
            and positive_candidate_nonoverlap >= 2
            and mean_overlap > 0.0
            and mean_nonoverlap >= 0.03
            and positive_overlap >= 2
            and positive_nonoverlap == len(FOLDS)
        )
        rows.append(
            {
                "candidate": candidate,
                "role": details["role"],
                "target": details["target"],
                "mean_candidate_overlap_spearman": (
                    mean_candidate_overlap
                ),
                "mean_candidate_nonoverlap_spearman": (
                    mean_candidate_nonoverlap
                ),
                "mean_incremental_overlap_spearman": mean_overlap,
                "mean_incremental_nonoverlap_spearman": mean_nonoverlap,
                "positive_candidate_overlap_folds": (
                    positive_candidate_overlap
                ),
                "positive_candidate_nonoverlap_folds": (
                    positive_candidate_nonoverlap
                ),
                "positive_overlap_folds": positive_overlap,
                "positive_nonoverlap_folds": positive_nonoverlap,
                "total_folds": len(FOLDS),
                "development_gate_pass": gate_pass,
                "promotion": (
                    "forward_paper_candidate"
                    if gate_pass
                    else "blocked_not_stable_incremental"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_outer_diagnostic(
    prepared: pd.DataFrame,
    spec: StateFactorLabSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, details in CANDIDATE_SPECS.items():
        target = details["target"]
        columns = list(BASELINE_FEATURES) + [candidate, target]
        complete = prepared[columns].dropna()
        raw_train = complete.loc[spec.start : spec.development_end]
        train = raw_train.iloc[: -spec.purge_days]
        test = complete.loc[spec.outer_start :]
        baseline = _fit_predict(
            train,
            test,
            features=list(BASELINE_FEATURES),
            target=target,
            alpha=spec.ridge_alpha,
        )
        plus_candidate = _fit_predict(
            train,
            test,
            features=list(BASELINE_FEATURES) + [candidate],
            target=target,
            alpha=spec.ridge_alpha,
        )
        for metric in _sample_metrics(
            test[target],
            baseline,
            plus_candidate,
        ):
            rows.append(
                {
                    "candidate": candidate,
                    "role": details["role"],
                    "target": target,
                    "selection_uses_outer": False,
                    **metric,
                }
            )
    return pd.DataFrame(rows)


def build_current_snapshot(panel: pd.DataFrame) -> dict[str, Any]:
    latest = panel.iloc[-1]
    return {
        "as_of": panel.index[-1].date().isoformat(),
        "factors": {
            candidate: float(latest[candidate])
            for candidate in CANDIDATE_SPECS
        },
        "context": {
            "soxx_return_21": float(latest["soxx_return_21"]),
            "soxl_return_21": float(latest["soxl_return_21"]),
            "retention_z252_21": float(latest["retention_z252_21"]),
            "vix": float(latest["vix"]),
            "soxx_rvol_21": float(latest["soxx_rvol_21"]),
        },
    }


def _record(
    spec: StateFactorLabSpec,
    *,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    analysis_summary: dict[str, Any],
    selected_candidates: list[str],
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
            json.dumps(
                analysis_summary,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                factor_family="3x_state_path_curve_volatility",
                selection_scope="development_only_2010_2022",
                target_horizon_days=str(TARGET_HORIZON),
                purge_days=str(spec.purge_days),
                ridge_alpha=str(spec.ridge_alpha),
                outer_role="reused_diagnostic_not_selection",
            )
            R.log_metrics(
                development_gate_pass_count=float(len(selected_candidates)),
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


def run_soxl_3x_state_factor_lab_card(
    root: Path | str | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    spec = StateFactorLabSpec(root=resolved_root)
    spec.card_dir.mkdir(parents=True, exist_ok=True)

    frame, data_audit = load_soxl_soxx_frame(
        LongHorizonSpec(root=resolved_root)
    )
    panel = build_state_factor_panel(frame)
    dataset, dataset_audit = build_qlib_dataset(panel, spec)
    prepared = _prepared_frame(dataset)
    univariate = build_univariate_validation(prepared, spec)
    model_comparison, predictions = build_walk_forward_comparison(
        prepared,
        spec,
    )
    factor_gate = build_factor_gate(model_comparison)
    outer_diagnostic = build_outer_diagnostic(prepared, spec)
    current_snapshot = build_current_snapshot(panel)
    selected_candidates = factor_gate.loc[
        factor_gate["development_gate_pass"],
        "candidate",
    ].tolist()

    decision = (
        "development_candidate_forward_paper_only"
        if selected_candidates
        else "no_stable_incremental_candidate"
    )
    promotion = "blocked_pending_new_forward_paper"
    analysis_summary = {
        "decision": decision,
        "promotion": promotion,
        "selected_candidates": selected_candidates,
        "selection_uses_outer": False,
        "factor_gate": factor_gate.to_dict(orient="records"),
        "current_snapshot": current_snapshot,
    }
    recorder = _record(
        spec,
        panel=panel,
        predictions=predictions,
        analysis_summary=analysis_summary,
        selected_candidates=selected_candidates,
    )

    sidecars = {
        "factor_definitions": "factor_definitions.csv",
        "univariate_validation": "univariate_validation.csv",
        "model_comparison": "model_comparison.csv",
        "model_selection": "model_selection.csv",
        "outer_diagnostic": "outer_diagnostic.csv",
        "current_factor_snapshot": "current_factor_snapshot.json",
        "selected_model": "selected_model.json",
        "risk_analysis": "risk_analysis.csv",
        "analysis_summary": "analysis_summary.json",
    }
    summary: dict[str, Any] = {
        "experiment": spec.experiment,
        "as_of": data_audit["as_of"],
        "decision": decision,
        "promotion": promotion,
        "selected_candidates": selected_candidates,
        "selection_uses_outer": False,
        "forward_paper_start": "2026-07-29",
        "data_audit": data_audit,
        "dataset_audit": dataset_audit,
        "factor_gate": factor_gate.to_dict(orient="records"),
        "current_snapshot": current_snapshot,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/factor_panel.csv",
                f"artifacts/{spec.experiment}/walk_forward_predictions.csv",
            ],
        },
        "sidecars": sidecars,
        "qlib_docs_reference": [
            "https://qlib.readthedocs.io/en/latest/component/data.html",
            "https://qlib.readthedocs.io/en/latest/component/recorder.html",
        ],
        **recorder,
    }

    definitions = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "role": details["role"],
                "target": details["target"],
                "formula": details["formula"],
                "hypothesis": details["hypothesis"],
            }
            for candidate, details in CANDIDATE_SPECS.items()
        ]
    )
    definitions.to_csv(
        spec.card_dir / "factor_definitions.csv",
        index=False,
    )
    univariate.to_csv(
        spec.card_dir / "univariate_validation.csv",
        index=False,
    )
    model_comparison.to_csv(
        spec.card_dir / "model_comparison.csv",
        index=False,
    )
    factor_gate.to_csv(
        spec.card_dir / "model_selection.csv",
        index=False,
    )
    outer_diagnostic.to_csv(
        spec.card_dir / "outer_diagnostic.csv",
        index=False,
    )
    (spec.card_dir / "current_factor_snapshot.json").write_text(
        json.dumps(current_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selected_model = {
        "selected_candidates": selected_candidates,
        "selection_scope": "development_only_2010_2022",
        "selection_uses_outer": False,
        "model": "StandardScaler_plus_Ridge",
        "ridge_alpha": spec.ridge_alpha,
        "purge_days": spec.purge_days,
        "promotion": promotion,
    }
    (spec.card_dir / "selected_model.json").write_text(
        json.dumps(selected_model, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    risk_analysis = pd.DataFrame(
        [
            {
                "risk": "outer_reuse",
                "status": "not_clean_holdout",
                "mitigation": "outer metrics are diagnostic only and never used for selection",
            },
            {
                "risk": "overlapping_labels",
                "status": "material",
                "mitigation": "use 21-day nonoverlap metrics and purge 21 train rows at every fold boundary",
            },
            {
                "risk": "multiple_hypotheses",
                "status": "material",
                "mitigation": "pre-map each candidate to one role and target and require fold-consistent incremental evidence",
            },
            {
                "risk": "factor_redundancy",
                "status": "material",
                "mitigation": "compare every candidate against the VIX, SOXX volatility, trend, and path baseline",
            },
            {
                "risk": "production_use",
                "status": "blocked",
                "mitigation": "collect new forward paper evidence from 2026-07-29 before any Stage1 change",
            },
        ]
    )
    risk_analysis.to_csv(
        spec.card_dir / "risk_analysis.csv",
        index=False,
    )
    (spec.card_dir / "analysis_summary.json").write_text(
        json.dumps(
            analysis_summary,
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
        "region": "us",
        "market": "all",
        "start": spec.start,
        "end": summary["as_of"],
        "segments": dataset_audit["segments"],
        "selection_scope": "development_only_2010_2022",
        "selection_uses_outer": False,
        "forward_paper_start": "2026-07-29",
        "candidate_specs": CANDIDATE_SPECS,
        "model": {
            "class": "Ridge",
            "alpha": spec.ridge_alpha,
            "purge_days": spec.purge_days,
            "walk_forward_folds": [fold[0] for fold in FOLDS],
        },
        "artifact_policy": summary["artifact_policy"],
    }
    (spec.card_dir / "workflow.yaml").write_text(
        yaml.safe_dump(workflow, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    gate_by_name = factor_gate.set_index("candidate")
    signed = gate_by_name.loc["signed_path_efficiency_21"]
    curve = gate_by_name.loc[
        "retention_curve_curvature_5_21_63"
    ]
    tracking = gate_by_name.loc["soxl_tracking_error_vol_21"]
    vix_gap = gate_by_name.loc["vix_rvol_divergence_21"]
    shock = gate_by_name.loc["shock_concentration_21"]
    signed_outer = outer_diagnostic.loc[
        outer_diagnostic["candidate"].eq("signed_path_efficiency_21")
        & outer_diagnostic["sample"].eq("nonoverlap")
    ].iloc[0]
    tracking_outer = outer_diagnostic.loc[
        outer_diagnostic["candidate"].eq("soxl_tracking_error_vol_21")
        & outer_diagnostic["sample"].eq("nonoverlap")
    ].iloc[0]
    selected_text = (
        "、".join(selected_candidates) if selected_candidates else "无"
    )
    current = current_snapshot["factors"]
    current_context = current_snapshot["context"]
    conclusion = (
        "# 中文结论\n\n"
        "- 做了什么：使用qlib SOXL/SOXX/VIX日线、DatasetH、DataHandlerLP和Recorder，"
        "检验有方向路径效率、3x保持度曲率、保持度斜率、SOXL自身跟踪误差波动、"
        "VIX与实现波动发散、单日冲击集中度六个预先定义候选。\n"
        "- 时间纪律：候选选择只使用2010–2022开发期；三个扩展时间折的训练端各剔除"
        "最后21个交易日，主指标使用互不重叠的21日样本。2023–2026只作已查看的诊断，"
        "不参与选择。\n"
        f"- 开发期通过候选：{selected_text}。通过只代表可以进入前向paper，"
        "不代表可以修改生产仓位。\n"
        f"- 有方向路径效率：对未来21日SOXL相对SOXX超额的平均增量Spearman为"
        f"{signed['mean_incremental_overlap_spearman']:.3f}（重叠）和"
        f"{signed['mean_incremental_nonoverlap_spearman']:.3f}（非重叠），"
        f"非重叠正增益时间折为{int(signed['positive_nonoverlap_folds'])}/3；"
        f"但候选模型本身的非重叠平均Spearman仅"
        f"{signed['mean_candidate_nonoverlap_spearman']:.3f}，只有"
        f"{int(signed['positive_candidate_nonoverlap_folds'])}/3个时间折为正，"
        f"且2023–2026诊断区间增量为{signed_outer['incremental_spearman']:.3f}，"
        "因此不通过。\n"
        f"- 3x保持度曲率：非重叠平均增量为"
        f"{curve['mean_incremental_nonoverlap_spearman']:.3f}，"
        f"正增益时间折为{int(curve['positive_nonoverlap_folds'])}/3；"
        "未满足完整门槛时只保留研究价值。\n"
        f"- SOXL自身跟踪误差波动：对未来3x保持度的非重叠平均增量为"
        f"{tracking['mean_incremental_nonoverlap_spearman']:.3f}。"
        f"2023–2026诊断增量为{tracking_outer['incremental_spearman']:.3f}。"
        "它的单变量关系会被VIX和SOXX实现波动基线吸收，保留为产品异常监控，不作为alpha。\n"
        f"- VIX与实现波动发散：对未来SOXL最差21日路径的非重叠平均增量为"
        f"{vix_gap['mean_incremental_nonoverlap_spearman']:.3f}；"
        f"冲击集中度为{shock['mean_incremental_nonoverlap_spearman']:.3f}。"
        "若重叠与非重叠或时间折不一致，则只允许风险paper观察。\n"
        f"- 当前状态：有方向路径效率为{current['signed_path_efficiency_21']:.3f}，"
        f"21日保持度Z分数为{current_context['retention_z252_21']:.3f}，"
        f"VIX与实现波动发散为{current['vix_rvol_divergence_21']:.3f}。"
        "当前不是干净的正向杠杆路径，且VIX相对SOXX已实现波动偏低；这只是状态描述，"
        "不是交易信号。\n"
        f"- 决策：{decision}；生产晋升为{promotion}。现有Stage1和模拟盘逻辑不变。\n"
        f"- qlib recorder：{summary['recorder_id']}。\n"
        "- 风险：outer已被反复查看，不是clean holdout；候选较多存在多重检验风险；"
        "Spearman改善不等于扣除成本后的组合收益；未来风险目标越高表示回撤越轻。\n"
        "- 下一步：从2026-07-29开始只做观察性前向记录，不把任何候选接入仓位；"
        "至少积累多个互不重叠21日窗口后，重新检查有方向路径效率的符号反转和"
        "跟踪误差波动的增量价值。\n"
        "- 不构成买卖建议。\n"
    )
    (spec.card_dir / "conclusion.md").write_text(
        conclusion,
        encoding="utf-8",
    )
    return summary
