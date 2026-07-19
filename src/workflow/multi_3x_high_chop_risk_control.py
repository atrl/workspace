"""Qlib-native transferable high-chop research workflow.

The workflow keeps model selection and probability calibration inside validation
segments. Outer SOXL years are consumed once for stitched evaluation only.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import make_pipeline

from src.backtest import RiskControlSpec, apply_multi_asset_risk_control, evaluate_multi_asset_risk_control
from src.data import QlibProviderConfig, build_qlib_dataset, build_qlib_store_from_frame, load_qlib_feature_frame
from src.factor import PATH_LABELS, TRANSFERABLE_FEATURES, build_multi_3x_high_chop_frame, purge_segment_tail


EXPERIMENT = "multi_3x_high_chop_risk_control_v1"
CARD_TYPE = "multi_3x_high_chop_risk_control_research_card"
QLIB_DOCS = "https://qlib.readthedocs.io/en/latest/"
RESEARCH_ETFS = (
    "SOXL",
    "TQQQ",
    "UPRO",
    "SPXL",
    "TECL",
    "UDOW",
    "TNA",
    "MIDU",
    "FAS",
    "LABU",
    "CURE",
    "NAIL",
    "RETL",
    "ERX",
    "DRN",
    "WANT",
)
UNDERLYING_GROUPS = {
    "SOXL": "SOXX",
    "TQQQ": "QQQ",
    "UPRO": "SPY",
    "SPXL": "SPY",
    "TECL": "XLK",
    "UDOW": "DIA",
    "TNA": "IWM",
    "MIDU": "MDY",
    "FAS": "XLF",
    "LABU": "XBI",
    "CURE": "XLV",
    "NAIL": "ITB",
    "RETL": "XRT",
    "ERX": "XLE",
    "DRN": "XLRE",
    "WANT": "XLY",
}
CALIBRATION_TEMPERATURES = (0.75, 1.0, 1.25, 1.5, 2.0)
MODEL_CANDIDATES = ("rule_baseline", "multinomial_logistic", "lightgbm_multiclass")


@dataclass(frozen=True)
class Multi3xHighChopRiskControlSpec:
    root: Path
    experiment: str = EXPERIMENT
    as_of: str = "2026-07-02"
    train_start: str = "2011-01-03"
    outer_test_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    label_horizon: int = 20
    cost_per_turnover: float = 0.0003
    stress_cost_per_turnover: float = 0.0010
    high_stress_cost_per_turnover: float = 0.0025
    min_class_samples: int = 50
    min_validation_class_samples: int = 20

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / self.experiment


def build_path_dataset(
    provider_uri: Path | str,
    *,
    instruments: list[str] | None = None,
    features: list[str] | None = None,
    train: tuple[str, str],
    valid: tuple[str, str],
    test: tuple[str, str],
):
    """Construct the official qlib DatasetH/DataHandlerLP/QlibDataLoader path."""

    feature_names = list(features or TRANSFERABLE_FEATURES)
    return build_qlib_dataset(
        QlibProviderConfig(provider_uri=Path(provider_uri), market="all", region="us", kernels=1),
        instruments=list(instruments or RESEARCH_ETFS),
        features=[name if name.startswith("$") else f"${name}" for name in feature_names],
        label="$path_label",
        train=train,
        valid=valid,
        test=test,
    )


def select_risk_control_candidate(metrics: pd.DataFrame) -> dict[str, Any]:
    """Select only from validation metrics with a deterministic tie break."""

    required = {"candidate", "validation_utility", "validation_log_loss"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"candidate metrics missing columns: {missing}")
    if metrics.empty:
        raise ValueError("candidate metrics are empty")
    grouped = (
        metrics.groupby("candidate", as_index=False)
        .agg(
            validation_utility=("validation_utility", "mean"),
            validation_log_loss=("validation_log_loss", "mean"),
            validation_fold_count=("candidate", "size"),
        )
        .sort_values(
            ["validation_utility", "validation_log_loss", "candidate"],
            ascending=[False, True, True],
            kind="mergesort",
        )
    )
    selected = grouped.iloc[0]
    return {
        "candidate": str(selected["candidate"]),
        "validation_utility": float(selected["validation_utility"]),
        "validation_log_loss": float(selected["validation_log_loss"]),
        "validation_fold_count": int(selected["validation_fold_count"]),
        "selection_metric": "validation_utility_desc_then_log_loss_asc",
        "selection_uses_outer_test": False,
    }


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0)
    scaled = np.power(clipped, 1.0 / float(temperature))
    return scaled / scaled.sum(axis=1, keepdims=True)


def select_validation_temperature(labels: pd.Series | np.ndarray, probabilities: np.ndarray) -> float:
    """Choose calibration temperature from validation labels only."""

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(PATH_LABELS) or len(y) != len(p):
        raise ValueError("validation probabilities must be N x 4 and align with labels")
    losses = {
        temperature: log_loss(y, _temperature_scale(p, temperature), labels=list(range(4)))
        for temperature in CALIBRATION_TEMPERATURES
    }
    return float(min(losses, key=lambda value: (losses[value], abs(value - 1.0))))


def evaluate_promotion_gates(
    metrics: dict[str, float],
    *,
    benchmark: dict[str, float],
    transfer_fold_pass_rate: float,
    annual_pass_count: int = 0,
    annual_fold_count: int = 4,
    worst_annual_drawdown: float | None = None,
    stress_10bps_passed: bool = False,
    return_to_drawdown_beats_benchmarks: bool = False,
) -> dict[str, Any]:
    """Apply the fixed, fail-closed promotion gates from the design."""

    max_drawdown = float(metrics.get("max_drawdown", -1.0))
    benchmark_drawdown = float(benchmark.get("max_drawdown", -1.0))
    relative_improvement = (
        (max_drawdown - benchmark_drawdown) / abs(benchmark_drawdown)
        if benchmark_drawdown < 0.0
        else -np.inf
    )
    worst_year = float(worst_annual_drawdown if worst_annual_drawdown is not None else -1.0)
    gates = {
        "drawdown_absolute": {"passed": max_drawdown >= -0.20, "value": max_drawdown, "threshold": -0.20},
        "drawdown_relative_improvement": {
            "passed": relative_improvement >= 0.20,
            "value": float(relative_improvement),
            "threshold": 0.20,
        },
        "annualized_volatility": {
            "passed": float(metrics.get("annualized_volatility", np.inf)) <= 0.35,
            "value": float(metrics.get("annualized_volatility", np.inf)),
            "threshold": 0.35,
        },
        "cvar_95": {"passed": float(metrics.get("cvar_95", -1.0)) >= -0.08, "value": float(metrics.get("cvar_95", -1.0)), "threshold": -0.08},
        "worst_five_session_return": {
            "passed": float(metrics.get("worst_five_session_return", -1.0)) >= -0.12,
            "value": float(metrics.get("worst_five_session_return", -1.0)),
            "threshold": -0.12,
        },
        "upside_capture": {"passed": float(metrics.get("upside_capture", 0.0)) >= 0.60, "value": float(metrics.get("upside_capture", 0.0)), "threshold": 0.60},
        "average_risky_weight": {
            "passed": float(metrics.get("average_risky_weight", 0.0)) >= 0.15,
            "value": float(metrics.get("average_risky_weight", 0.0)),
            "threshold": 0.15,
        },
        "return_to_drawdown_benchmarks": {"passed": bool(return_to_drawdown_beats_benchmarks)},
        "annual_fold_gate": {
            "passed": annual_fold_count >= 4 and annual_pass_count >= 3 and worst_year >= -0.25,
            "annual_pass_count": int(annual_pass_count),
            "annual_fold_count": int(annual_fold_count),
            "worst_annual_drawdown": worst_year,
        },
        "stress_10bps": {"passed": bool(stress_10bps_passed)},
        "transfer_fold_pass_rate": {
            "passed": float(transfer_fold_pass_rate) >= 0.60,
            "value": float(transfer_fold_pass_rate),
            "threshold": 0.60,
        },
    }
    all_passed = all(bool(item["passed"]) for item in gates.values())
    return {
        "gates": gates,
        "all_passed": all_passed,
        "promotion_decision": "promoted" if all_passed else "not_promoted_risk_control_gates_failed",
    }


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _aligned_probabilities(model: Any, features: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(model[-1].classes_ if hasattr(model, "__getitem__") else model.classes_, dtype=int)
    aligned = np.full((len(features), 4), 1e-9, dtype=float)
    for column, label in enumerate(classes):
        aligned[:, int(label)] = raw[:, column]
    return aligned / aligned.sum(axis=1, keepdims=True)


def _rule_probabilities(frame: pd.DataFrame) -> np.ndarray:
    chop = pd.to_numeric(frame["high_chop_intensity"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    ret = pd.to_numeric(frame["ret_20_pctile252"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    downside = pd.to_numeric(frame["downside_semivariance_20_pctile252"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    whipsaw = 0.10 + 0.65 * chop
    breakdown = 0.05 + 0.60 * downside * (1.0 - ret)
    breakout = 0.05 + 0.60 * ret * (1.0 - chop)
    benign = np.full(len(frame), 0.25)
    raw = np.column_stack([benign, breakout, breakdown, whipsaw])
    return raw / raw.sum(axis=1, keepdims=True)


def _fit_model(candidate: str, train: pd.DataFrame, feature_cols: list[str]):
    if candidate == "multinomial_logistic":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
        )
    elif candidate == "lightgbm_multiclass":
        from lightgbm import LGBMClassifier

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                objective="multiclass",
                num_class=4,
                n_estimators=60,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=30,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.5,
                random_state=43,
                n_jobs=1,
                verbose=-1,
            ),
        )
    else:
        raise ValueError(f"unsupported trainable candidate: {candidate}")
    model.fit(train[feature_cols], train["path_label"].astype(int))
    return model


def _candidate_probabilities(
    candidate: str,
    *,
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    if candidate == "rule_baseline":
        return _rule_probabilities(predict)
    return _aligned_probabilities(_fit_model(candidate, train, feature_cols), predict[feature_cols])


def run_multi_3x_high_chop_risk_control_card(*, root: Path | str | None = None) -> dict[str, Any]:
    """Run the full research card from already refreshed pair providers."""

    spec = Multi3xHighChopRiskControlSpec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: Multi3xHighChopRiskControlSpec) -> dict[str, Any]:
    # The execution body is intentionally below the public contracts so tests can
    # validate leakage and promotion semantics without triggering a live refresh.
    panel, source_audit = _load_research_panel(spec)
    featured = build_multi_3x_high_chop_frame(panel, label_horizon=spec.label_horizon)
    provider_manifest = build_qlib_store_from_frame(
        featured,
        provider_uri=spec.provider_uri,
        as_of=spec.as_of,
        source="qlib.data.D.features multi-provider normalized panel",
    )
    return _execute_nested_card(spec, featured, provider_manifest=provider_manifest, source_audit=source_audit)


def _load_research_panel(spec: Multi3xHighChopRiskControlSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    from src.workflow.multi_3x_etf_normalized_nested_wf import _load_pair_frame

    provider_map = {etf: etf.lower() for etf in RESEARCH_ETFS if etf != "SOXL"}
    provider_map["SOXL"] = "soxl"
    frames: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for etf in RESEARCH_ETFS:
        provider_uri = spec.root / "data" / "qlib" / provider_map[etf]
        config = QlibProviderConfig(provider_uri=provider_uri, market="all", region="us", kernels=1)
        underlying = UNDERLYING_GROUPS[etf]
        frame = _load_pair_frame(
            config,
            etf=etf,
            underlying=underlying,
            start=spec.train_start,
            end=spec.as_of,
        )
        frames.append(frame)
        audits[etf] = {
            "provider_uri": provider_uri.resolve().as_posix(),
            "underlying": underlying,
            "row_count": int(len(frame)),
            "min_datetime": pd.Timestamp(frame["datetime"].min()).date().isoformat(),
            "max_datetime": pd.Timestamp(frame["datetime"].max()).date().isoformat(),
            "source": "qlib.data.D.features",
        }
    panel = pd.concat(frames, ignore_index=True)
    common_end = panel.groupby("instrument")["datetime"].max().min()
    panel = panel.loc[pd.to_datetime(panel["datetime"]) <= pd.Timestamp(common_end)].copy()
    return panel.sort_values(["datetime", "instrument"]).reset_index(drop=True), audits


def _execute_nested_card(
    spec: Multi3xHighChopRiskControlSpec,
    featured: pd.DataFrame,
    *,
    provider_manifest: dict[str, Any],
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    provider_frame, readback_audit = _read_normalized_provider(spec)
    model_rows = provider_frame.dropna(subset=TRANSFERABLE_FEATURES).copy()
    model_rows["underlying_group"] = model_rows["instrument"].map(UNDERLYING_GROUPS)
    model_selection_parts: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    rule_prediction_parts: list[pd.DataFrame] = []
    dataset_evidence: list[dict[str, Any]] = []

    for outer_year in spec.outer_test_years:
        # A three-year validation window preserves the fixed 20-observation
        # minimum for the rare whipsaw class without weakening the class gate.
        train_end = f"{outer_year - 4}-12-31"
        valid_range = (f"{outer_year - 3}-01-01", f"{outer_year - 1}-12-31")
        test_end = min(pd.Timestamp(spec.as_of), pd.Timestamp(f"{outer_year}-12-31")).date().isoformat()
        test_range = (f"{outer_year}-01-01", test_end)
        dataset = build_path_dataset(
            spec.provider_uri,
            train=(spec.train_start, train_end),
            valid=valid_range,
            test=test_range,
        )
        prepared_rows = {
            segment: int(len(dataset.prepare(segment))) for segment in ("train", "valid", "test")
        }
        dataset_evidence.append(
            {
                "outer_year": int(outer_year),
                "qlib_dataset_api": dataset.qlib_dataset_api,
                "segments": dataset.qlib_segments,
                "prepared_rows": prepared_rows,
            }
        )

        train = _segment(model_rows, spec.train_start, train_end, purge_end=train_end, horizon=spec.label_horizon)
        valid = _segment(
            model_rows,
            valid_range[0],
            valid_range[1],
            purge_end=valid_range[1],
            horizon=spec.label_horizon,
        )
        test = _segment(
            model_rows.loc[model_rows["instrument"].eq("SOXL")],
            test_range[0],
            test_range[1],
            purge_end=test_range[1],
            horizon=spec.label_horizon,
        )
        fold_metrics = _inner_leave_one_group_metrics(train, valid, spec=spec, outer_year=outer_year)
        model_selection_parts.append(fold_metrics)
        eligible = fold_metrics.loc[fold_metrics["status"].eq("ok")].copy()
        if eligible.empty:
            raise ValueError(f"outer fold {outer_year} blocked: no eligible validation candidate")
        selected = select_risk_control_candidate(eligible)
        selected["outer_year"] = int(outer_year)
        selected_rows.append(selected)

        outer_pred, calibration = _outer_probabilities(
            train,
            valid,
            test,
            candidate=selected["candidate"],
            feature_cols=list(TRANSFERABLE_FEATURES),
            spec=spec,
        )
        calibration["outer_year"] = int(outer_year)
        calibration_rows.append(calibration)
        outer_pred["outer_year"] = int(outer_year)
        outer_pred["selected_candidate"] = selected["candidate"]
        prediction_parts.append(outer_pred)

        rule_pred, _ = _outer_probabilities(
            train,
            valid,
            test,
            candidate="rule_baseline",
            feature_cols=list(TRANSFERABLE_FEATURES),
            spec=spec,
        )
        rule_pred["outer_year"] = int(outer_year)
        rule_prediction_parts.append(rule_pred)

    model_selection = pd.concat(model_selection_parts, ignore_index=True)
    selected_models = pd.DataFrame(selected_rows)
    calibration_metrics = pd.DataFrame(calibration_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    rule_predictions = pd.concat(rule_prediction_parts, ignore_index=True).sort_values("datetime").reset_index(drop=True)

    controlled = apply_multi_asset_risk_control(_controller_inputs(predictions))
    portfolio_report, strategy_metrics = evaluate_multi_asset_risk_control(
        controlled, cost_per_turnover=spec.cost_per_turnover
    )
    strategy_metrics = _extend_risk_metrics(strategy_metrics, portfolio_report)
    rule_controlled = apply_multi_asset_risk_control(_controller_inputs(rule_predictions))
    rule_report, rule_metrics = evaluate_multi_asset_risk_control(
        rule_controlled, cost_per_turnover=spec.cost_per_turnover
    )
    rule_metrics = _extend_risk_metrics(rule_metrics, rule_report)

    benchmark_comparison = _benchmark_comparison(
        predictions,
        controlled,
        strategy_metrics,
        rule_metrics,
        cost_per_turnover=spec.cost_per_turnover,
    )
    annual_metrics = _annual_metrics(portfolio_report, controlled)
    regime_metrics = _regime_metrics(portfolio_report, controlled)
    window_metrics = _window_metrics(portfolio_report, controlled)
    stress_metrics = _cost_stress(controlled, spec)
    strategy_metrics["average_risky_weight"] = float(controlled["risky_weight"].mean())
    benchmark_ref = benchmark_comparison.set_index("benchmark").loc["base_constrained_ref"].to_dict()
    benchmark_for_gate = {
        "max_drawdown": float(benchmark_ref["max_drawdown"]),
        "annualized_volatility": float(benchmark_ref["annualized_volatility"]),
        "cvar_95": float(benchmark_ref["cvar_95"]),
    }
    selected_transfer = model_selection.merge(
        selected_models[["outer_year", "candidate"]],
        on=["outer_year", "candidate"],
        how="inner",
    )
    transfer_pass_rate = float(selected_transfer["transfer_pass"].mean()) if len(selected_transfer) else 0.0
    annual_pass = annual_metrics["passes_drawdown_volatility"].astype(bool)
    selected_rtd = float(strategy_metrics["annualized_return"] / max(abs(strategy_metrics["max_drawdown"]), 1e-9))
    benchmark_rtd = benchmark_comparison.loc[
        benchmark_comparison["benchmark"].isin(
            ["constant_soxl_average_exposure", "matched_beta_static", "soxx_buy_hold"]
        ),
        "return_to_drawdown",
    ]
    gates = evaluate_promotion_gates(
        strategy_metrics,
        benchmark=benchmark_for_gate,
        transfer_fold_pass_rate=transfer_pass_rate,
        annual_pass_count=int(annual_pass.sum()),
        annual_fold_count=int(len(annual_pass)),
        worst_annual_drawdown=float(annual_metrics["max_drawdown"].min()),
        stress_10bps_passed=bool(stress_metrics.loc[stress_metrics["cost_bps"].eq(10), "passes_core_risk"].all()),
        return_to_drawdown_beats_benchmarks=bool((selected_rtd > benchmark_rtd).all()),
    )

    return _record_and_write_card(
        spec,
        provider_manifest=provider_manifest,
        source_audit=source_audit,
        readback_audit=readback_audit,
        dataset_evidence=dataset_evidence,
        model_selection=model_selection,
        selected_models=selected_models,
        calibration_metrics=calibration_metrics,
        predictions=predictions,
        controlled=controlled,
        portfolio_report=portfolio_report,
        strategy_metrics=strategy_metrics,
        benchmark_comparison=benchmark_comparison,
        annual_metrics=annual_metrics,
        regime_metrics=regime_metrics,
        window_metrics=window_metrics,
        stress_metrics=stress_metrics,
        promotion=gates,
        transfer_pass_rate=transfer_pass_rate,
    )


def _read_normalized_provider(spec: Multi3xHighChopRiskControlSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    fields = [
        *[f"${name}" for name in TRANSFERABLE_FEATURES],
        "$path_label",
        "$open",
        "$underlying_open",
        "$_lagged_sigma_20",
    ]
    loaded = load_qlib_feature_frame(
        QlibProviderConfig(provider_uri=spec.provider_uri, market="all", region="us", kernels=1),
        instruments=list(RESEARCH_ETFS),
        fields=fields,
        start=spec.train_start,
        end=spec.as_of,
    )
    frame = loaded.reset_index().rename(columns={field: field.removeprefix("$") for field in fields})
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    audit = {
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "source": "qlib.data.D.features",
        "qlib_data_api": loaded.attrs["qlib_data_api"],
        "instruments": sorted(frame["instrument"].unique().tolist()),
        "fields": fields,
        "row_count": int(len(frame)),
        "max_datetime": frame["datetime"].max().date().isoformat(),
        "missing_feature_columns": [name for name in TRANSFERABLE_FEATURES if frame[name].dropna().empty],
    }
    if audit["missing_feature_columns"]:
        raise ValueError(f"qlib readback missing features: {audit['missing_feature_columns']}")
    return frame.sort_values(["datetime", "instrument"]).reset_index(drop=True), audit


def _segment(
    frame: pd.DataFrame,
    start: str,
    end: str,
    *,
    purge_end: str,
    horizon: int,
) -> pd.DataFrame:
    segment = frame.loc[frame["datetime"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    segment = purge_segment_tail(segment, end=purge_end, horizon=horizon)
    return segment.reset_index(drop=True)


def _has_class_support(frame: pd.DataFrame, minimum: int) -> bool:
    counts = frame["path_label"].dropna().astype(int).value_counts().reindex(range(4), fill_value=0)
    return bool(counts.ge(minimum).all())


def _probability_metrics(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    y = labels.astype(int).to_numpy()
    prediction = probabilities.argmax(axis=1)
    breakdown_actual = (y == PATH_LABELS["breakdown"]).astype(float)
    breakdown_probability = probabilities[:, PATH_LABELS["breakdown"]]
    predicted_breakdown = breakdown_probability >= 0.50
    actual_breakdown = breakdown_actual.astype(bool)
    recall = float((predicted_breakdown & actual_breakdown).sum() / max(actual_breakdown.sum(), 1))
    false_positive = float((predicted_breakdown & ~actual_breakdown).sum() / max((~actual_breakdown).sum(), 1))
    confidence = probabilities.max(axis=1)
    correctness = prediction == y
    calibration_error = 0.0
    for lower in np.linspace(0.0, 0.8, 5):
        mask = (confidence >= lower) & (confidence < lower + 0.2)
        if mask.any():
            calibration_error += float(mask.mean() * abs(confidence[mask].mean() - correctness[mask].mean()))
    loss = float(log_loss(y, probabilities, labels=list(range(4))))
    brier = float(np.mean((breakdown_probability - breakdown_actual) ** 2))
    balanced = float(balanced_accuracy_score(y, prediction))
    utility = -loss - brier - calibration_error + 0.25 * recall + 0.10 * balanced
    return {
        "validation_log_loss": loss,
        "breakdown_brier": brier,
        "breakdown_tail_recall": recall,
        "breakdown_false_positive_rate": false_positive,
        "calibration_error": float(calibration_error),
        "balanced_accuracy": balanced,
        "validation_utility": float(utility),
        "transfer_pass": bool(balanced >= 0.25 and recall >= 0.20),
    }


def _inner_leave_one_group_metrics(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    spec: Multi3xHighChopRiskControlSpec,
    outer_year: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = sorted(set(valid["underlying_group"].dropna()) - {"SOXX"})
    for group in groups:
        heldout = valid.loc[valid["underlying_group"].eq(group)].dropna(subset=["path_label"])
        fit = train.loc[
            train["underlying_group"].ne(group) & train["instrument"].ne("SOXL")
        ].dropna(subset=["path_label"])
        support = _has_class_support(fit, spec.min_class_samples) and _has_class_support(
            heldout, spec.min_validation_class_samples
        )
        for candidate in MODEL_CANDIDATES:
            base = {
                "outer_year": int(outer_year),
                "heldout_group": group,
                "candidate": candidate,
                "train_rows": int(len(fit)),
                "validation_rows": int(len(heldout)),
                "selection_uses_outer_test": False,
            }
            if not support:
                rows.append(
                    base
                    | {
                        "status": "blocked_insufficient_class_samples",
                        "validation_utility": np.nan,
                        "validation_log_loss": np.nan,
                        "transfer_pass": False,
                    }
                )
                continue
            raw = _candidate_probabilities(
                candidate,
                train=fit,
                predict=heldout,
                feature_cols=list(TRANSFERABLE_FEATURES),
            )
            temperature = select_validation_temperature(heldout["path_label"], raw)
            metrics = _probability_metrics(heldout["path_label"], _temperature_scale(raw, temperature))
            rows.append(base | {"status": "ok", "temperature": temperature} | metrics)
    return pd.DataFrame(rows)


def _outer_probabilities(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    *,
    candidate: str,
    feature_cols: list[str],
    spec: Multi3xHighChopRiskControlSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fit = train.loc[train["instrument"].ne("SOXL")].dropna(subset=["path_label"])
    calibration = valid.loc[valid["instrument"].ne("SOXL")].dropna(subset=["path_label"])
    if not _has_class_support(fit, spec.min_class_samples) or not _has_class_support(
        calibration, spec.min_validation_class_samples
    ):
        raise ValueError("outer fold blocked_insufficient_class_samples")
    raw_valid = _candidate_probabilities(candidate, train=fit, predict=calibration, feature_cols=feature_cols)
    temperature = select_validation_temperature(calibration["path_label"], raw_valid)
    validation_metrics = _probability_metrics(
        calibration["path_label"], _temperature_scale(raw_valid, temperature)
    )
    refit = pd.concat([fit, calibration], ignore_index=True)
    raw_test = _candidate_probabilities(candidate, train=refit, predict=test, feature_cols=feature_cols)
    probabilities = _temperature_scale(raw_test, temperature)
    output = test[
        [
            "datetime",
            "instrument",
            "open",
            "underlying_open",
            "_lagged_sigma_20",
            "ret_20_pctile252",
            "range_pos_252",
            "dist_high_252",
            "cross_asset_positive_breadth",
            "cross_asset_downside_breadth",
            "high_chop_intensity",
        ]
    ].copy()
    for name, label in PATH_LABELS.items():
        output[f"p_{name}"] = probabilities[:, label]
    return output, {"candidate": candidate, "temperature": temperature} | validation_metrics


def _controller_inputs(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.sort_values("datetime").reset_index(drop=True).copy()
    out["soxl_return"] = out["open"].pct_change(fill_method=None).fillna(0.0)
    out["soxx_return"] = out["underlying_open"].pct_change(fill_method=None).fillna(0.0)
    out["realized_vol_20"] = pd.to_numeric(out["_lagged_sigma_20"], errors="coerce").fillna(0.35).clip(0.05, 2.0)
    out["price_break"] = out["range_pos_252"].lt(0.30) | out["dist_high_252"].lt(-0.10)
    out["breadth_break"] = out["cross_asset_positive_breadth"].lt(0.30)
    return out


def _metrics_from_returns(returns: pd.Series, *, annualization: int = 252) -> dict[str, float]:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = (1.0 + clean).cumprod()
    total = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    ann_return = float((1.0 + total) ** (annualization / max(len(clean), 1)) - 1.0)
    ann_vol = float(clean.std(ddof=0) * np.sqrt(annualization))
    drawdown = equity.div(equity.cummax()).sub(1.0) if len(equity) else pd.Series(dtype=float)
    tail = clean.loc[clean <= clean.quantile(0.05)] if len(clean) else clean
    rolling_five = (1.0 + clean).rolling(5).apply(np.prod, raw=True).sub(1.0)
    return {
        "total_return": total,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "cvar_95": float(tail.mean()) if len(tail) else 0.0,
        "worst_five_session_return": float(rolling_five.min()) if rolling_five.notna().any() else 0.0,
        "underwater_day_ratio": float(drawdown.lt(0.0).mean()) if len(drawdown) else 0.0,
        "sharpe": float(ann_return / ann_vol) if ann_vol > 0.0 else 0.0,
    }


def _extend_risk_metrics(metrics: dict[str, float], report: pd.DataFrame) -> dict[str, float]:
    extended = dict(metrics) | _metrics_from_returns(report["strategy_return"])
    positive = report["soxl_return"].gt(0.0)
    benchmark_up = float(report.loc[positive, "soxl_return"].mean()) if positive.any() else 0.0
    strategy_up = float(report.loc[positive, "strategy_return"].mean()) if positive.any() else 0.0
    extended["upside_capture"] = strategy_up / benchmark_up if benchmark_up > 0.0 else 0.0
    return extended


def _static_controlled(inputs: pd.DataFrame, soxl_weight: pd.Series | float, soxx_weight: pd.Series | float) -> pd.DataFrame:
    out = inputs.copy()
    out["soxl_weight"] = soxl_weight
    out["soxx_weight"] = soxx_weight
    out["cash_weight"] = 1.0 - out["soxl_weight"] - out["soxx_weight"]
    return out


def _benchmark_comparison(
    predictions: pd.DataFrame,
    controlled: pd.DataFrame,
    selected_metrics: dict[str, float],
    rule_metrics: dict[str, float],
    *,
    cost_per_turnover: float,
) -> pd.DataFrame:
    inputs = _controller_inputs(predictions)
    avg_soxl = float(controlled["soxl_weight"].mean())
    avg_soxx = float(controlled["soxx_weight"].mean())
    base_weight = (
        (0.35 / inputs["realized_vol_20"].clip(lower=0.05)).clip(0.15, 1.0)
        * np.where(inputs["price_break"] | inputs["breadth_break"], 0.40, 1.0)
    ).clip(0.0, 1.0)
    throttle_weight = base_weight * (1.0 - 0.50 * inputs["high_chop_intensity"].clip(0.0, 1.0))
    benchmark_specs: list[tuple[str, pd.DataFrame, str]] = [
        ("soxl_buy_hold", _static_controlled(inputs, 1.0, 0.0), "SOXL next-open buy and hold"),
        ("soxx_buy_hold", _static_controlled(inputs, 0.0, 1.0), "SOXX next-open buy and hold"),
        (
            "constant_soxl_average_exposure",
            _static_controlled(inputs, avg_soxl, 0.0),
            "SOXL held at selected average SOXL weight",
        ),
        (
            "matched_beta_static",
            _static_controlled(inputs, avg_soxl, avg_soxx),
            "static selected average SOXL and SOXX weights",
        ),
        (
            "base_constrained_ref",
            _static_controlled(inputs, base_weight, 0.0),
            "recomputed point-in-time volatility and break constrained reference",
        ),
        (
            "selected_high_chop_base_throttle",
            _static_controlled(inputs, throttle_weight, 0.0),
            "base constrained reference with continuous high-chop throttle",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, benchmark_controlled, description in benchmark_specs:
        report, metrics = evaluate_multi_asset_risk_control(
            benchmark_controlled, cost_per_turnover=cost_per_turnover
        )
        full = _extend_risk_metrics(metrics, report)
        rows.append({"benchmark": name, "description": description} | full)
    rows.append({"benchmark": "rule_only_risk_controller", "description": "same constraints with rule probabilities"} | rule_metrics)
    selected_rtd = float(selected_metrics["annualized_return"] / max(abs(selected_metrics["max_drawdown"]), 1e-9))
    for row in rows:
        row["return_to_drawdown"] = float(row["annualized_return"] / max(abs(row["max_drawdown"]), 1e-9))
        row["selected_annualized_return_gap"] = float(selected_metrics["annualized_return"] - row["annualized_return"])
        row["selected_max_drawdown_improvement"] = float(selected_metrics["max_drawdown"] - row["max_drawdown"])
        row["selected_return_to_drawdown"] = selected_rtd
    return pd.DataFrame(rows)


def _annual_metrics(report: pd.DataFrame, controlled: pd.DataFrame) -> pd.DataFrame:
    merged = report.copy()
    merged["year"] = pd.to_datetime(merged["datetime"]).dt.year
    rows = []
    for year, group in merged.groupby("year"):
        metrics = _metrics_from_returns(group["strategy_return"])
        rows.append(
            {"year": int(year)}
            | metrics
            | {
                "average_soxl_weight": float(group["soxl_weight"].mean()),
                "average_soxx_weight": float(group["soxx_weight"].mean()),
                "passes_drawdown_volatility": bool(
                    metrics["max_drawdown"] >= -0.20 and metrics["annualized_volatility"] <= 0.35
                ),
            }
        )
    return pd.DataFrame(rows)


def _regime_metrics(report: pd.DataFrame, controlled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state, group in report.groupby("controller_state"):
        metrics = _metrics_from_returns(group["strategy_return"])
        rows.append(
            {"controller_state": state, "session_count": int(len(group))}
            | metrics
            | {
                "average_soxl_weight": float(group["soxl_weight"].mean()),
                "average_soxx_weight": float(group["soxx_weight"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _window_metrics(report: pd.DataFrame, controlled: pd.DataFrame) -> pd.DataFrame:
    windows = {
        "2023": ("2023-01-01", "2023-12-31"),
        "2024": ("2024-01-01", "2024-12-31"),
        "2025": ("2025-01-01", "2025-12-31"),
        "2026_ytd": ("2026-01-01", "2026-12-31"),
    }
    rows = []
    dates = pd.to_datetime(report["datetime"])
    for name, (start, end) in windows.items():
        group = report.loc[dates.between(pd.Timestamp(start), pd.Timestamp(end))]
        if group.empty:
            continue
        rows.append({"window": name, "start": start, "end": end, "session_count": int(len(group))} | _metrics_from_returns(group["strategy_return"]))
    return pd.DataFrame(rows)


def _cost_stress(controlled: pd.DataFrame, spec: Multi3xHighChopRiskControlSpec) -> pd.DataFrame:
    rows = []
    for bps, cost in [(3, spec.cost_per_turnover), (10, spec.stress_cost_per_turnover), (25, spec.high_stress_cost_per_turnover)]:
        report, metrics = evaluate_multi_asset_risk_control(controlled, cost_per_turnover=cost)
        full = _extend_risk_metrics(metrics, report)
        rows.append(
            {"cost_bps": bps}
            | full
            | {"passes_core_risk": bool(full["max_drawdown"] >= -0.20 and full["annualized_volatility"] <= 0.35)}
        )
    return pd.DataFrame(rows)


def _record_and_write_card(
    spec: Multi3xHighChopRiskControlSpec,
    **artifacts: Any,
) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    model_selection: pd.DataFrame = artifacts["model_selection"]
    selected_models: pd.DataFrame = artifacts["selected_models"]
    calibration_metrics: pd.DataFrame = artifacts["calibration_metrics"]
    benchmark_comparison: pd.DataFrame = artifacts["benchmark_comparison"]
    annual_metrics: pd.DataFrame = artifacts["annual_metrics"]
    regime_metrics: pd.DataFrame = artifacts["regime_metrics"]
    window_metrics: pd.DataFrame = artifacts["window_metrics"]
    stress_metrics: pd.DataFrame = artifacts["stress_metrics"]
    strategy_metrics: dict[str, float] = artifacts["strategy_metrics"]
    promotion: dict[str, Any] = artifacts["promotion"]

    model_selection.to_csv(card / "model_selection.csv", index=False)
    calibration_metrics.to_csv(card / "calibration_metrics.csv", index=False)
    benchmark_comparison.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    regime_metrics.to_csv(card / "regime_metrics.csv", index=False)
    window_metrics.to_csv(card / "window_metrics.csv", index=False)
    selected_payload = {
        "selection_uses_outer_test": False,
        "selection_metric": "validation_utility_desc_then_log_loss_asc",
        "outer_models": selected_models.to_dict(orient="records"),
    }
    (card / "selected_model.json").write_text(json.dumps(selected_payload, indent=2, default=str), encoding="utf-8")
    risk_rows = [
        {"metric": key, "value": value, "status": "observed"}
        for key, value in strategy_metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    risk_rows.extend(
        {
            "metric": f"gate_{name}",
            "value": item.get("value", item.get("annual_pass_count", np.nan)),
            "status": "pass" if item["passed"] else "fail",
        }
        for name, item in promotion["gates"].items()
    )
    pd.DataFrame(risk_rows).to_csv(card / "risk_analysis.csv", index=False)
    workflow_yaml = (
        f"experiment: {spec.experiment}\n"
        f"provider_uri: {spec.provider_uri.resolve().as_posix()}\n"
        "data_api: qlib.data.D.features\n"
        "dataset_api:\n  - DatasetH\n  - DataHandlerLP\n  - QlibDataLoader\n"
        "selection_uses_outer_test: false\n"
        "execution_timing: close_t_signal_next_open_t_plus_1\n"
        f"cost_per_turnover: {spec.cost_per_turnover}\n"
    )
    (card / "workflow.yaml").write_text(workflow_yaml, encoding="utf-8")

    failed_gates = [name for name, value in promotion["gates"].items() if not value["passed"]]
    failure_text = (
        f"# Failure Modes\n\n- promotion_decision: {promotion['promotion_decision']}\n"
        f"- failed_gates: {', '.join(failed_gates) if failed_gates else 'none'}\n"
        "- model_risk: 跨ETF可迁移不等于SOXL结构稳定，年度和underlying holdout结果必须同时审阅。\n"
        "- execution_risk: 仅验证次日开盘持有与固定单边换手成本，未包含滑点冲击。\n"
        "- data_risk: ETF起始日期不同，晚成立产品的早期fold会按类别样本门槛阻断。\n"
    )
    (card / "failure_modes.md").write_text(failure_text, encoding="utf-8")
    metrics = strategy_metrics
    conclusion = (
        "# 结论\n\n"
        f"- 执行内容：16只长期股票类3x ETF的归一化路径状态模型，按2023至2026年度样本外拼接，并用SOXL/SOXX/现金控制器执行。\n"
        f"- 关键指标：年化收益 {metrics['annualized_return']:.2%}，年化波动 {metrics['annualized_volatility']:.2%}，最大回撤 {metrics['max_drawdown']:.2%}，5% CVaR {metrics['cvar_95']:.2%}。\n"
        f"- 区间差异：高位震荡与下行确认采用独立状态；跨资产验证通过率为 {artifacts['transfer_pass_rate']:.1%}。\n"
        f"- 晋升决定：{promotion['promotion_decision']}；失败门槛为 {', '.join(failed_gates) if failed_gates else '无'}。\n"
        "- 风险：模型概率仍可能在SOXL独有半导体周期中失效，且固定成本未覆盖开盘跳空冲击。\n"
        "- 下一步：仅保留样本外风险改善稳定的状态，继续缩减无迁移增益特征并做paper验证，不进入真实交易。\n"
    )
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    import qlib
    from qlib.workflow import R

    _ = qlib
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix="multi_3x_high_chop_") as tmp:
        temp = Path(tmp)
        recorder_files = {
            "predictions.csv": artifacts["predictions"],
            "positions.csv": artifacts["controlled"],
            "portfolio_report.csv": artifacts["portfolio_report"],
            "all_model_selection.csv": model_selection,
            "annual_metrics.csv": annual_metrics,
            "cost_stress.csv": stress_metrics,
        }
        for name, frame in recorder_files.items():
            frame.to_csv(temp / name, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                qlib_docs=QLIB_DOCS,
                provider_uri=spec.provider_uri.resolve().as_posix(),
                research_etfs=",".join(RESEARCH_ETFS),
                model_candidates=",".join(MODEL_CANDIDATES),
                outer_test_years=",".join(map(str, spec.outer_test_years)),
                selection_uses_outer_test="false",
                execution_timing="close_t_signal_next_open_t_plus_1",
            )
            for path in temp.iterdir():
                recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            summary = {
                "experiment": spec.experiment,
                "card_type": CARD_TYPE,
                "status": "completed",
                "as_of": spec.as_of,
                "research_universe": list(RESEARCH_ETFS),
                "underlying_groups": UNDERLYING_GROUPS,
                "qlib_contract": {
                    "docs": QLIB_DOCS,
                    "data_api": ["qlib.init", "qlib.data.D.features"],
                    "dataset_api": ["DatasetH", "DataHandlerLP", "QlibDataLoader"],
                    "workflow_api": ["qlib.workflow.R", "Recorder"],
                    "dataset_evidence": artifacts["dataset_evidence"],
                    "provider_readback": artifacts["readback_audit"],
                },
                "data_lineage": {
                    "provider_manifest": artifacts["provider_manifest"],
                    "source_audit": artifacts["source_audit"],
                },
                "path_labels": PATH_LABELS,
                "transferable_features": list(TRANSFERABLE_FEATURES),
                "model_selection": selected_payload,
                "controller": asdict(RiskControlSpec()),
                "execution": {
                    "signal": "close_t",
                    "target_start": "next_open_t_plus_1",
                    "return": "open_to_open",
                    "cost_per_turnover": spec.cost_per_turnover,
                },
                "stitched_metrics": strategy_metrics,
                "transfer_fold_pass_rate": artifacts["transfer_pass_rate"],
                "promotion": promotion,
                "promotion_decision": promotion["promotion_decision"],
                "recorder": {
                    "experiment": spec.experiment,
                    "recorder_id": recorder.id,
                    "recorder_uri": recorder_uri.resolve().as_posix(),
                    "local_dir": Path(recorder.get_local_dir()).resolve().as_posix(),
                },
                "artifact_policy": {
                    "policy": "slim_card_recorder_first",
                    "workflow_config": "workflow.yaml",
                    "recorder_only_artifacts": [
                        "predictions.csv",
                        "positions.csv",
                        "portfolio_report.csv",
                        "all_model_selection.csv",
                        "annual_metrics.csv",
                        "cost_stress.csv",
                    ],
                },
                "compact_sidecars": [
                    "selected_model.json",
                    "model_selection.csv",
                    "backtest_benchmark_comparison.csv",
                    "risk_analysis.csv",
                    "regime_metrics.csv",
                    "window_metrics.csv",
                    "calibration_metrics.csv",
                    "failure_modes.md",
                ],
            }
            (card / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
            for name in summary["compact_sidecars"] + ["workflow.yaml", "summary.json", "conclusion.md"]:
                path = card / name
                if path.exists():
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    return summary
