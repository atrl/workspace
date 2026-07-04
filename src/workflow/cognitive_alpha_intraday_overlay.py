from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, init_qlib_provider, load_qlib_feature_frame


EXPERIMENT = "cognitive_alpha_intraday_execution_overlay_v1"
EXTERNAL_CONTEXT_OVERLAY_EXPERIMENT = "cognitive_alpha_intraday_external_context_overlay_v1"
REGIME_DIAGNOSTIC_EXPERIMENT = "cognitive_alpha_intraday_regime_overlay_diagnostic_v1"
DAILY_SOURCE_CARD = "research/cognitive_alpha_goal_codex_exec_v1"
INTRADAY_SOURCE_CARD = "research/agentic_alpha_intraday_15m_regime_gated_v1"
EXTERNAL_CONTEXT_INTRADAY_SOURCE_CARD = "research/agentic_alpha_intraday_external_context_label_v1"
DAILY_EXECUTION_TIMING = "close_t_signal_next_open_to_following_open"
OVERLAY_SIGNAL_TIMING = "15m_bar_close_to_next_bar_open"
PROB_COL = "prob_lgb_xgb_consensus"
GATE_COL = "regime_gate_v2_dislocation"
PREDICTION_AVAILABLE_GATE_COL = "prediction_available_gate"
VARIANTS = [
    "baseline_next_open",
    "overlay_a_signal_gate",
    "overlay_b_twap4",
    "overlay_c_risk_exit",
]
QLIB_DOCS = {
    "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
    "highfreq": "https://qlib.readthedocs.io/en/latest/component/highfreq.html",
    "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
}
QLIB_NATIVE_API = [
    "qlib.init",
    "qlib.data.D.features",
    "qlib.workflow.R.start",
    "qlib.workflow.R.log_params",
    "qlib.workflow.R.log_metrics",
    "qlib.workflow.Recorder.log_artifact",
]


@dataclass(frozen=True)
class CognitiveAlphaIntradayOverlaySpec:
    root: Path
    experiment: str = EXPERIMENT
    market: str = "all"
    region: str = "us"
    as_of: str = "2026-07-02"
    selection_cost_bps: float = 20.0
    cost_stress_bps: tuple[float, ...] = (10.0, 20.0, 50.0)
    signal_threshold: float | None = None
    risk_threshold: float | None = None
    source_daily_card: str = DAILY_SOURCE_CARD
    source_intraday_card: str = INTRADAY_SOURCE_CARD
    intraday_prob_col: str | None = None
    intraday_gate_col: str | None = GATE_COL
    intraday_gate_policy: str = "source_trade_gate"
    optimize_thresholds: bool = False

    @property
    def output_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def recorder_uri(self) -> Path:
        return self.output_dir / "qlib_recorder"

    @property
    def daily_card_dir(self) -> Path:
        return self.root / self.source_daily_card

    @property
    def intraday_card_dir(self) -> Path:
        return self.root / self.source_intraday_card

    @property
    def daily_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def intraday_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_15min"


def run_cognitive_alpha_intraday_execution_overlay_card(
    *,
    root: Path | str | None = None,
    signal_threshold: float | None = None,
    risk_threshold: float | None = None,
) -> dict[str, Any]:
    spec = CognitiveAlphaIntradayOverlaySpec(
        root=_resolve_root(root),
        signal_threshold=signal_threshold,
        risk_threshold=risk_threshold,
    )
    return _run_card(spec)


def run_cognitive_alpha_intraday_external_context_overlay_card(
    *,
    root: Path | str | None = None,
    signal_threshold: float | None = None,
    risk_threshold: float | None = None,
) -> dict[str, Any]:
    spec = CognitiveAlphaIntradayOverlaySpec(
        root=_resolve_root(root),
        experiment=EXTERNAL_CONTEXT_OVERLAY_EXPERIMENT,
        source_intraday_card=EXTERNAL_CONTEXT_INTRADAY_SOURCE_CARD,
        intraday_gate_col=None,
        intraday_gate_policy="all_predicted_15m_rows",
        optimize_thresholds=True,
        signal_threshold=signal_threshold,
        risk_threshold=risk_threshold,
    )
    return _run_card(spec)


def run_cognitive_alpha_intraday_regime_overlay_diagnostic_card(
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    spec = CognitiveAlphaIntradayOverlaySpec(
        root=_resolve_root(root),
        experiment=REGIME_DIAGNOSTIC_EXPERIMENT,
        source_intraday_card=EXTERNAL_CONTEXT_INTRADAY_SOURCE_CARD,
        intraday_gate_col=None,
        intraday_gate_policy="all_predicted_15m_rows",
        optimize_thresholds=True,
    )
    return _run_regime_diagnostic_card(spec)


def simulate_execution_overlays(
    daily: pd.DataFrame,
    *,
    daily_prices: pd.DataFrame,
    intraday: pd.DataFrame,
    signal_threshold: float,
    risk_threshold: float,
    cost_bps: float = 0.0,
    prob_col: str = PROB_COL,
    gate_col: str = GATE_COL,
) -> pd.DataFrame:
    daily_frame = _normalize_daily_positions(daily)
    prices = _normalize_daily_prices(daily_prices)
    bars = _normalize_intraday_bars(intraday, prob_col=prob_col, gate_col=gate_col)

    rows: list[dict[str, Any]] = []
    for daily_row in daily_frame.to_dict("records"):
        rows.extend(
            _simulate_daily_row(
                daily_row,
                prices=prices,
                intraday=bars,
                signal_threshold=signal_threshold,
                risk_threshold=risk_threshold,
                cost_bps=float(cost_bps),
                prob_col=prob_col,
                gate_col=gate_col,
            )
        )
    return pd.DataFrame(rows)


def summarize_overlay_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame(rows)
    grouped = trades.groupby(["cost_bps", "segment", "variant"], sort=True)
    for (cost_bps, segment, variant), group in grouped:
        group = group.sort_values("datetime").copy()
        returns = pd.to_numeric(group["strategy_return"], errors="coerce").fillna(0.0)
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0 if not equity.empty else pd.Series(dtype=float)
        std = float(returns.std(ddof=0))
        sharpe = 0.0 if std == 0.0 else float(returns.mean() / std * np.sqrt(252.0))
        active = group.loc[pd.to_numeric(group["target_position"], errors="coerce").abs() > 1e-9]
        rows.append(
            {
                "cost_bps": float(cost_bps),
                "segment": str(segment),
                "variant": str(variant),
                "periods": int(len(group)),
                "active_periods": int(len(active)),
                "trade_days": int(active["entry_date"].nunique()) if not active.empty else 0,
                "skipped_periods": int(group["skipped_reason"].notna().sum()),
                "avg_abs_position": float(pd.to_numeric(group["target_position"], errors="coerce").abs().mean()),
                "total_return_pct": float((equity.iloc[-1] - 1.0) * 100.0) if not equity.empty else 0.0,
                "max_drawdown_pct": float(drawdown.min() * 100.0) if not drawdown.empty else 0.0,
                "daily_sharpe": sharpe,
                "win_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
                "worst_period_return_pct": float(returns.min() * 100.0) if len(returns) else 0.0,
                "avg_period_return_bps": float(returns.mean() * 10000.0) if len(returns) else 0.0,
            }
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    baseline = metrics.loc[metrics["variant"] == "baseline_next_open", ["cost_bps", "segment", "total_return_pct", "max_drawdown_pct"]].rename(
        columns={
            "total_return_pct": "baseline_return_pct",
            "max_drawdown_pct": "baseline_max_drawdown_pct",
        }
    )
    metrics = metrics.merge(baseline, on=["cost_bps", "segment"], how="left")
    metrics["return_retention_pct"] = np.where(
        metrics["baseline_return_pct"].abs() > 1e-12,
        metrics["total_return_pct"] / metrics["baseline_return_pct"] * 100.0,
        np.nan,
    )
    metrics["drawdown_improvement_pct"] = metrics["max_drawdown_pct"] - metrics["baseline_max_drawdown_pct"]
    metrics["selection_score"] = (
        metrics["return_retention_pct"].fillna(0.0)
        + 2.0 * metrics["drawdown_improvement_pct"].fillna(0.0)
        + 0.25 * metrics["daily_sharpe"].fillna(0.0)
        - 0.10 * metrics["skipped_periods"].fillna(0.0)
    )
    return metrics.sort_values(["cost_bps", "segment", "selection_score"], ascending=[True, True, False]).reset_index(drop=True)


def evaluate_conditional_overlay_candidates(
    trades: pd.DataFrame,
    regime_features: pd.DataFrame,
    *,
    cost_bps: float,
    variants: tuple[str, ...] = (
        "overlay_a_signal_gate",
        "overlay_b_twap4",
        "overlay_c_risk_exit",
    ),
    condition_columns: tuple[str, ...] = (
        "probability",
        "target_position",
        "turnover",
        "raw_target",
    ),
    quantiles: tuple[float, ...] = tuple(float(value) for value in np.arange(0.05, 0.975, 0.025)),
    validation_segment: str = "validation",
    test_segment: str = "test",
    return_retention_low: float = 85.0,
    return_retention_high: float = 95.0,
    max_drawdown_floor: float = -18.0,
    max_skip_rate: float = 0.50,
    min_validation_coverage: float = 0.03,
) -> pd.DataFrame:
    frame = trades.copy()
    if "cost_bps" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["cost_bps"], errors="coerce").eq(float(cost_bps))].copy()
    if frame.empty:
        return pd.DataFrame()

    features = regime_features.copy()
    frame["datetime_key"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    features["datetime_key"] = pd.to_datetime(features["datetime"]).dt.normalize()

    baseline = _variant_return_frame(frame, variant="baseline_next_open")
    if baseline.empty:
        return pd.DataFrame()

    feature_columns = ["datetime_key", *[column for column in condition_columns if column in features.columns]]
    if "target_position" in features.columns and "target_position" not in feature_columns:
        feature_columns.append("target_position")
    aligned = baseline.merge(features[feature_columns].drop_duplicates("datetime_key"), on="datetime_key", how="left")
    validation_mask = aligned["segment"].eq(validation_segment).to_numpy()
    test_mask = aligned["segment"].eq(test_segment).to_numpy()
    active_mask = (
        pd.to_numeric(aligned.get("target_position", pd.Series(1.0, index=aligned.index)), errors="coerce")
        .fillna(0.0)
        .abs()
        .gt(1e-12)
        .to_numpy()
    )
    baseline_returns = pd.to_numeric(aligned["strategy_return"], errors="coerce").fillna(0.0).to_numpy()
    baseline_skipped = aligned["skipped_reason"].notna().to_numpy()
    baseline_validation = _segment_return_metrics(baseline_returns, validation_mask, baseline_skipped)
    baseline_test = _segment_return_metrics(baseline_returns, test_mask, baseline_skipped)

    rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_frame = _variant_return_frame(frame, variant=variant)
        if variant_frame.empty:
            continue
        variant_aligned = aligned[["datetime_key"]].merge(
            variant_frame[["datetime_key", "strategy_return", "skipped_reason"]],
            on="datetime_key",
            how="left",
        )
        variant_returns = (
            pd.to_numeric(variant_aligned["strategy_return"], errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )
        variant_skipped = variant_aligned["skipped_reason"].notna().to_numpy()
        for column in condition_columns:
            if column not in aligned.columns:
                continue
            values = pd.to_numeric(aligned[column], errors="coerce")
            validation_values = values.loc[validation_mask].dropna()
            if validation_values.nunique() < 2:
                continue
            for quantile in quantiles:
                threshold = float(validation_values.quantile(float(quantile)))
                for operator in ("ge", "le"):
                    condition_mask = (
                        values.ge(threshold) if operator == "ge" else values.le(threshold)
                    ).fillna(False).to_numpy()
                    coverage_base = validation_mask & active_mask
                    coverage = (
                        float(condition_mask[coverage_base].mean())
                        if coverage_base.any()
                        else 0.0
                    )
                    if coverage < float(min_validation_coverage):
                        continue
                    returns = np.where(condition_mask, variant_returns, baseline_returns)
                    skipped = np.where(condition_mask, variant_skipped, baseline_skipped)
                    validation = _segment_return_metrics(returns, validation_mask, skipped)
                    test = _segment_return_metrics(returns, test_mask, skipped)
                    validation_retention = _retention(
                        validation["total_return_pct"],
                        baseline_validation["total_return_pct"],
                    )
                    test_retention = _retention(test["total_return_pct"], baseline_test["total_return_pct"])
                    validation_dd_improvement = (
                        validation["max_drawdown_pct"] - baseline_validation["max_drawdown_pct"]
                    )
                    test_dd_improvement = test["max_drawdown_pct"] - baseline_test["max_drawdown_pct"]
                    passes_validation = bool(
                        return_retention_low <= validation_retention <= return_retention_high
                        and validation["max_drawdown_pct"] > max_drawdown_floor
                        and validation["skipped_periods"] <= validation["periods"] * max_skip_rate
                    )
                    passes_test = bool(
                        return_retention_low <= test_retention <= return_retention_high
                        and test["max_drawdown_pct"] > max_drawdown_floor
                        and test["skipped_periods"] <= test["periods"] * max_skip_rate
                    )
                    rows.append(
                        {
                            "cost_bps": float(cost_bps),
                            "variant": variant,
                            "condition_column": column,
                            "condition_operator": operator,
                            "condition_quantile": float(quantile),
                            "condition_threshold": threshold,
                            "condition_threshold_source": "validation_quantile",
                            "condition_name": f"{column}_{operator}_valid_q{quantile:.3f}_{threshold:.6g}",
                            "condition_valid_active_coverage": coverage,
                            "selection_segment": validation_segment,
                            "report_segment": test_segment,
                            "selection_uses_test_segment": False,
                            "validation_periods": validation["periods"],
                            "validation_return_pct": validation["total_return_pct"],
                            "validation_return_retention_pct": validation_retention,
                            "validation_max_drawdown_pct": validation["max_drawdown_pct"],
                            "validation_drawdown_improvement_pct": validation_dd_improvement,
                            "validation_daily_sharpe": validation["daily_sharpe"],
                            "validation_skipped_periods": validation["skipped_periods"],
                            "test_periods": test["periods"],
                            "test_return_pct": test["total_return_pct"],
                            "test_return_retention_pct": test_retention,
                            "test_max_drawdown_pct": test["max_drawdown_pct"],
                            "test_drawdown_improvement_pct": test_dd_improvement,
                            "test_daily_sharpe": test["daily_sharpe"],
                            "test_skipped_periods": test["skipped_periods"],
                            "passes_validation_target": passes_validation,
                            "passes_test_target": passes_test,
                            "passes_validation_and_test_target": bool(passes_validation and passes_test),
                            "selection_score": (
                                validation_retention
                                + 2.0 * validation_dd_improvement
                                + 0.25 * validation["daily_sharpe"]
                                - 0.10 * validation["skipped_periods"]
                            ),
                        }
                    )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return (
        result.drop_duplicates()
        .sort_values(
            [
                "passes_validation_target",
                "selection_score",
                "validation_drawdown_improvement_pct",
                "validation_return_pct",
            ],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )


def build_regime_feature_frame(
    daily: pd.DataFrame,
    *,
    daily_prices: pd.DataFrame,
) -> pd.DataFrame:
    daily_frame = _normalize_daily_positions(daily)
    prices = _normalize_daily_prices(daily_prices)
    price_index = pd.Index(prices["datetime"])
    rows: list[dict[str, Any]] = []
    for daily_row in daily_frame.to_dict("records"):
        decision_date = pd.Timestamp(daily_row["datetime"]).normalize()
        location = price_index.get_loc(decision_date) if decision_date in set(price_index) else None
        if isinstance(location, slice):
            location = int(location.start)
        entry_date = (
            prices.iloc[int(location) + 1]["datetime"]
            if location is not None and int(location) + 1 < len(prices)
            else pd.NaT
        )
        row = {
            "datetime": decision_date.date().isoformat(),
            "segment": str(daily_row.get("segment", "unknown")),
            "entry_date": _date_or_none(entry_date),
            "target_position": float(daily_row.get("target_position", 0.0) or 0.0),
        }
        for column in ["probability", "raw_target", "turnover"]:
            if column in daily_row:
                row[column] = float(daily_row.get(column, np.nan) or 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _variant_return_frame(trades: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    frame = trades.loc[trades["variant"].eq(variant)].copy()
    if frame.empty:
        return frame
    frame["datetime_key"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    return frame.sort_values("datetime_key").drop_duplicates("datetime_key").reset_index(drop=True)


def _segment_return_metrics(
    returns: np.ndarray,
    mask: np.ndarray,
    skipped: np.ndarray,
) -> dict[str, float | int]:
    segment_returns = np.nan_to_num(np.asarray(returns, dtype=float)[mask], nan=0.0)
    segment_skipped = np.asarray(skipped, dtype=bool)[mask]
    if len(segment_returns) == 0:
        return {
            "periods": 0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "daily_sharpe": 0.0,
            "skipped_periods": 0,
        }
    equity = np.cumprod(1.0 + segment_returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    std = float(np.std(segment_returns))
    sharpe = 0.0 if std == 0.0 else float(np.mean(segment_returns) / std * np.sqrt(252.0))
    return {
        "periods": int(len(segment_returns)),
        "total_return_pct": float((equity[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "daily_sharpe": sharpe,
        "skipped_periods": int(segment_skipped.sum()),
    }


def _retention(total_return_pct: float, baseline_return_pct: float) -> float:
    if abs(float(baseline_return_pct)) <= 1e-12:
        return float("nan")
    return float(total_return_pct) / float(baseline_return_pct) * 100.0


def _run_card(spec: CognitiveAlphaIntradayOverlaySpec) -> dict[str, Any]:
    from qlib.workflow import R

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.recorder_uri.mkdir(parents=True, exist_ok=True)

    daily_summary = _read_json(spec.daily_card_dir / "summary.json")
    intraday_summary = _read_json(spec.intraday_card_dir / "summary.json")
    workflow_yaml = _write_workflow_yaml(spec, daily_summary=daily_summary, intraday_summary=intraday_summary)
    signal_config = _intraday_signal_config(spec, intraday_summary=intraday_summary)

    daily_positions = _load_daily_selected_positions(daily_summary)
    as_of = str(daily_summary.get("as_of") or spec.as_of)
    daily_prices = _load_daily_open_prices(spec, start=str(daily_positions["datetime"].min().date()), end=as_of)
    intraday_predictions = _load_intraday_prediction_frame(
        spec,
        intraday_summary=intraday_summary,
        as_of=as_of,
        signal_config=signal_config,
    )
    thresholds = _thresholds_from_validation(
        intraday_predictions,
        intraday_summary=intraday_summary,
        signal_config=signal_config,
        signal_threshold=spec.signal_threshold,
        risk_threshold=spec.risk_threshold,
    )
    threshold_selection = pd.DataFrame()
    if spec.optimize_thresholds and spec.signal_threshold is None and spec.risk_threshold is None:
        thresholds, threshold_selection = _select_thresholds_from_validation(
            daily_positions,
            daily_prices=daily_prices,
            intraday_predictions=intraday_predictions,
            signal_config=signal_config,
            base_thresholds=thresholds,
            cost_bps=spec.selection_cost_bps,
        )

    trade_frames = []
    for cost_bps in sorted(set(float(value) for value in spec.cost_stress_bps)):
        trade_frames.append(
            simulate_execution_overlays(
                daily_positions,
                daily_prices=daily_prices,
                intraday=intraday_predictions,
                signal_threshold=thresholds["signal_threshold"],
                risk_threshold=thresholds["risk_threshold"],
                cost_bps=cost_bps,
                prob_col=signal_config["prob_col"],
                gate_col=signal_config["gate_col"],
            ).assign(cost_bps=float(cost_bps))
        )
    overlay_trades = pd.concat(trade_frames, ignore_index=True)
    cost_sensitivity = summarize_overlay_metrics(overlay_trades)
    model_comparison = (
        cost_sensitivity.loc[cost_sensitivity["cost_bps"].eq(float(spec.selection_cost_bps))]
        .sort_values(["segment", "selection_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    selected_model = _select_variant(model_comparison, cost_bps=spec.selection_cost_bps)
    selected_model["signal_threshold"] = float(thresholds["signal_threshold"])
    selected_model["risk_threshold"] = float(thresholds["risk_threshold"])
    walk_forward = _walk_forward_metrics(overlay_trades, cost_bps=spec.selection_cost_bps)
    analysis_summary = _analysis_summary(
        spec,
        daily_summary=daily_summary,
        intraday_summary=intraday_summary,
        daily_positions=daily_positions,
        daily_prices=daily_prices,
        intraday_predictions=intraday_predictions,
        thresholds=thresholds,
        signal_config=signal_config,
        model_comparison=model_comparison,
        threshold_selection=threshold_selection,
        selected_model=selected_model,
        walk_forward=walk_forward,
    )
    sidecars = _write_sidecars(
        spec,
        model_comparison=model_comparison,
        threshold_selection=threshold_selection,
        cost_sensitivity=cost_sensitivity,
        walk_forward=walk_forward,
        selected_model=selected_model,
        analysis_summary=analysis_summary,
    )

    upload_dir = Path(tempfile.mkdtemp(prefix=f"{spec.experiment}_recorder_"))
    try:
        execution_dir = upload_dir / "execution_overlay"
        execution_dir.mkdir(parents=True, exist_ok=True)
        overlay_trades.to_csv(execution_dir / "overlay_trades.csv", index=False)
        daily_positions.to_csv(execution_dir / "daily_selected_positions.csv", index=False)
        intraday_predictions.to_csv(execution_dir / "intraday_predictions_15m.csv", index=False)

        R.set_uri(spec.recorder_uri.resolve().as_posix())
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                experiment=spec.experiment,
                daily_source=spec.source_daily_card,
                intraday_source=spec.source_intraday_card,
                selection_segment="validation",
                report_segment="test",
                selection_cost_bps=float(spec.selection_cost_bps),
                signal_threshold=float(thresholds["signal_threshold"]),
                risk_threshold=float(thresholds["risk_threshold"]),
                selection_uses_test_segment=False,
            )
            R.log_metrics(
                selected_validation_return_pct=float(selected_model.get("validation_total_return_pct", 0.0)),
                selected_validation_max_drawdown_pct=float(selected_model.get("validation_max_drawdown_pct", 0.0)),
                selected_test_return_pct=float(selected_model.get("test_total_return_pct", 0.0)),
                selected_test_max_drawdown_pct=float(selected_model.get("test_max_drawdown_pct", 0.0)),
                selected_test_retention_pct=float(selected_model.get("test_return_retention_pct", 0.0)),
            )
            for file_name in [
                "overlay_trades.csv",
                "daily_selected_positions.csv",
                "intraday_predictions_15m.csv",
            ]:
                recorder.log_artifact((execution_dir / file_name).as_posix(), artifact_path="execution_overlay")
            for rel_path in sidecars.values():
                recorder.log_artifact((spec.output_dir / rel_path).as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir())
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)

    _remove_transient_code_snapshots(recorder_local_dir)
    recorder_artifacts = _recorder_files(recorder_local_dir)
    promotion_decision = _promotion_decision(selected_model, intraday_summary=intraday_summary)
    summary = {
        "experiment": spec.experiment,
        "card_type": "execution_overlay_research",
        "workflow_config": workflow_yaml.name,
        "promotion_decision": promotion_decision,
        "as_of": as_of,
        "market": spec.market,
        "region": spec.region,
        "source_daily_model": {
            "card": spec.source_daily_card,
            "recorder_id": daily_summary.get("recorder_id"),
            "execution_timing": daily_summary.get("position_model", {}).get("execution_timing", DAILY_EXECUTION_TIMING),
            "return_col": daily_summary.get("position_model", {}).get("return_col", "open_to_open_return"),
            "model_target_col": daily_summary.get("position_model", {}).get("model_target_col", "label_open_to_open_5d"),
            "selected_return_pct": daily_summary.get("position_model", {}).get("comparison", {}).get("selected_return_pct"),
            "selected_max_drawdown_pct": daily_summary.get("position_model", {}).get("comparison", {}).get("selected_max_drawdown_pct"),
        },
        "source_intraday_model": {
            "card": spec.source_intraday_card,
            "recorder_id": intraday_summary.get("recorder_id"),
            "selected_model": intraday_summary.get("selected_model", {}).get("model"),
            "threshold": thresholds["signal_threshold"],
            "risk_threshold": thresholds["risk_threshold"],
            "signal_prob_col": signal_config["prob_col"],
            "gate_col": signal_config["gate_col"],
            "gate_policy": signal_config["gate_policy"],
            "trade_gate": intraday_summary.get("trade_gate", {}).get("name"),
            "execution_policy": intraday_summary.get("execution_policy"),
            "label_policy_name": intraday_summary.get("label_policy", {}).get("name"),
            "external_context": intraday_summary.get("external_intraday_context", {}),
        },
        "provider_uris": _provider_uris(spec, intraday_summary=intraday_summary),
        "data_audit": analysis_summary["data_audit"],
        "execution_overlay": {
            "frequency": "15min",
            "decision_model_timing": "daily_close_t_signal",
            "daily_execution_timing": DAILY_EXECUTION_TIMING,
            "overlay_signal_timing": OVERLAY_SIGNAL_TIMING,
            "variants": VARIANTS,
            "open_to_open_baseline": True,
            "intraday_overlays_hold_overnight": False,
            "intraday_label_policy": intraday_summary.get("label_policy", {}).get("name")
            or intraday_summary.get("execution_policy", {}).get("label_policy"),
            "uses_external_intraday_context": bool(intraday_summary.get("external_intraday_context", {}).get("provider_uri")),
            "description": "daily target_position retained; 15m model only changes next-session entry/exit execution",
        },
        "walk_forward": {
            "folds": walk_forward["fold"].tolist() if "fold" in walk_forward else [],
            "rows": int(len(walk_forward)),
            "required_policy_satisfied": _required_overlay_policy_satisfied(
                intraday_summary=intraday_summary,
                walk_forward=walk_forward,
            ),
        },
        "model_selection": {
            "selection_segment": "validation",
            "report_segment": "test",
            "selection_metric": selected_model["selection_metric"],
            "selection_uses_test_segment": False,
            "selection_cost_bps": float(spec.selection_cost_bps),
            "selected_variant": selected_model["variant"],
            "candidate_count": len(VARIANTS),
            "threshold_search": {
                "enabled": bool(spec.optimize_thresholds),
                "selection_segment": "validation",
                "selection_uses_test_segment": False,
                "candidate_count": int(len(threshold_selection)),
                "target_candidate_count": int(threshold_selection["passes_target"].sum())
                if "passes_target" in threshold_selection
                else 0,
                "selected_signal_threshold": float(thresholds["signal_threshold"]),
                "selected_risk_threshold": float(thresholds["risk_threshold"]),
            },
        },
        "selected_model": selected_model,
        "cost_stress_bps": [float(value) for value in sorted(set(spec.cost_stress_bps))],
        "recorder_id": recorder_id,
        "recorder_uri": spec.recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": recorder_artifacts,
        "sidecars": sidecars,
        "artifact_policy": _artifact_policy(spec.experiment, sidecars),
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": QLIB_NATIVE_API,
    }
    (spec.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_conclusion(spec, summary=summary, selected_model=selected_model, model_comparison=model_comparison)
    return summary


def _run_regime_diagnostic_card(spec: CognitiveAlphaIntradayOverlaySpec) -> dict[str, Any]:
    from qlib.workflow import R

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.recorder_uri.mkdir(parents=True, exist_ok=True)

    daily_summary = _read_json(spec.daily_card_dir / "summary.json")
    intraday_summary = _read_json(spec.intraday_card_dir / "summary.json")
    workflow_yaml = _write_regime_workflow_yaml(spec, daily_summary=daily_summary, intraday_summary=intraday_summary)
    signal_config = _intraday_signal_config(spec, intraday_summary=intraday_summary)

    daily_positions = _load_daily_selected_positions(daily_summary)
    as_of = str(daily_summary.get("as_of") or spec.as_of)
    daily_prices = _load_daily_open_prices(spec, start=str(daily_positions["datetime"].min().date()), end=as_of)
    intraday_predictions = _load_intraday_prediction_frame(
        spec,
        intraday_summary=intraday_summary,
        as_of=as_of,
        signal_config=signal_config,
    )
    thresholds = _regime_overlay_thresholds(
        spec,
        intraday_predictions=intraday_predictions,
        intraday_summary=intraday_summary,
        signal_config=signal_config,
    )
    overlay_trades = simulate_execution_overlays(
        daily_positions,
        daily_prices=daily_prices,
        intraday=intraday_predictions,
        signal_threshold=thresholds["signal_threshold"],
        risk_threshold=thresholds["risk_threshold"],
        cost_bps=float(spec.selection_cost_bps),
        prob_col=signal_config["prob_col"],
        gate_col=signal_config["gate_col"],
    ).assign(cost_bps=float(spec.selection_cost_bps))
    regime_features = build_regime_feature_frame(daily_positions, daily_prices=daily_prices)
    condition_columns = tuple(
        column
        for column in ("probability", "target_position", "turnover", "raw_target")
        if column in regime_features.columns
    )
    policy_selection = evaluate_conditional_overlay_candidates(
        overlay_trades,
        regime_features,
        cost_bps=float(spec.selection_cost_bps),
        condition_columns=condition_columns,
    )
    model_comparison = summarize_overlay_metrics(overlay_trades)
    selected_model = _select_regime_policy(policy_selection)
    analysis_summary = _regime_analysis_summary(
        spec,
        daily_summary=daily_summary,
        intraday_summary=intraday_summary,
        daily_positions=daily_positions,
        daily_prices=daily_prices,
        intraday_predictions=intraday_predictions,
        thresholds=thresholds,
        signal_config=signal_config,
        model_comparison=model_comparison,
        policy_selection=policy_selection,
        selected_model=selected_model,
        condition_columns=condition_columns,
    )
    sidecars = _write_regime_sidecars(
        spec,
        model_comparison=model_comparison,
        policy_selection=policy_selection,
        selected_model=selected_model,
        analysis_summary=analysis_summary,
    )

    upload_dir = Path(tempfile.mkdtemp(prefix=f"{spec.experiment}_recorder_"))
    try:
        execution_dir = upload_dir / "regime_overlay"
        execution_dir.mkdir(parents=True, exist_ok=True)
        overlay_trades.to_csv(execution_dir / "overlay_trades.csv", index=False)
        regime_features.to_csv(execution_dir / "regime_features.csv", index=False)
        policy_selection.to_csv(execution_dir / "policy_selection.csv", index=False)

        R.set_uri(spec.recorder_uri.resolve().as_posix())
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                experiment=spec.experiment,
                daily_source=spec.source_daily_card,
                intraday_source=spec.source_intraday_card,
                selection_segment="validation",
                report_segment="test",
                selection_uses_test_segment=False,
                selection_cost_bps=float(spec.selection_cost_bps),
                signal_threshold=float(thresholds["signal_threshold"]),
                risk_threshold=float(thresholds["risk_threshold"]),
                condition_columns=",".join(condition_columns),
            )
            R.log_metrics(
                candidate_count=float(len(policy_selection)),
                validation_target_candidate_count=float(
                    policy_selection["passes_validation_target"].sum()
                    if "passes_validation_target" in policy_selection
                    else 0
                ),
                validation_and_test_target_candidate_count=float(
                    policy_selection["passes_validation_and_test_target"].sum()
                    if "passes_validation_and_test_target" in policy_selection
                    else 0
                ),
                selected_validation_return_retention_pct=float(
                    selected_model.get("validation_return_retention_pct") or 0.0
                ),
                selected_validation_max_drawdown_pct=float(
                    selected_model.get("validation_max_drawdown_pct") or 0.0
                ),
                selected_test_return_retention_pct=float(selected_model.get("test_return_retention_pct") or 0.0),
                selected_test_max_drawdown_pct=float(selected_model.get("test_max_drawdown_pct") or 0.0),
            )
            for file_name in [
                "overlay_trades.csv",
                "regime_features.csv",
                "policy_selection.csv",
            ]:
                recorder.log_artifact((execution_dir / file_name).as_posix(), artifact_path="regime_overlay")
            for rel_path in sidecars.values():
                recorder.log_artifact((spec.output_dir / rel_path).as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir())
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)

    _remove_transient_code_snapshots(recorder_local_dir)
    recorder_artifacts = _recorder_files(recorder_local_dir)
    promotion_decision = _regime_promotion_decision(selected_model, intraday_summary=intraday_summary)
    validation_target_count = int(
        policy_selection["passes_validation_target"].sum()
        if "passes_validation_target" in policy_selection
        else 0
    )
    validation_and_test_target_count = int(
        policy_selection["passes_validation_and_test_target"].sum()
        if "passes_validation_and_test_target" in policy_selection
        else 0
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": "intraday_regime_overlay_diagnostic",
        "workflow_config": workflow_yaml.name,
        "promotion_decision": promotion_decision,
        "as_of": as_of,
        "market": spec.market,
        "region": spec.region,
        "source_daily_model": {
            "card": spec.source_daily_card,
            "recorder_id": daily_summary.get("recorder_id"),
            "execution_timing": daily_summary.get("position_model", {}).get("execution_timing", DAILY_EXECUTION_TIMING),
            "return_col": daily_summary.get("position_model", {}).get("return_col", "open_to_open_return"),
        },
        "source_intraday_model": {
            "card": spec.source_intraday_card,
            "recorder_id": intraday_summary.get("recorder_id"),
            "selected_model": intraday_summary.get("selected_model", {}).get("model"),
            "signal_prob_col": signal_config["prob_col"],
            "gate_policy": signal_config["gate_policy"],
            "external_context": intraday_summary.get("external_intraday_context", {}),
        },
        "provider_uris": _provider_uris(spec, intraday_summary=intraday_summary),
        "execution_overlay": {
            "frequency": "15min",
            "decision_model_timing": "daily_close_t_signal",
            "daily_execution_timing": DAILY_EXECUTION_TIMING,
            "overlay_signal_timing": OVERLAY_SIGNAL_TIMING,
            "variants": VARIANTS,
            "open_to_open_baseline": True,
            "intraday_overlays_hold_overnight": False,
            "description": "baseline remains default; a validation-selected daily-regime condition can switch execution to a 15m overlay",
        },
        "regime_policy": {
            "selection_segment": "validation",
            "report_segment": "test",
            "selection_uses_test_segment": False,
            "selection_cost_bps": float(spec.selection_cost_bps),
            "condition_columns": list(condition_columns),
            "condition_threshold_source": "validation_quantile",
            "causal_scope": "daily_close_known_features_only",
            "excludes_first_hour_features_from_twap_selection": True,
            "target_return_retention_pct": [85.0, 95.0],
            "target_max_drawdown_floor_pct": -18.0,
            "candidate_count": int(len(policy_selection)),
            "validation_target_candidate_count": validation_target_count,
            "validation_and_test_target_candidate_count": validation_and_test_target_count,
        },
        "model_selection": {
            "selection_segment": "validation",
            "report_segment": "test",
            "selection_metric": "validation return_retention + 2*drawdown_improvement + 0.25*sharpe - skipped_penalty",
            "selection_uses_test_segment": False,
            "candidate_count": int(len(policy_selection)),
            "validation_target_candidate_count": validation_target_count,
            "validation_and_test_target_candidate_count": validation_and_test_target_count,
            "selected_variant": selected_model.get("variant"),
        },
        "selected_model": selected_model,
        "analysis_summary": analysis_summary,
        "recorder_id": recorder_id,
        "recorder_uri": spec.recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": recorder_artifacts,
        "sidecars": sidecars,
        "artifact_policy": _regime_artifact_policy(spec.experiment, sidecars),
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": QLIB_NATIVE_API,
    }
    (spec.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_regime_conclusion(spec, summary=summary, selected_model=selected_model)
    return summary


def _provider_uris(spec: CognitiveAlphaIntradayOverlaySpec, *, intraday_summary: dict[str, Any]) -> dict[str, str]:
    provider_uris = {
            "daily": spec.daily_provider_uri.resolve().as_posix(),
            "15min": spec.intraday_provider_uri.resolve().as_posix(),
    }
    external_uri = intraday_summary.get("provider_uris", {}).get("external_15min")
    if external_uri:
        provider_uris["external_context_15min"] = str(external_uri)
    return provider_uris


def _simulate_daily_row(
    daily_row: dict[str, Any],
    *,
    prices: pd.DataFrame,
    intraday: pd.DataFrame,
    signal_threshold: float,
    risk_threshold: float,
    cost_bps: float,
    prob_col: str,
    gate_col: str,
) -> list[dict[str, Any]]:
    decision_date = pd.Timestamp(daily_row["datetime"]).normalize()
    price_index = pd.Index(prices["datetime"])
    if decision_date not in set(price_index):
        return [_empty_variant_row(daily_row, variant, cost_bps, "missing_decision_open") for variant in VARIANTS]
    location = int(price_index.get_loc(decision_date))
    if isinstance(location, slice):
        location = int(location.start)
    entry_date = prices.iloc[location + 1]["datetime"] if location + 1 < len(prices) else pd.NaT
    exit_date = prices.iloc[location + 2]["datetime"] if location + 2 < len(prices) else pd.NaT
    target = float(daily_row.get("target_position", 0.0) or 0.0)
    segment = str(daily_row.get("segment", "unknown"))
    source_return = float(daily_row.get("strategy_return", 0.0) or 0.0)
    base_common = {
        "datetime": decision_date.date().isoformat(),
        "segment": segment,
        "target_position": target,
        "cost_bps": float(cost_bps),
        "entry_date": _date_or_none(entry_date),
        "exit_date": _date_or_none(exit_date),
        "source_daily_strategy_return": source_return,
    }
    if abs(target) <= 1e-12:
        return [
            {
                **base_common,
                "variant": variant,
                "entry_time": None,
                "exit_time": None,
                "entry_price": np.nan,
                "exit_price": np.nan,
                "intraday_signal_score": np.nan,
                "gross_return": 0.0,
                "strategy_return": 0.0,
                "skipped_reason": None,
            }
            for variant in VARIANTS
        ]
    if pd.isna(entry_date) or pd.isna(exit_date):
        return [_empty_variant_row(daily_row, variant, cost_bps, "missing_entry_or_exit_open") for variant in VARIANTS]

    entry_open = float(prices.iloc[location + 1]["open"])
    exit_open = float(prices.iloc[location + 2]["open"])
    entry_day_bars = intraday.loc[intraday["date"].eq(pd.Timestamp(entry_date).date().isoformat())].sort_values("datetime")
    execution_cost = abs(target) * cost_bps / 10000.0

    baseline_return = source_return - execution_cost
    rows = [
        {
            **base_common,
            "variant": "baseline_next_open",
            "entry_time": pd.Timestamp(entry_date).date().isoformat(),
            "exit_time": pd.Timestamp(exit_date).date().isoformat(),
            "entry_price": entry_open,
            "exit_price": exit_open,
            "intraday_signal_score": np.nan,
            "gross_return": source_return,
            "strategy_return": baseline_return,
            "skipped_reason": None,
        }
    ]

    signal_candidates = entry_day_bars.loc[
        entry_day_bars.get(gate_col, pd.Series(False, index=entry_day_bars.index)).fillna(False).astype(bool)
        & (pd.to_numeric(entry_day_bars.get(prob_col), errors="coerce") >= signal_threshold)
        & entry_day_bars["next_open"].notna()
    ]
    if signal_candidates.empty:
        rows.append(
            {
                **base_common,
                "variant": "overlay_a_signal_gate",
                "entry_time": None,
                "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                "entry_price": np.nan,
                "exit_price": exit_open,
                "intraday_signal_score": np.nan,
                "gross_return": 0.0,
                "strategy_return": 0.0,
                "skipped_reason": "no_15m_entry_signal",
            }
        )
    else:
        signal = signal_candidates.iloc[0]
        entry_price = float(signal["next_open"])
        gross_return = target * (exit_open / entry_price - 1.0)
        rows.append(
            {
                **base_common,
                "variant": "overlay_a_signal_gate",
                "entry_time": _timestamp_or_none(signal["next_datetime"]),
                "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                "entry_price": entry_price,
                "exit_price": exit_open,
                "intraday_signal_score": float(signal[prob_col]),
                "gross_return": gross_return,
                "strategy_return": gross_return - execution_cost,
                "skipped_reason": None,
            }
        )

    twap_bars = entry_day_bars.loc[entry_day_bars["open"].notna()].head(4)
    if twap_bars.empty:
        rows.append(
            {
                **base_common,
                "variant": "overlay_b_twap4",
                "entry_time": None,
                "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                "entry_price": np.nan,
                "exit_price": exit_open,
                "intraday_signal_score": np.nan,
                "gross_return": 0.0,
                "strategy_return": 0.0,
                "skipped_reason": "missing_intraday_twap_bars",
            }
        )
    else:
        entry_price = float(twap_bars["open"].mean())
        gross_return = target * (exit_open / entry_price - 1.0)
        rows.append(
            {
                **base_common,
                "variant": "overlay_b_twap4",
                "entry_time": "twap4_first_4_bars",
                "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                "entry_price": entry_price,
                "exit_price": exit_open,
                "intraday_signal_score": np.nan,
                "gross_return": gross_return,
                "strategy_return": gross_return - execution_cost,
                "skipped_reason": None,
            }
        )

    risk_candidates = entry_day_bars.loc[
        entry_day_bars.get(gate_col, pd.Series(False, index=entry_day_bars.index)).fillna(False).astype(bool)
        & (pd.to_numeric(entry_day_bars.get(prob_col), errors="coerce") <= risk_threshold)
        & entry_day_bars["close"].notna()
    ]
    if risk_candidates.empty:
        rows.append(
            {
                **base_common,
                "variant": "overlay_c_risk_exit",
                "entry_time": pd.Timestamp(entry_date).date().isoformat(),
                "exit_time": pd.Timestamp(exit_date).date().isoformat(),
                "entry_price": entry_open,
                "exit_price": exit_open,
                "intraday_signal_score": np.nan,
                "gross_return": target * (exit_open / entry_open - 1.0),
                "strategy_return": source_return - execution_cost,
                "skipped_reason": None,
            }
        )
    else:
        risk = risk_candidates.iloc[0]
        exit_price = float(risk["close"])
        gross_return = target * (exit_price / entry_open - 1.0)
        rows.append(
            {
                **base_common,
                "variant": "overlay_c_risk_exit",
                "entry_time": pd.Timestamp(entry_date).date().isoformat(),
                "exit_time": _timestamp_or_none(risk["datetime"]),
                "entry_price": entry_open,
                "exit_price": exit_price,
                "intraday_signal_score": float(risk[prob_col]),
                "gross_return": gross_return,
                "strategy_return": gross_return - execution_cost,
                "skipped_reason": None,
            }
        )
    return rows


def _normalize_daily_positions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"]).dt.normalize()
    if "segment" in result.columns:
        result["segment"] = result["segment"].replace({"valid": "validation"})
    else:
        result["segment"] = "unknown"
    for column in ["target_position", "strategy_return"]:
        if column not in result.columns:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result.sort_values("datetime").reset_index(drop=True)


def _normalize_daily_prices(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"]).dt.normalize()
    result["open"] = pd.to_numeric(result["open"], errors="coerce")
    return result.dropna(subset=["datetime", "open"]).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)


def _normalize_intraday_bars(frame: pd.DataFrame, *, prob_col: str, gate_col: str) -> pd.DataFrame:
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"])
    if "date" not in result.columns:
        result["date"] = result["datetime"].dt.date.astype(str)
    else:
        result["date"] = pd.to_datetime(result["date"]).dt.date.astype(str)
    for column in ["open", "close", prob_col]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if gate_col not in result.columns:
        result[gate_col] = False
    result[gate_col] = result[gate_col].fillna(False).astype(bool)
    result = result.sort_values(["date", "datetime"]).reset_index(drop=True)
    result["next_datetime"] = result.groupby("date")["datetime"].shift(-1)
    result["next_open"] = result.groupby("date")["open"].shift(-1)
    return result


def _load_daily_selected_positions(daily_summary: dict[str, Any]) -> pd.DataFrame:
    recorder_dir = Path(str(daily_summary["recorder_local_dir"]))
    path = recorder_dir / "artifacts" / "cognitive_alpha_position" / "positions.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["selected"].astype(bool)].copy()
    return _normalize_daily_positions(frame)


def _load_daily_open_prices(spec: CognitiveAlphaIntradayOverlaySpec, *, start: str, end: str) -> pd.DataFrame:
    config = QlibProviderConfig(
        provider_uri=spec.daily_provider_uri,
        region=spec.region,
        market=spec.market,
        freq="day",
        kernels=1,
    )
    init_qlib_provider(config)
    raw = load_qlib_feature_frame(config, instruments=["SOXL"], fields=["$open"], start=start, end=end)
    frame = raw.reset_index()
    frame = frame.loc[frame["instrument"].astype(str).str.upper().eq("SOXL"), ["datetime", "$open"]].rename(
        columns={"$open": "open"}
    )
    return _normalize_daily_prices(frame)


def _intraday_signal_config(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    intraday_summary: dict[str, Any],
) -> dict[str, str]:
    selected_model = str(intraday_summary.get("selected_model", {}).get("model") or "lgb_xgb_consensus")
    prob_col = str(spec.intraday_prob_col or f"prob_{selected_model}")
    if spec.intraday_gate_col is None:
        return {
            "prob_col": prob_col,
            "gate_col": PREDICTION_AVAILABLE_GATE_COL,
            "gate_policy": spec.intraday_gate_policy,
        }
    return {
        "prob_col": prob_col,
        "gate_col": spec.intraday_gate_col,
        "gate_policy": spec.intraday_gate_policy,
    }


def _load_intraday_prediction_frame(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    intraday_summary: dict[str, Any],
    as_of: str,
    signal_config: dict[str, str],
) -> pd.DataFrame:
    config = QlibProviderConfig(
        provider_uri=spec.intraday_provider_uri,
        region=spec.region,
        market=spec.market,
        freq="15min",
        kernels=1,
    )
    init_qlib_provider(config)
    raw = load_qlib_feature_frame(
        config,
        instruments=["SOXL"],
        fields=["$open", "$close"],
        start="2025-01-02 09:45:00",
        end=f"{as_of} 16:00:00",
    )
    bars = raw.reset_index()
    bars = bars.loc[bars["instrument"].astype(str).str.upper().eq("SOXL"), ["datetime", "$open", "$close"]].rename(
        columns={"$open": "open", "$close": "close"}
    )
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    bars["date"] = bars["datetime"].dt.date.astype(str)

    predictions_path = (
        Path(str(intraday_summary["recorder_local_dir"]))
        / "artifacts"
        / "model_selection"
        / "all_model_predictions.csv"
    )
    predictions = pd.read_csv(predictions_path)
    predictions["datetime"] = pd.to_datetime(predictions["datetime"])
    prob_col = signal_config["prob_col"]
    gate_col = signal_config["gate_col"]
    columns = ["datetime", "segment", prob_col]
    if "freq" in predictions.columns:
        columns.append("freq")
    if spec.intraday_gate_col is not None and spec.intraday_gate_col in predictions.columns:
        columns.append(spec.intraday_gate_col)
    predictions = predictions.loc[predictions["datetime"].between(bars["datetime"].min(), bars["datetime"].max()), columns].copy()
    merged = bars.merge(predictions, on="datetime", how="left")
    if spec.intraday_gate_col is None:
        merged[gate_col] = merged[prob_col].notna()
    elif spec.intraday_gate_col not in merged.columns:
        merged[gate_col] = False
    return _normalize_intraday_bars(merged, prob_col=prob_col, gate_col=gate_col)


def _thresholds_from_validation(
    intraday: pd.DataFrame,
    *,
    intraday_summary: dict[str, Any],
    signal_config: dict[str, str],
    signal_threshold: float | None,
    risk_threshold: float | None,
) -> dict[str, float]:
    selected_threshold = float(intraday_summary.get("selected_model", {}).get("threshold", 0.5))
    prob_col = signal_config["prob_col"]
    gate_col = signal_config["gate_col"]
    validation = intraday.loc[
        intraday.get("segment", pd.Series(index=intraday.index, dtype=object)).eq("valid")
        & intraday[gate_col].fillna(False).astype(bool)
        & intraday[prob_col].notna()
    ]
    if risk_threshold is None:
        if validation.empty:
            risk_value = max(0.0, selected_threshold - 0.05)
        else:
            risk_value = float(validation[prob_col].quantile(0.20))
    else:
        risk_value = float(risk_threshold)
    return {
        "signal_threshold": float(selected_threshold if signal_threshold is None else signal_threshold),
        "risk_threshold": risk_value,
    }


def _select_thresholds_from_validation(
    daily_positions: pd.DataFrame,
    *,
    daily_prices: pd.DataFrame,
    intraday_predictions: pd.DataFrame,
    signal_config: dict[str, str],
    base_thresholds: dict[str, float],
    cost_bps: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    prob_col = signal_config["prob_col"]
    gate_col = signal_config["gate_col"]
    validation_probs = intraday_predictions.loc[
        intraday_predictions.get("segment", pd.Series(index=intraday_predictions.index, dtype=object)).eq("valid")
        & intraday_predictions[gate_col].fillna(False).astype(bool)
        & intraday_predictions[prob_col].notna(),
        prob_col,
    ]
    if validation_probs.empty:
        return base_thresholds, pd.DataFrame()

    signal_grid = sorted(
        set(
            [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
            + [float(validation_probs.quantile(value)) for value in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]]
        )
    )
    risk_grid = sorted(
        set(
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
            + [float(validation_probs.quantile(value)) for value in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]]
        )
    )

    rows: list[dict[str, Any]] = []
    for signal_threshold in signal_grid:
        for risk_threshold in risk_grid:
            trades = simulate_execution_overlays(
                daily_positions,
                daily_prices=daily_prices,
                intraday=intraday_predictions,
                signal_threshold=float(signal_threshold),
                risk_threshold=float(risk_threshold),
                cost_bps=float(cost_bps),
                prob_col=prob_col,
                gate_col=gate_col,
            ).assign(cost_bps=float(cost_bps))
            metrics = summarize_overlay_metrics(trades)
            validation_metrics = metrics.loc[metrics["segment"].eq("validation")].copy()
            for row in validation_metrics.to_dict("records"):
                row.update(
                    {
                        "signal_threshold": float(signal_threshold),
                        "risk_threshold": float(risk_threshold),
                        "selection_uses_test_segment": False,
                    }
                )
                rows.append(row)

    selection = pd.DataFrame(rows)
    if selection.empty:
        return base_thresholds, selection
    selection["passes_target"] = (
        (selection["return_retention_pct"].fillna(-np.inf) >= 85.0)
        & (selection["return_retention_pct"].fillna(np.inf) <= 95.0)
        & (selection["max_drawdown_pct"].fillna(-np.inf) > -18.0)
        & (selection["skipped_periods"].fillna(0) <= selection["periods"].fillna(0) * 0.5)
    )
    target_candidates = selection.loc[selection["passes_target"]].copy()
    if target_candidates.empty:
        return base_thresholds, selection.sort_values(
            ["passes_target", "selection_score", "total_return_pct"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    selected = target_candidates.sort_values(
        [
            "drawdown_improvement_pct",
            "selection_score",
            "total_return_pct",
            "return_retention_pct",
            "signal_threshold",
            "risk_threshold",
        ],
        ascending=[False, False, False, False, True, True],
    ).iloc[0]
    thresholds = {
        "signal_threshold": float(selected["signal_threshold"]),
        "risk_threshold": float(selected["risk_threshold"]),
    }
    return thresholds, selection.sort_values(
        ["passes_target", "selection_score", "total_return_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _select_variant(model_comparison: pd.DataFrame, *, cost_bps: float) -> dict[str, Any]:
    validation = model_comparison.loc[model_comparison["segment"].eq("validation")].copy()
    if validation.empty:
        validation = model_comparison.copy()
    validation["passes_target"] = (
        (validation["return_retention_pct"].fillna(-np.inf) >= 85.0)
        & (validation["max_drawdown_pct"].fillna(-np.inf) > -18.0)
        & (validation["skipped_periods"].fillna(0) <= validation["periods"].fillna(0) * 0.5)
    )
    eligible = validation.loc[validation["passes_target"]]
    selected = (eligible if not eligible.empty else validation).sort_values(
        ["passes_target", "selection_score", "total_return_pct"],
        ascending=[False, False, False],
    ).iloc[0]
    variant = str(selected["variant"])
    test = model_comparison.loc[(model_comparison["segment"].eq("test")) & (model_comparison["variant"].eq(variant))]
    test_row = test.iloc[0] if not test.empty else selected
    return {
        "variant": variant,
        "selection_segment": "validation",
        "report_segment": "test",
        "selection_uses_test_segment": False,
        "selection_cost_bps": float(cost_bps),
        "selection_metric": "validation return_retention + 2*drawdown_improvement + 0.25*sharpe - skipped_penalty",
        "selection_score": float(selected["selection_score"]),
        "validation_passes_target": bool(selected["passes_target"]),
        "validation_total_return_pct": float(selected["total_return_pct"]),
        "validation_max_drawdown_pct": float(selected["max_drawdown_pct"]),
        "validation_return_retention_pct": float(selected["return_retention_pct"]),
        "validation_drawdown_improvement_pct": float(selected["drawdown_improvement_pct"]),
        "test_total_return_pct": float(test_row["total_return_pct"]),
        "test_max_drawdown_pct": float(test_row["max_drawdown_pct"]),
        "test_return_retention_pct": float(test_row["return_retention_pct"]),
        "test_drawdown_improvement_pct": float(test_row["drawdown_improvement_pct"]),
    }


def _walk_forward_metrics(trades: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    folds = [
        {
            "fold": "wf_2025_h2",
            "validation": ("2025-01-02", "2025-06-30"),
            "test": ("2025-07-01", "2025-12-31"),
        },
        {
            "fold": "wf_2026_ytd",
            "validation": ("2025-01-02", "2025-12-31"),
            "test": ("2026-01-02", "2026-07-02"),
        },
    ]
    rows: list[dict[str, Any]] = []
    base = trades.loc[trades["cost_bps"].eq(float(cost_bps))].copy()
    base["datetime_ts"] = pd.to_datetime(base["datetime"])
    for fold in folds:
        valid_start, valid_end = fold["validation"]
        test_start, test_end = fold["test"]
        valid_trades = base.loc[base["datetime_ts"].between(pd.Timestamp(valid_start), pd.Timestamp(valid_end))]
        test_trades = base.loc[base["datetime_ts"].between(pd.Timestamp(test_start), pd.Timestamp(test_end))]
        valid_metrics = summarize_overlay_metrics(valid_trades.drop(columns=["datetime_ts"], errors="ignore"))
        test_metrics = summarize_overlay_metrics(test_trades.drop(columns=["datetime_ts"], errors="ignore"))
        valid_selection = valid_metrics.loc[valid_metrics["segment"].isin(["validation", "test"])].copy()
        if valid_selection.empty:
            continue
        selected = valid_selection.sort_values(["selection_score", "total_return_pct"], ascending=False).iloc[0]
        variant = str(selected["variant"])
        test_row = test_metrics.loc[test_metrics["variant"].eq(variant)]
        if test_row.empty:
            continue
        test_row = test_row.iloc[0]
        rows.append(
            {
                "fold": fold["fold"],
                "cost_bps": float(cost_bps),
                "valid_start": valid_start,
                "valid_end": valid_end,
                "test_start": test_start,
                "test_end": test_end,
                "selected_variant": variant,
                "selection_uses_test_segment": False,
                "valid_total_return_pct": float(selected["total_return_pct"]),
                "valid_max_drawdown_pct": float(selected["max_drawdown_pct"]),
                "valid_return_retention_pct": float(selected["return_retention_pct"]),
                "test_total_return_pct": float(test_row["total_return_pct"]),
                "test_max_drawdown_pct": float(test_row["max_drawdown_pct"]),
                "test_return_retention_pct": float(test_row["return_retention_pct"]),
            }
        )
    return pd.DataFrame(rows)


def _analysis_summary(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    daily_summary: dict[str, Any],
    intraday_summary: dict[str, Any],
    daily_positions: pd.DataFrame,
    daily_prices: pd.DataFrame,
    intraday_predictions: pd.DataFrame,
    thresholds: dict[str, float],
    signal_config: dict[str, str],
    model_comparison: pd.DataFrame,
    threshold_selection: pd.DataFrame,
    selected_model: dict[str, Any],
    walk_forward: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "source": "daily_open2open_position_model_plus_15m_execution_overlay",
        "as_of": str(daily_summary.get("as_of") or spec.as_of),
        "selection_uses_test_segment": False,
    "thresholds": {
            "signal_threshold": thresholds["signal_threshold"],
            "signal_threshold_source": (
                "overlay_validation_threshold_search"
                if spec.optimize_thresholds
                else "source_intraday_selected_model_valid_threshold"
            ),
            "risk_threshold": thresholds["risk_threshold"],
            "risk_threshold_source": (
                "overlay_validation_threshold_search"
                if spec.optimize_thresholds
                else "validation_gated_probability_quantile_20pct_or_user_override"
            ),
            "signal_prob_col": signal_config["prob_col"],
            "gate_col": signal_config["gate_col"],
            "gate_policy": signal_config["gate_policy"],
            "threshold_search_enabled": bool(spec.optimize_thresholds),
        },
        "data_audit": {
            "daily_selected_position_rows": int(len(daily_positions)),
            "daily_selected_position_start": daily_positions["datetime"].min().date().isoformat(),
            "daily_selected_position_end": daily_positions["datetime"].max().date().isoformat(),
            "daily_price_rows": int(len(daily_prices)),
            "daily_price_start": daily_prices["datetime"].min().date().isoformat(),
            "daily_price_end": daily_prices["datetime"].max().date().isoformat(),
            "intraday_rows": int(len(intraday_predictions)),
            "intraday_prediction_rows": int(intraday_predictions[signal_config["prob_col"]].notna().sum()),
            "intraday_start": intraday_predictions["datetime"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "intraday_end": intraday_predictions["datetime"].max().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "source_daily_model": {
            "recorder_id": daily_summary.get("recorder_id"),
            "execution_timing": daily_summary.get("position_model", {}).get("execution_timing"),
            "feature_count": daily_summary.get("position_model", {}).get("feature_count"),
            "has_soxx_constituent_features": any(
                str(col).startswith("soxx_const_")
                for col in daily_summary.get("position_model", {}).get("feature_cols", [])
            ),
            "has_intraday_t1_features": any(
                str(col).startswith("intraday15m_")
                for col in daily_summary.get("position_model", {}).get("feature_cols", [])
            ),
        },
        "source_intraday_model": {
            "recorder_id": intraday_summary.get("recorder_id"),
            "selected_model": intraday_summary.get("selected_model", {}).get("model"),
            "trade_gate": intraday_summary.get("trade_gate"),
            "label_policy": intraday_summary.get("label_policy"),
            "external_context": intraday_summary.get("external_intraday_context"),
        },
        "selected_model": selected_model,
        "model_comparison_rows": int(len(model_comparison)),
        "threshold_selection_rows": int(0 if not spec.optimize_thresholds else len(threshold_selection)),
        "walk_forward_rows": int(len(walk_forward)),
    }


def _write_workflow_yaml(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    daily_summary: dict[str, Any],
    intraday_summary: dict[str, Any],
) -> Path:
    path = spec.output_dir / "workflow.yaml"
    path.write_text(
        "\n".join(
            [
                f"experiment: {spec.experiment}",
                "qlib:",
                f"  recorder_uri: {spec.recorder_uri.resolve().as_posix()}",
                f"  daily_provider_uri: {spec.daily_provider_uri.resolve().as_posix()}",
                f"  intraday_provider_uri: {spec.intraday_provider_uri.resolve().as_posix()}",
                "sources:",
                f"  daily_card: {spec.source_daily_card}",
                f"  daily_recorder_id: {daily_summary.get('recorder_id')}",
                f"  intraday_card: {spec.source_intraday_card}",
                f"  intraday_recorder_id: {intraday_summary.get('recorder_id')}",
                "selection:",
                "  segment: validation",
                "  report_segment: test",
                "  uses_test_segment: false",
                f"  selection_cost_bps: {float(spec.selection_cost_bps)}",
                f"  cost_stress_bps: {list(float(value) for value in sorted(set(spec.cost_stress_bps)))}",
                "execution:",
                f"  daily_timing: {DAILY_EXECUTION_TIMING}",
                f"  overlay_signal_timing: {OVERLAY_SIGNAL_TIMING}",
                "  variants:",
                *[f"    - {variant}" for variant in VARIANTS],
                "artifact_policy:",
                "  policy: slim_card_recorder_first",
                "  machine_entrypoint: summary.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_regime_workflow_yaml(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    daily_summary: dict[str, Any],
    intraday_summary: dict[str, Any],
) -> Path:
    path = spec.output_dir / "workflow.yaml"
    path.write_text(
        "\n".join(
            [
                f"experiment: {spec.experiment}",
                "qlib:",
                f"  recorder_uri: {spec.recorder_uri.resolve().as_posix()}",
                f"  daily_provider_uri: {spec.daily_provider_uri.resolve().as_posix()}",
                f"  intraday_provider_uri: {spec.intraday_provider_uri.resolve().as_posix()}",
                "sources:",
                f"  daily_card: {spec.source_daily_card}",
                f"  daily_recorder_id: {daily_summary.get('recorder_id')}",
                f"  intraday_card: {spec.source_intraday_card}",
                f"  intraday_recorder_id: {intraday_summary.get('recorder_id')}",
                "selection:",
                "  segment: validation",
                "  report_segment: test",
                "  uses_test_segment: false",
                f"  selection_cost_bps: {float(spec.selection_cost_bps)}",
                "regime_policy:",
                "  causal_scope: daily_close_known_features_only",
                "  condition_threshold_source: validation_quantile",
                "  target_return_retention_pct: [85.0, 95.0]",
                "  target_max_drawdown_floor_pct: -18.0",
                "artifact_policy:",
                "  policy: slim_card_recorder_first",
                "  machine_entrypoint: summary.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _regime_overlay_thresholds(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    intraday_predictions: pd.DataFrame,
    intraday_summary: dict[str, Any],
    signal_config: dict[str, str],
) -> dict[str, float]:
    overlay_summary_path = spec.root / "research" / EXTERNAL_CONTEXT_OVERLAY_EXPERIMENT / "summary.json"
    if overlay_summary_path.exists():
        overlay_summary = _read_json(overlay_summary_path)
        selected = overlay_summary.get("selected_model", {})
        if "signal_threshold" in selected and "risk_threshold" in selected:
            return {
                "signal_threshold": float(selected["signal_threshold"]),
                "risk_threshold": float(selected["risk_threshold"]),
                "source": "external_context_overlay_validation_search",
            }
    thresholds = _thresholds_from_validation(
        intraday_predictions,
        intraday_summary=intraday_summary,
        signal_config=signal_config,
        signal_threshold=spec.signal_threshold,
        risk_threshold=spec.risk_threshold,
    )
    thresholds["source"] = "source_intraday_validation_threshold"
    return thresholds


def _select_regime_policy(policy_selection: pd.DataFrame) -> dict[str, Any]:
    if policy_selection.empty:
        return {
            "variant": "baseline_next_open",
            "selection_segment": "validation",
            "report_segment": "test",
            "selection_uses_test_segment": False,
            "validation_passes_target": False,
            "test_passes_target": False,
        }
    validation_passes = policy_selection.loc[policy_selection["passes_validation_target"].astype(bool)].copy()
    source = validation_passes if not validation_passes.empty else policy_selection.copy()
    selected = source.sort_values(
        [
            "passes_validation_target",
            "selection_score",
            "validation_drawdown_improvement_pct",
            "validation_return_pct",
        ],
        ascending=[False, False, False, False],
    ).iloc[0]
    result = _json_ready_dict(selected.to_dict())
    result["validation_passes_target"] = bool(result.get("passes_validation_target", False))
    result["test_passes_target"] = bool(result.get("passes_test_target", False))
    result["selection_uses_test_segment"] = False
    return result


def _regime_analysis_summary(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    daily_summary: dict[str, Any],
    intraday_summary: dict[str, Any],
    daily_positions: pd.DataFrame,
    daily_prices: pd.DataFrame,
    intraday_predictions: pd.DataFrame,
    thresholds: dict[str, float],
    signal_config: dict[str, str],
    model_comparison: pd.DataFrame,
    policy_selection: pd.DataFrame,
    selected_model: dict[str, Any],
    condition_columns: tuple[str, ...],
) -> dict[str, Any]:
    validation_target_count = int(
        policy_selection["passes_validation_target"].sum()
        if "passes_validation_target" in policy_selection
        else 0
    )
    validation_and_test_count = int(
        policy_selection["passes_validation_and_test_target"].sum()
        if "passes_validation_and_test_target" in policy_selection
        else 0
    )
    return {
        "source": "daily_open2open_position_model_plus_causal_daily_regime_intraday_execution_overlay",
        "as_of": str(daily_summary.get("as_of") or spec.as_of),
        "selection_uses_test_segment": False,
        "thresholds": {
            "signal_threshold": float(thresholds["signal_threshold"]),
            "risk_threshold": float(thresholds["risk_threshold"]),
            "source": thresholds.get("source"),
            "signal_prob_col": signal_config["prob_col"],
            "gate_col": signal_config["gate_col"],
            "gate_policy": signal_config["gate_policy"],
        },
        "regime_policy": {
            "condition_columns": list(condition_columns),
            "condition_threshold_source": "validation_quantile",
            "causal_scope": "daily_close_known_features_only",
            "noncausal_first_hour_probe_excluded_from_selection": True,
            "validation_target_candidate_count": validation_target_count,
            "validation_and_test_target_candidate_count": validation_and_test_count,
        },
        "data_audit": {
            "daily_selected_position_rows": int(len(daily_positions)),
            "daily_selected_position_start": daily_positions["datetime"].min().date().isoformat(),
            "daily_selected_position_end": daily_positions["datetime"].max().date().isoformat(),
            "daily_price_rows": int(len(daily_prices)),
            "daily_price_start": daily_prices["datetime"].min().date().isoformat(),
            "daily_price_end": daily_prices["datetime"].max().date().isoformat(),
            "intraday_rows": int(len(intraday_predictions)),
            "intraday_prediction_rows": int(intraday_predictions[signal_config["prob_col"]].notna().sum()),
            "intraday_start": intraday_predictions["datetime"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "intraday_end": intraday_predictions["datetime"].max().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "source_daily_model": {
            "recorder_id": daily_summary.get("recorder_id"),
            "execution_timing": daily_summary.get("position_model", {}).get("execution_timing"),
            "feature_count": daily_summary.get("position_model", {}).get("feature_count"),
            "has_soxx_constituent_features": any(
                str(col).startswith("soxx_const_")
                for col in daily_summary.get("position_model", {}).get("feature_cols", [])
            ),
        },
        "source_intraday_model": {
            "recorder_id": intraday_summary.get("recorder_id"),
            "selected_model": intraday_summary.get("selected_model", {}).get("model"),
            "label_policy": intraday_summary.get("label_policy"),
            "external_context": intraday_summary.get("external_intraday_context"),
        },
        "model_comparison_rows": int(len(model_comparison)),
        "policy_selection_rows": int(len(policy_selection)),
        "selected_model": selected_model,
    }


def _write_regime_sidecars(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    model_comparison: pd.DataFrame,
    policy_selection: pd.DataFrame,
    selected_model: dict[str, Any],
    analysis_summary: dict[str, Any],
) -> dict[str, str]:
    sidecars = {
        "model_comparison": "model_comparison.csv",
        "policy_selection": "policy_selection.csv",
        "selected_model": "selected_model.json",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
    }
    model_comparison.to_csv(spec.output_dir / sidecars["model_comparison"], index=False)
    policy_selection.to_csv(spec.output_dir / sidecars["policy_selection"], index=False)
    (spec.output_dir / sidecars["selected_model"]).write_text(
        json.dumps(selected_model, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (spec.output_dir / sidecars["analysis_summary"]).write_text(
        json.dumps(analysis_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    failure_modes = (
        "# Failure Modes\n\n"
        "- Regime overlay 只允许使用 daily close 时已知的 probability、target_position、turnover、raw_target 做条件选择。\n"
        "- 第一小时走势或 15m 概率不能用于选择前 4 根 15m TWAP；这类条件会造成执行时序泄漏，已从正式选择中排除。\n"
        "- 即使 validation 有候选过 85%-95% retention 与 max_dd>-18% 门槛，test 仍必须独立报告，不能回填调参。\n"
        "- 当前外部 15m 历史仍不足以证明完整 2019-2026 walk-forward 推广。\n"
    )
    (spec.output_dir / sidecars["failure_modes"]).write_text(failure_modes, encoding="utf-8")
    return sidecars


def _regime_artifact_policy(experiment: str, sidecars: dict[str, str]) -> dict[str, Any]:
    return {
        "policy": "slim_card_recorder_first",
        "primary_artifact": f"{experiment}.ipynb",
        "machine_entrypoint": "summary.json",
        "workflow_config": "workflow.yaml",
        "kept_sidecars": sorted(sidecars.values()),
        "recorder_only_artifacts": {
            "overlay_trades": "artifacts/regime_overlay/overlay_trades.csv",
            "regime_features": "artifacts/regime_overlay/regime_features.csv",
            "policy_selection": "artifacts/regime_overlay/policy_selection.csv",
        },
        "reproducible_from_recorder": True,
    }


def _regime_promotion_decision(selected_model: dict[str, Any], *, intraday_summary: dict[str, Any]) -> str:
    if not bool(selected_model.get("validation_passes_target")):
        return "not_promoted_regime_overlay_validation_gate_failed"
    if not bool(selected_model.get("test_passes_target")):
        return "not_promoted_regime_overlay_test_gate_failed"
    if intraday_summary.get("walk_forward", {}).get("required_policy_satisfied") is False:
        return "not_promoted_external_intraday_history_insufficient"
    return "not_promoted_intraday_replay_missing"


def _write_regime_conclusion(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    summary: dict[str, Any],
    selected_model: dict[str, Any],
) -> None:
    policy = summary["regime_policy"]
    text = (
        "# 中文结论\n\n"
        "- 做了什么：建立 causal regime overlay 诊断卡，日线 LGB/XGB open-to-open 仓位仍是 baseline，"
        "外部 15m LGB 模型只作为执行 overlay；条件选择只允许使用 daily close 时已知的 probability、target_position、turnover、raw_target。\n"
        f"- 选择规则：只用 validation、{spec.selection_cost_bps:.0f}bp 成本和 validation quantile 阈值选条件；test 只报告，不参与调参。\n"
        f"- 候选数：{policy['candidate_count']}；validation 过 85%-95% retention 且 max_dd>-18% 的候选数="
        f"{policy['validation_target_candidate_count']}；validation 和 test 同时过门槛的候选数="
        f"{policy['validation_and_test_target_candidate_count']}。\n"
        f"- validation 选中：{selected_model.get('variant')} / {selected_model.get('condition_name')}，"
        f"retention={float(selected_model.get('validation_return_retention_pct') or 0.0):.2f}%，"
        f"max_dd={float(selected_model.get('validation_max_drawdown_pct') or 0.0):.2f}%。\n"
        f"- test 报告：retention={float(selected_model.get('test_return_retention_pct') or 0.0):.2f}%，"
        f"max_dd={float(selected_model.get('test_max_drawdown_pct') or 0.0):.2f}%。\n"
        f"- 是否晋升：{summary['promotion_decision']}。结论是当前 causal 日内 overlay 还不能替代日线 next-open 执行；"
        "下一步应优化日线整体仓位模型或做真正延迟入场/风险退出模型，而不是用看完第一小时后的特征证明开盘 TWAP。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
    )
    (spec.output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def _json_ready_dict(values: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (np.bool_, bool)):
            result[key] = bool(value)
        elif isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = None if np.isnan(value) else float(value)
        elif pd.isna(value):
            result[key] = None
        else:
            result[key] = value
    return result


def _write_sidecars(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    model_comparison: pd.DataFrame,
    threshold_selection: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    walk_forward: pd.DataFrame,
    selected_model: dict[str, Any],
    analysis_summary: dict[str, Any],
) -> dict[str, str]:
    sidecars = {
        "model_comparison": "model_comparison.csv",
        "selected_model": "selected_model.json",
        "cost_sensitivity": "cost_sensitivity.csv",
        "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
    }
    if spec.optimize_thresholds:
        sidecars["threshold_selection"] = "threshold_selection.csv"
    model_comparison.to_csv(spec.output_dir / sidecars["model_comparison"], index=False)
    if spec.optimize_thresholds:
        threshold_selection.to_csv(spec.output_dir / sidecars["threshold_selection"], index=False)
    cost_sensitivity.to_csv(spec.output_dir / sidecars["cost_sensitivity"], index=False)
    walk_forward.to_csv(spec.output_dir / sidecars["backtest_walk_forward_metrics"], index=False)
    (spec.output_dir / sidecars["selected_model"]).write_text(
        json.dumps(selected_model, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (spec.output_dir / sidecars["analysis_summary"]).write_text(
        json.dumps(analysis_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    failure_modes = (
        "# Failure Modes\n\n"
        "- Overlay 使用的是已有日级 target_position，不重新训练仓位模型；它只能改善成交和盘中退出，不能创造隔夜趋势收益。\n"
        "- 15m 信号来自 bar close，overlay A 只允许下一根 15m open 入场；没有使用同一根 bar 的未来价格。\n"
        "- 成本压力用固定 bps；尚未验证 bid/ask、排队、延迟、停牌、跳空和盘前盘后成交质量。\n"
        "- walk-forward 只在已有日级 selected positions 上重新选择执行 variant；不是完整重训日级 LGB/XGB 仓位模型。\n"
        "- 若 validation 选择 baseline，说明日内执行层没有足够证据替代 next-open 一次成交。\n"
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
            "overlay_trades": "artifacts/execution_overlay/overlay_trades.csv",
            "daily_positions": "artifacts/execution_overlay/daily_selected_positions.csv",
            "intraday_predictions": "artifacts/execution_overlay/intraday_predictions_15m.csv",
        },
        "reproducible_from_recorder": True,
    }


def _required_overlay_policy_satisfied(
    *,
    intraday_summary: dict[str, Any],
    walk_forward: pd.DataFrame,
) -> bool:
    source_required_policy = intraday_summary.get("walk_forward", {}).get("required_policy_satisfied", True)
    has_overlay_folds = not walk_forward.empty and set(walk_forward.get("fold", pd.Series(dtype=object))) >= {
        "wf_2025_h2",
        "wf_2026_ytd",
    }
    return bool(source_required_policy and has_overlay_folds)


def _promotion_decision(selected_model: dict[str, Any], *, intraday_summary: dict[str, Any]) -> str:
    if intraday_summary.get("walk_forward", {}).get("required_policy_satisfied") is False:
        return "not_promoted_external_intraday_history_insufficient"
    if selected_model["variant"] == "baseline_next_open":
        return "not_promoted_overlay_did_not_beat_daily_execution"
    if not bool(selected_model.get("validation_passes_target")):
        return "not_promoted_overlay_validation_gate_failed"
    if float(selected_model.get("test_return_retention_pct", 0.0)) < 85.0:
        return "not_promoted_test_return_retention_failed"
    return "not_promoted_intraday_replay_missing"


def _write_conclusion(
    spec: CognitiveAlphaIntradayOverlaySpec,
    *,
    summary: dict[str, Any],
    selected_model: dict[str, Any],
    model_comparison: pd.DataFrame,
) -> None:
    test_rows = model_comparison.loc[model_comparison["segment"].eq("test")].copy()
    selected_test = test_rows.loc[test_rows["variant"].eq(selected_model["variant"])]
    selected_test_row = selected_test.iloc[0] if not selected_test.empty else None
    baseline_test = test_rows.loc[test_rows["variant"].eq("baseline_next_open")]
    baseline_return = float(baseline_test.iloc[0]["total_return_pct"]) if not baseline_test.empty else 0.0
    selected_test_return = float(selected_test_row["total_return_pct"]) if selected_test_row is not None else 0.0
    selected_test_dd = float(selected_test_row["max_drawdown_pct"]) if selected_test_row is not None else 0.0
    threshold_search = summary.get("model_selection", {}).get("threshold_search", {})
    threshold_sentence = (
        f"- 阈值搜索：只用 validation 搜 {threshold_search.get('candidate_count', 0)} 个候选，"
        f"满足 85%-95% retention 且 max_dd>-18% 的候选数={threshold_search.get('target_candidate_count', 0)}。\n"
        if threshold_search.get("enabled")
        else ""
    )
    overlay_name = (
        "外部 15m 上下文执行 overlay"
        if spec.source_intraday_card == EXTERNAL_CONTEXT_INTRADAY_SOURCE_CARD
        else "日级模型 + 日内执行 overlay"
    )
    intraday_source_text = (
        "外部 15m 上下文 triple-barrier/execution-improvement 模型"
        if spec.source_intraday_card == EXTERNAL_CONTEXT_INTRADAY_SOURCE_CARD
        else "15m lgb_xgb_consensus"
    )
    text = (
        "# 中文结论\n\n"
        f"- 做了什么：建立“{overlay_name}”实验，日级 LGB/XGB 仓位模型仍决定整体 target_position，"
        f"{intraday_source_text}只负责下一交易日的入场过滤、前 4 根 15m TWAP 和盘中风险退出。\n"
        f"- 选择规则：只用 validation、{spec.selection_cost_bps:.0f}bp 成本选择 variant；test 只报告，"
        "没有用 test 调阈值或调参数。\n"
        f"- validation 选择：{selected_model['variant']}，retention={selected_model['validation_return_retention_pct']:.2f}%，"
        f"max_dd={selected_model['validation_max_drawdown_pct']:.2f}%。\n"
        f"{threshold_sentence}"
        f"- test 对照：baseline_next_open return={baseline_return:.2f}%，"
        f"selected return={selected_test_return:.2f}%、max_dd={selected_test_dd:.2f}%。\n"
        f"- 是否晋升：{summary['promotion_decision']}。如果 selected 仍是 baseline，结论就是当前 15m 执行层没有可靠超过 next-open 一次成交。\n"
        "- 风险：这不是独立日内 alpha；它依赖日级模型的趋势和隔夜收益来源。真实推广前还要补 bid/ask replay、成交队列、延迟和盘前盘后风险验证。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
    )
    (spec.output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def _empty_variant_row(daily_row: dict[str, Any], variant: str, cost_bps: float, reason: str) -> dict[str, Any]:
    date = pd.Timestamp(daily_row["datetime"]).date().isoformat()
    return {
        "datetime": date,
        "segment": str(daily_row.get("segment", "unknown")),
        "target_position": float(daily_row.get("target_position", 0.0) or 0.0),
        "cost_bps": float(cost_bps),
        "entry_date": None,
        "exit_date": None,
        "source_daily_strategy_return": float(daily_row.get("strategy_return", 0.0) or 0.0),
        "variant": variant,
        "entry_time": None,
        "exit_time": None,
        "entry_price": np.nan,
        "exit_price": np.nan,
        "intraday_signal_score": np.nan,
        "gross_return": 0.0,
        "strategy_return": 0.0,
        "skipped_reason": reason,
    }


def _date_or_none(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _timestamp_or_none(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_root(root: Path | str | None) -> Path:
    if root is None:
        return Path(__file__).resolve().parents[2]
    return Path(root).resolve()


def _recorder_files(recorder_dir: Path) -> list[str]:
    return sorted(path.relative_to(recorder_dir).as_posix() for path in recorder_dir.rglob("*") if path.is_file())


def _remove_transient_code_snapshots(recorder_dir: Path) -> None:
    for name in ["code_cached.txt", "code_diff.txt", "code_status.txt"]:
        path = recorder_dir / "artifacts" / name
        if path.exists():
            path.unlink()
