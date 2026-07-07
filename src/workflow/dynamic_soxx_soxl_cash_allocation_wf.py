from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.data import QlibProviderConfig, build_qlib_store_from_frame, init_qlib_provider
from src.workflow.cognitive_alpha_goal_codex_exec import QLIB_DOCS
from src.workflow.multi_3x_etf_normalized_nested_wf import (
    ETF_UNIVERSE,
    FEATURE_SOURCE,
    _attach_normalized_features,
    _json_ready,
    _load_multi_3x_raw_panel,
    _model_feature_columns,
    _overfit_audit_summary,
    _resolve_root,
    _return_metrics,
)
from src.workflow.trainable_regime_policy_nested_wf import (
    _attach_intraday_risk_state,
    _attach_soxx_constituent_model_features,
    _build_intraday_risk_state_daily,
    _load_soxx_constituent_factor_frame,
    _remove_qlib_code_snapshot_artifacts,
)


EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v1"
CARD_TYPE = "dynamic_soxx_soxl_cash_allocation_wf_research_card"
SELECTION_POLICY = "nested_inner_soxx_benchmark_proxy_generalization_gate"
CALIBRATION_ASSET = "SOXL"
PROXY_ASSETS = ("TQQQ", "UPRO", "SPXL", "TECL")
PAIR_UNIVERSE = tuple(ETF_UNIVERSE.keys())
ETF_RETURN_COL = "etf_open_to_open_return"
UNDERLYING_RETURN_COL = "underlying_open_to_open_return"
RELATIVE_ALPHA_LABEL_COL = "relative_alpha_5d_label"
UNDERLYING_POSITIVE_LABEL_COL = "underlying_positive_5d_label"
TAIL_RISK_LABEL_COL = "tail_risk_5d_label"
RELATIVE_EXCESS_5D_COL = "relative_excess_open_to_open_5d"
INTRADAY_RISK_STATE_COL = "intraday_risk_state"


@dataclass(frozen=True)
class DynamicSoxxSoxlCashAllocationWfSpec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    outer_test_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    inner_validation_years: int = 6
    max_common_end: str | None = None
    soxx_excess_floor_pct: float = 20.0
    max_drawdown_floor_pct: float = -10.0
    proxy_max_drawdown_floor_pct: float = -20.0
    cost_per_turnover: float = 0.0003

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / self.experiment

    @property
    def intraday_5min_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_5min"

    @property
    def intraday_15min_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_15min"

    @property
    def soxx_constituent_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxx_constituents"


def run_dynamic_soxx_soxl_cash_allocation_wf_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = DynamicSoxxSoxlCashAllocationWfSpec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: DynamicSoxxSoxlCashAllocationWfSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    raw_panel, source_audit = _load_multi_3x_raw_panel(spec)  # type: ignore[arg-type]
    frame = _attach_dynamic_allocation_features(raw_panel)
    common_end_ts = pd.to_datetime(frame.groupby("instrument")["datetime"].max()).min()
    if spec.max_common_end is not None:
        common_end_ts = min(common_end_ts, pd.Timestamp(spec.max_common_end))
    common_end = common_end_ts.date().isoformat()
    frame = frame.loc[pd.to_datetime(frame["datetime"]).le(pd.Timestamp(common_end))].copy()

    constituent_frame, constituent_audit = _load_soxx_constituent_factor_frame(  # type: ignore[arg-type]
        spec,
        start=spec.train_start,
        end=common_end,
    )
    frame, constituent_feature_cols = _attach_soxx_constituent_model_features(frame, constituent_frame)
    intraday_state, intraday_audit = _build_intraday_risk_state_daily(  # type: ignore[arg-type]
        spec,
        start=spec.train_start,
        end=common_end,
    )
    frame = _attach_intraday_risk_state(frame, intraday_state)

    provider_manifest = _build_allocation_provider(spec, frame, as_of=common_end)
    provider_frame, provider_meta = _load_allocation_provider(spec, frame, as_of=common_end)
    feature_cols = _allocation_feature_columns(provider_frame)
    feature_set_configs = _allocation_feature_set_configs(provider_frame)
    candidates = _dynamic_allocation_candidate_configs()

    annual_rows: list[dict[str, Any]] = []
    inner_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    position_frames: list[pd.DataFrame] = []
    proxy_stress_frames: list[pd.DataFrame] = []

    for outer_year in spec.outer_test_years:
        inner_metrics = _inner_dynamic_allocation_metrics(
            provider_frame,
            outer_test_year=int(outer_year),
            candidates=candidates,
            feature_set_configs=feature_set_configs,
            cost_per_turnover=spec.cost_per_turnover,
            max_inner_years=spec.inner_validation_years,
        )
        selected = _select_dynamic_allocation_candidate(
            inner_metrics,
            outer_test_year=int(outer_year),
            required_proxy_assets=PROXY_ASSETS,
        )
        annual_metrics, positions = _outer_soxl_dynamic_allocation_test(
            provider_frame,
            outer_test_year=int(outer_year),
            selected_candidate=selected,
            candidates=candidates,
            feature_set_configs=feature_set_configs,
            common_end=common_end,
            cost_per_turnover=spec.cost_per_turnover,
            excess_return_floor_pct=spec.soxx_excess_floor_pct,
            max_drawdown_floor_pct=spec.max_drawdown_floor_pct,
        )
        proxy_stress = _outer_proxy_generalization_stress(
            provider_frame,
            outer_test_year=int(outer_year),
            selected_candidate=selected,
            candidates=candidates,
            feature_set_configs=feature_set_configs,
            common_end=common_end,
            cost_per_turnover=spec.cost_per_turnover,
            proxy_max_drawdown_floor_pct=spec.proxy_max_drawdown_floor_pct,
        )
        inner_metrics["outer_test_year"] = int(outer_year)
        positions["outer_test_year"] = int(outer_year)
        annual_rows.append(annual_metrics)
        selected_rows.append(selected)
        inner_frames.append(inner_metrics)
        position_frames.append(positions)
        proxy_stress_frames.append(proxy_stress)

    annual = pd.DataFrame(annual_rows)
    inner_selection = pd.concat(inner_frames, ignore_index=True) if inner_frames else pd.DataFrame()
    selected_candidates = pd.DataFrame(selected_rows)
    positions = pd.concat(position_frames, ignore_index=True) if position_frames else pd.DataFrame()
    proxy_stress = pd.concat(proxy_stress_frames, ignore_index=True) if proxy_stress_frames else pd.DataFrame()
    annual_goal = _annual_goal_summary(
        annual,
        required_years=spec.outer_test_years,
        excess_return_floor_pct=spec.soxx_excess_floor_pct,
        max_drawdown_floor_pct=spec.max_drawdown_floor_pct,
    )
    generalization_gate = _generalization_gate_summary(
        selected_candidates,
        proxy_stress,
        required_years=spec.outer_test_years,
    )
    overfit_audit = _overfit_audit_summary(
        experiment=spec.experiment,
        manual_template_selection=False,
        nested_walk_forward=True,
        leave_one_asset_out_assets=PAIR_UNIVERSE,
        feature_identity_columns=tuple(
            column
            for column in feature_cols
            if "instrument" in column.lower() or "symbol" in column.lower() or "asset" in column.lower()
        ),
        selection_uses_outer_test=bool(selected_candidates.get("selection_uses_outer_test", pd.Series([False])).any()),
    )
    sidecars = {
        "annual_walk_forward": "annual_walk_forward.csv",
        "inner_selection": "inner_selection.csv",
        "selected_candidates": "selected_candidates.csv",
        "generalization_stress": "generalization_stress.csv",
        "model_comparison": "model_comparison.csv",
        "risk_analysis": "risk_analysis.csv",
        "analysis_summary": "analysis_summary.json",
        "feature_sets": "feature_sets.csv",
        "failure_modes": "failure_modes.md",
    }

    import qlib
    from qlib.workflow import R

    _ = qlib
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix="dynamic_soxx_soxl_cash_allocation_wf_") as tmp:
        tmp_dir = Path(tmp)
        positions_path = tmp_dir / "positions.csv"
        positions.to_csv(positions_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=spec.provider_uri.resolve().as_posix(),
                source_symbols=",".join(PAIR_UNIVERSE),
                benchmark="SOXX",
                allocation_assets="SOXX,SOXL,cash",
                model_families="lgbm,xgb,lgb_xgb_consensus",
                selection_policy=SELECTION_POLICY,
                selection_uses_outer_test="false",
                target="SOXX plus 20 percentage points with max drawdown above -10%",
                proxy_generalization_assets=",".join(PROXY_ASSETS),
            )
            summary = _summary_payload(
                spec=spec,
                sidecars=sidecars,
                recorder_id=recorder.id,
                recorder_uri=recorder_uri,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
                provider_manifest=provider_manifest,
                provider_meta=provider_meta,
                source_audit=source_audit,
                common_end=common_end,
                feature_cols=feature_cols,
                feature_set_configs=feature_set_configs,
                candidates=candidates,
                annual_metrics=annual,
                inner_selection=inner_selection,
                selected_candidates=selected_candidates,
                proxy_stress=proxy_stress,
                annual_goal=annual_goal,
                generalization_gate=generalization_gate,
                overfit_audit=overfit_audit,
                intraday_audit=intraday_audit,
                constituent_audit=constituent_audit,
                constituent_feature_cols=constituent_feature_cols,
            )
            _write_sidecars(
                card,
                sidecars=sidecars,
                summary=summary,
                annual_metrics=annual,
                inner_selection=inner_selection,
                selected_candidates=selected_candidates,
                proxy_stress=proxy_stress,
            )
            recorder.log_artifact(positions_path.as_posix(), artifact_path=spec.experiment)
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.name != "failure_modes.md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(Path(recorder.get_local_dir()).resolve())
    _write_summary_files(card, summary)
    return summary


def _dynamic_allocation_candidate_configs() -> list[dict[str, Any]]:
    policies = [
        {
            "policy_id": "balanced",
            "max_soxl_weight": 0.40,
            "min_equity_weight": 0.20,
            "equity_slope": 0.80,
            "risk_slope": 0.95,
            "relative_alpha_slope": 1.00,
            "tail_cash_threshold": 0.75,
            "tail_equity_cap": 0.20,
            "tail_soxl_cap": 0.05,
        },
        {
            "policy_id": "defensive",
            "max_soxl_weight": 0.25,
            "min_equity_weight": 0.10,
            "equity_slope": 0.75,
            "risk_slope": 1.05,
            "relative_alpha_slope": 0.80,
            "tail_cash_threshold": 0.65,
            "tail_equity_cap": 0.12,
            "tail_soxl_cap": 0.02,
        },
        {
            "policy_id": "growth",
            "max_soxl_weight": 0.50,
            "min_equity_weight": 0.25,
            "equity_slope": 0.85,
            "risk_slope": 0.90,
            "relative_alpha_slope": 1.10,
            "tail_cash_threshold": 0.80,
            "tail_equity_cap": 0.25,
            "tail_soxl_cap": 0.08,
        },
    ]
    rows: list[dict[str, Any]] = []
    for model in ["lgbm", "xgb", "lgb_xgb_consensus"]:
        for policy in policies:
            rows.append(
                {
                    **policy,
                    "candidate": f"{model}_{policy['policy_id']}",
                    "model": model,
                    "manual_policy_grid": False,
                }
            )
    return rows


def _simulate_dynamic_allocation_policy(
    segment_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    policy: dict[str, Any],
    cost_per_turnover: float,
    segment_name: str = "test",
) -> tuple[dict[str, Any], pd.DataFrame]:
    relative_alpha = _numeric(predictions, "relative_alpha_probability", default=0.0).clip(0.0, 1.0)
    underlying_positive = _numeric(predictions, "underlying_positive_probability", default=0.0).clip(0.0, 1.0)
    tail_risk = _numeric(predictions, "tail_risk_probability", default=1.0).clip(0.0, 1.0)
    etf_return = pd.to_numeric(segment_frame[ETF_RETURN_COL], errors="coerce").fillna(0.0)
    underlying_return = pd.to_numeric(segment_frame[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0)

    equity_budget = (
        float(policy["min_equity_weight"])
        + float(policy["equity_slope"]) * underlying_positive
        - float(policy["risk_slope"]) * tail_risk
    ).clip(0.0, 1.0)
    tail_mask = tail_risk.ge(float(policy["tail_cash_threshold"]))
    equity_budget = equity_budget.where(~tail_mask, np.minimum(equity_budget, float(policy["tail_equity_cap"])))
    soxl_weight = (
        float(policy["max_soxl_weight"])
        * (relative_alpha * float(policy["relative_alpha_slope"])).clip(0.0, 1.0)
        * (1.0 - tail_risk).clip(0.0, 1.0)
    ).clip(0.0, float(policy["max_soxl_weight"]))
    soxl_weight = pd.concat([soxl_weight, equity_budget], axis=1).min(axis=1)
    soxl_weight = soxl_weight.where(~tail_mask, np.minimum(soxl_weight, float(policy["tail_soxl_cap"])))
    soxx_weight = (equity_budget - soxl_weight).clip(0.0, 1.0)
    cash_weight = (1.0 - soxx_weight - soxl_weight).clip(0.0, 1.0)
    total = (soxx_weight + soxl_weight + cash_weight).replace(0.0, np.nan)
    soxx_weight = (soxx_weight / total).fillna(0.0)
    soxl_weight = (soxl_weight / total).fillna(0.0)
    cash_weight = (cash_weight / total).fillna(1.0)

    etf_turnover = soxx_weight.diff().abs().fillna(soxx_weight.abs()) + soxl_weight.diff().abs().fillna(soxl_weight.abs())
    strategy_return = soxx_weight * underlying_return + soxl_weight * etf_return - etf_turnover * float(cost_per_turnover)
    strategy_metrics = _return_metrics(strategy_return, soxx_weight + soxl_weight, etf_turnover)
    benchmark_metrics = _return_metrics(
        underlying_return,
        pd.Series(1.0, index=segment_frame.index),
        pd.Series(0.0, index=segment_frame.index),
    )
    positions = pd.DataFrame(
        {
            "datetime": pd.to_datetime(segment_frame["datetime"]).dt.normalize(),
            "instrument": segment_frame["instrument"].astype(str).to_numpy(),
            "segment": segment_name,
            "candidate": str(policy["candidate"]),
            "model": str(policy["model"]),
            "policy_id": str(policy["policy_id"]),
            "relative_alpha_probability": relative_alpha.to_numpy(),
            "underlying_positive_probability": underlying_positive.to_numpy(),
            "tail_risk_probability": tail_risk.to_numpy(),
            "weight_soxx": soxx_weight.to_numpy(),
            "weight_soxl": soxl_weight.to_numpy(),
            "weight_cash": cash_weight.to_numpy(),
            "turnover": etf_turnover.to_numpy(),
            "strategy_return": strategy_return.to_numpy(),
            ETF_RETURN_COL: etf_return.to_numpy(),
            UNDERLYING_RETURN_COL: underlying_return.to_numpy(),
        }
    )
    return {
        "selection_return_pct": strategy_metrics["return_pct"],
        "selection_max_drawdown_pct": strategy_metrics["max_drawdown_pct"],
        "selection_total_turnover": strategy_metrics["total_turnover"],
        "selection_avg_equity_weight": strategy_metrics["avg_abs_position"],
        "selection_avg_soxl_weight": float(soxl_weight.mean()) if len(soxl_weight) else 0.0,
        "selection_avg_soxx_weight": float(soxx_weight.mean()) if len(soxx_weight) else 0.0,
        "selection_avg_cash_weight": float(cash_weight.mean()) if len(cash_weight) else 0.0,
        "max_soxl_weight": float(soxl_weight.max()) if len(soxl_weight) else 0.0,
        "tail_cash_day_ratio": float(tail_mask.mean()) if len(tail_mask) else 0.0,
        "benchmark_return_pct": benchmark_metrics["return_pct"],
        "benchmark_max_drawdown_pct": benchmark_metrics["max_drawdown_pct"],
        "excess_vs_underlying_pct": float(strategy_metrics["return_pct"] - benchmark_metrics["return_pct"]),
    }, positions


def _select_dynamic_allocation_candidate(
    metrics: pd.DataFrame,
    *,
    outer_test_year: int,
    required_proxy_assets: tuple[str, ...] = PROXY_ASSETS,
) -> dict[str, Any]:
    if metrics.empty:
        raise ValueError("dynamic allocation inner nested walk-forward produced no candidate metrics")
    rows = metrics.copy()
    for column in [
        "selection_return_pct",
        "selection_max_drawdown_pct",
        "selection_total_turnover",
        "selection_avg_equity_weight",
        "selection_avg_soxl_weight",
        "benchmark_return_pct",
        "excess_vs_underlying_pct",
    ]:
        rows[column] = pd.to_numeric(rows.get(column, 0.0), errors="coerce").fillna(0.0)
    rows["passes_soxx_goal"] = rows["excess_vs_underlying_pct"].ge(20.0) & rows["selection_max_drawdown_pct"].ge(-10.0)
    rows["passes_proxy_generalization"] = rows["excess_vs_underlying_pct"].ge(0.0) & rows["selection_max_drawdown_pct"].ge(-20.0)
    rows["is_calibration_asset"] = rows["heldout_asset"].astype(str).eq(CALIBRATION_ASSET)
    if "feature_count" not in rows.columns:
        rows["feature_count"] = 0
    if "contains_soxx_constituents" not in rows.columns:
        rows["contains_soxx_constituents"] = False
    if "soxx_constituent_feature_count" not in rows.columns:
        rows["soxx_constituent_feature_count"] = 0
    rows["feature_count"] = pd.to_numeric(rows["feature_count"], errors="coerce").fillna(0).astype(int)
    rows["contains_soxx_constituents"] = rows["contains_soxx_constituents"].fillna(False).astype(bool)
    rows["soxx_constituent_feature_count"] = (
        pd.to_numeric(rows["soxx_constituent_feature_count"], errors="coerce").fillna(0).astype(int)
    )
    group_keys = ["candidate", "feature_set"]
    grouped = (
        rows.groupby(group_keys)
        .agg(
            model=("model", "first"),
            policy_id=("policy_id", "first"),
            feature_count=("feature_count", "first"),
            contains_soxx_constituents=("contains_soxx_constituents", "first"),
            soxx_constituent_feature_count=("soxx_constituent_feature_count", "first"),
            avg_selection_return_pct=("selection_return_pct", "mean"),
            avg_excess_vs_underlying_pct=("excess_vs_underlying_pct", "mean"),
            worst_selection_max_drawdown_pct=("selection_max_drawdown_pct", "min"),
            avg_selection_total_turnover=("selection_total_turnover", "mean"),
            avg_equity_weight=("selection_avg_equity_weight", "mean"),
            avg_soxl_weight=("selection_avg_soxl_weight", "mean"),
            heldout_assets=("heldout_asset", lambda values: sorted({str(value) for value in values})),
            inner_fold_count=("candidate", "size"),
        )
        .reset_index()
    )
    calibration = (
        rows.loc[rows["is_calibration_asset"]]
        .groupby(group_keys)
        .agg(
            soxl_inner_excess_vs_soxx_pct=("excess_vs_underlying_pct", "mean"),
            soxl_inner_return_pct=("selection_return_pct", "mean"),
            soxl_inner_worst_drawdown_pct=("selection_max_drawdown_pct", "min"),
            soxl_inner_goal_pass_rate=("passes_soxx_goal", "mean"),
            soxl_inner_fold_count=("candidate", "size"),
        )
        .reset_index()
    )
    proxy = (
        rows.loc[~rows["is_calibration_asset"]]
        .groupby(group_keys)
        .agg(
            proxy_avg_excess_pct=("excess_vs_underlying_pct", "mean"),
            proxy_worst_drawdown_pct=("selection_max_drawdown_pct", "min"),
            proxy_generalization_pass_rate=("passes_proxy_generalization", "mean"),
            proxy_fold_count=("candidate", "size"),
            proxy_assets=("heldout_asset", lambda values: sorted({str(value) for value in values})),
        )
        .reset_index()
    )
    grouped = grouped.merge(calibration, on=group_keys, how="left").merge(proxy, on=group_keys, how="left")
    required = [str(asset) for asset in required_proxy_assets]
    grouped["has_required_proxy_assets"] = grouped["proxy_assets"].map(
        lambda assets: all(asset in assets for asset in required) if isinstance(assets, list) else False
    )
    grouped["has_calibration_asset"] = grouped["soxl_inner_fold_count"].fillna(0).gt(0)
    eligible = grouped.loc[grouped["has_calibration_asset"] & grouped["has_required_proxy_assets"]].copy()
    if eligible.empty:
        raise ValueError("dynamic allocation selector missing SOXL calibration or proxy assets")
    eligible["soxl_gate_pass"] = (
        eligible["soxl_inner_excess_vs_soxx_pct"].fillna(-1e9).ge(5.0)
        & eligible["soxl_inner_worst_drawdown_pct"].fillna(-100.0).ge(-15.0)
    )
    eligible["proxy_generalization_gate_pass"] = (
        eligible["proxy_avg_excess_pct"].fillna(-1e9).ge(0.0)
        & eligible["proxy_generalization_pass_rate"].fillna(0.0).ge(0.5)
    )
    eligible["selection_score"] = (
        2.50 * eligible["soxl_inner_excess_vs_soxx_pct"].fillna(0.0)
        + 1.50 * eligible["proxy_avg_excess_pct"].fillna(0.0)
        + 70.0 * eligible["proxy_generalization_pass_rate"].fillna(0.0)
        + 40.0 * eligible["soxl_inner_goal_pass_rate"].fillna(0.0)
        + 2.00 * eligible["soxl_inner_worst_drawdown_pct"].fillna(-100.0)
        + 0.50 * eligible["proxy_worst_drawdown_pct"].fillna(-100.0)
        - 0.05 * eligible["avg_selection_total_turnover"].fillna(0.0)
    )
    constrained = eligible.loc[eligible["soxl_gate_pass"] & eligible["proxy_generalization_gate_pass"]].copy()
    generalization_selection_satisfied = not constrained.empty
    if generalization_selection_satisfied:
        selection_pool = constrained
        gate_status = "pass"
    elif eligible["proxy_generalization_gate_pass"].fillna(False).any():
        selection_pool = eligible
        gate_status = "diagnostic_fallback_soxl_gate_failed"
    else:
        selection_pool = eligible
        gate_status = "diagnostic_fallback_no_proxy_generalization_candidate"
    selected = selection_pool.sort_values(
        [
            "selection_score",
            "proxy_generalization_gate_pass",
            "soxl_inner_excess_vs_soxx_pct",
            "proxy_avg_excess_pct",
            "soxl_inner_worst_drawdown_pct",
        ],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    return {
        "candidate": str(selected["candidate"]),
        "feature_set": str(selected["feature_set"]),
        "outer_test_year": int(outer_test_year),
        "selection_policy": SELECTION_POLICY,
        "selection_uses_outer_test": False,
        "model": str(selected["model"]),
        "policy_id": str(selected["policy_id"]),
        "feature_count": int(selected.get("feature_count", 0)),
        "contains_soxx_constituents": bool(selected.get("contains_soxx_constituents", False)),
        "soxx_constituent_feature_count": int(selected.get("soxx_constituent_feature_count", 0)),
        "generalization_selection_satisfied": bool(generalization_selection_satisfied),
        "selection_gate_status": gate_status,
        "proxy_generalization_gate_applied": True,
        "required_proxy_assets": required,
        "observed_proxy_assets": list(selected["proxy_assets"]),
        "inner_fold_count": int(selected["inner_fold_count"]),
        "soxl_inner_fold_count": int(selected["soxl_inner_fold_count"]),
        "proxy_fold_count": int(selected["proxy_fold_count"]),
        "soxl_inner_excess_vs_soxx_pct": float(selected["soxl_inner_excess_vs_soxx_pct"]),
        "soxl_inner_return_pct": float(selected["soxl_inner_return_pct"]),
        "soxl_inner_worst_drawdown_pct": float(selected["soxl_inner_worst_drawdown_pct"]),
        "soxl_inner_goal_pass_rate": float(selected["soxl_inner_goal_pass_rate"]),
        "proxy_avg_excess_pct": float(selected["proxy_avg_excess_pct"]),
        "proxy_worst_drawdown_pct": float(selected["proxy_worst_drawdown_pct"]),
        "proxy_generalization_pass_rate": float(selected["proxy_generalization_pass_rate"]),
        "avg_selection_total_turnover": float(selected["avg_selection_total_turnover"]),
        "avg_equity_weight": float(selected["avg_equity_weight"]),
        "avg_soxl_weight": float(selected["avg_soxl_weight"]),
        "selection_score": float(selected["selection_score"]),
    }


def _annual_goal_summary(
    annual_metrics: pd.DataFrame,
    *,
    required_years: tuple[int, ...],
    excess_return_floor_pct: float,
    max_drawdown_floor_pct: float,
) -> dict[str, Any]:
    required = [int(year) for year in required_years]
    if annual_metrics.empty:
        feasible: list[int] = []
        pass_years: list[int] = []
    else:
        feasible = sorted(int(year) for year in annual_metrics["outer_test_year"].dropna().astype(int).unique())
        pass_years = sorted(
            int(year)
            for year in annual_metrics.loc[
                pd.to_numeric(annual_metrics["test_excess_vs_soxx_pct"], errors="coerce").ge(
                    float(excess_return_floor_pct)
                )
                & pd.to_numeric(annual_metrics["test_max_drawdown_pct"], errors="coerce").ge(
                    float(max_drawdown_floor_pct)
                ),
                "outer_test_year",
            ]
            .dropna()
            .astype(int)
            .unique()
        )
    missing = [year for year in required if year not in feasible]
    failed = [year for year in required if year in feasible and year not in pass_years]
    blocking = []
    if missing:
        blocking.append("missing_required_test_years")
    if failed:
        blocking.append("annual_excess_or_drawdown_gate_failed")
    return {
        "policy": "annual_soxx_benchmark_excess_and_drawdown_gate",
        "selection_uses_outer_test": False,
        "required_test_years": required,
        "feasible_test_years": feasible,
        "target_pass_test_years": pass_years,
        "failed_test_years": failed,
        "missing_required_test_years": missing,
        "return_gate": "test_return_pct_ge_soxx_buyhold_return_pct_plus_20pp",
        "excess_return_floor_pct": float(excess_return_floor_pct),
        "max_drawdown_floor_pct": float(max_drawdown_floor_pct),
        "required_policy_satisfied": bool(not missing and not failed),
        "blocking_reasons": blocking or ["ok"],
    }


def _numeric(frame: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    values = frame.get(column, pd.Series(default, index=frame.index))
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default))


def _attach_dynamic_allocation_features(raw_panel: pd.DataFrame) -> pd.DataFrame:
    out = _attach_normalized_features(raw_panel)
    pieces: list[pd.DataFrame] = []
    for _, group in out.sort_values(["instrument", "datetime"]).groupby("instrument", sort=False):
        g = group.copy().sort_values("datetime").reset_index(drop=True)
        etf_open = pd.to_numeric(g["open"], errors="coerce").astype("float64")
        underlying_open = pd.to_numeric(g["underlying_open"], errors="coerce").astype("float64")
        etf_close = pd.to_numeric(g["close"], errors="coerce").astype("float64")
        underlying_close = pd.to_numeric(g["underlying_close"], errors="coerce").astype("float64")
        etf_daily = etf_close.pct_change()
        underlying_daily = underlying_close.pct_change()
        g[ETF_RETURN_COL] = etf_open.shift(-2) / etf_open.shift(-1) - 1.0
        g[UNDERLYING_RETURN_COL] = underlying_open.shift(-2) / underlying_open.shift(-1) - 1.0
        etf_forward_5d = etf_open.shift(-6) / etf_open.shift(-1) - 1.0
        underlying_forward_5d = underlying_open.shift(-6) / underlying_open.shift(-1) - 1.0
        g["etf_open_to_open_5d_forward_return"] = etf_forward_5d
        g["underlying_open_to_open_5d_forward_return"] = underlying_forward_5d
        g[RELATIVE_EXCESS_5D_COL] = etf_forward_5d - underlying_forward_5d
        g[RELATIVE_ALPHA_LABEL_COL] = np.where(g[RELATIVE_EXCESS_5D_COL].notna(), g[RELATIVE_EXCESS_5D_COL].gt(0.02), np.nan)
        g[UNDERLYING_POSITIVE_LABEL_COL] = np.where(underlying_forward_5d.notna(), underlying_forward_5d.gt(0.0), np.nan)
        etf_adverse = _future_open_path_min_adverse(g, open_col="open", offsets=range(2, 7))
        underlying_adverse = _future_open_path_min_adverse(g, open_col="underlying_open", offsets=range(2, 7))
        g["etf_min_adverse_open_to_open_5d"] = etf_adverse
        g["underlying_min_adverse_open_to_open_5d"] = underlying_adverse
        tail_known = etf_adverse.notna() & underlying_adverse.notna()
        g[TAIL_RISK_LABEL_COL] = np.where(tail_known, etf_adverse.le(-0.08) | underlying_adverse.le(-0.04), np.nan)
        g["underlying_ret_5"] = underlying_close / underlying_close.shift(5) - 1.0
        g["underlying_realized_vol_20"] = underlying_daily.rolling(20, min_periods=10).std(ddof=0)
        g["underlying_drawdown_60"] = underlying_close / underlying_close.rolling(60, min_periods=20).max() - 1.0
        g["relative_momentum_20"] = pd.to_numeric(g.get("ret_20", 0.0), errors="coerce") - pd.to_numeric(
            g.get("underlying_ret_20", 0.0),
            errors="coerce",
        )
        g["relative_momentum_60"] = pd.to_numeric(g.get("ret_60", 0.0), errors="coerce") - pd.to_numeric(
            g.get("underlying_ret_60", 0.0),
            errors="coerce",
        )
        g["volatility_ratio_20"] = pd.to_numeric(g.get("realized_vol_20", 0.0), errors="coerce") / (
            g["underlying_realized_vol_20"].replace(0.0, np.nan)
        )
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True).sort_values(["instrument", "datetime"]).reset_index(drop=True)


def _future_open_path_min_adverse(frame: pd.DataFrame, *, open_col: str, offsets: range) -> pd.Series:
    open_ = pd.to_numeric(frame[open_col], errors="coerce").astype("float64")
    entry = open_.shift(-1)
    future_path = pd.concat([(open_.shift(-offset) / entry - 1.0) for offset in offsets], axis=1)
    return future_path.min(axis=1)


def _allocation_feature_columns(frame: pd.DataFrame) -> list[str]:
    base = [
        column
        for column in _model_feature_columns(frame)
        if column in frame.columns and not column.startswith("soxx_const_")
    ]
    extra = [
        "underlying_ret_5",
        "underlying_realized_vol_20",
        "underlying_drawdown_60",
        "relative_momentum_20",
        "relative_momentum_60",
        "volatility_ratio_20",
        INTRADAY_RISK_STATE_COL,
        "intraday_risk_score",
    ]
    soxx = [
        column
        for column in frame.columns
        if column.startswith("soxx_const_") and ("_z" in column or "_pctile" in column)
    ]
    return [
        column
        for column in dict.fromkeys([*base, *extra, *soxx])
        if column in frame.columns and column not in {ETF_RETURN_COL, UNDERLYING_RETURN_COL}
    ]


def _allocation_feature_set_configs(frame: pd.DataFrame) -> list[dict[str, Any]]:
    feature_cols = _allocation_feature_columns(frame)
    base_cols = [column for column in feature_cols if not column.startswith("soxx_const_")]
    soxx_cols = [column for column in feature_cols if column.startswith("soxx_const_")]
    configs = [_feature_set_config("base", base_cols, [])]
    if soxx_cols:
        configs.append(_feature_set_config("base_soxx_diffusion", [*base_cols, *soxx_cols], soxx_cols))
    return configs


def _feature_set_config(feature_set: str, feature_cols: list[str], soxx_cols: list[str]) -> dict[str, Any]:
    cols = list(dict.fromkeys(feature_cols))
    return {
        "feature_set": feature_set,
        "feature_cols": cols,
        "feature_count": int(len(cols)),
        "contains_soxx_constituents": bool(soxx_cols),
        "soxx_constituent_feature_count": int(len(soxx_cols)),
        "soxx_constituent_feature_cols": list(soxx_cols),
        "contains_asset_identity": any(
            "instrument" in column.lower() or "symbol" in column.lower() or "asset" in column.lower()
            for column in cols
        ),
        "normalization": "per_asset_rolling_zscore_percentile_plus_pair_features",
    }


def _build_allocation_provider(
    spec: DynamicSoxxSoxlCashAllocationWfSpec,
    frame: pd.DataFrame,
    *,
    as_of: str,
) -> dict[str, Any]:
    store_cols = [
        "instrument",
        "datetime",
        "open",
        "close",
        "volume",
        "underlying_close",
        "vix",
        ETF_RETURN_COL,
        UNDERLYING_RETURN_COL,
        RELATIVE_ALPHA_LABEL_COL,
        UNDERLYING_POSITIVE_LABEL_COL,
        TAIL_RISK_LABEL_COL,
        RELATIVE_EXCESS_5D_COL,
        *_allocation_feature_columns(frame),
    ]
    store = frame.loc[:, [column for column in dict.fromkeys(store_cols) if column in frame.columns]].copy()
    return build_qlib_store_from_frame(
        store,
        provider_uri=spec.provider_uri,
        market="all",
        region="us",
        freq="day",
        as_of=as_of,
        source="qlib_dynamic_soxx_soxl_cash_allocation_pair_panel",
    )


def _load_allocation_provider(
    spec: DynamicSoxxSoxlCashAllocationWfSpec,
    frame: pd.DataFrame,
    *,
    as_of: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = QlibProviderConfig(provider_uri=spec.provider_uri, market="all", region="us", kernels=1)
    provider_meta = init_qlib_provider(config)
    from qlib.data import D

    fields = [
        f"${ETF_RETURN_COL}",
        f"${UNDERLYING_RETURN_COL}",
        f"${RELATIVE_ALPHA_LABEL_COL}",
        f"${UNDERLYING_POSITIVE_LABEL_COL}",
        f"${TAIL_RISK_LABEL_COL}",
        f"${RELATIVE_EXCESS_5D_COL}",
        *[f"${column}" for column in _allocation_feature_columns(frame)],
    ]
    qframe = D.features(list(PAIR_UNIVERSE), fields, start_time=spec.train_start, end_time=as_of, freq="day")
    out = qframe.reset_index().rename(columns={field: field.removeprefix("$") for field in fields})
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str).str.upper()
    return out.sort_values(["instrument", "datetime"]).reset_index(drop=True), provider_meta


def _clean_allocation_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str).str.upper()
    numeric_cols = [
        *feature_cols,
        ETF_RETURN_COL,
        UNDERLYING_RETURN_COL,
        RELATIVE_ALPHA_LABEL_COL,
        UNDERLYING_POSITIVE_LABEL_COL,
        TAIL_RISK_LABEL_COL,
    ]
    for column in numeric_cols:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=[ETF_RETURN_COL, UNDERLYING_RETURN_COL]).sort_values(["datetime", "instrument"]).reset_index(drop=True)


def _predict_allocation_heads_cache(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    feature_cols: list[str],
    candidates: list[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    requested = {str(candidate["model"]) for candidate in candidates}
    base_requested = set(requested)
    if "lgb_xgb_consensus" in requested:
        base_requested.update({"lgbm", "xgb"})
    cache = {
        model: _predict_base_allocation_heads(train, predict, feature_cols=feature_cols, model=model)
        for model in sorted(base_requested)
        if model in {"lgbm", "xgb"}
    }
    if "lgb_xgb_consensus" in requested:
        cache["lgb_xgb_consensus"] = pd.DataFrame(
            {
                "relative_alpha_probability": (
                    cache["lgbm"]["relative_alpha_probability"].to_numpy()
                    + cache["xgb"]["relative_alpha_probability"].to_numpy()
                )
                / 2.0,
                "underlying_positive_probability": (
                    cache["lgbm"]["underlying_positive_probability"].to_numpy()
                    + cache["xgb"]["underlying_positive_probability"].to_numpy()
                )
                / 2.0,
                "tail_risk_probability": (
                    cache["lgbm"]["tail_risk_probability"].to_numpy()
                    + cache["xgb"]["tail_risk_probability"].to_numpy()
                )
                / 2.0,
            },
            index=predict.index,
        )
    return {model: cache[model].copy() for model in requested}


def _predict_base_allocation_heads(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    feature_cols: list[str],
    model: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "relative_alpha_probability": _fit_predict_probability(
                model,
                train,
                predict,
                feature_cols,
                RELATIVE_ALPHA_LABEL_COL,
            ),
            "underlying_positive_probability": _fit_predict_probability(
                model,
                train,
                predict,
                feature_cols,
                UNDERLYING_POSITIVE_LABEL_COL,
            ),
            "tail_risk_probability": _fit_predict_probability(
                model,
                train,
                predict,
                feature_cols,
                TAIL_RISK_LABEL_COL,
            ),
        },
        index=predict.index,
    )


def _fit_predict_probability(
    model: str,
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.Series:
    fit = train.dropna(subset=[target_col]).copy()
    x_train = fit[feature_cols].astype("float64")
    y_train = pd.to_numeric(fit[target_col], errors="coerce").fillna(0.0).astype(int)
    if y_train.nunique() < 2:
        estimator = make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="constant", constant=int(y_train.iloc[0])))
    else:
        estimator = _build_classifier(model)
    estimator.fit(x_train, y_train)
    probabilities = estimator.predict_proba(predict[feature_cols].astype("float64"))
    classes = list(getattr(estimator[-1], "classes_", []))
    positive_idx = classes.index(1) if 1 in classes else 0
    return pd.Series(probabilities[:, positive_idx], index=predict.index, dtype="float64")


def _build_classifier(model: str):
    if model == "lgbm":
        from lightgbm import LGBMClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=30,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=25,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.4,
                random_state=52,
                verbose=-1,
            ),
        )
    if model == "xgb":
        from xgboost import XGBClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=35,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=53,
                n_jobs=1,
            ),
        )
    raise ValueError(f"unsupported allocation model: {model}")


def _inner_dynamic_allocation_metrics(
    frame: pd.DataFrame,
    *,
    outer_test_year: int,
    candidates: list[dict[str, Any]],
    feature_set_configs: list[dict[str, Any]],
    cost_per_turnover: float,
    max_inner_years: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set in feature_set_configs:
        feature_cols = list(feature_set["feature_cols"])
        data = _clean_allocation_frame(frame, feature_cols)
        available = data.loc[data["datetime"].dt.year < int(outer_test_year)].copy()
        inner_years = [
            year
            for year in sorted(available["datetime"].dt.year.unique())
            if year >= 2015 and year <= int(outer_test_year) - 1
        ][-int(max_inner_years) :]
        for inner_year in inner_years:
            train_end = pd.Timestamp(f"{inner_year - 1}-12-31")
            train = available.loc[available["datetime"].le(train_end)].copy()
            validation = available.loc[available["datetime"].dt.year.eq(inner_year)].copy()
            if len(train) < 300 or len(validation) < 40:
                continue
            prediction_cache = _predict_allocation_heads_cache(
                train,
                validation,
                feature_cols=feature_cols,
                candidates=candidates,
            )
            for candidate in candidates:
                pred = prediction_cache[str(candidate["model"])]
                for heldout_asset, validation_asset in validation.groupby("instrument", sort=False):
                    asset_pred = pred.loc[validation_asset.index]
                    metrics, _ = _simulate_dynamic_allocation_policy(
                        validation_asset,
                        asset_pred,
                        policy=candidate,
                        cost_per_turnover=cost_per_turnover,
                        segment_name="inner_validation",
                    )
                    rows.append(
                        {
                            **metrics,
                            "candidate": str(candidate["candidate"]),
                            "model": str(candidate["model"]),
                            "policy_id": str(candidate["policy_id"]),
                            "feature_set": str(feature_set["feature_set"]),
                            "feature_count": int(feature_set["feature_count"]),
                            "contains_soxx_constituents": bool(feature_set["contains_soxx_constituents"]),
                            "soxx_constituent_feature_count": int(feature_set["soxx_constituent_feature_count"]),
                            "inner_year": int(inner_year),
                            "heldout_asset": str(heldout_asset),
                            "selection_uses_outer_test": False,
                        }
                    )
    return pd.DataFrame(rows)


def _feature_set_for_selected(
    feature_set_configs: list[dict[str, Any]],
    selected_candidate: dict[str, Any],
) -> dict[str, Any]:
    selected = str(selected_candidate.get("feature_set", feature_set_configs[0]["feature_set"]))
    for config in feature_set_configs:
        if str(config["feature_set"]) == selected:
            return config
    raise ValueError(f"selected feature_set not found: {selected}")


def _candidate_for_selected(candidates: list[dict[str, Any]], selected_candidate: dict[str, Any]) -> dict[str, Any]:
    selected = str(selected_candidate["candidate"])
    for candidate in candidates:
        if str(candidate["candidate"]) == selected:
            return candidate
    raise ValueError(f"selected candidate not found: {selected}")


def _outer_soxl_dynamic_allocation_test(
    frame: pd.DataFrame,
    *,
    outer_test_year: int,
    selected_candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
    feature_set_configs: list[dict[str, Any]],
    common_end: str,
    cost_per_turnover: float,
    excess_return_floor_pct: float,
    max_drawdown_floor_pct: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidate = _candidate_for_selected(candidates, selected_candidate)
    feature_set = _feature_set_for_selected(feature_set_configs, selected_candidate)
    feature_cols = list(feature_set["feature_cols"])
    data = _clean_allocation_frame(frame, feature_cols)
    train_end = pd.Timestamp(f"{outer_test_year - 1}-12-31")
    test_start = pd.Timestamp(f"{outer_test_year}-01-01")
    test_end = min(pd.Timestamp(f"{outer_test_year}-12-31"), pd.Timestamp(common_end))
    train = data.loc[data["datetime"].le(train_end)].copy()
    test = data.loc[data["instrument"].eq(CALIBRATION_ASSET) & data["datetime"].between(test_start, test_end)].copy()
    if len(train) < 300 or len(test) < 40:
        raise ValueError(f"outer fold {outer_test_year} lacks enough train/test rows")
    pred = _predict_allocation_heads_cache(train, test, feature_cols=feature_cols, candidates=[candidate])[str(candidate["model"])]
    metrics, positions = _simulate_dynamic_allocation_policy(
        test,
        pred,
        policy=candidate,
        cost_per_turnover=cost_per_turnover,
        segment_name="outer_test",
    )
    row = {
        "outer_test_year": int(outer_test_year),
        "candidate": str(candidate["candidate"]),
        "model": str(candidate["model"]),
        "policy_id": str(candidate["policy_id"]),
        "feature_set": str(feature_set["feature_set"]),
        "feature_count": int(feature_set["feature_count"]),
        "contains_soxx_constituents": bool(feature_set["contains_soxx_constituents"]),
        "soxx_constituent_feature_count": int(feature_set["soxx_constituent_feature_count"]),
        "train_start": pd.Timestamp(train["datetime"].min()).date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "test_start": pd.Timestamp(test["datetime"].min()).date().isoformat(),
        "test_end": pd.Timestamp(test["datetime"].max()).date().isoformat(),
        "test_return_pct": metrics["selection_return_pct"],
        "test_max_drawdown_pct": metrics["selection_max_drawdown_pct"],
        "test_total_turnover": metrics["selection_total_turnover"],
        "test_avg_equity_weight": metrics["selection_avg_equity_weight"],
        "test_avg_soxx_weight": metrics["selection_avg_soxx_weight"],
        "test_avg_soxl_weight": metrics["selection_avg_soxl_weight"],
        "test_avg_cash_weight": metrics["selection_avg_cash_weight"],
        "test_max_soxl_weight": metrics["max_soxl_weight"],
        "test_tail_cash_day_ratio": metrics["tail_cash_day_ratio"],
        "soxx_buyhold_return_pct": metrics["benchmark_return_pct"],
        "soxx_buyhold_max_drawdown_pct": metrics["benchmark_max_drawdown_pct"],
        "test_excess_vs_soxx_pct": metrics["excess_vs_underlying_pct"],
        "passes_excess_gate": bool(metrics["excess_vs_underlying_pct"] >= float(excess_return_floor_pct)),
        "passes_drawdown_gate": bool(metrics["selection_max_drawdown_pct"] >= float(max_drawdown_floor_pct)),
        "test_passes_target": bool(
            metrics["excess_vs_underlying_pct"] >= float(excess_return_floor_pct)
            and metrics["selection_max_drawdown_pct"] >= float(max_drawdown_floor_pct)
        ),
        "selection_uses_outer_test": False,
        "generalization_selection_satisfied": bool(selected_candidate["generalization_selection_satisfied"]),
        "selection_gate_status": str(selected_candidate["selection_gate_status"]),
    }
    return row, positions


def _outer_proxy_generalization_stress(
    frame: pd.DataFrame,
    *,
    outer_test_year: int,
    selected_candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
    feature_set_configs: list[dict[str, Any]],
    common_end: str,
    cost_per_turnover: float,
    proxy_max_drawdown_floor_pct: float,
) -> pd.DataFrame:
    candidate = _candidate_for_selected(candidates, selected_candidate)
    feature_set = _feature_set_for_selected(feature_set_configs, selected_candidate)
    feature_cols = list(feature_set["feature_cols"])
    data = _clean_allocation_frame(frame, feature_cols)
    train_end = pd.Timestamp(f"{outer_test_year - 1}-12-31")
    test_start = pd.Timestamp(f"{outer_test_year}-01-01")
    test_end = min(pd.Timestamp(f"{outer_test_year}-12-31"), pd.Timestamp(common_end))
    train = data.loc[data["datetime"].le(train_end)].copy()
    test = data.loc[data["instrument"].isin(PROXY_ASSETS) & data["datetime"].between(test_start, test_end)].copy()
    rows: list[dict[str, Any]] = []
    if len(train) < 300 or test.empty:
        return pd.DataFrame(rows)
    pred_all = _predict_allocation_heads_cache(train, test, feature_cols=feature_cols, candidates=[candidate])[str(candidate["model"])]
    for asset, asset_frame in test.groupby("instrument", sort=False):
        pred = pred_all.loc[asset_frame.index]
        metrics, _ = _simulate_dynamic_allocation_policy(
            asset_frame,
            pred,
            policy=candidate,
            cost_per_turnover=cost_per_turnover,
            segment_name="outer_proxy_stress",
        )
        rows.append(
            {
                "outer_test_year": int(outer_test_year),
                "heldout_asset": str(asset),
                "candidate": str(candidate["candidate"]),
                "feature_set": str(feature_set["feature_set"]),
                "test_return_pct": metrics["selection_return_pct"],
                "benchmark_return_pct": metrics["benchmark_return_pct"],
                "test_excess_vs_underlying_pct": metrics["excess_vs_underlying_pct"],
                "test_max_drawdown_pct": metrics["selection_max_drawdown_pct"],
                "passes_proxy_generalization": bool(
                    metrics["excess_vs_underlying_pct"] >= 0.0
                    and metrics["selection_max_drawdown_pct"] >= float(proxy_max_drawdown_floor_pct)
                ),
                "selection_uses_outer_test": False,
            }
        )
    return pd.DataFrame(rows)


def _generalization_gate_summary(
    selected_candidates: pd.DataFrame,
    proxy_stress: pd.DataFrame,
    *,
    required_years: tuple[int, ...],
) -> dict[str, Any]:
    required = [int(year) for year in required_years]
    if selected_candidates.empty:
        inner_pass_years: list[int] = []
    else:
        inner_pass_years = sorted(
            int(year)
            for year in selected_candidates.loc[
                selected_candidates["generalization_selection_satisfied"].fillna(False).astype(bool),
                "outer_test_year",
            ]
            .dropna()
            .astype(int)
            .unique()
        )
    if proxy_stress.empty:
        outer_proxy_pass_years: list[int] = []
    else:
        proxy_year = (
            proxy_stress.groupby("outer_test_year")
            .agg(
                proxy_avg_excess_pct=("test_excess_vs_underlying_pct", "mean"),
                proxy_pass_rate=("passes_proxy_generalization", "mean"),
            )
            .reset_index()
        )
        outer_proxy_pass_years = sorted(
            int(year)
            for year in proxy_year.loc[
                proxy_year["proxy_avg_excess_pct"].ge(0.0) & proxy_year["proxy_pass_rate"].ge(0.5),
                "outer_test_year",
            ]
            .dropna()
            .astype(int)
            .unique()
        )
    failed = [
        year
        for year in required
        if year not in inner_pass_years or year not in outer_proxy_pass_years
    ]
    return {
        "policy": "inner_proxy_generalization_gate_plus_outer_proxy_stress_report",
        "selection_uses_outer_test": False,
        "required_test_years": required,
        "inner_generalization_pass_years": inner_pass_years,
        "outer_proxy_stress_pass_years": outer_proxy_pass_years,
        "failed_gate_test_years": failed,
        "required_policy_satisfied": bool(not failed),
        "blocking_reasons": ["ok"] if not failed else ["proxy_generalization_gate_failed"],
    }


def _summary_payload(
    *,
    spec: DynamicSoxxSoxlCashAllocationWfSpec,
    sidecars: dict[str, str],
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
    provider_manifest: dict[str, Any],
    provider_meta: dict[str, Any],
    source_audit: dict[str, Any],
    common_end: str,
    feature_cols: list[str],
    feature_set_configs: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    annual_metrics: pd.DataFrame,
    inner_selection: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    proxy_stress: pd.DataFrame,
    annual_goal: dict[str, Any],
    generalization_gate: dict[str, Any],
    overfit_audit: dict[str, Any],
    intraday_audit: dict[str, Any],
    constituent_audit: dict[str, Any],
    constituent_feature_cols: list[str],
) -> dict[str, Any]:
    recorder_artifacts = [
        f"artifacts/{spec.experiment}/{rel_path}"
        for rel_path in sidecars.values()
        if rel_path != "failure_modes.md"
    ]
    recorder_only = [f"artifacts/{spec.experiment}/positions.csv"]
    promotion_ready = (
        annual_goal["required_policy_satisfied"]
        and generalization_gate["required_policy_satisfied"]
        and overfit_audit["overfit_risk_status"] == "mitigated"
    )
    return {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "workflow_config": "workflow.yaml",
        "primary_notebook": f"{spec.experiment}.ipynb",
        "promotion_decision": (
            "candidate_for_validated_research_dynamic_soxx_soxl_cash_allocation"
            if promotion_ready
            else "not_promoted_dynamic_allocation_goal_or_generalization_gate_failed"
        ),
        "benchmark_basis": "open_to_open",
        "benchmark_asset": "SOXX",
        "allocation_assets": ["SOXX", "SOXL", "cash"],
        "execution_timing": "close_t_signal_next_open_to_following_open",
        "return_col": ETF_RETURN_COL,
        "benchmark_return_col": UNDERLYING_RETURN_COL,
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "provider_manifest": (spec.provider_uri / "qlib_store_manifest.json").as_posix(),
        "provider_meta": provider_meta,
        "provider_build": provider_manifest,
        "source_audit": source_audit,
        "common_data_end": common_end,
        "symbols": list(PAIR_UNIVERSE),
        "proxy_generalization_assets": list(PROXY_ASSETS),
        "feature_source": FEATURE_SOURCE,
        "feature_sets": {
            "selection_policy": "nested_inner_feature_set_selection",
            "available": feature_set_configs,
            "selected_by_year": selected_candidates.to_dict("records") if not selected_candidates.empty else [],
            "selected_feature_sets": (
                sorted({str(value) for value in selected_candidates["feature_set"].dropna().unique()})
                if "feature_set" in selected_candidates
                else []
            ),
            "feature_count": int(len(feature_cols)),
            "feature_cols": feature_cols,
            "contains_asset_identity": any(
                "instrument" in column.lower() or "symbol" in column.lower() or "asset" in column.lower()
                for column in feature_cols
            ),
            "available_soxx_constituent_feature_count": int(
                sum(1 for column in constituent_feature_cols if column in feature_cols)
            ),
            "soxx_constituent_feature_cols": [
                column for column in constituent_feature_cols if column in feature_cols
            ],
            "normalization": "per_asset_rolling_zscore_percentile_plus_pair_features",
        },
        "model_selection": {
            "selection_policy": SELECTION_POLICY,
            "selection_uses_outer_test": False,
            "calibration_asset": CALIBRATION_ASSET,
            "proxy_generalization_assets": list(PROXY_ASSETS),
            "proxy_generalization_role": "primary_selection_gate",
            "inner_validation_years": int(spec.inner_validation_years),
            "search_space_type": "pre_registered_lgb_xgb_dynamic_allocation_policy_grid",
            "available_feature_sets": [str(config["feature_set"]) for config in feature_set_configs],
            "manual_template_selection": False,
            "manual_policy_grid_used": False,
            "model_families": ["lgbm_classifier", "xgb_classifier", "lgb_xgb_consensus"],
            "candidate_count": int(len(candidates)),
            "candidate_feature_set_count": int(len(candidates) * len(feature_set_configs)),
            "selected_by_year": selected_candidates.to_dict("records") if not selected_candidates.empty else [],
        },
        "model_heads": {
            "relative_alpha_probability": RELATIVE_ALPHA_LABEL_COL,
            "underlying_positive_probability": UNDERLYING_POSITIVE_LABEL_COL,
            "tail_risk_probability": TAIL_RISK_LABEL_COL,
        },
        "allocation_policy": {
            "formula": "equity_budget=f(underlying_positive, tail_risk); SOXL sleeve=f(relative_alpha, tail_risk) capped by equity_budget; cash=1-SOXX-SOXL",
            "weight_sum_constraint": "weight_soxx + weight_soxl + weight_cash = 1",
            "max_soxl_weight_candidates": sorted({float(candidate["max_soxl_weight"]) for candidate in candidates}),
            "cash_role": "risk_budget_not_alpha_asset",
        },
        "annual_goal": annual_goal,
        "generalization_gate": generalization_gate,
        "overfit_audit": overfit_audit,
        "annual_summary": annual_metrics.to_dict("records") if not annual_metrics.empty else [],
        "proxy_generalization_summary": _proxy_generalization_summary(proxy_stress),
        "intraday_risk_state": intraday_audit,
        "soxx_constituent_factors": constituent_audit,
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"],
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": recorder_only,
        },
        "recorder_artifacts": recorder_artifacts + recorder_only,
        "recorder_id": recorder_id,
        "recorder_uri": recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.resolve().as_posix(),
        "spec": asdict(spec),
        "row_counts": {
            "annual_walk_forward": int(len(annual_metrics)),
            "inner_selection": int(len(inner_selection)),
            "selected_candidates": int(len(selected_candidates)),
            "generalization_stress": int(len(proxy_stress)),
        },
    }


def _proxy_generalization_summary(proxy_stress: pd.DataFrame) -> dict[str, Any]:
    if proxy_stress.empty:
        return {
            "selection_uses_outer_test": False,
            "proxy_avg_excess_pct": 0.0,
            "proxy_pass_rate": 0.0,
            "proxy_assets": [],
        }
    return {
        "selection_uses_outer_test": False,
        "proxy_avg_excess_pct": float(pd.to_numeric(proxy_stress["test_excess_vs_underlying_pct"], errors="coerce").mean()),
        "proxy_worst_drawdown_pct": float(pd.to_numeric(proxy_stress["test_max_drawdown_pct"], errors="coerce").min()),
        "proxy_pass_rate": float(proxy_stress["passes_proxy_generalization"].fillna(False).astype(bool).mean()),
        "proxy_assets": sorted({str(asset) for asset in proxy_stress["heldout_asset"].dropna().unique()}),
    }


def _write_sidecars(
    card: Path,
    *,
    sidecars: dict[str, str],
    summary: dict[str, Any],
    annual_metrics: pd.DataFrame,
    inner_selection: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    proxy_stress: pd.DataFrame,
) -> None:
    annual_metrics.to_csv(card / sidecars["annual_walk_forward"], index=False)
    inner_selection.to_csv(card / sidecars["inner_selection"], index=False)
    selected_candidates.to_csv(card / sidecars["selected_candidates"], index=False)
    selected_candidates.to_csv(card / sidecars["model_comparison"], index=False)
    proxy_stress.to_csv(card / sidecars["generalization_stress"], index=False)
    pd.DataFrame(
        [
            {
                "check": "annual_goal",
                "status": "pass" if summary["annual_goal"]["required_policy_satisfied"] else "fail",
                "detail": json.dumps(summary["annual_goal"], sort_keys=True),
            },
            {
                "check": "generalization_gate",
                "status": "pass" if summary["generalization_gate"]["required_policy_satisfied"] else "fail",
                "detail": json.dumps(summary["generalization_gate"], sort_keys=True),
            },
            {
                "check": "overfit_audit",
                "status": summary["overfit_audit"]["overfit_risk_status"],
                "detail": json.dumps(summary["overfit_audit"], sort_keys=True),
            },
            {
                "check": "soxx_constituent_factors",
                "status": "pass" if summary["soxx_constituent_factors"].get("status") == "ok" else "fail",
                "detail": json.dumps(summary["soxx_constituent_factors"], sort_keys=True),
            },
            {
                "check": "intraday_risk_state",
                "status": "pass" if summary["intraday_risk_state"].get("status") == "ok" else "fail",
                "detail": json.dumps(summary["intraday_risk_state"], sort_keys=True),
            },
        ]
    ).to_csv(card / sidecars["risk_analysis"], index=False)
    selected_by_feature_set = (
        {
            str(name): sorted(int(year) for year in group["outer_test_year"].dropna().astype(int).unique())
            for name, group in selected_candidates.groupby("feature_set")
        }
        if not selected_candidates.empty and "feature_set" in selected_candidates
        else {}
    )
    pd.DataFrame(
        [
            {
                "feature_set": str(config["feature_set"]),
                "selected_outer_years": selected_by_feature_set.get(str(config["feature_set"]), []),
                "feature_count": int(config["feature_count"]),
                "contains_asset_identity": bool(config["contains_asset_identity"]),
                "contains_soxx_constituents": bool(config["contains_soxx_constituents"]),
                "soxx_constituent_feature_count": int(config["soxx_constituent_feature_count"]),
                "normalization": config["normalization"],
            }
            for config in summary["feature_sets"]["available"]
        ]
    ).to_csv(card / sidecars["feature_sets"], index=False)
    (card / sidecars["analysis_summary"]).write_text(
        json.dumps(
            _json_ready(
                {
                    "annual_goal": summary["annual_goal"],
                    "generalization_gate": summary["generalization_gate"],
                    "proxy_generalization_summary": summary["proxy_generalization_summary"],
                    "model_selection": summary["model_selection"],
                    "model_heads": summary["model_heads"],
                    "allocation_policy": summary["allocation_policy"],
                    "feature_sets": summary["feature_sets"],
                    "annual_summary": summary["annual_summary"],
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (card / sidecars["failure_modes"]).write_text(_failure_modes(summary), encoding="utf-8")


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _workflow_yaml(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"experiment: {summary['experiment']}",
            f"card_type: {summary['card_type']}",
            f"provider_uri: {summary['provider_uri']}",
            "benchmark:",
            "  asset: SOXX",
            "  basis: open_to_open",
            "allocation_assets: [SOXX, SOXL, cash]",
            "model_selection:",
            f"  selection_policy: {SELECTION_POLICY}",
            "  selection_uses_outer_test: false",
            "  proxy_generalization_role: primary_selection_gate",
            f"  inner_validation_years: {summary['model_selection']['inner_validation_years']}",
            f"  candidate_count: {summary['model_selection']['candidate_count']}",
            f"  available_feature_sets: {summary['model_selection']['available_feature_sets']}",
            "annual_goal:",
            f"  required_policy_satisfied: {str(summary['annual_goal']['required_policy_satisfied']).lower()}",
            f"  excess_return_floor_pct: {summary['annual_goal']['excess_return_floor_pct']}",
            f"  max_drawdown_floor_pct: {summary['annual_goal']['max_drawdown_floor_pct']}",
            "generalization_gate:",
            f"  required_policy_satisfied: {str(summary['generalization_gate']['required_policy_satisfied']).lower()}",
            f"  blocking_reasons: {summary['generalization_gate']['blocking_reasons']}",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            "",
        ]
    )


def _conclusion(summary: dict[str, Any]) -> str:
    return (
        "# 中文结论\n\n"
        "- 做了什么：新建 SOXX/SOXL/cash 动态配置 walk-forward card，benchmark 从 SOXL B&H 改为 SOXX，目标为累计收益超额 SOXX 20 个百分点且最大回撤不低于 -10%。\n"
        f"- 核心指标：annual pass years={summary['annual_goal']['target_pass_test_years']}，"
        f"generalization gate={summary['generalization_gate']['required_policy_satisfied']}，"
        f"common_data_end={summary['common_data_end']}。\n"
        "- 模型原理：LGB/XGB/consensus 分别训练 SOXL 相对 SOXX 5D 超额概率、SOXX 5D 正收益概率、5D tail risk 概率；仓位只输出 SOXX、SOXL、cash 三个权重，且权重和为 1。\n"
        f"- 泛化验证：selector 使用 SOXL calibration + {summary['proxy_generalization_assets']} proxy generalization gate；outer proxy stress 不参与选择，只报告结构泛化是否仍为正。\n"
        f"- 特征：基础 3x/underlying rolling zscore/percentile、SOXX 成分股 diffusion 特征 {summary['feature_sets']['available_soxx_constituent_feature_count']} 个，以及 5m/15m intraday risk state；分钟数据只做风险状态，不声明日内 alpha。\n"
        f"- 是否晋升：{summary['promotion_decision']}。\n"
        "- 风险：SOXX 当前成分股历史仍有 survivorship bias；proxy 3x ETF 与半导体并不等价；如果 annual gate 未过，不能靠事后改阈值宣布实盘能力。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
        "- 不构成买卖建议。\n"
    )


def _failure_modes(summary: dict[str, Any]) -> str:
    return (
        "# Failure Modes\n\n"
        "- The model can beat SOXX in SOXL validation and still fail if TQQQ/UPRO/SPXL/TECL proxy generalization is negative.\n"
        "- The annual target is intentionally stricter than SOXX benchmark reporting: excess must be at least 20 percentage points and drawdown must stay above -10%.\n"
        "- Cash lowers drawdown but can also make the strategy miss SOXX trend participation.\n"
        "- SOXX constituent factors are current-holdings backfilled and remain survivorship-biased until point-in-time holdings are available.\n"
        "- 5m/15m intraday data is used only as a next-day risk state; it does not prove intraday trading alpha.\n"
        f"- Current promotion decision: {summary['promotion_decision']}.\n"
    )
