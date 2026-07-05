from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.workflow.cognitive_alpha_goal_codex_exec import (
    BASE_CONSTRAINED_REF_COL,
    LABEL_OPEN_TO_OPEN_5D_COL,
    OPEN_TO_OPEN_RETURN_COL,
    QLIB_NATIVE_API,
    _attach_open_to_open_execution_context,
    _dynamic_open_to_open_benchmarks,
)
from src.workflow.cognitive_alpha_position import CognitiveAlphaPositionSpec, run_cognitive_alpha_position_model
from src.workflow.meta_label_risk_budget_open2open import _build_meta_frame_bundle


EXPERIMENT = "agentic_alpha_hybrid_intraday15m_open2open_v1"
CARD_TYPE = "hybrid_intraday15m_open2open_research_card"
HYBRID_INTRADAY_REGIME_GATE_COL = "hybrid_intraday15m_regime_risk_gate"


@dataclass(frozen=True)
class HybridIntraday15mOpen2OpenSpec:
    root: Path
    experiment: str = EXPERIMENT
    previous_model_return_floor_card: str = "cognitive_alpha_goal_codex_exec_v1"
    train: tuple[str, str] = ("2019-01-02", "2024-12-31")
    validation: tuple[str, str] = ("2025-01-02", "2025-12-31")
    test_start: str = "2026-01-02"
    opportunity_threshold: float = 0.01
    take_label_return_threshold: float = 0.0
    tail_loss_return_threshold: float = -0.05
    validation_max_drawdown_floor: float = -20.0
    validation_max_daily_loss_floor: float = -12.0
    validation_cvar_5pct_floor: float = -8.0
    validation_worst_5_daily_loss_floor: float = -6.0
    max_validation_underwater_day_ratio: float = 0.95
    test_max_drawdown_floor: float = -20.0
    test_max_daily_loss_floor: float = -12.0
    test_cvar_5pct_floor: float = -8.0
    test_worst_5_daily_loss_floor: float = -6.0
    max_test_underwater_day_ratio: float = 0.95

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def soxx_constituent_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxx_constituents"

    @property
    def intraday_15m_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_15min"


def run_hybrid_intraday15m_open2open_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = HybridIntraday15mOpen2OpenSpec(root=_resolve_root(root))
    return _run_card(spec)


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _attach_hybrid_intraday_regime_gate(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    late_ret = _numeric(out, "intraday15m_soxl_late_ret_lag1", default=0.0)
    dislocation = _numeric(out, "intraday15m_soxl_minus_3x_soxx_ret_lag1", default=0.0)
    vol_ratio = _numeric(out, "intraday15m_realized_vol_ratio_soxl_soxx_lag1", default=1.0)
    range_ratio = _numeric(out, "intraday15m_range_ratio_soxl_soxx_lag1", default=1.0)
    breadth = _numeric(out, "soxx_const_breadth_ret5_pos_lag1", default=1.0)
    downside = _numeric(out, "soxx_const_downside_ret5_lt_m5_lag1", default=0.0)
    stress = (
        late_ret.lt(-0.03)
        | dislocation.lt(-0.006)
        | (vol_ratio.gt(3.25) & range_ratio.gt(3.25))
        | (breadth.lt(0.45) & downside.gt(0.25))
    )
    out[HYBRID_INTRADAY_REGIME_GATE_COL] = stress.fillna(False).astype("float64")
    return out


def _position_spec_for_hybrid(
    feature_cols: list[str],
    *,
    spec: HybridIntraday15mOpen2OpenSpec,
    test: tuple[str, str],
) -> CognitiveAlphaPositionSpec:
    model_specs = [
        {"model": "lgbm_classifier", "selection_disabled": True},
        {"model": "lgbm_regressor", "task": "regression", "selection_disabled": True},
        {"model": "xgb_classifier", "selection_disabled": True},
        {"model": "xgb_regressor", "task": "regression", "selection_disabled": True},
        {
            "model": "lgb_xgb_consensus",
            "task": "consensus",
            "members": ["lgbm_classifier", "lgbm_regressor", "xgb_classifier", "xgb_regressor"],
            "aggregation": "mean",
        },
    ]
    common_objective = {
        "objective_baseline_position_col": BASE_CONSTRAINED_REF_COL,
        "objective_excess_return_weight": 0.10,
        "objective_excess_return_to_drawdown_weight": 8.0,
        "objective_return_weight": 0.02,
        "objective_return_to_drawdown_weight": 8.0,
        "objective_min_avg_abs_position": 0.08,
        "objective_max_drawdown_floor": spec.validation_max_drawdown_floor,
        "objective_max_daily_loss_floor": spec.validation_max_daily_loss_floor,
        "objective_cvar_5pct_floor": spec.validation_cvar_5pct_floor,
        "objective_worst_5_daily_loss_floor": spec.validation_worst_5_daily_loss_floor,
        "objective_max_underwater_day_ratio": spec.max_validation_underwater_day_ratio,
        "exposure_shortfall_penalty": 20.0,
        "drawdown_floor_penalty": 2.5,
        "daily_loss_floor_penalty": 3.0,
        "cvar_floor_penalty": 2.0,
        "worst_5_floor_penalty": 1.5,
        "underwater_penalty": 15.0,
        "turnover_penalty": 1.0,
        "reversal_penalty": 1.5,
        "trade_penalty": 0.10,
    }
    position_specs: list[dict[str, Any]] = []
    for low, high, min_throttle, risk_cap in [
        (0.40, 0.68, 0.20, 0.35),
        (0.45, 0.70, 0.35, 0.50),
        (0.50, 0.75, 0.50, 0.65),
    ]:
        position_specs.append(
            {
                "position_mode": "base_throttle",
                "base_position_col": BASE_CONSTRAINED_REF_COL,
                "threshold": 0.50,
                "low_probability": low,
                "high_probability": high,
                "min_throttle": min_throttle,
                "max_throttle": 1.15,
                "scale": 1.0,
                "max_position": 1.0,
                "base_cap": 1.0,
                "rebalance_threshold": 0.05,
                "max_daily_change": 0.50,
                "min_hold_days": 2,
                "risk_gate_col": HYBRID_INTRADAY_REGIME_GATE_COL,
                "risk_gate_threshold": 0.5,
                "risk_gate_direction": "above",
                "risk_gate_max_position": risk_cap,
                **common_objective,
            }
        )
    for threshold, boost, risk_cap, max_daily_change in [
        (0.50, 0.20, 0.35, 0.50),
        (0.55, 0.35, 0.50, 0.50),
        (0.60, 0.50, 0.65, 0.75),
        (0.65, 0.75, 0.75, None),
    ]:
        position_specs.append(
            {
                "position_mode": "base_plus_probability",
                "base_position_col": BASE_CONSTRAINED_REF_COL,
                "threshold": threshold,
                "base_scale": 1.0,
                "boost_scale": boost,
                "scale": 1.0,
                "max_position": 1.0,
                "base_cap": 1.0,
                "rebalance_threshold": 0.05,
                "max_daily_change": max_daily_change,
                "min_hold_days": 3,
                "risk_gate_col": HYBRID_INTRADAY_REGIME_GATE_COL,
                "risk_gate_threshold": 0.5,
                "risk_gate_direction": "above",
                "risk_gate_max_position": risk_cap,
                **common_objective,
            }
        )
    return CognitiveAlphaPositionSpec(
        feature_cols=feature_cols,
        train=spec.train,
        validation=spec.validation,
        test=test,
        model_specs=model_specs,
        position_specs=position_specs,
        return_col=OPEN_TO_OPEN_RETURN_COL,
        label_col=LABEL_OPEN_TO_OPEN_5D_COL,
        min_validation_trade_count=1,
        min_validation_avg_abs_position=0.03,
        max_validation_zero_position_day_ratio=0.98,
        validation_max_drawdown_floor=spec.validation_max_drawdown_floor,
        validation_max_daily_loss_floor=spec.validation_max_daily_loss_floor,
        validation_cvar_5pct_floor=spec.validation_cvar_5pct_floor,
        validation_worst_5_daily_loss_floor=spec.validation_worst_5_daily_loss_floor,
        max_validation_underwater_day_ratio=spec.max_validation_underwater_day_ratio,
        max_validation_total_turnover=35.0,
        max_validation_extreme_loss_day_count=7,
        max_validation_reversal_ratio=0.55,
        min_test_avg_abs_position=0.03,
        max_test_zero_position_day_ratio=0.98,
        test_max_drawdown_floor=spec.test_max_drawdown_floor,
        test_max_daily_loss_floor=spec.test_max_daily_loss_floor,
        test_cvar_5pct_floor=spec.test_cvar_5pct_floor,
        test_worst_5_daily_loss_floor=spec.test_worst_5_daily_loss_floor,
        max_test_underwater_day_ratio=spec.max_test_underwater_day_ratio,
        max_test_total_turnover=45.0,
        max_test_extreme_loss_day_count=7,
        constrained_reference_name=BASE_CONSTRAINED_REF_COL,
        extreme_loss_threshold=-0.05,
        walk_forward_folds=_annual_walk_forward_folds(test_end=test[1]),
    )


def _annual_walk_forward_folds(*, test_end: str) -> list[dict[str, Any]]:
    return [
        {
            "fold": "wf_2023",
            "train": ("2019-01-02", "2021-12-31"),
            "validation": ("2022-01-03", "2022-12-30"),
            "test": ("2023-01-03", "2023-12-29"),
        },
        {
            "fold": "wf_2024",
            "train": ("2019-01-02", "2022-12-30"),
            "validation": ("2023-01-03", "2023-12-29"),
            "test": ("2024-01-02", "2024-12-31"),
        },
        {
            "fold": "wf_2025",
            "train": ("2019-01-02", "2023-12-29"),
            "validation": ("2024-01-02", "2024-12-31"),
            "test": ("2025-01-02", "2025-12-31"),
        },
        {
            "fold": "wf_2026",
            "train": ("2019-01-02", "2024-12-31"),
            "validation": ("2025-01-02", "2025-12-31"),
            "test": ("2026-01-02", test_end),
        },
    ]


def _hybrid_goal_summary(
    walk_forward: pd.DataFrame,
    *,
    required_years: tuple[int, ...] = (2023, 2024, 2025, 2026),
    max_drawdown_floor: float = -20.0,
) -> dict[str, Any]:
    rows = walk_forward.copy()
    if "test_year" not in rows.columns and "test_start" in rows.columns:
        rows["test_year"] = pd.to_datetime(rows["test_start"]).dt.year
    required = [int(year) for year in required_years]
    if rows.empty:
        feasible_years: list[int] = []
        target_pass_years: list[int] = []
    else:
        rows["beats_buyhold"] = rows["test_return_pct"].astype(float) > rows["buyhold_return_pct"].astype(float)
        rows["passes_drawdown"] = rows["test_max_drawdown_pct"].astype(float) >= float(max_drawdown_floor)
        rows["test_passes_target"] = rows["beats_buyhold"].astype(bool) & rows["passes_drawdown"].astype(bool)
        feasible_years = sorted(int(year) for year in rows["test_year"].dropna().astype(int).unique())
        target_pass_years = sorted(
            int(year)
            for year in rows.loc[rows["test_passes_target"].astype(bool), "test_year"].dropna().astype(int).unique()
        )
    missing = [year for year in required if year not in feasible_years]
    blocking_reasons = []
    if missing:
        blocking_reasons.append("missing_required_test_years")
    if target_pass_years != required:
        blocking_reasons.append("annual_target_gate_failed")
    return {
        "policy": "annual_train_to_T_minus_2_valid_T_minus_1_test_T",
        "selection_uses_test_segment": False,
        "required_test_years": required,
        "feasible_test_years": feasible_years,
        "missing_required_test_years": missing,
        "target_pass_test_years": target_pass_years,
        "max_drawdown_floor": float(max_drawdown_floor),
        "return_gate": "test_return_pct_gt_soxl_buyhold_open2open",
        "required_policy_satisfied": bool(not missing and target_pass_years == required),
        "blocking_reasons": blocking_reasons or ["ok"],
    }


def _run_card(spec: HybridIntraday15mOpen2OpenSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    bundle = _build_hybrid_bundle(spec)
    benchmarks = _dynamic_open_to_open_benchmarks(bundle["frame"], test=bundle["test"])
    position_spec = _position_spec_for_hybrid(bundle["feature_cols"], spec=spec, test=bundle["test"])
    position_result = run_cognitive_alpha_position_model(bundle["frame"], spec=position_spec, benchmarks=benchmarks)
    _attach_open_to_open_execution_context(position_result.summary, position_result.latest_signal)
    walk_forward = _attach_buyhold_to_walk_forward(position_result.walk_forward_metrics, bundle["frame"])
    hybrid_goal = _hybrid_goal_summary(walk_forward)

    sidecars = {
        "selected_model": "selected_model.json",
        "model_selection": "model_selection.csv",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
        "risk_analysis": "risk_analysis.csv",
        "analysis_summary": "analysis_summary.json",
        "feature_sets": "feature_sets.csv",
        "latest_signal": "latest_signal.json",
        "failure_modes": "failure_modes.md",
    }

    import qlib
    from qlib.workflow import R

    _ = qlib
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.as_posix())
    with tempfile.TemporaryDirectory(prefix="hybrid_intraday15m_open2open_") as tmp:
        tmp_dir = Path(tmp)
        positions_path = tmp_dir / "positions.csv"
        wf_positions_path = tmp_dir / "walk_forward_positions.csv"
        position_result.positions.to_csv(positions_path, index=False)
        position_result.walk_forward_positions.to_csv(wf_positions_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=bundle["provider_uri"],
                market=bundle["market"],
                region=bundle["region"],
                instrument="SOXL",
                train_start=spec.train[0],
                train_end=spec.train[1],
                validation_start=spec.validation[0],
                validation_end=spec.validation[1],
                test_start=bundle["test"][0],
                test_end=bundle["test"][1],
                return_col=OPEN_TO_OPEN_RETURN_COL,
                label_col=LABEL_OPEN_TO_OPEN_5D_COL,
                intraday_15m_role="lagged_model_features_and_regime_risk_gate",
                model_families="lgbm_classifier,lgbm_regressor,xgb_classifier,xgb_regressor,lgb_xgb_consensus",
            )
            summary = _summary_payload(
                spec=spec,
                bundle=bundle,
                sidecars=sidecars,
                position_result=position_result,
                walk_forward=walk_forward,
                hybrid_goal=hybrid_goal,
                recorder_id=recorder.id,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
                recorder_uri=recorder_uri,
            )
            _write_sidecars(card, sidecars=sidecars, position_result=position_result, walk_forward=walk_forward, summary=summary)
            recorder.log_artifact(positions_path.as_posix(), artifact_path="hybrid_intraday15m_open2open")
            recorder.log_artifact(wf_positions_path.as_posix(), artifact_path="hybrid_intraday15m_open2open")
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.name != "failure_modes.md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
    _write_summary_files(card, summary)
    return summary


def _build_hybrid_bundle(spec: HybridIntraday15mOpen2OpenSpec) -> dict[str, Any]:
    meta_bundle = _build_meta_frame_bundle(spec)  # type: ignore[arg-type]
    frame = _attach_hybrid_intraday_regime_gate(meta_bundle.frame)
    feature_cols = [*meta_bundle.feature_cols, HYBRID_INTRADAY_REGIME_GATE_COL]
    feature_cols = [column for column in dict.fromkeys(feature_cols) if column in frame.columns]
    return {
        "frame": frame,
        "manifest": meta_bundle.manifest,
        "provider_uri": meta_bundle.provider_meta["provider_uri"],
        "market": meta_bundle.manifest["market"],
        "region": meta_bundle.manifest["region"],
        "as_of": meta_bundle.as_of,
        "test": meta_bundle.test,
        "feature_cols": feature_cols,
        "intraday_15m_factors": {
            **meta_bundle.intraday_15m_factor_summary,
            "role": "lagged_model_features_and_regime_risk_gate",
        },
        "constituent_factors": meta_bundle.constituent_factor_summary,
        "feature_groups": {
            **meta_bundle.feature_groups,
            "hybrid_intraday15m_open2open": feature_cols,
        },
    }


def _attach_buyhold_to_walk_forward(walk_forward: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    out = walk_forward.copy()
    if out.empty:
        return out
    frame = frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    frame[OPEN_TO_OPEN_RETURN_COL] = pd.to_numeric(frame[OPEN_TO_OPEN_RETURN_COL], errors="coerce").fillna(0.0)
    buyhold_returns = []
    buyhold_drawdowns = []
    test_years = []
    for _, row in out.iterrows():
        start = pd.Timestamp(row["test_start"])
        end = pd.Timestamp(row["test_end"])
        segment = frame.loc[(frame["datetime"] >= start) & (frame["datetime"] <= end)]
        returns = segment[OPEN_TO_OPEN_RETURN_COL].astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0 if not equity.empty else pd.Series(dtype="float64")
        buyhold_returns.append(float((equity.iloc[-1] - 1.0) * 100.0) if not equity.empty else 0.0)
        buyhold_drawdowns.append(float(drawdown.min() * 100.0) if not drawdown.empty else 0.0)
        test_years.append(int(start.year))
    out["test_year"] = test_years
    out["buyhold_return_pct"] = buyhold_returns
    out["buyhold_max_drawdown_pct"] = buyhold_drawdowns
    summary_rows = _hybrid_goal_summary(out)
    pass_years = set(summary_rows["target_pass_test_years"])
    out["beats_buyhold"] = out["test_return_pct"].astype(float) > out["buyhold_return_pct"].astype(float)
    out["passes_drawdown"] = out["test_max_drawdown_pct"].astype(float) >= -20.0
    out["test_passes_target"] = out["test_year"].astype(int).isin(pass_years)
    return out


def _summary_payload(
    *,
    spec: HybridIntraday15mOpen2OpenSpec,
    bundle: dict[str, Any],
    sidecars: dict[str, str],
    position_result: Any,
    walk_forward: pd.DataFrame,
    hybrid_goal: dict[str, Any],
    recorder_id: str,
    recorder_local_dir: Path,
    recorder_uri: Path,
) -> dict[str, Any]:
    selected = position_result.summary["selected_candidate"]
    return {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "benchmark_basis": "open_to_open",
        "execution_timing": "close_t_signal_next_open_to_following_open",
        "return_col": OPEN_TO_OPEN_RETURN_COL,
        "model_target_col": LABEL_OPEN_TO_OPEN_5D_COL,
        "as_of": bundle["as_of"],
        "provider_uri": bundle["provider_uri"],
        "qlib_docs": {
            "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
            "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
        },
        "qlib_native_api": [*QLIB_NATIVE_API, "LGBMClassifier", "XGBClassifier"],
        "model_selection": {
            "selected_model": selected.get("model"),
            "selection_metric": "validation_risk_adjusted_excess_return_with_intraday15m_gate",
            "selection_uses_test_segment": False,
            "model_families": position_result.summary.get("model_families", []),
        },
        "position_model": position_result.summary,
        "hybrid_goal": hybrid_goal,
        "intraday_15m_factors": bundle["intraday_15m_factors"],
        "constituent_factors": bundle["constituent_factors"],
        "hybrid_regime_gate": {
            "column": HYBRID_INTRADAY_REGIME_GATE_COL,
            "role": "lagged_15m_and_soxx_constituent_risk_cap",
            "lag_days": 1,
            "active_days": int(bundle["frame"][HYBRID_INTRADAY_REGIME_GATE_COL].astype(bool).sum()),
            "coverage_days": int(bundle["frame"][HYBRID_INTRADAY_REGIME_GATE_COL].notna().sum()),
        },
        "feature_sets": {
            "selected": "hybrid_intraday15m_open2open",
            "feature_count": int(len(bundle["feature_cols"])),
            "feature_cols": bundle["feature_cols"],
        },
        "sidecars": sidecars,
        "promotion_decision": (
            "candidate_for_validated_research_hybrid_intraday15m_open2open"
            if hybrid_goal["required_policy_satisfied"]
            else "not_promoted_hybrid_intraday15m_walk_forward_goal_failed"
        ),
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                "artifacts/hybrid_intraday15m_open2open/positions.csv",
                "artifacts/hybrid_intraday15m_open2open/walk_forward_positions.csv",
            ],
        },
        "recorder_id": recorder_id,
        "recorder_uri": recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.resolve().as_posix(),
        "spec": _json_ready(asdict(spec)),
        "walk_forward_rows": int(len(walk_forward)),
    }


def _write_sidecars(
    card: Path,
    *,
    sidecars: dict[str, str],
    position_result: Any,
    walk_forward: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    card.mkdir(parents=True, exist_ok=True)
    position_result.metrics.to_csv(card / sidecars["model_selection"], index=False)
    position_result.benchmark_comparison.to_csv(card / sidecars["backtest_benchmark_comparison"], index=False)
    walk_forward.to_csv(card / sidecars["backtest_walk_forward_metrics"], index=False)
    (card / sidecars["selected_model"]).write_text(
        json.dumps(position_result.summary["selected_candidate"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (card / sidecars["latest_signal"]).write_text(
        json.dumps(position_result.latest_signal, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "check": "annual_walk_forward_target",
                "status": "pass" if summary["hybrid_goal"]["required_policy_satisfied"] else "fail",
                "value": summary["hybrid_goal"]["target_pass_test_years"],
                "detail": json.dumps(summary["hybrid_goal"], sort_keys=True),
            },
            {
                "check": "direct_buyhold_comparison_latest_test",
                "status": "pass" if position_result.summary["comparison"].get("beats_soxl_buyhold") else "fail",
                "value": position_result.summary["comparison"].get("selected_vs_soxl_buyhold_return_gap_pct"),
                "detail": json.dumps(position_result.summary["comparison"], sort_keys=True),
            },
        ]
    ).to_csv(card / sidecars["risk_analysis"], index=False)
    (card / sidecars["analysis_summary"]).write_text(
        json.dumps(
            {
                "hybrid_goal": summary["hybrid_goal"],
                "selected_candidate": position_result.summary["selected_candidate"],
                "comparison": position_result.summary["comparison"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "feature_set": "hybrid_intraday15m_open2open",
                "feature_count": summary["feature_sets"]["feature_count"],
                "contains_intraday15m": any(
                    str(column).startswith("intraday15m_") for column in summary["feature_sets"]["feature_cols"]
                ),
                "contains_soxx_constituents": any(
                    str(column).startswith("soxx_const_") for column in summary["feature_sets"]["feature_cols"]
                ),
            }
        ]
    ).to_csv(card / sidecars["feature_sets"], index=False)
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
            "benchmark_basis: open_to_open",
            f"return_col: {summary['return_col']}",
            f"model_target_col: {summary['model_target_col']}",
            "model_selection:",
            f"  selected_model: {summary['model_selection']['selected_model']}",
            f"  selection_uses_test_segment: {str(summary['model_selection']['selection_uses_test_segment']).lower()}",
            "intraday_15m_factors:",
            f"  provider_uri: {summary['intraday_15m_factors']['provider_uri']}",
            f"  role: {summary['intraday_15m_factors']['role']}",
            f"  lag_days: {summary['intraday_15m_factors']['lag_days']}",
            "hybrid_goal:",
            f"  required_policy_satisfied: {str(summary['hybrid_goal']['required_policy_satisfied']).lower()}",
            f"  target_pass_test_years: {summary['hybrid_goal']['target_pass_test_years']}",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            "",
        ]
    )


def _conclusion(summary: dict[str, Any]) -> str:
    selected = summary["position_model"]["selected_candidate"]["test"]
    return (
        "# 中文结论\n\n"
        "- 做了什么：实现 daily open2open 主仓位 + lagged 15m regime 因子 + SOXX 成分股特征的 LGB/XGB hybrid 年度 walk-forward 卡。\n"
        f"- 核心指标：latest test return_pct={selected['return_pct']:.4f}，max_drawdown_pct={selected['max_drawdown_pct']:.4f}，"
        f"annual target pass years={summary['hybrid_goal']['target_pass_test_years']}。\n"
        "- regime/window 差异：15m 数据只作为 T+1 日级特征和风险 cap，不使用同日未完成 bar，不做独立日内总仓位。\n"
        f"- 是否晋升：{summary['promotion_decision']}。\n"
        "- 风险：SOXX 成分股仍带有持仓快照偏差；15m 因子是 bar 聚合特征，不含 bid/ask 队列成交验证；不构成买卖建议。\n"
        "- 下一步：若年度目标仍失败，优先检查失败年份的买入保持率和 15m risk gate 是否过度削减趋势暴露。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
    )


def _failure_modes(summary: dict[str, Any]) -> str:
    return (
        "# Failure Modes\n\n"
        "- Annual target requires every required fold to beat SOXL open-to-open buyhold and keep max drawdown above -20%.\n"
        "- 15m factors are lagged daily-safe inputs; this card does not prove live same-bar intraday execution.\n"
        "- SOXX constituent features may inherit snapshot or partial-history bias until point-in-time holdings are complete.\n"
        f"- Current blocking reasons: {', '.join(summary['hybrid_goal']['blocking_reasons'])}.\n"
    )


def _numeric(frame: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if pd.isna(value) if not isinstance(value, (str, bytes, bool)) else False:
        return None
    return value
