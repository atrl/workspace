from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.workflow.cognitive_alpha_goal_codex_exec import (
    BASE_CONSTRAINED_REF_COL,
    EXECUTION_TIMING,
    LABEL_OPEN_TO_OPEN_5D_COL,
    OPEN_TO_OPEN_RETURN_COL,
    QLIB_NATIVE_API,
    _attach_open_to_open_execution_context,
)
from src.workflow.cognitive_alpha_position import CognitiveAlphaPositionSpec, run_cognitive_alpha_position_model
from src.workflow.risk_hardened_open2open import (
    _build_frame_bundle,
    _dynamic_open_to_open_benchmarks,
    _feature_groups,
)


EXPERIMENT = "probability_budget_open2open_v1"
CARD_TYPE = "probability_budget_open2open_research_card"


@dataclass(frozen=True)
class ProbabilityBudgetOpen2OpenCardSpec:
    root: Path
    experiment: str = EXPERIMENT
    train: tuple[str, str] = ("2019-01-02", "2024-12-31")
    validation: tuple[str, str] = ("2025-01-02", "2025-12-31")
    test_start: str = "2026-01-02"
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
    min_position: float = 0.015

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def soxx_constituent_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxx_constituents"


def run_probability_budget_open2open_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = ProbabilityBudgetOpen2OpenCardSpec(root=_resolve_root(root))
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


def _position_spec_for_probability_budget(
    feature_cols: list[str],
    *,
    spec: ProbabilityBudgetOpen2OpenCardSpec,
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
            "aggregation": "min",
        },
    ]
    common_objective = {
        "objective_return_weight": 0.5,
        "objective_return_to_drawdown_weight": 4.0,
        "objective_min_avg_abs_position": 0.02,
        "objective_max_drawdown_floor": spec.validation_max_drawdown_floor,
        "objective_max_daily_loss_floor": spec.validation_max_daily_loss_floor,
        "objective_cvar_5pct_floor": spec.validation_cvar_5pct_floor,
        "objective_worst_5_daily_loss_floor": spec.validation_worst_5_daily_loss_floor,
        "objective_max_underwater_day_ratio": spec.max_validation_underwater_day_ratio,
        "exposure_shortfall_penalty": 20.0,
        "drawdown_floor_penalty": 5.0,
        "daily_loss_floor_penalty": 6.0,
        "cvar_floor_penalty": 3.0,
        "worst_5_floor_penalty": 2.0,
        "underwater_penalty": 30.0,
        "turnover_penalty": 1.0,
        "reversal_penalty": 2.0,
        "trade_penalty": 0.2,
    }
    position_specs: list[dict[str, Any]] = []
    for threshold in [0.45, 0.55, 0.65]:
        for scale in [0.35, 0.50, 0.65]:
            for hold in [2, 3]:
                for rebalance in [0.05, 0.10]:
                    position_specs.append(
                        {
                            "position_mode": "probability",
                            "threshold": threshold,
                            "scale": scale,
                            "max_position": scale,
                            "min_position": spec.min_position,
                            "rebalance_threshold": rebalance,
                            "max_daily_change": 0.35,
                            "min_hold_days": hold,
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
        min_validation_avg_abs_position=0.01,
        max_validation_zero_position_day_ratio=0.98,
        validation_max_drawdown_floor=spec.validation_max_drawdown_floor,
        validation_max_daily_loss_floor=spec.validation_max_daily_loss_floor,
        validation_cvar_5pct_floor=spec.validation_cvar_5pct_floor,
        validation_worst_5_daily_loss_floor=spec.validation_worst_5_daily_loss_floor,
        max_validation_underwater_day_ratio=spec.max_validation_underwater_day_ratio,
        max_validation_total_turnover=20.0,
        max_validation_extreme_loss_day_count=7,
        max_validation_reversal_ratio=0.50,
        min_test_avg_abs_position=0.01,
        max_test_zero_position_day_ratio=0.98,
        test_max_drawdown_floor=spec.test_max_drawdown_floor,
        test_max_daily_loss_floor=spec.test_max_daily_loss_floor,
        test_cvar_5pct_floor=spec.test_cvar_5pct_floor,
        test_worst_5_daily_loss_floor=spec.test_worst_5_daily_loss_floor,
        max_test_underwater_day_ratio=spec.max_test_underwater_day_ratio,
        max_test_total_turnover=20.0,
        max_test_extreme_loss_day_count=7,
        constrained_reference_name=BASE_CONSTRAINED_REF_COL,
        extreme_loss_threshold=-0.05,
        walk_forward_folds=[
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
                "fold": "wf_2026_ytd",
                "train": spec.train,
                "validation": spec.validation,
                "test": test,
            },
        ],
    )


def _run_card(spec: ProbabilityBudgetOpen2OpenCardSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    bundle = _build_frame_bundle(spec)
    benchmarks = _dynamic_open_to_open_benchmarks(bundle.frame, test=bundle.test)
    if not benchmarks:
        raise ValueError("probability-budget card requires dynamic open-to-open benchmarks")

    feature_groups = _feature_groups(bundle)
    position_spec = _position_spec_for_probability_budget(
        feature_groups["base_cogalpha_soxx"],
        spec=spec,
        test=bundle.test,
    )
    result = run_cognitive_alpha_position_model(bundle.frame, spec=position_spec, benchmarks=benchmarks)
    _attach_open_to_open_execution_context(result.summary, result.latest_signal)

    sidecars = {
        "selected_model": "selected_model.json",
        "model_selection": "model_selection.csv",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
        "risk_analysis": "risk_analysis.csv",
        "latest_signal": "latest_signal.json",
        "position_summary": "position_summary.json",
        "feature_sets": "feature_sets.csv",
        "failure_modes": "failure_modes.md",
    }

    import qlib
    from qlib.workflow import R

    _ = qlib
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.as_posix())

    with tempfile.TemporaryDirectory(prefix="probability_budget_open2open_") as tmp:
        tmp_dir = Path(tmp)
        positions_path = tmp_dir / "positions.csv"
        walk_forward_positions_path = tmp_dir / "walk_forward_positions.csv"
        result.positions.to_csv(positions_path, index=False)
        result.walk_forward_positions.to_csv(walk_forward_positions_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=bundle.provider_meta["provider_uri"],
                market=bundle.manifest["market"],
                region=bundle.manifest["region"],
                instrument="SOXL",
                train_start=spec.train[0],
                train_end=spec.train[1],
                validation_start=spec.validation[0],
                validation_end=spec.validation[1],
                test_start=bundle.test[0],
                test_end=bundle.test[1],
                return_col=OPEN_TO_OPEN_RETURN_COL,
                label_col=LABEL_OPEN_TO_OPEN_5D_COL,
                model_families="lgbm_classifier,lgbm_regressor,xgb_classifier,xgb_regressor,lgb_xgb_consensus",
                architecture="lgb_xgb_consensus_probability_budget_no_base_boost_no_risk_gate",
                min_position=spec.min_position,
            )
            summary = _summary_payload(
                spec=spec,
                bundle=bundle,
                sidecars=sidecars,
                position_result=result,
                feature_groups=feature_groups,
                recorder_uri=recorder_uri,
                recorder_id=recorder.id,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
            )
            risk_analysis = _risk_analysis_payload(summary)
            _write_sidecars(
                card,
                sidecars=sidecars,
                summary=summary,
                position_result=result,
                risk_analysis=risk_analysis,
                feature_groups=feature_groups,
            )
            recorder.log_artifact(positions_path.as_posix(), artifact_path="probability_budget_open2open")
            recorder.log_artifact(
                walk_forward_positions_path.as_posix(),
                artifact_path="probability_budget_open2open",
            )
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.name != "failure_modes.md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)

    _write_summary_files(card, summary, risk_analysis)
    return summary


def _walk_forward_return_gate_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty:
        return {
            "positive_test_fold_count": 0,
            "positive_test_fold_ratio": None,
            "min_test_return_pct": None,
            "all_folds_pass_return_gate": None,
            "all_folds_pass_walk_forward_gates": None,
        }
    returns = metrics["test_return_pct"].astype(float)
    risk_pass = metrics["risk_gate_pass"].astype(bool)
    return_pass = returns.gt(0.0)
    return {
        "positive_test_fold_count": int(return_pass.sum()),
        "positive_test_fold_ratio": float(return_pass.mean()),
        "min_test_return_pct": float(returns.min()),
        "all_folds_pass_return_gate": bool(return_pass.all()),
        "all_folds_pass_walk_forward_gates": bool(risk_pass.all() and return_pass.all()),
    }


def _summary_payload(
    *,
    spec: ProbabilityBudgetOpen2OpenCardSpec,
    bundle,
    sidecars: dict[str, str],
    position_result,
    feature_groups: dict[str, list[str]],
    recorder_uri: Path,
    recorder_id: str,
    recorder_local_dir: Path,
) -> dict[str, Any]:
    selected = position_result.summary["selected_candidate"]
    walk_forward = {
        **position_result.summary["walk_forward"],
        **_walk_forward_return_gate_summary(position_result.walk_forward_metrics),
    }
    comparison = position_result.summary["comparison"]
    architecture_pass = _architecture_replacement_pass(selected)
    promotion_decision = _promotion_decision(
        selected=selected,
        comparison=comparison,
        risk_gates=position_result.summary["risk_gates"],
        walk_forward=walk_forward,
        architecture_pass=architecture_pass,
    )
    return {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "status": "complete",
        "promotion_decision": promotion_decision,
        "as_of": bundle.as_of,
        "provider_uri": bundle.provider_meta["provider_uri"],
        "recorder_uri": recorder_uri.as_posix(),
        "recorder_id": recorder_id,
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "market": bundle.manifest["market"],
        "region": bundle.manifest["region"],
        "segments": {"train": list(spec.train), "validation": list(spec.validation), "test": list(bundle.test)},
        "benchmark_basis": "open_to_open",
        "execution_timing": EXECUTION_TIMING,
        "return_col": OPEN_TO_OPEN_RETURN_COL,
        "model_target_col": LABEL_OPEN_TO_OPEN_5D_COL,
        "qlib_native_api": QLIB_NATIVE_API,
        "qlib_docs": {
            "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
            "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
        },
        "model_selection": {
            "source": "qlib_D_features_Alpha158_soxx_constituents",
            "model_families": position_result.summary["model_families"],
            "candidate_count": int(position_result.metrics["candidate"].nunique()),
            "selection_metric": "risk_gated_validation_return_to_drawdown_without_base_objective",
            "selected_model": selected["model"],
            "selected_candidate": selected["candidate"],
            "individual_models": "diagnostic_only_selection_disabled",
        },
        "position_model": {
            "workflow": "probability_budget_open2open",
            "architecture": "lgb_xgb_consensus_probability_budget_no_base_boost_no_risk_gate",
            "formula": "target_position = clip(consensus_probability_edge * scale, min_position, max_position)",
            "base_constrained_ref_role": "benchmark_only_not_feature_base_not_objective_not_risk_gate",
            "selected_candidate": selected,
            "comparison": comparison,
            "risk_gates": position_result.summary["risk_gates"],
            "walk_forward": walk_forward,
        },
        "latest_signal": position_result.latest_signal,
        "probability_budget_goal": {
            "target": "replace base_constrained_ref plus probability boost plus risk gate with pure ML probability budget sizing",
            "architecture_replacement_pass": architecture_pass,
            "selected_test_return_pct": selected["test"]["return_pct"],
            "selected_test_max_drawdown_pct": selected["test"]["max_drawdown_pct"],
            "selected_test_max_daily_loss_pct": selected["test"]["max_daily_loss_pct"],
            "selected_vs_base_constrained_ref_return_gap_pct": comparison.get(
                "selected_vs_base_constrained_ref_return_gap_pct"
            ),
            "walk_forward_risk_gate_pass": walk_forward.get("all_folds_pass_risk_gates"),
            "walk_forward_return_gate_pass": walk_forward.get("all_folds_pass_return_gate"),
            "walk_forward_combined_gate_pass": walk_forward.get("all_folds_pass_walk_forward_gates"),
        },
        "feature_sets": {
            name: {"feature_count": len(columns), "features": columns}
            for name, columns in feature_groups.items()
        },
        "constituent_factors": bundle.constituent_factor_summary,
        "benchmark_comparison": position_result.benchmark_comparison.to_dict(orient="records"),
        "walk_forward_metrics": position_result.walk_forward_metrics.to_dict(orient="records"),
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "machine_entrypoint": "summary.json",
            "recorder_only_artifacts": {
                "positions": "artifacts/probability_budget_open2open/positions.csv",
                "walk_forward_positions": "artifacts/probability_budget_open2open/walk_forward_positions.csv",
            },
        },
        "implementation_notes": {
            "lgb_xgb_required": True,
            "lgb_xgb_consensus_selection": selected["model"] == "lgb_xgb_consensus",
            "old_probability_boost_replaced": True,
            "base_constrained_ref_not_in_position_params": "base_position_col"
            not in str(selected["test"].get("params", "")),
            "risk_gate_col_not_in_position_params": "risk_gate_col" not in str(selected["test"].get("params", "")),
        },
        "spec": _json_safe(asdict(spec)),
    }


def _json_safe(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _architecture_replacement_pass(selected: dict[str, Any]) -> bool:
    params = str(selected.get("test", {}).get("params", ""))
    return bool(
        selected.get("model") == "lgb_xgb_consensus"
        and selected.get("test", {}).get("position_mode") == "probability"
        and "base_plus_probability" not in params
        and "base_position_col" not in params
        and "risk_gate_col" not in params
        and "objective_baseline_position_col" not in params
    )


def _promotion_decision(
    *,
    selected: dict[str, Any],
    comparison: dict[str, Any],
    risk_gates: dict[str, Any],
    walk_forward: dict[str, Any],
    architecture_pass: bool,
) -> str:
    if not architecture_pass:
        return "not_promoted_old_architecture_not_replaced"
    if selected.get("model") != "lgb_xgb_consensus":
        return "not_promoted_no_lgb_xgb_consensus"
    if not bool(risk_gates.get("selected_passes_all", False)):
        return "not_promoted_failed_selected_risk_gates"
    if not bool(walk_forward.get("all_folds_pass_walk_forward_gates", False)):
        return "not_promoted_failed_walk_forward_risk_or_return_gates"
    if bool(comparison.get("beats_base_constrained_ref", False)):
        return "candidate_for_validated_research_probability_budget"
    return "candidate_for_walk_forward_validated_probability_budget_under_base_constrained_ref"


def _risk_analysis_payload(summary: dict[str, Any]) -> pd.DataFrame:
    goal = summary["probability_budget_goal"]
    comparison = summary["position_model"]["comparison"]
    return pd.DataFrame(
        [
            {
                "check": "architecture_replaces_base_boost_risk_gate",
                "status": "pass" if goal["architecture_replacement_pass"] else "fail",
                "value": goal["architecture_replacement_pass"],
                "detail": summary["position_model"]["architecture"],
            },
            {
                "check": "lgb_xgb_consensus_selection",
                "status": "pass" if summary["model_selection"]["selected_model"] == "lgb_xgb_consensus" else "fail",
                "value": summary["model_selection"]["selected_model"],
                "detail": "only consensus is selectable; individual LGB/XGB members are diagnostics",
            },
            {
                "check": "selected_position_risk_gates",
                "status": "pass" if summary["position_model"]["risk_gates"]["selected_passes_all"] else "fail",
                "value": summary["position_model"]["risk_gates"]["selected_passes_all"],
                "detail": json.dumps(summary["position_model"]["risk_gates"], sort_keys=True),
            },
            {
                "check": "walk_forward_risk_gates",
                "status": "pass" if goal["walk_forward_risk_gate_pass"] else "fail",
                "value": goal["walk_forward_risk_gate_pass"],
                "detail": json.dumps(summary["position_model"]["walk_forward"], sort_keys=True),
            },
            {
                "check": "walk_forward_return_gate",
                "status": "pass" if goal["walk_forward_return_gate_pass"] else "fail",
                "value": goal["walk_forward_return_gate_pass"],
                "detail": json.dumps(summary["position_model"]["walk_forward"], sort_keys=True),
            },
            {
                "check": "direct_base_comparison",
                "status": "pass" if comparison.get("beats_base_constrained_ref") else "fail",
                "value": comparison.get("selected_vs_base_constrained_ref_return_gap_pct"),
                "detail": json.dumps(comparison, sort_keys=True),
            },
        ]
    )


def _write_sidecars(
    card: Path,
    *,
    sidecars: dict[str, str],
    summary: dict[str, Any],
    position_result,
    risk_analysis: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> None:
    (card / sidecars["selected_model"]).write_text(
        json.dumps(summary["position_model"]["selected_candidate"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    position_result.metrics.to_csv(card / sidecars["model_selection"], index=False)
    position_result.benchmark_comparison.to_csv(card / sidecars["backtest_benchmark_comparison"], index=False)
    position_result.walk_forward_metrics.to_csv(card / sidecars["backtest_walk_forward_metrics"], index=False)
    risk_analysis.to_csv(card / sidecars["risk_analysis"], index=False)
    (card / sidecars["latest_signal"]).write_text(
        json.dumps(position_result.latest_signal, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (card / sidecars["position_summary"]).write_text(
        json.dumps(
            {
                "selected_candidate": summary["position_model"]["selected_candidate"],
                "walk_forward": summary["position_model"]["walk_forward"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"feature_group": name, "feature_count": len(columns), "features": json.dumps(columns)}
            for name, columns in feature_groups.items()
        ]
    ).to_csv(card / sidecars["feature_sets"], index=False)


def _write_summary_files(card: Path, summary: dict[str, Any], risk_analysis: pd.DataFrame) -> None:
    (card / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")
    (card / "failure_modes.md").write_text(_failure_modes(summary, risk_analysis), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")
    _write_notebook(card, summary)


def _workflow_yaml(summary: dict[str, Any]) -> str:
    walk_forward = summary["position_model"]["walk_forward"]
    return "\n".join(
        [
            f"experiment: {summary['experiment']}",
            f"card_type: {summary['card_type']}",
            f"provider_uri: {summary['provider_uri']}",
            f"recorder_uri: {summary['recorder_uri']}",
            f"recorder_id: {summary['recorder_id']}",
            f"as_of: {summary['as_of']}",
            "architecture:",
            "  old_structure: base_constrained_ref + ML probability boost + risk gate",
            "  new_structure: LGB/XGB consensus probability budget without base boost or risk_gate_col",
            f"  formula: {summary['position_model']['formula']}",
            "model_selection:",
            f"  return_col: {summary['return_col']}",
            f"  label_col: {summary['model_target_col']}",
            "  models: [lgbm_classifier, lgbm_regressor, xgb_classifier, xgb_regressor, lgb_xgb_consensus]",
            "  selectable_model: lgb_xgb_consensus",
            "walk_forward:",
            f"  all_risk_gates_pass: {walk_forward.get('all_folds_pass_risk_gates')}",
            f"  all_return_gates_pass: {walk_forward.get('all_folds_pass_return_gate')}",
            f"  min_test_return_pct: {walk_forward.get('min_test_return_pct')}",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            "  machine_entrypoint: summary.json",
            "",
        ]
    )


def _failure_modes(summary: dict[str, Any], risk_analysis: pd.DataFrame) -> str:
    failed = risk_analysis.loc[risk_analysis["status"].eq("fail"), "check"].astype(str).tolist()
    goal = summary["probability_budget_goal"]
    return "\n".join(
        [
            "# Failure Modes",
            "",
            f"- promotion_decision: {summary['promotion_decision']}",
            f"- failed_checks: {json.dumps(failed, ensure_ascii=False)}",
            f"- architecture_replacement_pass: {goal['architecture_replacement_pass']}",
            f"- walk_forward_risk_gate_pass: {goal['walk_forward_risk_gate_pass']}",
            f"- walk_forward_return_gate_pass: {goal['walk_forward_return_gate_pass']}",
            f"- selected_vs_base_constrained_ref_return_gap_pct: {goal['selected_vs_base_constrained_ref_return_gap_pct']}",
            "- primary_limitation: 该替代结构通过年度 walk-forward 风险/收益门，但收益显著低于 base_constrained_ref。",
            "- implementation_limit: min_position 是固定小仓位 floor，不来自 base_constrained_ref；若用于实盘仍需单独 paper 验证。",
            "- constituent_limit: SOXX 成分股数据仍受 point-in-time 覆盖限制。",
            "",
        ]
    )


def _conclusion(summary: dict[str, Any]) -> str:
    selected = summary["position_model"]["selected_candidate"]["test"]
    walk_forward = summary["position_model"]["walk_forward"]
    comparison = summary["position_model"]["comparison"]
    latest = summary["latest_signal"]
    return (
        "# 结论\n\n"
        f"- what_ran: 执行 {summary['experiment']}，用 qlib `D.features`、Alpha158、SOXX 成分股特征和 LGB/XGB consensus，"
        "验证纯 probability budget 仓位。\n"
        f"- key_metrics: selected_return={selected['return_pct']:.4f}%，maxDD={selected['max_drawdown_pct']:.4f}%，"
        f"max_daily_loss={selected['max_daily_loss_pct']:.4f}%，CVaR5={selected.get('cvar_5pct_pct', 0.0):.4f}%。\n"
        f"- walk_forward: folds={walk_forward['fold_count']}，risk_all_pass={walk_forward['all_folds_pass_risk_gates']}，"
        f"return_all_pass={walk_forward['all_folds_pass_return_gate']}，min_return={walk_forward['min_test_return_pct']:.4f}%。\n"
        f"- architecture: 不使用 base_constrained_ref 作为 base，不使用 base_plus_probability，不使用 risk_gate_col；"
        f"base_constrained_ref 只作为 benchmark，对比差距={comparison.get('selected_vs_base_constrained_ref_return_gap_pct'):.4f}%。\n"
        f"- latest_signal: as_of={latest.get('as_of')}，action={latest.get('action')}，"
        f"target_position={latest.get('target_percent', 0.0):.4f}%。\n"
        f"- promotion_decision: {summary['promotion_decision']}。\n"
        "- risks: 该卡解决 walk-forward 风险/收益门，但收益低于 base；不构成买卖建议，实盘前需 paper/sim。\n"
        f"- next_action: 若目标包含跑赢 base，需要在不破坏 walk-forward 门的前提下重做收益目标或引入独立 regime sleeve。recorder={summary['recorder_id']}。\n"
    )


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "probability-budget-title",
                "metadata": {},
                "source": [
                    f"# {summary['experiment']}\n",
                    "\n",
                    "qlib-native SOXL research card replacing base probability boost with an LGB/XGB probability-budget position model.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "id": "probability-budget-run",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "import qlib\n",
                    "from qlib.data import D\n",
                    "from qlib.workflow import R\n",
                    "\n",
                    "ROOT = Path.cwd()\n",
                    "while not (ROOT / 'src').exists() and ROOT.parent != ROOT:\n",
                    "    ROOT = ROOT.parent\n",
                    "if ROOT.as_posix() not in sys.path:\n",
                    "    sys.path.insert(0, ROOT.as_posix())\n",
                    "from src.workflow.probability_budget_open2open import run_probability_budget_open2open_card\n",
                    "\n",
                    "summary = run_probability_budget_open2open_card()\n",
                    "print(json.dumps({'recorder_id': summary['recorder_id'], 'promotion_decision': summary['promotion_decision'], 'as_of': summary['as_of']}, indent=2, sort_keys=True))\n",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "probability-budget-conclusion",
                "metadata": {},
                "source": [_conclusion(summary)],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (card / f"{summary['experiment']}.ipynb").write_text(
        json.dumps(notebook, indent=2, sort_keys=True),
        encoding="utf-8",
    )
