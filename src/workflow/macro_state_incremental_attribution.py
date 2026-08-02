from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig
from src.factor import (
    BROAD_DOLLAR_FEATURES,
    RATE_VOL_FEATURES,
    REAL_RATE_FEATURES,
    build_macro_state_factor_frame,
)
from src.workflow.cognitive_alpha_position import run_cognitive_alpha_position_model
from src.workflow.macro_state_factor_ablation import (
    BASE_GROUP,
    MacroStateFactorAblationSpec,
    _attach_macro_features,
    _ensure_macro_provider,
)
from src.workflow.probability_budget_open2open import (
    _position_spec_for_probability_budget,
    _resolve_root,
)
from src.workflow.risk_hardened_open2open import (
    LABEL_OPEN_TO_OPEN_5D_COL,
    _build_frame_bundle,
    _dynamic_open_to_open_benchmarks,
    _existing_unique,
    _feature_groups,
)


EXPERIMENT = "macro_state_incremental_attribution_v1"
CARD_TYPE = "macro_state_incremental_attribution_research_card"
RECORDER_ARTIFACT_PATH = "macro_state_incremental_attribution"
MACRO_FEATURES = [*REAL_RATE_FEATURES, *BROAD_DOLLAR_FEATURES, *RATE_VOL_FEATURES]
BOOTSTRAP_BLOCK_DAYS = 10
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20260802

FROZEN_POSITION_SPEC: dict[str, Any] = {
    "position_mode": "probability",
    "threshold": 0.45,
    "scale": 0.65,
    "max_position": 0.65,
    "min_position": 0.015,
    "rebalance_threshold": 0.10,
    "max_daily_change": 0.35,
    "min_hold_days": 3,
    "objective_return_weight": 0.5,
    "objective_return_to_drawdown_weight": 4.0,
    "objective_min_avg_abs_position": 0.02,
    "objective_max_drawdown_floor": -20.0,
    "objective_max_daily_loss_floor": -12.0,
    "objective_cvar_5pct_floor": -8.0,
    "objective_worst_5_daily_loss_floor": -6.0,
    "objective_max_underwater_day_ratio": 0.95,
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


@dataclass(frozen=True)
class MacroStateIncrementalAttributionSpec(MacroStateFactorAblationSpec):
    experiment: str = EXPERIMENT


def run_macro_state_incremental_attribution_card(*, root: Path | str | None = None) -> dict[str, Any]:
    return _run_card(MacroStateIncrementalAttributionSpec(root=_resolve_root(root)))


def _frozen_position_model_spec(columns: list[str], *, spec: MacroStateIncrementalAttributionSpec, test):
    position_spec = _position_spec_for_probability_budget(columns, spec=spec, test=test)
    return replace(position_spec, position_specs=[dict(FROZEN_POSITION_SPEC)])


def _selected_positions(result, *, walk_forward: bool = False) -> pd.DataFrame:
    source = result.walk_forward_positions if walk_forward else result.positions
    if source.empty:
        return source.copy()
    selected = source.loc[source["selected"].astype(bool)].copy()
    segment_col = "fold_segment" if walk_forward else "segment"
    return selected.loc[selected[segment_col].eq("test")].copy()


def _paired_block_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    block_days: int = BOOTSTRAP_BLOCK_DAYS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    paired = candidate[["datetime", "strategy_return"]].merge(
        reference[["datetime", "strategy_return"]],
        on="datetime",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("paired bootstrap requires aligned OOS returns")
    candidate_returns = paired["strategy_return_candidate"].to_numpy(dtype=float)
    reference_returns = paired["strategy_return_reference"].to_numpy(dtype=float)
    observed = 100.0 * (
        np.prod(1.0 + candidate_returns) - np.prod(1.0 + reference_returns)
    )
    n_obs = len(paired)
    block_days = max(1, min(int(block_days), n_obs))
    block_count = int(np.ceil(n_obs / block_days))
    rng = np.random.default_rng(seed)
    deltas = np.empty(int(samples), dtype=float)
    offsets = np.arange(block_days)
    for index in range(int(samples)):
        starts = rng.integers(0, n_obs, size=block_count)
        sampled = ((starts[:, None] + offsets[None, :]) % n_obs).reshape(-1)[:n_obs]
        deltas[index] = 100.0 * (
            np.prod(1.0 + candidate_returns[sampled])
            - np.prod(1.0 + reference_returns[sampled])
        )
    return {
        "paired_oos_days": int(n_obs),
        "paired_return_delta_pct": float(observed),
        "bootstrap_ci_low_pct": float(np.quantile(deltas, 0.025)),
        "bootstrap_ci_high_pct": float(np.quantile(deltas, 0.975)),
        "bootstrap_probability_positive": float(np.mean(deltas > 0.0)),
        "bootstrap_samples": int(samples),
        "bootstrap_block_days": int(block_days),
    }


def _metric_row(feature: str, result, baseline_result) -> dict[str, Any]:
    selected = result.summary["selected_candidate"]
    baseline = baseline_result.summary["selected_candidate"]
    validation = selected["validation"]
    test = selected["test"]
    baseline_validation = baseline["validation"]
    baseline_test = baseline["test"]
    paired = _paired_block_bootstrap(_selected_positions(result), _selected_positions(baseline_result))
    folds = result.walk_forward_metrics[["fold", "test_return_pct"]].merge(
        baseline_result.walk_forward_metrics[["fold", "test_return_pct"]],
        on="fold",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    fold_delta = folds["test_return_pct_candidate"] - folds["test_return_pct_baseline"]
    validation_score_delta = float(validation["selection_objective_score"]) - float(
        baseline_validation["selection_objective_score"]
    )
    test_return_delta = float(test["return_pct"]) - float(baseline_test["return_pct"])
    row = {
        "feature": feature,
        "feature_count": int(result.summary["feature_count"]),
        "frozen_candidate": selected["candidate"],
        "validation_selection_objective_score": float(validation["selection_objective_score"]),
        "validation_score_delta": validation_score_delta,
        "validation_return_pct": float(validation["return_pct"]),
        "validation_return_delta_pct": float(validation["return_pct"]) - float(baseline_validation["return_pct"]),
        "test_return_pct": float(test["return_pct"]),
        "test_return_delta_pct": test_return_delta,
        "test_max_drawdown_pct": float(test["max_drawdown_pct"]),
        "test_max_drawdown_delta_pct": float(test["max_drawdown_pct"]) - float(baseline_test["max_drawdown_pct"]),
        "test_total_turnover": float(test["total_turnover"]),
        "test_turnover_delta": float(test["total_turnover"]) - float(baseline_test["total_turnover"]),
        "validation_risk_gate_pass": bool(result.summary["risk_gates"]["selected_validation_passes"]),
        "test_risk_gate_pass": bool(result.summary["risk_gates"]["selected_test_passes"]),
        "walk_forward_positive_delta_folds": int((fold_delta > 0.0).sum()),
        "walk_forward_fold_count": int(len(folds)),
        "walk_forward_mean_return_delta_pct": float(fold_delta.mean()),
        "walk_forward_min_return_delta_pct": float(fold_delta.min()),
        **paired,
    }
    row["strict_incremental_gate_pass"] = bool(
        validation_score_delta > 0.0
        and row["validation_risk_gate_pass"]
        and row["test_risk_gate_pass"]
        and row["walk_forward_positive_delta_folds"] >= 3
        and row["bootstrap_ci_low_pct"] > 0.0
    )
    return row


def _drop_one_row(feature: str, full_result, without_result) -> dict[str, Any]:
    full = full_result.summary["selected_candidate"]
    without = without_result.summary["selected_candidate"]
    paired = _paired_block_bootstrap(_selected_positions(full_result), _selected_positions(without_result))
    folds = full_result.walk_forward_metrics[["fold", "test_return_pct"]].merge(
        without_result.walk_forward_metrics[["fold", "test_return_pct"]],
        on="fold",
        suffixes=("_full", "_without"),
        validate="one_to_one",
    )
    fold_delta = folds["test_return_pct_full"] - folds["test_return_pct_without"]
    return {
        "feature": feature,
        "full_feature_count": int(full_result.summary["feature_count"]),
        "without_feature_count": int(without_result.summary["feature_count"]),
        "validation_score_contribution": float(full["validation"]["selection_objective_score"])
        - float(without["validation"]["selection_objective_score"]),
        "validation_return_contribution_pct": float(full["validation"]["return_pct"])
        - float(without["validation"]["return_pct"]),
        "test_return_contribution_pct": float(full["test"]["return_pct"]) - float(without["test"]["return_pct"]),
        "test_max_drawdown_contribution_pct": float(full["test"]["max_drawdown_pct"])
        - float(without["test"]["max_drawdown_pct"]),
        "walk_forward_positive_contribution_folds": int((fold_delta > 0.0).sum()),
        "walk_forward_fold_count": int(len(folds)),
        "walk_forward_mean_return_contribution_pct": float(fold_delta.mean()),
        **paired,
    }


def _feature_diagnostics(
    frame: pd.DataFrame,
    base: list[str],
    macro: list[str],
    *,
    spec,
    test: tuple[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    datetimes = pd.to_datetime(frame["datetime"])
    train_mask = datetimes.between(*map(pd.Timestamp, spec.train))
    validation_mask = datetimes.between(*map(pd.Timestamp, spec.validation))
    test_mask = datetimes.between(*map(pd.Timestamp, test))
    for feature in macro:
        train = frame.loc[train_mask, [feature, *base]].apply(pd.to_numeric, errors="coerce")
        correlations = train[base].corrwith(train[feature]).abs().dropna()
        top_base = str(correlations.idxmax()) if not correlations.empty else ""
        rows.append(
            {
                "feature": feature,
                "train_missing_ratio": float(frame.loc[train_mask, feature].isna().mean()),
                "validation_pearson_ic": float(frame.loc[validation_mask, feature].corr(frame.loc[validation_mask, LABEL_OPEN_TO_OPEN_5D_COL])),
                "validation_rank_ic": float(frame.loc[validation_mask, feature].corr(frame.loc[validation_mask, LABEL_OPEN_TO_OPEN_5D_COL], method="spearman")),
                "test_pearson_ic": float(frame.loc[test_mask, feature].corr(frame.loc[test_mask, LABEL_OPEN_TO_OPEN_5D_COL])),
                "test_rank_ic": float(frame.loc[test_mask, feature].corr(frame.loc[test_mask, LABEL_OPEN_TO_OPEN_5D_COL], method="spearman")),
                "train_max_abs_base_correlation": float(correlations.max()) if not correlations.empty else None,
                "train_top_correlated_base_feature": top_base,
            }
        )
    return pd.DataFrame(rows)


def _tag_positions(result, *, run_type: str, feature: str, walk_forward: bool = False) -> pd.DataFrame:
    frame = _selected_positions(result, walk_forward=walk_forward)
    frame.insert(0, "attribution_feature", feature)
    frame.insert(0, "attribution_run_type", run_type)
    return frame


def _run_card(spec: MacroStateIncrementalAttributionSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    base_bundle = _build_frame_bundle(spec)
    benchmarks = _dynamic_open_to_open_benchmarks(base_bundle.frame, test=base_bundle.test)
    if not benchmarks:
        raise ValueError("incremental attribution requires dynamic open-to-open benchmarks")
    macro_manifest = _ensure_macro_provider(spec, as_of=base_bundle.as_of)
    macro_factors = build_macro_state_factor_frame(
        QlibProviderConfig(provider_uri=spec.macro_state_provider_uri, region="us", market="all", kernels=1),
        start=spec.train[0],
        end=base_bundle.as_of,
    )
    frame = _attach_macro_features(base_bundle.frame, macro_factors)
    bundle = replace(base_bundle, frame=frame)
    base = _existing_unique(_feature_groups(bundle)[BASE_GROUP], frame)
    macro = _existing_unique(MACRO_FEATURES, frame)
    if len(macro) != len(MACRO_FEATURES):
        raise ValueError(f"expected {len(MACRO_FEATURES)} macro features, got {len(macro)}")

    def run(columns: list[str]):
        return run_cognitive_alpha_position_model(
            frame,
            spec=_frozen_position_model_spec(columns, spec=spec, test=bundle.test),
            benchmarks=benchmarks,
        )

    baseline_result = run(base)
    full_result = run([*base, *macro])
    add_results: dict[str, Any] = {}
    add_rows: list[dict[str, Any]] = []
    for feature in macro:
        result = run([*base, feature])
        add_results[feature] = result
        add_rows.append(_metric_row(feature, result, baseline_result))
    add_one = pd.DataFrame(add_rows).sort_values(
        ["strict_incremental_gate_pass", "validation_score_delta", "feature"],
        ascending=[False, False, True],
    )

    drop_results: dict[str, Any] = {}
    drop_rows: list[dict[str, Any]] = []
    for feature in macro:
        result = run([*base, *[item for item in macro if item != feature]])
        drop_results[feature] = result
        drop_rows.append(_drop_one_row(feature, full_result, result))
    drop_one = pd.DataFrame(drop_rows).sort_values(
        ["validation_score_contribution", "feature"], ascending=[False, True]
    )
    diagnostics = _feature_diagnostics(frame, base, macro, spec=spec, test=bundle.test)

    baseline = baseline_result.summary["selected_candidate"]
    full = full_result.summary["selected_candidate"]
    qualified = add_one.loc[add_one["strict_incremental_gate_pass"].astype(bool), "feature"].astype(str).tolist()
    promotion_decision = (
        "candidate_features_for_next_validated_research" if qualified else "not_promoted_no_strict_incremental_feature"
    )
    sidecars = {
        "add_one_attribution": "add_one_attribution.csv",
        "drop_one_attribution": "drop_one_attribution.csv",
        "feature_diagnostics": "feature_diagnostics.csv",
        "failure_modes": "failure_modes.md",
    }

    from qlib.workflow import R

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.as_posix())
    with tempfile.TemporaryDirectory(prefix="macro_state_incremental_attribution_") as tmp:
        tmp_dir = Path(tmp)
        position_frames = [
            _tag_positions(baseline_result, run_type="baseline", feature="baseline"),
            _tag_positions(full_result, run_type="full", feature="all_macro_state"),
        ]
        walk_forward_frames = [
            _tag_positions(baseline_result, run_type="baseline", feature="baseline", walk_forward=True),
            _tag_positions(full_result, run_type="full", feature="all_macro_state", walk_forward=True),
        ]
        for feature, result in add_results.items():
            position_frames.append(_tag_positions(result, run_type="add_one", feature=feature))
            walk_forward_frames.append(_tag_positions(result, run_type="add_one", feature=feature, walk_forward=True))
        for feature, result in drop_results.items():
            position_frames.append(_tag_positions(result, run_type="full_minus_one", feature=feature))
            walk_forward_frames.append(
                _tag_positions(result, run_type="full_minus_one", feature=feature, walk_forward=True)
            )
        positions_path = tmp_dir / "attribution_positions.csv"
        walk_forward_path = tmp_dir / "attribution_walk_forward_positions.csv"
        pd.concat(position_frames, ignore_index=True).to_csv(positions_path, index=False)
        pd.concat(walk_forward_frames, ignore_index=True).to_csv(walk_forward_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=bundle.provider_meta["provider_uri"],
                macro_state_provider_uri="data/qlib/macro_state_v1",
                base_feature_count=len(base),
                macro_feature_count=len(macro),
                frozen_candidate=baseline["candidate"],
                bootstrap_samples=BOOTSTRAP_SAMPLES,
                bootstrap_block_days=BOOTSTRAP_BLOCK_DAYS,
                selection_uses_outer_test=False,
            )
            add_one.to_csv(card / sidecars["add_one_attribution"], index=False)
            drop_one.to_csv(card / sidecars["drop_one_attribution"], index=False)
            diagnostics.to_csv(card / sidecars["feature_diagnostics"], index=False)
            recorder.log_artifact(positions_path.as_posix(), artifact_path=RECORDER_ARTIFACT_PATH)
            recorder.log_artifact(walk_forward_path.as_posix(), artifact_path=RECORDER_ARTIFACT_PATH)
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.suffix != ".md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)

    recorder_local_dir = Path(recorder.get_local_dir()).resolve()
    summary = {
        "status": "complete",
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "as_of": bundle.as_of,
        "provider_uri": "data/qlib/soxl",
        "macro_state_provider_uri": "data/qlib/macro_state_v1",
        "macro_state_availability_policy": macro_manifest.get("availability_policy"),
        "segments": {"train": list(spec.train), "validation": list(spec.validation), "test": list(bundle.test)},
        "base_feature_count": len(base),
        "macro_feature_count": len(macro),
        "frozen_model": {
            "candidate": baseline["candidate"],
            "model": baseline["model"],
            "members": ["lgbm_classifier", "lgbm_regressor", "xgb_classifier", "xgb_regressor"],
            "aggregation": "min",
            "position_spec": FROZEN_POSITION_SPEC,
        },
        "baseline": {
            "validation_selection_objective_score": baseline["validation"]["selection_objective_score"],
            "validation_return_pct": baseline["validation"]["return_pct"],
            "test_return_pct": baseline["test"]["return_pct"],
            "test_max_drawdown_pct": baseline["test"]["max_drawdown_pct"],
        },
        "full_macro": {
            "validation_selection_objective_score": full["validation"]["selection_objective_score"],
            "validation_return_pct": full["validation"]["return_pct"],
            "test_return_pct": full["test"]["return_pct"],
            "test_max_drawdown_pct": full["test"]["max_drawdown_pct"],
        },
        "strict_gate": {
            "definition": "positive validation score delta, validation/test risk gates pass, at least 3 of 4 positive walk-forward return deltas, and paired outer-test 95pct bootstrap CI above zero",
            "qualified_features": qualified,
            "qualified_feature_count": len(qualified),
            "selection_uses_outer_test": False,
            "outer_test_role": "report_and_research_gate_only",
        },
        "promotion_decision": promotion_decision,
        "bootstrap": {
            "method": "paired_circular_moving_block_bootstrap_of_compounded_return_difference",
            "samples": BOOTSTRAP_SAMPLES,
            "block_days": BOOTSTRAP_BLOCK_DAYS,
            "seed": BOOTSTRAP_SEED,
        },
        "sidecars": sidecars,
        "recorder_id": recorder.id,
        "recorder_uri": f"research/{EXPERIMENT}/qlib_recorder",
        "recorder_local_dir": (
            f"research/{EXPERIMENT}/qlib_recorder/{recorder_local_dir.parent.name}/{recorder.id}"
        ),
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": {
                "positions": f"artifacts/{RECORDER_ARTIFACT_PATH}/attribution_positions.csv",
                "walk_forward_positions": f"artifacts/{RECORDER_ARTIFACT_PATH}/attribution_walk_forward_positions.csv",
            },
        },
    }
    _write_summary_files(card, summary, add_one, drop_one)
    return summary


def _write_summary_files(card: Path, summary: dict[str, Any], add_one: pd.DataFrame, drop_one: pd.DataFrame) -> None:
    (card / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")
    (card / "failure_modes.md").write_text(_failure_modes(summary), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion(summary, add_one, drop_one), encoding="utf-8")
    _write_notebook(card, summary)


def _workflow_yaml(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"experiment: {summary['experiment']}",
            f"card_type: {summary['card_type']}",
            f"as_of: {summary['as_of']}",
            "attribution:",
            "  design: frozen_model_add_one_and_full_minus_one",
            "  selection_uses_outer_test: false",
            f"  bootstrap_samples: {summary['bootstrap']['samples']}",
            f"  bootstrap_block_days: {summary['bootstrap']['block_days']}",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            "  machine_entrypoint: summary.json",
            "",
        ]
    )


def _failure_modes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Failure Modes",
            "",
            f"- promotion_decision: {summary['promotion_decision']}",
            "- multiple_testing_limit: 18 add-one and 18 conditional drop-one comparisons are exploratory; no family-wise error correction is claimed.",
            "- short_outer_test_limit: paired confidence intervals use the 2026 YTD outer-test window and remain low-power.",
            "- retraining_limit: model architecture and sizing are frozen, but model weights are retrained for every feature set.",
            "- revision_limit: Treasury curves and the broad-dollar mirror are current-vintage history, not revision-vintage archives.",
            "- interpretation_limit: IC and correlation diagnostics do not establish causal macro effects.",
            "",
        ]
    )


def _conclusion(summary: dict[str, Any], add_one: pd.DataFrame, drop_one: pd.DataFrame) -> str:
    top_add = add_one.sort_values("validation_score_delta", ascending=False).iloc[0]
    top_drop = drop_one.sort_values("validation_score_contribution", ascending=False).iloc[0]
    return (
        "# 结论\n\n"
        f"- what_ran: 冻结 63 因子基线的模型家族和仓位规则，对 {summary['macro_feature_count']} 个宏观特征逐个 add-one，并从全量宏观模型逐个 drop-one。\n"
        f"- period: train={summary['segments']['train']}，validation={summary['segments']['validation']}，outer_test={summary['segments']['test']}，另含 2023/2024/2025/2026YTD 四个 walk-forward 测试折。\n"
        f"- baseline: validation_score={summary['baseline']['validation_selection_objective_score']:.6f}，test_return={summary['baseline']['test_return_pct']:.4f}%，maxDD={summary['baseline']['test_max_drawdown_pct']:.4f}%。\n"
        f"- best_add_one_by_validation: {top_add['feature']}，validation_score_delta={top_add['validation_score_delta']:.6f}，test_return_delta={top_add['test_return_delta_pct']:.4f}%，paired_CI=[{top_add['bootstrap_ci_low_pct']:.4f}%, {top_add['bootstrap_ci_high_pct']:.4f}%]。\n"
        f"- best_conditional_drop_one: {top_drop['feature']}，validation_score_contribution={top_drop['validation_score_contribution']:.6f}，test_return_contribution={top_drop['test_return_contribution_pct']:.4f}%。\n"
        f"- strict_gate: qualified={json.dumps(summary['strict_gate']['qualified_features'], ensure_ascii=False)}。\n"
        f"- promotion_decision: {summary['promotion_decision']}。\n"
        "- risks: 18 次探索存在多重检验问题；2026YTD 配对区间样本较短；模型权重会随特征集重训，因此结论是固定架构/仓位下的预测增量，不是因果效应。\n"
        "- next_action: 只有 strict gate 通过的列才进入独立 validated research；若为空，保持当前 63 因子基线不变。\n"
    )


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8") if (card / "conclusion.md").exists() else ""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "macro-incremental-title",
                "metadata": {},
                "source": [f"# {summary['experiment']}\n", "\n", conclusion],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "id": "macro-incremental-run",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "import qlib\n",
                    "from qlib.data import D\n",
                    "from qlib.data.dataset import DatasetH\n",
                    "from qlib.workflow import R\n",
                    "\n",
                    "ROOT = Path.cwd()\n",
                    "while not (ROOT / 'src').exists() and ROOT.parent != ROOT:\n",
                    "    ROOT = ROOT.parent\n",
                    "if ROOT.as_posix() not in sys.path:\n",
                    "    sys.path.insert(0, ROOT.as_posix())\n",
                    "from src.workflow.macro_state_incremental_attribution import run_macro_state_incremental_attribution_card\n",
                    "summary = run_macro_state_incremental_attribution_card(root=ROOT)\n",
                    "print(json.dumps({'recorder_id': summary['recorder_id'], 'promotion_decision': summary['promotion_decision'], 'qualified_features': summary['strict_gate']['qualified_features']}, indent=2, sort_keys=True))\n",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "macro-incremental-conclusion",
                "metadata": {},
                "source": [conclusion],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (card / f"{summary['experiment']}.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
