from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import QlibProviderConfig, build_macro_state_qlib_store
from src.factor import (
    BROAD_DOLLAR_FEATURES,
    RATE_VOL_FEATURES,
    REAL_RATE_FEATURES,
    build_macro_state_factor_frame,
)
from src.workflow.cognitive_alpha_goal_codex_exec import _attach_open_to_open_execution_context
from src.workflow.cognitive_alpha_position import run_cognitive_alpha_position_model
from src.workflow.probability_budget_open2open import (
    ProbabilityBudgetOpen2OpenCardSpec,
    _position_spec_for_probability_budget,
    _resolve_root,
    _risk_analysis_payload,
    _summary_payload,
    _walk_forward_return_gate_summary,
    _write_sidecars,
)
from src.workflow.risk_hardened_open2open import (
    _build_frame_bundle,
    _dynamic_open_to_open_benchmarks,
    _existing_unique,
    _feature_groups,
)


EXPERIMENT = "macro_state_factor_ablation_v1"
CARD_TYPE = "macro_state_factor_ablation_research_card"
RECORDER_ARTIFACT_PATH = "macro_state_factor_ablation"
BASE_GROUP = "base_cogalpha_soxx"
MACRO_GROUPS = {
    "base_plus_real_rates": list(REAL_RATE_FEATURES),
    "base_plus_broad_dollar": list(BROAD_DOLLAR_FEATURES),
    "base_plus_rate_vol": list(RATE_VOL_FEATURES),
    "base_plus_all_macro_state": [*REAL_RATE_FEATURES, *BROAD_DOLLAR_FEATURES, *RATE_VOL_FEATURES],
}


@dataclass(frozen=True)
class MacroStateFactorAblationSpec(ProbabilityBudgetOpen2OpenCardSpec):
    experiment: str = EXPERIMENT

    @property
    def macro_state_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "macro_state_v1"


def run_macro_state_factor_ablation_card(*, root: Path | str | None = None) -> dict[str, Any]:
    return _run_card(MacroStateFactorAblationSpec(root=_resolve_root(root)))


def _macro_feature_groups(
    base_groups: dict[str, list[str]],
    *,
    frame: pd.DataFrame,
) -> dict[str, list[str]]:
    base = _existing_unique(base_groups[BASE_GROUP], frame)
    groups = {BASE_GROUP: base}
    for name, macro_columns in MACRO_GROUPS.items():
        groups[name] = _existing_unique([*base, *macro_columns], frame)
    return groups


def _attach_macro_features(frame: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    feature_cols = list(factors.attrs.get("feature_cols", []))
    if not feature_cols:
        raise ValueError("macro-state factor frame produced no feature columns")
    left = frame.copy()
    right = factors[["datetime", *feature_cols]].copy()
    left["datetime"] = pd.to_datetime(left["datetime"]).dt.normalize()
    right["datetime"] = pd.to_datetime(right["datetime"]).dt.normalize()
    return left.merge(right, on="datetime", how="left")


def _ablation_row(group: str, result, feature_count: int) -> dict[str, Any]:
    selected = result.summary["selected_candidate"]
    validation = selected["validation"]
    test = selected["test"]
    walk_forward = {
        **result.summary["walk_forward"],
        **_walk_forward_return_gate_summary(result.walk_forward_metrics),
    }
    return {
        "feature_group": group,
        "feature_count": int(feature_count),
        "selected_candidate": selected["candidate"],
        "selected_model": selected["model"],
        "validation_selection_objective_score": float(validation["selection_objective_score"]),
        "validation_return_pct": float(validation["return_pct"]),
        "validation_max_drawdown_pct": float(validation["max_drawdown_pct"]),
        "validation_risk_gate_pass": bool(result.summary["risk_gates"]["selected_validation_passes"]),
        "test_return_pct": float(test["return_pct"]),
        "test_max_drawdown_pct": float(test["max_drawdown_pct"]),
        "test_risk_gate_pass": bool(result.summary["risk_gates"]["selected_test_passes"]),
        "walk_forward_risk_all_pass": bool(walk_forward["all_folds_pass_risk_gates"]),
        "walk_forward_return_all_pass": bool(walk_forward["all_folds_pass_return_gate"]),
        "walk_forward_min_return_pct": float(walk_forward["min_test_return_pct"]),
    }


def _select_feature_group(ablation: pd.DataFrame) -> str:
    """Choose only with validation score and validation gates; never use outer-test metrics."""

    eligible = ablation.loc[ablation["validation_risk_gate_pass"].astype(bool)]
    pool = eligible if not eligible.empty else ablation
    selected = pool.sort_values(
        ["validation_selection_objective_score", "validation_return_pct", "feature_group"],
        ascending=[False, False, True],
    ).iloc[0]
    return str(selected["feature_group"])


def _promotion_decision(selected_group: str, ablation: pd.DataFrame, selected_result) -> str:
    selected_row = ablation.loc[ablation["feature_group"].eq(selected_group)].iloc[0]
    base_row = ablation.loc[ablation["feature_group"].eq(BASE_GROUP)].iloc[0]
    walk_forward = _walk_forward_return_gate_summary(selected_result.walk_forward_metrics)
    if selected_group == BASE_GROUP:
        return "not_promoted_validation_selected_baseline"
    if not bool(selected_row["validation_risk_gate_pass"]):
        return "not_promoted_no_validation_risk_eligible_group"
    if not bool(selected_row["test_risk_gate_pass"]):
        return "not_promoted_failed_outer_test_risk_gates"
    if not bool(walk_forward["all_folds_pass_walk_forward_gates"]):
        return "not_promoted_failed_walk_forward_risk_or_return_gates"
    if float(selected_row["test_return_pct"]) <= float(base_row["test_return_pct"]):
        return "not_promoted_no_outer_test_improvement_over_baseline"
    return "candidate_for_validated_macro_state_research"


def _ensure_macro_provider(spec: MacroStateFactorAblationSpec, *, as_of: str) -> dict[str, Any]:
    manifest_path = spec.macro_state_provider_uri / "qlib_store_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("calendar_end")) >= str(as_of):
            return manifest
    return build_macro_state_qlib_store(root=spec.root, start=spec.train[0], end=as_of)


def _run_card(spec: MacroStateFactorAblationSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    base_bundle = _build_frame_bundle(spec)
    benchmarks = _dynamic_open_to_open_benchmarks(base_bundle.frame, test=base_bundle.test)
    if not benchmarks:
        raise ValueError("macro-state ablation requires dynamic open-to-open benchmarks")
    macro_manifest = _ensure_macro_provider(spec, as_of=base_bundle.as_of)
    macro_factors = build_macro_state_factor_frame(
        QlibProviderConfig(provider_uri=spec.macro_state_provider_uri, region="us", market="all", kernels=1),
        start=spec.train[0],
        end=base_bundle.as_of,
    )
    frame = _attach_macro_features(base_bundle.frame, macro_factors)
    bundle = replace(base_bundle, frame=frame)
    feature_groups = _macro_feature_groups(_feature_groups(bundle), frame=frame)

    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for group, columns in feature_groups.items():
        result = run_cognitive_alpha_position_model(
            frame,
            spec=_position_spec_for_probability_budget(columns, spec=spec, test=bundle.test),
            benchmarks=benchmarks,
        )
        _attach_open_to_open_execution_context(result.summary, result.latest_signal)
        results[group] = result
        rows.append(_ablation_row(group, result, len(columns)))
    ablation = pd.DataFrame(rows)
    selected_group = _select_feature_group(ablation)
    selected_result = results[selected_group]

    sidecars = {
        "selected_model": "selected_model.json",
        "model_selection": "model_selection.csv",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
        "risk_analysis": "risk_analysis.csv",
        "latest_signal": "latest_signal.json",
        "position_summary": "position_summary.json",
        "feature_sets": "feature_sets.csv",
        "macro_ablation": "macro_ablation.csv",
        "failure_modes": "failure_modes.md",
    }

    from qlib.workflow import R

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.as_posix())
    with tempfile.TemporaryDirectory(prefix="macro_state_factor_ablation_") as tmp:
        tmp_dir = Path(tmp)
        positions_path = tmp_dir / "positions.csv"
        walk_forward_path = tmp_dir / "walk_forward_positions.csv"
        selected_result.positions.to_csv(positions_path, index=False)
        selected_result.walk_forward_positions.to_csv(walk_forward_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=bundle.provider_meta["provider_uri"],
                macro_state_provider_uri="data/qlib/macro_state_v1",
                selected_feature_group=selected_group,
                feature_group_selection="validation_only",
                selection_uses_outer_test=False,
                train_start=spec.train[0],
                train_end=spec.train[1],
                validation_start=spec.validation[0],
                validation_end=spec.validation[1],
                test_start=bundle.test[0],
                test_end=bundle.test[1],
            )
            summary = _summary_payload(
                spec=spec,
                bundle=bundle,
                sidecars=sidecars,
                position_result=selected_result,
                feature_groups=feature_groups,
                recorder_uri=recorder_uri,
                recorder_id=recorder.id,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
            )
            summary = _augment_summary(
                summary,
                selected_group=selected_group,
                ablation=ablation,
                macro_manifest=macro_manifest,
                factor_attrs=macro_factors.attrs,
                selected_result=selected_result,
            )
            risk_analysis = _risk_analysis_payload(summary)
            _write_sidecars(
                card,
                sidecars=sidecars,
                summary=summary,
                position_result=selected_result,
                risk_analysis=risk_analysis,
                feature_groups=feature_groups,
            )
            ablation.to_csv(card / sidecars["macro_ablation"], index=False)
            recorder.log_artifact(positions_path.as_posix(), artifact_path=RECORDER_ARTIFACT_PATH)
            recorder.log_artifact(walk_forward_path.as_posix(), artifact_path=RECORDER_ARTIFACT_PATH)
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.name != "failure_modes.md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)

    _write_summary_files(card, summary, risk_analysis)
    return summary


def _augment_summary(
    summary: dict[str, Any],
    *,
    selected_group: str,
    ablation: pd.DataFrame,
    macro_manifest: dict[str, Any],
    factor_attrs: dict[str, Any],
    selected_result,
) -> dict[str, Any]:
    out = dict(summary)
    out["card_type"] = CARD_TYPE
    out["provider_uri"] = "data/qlib/soxl"
    out["recorder_uri"] = f"research/{EXPERIMENT}/qlib_recorder"
    out["recorder_local_dir"] = (
        f"research/{EXPERIMENT}/qlib_recorder/"
        f"{Path(str(summary['recorder_local_dir'])).parent.name}/{summary['recorder_id']}"
    )
    if isinstance(out.get("constituent_factors"), dict):
        out["constituent_factors"] = {
            **out["constituent_factors"],
            "provider_uri": "data/qlib/soxx_constituents",
            "qlib_store_manifest": "data/qlib/soxx_constituents/qlib_store_manifest.json",
            "raw_manifest": "data/raw/soxx_constituents/manifest.json",
        }
    out["spec"] = {**out.get("spec", {}), "root": "."}
    out["promotion_decision"] = _promotion_decision(selected_group, ablation, selected_result)
    out["model_selection"] = {
        **out["model_selection"],
        "source": "qlib_D_features_Alpha158_soxx_constituents_macro_state",
        "selected_feature_group": selected_group,
        "feature_group_selection": "validation_selection_objective_score_with_validation_risk_gate",
        "selection_uses_outer_test": False,
    }
    out["macro_state"] = {
        "source": factor_attrs.get("source"),
        "provider_uri": "data/qlib/macro_state_v1",
        "raw_manifest": macro_manifest.get("raw_manifest"),
        "calendar_start": macro_manifest.get("calendar_start"),
        "calendar_end": macro_manifest.get("calendar_end"),
        "availability_policy": macro_manifest.get("availability_policy"),
        "series": macro_manifest.get("series", {}),
        "feature_groups": factor_attrs.get("feature_groups", {}),
        "feature_count": len(factor_attrs.get("feature_cols", [])),
        "limitations": macro_manifest.get("limitations", []),
    }
    out["macro_ablation"] = ablation.to_dict(orient="records")
    out["artifact_policy"] = {
        **out["artifact_policy"],
        "recorder_only_artifacts": {
            "positions": f"artifacts/{RECORDER_ARTIFACT_PATH}/positions.csv",
            "walk_forward_positions": f"artifacts/{RECORDER_ARTIFACT_PATH}/walk_forward_positions.csv",
        },
    }
    return out


def _write_summary_files(card: Path, summary: dict[str, Any], risk_analysis: pd.DataFrame) -> None:
    (card / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")
    (card / "failure_modes.md").write_text(_failure_modes(summary, risk_analysis), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")
    _write_notebook(card, summary)


def _workflow_yaml(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"experiment: {summary['experiment']}",
            f"card_type: {summary['card_type']}",
            f"as_of: {summary['as_of']}",
            "feature_group_selection:",
            "  source: validation_only",
            "  selection_uses_outer_test: false",
            f"  selected: {summary['model_selection']['selected_feature_group']}",
            "macro_state:",
            f"  provider_uri: {summary['macro_state']['provider_uri']}",
            f"  availability_policy: {summary['macro_state']['availability_policy']}",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            "  machine_entrypoint: summary.json",
            "",
        ]
    )


def _failure_modes(summary: dict[str, Any], risk_analysis: pd.DataFrame) -> str:
    failed = risk_analysis.loc[risk_analysis["status"].eq("fail"), "check"].astype(str).tolist()
    return "\n".join(
        [
            "# Failure Modes",
            "",
            f"- promotion_decision: {summary['promotion_decision']}",
            f"- failed_checks: {json.dumps(failed, ensure_ascii=False)}",
            "- revision_limit: Treasury curves and the broad-dollar mirror are current-vintage history, not revision-vintage archives.",
            "- inflation_proxy_limit: 5Y5Y inflation is derived from Treasury par curves and is not FRED T5YIFR.",
            "- selection_limit: feature groups are selected on validation only; outer-test results are report-only.",
            "- interpretation_limit: macro state factors are model inputs, not standalone trading rules.",
            "",
        ]
    )


def _conclusion(summary: dict[str, Any]) -> str:
    group = summary["model_selection"]["selected_feature_group"]
    row = next(item for item in summary["macro_ablation"] if item["feature_group"] == group)
    baseline = next(item for item in summary["macro_ablation"] if item["feature_group"] == BASE_GROUP)
    return (
        "# 结论\n\n"
        f"- what_ran: 执行 {summary['experiment']}，对实际利率/通胀预期、广义美元、利率波动及全量组合做 qlib 消融。\n"
        f"- selection: 只按验证集选择，selected_group={group}，validation_score={row['validation_selection_objective_score']:.6f}；外层测试未参与选组。\n"
        f"- key_metrics: selected_test_return={row['test_return_pct']:.4f}%，selected_maxDD={row['test_max_drawdown_pct']:.4f}%，"
        f"baseline_test_return={baseline['test_return_pct']:.4f}%。\n"
        f"- walk_forward: risk_all_pass={row['walk_forward_risk_all_pass']}，return_all_pass={row['walk_forward_return_all_pass']}，"
        f"min_return={row['walk_forward_min_return_pct']:.4f}%。\n"
        f"- promotion_decision: {summary['promotion_decision']}。\n"
        "- risks: 5Y5Y 通胀项是财政部名义/实际曲线推导 proxy，并非 FRED T5YIFR；发布映射是保守可用性假设。\n"
        f"- next_action: 仅当非基线组同时通过外层测试和 walk-forward 门时，才进入下一层 validated research；recorder={summary['recorder_id']}。\n"
    )


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "macro-state-title",
                "metadata": {},
                "source": [f"# {summary['experiment']}\n", "\n", _conclusion(summary)],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "id": "macro-state-run",
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
                    "from src.workflow.macro_state_factor_ablation import run_macro_state_factor_ablation_card\n",
                    "summary = run_macro_state_factor_ablation_card(root=ROOT)\n",
                    "print(json.dumps({'recorder_id': summary['recorder_id'], 'promotion_decision': summary['promotion_decision'], 'selected_feature_group': summary['model_selection']['selected_feature_group']}, indent=2, sort_keys=True))\n",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "macro-state-conclusion",
                "metadata": {},
                "source": [_conclusion(summary)],
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
