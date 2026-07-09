from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import QlibProviderConfig, build_hardware_cycle_qlib_store
from src.factor import build_hardware_cycle_factor_frame
from src.workflow.cognitive_alpha_goal_codex_exec import (
    LABEL_OPEN_TO_OPEN_5D_COL,
    OPEN_TO_OPEN_RETURN_COL,
    _attach_open_to_open_execution_context,
)
from src.workflow.cognitive_alpha_position import run_cognitive_alpha_position_model
from src.workflow.probability_budget_open2open import (
    ProbabilityBudgetOpen2OpenCardSpec,
    _failure_modes,
    _position_spec_for_probability_budget,
    _resolve_root,
    _risk_analysis_payload,
    _summary_payload,
    _walk_forward_return_gate_summary,
    _write_sidecars,
)
from src.workflow.risk_hardened_open2open import _build_frame_bundle, _dynamic_open_to_open_benchmarks, _existing_unique, _feature_groups


HARDWARE_EXPERIMENT = "hardware_cycle_probability_budget_open2open_v1"
HARDWARE_CARD_TYPE = "hardware_cycle_probability_budget_open2open_research_card"
HARDWARE_RECORDER_ARTIFACT_PATH = "hardware_cycle_probability_budget_open2open"
HARDWARE_FEATURE_GROUP = "base_cogalpha_soxx_hardware"


@dataclass(frozen=True)
class HardwareCycleProbabilityBudgetOpen2OpenCardSpec(ProbabilityBudgetOpen2OpenCardSpec):
    experiment: str = HARDWARE_EXPERIMENT

    @property
    def hardware_cycle_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "hardware_cycle_v1"


def run_hardware_cycle_probability_budget_open2open_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = HardwareCycleProbabilityBudgetOpen2OpenCardSpec(root=_resolve_root(root))
    return _run_card(spec)


def _hardware_cycle_feature_groups(
    base_groups: dict[str, list[str]],
    *,
    hardware_feature_cols: list[str],
    frame: pd.DataFrame,
) -> dict[str, list[str]]:
    groups = dict(base_groups)
    groups[HARDWARE_FEATURE_GROUP] = _existing_unique(
        [*groups["base_cogalpha_soxx"], *hardware_feature_cols],
        frame,
    )
    return groups


def _attach_hardware_cycle_model_features(
    frame: pd.DataFrame,
    hardware_factor_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = list(hardware_factor_frame.attrs.get("feature_cols", []))
    if not feature_cols:
        raise ValueError("hardware-cycle factor frame produced no feature columns")
    out = frame.copy()
    factors = hardware_factor_frame[["datetime", *feature_cols]].copy()
    factors["datetime"] = pd.to_datetime(factors["datetime"]).dt.normalize()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out = out.merge(factors, on="datetime", how="left")
    return out, _existing_unique(feature_cols, out)


def _ensure_hardware_cycle_qlib_provider(
    spec: HardwareCycleProbabilityBudgetOpen2OpenCardSpec,
    *,
    as_of: str,
) -> dict[str, Any]:
    manifest_path = spec.hardware_cycle_provider_uri / "qlib_store_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("calendar_end")) >= str(as_of):
            return manifest
    return build_hardware_cycle_qlib_store(root=spec.root, start=spec.train[0], end=as_of)


def _hardware_summary_payload(
    *,
    provider_manifest: dict[str, Any],
    factor_attrs: dict[str, Any],
) -> dict[str, Any]:
    feature_cols = list(factor_attrs.get("feature_cols", []))
    return {
        "source": factor_attrs.get("source", "qlib.data.D.features"),
        "provider_uri": factor_attrs.get("provider_uri", provider_manifest.get("provider_uri")),
        "qlib_store_manifest": str(Path(str(provider_manifest.get("provider_uri", ""))) / "qlib_store_manifest.json"),
        "raw_manifest": provider_manifest.get("raw_manifest"),
        "calendar_start": provider_manifest.get("calendar_start"),
        "calendar_end": provider_manifest.get("calendar_end"),
        "feature_cols": feature_cols,
        "feature_count": len(feature_cols),
        "lag_days": factor_attrs.get("lag_days", 1),
        "limitations": list(factor_attrs.get("limitations", [])),
    }


def _run_card(spec: HardwareCycleProbabilityBudgetOpen2OpenCardSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    base_bundle = _build_frame_bundle(spec)
    benchmarks = _dynamic_open_to_open_benchmarks(base_bundle.frame, test=base_bundle.test)
    if not benchmarks:
        raise ValueError("hardware-cycle probability-budget card requires dynamic open-to-open benchmarks")
    hardware_manifest = _ensure_hardware_cycle_qlib_provider(spec, as_of=base_bundle.as_of)
    hardware_end = min(pd.Timestamp(base_bundle.as_of), pd.Timestamp(str(hardware_manifest["calendar_end"]))).date().isoformat()
    hardware_factors = build_hardware_cycle_factor_frame(
        QlibProviderConfig(
            provider_uri=spec.hardware_cycle_provider_uri,
            region=str(hardware_manifest["region"]),
            market=str(hardware_manifest["market"]),
            kernels=1,
        ),
        start=spec.train[0],
        end=hardware_end,
        lag_days=1,
    )
    frame, hardware_feature_cols = _attach_hardware_cycle_model_features(base_bundle.frame, hardware_factors)
    bundle = replace(base_bundle, frame=frame)
    feature_groups = _hardware_cycle_feature_groups(
        _feature_groups(bundle),
        hardware_feature_cols=hardware_feature_cols,
        frame=frame,
    )
    position_spec = _position_spec_for_probability_budget(
        feature_groups[HARDWARE_FEATURE_GROUP],
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

    with tempfile.TemporaryDirectory(prefix="hardware_cycle_probability_budget_") as tmp:
        tmp_dir = Path(tmp)
        positions_path = tmp_dir / "positions.csv"
        walk_forward_positions_path = tmp_dir / "walk_forward_positions.csv"
        result.positions.to_csv(positions_path, index=False)
        result.walk_forward_positions.to_csv(walk_forward_positions_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=bundle.provider_meta["provider_uri"],
                hardware_cycle_provider_uri=spec.hardware_cycle_provider_uri.as_posix(),
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
                architecture="lgb_xgb_consensus_probability_budget_with_hardware_cycle_features",
                feature_group=HARDWARE_FEATURE_GROUP,
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
            summary = _augment_summary_with_hardware_cycle(
                summary,
                hardware_manifest=hardware_manifest,
                hardware_factor_attrs={
                    **hardware_factors.attrs,
                    "feature_cols": hardware_feature_cols,
                },
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
            recorder.log_artifact(positions_path.as_posix(), artifact_path=HARDWARE_RECORDER_ARTIFACT_PATH)
            recorder.log_artifact(walk_forward_positions_path.as_posix(), artifact_path=HARDWARE_RECORDER_ARTIFACT_PATH)
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.name != "failure_modes.md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)

    _write_hardware_summary_files(card, summary, risk_analysis)
    return summary


def _augment_summary_with_hardware_cycle(
    summary: dict[str, Any],
    *,
    hardware_manifest: dict[str, Any],
    hardware_factor_attrs: dict[str, Any],
) -> dict[str, Any]:
    out = dict(summary)
    out["card_type"] = HARDWARE_CARD_TYPE
    out["model_selection"] = {
        **out["model_selection"],
        "source": "qlib_D_features_Alpha158_soxx_constituents_hardware_cycle",
        "feature_group": HARDWARE_FEATURE_GROUP,
    }
    out["position_model"] = {
        **out["position_model"],
        "workflow": "hardware_cycle_probability_budget_open2open",
        "architecture": "lgb_xgb_consensus_probability_budget_with_hardware_cycle_features",
        "formula": "target_position = clip(consensus_probability_edge_with_hardware_cycle * scale, min_position, max_position)",
    }
    out["hardware_cycle"] = _hardware_summary_payload(
        provider_manifest=hardware_manifest,
        factor_attrs=hardware_factor_attrs,
    )
    out["probability_budget_goal"] = {
        **out["probability_budget_goal"],
        "target": "test whether lagged hardware-cycle and cloud-GPU pricing features improve pure probability-budget sizing",
        "hardware_feature_count": out["hardware_cycle"]["feature_count"],
        **_walk_forward_return_gate_summary(pd.DataFrame(out["walk_forward_metrics"])),
    }
    out["artifact_policy"] = {
        **out["artifact_policy"],
        "recorder_only_artifacts": {
            "positions": f"artifacts/{HARDWARE_RECORDER_ARTIFACT_PATH}/positions.csv",
            "walk_forward_positions": f"artifacts/{HARDWARE_RECORDER_ARTIFACT_PATH}/walk_forward_positions.csv",
        },
    }
    return out


def _write_hardware_summary_files(card: Path, summary: dict[str, Any], risk_analysis: pd.DataFrame) -> None:
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
            f"hardware_cycle_provider_uri: {summary['hardware_cycle']['provider_uri']}",
            f"recorder_uri: {summary['recorder_uri']}",
            f"recorder_id: {summary['recorder_id']}",
            f"as_of: {summary['as_of']}",
            "architecture:",
            "  base_structure: LGB/XGB consensus probability budget",
            "  added_features: lagged hardware-cycle and cloud-GPU pricing factors",
            f"  feature_group: {summary['model_selection']['feature_group']}",
            "model_selection:",
            f"  return_col: {summary['return_col']}",
            f"  label_col: {summary['model_target_col']}",
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


def _conclusion(summary: dict[str, Any]) -> str:
    selected = summary["position_model"]["selected_candidate"]["test"]
    latest = summary["latest_signal"]
    hardware = summary["hardware_cycle"]
    walk_forward = summary["position_model"]["walk_forward"]
    comparison = summary["position_model"]["comparison"]
    return (
        "# 结论\n\n"
        f"- what_ran: 执行 {summary['experiment']}，在纯 probability budget 仓位模型中加入 "
        f"{hardware['feature_count']} 个硬件周期/云 GPU 滞后特征。\n"
        f"- key_metrics: selected_return={selected['return_pct']:.4f}%，maxDD={selected['max_drawdown_pct']:.4f}%，"
        f"max_daily_loss={selected['max_daily_loss_pct']:.4f}%，CVaR5={selected.get('cvar_5pct_pct', 0.0):.4f}%。\n"
        f"- walk_forward: folds={walk_forward['fold_count']}，risk_all_pass={walk_forward['all_folds_pass_risk_gates']}，"
        f"return_all_pass={walk_forward.get('all_folds_pass_return_gate')}，min_return={walk_forward.get('min_test_return_pct'):.4f}%。\n"
        f"- hardware_cycle: provider={hardware['provider_uri']}，calendar={hardware['calendar_start']}..{hardware['calendar_end']}，"
        f"lag_days={hardware['lag_days']}。\n"
        f"- latest_signal: as_of={latest.get('as_of')}，action={latest.get('action')}，"
        f"target_position={latest.get('target_percent', 0.0):.4f}%。\n"
        f"- benchmark_gap: selected_vs_base_constrained_ref_return_gap_pct={comparison.get('selected_vs_base_constrained_ref_return_gap_pct'):.4f}%。\n"
        f"- promotion_decision: {summary['promotion_decision']}。\n"
        "- risks: 云 GPU 直接价格历史较短且是外部重构指数；若收益提升，只能作为研究候选，生产前仍需 point-in-time 数据归档。\n"
        f"- next_action: 对比 probability_budget_open2open_v1 的收益/风险门，确认硬件周期特征是否真实改善 WF。recorder={summary['recorder_id']}。\n"
    )


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "hardware-cycle-title",
                "metadata": {},
                "source": [f"# {summary['experiment']}\n", "\n", _conclusion(summary)],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "id": "hardware-cycle-run",
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
                    "from src.workflow.hardware_cycle_probability_budget_open2open import run_hardware_cycle_probability_budget_open2open_card\n",
                    "\n",
                    "# qlib.init, D.features and R.start are executed inside the workflow.\n",
                    "summary = run_hardware_cycle_probability_budget_open2open_card(root=ROOT)\n",
                    "print(json.dumps({'recorder_id': summary['recorder_id'], 'promotion_decision': summary['promotion_decision'], 'as_of': summary['as_of']}, indent=2, sort_keys=True))\n",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "hardware-cycle-conclusion",
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
