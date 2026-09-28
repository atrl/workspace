"""Bounded retraining of the previously fixed PIT allocator with new provenance."""
from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import QlibProviderConfig
from src.factor import build_constituent_2x_tail_training_frame, load_soxx_pit_leveraged_training_frame
from src.workflow.direct_continuous_soxx_allocation import DirectContinuousAllocationSpec, _load_frame
from src.workflow.pit_constituent_auxiliary_training_allocation import (
    CANDIDATES, EXPERIMENT as ORIGINAL_EXPERIMENT, PitAuxiliaryAllocationSpec,
    _build_auxiliary_dataset, _build_primary_dataset, _candidate_predictions,
    _evaluate_holdings_aware, _profile, _soxx_common_training_frame,
)
from src.workflow.soxl_objective_feasibility_audit import _json_ready
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "pit_constituent_auxiliary_training_allocation_retrained_20260928"
FIXED_CANDIDATE = "pit_only_aux_005"
MISSING_PREDECESSOR = "3144ec23c3ee4478b6b1e699fb8203d3"
PRIMARY_LABELS = [
    "soxl_forward_open_return", "soxx_forward_open_return", "tail_event_5d",
    "tail_cash_target_5d", "relative_alpha_5d",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mature_training_labels(frame: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A five-return label is known at the fifth row's exit open, not its signal close."""
    result = frame.copy()
    label_end = pd.to_datetime(result["exit_date"]).shift(-4)
    training = result.index <= pd.Timestamp(cutoff)
    unavailable = training & (label_end.isna() | label_end.gt(pd.Timestamp(cutoff)))
    result.loc[unavailable, PRIMARY_LABELS] = float("nan")
    matured = training & ~unavailable
    if not matured.any():
        raise ValueError("blocked: no primary training labels are realized by the fixed cutoff")
    return result, {
        "training_information_cutoff": cutoff,
        "primary_label_timing": "five T+1 open to T+2 open returns; final realization is fifth exit_date",
        "excluded_cross_boundary_signal_rows": int(unavailable.sum()),
        "last_mature_training_signal": result.index[matured].max().date().isoformat(),
        "last_training_label_realization": label_end.loc[matured].max().date().isoformat(),
    }


def _training_auxiliary(spec: PitAuxiliaryAllocationSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((spec.auxiliary_provider_uri / "qlib_store_manifest.json").read_text())
    # Construct forward labels only from raw observations already available at
    # the fixed training cutoff. Calibration on later 2x products is unnecessary.
    raw = load_soxx_pit_leveraged_training_frame(
        QlibProviderConfig(provider_uri=spec.auxiliary_provider_uri),
        instruments=list(manifest["instruments"]),
        start=manifest["calendar_start"], end=spec.development_end,
    )
    auxiliary = build_constituent_2x_tail_training_frame(raw)
    auxiliary = auxiliary.loc[
        auxiliary["training_eligible"]
        & auxiliary["synthetic_2x_crash_label"].notna()
    ].copy()
    if auxiliary.empty or pd.to_datetime(raw["datetime"]).max() > pd.Timestamp(spec.development_end):
        raise ValueError("blocked: invalid bounded auxiliary training source")
    return auxiliary, {
        "source_api": "qlib.data.D.features",
        "provider_uri": str(spec.auxiliary_provider_uri),
        "raw_observation_end": pd.to_datetime(raw["datetime"]).max().date().isoformat(),
        "rows": len(auxiliary),
        "label_construction": "synthetic 2x forward labels built after truncating raw observations at training cutoff",
        "union_backfill_used": False,
        "post_cutoff_actual_2x_calibration_used": False,
    }


def validate_frozen_bundle(card: Path) -> dict[str, Any]:
    manifest = json.loads((card / "frozen_bundle_manifest.json").read_text())
    for artifact in [*manifest["artifacts"], *manifest.get("recorder_metadata", [])]:
        path = card / artifact["path"]
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            raise ValueError(f"blocked: frozen retraining artifact missing or changed: {artifact['path']}")
    return manifest


def run_fixed_pit_retraining(
    *, root: Path | str, as_of: str, experiment: str = EXPERIMENT,
) -> dict[str, Any]:
    """Fit exactly one fixed candidate; publish new evidence without promotion."""
    from qlib.workflow import R

    root = Path(root).resolve()
    if experiment == ORIGINAL_EXPERIMENT or Path(experiment).name != experiment:
        raise ValueError("blocked: retraining requires a separate card identity")
    if pd.Timestamp(as_of) < pd.Timestamp("2023-01-03") or pd.Timestamp(as_of) > pd.Timestamp.today().normalize():
        raise ValueError("blocked: invalid retraining diagnostic as_of")
    spec = PitAuxiliaryAllocationSpec(root=root, experiment=experiment, outer=("2023-01-03", as_of))
    card = spec.card_dir
    if (card / "summary.json").exists():
        summary = json.loads((card / "summary.json").read_text())
        if summary.get("as_of") != as_of or summary.get("experiment") != experiment:
            raise ValueError("blocked: existing retraining card has different identity; use a new experiment")
        bundle = validate_frozen_bundle(card)
        if summary.get("recorder_id") != bundle.get("recorder_id"):
            raise ValueError("blocked: retraining summary and frozen recorder identity differ")
        return summary
    card.mkdir(parents=True, exist_ok=True)
    candidate = next(value for value in CANDIDATES if value.name == FIXED_CANDIDATE)
    provider_names = ("soxl", "soxl_soxx_soxs", "dynamic_soxx_soxl_cash_allocation_wf_v1", "soxx_pit_leveraged_training_v1")
    source_files = [root / "data/qlib" / name / "qlib_store_manifest.json" for name in provider_names]
    source_hashes = {str(path.relative_to(root)): _sha256(path) for path in source_files}
    implementation = root / "src/workflow/pit_constituent_auxiliary_training_allocation.py"
    source_hashes[str(implementation.relative_to(root))] = _sha256(implementation)
    provenance = {
        "kind": "user_authorized_fixed_candidate_retraining",
        "replaces_missing_artifact_identity": False,
        "missing_predecessor_recorder_id": MISSING_PREDECESSOR,
        "candidate": asdict(candidate),
        "training_spec": _json_ready(asdict(spec) | {"root": str(root)}),
        "candidate_search_count": 1,
        "selection_policy": "previously_fixed_candidate_no_new_search",
        "selection_uses_outer_test": False,
        "same_weights_as_missing_predecessor_claimed": False,
        "training_implementation": "current holdings-aware turnover implementation; new model identity",
        "source_sha256": source_hashes,
    }
    # Provider loading switches between Qlib stores. Complete it before opening
    # the recorder; Qlib prohibits provider reinitialization during an active run.
    print("Loading fixed-candidate qlib datasets", flush=True)
    market, market_audit, features = _load_frame(DirectContinuousAllocationSpec(
        root=root, train_start=spec.train_start, development_years=spec.development_years,
        development_end=spec.development_end, outer=spec.outer, purge_rows=spec.purge_rows,
    ))
    market = market.loc[market.index <= pd.Timestamp(as_of)].copy()
    market, label_audit = _mature_training_labels(market, spec.development_end)
    primary_dataset, primary_audit = _build_primary_dataset(market, features, spec)
    common = _soxx_common_training_frame(spec, end=as_of).reindex(market.index)
    auxiliary, auxiliary_audit = _training_auxiliary(spec)
    auxiliary_dataset, auxiliary_dataset_audit = _build_auxiliary_dataset(auxiliary, spec)
    provenance.update(primary_label_audit=label_audit, primary_dataset=primary_audit,
                      auxiliary=auxiliary_audit, auxiliary_dataset=auxiliary_dataset_audit,
                      market_audit=market_audit)
    print("Fitting 2016-2022 walk-forward models and one outer refit", flush=True)
    development, outer, states, diagnostics = _candidate_predictions(
        primary_dataset=primary_dataset, auxiliary_dataset=auxiliary_dataset, market=market,
        soxx_common=common, primary_features=features, candidate=candidate, spec=spec,
    )
    if any(_sha256(root / name) != expected for name, expected in source_hashes.items()):
        raise ValueError("blocked: retraining input manifests or implementation changed during fitting")
    comparisons, returns = [], []
    for segment, prediction in (("development_walk_forward", development), ("outer_diagnostic", outer)):
        for cost in spec.costs_bps:
            metrics, result, _ = _evaluate_holdings_aware(
                market=market, predictions=prediction, profile=_profile(candidate), cost_bps=cost,
            )
            metadata = {"segment": segment, "candidate": candidate.name, "cost_bps": cost,
                        "selection_uses_outer_test": False}
            comparisons.append(metadata | metrics)
            returns.append(result.assign(**metadata))
    comparison = pd.DataFrame(comparisons)
    recorder_uri = card / "qlib_recorder"
    R.set_uri(str(recorder_uri))
    with R.start(experiment_name=experiment):
        recorder = R.get_recorder()
        recorder_dir = Path(recorder.get_local_dir()).resolve()
        R.log_params(candidate=FIXED_CANDIDATE, candidate_search_count=1,
                     train_start=spec.train_start, training_information_cutoff=spec.development_end,
                     diagnostic_as_of=as_of, selection_uses_outer_test="false", seed=spec.seed)
        provenance["recorder_id"] = recorder.id
        provenance["recorder_local_dir"] = str(recorder_dir)
        selected = {"candidate": candidate.name, "candidate_parameters": asdict(candidate),
                    "selection_policy": provenance["selection_policy"], "selection_uses_outer_test": False,
                    "recorder_id": recorder.id, "training_information_cutoff": spec.development_end}
        with tempfile.TemporaryDirectory(prefix="fixed_pit_retrain_") as temporary:
            temp = Path(temporary)
            (temp / "models").mkdir()
            with (temp / "models" / f"{candidate.name}.pkl").open("wb") as handle:
                pickle.dump(states, handle)
            pd.concat([development, outer]).sort_index().reset_index(names="signal_date").to_csv(temp / "predictions.csv", index=False)
            pd.concat(returns, ignore_index=True).to_csv(temp / "strategy_returns.csv", index=False)
            diagnostics.to_csv(temp / "training_diagnostics.csv", index=False)
            (temp / "selected_model.json").write_text(json.dumps(selected, indent=2) + "\n")
            (temp / "provenance.json").write_text(json.dumps(_json_ready(provenance), indent=2) + "\n")
            for path in temp.rglob("*"):
                if path.is_file():
                    recorder.log_artifact(str(path), artifact_path=str(Path(experiment) / path.parent.relative_to(temp)))
        artifact_dir = recorder_dir / "artifacts" / experiment
        required = [f"models/{candidate.name}.pkl", "selected_model.json", "predictions.csv",
                    "strategy_returns.csv", "training_diagnostics.csv", "provenance.json"]
        bundle = {
            "version": 1, "experiment": experiment, "recorder_id": recorder.id,
            "training_window": [spec.train_start, spec.development_end], "diagnostic_as_of": as_of,
            "source_sha256": source_hashes,
            "artifacts": [{"path": str((artifact_dir / name).relative_to(card)),
                           "sha256": _sha256(artifact_dir / name), "size_bytes": (artifact_dir / name).stat().st_size}
                          for name in required],
        }
        summary = {
            "experiment": experiment, "card_type": "fixed_candidate_retraining",
            "decision": "fixed_candidate_retrained_not_promoted_pending_stage1_validation",
            "promotion_decision": "not_promoted", "system_of_record": "qlib",
            "provider_uri": str(spec.daily_provider_uri), "as_of": as_of,
            "selected_model": selected, "source_audit": provenance,
            "recorder_id": recorder.id, "recorder_uri": str(recorder_uri), "recorder_local_dir": str(recorder_dir),
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                "recorder_only_artifacts": [f"artifacts/{experiment}/{name}" for name in required]},
            "frozen_bundle_manifest": "frozen_bundle_manifest.json",
            "qlib_docs_reference": ["https://qlib.readthedocs.io/en/latest/component/data.html",
                                    "https://qlib.readthedocs.io/en/latest/component/recorder.html"],
        }
        for name, payload in (("summary.json", summary), ("selected_model.json", selected), ("frozen_bundle_manifest.json", bundle)):
            (card / name).write_text(json.dumps(_json_ready(payload), indent=2) + "\n")
        comparison.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
        diagnostics.to_csv(card / "training_summary.csv", index=False)
        dev = comparison.loc[(comparison.segment == "development_walk_forward") & (comparison.cost_bps == 10)].iloc[0]
        out = comparison.loc[(comparison.segment == "outer_diagnostic") & (comparison.cost_bps == 10)].iloc[0]
        conclusion = (
            "# 新身份固定候选重训结论\n\n"
            f"- 固定 `{candidate.name}`，仅训练该候选，2016–2022 年度 walk-forward + 截至 2022-12-30 的 outer refit。\n"
            f"- 原 recorder `{MISSING_PREDECESSOR}` 的权重未找回；这是新 recorder `{recorder.id}`，不宣称等同旧权重。\n"
            f"- 训练标签实现日不晚于 2022-12-30；排除跨截止信号 {label_audit['excluded_cross_boundary_signal_rows']} 行，最后成熟训练信号 {label_audit['last_mature_training_signal']}。\n"
            f"- 10bps 开发期收益 {dev.return_pct:.2f}%，MaxDD {dev.max_drawdown_pct:.2f}%；外层诊断收益 {out.return_pct:.2f}%，MaxDD {out.max_drawdown_pct:.2f}%。\n"
            "- 外层仅诊断，不选候选或参数。基础模型重训完成不代表 Stage1/HOLD 动作层验证通过；当前不晋升、不切换模拟盘。\n"
            "- 后续：由 Stage1 动作层做独立开发期门禁和外层诊断；冻结文件与 SHA256 见 frozen_bundle_manifest.json。\n"
        )
        (card / "conclusion.md").write_text(conclusion)
        import yaml
        workflow = {"experiment": experiment, "provider_uri": str(spec.daily_provider_uri), "region": "us",
                    "as_of": as_of, "recorder_uri": str(recorder_uri), "candidate": asdict(candidate),
                    "training_spec": provenance["training_spec"], "selection_uses_outer_test": False,
                    "artifact_policy": summary["artifact_policy"]}
        (card / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False))
        for name in ("summary.json", "frozen_bundle_manifest.json", "backtest_benchmark_comparison.csv",
                     "training_summary.csv", "conclusion.md", "workflow.yaml"):
            recorder.log_artifact(str(card / name), artifact_path=experiment)
        R.log_metrics(development_return_pct=float(dev.return_pct), outer_return_pct=float(out.return_pct),
                      development_max_drawdown_pct=float(dev.max_drawdown_pct), outer_max_drawdown_pct=float(out.max_drawdown_pct))
        _remove_qlib_code_snapshot_artifacts(recorder_dir)
        validate_frozen_bundle(card)
    # Qlib writes FINISHED status when the context exits. Seal the metadata
    # afterwards so a restored bundle is a readable native Qlib recorder too.
    metadata_paths = [recorder_dir / "meta.yaml", recorder_dir.parent / "meta.yaml"]
    for directory in ("params", "metrics", "tags"):
        metadata_paths.extend(path for path in (recorder_dir / directory).glob("*") if path.is_file())
    bundle["recorder_metadata"] = [
        {"path": str(path.relative_to(card)), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in sorted(metadata_paths)
    ]
    (card / "frozen_bundle_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n")
    recorder.log_artifact(str(card / "frozen_bundle_manifest.json"), artifact_path=experiment)
    validate_frozen_bundle(card)
    return _json_ready(summary)
