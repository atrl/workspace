"""Workflow runner for the multi-session label audit (diagnostic only).

Reads frozen control models from the saved Recorder (no fitting), rebuilds BOTH
the 10bp and 25bp parent own-state accounts, verifies each observed parent state
exactly (rtol=0) against the saved control replay, builds fixed 5-session
counterfactual labels (label fee always 25bp) for each parent cost, and writes a
slim card plus recorder-only per-cost trajectory artifacts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis import multisession_label as M

EXPERIMENT = "stage1_multisession_label_audit_20261010"
SCHEMA_VERSION = "stage1_multisession_label_audit_v1"
COSTS_BPS = (10.0, 25.0)
QLIB_DOCS = ["https://qlib.readthedocs.io/en/latest/component/data.html",
             "https://qlib.readthedocs.io/en/latest/component/recorder.html"]


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _resolve_root(root) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _lineage(root: Path) -> dict:
    lineage = {}
    for name, rel in M.EXPECTED_PROVIDER_SHA256.items():
        data = json.loads((root / "data/qlib" / name / "qlib_store_manifest.json").read_text())
        lineage[name] = {"provider_uri": str((root / "data/qlib" / name).resolve()), "region": data.get("region"),
                         "as_of": data.get("as_of"), "calendar_start": data.get("calendar_start"),
                         "calendar_end": data.get("calendar_end"), "manifest_sha256": rel}
    return lineage


def run_multisession_label_audit(*, root=None, card_dir=None) -> dict[str, Any]:
    spec_root = _resolve_root(root)
    target_card = Path(card_dir) if card_dir is not None else (spec_root / "research" / EXPERIMENT)
    target_card.mkdir(parents=True, exist_ok=True)
    plan_path = spec_root / "research" / EXPERIMENT / "preregistration_plan.json"
    _require(plan_path.is_file(), "preregistration plan is missing")

    control = M.load_frozen_control(spec_root)
    panel, common = control["panel"], control["common"]
    cutoffs = {int(d["model_year"]): d["information_cutoff_exclusive"] for d in control["diagnostics"]}

    # --- FAST phase: observed parent states + exact parity (seconds) BEFORE labels ---
    parents_by_cost, parity_by_cost = {}, {}
    for cost in COSTS_BPS:
        parents = M.observed_parent(panel, models=control["models"], common=common, cost_bps=cost)
        parents_by_cost[int(cost)] = parents
        parity_by_cost[int(cost)] = {}
        for segment, span in M.SEGMENT_LABELS.items():
            saved = control["replays"][f"control_all_state::{span}::{int(cost)}"]
            parity_by_cost[int(cost)][span] = M.verify_observed_parity(
                parents[parents["segment"] == segment].reset_index(drop=True), saved, label=f"{span}@{cost:g}bp")
    training_label_parity = _training_label_parity(control, parents_by_cost[10])

    # --- EXPENSIVE phase: 5-session labels + bootstrap, only after parity passes ---
    results = {}
    for cost in COSTS_BPS:
        parents = parents_by_cost[int(cost)]
        labels, trajectories = M.five_day_labels(panel, parents, models=control["models"], common=common)
        metrics = M.summarize(labels, labels)
        results[int(cost)] = {"parents": parents, "labels": labels, "trajectories": trajectories,
                              "metrics": metrics, "parity": parity_by_cost[int(cost)]}

    summary = {
        "experiment": EXPERIMENT, "schema_version": SCHEMA_VERSION, "card_status": "active_research",
        "diagnostic_only": True, "models_trained": 0, "promotion_eligible": False, "objective_achieved": False,
        "plan_sha256": M._file_or_tree_sha(plan_path),
        "data_lineage": _lineage(spec_root),
        "source_identity": {"prepared_sha256": control["prepared_sha256"],
                            "control_object_sha256": control["object_sha256"],
                            "provider_sha256": control["provider_sha256"],
                            "control_recorder": M.CONTROL_RECORDER,
                            "carry_forward_from_year": control["carry_forward_from_year"]},
        "model_information_cutoffs": cutoffs,
        "parent_costs_bps": [int(c) for c in COSTS_BPS], "label_fee_bps": 25.0,
        "segment_window_status": {int(c): results[int(c)]["metrics"]["window_status_counts"] for c in COSTS_BPS},
        "training_label_parity": training_label_parity,
        "observed_parent_parity": {int(c): results[int(c)]["parity"] for c in COSTS_BPS},
        "native": {"status": "needs_data", "reason": "no native cash/quantity reconciliation executed this diagnostic round"},
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": ["parents_10", "labels_10", "trajectories_10",
                                                        "parents_25", "labels_25", "trajectories_25",
                                                        "diagnostics_control"]},
        "segments": {int(c): results[int(c)]["metrics"]["segments"] for c in COSTS_BPS},
        "next_action": ("read-only diagnostic complete; do NOT train a 5-session candidate without a separate "
                        "pre-registration; outer is reused_diagnostic, never a fresh out-of-sample pass"),
    }
    summary.update(_record(spec_root=spec_root, card_dir=target_card, summary=summary, results=results,
                           diagnostics=control["diagnostics"], plan_path=plan_path))
    _write_freeze_manifest(target_card)
    _write_card(summary, card_dir=target_card)
    return summary


def _write_freeze_manifest(card_dir: Path) -> None:
    manifest = {
        "experiment": EXPERIMENT, "plan_pointer": "preregistration_plan.json",
        "control_recorder": M.CONTROL_RECORDER, "control_card": M.CONTROL_CARD,
        "model_object_sha256": {k: v for k, v in M.EXPECTED_OBJECT_SHA256.items() if k.startswith("model_")},
        "replay_training_diagnostics_sha256": {k: v for k, v in M.EXPECTED_OBJECT_SHA256.items() if not k.startswith("model_")},
        "carry_forward_from_year": M.CARRY_FORWARD_FROM_YEAR,
        "provider_manifest_sha256": M.EXPECTED_PROVIDER_SHA256,
        "prepared_inputs_sha256": M.EXPECTED_PREPARED_SHA256,
        "statistical_corrections": {
            "bootstrap": "fixed 20-session moving blocks over the full official calendar, resampled inside the statistic (previous pooled bootstrap was invalid)",
            "cost_accounts": "both 10bp and 25bp parent own-state accounts computed; label fee fixed at 25bp",
            "five_day_choice": "unchanged fixed 5-session choice; no horizon/sign/threshold selection",
        },
    }
    (card_dir / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _training_label_parity(control: dict, ten_parents: pd.DataFrame) -> dict:
    """Fast 10bp source-vs-observed label parity over all 2690 rows."""
    training = control["training_rows"].sort_values("signal_date").reset_index(drop=True)
    ten_parents = ten_parents.sort_values("signal_date").reset_index(drop=True)
    _require(len(training) == 2690 and len(ten_parents) == 2690, "full-calendar rows must be retained")
    _require(not training["signal_date"].duplicated().any() and not ten_parents["signal_date"].duplicated().any(),
             "duplicate signal_date")
    merged = ten_parents.merge(training[["signal_date", "execute_advantage_bps", "label_status", "label_available_at"]],
                               on="signal_date", suffixes=("", "_tr"), how="left", validate="one_to_one")
    _require(len(merged) == 2690, "merge coverage failed")
    finite = pd.to_numeric(merged["execute_advantage_bps_tr"], errors="coerce").notna()
    _require(int(finite.sum()) == 2688, f"finite labelled rows expected 2688, got {int(finite.sum())}")
    values_exact = bool(np.allclose(merged.loc[finite, "execute_advantage_bps"].to_numpy(float),
                                    merged.loc[finite, "execute_advantage_bps_tr"].to_numpy(float), rtol=0, atol=1e-12))
    available_exact = bool((pd.to_datetime(merged.loc[finite, "label_available_at"], utc=True).to_numpy()
                            == pd.to_datetime(merged.loc[finite, "label_available_at_tr"], utc=True).to_numpy()).all())
    status_match = bool((merged["label_status"].to_numpy() == merged["label_status_tr"].to_numpy()).all())
    _require(values_exact and available_exact and status_match,
             f"blocked: 10bp training-row label parity failed values={values_exact} available={available_exact} status={status_match}")
    return {"n_rows": int(len(merged)), "n_finite": int(finite.sum()), "values_exact": values_exact,
            "label_available_at_exact": available_exact, "label_status_match": status_match,
            "boundary_unknown_rows": [str(d.date()) for d in pd.to_datetime(merged.loc[~finite, "signal_date"])],
            "canonical_unknown_identity": {k: int(v) for k, v in
                                           ten_parents["label_identity"].value_counts().items()},
            "scope": "10bp training book only; 25bp original label has no saved training-row control and is recomputed from its own 25bp state"}


def _record(*, spec_root, card_dir, summary, results, diagnostics, plan_path) -> dict[str, Any]:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import logging
    import warnings
    for name in ("qlib", "mlflow", "qlib.data", "qlib.workflow"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    import qlib
    from qlib.workflow import R

    recorder_uri = (Path(card_dir) / "qlib_recorder").resolve()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((spec_root / "data/qlib/soxl_soxx_soxs").resolve()), region="us")
    saved = {"diagnostics_control": diagnostics}
    for cost, block in results.items():
        saved[f"parents_{cost}"] = block["parents"]
        saved[f"labels_{cost}"] = block["labels"]
        saved[f"trajectories_{cost}"] = block["trajectories"]
    with R.start(experiment_name=EXPERIMENT, recorder_name="multisession_audit", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(schema_version=SCHEMA_VERSION, plan_sha256=summary["plan_sha256"],
                            control_recorder=M.CONTROL_RECORDER, models_trained=0)
        recorder.save_objects(**saved)
        recorder.log_artifact(str(plan_path), artifact_path="plan")
        recorder_id, experiment_id = recorder.id, str(recorder.experiment_id)
        local_dir = recorder_uri / experiment_id / recorder_id
    return {"recorder_id": recorder_id, "experiment_id": experiment_id, "recorder_uri": str(recorder_uri),
            "recorder_local_dir": str(local_dir), "saved_objects": sorted(saved), "qlib_docs_reference": QLIB_DOCS}


def _write_card(summary: dict, *, card_dir: Path) -> None:
    (card_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    rows = []
    for cost, segments in summary["segments"].items():
        for span, block in segments.items():
            if block.get("label_comparison"):
                rows.append({"cost_bps": cost, "segment": span, **block["label_comparison"],
                             "eligible_available_sessions": block["eligible_available_sessions"]})
    if rows:
        pd.DataFrame(rows).to_csv(card_dir / "label_comparison.csv", index=False)
    ic_rows = []
    for cost, segments in summary["segments"].items():
        for span, block in segments.items():
            ic_rows.append({"cost_bps": cost, "segment": span, **block["oos_ic_overall"]})
    pd.DataFrame(ic_rows).to_csv(card_dir / "oos_ic_overall.csv", index=False)
    for cost, segments in summary["segments"].items():
        for span, block in segments.items():
            pd.DataFrame(block["eligible_available_by_year"].items(), columns=["year", "eligible"]).to_csv(
                card_dir / f"eligible_by_year_c{cost}_{span}.csv", index=False)
            pd.DataFrame({int(y): v for y, v in block["oos_ic_by_year"].items()}).T.to_csv(
                card_dir / f"oos_ic_by_year_c{cost}_{span}.csv")
    # Remove stale single-cost sidecars from earlier iterations of this card.
    for pattern in ("eligible_by_year_*.csv", "oos_ic_by_year_*.csv"):
        for stale in card_dir.glob(pattern):
            if "_c" not in stale.name:
                stale.unlink()
    workflow = {"schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
                "plan_sha256": summary["plan_sha256"], "diagnostic_only": True, "models_trained": 0,
                "control_recorder": M.CONTROL_RECORDER, "horizon_sessions": M.HORIZON,
                "parent_costs_bps": summary["parent_costs_bps"], "label_fee_bps": summary["label_fee_bps"],
                "data_lineage": summary["data_lineage"], "source_identity": summary["source_identity"],
                "model_information_cutoffs": summary["model_information_cutoffs"],
                "observed_parent_parity": summary["observed_parent_parity"],
                "training_label_parity": summary["training_label_parity"],
                "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS}
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    lines = [
        "# multi-session label 审计（diagnostic-only，中文结论）\n",
        "- 0 新模型；不训练候选、不晋升；控制模型从已保存 Recorder 读回（全部年度 hash 绑定），本轮不调用任何 fit。\n",
        f"- plan sha `{summary['plan_sha256'][:12]}`；control recorder `{M.CONTROL_RECORDER}`；"
        "label fee=25bp；父账户成本 10bp 与 25bp 各自独立。\n",
        f"- observed-parent 精确 parity（rtol=0）：{summary['observed_parent_parity']}；"
        f"10bp training-row label parity values/available/status：{summary['training_label_parity']['values_exact']}/"
        f"{summary['training_label_parity']['label_available_at_exact']}/{summary['training_label_parity']['label_status_match']}"
        f"（rows={summary['training_label_parity']['n_rows']}, finite={summary['training_label_parity']['n_finite']}, "
        f"boundary_unknown={summary['training_label_parity']['boundary_unknown_rows']}）。\n",
    ]
    for cost, segments in summary["segments"].items():
        for span, block in segments.items():
            lines.append(f"## cost {cost}bp / segment {span}（独立 bootstrap，不跨段拼接）\n")
            c = block.get("label_comparison")
            lines.append(f"- eligible available {block['eligible_available_sessions']}；by year {block['eligible_available_by_year']}。\n")
            if c:
                lines.append(f"- sign disagreement {c['sign_disagreement_rate']:.4f}；orig mean {c['orig_mean_bps']:.2f}bp vs "
                             f"5d mean {c['label_5d_mean_bps']:.2f}bp；std {c['orig_std_bps']:.1f} vs {c['label_5d_std_bps']:.1f}；"
                             f"Spearman {c['spearman_corr']}。\n")
                lines.append(f"- branch risk_state/clock/recovery 分歧率 {c['risk_state_disagreement_rate']:.4f} / "
                             f"{c['risk_clock_disagreement_rate']:.4f} / {c['recovery_disagreement_rate']:.4f}。\n")
            lines.append(f"- 严格因果 OOS（20-session block）：{block['oos_ic_overall']}；by year="
                         + ", ".join(f"{y}:{v['ic']}" for y, v in block["oos_ic_by_year"].items()) + "。\n")
    lines += [
        f"- native：{summary['native']['status']}（{summary['native']['reason']}）。\n",
        "- 结论边界：本卡只诊断标签口径差异与因果 OOS 信息，未证明新 alpha；outer 为 reused_diagnostic（非 fresh）。\n",
        f"- 下一动作：{summary['next_action']}。\n",
    ]
    if summary.get("recorder_id"):
        lines.append(f"- 子 Recorder：{summary['recorder_id']}；parents/labels/trajectories 按 cost 分别保存。\n")
    return "".join(lines)
