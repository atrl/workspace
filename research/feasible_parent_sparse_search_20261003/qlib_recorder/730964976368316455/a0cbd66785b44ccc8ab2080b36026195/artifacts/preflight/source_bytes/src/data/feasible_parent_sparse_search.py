"""Read-only, hash-bound Phase7 parents over the unchanged native input bundle.

Parents are retained policies, never the last optimizer iterate.  This module
does not embed, replay, fit, or repair a policy.  Its compact audit points to the
original full evidence; the new optimizer must revalidate forward feasibility.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.constrained_net_excess import _parent_header, _check_parent_code
from src.data.sparse_holdings_net_excess import load_sparse_inputs, prepare_folds, variant_mapping
from src.workflow.anchor_checkpoint_feasibility import checked_parent_load, _same_evidence


SEEDS = (17, 41, 71)


def _require(value, message):
    if not bool(value):
        raise ValueError("blocked: " + message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def _no_active_recorder():
    from qlib.workflow import R
    try:
        active = R.exp_manager.active_experiment
    except AttributeError:  # Qlib has not been initialized yet.
        active = None
    _require(active is None, "close the writable Recorder before loading frozen parent evidence")


def _artifact_bytes(recorder, manifest, artifact):
    relative = Path(artifact)
    _require(not relative.is_absolute() and ".." not in relative.parts, "invalid parent text artifact path")
    base = (Path(recorder.get_local_dir()) / "artifacts").resolve()
    path = (base / relative).resolve()
    _require(path.is_relative_to(base), "parent text artifact escapes Recorder")
    run = Path(recorder.get_local_dir()).relative_to(Path(recorder.uri))
    wanted = str(Path("qlib_recorder") / run / "artifacts" / relative)
    matches = [entry for entry in manifest["entries"] if entry["path"] == wanted]
    _require(len(matches) == 1, "parent text artifact missing or duplicated in manifest")
    entry = matches[0]
    raw = path.read_bytes()
    _require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"],
             "parent text artifact bytes differ")
    return raw, {"artifact": artifact, "sha256": entry["sha256"], "bytes": entry["bytes"],
                 "parent_manifest_path": wanted}


def _context(root, plan):
    """The caller has initialized the native provider, with no active run."""
    from qlib.workflow import R
    from src.workflow.constrained_net_excess import training_spec

    _no_active_recorder()
    source = plan["parent_model_source"]
    adapted = deepcopy(plan)
    adapted["parent_input_source"] = source
    card, manifest, summary, summary_bytes = _parent_header(root, adapted)
    with R.uri_context(source["recorder_uri"]):
        recorder = R.get_recorder(experiment_name=source["experiment"], recorder_id=source["recorder_id"])
    _require(recorder.client.get_run(recorder.id).info.status == "FINISHED", "Phase7 parent Recorder is not FINISHED")
    registration, registration_id = _artifact_bytes(recorder, manifest, "inputs/selection_plan.json")
    _require(hashlib.sha256(registration).hexdigest() == source["selection_plan_sha256"]
             == summary["plan_sha256"], "Phase7 parent plan identity differs")
    original_plan = json.loads(registration)
    for key in ("training", "constraints"):
        _require(original_plan[key] == plan["parent_reproduction_protocol"][key],
                 "declared parent reproduction protocol differs from original " + key)
    hashes, identity = checked_parent_load(recorder, card, manifest, "inputs/source_hashes.pkl")
    _check_parent_code(root, hashes, summary)
    pins = {"model_source_sha256": "src/model/constrained_net_excess.py",
            "base_model_source_sha256": "src/model/benchmark_anchor_utility.py",
            "workflow_source_sha256": "src/workflow/constrained_net_excess.py",
            "risk_source_sha256": "src/objective/benchmark_tail_constraints.py"}
    _require(all(hashes.get(path) == source[key] for key, path in pins.items()), "Phase7 implementation pins differ")
    source_audit, source_id = checked_parent_load(recorder, card, manifest, "inputs/source_audit.pkl")
    _require(_same_evidence(source_audit, summary["source_audit"]), "parent source audit differs from archived summary")
    _require(source_audit["parent_recorder_id"] == plan["parent_input_source"]["recorder_id"],
             "parent policy used a different frozen input Recorder")
    return {"recorder": recorder, "card": card, "manifest": manifest, "summary": summary,
            "summary_bytes": summary_bytes, "plan": original_plan, "spec": asdict(training_spec(original_plan)),
            "hashes": hashes, "source_audit": source_audit, "artifacts": [registration_id, identity, source_id]}


def _load(context, name):
    obj, identity = checked_parent_load(context["recorder"], context["card"], context["manifest"], name)
    context["artifacts"].append(identity)
    return obj, identity


def _finish_context(root, plan, context):
    _require((context["card"] / "summary.json").read_bytes() == context["summary_bytes"]
             and _sha(context["card"] / "artifact_manifest.json") == plan["parent_model_source"]["artifact_manifest_sha256"],
             "parent card changed during loading")
    _check_parent_code(root, context["hashes"], context["summary"])
    base = Path(context["recorder"].get_local_dir()) / "artifacts"
    _require(all(_sha(base / item["artifact"]) == item["sha256"] for item in context["artifacts"]),
             "parent artifact changed during loading")


def _check_native_identity(bundle, context):
    for key in ("parent_recorder_id", "as_of", "native_frame_sha256", "raw_frame_sha256",
                "prices_frame_sha256", "forecasts_frame_sha256"):
        _require(bundle["source_audit"][key] == context["source_audit"][key],
                 "cached parent and current native input identity differ: " + key)


def _check_fold(fold, record, *, variant, year):
    _require(record["variant"] == variant and record["model_year"] == year,
             "cached parent variant/model year differs")
    _require(record["invalid_training"] is False and record["numerical_failure"] is False
             and record["invalid_training_dates"] == [] and fold["invalid_training"] is False,
             "cached parent has source or numerical failure")
    _require(_same_evidence(record["scaler"], fold["scaler"])
             and record["scaler_sha256"] == fold["scaler"]["scaler_sha256"], "cached parent scaler binding differs")
    _require(record["training_feature_sha256"] == _array_sha(fold["features"])
             and record["training_return_sha256"] == _array_sha(fold["returns"]), "cached parent input matrix hashes differ")
    cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    rows = fold["rows"]
    _require(pd.Timestamp(fold["audit"]["cutoff"]) == cutoff and pd.Timestamp(record["scaler"]["cutoff"]) == cutoff
             and rows.feature_available_at.lt(cutoff).all() and rows.return_available_at.lt(cutoff).all(),
             "cached parent training data are not strictly mature at original cutoff")
    _require(set(record["seed_models"]) == set(SEEDS), "cached parent seed inventory differs")


def _check_original_rows(fold, previous):
    _require(fold["rows"].equals(previous["rows"])
             and np.array_equal(fold["features"], previous["features"])
             and np.array_equal(fold["returns"], previous["returns"])
             and all(_same_evidence(fold[key], previous[key]) for key in
                     ("scaler", "audit", "transform_audit", "invalid_training", "invalid_training_dates")),
             "native mature dates/scaler/matrices differ from original Phase7 parent fold")


def _check_seed(payload, audit, index, per_fit_payload, *, old_variant, year, seed, fold, expected_spec):
    from src.model.benchmark_anchor_utility import BenchmarkAnchorPolicy, policy_from_dict, policy_to_dict

    _require(audit["variant"] == old_variant and audit["year"] == year and audit["seed"] == seed,
             "cached parent training audit identity differs")
    _require(audit["spec"] == expected_spec and audit["training_rows"] == len(fold["rows"]),
             "cached parent original full spec or row count differs")
    _require(audit["failure"] is None and audit["qualification_eligible"] is True
             and audit["training_feasibility_passed"] is True and audit["training_feasible"] is True,
             "cached parent is not a healthy retained fit")
    for field, values in (("training_feature_sha256", fold["features"]), ("training_return_sha256", fold["returns"])):
        _require(audit[field] == _array_sha(values), "parent audit consumed different mature inputs")
    for key, value in index.items():
        if key != "recorder_artifact":
            _require(key in audit and _same_evidence(value, audit[key]), "parent compact/full training audits disagree: " + key)
    epochs = audit["epochs"]
    _require([row["epoch"] for row in epochs] == list(range(expected_spec["updates"] + 1))
             and _same_evidence(epochs, audit["history"]), "parent checkpoint history is incomplete or inconsistent")
    _require(audit["optimizer_diagnostics"]["updates_completed"] == expected_spec["updates"],
             "healthy parent did not complete the original update schedule")
    best_epoch, best_F = "anchor", 0.
    for row in epochs:
        _require(isinstance(row["feasible"], (bool, np.bool_))
                 and isinstance(row["F"], (int, float, np.integer, np.floating))
                 and not isinstance(row["F"], (bool, np.bool_)) and np.isfinite(row["F"]),
                 "invalid parent checkpoint feasibility/objective")
        if row["feasible"] and row["F"] > best_F:
            best_epoch, best_F = row["epoch"], row["F"]
    _require(audit["selected_epoch"] == best_epoch and audit["F"] == best_F,
             "parent did not retain the first strictly best feasible F checkpoint")
    restored = policy_from_dict(payload)
    _require(payload["seed"] == seed and _same_evidence(payload, per_fit_payload),
             "bundle parent differs from original retained per-fit model")
    selected = audit["selected_epoch"]
    if selected == "anchor":
        expected = policy_to_dict(BenchmarkAnchorPolicy(seed=seed, anchor=True))
    else:
        _require(isinstance(selected, (int, np.integer)) and not isinstance(selected, (bool, np.bool_))
                 and 0 <= selected <= expected_spec["updates"], "invalid parent retained epoch")
        expected = epochs[int(selected)]["checkpoint"]
    _require(_same_evidence(payload, expected), "parent payload is not the declared retained checkpoint")
    _require(isinstance(audit["null_selected"], (bool, np.bool_))
             and bool(audit["null_selected"]) == bool(np.count_nonzero(payload["state_dict"]["theta"]) == 0),
             "parent null identity contradicts retained parameters")
    _require(audit["initial_tensor_sha256"] == policy_to_dict(BenchmarkAnchorPolicy(seed=seed))["tensor_sha256"],
             "parent initialization identity differs from its original seed/source")
    F = audit["centered_worst_utility"]
    components = audit["centered_utility_by_cost"]
    costs = set(expected_spec["cost_bps"])
    _require(set(components) == costs and np.isfinite(list(components.values())).all()
             and isinstance(F, (int, float, np.integer, np.floating)) and not isinstance(F, (bool, np.bool_))
             and np.isfinite(F) and F >= 0 and F == min(components.values())
             and audit["F"] == audit["selected_F"] == F, "cached retained parent F is invalid")
    risks = audit["selected_risks_by_cost"]
    _require(set(risks) == costs, "cached parent cost/risk coverage differs")
    for risk in risks.values():
        values = np.array([risk[k] for k in ("max_drawdown_pct", "cvar5_pct", "benchmark_max_drawdown_pct", "benchmark_cvar5_pct")], float)
        gaps, normalizers = np.asarray(risk["raw_percentage_point_gaps"], float), np.asarray(risk["normalizers"], float)
        _require(np.isfinite(values).all() and gaps.shape == (2,) and np.isfinite(gaps).all()
                 and np.allclose(gaps, values[2:] - values[:2], atol=1e-12, rtol=0)
                 and (gaps <= expected_spec["risk_tolerance_percentage_points"]).all()
                 and normalizers.shape == (2,) and np.isfinite(normalizers).all() and (normalizers > 0).all()
                 and np.allclose(normalizers, -values[2:] / 100, atol=1e-14, rtol=0)
                 and risk["normalizers_valid_for_optimization"] is True and risk["feasible"] is True
                 and isinstance(risk["risk_pass"], (list, tuple)) and len(risk["risk_pass"]) == 2
                 and all(isinstance(flag, (bool, np.bool_)) and bool(flag) for flag in risk["risk_pass"]),
                 "cached retained parent raw risk/normalizer evidence is invalid")
    fields = ("variant", "year", "seed", "spec", "training_rows", "selected_epoch", "failure", "qualification_eligible",
              "training_feasibility_passed", "null_selected", "F", "centered_utility_by_cost", "legacy_J",
              "selected_risks_by_cost", "training_feature_sha256", "training_return_sha256", "initial_tensor_sha256")
    return policy_to_dict(restored), {**{key: deepcopy(audit[key]) for key in fields},
                                    "retained_tensor_sha256": payload["tensor_sha256"],
                                    "training_cutoff": fold["audit"]["cutoff"],
                                    "training_dates_sha256": fold["audit"]["eligible_dates_sha256"],
                                    "scaler_sha256": fold["scaler"]["scaler_sha256"],
                                    "scope": "Original retained parent metadata only; corrected-kernel feasibility revalidation still required."}


def _parents(context, folds, stored, index, *, mapping):
    models, audits, bundles = {}, {}, {}
    for (new_variant, year), fold in folds.items():
        old = mapping[new_variant]
        _require(old in stored and year in stored[old], "expected cached parent fold is missing; refit prohibited")
        record = stored[old][year]
        _check_fold(fold, record, variant=old, year=year)
        bundles[(new_variant, year)] = {key: deepcopy(record[key]) for key in
            ("variant", "model_year", "scaler", "scaler_sha256", "training_feature_sha256", "training_return_sha256", "ensemble_audit")}
        for seed in SEEDS:
            identity = (old, year, seed)
            _require(identity in index, "cached parent training audit index is missing")
            row = index[identity]
            expected_path = f"fit_checkpoints/{old}/{year}/{seed}/training_audit.pkl"
            _require(row["recorder_artifact"] == expected_path, "parent audit index points to another fit")
            audit, audit_id = _load(context, expected_path)
            per_fit, model_id = _load(context, f"fit_checkpoints/{old}/{year}/{seed}/model.pkl")
            model, compact = _check_seed(record["seed_models"][seed], audit, row, per_fit,
                old_variant=old, year=year, seed=seed, fold=fold, expected_spec=context["spec"])
            compact["source_artifacts"] = [audit_id, model_id]
            compact["parent_recorder_id"] = context["recorder"].id
            models[(new_variant, year, seed)], audits[(new_variant, year, seed)] = model, compact
    return {"parent_seed_models": models, "parent_seed_audits": audits, "parent_fold_bundles": bundles}


def _index(context):
    rows, _ = _load(context, "analysis/training_diagnostics.pkl")
    identities = [(row["variant"], row["year"], row["seed"]) for row in rows]
    _require(len(identities) == len(set(identities)), "duplicate parent audit identities")
    return dict(zip(identities, rows))


def load_feasible_parent_inputs(root, plan):
    """Native inputs plus development parents only; no writable R or fitting."""
    _no_active_recorder()
    _require(tuple(plan["training"]["seed_ids"]) == SEEDS, "new search seed identities differ from pinned parents")
    root = Path(root).resolve()
    bundle = load_sparse_inputs(root, plan)
    context = _context(root, plan)
    _check_native_identity(bundle, context)
    stored, _ = _load(context, "models/development.pkl")
    original_folds, _ = _load(context, "inputs/development_fold_inputs.pkl")
    mapping = variant_mapping(plan)
    expected = {(old, year) for old in mapping.values() for year in plan["training"]["development_signal_years"]}
    _require({(name, year) for name, records in stored.items() for year in records} == expected,
             "cached development parent coverage differs")
    _require(set(original_folds) == expected, "original Phase7 development input coverage differs")
    for (name, year), fold in bundle["development_folds"].items():
        _check_original_rows(fold, original_folds[(mapping[name], year)])
    parents = _parents(context, bundle["development_folds"], stored, _index(context), mapping=mapping)
    _finish_context(root, plan, context)
    audit = {"parent_recorder_id": context["recorder"].id, "parent_plan_sha256": plan["parent_model_source"]["selection_plan_sha256"],
             "parent_source_sha256": context["hashes"], "development_parent_count": len(parents["parent_seed_models"]),
             "outer_parent_models_loaded": False, "parent_fit_performed": False, "metadata_only_no_policy_replay": True}
    return {**bundle, **parents, "parent_source_artifacts": context["artifacts"], "parent_source_audit": audit,
            "parent_original_plan": context["plan"]}


def load_cached_outer_parents(root, plan, bundle, *, variant, years, selection_seal):
    """Only scheduled outer models after a matching persisted development seal.

    Missing families are reported for the runner's separately recorded original
    protocol reproduction. Missing artifacts for a materialized family fail.
    The caller must close/resume its writable Recorder around this read.
    """
    _no_active_recorder()
    root = Path(root).resolve()
    if isinstance(selection_seal, dict):
        path = root / "research" / plan["experiment"] / "selection_seal.json"
        _require(path.is_file() and json.loads(path.read_text()) == selection_seal,
                 "outer parents require a persisted matching development seal")
        seal = selection_seal
    else:
        path = Path(selection_seal)
        seal = json.loads(path.read_text())
    plan_path = root / "research" / plan["experiment"] / "selection_plan.json"
    _require(json.loads(plan_path.read_text()) == plan and seal["plan_sha256"] == _sha(plan_path)
             and seal["selection_uses_diagnostic_signal_years"] is False and seal["selected"]["variant"] == variant,
             "outer parents precede or disagree with the development selection seal")
    years = list(years)
    _require(years and len(years) == len(set(years)) and all(isinstance(y, int) and not isinstance(y, bool) for y in years)
             and set(years).issubset(plan["training"]["diagnostic_signal_years"]), "invalid scheduled outer parent years")
    mapping = variant_mapping(plan)
    _require(variant in mapping, "unknown scheduled outer parent family")
    folds = prepare_folds(bundle["dataset"], [variant], years, plan)
    context = _context(root, plan)
    _check_native_identity(bundle, context)
    stored, _ = _load(context, "models/selected_annual_refits.pkl")
    original_selected = context["summary"]["selected"]["variant"]
    _require(set(stored) == {original_selected}
             and set(stored[original_selected]) == set(context["plan"]["training"]["diagnostic_signal_years"]),
             "materialized outer parent coverage differs")
    index = _index(context)
    old = mapping[variant]
    if old in stored:
        original_folds, _ = _load(context, "inputs/diagnostic_fold_inputs.pkl")
        for (name, year), fold in folds.items():
            _require((old, year) in original_folds, "materialized outer parent input fold missing")
            previous = original_folds[(old, year)]
            _check_original_rows(fold, previous)
        parents = _parents(context, folds, stored, index, mapping=mapping)
        missing = []
    else:
        _require(not any((old, year, seed) in index for year in years for seed in SEEDS),
                 "outer parent metadata exists but cached family is missing")
        for year in years:
            fragment = f"/artifacts/fit_checkpoints/{old}/{year}/"
            _require(not any(fragment in entry["path"] for entry in context["manifest"]["entries"]),
                     "outer parent artifacts exist; do not replace missing cache by refitting")
        parents = {"parent_seed_models": {}, "parent_seed_audits": {}, "parent_fold_bundles": {}}
        missing = [(variant, year, seed) for year in years for seed in SEEDS]
    _finish_context(root, plan, context)
    return {**parents, "folds": folds, "genuinely_missing": missing, "parent_source_artifacts": context["artifacts"],
            "selection_seal_sha256": _sha(path), "parent_original_plan": context["plan"],
            "parent_source_audit": {"parent_recorder_id": context["recorder"].id,
                "cached_parent_count": len(parents["parent_seed_models"]), "genuinely_missing_count": len(missing),
                "scheduled_variant": variant, "scheduled_years": years, "parent_fit_performed": False}}


__all__ = ["load_feasible_parent_inputs", "load_cached_outer_parents"]
