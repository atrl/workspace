"""One bounded, post-fit repair of the sealed FAILED P2 Recorder.

The old run is read-only. A distinct repair GO is required before any P1 or
FAILED-run object is deserialized, and the new run has its own Qlib Recorder.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from src.workflow import path_area_bridge as p2


FAILED_RID = "004c3320e6214a14bfdbd06425379514"
FAILED_EID = "744316177731782220"
FAILURE_MANIFEST_SHA256 = "1ee03974f3abfac3db38b6b0a988166a3544d0d9e4209c9c1b549ef551ba9143"
FAILURE_SUMMARY_SHA256 = "69451895cb53eb04ec46bfc3bb3c1beb571e45fc9ab9f13cca87196f8886315d"
FAILURE_METADATA_SHA256 = "c39877e64eb267828ec047a31909f87e5f6120c08f1c665c4c3fa30759869ec5"
FAILED_STATE_SHA256 = "65aa49497d5053c8c25c3af7ed6d6c21d5f3f6d3f7d4780d84ed0678cd92a018"
ORIGINAL_IMPLEMENTATION_SHA256 = "456c3b342c36a4ce016dacbdf2949e8c9e307d8eb97b9b9788c2f9941a760d15"
REPAIR_CARD = Path("research/path_area_action_bridge_20261018/P2/repair_1")
REPAIR_EXPERIMENT = "path_area_action_bridge_20261018_P2_repair_1"
REPAIR_RECORDER_NAME = "path_area_action_bridge_P2_repair_1"
REPAIR1_FAILURE_RECEIPT_SHA256 = "e70fbd1895b846f626ab39699a7aa87a80afc923dfe23270c46864110c17c798"


@dataclass(frozen=True)
class RepairVersion:
    number: int

    @property
    def card(self) -> Path:
        return REPAIR_CARD.parent / f"repair_{self.number}"

    @property
    def phase(self) -> str:
        return f"P2_repair_{self.number}"

    @property
    def experiment(self) -> str:
        return f"path_area_action_bridge_20261018_P2_repair_{self.number}"

    @property
    def recorder_name(self) -> str:
        return f"path_area_action_bridge_P2_repair_{self.number}"


REPAIR_1 = RepairVersion(1)
REPAIR_2 = RepairVersion(2)
SAVED_STATE_NAMES = tuple(f"models/new_{year}_{seed}" for year in (2018, 2019)
                          for seed in (71, 172, 273))
SEED_PREDICTION_NAMES = tuple(f"OOF_seed_predictions/year_{year}_seed_{seed}"
                              for year in range(2018, 2023) for seed in (71, 172, 273))
CHECKPOINT_NAMES = ("barrier/fit_barrier", *SAVED_STATE_NAMES,
                    *SEED_PREDICTION_NAMES, "OOF/raw_equal_weight_predictions")


def _read_seal(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate all sealed bytes before a Qlib checkpoint object can be read."""
    card = root / p2.P2_CARD
    fixed = {
        "failure_manifest.json": FAILURE_MANIFEST_SHA256,
        "failure_summary.json": FAILURE_SUMMARY_SHA256,
        "failure_metadata_receipt.json": FAILURE_METADATA_SHA256,
        "P2_execution_state.json": FAILED_STATE_SHA256,
    }
    for name, digest in fixed.items():
        path = card / name
        if not path.is_file() or path.is_symlink() or p2._sha_file(path) != digest:
            raise ValueError("blocked: sealed failed-run metadata changed: " + name)
    manifest = json.loads((card / "failure_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("recorder_id") != FAILED_RID
            or manifest.get("experiment_id") != FAILED_EID
            or manifest.get("recorder_status") != "FAILED"
            or manifest.get("original_implementation_sha256") != ORIGINAL_IMPLEMENTATION_SHA256):
        raise ValueError("blocked: failed-run seal identity differs")
    original_identity = root / "research/path_area_action_bridge_20261018/P2_implementation_identity.json"
    if json.loads(original_identity.read_text(encoding="utf-8")).get("sha256") != ORIGINAL_IMPLEMENTATION_SHA256:
        raise ValueError("blocked: original implementation identity differs")
    state = json.loads((card / "P2_execution_state.json").read_text(encoding="utf-8"))
    if not (state.get("status") == "failed" and state.get("recorder_id") == FAILED_RID
            and state.get("plan_sha256") == p2.PLAN_SHA256
            and state.get("new_allocation_fit_attempted") == 6
            and state.get("new_allocation_fit_completed") == 6
            and state.get("training_forward_calls_completed") == 1800
            and state.get("OOF_prediction_calls_completed") == 15
            and state.get("shadow_policy_paths") == 0
            and state.get("label_pairs") == 0 and state.get("native_accounts") == 0
            and state.get("fit_barrier_sha256") == "29b87bbbbf4dd643d4c85168581ed44d99ce5085d08472134c32573dfda9b6fc"
            and state.get("exception_type") == "KeyError" and state.get("exception") == "'tariff'"):
        raise ValueError("blocked: failed-run completed stage or failure boundary differs")
    run_relative = Path(manifest["run_metadata_path"])
    run_prefix = Path("qlib_recorder") / FAILED_EID / FAILED_RID
    if (run_relative != run_prefix / "meta.yaml" or not (card / run_relative).is_file()
            or (card / run_relative).is_symlink()
            or p2._sha_file(card / run_relative) != manifest["run_metadata_sha256"]):
        raise ValueError("blocked: FAILED run metadata changed")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 34:
        raise ValueError("blocked: exact failed-run artifact inventory is required")
    recorded = {}
    for item in entries:
        relative = Path(item["path"])
        if (relative.is_absolute() or ".." in relative.parts or relative.parts[:4] !=
                ("qlib_recorder", FAILED_EID, FAILED_RID, "artifacts") or len(relative.parts) < 5):
            raise ValueError("blocked: failed-run artifact path escapes its Recorder")
        name = Path(*relative.parts[4:]).as_posix()
        if name in recorded or not isinstance(item.get("bytes"), int):
            raise ValueError("blocked: duplicate or invalid failed-run artifact entry")
        path = card / relative
        if (not path.is_file() or any(parent.is_symlink() for parent in (path, *path.parents)
                    if parent.is_relative_to(card)) or path.stat().st_size != item["bytes"]
                or p2._sha_file(path) != item["sha256"]):
            raise ValueError("blocked: failed-run artifact differs before unpickling: " + name)
        recorded[name] = item
    actual = {path.relative_to(card / run_prefix / "artifacts").as_posix()
              for path in (card / run_prefix / "artifacts").rglob("*") if path.is_file()}
    if actual != set(recorded) or not set(CHECKPOINT_NAMES).issubset(recorded):
        raise ValueError("blocked: failed-run Recorder inventory is incomplete or expanded")
    barrier = json.loads((card / "fit_barrier.json").read_text(encoding="utf-8"))
    if (barrier.get("status") != "passed" or barrier.get("sha256") != state["fit_barrier_sha256"]
            or len(barrier.get("completed_slots", [])) != 6):
        raise ValueError("blocked: completed six-state barrier differs")
    return manifest, recorded


def _validate_aggregate_checkpoint(predictions: Any) -> None:
    """Check basic shape; exact rows and values are proved against P1 routes later."""
    columns = ["soxl_weight", "soxx_weight", "cash_weight",
               "tail_probability", "alpha_probability"]
    if (not isinstance(predictions, pd.DataFrame) or predictions.empty
            or not predictions.index.is_unique or list(predictions.columns) != columns):
        raise ValueError("blocked: saved aggregate OOF prediction schema differs")


def _record_checkpoint_progress(state: dict[str, Any], state_path: Path,
                                event: str, name: str) -> None:
    if event == "attempt":
        state["checkpoint_artifacts_load_attempted"] += 1
        state["current_checkpoint_artifact"] = name
    elif event == "completed":
        state["checkpoint_artifacts_loaded"] += 1
        state["current_checkpoint_artifact"] = None
    elif event == "unknown_partial":
        state["checkpoint_partial_load_unknown_on_exception"] = True
    else:
        raise ValueError("blocked: unknown checkpoint progress event")
    p2._atomic_json(state_path, state)


def _load_failed_checkpoint(root: Path, recorded: dict[str, Any], *,
                            progress: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the pinned FAILED run without starting or resuming it."""
    from qlib.workflow import R

    old_uri = str(root / p2.P2_CARD / "qlib_recorder")
    original_uri = R.get_uri()
    active = R.exp_manager.active_experiment
    if active is not None and active.active_recorder is not None:
        raise ValueError("blocked: checkpoint read requires no active writable Recorder")
    loaded = {}
    with R.uri_context(old_uri):
        recorder = R.get_recorder(experiment_name="path_area_action_bridge_20261018_P2",
                                  recorder_id=FAILED_RID)
        run = recorder.client.get_run(recorder.id)
        expected_local = (root / p2.P2_CARD / "qlib_recorder" / FAILED_EID / FAILED_RID).resolve()
        expected_artifacts = (expected_local / "artifacts").resolve()
        if (recorder.id != FAILED_RID or str(recorder.experiment_id) != FAILED_EID
                or run.info.status != "FAILED" or run.info.run_id != FAILED_RID
                or Path(recorder.get_local_dir()).resolve() != expected_local
                or Path(str(recorder.artifact_uri).removeprefix("file://")).resolve() != expected_artifacts):
            raise ValueError("blocked: Qlib FAILED Recorder identity/status differs")
        artifact_root = root / p2.P2_CARD / "qlib_recorder" / FAILED_EID / FAILED_RID / "artifacts"
        for name in CHECKPOINT_NAMES:
            path = artifact_root / name
            if p2._sha_file(path) != recorded[name]["sha256"]:
                raise ValueError("blocked: checkpoint changed before unpickling: " + name)
            if progress is not None:
                progress("attempt", name)
            try:
                value = recorder.load_object(name)
            except BaseException:
                if progress is not None:
                    progress("unknown_partial", name)
                raise
            loaded[name] = value
            if progress is not None:
                progress("completed", name)
        if recorder.client.get_run(recorder.id).info.status != "FAILED":
            raise ValueError("blocked: FAILED Recorder status changed during checkpoint read")
    if R.get_uri() != original_uri or R.exp_manager.active_experiment is not active:
        raise ValueError("blocked: Qlib Recorder context changed after checkpoint read")
    for name in CHECKPOINT_NAMES:
        path = root / p2.P2_CARD / "qlib_recorder" / FAILED_EID / FAILED_RID / "artifacts" / name
        if p2._sha_file(path) != recorded[name]["sha256"]:
            raise ValueError("blocked: checkpoint changed during read: " + name)
    barrier = loaded["barrier/fit_barrier"]
    if (barrier.get("sha256") != "29b87bbbbf4dd643d4c85168581ed44d99ce5085d08472134c32573dfda9b6fc"
            or len(barrier.get("completed_slots", [])) != 6):
        raise ValueError("blocked: deserialized fit barrier differs")
    predictions = loaded["OOF/raw_equal_weight_predictions"]
    _validate_aggregate_checkpoint(predictions)
    return loaded, {"recorder_id": FAILED_RID, "experiment_id": FAILED_EID,
        "recorder_status": "FAILED", "failure_manifest_sha256": FAILURE_MANIFEST_SHA256,
        "checkpoint_artifacts": {name: recorded[name]["sha256"] for name in CHECKPOINT_NAMES},
        "hashes_checked_before_unpickling": True, "old_recorder_read_only": True}


def _verify_saved_predictions(checkpoint: dict[str, Any], inputs: dict[str, Any]) -> pd.DataFrame:
    """Assemble stored outputs only, proving the aggregate without model forward."""
    from src.model.minute_pit_allocator import _flat_index

    timing = inputs["timing"]
    annual = []
    for year in range(2018, 2023):
        route = inputs["fold_package"]["folds"][year]["prediction_index"]
        dates = (pd.DatetimeIndex(route.get_level_values("datetime"))
                 if isinstance(route, pd.MultiIndex) else pd.DatetimeIndex(route))
        if not len(dates) or not timing.loc[dates, "trade_year"].eq(year).all():
            raise ValueError("blocked: P1 frozen OOF route differs")
        seed_outputs = []
        for seed in (71, 172, 273):
            value = checkpoint[f"OOF_seed_predictions/year_{year}_seed_{seed}"]
            if not isinstance(value, pd.DataFrame):
                raise ValueError("blocked: saved per-seed OOF is not a frame")
            flat = _flat_index(value)
            if flat.index.has_duplicates or set(flat.index) != set(dates):
                raise ValueError("blocked: saved per-seed OOF index differs")
            seed_outputs.append(flat.reindex(dates))
        ensemble = p2._equal_weight_ensemble(seed_outputs)
        annual.append(ensemble)
    assembled = pd.concat(annual).sort_index().loc[:,
        ["soxl_weight", "soxx_weight", "cash_weight", "tail_probability", "alpha_probability"]]
    saved = checkpoint["OOF/raw_equal_weight_predictions"]
    if (assembled.index.has_duplicates or set(assembled.index) !=
            set(timing.index[timing.trade_year.between(2018, 2022)]) or
            any(pd.Timestamp(value).year >= 2023 for value in assembled.index)):
        raise ValueError("blocked: saved OOF grid extends beyond P2")
    pd.testing.assert_frame_equal(assembled, saved, check_exact=True)
    return saved


def _verify_repair1_failure(root: Path) -> dict[str, Any]:
    path = root / REPAIR_1.card / "failure_receipt.json"
    if not path.is_file() or p2._sha_file(path) != REPAIR1_FAILURE_RECEIPT_SHA256:
        raise ValueError("blocked: sealed repair_1 failure receipt differs")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    raw = root / receipt["raw_state_path"]
    if (receipt.get("status") != "failed_implementation_before_P1_reload"
            or receipt.get("corrected_checkpoint_objects_loaded") != 23
            or receipt.get("new_P1_artifact_loads") != 0
            or receipt.get("new_Recorder_created") is not False
            or p2._sha_file(raw) != receipt.get("raw_state_sha256")):
        raise ValueError("blocked: repair_1 failure boundary differs")
    return receipt


def _require_repair_go(root: Path, repair_identity: str,
                       version: RepairVersion = REPAIR_1) -> dict[str, Any]:
    path = root / version.card / "repair_execution_authorization.json"
    if not path.is_file():
        raise PermissionError("blocked: separate P2 repair execution GO is required")
    go = json.loads(path.read_text(encoding="utf-8"))
    repair_plan_sha256 = p2._sha_file(root / version.card / "repair_plan.json")
    budget = {"new_allocation_fits": 0, "new_model_forward_calls": 0,
        "saved_new_models_loaded": 6, "saved_seed_predictions_loaded": 15,
        "parent_artifact_loads_maximum": 9, "common_shadow_policy_paths": 1,
        "common_shadow_native_accounts": 1, "action_classifier_fits": 0,
        "evaluation_2023_plus_forward_or_accounts": 0, "new_market_source_queries": 0,
        "new_market_HTTP": 0, "native_derived_point_provider_builds": 1,
        "native_derived_point_provider_readbacks": 1,
        "area_recomputations_maximum": 1}
    if (go.get("approved") is not True or go.get("owner") != "/root"
            or go.get("scope") != f"actual_P2_repair_{version.number}_execution"
            or go.get("P3") is not False or go.get("plan_sha256") != p2.PLAN_SHA256
            or go.get("failure_manifest_sha256") != FAILURE_MANIFEST_SHA256
            or go.get("repair_plan_sha256") != repair_plan_sha256
            or go.get("repair_implementation_sha256") != repair_identity
            or go.get("repair_implementation_review_sha256") !=
                p2._sha_file(root / version.card / "repair_implementation_review.json")
            or (version.number == 2 and go.get("repair_1_failure_receipt_sha256") !=
                REPAIR1_FAILURE_RECEIPT_SHA256)
            or go.get("budget") != budget):
        raise PermissionError("blocked: P2 repair GO differs from sealed scope and budgets")
    return {"path": str(path), "sha256": p2._sha_file(path), "receipt": go}


def repair_implementation_identity(root: str | Path | None = None, *,
                                   version: RepairVersion = REPAIR_1) -> dict[str, Any]:
    """Code-only identity for review; never deserializes a source object."""
    workspace = p2._resolve_root(root)
    _, plan = p2._load_plan(workspace)
    hashes = dict(p2._implementation_identity(workspace, plan)["files"])
    files = ("src/workflow/path_area_bridge_recovery.py",
             str(version.card / "repair_plan.json"),
             str(version.card / "workflow.yaml"),
             str(version.card / f"repair_{version.number}.ipynb"),
             "tests/test_path_area_bridge_recovery.py")
    for name in files:
        path = workspace / name
        hashes[name] = (p2._notebook_source_identity(path) if path.suffix == ".ipynb"
                        else p2._sha_file(path))
    return {"files": hashes, "sha256": p2._canonical_sha(hashes)}


def _require_unused_repair_card(card: Path, version: RepairVersion = REPAIR_1) -> None:
    allowed = {"workflow.yaml", "repair_plan.json", f"repair_{version.number}.ipynb",
               "repair_execution_authorization.json", "repair_implementation_review.json"}
    if not card.is_dir() or any(path.name not in allowed for path in card.iterdir()):
        raise FileExistsError(f"blocked: repair_{version.number} already has execution artifacts; no automatic retry")


def _run_repair(*, root: str | Path | None = None,
                version: RepairVersion) -> dict[str, Any]:
    """Execute only the uncompleted P2 stages after an exact separate repair GO."""
    root = p2._resolve_root(root)
    _, plan = p2._load_plan(root)
    inherited = p2._verify_inherited(root, plan)
    repair_identity = repair_implementation_identity(root, version=version)
    if version.number == 2:
        _verify_repair1_failure(root)
    review_path = root / version.card / "repair_implementation_review.json"
    if not review_path.is_file():
        raise PermissionError("blocked: scoped P2 repair implementation review is required")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    verdict = ("APPROVE_P2_REPAIR_FOR_SEPARATE_EXECUTION_GO" if version.number == 1
               else "APPROVE_P2_REPAIR_2_FOR_SEPARATE_EXECUTION_GO")
    if (review.get("verdict") != verdict
            or review.get("plan_sha256") != p2.PLAN_SHA256
            or review.get("failure_manifest_sha256") != FAILURE_MANIFEST_SHA256
            or (version.number == 2 and review.get("repair_1_failure_receipt_sha256") !=
                REPAIR1_FAILURE_RECEIPT_SHA256)
            or review.get("repair_implementation_sha256") != repair_identity["sha256"]
            or review.get("material_blockers") != []):
        raise PermissionError("blocked: scoped P2 repair review does not bind current code")
    go = _require_repair_go(root, repair_identity["sha256"], version)
    manifest, recorded = _read_seal(root)
    card = root / version.card
    _require_unused_repair_card(card, version)
    state_path = card / "repair_execution_state.json"
    state = {"phase": version.phase, "status": "started", "failed_recorder_id": FAILED_RID,
             "failure_manifest_sha256": FAILURE_MANIFEST_SHA256, "plan_sha256": p2.PLAN_SHA256,
             "execution_authorization_sha256": go["sha256"], "source_artifacts_loaded": 0,
             "checkpoint_artifacts_requested": len(CHECKPOINT_NAMES),
             "checkpoint_artifacts_load_attempted": 0,
             "checkpoint_artifacts_loaded": 0, "current_checkpoint_artifact": None,
             "checkpoint_partial_load_unknown_on_exception": False,
             "checkpoint_stage_completed": False,
             "new_allocation_fit_attempted": 0,
             "OOF_prediction_calls_attempted": 0, "area_calculation_attempts": 0,
             "shadow_policy_path_attempts": 0, "shadow_policy_paths": 0,
             "local_label_generation_attempts": 0, "native_account_attempts": 0,
             "native_accounts": 0, "new_market_HTTP": 0, "new_market_source_queries": 0,
             "action_classifier_fits": 0, "evaluation_2023_plus_accounts": 0}
    p2._atomic_json(state_path, state)
    provider = None
    recorder = None
    try:
        import qlib
        from qlib.workflow import R
        from src.core.recorder_sources import load_verified_recorder_objects
        from src.data.path_area_source_preflight import frame_digest
        from src.core.native_evidence import finalize_evidence, prepare_groups
        from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
        from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
        from src.workflow.pit_constituent_auxiliary_training_allocation import AuxiliaryTrainingCandidate
        from src.model.minute_pit_allocator import TARGETS, MinutePitAllocator

        p1_plan = json.loads((root / plan["inherits"]["P1_plan"]["path"]).read_text())
        tariff = p1_plan["tariff"]
        if not isinstance(tariff, dict) or not tariff.get("tiers"):
            raise ValueError("blocked: frozen P1 tariff is absent or malformed")
        daily_provider = root / p1_plan["sources"]["daily_calendar"]["provider_uri"]
        qlib.init(provider_uri=str(daily_provider), region="us", kernels=1)
        def checkpoint_progress(event: str, name: str) -> None:
            _record_checkpoint_progress(state, state_path, event, name)

        checkpoint, checkpoint_audit = _load_failed_checkpoint(root, recorded,
                                                               progress=checkpoint_progress)
        if state["checkpoint_artifacts_loaded"] != len(CHECKPOINT_NAMES):
            raise ValueError("blocked: incomplete FAILED checkpoint load")
        state["checkpoint_stage_completed"] = True
        p2._atomic_json(state_path, state)
        barrier = checkpoint["barrier/fit_barrier"]
        new_models = {}
        for year in (2018, 2019):
            for seed in (71, 172, 273):
                model = checkpoint[f"models/new_{year}_{seed}"]
                if (type(model) is not MinutePitAllocator or model.seed != seed + year
                        or len(model.base_features) != 39 or model.extra_features
                        or model.spec.epochs != 300):
                    raise ValueError("blocked: restored six-state allocator differs")
                slot = next((row for row in barrier["completed_slots"]
                             if row["trade_year"] == year and row["base_seed"] == seed), None)
                if slot is None or p2._model_state_sha(model) != slot["state_sha256"]:
                    raise ValueError("blocked: restored fitted-state digest differs from barrier")
                new_models[(year, seed)] = model
        source_card = root / p2.P1_CARD
        expected = plan["source_admission"]["read_allowlist"]
        source_objects, source_audit = load_verified_recorder_objects(
            source_card, expected_recorder_id=plan["P1_evidence"]["recorder_id"],
            artifacts=list(expected), expected_artifact_sha256=expected,
            expected_manifest_sha256=plan["P1_evidence"]["manifest_sha256"])
        state["source_artifacts_loaded"] = 9
        p2._atomic_json(state_path, state)
        p2._check_loaded_source(source_objects, source_audit, plan)
        inputs = p2._fit_barrier_inputs(source_objects, plan)
        saved_models = p2._verify_saved_models(inputs, plan)
        predictions = _verify_saved_predictions(checkpoint, inputs)
        start = pd.Timestamp(plan["common_training_shadow"]["start_trade_date"])
        preliminary = inputs["timing"].loc[inputs["timing"].trade_year.between(2018, 2022)]
        terminal = pd.Timestamp(preliminary.exit_date.iloc[-1]).normalize()
        point_prices, provider, point_identity = p2._build_point_provider(root, inputs, plan, terminal)
        state["native_derived_point_provider_builds"] = 1
        state["native_derived_point_provider_readbacks"] = 1
        p2._atomic_json(state_path, state)
        h1_projection = {"dataset_artifact": "inputs/admitted_source_objects:PG_train_only/inputs/training_dataset",
            "columns": ["soxl_forward_open_return", "soxx_forward_open_return", "return_available_at"],
            "signal_index_sha256": p2._canonical_sha([str(value) for value in inputs["h1_timing"].index]),
            "two_original_H1_arrays_sha256": frame_digest(inputs["h1_returns"]),
            "five_original_targets_sha256_before": inputs["original_target_digest"],
            "five_original_targets_sha256_after": frame_digest(inputs["primary"].loc[:,
                [("label", name) for name in TARGETS]]), "price_derived_returns": False}
        if h1_projection["five_original_targets_sha256_before"] != h1_projection["five_original_targets_sha256_after"]:
            raise ValueError("blocked: original five target arrays changed")
        source_identity = {"P1_recorder_id": source_audit["recorder_id"],
            "P1_manifest_sha256": source_audit["verification_manifest_sha256"],
            "P1_artifacts": {name: row["sha256"] for name, row in source_audit["artifacts"].items()},
            "P1_fold_dataset_identity": {str(year): p2._dataframe_identity(prepare_groups(
                inputs["datasets"][f"fold_{year}"], "train")) for year in (2018, 2019)},
            "P1_empty_aux_identity": {str(year): p2._dataframe_identity(prepare_groups(
                inputs["datasets"][f"empty_aux_{year}"], "train")) for year in (2018, 2019)},
            "P1_input_view": inputs["source_frame_identity"],
            "P1_primary_training_frame": inputs["primary_frame_identity"],
            "P1_minute_frame": inputs["minute_frame_identity"],
            "P1_coverage": inputs["coverage_identity"], "H1_projection": h1_projection,
            "derived_point_provider": point_identity, "inherited_evidence": inherited,
            "failed_checkpoint": checkpoint_audit,
            "repair_implementation_identity_sha256": repair_identity["sha256"]}
        source_identity["phase_identity_sha256"] = p2._canonical_sha(source_identity)
        p1_binding = json.loads((source_card / "nas_binding.json").read_text(encoding="utf-8"))
        binding_path = card / "nas_binding.json"
        p2._atomic_json(binding_path, {"schema_version": "recorder_evidence_binding_v1",
            "authority": p1_binding["authority"], "ssh": p1_binding["ssh"],
            "source_id": source_identity["phase_identity_sha256"],
            "as_of": str(terminal.date()), "phase": version.phase, "plan_sha256": p2.PLAN_SHA256})
        recovery_spec = IntradayRecoveryDirectAllocationSpec(root=root, **{
            key: value for key, value in plan["nonlearned_policy_identity"]["recovery_spec"].items()
            if key in IntradayRecoveryDirectAllocationSpec.__dataclass_fields__ and key != "root"})
        action_spec = Stage1OpenActionValueSpec(root=root, **{
            key: value for key, value in plan["nonlearned_policy_identity"]["action_spec"].items()
            if key in Stage1OpenActionValueSpec.__dataclass_fields__ and key != "root"})
        candidate = AuxiliaryTrainingCandidate(name="pit_only_aux_005", auxiliary_weight=.05,
                                              include_union_backfill=False)
        tracking = card / "qlib_recorder"
        qlib.init(provider_uri=str(daily_provider), region="us", kernels=1)
        with R.start(experiment_name=version.experiment, recorder_name=version.recorder_name,
                     uri=str(tracking)):
            recorder = R.get_recorder()
            recorder.log_params(plan_sha256=p2.PLAN_SHA256,
                failure_manifest_sha256=FAILURE_MANIFEST_SHA256,
                failed_recorder_id=FAILED_RID, source_identity_sha256=source_identity["phase_identity_sha256"],
                repair_implementation_sha256=repair_identity["sha256"],
                new_allocation_fits=0, new_model_forward_calls=0, native_accounts=1)
            recorder.save_objects(artifact_path="sources", identity=source_identity,
                                  load_audit=source_audit, H1_projection=h1_projection,
                                  failed_checkpoint_audit=checkpoint_audit)
            code_snapshot = {name: (root / name).read_text(encoding="utf-8")
                             for name in repair_identity["files"]
                             if not name.startswith("environment:")}
            recorder.save_objects(artifact_path="implementation", code=code_snapshot,
                                  identity=repair_identity)
            recorder.save_objects(artifact_path="barrier", fit_barrier=barrier)
            for year in (2018, 2019):
                for seed in (71, 172, 273):
                    recorder.save_objects(artifact_path="models", **{
                        f"new_{year}_{seed}": new_models[(year, seed)]})
            for year in range(2018, 2023):
                for seed in (71, 172, 273):
                    key = f"year_{year}_seed_{seed}"
                    recorder.save_objects(artifact_path="OOF_seed_predictions", **{
                        key: checkpoint[f"OOF_seed_predictions/{key}"]})
            recorder.save_objects(artifact_path="OOF", raw_equal_weight_predictions=predictions)
            result = p2._complete_postfit(root=root, card=card, plan=plan, inputs=inputs,
                predictions=predictions, saved_models=saved_models, point_prices=point_prices,
                provider=provider, start=start, terminal=terminal, state=state,
                state_path=state_path, recorder=recorder, source_identity=source_identity,
                source_audit=source_audit, implementation_identity=repair_identity,
                implementation_review={"sha256": p2._sha_file(card / "repair_implementation_review.json")},
                h1_projection=h1_projection, fit_barrier=barrier, tracking=tracking,
                daily_provider=daily_provider, tariff=tariff, candidate=candidate,
                recovery_spec=recovery_spec, action_spec=action_spec,
                experiment_name=version.experiment, phase=version.phase)
            result.update(repair_of_failed_recorder=FAILED_RID,
                          repair_failure_manifest_sha256=FAILURE_MANIFEST_SHA256,
                          repair_implementation_sha256=repair_identity["sha256"],
                          prior_failed_run={"new_allocation_fits": 6,
                              "training_epochs": 1800, "OOF_model_forward_calls": 15,
                              "P1_artifact_loads": 9, "derived_provider_builds": 1,
                              "derived_provider_readbacks": 1,
                              "deterministic_smoothing_and_area_passes": 1},
                          repair_run={"new_allocation_fits": 0,
                              "new_model_forward_calls": 0,
                              "checkpoint_artifacts_loaded": len(CHECKPOINT_NAMES),
                              "P1_artifact_loads": 9,
                              "derived_provider_builds": 1, "derived_provider_readbacks": 1,
                              "deterministic_smoothing_and_area_passes": 1,
                              "physical_shadow_paths": 1, "native_accounts": 1},
                          cumulative_work={"new_allocation_fits": 6,
                              "training_epochs": 1800, "OOF_model_forward_calls": 15,
                              "P1_artifact_loads": 18, "derived_provider_builds": 2,
                              "derived_provider_readbacks": 2,
                              "deterministic_smoothing_and_area_passes": 2,
                              "physical_shadow_paths": 1, "native_accounts": 1})
            if version.number == 2:
                result["prior_repair_1"] = {"checkpoint_artifacts_loaded": 23,
                    "P1_artifact_loads": 0, "derived_provider_builds": 0,
                    "deterministic_smoothing_and_area_passes": 0,
                    "physical_shadow_paths": 0, "native_accounts": 0,
                    "failure_receipt_sha256": REPAIR1_FAILURE_RECEIPT_SHA256}
                result["cumulative_work"]["checkpoint_artifact_loads"] = 46
        result = finalize_evidence(recorder, card=card, summary=result,
            transport_binding=binding_path, workspace_root=root, include_code_artifacts=True)
        shutil.rmtree(provider)
        provider = None
        state.update(status="completed", recorder_id=str(recorder.id),
                     summary_sha256=p2._sha_file(card / "summary.json"),
                     verification_manifest_sha256=p2._sha_file(card / "verification_manifest.json"))
        p2._atomic_json(state_path, state)
        return result
    except BaseException as error:
        state.update(status="failed", exception_type=type(error).__name__, exception=str(error))
        if recorder is not None:
            state["recorder_id"] = str(recorder.id)
            try:
                recorder.save_objects(artifact_path="execution", failure=state)
            except BaseException:
                pass
        p2._atomic_json(state_path, state)
        if provider is not None:
            shutil.rmtree(provider, ignore_errors=True)
        raise


def run_path_area_bridge_repair(*, root: str | Path | None = None) -> dict[str, Any]:
    """Historical repair_1 entry; its sealed code/failed attempt remain archived."""
    return _run_repair(root=root, version=REPAIR_1)


def run_path_area_bridge_repair_2(*, root: str | Path | None = None) -> dict[str, Any]:
    """Run a separately reviewed, post-fit repair_2 only after its exact GO."""
    return _run_repair(root=root, version=REPAIR_2)
