"""Frozen P2 causal OOF path-area and local-action bridge.

The callable requires the separately issued P2 execution receipt. Its default
path is intentionally non-executing: design/implementation authorization alone
does not permit Recorder source loads, model fits, predictions, path statistics,
physical shadow simulation, or local labels.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.path_area_teacher import (
    clocked_smooth_targets,
    label_local_actions,
    run_training_shadow,
)
from src.data.path_area_bridge import (
    AREA_COLUMN,
    CONTROL_COLUMNS,
    PATH_COLUMNS,
    build_path_area_inputs,
    build_raw_training_point_prices,
)


PLAN_SHA256 = "db509d12808a4cccef4ddd539cbb0bf9918d1815726dcf0d0e9e7069d0438171"
IMPLEMENTATION_AUTH_SHA256 = "91b1b621f1809f0bbb2f9f8a9624361113ac39ccd9383c44163512907386130f"
DESIGN_REVIEW_SHA256 = "2663e6f5ad1a9917828de5956c5dcdbe5ce58525b2253836fe7808cf0057336d"
P1_CARD = Path("research/path_area_action_bridge_20261018/P1")
P2_CARD = Path("research/path_area_action_bridge_20261018/P2")
P1_ARTIFACTS = (
    "inputs/datasets",
    "inputs/admitted_source_objects",
    "inputs/daily_calendar",
    "analysis/fold_package",
    "analysis/model_eligibility",
    "analysis/membership_qualification",
    "analysis/minute_coverage",
    "analysis/corporate_qualification",
    "inputs/source_read_audits",
)
IMPLEMENTATION_CONTRACT = {
    "source_objects": "one verified P1 Recorder read of exactly the frozen nine-artifact allowlist",
    "H1_projection": {
        "dataset": "inputs/admitted_source_objects:PG_train_only/inputs/training_dataset",
        "columns": ["soxl_forward_open_return", "soxx_forward_open_return"],
        "clock": "the same native PG meta.return_available_at, aligned by original signal index",
        "guard": "reuse only the previous OOF row after return_available_at<=current 09:46 NY",
        "price_derived_returns": False,
    },
    "provider_marks": {
        "account_grid": "09:45 close; 09:47 fill from 09:48.open; 10:32 mark from 10:33.open; scheduled close",
        "executor_guard": "first XNYS session after account terminal, exact right-edge 09:31 $open for both assets stored at 09:30; missing/invalid stops",
        "executor_guard_scope": "provider calendar only; excluded from point_prices, orders, valuation, labels, and native comparison",
        "provider_builds": 1,
        "native_provider_readbacks": 1,
    },
    "decision_trace_mapping": {
        "source_target_*": "smoothed equal-weight original five-output OOF target before risk projection",
        "projected_target_*": "prepare_on_policy_open desired_open_weights after frozen deterministic risk guards",
        "postfill_*_weight": "physical post-settlement q/raw_mark, SOXX, and cash+receivable over postfill NAV",
        "filled_order_count": "number of actual nonzero settle_raw order fills at 09:47",
        "total_order_fees_usd": "sum of frozen Tiered fees plus 2bp slippage on filled orders",
        "points": "physical quantities, spendable cash, dividend receivable and NAV at each of four sparse native marks",
    },
    "fit_barrier": "all six fixed 2018/2019 new model states and native Dataset/scaler/target identities sealed before any OOF forward",
    "model_prediction_route": "three equal-weight seeds only on each P1 fold_package prediction_index selected by actual trade_year",
    "area": "fixed signed SOXX-x/SOXL-y 09:30-09:45 left-fold arithmetic; NaN controls and mask0 for missing prefix",
    "accounts": "one continuous 100000 USD 2bp physical teacher and one native reconciliation",
    "new_source_queries": 0,
    "action_classifier_fits": 0,
    "evaluation_2023_plus": 0,
    "P3": False,
}

IMPLEMENTATION_FILES = (
    "src/core/recorder_sources.py",
    "src/core/native_evidence.py",
    "src/analysis/decision_bridge.py",
    "src/data/path_area_bridge.py",
    "src/backtest/path_area_teacher.py",
    "src/workflow/path_area_bridge.py",
    "src/data/path_area_source_preflight.py",
    "src/model/minute_pit_allocator.py",
    "src/workflow/minute_pit_allocator.py",
    "src/backtest/raw_corporate_policy.py",
    "src/backtest/ibkr_tiered.py",
    "src/backtest/native_raw_corporate.py",
    "src/workflow/stage1_open_action_value.py",
    "src/workflow/intraday_recovery_direct_allocation.py",
    "src/workflow/pit_constituent_auxiliary_training_allocation.py",
    "src/data/minute_price_basis.py",
    "src/data/full_path_prices.py",
    "tests/test_path_area_bridge.py",
    "research/path_area_action_bridge_20261018/path_area_action_bridge_20261018.ipynb",
    "research/path_area_action_bridge_20261018/P2/workflow.yaml",
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str,
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return _sha_bytes(raw)


def _notebook_source_identity(path: Path) -> str:
    """Hash executable/document source cells, excluding all outputs and conclusion prose."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [{"id": cell.get("id"), "cell_type": cell["cell_type"],
              "source": cell.get("source", [])}
             for cell in notebook.get("cells", []) if cell.get("id") != "path-p2-conclusion"]
    return _canonical_sha(cells)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resolve_root(root: str | Path | None) -> Path:
    if root is not None:
        resolved = Path(root).resolve()
        if (resolved / "AGENTS.md").is_file() and (resolved / "SPEC.md").is_file():
            return resolved
        raise ValueError("blocked: explicit root is not the SOXL workspace")
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").is_file() and (current / "SPEC.md").is_file():
            return current
        current = current.parent
    raise ValueError("blocked: could not resolve the SOXL workspace root")


def _load_plan(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root / "research/path_area_action_bridge_20261018/P2_plan.json"
    if _sha_file(path) != PLAN_SHA256:
        raise ValueError("blocked: frozen P2 plan SHA-256 differs")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("phase") != "P2" or plan.get("frozen") is not True:
        raise ValueError("blocked: P2 plan is not frozen")
    return path, plan


def _require_execution_go(root: Path, plan: dict[str, Any], implementation_sha256: str,
                          implementation_review_sha256: str,
                          path: str | Path | None) -> dict[str, Any]:
    go_path = Path(path) if path is not None else root / P2_CARD / "P2_execution_authorization.json"
    if not go_path.is_absolute():
        go_path = root / go_path
    if not go_path.is_file():
        raise PermissionError("blocked: a separately reviewed P2_execution_authorization.json is required")
    go = json.loads(go_path.read_text(encoding="utf-8"))
    expected_budget = {
        "new_allocation_fits_maximum": 6,
        "saved_models_reused": 9,
        "common_shadow_policy_paths": 1,
        "common_shadow_native_accounts": 1,
        "action_classifier_fits": 0,
        "evaluation_2023_plus_forward_or_accounts": 0,
        "new_market_source_queries": 0,
        "new_market_HTTP": 0,
        "parent_artifact_loads_maximum": 9,
        "native_derived_point_provider_builds": 1,
        "native_derived_point_provider_readbacks": 1,
        "local_branch_pairs_maximum": plan["budgets"]["local_branch_pairs_maximum"],
    }
    if (go.get("approved") is not True or go.get("owner") != "/root"
            or go.get("scope") != "actual_P2_bridge_execution"
            or go.get("plan_sha256") != PLAN_SHA256
            or go.get("design_review_sha256") != DESIGN_REVIEW_SHA256
            or go.get("implementation_identity_sha256") != implementation_sha256
            or go.get("implementation_review_sha256") != implementation_review_sha256
            or go.get("budget") != expected_budget
            or go.get("P3") is not False):
        raise PermissionError("blocked: P2 execution receipt does not match the frozen scope and exact budgets")
    return {"path": str(go_path.resolve()), "sha256": _sha_file(go_path), "receipt": go}


def _verify_implementation_review(root: Path, plan: dict[str, Any], implementation: dict[str, Any]) -> dict[str, Any]:
    path = root / "research/path_area_action_bridge_20261018/P2_implementation_review.json"
    if not path.is_file():
        raise PermissionError("blocked: independent P2 implementation review receipt is missing")
    review = json.loads(path.read_text(encoding="utf-8"))
    if (review.get("verdict") != "APPROVE_P2_IMPLEMENTATION_FOR_SEPARATE_EXECUTION_GO"
            or review.get("plan_sha256") != PLAN_SHA256
            or review.get("design_review_sha256") != DESIGN_REVIEW_SHA256
            or review.get("implementation_identity_sha256") != implementation["sha256"]
            or review.get("material_blockers") != []):
        raise PermissionError("blocked: independent P2 implementation review is not bound to current code")
    return {"path": str(path), "sha256": _sha_file(path), "receipt": review}


def _verify_inherited(root: Path, plan: dict[str, Any]) -> dict[str, str]:
    verified = {}
    for item in plan["inherits"].values():
        if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
            continue
        path = root / item["path"]
        actual = _sha_file(path)
        if actual != item["sha256"]:
            raise ValueError("blocked: inherited P2 evidence changed: " + item["path"])
        verified[item["path"]] = actual
    review = root / "research/path_area_action_bridge_20261018/P2_design_review.json"
    if _sha_file(review) != DESIGN_REVIEW_SHA256:
        raise ValueError("blocked: scoped P2 design review changed")
    implementation_auth = root / "research/path_area_action_bridge_20261018/P2_implementation_authorization.json"
    if _sha_file(implementation_auth) != IMPLEMENTATION_AUTH_SHA256:
        raise ValueError("blocked: P2 implementation authorization changed")
    return verified


def _implementation_identity(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    identities = {}
    for relative in IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError("blocked: implementation source is missing: " + relative)
        identities[relative] = (_notebook_source_identity(path)
                                if path.suffix == ".ipynb" else _sha_file(path))
    for relative, expected in plan["nonlearned_policy_identity"]["source_code_sha256"].items():
        if identities.get(relative, _sha_file(root / relative)) != expected:
            raise ValueError("blocked: frozen nonlearned policy source identity changed: " + relative)
        identities.setdefault(relative, expected)
    p1_plan_path = root / plan["inherits"]["P1_plan"]["path"]
    p1_plan = json.loads(p1_plan_path.read_text(encoding="utf-8"))
    for relative, expected in p1_plan.get("source_clock_identity", {}).items():
        if _sha_file(root / relative) != expected:
            raise ValueError("blocked: frozen path source clock identity changed: " + relative)
        identities[relative] = expected
    authority = p1_plan["exchange_session_authority"]
    package = authority["package"]
    if importlib.metadata.version(package) != authority["version"]:
        raise ValueError("blocked: exchange-calendar package version differs from P1")
    for name, item in authority["module_identities"].items():
        path = Path(item["path"])
        if not path.is_file() or _sha_file(path) != item["sha256"]:
            raise ValueError("blocked: frozen XNYS schedule implementation changed: " + name)
        identities["environment:" + name] = item["sha256"]
    return {"files": identities, "sha256": _canonical_sha(identities)}


def current_path_area_implementation_identity(root: str | Path | None = None) -> dict[str, Any]:
    """Read-only code/environment identity for independent review and actual GO."""
    workspace = _resolve_root(root)
    _, plan = _load_plan(workspace)
    return _implementation_identity(workspace, plan)


def _flat_dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(frame.index, pd.MultiIndex) or list(frame.index.names) != ["datetime", "instrument"]:
        raise ValueError("blocked: prepared Qlib frame index differs")
    dates = pd.DatetimeIndex(frame.index.get_level_values("datetime"))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    if dates.has_duplicates:
        raise ValueError("blocked: original PG Dataset has duplicate signal dates")
    return dates


def _single_datetime_frame(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 2:
        raise ValueError("blocked: P1 DatasetH lost grouped columns")
    if group not in frame.columns.get_level_values(0):
        raise ValueError("blocked: P1 DatasetH group is missing: " + group)
    result = frame[group].copy()
    dates = _flat_dates(frame)
    result.index = dates
    return result


def _frame_digest(frame: pd.DataFrame) -> str:
    from src.data.path_area_source_preflight import frame_digest
    return frame_digest(frame)


def _model_state_sha(model: Any) -> str:
    digest = hashlib.sha256()
    state = model.network.state_dict()
    for name in sorted(state):
        array = state[name].detach().cpu().numpy()
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    digest.update(_canonical_sha({"primary_scaler": model.primary_scaler,
                                  "common_scaler": model.common_scaler,
                                  "extra_scaler": model.extra_scaler,
                                  "audit": model.audit,
                                  "metadata": model.state_metadata()}).encode("ascii"))
    return digest.hexdigest()


def _dataframe_identity(frame: pd.DataFrame) -> dict[str, Any]:
    return {"rows": len(frame), "columns": len(frame.columns), "sha256": _frame_digest(frame),
            "index_names": list(frame.index.names), "column_names": list(frame.columns.names)}


def _fit_barrier_inputs(objects: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Directly prepare only admitted native P1 objects; never query a provider."""
    from src.core.native_evidence import prepare_groups
    from src.model.minute_pit_allocator import TARGETS

    datasets = objects["inputs/datasets"]
    admitted = objects["inputs/admitted_source_objects"]
    pg = admitted["PG_train_only"]
    original = admitted["original_OOF_candidates"]
    if set(("inputs/training_dataset", "inputs/auxiliary_dataset")).difference(pg):
        raise ValueError("blocked: admitted PG training DatasetH objects are incomplete")
    primary = prepare_groups(pg["inputs/training_dataset"], "train")
    auxiliary = prepare_groups(pg["inputs/auxiliary_dataset"], "train")
    target_columns = [("label", column) for column in TARGETS]
    original_target_digest = _frame_digest(primary.loc[:, target_columns])
    if primary.shape != (3014, 85) or auxiliary.shape != (20741, 20):
        raise ValueError("blocked: admitted PG DatasetH shape differs from P1")

    timing_dataset = datasets["original_input_timing"]
    timing_source = prepare_groups(timing_dataset, "train")
    timing = _single_datetime_frame(timing_source, "meta")
    timing.index = pd.DatetimeIndex(timing.index)
    timing.index.name = "signal_date"
    fold_package = objects["analysis/fold_package"]
    fold_timing = fold_package["timing"].copy()
    fold_timing.index = pd.DatetimeIndex(fold_timing.index)
    fold_timing.index.name = "signal_date"
    if not isinstance(fold_timing.index, pd.DatetimeIndex) or not fold_timing.index.equals(timing.index):
        raise ValueError("blocked: P1 fold timing differs from original Qlib input index")
    for name in ("decision_at", "return_available_at", "label_available_at"):
        if name not in timing or not timing[name].equals(fold_timing[name]):
            raise ValueError("blocked: P1 metadata clock differs: " + name)
        if not isinstance(timing[name].dtype, pd.DatetimeTZDtype) or str(timing[name].dt.tz) != "America/New_York":
            raise ValueError("blocked: original PG clock lost America/New_York timezone: " + name)
    expected_decision = pd.to_datetime(fold_timing.trade_date).dt.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=29)
    expected_return = pd.to_datetime(fold_timing.exit_date).dt.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
    if not fold_timing.decision_at.equals(expected_decision) or not fold_timing.return_available_at.equals(expected_return):
        raise ValueError("blocked: P1 original allocation/H1 clocks differ from their pinned session mapping")
    pg_meta = _single_datetime_frame(primary, "meta")
    pg_labels = _single_datetime_frame(primary, "label")
    if not pg_meta.index.equals(timing.index) or not pg_labels.index.equals(timing.index):
        raise ValueError("blocked: PG H1 projection does not align to P1 input timing")
    h1_columns = ["soxl_forward_open_return", "soxx_forward_open_return"]
    h1_returns = pg_labels.loc[:, h1_columns].copy()
    h1_timing = fold_timing.copy()
    h1_timing.index = pd.DatetimeIndex(h1_timing.index)
    h1_timing.index.name = "signal_date"
    for column in h1_columns:
        h1_timing[column] = h1_returns[column].to_numpy()
    if not pg_meta["return_available_at"].equals(h1_timing["return_available_at"]):
        raise ValueError("blocked: projected original H1 availability differs from PG metadata")
    if _frame_digest(primary.loc[:, target_columns]) != original_target_digest:
        raise ValueError("blocked: original five target arrays changed during H1 projection")

    # P1's original full input frame contains 61 primary features, 16 common
    # features, five labels and three clocks. The P2 signal view retains only
    # feature/common/meta, exactly as admitted in the native DatasetH.
    full_timing = timing_source.copy()
    if len(full_timing) != 3014 or full_timing.shape[1] != 80:
        raise ValueError("blocked: P1 prediction-input view differs from the native 3014x80 contract")
    minute_frame = prepare_groups(datasets["bounded_core_minutes"], "all")
    coverage = objects["analysis/minute_coverage"].copy()
    if len(coverage) != 3014 or "prefix_mask" not in coverage:
        raise ValueError("blocked: P1 prefix coverage object differs")
    calendar = pd.DatetimeIndex(objects["inputs/daily_calendar"])
    if calendar.tz is not None or not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("blocked: P1 daily Qlib calendar differs")
    return {"datasets": datasets, "admitted": admitted, "primary": primary,
            "auxiliary": auxiliary, "timing_source": timing_source,
            "input_frame": full_timing, "timing": fold_timing,
            "h1_timing": h1_timing, "h1_returns": h1_returns,
            "original_target_digest": original_target_digest,
            "fold_package": fold_package, "minute_frame": minute_frame,
            "coverage": coverage, "calendar": calendar,
            "original_models": original, "model_eligibility": objects["analysis/model_eligibility"],
            "membership": objects["analysis/membership_qualification"],
            "corporate_qualification": objects["analysis/corporate_qualification"],
            "source_read_audits": objects["inputs/source_read_audits"],
            "source_frame_identity": _dataframe_identity(timing_source),
            "primary_frame_identity": _dataframe_identity(primary),
            "minute_frame_identity": _dataframe_identity(minute_frame),
            "coverage_identity": _dataframe_identity(coverage)}


def _check_loaded_source(objects: dict[str, Any], audit: dict[str, Any], plan: dict[str, Any]) -> None:
    expected = plan["source_admission"]["read_allowlist"]
    if set(expected) != set(P1_ARTIFACTS) or set(audit.get("artifacts", {})) != set(expected):
        raise ValueError("blocked: P1 source reader did not load the exact nine admitted artifacts")
    if audit.get("recorder_id") != plan["P1_evidence"]["recorder_id"]:
        raise ValueError("blocked: P1 source Recorder differs")
    if audit.get("verification_manifest_sha256") != plan["P1_evidence"]["manifest_sha256"]:
        raise ValueError("blocked: P1 verification manifest differs")
    if audit.get("hashes_checked_before_unpickling") is not True:
        raise ValueError("blocked: P1 source loader did not verify hashes before deserialization")
    for name, expected_hash in expected.items():
        if audit["artifacts"][name]["sha256"] != expected_hash:
            raise ValueError("blocked: P1 artifact identity differs: " + name)
    if objects.get("inputs/admitted_source_objects") is None or objects.get("inputs/datasets") is None:
        raise ValueError("blocked: P1 evidence payload is incomplete")


def _verify_saved_models(inputs: dict[str, Any], plan: dict[str, Any]) -> dict[tuple[int, int], Any]:
    from src.model.minute_pit_allocator import MinutePitAllocator

    expected = []
    for year in (2020, 2021, 2022):
        for seed in (71, 172, 273):
            expected.append((year, seed))
    qualification = inputs["model_eligibility"]
    by_key = {(int(row["year"]), int(row["base_seed"])): row for row in qualification}
    models = {}
    original = inputs["original_models"]
    for year, seed in expected:
        record = by_key.get((year, seed))
        if not record or record.get("static_qualified") is not True:
            raise ValueError("blocked: one saved OOF model lacks P1 static qualification")
        artifact = f"models/base_{year}_{seed}"
        model = original.get(artifact)
        if type(model) is not MinutePitAllocator or model.seed != seed + year:
            raise ValueError("blocked: saved original allocator identity differs")
        if model.extra_features or len(model.base_features) != 39 or model.spec.epochs != 300:
            raise ValueError("blocked: saved original allocator contract differs")
        models[(year, seed)] = model
    if len(models) != 9:
        raise ValueError("blocked: exact nine original OOF states are required")
    return models


def _fit_new_states(root: Path, inputs: dict[str, Any], plan: dict[str, Any], recorder: Any,
                    execution_state: dict[str, Any], state_path: Path) -> tuple[dict[tuple[int, int], Any], dict[str, Any]]:
    from src.model.minute_pit_allocator import MinutePitAllocator
    from src.core.native_evidence import prepare_groups

    datasets = inputs["datasets"]
    folds = inputs["fold_package"]["folds"]
    originals = inputs["original_models"]
    template = originals["models/base_2020_71"]
    expected_slots = [tuple((int(item["trade_year"]), int(item["base_seed"]), int(item["model_seed"])))
                      for item in plan["OOF_training"]["new_slots"]]
    if len(expected_slots) != 6 or len(set(expected_slots)) != 6:
        raise ValueError("blocked: frozen supplemental fit budget is not six unique slots")
    fitted, barrier_slots = {}, []
    for year, base_seed, model_seed in expected_slots:
        if model_seed != year + base_seed:
            raise ValueError("blocked: supplemental model seed differs from year+base_seed")
        fold = folds[year]
        train_dataset = datasets[f"fold_{year}"]
        auxiliary_dataset = datasets[f"empty_aux_{year}"]
        train_frame = prepare_groups(train_dataset, "train")
        aux_frame = prepare_groups(auxiliary_dataset, "train")
        target_frame = train_frame.loc[:, [("label", name) for name in __import__(
            "src.model.minute_pit_allocator", fromlist=["TARGETS"]).TARGETS]]
        target_digest = _frame_digest(target_frame)
        train_identity = _dataframe_identity(train_frame)
        aux_identity = _dataframe_identity(aux_frame)
        if target_digest != fold["five_targets_sha256"]:
            raise ValueError("blocked: P1 earlyfold original target digest differs")
        if aux_frame.shape != (0, 20) or len(aux_frame.columns) != 20:
            raise ValueError("blocked: earlyfold typed-empty auxiliary DatasetH changed")
        membership = inputs["membership"].get(year, inputs["membership"].get(str(year), {}))
        if membership.get("status") != "proven_empty":
            raise ValueError("blocked: earlyfold empty-PIT qualification differs")
        spec = asdict(template.spec)
        spec["root"] = str(root)
        cutoff = pd.Timestamp(fold["cutoff"])
        model = MinutePitAllocator(base_features=template.base_features,
            extra_features=[], feature_masks=template.feature_masks, spec=spec,
            seed=model_seed, cutoff=cutoff.isoformat(), train_end=pd.Timestamp(fold["train_end"]).isoformat(),
            minimum_observations=template.minimum_observations)
        execution_state["new_allocation_fit_attempted"] += 1
        execution_state["current_fit_in_progress"] = {"trade_year": year, "base_seed": base_seed,
            "model_seed": model_seed, "epochs": int(model.spec.epochs),
            "exact_epochs_completed_if_interrupted": "unknown"}
        execution_state["incomplete_fit_forward_count_unknown"] = True
        _atomic_json(state_path, execution_state)
        # One call per frozen slot; an exception is terminal and never retried.
        try:
            model.fit(train_dataset, auxiliary_dataset=auxiliary_dataset)
        except BaseException as error:
            partial = {"trade_year": year, "base_seed": base_seed, "model_seed": model_seed,
                       "fit_attempted": True, "fit_completed": False,
                       "exception_type": type(error).__name__, "exception": str(error),
                       "epochs_completed_exact": "unknown"}
            recorder.save_objects(artifact_path="fit_attempts",
                                  **{f"failed_{year}_{base_seed}": {"model": model, "receipt": partial}})
            raise
        execution_state["new_allocation_fit_completed"] += 1
        execution_state["training_forward_calls_completed"] += int(model.spec.epochs)
        execution_state["current_fit_in_progress"] = None
        execution_state["incomplete_fit_forward_count_unknown"] = False
        _atomic_json(state_path, execution_state)
        if model.primary_scaler != fold["primary_scaler"] or model.common_scaler != fold["common_scaler"]:
            raise ValueError("blocked: earlyfold fitted scaler differs from the P1 fixed scaler")
        if int(model.audit["training_rows"]) != int(fold["training_rows"]):
            raise ValueError("blocked: earlyfold training rows differ from P1 eligibility")
        state_hash = _model_state_sha(model)
        key = (year, base_seed)
        fitted[key] = model
        recorder.save_objects(artifact_path="models", **{f"new_{year}_{base_seed}": model})
        barrier_slots.append({"trade_year": year, "base_seed": base_seed, "model_seed": model_seed,
            "status": "fit_completed", "state_sha256": state_hash,
            "train_dataset": train_identity, "empty_aux_dataset": aux_identity,
            "five_targets_sha256": target_digest,
            "primary_scaler_matches_P1": True, "common_scaler_matches_P1": True,
            "training_rows": int(model.audit["training_rows"]),
            "audit": model.audit})
    if len(fitted) != 6 or execution_state["new_allocation_fit_completed"] != 6:
        raise ValueError("blocked: all six fixed supplemental fits must complete before predictions")
    barrier = {"status": "passed", "plan_sha256": PLAN_SHA256,
        "expected_slots": [{"trade_year": y, "base_seed": b, "model_seed": s} for y, b, s in expected_slots],
        "completed_slots": barrier_slots, "fitted_state_count": len(fitted),
        "maximum_new_fits": 6, "partial_ensemble_permitted": False,
        "all_states_sealed_before_prediction": True}
    barrier["sha256"] = _canonical_sha(barrier)
    recorder.save_objects(artifact_path="barrier", fit_barrier=barrier,
                          train_dataset_identity={str(y): next(x["train_dataset"] for x in barrier_slots if x["trade_year"] == y)
                                                  for y in (2018, 2019)},
                          empty_aux_dataset_identity={str(y): next(x["empty_aux_dataset"] for x in barrier_slots if x["trade_year"] == y)
                                                      for y in (2018, 2019)})
    _atomic_json(state_path.parent / "fit_barrier.json", barrier)
    execution_state["fit_barrier_sha256"] = barrier["sha256"]
    execution_state["fit_barrier_status"] = "passed"
    _atomic_json(state_path, execution_state)
    return fitted, barrier


def _predict_oof(inputs: dict[str, Any], saved_models: dict[tuple[int, int], Any],
                 new_models: dict[tuple[int, int], Any], plan: dict[str, Any], *,
                 recorder: Any, execution_state: dict[str, Any], state_path: Path) -> pd.DataFrame:
    from src.core.native_evidence import native_dataset, prepare_groups
    from src.model.minute_pit_allocator import _flat_index

    timing = inputs["timing"]
    route = {year: inputs["fold_package"]["folds"][year]["prediction_index"]
             for year in (2018, 2019, 2020, 2021, 2022)}
    frame = inputs["input_frame"]
    dates = pd.DatetimeIndex(frame.index.get_level_values("datetime"))
    complete = {}
    all_outputs = []
    for year, index in route.items():
        target_dates = pd.DatetimeIndex(index.get_level_values("datetime")) if isinstance(index, pd.MultiIndex) else pd.DatetimeIndex(index)
        if not len(target_dates) or not target_dates.isin(timing.index).all():
            raise ValueError("blocked: P1 frozen trade-year prediction index is unavailable")
        if not timing.loc[target_dates, "trade_year"].eq(year).all():
            raise ValueError("blocked: P1 fold prediction index is not routed by actual trade_year")
        selected = frame.loc[dates.isin(target_dates)].copy()
        if len(selected) != len(target_dates):
            raise ValueError("blocked: routed native input rows differ from P1 fold index")
        dataset = native_dataset(selected, segments={"predict": (None, None)})
        seed_outputs = []
        for base_seed in (71, 172, 273):
            model = new_models[(year, base_seed)] if year in (2018, 2019) else saved_models[(year, base_seed)]
            slot = {"trade_year": year, "base_seed": base_seed}
            execution_state["OOF_prediction_calls_attempted"] += 1
            execution_state["current_OOF_prediction_slot"] = slot
            _atomic_json(state_path, execution_state)
            try:
                prediction = model.predict(dataset, segment="predict")
            except BaseException as error:
                partial = {**slot, "status": "failed", "exception_type": type(error).__name__,
                           "exception": str(error), "call_attempted": True}
                try:
                    recorder.save_objects(artifact_path="OOF_attempts",
                        **{f"failed_{year}_{base_seed}": partial})
                except BaseException:
                    pass
                finally:
                    execution_state["current_OOF_prediction_slot"] = None
                    _atomic_json(state_path, execution_state)
                raise
            execution_state["OOF_prediction_calls_completed"] += 1
            execution_state["OOF_prediction_rows"] += len(prediction)
            execution_state["current_OOF_prediction_slot"] = None
            _atomic_json(state_path, execution_state)
            recorder.save_objects(artifact_path="OOF_seed_predictions",
                **{f"year_{year}_seed_{base_seed}": prediction})
            prediction = _flat_index(prediction)
            if prediction.index.has_duplicates or not prediction.index.isin(target_dates).all():
                raise ValueError("blocked: model output exceeded its actual trade-year rows")
            if set(prediction.index) != set(target_dates):
                raise ValueError("blocked: model prediction index differs from P1 trade-year rows")
            prediction = prediction.reindex(target_dates)
            seed_outputs.append(prediction)
        ensemble = _equal_weight_ensemble(seed_outputs)
        complete[year] = ensemble
        all_outputs.append(ensemble)
    prediction = pd.concat(all_outputs).sort_index()
    if prediction.index.has_duplicates or set(prediction.index) != set(timing.index[timing.trade_year.between(2018, 2022)]):
        raise ValueError("blocked: OOF output calendar differs from admitted trade-year rows")
    if any(pd.Timestamp(value).year >= 2023 for value in prediction.index):
        raise ValueError("blocked: a P2 model forward reached 2023 or later")
    return prediction.loc[:, ["soxl_weight", "soxx_weight", "cash_weight", "tail_probability", "alpha_probability"]]


def _equal_weight_ensemble(seed_outputs: list[pd.DataFrame]) -> pd.DataFrame:
    """Original three-seed P2 arithmetic, shared by forward and saved-output reads."""
    if len(seed_outputs) != 3 or any(not seed_outputs[0].index.equals(part.index)
                                     for part in seed_outputs[1:]):
        raise ValueError("blocked: three-seed prediction calendars differ")
    ensemble = sum(seed_outputs) / 3.0
    weights = ["soxl_weight", "soxx_weight", "cash_weight"]
    weight_sum = ensemble.loc[:, weights].sum(axis=1)
    if not np.isfinite(ensemble.to_numpy(dtype=float)).all() or (weight_sum <= 0).any():
        raise ValueError("blocked: original five-output ensemble is nonfinite")
    ensemble.loc[:, weights] = ensemble.loc[:, weights].div(weight_sum, axis=0)
    ensemble = ensemble.copy()
    ensemble.index.name = "signal_date"
    return ensemble


def _event_views(admitted: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = admitted["corporate_events_metadata"]
    cash = events["data/cash_events.pkl"].copy()
    split = events["data/split_events.pkl"].copy()
    cash["ex_date"] = pd.to_datetime(cash.ex_date).dt.tz_localize(None).dt.normalize()
    split["effective_trading_date"] = pd.to_datetime(split.effective_trading_date).dt.tz_localize(None).dt.normalize()
    cash = cash.loc[cash.instrument.isin(("SOXL", "SOXX")) & cash.ex_date.between(start, end)].copy()
    split = split.loc[split.instrument.isin(("SOXL", "SOXX"))
                      & split.effective_trading_date.between(start, end)].copy()
    return cash, split


def _point_provider_identity(provider: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(item for item in provider.rglob("*") if item.is_file()):
        entries.append({"path": path.relative_to(provider).as_posix(), "sha256": _sha_file(path),
                        "bytes": path.stat().st_size})
    return {"entries": entries, "sha256": _canonical_sha(entries)}


def _build_point_provider(root: Path, inputs: dict[str, Any], plan: dict[str, Any],
                          terminal: pd.Timestamp) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    import exchange_calendars as xc
    import qlib
    from qlib.data import D
    from src.data import build_qlib_store_from_frame

    start = pd.Timestamp(plan["common_training_shadow"]["start_trade_date"])
    p1_plan = json.loads((root / plan["inherits"]["P1_plan"]["path"]).read_text())
    authority = p1_plan["exchange_session_authority"]
    cal = xc.get_calendar(authority["calendar"], start=str(start.date()), end=str(terminal.date()))
    session_schedule = cal.schedule.loc[start:terminal]
    points = build_raw_training_point_prices(inputs["minute_frame"], session_schedule,
                                              start=start, end=terminal)
    raw = inputs["minute_frame"]["feature"]
    p1_plan = json.loads((root / plan["inherits"]["P1_plan"]["path"]).read_text())
    authority = p1_plan["exchange_session_authority"]
    calendar = xc.get_calendar(authority["calendar"], start=str(terminal.date()),
                               end=authority["schedule_end"]).schedule
    session_index = pd.DatetimeIndex(calendar.index).tz_localize(None).normalize()
    next_sessions = session_index[session_index > terminal]
    if not len(next_sessions):
        raise ValueError("needs_data: no next XNYS session inside the admitted P1 source interval")
    guard_day = next_sessions[0]
    guard_source_stamp = guard_day + pd.Timedelta(hours=9, minutes=31)
    guard_stamp = guard_day + pd.Timedelta(hours=9, minutes=30)
    guard_values = {}
    for symbol in ("SOXL", "SOXX"):
        key = (guard_source_stamp, symbol)
        if key not in raw.index:
            raise ValueError("needs_data: first next-session raw open is missing for the Qlib guard")
        row = raw.loc[key]
        price = row["$open"]
        if int(row["$price_valid"]) != 1 or not np.isfinite(price) or float(price) <= 0:
            raise ValueError("needs_data: first next-session raw open is invalid for the Qlib guard")
        guard_values[symbol] = float(np.float64(price))
    if guard_stamp is None or guard_values is None or guard_stamp <= points.index[-1]:
        raise ValueError("needs_data: no admitted next raw open is available for the Qlib executor guard")
    if guard_source_stamp.year > 2022 or guard_stamp.year > 2022:
        raise ValueError("blocked: executor guard would extend beyond the frozen P2 training source")
    records = []
    for stamp, row in points.iterrows():
        for symbol in ("SOXL", "SOXX"):
            price = float(row[symbol])
            records.append({"datetime": stamp, "instrument": symbol, "open": price,
                "high": price, "low": price, "close": price,
                "execution_price": price, "raw_execution_price": price})
    for symbol in ("SOXL", "SOXX"):
        price = guard_values[symbol]
        records.append({"datetime": guard_stamp, "instrument": symbol, "open": price,
            "high": price, "low": price, "close": price,
            "execution_price": price, "raw_execution_price": price})
    provider = Path(tempfile.mkdtemp(prefix="soxl-path-area-p2-provider-"))
    manifest = build_qlib_store_from_frame(pd.DataFrame(records), provider_uri=provider,
        freq="1min", region="us", as_of=str(guard_stamp.date()),
        field_map={name: name for name in ("open", "high", "low", "close", "execution_price", "raw_execution_price")},
        source="P1 bounded_core_minutes DatasetH; P2 scheduled raw event marks")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    observed = D.features(["SOXL", "SOXX"], ["$raw_execution_price"],
        start_time=points.index[0], end_time=guard_stamp, freq="1min", disk_cache=0)
    expected_all = points.copy()
    expected_all.loc[guard_stamp, ["SOXL", "SOXX"]] = [guard_values["SOXL"], guard_values["SOXX"]]
    expected_all = expected_all.sort_index()
    readback = observed["$raw_execution_price"].unstack("instrument").reindex(
        index=expected_all.index, columns=["SOXL", "SOXX"])
    expected = expected_all.astype(np.float32).astype(np.float64)
    pd.testing.assert_frame_equal(readback.astype(np.float64), expected,
                                  check_names=False, check_exact=True)
    identity = {"manifest": manifest, "provider_tree": _point_provider_identity(provider),
        "rows": len(points), "first_point": str(points.index[0]), "last_point": str(points.index[-1]),
        "native_readback": "passed_exact_float32", "provider_path": str(provider),
        "provider_build_count": 1, "derived_only_from_P1_bounded_core_minutes": True,
        "native_readback_count": 1,
        "executor_guard": {"policy_timestamp": str(guard_stamp),
            "source_right_edge_timestamp": str(guard_source_stamp),
            "raw_open_values": guard_values, "from_admitted_next_raw_quote": True,
            "outside_account_interval": True, "outside_point_prices": True,
            "excluded_from_orders_valuation_labels_and_native_comparison": True}}
    return points, provider, identity


def _build_label_dataset(shadow: dict[str, Any], labels: pd.DataFrame):
    from src.core.native_evidence import native_dataset
    from src.workflow.stage1_open_action_value import ACTION_FEATURES

    decisions = shadow["decisions"].copy()
    if not decisions.index.equals(labels.index):
        raise ValueError("blocked: local label and common teacher rows differ")
    features = decisions.loc[:, list(ACTION_FEATURES) + list(PATH_COLUMNS) + ["prefix_mask"]].copy()
    outcome = pd.DataFrame({"local_execute_delta": labels["execute_delta"]}, index=labels.index)
    meta = pd.DataFrame({"trade_date": pd.to_datetime(labels.index),
                         "label_available_at": labels["label_available_at"],
                         "prefix_mask": labels["prefix_mask"].astype("int8")}, index=labels.index)
    signal_dates = pd.DatetimeIndex(decisions.signal_date)
    instrument = ["SOXL_SOXX_CASH"] * len(decisions)
    groups = pd.concat({"feature": features, "label": outcome, "meta": meta}, axis=1)
    groups.index = pd.MultiIndex.from_arrays([signal_dates, instrument], names=["datetime", "instrument"])
    groups = groups.sort_index()
    dataset = native_dataset(groups, segments={"all": (None, None)})
    return dataset, groups


def _support_report(decisions: pd.DataFrame, labels: pd.DataFrame, plan: dict[str, Any]) -> dict[str, Any]:
    cutoff = pd.Timestamp("2023-01-01 00:00", tz="America/New_York")
    report = {}
    merged = decisions.join(labels[["label_status", "label_available_at", "execute_delta",
                                   "next_common_teacher_target_present"]],
                            rsuffix="_label")
    for year in range(2018, 2023):
        rows = merged.loc[pd.DatetimeIndex(merged.index).year == year]
        report[str(year)] = {
            "shadow_rows": int(len(rows)),
            "eligible_rows": int(rows.eligible.astype(bool).sum()),
            "mandatory_derisk_rows": int(rows.mandatory_derisk.astype(bool).sum()),
            "large_transition_rows": int(rows.large_transition.astype(bool).sum()),
            "mandatory_position_repair_rows": int(rows.mandatory_position_repair.astype(bool).sum()),
            "nonfinite_original22_rows": int(rows.eligibility_reasons.map(lambda x: "nonfinite_original22" in x).sum()),
            "prefix_mask_1_rows": int(rows.prefix_mask.eq(1).sum()),
            "prefix_mask_0_rows": int(rows.prefix_mask.eq(0).sum()),
            "next_common_teacher_target_rows": int(rows.next_common_teacher_target_present.astype(bool).sum()),
            "complete_prefix_marks_next_target_intersection": int((rows.prefix_mask.eq(1)
                & rows.next_common_teacher_target_present.astype(bool)).sum()),
            "mature_action_labels": int(rows.label_status.eq("mature").sum()),
            "unknown_action_labels": int(rows.label_status.eq("unknown").sum()),
            "label_available_rows": int(rows.label_available_at.notna().sum()),
            "label_available_before_2023": int((rows.label_available_at.notna()
                & (rows.label_available_at < cutoff)).sum()),
        }
    pre2022 = merged.loc[pd.DatetimeIndex(merged.index).year < 2022]
    eligible_mature = pre2022.loc[
        pre2022.eligible.astype(bool)
        & pre2022.label_status.eq("mature")
        & np.isfinite(pre2022.execute_delta.to_numpy(dtype=float))
        & pre2022.label_available_at.lt(cutoff)
    ]
    minimum = int(plan["support_and_stop"]["minimum_pre2022_eligible_mature_action_rows"])
    return {"by_trade_year": report,
            "pre2022_eligible_mature_action_rows": int(len(eligible_mature)),
            "minimum_required": minimum,
            "gate": "count_only_pass" if len(eligible_mature) >= minimum else "needs_data_below_fixed_minimum",
            "prefix_mask_required_for_label_count": False,
            "classification_fits": 0,
            "economic_or_independent_validation_claim": False}


def _execution_state(card: Path, go: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    state_path = card / "P2_execution_state.json"
    if state_path.exists():
        raise FileExistsError("blocked: P2 execution state already exists; no automatic resume or retry")
    state = {"phase": "P2", "status": "started", "plan_sha256": PLAN_SHA256,
        "execution_authorization_sha256": go["sha256"], "source_artifact_request_count": 9,
        "source_loader_calls_attempted": 0, "source_artifacts_loaded": 0,
        "source_artifact_load_count_unknown_on_exception": False,
        "new_allocation_fit_attempted": 0,
        "new_allocation_fit_completed": 0, "saved_models_reused": 0,
        "OOF_prediction_calls": 0, "OOF_prediction_calls_attempted": 0,
        "OOF_prediction_calls_completed": 0, "current_OOF_prediction_slot": None,
        "OOF_prediction_rows": 0,
        "training_forward_calls_completed": 0, "current_fit_in_progress": None,
        "incomplete_fit_forward_count_unknown": False,
        "area_rows": 0, "label_pairs": 0, "shadow_policy_paths": 0,
        "area_calculation_attempts": 0, "area_calculation_status": "not_started",
        "local_label_generation_attempts": 0, "local_label_generation_status": "not_started",
        "shadow_policy_path_attempts": 0, "native_account_attempts": 0,
        "native_accounts": 0, "new_market_HTTP": 0, "new_market_source_queries": 0,
        "native_derived_point_provider_builds": 0, "native_derived_point_provider_readbacks": 0,
        "action_classifier_fits": 0, "evaluation_2023_plus_accounts": 0,
        "source_loaded": False, "P3_authorized": False}
    _atomic_json(state_path, state)
    return state, state_path


def _complete_postfit(*, root: Path, card: Path, plan: dict[str, Any], inputs: dict[str, Any],
                      predictions: pd.DataFrame, saved_models: dict[tuple[int, int], Any],
                      point_prices: pd.DataFrame, provider: Path, start: pd.Timestamp,
                      terminal: pd.Timestamp, state: dict[str, Any], state_path: Path,
                      recorder: Any, source_identity: dict[str, Any], source_audit: dict[str, Any],
                      implementation_identity: dict[str, Any], implementation_review: dict[str, Any],
                      h1_projection: dict[str, Any], fit_barrier: dict[str, Any],
                      tracking: Path, daily_provider: Path, tariff: dict[str, Any],
                      candidate: Any, recovery_spec: Any, action_spec: Any,
                      experiment_name: str, phase: str) -> dict[str, Any]:
    """Continue one frozen P2 run after predictions; no fit or model forward path."""
    from copy import deepcopy
    from src.backtest.native_raw_corporate import native_raw_validate
    from src.backtest.raw_corporate_policy import make_schedule
    from src.analysis.decision_bridge import build_decision_bridge
    transport_binding = card / "nas_binding.json"
    oof_h1_timing = inputs["h1_timing"].loc[predictions.index].copy()
    smoothed = clocked_smooth_targets(predictions, oof_h1_timing,
                                      spec=saved_models[(2020, 71)].spec, candidate=candidate)
    timing_by_signal = inputs["timing"].copy()
    timing_by_signal.index.name = "signal_date"
    teacher = smoothed.reset_index().merge(timing_by_signal.reset_index(),
        left_on="signal_date", right_on="signal_date", how="left", validate="one_to_one",
        suffixes=("", "_source"))
    state["area_calculation_attempts"] += 1
    state["area_calculation_status"] = "in_progress"
    _atomic_json(state_path, state)
    try:
        path_inputs = build_path_area_inputs(inputs["minute_frame"], timing_by_signal,
                                             inputs["coverage"])
    except BaseException:
        state["area_calculation_status"] = "failed_partial_row_count_unknown_no_retry"
        _atomic_json(state_path, state)
        raise
    teacher = teacher.merge(path_inputs.reset_index(), on=["signal_date", "trade_date"],
                            how="left", validate="one_to_one")
    if len(teacher) != len(smoothed) or teacher.trade_date.isna().any():
        raise ValueError("blocked: causal target/path join differs from P1 OOF grid")
    state["area_rows"] = int(teacher.prefix_mask.sum())
    state["area_calculation_status"] = "completed"
    _atomic_json(state_path, state)

    cash_events, split_events = _event_views(inputs["admitted"], start, terminal)
    schedule = make_schedule(cash_events, split_events,
        pd.DatetimeIndex(inputs["calendar"][(inputs["calendar"] >= start) & (inputs["calendar"] <= terminal)]),
        start, terminal)
    import exchange_calendars as xc
    p1_plan = json.loads((root / plan["inherits"]["P1_plan"]["path"]).read_text())
    session_schedule = xc.get_calendar(p1_plan["exchange_session_authority"]["calendar"],
        start=str(start.date()), end=str(terminal.date())).schedule.loc[start:terminal]
    state["shadow_policy_path_attempts"] += 1
    _atomic_json(state_path, state)
    shadow = run_training_shadow(teacher, point_prices=point_prices, schedule=schedule,
        sessions=pd.DatetimeIndex(session_schedule.index).tz_localize(None),
        recovery_spec=recovery_spec, action_spec=action_spec,
        tariff=tariff, initial_cash=float(plan["common_training_shadow"]["initial_cash_usd"]),
        slippage_bps=float(plan["common_training_shadow"]["slippage_bps"]),
        start_trade_date=plan["common_training_shadow"]["start_trade_date"])
    state["shadow_policy_paths"] = 1
    _atomic_json(state_path, state)
    state["local_label_generation_attempts"] += 1
    state["local_label_generation_status"] = "in_progress"
    _atomic_json(state_path, state)
    try:
        labels = label_local_actions(shadow, point_prices=point_prices, schedule=schedule,
        sessions=pd.DatetimeIndex(session_schedule.index).tz_localize(None),
        cutoff=pd.Timestamp("2023-01-01 00:00", tz="America/New_York"))
    except BaseException:
        state["local_label_generation_status"] = "failed_partial_pair_count_unknown_no_retry"
        _atomic_json(state_path, state)
        raise
    state["local_label_generation_status"] = "completed"
    state["label_pairs"] = int(labels.label_status.eq("mature").sum())
    _atomic_json(state_path, state)
    label_dataset, label_frame = _build_label_dataset(shadow, labels)
    support = _support_report(shadow["decisions"], labels, plan)
    prebranch_state = []
    for day, context in shadow["_branch_contexts"].items():
        ledger = context["state"]["income"]
        fee_state = context["state"]["fees"]
        prebranch_state.append({"trade_date": day, "signal_date": context["signal_date"],
            "q_soxl": float(context["state"]["q"][0]), "q_soxx": float(context["state"]["q"][1]),
            "cash": float(context["state"]["cash"]), "receivable": ledger.value,
            "receivables": deepcopy(ledger.receivables), "applied_event_ids": sorted(ledger.applied),
            "monthly_fee_shares": dict(fee_state.monthly_shares),
            "prebranch_nav": float(context["prebranch_nav"]), "target": context["target"].copy()})
    recorder.save_objects(artifact_path="shadow_python", decisions=shadow["decisions"],
        targets=shadow["targets"], trace=shadow["trace"], replay=shadow["replay"],
        orders=shadow["orders"], points=shadow["points"],
        corporate_receipts=shadow["corporate_receipts"], prebranch_state=prebranch_state,
        python_audit=shadow["audit"], path_area_inputs=teacher.loc[:,
            ["signal_date", "trade_date", *PATH_COLUMNS, "prefix_mask"]])
    recorder.save_objects(artifact_path="local_labels", labels=labels,
        action_label_dataset=label_dataset, action_label_frame=label_frame)
    state["python_shadow_artifacts_saved"] = True
    state["local_label_artifacts_saved"] = True
    _atomic_json(state_path, state)

    state["native_account_attempts"] += 1
    _atomic_json(state_path, state)
    native = native_raw_validate(shadow, provider_uri=provider, point_prices=point_prices,
        schedule=schedule, tariff=tariff,
        cost=float(plan["common_training_shadow"]["slippage_bps"]),
        initial_cash=float(plan["common_training_shadow"]["initial_cash_usd"]))
    if native["audit"].get("status") != "passed":
        raise ValueError("blocked: native Qlib physical-shadow reconciliation did not pass")
    state["native_accounts"] = 1
    _atomic_json(state_path, state)
    shadow["audit"] = {**shadow["audit"], **native["audit"],
        "python_to_native_status": "passed", "certifies_alpha": False,
        "certifies_PIT_or_first_receipt": False}
    bridge_trace = shadow["trace"].reset_index().copy()
    bridge_trace["eligibility_reasons"] = bridge_trace["eligibility_reasons"].map(
        lambda value: "|".join(value) if isinstance(value, (list, tuple)) else str(value))
    bridge_source_identity = {**source_identity,
        "current_P2_recorder_id": str(recorder.id),
        "current_P2_experiment_id": str(recorder.experiment_id)}
    decision_bridge = build_decision_bridge(
        bridge_trace, native["authority"], native_audit=native["audit"],
        source_identity=bridge_source_identity,
        fields={"source_target": ["source_target_soxl_weight", "source_target_soxx_weight", "source_target_cash_weight"],
            "projected_target": ["projected_target_soxl_weight", "projected_target_soxx_weight", "projected_target_cash_weight"],
            "postfill_weights": ["postfill_soxl_weight", "postfill_soxx_weight", "postfill_cash_weight"],
            "order_count": "filled_order_count", "fees_usd": "total_order_fees_usd"},
        native_calendar=inputs["calendar"])
    recorder.save_objects(artifact_path="native", native_report=native["native_report"],
        native_positions=native["native_positions"], native_orders=native["native_orders"],
        native_points=native["points"], native_corporate_receipts=native["native_corporate_receipts"],
        native_reconciliation=native["reconciliation"], native_audit=native["audit"],
        teacher_native_audit=shadow["audit"],
        decision_bridge_daily=decision_bridge["daily"],
        decision_bridge_periods=decision_bridge["periods"],
        decision_bridge_rolling_126=decision_bridge["rolling_126"])
    repair = phase.startswith("P2_repair_")
    recorder.log_metrics(completed_new_allocation_fits=0 if repair else 6,
        reused_original_models=9, oof_prediction_rows=0 if repair else len(predictions),
        checkpoint_oof_prediction_rows=len(predictions) if repair else 0,
        shadow_sessions=len(shadow["replay"]), native_reconciliation_pass=1,
        mature_action_labels=int(support["pre2022_eligible_mature_action_rows"]))
    result = {"experiment": experiment_name, "phase": phase,
        "status": "completed_training_bridge_no_economic_evaluation",
        "decision": support["gate"], "plan_sha256": PLAN_SHA256,
        "design_review_sha256": DESIGN_REVIEW_SHA256,
        "implementation_review_sha256": implementation_review["sha256"],
        "source_identity_sha256": source_identity["phase_identity_sha256"],
        "implementation_identity_sha256": implementation_identity["sha256"],
        "region": "us", "recorder_uri": str(tracking),
        "provider_uri": {"P1_daily_calendar": str(daily_provider),
            "source_dataset_authority": "verified P1 Recorder native DatasetH objects",
            "derived_native_point_provider": "temporary P2 provider; removed after one exact Qlib readback and native reconciliation"},
        "native_dataset_authority": {"P1_training_inputs": "P1 verified Recorder inputs/datasets and admitted PG training DatasetH",
            "P2_action_labels": "native DatasetH saved in this P2 Recorder",
            "derived_provider_status": "temporary execution input deleted after reconciliation; manifest/readback/report/positions retained"},
        "P1_recorder_id": source_audit["recorder_id"],
        "P1_manifest_sha256": source_audit["verification_manifest_sha256"],
        "source_status": plan["source_admission"]["mode"],
        "source_snapshot_lineage": {"source_byte_identity_sha256": plan["P1_evidence"]["source_byte_identity_sha256"],
            "P1_archive_sha256": plan["P1_evidence"]["archive_sha256"],
            "P1_manifest_sha256": plan["P1_evidence"]["manifest_sha256"],
            "new_market_HTTP": 0, "new_market_source_queries": 0},
        "claims_false": plan["source_admission"]["claims_false"],
        "fit_barrier_sha256": fit_barrier["sha256"],
        "new_allocation_fits": 0 if repair else 6, "reused_models": 9,
        "new_training_epochs": 0 if repair else 1800,
        "training_forward_calls_completed": 0 if repair else 1800,
        "OOF_prediction_calls": 0 if repair else 15,
        "action_classifier_fits": 0, "native_accounts": 1,
        "native_reconciliation": native["audit"],
        "support": support,
        "area_masked_rows": int(teacher.prefix_mask.eq(1).sum()),
        "area_unknown_rows": int(teacher.prefix_mask.eq(0).sum()),
        "source_H1_projection": h1_projection,
        "training_shadow": {"sessions": len(shadow["replay"]),
            "points": len(shadow["points"]), "orders": len(shadow["orders"]),
            "order_notional_usd": float(shadow["orders"].notional.sum()) if len(shadow["orders"]) else 0.0,
            "order_fees_usd": float(shadow["orders"].total.sum()) if len(shadow["orders"]) else 0.0,
            "terminal_nav": float(shadow["replay"].native_expected_nav.iloc[-1]),
            "policy_paths": 1, "slippage_bps": 2,
            "purpose": "training_reference_only_not_candidate_economics"},
        "decision_chain": decision_bridge["summary"],
        "artifact_policy": {"policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": list(plan["recorder_delivery"]["recorder_only"])},
        "no_2023_plus_forward_or_accounts": True,
        "new_market_HTTP": 0, "new_market_source_queries": 0,
        "P3_authorized": False, "promotion_permission": False,
        "historical_PIT_qualified": False, "independent_validation_qualified": False,
        "alpha_or_economic_success": False}
    recorder.log_metrics(bridge_status_code=1, output_row_count=len(labels))
    return result


def run_path_area_bridge(*, root: str | Path | None = None,
                         execution_authorization: str | Path | None = None) -> dict[str, Any]:
    """Execute P2 only after a distinct actual-run authorization receipt.

    This implementation-GO turn never calls this function. It is provided for
    the later actual execution GO and refuses to consume its fit/prediction/
    account budgets under design or implementation authorization alone.
    """
    root = _resolve_root(root)
    plan_path, plan = _load_plan(root)
    inherited = _verify_inherited(root, plan)
    implementation_identity = _implementation_identity(root, plan)
    implementation_review = _verify_implementation_review(root, plan, implementation_identity)
    go = _require_execution_go(root, plan, implementation_identity["sha256"],
                               implementation_review["sha256"], execution_authorization)
    card = root / P2_CARD
    static_files = {"workflow.yaml", "conclusion.md", "P2_execution_authorization.json",
                    "P2_job_handle.json", "P2_process_exit.json"}
    if card.exists() and any(item.name not in static_files for item in card.iterdir()):
        raise FileExistsError("blocked: P2 execution artifacts already exist; refusing resume or overwrite")
    card.mkdir(parents=True, exist_ok=True)
    state, state_path = _execution_state(card, go)
    recorder = None
    provider = None
    try:
        import qlib
        from qlib.workflow import R
        from src.core.recorder_sources import load_verified_recorder_objects
        from src.data.path_area_source_preflight import frame_digest
        from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
        from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
        from src.workflow.pit_constituent_auxiliary_training_allocation import (
            AuxiliaryTrainingCandidate, PitAuxiliaryAllocationSpec,
        )
        from src.backtest.native_raw_corporate import native_raw_validate
        from src.backtest.raw_corporate_policy import make_schedule
        from src.core.native_evidence import finalize_evidence
        from src.analysis.decision_bridge import build_decision_bridge

        p1_plan = json.loads((root / plan["inherits"]["P1_plan"]["path"]).read_text())
        tariff = p1_plan["tariff"]
        if not isinstance(tariff, dict) or not tariff.get("tiers"):
            raise ValueError("blocked: frozen inherited P1 tariff is absent or malformed")
        daily_provider = root / p1_plan["sources"]["daily_calendar"]["provider_uri"]
        qlib.init(provider_uri=str(daily_provider), region="us", kernels=1)
        source_card = root / P1_CARD
        expected = plan["source_admission"]["read_allowlist"]
        state["source_loader_calls_attempted"] = 1
        state["source_artifact_load_count_unknown_on_exception"] = True
        _atomic_json(state_path, state)
        source_objects, source_audit = load_verified_recorder_objects(
            source_card, expected_recorder_id=plan["P1_evidence"]["recorder_id"],
            artifacts=list(expected), expected_artifact_sha256=expected,
            expected_manifest_sha256=plan["P1_evidence"]["manifest_sha256"])
        state.update(source_artifacts_loaded=9, source_loaded=True)
        state["source_artifact_load_count_unknown_on_exception"] = False
        _atomic_json(state_path, state)
        _check_loaded_source(source_objects, source_audit, plan)
        inputs = _fit_barrier_inputs(source_objects, plan)
        saved_models = _verify_saved_models(inputs, plan)
        state["saved_models_reused"] = len(saved_models)
        _atomic_json(state_path, state)

        start = pd.Timestamp(plan["common_training_shadow"]["start_trade_date"])
        preliminary_timing = inputs["timing"].loc[
            inputs["timing"].trade_year.between(2018, 2022)]
        if preliminary_timing.empty:
            raise ValueError("needs_data: P1 trade-year rows are absent")
        terminal = pd.Timestamp(preliminary_timing.exit_date.iloc[-1]).normalize()
        point_prices, provider, point_identity = _build_point_provider(root, inputs, plan, terminal)
        state["native_derived_point_provider_builds"] = 1
        state["native_derived_point_provider_readbacks"] = 1
        _atomic_json(state_path, state)
        h1_projection = {"dataset_artifact": "inputs/admitted_source_objects:PG_train_only/inputs/training_dataset",
            "columns": ["soxl_forward_open_return", "soxx_forward_open_return", "return_available_at"],
            "signal_index_sha256": _canonical_sha([str(value) for value in inputs["h1_timing"].index]),
            "two_original_H1_arrays_sha256": frame_digest(inputs["h1_returns"]),
            "five_original_targets_sha256_before": inputs["original_target_digest"],
            "five_original_targets_sha256_after": frame_digest(inputs["primary"].loc[:, [
                ("label", name) for name in __import__("src.model.minute_pit_allocator", fromlist=["TARGETS"]).TARGETS]]),
            "price_derived_returns": False}
        if h1_projection["five_original_targets_sha256_before"] != h1_projection["five_original_targets_sha256_after"]:
            raise ValueError("blocked: original five-target digest changed")

        source_identity = {
            "P1_recorder_id": source_audit["recorder_id"],
            "P1_manifest_sha256": source_audit["verification_manifest_sha256"],
            "P1_artifacts": {name: row["sha256"] for name, row in source_audit["artifacts"].items()},
            "P1_fold_dataset_identity": {str(year): _dataframe_identity(
                __import__("src.core.native_evidence", fromlist=["prepare_groups"]).prepare_groups(
                    inputs["datasets"][f"fold_{year}"], "train")) for year in (2018, 2019)},
            "P1_empty_aux_identity": {str(year): _dataframe_identity(
                __import__("src.core.native_evidence", fromlist=["prepare_groups"]).prepare_groups(
                    inputs["datasets"][f"empty_aux_{year}"], "train")) for year in (2018, 2019)},
            "P1_input_view": inputs["source_frame_identity"],
            "P1_primary_training_frame": inputs["primary_frame_identity"],
            "P1_minute_frame": inputs["minute_frame_identity"],
            "P1_coverage": inputs["coverage_identity"],
            "H1_projection": h1_projection,
            "derived_point_provider": point_identity,
            "inherited_evidence": inherited,
            "implementation_identity_sha256": implementation_identity["sha256"],
            "implementation_review_sha256": implementation_review["sha256"],
            "implementation_contract": IMPLEMENTATION_CONTRACT,
        }
        source_identity["phase_identity_sha256"] = _canonical_sha(source_identity)
        # The P2 Recorder is created only after every source object, clock,
        # exact P1 artifact, derived point provider, and current source code has
        # a sealed identity.
        from src.core.native_evidence import native_dataset, prepare_groups
        tracking = card / "qlib_recorder"
        p1_binding_path = source_card / "nas_binding.json"
        p1_binding = json.loads(p1_binding_path.read_text(encoding="utf-8"))
        transport_binding = card / "nas_binding.json"
        binding = {"schema_version": "recorder_evidence_binding_v1",
            "authority": p1_binding["authority"], "ssh": p1_binding["ssh"],
            "source_id": source_identity["phase_identity_sha256"], "as_of": str(terminal.date()),
            "phase": "P2", "plan_sha256": PLAN_SHA256}
        _atomic_json(transport_binding, binding)

        recovery_spec = IntradayRecoveryDirectAllocationSpec(
            root=root, **{key: value for key, value in plan["nonlearned_policy_identity"]["recovery_spec"].items()
                          if key in IntradayRecoveryDirectAllocationSpec.__dataclass_fields__ and key != "root"})
        action_spec = Stage1OpenActionValueSpec(
            root=root, **{key: value for key, value in plan["nonlearned_policy_identity"]["action_spec"].items()
                          if key in Stage1OpenActionValueSpec.__dataclass_fields__ and key != "root"})
        candidate = AuxiliaryTrainingCandidate(name="pit_only_aux_005", auxiliary_weight=.05,
                                               include_union_backfill=False)
        qlib.init(provider_uri=str(daily_provider), region="us", kernels=1)
        with R.start(experiment_name=plan["experiment"] + "_P2", recorder_name="path_area_action_bridge_P2",
                     uri=str(tracking)):
            recorder = R.get_recorder()
            recorder.log_params(plan_sha256=PLAN_SHA256,
                source_identity_sha256=source_identity["phase_identity_sha256"],
                implementation_identity_sha256=implementation_identity["sha256"],
                implementation_review_sha256=implementation_review["sha256"],
                p1_recorder_id=source_audit["recorder_id"], new_fit_maximum=6,
                reused_model_count=9, action_classifier_fits=0, native_accounts=1)
            recorder.save_objects(artifact_path="sources", identity=source_identity,
                                  load_audit=source_audit, H1_projection=h1_projection)
            code_snapshot = {relative: (root / relative).read_text(encoding="utf-8")
                             for relative in implementation_identity["files"] if not relative.startswith("environment:")}
            recorder.save_objects(artifact_path="implementation", code=code_snapshot,
                                  identity=implementation_identity)
            new_models, fit_barrier = _fit_new_states(root, inputs, plan, recorder, state, state_path)
            state["new_allocation_fit_attempted"] = 6
            state["new_allocation_fit_completed"] = 6
            state["fit_barrier_status"] = "passed"
            _atomic_json(state_path, state)

            predictions = _predict_oof(inputs, saved_models, new_models, plan,
                recorder=recorder, execution_state=state, state_path=state_path)
            state["OOF_prediction_calls"] = state["OOF_prediction_calls_completed"]
            recorder.save_objects(artifact_path="OOF", raw_equal_weight_predictions=predictions)
            _atomic_json(state_path, state)
            result = _complete_postfit(root=root, card=card, plan=plan, inputs=inputs,
                predictions=predictions, saved_models=saved_models, point_prices=point_prices,
                provider=provider, start=start, terminal=terminal, state=state,
                state_path=state_path, recorder=recorder, source_identity=source_identity,
                source_audit=source_audit, implementation_identity=implementation_identity,
                implementation_review=implementation_review, h1_projection=h1_projection,
                fit_barrier=fit_barrier, tracking=tracking, daily_provider=daily_provider,
                tariff=tariff, candidate=candidate, recovery_spec=recovery_spec,
                action_spec=action_spec, experiment_name=plan["experiment"] + "_P2", phase="P2")
        result = finalize_evidence(recorder, card=card, summary=result,
            transport_binding=transport_binding, workspace_root=root, include_code_artifacts=True)
        import shutil
        shutil.rmtree(provider)
        provider = None
        state.update(status="completed", recorder_id=str(recorder.id), summary_sha256=_sha_file(card / "summary.json"),
                     verification_manifest_sha256=_sha_file(card / "verification_manifest.json"))
        _atomic_json(state_path, state)
        return result
    except BaseException as error:
        state.update(status="failed", exception_type=type(error).__name__, exception=str(error))
        if recorder is not None:
            state["recorder_id"] = str(recorder.id)
            try:
                recorder.save_objects(artifact_path="execution", failure=state)
            except BaseException:
                pass
        _atomic_json(state_path, state)
        if provider is not None:
            import shutil
            shutil.rmtree(provider, ignore_errors=True)
        raise
