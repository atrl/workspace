"""One authorized P3 action experiment with a durable train/evaluation barrier.

Importing or inspecting this module does not open research objects. ``run`` is
single-use: any started attempt, including a source or cache failure, requires
separate review rather than an automatic restart.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.native_evidence import _atomic_json, finalize_evidence, prepare_groups
from src.core.recorder_sources import load_verified_recorder_objects
from src.model.path_area_action import feature_order, fit_one_arm, preflight_training_rows


BASE = Path("research/path_area_action_bridge_20261018")
CARD = BASE / "P3"
NAME = "path_area_action_bridge_20261018_P3"
CANDIDATE, CONTROL = "C_path_area", "B_path_control"
FAMILIES = ("A_active_research_replay", "B_shared", CONTROL, CANDIDATE,
            "SOXX", "strong9097", "additional27baa")
COMPARATORS = ("SOXX", CONTROL, "A_active_research_replay", "B_shared", "strong9097")
DATES = ("signal_date", "trade_date", "exit_date")
OUTPUTS = ("soxl_weight", "soxx_weight", "cash_weight", "tail_probability", "alpha_probability")
NOTEBOOK = CARD / "path_area_action_bridge_p3.ipynb"
CODE_FILES = (
    "src/workflow/path_area_action_bridge_p3.py", "src/model/path_area_action.py",
    "src/backtest/path_area_action.py", "src/data/path_area_evaluation.py",
    "src/data/path_area_bridge.py", "src/core/native_evidence.py",
    "src/workflow/path_area_source_preflight.py", "src/data/path_area_source_preflight.py",
    "src/core/recorder_sources.py", "src/data/qlib_native.py", "src/data/nas_market_data.py",
    "src/backtest/stage1_domain_heads_raw.py", "src/backtest/raw_corporate_policy.py",
    "src/backtest/native_raw_corporate.py", "src/backtest/qlib_continuous_position.py",
    "src/backtest/strategy.py", "src/backtest/ibkr_tiered.py",
    "src/workflow/stage1_open_action_value.py", "src/workflow/intraday_recovery_direct_allocation.py",
    "src/workflow/stage1_domain_heads.py", "src/analysis/stage1_domain_heads.py",
    "src/analysis/integrated_allocation.py", "src/analysis/soxx_objective.py",
    "src/analysis/decision_bridge.py", str(CARD / "workflow.yaml"),
    "tests/test_path_area_action.py", "tests/test_path_area_action_runtime.py",
    "tests/test_path_area_action_bridge_p3.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _jsonable(value: Any) -> Any:
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (Path, pd.Timestamp)):
            return str(item)
        raise TypeError("non-JSON evidence value: " + type(item).__name__)
    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("blocked: " + message)


def _no_global_writer() -> None:
    from qlib.workflow import R
    try:
        active = R.exp_manager.active_experiment
    except AttributeError:
        active = None
    _require(active is None or active.active_recorder is None, "existing active global Recorder")


def _notebook_source(path: Path) -> dict:
    notebook = json.loads(path.read_text())
    # Result prose/outputs do not change the executable implementation seal.
    return {"cells": [{"cell_type": cell["cell_type"], "source": cell.get("source", [])}
                      for cell in notebook["cells"]
                      if not (cell["cell_type"] == "markdown"
                              and cell.get("metadata", {}).get("role") == "conclusion")]}


def current_implementation_identity(root: str | Path) -> dict:
    """Code bytes and normalized executable notebook, with one aggregate SHA."""
    root = Path(root)
    files = {name: sha(root / name) for name in CODE_FILES}
    notebook = _notebook_source(root / NOTEBOOK)
    files[str(NOTEBOOK)] = canonical_sha(notebook)
    return {"files": files, "notebook_normalization": "exclude outputs, execution counts, metadata and conclusion-role cell",
            "sha256": canonical_sha(files)}


def _authorize(root: Path, card: Path, plan: dict, identity: dict) -> dict:
    """Only JSON/code reads occur before this exact plan/review/GO check."""
    digest = sha(root / BASE / "P3_plan.json")
    design_path, review_path = root / BASE / "P3_design_review.json", root / BASE / "P3_implementation_review.json"
    go_path = card / "P3_execution_authorization.json"
    design, review, go = [json.loads(path.read_text()) for path in (design_path, review_path, go_path)]
    _require(plan.get("frozen") is True and plan.get("phase") == "P3", "P3 plan is not frozen")
    _require(design.get("verdict") == "APPROVE_P3_FROZEN_DESIGN_FOR_SCOPED_IMPLEMENTATION"
             and design.get("plan_sha256") == digest, "independent design review differs")
    _require(review.get("verdict") == "APPROVE_P3_IMPLEMENTATION_FOR_SEPARATE_EXECUTION_GO"
             and review.get("plan_sha256") == digest
             and review.get("implementation_sha256") == identity["sha256"]
             and review.get("material_blockers") == [], "implementation review differs")
    _require(go.get("approved") is True and go.get("owner") == "root"
             and go.get("scope") == "actual_P3_execution" and go.get("maximum_attempts") == 1
             and go.get("plan_sha256") == digest and go.get("implementation_sha256") == identity["sha256"]
             and go.get("design_review_sha256") == sha(design_path)
             and go.get("implementation_review_sha256") == sha(review_path)
             and go.get("budget") == plan["costs_and_budgets"], "separate root execution GO missing or changed")
    _require(not (card / "run_state.json").exists(), "P3 attempt already exists; automatic resume/refit forbidden")
    _require(not (card / "summary.json").exists(), "P3 card is already sealed")
    for item in plan["code_reuse_and_implementation_boundary"]["reuse_candidates"]:
        _require(sha(root / item["path"]) == item["sha256"], "frozen reuse code changed: " + item["path"])
    policy = plan["objective_policy"]
    _require(policy["candidate_family"] == CANDIDATE and policy["phase"] == "P3"
             and canonical_sha(policy) == plan["objective_policy_sha256"], "frozen SOXX objective policy differs")
    return {"plan_sha256": digest, "implementation": identity,
            "design_review_sha256": sha(design_path), "implementation_review_sha256": sha(review_path),
            "go_sha256": sha(go_path), "go": go}


def _checkpoint(state: dict, path: Path, rec=None) -> None:
    _atomic_json(path, state)
    if rec is not None:
        rec.save_objects(artifact_path="execution", checkpoint=state.copy())


def _outside_writer(rec, operation):
    """Close the detached MLflow writer around source/native initialization."""
    from qlib.workflow.recorder import Recorder
    _no_global_writer()
    if rec is None:
        return operation()
    rec.end_run(status=Recorder.STATUS_FI)
    try:
        return operation()
    finally:
        # Resume the same Recorder only to attach the result/failure checkpoint.
        # This is a lifecycle transition inside one attempt, never job resume.
        rec.start_run()


def _claim(state: dict, path: Path, kind: str, key: str, maximum: int, rec=None) -> None:
    attempts = state[kind + "_attempts"]
    _require(key not in attempts and len(attempts) < maximum, kind + " attempt consumed; no automatic retry")
    attempts.append(key)
    state["phase"] = kind
    _checkpoint(state, path, rec)


def _source_binding(source: dict, artifacts: dict | None = None) -> dict:
    hashes = artifacts if artifacts is not None else source["artifacts"]
    return {"card": source["card"], "recorder_id": source["recorder_id"],
            "archive": source.get("archive", source.get("archive_sha256")),
            "inventory_sha256": source.get("inventory_sha256", source.get("verification_manifest_sha256")),
            "artifacts": list(hashes), "artifact_sha256": hashes}


def _read(root, source, state, state_path, rec, key):
    _no_global_writer()
    _claim(state, state_path, "source", key, 10, rec)
    card = root / source["card"]
    binding = json.loads((card / "nas_binding.json").read_text())
    _require(binding["evidence"]["sha256"] == source["archive"], "source NAS archive changed: " + key)
    try:
        objects, audit = _outside_writer(rec, lambda: load_verified_recorder_objects(card,
            expected_recorder_id=source["recorder_id"], artifacts=source["artifacts"],
            expected_manifest_sha256=source["inventory_sha256"],
            expected_artifact_sha256=source["artifact_sha256"]))
    except BaseException:
        # The loader verifies the complete inventory before opening objects,
        # but a deserialize error can occur after an earlier object loaded.
        state["partial_object_load_count_unknown"] = True
        state["partial_object_load_failed_source"] = key
        _checkpoint(state, state_path, rec)
        raise
    state["source_objects_read"] += len(objects)
    state["source_done"].append(key)
    if rec is not None:
        rec.save_objects(artifact_path="sources/audits", **{key: audit})
    _checkpoint(state, state_path, rec)
    return objects, audit


def _training(objects: dict):
    """Reconcile signal-index Dataset with actual trade-index P2 evidence."""
    dataset = objects["local_labels/action_label_dataset"]
    grouped = prepare_groups(dataset, "all")
    pd.testing.assert_frame_equal(grouped, objects["local_labels/action_label_frame"], check_exact=True)
    _require(list(grouped.index.names) == ["datetime", "instrument"]
             and not grouped.index.has_duplicates
             and set(grouped.index.get_level_values("instrument")) == {"SOXL_SOXX_CASH"}, "training native index differs")
    trade = pd.DatetimeIndex(pd.to_datetime(grouped[("meta", "trade_date")]))
    signal = pd.DatetimeIndex(grouped.index.get_level_values("datetime"))
    _require(not trade.has_duplicates and trade.is_monotonic_increasing
             and trade.tz is None and trade.equals(trade.normalize()), "training actual trade dates differ")
    labels, decisions = objects["local_labels/labels"], objects["shadow_python/decisions"]
    for frame in (labels, decisions):
        pd.testing.assert_index_equal(pd.DatetimeIndex(frame.index), trade, check_names=False)
        pd.testing.assert_index_equal(pd.DatetimeIndex(frame.signal_date), signal, check_names=False)
    _require(tuple(grouped["feature"].columns) != (), "empty training feature group")
    features = grouped["feature"].loc[:, list(feature_order())].copy()
    features.index = trade
    delta = grouped[("label", "local_execute_delta")].copy()
    delta.index = trade
    pd.testing.assert_series_equal(delta, labels.execute_delta.rename(delta.name), check_names=False, check_exact=True)
    expected = decisions.loc[:, list(feature_order())].copy()
    expected.index = trade
    pd.testing.assert_frame_equal(features, expected, check_exact=True, check_names=False)
    pd.testing.assert_frame_equal(features, labels.loc[:, list(feature_order())], check_exact=True, check_names=False)
    maturity = grouped[("meta", "label_available_at")].copy()
    maturity.index = trade
    pd.testing.assert_series_equal(maturity, labels.label_available_at, check_names=False, check_exact=True)
    pd.testing.assert_series_equal(labels.eligible, decisions.eligible, check_names=False, check_exact=True)
    path = objects["shadow_python/path_area_inputs"].copy()
    _require("signal_date" in path and "trade_date" in path, "P2 path superset has no explicit date mapping")
    path_signal, path_trade = pd.DatetimeIndex(path.signal_date), pd.DatetimeIndex(path.trade_date)
    _require(path_signal.is_unique and path_trade.is_unique and path_signal.is_monotonic_increasing
             and path_trade.is_monotonic_increasing and path_signal.tz is None and path_trade.tz is None
             and (path_trade < pd.Timestamp("2023-01-01")).all()
             and signal.isin(path_signal).all(), "P2 path superset/date boundary differs")
    # P2 path inputs cover an earlier warmup as well as the 1057 shadow rows.
    # Admission is on the exact native signal grid, never a positional slice.
    path = path.set_index("signal_date").loc[signal]
    pd.testing.assert_index_equal(pd.DatetimeIndex(path.trade_date), trade, check_names=False)
    from src.data.path_area_bridge import AREA_COLUMN, CONTROL_COLUMNS
    names = [*CONTROL_COLUMNS, "prefix_mask", AREA_COLUMN]
    pd.testing.assert_frame_equal(path[names].reset_index(drop=True), features[names].reset_index(drop=True),
                                  check_names=False, check_exact=True)
    metadata = pd.DataFrame({"trade_date": trade, "eligible": labels.eligible.to_numpy(),
        "label_status": labels.label_status.to_numpy(), "label_available_at": maturity.to_numpy()}, index=trade)
    preflight = preflight_training_rows(features, delta, metadata)
    return preflight, {"input_rows": preflight.input_rows, "fit_rows": preflight.row_count,
        "class_counts": preflight.class_counts, "pre_2022_rows": preflight.pre_2022_rows,
        "pre_2022_class_counts": preflight.pre_2022_class_counts, "eligible_rows": preflight.eligible_rows,
        "mature_rows": preflight.mature_rows, "observed_area_rows": preflight.observed_area_rows,
        "failures": list(preflight.failures), "feature_order": list(feature_order()),
        "fit_row_ids_sha256": hashlib.sha256(pd.util.hash_pandas_object(preflight.features.index).values.tobytes()).hexdigest(),
        "B_C_identical_row_ids_and_labels": True, "cutoff": str(preflight.cutoff),
        "pre_2022_cutoff": str(preflight.pre_2022_cutoff),
        "software": {name: importlib.metadata.version(name) for name in ("pyqlib", "scikit-learn", "numpy", "pandas")}}


def _fit_models(preflight, rec, state, state_path):
    _require(preflight.ready, "training support failed: " + ",".join(preflight.failures))
    _require(all(preflight.pre_2022_class_counts.values()), "strict pre-2022 cohort lacks both classes")
    artifact_root = Path(rec.get_local_dir()) / "artifacts"
    for arm in ("B", "C"):
        _claim(state, state_path, "fit", arm, 2, rec)
        fitted = fit_one_arm(preflight, arm)
        # Save each successful model immediately, before another fit can start.
        rec.save_objects(artifact_path="models", **{arm + "_action_model": fitted.pipeline,
            arm + "_preprocessor": fitted.pipeline[:-1], arm + "_fit_metadata": asdict(fitted) | {"pipeline": None}})
        for suffix in ("action_model", "preprocessor"):
            key = arm + "_" + suffix
            state["model_sha256"][key] = sha(artifact_root / "models" / key)
        state["fit_done"].append(arm)
        _checkpoint(state, state_path, rec)
    _require(state["fit_done"] == ["B", "C"] and len(state["model_sha256"]) == 4, "two-model barrier incomplete")
    models = {}
    for arm in ("B", "C"):
        for suffix in ("action_model", "preprocessor"):
            key = arm + "_" + suffix
            _require(sha(artifact_root / "models" / key) == state["model_sha256"][key], "sealed model bytes differ")
            value = rec.load_object("models/" + key)
            _require(int(value.n_features_in_) == 32, "sealed pipeline width differs")
            if suffix == "action_model":
                models[arm] = value
    state.update(models_sealed=True, phase="models_fixed_before_evaluation", evaluation_unlocked=True)
    rec.save_objects(artifact_path="execution", model_barrier={"model_sha256": state["model_sha256"],
        "fit_attempts": state["fit_attempts"], "fit_done": state["fit_done"],
        "evaluation_objects_read_before_barrier": 0})
    _checkpoint(state, state_path, rec)
    return models


def _analysis_plan(root, plan):
    source = plan["sources_and_phase_allowlists"]["P3b_evaluation_only_after_barrier"]["source_plan"]
    _require(sha(root / source["path"]) == source["sha256"], "source clock plan changed")
    original = json.loads((root / source["path"]).read_text())
    accounts = {**original["accounts"], "families": list(FAMILIES)}
    gates = {**original["gates"], "required_comparators": list(COMPARATORS)}
    gates["paired_uncertainty"] = {**gates["paired_uncertainty"], "mean_block": 63,
                                 "resamples": 10000, "alpha": .025 / 15, "seed": 20261014}
    _require(accounts["initial_cash"] == plan["costs_and_budgets"]["initial_cash_usd"]
             and original["tariff"] == plan["costs_and_budgets"]["tariff"], "frozen source cost/capital differs")
    return {**original, "accounts": accounts, "gates": gates,
            "provider_output": "data/qlib/path_area_action_bridge_20261018_P3_execution"}


def _evaluation(root, plan, state, state_path, rec):
    _require(state.get("models_sealed") is True and state["fit_done"] == ["B", "C"], "evaluation before model seal")
    spec = plan["sources_and_phase_allowlists"]["P3b_evaluation_only_after_barrier"]
    action = spec["original_recovery_and_account_sources"]
    target_spec = spec["original_allocation_target_stream"]
    target_hashes = {target_spec["artifact"]: target_spec["artifact_sha256"],
                     **target_spec["required_tail_audit"]["artifacts"]}
    sources = {
        "B_targets": _source_binding(target_spec, target_hashes),
        "structural": _source_binding(action["structural_evaluation_context"]),
        "recovery": _source_binding(action["recovery_selection_and_models"]),
        "raw": _source_binding(action["raw_account_grid_and_marks"]),
        **{role: source for role, source in spec["reference_sources"].items()},
    }
    objects, audits = {}, {}
    for role, source in sources.items():
        objects[role], audits[role] = _read(root, source, state, state_path, rec, "P3b_" + role)
    context = objects["structural"]
    frame, market = context["inputs/frame"].reset_index(drop=True).copy(), context["inputs/market"].reset_index(drop=True)
    _require(len(frame) == len(market) == 944, "structural account row count differs")
    pd.testing.assert_frame_equal(frame[list(DATES)], market[list(DATES)], check_exact=True)
    signal, trade = pd.DatetimeIndex(frame.signal_date), pd.DatetimeIndex(frame.trade_date)
    _require(signal.is_unique and trade.is_unique and signal.is_monotonic_increasing
             and trade.is_monotonic_increasing, "structural row identity differs")
    target = objects["B_targets"]["analysis/smoothed_targets"]["B_shared"]
    _require(tuple(target.columns) == OUTPUTS and len(target) == 944, "saved B target schema differs")
    pd.testing.assert_index_equal(pd.DatetimeIndex(target.index), signal, check_names=False)
    _require(np.isfinite(target.to_numpy(float)).all(), "saved B targets are nonfinite")
    prediction_index = None
    for seed in (71, 172, 273):
        prediction = objects["B_targets"][f"predictions/B_shared_{seed}"]
        index = pd.DatetimeIndex(prediction.index)
        _require(len(prediction) == 946 and index.is_unique and index.is_monotonic_increasing
                 and len(index.difference(signal)) == 2 and signal.isin(index).all(), "saved allocation tail proof differs")
        if prediction_index is not None:
            pd.testing.assert_index_equal(index, prediction_index, check_names=False)
        prediction_index = index
    # The saved prediction axis establishes signal->next actual session mapping.
    pd.testing.assert_index_equal(prediction_index[:944], signal, check_names=False)
    pd.testing.assert_index_equal(prediction_index[1:945], trade, check_names=False)
    for column in OUTPUTS:
        frame[column] = target[column].to_numpy(copy=True)
        np.testing.assert_array_equal(frame[column], target[column])
    points, schedule = objects["raw"]["inputs/raw_points"], objects["raw"]["inputs/schedule"]
    strong = objects["strong_native"]
    pd.testing.assert_frame_equal(points, strong["inputs/raw_points"], check_exact=True)
    _require(len(points) == 3778 and points.gt(0).all().all(), "raw point grid differs")
    _require(sources["raw"]["artifact_sha256"]["inputs/schedule"] ==
             sources["strong_native"]["artifact_sha256"]["inputs/schedule"], "reference corporate schedule differs")
    analysis_plan = _analysis_plan(root, plan)
    raw_plan = strong["inputs/plan"]
    raw_accounts = raw_plan.get("accounts", raw_plan)
    _require(raw_plan["tariff"] == analysis_plan["tariff"]
             and raw_accounts["initial_cash"] == 100000
             and raw_accounts["slippage_bps"] == [0, 2, 5], "reference tariff/initial state differs")
    references = {}
    for item in spec["reference_books"]:
        book = objects[item["source_role"]][item["artifact"]]
        pd.testing.assert_frame_equal(book["authority"][list(DATES)].reset_index(drop=True), frame[list(DATES)], check_exact=True)
        _require(pd.DatetimeIndex(book["points"].stamp).equals(points.index)
                 and book["audit"]["status"] == "passed", "reused reference native/grid differs")
        np.testing.assert_allclose(book["points"].iloc[0][["cash", "receivable", "q_soxl", "q_soxx", "nav"]].to_numpy(float),
                                   [100000, 0, 0, 0, 100000], rtol=0, atol=1e-7)
        references[(item["family"], item["cost"])] = book
    _require(len(references) == 15, "exact fifteen reference books required")
    release = plan["clock_and_account_contract"]["active_release"]
    _require(sha(root / release["path"]) == release["sha256"], "frozen release changed")
    recovery = objects["recovery"]
    controller, source_plan = recovery["controller/frozen_models"], recovery["inputs/plan"]
    _require("frozen_recovery_model" in controller, "frozen recovery model missing")
    # Prefix helper owns the one native daily-calendar and core-feature query.
    from src.data.path_area_evaluation import read_evaluation_prefix
    _claim(state, state_path, "prefix", "native_prefix_and_daily_calendar", 1, rec)
    def native_attempt(kind):
        _require(kind in ("D.calendar", "D.features"), "unexpected source native call")
        field = "source_" + kind.replace(".", "_") + "_attempts"
        _require(state[field] == 0, "native source call budget consumed")
        state[field] += 1
        _checkpoint(state, state_path)
    prefix = _outside_writer(rec, lambda: read_evaluation_prefix(root, plan, frame, on_native_attempt=native_attempt))
    retained = prepare_groups(prefix["prefix_dataset"], "all")["feature"]
    for day in trade[prefix["path_inputs"].prefix_mask.eq(1).to_numpy()]:
        stamp = day + pd.Timedelta(hours=9, minutes=45)
        mark = retained.loc[(stamp, ["SOXL", "SOXX"]), "$close"]
        mark = mark.droplevel("datetime").reindex(["SOXL", "SOXX"])
        np.testing.assert_array_equal(mark.to_numpy(float), points.loc[stamp, ["SOXL", "SOXX"]].to_numpy(float))
    state["prefix_done"].append("native_prefix_and_daily_calendar")
    state["native_source_counts"] = prefix["audit"]
    rec.save_objects(artifact_path="evaluation", prefix_dataset=prefix["prefix_dataset"],
                     path_dataset=prefix["path_dataset"], prefix_audit=prefix["audit"],
                     daily_calendar=prefix["daily_calendar"], path_inputs=prefix["path_inputs"],
                     source_audits=audits, B_target=target, B_seed_predictions={seed: objects["B_targets"][f"predictions/B_shared_{seed}"] for seed in (71, 172, 273)})
    _checkpoint(state, state_path, rec)
    return {"plan": analysis_plan, "frame": frame, "market": market, "points": points, "schedule": schedule,
            "guard": objects["raw"]["inputs/repair_1_calendar_guard"], "controller": controller,
            "selection": recovery["controller/source_recovery_selection"], "recovery_spec": source_plan["recovery_spec"],
            "action_spec": source_plan["action_spec"], "references": references, "prefix": prefix["path_inputs"],
            "daily_calendar": prefix["daily_calendar"], "source_audits": audits}


class _CountForecast:
    """Transparent fixed-model forward proxy; persist attempts before calling."""
    def __init__(self, model, state, state_path, kind):
        self.model, self.state, self.state_path, self.kind = model, state, state_path, kind

    def predict_proba(self, values):
        self.state[self.kind] += 1
        _checkpoint(self.state, self.state_path)
        return self.model.predict_proba(values)


def _execution_provider(root, prepared, state, state_path, rec):
    """One scoped derived build/readback with durable attempts before each API."""
    import qlib
    from qlib.data import D
    from src.data import build_qlib_store_from_frame
    _no_global_writer()
    points = prepared["points"]
    guard = prepared["guard"]["$raw_execution_price"].unstack("instrument")[["SOXL", "SOXX"]]
    _require(guard.index[0] > points.index[-1], "native guard must be strictly after valuation endpoint")
    combined = pd.concat([points, guard]).sort_index()
    _require(combined.index.is_unique, "derived raw grid has duplicates")
    provider = root / prepared["plan"]["provider_output"]
    _require(not provider.exists(), "derived provider already exists; no silent overwrite or rebuild")
    fields = ["open", "high", "low", "close", "execution_price", "raw_execution_price"]
    rows = [{"datetime": stamp, "instrument": symbol, **{field: float(value) for field in fields}}
            for symbol in ("SOXL", "SOXX") for stamp, value in combined[symbol].items()]
    state["derived_native_provider_build_attempts"] += 1
    _checkpoint(state, state_path)
    manifest = build_qlib_store_from_frame(pd.DataFrame(rows), provider_uri=provider, freq="1min", region="us",
        as_of="2026-10-09", field_map={name: name for name in fields},
        source="P3 exact hash-verified raw Recorder marks; following guard is excluded from policy and NAV")
    state["derived_native_provider_builds"] += 1
    _checkpoint(state, state_path)
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    state["derived_native_provider_readback_attempts"] += 1
    _checkpoint(state, state_path)
    observed = D.features(["SOXL", "SOXX"], ["$raw_execution_price"], start_time=combined.index[0],
                          end_time=combined.index[-1], freq="1min")["$raw_execution_price"].unstack("instrument")
    pd.testing.assert_index_equal(observed.index, combined.index, check_names=False)
    np.testing.assert_array_equal(observed[["SOXL", "SOXX"]].to_numpy(), combined.to_numpy())
    state["derived_native_provider_readbacks"] += 1
    _checkpoint(state, state_path)
    return provider, manifest


def _accounts(root, card, plan, prepared, models, rec, state, state_path, authorization):
    from src.backtest.path_area_action import replay_path_area_action
    from src.backtest.native_raw_corporate import native_raw_validate
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
    from src.analysis.stage1_domain_heads import analyze_paired_accounts
    from src.analysis.soxx_objective import evaluate_soxx_objective
    from src.analysis.decision_bridge import build_decision_bridge
    _claim(state, state_path, "provider", "derived_provider_build_and_readback", 1, rec)
    provider, manifest = _outside_writer(rec, lambda: _execution_provider(root, prepared, state, state_path, rec))
    state["provider_done"].append("derived_provider_build_and_readback")
    rec.save_objects(artifact_path="evaluation", native_provider_manifest=manifest)
    _checkpoint(state, state_path, rec)
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=root, **prepared["recovery_spec"])
    action_spec = Stage1OpenActionValueSpec(root=root, **prepared["action_spec"])
    books = dict(prepared["references"])
    execution = {"initial_cash": 100000, "tariff": plan["costs_and_budgets"]["tariff"], "slippage_bps": [0, 2, 5]}
    for arm, family in (("B", CONTROL), ("C", CANDIDATE)):
        for cost in (0, 2, 5):
            key = f"{family}_{cost}"
            _claim(state, state_path, "path", key, 6, rec)
            controller = {**prepared["controller"], "frozen_recovery_model": _CountForecast(
                prepared["controller"]["frozen_recovery_model"], state, state_path, "recovery_forecast_attempts")}
            path = replay_path_area_action(prepared["frame"].copy(), prefix=prepared["prefix"],
                pipeline=_CountForecast(models[arm], state, state_path, "action_forecast_attempts"), arm=arm,
                point_prices=prepared["points"], schedule=prepared["schedule"], plan=execution,
                controller=controller, recovery_spec=recovery_spec, action_spec=action_spec,
                selection=prepared["selection"], slippage_bps=cost)
            rec.save_objects(artifact_path="paths", **{key: path})
            state["path_done"].append(key)
            _checkpoint(state, state_path, rec)
            _claim(state, state_path, "native", key, 6, rec)
            book = _outside_writer(rec, lambda: native_raw_validate(path, provider_uri=provider, point_prices=prepared["points"],
                schedule=prepared["schedule"], tariff=execution["tariff"], cost=cost, initial_cash=100000))
            rec.save_objects(artifact_path="native", **{key: book})
            state["native_done"].append(key)
            _checkpoint(state, state_path, rec)
            bridge = build_decision_bridge(path["trace"], book["authority"], native_audit=book["audit"],
                source_identity={"plan_sha256": authorization["plan_sha256"], "recorder_id": rec.id,
                                 "model_sha256": state["model_sha256"][arm + "_action_model"]},
                fields={"source_target": [f"source_target_{asset}_weight" for asset in ("soxl", "soxx", "cash")],
                        "projected_target": [f"projected_target_{asset}_weight" for asset in ("soxl", "soxx", "cash")],
                        "postfill_weights": [f"postfill_{asset}_weight" for asset in ("soxl", "soxx", "cash")],
                        "order_count": "filled_order_count", "fees_usd": "total_order_fees_usd"},
                native_calendar=prepared["daily_calendar"])
            rec.save_objects(artifact_path="decision_chain", **{key: bridge})
            books[(family, cost)] = book
            rec.log_metrics(completed_native_accounts=len(state["native_done"]))
            print("completed P3 native account", key, flush=True)
    _claim(state, state_path, "analysis", "fixed_15_pair_bootstrap", 1, rec)
    results = analyze_paired_accounts(books, pd.DataFrame({"trade_date": prepared["daily_calendar"]}),
        prepared["plan"], candidate_family=CANDIDATE, required_comparators=COMPARATORS, expected_families=FAMILIES)
    uncertainty = results["uncertainty"].copy()
    for key, value in {"mean_block": 63, "resamples": 10000, "random_seed": 20261014,
                       "alpha": .025 / 15}.items():
        _require(key in uncertainty and uncertainty[key].eq(value).all(), "actual bootstrap setting differs: " + key)
    uncertainty["family_size"] = 15
    results["uncertainty"] = uncertainty
    provenance = {"phase": "P3", "candidate_family": CANDIDATE,
        "frozen_plan_sha256": authorization["plan_sha256"], "objective_policy": plan["objective_policy"],
        "frozen_plan_policy_sha256": plan["objective_policy_sha256"], "receipt_refs": [
            {"path": str(card / "P3_execution_authorization.json"), "sha256": authorization["go_sha256"]},
            {"path": str(root / BASE / "P3_implementation_review.json"), "sha256": authorization["implementation_review_sha256"]}]}
    assessment = evaluate_soxx_objective(results, candidate_family=CANDIDATE, provenance=provenance)
    # C-vs-B uses the same predeclared paired draws already present; no B rerun.
    cb = results["paired_comparison"].loc[results["paired_comparison"].comparator.eq(CONTROL)]
    cb_ci = uncertainty.loc[uncertainty.comparator.eq(CONTROL)]
    increment = bool(len(cb) == len(cb_ci) == 3 and cb.relative_wealth.gt(1 + 1e-8).all()
                     and cb_ci.daily_lower_bound.gt(0).all())
    rec.save_objects(artifact_path="analysis", results=results, soxx_objective=assessment,
                     signed_area_increment={"diagnostic_only": True, "all_costs_net_and_lower_positive": increment,
                                            "paired": cb, "uncertainty": cb_ci})
    for name in ("metrics", "paired_comparison", "annual_metrics", "quarterly_metrics", "rolling_summary", "uncertainty"):
        _require(len(results[name]) <= 500, "slim summary exceeds fixed size")
        results[name].to_csv(card / (name + ".csv"), index=False)
    state["analysis_done"].append("fixed_15_pair_bootstrap")
    state["phase"] = "complete_before_archive"
    _checkpoint(state, state_path, rec)
    return _jsonable({"experiment": NAME, "phase": "P3", "plan_sha256": authorization["plan_sha256"],
        "implementation_sha256": authorization["implementation"]["sha256"],
        "decision": "historical_pass_needs_forward_and_risk" if assessment["historical_pass"] else "not_promoted",
        "soxx_objective": {key: value for key, value in assessment.items() if key not in ("diagnostic_comparisons",)},
        "signed_area_increment_diagnostic_pass": increment, "legacy_gates_diagnostic_only": results["gates"],
        "candidate_metrics": results["metrics"].loc[results["metrics"].family.eq(CANDIDATE)].to_dict(orient="records"),
        "objective_achieved": False, "promotion_approved": False, "independent_forward_qualified": False,
        "historical_PIT_qualified": False, "fullminute_risk_qualified": False,
        "source_qualification_limits": plan["source_qualification_limits"], "actual_counts": state.copy(),
        "new_action_fits": len(state["fit_done"]), "new_native_accounts": len(state["native_done"]),
        "reused_native_accounts": 15, "total_comparison_accounts": 21, "source_HTTP": 0,
        "recorder_uri": str(card / "qlib_recorder"), "artifact_policy": plan["artifact_policy"],
        "workflow_config": "workflow.yaml", "production_changed": False})


def _transport(root, card, plan):
    source = plan["sources_and_phase_allowlists"]["P3a_training_only"]["recorder"]
    inherited = json.loads((root / source["card"] / "nas_binding.json").read_text())
    binding = {"schema_version": "recorder_evidence_binding_v1", "authority": inherited["authority"],
               "ssh": inherited["ssh"], "source_id": NAME, "source_plan": "../P3_plan.json"}
    _atomic_json(card / "nas_binding.json", binding)
    return card / "nas_binding.json"


def _conclusion(card, summary):
    metrics = "\n".join(f"- {int(row['slippage_bps'])}bp：相对 SOXX 财富 {row['relative_wealth']:.6f}，"
        f"收盘最大回撤 {row['close_max_drawdown_pct']:.2f}%，稀疏价格点最大回撤 {row['sampled_point_max_drawdown_pct']:.2f}%，"
        f"CVaR5 {row['cvar5_pct']:.4f}%。" for row in summary["candidate_metrics"])
    text = ("# P3 路径面积动作实验\n\n"
        f"结果：{summary['decision']}。Qlib Recorder：{summary['recorder_id']}。\n\n"
        f"已完成 {summary['new_action_fits']} 次动作模型拟合、{summary['new_native_accounts']} 个独立账户原生核对，复用 15 个固定参考账户。"
        "C 为唯一登记候选；B 仅用于匹配归因，不进行结果后替换。完整年度、季度及所有 126 日窗口与决策链保存在 Recorder。\n\n"
        f"SOXX v2 历史硬门槛：{summary['soxx_objective']['status']}；"
        f"面积增量诊断：{summary['signed_area_increment_diagnostic_pass']}。"
        "\n\n" + metrics + "\n\n"
        "分钟来源为有条件的回溯来源，不证明历史首次接收或 PIT；稀疏原始价格点不构成全分钟风险认证。"
        "历史通过仍需独立前向证据与风险资格，本阶段不批准提升或生产发布。固定版本失败后闭合，不调阈值、窗口或符号。\n")
    (card / "conclusion.md").write_text(text)
    notebook_path = card / NOTEBOOK.name
    notebook = json.loads(notebook_path.read_text())
    for cell in notebook["cells"]:
        if cell.get("metadata", {}).get("role") == "conclusion":
            cell["source"] = text.splitlines(keepends=True)
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")


def run(*, root: str | Path, card: str | Path | None = None) -> dict:
    """Execute exactly once only after the root's separately sealed GO."""
    root = Path(root).resolve()
    card = Path(card).resolve() if card is not None else root / CARD
    _require(card == root / CARD, "P3 card path must be canonical")
    plan = json.loads((root / BASE / "P3_plan.json").read_text())
    identity = current_implementation_identity(root)
    authorization = _authorize(root, card, plan, identity)
    state_path = card / "run_state.json"
    state = {"plan_sha256": authorization["plan_sha256"], "implementation_sha256": identity["sha256"],
        "phase": "authorized", "source_objects_read": 0, "action_forecast_attempts": 0,
        "recovery_forecast_attempts": 0, "allocation_forecast_attempts": 0, "model_sha256": {},
        "derived_native_provider_build_attempts": 0, "derived_native_provider_builds": 0,
        "derived_native_provider_readback_attempts": 0, "derived_native_provider_readbacks": 0,
        "source_D_calendar_attempts": 0, "source_D_features_attempts": 0,
        **{kind + suffix: [] for kind in ("source", "fit", "prefix", "provider", "path", "native", "analysis")
           for suffix in ("_attempts", "_done")}}
    _checkpoint(state, state_path)
    import qlib
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder
    rec = None
    try:
        # Initialization and all source reads occur with no global writable R.
        _no_global_writer()
        qlib.init(provider_uri=str(root / plan["sources_and_phase_allowlists"]["P3b_evaluation_only_after_barrier"]
                                   ["prefix_feature_source"]["provider_uri"]), region="us", kernels=1)
        # Source identity and bounded P3a reads precede the first writable run.
        # A prewriter failure still leaves the durable single-attempt ledger.
        training_spec = plan["sources_and_phase_allowlists"]["P3a_training_only"]
        source = _source_binding(training_spec["recorder"], training_spec["artifact_allowlist_sha256"])
        objects, audit = _read(root, source, state, state_path, None, "P3a_training_only")
        with R.start(experiment_name=NAME, recorder_name="single_fixed_P3", uri=str(card / "qlib_recorder")):
            rec = R.get_recorder()
            state.update(recorder_id=rec.id, experiment_id=str(rec.experiment_id))
            rec.save_objects(artifact_path="inputs", plan=plan, authorization=authorization)
        # Detached Recorder lifecycle survives qlib.init in native validators.
        rec.start_run()
        snapshot = {name: (root / name).read_bytes() for name in CODE_FILES}
        snapshot[str(NOTEBOOK)] = json.dumps(_notebook_source(root / NOTEBOOK), sort_keys=True).encode()
        rec.save_objects(artifact_path="sources", implementation_snapshot=snapshot, implementation_identity=identity)
        transport = _transport(root, card, plan)
        rec.save_objects(artifact_path="sources/audits", P3a_training_only=audit)
        preflight, fit_audit = _training(objects)
        rec.save_objects(artifact_path="inputs", P3a_fit_audit=fit_audit, P3a_source_identity=audit,
                         P3a_training_dataset=objects["local_labels/action_label_dataset"])
        models = _fit_models(preflight, rec, state, state_path)
        prepared = _evaluation(root, plan, state, state_path, rec)
        summary = _accounts(root, card, plan, prepared, models, rec, state, state_path, authorization)
        rec.save_objects(artifact_path="analysis", summary=summary)
        rec.log_metrics(new_action_fits=len(state["fit_done"]), new_native_accounts=len(state["native_done"]),
                        action_forecast_attempts=state["action_forecast_attempts"],
                        historical_soxx_pass=int(summary["soxx_objective"]["historical_pass"]))
        rec.end_run(status=Recorder.STATUS_FI)
        summary = finalize_evidence(rec, card=card, summary=summary, transport_binding=transport,
                                    workspace_root=root, include_code_artifacts=True)
        _conclusion(card, summary)
        state.update(completed=True, phase="completed")
        _checkpoint(state, state_path)
        return summary
    except BaseException as error:
        state.update(failed=True, failure={"type": type(error).__name__, "message": str(error)},
                     failure_phase=state["phase"], automatic_resume_forbidden=True)
        if state["source_D_features_attempts"] and not state["prefix_done"]:
            state["native_source_partial_slice_count_unknown"] = True
        _checkpoint(state, state_path)
        if rec is not None:
            try:
                rec.save_objects(artifact_path="execution", failure=state.copy())
                rec.log_metrics(actual_fit_attempts=len(state["fit_attempts"]), actual_native_attempts=len(state["native_attempts"]))
                rec.end_run(status=Recorder.STATUS_FA)
                from src.data.nas_market_data import archive_nas_recorder
                state["failure_nas_evidence"] = archive_nas_recorder(_transport(root, card, plan),
                    workspace_root=root, card=card, recorder_id=rec.id, experiment_id=str(rec.experiment_id))
            except BaseException as archive_error:
                state["failure_archive_error"] = {"type": type(archive_error).__name__, "message": str(archive_error)}
            _checkpoint(state, state_path)
        raise
