"""Native-input/Recorder identity tests; cached synthetic policies are never fit."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pytest

from test_constrained_net_excess_data import frozen_parent, synthetic_source
from src.data.feasible_parent_sparse_search import load_feasible_parent_inputs, load_cached_outer_parents
from src.model.benchmark_anchor_utility import BenchmarkAnchorPolicy, policy_to_dict
from src.objective.benchmark_tail_constraints import evaluate_risk
from src.workflow.constrained_net_excess import training_spec

ROOT = Path(__file__).parents[1]
NEW_CARD = "feasible_parent_sparse_search_20261003"
SEEDS = (17, 41, 71)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.asarray(value, np.float64).tobytes()).hexdigest()


@pytest.fixture
def cached_parent(frozen_parent):
    """Native Phase5 fixture plus a separate manifest-bound Phase7-like cache."""
    from qlib.workflow import R
    root, input_plan, objects, _ = frozen_parent
    plan = json.loads((ROOT / "research" / NEW_CARD / "selection_plan.json").read_text())
    plan["source"] = input_plan["source"]
    plan["parent_input_source"] = input_plan["parent_input_source"]
    plan["training"]["development_signal_years"] = [2016]
    # No outer market years are evaluated; their cache inventory is a metadata fixture.
    plan["training"]["diagnostic_signal_years"] = [2017]
    parent_plan = json.loads((ROOT / "research/constrained_net_excess_20261003/selection_plan.json").read_text())
    parent_plan["source"] = input_plan["source"]
    parent_plan["training"]["development_signal_years"] = [2016]
    parent_plan["training"]["diagnostic_signal_years"] = [2017]
    plan["parent_reproduction_protocol"]["training"] = deepcopy(parent_plan["training"])
    plan["parent_reproduction_protocol"]["constraints"] = deepcopy(parent_plan["constraints"])
    expected_spec = asdict(training_spec(parent_plan))
    card = root / "research/synthetic_phase7"; card.mkdir()
    (card / "selection_plan.json").write_text(json.dumps(parent_plan))
    actual_summary = json.loads((ROOT / "research/constrained_net_excess_20261003/summary.json").read_text())
    source_hashes = actual_summary["source_sha256"]
    for name, digest in source_hashes.items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
        assert sha(path) == digest
    from src.data.constrained_net_excess import load_constrained_inputs
    input_bundle = load_constrained_inputs(root, input_plan)
    source_audit = input_bundle["source_audit"]
    development, audit_index, audit_objects = {}, [], {}
    for old in ("constant", "price", "price_vol"):
        fold = objects["parent_dev_folds"][(old, 2016)]
        seed_models = {}
        for seed in SEEDS:
            model = policy_to_dict(BenchmarkAnchorPolicy(seed=seed, anchor=True)); seed_models[seed] = model
            risks = {}
            for cost in (10, 25):
                net = 1 + fold["returns"][:, 1].copy(); net[0] /= 1 + cost / 10000
                risk = evaluate_risk(np.log(net))
                risks[cost] = {"max_drawdown_pct": risk["max_drawdown_pct"], "cvar5_pct": risk["cvar5_pct"],
                    "benchmark_max_drawdown_pct": risk["max_drawdown_pct"], "benchmark_cvar5_pct": risk["cvar5_pct"],
                    "raw_percentage_point_gaps": [0., 0.], "normalizers": [risk["drawdown_loss"], risk["cvar5_loss"]],
                    "normalizers_valid_for_optimization": True, "risk_pass": [True, True], "feasible": True}
            epochs = [{"epoch": epoch, "checkpoint": deepcopy(model), "F": 0., "feasible": True} for epoch in range(121)]
            audit = {"variant": old, "year": 2016, "seed": seed, "spec": expected_spec,
                "training_rows": len(fold["rows"]), "selected_epoch": "anchor", "failure": None,
                "qualification_eligible": True, "training_feasible": True, "training_feasibility_passed": True,
                "null_selected": True, "F": 0., "selected_F": 0., "centered_worst_utility": 0.,
                "centered_utility_by_cost": {10: 0., 25: 0.}, "legacy_J": 0., "selected_risks_by_cost": risks,
                "training_feature_sha256": _array_sha(fold["features"]), "training_return_sha256": _array_sha(fold["returns"]),
                "initial_tensor_sha256": policy_to_dict(BenchmarkAnchorPolicy(seed=seed))["tensor_sha256"],
                "optimizer_diagnostics": {"updates_completed": 120}, "epochs": epochs, "history": epochs}
            path = f"fit_checkpoints/{old}/2016/{seed}"
            audit_objects[path + "/training_audit.pkl"] = audit
            audit_objects[path + "/model.pkl"] = model
            audit_index.append({key: audit[key] for key in ("variant", "year", "seed", "selected_epoch", "failure")}
                               | {"recorder_artifact": path + "/training_audit.pkl"})
        development[old] = {2016: {"variant": old, "model_year": 2016, "scaler": fold["scaler"],
            "scaler_sha256": fold["scaler"]["scaler_sha256"], "training_feature_sha256": _array_sha(fold["features"]),
            "training_return_sha256": _array_sha(fold["returns"]), "invalid_training": False,
            "numerical_failure": False, "invalid_training_dates": [], "seed_models": seed_models,
            "ensemble_audit": {"anchor_selected": True}}}
    uri = root / "phase7_recorder"
    with R.start(experiment_name="synthetic_phase7", uri=str(uri)):
        rec = R.get_recorder(); recdir = Path(rec.get_local_dir())
        rec.log_artifact(str(card / "selection_plan.json"), artifact_path="inputs")
        all_objects = {"inputs/source_hashes.pkl": source_hashes, "inputs/source_audit.pkl": source_audit,
            "inputs/development_fold_inputs.pkl": objects["parent_dev_folds"], "models/development.pkl": development,
            "models/selected_annual_refits.pkl": {"price_vol": {2017: {}}}, "analysis/training_diagnostics.pkl": audit_index,
            **audit_objects}
        for name, obj in all_objects.items():
            path = Path(name); rec.save_objects(artifact_path=str(path.parent), **{path.name: obj})
        summary = {"experiment": "synthetic_phase7", "recorder_id": rec.id, "source": plan["source"],
            "source_sha256": source_hashes, "source_audit": source_audit, "plan_sha256": sha(card / "selection_plan.json"),
            "selected": {"variant": "price_vol"}}
        (card / "summary.json").write_text(json.dumps(summary)); rec.log_artifact(str(card / "summary.json"), artifact_path="card")
    def rebind(artifact=None, value=None):
        if artifact:
            with (recdir / "artifacts" / artifact).open("wb") as stream:
                pickle.dump(value, stream)
        entries = [{"path": str(Path("qlib_recorder") / recdir.relative_to(uri) / p.relative_to(recdir)),
                    "sha256": sha(p), "bytes": p.stat().st_size}
                   for p in sorted((recdir / "artifacts").rglob("*")) if p.is_file()]
        (card / "artifact_manifest.json").write_text(json.dumps({"entries": entries}))
        plan["parent_model_source"].update({"experiment": "synthetic_phase7", "recorder_id": rec.id,
            "recorder_uri": str(uri), "artifact_manifest_sha256": sha(card / "artifact_manifest.json"),
            "selection_plan_sha256": sha(card / "selection_plan.json")})
        current = root / "research" / NEW_CARD; current.mkdir(exist_ok=True)
        (current / "selection_plan.json").write_text(json.dumps(plan))
    rebind()
    return root, plan, all_objects, recdir, rebind


def test_loads_original_retained_parent_and_mature_bindings_without_fitting(cached_parent):
    root, plan, objects, _, _ = cached_parent
    result = load_feasible_parent_inputs(root, plan)
    assert len(result["parent_seed_models"]) == 9 and len(result["parent_fold_bundles"]) == 3
    assert result["parent_source_audit"]["parent_fit_performed"] is False
    assert result["parent_source_audit"]["outer_parent_models_loaded"] is False
    for (name, year, seed), payload in result["parent_seed_models"].items():
        audit = result["parent_seed_audits"][(name, year, seed)]
        assert audit["selected_epoch"] == "anchor" and audit["F"] == 0.
        assert audit["retained_tensor_sha256"] == payload["tensor_sha256"]
        assert audit["training_dates_sha256"] == result["development_folds"][(name, year)]["audit"]["eligible_dates_sha256"]
        assert "epochs" not in audit and "training_paths" not in audit
    assert all("selected_annual_refits" not in item["artifact"] for item in result["parent_source_artifacts"])


def test_corrupt_cache_rejected_before_pickle_deserialization(cached_parent):
    root, plan, _, recdir, _ = cached_parent
    (recdir / "artifacts/models/development.pkl").write_bytes(b"corrupt pickle")
    with pytest.raises(ValueError, match="before deserialization"):
        load_feasible_parent_inputs(root, plan)


@pytest.mark.parametrize("fault", ["wrong_checkpoint", "wrong_spec", "wrong_input", "unhealthy", "risk", "risk_flag", "missing_epoch", "best_epoch"])
def test_manifest_valid_but_semantically_wrong_parent_is_rejected(cached_parent, fault):
    root, plan, objects, _, rebind = cached_parent
    artifact = "fit_checkpoints/price/2016/17/training_audit.pkl"
    changed = deepcopy(objects[artifact])
    if fault == "wrong_checkpoint":
        changed["selected_epoch"] = 0
        changed["epochs"][0]["checkpoint"] = policy_to_dict(BenchmarkAnchorPolicy(seed=17))
        # Keep the small/full audit index consistent; test selected-model binding.
        index = deepcopy(objects["analysis/training_diagnostics.pkl"])
        next(row for row in index if row["variant"] == "price" and row["seed"] == 17)["selected_epoch"] = 0
        rebind("analysis/training_diagnostics.pkl", index)
    if fault == "wrong_spec": changed["spec"]["updates"] = 119
    if fault == "wrong_input": changed["training_feature_sha256"] = "changed"
    if fault == "unhealthy": changed["qualification_eligible"] = False
    if fault == "risk": changed["selected_risks_by_cost"][25]["raw_percentage_point_gaps"] = [1., 1.]
    if fault == "risk_flag": changed["selected_risks_by_cost"][25]["risk_pass"] = [True, "true"]
    if fault == "missing_epoch": changed["epochs"].pop()
    if fault == "best_epoch": changed["epochs"][3]["F"] = .01
    rebind(artifact, changed)
    with pytest.raises(ValueError, match="blocked"):
        load_feasible_parent_inputs(root, plan)


def test_original_phase7_dates_cannot_be_replaced_by_same_value_rows(cached_parent):
    root, plan, objects, _, rebind = cached_parent
    folds = deepcopy(objects["inputs/development_fold_inputs.pkl"])
    rows = folds[("constant", 2016)]["rows"]
    rows.iloc[0, rows.columns.get_loc("signal_date")] += np.timedelta64(1, "D")
    rebind("inputs/development_fold_inputs.pkl", folds)
    with pytest.raises(ValueError, match="mature dates"):
        load_feasible_parent_inputs(root, plan)


def test_outer_cache_requires_matching_persisted_seal_and_never_fits_missing_family(cached_parent):
    root, plan, _, _, _ = cached_parent
    bundle = load_feasible_parent_inputs(root, plan)
    card = root / "research" / NEW_CARD
    seal = {"plan_sha256": sha(card / "selection_plan.json"), "selection_uses_diagnostic_signal_years": False,
            "selected": {"variant": "price_holdings"}}
    with pytest.raises(ValueError, match="persisted"):
        load_cached_outer_parents(root, plan, bundle, variant="price_holdings", years=[2017], selection_seal=seal)
    (card / "selection_seal.json").write_text(json.dumps(seal))
    outer = load_cached_outer_parents(root, plan, bundle, variant="price_holdings", years=[2017], selection_seal=seal)
    assert outer["parent_seed_models"] == {}
    assert outer["genuinely_missing"] == [("price_holdings", 2017, seed) for seed in SEEDS]
    assert outer["parent_source_audit"]["parent_fit_performed"] is False
    with pytest.raises(ValueError, match="selection seal"):
        load_cached_outer_parents(root, plan, bundle, variant="holdings_only", years=[2017], selection_seal=seal)


def test_parent_reads_refuse_active_writable_recorder(cached_parent):
    from qlib.workflow import R
    root, plan, _, _, _ = cached_parent
    with R.start(experiment_name="current_child", uri=str(root / "child")):
        child = R.get_recorder().id
        with pytest.raises(ValueError, match="close the writable Recorder"):
            load_feasible_parent_inputs(root, plan)
        assert R.get_recorder().id == child


@pytest.mark.parametrize("fault", ["existing_family_missing_input", "existing_metadata_without_family"])
def test_missing_outer_cache_is_not_reported_as_never_materialized(cached_parent, fault):
    root, plan, objects, _, rebind = cached_parent
    if fault == "existing_metadata_without_family":
        index = deepcopy(objects["analysis/training_diagnostics.pkl"])
        index.append({"variant": "price", "year": 2017, "seed": 17})
        rebind("analysis/training_diagnostics.pkl", index)
    bundle = load_feasible_parent_inputs(root, plan)
    name = "price_vol_holdings" if fault == "existing_family_missing_input" else "price_holdings"
    card = root / "research" / NEW_CARD
    seal = {"plan_sha256": sha(card / "selection_plan.json"), "selection_uses_diagnostic_signal_years": False,
            "selected": {"variant": name}}
    (card / "selection_seal.json").write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="missing|absent"):
        load_cached_outer_parents(root, plan, bundle, variant=name, years=[2017], selection_seal=seal)
