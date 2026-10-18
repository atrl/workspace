"""One frozen FINRA short-mark composition intervention; native Qlib evidence."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json
from src.model.minute_pit_allocator import MinutePitAllocator, TARGETS, _flat_index
from src.model.finra_allocator import FinraAllocator, initialization_contract
from src.data.finra_allocation_features import FINRA_FEATURES, build_finra_allocation_features
from src.data.finra_short_mix_schema import CACHE_FIELDS
from src.workflow.minute_pit_allocator import _primary_dataset, _auxiliary_dataset, _prediction_dates, _ensemble
from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights
from src.workflow.stage1_runtime_release import load_runtime_release

NAME = "stage1_finra_short_mix_allocation_20261018"
ARMS = {"B_FINRAcontext": "neutral", "C_FINRA_short_mix": "observed"}
DATES = ["signal_date", "trade_date", "exit_date"]
OUTPUTS = ["soxl_weight", "soxx_weight", "cash_weight", "tail_probability", "alpha_probability"]
REFERENCES = {"SOXX": "raw_SOXX", "strong9097": "raw_strong9097", "additional27baa": "raw_candidate27baa"}
CODE_FILES = [
    "src/workflow/finra_allocation.py", "src/model/finra_allocator.py",
    "src/data/finra_allocation_features.py", "src/analysis/finra_allocation.py",
    "src/data/finra_short_mix_schema.py",
    "src/data/qlib_native.py", "src/data/nas_market_data.py",
    "src/model/minute_pit_allocator.py", "src/backtest/stage1_domain_heads_raw.py",
    "src/workflow/minute_pit_allocator.py", "src/analysis/integrated_allocation.py",
    "src/backtest/raw_corporate_policy.py", "src/backtest/native_raw_corporate.py",
    "src/backtest/ibkr_tiered.py", "src/workflow/stage1_open_action_value.py",
    "src/workflow/stage1_runtime_release.py",
    "src/workflow/intraday_recovery_direct_allocation.py",
    "src/workflow/pit_constituent_auxiliary_training_allocation.py",
    "src/core/recorder_sources.py", "src/analysis/stage1_domain_heads.py",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity(root):
    return {name: sha(root / name) for name in CODE_FILES}


def _read_source(root, source):
    from src.data.nas_market_data import materialize_nas_evidence
    card = root / source["card"]
    binding = json.loads((card / "nas_binding.json").read_text())
    if binding["evidence"]["sha256"] != source["archive"]:
        raise ValueError("frozen source archive differs")
    summary = json.loads((card / "summary.json").read_text())
    if not Path(summary["recorder_local_dir"]).is_dir():
        materialize_nas_evidence(card / "nas_binding.json", card=card)
    return load_verified_recorder_objects(
        card, expected_recorder_id=source["recorder_id"], artifacts=source["artifacts"],
        expected_manifest_sha256=source["inventory_sha256"],
        expected_artifact_sha256=source["artifact_sha256"],
    )


def _new_model(original, mode, root):
    return FinraAllocator(mode=mode, base_features=original.base_features,
        spec={**asdict(original.spec), "root": str(root)}, seed=original.seed,
        cutoff=str(original.cutoff), train_end=str(original.train_end.date()),
        minimum_observations=original.minimum_observations)


def _source_qualification(plan, review, summary):
    """Accept conditional historical design only; this never approves a fit."""
    source = plan["sources"]["finra"]
    expected = {
        "verdict": "ACCEPT_CONDITIONAL_HISTORY_MODEL_DESIGN",
        "conditional_model_design_eligible": True,
        "model_execution_approved": False,
        "source_recorder_id": source["recorder_id"],
        "source_plan_sha256": plan["source_qualification"]["source_plan_sha256"],
        "archive_sha256": source["archive"],
    }
    if any(review.get(key) != value for key, value in expected.items()):
        raise ValueError("source review does not authorize this conditional historical design")
    if (summary.get("raw_daily_complete") is not True
            or summary.get("new_model_fits") != 0 or summary.get("new_accounts") != 0
            or summary.get("return_labels_computed") != 0
            or summary.get("historical_PIT_qualified") is not False
            or summary.get("independent_forward_qualified") is not False
            or summary.get("recorder_id") != source["recorder_id"]
            or summary.get("plan_sha256") != expected["source_plan_sha256"]):
        raise ValueError("source summary identity/scope/qualification differs")


def _source_dataset(panel):
    """Rebuild native source access from the hash-verified float64 Recorder panel.

    The sealed source/dataset object lacks handler caches; it is retained as
    historical evidence, never silently treated as a self-contained dataset.
    This reconstruction adds no labels, processors or new observations.
    """
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    if list(panel.columns) != ["datetime", "instrument", *CACHE_FIELDS]:
        raise ValueError("source panel schema differs from the six frozen source fields")
    frame = panel.set_index(["datetime", "instrument"]).loc[:, list(CACHE_FIELDS)].sort_index()
    if frame.index.has_duplicates or any(dtype != np.dtype("float64") for dtype in frame.dtypes):
        raise ValueError("source panel is not unique float64 source data")
    dataset = DatasetH(handler=DataHandlerLP.from_df(pd.concat({"feature": frame}, axis=1)),
                       segments={"all": ("2018-08-01", "2026-10-09")})
    dataset.config(dump_all=True, recursive=True)
    return dataset


def _finra_features(root, plan, inputs, source):
    """Exact source Dataset -> six formulas -> float32 Qlib -> named feature frame."""
    import qlib
    from qlib.data import D
    from src.data import build_qlib_store_from_frame
    source_dataset = _source_dataset(source["source/panel_float64"])
    prepared = source_dataset.prepare("all", col_set="feature")
    features, metadata = build_finra_allocation_features(prepared,
        signal_index=inputs["features"].index, decision_at=inputs["timing"].decision_at)
    rows = features.rename_axis("datetime").reset_index()
    rows["instrument"] = "SOXL"
    provider = root / plan["feature_provider_output"]
    manifest = build_qlib_store_from_frame(rows, provider_uri=provider, freq="day", region="us",
        as_of="2026-10-09", field_map={name.lower(): name for name in FINRA_FEATURES},
        source="Verified FINRA source Dataset float64 formulas; conditional current-byte historical research")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    expressions = ["$" + name.lower() for name in FINRA_FEATURES]
    observed = D.features(["SOXL"], expressions, start_time=features.index[0],
                          end_time=features.index[-1]).droplevel("instrument")
    observed = observed.rename(columns=dict(zip(expressions, FINRA_FEATURES))).reindex(features.index)
    observed = observed.loc[:, list(FINRA_FEATURES)]
    np.testing.assert_allclose(observed.to_numpy(), features.to_numpy(np.float32), rtol=0, atol=0, equal_nan=True)
    quantization = []
    for name in FINRA_FEATURES:
        exact, cached = features[name].to_numpy(float), observed[name].to_numpy(float)
        valid = np.isfinite(exact) & np.isfinite(cached)
        quantization.append({"field": name, "finite_rows": int(valid.sum()),
            "max_absolute_error": float(np.max(abs(exact[valid] - cached[valid]))) if valid.any() else None})
    if set(FINRA_FEATURES).intersection(inputs["features"].columns):
        raise ValueError("FINRA coordinates already present")
    inputs["features"] = inputs["features"].join(observed, how="left", validate="one_to_one")
    return features, metadata, manifest, pd.DataFrame(quantization), source_dataset


def prepare(*, root, card):
    """Read and count pinned Recorder inputs. No training, new labels or accounts."""
    import qlib
    root, card = Path(root), Path(card)
    plan = json.loads((card / "plan.json").read_text())
    digest = sha(card / "plan.json")
    if not plan.get("frozen_before_fits") or "finra" not in plan["sources"]:
        raise ValueError("model plan/source identity not frozen; no preflight or execution")
    review = plan["source_result_review"]
    if sha(root / review["path"]) != review["sha256"]:
        raise ValueError("source-result review identity differs")
    source_review = json.loads((root / review["path"]).read_text())
    release = root / plan["active_release"]["path"]
    if sha(release) != plan["active_release"]["sha256"]:
        raise ValueError("active release changed")
    loaded = load_runtime_release(root, release)
    qlib.init(provider_uri=str(root / plan["provider_init"]), region="us", kernels=1)
    sources, audits = {}, {}
    for role, source in plan["sources"].items():
        sources[role], audits[role] = _read_source(root, source)
    _source_qualification(plan, source_review, sources["finra"]["analysis/summary"])
    train, action, raw = sources["training"], sources["action"], sources["raw"]
    original = train["models/base_outer_71"]
    if original.extra_features or len(original.base_features) != 39 or original.spec.epochs != 300:
        raise ValueError("training source is not original nonlinear39-input base")
    inputs = {name: train["inputs/" + key] for name, key in [
        ("frame", "daily_frame"), ("features", "features"), ("common", "common"), ("timing", "timing")]}
    inputs["spec"] = original.spec
    finra_features, finra_metadata, feature_manifest, feature_quantization, source_dataset = _finra_features(root, plan, inputs, sources["finra"])
    dataset = _primary_dataset(inputs, train_end=str(original.train_end.date()),
                               predict_bounds=train["inputs/plan"]["outer_window"])
    auxiliary = _auxiliary_dataset(train["inputs/auxiliary"]["outer"], str(original.train_end.date()))
    rows = _flat_index(dataset.prepare("train", col_set=["feature", "label", "common", "meta"]))
    maturity = pd.to_datetime(rows[("meta", "label_available_at")])
    eligible = (maturity.notna() & maturity.lt(original.cutoff) & (rows.index <= original.train_end)
                & np.isfinite(rows["label"][list(TARGETS)].to_numpy(float)).all(axis=1))
    aux = auxiliary.prepare("train", col_set=["feature", "label", "meta"])
    if int(eligible.sum()) != 3014 or len(aux) != 20741 or len(rows["common"].columns) != 16:
        raise ValueError("frozen training domain differs")
    if pd.to_datetime(aux[("meta", "observations_end")]).max() > original.train_end:
        raise ValueError("auxiliary labels not mature")
    if str(original.cutoff) != str(pd.Timestamp(plan["training"]["cutoff"])):
        raise ValueError("cutoff differs")
    frame = raw["inputs/frame"].reset_index(drop=True).copy()
    market = action["inputs/market"].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[DATES], market[DATES])
    if len(frame) != plan["accounts"]["expected_sessions"]:
        raise ValueError("research account calendar differs")
    account_signals = pd.DatetimeIndex(frame.signal_date)
    account_metadata = finra_metadata.reindex(account_signals)
    if not account_metadata.decision_executable.eq(True).all():
        raise ValueError("unknown FINRA decision clocks cannot enter an account")
    account_decision_dates = pd.DatetimeIndex(account_metadata.decision_at).tz_localize(None).normalize()
    if not account_decision_dates.equals(pd.DatetimeIndex(frame.trade_date)):
        raise ValueError("FINRA decision date differs from frozen next-session trade date")
    early = [name for name in market if name.startswith("early_")]
    for name in early:
        np.testing.assert_array_equal(frame[name], market[name])
    targets = action["inputs/source_targets"]["frozen_current"]
    if not targets.index.equals(pd.DatetimeIndex(frame.signal_date)) or list(targets) != OUTPUTS:
        raise ValueError("active target identity/calendar differs")
    # Explicitly discard the NVDA-source probabilities/weights for A/B/C.
    for name in OUTPUTS:
        frame[name] = targets[name].to_numpy(copy=True)
    raw_plan = raw["inputs/plan"]
    if raw_plan["tariff"] != plan["tariff"] or raw_plan["initial_cash"] != plan["accounts"]["initial_cash"]:
        raise ValueError("raw reference fee or starting capital differs")
    if raw_plan["slippage_bps"] != plan["accounts"]["slippage_bps"]:
        raise ValueError("raw reference cost set differs")
    points = raw["inputs/raw_points"]
    if len(points) != 4 * len(frame) + 2 or not points.gt(0).all().all():
        raise ValueError("invalid pinned raw point grid")
    reference_books = {}
    for family, source_family in REFERENCES.items():
        for cost in plan["accounts"]["slippage_bps"]:
            book = raw[f"native/{source_family}_{cost}"]
            pd.testing.assert_frame_equal(book["authority"][DATES].reset_index(drop=True), frame[DATES])
            if not pd.DatetimeIndex(book["points"].stamp).equals(points.index):
                raise ValueError("raw reference valuation grid differs")
            np.testing.assert_allclose(book["points"].iloc[0].nav, plan["accounts"]["initial_cash"], rtol=0, atol=1e-8)
            if book["audit"]["status"] != "passed":
                raise ValueError("raw reference native validation not passed")
            reference_books[(family, cost)] = book
    domain = sources["current_shared"]
    old_plan = domain["inputs/plan"]
    if old_plan["active_release"] != plan["active_release"]:
        raise ValueError("A/current B frozen active model chain differs")
    if old_plan["tariff"] != plan["tariff"]:
        raise ValueError("A/current B fee contract differs")
    for key in ["initial_cash", "slippage_bps", "expected_sessions", "clock", "terminal"]:
        if old_plan["accounts"][key] != plan["accounts"][key]:
            raise ValueError("A/current B account contract differs: " + key)
    pd.testing.assert_frame_equal(domain["inputs/raw_points"], points)
    if domain["inputs/schedule"] != raw["inputs/schedule"]:
        raise ValueError("A/current B corporate-action schedule differs")
    for family in ["A_active_research_replay", "B_shared"]:
        for cost in plan["accounts"]["slippage_bps"]:
            book = domain[f"native/{family}_{cost}"]
            pd.testing.assert_frame_equal(book["authority"][DATES].reset_index(drop=True), frame[DATES])
            if not pd.DatetimeIndex(book["points"].stamp).equals(points.index) or book["audit"]["status"] != "passed":
                raise ValueError("A/current B native source contract differs")
            np.testing.assert_allclose(book["points"].iloc[0].nav, plan["accounts"]["initial_cash"], rtol=0, atol=1e-8)
            reference_books[(family, cost)] = book
    # The runtime loader verifies exact sealed b472 classifiers and f696 bundle.
    selected = loaded["action_summary"]["source_stage1"]["selected_model"]
    if selected != action["controller/source_recovery_selection"]:
        raise ValueError("active recovery selection differs from saved source")
    action_spec = action["inputs/plan"]["action_spec"]
    for name in ["logistic_c", "execute_probability_threshold", "maximum_skippable_turnover", "mandatory_derisk_beta_delta"]:
        if action_spec[name] != loaded["action_summary"]["selected_model"][name]:
            raise ValueError("active action contract differs: " + name)
    initialization = []
    for seed in plan["training"]["seeds"]:
        old = train[f"models/base_outer_{seed}"]
        left, right = _new_model(old, "neutral", root), _new_model(old, "observed", root)
        initialization.append({"seed": seed, **initialization_contract(old, left, right)})
    receipt = {"plan_sha256": digest, "primary_rows": int(eligible.sum()), "auxiliary_rows": len(aux),
               "primary_last_mature": str(maturity.loc[eligible].max()),
               "aux_first_signal": str(aux.index.get_level_values("datetime").min()),
               "aux_last_signal": str(aux.index.get_level_values("datetime").max()),
               "aux_observations_end_max": str(pd.to_datetime(aux[("meta", "observations_end")]).max()),
               "source_hashes_checked": True, "source_api": ["Recorder.load_object", "DatasetH.prepare"],
               "main_inputs": 39, "additional_allocation_inputs": 6, "common_inputs": 16, "account_sessions": len(frame),
               "early_fields_exact_source_equality": early, "reused_reference_accounts": len(reference_books),
               "active_bundle_and_classifier_hashes_verified": True, "initialization": initialization,
               "new_model_fits": 0, "new_accounts": 0, "dirty_runtime_providers_used_as_input": False,
               "FINRA_ready_primary_training_rows": int(rows.loc[eligible, ("feature", "finra_both_ready")].eq(1).sum()),
               "account_decision_clocks_all_executable": True,
               "nonexecutable_feature_rows_retained": int((~finra_metadata.decision_executable).sum()),
               "historical_PIT_qualified": False, "independent_forward_qualified": False,
               "source_dataset_access": "verified source/panel_float64 -> DataHandlerLP.from_df -> DatasetH.prepare; sealed source/dataset is not self-contained",
               "feature_quantization": feature_quantization.to_dict("records")}
    atomic_json(receipt, card / "preflight.json")
    return dict(plan=plan, plan_sha256=digest, sources=sources, source_audits=audits, dataset=dataset,
                auxiliary=auxiliary, frame=frame, market=market, points=points,
                schedule=raw["inputs/schedule"], guard=raw["inputs/repair_1_calendar_guard"],
                references=reference_books, controller=loaded["models"], selection=selected,
                recovery_spec=action["inputs/plan"]["recovery_spec"], action_spec=action_spec,
                receipt=receipt, original=original, daily_calendar_frame=inputs["frame"],
                finra_features=finra_features, finra_metadata=finra_metadata,
                feature_manifest=feature_manifest, feature_quantization=feature_quantization,
                source_dataset=source_dataset)


def _approval(root, card, digest):
    review_path, execution_path = card / "design_review.json", card / "execution_approval.json"
    if not review_path.exists() or not execution_path.exists():
        raise ValueError("formal fits/accounts require independent design and explicit execution approval")
    review, execution = json.loads(review_path.read_text()), json.loads(execution_path.read_text())
    if review.get("plan_sha256") != digest or review.get("verdict") not in {"APPROVE_DESIGN", "APPROVE_EXECUTION"}:
        raise ValueError("independent design approval does not match frozen plan")
    if execution.get("plan_sha256") != digest or execution.get("approved") is not True:
        raise ValueError("explicit execution approval missing")
    if execution.get("code_sha256") != code_identity(root):
        raise ValueError("execution code identity changed")
    return {"review_sha256": sha(review_path), "execution_approval_sha256": sha(execution_path),
            "code_sha256": execution["code_sha256"]}


def _provider(root, prepared):
    import qlib
    from qlib.data import D
    from src.data import build_qlib_store_from_frame
    points = prepared["points"]
    guard = prepared["guard"]["$raw_execution_price"].unstack("instrument")[["SOXL", "SOXX"]]
    if guard.index[0] <= points.index[-1]:
        raise ValueError("native guard must be strictly after valuation endpoint")
    combined = pd.concat([points, guard]).sort_index()
    rows = [{"datetime": stamp, "instrument": symbol,
             **{field: float(value) for field in ["open", "high", "low", "close", "execution_price", "raw_execution_price"]}}
            for symbol in ["SOXL", "SOXX"] for stamp, value in combined[symbol].items()]
    provider = root / prepared["plan"]["provider_output"]
    fields = ["open", "high", "low", "close", "execution_price", "raw_execution_price"]
    manifest = build_qlib_store_from_frame(pd.DataFrame(rows), provider_uri=provider, freq="1min", region="us",
        as_of="2026-10-09", field_map={name: name for name in fields},
        source="Exact hash-verified raw Recorder quotes; following guard is not used for policy or NAV")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    observed = D.features(["SOXL", "SOXX"], ["$raw_execution_price"], start_time=combined.index[0],
                          end_time=combined.index[-1], freq="1min")["$raw_execution_price"].unstack("instrument")
    pd.testing.assert_index_equal(observed.index, combined.index, check_names=False)
    np.testing.assert_array_equal(observed[["SOXL", "SOXX"]].to_numpy(), combined.to_numpy())
    return provider, manifest


def run(*, root, card, prepared=None):
    """Run only after independent approval; attempts are durable and never retried silently."""
    from qlib.workflow import R
    from src.analysis.stage1_domain_heads import information_metrics
    root, card = Path(root), Path(card)
    digest = sha(card / "plan.json")
    approval = _approval(root, card, digest)
    p = prepare(root=root, card=card) if prepared is None else prepared
    if p["plan_sha256"] != digest:
        raise ValueError("prepared data plan differs")
    plan = p["plan"]
    state_path = card / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "plan_sha256": digest, "fit_attempts": [], "fits_done": [], "path_attempts": [],
        "paths_done": [], "native_attempts": [], "native_done": []}
    if state["plan_sha256"] != digest:
        raise ValueError("attempt log belongs to another plan")
    tracking = card / "qlib_recorder"
    kwargs = {"experiment_name": NAME, "uri": str(tracking)}
    if state.get("recorder_id"):
        kwargs.update(recorder_id=state["recorder_id"], resume=True)
    predictions, training = {}, []
    with R.start(**kwargs):
        rec = R.get_recorder()
        state.update(recorder_id=rec.id, experiment_id=str(rec.experiment_id))
        atomic_json(state, state_path)
        if not state.get("sources_saved"):
            for dataset in (p["dataset"], p["auxiliary"], p["source_dataset"]):
                dataset.config(dump_all=True, recursive=True)
            rec.save_objects(artifact_path="inputs", plan=plan, source_audits=p["source_audits"],
                preflight=p["receipt"], approval=approval, dataset=p["dataset"], auxiliary_dataset=p["auxiliary"],
                frame=p["frame"], market=p["market"], raw_points=p["points"], schedule=p["schedule"],
                guard=p["guard"], controller=p["controller"], selection=p["selection"],
                recovery_spec=p["recovery_spec"], action_spec=p["action_spec"],
                finra_features=p["finra_features"], finra_metadata=p["finra_metadata"],
                feature_manifest=p["feature_manifest"], feature_quantization=p["feature_quantization"],
                source_dataset=p["source_dataset"])
            rec.save_objects(artifact_path="sources", **{name.replace("/", "__"): (root / name).read_text() for name in CODE_FILES})
            state["sources_saved"] = True
            atomic_json(state, state_path)
        for family, mode in ARMS.items():
            for seed in plan["training"]["seeds"]:
                key = f"{family}_{seed}"
                original = p["sources"]["training"][f"models/base_outer_{seed}"]
                if key in state["fits_done"]:
                    model = rec.load_object("models/" + key)
                else:
                    if key in state["fit_attempts"] or len(state["fit_attempts"]) >= 6:
                        raise ValueError("fit attempt already consumed or fixed budget exhausted")
                    state["fit_attempts"].append(key)
                    atomic_json(state, state_path)
                    model = _new_model(original, mode, root)
                    model.fit(p["dataset"], auxiliary_dataset=p["auxiliary"])
                    for name in ["training_rows", "auxiliary_rows", "last_mature_label_at", "epochs", "checkpoint_policy"]:
                        if model.audit[name] != original.audit[name]:
                            raise ValueError("paired training contract differs: " + name)
                    if model.primary_scaler != original.primary_scaler or model.common_scaler != original.common_scaler:
                        raise ValueError("matched source scaler differs")
                    if mode == "observed":
                        paired_control = rec.load_object(f"models/B_FINRAcontext_{seed}")
                        if model.extra_scaler != paired_control.extra_scaler:
                            raise ValueError("paired FINRA training-only scalers differ")
                    rec.save_objects(artifact_path="models", **{key: model})
                    state["fits_done"].append(key)
                    atomic_json(state, state_path)
                prediction = _prediction_dates(model.predict(p["dataset"]))
                predictions[(family, seed)] = prediction
                rec.save_objects(artifact_path="predictions", **{key: prediction})
                training.append({"family": family, "seed": seed, **model.audit})
                rec.log_metrics(completed_fits=len(state["fits_done"]))
                print("completed fit", key, flush=True)
        ensembles = {family: _ensemble(predictions, family, plan["training"]["seeds"]) for family in ARMS}
        dates = pd.DatetimeIndex(p["market"].signal_date)
        returns = p["market"].set_index("signal_date")[["soxl_return", "soxx_return"]]
        targets = {family: _smooth_weights(values.reindex(dates), spec=p["original"].spec,
                     candidate=p["original"].candidate, asset_returns=returns) for family, values in ensembles.items()}
        info = information_metrics(predictions, ensembles, p["dataset"])
        rec.save_objects(artifact_path="analysis", training=pd.DataFrame(training), ensembles=ensembles,
                         smoothed_targets=targets, classification=info)
    # Resume the SAME Recorder for accounting. Native validators reinitialize
    # Qlib, so the lower-level public Recorder lifecycle is used without an
    # active R experiment. Any second-stage failure is recorded as FAILED.
    from qlib.workflow.recorder import Recorder
    rec.start_run()
    try:
        return _complete_accounts(root=root, card=card, p=p, plan=plan, digest=digest,
            rec=rec, state=state, state_path=state_path, targets=targets,
            training=training, info=info, tracking=tracking)
    except BaseException:
        rec.end_run(status=Recorder.STATUS_FA)
        raise


def _complete_accounts(*, root, card, p, plan, digest, rec, state, state_path,
                       targets, training, info, tracking):
    from qlib.workflow.recorder import Recorder
    from src.backtest.stage1_domain_heads_raw import replay_stage1_raw
    from src.backtest.native_raw_corporate import native_raw_validate
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
    from src.analysis.finra_allocation import analyze_accounts
    from src.data.nas_market_data import archive_nas_recorder
    # No information-score gate here: all fixed economic paths must be executed.
    provider, manifest = _provider(root, p)
    rec.save_objects(artifact_path="inputs", native_provider_manifest=manifest)
    recovery = IntradayRecoveryDirectAllocationSpec(root=root, **p["recovery_spec"])
    action = Stage1OpenActionValueSpec(root=root, **p["action_spec"])
    execution = {"initial_cash": plan["accounts"]["initial_cash"], "tariff": plan["tariff"]}
    books = dict(p["references"])
    for family in ARMS:
        frame = p["frame"].copy()
        if family in targets:
            for name in OUTPUTS:
                frame[name] = targets[family][name].to_numpy(copy=True)
        for cost in plan["accounts"]["slippage_bps"]:
            key = f"{family}_{cost}"
            if key in state["paths_done"]:
                path = rec.load_object("paths/" + key)
            else:
                if key in state["path_attempts"] or len(state["path_attempts"]) >= 6:
                    raise ValueError("path attempt consumed or budget exhausted")
                state["path_attempts"].append(key)
                atomic_json(state, state_path)
                path = replay_stage1_raw(frame, point_prices=p["points"], schedule=p["schedule"], plan=execution,
                    controller=p["controller"], recovery_spec=recovery, action_spec=action,
                    selection=p["selection"], slippage_bps=cost)
                rec.save_objects(artifact_path="paths", **{key: path})
                state["paths_done"].append(key)
                atomic_json(state, state_path)
            if key in state["native_done"]:
                book = rec.load_object("native/" + key)
            else:
                if key in state["native_attempts"] or len(state["native_attempts"]) >= 6:
                    raise ValueError("native attempt consumed or budget exhausted")
                state["native_attempts"].append(key)
                atomic_json(state, state_path)
                book = native_raw_validate(path, provider_uri=provider, point_prices=p["points"], schedule=p["schedule"],
                    tariff=plan["tariff"], cost=cost, initial_cash=plan["accounts"]["initial_cash"])
                rec.save_objects(artifact_path="native", **{key: book})
                state["native_done"].append(key)
                atomic_json(state, state_path)
            books[(family, cost)] = book
            rec.log_metrics(completed_native_accounts=len(state["native_done"]))
            print("completed native account", key, flush=True)
    results = analyze_accounts(books, p["daily_calendar_frame"], plan)
    for name, value in results.items():
        if isinstance(value, pd.DataFrame):
            small_tables = {"metrics", "paired_comparison", "annual_metrics", "quarterly_metrics",
                            "rolling_summary", "uncertainty", "seed_summary"}
            if name in small_tables:
                if len(value) > 500:
                    raise ValueError("unexpectedly large slim sidecar: " + name)
                value.to_csv(card / (name + ".csv"), index=False)
            rec.save_objects(artifact_path="analysis", **{name: value})
        elif name == "rolling":
            rec.save_objects(artifact_path="analysis/rolling", **value)
    pd.DataFrame(training).to_csv(card / "training_summary.csv", index=False)
    info.to_csv(card / "information_metrics.csv", index=False)
    economic = all(v for k, v in results["gates"].items() if k not in ["qualified_independent_forward", "fresh_fullminute_risk_qualified"])
    summary = {"experiment": NAME, "plan_sha256": digest, "decision": "historical_economic_pass_needs_risk_and_independent_forward" if economic else "not_promoted_fixed_FINRA_short_mix_v1",
        "objective_achieved": False, "gates": results["gates"], "coarse_economic_gates_pass": economic,
        "new_fits": len(state["fits_done"]), "new_native_accounts": len(state["native_done"]),
        "reused_native_accounts": 15, "total_comparison_accounts": len(books), "sessions_each": len(p["frame"]),
        "recorder_id": rec.id, "experiment_id": str(rec.experiment_id), "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / str(rec.experiment_id) / rec.id), "provider_uri": str(provider),
        "region": "us", "as_of": "2026-10-09", "history_status": "all_history_reused_not_independent",
        "historical_FINRA_PIT_qualified": False, "FINRA_vintage_status": "current bytes; unknown historical revisions",
        "production_changed": False, "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "sources/*", "models/*", "predictions/*", "paths/*", "native/*", "analysis/rolling/*"]}}
    rec.save_objects(artifact_path="analysis", summary=summary, gates=results["gates"])
    artifact_root = Path(summary["recorder_local_dir"]) / "artifacts"
    rec.log_metrics(registered_artifact_count=sum(path.is_file() for path in artifact_root.rglob("*")))
    rec.end_run(status=Recorder.STATUS_FI)
    atomic_json({"recorder_id": rec.id, "entries": [{"path": str(path.relative_to(card)), "sha256": sha(path), "bytes": path.stat().st_size}
                 for path in sorted(artifact_root.rglob("*")) if path.is_file()]}, card / "verification_manifest.json")
    binding = json.loads((root / plan["sources"]["training"]["card"] / "nas_binding.json").read_text())
    binding.pop("evidence", None)
    atomic_json(binding, card / "nas_binding.json")
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card,
                                                   recorder_id=rec.id, experiment_id=str(rec.experiment_id))
    atomic_json(summary, card / "summary.json")
    metric = results["metrics"]
    lines = ["# Stage1 FINRA 场外卖空标记占比差：固定研究", "",
        "完成B_FINRAcontext与C_FINRA_short_mix各3个seed，共6次固定300轮拟合。主3014条、辅助20741条；原39主输入、16共享输入、损失、样本与Stage1控制链不变。仅主allocation接6项新坐标，B/C只有q信息不同。",
        "", "新增B/C各0/2/5bp共6个原生账户；严格复用A、原shared B、SOXX、9097、27baa共15账户；每账户944区间。各账户保留原policy与独立成本状态，原始股数及分红应收经Qlib核对。",
        "", "|方案|滑点bp|累计净收益%|收盘最大回撤%|CVaR5%|", "|---|---:|---:|---:|---:|"]
    for row in metric.itertuples():
        lines.append(f"|{row.family}|{row.slippage_bps}|{row.return_pct:.3f}|{row.close_max_drawdown_pct:.3f}|{row.cvar5_pct:.3f}|")
    failed = [name for name, passed in results["gates"].items() if not passed]
    lines += ["", "决定：**" + summary["decision"] + "**。", "失败或未取得的门：" + ", ".join(failed) + "。",
        "", "所有年度、季度、126交易日窗口及每个seed分类诊断均保留；不择窗、择seed或按分类分数取消账户。全窗财富和配对CI对5个required比较，年度/rolling门相对SOXX。首尾partial年度单列。",
        "", "FINRA ShortVolume包含ShortExemptVolume，CNMS仅场外RTH子集；q不是shortinterest、情绪、净库存或作者operationalshorting复现。18:00仅历史可用性假设，当前字节修订与首次接收未认证。结果不能授予PIT或前向资格。",
        "", "采样回撤不认证全分钟风险，重复市场历史不独立。生产未改，稳定超额目标未完成。",
        "", ("经济门通过仍须另获来源、风险与独立前向资格。" if economic else "关闭固定q_v1，不反号、平滑、择窗口、调损失或控制器救援。"),
        "", "Qlib Recorder：`" + rec.id + "`；来源与NAS身份见summary.json、verification_manifest.json、nas_binding.json。"]
    (card / "conclusion.md").write_text("\n".join(lines) + "\n")
    (card / "workflow.yaml").write_text(
        "experiment: " + NAME + "\nentrypoint: src.workflow.finra_allocation.run\nphase: completed\nprovider_uri: "
        + str(provider) + "\nrecorder_uri: " + str(tracking) + "\nartifact_policy: slim_card_recorder_first\n")
    state["completed"] = True
    atomic_json(state, state_path)
    return summary
