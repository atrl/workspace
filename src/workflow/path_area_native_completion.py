"""Native-only continuation of the sealed P2 repair_2 completed stages.

No allocation model, area builder, Python account or label generator is called.
The native account runs before opening the new output Recorder because its
Qlib initialization must not replace an active Recorder manager.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd

from src.workflow import path_area_bridge as p2

CARD = p2.P2_CARD / "repair_4"
SOURCE_CARD = p2.P2_CARD / "repair_2"
SOURCE_RID = "ecce5facd2424cc6a8e79e102d89cdee"
SOURCE_EID = "517546296839356582"
EXPERIMENT = "path_area_action_bridge_20261018_P2_repair_4"
SHADOW_NAMES = ("trace", "corporate_receipts", "targets", "prebranch_state", "decisions",
                "python_audit", "path_area_inputs", "points", "replay", "orders")
LABEL_NAMES = ("labels", "action_label_frame", "action_label_dataset")
CHECKPOINT_NAMES = tuple(f"shadow_python/{name}" for name in SHADOW_NAMES) + tuple(
    f"local_labels/{name}" for name in LABEL_NAMES) + (
        "sources/identity", "sources/load_audit", "sources/H1_projection")
P1_NAMES = ("inputs/datasets", "inputs/admitted_source_objects", "inputs/daily_calendar")
BUDGET = {"new_allocation_fits": 0, "new_model_forward_calls": 0,
          "area_recomputations": 0, "common_shadow_policy_paths": 0,
          "local_action_label_passes": 0, "common_shadow_native_accounts": 1,
          "checkpoint_objects_loaded": len(CHECKPOINT_NAMES), "parent_artifact_loads_maximum": 3,
          "native_derived_point_provider_builds": 1, "native_derived_point_provider_readbacks": 1,
          "action_classifier_fits": 0, "evaluation_2023_plus_forward_or_accounts": 0,
          "new_market_source_queries": 0, "new_market_HTTP": 0}


def native_completion_implementation_identity(root=None):
    root = p2._resolve_root(root)
    _, plan = p2._load_plan(root)
    files = dict(p2._implementation_identity(root, plan)["files"])
    for name in ("src/workflow/path_area_native_completion.py", "src/core/completed_checkpoints.py",
                 "tests/test_path_area_native_completion.py", str(CARD / "repair_plan.json"),
                 str(CARD / "workflow.yaml"), str(CARD / "repair_4.ipynb")):
        path = root / name
        files[name] = p2._notebook_source_identity(path) if path.suffix == ".ipynb" else p2._sha_file(path)
    return {"files": files, "sha256": p2._canonical_sha(files)}


def _verified_code_snapshot(root, identity):
    """Retain exact UTF-8 source bytes, bound to the reviewed implementation."""
    if p2._canonical_sha(identity["files"]) != identity["sha256"]:
        raise ValueError("blocked: source snapshot implementation identity differs")
    snapshot = {}
    for name, expected in identity["files"].items():
        if name.startswith("environment:"):
            continue
        path = root / name
        payload = path.read_bytes()
        source = payload.decode("utf-8")
        if path.suffix == ".ipynb":
            notebook = json.loads(source)
            cells = [{"id": cell.get("id"), "cell_type": cell["cell_type"],
                      "source": cell.get("source", [])}
                     for cell in notebook.get("cells", [])
                     if cell.get("id") != "path-p2-conclusion"]
            observed = p2._canonical_sha(cells)
        else:
            observed = p2._sha_bytes(payload)
        if observed != expected:
            raise ValueError("blocked: source snapshot differs from frozen implementation: " + name)
        snapshot[name] = source
    return snapshot


def _no_writer():
    from qlib.workflow import R
    active = R.exp_manager.active_experiment
    if active is not None and active.active_recorder is not None:
        raise ValueError("blocked: native continuation requires no active writable Recorder")


def _authorization(root, identity):
    card = root / CARD
    repair_plan = json.loads((card / "repair_plan.json").read_text())
    review_path = card / "repair_implementation_review.json"
    go_path = card / "repair_execution_authorization.json"
    if not review_path.is_file() or not go_path.is_file():
        raise PermissionError("blocked: repair_4 scoped review and separate native execution GO required")
    review, go = (json.loads(path.read_text()) for path in (review_path, go_path))
    bindings = {"plan_sha256": p2.PLAN_SHA256,
                "failure_manifest_sha256": repair_plan["failure_manifest_sha256"],
                "repair_3_failure_receipt_sha256": repair_plan["repair_3_failure_receipt_sha256"],
                "repair_plan_sha256": p2._sha_file(card / "repair_plan.json"),
                "repair_implementation_sha256": identity["sha256"]}
    if (any(review.get(key) != value or go.get(key) != value for key, value in bindings.items())
            or review.get("verdict") != "APPROVE_P2_REPAIR_4_FOR_SEPARATE_EXECUTION_GO"
            or review.get("material_blockers") != [] or go.get("approved") is not True
            or go.get("owner") != "/root" or go.get("scope") != "actual_P2_repair_4_execution"
            or go.get("P3") is not False or go.get("budget") != BUDGET
            or go.get("repair_implementation_review_sha256") != p2._sha_file(review_path)):
        raise PermissionError("blocked: repair_4 authorization differs from code, seal or bounded scope")
    allowed = {"repair_plan.json", "workflow.yaml", "repair_4.ipynb",
               "repair_implementation_review.json", "repair_execution_authorization.json",
               "checkpoint_schema_inspection.json"}
    if any(path.name not in allowed for path in card.iterdir()):
        raise FileExistsError("blocked: repair_4 already has execution evidence; no automatic retry")
    return repair_plan, {"sha256": p2._sha_file(go_path)}, {"sha256": p2._sha_file(review_path)}


def _unpack_cached(objects):
    from src.core.native_evidence import prepare_groups
    shadow = {name: objects[f"shadow_python/{name}"] for name in SHADOW_NAMES}
    shadow["audit"] = shadow.pop("python_audit")
    labels = {name: objects[f"local_labels/{name}"] for name in LABEL_NAMES}
    # These are the real native_raw_validate interface fields, not a new replay schema.
    required = {"points": ("stamp", "nav", "cash", "receivable", "q_soxl", "q_soxx"),
                "orders": ("stamp", "symbol", "raw_shares", "sell", "total"),
                "replay": ("trade_date", "exit_date", "native_expected_nav")}
    for name, columns in required.items():
        if not isinstance(shadow[name], pd.DataFrame) or not set(columns).issubset(shadow[name].columns):
            raise ValueError("blocked: cached native interface schema differs: " + name)
    stamps = pd.DatetimeIndex(shadow["points"].stamp)
    if (stamps.empty or stamps.has_duplicates or not stamps.is_monotonic_increasing
            or stamps.tz is not None or (stamps.year >= 2023).any()):
        raise ValueError("blocked: cached native point grid differs")
    if not labels["labels"].index.equals(shadow["decisions"].index):
        raise ValueError("blocked: cached labels and decisions differ")
    dates = pd.DatetimeIndex(shadow["decisions"].index)
    if (dates.empty or dates.name != "trade_date" or dates.tz is not None
            or dates.has_duplicates or not dates.is_monotonic_increasing
            or not dates.equals(pd.DatetimeIndex(shadow["replay"].trade_date))
            or not dates.equals(shadow["trace"].index)):
        raise ValueError("blocked: cached decision/native/trace date grids differ")
    pd.testing.assert_frame_equal(prepare_groups(labels["action_label_dataset"], "all"),
                                  labels["action_label_frame"], check_exact=True)
    _bridge_trace(shadow)
    return shadow, labels


def _bridge_trace(shadow):
    trace = shadow["trace"].reset_index().copy()
    trace["eligibility_reasons"] = trace.eligibility_reasons.map(
        lambda value: "|".join(value) if isinstance(value, (list, tuple)) else str(value))
    # Fail before the account on any cached list/dict field the report cannot hash.
    pd.util.hash_pandas_object(trace, index=True)
    return trace


def _completed_boundary(root, repair_plan):
    card = root / SOURCE_CARD
    manifest_path = card / "failure_manifest.json"
    if p2._sha_file(manifest_path) != repair_plan["failure_manifest_sha256"]:
        raise ValueError("blocked: completed-stage seal differs")
    manifest = json.loads(manifest_path.read_text())
    state_path = card / "repair_execution_state.json"
    if (manifest.get("completed_stage") != "python_shadow_and_local_labels"
            or manifest.get("failed_stage") != "native_qlib_initialization_inside_writer"
            or manifest.get("execution_state_path") != state_path.name
            or p2._sha_file(state_path) != manifest.get("execution_state_sha256")):
        raise ValueError("blocked: completed-stage failure boundary differs")
    previous = json.loads(state_path.read_text())
    if (previous.get("status") != "failed" or previous.get("recorder_id") != SOURCE_RID
            or previous.get("shadow_policy_paths") != 1 or previous.get("native_accounts") != 0
            or previous.get("label_pairs") != 352
            or previous.get("exception_type") != "RecorderInitializationError"
            or previous.get("python_shadow_artifacts_saved") is not True
            or previous.get("local_label_artifacts_saved") is not True
            or previous.get("area_calculation_status") != "completed"
            or previous.get("local_label_generation_status") != "completed"):
        raise ValueError("blocked: completed-stage counters differ")


def _prior_failure_boundary(root, repair_plan):
    card = root / p2.P2_CARD / "repair_3"
    receipt_path = card / "failure_receipt.json"
    if p2._sha_file(receipt_path) != repair_plan["repair_3_failure_receipt_sha256"]:
        raise ValueError("blocked: repair_3 immutable failure receipt differs")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status") != "closed_failed_before_native"
            or receipt.get("new_recorder_created") is not False
            or receipt.get("counts", {}).get("checkpoint_artifacts_loaded") != 16
            or receipt.get("counts", {}).get("source_artifacts_loaded") != 3
            or receipt.get("counts", {}).get("native_account_attempts") != 0
            or p2._sha_file(card / "repair_execution_state.json") != receipt.get("execution_state_sha256")):
        raise ValueError("blocked: repair_3 closed failure boundary differs")


def _provider_manifest_integrity(identity):
    """Verify location-dependent hashes against that provider's own metadata."""
    from src.core import stable_hash
    manifest = identity["manifest"]
    audit_payload = {"provider_uri": manifest["provider_uri"], "market": manifest["market"],
        "region": manifest["region"], "instruments": manifest["instruments"], "fields": manifest["fields"],
        "start": manifest["calendar_start"], "end": manifest["calendar_end"],
        "max_datetime": manifest["max_datetime"], "row_count": manifest["row_count"]}
    build_payload = {key: value for key, value in manifest.items()
                     if key not in ("operation", "status", "hash", "source_api")}
    if (str(Path(identity["provider_path"]).resolve()) != manifest["provider_uri"]
            or stable_hash(audit_payload) != manifest["audit_snapshot_hash"]
            or stable_hash(build_payload) != manifest["hash"]):
        raise ValueError("blocked: provider own-URI audit or manifest integrity differs")


def _verify_quote_identity(old, new):
    # Provider path and manifest hash include the temporary directory name.
    # Calendar/instrument/feature bytes and the content snapshot must be exact.
    def quote_files(identity):
        return {row["path"]: (row["sha256"], row["bytes"])
                for row in identity["provider_tree"]["entries"]
                if row["path"].startswith(("features/", "calendars/", "instruments/"))}
    if len(quote_files(old)) != 14 or len(quote_files(new)) != 14:
        raise ValueError("blocked: native quote inventory requires exactly 14 files")
    keys = ("rows", "first_point", "last_point", "native_readback", "executor_guard")
    _provider_manifest_integrity(old)
    _provider_manifest_integrity(new)
    def source_manifest(identity):
        return {key: value for key, value in identity["manifest"].items()
                if key not in ("provider_uri", "audit_snapshot_hash", "hash")}
    if (any(old.get(key) != new.get(key) for key in keys)
            or not quote_files(old) or quote_files(old) != quote_files(new)
            or source_manifest(old) != source_manifest(new)):
        raise ValueError("blocked: rebuilt native provider quote/source identity differs")
    return {"status": "passed", "exact_calendar_instrument_feature_files": len(quote_files(old)),
            "own_URI_audit_and_manifest_hashes_verified": True,
            "location_independent_manifest_fields_equal": True,
            "guard_and_point_shape_equal": True}


def _native_then_record(*, shadow, labels, provider, point_prices, schedule, tariff,
                        plan, calendar, tracking, source_identity, state, persist,
                        implementation_identity, code_snapshot, source_audit, checkpoint_audit):
    from qlib.workflow import R
    from src.backtest.native_raw_corporate import native_raw_validate
    from src.analysis.decision_bridge import build_decision_bridge

    _no_writer()
    state["native_account_attempts"] += 1
    persist()
    native = native_raw_validate(shadow, provider_uri=provider, point_prices=point_prices,
        schedule=schedule, tariff=tariff, cost=float(plan["common_training_shadow"]["slippage_bps"]),
        initial_cash=float(plan["common_training_shadow"]["initial_cash_usd"]))
    if native["audit"].get("status") != "passed":
        raise ValueError("blocked: native Qlib reconciliation failed")
    state["native_accounts"] = 1
    state["native_status"] = "passed_before_output_recorder"
    persist()
    _no_writer()
    # All init/provider/source operations have now completed. No init in this block.
    with R.start(experiment_name=EXPERIMENT, recorder_name="path_area_action_bridge_P2_repair_4",
                 uri=str(tracking)):
        recorder = R.get_recorder()
        state["recorder_id"] = str(recorder.id)
        persist()
        combined = {**shadow["audit"], **native["audit"], "python_to_native_status": "passed",
                    "certifies_alpha": False, "certifies_PIT_or_first_receipt": False}
        trace = _bridge_trace(shadow)
        bridge = build_decision_bridge(trace, native["authority"], native_audit=native["audit"],
            source_identity={**source_identity, "current_P2_recorder_id": str(recorder.id),
                             "current_P2_experiment_id": str(recorder.experiment_id)},
            fields={"source_target": ["source_target_soxl_weight", "source_target_soxx_weight", "source_target_cash_weight"],
                    "projected_target": ["projected_target_soxl_weight", "projected_target_soxx_weight", "projected_target_cash_weight"],
                    "postfill_weights": ["postfill_soxl_weight", "postfill_soxx_weight", "postfill_cash_weight"],
                    "order_count": "filled_order_count", "fees_usd": "total_order_fees_usd"},
            native_calendar=calendar)
        recorder.log_params(plan_sha256=p2.PLAN_SHA256, failed_recorder_id=SOURCE_RID,
            repair_implementation_sha256=implementation_identity["sha256"],
            native_executed_before_writer=True, new_allocation_fits=0, new_model_forward_calls=0)
        recorder.save_objects(artifact_path="sources", identity=source_identity,
                              load_audit=source_audit, completed_checkpoint_audit=checkpoint_audit)
        recorder.save_objects(artifact_path="implementation", identity=implementation_identity,
                              code=code_snapshot)
        recorder.save_objects(artifact_path="shadow_python", **{
            name: shadow["audit"] if name == "python_audit" else shadow[name] for name in SHADOW_NAMES})
        recorder.save_objects(artifact_path="local_labels", **labels)
        recorder.save_objects(artifact_path="native", native_report=native["native_report"],
            native_positions=native["native_positions"], native_orders=native["native_orders"],
            native_points=native["points"], native_corporate_receipts=native["native_corporate_receipts"],
            native_reconciliation=native["reconciliation"], native_audit=native["audit"],
            teacher_native_audit=combined, decision_bridge_daily=bridge["daily"],
            decision_bridge_periods=bridge["periods"], decision_bridge_rolling_126=bridge["rolling_126"])
        recorder.log_metrics(native_reconciliation_pass=1, shadow_sessions=len(shadow["replay"]),
                             mature_action_labels=int(labels["labels"].label_status.eq("mature").sum()))
    return recorder, native, bridge


def run_path_area_native_completion(*, root=None, repair_number=4):
    if repair_number != 4:
        raise ValueError("blocked: closed repair versions cannot execute; only reviewed repair_4 is supported")
    root = p2._resolve_root(root)
    _, plan = p2._load_plan(root)
    inherited = p2._verify_inherited(root, plan)
    identity = native_completion_implementation_identity(root)
    repair_plan, go, review = _authorization(root, identity)
    _completed_boundary(root, repair_plan)
    _prior_failure_boundary(root, repair_plan)
    code_snapshot = _verified_code_snapshot(root, identity)
    card = root / CARD
    state_path = card / "repair_execution_state.json"
    state = {"phase": "P2_repair_4", "status": "started", "plan_sha256": p2.PLAN_SHA256,
        "execution_authorization_sha256": go["sha256"], "native_account_attempts": 0,
        "native_accounts": 0, "checkpoint_artifacts_load_attempted": 0,
        "checkpoint_artifacts_loaded": 0, "source_artifacts_loaded": 0,
        "new_allocation_fits": 0, "new_model_forward_calls": 0, "area_calculations": 0,
        "shadow_policy_paths": 0, "local_action_label_passes": 0,
        "new_market_HTTP": 0, "new_market_source_queries": 0}
    def persist():
        p2._atomic_json(state_path, state)
    persist()
    provider = None
    try:
        import qlib
        from src.core.completed_checkpoints import read_completed_checkpoint
        from src.core.native_evidence import finalize_evidence, prepare_groups
        from src.core.recorder_sources import load_verified_recorder_objects
        from src.backtest.raw_corporate_policy import make_schedule
        p1_plan = json.loads((root / plan["inherits"]["P1_plan"]["path"]).read_text())
        qlib.init(provider_uri=str(root / p1_plan["sources"]["daily_calendar"]["provider_uri"]),
                  region="us", kernels=1)
        def progress(event, name):
            if event == "attempt":
                state["checkpoint_artifacts_load_attempted"] += 1
                state["current_checkpoint_artifact"] = name
            elif event == "completed":
                state["checkpoint_artifacts_loaded"] += 1
                state["current_checkpoint_artifact"] = None
            else:
                state["checkpoint_partial_load_unknown_on_exception"] = True
            persist()
        objects, checkpoint_audit = read_completed_checkpoint(root / SOURCE_CARD,
            manifest_sha256=repair_plan["failure_manifest_sha256"], recorder_id=SOURCE_RID,
            experiment_id=SOURCE_EID, experiment_name="path_area_action_bridge_20261018_P2_repair_2",
            artifacts=CHECKPOINT_NAMES, progress=progress)
        shadow, labels = _unpack_cached(objects)
        expected = {name: plan["source_admission"]["read_allowlist"][name] for name in P1_NAMES}
        source, source_audit = load_verified_recorder_objects(root / p2.P1_CARD,
            expected_recorder_id=plan["P1_evidence"]["recorder_id"], artifacts=list(P1_NAMES),
            expected_artifact_sha256=expected,
            expected_manifest_sha256=plan["P1_evidence"]["manifest_sha256"])
        state["source_artifacts_loaded"] = len(P1_NAMES)
        persist()
        minute_frame = prepare_groups(source["inputs/datasets"]["bounded_core_minutes"], "all")
        previous_source = objects["sources/identity"]
        if p2._dataframe_identity(minute_frame) != previous_source["P1_minute_frame"]:
            raise ValueError("blocked: P1 minute DatasetH identity differs from cached shadow source")
        calendar = pd.DatetimeIndex(source["inputs/daily_calendar"])
        if calendar.tz is not None or calendar.has_duplicates or not calendar.is_monotonic_increasing:
            raise ValueError("blocked: P1 native calendar differs")
        dates = shadow["decisions"].index
        if not dates.equals(calendar[(calendar >= dates[0]) & (calendar <= dates[-1])]):
            raise ValueError("blocked: cached decisions omit or add native sessions")
        start = pd.Timestamp(plan["common_training_shadow"]["start_trade_date"])
        terminal = pd.DatetimeIndex(shadow["points"].stamp).max().normalize()
        state["provider_build_attempts"] = 1
        persist()
        point_prices, provider, point_identity = p2._build_point_provider(
            root, {"minute_frame": minute_frame}, plan, terminal)
        state.update(native_derived_point_provider_builds=1, native_derived_point_provider_readbacks=1)
        persist()
        comparison_path = card / "quote_comparison.json"
        comparison = {"status": "pending", "old_identity": previous_source["derived_point_provider"],
                      "new_identity": point_identity, "native_account_attempts": 0}
        p2._atomic_json(comparison_path, comparison)
        try:
            comparison.update(_verify_quote_identity(previous_source["derived_point_provider"], point_identity))
        except BaseException as error:
            comparison.update(status="failed", exception_type=type(error).__name__, exception=str(error))
            p2._atomic_json(comparison_path, comparison)
            raise
        p2._atomic_json(comparison_path, comparison)
        state["quote_comparison_sha256"] = p2._sha_file(comparison_path)
        persist()
        if not point_prices.index.equals(pd.DatetimeIndex(shadow["points"].stamp)):
            raise ValueError("blocked: rebuilt provider grid differs from cached Python points")
        cash, split = p2._event_views(source["inputs/admitted_source_objects"], start, terminal)
        # Corporate schedule comes exclusively from pinned P1 metadata, never Python receipts.
        schedule = make_schedule(cash, split, calendar[(calendar >= start) & (calendar <= terminal)], start, terminal)
        source_identity = {"prior_completed_stage_source": previous_source,
            "P1_recorder_id": source_audit["recorder_id"],
            "P1_manifest_sha256": source_audit["verification_manifest_sha256"],
            "P1_artifacts": expected, "derived_point_provider": point_identity,
            "inherited_evidence": inherited, "completed_checkpoint": checkpoint_audit,
            "H1_projection": objects["sources/H1_projection"]}
        source_identity["phase_identity_sha256"] = p2._canonical_sha(source_identity)
        binding = json.loads((root / p2.P1_CARD / "nas_binding.json").read_text())
        binding_path = card / "nas_binding.json"
        p2._atomic_json(binding_path, {**binding, "source_id": source_identity["phase_identity_sha256"],
                                     "phase": "P2_repair_4", "as_of": str(terminal.date())})
        tracking = card / "qlib_recorder"
        recorder, native, bridge = _native_then_record(shadow=shadow, labels=labels,
            provider=provider, point_prices=point_prices, schedule=schedule, tariff=p1_plan["tariff"],
            plan=plan, calendar=calendar, tracking=tracking, source_identity=source_identity,
            state=state, persist=persist, implementation_identity=identity,
            code_snapshot=code_snapshot, source_audit=source_audit, checkpoint_audit=checkpoint_audit)
        support = p2._support_report(shadow["decisions"], labels["labels"], plan)
        result = {"experiment": EXPERIMENT, "phase": "P2_repair_4",
            "status": "completed_training_bridge_no_economic_evaluation", "decision": support["gate"],
            "plan_sha256": p2.PLAN_SHA256, "implementation_identity_sha256": identity["sha256"],
            "implementation_review_sha256": review["sha256"], "region": "us", "recorder_uri": str(tracking),
            "repair_of_failed_recorder": SOURCE_RID, "completed_checkpoint": checkpoint_audit,
            "source_identity_sha256": source_identity["phase_identity_sha256"],
            "source_snapshot_lineage": plan["P1_evidence"], "source_status": plan["source_admission"]["mode"],
            "claims_false": plan["source_admission"]["claims_false"], "support": support,
            "native_reconciliation": native["audit"], "decision_chain": bridge["summary"],
            "native_executed_before_output_recorder": True, "repair_run": BUDGET,
            "new_allocation_fits": 0, "new_training_epochs": 0, "OOF_prediction_calls": 0,
            "action_classifier_fits": 0, "native_accounts": 1,
            "no_2023_plus_forward_or_accounts": True,
            "native_dataset_authority": {"P2_action_labels": "cached native DatasetH copied without label generation",
                "source_training_states": "original FAILED P2 Recorder retained by immutable checkpoint reference",
                "derived_provider_status": "temporary exact-quote provider removed after reconciliation"},
            "cumulative_work": {"new_allocation_fits": 6, "training_epochs": 1800,
                "OOF_model_forward_calls": 15, "P1_artifact_loads": 24,
                "derived_provider_builds": 4, "derived_provider_readbacks": 4,
                "deterministic_smoothing_and_area_passes": 2, "physical_shadow_paths": 1,
                "local_action_label_passes": 1, "native_accounts": 1,
                "checkpoint_object_loads": 95,
                "prior_native_initialization_failures_before_account_start": 1},
            "area_masked_rows": int(shadow["path_area_inputs"].prefix_mask.eq(1).sum()),
            "area_unknown_rows": int(shadow["path_area_inputs"].prefix_mask.eq(0).sum()),
            "new_market_HTTP": 0, "new_market_source_queries": 0, "P3_authorized": False,
            "promotion_permission": False, "historical_PIT_qualified": False,
            "independent_validation_qualified": False, "alpha_or_economic_success": False,
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": ["shadow_python", "local_labels", "native", "sources", "implementation"]}}
        result = finalize_evidence(recorder, card=card, summary=result, transport_binding=binding_path,
                                   workspace_root=root, include_code_artifacts=True)
        conclusion = ("## 中文结论\n\n"
            f"P2 repair_4 原生对账通过；新 Recorder `{result['recorder_id']}`。复用 repair_2 的"
            "完整 Python 影子路径、面积输入与局部标签；新增拟合、模型预测、面积计算、Python 账户和标签生成均为0。"
            f"本次仅构建/读回1份相同报价 provider、运行1个 native 账户，对账最大相对误差"
            f"{native['audit']['max_relative_error']:.3g}；成熟局部标签"
            f"{int(labels['labels'].label_status.eq('mature').sum())}，2022年前合格成熟标签"
            f"{support['pre2022_eligible_mature_action_rows']}，支持计数门为 `{support['gate']}`。\n\n"
            "各年度、季度及126日窗口仅为训练参考路径描述，保存在 Recorder/native；"
            "不作为策略经济收益或独立验证。原 FAILED 状态与文件保持不变。"
            "历史 PIT、首次收据、SOXX费用后超额与晋升资格均未取得。"
            "下一步为独立结果审查；P3须另行冻结设计与执行授权。\n")
        (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
        notebook_path = card / "repair_4.ipynb"
        notebook = json.loads(notebook_path.read_text())
        for cell in notebook["cells"]:
            if cell.get("id") == "path-p2-conclusion":
                cell["source"] = conclusion.splitlines(keepends=True)
        p2._atomic_json(notebook_path, notebook)
        state.update(status="completed", summary_sha256=p2._sha_file(card / "summary.json"))
        persist()
        return result
    except BaseException as error:
        state.update(status="failed", exception_type=type(error).__name__, exception=str(error))
        persist()
        raise
    finally:
        if provider is not None:
            shutil.rmtree(provider)
