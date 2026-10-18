"""One frozen auxiliary-domain intervention with native Qlib raw accounts."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json
from src.model.minute_pit_allocator import MinutePitAllocator, TARGETS, _flat_index
from src.model.stage1_domain_heads import DomainHeadAllocator, initialization_contract
from src.workflow.minute_pit_allocator import _primary_dataset, _auxiliary_dataset, _prediction_dates, _ensemble
from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights
from src.workflow.stage1_runtime_release import load_runtime_release

NAME = "stage1_domain_heads_20261018"
DATES = ["signal_date", "trade_date", "exit_date"]
OUTPUTS = ["soxl_weight", "soxx_weight", "cash_weight", "tail_probability", "alpha_probability"]
REFERENCES = {"SOXX": "raw_SOXX", "strong9097": "raw_strong9097", "additional27baa": "raw_candidate27baa"}
CODE_FILES = [
    "src/workflow/stage1_domain_heads.py", "src/model/stage1_domain_heads.py",
    "src/model/minute_pit_allocator.py", "src/backtest/stage1_domain_heads_raw.py",
    "src/workflow/minute_pit_allocator.py", "src/analysis/integrated_allocation.py",
    "src/backtest/raw_corporate_policy.py", "src/backtest/native_raw_corporate.py",
    "src/backtest/ibkr_tiered.py", "src/workflow/stage1_open_action_value.py",
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


def _new_model(original, cls, root):
    return cls(base_features=original.base_features, extra_features=[], feature_masks=original.feature_masks,
               spec={**asdict(original.spec), "root": str(root)}, seed=original.seed,
               cutoff=str(original.cutoff), train_end=str(original.train_end.date()),
               minimum_observations=original.minimum_observations)


def prepare(*, root, card):
    """Read and count pinned Recorder inputs. No training, new labels or accounts."""
    import qlib
    root, card = Path(root), Path(card)
    plan = json.loads((card / "plan.json").read_text())
    digest = sha(card / "plan.json")
    release = root / plan["active_release"]["path"]
    if sha(release) != plan["active_release"]["sha256"]:
        raise ValueError("active release changed")
    loaded = load_runtime_release(root, release)
    qlib.init(provider_uri=str(root / plan["provider_init"]), region="us", kernels=1)
    sources, audits = {}, {}
    for role, source in plan["sources"].items():
        sources[role], audits[role] = _read_source(root, source)
    train, action, raw = sources["training"], sources["action"], sources["raw"]
    original = train["models/base_outer_71"]
    if original.extra_features or len(original.base_features) != 39 or original.spec.epochs != 300:
        raise ValueError("training source is not original nonlinear39-input base")
    inputs = {name: train["inputs/" + key] for name, key in [
        ("frame", "daily_frame"), ("features", "features"), ("common", "common"), ("timing", "timing")]}
    inputs["spec"] = original.spec
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
        left, right = _new_model(old, MinutePitAllocator, root), _new_model(old, DomainHeadAllocator, root)
        initialization.append({"seed": seed, **initialization_contract(left, right)})
    receipt = {"plan_sha256": digest, "primary_rows": int(eligible.sum()), "auxiliary_rows": len(aux),
               "primary_last_mature": str(maturity.loc[eligible].max()),
               "aux_first_signal": str(aux.index.get_level_values("datetime").min()),
               "aux_last_signal": str(aux.index.get_level_values("datetime").max()),
               "aux_observations_end_max": str(pd.to_datetime(aux[("meta", "observations_end")]).max()),
               "source_hashes_checked": True, "source_api": ["Recorder.load_object", "DatasetH.prepare"],
               "main_inputs": 39, "common_inputs": 16, "account_sessions": len(frame),
               "early_fields_exact_source_equality": early, "reused_reference_accounts": len(reference_books),
               "active_bundle_and_classifier_hashes_verified": True, "initialization": initialization,
               "new_model_fits": 0, "new_accounts": 0, "dirty_runtime_providers_used_as_input": False}
    atomic_json(receipt, card / "preflight.json")
    return dict(plan=plan, plan_sha256=digest, sources=sources, source_audits=audits, dataset=dataset,
                auxiliary=auxiliary, frame=frame, market=market, points=points,
                schedule=raw["inputs/schedule"], guard=raw["inputs/repair_1_calendar_guard"],
                references=reference_books, controller=loaded["models"], selection=selected,
                recovery_spec=action["inputs/plan"]["recovery_spec"], action_spec=action_spec,
                receipt=receipt, original=original, daily_calendar_frame=inputs["frame"])


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
            rec.save_objects(artifact_path="inputs", plan=plan, source_audits=p["source_audits"],
                preflight=p["receipt"], approval=approval, dataset=p["dataset"], auxiliary_dataset=p["auxiliary"],
                frame=p["frame"], market=p["market"], raw_points=p["points"], schedule=p["schedule"],
                guard=p["guard"], controller=p["controller"], selection=p["selection"],
                recovery_spec=p["recovery_spec"], action_spec=p["action_spec"])
            rec.save_objects(artifact_path="sources", **{name.replace("/", "__"): (root / name).read_text() for name in CODE_FILES})
            state["sources_saved"] = True
            atomic_json(state, state_path)
        for family, cls in [("B_shared", MinutePitAllocator), ("C_domain", DomainHeadAllocator)]:
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
                    model = _new_model(original, cls, root)
                    model.fit(p["dataset"], auxiliary_dataset=p["auxiliary"])
                    for name in ["training_rows", "auxiliary_rows", "last_mature_label_at", "epochs", "checkpoint_policy"]:
                        if model.audit[name] != original.audit[name]:
                            raise ValueError("paired training contract differs: " + name)
                    if model.primary_scaler != original.primary_scaler or model.common_scaler != original.common_scaler:
                        raise ValueError("matched source scaler differs")
                    rec.save_objects(artifact_path="models", **{key: model})
                    state["fits_done"].append(key)
                    atomic_json(state, state_path)
                prediction = _prediction_dates(model.predict(p["dataset"]))
                predictions[(family, seed)] = prediction
                if family == "B_shared":
                    saved = p["sources"]["training"]["analysis/outer_predictions"][("base", seed)]
                    pd.testing.assert_index_equal(prediction.index, saved.index)
                    np.testing.assert_allclose(prediction, saved, rtol=1e-6, atol=1e-7)
                rec.save_objects(artifact_path="predictions", **{key: prediction})
                training.append({"family": family, "seed": seed, **model.audit})
                rec.log_metrics(completed_fits=len(state["fits_done"]))
                print("completed fit", key, flush=True)
        ensembles = {family: _ensemble(predictions, family, plan["training"]["seeds"]) for family in ["B_shared", "C_domain"]}
        dates = pd.DatetimeIndex(p["market"].signal_date)
        returns = p["market"].set_index("signal_date")[["soxl_return", "soxx_return"]]
        targets = {family: _smooth_weights(values.reindex(dates), spec=p["original"].spec,
                     candidate=p["original"].candidate, asset_returns=returns) for family, values in ensembles.items()}
        np.testing.assert_allclose(targets["B_shared"], p["sources"]["action"]["inputs/source_targets"]["base"], rtol=1e-6, atol=1e-7)
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
    from src.analysis.stage1_domain_heads import analyze_accounts
    from src.data.nas_market_data import archive_nas_recorder
    # No information-score gate here: all fixed economic paths must be executed.
    provider, manifest = _provider(root, p)
    rec.save_objects(artifact_path="inputs", native_provider_manifest=manifest)
    recovery = IntradayRecoveryDirectAllocationSpec(root=root, **p["recovery_spec"])
    action = Stage1OpenActionValueSpec(root=root, **p["action_spec"])
    execution = {"initial_cash": plan["accounts"]["initial_cash"], "tariff": plan["tariff"]}
    books = dict(p["references"])
    for family in ["A_active_research_replay", "B_shared", "C_domain"]:
        frame = p["frame"].copy()
        if family in targets:
            for name in OUTPUTS:
                frame[name] = targets[family][name].to_numpy(copy=True)
        for cost in plan["accounts"]["slippage_bps"]:
            key = f"{family}_{cost}"
            if key in state["paths_done"]:
                path = rec.load_object("paths/" + key)
            else:
                if key in state["path_attempts"] or len(state["path_attempts"]) >= 9:
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
                if key in state["native_attempts"] or len(state["native_attempts"]) >= 9:
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
    summary = {"experiment": NAME, "plan_sha256": digest, "decision": "historical_economic_pass_needs_risk_and_independent_forward" if economic else "not_promoted_fixed_stage1_domain_heads",
        "objective_achieved": False, "gates": results["gates"], "coarse_economic_gates_pass": economic,
        "new_fits": len(state["fits_done"]), "new_native_accounts": len(state["native_done"]),
        "reused_native_accounts": 9, "total_comparison_accounts": len(books), "sessions_each": len(p["frame"]),
        "recorder_id": rec.id, "experiment_id": str(rec.experiment_id), "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / str(rec.experiment_id) / rec.id), "provider_uri": str(provider),
        "region": "us", "as_of": "2026-10-09", "history_status": "all_history_reused_not_independent",
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
    lines = ["# Stage1分监督域输出头：固定配对实验", "",
        "完成B共享头与C分域头各3个seed、共6次固定300轮拟合。主3014条、辅助20741条；39主输入与16共享输入、标签、样本权重及原Stage1路由/风险/恢复均固定。新增18参数，仍允许辅助梯度更新共享编码器。",
        "", "A是现役f696+b472在同源原始股数、09:45已知/09:47成交/10:32恢复研究合同下重放，不是运行中09:31/10:31实盘复现。新增9个A/B/C独立费用原生账户，严格同口径复用SOXX/9097/27baa九账户；每账户944区间。",
        "", "|方案|滑点bp|累计净收益%|收盘最大回撤%|CVaR5%|", "|---|---:|---:|---:|---:|"]
    for row in metric.itertuples():
        lines.append(f"|{row.family}|{row.slippage_bps}|{row.return_pct:.3f}|{row.close_max_drawdown_pct:.3f}|{row.cvar5_pct:.3f}|")
    failed = [name for name, passed in results["gates"].items() if not passed]
    lines += ["", "决定：**" + summary["decision"] + "**。", "失败或未取得的门：" + ", ".join(failed) + "。",
        "", "全部年度、季度、126交易日窗口与三seed分类结果均保留；不择窗、择seed或按分类分数取消账户。首尾不完整年度单列，复用历史与重叠窗口不构成独立验证。参数分离和全局梯度裁剪/优化器变化共同构成本次干预，不能把差异解释成已证明的唯一负迁移原因。",
        "", "净收益来自原始股数/现金/分红应收/费用独立账本并经Qlib核对。采样回撤不认证全分钟风险；公开历史来源与纸面公司行动约定不等于历史PIT或真实券商结算。现役生产未改。",
        "", ("下一步仅补齐原定风险与独立前向资格，不宣称稳定超额目标已达到。" if economic else "关闭本固定分域头版本，不调整head数量、损失系数、epoch、seed、窗口或控制器救援；整体稳定超额目标仍未实现。"),
        "", "Qlib Recorder：`" + rec.id + "`；来源与NAS身份见summary.json、verification_manifest.json、nas_binding.json。"]
    (card / "conclusion.md").write_text("\n".join(lines) + "\n")
    (card / "workflow.yaml").write_text(
        "experiment: " + NAME + "\nentrypoint: src.workflow.stage1_domain_heads.run\nphase: completed\nprovider_uri: "
        + str(provider) + "\nrecorder_uri: " + str(tracking) + "\nartifact_policy: slim_card_recorder_first\n")
    state["completed"] = True
    atomic_json(state, state_path)
    return summary
