"""Seal an independent paper observation release from retained Qlib evidence."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

PRODUCTION = "stage1_regime_paper_20261001"
RESEARCH = "stage1_regime_deployment_20261001"
CANONICAL_ROOT = Path("/Users/rziiiii/dev/workspace")
CODE_FILES = (
    "src/workflow/stage1_regime_action.py", "src/workflow/stage1_regime_paper.py",
    "src/schedule/stage1_regime_paper.py", "src/schedule/us_open_quotes.py",
    "src/workflow/stage1_economic_action.py", "src/workflow/stage1_economic_training.py",
    "src/workflow/stage1_open_action_value.py", "src/workflow/intraday_recovery_direct_allocation.py",
    "src/workflow/stage1_futu_sim.py", "src/backtest/position_accounting.py",
    "src/analysis/stage1_regime_validation.py", "src/workflow/stage1_regime_release.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n")


def seal_paper_release(*, root, research=RESEARCH):
    import qlib
    from qlib.workflow import R
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
    from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts

    root = Path(root).resolve()
    source_card = root / "research" / research
    source = json.loads((source_card / "summary.json").read_text())
    if source["promotion"].get("native_reconciliation_pass") is not True:
        raise ValueError("blocked: native Qlib reconciliation is not verified")
    native = source.get("native_qlib_audit", {})
    error, tolerance = float(native.get("maximum_session_return_error", float("nan"))), float(native.get("tolerance", float("nan")))
    if native.get("status") != "passed" or not math.isfinite(error) or not math.isfinite(tolerance) or not 0 <= error <= tolerance <= 1e-7:
        raise ValueError("blocked: inconsistent native Qlib reconciliation evidence")
    if source["baseline_reproduction_pass"] is not True:
        raise ValueError("blocked: original baseline identity did not reproduce")
    for path, expected in source["source_sha256"].items():
        if sha256(root / path) != expected:
            raise ValueError(f"blocked: research implementation changed: {path}")
    qlib.init(provider_uri=str(root / "data/qlib/soxl_soxx_soxs"), region="us", kernels=1)
    with R.uri_context(source["recorder_uri"]):
        source_recorder = R.get_recorder(experiment_name=source["experiment"], recorder_id=source["recorder_id"])
        artifact_root = Path(source_recorder.get_local_dir()).resolve() / "artifacts"
        expected_root = Path(source["recorder_local_dir"]).resolve() / "artifacts"
        if artifact_root != expected_root:
            raise ValueError("blocked: research recorder artifact location differs")
        needed = ("models/selected_action.pkl", "models/selected_spec.pkl", "inputs/prepared_inputs.pkl", "training/expanded_model_training.pkl")
        for name in needed:
            expected = source.get("sealed_source_objects", {}).get(name)
            if not expected or sha256(artifact_root / name) != expected:
                raise ValueError(f"blocked: unsealed or changed research object: {name}")
        action_model = source_recorder.load_object("models/selected_action.pkl")
        action_spec = source_recorder.load_object("models/selected_spec.pkl")
        inputs = source_recorder.load_object("inputs/prepared_inputs.pkl")
        training = source_recorder.load_object("training/expanded_model_training.pkl")
    if action_model is None or asdict(action_spec) != source["selected_spec"]:
        raise ValueError("blocked: retained action model/spec mismatch")
    if getattr(action_model, "regime_action_spec", None) != asdict(action_spec):
        raise ValueError("blocked: action estimator specification differs from release")
    baseline_model = inputs["legacy_models"][2026]
    recovery_model = inputs["recovery_models"][2026]
    if baseline_model is None or recovery_model is None:
        raise ValueError("blocked: paired paper references unavailable")
    final_fit = training["diagnostics"].loc[training["diagnostics"].model_year.eq(training["final_model_year"])].iloc[-1]
    model_audit = getattr(action_model, "regime_training_audit", {})
    for key in ("model_year", "training_rows", "training_signal_end", "label_available_at_max", "information_cutoff_exclusive", "training_sha256", "feature_schema_sha256"):
        if model_audit.get(key) != final_fit[key]:
            raise ValueError("blocked: retained estimator training identity differs from diagnostics")
    label_time = datetime.fromisoformat(final_fit.label_available_at_max)
    cutoff = datetime.fromisoformat(final_fit.information_cutoff_exclusive)
    if label_time.tzinfo is None or cutoff.tzinfo is None or not label_time < cutoff <= datetime(2026, 1, 1, 5, tzinfo=timezone.utc):
        raise ValueError("blocked: paper model consumed information beyond the sealed training cutoff")
    card = root / "production" / PRODUCTION
    card.mkdir(parents=True, exist_ok=True)
    existing = card / "release_manifest.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        if previous.get("source_research_summary_sha256") != sha256(source_card / "summary.json") or previous.get("code_sha256") != {path: sha256(root / path) for path in CODE_FILES}:
            raise ValueError("blocked: existing paper release is immutable; use a new release identity")
        summary = json.loads((card / "summary.json").read_text())
        if summary.get("release_manifest_sha256") != sha256(existing):
            raise ValueError("blocked: retained paper manifest changed")
        return summary
    canonical_card = CANONICAL_ROOT / "production" / PRODUCTION
    uri = canonical_card / "qlib_recorder"
    uri.mkdir(parents=True, exist_ok=True)
    recovery_spec = asdict(IntradayRecoveryDirectAllocationSpec(root=root))
    baseline_spec = asdict(Stage1OpenActionValueSpec(root=root))
    recovery_spec.pop("root"); baseline_spec.pop("root")
    forward_plan = {
        "registered_before_first_forward_observation": True,
        "status": "pending", "minimum_complete_paired_sessions": 60,
        "minimum_sessions_per_regime": 15, "minimum_regime_transitions": 3,
        "paired": "candidate and incumbent controller, each from same initialcash and captured quotes at both10/25bp; separate own holdings and risk clock",
        "quality": "No unresolved slot gaps, data-age, model/hash, accounting or timing failures; incomplete sessions excluded and explicitly reported, never retroactively traded.",
        "performance": "At both10/25bp net terminal wealth at least paired baseline; maximum drawdown,CVaR5,worst daily loss no worse; turnover no higher. Historical failure gates remain binding for active replacement.",
        "future_data": "Only events at or after release not_before; model/threshold changes start a new release and evidence window.",
        "scope": "Minimum observation criteria, not statistical proof or automatic broker authorization. Fractional price-return books do not verify real fills, dividends or corporate actions.",
    }
    R.set_uri(str(uri))
    with R.start(experiment_name=PRODUCTION):
        recorder = R.get_recorder()
        R.log_params(research_recorder_id=source["recorder_id"], execution_mode="INDEPENDENT_PAPER", research_promotion_status="not_promoted", candidate=source["selected"]["variant"])
        R.save_objects(artifact_path="models", **{"action_model.pkl": action_model, "recovery_model.pkl": recovery_model, "baseline_action_model.pkl": baseline_model})
        R.save_objects(artifact_path="evidence", **{"research_summary.pkl": source, "forward_plan.pkl": forward_plan, "training_diagnostics.pkl": training["diagnostics"]})
        recdir = Path(recorder.get_local_dir())
    _remove_qlib_code_snapshot_artifacts(recdir)
    manifest = {
        "schema_version": 1, "release_id": PRODUCTION,
        "execution_mode": "INDEPENDENT_PAPER", "research_promotion_status": "not_promoted",
        "not_before": datetime.now(timezone.utc).isoformat(), "initial_cash": 100000., "cost_bps": [10., 25.],
        "recorder_uri": str(uri.relative_to(CANONICAL_ROOT)), "experiment_name": PRODUCTION, "recorder_id": recorder.id,
        "daily_provider_uri": "data/qlib/soxl", "regime_provider_uri": "data/qlib/soxl_soxx_soxs", "intraday_provider_uri": "data/qlib/soxl_intraday_15min",
        "allocation_source": source["allocation_source"], "regime_action_spec": asdict(action_spec),
        "recovery_spec": recovery_spec, "baseline_action_spec": baseline_spec,
        "selected_recovery_model": inputs["selected_model"],
        "label_available_at_max": final_fit.label_available_at_max,
        "training_information_cutoff": final_fit.information_cutoff_exclusive,
        "source_research_recorder_id": source["recorder_id"], "source_research_summary_sha256": sha256(source_card / "summary.json"),
        "historical_gates_pass": bool(source["selected"]["historical_all_pass"]),
        "forward_plan": forward_plan,
        "objects": {name: {"path": f"models/{name}.pkl", "sha256": sha256(recdir / f"artifacts/models/{name}.pkl")} for name in ("action_model", "recovery_model", "baseline_action_model")},
        "code_sha256": {path: sha256(root / path) for path in CODE_FILES},
    }
    write_json(card / "release_manifest.json", manifest)
    write_json(card / "forward_plan.json", forward_plan)
    summary = {
        "experiment": PRODUCTION, "decision": "independent_paper_observation_only_not_promoted",
        "research_recorder_id": source["recorder_id"], "candidate": source["selected"]["variant"],
        "historical_gates_pass": bool(source["selected"]["historical_all_pass"]),
        "recorder_id": recorder.id, "recorder_uri": str(uri), "recorder_local_dir": str(recdir),
        "recorder_relative_dir": str(recdir.relative_to(canonical_card)),
        "provider_uri": "data/qlib/soxl", "region": "us", "market": "all", "source_as_of": source["source_as_of"],
        "release_manifest": "release_manifest.json", "release_manifest_sha256": sha256(card / "release_manifest.json"),
        "prepared": True, "activated": False, "new_forward_sessions": 0,
        "orders_submitted": 0, "active_broker_release_changed": False,
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml", "recorder_only_artifacts": ["models/*.pkl", "evidence/*.pkl", "forward/*"]},
        "qlib_docs_reference": source["qlib_docs_reference"],
    }
    write_json(card / "summary.json", summary)
    import yaml
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": PRODUCTION, "provider_uri": summary["provider_uri"], "region": "us", "recorder_uri": str(uri), "research": f"research/{research}", "model_selection": source["selected"], "release_manifest": "release_manifest.json", "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    conclusion = f"""# 独立连续 paper 部署结论

研究候选为 {source['selected']['variant']}，来源 QLib recorder {source['recorder_id']}。历史门全部通过={source['selected']['historical_all_pass']}，保持未晋升。

准备四本独立连续账户：候选和现有动作基准各用10/25bp费用，从相同100000现金起点、相同已观测报价计算各自持仓、净值、回撤时钟和恢复动作。分数股及费用均为模拟假设，不声称真实模拟成交。

本卡仅用于独立观察；不替换现有SIM、不接入真实下单。全新前向证据仍为0。至少60个完整配对交易日、每类区间至少15日及3次区间转换，并通过净费收益、尾部风险和换手比较后，才具备进一步评审的最低证据；历史失败门不会被前向观察自动豁免。

行情缺失、时序错误、模型身份改变、未完成的时段或账本失败均停止推进，不事后补造交易。现金分配/拆股等需另行核实处理。

下一步：激活已封存观察任务，核验今晚实际开盘及首小时事件；持续记录相同报价下的区间差异。准备完成不代表已积累前向表现。

QLib recorder：{recorder.id}。
"""
    (card / "conclusion.md").write_text(conclusion)
    return summary
