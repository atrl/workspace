from __future__ import annotations

import hashlib
import json
import os
import shlex
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _h5(series: pd.Series) -> pd.Series:
    parts = [series.astype(float).shift(-i) for i in range(5)]
    valid = np.logical_and.reduce(
        [np.isfinite(part.to_numpy(float)) & (part.to_numpy(float) > -1) for part in parts]
    )
    return np.log1p(pd.concat(parts, axis=1)).sum(axis=1).where(valid)


def run(root: Path) -> dict:
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.analysis.seed_disagreement_information import full_calendar_bootstrap
    from src.data.nas_market_data import _settings, _ssh, archive_nas_recorder

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    root = Path(root).resolve()
    card = root / "research/stage1_0945_signed_volume_cash_h5_information_20261023"
    plan = json.loads((card / "plan.json").read_text())
    source = plan["source"]
    parent = root / source["parent_card"]
    assert plan["frozen"] and not (card / "run_state.json").exists()
    feature_card = root / source["feature_card"]
    assert _sha(feature_card / "summary.json") == source["feature_summary_sha256"]
    assert _sha(parent / "summary.json") == source["parent_summary_sha256"]
    assert _sha(parent / "verification_manifest.json") == source["parent_verification_manifest_sha256"]
    assert _sha(root / "src/analysis/seed_disagreement_information.py") == source["bootstrap_module_sha256"]
    feature_summary = json.loads((feature_card / "summary.json").read_text())
    assert feature_summary["recorder_id"] == source["feature_recorder_id"]
    assert feature_summary["NAS_evidence"]["sha256"] == source["feature_NAS_sha256"]
    assert json.loads((feature_card / "nas_binding.json").read_text())["evidence"]["sha256"] == source["feature_NAS_sha256"]
    assert json.loads((feature_card / "result.json").read_text())["decision"] == "conditional_source_and_novelty_pass_no_information_or_account"
    parent_summary = json.loads((parent / "summary.json").read_text())
    assert parent_summary["recorder_id"] == source["parent_recorder_id"]
    assert parent_summary["nas_evidence"]["sha256"] == source["parent_NAS_sha256"]
    assert json.loads((parent / "nas_binding.json").read_text())["evidence"]["sha256"] == source["parent_NAS_sha256"]

    state = {"status": "preflight", "plan_sha256": _sha(card / "plan.json"),
             "source_Recorder_object_loads": 0, "Qlib_D_feature_calls": 0,
             "Qlib_D_calendar_calls": 0, "DatasetH_builds": 0,
             "model_fits": 0, "bootstrap_attempts": 0, "native_accounts": 0}

    def save_state() -> None:
        (card / "run_state.json").write_text(json.dumps(state, indent=2) + "\n")

    save_state()
    qlib.init(provider_uri=str((root / source["provider_uri"]).resolve()), region="us", kernels=1)
    calendar = D.calendar(start_time="2018-10-12 09:31", end_time="2026-10-09 16:00", freq="1min")
    state["Qlib_D_calendar_calls"] = 1
    assert len(calendar) > 700000
    with R.uri_context(str((parent / "qlib_recorder").resolve())):
        recorder = R.get_recorder(experiment_name=parent.name, recorder_id=source["parent_recorder_id"])
        features = recorder.load_object("inputs/features")
        frame = recorder.load_object("inputs/daily_frame")
        timing = recorder.load_object("inputs/timing")
    with R.uri_context(str((feature_card / "qlib_recorder").resolve())):
        recorder = R.get_recorder(experiment_name=feature_card.name, recorder_id=source["feature_recorder_id"])
        daily_source = recorder.load_object("analysis/daily_source")
    state["source_Recorder_object_loads"] = 4
    assert features.index.equals(frame.index) and frame.index.equals(timing.index)
    base = list(parent_summary["input_audit"]["original_features"])
    assert len(base) == 39 and set(base) <= set(features.columns)
    assert daily_source.signal_date.is_unique and daily_source.signal_date.is_monotonic_increasing
    source_rows = daily_source.set_index("signal_date").reindex(pd.DatetimeIndex(frame.trade_date))
    source_rows.index = frame.index
    available = source_rows.reason.eq("ok")
    in_scope = frame.trade_date.between(plan["cohorts"]["train_trade_start"], plan["cohorts"]["held_trade_end"])
    assert source_rows.loc[in_scope, "reason"].notna().all()
    assert (source_rows.loc[in_scope, "source_last_bar"] < source_rows.loc[in_scope, "decision_at"]).all()
    # The source card's assumed 09:46 clock is not the frozen upstream allocator
    # clock. Check the parent Recorder before constructing any model dataset.
    known = in_scope & available
    source_last = pd.to_datetime(source_rows.loc[known, "source_last_bar"]).dt.tz_localize("America/New_York")
    allocator_decision = pd.to_datetime(timing.loc[known, "decision_at"])
    if source_last.ge(allocator_decision).any():
        raise ValueError("invalid_for_frozen_Stage1_allocator_0929_clock")
    controls = pd.DataFrame(index=frame.index)
    controls["opening_return_0931_0945"] = source_rows.opening_return_0931_0945.astype(float)
    controls["log_unsigned_volume_14bars"] = np.log1p(source_rows.unsigned_volume_14bars.astype(float))
    controls["source_known"] = available.astype(float)
    coordinate = source_rows.signed_volume_pressure.astype(float).rename("soxx_0945_signed_volume_pressure")
    assert coordinate.loc[frame.trade_date.between(plan["cohorts"]["held_trade_start"],plan["cohorts"]["held_trade_end"])].notna().all()

    label = -_h5(frame.soxx_forward_open_return)
    mature = pd.to_datetime(timing.return_available_at.shift(-4), utc=True)
    train_cutoff = pd.Timestamp(plan["cohorts"]["train_label_available_before"]).tz_convert("UTC")
    held_terminal = pd.Timestamp(plan["cohorts"]["held_label_available_before"]).tz_convert("UTC")
    train = (frame.trade_date.between(plan["cohorts"]["train_trade_start"],plan["cohorts"]["train_trade_end"]) & (mature < train_cutoff) & label.notna())
    held = (frame.trade_date.between(plan["cohorts"]["held_trade_start"],plan["cohorts"]["held_trade_end"]) & (mature < held_terminal) & label.notna())
    assert int(train.sum()) >= plan["cohorts"]["minimum_train_mature"]
    assert int(held.sum()) >= plan["cohorts"]["minimum_held_mature"]
    static = pd.concat({
        "feature": features[base].join(controls).assign(soxx_0945_signed_volume_pressure=coordinate),
        "label": pd.DataFrame({"cash_minus_SOXX_H5": label}),
        "meta": pd.DataFrame({"train_mature": train, "held_mature": held}),
    }, axis=1)
    static.index = pd.MultiIndex.from_arrays(
        [frame.index, ["SOXL_SOXX_CASH"] * len(frame)], names=["datetime", "instrument"]
    )
    dataset = DatasetH(
        handler=DataHandlerLP.from_df(static),
        segments={"train": [plan["cohorts"]["train_signal_start"],plan["cohorts"]["train_signal_end"]],
                  "held": [plan["cohorts"]["held_signal_start"],plan["cohorts"]["held_signal_end"]]},
    )
    state["DatasetH_builds"] = 1
    training = dataset.prepare("train", col_set=["feature", "label", "meta"])
    evaluation = dataset.prepare("held", col_set=["feature", "label", "meta"])
    training = training.loc[training[("meta", "train_mature")].astype(bool)]
    evaluation = evaluation.loc[evaluation[("meta", "held_mature")].astype(bool)]
    assert len(training) >= 1000 and len(evaluation) >= 900
    baseline_names = base + ["opening_return_0931_0945", "log_unsigned_volume_14bars", "source_known"]
    names = baseline_names + ["soxx_0945_signed_volume_pressure"]
    xtr, xhe = training["feature"], evaluation["feature"]
    ytr = training[("label", "cash_minus_SOXX_H5")].to_numpy(float)
    yhe = evaluation[("label", "cash_minus_SOXX_H5")].to_numpy(float)
    assert np.isfinite(ytr).all() and np.isfinite(yhe).all()
    preflight = {"status": "matched_Qlib_source_and_label_maturity_passed",
                 "train_mature": len(training), "held_mature": len(evaluation),
                 "candidate_train_known": int(xtr.soxx_0945_signed_volume_pressure.notna().sum()),
                 "candidate_held_known": int(xhe.soxx_0945_signed_volume_pressure.notna().sum()),
                 "source_Recorder_object_loads": 4, "Qlib_D_feature_calls": 0,
                 "Qlib_D_calendar_calls": 1, "DatasetH_builds": 1,
                 "parent_recorder_id": source["parent_recorder_id"]}
    (card / "preflight_result.json").write_text(json.dumps(preflight, indent=2) + "\n")

    state["status"] = "fit_attempted"
    save_state()
    baseline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    candidate = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    baseline.fit(xtr[baseline_names], ytr)
    state["model_fits"] = 1
    save_state()
    candidate.fit(xtr[names], ytr)
    state["model_fits"] = 2
    save_state()
    p0, p1 = baseline.predict(xhe[baseline_names]), candidate.predict(xhe[names])
    delta = (yhe - p0) ** 2 - (yhe - p1) ** 2
    score = pd.DataFrame({"signal_date": pd.DatetimeIndex(evaluation.index.get_level_values("datetime")),
                          "target": yhe, "baseline": p0, "candidate": p1,
                          "paired_MSE_improvement": delta})
    score["trade_date"] = frame.trade_date.reindex(score.signal_date).to_numpy()
    assert len(score) == 940 and score.signal_date.is_monotonic_increasing
    half = len(score) // 2
    metrics = {"coefficient_standardized": float(candidate.named_steps["ridge"].coef_[-1]),
               "all_held_MSE_baseline": float(np.mean((yhe - p0) ** 2)),
               "all_held_MSE_candidate": float(np.mean((yhe - p1) ** 2)),
               "all_held_MSE_train_mean": float(np.mean((yhe - np.mean(ytr)) ** 2)),
               "all_held_improvement_vs_baseline": float(np.mean(delta)),
               "all_held_improvement_vs_train_mean": float(np.mean((yhe - np.mean(ytr)) ** 2 - (yhe - p1) ** 2)),
               "first_half_improvement": float(np.mean(delta[:half])),
               "second_half_improvement": float(np.mean(delta[half:])),
               "2024_improvement": float(score.loc[score.trade_date.dt.year == 2024, "paired_MSE_improvement"].mean()),
               "2025_improvement": float(score.loc[score.trade_date.dt.year == 2025, "paired_MSE_improvement"].mean())}
    full = pd.Series(np.nan, index=frame.loc["2023-01-03":"2026-10-07"].index, dtype=float)
    full.loc[score.signal_date.to_numpy()] = delta
    assert len(full) == 944 and int(full.notna().sum()) == 940
    bootstrap, _, means = full_calendar_bootstrap(
        full.to_numpy(float), mean_block=63, resamples=2000, seed=20261023, alpha=0.025/30)
    state["bootstrap_attempts"] = 1
    save_state()
    gates = {"negative_coefficient": metrics["coefficient_standardized"] < 0,
             "beats_baseline": metrics["all_held_improvement_vs_baseline"] > 0,
             "beats_train_mean": metrics["all_held_improvement_vs_train_mean"] > 0,
             "both_halves": metrics["first_half_improvement"] > 0 and metrics["second_half_improvement"] > 0,
             "both_complete_years": metrics["2024_improvement"] > 0 and metrics["2025_improvement"] > 0,
             "positive_block_lower": bootstrap["lower_bound"] > 0}
    decision = ("information_gate_passed_requires_full_0bp_account" if all(gates.values())
                else "not_promoted_fixed_0945_signed_volume_cash_H5_information_failed")

    (card / "qlib_recorder").mkdir(exist_ok=True)
    R.set_uri(str((card / "qlib_recorder").resolve()))
    with R.start(experiment_name=card.name):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, preflight=preflight, dataset=dataset)
        recorder.save_objects(artifact_path="models", baseline=baseline, candidate=candidate)
        recorder.save_objects(artifact_path="predictions", held_scored=score)
        recorder.save_objects(artifact_path="analysis", full_held_calendar=full,
                              metrics=metrics, gates=gates, bootstrap=bootstrap,
                              bootstrap_replicate_means=means)
        R.log_params(feature="soxx_0945_signed_volume_pressure", parent_recorder_id=source["parent_recorder_id"])
        R.log_metrics(held_MSE_improvement=metrics["all_held_improvement_vs_baseline"],
                      block_lower=bootstrap["lower_bound"])
        rid, eid = recorder.id, str(recorder.experiment_id)
    summary = {"experiment": card.name, "decision": decision,
               "promotion": "not_promoted_information_only", "objective_achieved": False,
               "plan_sha256": _sha(card / "plan.json"), "source_provider_uri": source["provider_uri"],
               "feature_source_recorder_id": source["feature_recorder_id"],
               "parent_recorder_id": source["parent_recorder_id"], "recorder_id": rid,
               "experiment_id": eid, "metrics": metrics, "bootstrap": bootstrap,
               "gates": gates, "model_fits": 2, "native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first",
                                   "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "models/*", "predictions/*", "analysis/*"]},
               "limitations": ["2023-2026 reuse is development feedback, not independent forward evidence.",
                               "H5 labels omit funded Tiered fees and account feedback.",
                               "Historical 09:45 bar finality and first receipt remain unqualified; no SOXL relative receiver is tested here."]}
    state.update(status="finished", recorder_id=rid, experiment_id=eid)
    save_state()
    binding = json.loads((parent / "nas_binding.json").read_text())
    binding.pop("evidence", None)
    (card / "nas_binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    evidence = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
                                    card=card, recorder_id=rid, experiment_id=eid)
    remote = _ssh(_settings(root)["ssh"], "sha256sum " + shlex.quote(evidence["nas_path"])).split()[0]
    assert remote == evidence["sha256"]
    summary["NAS_evidence"] = evidence
    summary["NAS_readback"] = {"verified": True, "sha256": remote}
    (card / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    conclusion = (
        "# Stage1 09:45 SOXX 方向加权成交量现金 H5 信息检验\n\n"
        "完整冻结 Stage1 A0 为母本，SOXX 为最终硬门。本卡检验现有 NAS/Qlib 09:31–09:45 SOXX 方向加权成交量对 H5 现金优势的增量信息，"
        "对照同时含原39项、同前缀开盘收益、无符号成交量及已知标记；成熟标签和 Ridge 配对不变。\n\n"
        f"训练成熟 {len(training)} 行，复用历史评价 {len(evaluation)} 行；2 拟合、1 区块 bootstrap、0 账户。"
        f"新增标准化系数 {metrics['coefficient_standardized']:.8f}。候选 MSE {metrics['all_held_MSE_candidate']:.9f}，"
        f"原39对照 {metrics['all_held_MSE_baseline']:.9f}，训练均值 {metrics['all_held_MSE_train_mean']:.9f}。"
        f"对照改善 {metrics['all_held_improvement_vs_baseline']:.9f}，区块下界 {bootstrap['lower_bound']:.9f}。"
        f"前/后半改善 {metrics['first_half_improvement']:.9f}/{metrics['second_half_improvement']:.9f}，"
        f"2024/2025 改善 {metrics['2024_improvement']:.9f}/{metrics['2025_improvement']:.9f}。"
        f"六项固定门 {sum(gates.values())}/6；决定 `{decision}`。\n\n"
        "这不是费用后 SOXX 超额；无完整 0bp 账户或独立前向。失败版不翻转符号或调整前缀/窗口。"
        f"Recorder `{rid}`，NAS SHA `{remote}` 经验证。\n")
    (card / "conclusion.md").write_text(conclusion)
    return summary
