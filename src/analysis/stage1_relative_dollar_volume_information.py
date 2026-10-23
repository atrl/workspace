"""One frozen conditional H5 SOXL-relative information screen."""

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
    valid = np.logical_and.reduce([
        np.isfinite(part.to_numpy(float)) & (part.to_numpy(float) > -1)
        for part in parts
    ])
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
    from src.analysis.stage1_relative_dollar_volume_preflight import _shock
    from src.data.nas_market_data import _settings, _ssh, archive_nas_recorder

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    root = Path(root).resolve()
    card = root / "research/stage1_relative_dollar_volume_h5_information_20261023"
    if (card / "run_state.json").exists():
        raise ValueError("fixed H5 information attempt already started")
    plan = json.loads((card / "plan.json").read_text())
    source = plan["source"]
    assert plan["frozen"]
    source_card = root / source["coordinate_card"]
    parent_card = root / source["parent_card"]
    for relative, expected in [
        (source["coordinate_card"] + "/summary.json", source["coordinate_summary_sha256"]),
        (source["coordinate_card"] + "/result.json", source["coordinate_result_sha256"]),
        (source["coordinate_provider_uri"] + "/qlib_store_manifest.json", source["coordinate_provider_manifest_sha256"]),
        ("src/analysis/stage1_relative_dollar_volume_preflight.py", source["coordinate_module_sha256"]),
        (source["parent_card"] + "/summary.json", source["parent_summary_sha256"]),
        (source["parent_card"] + "/verification_manifest.json", source["parent_verification_manifest_sha256"]),
        (source["parent_provider_uri"] + "/qlib_store_manifest.json", source["parent_provider_manifest_sha256"]),
        ("src/analysis/seed_disagreement_information.py", source["bootstrap_module_sha256"]),
    ]:
        assert _sha(root / relative) == expected, relative
    source_summary = json.loads((source_card / "summary.json").read_text())
    parent_summary = json.loads((parent_card / "summary.json").read_text())
    assert source_summary["recorder_id"] == source["coordinate_recorder_id"]
    assert source_summary["NAS_evidence"]["sha256"] == source["coordinate_NAS_sha256"]
    assert source_summary["decision"] == "conditional_source_novelty_pass_first_public_unknown"
    assert json.loads((source_card / "nas_scoped_restore.json").read_text())["status"] == "NAS_scoped_restore_and_native_Qlib_readback_passed"
    assert parent_summary["recorder_id"] == source["parent_recorder_id"]
    assert parent_summary["nas_evidence"]["sha256"] == source["parent_NAS_sha256"]

    state = {"status": "preflight", "plan_sha256": _sha(card / "plan.json"),
             "source_Recorder_object_loads": 0, "DatasetH_builds": 0,
             "model_fits": 0, "bootstrap_attempts": 0, "native_accounts": 0}

    def save_state() -> None:
        (card / "run_state.json").write_text(json.dumps(state, indent=2) + "\n")

    save_state()
    qlib.init(provider_uri=str((root / source["coordinate_provider_uri"]).resolve()),
              region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(start_time="2011-01-03", end_time="2026-10-09", freq="day"))
    native = D.features(["SOXL", "SOXX"], ["$dollar_volume_proxy"],
        start_time="2011-01-03", end_time="2026-10-09", freq="day")
    pair = native["$dollar_volume_proxy"].unstack("instrument").reindex(calendar)
    assert len(calendar) == 3966 and set(pair.columns) == {"SOXL", "SOXX"}
    with R.uri_context(str((source_card / "qlib_recorder").resolve())):
        source_recorder = R.get_recorder(experiment_id=source["coordinate_experiment_id"],
                                         recorder_id=source["coordinate_recorder_id"])
        relative = source_recorder.load_object("analysis/relative_shock")
        soxx_shock = source_recorder.load_object("analysis/SOXX_shock")
    state["source_Recorder_object_loads"] = 2
    pd.testing.assert_index_equal(relative.index, calendar, check_names=False)
    np.testing.assert_allclose(relative.to_numpy(float),
        (_shock(pair.SOXL) - _shock(pair.SOXX)).to_numpy(float),
        rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(soxx_shock.to_numpy(float), _shock(pair.SOXX).to_numpy(float),
                               rtol=0, atol=1e-9, equal_nan=True)
    with R.uri_context(str((parent_card / "qlib_recorder").resolve())):
        parent = R.get_recorder(experiment_id=source["parent_experiment_id"],
                                recorder_id=source["parent_recorder_id"])
        features = parent.load_object("inputs/features")
        frame = parent.load_object("inputs/daily_frame")
        timing = parent.load_object("inputs/timing")
    state["source_Recorder_object_loads"] = 5
    assert features.index.equals(frame.index) and frame.index.equals(timing.index)
    assert frame.index.equals(calendar)
    base = list(parent_summary["input_audit"]["original_features"])
    assert len(base) == 39 and set(base) <= set(features.columns)
    label = _h5(frame.soxl_forward_open_return) - _h5(frame.soxx_forward_open_return)
    maturity = pd.to_datetime(timing.return_available_at.shift(-4), utc=True)
    receiver = plan["receiver"]
    train_cutoff = pd.Timestamp(receiver["train_label_available_before"]).tz_convert("UTC")
    held_cutoff = pd.Timestamp(receiver["held_label_available_before"]).tz_convert("UTC")
    train_mature = ((frame.index >= pd.Timestamp(receiver["train_signals"][0])) &
                    (frame.index <= pd.Timestamp(receiver["train_signals"][1])) &
                    (maturity < train_cutoff) & label.notna())
    held_mature = ((frame.index >= pd.Timestamp(receiver["reused_held_signals"][0])) &
                   (frame.index <= pd.Timestamp(receiver["reused_held_signals"][1])) &
                   (maturity < held_cutoff) & label.notna())
    assert int(train_mature.sum()) == 3014 and int(held_mature.sum()) == 940
    assert int(relative.loc[train_mature].notna().sum()) >= plan["gates"]["source_train_known_min"]
    assert int(relative.loc[held_mature].notna().sum()) >= plan["gates"]["source_held_mature_known_min"]
    static = pd.concat({
        "feature": features[base].assign(SOXX_dollar_volume_shock20=soxx_shock,
                                          SOXL_minus_SOXX_dollar_volume_shock20=relative),
        "label": pd.DataFrame({"SOXL_relative_H5": label}),
        "meta": pd.DataFrame({"train_mature": train_mature,
                              "held_mature": held_mature,
                              "source_known": relative.notna() & soxx_shock.notna()}),
    }, axis=1)
    static.index = pd.MultiIndex.from_arrays(
        [frame.index, ["SOXL_SOXX_CASH"] * len(frame)], names=["datetime", "instrument"])
    dataset = DatasetH(handler=DataHandlerLP.from_df(static),
        segments={"train": receiver["train_signals"], "held": receiver["reused_held_signals"]})
    state["DatasetH_builds"] = 1
    training = dataset.prepare("train", col_set=["feature", "label", "meta"])
    evaluation = dataset.prepare("held", col_set=["feature", "label", "meta"])
    training = training.loc[training[("meta", "train_mature")].astype(bool) & training[("meta", "source_known")].astype(bool)]
    evaluation = evaluation.loc[evaluation[("meta", "held_mature")].astype(bool) & evaluation[("meta", "source_known")].astype(bool)]
    assert len(training) >= plan["gates"]["source_train_known_min"]
    assert len(evaluation) >= plan["gates"]["source_held_mature_known_min"]
    control_names = base + ["SOXX_dollar_volume_shock20"]
    candidate_names = control_names + ["SOXL_minus_SOXX_dollar_volume_shock20"]
    xtr, xhe = training["feature"], evaluation["feature"]
    ytr = training[("label", "SOXL_relative_H5")].to_numpy(float)
    yhe = evaluation[("label", "SOXL_relative_H5")].to_numpy(float)
    assert np.isfinite(ytr).all() and np.isfinite(yhe).all()
    preflight = {"status": "matched_Qlib_source_and_label_maturity_passed",
                 "train_mature_full": int(train_mature.sum()), "held_mature_full": int(held_mature.sum()),
                 "train_matched_source_known": len(training), "held_matched_source_known": len(evaluation),
                 "source_Recorder_object_loads": 5, "DatasetH_builds": 1,
                 "historical_first_public_qualified": False}
    (card / "preflight_result.json").write_text(json.dumps(preflight, indent=2) + "\n")
    state["status"] = "fit_attempted"
    save_state()

    def model():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))

    original = model()
    activity = model()
    candidate = model()
    original.fit(xtr[base], ytr)
    state["model_fits"] = 1
    save_state()
    activity.fit(xtr[control_names], ytr)
    state["model_fits"] = 2
    save_state()
    candidate.fit(xtr[candidate_names], ytr)
    state["model_fits"] = 3
    save_state()
    p_original = original.predict(xhe[base])
    p_activity = activity.predict(xhe[control_names])
    p_candidate = candidate.predict(xhe[candidate_names])
    d_activity = (yhe - p_activity) ** 2 - (yhe - p_candidate) ** 2
    d_original = (yhe - p_original) ** 2 - (yhe - p_candidate) ** 2
    d_mean = (yhe - np.mean(ytr)) ** 2 - (yhe - p_candidate) ** 2
    score = pd.DataFrame({"signal_date": pd.DatetimeIndex(evaluation.index.get_level_values("datetime")),
                          "target": yhe, "original39": p_original,
                          "activity_control": p_activity, "candidate": p_candidate,
                          "improvement_vs_activity": d_activity,
                          "improvement_vs_original39": d_original,
                          "improvement_vs_train_mean": d_mean})
    assert score.signal_date.is_monotonic_increasing
    half = len(score) // 2
    metrics = {"standardized_relative_coefficient": float(candidate.named_steps["ridge"].coef_[-1]),
               "MSE_original39": float(np.mean((yhe - p_original) ** 2)),
               "MSE_activity_control": float(np.mean((yhe - p_activity) ** 2)),
               "MSE_candidate": float(np.mean((yhe - p_candidate) ** 2)),
               "MSE_train_mean": float(np.mean((yhe - np.mean(ytr)) ** 2)),
               "improvement_vs_activity": float(np.mean(d_activity)),
               "improvement_vs_original39": float(np.mean(d_original)),
               "improvement_vs_train_mean": float(np.mean(d_mean)),
               "first_half_vs_activity": float(np.mean(d_activity[:half])),
               "second_half_vs_activity": float(np.mean(d_activity[half:])),
               "2024_vs_activity": float(score.loc[score.signal_date.dt.year.eq(2024), "improvement_vs_activity"].mean()),
               "2025_vs_activity": float(score.loc[score.signal_date.dt.year.eq(2025), "improvement_vs_activity"].mean())}
    full = pd.Series(np.nan, index=frame.loc["2023-01-03":"2026-10-07"].index, dtype=float)
    full.loc[score.signal_date.to_numpy()] = d_activity
    assert len(full) == plan["gates"]["candidate_vs_activity_control_stationary_block_lower_positive"]["calendar_rows"]
    fixed = plan["gates"]["candidate_vs_activity_control_stationary_block_lower_positive"]
    bootstrap, _, means = full_calendar_bootstrap(full.to_numpy(float),
        mean_block=fixed["mean_block"], resamples=fixed["resamples"],
        seed=fixed["seed"], alpha=fixed["one_sided_alpha"])
    state["bootstrap_attempts"] = 1
    save_state()
    gates = {"negative_coefficient": metrics["standardized_relative_coefficient"] < 0,
             "beats_activity": metrics["improvement_vs_activity"] > 0,
             "beats_original39": metrics["improvement_vs_original39"] > 0,
             "beats_train_mean": metrics["improvement_vs_train_mean"] > 0,
             "both_halves": metrics["first_half_vs_activity"] > 0 and metrics["second_half_vs_activity"] > 0,
             "both_complete_years": metrics["2024_vs_activity"] > 0 and metrics["2025_vs_activity"] > 0,
             "positive_block_lower": bootstrap["lower_bound"] > 0}
    decision = ("conditional_information_pass_source_PIT_needed_before_account" if all(gates.values())
                else "not_promoted_fixed_relative_dollar_volume_H5_information_failed")
    with R.start(experiment_name=card.name, uri=str((card / "qlib_recorder").resolve())):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, preflight=preflight, dataset=dataset)
        recorder.save_objects(artifact_path="models", original39=original,
                              activity_control=activity, candidate=candidate)
        recorder.save_objects(artifact_path="predictions", held_scored=score)
        recorder.save_objects(artifact_path="analysis", full_held_calendar=full,
                              metrics=metrics, gates=gates, bootstrap=bootstrap,
                              bootstrap_replicate_means=means)
        R.log_params(feature="SOXL_minus_SOXX_dollar_volume_shock20", parent="f696_A0",
                     source_first_public_qualified=False)
        R.log_metrics(held_MSE_improvement_vs_activity=metrics["improvement_vs_activity"],
                      block_lower=bootstrap["lower_bound"])
        rid, eid = recorder.id, str(recorder.experiment_id)
    state.update(status="finished", recorder_id=rid, experiment_id=eid)
    save_state()
    summary = {"experiment": card.name, "decision": decision,
               "promotion": "not_promoted_information_only", "objective_achieved": False,
               "plan_sha256": _sha(card / "plan.json"), "source_recorder_id": source["coordinate_recorder_id"],
               "parent_recorder_id": source["parent_recorder_id"],
               "recorder_id": rid, "experiment_id": eid,
               "preflight": preflight, "metrics": metrics, "bootstrap": bootstrap,
               "gates": gates, "model_fits": 3, "native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first",
                                   "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "models/*", "predictions/*", "analysis/*"]},
               "limitations": ["Historical first-public Futu close and volume vintages are unqualified.",
                               "2023-2026 reuse is development feedback, not independent forward evidence.",
                               "H5 price labels omit funded Tiered fees and own-state account feedback."]}
    binding = json.loads((source_card / "nas_binding.json").read_text())
    binding.pop("evidence", None)
    (card / "nas_binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    archived = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
                                    card=card, recorder_id=rid, experiment_id=eid)
    remote = _ssh(_settings(root)["ssh"], "sha256sum " + shlex.quote(archived["nas_path"])).split()[0]
    assert remote == archived["sha256"]
    summary["NAS_evidence"] = archived
    summary["NAS_readback"] = {"verified": True, "sha256": remote}
    (card / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nentrypoint: " + card.name + ".ipynb\n"
        "phase: completed_conditional_information_screen\n")
    (card / "conclusion.md").write_text(
        "# Stage1 相对成交额近似异常：固定 H5 信息门\n\n"
        "冻结原 f696 A0 完整链为研究父本，SOXX 为最终硬门。"
        "条件性检验 SOXL 相对 SOXX 前一完成日成交额近似异常，"
        "与原39项加 SOXX 单边活动对照使用完全相同的成熟日期、Ridge 参数。"
        "当前 Futu 历史首次公开版本尚未核证，故此结果仅是复用历史开发反馈。\n\n"
        f"训练共同成熟 {len(training)} 日、评价 {len(evaluation)} 日；3拟合、1固定区块检验、0资金账户。"
        f"新增标准化系数 {metrics['standardized_relative_coefficient']:.8f}（预定负号）。"
        f"候选/活动对照/原39项/训练均值 MSE 为 "
        f"{metrics['MSE_candidate']:.9f}/{metrics['MSE_activity_control']:.9f}/"
        f"{metrics['MSE_original39']:.9f}/{metrics['MSE_train_mean']:.9f}。"
        f"候选相对活动对照 MSE 改善 {metrics['improvement_vs_activity']:.9f}，"
        f"区块下界 {bootstrap['lower_bound']:.9f}；前后半改善 "
        f"{metrics['first_half_vs_activity']:.9f}/{metrics['second_half_vs_activity']:.9f}，"
        f"2024/2025 为 {metrics['2024_vs_activity']:.9f}/{metrics['2025_vs_activity']:.9f}。"
        f"七项门 {sum(gates.values())}/7；决定 `{decision}`。\n\n"
        "信息分数不是费后 SOXX 超额；若失败，固定版本关闭，不反转符号或调窗口。"
        "即使通过，历史首次公开来源需先认证，之后仍须另冻完整 Stage1 自身状态0bp账户、2/5bp和独立前向。"
        f"Recorder `{rid}`，NAS SHA `{remote}`。\n")
    return summary
