"""Fixed original39-matched cash H5 test for prior SOXX day/night tug."""

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
    from src.data.nas_market_data import _settings, _ssh, archive_nas_recorder

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    root = Path(root).resolve()
    card = root / "research/stage1_soxx_prior_session_tug_cash_h5_information_20261023"
    if (card / "run_state.json").exists():
        raise ValueError("fixed cash H5 attempt already started")
    plan = json.loads((card / "plan.json").read_text())
    src = plan["source"]
    assert plan["frozen"]
    feature_card = root / src["feature_card"]
    parent_card = root / src["parent_card"]
    for relative, digest in [
        (src["feature_card"] + "/summary.json", src["feature_summary_sha256"]),
        (src["feature_card"] + "/result.json", src["feature_result_sha256"]),
        ("src/analysis/stage1_prior_session_tug_source.py", src["feature_module_sha256"]),
        (src["provider_uri"] + "/qlib_store_manifest.json", src["provider_manifest_sha256"]),
        (src["parent_card"] + "/summary.json", src["parent_summary_sha256"]),
        (src["parent_card"] + "/verification_manifest.json", src["parent_verification_manifest_sha256"]),
        ("src/analysis/seed_disagreement_information.py", src["bootstrap_module_sha256"]),
    ]:
        assert _sha(root / relative) == digest, relative
    feature_summary = json.loads((feature_card / "summary.json").read_text())
    parent_summary = json.loads((parent_card / "summary.json").read_text())
    assert feature_summary["recorder_id"] == src["feature_recorder_id"]
    assert feature_summary["NAS_evidence"]["sha256"] == src["feature_NAS_sha256"]
    assert feature_summary["decision"] == "source_novelty_pass_needs_separate_cash_H5_information"
    assert json.loads((feature_card / "nas_scoped_restore.json").read_text())["status"] == "NAS_scoped_restore_native_source_feature_passed"
    assert parent_summary["recorder_id"] == src["parent_recorder_id"]
    assert parent_summary["nas_evidence"]["sha256"] == src["parent_NAS_sha256"]
    state = {"status": "preflight", "plan_sha256": _sha(card / "plan.json"),
             "source_Recorder_object_loads": 0, "DatasetH_builds": 0,
             "model_fits": 0, "bootstrap_attempts": 0, "native_accounts": 0}

    def save_state() -> None:
        (card / "run_state.json").write_text(json.dumps(state, indent=2) + "\n")

    save_state()
    qlib.init(provider_uri=str((root / src["provider_uri"]).resolve()), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(start_time="2010-03-11", end_time="2026-10-09", freq="day"))
    bars = D.features(["SOXX"], ["$open", "$close"],
        start_time="2010-03-11", end_time="2026-10-09", freq="day")
    open_ = bars["$open"].droplevel("instrument").reindex(calendar)
    close = bars["$close"].droplevel("instrument").reindex(calendar)
    expected_feature = (np.log(close.div(open_)) - np.log(open_.div(close.shift(1))))
    expected_total = np.log(close.div(close.shift(1)))
    with R.uri_context(str((feature_card / "qlib_recorder").resolve())):
        feature_recorder = R.get_recorder(experiment_id=src["feature_experiment_id"],
            recorder_id=src["feature_recorder_id"])
        feature = feature_recorder.load_object("analysis/feature")
        total = feature_recorder.load_object("analysis/total")
    state["source_Recorder_object_loads"] = 2
    expected_feature = expected_feature.reindex(feature.index)
    expected_total = expected_total.reindex(total.index)
    np.testing.assert_allclose(feature.to_numpy(float), expected_feature.to_numpy(float),
                               rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(total.to_numpy(float), expected_total.to_numpy(float),
                               rtol=0, atol=1e-9, equal_nan=True)
    with R.uri_context(str((parent_card / "qlib_recorder").resolve())):
        parent = R.get_recorder(experiment_id=src["parent_experiment_id"],
                                recorder_id=src["parent_recorder_id"])
        features = parent.load_object("inputs/features")
        frame = parent.load_object("inputs/daily_frame")
        timing = parent.load_object("inputs/timing")
    state["source_Recorder_object_loads"] = 5
    assert features.index.equals(frame.index) and frame.index.equals(timing.index)
    assert feature.index.equals(frame.index) and total.index.equals(frame.index)
    base = list(parent_summary["input_audit"]["original_features"])
    assert len(base) == 39 and set(base) <= set(features.columns)
    receiver = plan["receiver"]
    label = (-_h5(frame.soxx_forward_open_return)).rename("cash_advantage_H5")
    maturity = pd.to_datetime(timing.return_available_at.shift(-4), utc=True)
    train_cutoff = pd.Timestamp(receiver["train_label_available_before"]).tz_convert("UTC")
    held_cutoff = pd.Timestamp(receiver["held_label_available_before"]).tz_convert("UTC")
    train_mature = ((frame.index >= pd.Timestamp(receiver["train_signals"][0])) &
                    (frame.index <= pd.Timestamp(receiver["train_signals"][1])) &
                    (maturity < train_cutoff) & label.notna())
    held_mature = ((frame.index >= pd.Timestamp(receiver["reused_held_signals"][0])) &
                   (frame.index <= pd.Timestamp(receiver["reused_held_signals"][1])) &
                   (maturity < held_cutoff) & label.notna())
    assert int(train_mature.sum()) == 3014 and int(held_mature.sum()) == 940
    static = pd.concat({
        "feature": features[base].assign(SOXX_total_log_return1=total,
                                          SOXX_prior_session_intraday_minus_overnight_log_return=feature),
        "label": pd.DataFrame({"cash_advantage_H5": label}),
        "meta": pd.DataFrame({"train_mature": train_mature,
                              "held_mature": held_mature,
                              "source_known": feature.notna() & total.notna()}),
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
    assert len(training) == 3014 and len(evaluation) == 940
    control_names = base + ["SOXX_total_log_return1"]
    candidate_names = control_names + ["SOXX_prior_session_intraday_minus_overnight_log_return"]
    xtr, xhe = training["feature"], evaluation["feature"]
    ytr = training[("label", "cash_advantage_H5")].to_numpy(float)
    yhe = evaluation[("label", "cash_advantage_H5")].to_numpy(float)
    assert np.isfinite(ytr).all() and np.isfinite(yhe).all()
    preflight = {"status": "matched_Qlib_source_and_cash_label_maturity_passed",
                 "train_mature": len(training), "held_mature": len(evaluation),
                 "source_Recorder_object_loads": 5, "DatasetH_builds": 1}
    (card / "preflight_result.json").write_text(json.dumps(preflight, indent=2) + "\n")
    state["status"] = "fit_attempted"
    save_state()

    def model():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))

    original = model()
    control = model()
    candidate = model()
    original.fit(xtr[base], ytr)
    state["model_fits"] = 1
    save_state()
    control.fit(xtr[control_names], ytr)
    state["model_fits"] = 2
    save_state()
    candidate.fit(xtr[candidate_names], ytr)
    state["model_fits"] = 3
    save_state()
    p_original, p_control, p_candidate = (
        original.predict(xhe[base]), control.predict(xhe[control_names]),
        candidate.predict(xhe[candidate_names]))
    d_control = (yhe - p_control) ** 2 - (yhe - p_candidate) ** 2
    d_original = (yhe - p_original) ** 2 - (yhe - p_candidate) ** 2
    d_mean = (yhe - np.mean(ytr)) ** 2 - (yhe - p_candidate) ** 2
    score = pd.DataFrame({"signal_date": pd.DatetimeIndex(evaluation.index.get_level_values("datetime")),
                          "target": yhe, "original39": p_original,
                          "total_return_control": p_control, "candidate": p_candidate,
                          "improvement_vs_control": d_control,
                          "improvement_vs_original39": d_original,
                          "improvement_vs_train_mean": d_mean})
    assert score.signal_date.is_monotonic_increasing
    half = len(score) // 2
    metrics = {"standardized_tug_coefficient": float(candidate.named_steps["ridge"].coef_[-1]),
               "MSE_original39": float(np.mean((yhe - p_original) ** 2)),
               "MSE_total_return_control": float(np.mean((yhe - p_control) ** 2)),
               "MSE_candidate": float(np.mean((yhe - p_candidate) ** 2)),
               "MSE_train_mean": float(np.mean((yhe - np.mean(ytr)) ** 2)),
               "improvement_vs_control": float(np.mean(d_control)),
               "improvement_vs_original39": float(np.mean(d_original)),
               "improvement_vs_train_mean": float(np.mean(d_mean)),
               "first_half_vs_control": float(np.mean(d_control[:half])),
               "second_half_vs_control": float(np.mean(d_control[half:])),
               "2024_vs_control": float(score.loc[score.signal_date.dt.year.eq(2024), "improvement_vs_control"].mean()),
               "2025_vs_control": float(score.loc[score.signal_date.dt.year.eq(2025), "improvement_vs_control"].mean())}
    full = pd.Series(np.nan, index=frame.loc["2023-01-03":"2026-10-07"].index, dtype=float)
    full.loc[score.signal_date.to_numpy()] = d_control
    fixed = plan["gate"]["candidate_vs_total_return_control_full_calendar_block_lower_positive"]
    assert len(full) == fixed["calendar_rows"] and int(full.notna().sum()) == 940
    bootstrap, _, means = full_calendar_bootstrap(full.to_numpy(float),
        mean_block=fixed["mean_block"], resamples=fixed["resamples"],
        seed=fixed["seed"], alpha=fixed["one_sided_alpha"])
    state["bootstrap_attempts"] = 1
    save_state()
    gates = {"negative_coefficient": metrics["standardized_tug_coefficient"] < 0,
             "beats_total_return_control": metrics["improvement_vs_control"] > 0,
             "beats_original39": metrics["improvement_vs_original39"] > 0,
             "beats_train_mean": metrics["improvement_vs_train_mean"] > 0,
             "both_halves": metrics["first_half_vs_control"] > 0 and metrics["second_half_vs_control"] > 0,
             "both_complete_years": metrics["2024_vs_control"] > 0 and metrics["2025_vs_control"] > 0,
             "positive_block_lower": bootstrap["lower_bound"] > 0}
    decision = ("information_gate_passed_requires_full_f696_0bp_account" if all(gates.values())
                else "not_promoted_fixed_prior_session_tug_cash_H5_information_failed")
    with R.start(experiment_name=card.name, uri=str((card / "qlib_recorder").resolve())):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, preflight=preflight, dataset=dataset)
        recorder.save_objects(artifact_path="models", original39=original,
                              total_return_control=control, candidate=candidate)
        recorder.save_objects(artifact_path="predictions", held_scored=score)
        recorder.save_objects(artifact_path="analysis", full_held_calendar=full,
                              metrics=metrics, gates=gates, bootstrap=bootstrap,
                              bootstrap_replicate_means=means)
        R.log_params(feature="SOXX_prior_session_intraday_minus_overnight_log_return", parent="f696_A0")
        R.log_metrics(held_MSE_improvement_vs_control=metrics["improvement_vs_control"],
                      block_lower=bootstrap["lower_bound"])
        rid, eid = recorder.id, str(recorder.experiment_id)
    state.update(status="finished", recorder_id=rid, experiment_id=eid)
    save_state()
    summary = {"experiment": card.name, "decision": decision,
               "promotion": "not_promoted_information_only", "objective_achieved": False,
               "plan_sha256": _sha(card / "plan.json"),
               "source_recorder_id": src["feature_recorder_id"],
               "parent_recorder_id": src["parent_recorder_id"],
               "recorder_id": rid, "experiment_id": eid,
               "preflight": preflight, "metrics": metrics, "bootstrap": bootstrap,
               "gates": gates, "model_fits": 3, "native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "models/*", "predictions/*", "analysis/*"]},
               "limitations": ["2023-2026 reuse is development feedback, not independent forward evidence.",
                               "H5 SOXX price labels omit funded Tiered fees and own-state feedback.",
                               "The full f696 0bp economic and risk gates are not tested by this information screen."]}
    binding = json.loads((feature_card / "nas_binding.json").read_text())
    binding.pop("evidence", None)
    (card / "nas_binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    evidence = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
                                    card=card, recorder_id=rid, experiment_id=eid)
    remote = _ssh(_settings(root)["ssh"], "sha256sum " + shlex.quote(evidence["nas_path"])).split()[0]
    assert remote == evidence["sha256"]
    summary["NAS_evidence"] = evidence
    summary["NAS_readback"] = {"verified": True, "sha256": remote}
    (card / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nentrypoint: " + card.name + ".ipynb\n"
        "phase: completed_cash_H5_information_screen\n")
    (card / "conclusion.md").write_text(
        "# Stage1 SOXX 前一完成日日内减隔夜：现金优势 H5 信息门\n\n"
        "原 f696 A0 完整链为母本，只检验次日09:29已知的 SOXX 日频开收盘分量。"
        "原39项、原39项加 SOXX 同日总收益控制、再加日内减隔夜候选使用完全相同的3014/940成熟日期、"
        "Ridge和固定区块检验。无新闻、无新行情、无经济账户。\n\n"
        f"新增标准化系数 {metrics['standardized_tug_coefficient']:.8f}（预定负号）；"
        f"候选/总收益控制/原39项/训练均值 MSE 为 {metrics['MSE_candidate']:.9f}/"
        f"{metrics['MSE_total_return_control']:.9f}/{metrics['MSE_original39']:.9f}/"
        f"{metrics['MSE_train_mean']:.9f}。对总收益控制改善 {metrics['improvement_vs_control']:.9f}，"
        f"区块下界 {bootstrap['lower_bound']:.9f}；前后半改善 "
        f"{metrics['first_half_vs_control']:.9f}/{metrics['second_half_vs_control']:.9f}，"
        f"2024/2025 为 {metrics['2024_vs_control']:.9f}/{metrics['2025_vs_control']:.9f}。"
        f"七项门 {sum(gates.values())}/7，决定 `{decision}`。\n\n"
        "信息门并非 SOXX 费用后资金收益。若失败，此固定版本不改符号、公式、年度或窗口，"
        "不运行完整0bp账户；通过才允许另冻 f696 自身状态0bp Tiered账户，再看2/5bp与独立前向。"
        f"Recorder `{rid}`，NAS SHA `{remote}`。\n")
    return summary
