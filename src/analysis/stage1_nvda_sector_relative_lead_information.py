"""One fixed SOXL-relative H5 information screen for NVDA sector leadership."""

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
    card = root / "research/stage1_nvda_sector_relative_lead_h5_information_20261023"
    if (card / "run_state.json").exists():
        raise ValueError("fixed NVDA leadership information attempt already started")
    plan = json.loads((card / "plan.json").read_text())
    src = plan["source"]
    assert plan["frozen"]
    feature_card = root / src["feature_card"]
    parent_card = root / src["parent_card"]
    for relative, expected in [
        (src["feature_card"] + "/summary.json", src["feature_summary_sha256"]),
        (src["feature_card"] + "/result.json", src["feature_result_sha256"]),
        (src["feature_provider_uri"] + "/qlib_store_manifest.json", src["feature_provider_manifest_sha256"]),
        ("src/analysis/stage1_nvda_sector_relative_lead_source.py", src["feature_module_sha256"]),
        (src["parent_card"] + "/summary.json", src["parent_summary_sha256"]),
        ("src/analysis/seed_disagreement_information.py", src["bootstrap_module_sha256"]),
    ]:
        assert _sha(root / relative) == expected, relative
    fs = json.loads((feature_card / "summary.json").read_text())
    ps = json.loads((parent_card / "summary.json").read_text())
    assert fs["recorder_id"] == src["feature_recorder_id"]
    assert fs["NAS_evidence"]["sha256"] == src["feature_NAS_sha256"]
    assert fs["decision"] == "conditional_source_novelty_pass_first_public_pending"
    assert json.loads((feature_card / "nas_scoped_restore.json").read_text())["status"] == "NAS_scoped_restore_native_NVDA_lead_passed"
    assert ps["recorder_id"] == src["parent_recorder_id"]
    assert ps["nas_evidence"]["sha256"] == src["parent_NAS_sha256"]
    state = {"status": "preflight", "plan_sha256": _sha(card / "plan.json"),
             "source_Recorder_object_loads": 0, "DatasetH_builds": 0,
             "model_fits": 0, "bootstrap_attempts": 0, "native_accounts": 0}

    def save_state() -> None:
        (card / "run_state.json").write_text(json.dumps(state, indent=2) + "\n")

    save_state()
    qlib.init(provider_uri=str((root / src["feature_provider_uri"]).resolve()), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(start_time="2010-03-11", end_time="2026-10-09", freq="day"))
    prices = D.features(["NVDA", "SOXX"], ["$close"],
        start_time="2010-03-11", end_time="2026-10-09", freq="day")["$close"].unstack("instrument").reindex(calendar)
    assert prices.SOXX.notna().all()
    expected = (np.log(prices.NVDA.div(prices.NVDA.shift(20))) -
                np.log(prices.SOXX.div(prices.SOXX.shift(20))))
    with R.uri_context(str((feature_card / "qlib_recorder").resolve())):
        feature = R.get_recorder(experiment_id=src["feature_experiment_id"],
                                 recorder_id=src["feature_recorder_id"]).load_object("analysis/lead")
    state["source_Recorder_object_loads"] = 1
    np.testing.assert_allclose(feature.to_numpy(float), expected.reindex(feature.index).to_numpy(float),
                               rtol=0, atol=1e-9, equal_nan=True)
    with R.uri_context(str((parent_card / "qlib_recorder").resolve())):
        parent = R.get_recorder(experiment_id=src["parent_experiment_id"],
                                recorder_id=src["parent_recorder_id"])
        features = parent.load_object("inputs/features")
        frame = parent.load_object("inputs/daily_frame")
        timing = parent.load_object("inputs/timing")
    state["source_Recorder_object_loads"] = 4
    assert features.index.equals(frame.index) and frame.index.equals(timing.index)
    assert feature.index.equals(frame.index)
    base = list(ps["input_audit"]["original_features"])
    assert len(base) == 39 and set(base) <= set(features.columns)
    receiver = plan["receiver"]
    label = _h5(frame.soxl_forward_open_return) - _h5(frame.soxx_forward_open_return)
    maturity = pd.to_datetime(timing.return_available_at.shift(-4), utc=True)
    train_cutoff = pd.Timestamp(receiver["train_label_available_before"]).tz_convert("UTC")
    held_cutoff = pd.Timestamp(receiver["held_label_available_before"]).tz_convert("UTC")
    train_mature = ((frame.index >= pd.Timestamp(receiver["train_signals"][0])) &
                    (frame.index <= pd.Timestamp(receiver["train_signals"][1])) &
                    (maturity < train_cutoff) & label.notna())
    held_mature = ((frame.index >= pd.Timestamp(receiver["reused_held_signals"][0])) &
                   (frame.index <= pd.Timestamp(receiver["reused_held_signals"][1])) &
                   (maturity < held_cutoff) & label.notna())
    assert int(train_mature.sum()) == receiver["expected_train_mature"]
    assert int(held_mature.sum()) == receiver["expected_held_mature"]
    assert feature.loc[train_mature].notna().all() and feature.loc[held_mature].notna().all()
    static = pd.concat({
        "feature": features[base].assign(NVDA_minus_SOXX_log_momentum20=feature),
        "label": pd.DataFrame({"SOXL_relative_H5": label}),
        "meta": pd.DataFrame({"train_mature": train_mature, "held_mature": held_mature}),
    }, axis=1)
    static.index = pd.MultiIndex.from_arrays(
        [frame.index, ["SOXL_SOXX_CASH"] * len(frame)], names=["datetime", "instrument"])
    dataset = DatasetH(handler=DataHandlerLP.from_df(static),
        segments={"train": receiver["train_signals"], "held": receiver["reused_held_signals"]})
    state["DatasetH_builds"] = 1
    training = dataset.prepare("train", col_set=["feature", "label", "meta"])
    evaluation = dataset.prepare("held", col_set=["feature", "label", "meta"])
    training = training.loc[training[("meta", "train_mature")].astype(bool)]
    evaluation = evaluation.loc[evaluation[("meta", "held_mature")].astype(bool)]
    assert len(training) == 3014 and len(evaluation) == 940
    xtr, xhe = training["feature"], evaluation["feature"]
    ytr = training[("label", "SOXL_relative_H5")].to_numpy(float)
    yhe = evaluation[("label", "SOXL_relative_H5")].to_numpy(float)
    assert np.isfinite(ytr).all() and np.isfinite(yhe).all()
    preflight = {"status": "matched_Qlib_source_and_relative_label_maturity_passed",
                 "train_mature": len(training), "held_mature": len(evaluation),
                 "source_Recorder_object_loads": 4, "DatasetH_builds": 1,
                 "historical_first_public_qualified": False}
    (card / "preflight_result.json").write_text(json.dumps(preflight, indent=2) + "\n")

    def model():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))

    baseline = model()
    candidate = model()
    names = base + ["NVDA_minus_SOXX_log_momentum20"]
    state["status"] = "fit_attempted"
    save_state()
    baseline.fit(xtr[base], ytr)
    state["model_fits"] = 1
    save_state()
    candidate.fit(xtr[names], ytr)
    state["model_fits"] = 2
    save_state()
    p0, p1 = baseline.predict(xhe[base]), candidate.predict(xhe[names])
    delta = (yhe - p0) ** 2 - (yhe - p1) ** 2
    delta_mean = (yhe - np.mean(ytr)) ** 2 - (yhe - p1) ** 2
    score = pd.DataFrame({"signal_date": pd.DatetimeIndex(evaluation.index.get_level_values("datetime")),
                          "target": yhe, "baseline": p0, "candidate": p1,
                          "paired_improvement": delta, "improvement_vs_train_mean": delta_mean})
    assert score.signal_date.is_monotonic_increasing
    half = len(score) // 2
    metrics = {"standardized_NVDA_lead_coefficient": float(candidate.named_steps["ridge"].coef_[-1]),
               "MSE_baseline": float(np.mean((yhe - p0) ** 2)),
               "MSE_candidate": float(np.mean((yhe - p1) ** 2)),
               "MSE_train_mean": float(np.mean((yhe - np.mean(ytr)) ** 2)),
               "improvement_vs_baseline": float(np.mean(delta)),
               "improvement_vs_train_mean": float(np.mean(delta_mean)),
               "first_half_improvement": float(np.mean(delta[:half])),
               "second_half_improvement": float(np.mean(delta[half:])),
               "2024_improvement": float(score.loc[score.signal_date.dt.year.eq(2024), "paired_improvement"].mean()),
               "2025_improvement": float(score.loc[score.signal_date.dt.year.eq(2025), "paired_improvement"].mean())}
    full = pd.Series(np.nan, index=frame.loc["2023-01-03":"2026-10-07"].index, dtype=float)
    full.loc[score.signal_date.to_numpy()] = delta
    fixed = plan["gate"]["candidate_minus_original39_full_calendar_stationary_block_lower_positive"]
    assert len(full) == fixed["calendar_rows"] and int(full.notna().sum()) == 940
    bootstrap, _, means = full_calendar_bootstrap(full.to_numpy(float),
        mean_block=fixed["mean_block"], resamples=fixed["resamples"],
        seed=fixed["seed"], alpha=fixed["one_sided_alpha"])
    state["bootstrap_attempts"] = 1
    save_state()
    gates = {"negative_coefficient": metrics["standardized_NVDA_lead_coefficient"] < 0,
             "beats_original39": metrics["improvement_vs_baseline"] > 0,
             "beats_train_mean": metrics["improvement_vs_train_mean"] > 0,
             "both_halves": metrics["first_half_improvement"] > 0 and metrics["second_half_improvement"] > 0,
             "both_complete_years": metrics["2024_improvement"] > 0 and metrics["2025_improvement"] > 0,
             "positive_block_lower": bootstrap["lower_bound"] > 0}
    decision = ("conditional_information_pass_source_PIT_needed_before_account" if all(gates.values())
                else "not_promoted_fixed_NVDA_sector_relative_lead_H5_information_failed")
    with R.start(experiment_name=card.name, uri=str((card / "qlib_recorder").resolve())):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, preflight=preflight, dataset=dataset)
        recorder.save_objects(artifact_path="models", baseline=baseline, candidate=candidate)
        recorder.save_objects(artifact_path="predictions", held_scored=score)
        recorder.save_objects(artifact_path="analysis", full_held_calendar=full,
                              metrics=metrics, gates=gates, bootstrap=bootstrap,
                              bootstrap_replicate_means=means)
        R.log_params(feature="NVDA_minus_SOXX_log_momentum20", parent="f696_A0",
                     source_first_public_qualified=False)
        R.log_metrics(held_MSE_improvement=metrics["improvement_vs_baseline"],
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
               "gates": gates, "model_fits": 2, "native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "models/*", "predictions/*", "analysis/*"]},
               "limitations": ["Early Yahoo Qlib and later Futu peer histories are retrospective, not proven first-public vintages.",
                               "NVDA relative price leadership is not actual SOXX holdings contribution.",
                               "H5 price labels omit funded Tiered fees and own-state account feedback."]}
    binding = json.loads((feature_card / "nas_binding.json").read_text())
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
        "# Stage1 NVDA相对SOXX单股领导性：固定H5信息门\n\n"
        "冻结原f696 A0完整链为母本；SOXX为最终硬门。用前20会话NVDA减SOXX对数动量"
        "检验窄幅龙头上涨后的SOXL相对收益脆弱性，原39项Ridge和加一项候选同日配对。"
        "当前历史首次公开版本未知，结果仅属复用历史条件性反馈。\n\n"
        f"训练3014、评价940成熟日；2拟合、1固定区块检验、0资金账户。新增系数 "
        f"{metrics['standardized_NVDA_lead_coefficient']:.8f}（预定负号）；"
        f"候选/原39/训练均值MSE为 {metrics['MSE_candidate']:.9f}/"
        f"{metrics['MSE_baseline']:.9f}/{metrics['MSE_train_mean']:.9f}。"
        f"候选相对原39改善 {metrics['improvement_vs_baseline']:.9f}，"
        f"区块下界 {bootstrap['lower_bound']:.9f}；前/后半改善 "
        f"{metrics['first_half_improvement']:.9f}/{metrics['second_half_improvement']:.9f}，"
        f"2024/2025为 {metrics['2024_improvement']:.9f}/{metrics['2025_improvement']:.9f}。"
        f"六项门 {sum(gates.values())}/6，决定 `{decision}`。\n\n"
        "信息误差不等于费后SOXX收益；失败版不改符号、股票、期限或年份。"
        "即使通过也须先核实来源首发时点，再另冻f696 0bp Tiered账户。"
        f"Recorder `{rid}`，NAS SHA `{remote}`。\n")
    return summary
