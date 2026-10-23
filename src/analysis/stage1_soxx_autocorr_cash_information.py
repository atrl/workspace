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
    card = root / "research/stage1_soxx_autocorr_cash_h5_information_20261023"
    plan = json.loads((card / "plan.json").read_text())
    source = plan["source"]
    parent = root / source["parent_card"]
    assert plan["frozen"] and not (card / "run_state.json").exists()
    assert _sha(root / source["provider_uri"] / "qlib_store_manifest.json") == source["provider_manifest_sha256"]
    assert _sha(parent / "summary.json") == source["parent_summary_sha256"]
    assert _sha(parent / "verification_manifest.json") == source["parent_verification_manifest_sha256"]
    assert _sha(root / "src/analysis/seed_disagreement_information.py") == source["bootstrap_module_sha256"]
    parent_summary = json.loads((parent / "summary.json").read_text())
    assert parent_summary["recorder_id"] == source["parent_recorder_id"]
    assert parent_summary["nas_evidence"]["sha256"] == source["parent_NAS_archive_sha256"]
    assert json.loads((parent / "nas_binding.json").read_text())["evidence"]["sha256"] == source["parent_NAS_archive_sha256"]

    state = {"status": "preflight", "plan_sha256": _sha(card / "plan.json"),
             "source_Recorder_object_loads": 0, "Qlib_D_feature_calls": 0,
             "Qlib_D_calendar_calls": 0, "DatasetH_builds": 0,
             "model_fits": 0, "bootstrap_attempts": 0, "native_accounts": 0}

    def save_state() -> None:
        (card / "run_state.json").write_text(json.dumps(state, indent=2) + "\n")

    save_state()
    qlib.init(provider_uri=str((root / source["provider_uri"]).resolve()), region="us", kernels=1)
    calendar = D.calendar(start_time="2010-03-11", end_time="2026-10-09", freq="day")
    state["Qlib_D_calendar_calls"] = 1
    assert len(calendar) > 4000
    with R.uri_context(str((parent / "qlib_recorder").resolve())):
        recorder = R.get_recorder(experiment_name=parent.name, recorder_id=source["parent_recorder_id"])
        features = recorder.load_object("inputs/features")
        frame = recorder.load_object("inputs/daily_frame")
        timing = recorder.load_object("inputs/timing")
    state["source_Recorder_object_loads"] = 3
    assert features.index.equals(frame.index) and frame.index.equals(timing.index)
    base = list(parent_summary["input_audit"]["original_features"])
    assert len(base) == 39 and set(base) <= set(features.columns)

    close = D.features(
        ["SOXX"], ["$close"], start_time="2010-03-11",
        end_time="2026-10-09", freq="day"
    )["$close"].droplevel("instrument").sort_index()
    state["Qlib_D_feature_calls"] = 1
    assert close.notna().all()
    ret = close.pct_change(fill_method=None)
    coordinate = ret.rolling(10, min_periods=10).corr(ret.shift(1))
    coordinate = coordinate.rename("soxx_return_autocorr10").reindex(features.index)
    assert coordinate.loc["2011-01-03":"2026-10-07"].notna().all()

    label = -_h5(frame.soxx_forward_open_return)
    mature = pd.to_datetime(timing.return_available_at.shift(-4), utc=True)
    train_cutoff = pd.Timestamp(plan["cohorts"]["train_label_available_before"]).tz_convert("UTC")
    held_terminal = pd.Timestamp(plan["cohorts"]["reused_held_label_available_before"]).tz_convert("UTC")
    train = ((frame.index >= pd.Timestamp("2011-01-03")) &
             (frame.index <= pd.Timestamp("2022-12-30")) &
             (mature < train_cutoff) & label.notna())
    held = ((frame.index >= pd.Timestamp("2023-01-03")) &
            (frame.index <= pd.Timestamp("2026-10-07")) &
            (mature < held_terminal) & label.notna())
    assert int(train.sum()) == plan["cohorts"]["expected_train_mature"]
    assert int(held.sum()) == plan["cohorts"]["expected_held_mature"]
    static = pd.concat({
        "feature": features[base].assign(soxx_return_autocorr10=coordinate),
        "label": pd.DataFrame({"cash_minus_SOXX_H5": label}),
        "meta": pd.DataFrame({"train_mature": train, "held_mature": held}),
    }, axis=1)
    static.index = pd.MultiIndex.from_arrays(
        [frame.index, ["SOXL_SOXX_CASH"] * len(frame)], names=["datetime", "instrument"]
    )
    dataset = DatasetH(
        handler=DataHandlerLP.from_df(static),
        segments={"train": plan["cohorts"]["train_signal"],
                  "held": plan["cohorts"]["reused_held_signal"]},
    )
    state["DatasetH_builds"] = 1
    training = dataset.prepare("train", col_set=["feature", "label", "meta"])
    evaluation = dataset.prepare("held", col_set=["feature", "label", "meta"])
    training = training.loc[training[("meta", "train_mature")].astype(bool)]
    evaluation = evaluation.loc[evaluation[("meta", "held_mature")].astype(bool)]
    assert len(training) == 3014 and len(evaluation) == 940
    names = base + ["soxx_return_autocorr10"]
    xtr, xhe = training["feature"], evaluation["feature"]
    ytr = training[("label", "cash_minus_SOXX_H5")].to_numpy(float)
    yhe = evaluation[("label", "cash_minus_SOXX_H5")].to_numpy(float)
    assert np.isfinite(ytr).all() and np.isfinite(yhe).all()
    preflight = {"status": "matched_Qlib_source_and_label_maturity_passed",
                 "train_mature": len(training), "held_mature": len(evaluation),
                 "candidate_train_known": int(xtr.soxx_return_autocorr10.notna().sum()),
                 "candidate_held_known": int(xhe.soxx_return_autocorr10.notna().sum()),
                 "parent_Recorder_object_loads": 3, "Qlib_D_feature_calls": 1,
                 "Qlib_D_calendar_calls": 1, "DatasetH_builds": 1,
                 "parent_recorder_id": source["parent_recorder_id"]}
    (card / "preflight_result.json").write_text(json.dumps(preflight, indent=2) + "\n")

    state["status"] = "fit_attempted"
    save_state()
    baseline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    candidate = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    baseline.fit(xtr[base], ytr)
    state["model_fits"] = 1
    save_state()
    candidate.fit(xtr[names], ytr)
    state["model_fits"] = 2
    save_state()
    p0, p1 = baseline.predict(xhe[base]), candidate.predict(xhe[names])
    delta = (yhe - p0) ** 2 - (yhe - p1) ** 2
    score = pd.DataFrame({"signal_date": pd.DatetimeIndex(evaluation.index.get_level_values("datetime")),
                          "target": yhe, "baseline": p0, "candidate": p1,
                          "paired_MSE_improvement": delta})
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
               "2024_improvement": float(score.loc[score.signal_date.dt.year == 2024, "paired_MSE_improvement"].mean()),
               "2025_improvement": float(score.loc[score.signal_date.dt.year == 2025, "paired_MSE_improvement"].mean())}
    full = pd.Series(np.nan, index=frame.loc["2023-01-03":"2026-10-07"].index, dtype=float)
    full.loc[score.signal_date.to_numpy()] = delta
    assert len(full) == 944 and int(full.notna().sum()) == 940
    bootstrap, _, means = full_calendar_bootstrap(
        full.to_numpy(float), mean_block=63, resamples=2000, seed=20261023, alpha=0.025)
    state["bootstrap_attempts"] = 1
    save_state()
    gates = {"negative_coefficient": metrics["coefficient_standardized"] < 0,
             "beats_baseline": metrics["all_held_improvement_vs_baseline"] > 0,
             "beats_train_mean": metrics["all_held_improvement_vs_train_mean"] > 0,
             "both_halves": metrics["first_half_improvement"] > 0 and metrics["second_half_improvement"] > 0,
             "both_complete_years": metrics["2024_improvement"] > 0 and metrics["2025_improvement"] > 0,
             "positive_block_lower": bootstrap["lower_bound"] > 0}
    decision = ("information_gate_passed_requires_full_0bp_account" if all(gates.values())
                else "not_promoted_fixed_SOXX_autocorr_cash_H5_information_failed")

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
        R.log_params(feature="soxx_return_autocorr10", parent_recorder_id=source["parent_recorder_id"])
        R.log_metrics(held_MSE_improvement=metrics["all_held_improvement_vs_baseline"],
                      block_lower=bootstrap["lower_bound"])
        rid, eid = recorder.id, str(recorder.experiment_id)
    summary = {"experiment": card.name, "decision": decision,
               "promotion": "not_promoted_information_only", "objective_achieved": False,
               "plan_sha256": _sha(card / "plan.json"), "source_provider_uri": source["provider_uri"],
               "source_provider_manifest_sha256": source["provider_manifest_sha256"],
               "parent_recorder_id": source["parent_recorder_id"], "recorder_id": rid,
               "experiment_id": eid, "metrics": metrics, "bootstrap": bootstrap,
               "gates": gates, "model_fits": 2, "native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first",
                                   "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "models/*", "predictions/*", "analysis/*"]},
               "limitations": ["2023-2026 reuse is development feedback, not independent forward evidence.",
                               "H5 labels omit funded Tiered fees and account feedback.",
                               "No SOXL relative receiver is tested here."]}
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
        "# Stage1 SOXX 自相关现金 H5 信息筛选\n\n"
        "完整冻结 Stage1 A0 为母本，SOXX 为最终硬门。本卡只检验现有 Qlib 收盘价形成的 10 日 SOXX 收益自相关，"
        "原39项、同一成熟标签和 Ridge 对照不变。\n\n"
        f"训练成熟 {len(training)} 行，复用历史评价 {len(evaluation)} 行；2 拟合、1 区块 bootstrap、0 账户。"
        f"新增标准化系数 {metrics['coefficient_standardized']:.8f}。候选 MSE {metrics['all_held_MSE_candidate']:.9f}，"
        f"原39对照 {metrics['all_held_MSE_baseline']:.9f}，训练均值 {metrics['all_held_MSE_train_mean']:.9f}。"
        f"对照改善 {metrics['all_held_improvement_vs_baseline']:.9f}，区块下界 {bootstrap['lower_bound']:.9f}。"
        f"前/后半改善 {metrics['first_half_improvement']:.9f}/{metrics['second_half_improvement']:.9f}，"
        f"2024/2025 改善 {metrics['2024_improvement']:.9f}/{metrics['2025_improvement']:.9f}。"
        f"六项固定门 {sum(gates.values())}/6；决定 `{decision}`。\n\n"
        "这不是费用后 SOXX 超额；无完整 0bp 账户或独立前向。失败版不翻转符号或调整窗口。"
        f"Recorder `{rid}`，NAS SHA `{remote}` 经验证。\n")
    (card / "conclusion.md").write_text(conclusion)
    return summary
