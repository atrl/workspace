"""One frozen 4-of-5 broad rebound interaction screen for Stage1 cash H5."""

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
    card = root / "research/stage1_broad_rebound_confirmation_cash_h5_20261023"
    if (card / "run_state.json").exists() or (card / "summary.json").exists():
        raise ValueError("fixed broad-rebound screen already attempted")
    plan = json.loads((card / "plan.json").read_text())
    source = plan["source"]
    assert plan["frozen"]
    parent_card = root / source["parent_card"]
    assert _sha(parent_card / "summary.json") == source["parent_summary_sha256"]
    assert _sha(root / source["provider_uri"] / "qlib_store_manifest.json") == source["provider_manifest_sha256"]
    assert _sha(root / "src/analysis/seed_disagreement_information.py") == source["bootstrap_module_sha256"]
    parent_summary = json.loads((parent_card / "summary.json").read_text())
    assert parent_summary["recorder_id"] == source["parent_recorder_id"]
    assert parent_summary["nas_evidence"]["sha256"] == source["parent_NAS_sha256"]
    qlib.init(provider_uri=str((root / source["provider_uri"]).resolve()), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(start_time="2011-01-03", end_time="2026-10-09", freq="day"))
    close = D.features(["SOXX"], ["$close"], start_time="2011-01-03",
                       end_time="2026-10-09", freq="day")["$close"].droplevel("instrument")
    assert calendar.equals(close.index) and close.notna().all()
    with R.uri_context(str((parent_card / "qlib_recorder").resolve())):
        recorder = R.get_recorder(experiment_id=source["parent_experiment_id"],
                                  recorder_id=source["parent_recorder_id"])
        features = recorder.load_object("inputs/features")
    pd.testing.assert_index_equal(features.index, calendar, check_names=False)
    base = list(parent_summary["input_audit"]["original_features"])
    assert len(base) == 39 and set(base) <= set(features.columns)
    a = pd.to_numeric(features.soxx_ret5, errors="coerce")
    b = pd.to_numeric(features.cross_breadth_pos_20, errors="coerce")
    known = a.notna() & b.notna()
    indicator = pd.Series(np.where(known, (a.gt(0) & b.ge(.8)).astype(float), np.nan),
                          index=features.index, name=plan["feature"]["name"])
    train_source = indicator.loc["2011-01-03":"2022-12-30"]
    held_source = indicator.loc["2023-01-03":"2026-10-09"]
    corr = features.loc[train_source.index, base].assign(**{indicator.name: train_source}).corr(method="spearman")[indicator.name]
    max_corr = float(corr.loc[base].abs().max())
    fixed_prefit = plan["prefit_gates"]
    prefit_gates = {
        "train_positive_support": int(train_source.eq(1).sum()) >= fixed_prefit["train_indicator_positive_min"],
        "held_positive_support": int(held_source.eq(1).sum()) >= fixed_prefit["held_indicator_positive_min"],
        "full_calendar_known": bool(indicator.notna().all()),
        "nonredundant_to_original39": max_corr < fixed_prefit["train_max_abs_spearman_original39_below"],
    }
    preflight = {"status": "source_novelty_pass" if all(prefit_gates.values()) else "closed_before_future_label_read",
                 "calendar_rows": len(calendar), "train_rows": len(train_source),
                 "train_positive": int(train_source.eq(1).sum()),
                 "held_rows": len(held_source), "held_positive": int(held_source.eq(1).sum()),
                 "max_abs_spearman_original39": max_corr, "gates": prefit_gates,
                 "future_label_reads": 0, "model_fits": 0, "accounts": 0,
                 "plan_sha256": _sha(card / "plan.json")}
    (card / "preflight_result.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if not all(prefit_gates.values()):
        summary = {"experiment": card.name, "decision": "closed_before_fit_broad_rebound_source_or_novelty_failed",
                   "objective_achieved": False, "preflight": preflight, "new_model_fits": 0,
                   "new_native_accounts": 0, "artifact_policy": {"policy": "slim_card_recorder_first",
                   "workflow_config": "workflow.yaml", "recorder_only_artifacts": []}}
        (card / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (card / "workflow.yaml").write_text("experiment: " + card.name + "\nphase: closed_prefit_no_fit\n")
        (card / "conclusion.md").write_text("# Stage1 广泛反弹确认：拟合前关闭\n\n"
            + json.dumps(prefit_gates, ensure_ascii=False) + "。0未来标签读取、0拟合、0账户。\n")
        return summary
    state = {"status": "preflight_passed", "plan_sha256": preflight["plan_sha256"],
             "model_fits": 0, "bootstrap_attempts": 0, "native_accounts": 0}

    def save_state() -> None:
        (card / "run_state.json").write_text(json.dumps(state, indent=2) + "\n")

    save_state()
    with R.uri_context(str((parent_card / "qlib_recorder").resolve())):
        parent = R.get_recorder(experiment_id=source["parent_experiment_id"],
                                recorder_id=source["parent_recorder_id"])
        frame = parent.load_object("inputs/daily_frame")
        timing = parent.load_object("inputs/timing")
    assert frame.index.equals(features.index) and timing.index.equals(features.index)
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
    assert int(train_mature.sum()) == receiver["expected_train_mature"]
    assert int(held_mature.sum()) == receiver["expected_held_mature"]
    static = pd.concat({
        "feature": features[base].assign(**{indicator.name: indicator}),
        "label": pd.DataFrame({"cash_advantage_H5": label}),
        "meta": pd.DataFrame({"train_mature": train_mature, "held_mature": held_mature}),
    }, axis=1)
    static.index = pd.MultiIndex.from_arrays(
        [frame.index, ["SOXL_SOXX_CASH"] * len(frame)], names=["datetime", "instrument"])
    dataset = DatasetH(handler=DataHandlerLP.from_df(static),
        segments={"train": receiver["train_signals"], "held": receiver["reused_held_signals"]})
    training = dataset.prepare("train", col_set=["feature", "label", "meta"])
    evaluation = dataset.prepare("held", col_set=["feature", "label", "meta"])
    training = training.loc[training[("meta", "train_mature")].astype(bool)]
    evaluation = evaluation.loc[evaluation[("meta", "held_mature")].astype(bool)]
    assert len(training) == 3014 and len(evaluation) == 940
    xtr, xhe = training["feature"], evaluation["feature"]
    ytr = training[("label", "cash_advantage_H5")].to_numpy(float)
    yhe = evaluation[("label", "cash_advantage_H5")].to_numpy(float)
    assert np.isfinite(ytr).all() and np.isfinite(yhe).all()

    def model():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))

    names = base + [indicator.name]
    baseline = model()
    candidate = model()
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
    mean_delta = (yhe - np.mean(ytr)) ** 2 - (yhe - p1) ** 2
    score = pd.DataFrame({"signal_date": pd.DatetimeIndex(evaluation.index.get_level_values("datetime")),
                          "target": yhe, "baseline": p0, "candidate": p1,
                          "paired_improvement": delta, "improvement_vs_train_mean": mean_delta})
    assert score.signal_date.is_monotonic_increasing
    half = len(score) // 2
    metrics = {"standardized_indicator_coefficient": float(candidate.named_steps["ridge"].coef_[-1]),
               "MSE_baseline": float(np.mean((yhe - p0) ** 2)),
               "MSE_candidate": float(np.mean((yhe - p1) ** 2)),
               "MSE_train_mean": float(np.mean((yhe - np.mean(ytr)) ** 2)),
               "improvement_vs_baseline": float(np.mean(delta)),
               "improvement_vs_train_mean": float(np.mean(mean_delta)),
               "first_half_improvement": float(np.mean(delta[:half])),
               "second_half_improvement": float(np.mean(delta[half:])),
               "2024_improvement": float(score.loc[score.signal_date.dt.year.eq(2024), "paired_improvement"].mean()),
               "2025_improvement": float(score.loc[score.signal_date.dt.year.eq(2025), "paired_improvement"].mean())}
    full = pd.Series(np.nan, index=frame.loc["2023-01-03":"2026-10-07"].index, dtype=float)
    full.loc[score.signal_date.to_numpy()] = delta
    fixed_block = plan["information_gate"]["paired_full_calendar_stationary_block_lower_positive"]
    assert len(full) == fixed_block["calendar_rows"] and int(full.notna().sum()) == 940
    bootstrap, _, means = full_calendar_bootstrap(full.to_numpy(float),
        mean_block=fixed_block["mean_block"], resamples=fixed_block["resamples"],
        seed=fixed_block["seed"], alpha=fixed_block["one_sided_alpha"])
    state["bootstrap_attempts"] = 1
    save_state()
    gates = {"negative_coefficient": metrics["standardized_indicator_coefficient"] < 0,
             "beats_original39": metrics["improvement_vs_baseline"] > 0,
             "beats_train_mean": metrics["improvement_vs_train_mean"] > 0,
             "both_halves": metrics["first_half_improvement"] > 0 and metrics["second_half_improvement"] > 0,
             "both_complete_years": metrics["2024_improvement"] > 0 and metrics["2025_improvement"] > 0,
             "positive_block_lower": bootstrap["lower_bound"] > 0}
    decision = ("information_gate_passed_requires_f696_0bp_account" if all(gates.values())
                else "not_promoted_fixed_broad_rebound_confirmation_cash_H5_failed")
    with R.start(experiment_name=card.name, uri=str((card / "qlib_recorder").resolve())):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, preflight=preflight, dataset=dataset)
        recorder.save_objects(artifact_path="models", baseline=baseline, candidate=candidate)
        recorder.save_objects(artifact_path="predictions", held_scored=score)
        recorder.save_objects(artifact_path="analysis", full_held_calendar=full,
                              metrics=metrics, gates=gates, bootstrap=bootstrap,
                              bootstrap_replicate_means=means, indicator=indicator)
        R.log_params(feature=indicator.name, parent="f696_A0", news_data=False)
        R.log_metrics(held_MSE_improvement=metrics["improvement_vs_baseline"],
                      block_lower=bootstrap["lower_bound"])
        rid, eid = recorder.id, str(recorder.experiment_id)
    state.update(status="finished", recorder_id=rid, experiment_id=eid)
    save_state()
    summary = {"experiment": card.name, "decision": decision,
               "promotion": "not_promoted_information_only", "objective_achieved": False,
               "plan_sha256": _sha(card / "plan.json"), "parent_recorder_id": source["parent_recorder_id"],
               "recorder_id": rid, "experiment_id": eid,
               "preflight": preflight, "metrics": metrics, "bootstrap": bootstrap,
               "gates": gates, "model_fits": 2, "native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "models/*", "predictions/*", "analysis/*"]},
               "limitations": ["The feature is an explicit interaction of existing original39 inputs, not new raw market information.",
                               "2023-2026 is reused development history, not independent forward evidence.",
                               "H5 price loss is not funded Tiered account value."]}
    binding = json.loads((parent_card / "nas_binding.json").read_text())
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
        "phase: completed_existing_source_interaction_H5_screen\n")
    (card / "conclusion.md").write_text(
        "# Stage1 广泛反弹确认：固定现金优势 H5 信息检验\n\n"
        "原 f696 A0 完整链为母本。只从已有原39项构造 SOXX五日收益为正且五个跨资产"
        "底层ETF中至少四个20日收益为正的固定联合指示器；信号日收盘后、次交易日09:29前可用。"
        "这不是新原始行情或事后挑选的区间开关。\n\n"
        f"训练/评价指标为 {preflight['train_positive']}/{len(train_source)} 与 "
        f"{preflight['held_positive']}/{len(held_source)}，最高原39项秩相关 {max_corr:.4f}。"
        f"两次匹配Ridge、一次固定区块检验、0资金账户。新增系数 "
        f"{metrics['standardized_indicator_coefficient']:.8f}（预定负号）。"
        f"候选/原39/训练均值MSE为 {metrics['MSE_candidate']:.9f}/"
        f"{metrics['MSE_baseline']:.9f}/{metrics['MSE_train_mean']:.9f}；"
        f"对照改善 {metrics['improvement_vs_baseline']:.9f}，区块下界 {bootstrap['lower_bound']:.9f}。"
        f"前/后半改善 {metrics['first_half_improvement']:.9f}/{metrics['second_half_improvement']:.9f}，"
        f"完整2024/2025为 {metrics['2024_improvement']:.9f}/{metrics['2025_improvement']:.9f}。"
        f"六项信息门 {sum(gates.values())}/6，决定 `{decision}`。\n\n"
        "信息误差不能替代SOXX费用后超额。失败版不换篮子、门槛、期限或年份；"
        "只有全部通过才可能另冻完整f696自身状态0bp Tiered账户、再看2/5bp与独立前向。"
        f"Recorder `{rid}`，NAS SHA `{remote}`。\n")
    return summary
