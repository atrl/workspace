"""Fixed 09:46 peer breadth on funded joint cash/cap local value."""

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
    card = root / "research/stage1_0946_peer_breadth_joint_value_information_20261023"
    if (card / "run_state.json").exists() or (card / "summary.json").exists():
        raise ValueError("fixed 09:46 peer joint-value attempt already started")
    plan = json.loads((card / "plan.json").read_text())
    src = plan["source"]
    assert plan["frozen"]
    cards = {role: root / src[role + "_card"] for role in ("local_label", "training_reference", "peer")}
    for role in cards:
        assert _sha(cards[role] / "summary.json") == src[role + "_summary_sha256"]
        summary = json.loads((cards[role] / "summary.json").read_text())
        assert summary["recorder_id"] == src[role + "_recorder_id"]
        nas = summary.get("NAS_evidence") or summary.get("nas_evidence")
        assert nas["sha256"] == src[role + "_NAS_sha256"]
    for role, artifact in [
        ("local_label", "analysis/labels"),
        ("training_reference", "inputs/source_frame"),
        ("training_reference", "paths/reference_0"),
        ("peer", "analysis/daily"),
    ]:
        key = {"analysis/labels": "local_label_artifact_sha256",
               "inputs/source_frame": "training_frame_artifact_sha256",
               "paths/reference_0": "training_path_artifact_sha256",
               "analysis/daily": "peer_daily_artifact_sha256"}[artifact]
        local = cards[role] / "qlib_recorder" / src[role + "_experiment_id"] / src[role + "_recorder_id"] / "artifacts" / artifact
        assert _sha(local) == src[key]
    assert _sha(root / src["provider_uri"] / "qlib_store_manifest.json") == src["provider_manifest_sha256"]
    assert _sha(root / "src/analysis/seed_disagreement_information.py") == src["bootstrap_module_sha256"]
    parent_summary_path = root / "research/minute_information_pit_allocator_20261012/summary.json"
    assert _sha(parent_summary_path) == src["original39_parent_summary_sha256"]
    parent_summary = json.loads(parent_summary_path.read_text())
    base = list(parent_summary["input_audit"]["original_features"])
    assert len(base) == 39
    qlib.init(provider_uri=str((root / src["provider_uri"]).resolve()), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(start_time="2018-10-15", end_time="2022-12-30", freq="day"))
    assert len(calendar) == 1061
    with R.uri_context(str((cards["training_reference"] / "qlib_recorder").resolve())):
        rec = R.get_recorder(experiment_id=src["training_reference_experiment_id"],
                             recorder_id=src["training_reference_recorder_id"])
        frame = rec.load_object("inputs/source_frame")
        path = rec.load_object("paths/reference_0")
    with R.uri_context(str((cards["peer"] / "qlib_recorder").resolve())):
        peer = R.get_recorder(experiment_id=src["peer_experiment_id"],
                              recorder_id=src["peer_recorder_id"]).load_object("analysis/daily")
    with R.uri_context(str((cards["local_label"] / "qlib_recorder").resolve())):
        labels = R.get_recorder(experiment_id=src["local_label_experiment_id"],
                                recorder_id=src["local_label_recorder_id"]).load_object("analysis/labels")
    assert len(frame) == len(labels) == len(path["replay"]) == len(path["targets"]) == 1061

    def indexed(value: pd.DataFrame) -> pd.DataFrame:
        copy = value.copy()
        copy["trade_date"] = pd.to_datetime(copy.trade_date).dt.normalize()
        if copy.trade_date.duplicated().any():
            raise ValueError("duplicate trade date in 09:46 source")
        return copy.set_index("trade_date").reindex(calendar)

    frame = indexed(frame)
    replay = indexed(path["replay"])
    targets = indexed(path["targets"])
    labels = indexed(labels)
    peer = indexed(peer)
    assert frame.signal_date.notna().all() and replay.risk_state.notna().all()
    assert peer.source_known.fillna(False).all() and peer.peer_positive_fraction_3.notna().all()
    assert (pd.to_datetime(peer.assumed_available_at) < pd.to_datetime(peer.receiver_decision_at)).all()
    assert pd.to_datetime(peer.receiver_decision_at).equals(pd.to_datetime(labels.decision_at))
    assert set(base) <= set(frame.columns)
    state_names = ["portfolio_drawdown", "current_beta_cap", "risk_state",
                   "raw_beta", "projected_cash", "delta_target_beta",
                   "early_ret_soxx", "early_ret_soxl"]
    state = pd.DataFrame({
        "portfolio_drawdown": replay.portfolio_drawdown,
        "current_beta_cap": replay.current_beta_cap,
        "risk_state": replay.risk_state.astype(float),
        "raw_beta": 3.0 * frame.soxl_weight + frame.soxx_weight,
        "projected_cash": targets.cash_weight,
        "delta_target_beta": labels.delta_target_beta,
        "early_ret_soxx": frame.early_ret_soxx,
        "early_ret_soxl": frame.early_ret_soxl,
    }, index=calendar).astype(float)
    assert not np.isinf(state.to_numpy()).any()
    coordinate = peer.peer_positive_fraction_3.astype(float).rename("peer_positive_fraction_3")
    assert coordinate.between(0, 1).all()
    baseline_names = [*base, *state_names]
    all_features = frame[base].join(state).join(coordinate)
    train_dates = calendar[calendar < pd.Timestamp("2021-01-01")]
    corr = all_features.loc[train_dates, baseline_names + [coordinate.name]].corr(method="spearman")[coordinate.name]
    max_corr = float(corr.loc[baseline_names].abs().max())
    preflight = {"status": "source_clock_and_novelty_pass" if max_corr < plan["source_gate"]["train_max_abs_spearman_added_vs_baseline_below"] else "closed_before_fit_source_redundant",
                 "calendar_rows": len(calendar), "peer_known": int(coordinate.notna().sum()),
                 "max_abs_spearman_baseline": max_corr,
                 "source_Recorder_object_loads": 4, "future_label_values_used_for_selection": False,
                 "model_fits": 0, "native_accounts": 0, "plan_sha256": _sha(card / "plan.json")}
    (card / "preflight_result.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if preflight["status"] != "source_clock_and_novelty_pass":
        summary = {"experiment": card.name, "decision": preflight["status"],
                   "objective_achieved": False, "preflight": preflight,
                   "new_model_fits": 0, "new_native_accounts": 0,
                   "artifact_policy": {"policy": "slim_card_recorder_first",
                                       "workflow_config": "workflow.yaml", "recorder_only_artifacts": []}}
        (card / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (card / "workflow.yaml").write_text("experiment: " + card.name + "\nphase: source_redundant_no_fit\n")
        (card / "conclusion.md").write_text("# 09:46 同业广度联合价值：拟合前停止\n\n"
            f"来源与时钟核对完成，新增同业广度和原状态最高训练秩相关 {max_corr:.4f}，"
            "超过预定去重门；0拟合、0账户。\n")
        return summary
    receiver = plan["receiver"]
    available = pd.to_datetime(labels.label_available_at)
    train_cutoff = pd.Timestamp(receiver["train_label_available_before"]).tz_localize(None)
    eval_cutoff = pd.Timestamp(receiver["evaluation_label_available_before"]).tz_localize(None)
    y = labels.joint_local_fee_net_advantage.astype(float)
    eligible = labels.eligible.astype(bool) & y.notna()
    train = eligible & (calendar >= pd.Timestamp(receiver["train_trade_dates"][0])) & (
        calendar <= pd.Timestamp(receiver["train_trade_dates"][1])) & (available < train_cutoff)
    evaluation = eligible & (calendar >= pd.Timestamp(receiver["evaluation_trade_dates"][0])) & (
        calendar <= pd.Timestamp(receiver["evaluation_trade_dates"][1])) & (available < eval_cutoff)
    assert int(train.sum()) == receiver["expected_train_mature_eligible"]
    assert int(evaluation.sum()) == receiver["expected_eval_mature_eligible"]
    assert int(evaluation[calendar.year == 2021].sum()) >= receiver["minimum_eval_2021"]
    assert int(evaluation[calendar.year == 2022].sum()) >= receiver["minimum_eval_2022"]
    static = pd.concat({"feature": all_features,
                        "label": pd.DataFrame({"joint_local_fee_net_advantage": y}),
                        "meta": pd.DataFrame({"train_mature": train, "eval_mature": evaluation})}, axis=1)
    static.index = pd.MultiIndex.from_arrays(
        [calendar, ["SOXL_SOXX_CASH"] * len(calendar)], names=["datetime", "instrument"])
    dataset = DatasetH(handler=DataHandlerLP.from_df(static),
        segments={"train": receiver["train_trade_dates"], "eval": receiver["evaluation_trade_dates"]})
    training = dataset.prepare("train", col_set=["feature", "label", "meta"])
    held = dataset.prepare("eval", col_set=["feature", "label", "meta"])
    training = training.loc[training[("meta", "train_mature")].astype(bool)]
    held = held.loc[held[("meta", "eval_mature")].astype(bool)]
    assert len(training) == receiver["expected_train_mature_eligible"]
    assert len(held) == receiver["expected_eval_mature_eligible"]
    xtr, xhe = training["feature"], held["feature"]
    ytr = training[("label", "joint_local_fee_net_advantage")].to_numpy(float)
    yhe = held[("label", "joint_local_fee_net_advantage")].to_numpy(float)
    assert np.isfinite(ytr).all() and np.isfinite(yhe).all()
    run_state = {"status": "fit_attempted", "plan_sha256": preflight["plan_sha256"],
                 "model_fits": 0, "bootstrap_attempts": 0, "native_accounts": 0}

    def save_state() -> None:
        (card / "run_state.json").write_text(json.dumps(run_state, indent=2) + "\n")

    save_state()

    def model():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))

    baseline = model()
    candidate = model()
    candidate_names = baseline_names + [coordinate.name]
    baseline.fit(xtr[baseline_names], ytr)
    run_state["model_fits"] = 1
    save_state()
    candidate.fit(xtr[candidate_names], ytr)
    run_state["model_fits"] = 2
    save_state()
    p0, p1 = baseline.predict(xhe[baseline_names]), candidate.predict(xhe[candidate_names])
    delta = (yhe - p0) ** 2 - (yhe - p1) ** 2
    mean_delta = (yhe - np.mean(ytr)) ** 2 - (yhe - p1) ** 2
    score = pd.DataFrame({"trade_date": pd.DatetimeIndex(held.index.get_level_values("datetime")),
                          "target": yhe, "baseline": p0, "candidate": p1,
                          "paired_improvement": delta, "improvement_vs_train_mean": mean_delta})
    assert score.trade_date.is_monotonic_increasing
    by_year = {str(year): float(score.loc[score.trade_date.dt.year.eq(year), "paired_improvement"].mean())
               for year in (2021, 2022)}
    rank = {str(year): float(score.loc[score.trade_date.dt.year.eq(year), ["candidate", "target"]].corr(method="spearman").iloc[0, 1])
            for year in (2021, 2022)}
    metrics = {"standardized_peer_coefficient": float(candidate.named_steps["ridge"].coef_[-1]),
               "MSE_baseline": float(np.mean((yhe - p0) ** 2)),
               "MSE_candidate": float(np.mean((yhe - p1) ** 2)),
               "MSE_train_mean": float(np.mean((yhe - np.mean(ytr)) ** 2)),
               "paired_improvement": float(np.mean(delta)),
               "improvement_vs_train_mean": float(np.mean(mean_delta)),
               "annual_paired_improvement": by_year,
               "annual_candidate_rank_spearman": rank}
    full_calendar = calendar[(calendar >= pd.Timestamp("2021-01-01")) & (calendar <= pd.Timestamp("2022-12-30"))]
    assert len(full_calendar) == 503
    full = pd.Series(np.nan, index=full_calendar, dtype=float)
    full.loc[score.trade_date.to_numpy()] = delta
    assert int(full.notna().sum()) == len(score)
    fixed = plan["information_gate"]["paired_full_503_day_calendar_stationary_block_lower_positive"]
    bootstrap, _, means = full_calendar_bootstrap(full.to_numpy(float),
        mean_block=fixed["mean_block"], resamples=fixed["resamples"],
        seed=fixed["seed"], alpha=fixed["one_sided_alpha"])
    run_state["bootstrap_attempts"] = 1
    save_state()
    gates = {"positive_coefficient": metrics["standardized_peer_coefficient"] > 0,
             "beats_baseline": metrics["paired_improvement"] > 0,
             "beats_train_mean": metrics["improvement_vs_train_mean"] > 0,
             "both_eval_years": all(value > 0 for value in by_year.values()),
             "both_year_ranks": all(np.isfinite(value) and value > 0 for value in rank.values()),
             "positive_block_lower": bootstrap["lower_bound"] > 0}
    decision = ("conditional_local_value_information_pass_source_PIT_needed_before_f696_account" if all(gates.values())
                else "not_promoted_fixed_0946_peer_breadth_joint_value_information_failed")
    with R.start(experiment_name=card.name, uri=str((card / "qlib_recorder").resolve())):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, preflight=preflight, dataset=dataset)
        recorder.save_objects(artifact_path="models", baseline=baseline, candidate=candidate)
        recorder.save_objects(artifact_path="predictions", held_scored=score)
        recorder.save_objects(artifact_path="analysis", full_eval_calendar=full,
                              metrics=metrics, gates=gates, bootstrap=bootstrap,
                              bootstrap_replicate_means=means)
        R.log_params(feature="peer_positive_fraction_3", receiver="joint_cash_cap_0946_fee_net_local_value",
                     parent="original_architecture_OOF_then_f696_only_if_qualified")
        R.log_metrics(eval_paired_MSE_improvement=metrics["paired_improvement"],
                      block_lower=bootstrap["lower_bound"])
        rid, eid = recorder.id, str(recorder.experiment_id)
    run_state.update(status="finished", recorder_id=rid, experiment_id=eid)
    save_state()
    summary = {"experiment": card.name, "decision": decision,
               "promotion": "not_promoted_information_only", "objective_achieved": False,
               "plan_sha256": _sha(card / "plan.json"),
               "source_recorder_ids": {role: src[role + "_recorder_id"] for role in cards},
               "recorder_id": rid, "experiment_id": eid,
               "preflight": preflight, "train_mature": len(training), "eval_mature": len(held),
               "metrics": metrics, "bootstrap": bootstrap, "gates": gates,
               "model_fits": 2, "native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "models/*", "predictions/*", "analysis/*"]},
               "limitations": ["The OOF training scaffold is not the exact 2023+ f696 own-state policy.",
                               "09:45 bar completion is an assumed research clock; original historical first receipt is unqualified.",
                               "One-step local fee-net value is not a continuous funded SOXX account or independent forward evidence."]}
    binding = json.loads((cards["local_label"] / "nas_binding.json").read_text())
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
        "phase: completed_conditional_local_value_information_screen\n")
    (card / "conclusion.md").write_text(
        "# Stage1 09:46同业广度：联合现金/风险上限局部净价值信息门\n\n"
        "在原架构2018–2022 OOF 0bp Tiered训练账本上，同起点、次日共同目标复位的现金转SOXX"
        "局部净价值标签已冻结。原39项、当时风险状态/目标和同日SOXX/SOXL早盘收益为对照，"
        "只新增09:45 AMD/NVDA/SMH同涨比例；来源首收仍未核证。\n\n"
        f"训练/评价成熟可行动 {len(training)}/{len(held)}，最高对照训练秩相关 {max_corr:.4f}；"
        f"2拟合、1固定区块检验、0完整账户。新增系数 {metrics['standardized_peer_coefficient']:.9f}"
        "（预定正号）。"
        f"候选/对照/训练均值MSE为 {metrics['MSE_candidate']:.10f}/"
        f"{metrics['MSE_baseline']:.10f}/{metrics['MSE_train_mean']:.10f}。"
        f"配对改善 {metrics['paired_improvement']:.10f}，修正区块下界 {bootstrap['lower_bound']:.10f}；"
        f"2021/2022改善 {by_year['2021']:.10f}/{by_year['2022']:.10f}，"
        f"候选排序相关 {rank['2021']:.4f}/{rank['2022']:.4f}。"
        f"六项门 {sum(gates.values())}/6，决定 `{decision}`。\n\n"
        "局部金额不代表完整f696候选账户，失败不改同业篮子/符号/仓位幅度或年度救援；"
        "通过也须历史来源时点和独立完整0bp Tiered自有资金账户。"
        f"Recorder `{rid}`，NAS SHA `{remote}`。\n")
    return summary
