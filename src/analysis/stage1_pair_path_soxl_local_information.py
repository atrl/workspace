"""Frozen 09:46 paired ETF minute-path information screen for legal SOXL deployment."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path) -> dict:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
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
    from src.data.nas_market_data import _settings, archive_nas_recorder, materialize_nas_evidence

    root = Path(root).resolve()
    card = root / "research/stage1_0946_pair_path_soxl_local_information_20261025"
    if (card / "run_state.json").exists() or (card / "summary.json").exists():
        raise ValueError("fixed pair-path information attempt already started")
    plan = json.loads((card / "plan.json").read_text())
    assert plan["frozen"]
    src = plan["source"]
    cards = {role: root / src[role + "_card"] for role in
             ("pair_source", "local_label", "legality_audit", "training_reference")}
    for role, path in cards.items():
        assert _sha(path / "summary.json") == src[role + "_summary_sha256"]
    pair_summary = json.loads((cards["pair_source"] / "summary.json").read_text())
    assert pair_summary["decision"] == "source_and_novelty_pass_needs_separate_0946_local_value_information_and_first_receipt"
    pair_result = json.loads((cards["pair_source"] / "result.json").read_text())
    assert pair_result["coverage"]["held_known"] >= plan["source_gate"]["held_f696_known_min"]
    assert pair_result["coverage"]["train_unknown_trade_dates"] == ["2020-03-16"]
    legal_audit = json.loads((cards["legality_audit"] / "result.json").read_text())
    assert legal_audit["legal_mature"] == 540
    for role in ("local_label", "training_reference"):
        summary = json.loads((cards[role] / "summary.json").read_text())
        assert summary["recorder_id"] == src[role + "_recorder_id"]
        nas = summary.get("nas_evidence") or summary.get("NAS_evidence")
        assert nas["sha256"] == src[role + "_nas_sha256"]
    for role, artifact, key in [
        ("local_label", "analysis/labels", "local_label_artifact_sha256"),
        ("training_reference", "inputs/source_frame", "training_frame_artifact_sha256"),
        ("training_reference", "paths/reference_0", "training_path_artifact_sha256"),
    ]:
        local = (cards[role] / "qlib_recorder" / src[role + "_experiment_id"] /
                 src[role + "_recorder_id"] / "artifacts" / artifact)
        assert _sha(local) == src[key]
    assert _sha(root / src["minute_provider"] / "qlib_store_manifest.json") == src["minute_provider_manifest_sha256"]
    assert _sha(root / "research/minute_information_pit_allocator_20261012/summary.json") == src["original39_parent_summary_sha256"]
    assert _sha(root / "src/analysis/seed_disagreement_information.py") == src["bootstrap_module_sha256"]
    base = json.loads((root / "research/minute_information_pit_allocator_20261012/summary.json").read_text())["input_audit"]["original_features"]
    assert len(base) == 39
    qlib.init(provider_uri=str((root / src["minute_provider"]).resolve()), region="us", kernels=1,
              expression_cache=None, dataset_cache=None)
    with R.uri_context(str((cards["training_reference"] / "qlib_recorder").resolve())):
        ref = R.get_recorder(experiment_id=src["training_reference_experiment_id"],
                             recorder_id=src["training_reference_recorder_id"])
        assert ref.status == "FINISHED"
        frame = ref.load_object("inputs/source_frame")
        path = ref.load_object("paths/reference_0")
    with R.uri_context(str((cards["local_label"] / "qlib_recorder").resolve())):
        lab_rec = R.get_recorder(experiment_id=src["local_label_experiment_id"],
                                 recorder_id=src["local_label_recorder_id"])
        assert lab_rec.status == "FINISHED"
        labels = lab_rec.load_object("analysis/labels")
    assert len(frame) == len(path["replay"]) == len(path["targets"]) == len(labels) == 1061

    def indexed(value: pd.DataFrame) -> pd.DataFrame:
        copy = value.copy()
        copy["trade_date"] = pd.to_datetime(copy.trade_date).dt.normalize()
        assert copy.trade_date.is_unique and copy.trade_date.is_monotonic_increasing
        return copy.set_index("trade_date")

    frame, replay, targets, labels = map(indexed, (frame, path["replay"], path["targets"], labels))
    calendar = pd.DatetimeIndex(frame.index)
    assert calendar.equals(replay.index) and calendar.equals(targets.index) and calendar.equals(labels.index)
    assert len(calendar) == 1061 and calendar[0] == pd.Timestamp("2018-10-15") and calendar[-1] == pd.Timestamp("2022-12-30")
    bars = D.features(["SOXL", "SOXX"], ["$open", "$close", "$price_valid"],
                      start_time="2018-10-15 09:31:00", end_time="2022-12-30 09:45:00", freq="1min").reset_index()
    assert {"instrument", "datetime", "$open", "$close", "$price_valid"}.issubset(bars)
    bars["datetime"] = pd.to_datetime(bars.datetime)
    clock = bars.datetime.dt.strftime("%H:%M")
    opening = bars.loc[clock.between("09:31", "09:45")].copy()
    opening["day"] = opening.datetime.dt.normalize()
    energy = {}
    invalid = []
    for day, group in opening.groupby("day", sort=True):
        if day not in calendar:
            continue
        expected = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=15, freq="min")
        parts = {name: group.loc[group.instrument.eq(name)].sort_values("datetime") for name in ("SOXL", "SOXX")}
        valid = True
        for part in parts.values():
            valid &= len(part) == 15 and pd.DatetimeIndex(part.datetime).equals(expected)
            if len(part) == 15:
                values = part[["$open", "$close", "$price_valid"]].to_numpy(float)
                valid &= bool(np.isfinite(values).all() and (values[:, :2] > 0).all() and (values[:, 2] == 1).all())
        if not valid:
            invalid.append(str(day.date()))
            continue
        soxl, soxx = parts["SOXL"], parts["SOXX"]
        r_l = np.log(soxl["$close"].to_numpy(float) / soxl["$open"].to_numpy(float))
        r_x = np.log(soxx["$close"].to_numpy(float) / soxx["$open"].to_numpy(float))
        energy[day] = float(np.square(r_l - 3 * r_x).sum() /
                            (np.square(r_l).sum() + 9 * np.square(r_x).sum() + 1e-12))
    coordinate = pd.Series(energy, name="pair_path_divergence", dtype=float).reindex(calendar)
    assert invalid == ["2020-03-16"] and int(coordinate.notna().sum()) == 1060
    assert coordinate.dropna().between(0, 2.01).all()
    assert (pd.to_datetime(labels.decision_at).dt.hour == 9).all()
    state_names = ["portfolio_drawdown", "current_beta_cap", "risk_state", "raw_beta",
                   "projected_cash", "delta_target_beta", "early_ret_soxx", "early_ret_soxl"]
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
    baseline_names = [*base, *state_names]
    all_features = frame[base].join(state).join(coordinate)
    train_dates = calendar[calendar < pd.Timestamp("2021-01-01")]
    max_corr = float(all_features.loc[train_dates, baseline_names + [coordinate.name]].corr(method="spearman")
                     [coordinate.name].loc[baseline_names].abs().max())
    preflight = {"status": "source_clock_and_novelty_pass" if max_corr < plan["source_gate"]["train_max_abs_spearman_added_vs_baseline_below"] else "closed_before_fit_source_redundant",
                 "source_known": int(coordinate.notna().sum()), "unknown_dates": [str(x.date()) for x in coordinate[coordinate.isna()].index],
                 "max_abs_spearman_baseline": max_corr, "plan_sha256": _sha(card / "plan.json"),
                 "future_label_values_used_for_source_selection": False, "model_fits": 0, "native_accounts": 0}
    (card / "preflight_result.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if preflight["status"] != "source_clock_and_novelty_pass":
        summary = {"experiment": card.name, "decision": preflight["status"], "objective_achieved": False,
                   "plan_sha256": preflight["plan_sha256"], "preflight": preflight, "model_fits": 0,
                   "native_accounts": 0, "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml", "recorder_only_artifacts": []}}
        (card / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    receiver = plan["receiver"]
    y = labels.soxl_local_fee_net_advantage.astype(float)
    legal = labels.eligible.astype(bool) & y.notna() & (
        targets.soxl_weight.astype(float) + labels.delta_SOXL_weight.astype(float) <= 0.30 + 1e-10)
    assert int(legal.sum()) == 540
    available = pd.to_datetime(labels.label_available_at)
    train_cutoff = pd.Timestamp(receiver["train_label_available_before"]).tz_localize(None)
    eval_cutoff = pd.Timestamp(receiver["evaluation_label_available_before"]).tz_localize(None)
    train = legal & coordinate.notna() & (calendar <= pd.Timestamp("2020-12-31")) & (available < train_cutoff)
    evaluation = legal & coordinate.notna() & (calendar >= pd.Timestamp("2021-01-01")) & (available < eval_cutoff)
    assert int(train.sum()) >= receiver["minimum_train_mature_legal_source_known"]
    assert int(evaluation.sum()) >= receiver["minimum_eval_mature_legal_source_known"]
    assert int(evaluation[calendar.year == 2021].sum()) >= receiver["minimum_eval_2021"]
    assert int(evaluation[calendar.year == 2022].sum()) >= receiver["minimum_eval_2022"]
    static = pd.concat({"feature": all_features,
                        "label": pd.DataFrame({"soxl_local_fee_net_advantage": y}),
                        "meta": pd.DataFrame({"train_mature": train, "eval_mature": evaluation})}, axis=1)
    static.index = pd.MultiIndex.from_arrays([calendar, ["SOXL_SOXX_CASH"] * len(calendar)], names=["datetime", "instrument"])
    dataset = DatasetH(handler=DataHandlerLP.from_df(static), segments={"train": receiver["train_trade_dates"],
                                                           "eval": receiver["evaluation_trade_dates"]})
    training = dataset.prepare("train", col_set=["feature", "label", "meta"])
    held = dataset.prepare("eval", col_set=["feature", "label", "meta"])
    training = training.loc[training[("meta", "train_mature")].astype(bool)]
    held = held.loc[held[("meta", "eval_mature")].astype(bool)]
    assert len(training) == int(train.sum()) and len(held) == int(evaluation.sum())
    xtr, xhe = training["feature"], held["feature"]
    ytr = training[("label", "soxl_local_fee_net_advantage")].to_numpy(float)
    yhe = held[("label", "soxl_local_fee_net_advantage")].to_numpy(float)
    assert np.isfinite(ytr).all() and np.isfinite(yhe).all()
    state_path = card / "run_state.json"
    run_state = {"status": "fit_attempted", "plan_sha256": preflight["plan_sha256"],
                 "model_fits": 0, "bootstrap_attempts": 0, "native_accounts": 0}
    state_path.write_text(json.dumps(run_state, indent=2) + "\n")

    def save_state() -> None:
        state_path.write_text(json.dumps(run_state, indent=2) + "\n")

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
    by_year = {str(year): float(score.loc[score.trade_date.dt.year.eq(year), "paired_improvement"].mean())
               for year in (2021, 2022)}
    rank = {str(year): float(score.loc[score.trade_date.dt.year.eq(year), ["candidate", "target"]].corr(method="spearman").iloc[0, 1])
            for year in (2021, 2022)}
    metrics = {"standardized_pair_path_coefficient": float(candidate.named_steps["ridge"].coef_[-1]),
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
    fixed = plan["information_gate"]["paired_full_503_day_calendar_stationary_block_lower_positive"]
    bootstrap, _, means = full_calendar_bootstrap(full.to_numpy(float), mean_block=fixed["mean_block"],
        resamples=fixed["resamples"], seed=fixed["seed"], alpha=fixed["one_sided_alpha"])
    run_state["bootstrap_attempts"] = 1
    save_state()
    gates = {"negative_coefficient": metrics["standardized_pair_path_coefficient"] < 0,
             "beats_baseline": metrics["paired_improvement"] > 0,
             "beats_train_mean": metrics["improvement_vs_train_mean"] > 0,
             "both_eval_years": all(value > 0 for value in by_year.values()),
             "both_year_ranks": all(np.isfinite(value) and value > 0 for value in rank.values()),
             "positive_block_lower": bootstrap["lower_bound"] > 0}
    decision = ("conditional_SOXL_local_information_pass_source_PIT_needed_before_f696_account" if all(gates.values())
                else "not_promoted_fixed_pair_path_SOXL_local_information_failed")
    with R.start(experiment_name=card.name, uri=str((card / "qlib_recorder").resolve())):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, preflight=preflight, dataset=dataset)
        recorder.save_objects(artifact_path="models", baseline=baseline, candidate=candidate)
        recorder.save_objects(artifact_path="predictions", held_scored=score)
        recorder.save_objects(artifact_path="analysis", full_eval_calendar=full,
                              metrics=metrics, gates=gates, bootstrap=bootstrap,
                              bootstrap_replicate_means=means)
        R.log_params(feature="pair_path_divergence", receiver="legal_SOXL_cash_0946_fee_net_local_value",
                     parent="original_architecture_OOF_then_f696_only_if_qualified")
        R.log_metrics(eval_paired_MSE_improvement=metrics["paired_improvement"],
                      block_lower=bootstrap["lower_bound"])
        rid, eid = recorder.id, str(recorder.experiment_id)
    run_state.update(status="finished", recorder_id=rid, experiment_id=eid)
    save_state()
    summary = {"experiment": card.name, "decision": decision, "promotion": "not_promoted_information_only",
               "objective_achieved": False, "plan_sha256": _sha(card / "plan.json"),
               "source_recorder_ids": {role: src[role + "_recorder_id"] for role in ("local_label", "training_reference")},
               "recorder_id": rid, "experiment_id": eid, "preflight": preflight,
               "train_mature_legal": len(training), "eval_mature_legal": len(held),
               "metrics": metrics, "bootstrap": bootstrap, "gates": gates,
               "model_fits": 2, "native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "models/*", "predictions/*", "analysis/*"]},
               "limitations": ["Original-architecture OOF local-value scaffold is not exact f696 own-state policy.",
                               "09:45 completion is an assumed research clock; historical first receipt is unqualified.",
                               "One-step local fee-net value is not a continuous funded SOXX account."]}
    binding = {"schema_version": "1", "authority": "yanxuan_nas",
               "source_id": card.name, "as_of": "2026-10-25", "source_plan": "plan.json"}
    (card / "nas_binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    archived = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
                                    card=card, recorder_id=rid, experiment_id=eid)
    with tempfile.TemporaryDirectory(prefix="stage1-pair-path-soxl-restore-") as temporary:
        temporary = Path(temporary)
        manifest = temporary / "manifest.json"
        manifest.write_text(json.dumps({"evidence": archived, "ssh": _settings(root)["ssh"]}))
        restored = temporary / "card"
        restored.mkdir()
        materialize_nas_evidence(manifest, card=restored)
        with R.uri_context(str(restored / "qlib_recorder")):
            cold = R.get_recorder(experiment_id=eid, recorder_id=rid)
            assert cold.status == "FINISHED"
            assert cold.load_object("analysis/metrics") == metrics
            assert cold.load_object("analysis/gates") == gates
    summary["NAS_evidence"] = archived
    summary["NAS_readback"] = {"verified": True, "archive_sha256": archived["sha256"],
                               "scoped_metrics_and_gates_equal": True}
    (card / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (card / "conclusion.md").write_text(
        "# Stage1 09:46 双 ETF 分钟路径分歧：合法 SOXL 局部净价值信息门\n\n"
        "在原架构2018–2022 OOF 0bp Tiered训练账本上，保留全部1061日、仅用540个合法成熟现金转SOXL标签；"
        "原39项、账户风险/目标与同日早盘收益为对照。09:45完成的双ETF路径分歧能否于09:46送达仅按研究时钟假定，历史首次接收未证实。\n\n"
        f"训练/评价成熟合法且来源已知 {len(training)}/{len(held)}，新增值与对照最高训练秩相关 {max_corr:.4f}；"
        f"2拟合、1固定区块检验、0候选资金账户。新增标准化系数 {metrics['standardized_pair_path_coefficient']:.9f}（预定负号）。"
        f"候选/对照/训练均值MSE {metrics['MSE_candidate']:.10f}/{metrics['MSE_baseline']:.10f}/"
        f"{metrics['MSE_train_mean']:.10f}；配对改善 {metrics['paired_improvement']:.10f}，"
        f"校正区块下界 {bootstrap['lower_bound']:.10f}；2021/2022改善 {by_year['2021']:.10f}/{by_year['2022']:.10f}，"
        f"排序相关 {rank['2021']:.4f}/{rank['2022']:.4f}。六项门 {sum(gates.values())}/6，决定 `{decision}`。\n\n"
        "此局部训练标签不能代替精确f696自身完整账户。失败版本不改符号、分钟窗口、风险阈值、动作幅度或年份救援；"
        "信息通过也仍需首次历史送达资格和单独冻结的完整0bp含费资金账户。"
        f"Recorder `{rid}`，NAS SHA `{archived['sha256']}` 已限定冷恢复。Goal仍未达成。\n")
    return summary
