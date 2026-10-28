"""Matched 09:46 volume timing increment on exact-parent local action value."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.minute_intake import atomic_json
from src.model.f696_joint_local_value import F696JointLocalValue
from src.workflow.minute_action_value import _source
from src.workflow.stage1_f696_joint_information import _dataset, _scores, _stationary_lower


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, root: Path, card: Path, plan: dict) -> dict:
    import qlib
    from qlib.workflow import R

    root, card = Path(root).resolve(), Path(card).resolve()
    digest = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        old = json.loads((card / "summary.json").read_text())
        if old["plan_sha256"] != digest:
            raise ValueError("blocked: frozen 09:46 information plan changed")
        return old
    for source in plan["sources"].values():
        if _sha(root / source["card"] / "summary.json") != source["summary_sha256"]:
            raise ValueError("blocked: matched 09:46 source summary changed")
    qlib.init(provider_uri=str(root / plan["provider_uri"]), region="us", kernels=1)
    labels, la = _source(root, plan["sources"]["labels"], ["analysis/labels"])
    reference, ra = _source(root, plan["sources"]["frame"], ["inputs/covered_frame"])
    volume, va = _source(root, plan["sources"]["volume"], ["analysis/coordinate"])
    label_frame = labels["analysis/labels"]
    original = reference["inputs/covered_frame"]
    coordinate = volume["analysis/coordinate"]
    if (len(label_frame) != len(original) or len(coordinate) != len(original)
            or not coordinate.source_known.all()):
        raise ValueError("needs_data: 09:46 coordinate does not cover exact parent calendar")
    joined = label_frame.merge(original[["signal_date", *plan["prior_market_features"], "feature_available_at"]],
        on="signal_date", validate="one_to_one")
    joined = joined.merge(coordinate[["trade_date", *plan["early_features"], plan["added_feature"],
        "first_available_at"]], on="trade_date", validate="one_to_one")
    if (not pd.to_datetime(joined.feature_available_at).lt(pd.to_datetime(joined.decision_at)).all()
            or not pd.to_datetime(joined.first_available_at).le(pd.to_datetime(joined.decision_at)).all()):
        raise ValueError("blocked: 09:46 feature availability differs")
    fields = [*plan["state_features"], *plan["prior_market_features"],
              *plan["early_features"], plan["added_feature"]]
    if joined[fields].isna().any().any() or len(joined) != plan["expected_reference_rows"]:
        raise ValueError("needs_data: matched 09:46 feature matrix incomplete")
    tracking = card / "qlib_recorder"
    progress_path = card / "run_state.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "plan_sha256": digest, "fit_attempts": [], "fits_done": []}
    if progress["plan_sha256"] != digest:
        raise ValueError("blocked: prior 09:46 fit belongs to a different plan")
    kwargs = {"experiment_name": card.name, "uri": str(tracking)}
    if progress.get("recorder_id"):
        kwargs.update(recorder_id=progress["recorder_id"], resume=True)
    with R.start(**kwargs):
        recorder = R.get_recorder()
        progress.update(recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
        atomic_json(progress, progress_path)
        recorder.save_objects(artifact_path="inputs", plan=plan,
            source_audits={"labels": la, "frame": ra, "volume": va})
        recorder.save_objects(artifact_path="sources", workflow=Path(__file__).read_text(),
            model=(root / "src/model/f696_joint_local_value.py").read_text())
        all_years, predictions, gates, audits = [], {}, {}, []
        for route, config in plan["routes"].items():
            data = joined.loc[joined.eligible & joined.route.eq(route)].sort_values("trade_date")
            cutoff = config["cutoff"]
            inputs = {"control": [*plan["state_features"], *plan["prior_market_features"],
                                   *plan["early_features"]]}
            inputs["candidate"] = [*inputs["control"], plan["added_feature"]]
            scores, models = {}, {}
            for arm, features in inputs.items():
                name = route + "_" + arm
                dataset = _dataset(data, features=features, cutoff=cutoff)
                if name in progress["fits_done"]:
                    model = recorder.load_object("models/" + name)
                else:
                    if name in progress["fit_attempts"] or len(progress["fit_attempts"]) >= plan["budget"]["model_fit_attempts"]:
                        raise ValueError("blocked: frozen 09:46 fit attempt consumed")
                    progress["fit_attempts"].append(name)
                    atomic_json(progress, progress_path)
                    model = F696JointLocalValue(features=features, cutoff=cutoff, seed=plan["seed"]).fit(dataset)
                    recorder.save_objects(artifact_path="models", **{name: model})
                    progress["fits_done"].append(name)
                    atomic_json(progress, progress_path)
                models[arm] = model
                scores[arm] = model.predict(dataset)
                audits.append({"route": route, "arm": arm, **model.audit})
                recorder.log_metrics(completed_model_fits=float(len(progress["fits_done"])))
            truth = data.loc[data.trade_date.ge(pd.Timestamp(cutoff)), ["trade_date", "local_advantage"]]
            index = pd.MultiIndex.from_arrays([truth.trade_date, ["SOXL_SOXX_CASH"] * len(truth)],
                                              names=["datetime", "instrument"])
            actual = truth.local_advantage.to_numpy(float)
            control = scores["control"].reindex(index).to_numpy(float)
            candidate = scores["candidate"].reindex(index).to_numpy(float)
            if len(truth) != config["expected_evaluation_rows"] or not np.isfinite(np.c_[actual, control, candidate]).all():
                raise ValueError("blocked: fixed 09:46 evaluation cohort differs")
            mean = models["control"].training_mean
            pooled = _scores(actual, candidate, control, mean)
            gain = (actual - control) ** 2 - (actual - candidate) ** 2
            lower = _stationary_lower(gain, mean_block=21, repetitions=3000,
                                      seed=plan["seed"] + config["seed_offset"])
            annual = []
            for year in config["required_years"]:
                take = truth.trade_date.dt.year.eq(year).to_numpy()
                annual.append({"route": route, "year": year,
                    **_scores(actual[take], candidate[take], control[take], mean)})
            passed = (pooled["candidate_mse"] < pooled["control_mse"]
                and pooled["candidate_mse"] < pooled["train_mean_mse"]
                and pooled["candidate_auc"] > .5
                and pooled["predicted_positive"] >= 5 and pooled["predicted_negative"] >= 5
                and lower > 0 and all(row["rows"] >= config["minimum_year_rows"]
                    and row["candidate_mse"] < row["control_mse"]
                    and row["candidate_auc"] > .5 for row in annual))
            gates[route] = {"passed": bool(passed), "pooled": pooled,
                "bootstrap_lower_mse_gain": lower, "required_years": config["required_years"]}
            all_years.extend(annual)
            predictions[route] = pd.DataFrame({"trade_date": truth.trade_date.to_numpy(),
                "actual": actual, "control": control, "candidate": candidate})
        joint_pass = all(x["passed"] for x in gates.values())
        summary = {"experiment": card.name, "plan_sha256": digest,
            "decision": "09_46_joint_information_pass_needs_source_receipt_and_0bp_design" if joint_pass
                        else "not_promoted_fixed_0946_volume_timing_information_failed",
            "objective_achieved": False, "historical_research_passed": False,
            "gates": gates, "historical_first_receipt_qualified": False,
            "new_model_fits": len(progress["fits_done"]), "new_native_accounts": 0,
            "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
            "recorder_uri": str(tracking),
            "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
            "provider_uri": str(root / plan["provider_uri"]),
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": ["inputs/*", "sources/*", "models/*", "analysis/*"]}}
        recorder.save_objects(artifact_path="analysis", summary=summary,
            annual=pd.DataFrame(all_years), predictions=predictions, training_audit=pd.DataFrame(audits))
        recorder.log_metrics(joint_information_pass=float(joint_pass))
    pd.DataFrame(all_years).to_csv(card / "annual_information.csv", index=False)
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": card.name,
        "provider_uri": summary["provider_uri"], "recorder_uri": str(tracking),
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    (card / "conclusion.md").write_text("# 精确f696的09:46成交量时序残差信息门\n\n"
        "固定四次LightGBM拟合；状态、日线六项和早盘普通涨跌在两组共同控制中，候选只加一个已通过来源门的成交量时序残差。"
        "两类同状态Tiered局部标签分别按事前时间切分和相同MSE、AUC、逐年、区块下界硬门检验。"
        f"决定：`{summary['decision']}`。数值见 `summary.json` 与 `annual_information.csv`。"
        "旧日09:46首次实际接收未认证；来源通过或局部信息改善都不能替代自身完整账户和独立前向证据。"
        "固定失败后不改分钟窗口、符号、模型深度或年份。\n")
    atomic_json(summary, card / "summary.json")
    return summary
