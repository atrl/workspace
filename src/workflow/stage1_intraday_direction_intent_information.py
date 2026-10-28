"""Funded-state direction screen with all minute buy intentions and fixed execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json
from src.model.stage1_intraday_cash_timing_ridge import CashTimingRidge
from src.workflow.minute_action_value import _source


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(frame: pd.DataFrame, features: list[str], cutoff: str) -> DatasetH:
    packed = pd.concat({"feature": frame[features],
        "label": frame[["intent_local_advantage"]],
        "meta": frame[["trade_date", "decision_at", "label_available_at"]]}, axis=1)
    packed[("meta", "feature_available_at")] = frame.decision_at
    packed.index = pd.MultiIndex.from_arrays([frame.decision_at,
        ["SOXL_SOXX_CASH"] * len(frame)], names=["datetime", "instrument"])
    return DatasetH(handler=DataHandlerLP.from_df(packed),
        segments={"train": (None, pd.Timestamp(cutoff) - pd.Timedelta(nanoseconds=1)),
                  "test": (pd.Timestamp(cutoff), None)})


def _bootstrap_lower(values: np.ndarray, *, block: int, repetitions: int,
                     alpha: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    size = len(values)
    index = rng.integers(size, size=repetitions)
    sums = np.zeros(repetitions)
    for step in range(size):
        if step:
            refresh = rng.random(repetitions) < 1. / block
            index = np.where(refresh, rng.integers(size, size=repetitions), (index + 1) % size)
        sums += values[index]
    return float(np.quantile(sums / size, alpha))


def _replay_intentions(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    data = frame[["trade_date", "decision_at", "fill_at", "funded_change",
                  "intent_local_advantage"]].copy()
    data["score"] = np.asarray(scores, float)
    if not np.isfinite(data.score).all():
        raise ValueError("blocked: nonfinite direction score")
    rows = []
    for day, part in data.sort_values(["trade_date", "decision_at"]).groupby("trade_date", sort=False):
        next_allowed = pd.Timestamp(day)
        attempts = failed = 0
        chosen = None
        for row in part.itertuples(index=False):
            if row.decision_at < next_allowed or row.score <= 0:
                continue
            attempts += 1
            next_allowed = row.fill_at + pd.Timedelta(minutes=1)
            if row.funded_change:
                chosen = row
                break
            failed += 1
        rows.append({"trade_date": day, "attempts": attempts,
            "failed_attempts": failed, "funded": chosen is not None,
            "decision_at": chosen.decision_at if chosen is not None else pd.NaT,
            "local_value": float(chosen.intent_local_advantage) if chosen is not None else 0.})
    return pd.DataFrame(rows)


def run(*, root: Path, card: Path, plan: dict) -> dict:
    import qlib
    import yaml
    from qlib.data import D
    from qlib.workflow import R

    root, card = Path(root).resolve(), Path(card).resolve()
    digest = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        old = json.loads((card / "summary.json").read_text())
        if old["plan_sha256"] != digest:
            raise ValueError("blocked: frozen direction plan differs")
        return old
    source = plan["label_source"]
    label_card = root / source["card"]
    if (_sha(label_card / "summary.json") != source["summary_sha256"]
            or _sha(label_card / "verification_manifest.json") != source["inventory_sha256"]
            or _sha(root / plan["provider_uri"] / "qlib_store_manifest.json") != plan["provider_manifest_sha256"]):
        raise ValueError("blocked: direction source identity differs")
    qlib.init(provider_uri=str(root / plan["provider_uri"]), region="us", kernels=1)
    objects, la = load_verified_recorder_objects(label_card,
        expected_recorder_id=source["recorder_id"], artifacts=["analysis/labels"],
        expected_manifest_sha256=source["inventory_sha256"])
    parent, pa = _source(root, plan["parent_source"], ["analysis/native/dated_f696_0"])
    source_rows = objects["analysis/labels"].copy()
    if len(source_rows) != plan["expected_source_slots"]:
        raise ValueError("blocked: full minute source-slot count differs")
    data = source_rows.loc[source_rows.cash_deployment_weight.gt(1e-10)].copy()
    data = data.sort_values(["trade_date", "decision_at"])
    if (len(data) != plan["expected_intent_rows"]
            or int(data.funded_change.sum()) != plan["expected_funded_rows"]):
        raise ValueError("blocked: decision-known intent cohort differs")
    data["intent_local_advantage"] = data.local_advantage.where(data.funded_change, 0.).astype(float)
    data.loc[~data.funded_change, "label_available_at"] = (
        data.loc[~data.funded_change, "fill_at"] + pd.Timedelta(minutes=1))
    if (data.label_available_at.le(data.decision_at).any()
            or not np.isfinite(data.intent_local_advantage).all()):
        raise ValueError("blocked: intent label maturity or value invalid")
    bars = D.features(["SOXX"], ["$close"],
        start_time="2018-10-15 09:31:00", end_time="2022-12-22 16:00:00", freq="1min")
    soxx = bars.xs("SOXX", level="instrument")["$close"].astype(float)
    day = soxx.index.normalize()
    ret15 = np.log(soxx) - np.log(soxx).groupby(day).shift(15)
    data["soxx_ret15"] = ret15.reindex(pd.DatetimeIndex(data.bar_end_at)).to_numpy(float)
    data["minute_of_session"] = (data.decision_at - data.trade_date
        - pd.Timedelta(hours=9, minutes=30)).dt.total_seconds() / 60
    features = [*plan["control_features"], plan["candidate_added_feature"]]
    if not np.isfinite(data[features].to_numpy(float)).all():
        raise ValueError("needs_data: completed minute direction feature missing")
    calendar = pd.DatetimeIndex(parent["analysis/native/dated_f696_0"]["authority"].trade_date)
    tracking = card / "qlib_recorder"
    progress_path = card / "run_state.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "plan_sha256": digest, "fit_attempts": [], "fits_done": []}
    if progress["plan_sha256"] != digest:
        raise ValueError("blocked: prior direction fit belongs to another plan")
    kwargs = {"experiment_name": card.name, "uri": str(tracking)}
    if progress.get("recorder_id"):
        kwargs.update(recorder_id=progress["recorder_id"], resume=True)
    with R.start(**kwargs):
        recorder = R.get_recorder()
        progress.update(recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
        atomic_json(progress, progress_path)
        recorder.save_objects(artifact_path="inputs", plan=plan,
            source_audits={"labels": la, "parent": pa})
        recorder.save_objects(artifact_path="sources", workflow=Path(__file__).read_text(),
            model=(root / "src/model/stage1_intraday_cash_timing_ridge.py").read_text())
        annual, parts, predictions, model_audits = [], [], {}, []
        for year in plan["evaluation_years"]:
            cutoff = f"{year}-01-01"
            cohort = data.loc[data.trade_date.dt.year.eq(year)].copy()
            if cohort.trade_date.nunique() < 50:
                raise ValueError("needs_data: direction evaluation year sparse")
            scores = {}
            for arm, selected_features in (("control", plan["control_features"]),
                                           ("candidate", features)):
                key = f"{year}_{arm}"
                dataset = _dataset(data, selected_features, cutoff)
                if key in progress["fits_done"]:
                    model = recorder.load_object("models/" + key)
                else:
                    if (key in progress["fit_attempts"] or
                            len(progress["fit_attempts"]) >= plan["budget"]["new_model_fit_attempts"]):
                        raise ValueError("blocked: frozen direction fit budget consumed")
                    progress["fit_attempts"].append(key)
                    atomic_json(progress, progress_path)
                    model = CashTimingRidge(features=selected_features,
                        label="intent_local_advantage", cutoff=cutoff, alpha=10.0).fit(dataset)
                    recorder.save_objects(artifact_path="models", **{key: model})
                    progress["fits_done"].append(key)
                    atomic_json(progress, progress_path)
                model_audits.append({"year": year, "arm": arm, **model.audit})
                pred = model.predict(dataset)
                target_index = pd.MultiIndex.from_arrays([cohort.decision_at,
                    ["SOXL_SOXX_CASH"] * len(cohort)], names=["datetime", "instrument"])
                selected = pred.reindex(target_index)
                if selected.isna().any():
                    raise ValueError("blocked: direction prediction calendar differs")
                scores[arm] = selected.to_numpy(float)
                predictions[key] = pd.DataFrame({"trade_date": cohort.trade_date.to_numpy(),
                    "decision_at": cohort.decision_at.to_numpy(), "actual": cohort.intent_local_advantage.to_numpy(float),
                    "funded_change": cohort.funded_change.to_numpy(bool), "score": scores[arm]})
                recorder.log_metrics(completed_model_fits=float(len(progress["fits_done"])))
            evaluation = pd.DataFrame({"trade_date": calendar[calendar.year == year]})
            for arm in ("control", "candidate"):
                replay = _replay_intentions(cohort, scores[arm]).set_index("trade_date")
                evaluation[arm + "_value"] = evaluation.trade_date.map(replay.local_value).fillna(0.).to_numpy(float)
                evaluation[arm + "_decision_at"] = evaluation.trade_date.map(replay.decision_at)
                evaluation[arm + "_attempts"] = evaluation.trade_date.map(replay.attempts).fillna(0).to_numpy(int)
                evaluation[arm + "_failed_attempts"] = evaluation.trade_date.map(replay.failed_attempts).fillna(0).to_numpy(int)
            changed = evaluation.candidate_decision_at.fillna(pd.Timestamp("1900-01-01")).ne(
                evaluation.control_decision_at.fillna(pd.Timestamp("1900-01-01")))
            annual.append({"year": year, "calendar_days": len(evaluation),
                "intent_days": int(cohort.trade_date.nunique()),
                "intent_rows": len(cohort), "funded_source_rows": int(cohort.funded_change.sum()),
                "candidate_funded_days": int(evaluation.candidate_decision_at.notna().sum()),
                "control_funded_days": int(evaluation.control_decision_at.notna().sum()),
                "candidate_attempts": int(evaluation.candidate_attempts.sum()),
                "candidate_failed_attempts": int(evaluation.candidate_failed_attempts.sum()),
                "changed_funded_action_days": int(changed.sum()),
                "different_score_rows": int((np.abs(scores["candidate"] - scores["control"]) > 1e-12).sum()),
                "candidate_local_value_sum": float(evaluation.candidate_value.sum()),
                "control_local_value_sum": float(evaluation.control_value.sum()),
                "candidate_increment_sum": float((evaluation.candidate_value - evaluation.control_value).sum())})
            evaluation["year"] = year
            parts.append(evaluation)
        daily = pd.concat(parts, ignore_index=True).sort_values("trade_date")
        gain = daily.candidate_value.to_numpy(float) - daily.control_value.to_numpy(float)
        lower = _bootstrap_lower(gain, block=plan["bootstrap"]["mean_block_days"],
            repetitions=plan["bootstrap"]["resamples"],
            alpha=plan["bootstrap"]["one_sided_alpha"], seed=plan["bootstrap"]["seed"])
        coefficient = {row["year"]: row["coefficients"][plan["candidate_added_feature"]]
            for row in model_audits if row["arm"] == "candidate"}
        passed = (all(row["candidate_funded_days"] >= 10
            and row["candidate_local_value_sum"] > 0
            and row["candidate_increment_sum"] > 0
            and row["changed_funded_action_days"] > 0
            and row["different_score_rows"] > 0
            and coefficient[row["year"]] > 0 for row in annual) and lower > 0)
        summary = {"experiment": card.name, "plan_sha256": digest,
            "decision": "minute_direction_intent_information_pass_needs_execution_and_account" if passed
                        else "not_promoted_minute_direction_intent_information_failed",
            "objective_achieved": False, "historical_research_passed": False,
            "information_gate_pass": bool(passed), "annual": annual,
            "added_feature_coefficients": coefficient,
            "bootstrap_lower_mean_increment": lower,
            "new_model_fits": len(progress["fits_done"]), "new_native_accounts": 0,
            "label_nas_receipt_qualified": False, "historical_first_receipt_qualified": False,
            "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
            "recorder_uri": str(tracking),
            "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
            "provider_uri": str(root / plan["provider_uri"]),
            "artifact_policy": {"policy": "slim_card_recorder_first",
                "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": ["inputs/*", "sources/*", "models/*", "analysis/*"]}}
        recorder.save_objects(artifact_path="analysis", summary=summary, daily=daily,
            predictions=predictions, model_audits=pd.DataFrame(model_audits))
        recorder.log_metrics(information_gate_pass=float(passed))
    pd.DataFrame(annual).to_csv(card / "annual_information.csv", index=False)
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": card.name,
        "provider_uri": summary["provider_uri"], "recorder_uri": str(tracking),
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    (card / "conclusion.md").write_text("# 分钟现金部署方向模型的信息门\n\n"
        "固定执行为下一分钟开盘，从当前状态学习现金买SOXX相对HOLD的费用后局部价值。"
        "训练和评价覆盖所有当时有目标空间的分钟意向，未成交/整股不足保留为零价值并按结果时钟继续，"
        "不按事后成功成交筛选。候选只加SOXX已完成15分钟收益。"
        f"逐年完整日历见annual_information.csv，配对区块下界{lower:.8g}，"
        f"决定`{summary['decision']}`。本卡不训练执行择时或跑完整候选账户；"
        "历史分钟首收及新标签NAS工件回执未认证，SOXX超额未证明。\n")
    atomic_json(summary, card / "summary.json")
    return summary
