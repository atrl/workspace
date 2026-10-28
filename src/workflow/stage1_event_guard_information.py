"""Bounded walk-forward first-passage information screen for the minute guard."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from src.data.minute_intake import atomic_json
from src.model.stage1_event_guard_value import EventGuardValue
from src.workflow.minute_action_value import _source


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(frame: pd.DataFrame, features: list[str], cutoff: str) -> DatasetH:
    packed = pd.concat({"feature": frame[features],
        "label": frame[["local_advantage"]],
        "meta": frame[["trade_date", "decision_at", "label_available_at"]]}, axis=1)
    packed[("meta", "feature_available_at")] = frame.decision_at
    packed.index = pd.MultiIndex.from_arrays([frame.decision_at,
        ["SOXL_SOXX_CASH"] * len(frame)], names=["datetime", "instrument"])
    return DatasetH(handler=DataHandlerLP.from_df(packed),
        segments={"train": (None, pd.Timestamp(cutoff) - pd.Timedelta(nanoseconds=1)),
                  "test": (pd.Timestamp(cutoff), None)})


def _bootstrap_lower(values: np.ndarray, *, mean_block: int, repetitions: int,
                     alpha: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    size = len(values)
    index = rng.integers(size, size=repetitions)
    sums = np.zeros(repetitions)
    for step in range(size):
        if step:
            refresh = rng.random(repetitions) < 1. / mean_block
            index = np.where(refresh, rng.integers(size, size=repetitions), (index + 1) % size)
        sums += values[index]
    return float(np.quantile(sums / size, alpha))


def _first_passage(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    data = frame[["trade_date", "decision_at", "local_advantage"]].copy()
    data["score"] = np.asarray(score, float)
    if not np.isfinite(data.score).all():
        raise ValueError("blocked: event first-passage score is nonfinite")
    crossing = data.loc[data.score.gt(0)].sort_values(["trade_date", "decision_at"])
    chosen = crossing.groupby("trade_date", sort=False).head(1)
    return chosen[["trade_date", "decision_at", "local_advantage", "score"]]


def run(*, root: Path, card: Path, plan: dict) -> dict:
    import qlib
    import yaml
    from qlib.workflow import R

    root, card = Path(root).resolve(), Path(card).resolve()
    digest = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        old = json.loads((card / "summary.json").read_text())
        if old["plan_sha256"] != digest:
            raise ValueError("blocked: frozen event information plan changed")
        return old
    for source in plan["sources"].values():
        if _sha(root / source["card"] / "summary.json") != source["summary_sha256"]:
            raise ValueError("blocked: event information source summary differs")
    qlib.init(provider_uri=str(root / plan["provider_uri"]), region="us", kernels=1)
    labels, la = _source(root, plan["sources"]["labels"], ["analysis/labels"])
    parent, pa = _source(root, plan["sources"]["parent"],
        ["analysis/native/dated_f696_0"])
    source = labels["analysis/labels"]
    source = source.loc[source.funded_change].sort_values(["trade_date", "decision_at"]).copy()
    if len(source) != plan["expected_funded_rows"]:
        raise ValueError("blocked: event label cohort differs")
    calendar = pd.DatetimeIndex(parent["analysis/native/dated_f696_0"]["authority"].trade_date)
    tracking = card / "qlib_recorder"
    progress_path = card / "run_state.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "plan_sha256": digest, "fit_attempts": [], "fits_done": []}
    if progress["plan_sha256"] != digest:
        raise ValueError("blocked: previous event information run belongs to another plan")
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
            model=(root / "src/model/stage1_event_guard_value.py").read_text())
        parts, model_audits, years, predictions = [], [], [], {}
        for year in plan["evaluation_years"]:
            cutoff = f"{year}-01-01"
            year_frame = source.loc[source.trade_date.dt.year.eq(year)].copy()
            if year_frame.trade_date.nunique() < plan["minimum_year_source_days"]:
                raise ValueError("needs_data: event receiver evaluation support differs")
            year_predictions = {}
            for arm, features in (("control", plan["control_features"]),
                                  ("candidate", [*plan["control_features"], plan["added_feature"]])):
                key = f"{year}_{arm}"
                dataset = _dataset(source, features, cutoff)
                if key in progress["fits_done"]:
                    model = recorder.load_object("models/" + key)
                else:
                    if (key in progress["fit_attempts"]
                            or len(progress["fit_attempts"]) >= plan["budget"]["model_fit_attempts"]):
                        raise ValueError("blocked: event value fit attempt consumed")
                    progress["fit_attempts"].append(key)
                    atomic_json(progress, progress_path)
                    model = EventGuardValue(features=features, cutoff=cutoff,
                        seed=plan["seed"]).fit(dataset)
                    recorder.save_objects(artifact_path="models", **{key: model})
                    progress["fits_done"].append(key)
                    atomic_json(progress, progress_path)
                model_audits.append({"year": year, "arm": arm, **model.audit})
                score = model.predict(dataset).reindex(pd.MultiIndex.from_arrays(
                    [year_frame.decision_at, ["SOXL_SOXX_CASH"] * len(year_frame)],
                    names=["datetime", "instrument"]))
                if score.isna().any():
                    raise ValueError("blocked: event prediction calendar incomplete")
                year_predictions[arm] = score.to_numpy(float)
                recorder.log_metrics(completed_model_fits=float(len(progress["fits_done"])))
            evaluation = pd.DataFrame({"trade_date": calendar[calendar.year == year]})
            for arm in ("control", "candidate"):
                chosen = _first_passage(year_frame, year_predictions[arm])
                value = chosen.set_index("trade_date").local_advantage
                evaluation[arm + "_value"] = evaluation.trade_date.map(value).fillna(0.).to_numpy(float)
                evaluation[arm + "_triggered"] = evaluation.trade_date.isin(chosen.trade_date)
                predictions[f"{year}_{arm}"] = pd.DataFrame({"trade_date": year_frame.trade_date.to_numpy(),
                    "decision_at": year_frame.decision_at.to_numpy(),
                    "actual": year_frame.local_advantage.to_numpy(float),
                    "score": year_predictions[arm]})
            evaluation["year"] = year
            parts.append(evaluation)
            years.append({"year": year, "days": len(evaluation),
                "candidate_trigger_days": int(evaluation.candidate_triggered.sum()),
                "control_trigger_days": int(evaluation.control_triggered.sum()),
                "candidate_local_value_sum": float(evaluation.candidate_value.sum()),
                "control_local_value_sum": float(evaluation.control_value.sum()),
                "candidate_increment_sum": float((evaluation.candidate_value - evaluation.control_value).sum())})
        daily = pd.concat(parts, ignore_index=True).sort_values("trade_date")
        gain = daily.candidate_value.to_numpy(float) - daily.control_value.to_numpy(float)
        lower = _bootstrap_lower(gain, mean_block=plan["bootstrap"]["mean_block_days"],
            repetitions=plan["bootstrap"]["resamples"],
            alpha=plan["bootstrap"]["one_sided_alpha"], seed=plan["bootstrap"]["seed"])
        passed = bool(all(row["candidate_trigger_days"] >= plan["minimum_year_trigger_days"]
            and row["candidate_local_value_sum"] > 0 and row["candidate_increment_sum"] > 0
            for row in years) and lower > 0)
        summary = {"experiment": card.name, "plan_sha256": digest,
            "decision": "event_guard_information_pass_needs_full_account" if passed
                        else "not_promoted_fixed_event_guard_information_failed",
            "objective_achieved": False, "historical_research_passed": False,
            "information_gate_pass": passed, "annual": years,
            "bootstrap_lower_mean_increment": lower,
            "new_model_fits": len(progress["fits_done"]), "new_native_accounts": 0,
            "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
            "recorder_uri": str(tracking),
            "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
            "provider_uri": str(root / plan["provider_uri"]),
            "historical_first_receipt_qualified": False,
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": ["inputs/*", "sources/*", "models/*", "analysis/*"]}}
        recorder.save_objects(artifact_path="analysis", summary=summary, daily=daily,
            predictions=predictions, model_audits=pd.DataFrame(model_audits))
        recorder.log_metrics(information_gate_pass=float(passed))
    pd.DataFrame(years).to_csv(card / "annual_information.csv", index=False)
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": card.name,
        "provider_uri": summary["provider_uri"], "recorder_uri": str(tracking),
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    (card / "conclusion.md").write_text("# 事件驱动减险首次触发信息门\n\n"
        "固定2021/2022逐年向前、同模型状态及普通5/15分钟涨跌控制，候选只加15分钟相对下行波动占比。"
        "每日日内第一条预测净切换价值>0的可成交分钟触发，其余日期在完整日历记0。"
        f"年度动作与净局部价值见`annual_information.csv`；"
        f"完整日历63日区块配对改善下界{lower:.8g}，决定`{summary['decision']}`。"
        "点位预测矩阵仅在Recorder，不按评价年修改符号、分钟窗口或阈值。"
        "局部信息不是自身状态完整账户；历史首次分钟接收仍未认证。\n")
    atomic_json(summary, card / "summary.json")
    return summary
