"""Walk-forward minute-timing information gate for funded cash-to-SOXX actions."""
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


def _dataset(frame: pd.DataFrame, features: list[str], label: str, cutoff: str) -> DatasetH:
    packed = pd.concat({"feature": frame[features], "label": frame[[label]],
        "meta": frame[["trade_date", "decision_at", "label_available_at"]]}, axis=1)
    packed[("meta", "feature_available_at")] = frame.decision_at
    packed.index = pd.MultiIndex.from_arrays([frame.decision_at,
        ["SOXL_SOXX_CASH"] * len(frame)], names=["datetime", "instrument"])
    return DatasetH(handler=DataHandlerLP.from_df(packed),
        segments={"train": (None, pd.Timestamp(cutoff) - pd.Timedelta(nanoseconds=1)),
                  "test": (pd.Timestamp(cutoff), None)})


def _minute_features(bars: pd.DataFrame) -> pd.DataFrame:
    parts = {}
    for symbol in ("SOXL", "SOXX"):
        item = bars.xs(symbol, level="instrument").copy()
        day = item.index.normalize()
        log_price = np.log(item["$close"].astype(float))
        for width in (5, 15):
            item[f"{symbol.lower()}_ret{width}"] = log_price - log_price.groupby(day).shift(width)
        if symbol == "SOXX":
            raw_close = item["$raw_close"].astype(float)
            volume = item["$raw_volume"].astype(float)
            weighted = (raw_close * volume).groupby(day).rolling(15, min_periods=15).sum().droplevel(0)
            count = volume.groupby(day).rolling(15, min_periods=15).sum().droplevel(0)
            item["soxx_vwap_gap15"] = np.log(raw_close / (weighted / count))
        columns = [f"{symbol.lower()}_ret5", f"{symbol.lower()}_ret15"]
        if symbol == "SOXX":
            columns.append("soxx_vwap_gap15")
        parts[symbol] = item[columns]
    return parts["SOXL"].join(parts["SOXX"], how="inner")


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


def _first_passage(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    data = frame[["trade_date", "decision_at", "local_advantage"]].copy()
    data["score"] = np.asarray(score, float)
    if not np.isfinite(data.score).all():
        raise ValueError("blocked: nonfinite minute timing score")
    return data.loc[data.score.gt(0)].sort_values(["trade_date", "decision_at"]).groupby(
        "trade_date", sort=False).head(1)


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
            raise ValueError("blocked: frozen timing information plan changed")
        return old
    source = plan["label_source"]
    label_card = root / source["card"]
    if (_sha(label_card / "summary.json") != source["summary_sha256"]
            or _sha(label_card / "verification_manifest.json") != source["inventory_sha256"]
            or _sha(root / plan["provider_uri"] / "qlib_store_manifest.json") != plan["provider_manifest_sha256"]):
        raise ValueError("blocked: timing information source identity differs")
    qlib.init(provider_uri=str(root / plan["provider_uri"]), region="us", kernels=1)
    label_objects, la = load_verified_recorder_objects(label_card,
        expected_recorder_id=source["recorder_id"], artifacts=["analysis/labels"],
        expected_manifest_sha256=source["inventory_sha256"])
    parent, pa = _source(root, plan["parent_source"], ["analysis/native/dated_f696_0"])
    labels = label_objects["analysis/labels"]
    labels = labels.loc[labels.funded_change].sort_values(["trade_date", "decision_at"]).copy()
    if len(labels) != plan["expected_funded_rows"]:
        raise ValueError("blocked: funded minute label source count differs")
    bars = D.features(["SOXL", "SOXX"],
        ["$close", "$raw_close", "$raw_volume"],
        start_time="2018-10-15 09:31:00", end_time="2022-12-22 16:00:00", freq="1min")
    minute = _minute_features(bars)
    aligned = minute.reindex(pd.DatetimeIndex(labels.bar_end_at))
    if aligned.isna().any().any() or len(aligned) != len(labels):
        raise ValueError("needs_data: funded minute feature coverage differs")
    aligned.index = labels.index
    data = labels.join(aligned)
    first = data.groupby("trade_date", sort=False).head(1).copy()
    anchor = first.set_index("trade_date")
    data["delta_local_advantage"] = data.local_advantage - data.trade_date.map(anchor.local_advantage)
    data["elapsed_minutes"] = (data.decision_at - data.trade_date.map(anchor.decision_at)).dt.total_seconds() / 60
    for name in ("soxl_ret5", "soxx_ret5", "soxl_ret15", "soxx_ret15", "soxx_vwap_gap15"):
        data[name + "_delta"] = data[name] - data.trade_date.map(anchor[name])
    if (not np.isfinite(data[[*plan["base_features"], *plan["common_delta_features"],
            plan["added_delta_feature"], "delta_local_advantage"]].to_numpy(float)).all()
            or not data.loc[first.index, "delta_local_advantage"].abs().le(1e-12).all()):
        raise ValueError("blocked: minute feature/first-anchor parity differs")
    first = data.loc[first.index].copy()
    calendar = pd.DatetimeIndex(parent["analysis/native/dated_f696_0"]["authority"].trade_date)
    tracking = card / "qlib_recorder"
    progress_path = card / "run_state.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "plan_sha256": digest, "fit_attempts": [], "fits_done": []}
    if progress["plan_sha256"] != digest:
        raise ValueError("blocked: prior timing fit belongs to another plan")
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
        years, parts, predictions, model_audits = [], [], {}, []
        for year in plan["evaluation_years"]:
            cutoff = f"{year}-01-01"
            cohort = data.loc[data.trade_date.dt.year.eq(year)].copy()
            if cohort.trade_date.nunique() < 50:
                raise ValueError("needs_data: minute evaluation year too sparse")
            model_specs = [
                ("base", first, plan["base_features"], "local_advantage"),
                ("control", data, plan["common_delta_features"], "delta_local_advantage"),
                ("candidate", data, [*plan["common_delta_features"], plan["added_delta_feature"]],
                 "delta_local_advantage")]
            output = {}
            for arm, frame, features, label in model_specs:
                key = f"{year}_{arm}"
                dataset = _dataset(frame, features, label, cutoff)
                if key in progress["fits_done"]:
                    model = recorder.load_object("models/" + key)
                else:
                    if (key in progress["fit_attempts"] or
                            len(progress["fit_attempts"]) >= plan["budget"]["new_model_fit_attempts"]):
                        raise ValueError("blocked: frozen minute fit budget consumed")
                    progress["fit_attempts"].append(key)
                    atomic_json(progress, progress_path)
                    model = CashTimingRidge(features=features, label=label,
                        cutoff=cutoff, alpha=10.0).fit(dataset)
                    recorder.save_objects(artifact_path="models", **{key: model})
                    progress["fits_done"].append(key)
                    atomic_json(progress, progress_path)
                model_audits.append({"year": year, "arm": arm, **model.audit})
                predictions_all = model.predict(dataset)
                cohort_frame = first.loc[first.trade_date.dt.year.eq(year)] if arm == "base" else cohort
                target_index = pd.MultiIndex.from_arrays([cohort_frame.decision_at,
                    ["SOXL_SOXX_CASH"] * len(cohort_frame)], names=["datetime", "instrument"])
                selected = predictions_all.reindex(target_index)
                if selected.isna().any():
                    raise ValueError("blocked: minute prediction calendar differs")
                output[arm] = pd.Series(selected.to_numpy(float), index=cohort_frame.index)
                recorder.log_metrics(completed_model_fits=float(len(progress["fits_done"])))
            base_by_day = pd.Series(output["base"].to_numpy(float),
                index=first.loc[first.trade_date.dt.year.eq(year)].trade_date)
            eval_day = pd.DataFrame({"trade_date": calendar[calendar.year == year]})
            scores = {}
            for arm in ("control", "candidate"):
                score = cohort.trade_date.map(base_by_day).to_numpy(float) + output[arm].to_numpy(float)
                scores[arm] = score
                chosen = _first_passage(cohort, score)
                eval_day[arm + "_value"] = eval_day.trade_date.map(
                    chosen.set_index("trade_date").local_advantage).fillna(0.).to_numpy(float)
                eval_day[arm + "_decision_at"] = eval_day.trade_date.map(
                    chosen.set_index("trade_date").decision_at)
                predictions[f"{year}_{arm}"] = pd.DataFrame({
                    "trade_date": cohort.trade_date.to_numpy(),
                    "decision_at": cohort.decision_at.to_numpy(),
                    "actual": cohort.local_advantage.to_numpy(float), "score": score})
            first_time = cohort.groupby("trade_date").decision_at.min()
            triggered = eval_day.candidate_decision_at.notna()
            candidate_times = eval_day.loc[triggered, ["trade_date", "candidate_decision_at"]]
            later = int(candidate_times.candidate_decision_at.gt(
                candidate_times.trade_date.map(first_time)).sum())
            differing = (eval_day.candidate_decision_at.fillna(pd.Timestamp("1900-01-01"))
                         != eval_day.control_decision_at.fillna(pd.Timestamp("1900-01-01")))
            difference = np.abs(scores["candidate"] - scores["control"])
            years.append({"year": year, "calendar_days": len(eval_day),
                "funded_source_days": int(cohort.trade_date.nunique()),
                "candidate_trigger_days": int(triggered.sum()),
                "control_trigger_days": int(eval_day.control_decision_at.notna().sum()),
                "candidate_later_than_first_days": later,
                "changed_action_days": int(differing.sum()),
                "different_score_rows": int((difference > 1e-12).sum()),
                "candidate_local_value_sum": float(eval_day.candidate_value.sum()),
                "control_local_value_sum": float(eval_day.control_value.sum()),
                "candidate_increment_sum": float((eval_day.candidate_value - eval_day.control_value).sum())})
            eval_day["year"] = year
            parts.append(eval_day)
        daily = pd.concat(parts, ignore_index=True).sort_values("trade_date")
        gain = daily.candidate_value.to_numpy(float) - daily.control_value.to_numpy(float)
        lower = _bootstrap_lower(gain, block=plan["bootstrap"]["mean_block_days"],
            repetitions=plan["bootstrap"]["resamples"],
            alpha=plan["bootstrap"]["one_sided_alpha"], seed=plan["bootstrap"]["seed"])
        coefficient = {x["year"]: x["coefficients"][plan["added_delta_feature"]]
            for x in model_audits if x["arm"] == "candidate"}
        passed = (all(row["candidate_trigger_days"] >= 10
            and row["candidate_local_value_sum"] > 0
            and row["candidate_increment_sum"] > 0
            and row["changed_action_days"] > 0
            and row["different_score_rows"] > 0
            and row["candidate_later_than_first_days"] > 0
            and coefficient[row["year"]] < 0 for row in years) and lower > 0)
        summary = {"experiment": card.name, "plan_sha256": digest,
            "decision": "minute_cash_timing_information_pass_needs_full_account" if passed
                        else "not_promoted_minute_cash_timing_information_failed",
            "objective_achieved": False, "historical_research_passed": False,
            "information_gate_pass": bool(passed), "annual": years,
            "added_feature_coefficients": coefficient,
            "bootstrap_lower_mean_increment": lower,
            "new_model_fits": len(progress["fits_done"]), "new_native_accounts": 0,
            "label_nas_receipt_qualified": False,
            "historical_first_receipt_qualified": False,
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
    pd.DataFrame(years).to_csv(card / "annual_information.csv", index=False)
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": card.name,
        "provider_uri": summary["provider_uri"], "recorder_uri": str(tracking),
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    (card / "conclusion.md").write_text("# 分钟级现金部署时机模型信息门\n\n"
        "以每天首个合法整股买入分钟为点时基线，分别拟合状态基础价值与之后的分钟价值变化。"
        "对照用普通5/15分钟收益和经过分钟，候选只加SOXX最近15根已完成bar的成交量加权价格偏离。"
        f"2021/2022年度与完整日历见annual_information.csv；区块增量下界{lower:.8g}，"
        f"决定`{summary['decision']}`。旧保护标签、独立树模型均未复用。"
        "本卡零候选完整账户，历史首收和新标签NAS工件收据未认证；SOXX超额未证明。\n")
    atomic_json(summary, card / "summary.json")
    return summary
