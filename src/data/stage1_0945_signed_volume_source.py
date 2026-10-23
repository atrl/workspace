"""Qlib source-only 09:45 SOXX signed-volume participation preflight."""

from __future__ import annotations

import hashlib
import json
import shlex
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_daily_prefix(raw: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Keep each source session, including days whose full prefix is unknown."""
    result = []
    frame = raw.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    if frame.duplicated(["datetime", "instrument"]).any():
        raise ValueError("duplicate native minute observation")
    if set(frame.instrument.unique()) != {"SOXX"}:
        raise ValueError("wrong source instrument")
    frame = frame.loc[frame.datetime.dt.time.between(time(9, 31), time(9, 45))]
    frame["date"] = frame.datetime.dt.normalize()
    groups = {date: group.sort_values("datetime") for date, group in frame.groupby("date")}
    for date in dates:
        date = pd.Timestamp(date).normalize()
        group = groups.get(date)
        reason = "ok"
        pressure = opening_return = unsigned_volume = float("nan")
        count = 0 if group is None else len(group)
        expected = pd.date_range(date + pd.Timedelta(hours=9, minutes=31), periods=15, freq="min")
        if group is None or count != 15 or not pd.DatetimeIndex(group.datetime).equals(expected):
            reason = "missing_or_duplicate_prefix_bar"
        else:
            close = group["$close"].to_numpy(float)
            volume = group["$volume"].to_numpy(float)
            valid = group["$price_valid"].to_numpy(float)
            if (not np.isfinite(close).all() or not (close > 0).all() or
                    not np.isfinite(volume).all() or not (volume >= 0).all() or
                    not np.isfinite(valid).all() or not np.equal(valid, 1).all()):
                reason = "invalid_close_volume_or_price_flag"
            else:
                signed_volume = volume[1:]
                denominator = float(signed_volume.sum())
                if denominator <= 0:
                    reason = "zero_signed_bar_volume"
                else:
                    pressure = float(np.dot(np.sign(np.diff(close)), signed_volume) / denominator)
                    opening_return = float(close[-1] / close[0] - 1)
                    unsigned_volume = denominator
        result.append({"signal_date": date, "source_last_bar": expected[-1],
                       "decision_at": date + pd.Timedelta(hours=9, minutes=46),
                       "bars_present": count, "reason": reason,
                       "signed_volume_pressure": pressure,
                       "opening_return_0931_0945": opening_return,
                       "unsigned_volume_14bars": unsigned_volume})
    return pd.DataFrame(result)


def run(root: Path) -> dict:
    import qlib
    from qlib.data import D
    from qlib.workflow import R

    from src.data.minute_intake import atomic_json
    from src.data.nas_market_data import _settings, _ssh, archive_nas_recorder

    root = Path(root).resolve()
    card = root / "research/stage1_0945_signed_volume_cash_source_20261023"
    plan = json.loads((card / "plan.json").read_text())
    assert plan["frozen"] and not (card / "summary.json").exists()
    digest = _sha(card / "plan.json")
    provider = root / plan["source"]["provider_uri"]
    assert _sha(provider / "qlib_store_manifest.json") == plan["source"]["manifest_sha256"]
    manifest = json.loads((provider / "qlib_store_manifest.json").read_text())
    assert manifest["as_of"] == plan["source"]["as_of"]
    assert plan["source"]["nas_source_id"] in manifest["source"]
    qlib.init(provider_uri=str(provider.resolve()), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(
        start_time="2018-10-12 09:31:00", end_time="2026-10-09 16:00:00", freq="1min"))
    dates = calendar.normalize().unique()
    raw = D.features(["SOXX"], plan["source"]["fields"],
        start_time="2018-10-12 09:31:00", end_time="2026-10-09 16:00:00", freq="1min")
    daily = build_daily_prefix(raw.reset_index(), dates)
    assert daily.signal_date.is_unique and daily.signal_date.is_monotonic_increasing
    assert len(daily) == len(dates)
    assert daily.loc[daily.reason.eq("ok"), "signed_volume_pressure"].between(-1, 1).all()
    assert (daily.source_last_bar < daily.decision_at).all()
    train = daily.loc[daily.signal_date.between(plan["cohorts"]["train_start"], plan["cohorts"]["train_end"])]
    held = daily.loc[daily.signal_date.between(plan["cohorts"]["held_start"], plan["cohorts"]["held_end"])]
    train_known = train.loc[train.reason.eq("ok")]
    held_known = held.loc[held.reason.eq("ok")]
    spearman_return = float(train_known.signed_volume_pressure.corr(
        train_known.opening_return_0931_0945, method="spearman"))
    spearman_volume = float(train_known.signed_volume_pressure.corr(
        train_known.unsigned_volume_14bars, method="spearman"))
    gates = {
        "train_support": len(train_known) >= plan["cohorts"]["minimum_train_known"],
        "held_support": len(held_known) >= plan["cohorts"]["minimum_held_known"],
        "bounded_coordinate": bool(daily.loc[daily.reason.eq("ok"), "signed_volume_pressure"].between(-1, 1).all()),
        "not_return_duplicate": abs(spearman_return) < .80,
    }
    decision = ("conditional_source_and_novelty_pass_no_information_or_account" if all(gates.values())
                else "closed_before_information_signed_volume_source_or_novelty_failed")
    result = {"experiment": card.name, "decision": decision,
              "plan_sha256": digest, "provider_uri": str(provider),
              "provider_manifest_sha256": plan["source"]["manifest_sha256"],
              "sessions": len(daily), "train_calendar": len(train), "train_known": len(train_known),
              "held_calendar": len(held), "held_known": len(held_known),
              "unknown_reasons": daily.loc[daily.reason.ne("ok"), "reason"].value_counts().to_dict(),
              "train_spearman_with_opening_return": spearman_return,
              "train_spearman_with_unsigned_volume": spearman_volume,
              "train_pressure_min": float(train_known.signed_volume_pressure.min()),
              "train_pressure_max": float(train_known.signed_volume_pressure.max()),
              "gates": gates, "Qlib_D_calendar_calls": 1, "Qlib_D_feature_calls": 1,
              "new_market_requests": 0, "DatasetH_builds": 0, "model_fits": 0,
              "native_accounts": 0, "future_return_reads": 0}
    R.set_uri(str((card / "qlib_recorder").resolve()))
    with R.start(experiment_name=card.name):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, provider_manifest=manifest)
        recorder.save_objects(artifact_path="analysis", daily_source=daily, result=result)
        R.log_params(coordinate="SOXX_0931_0945_signed_volume_pressure", no_future_labels=True)
        R.log_metrics(train_known=len(train_known), held_known=len(held_known),
                      train_spearman_with_opening_return=spearman_return)
        rid, eid = recorder.id, str(recorder.experiment_id)
    result["recorder_id"] = rid
    result["experiment_id"] = eid
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    source_binding = json.loads((root / "research/minute_information_pit_allocator_20261012/nas_binding.json").read_text())
    source_binding.pop("evidence", None)
    atomic_json(source_binding, card / "nas_binding.json")
    nas = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
                               card=card, recorder_id=rid, experiment_id=eid)
    remote = _ssh(_settings(root)["ssh"], "sha256sum " + shlex.quote(nas["nas_path"])).split()[0]
    assert remote == nas["sha256"]
    summary = {"experiment": card.name, "decision": decision,
               "objective_achieved": False, "plan_sha256": digest,
               "result": "result.json", "recorder_id": rid, "experiment_id": eid,
               "NAS_evidence": nas, "NAS_readback": {"verified": True, "sha256": remote},
               "model_fits": 0, "native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first",
                                   "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "analysis/daily_source", "analysis/result"]},
               "limitations": plan["limitations"]}
    atomic_json(summary, card / "summary.json")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nplan: plan.json\n"
        "provider_uri: " + plan["source"]["provider_uri"] + "\n"
        "phase: source_only_no_fit_no_account\n")
    conclusion = (
        "# Stage1 09:45 SOXX 方向加权成交量来源与去重门\n\n"
        "冻结完整 Stage1 09:46 配置时钟，只用 NAS 来源的 Qlib SOXX 09:31–09:45 已完成分钟价量。"
        "逐日保留缺失和无效掩码，0 未来收益读取、0 模型拟合、0 账户。\n\n"
        f"训练 {len(train_known)}/{len(train)} 日已知，复用持有期 {len(held_known)}/{len(held)} 日已知。"
        f"训练期与同前缀开盘收益 Spearman {spearman_return:.4f}，与无符号成交量 {spearman_volume:.4f}。"
        f"四项来源/去重门 {sum(gates.values())}/4；决定 `{decision}`。"
        f"无效原因 {result['unknown_reasons']}。\n\n"
        "数值存在与低相关不能证明未来现金优势、完整 0bp SOXX 超额或历史首收；"
        "若来源门通过，仍须另冻匹配信息任务和后续自身状态 Tiered 账户。"
        f"Recorder `{rid}`，NAS SHA `{remote}`。\n")
    (card / "conclusion.md").write_text(conclusion)
    return summary
