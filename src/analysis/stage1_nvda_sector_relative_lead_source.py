"""Fixed NVDA/SOXX daily leadership source bridge and novelty preflight."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path) -> dict:
    import qlib
    from qlib.data import D
    from qlib.workflow import R

    from src.data.nas_market_data import _settings, _ssh, archive_nas_recorder
    from src.data.qlib_native import build_qlib_store_from_frame

    root = Path(root).resolve()
    card = root / "research/stage1_nvda_sector_relative_lead_source_20261023"
    if (card / "summary.json").exists():
        raise ValueError("fixed NVDA leadership source screen already completed")
    plan = json.loads((card / "plan.json").read_text())
    source = plan["source"]
    assert plan["frozen"]
    for relative, expected in [
        (source["old_NVDA_provider"] + "/qlib_store_manifest.json", source["old_provider_manifest_sha256"]),
        (source["old_NVDA_provider"] + "/features/nvda/close.day.bin", source["old_NVDA_close_bin_sha256"]),
        (source["peer_card"] + "/summary.json", source["peer_summary_sha256"]),
        (source["peer_card"] + "/verification_manifest.json", source["peer_verification_manifest_sha256"]),
        (source["peer_raw_NVDA_5min"], source["peer_raw_NVDA_5min_sha256"]),
        (source["peer_NVDA_actions"], source["peer_NVDA_actions_sha256"]),
        (source["Stage1_SOXX_provider"] + "/qlib_store_manifest.json", source["Stage1_provider_manifest_sha256"]),
        (source["original39_card"] + "/summary.json", source["original39_summary_sha256"]),
    ]:
        assert _sha(root / relative) == expected, relative
    peer_summary = json.loads((root / source["peer_card"] / "summary.json").read_text())
    assert peer_summary["nas_evidence"]["sha256"] == source["peer_recorder_NAS_sha256"]
    settings = _settings(root)
    snapshot_root = settings["snapshot_parent"].rstrip("/") + "/" + source["NAS_verified_snapshot_source_id"]
    binding = json.loads(_ssh(settings["ssh"], "cat " + shlex.quote(snapshot_root + "/nas_binding.json")))
    assert binding["verified"] is True and binding["source_id"] == source["NAS_verified_snapshot_source_id"]
    entry = next(item for item in binding["objects"] if item["snapshot_path"] == source["peer_raw_NVDA_5min"])
    assert entry["sha256"] == source["NAS_peer_object_sha256"]
    remote = entry.get("snapshot_root", binding["snapshot_root"]).rstrip("/") + "/" + entry["snapshot_path"]
    assert _ssh(settings["ssh"], "sha256sum " + shlex.quote(remote)).split()[0] == entry["sha256"]
    qlib.init(provider_uri=str((root / source["old_NVDA_provider"]).resolve()), region="us", kernels=1)
    old = D.features(["NVDA"], ["$close"], start_time="2010-03-11",
                     end_time=source["old_provider_as_of"], freq="day")["$close"].droplevel("instrument").sort_index()
    assert old.notna().all() and old.gt(0).all()
    peer_raw = pd.read_parquet(root / source["peer_raw_NVDA_5min"], columns=["time_key", "close"])
    peer_stamp = pd.to_datetime(peer_raw.time_key)
    assert peer_stamp.is_monotonic_increasing
    peer = peer_raw.assign(date=peer_stamp.dt.normalize(), stamp=peer_stamp).groupby("date", sort=True).agg(
        close=("close", "last"), last_stamp=("stamp", "last"), bars=("close", "size"))
    peer_close = peer.close.astype(float)
    assert peer_close.gt(0).all() and np.isfinite(peer_close.to_numpy()).all()
    seam = pd.DatetimeIndex(old.loc["2026-06-15":"2026-07-02"].index)
    assert len(seam) == plan["gates"]["seam_common_sessions"]
    seam_error = float(np.abs(old.reindex(seam).to_numpy(float) - peer_close.reindex(seam).to_numpy(float)).max())
    seam_ret_error = float(np.abs(old.reindex(seam).pct_change(fill_method=None).to_numpy(float)[1:] -
                                  peer_close.reindex(seam).pct_change(fill_method=None).to_numpy(float)[1:]).max())
    assert seam_error <= .0001 and seam_ret_error <= .000001
    actions = pd.read_parquet(root / source["peer_NVDA_actions"])
    actions["ex_div_date"] = pd.to_datetime(actions.ex_div_date)
    tail_actions = actions.loc[actions.ex_div_date.between("2026-07-03", "2026-10-07")]
    assert tail_actions.split_ratio.isna().all()
    qlib.init(provider_uri=str((root / source["Stage1_SOXX_provider"]).resolve()), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(start_time="2010-03-11", end_time="2026-10-09", freq="day"))
    soxx = D.features(["SOXX"], ["$close"], start_time="2010-03-11",
                      end_time="2026-10-09", freq="day")["$close"].droplevel("instrument").reindex(calendar)
    assert soxx.notna().all() and soxx.gt(0).all()
    tail_dates = calendar[(calendar > pd.Timestamp("2026-07-02")) & (calendar <= pd.Timestamp("2026-10-07"))]
    assert peer_close.reindex(tail_dates).notna().all()
    assert peer.last_stamp.reindex(tail_dates).dt.strftime("%H:%M:%S").eq("16:00:00").all()
    assert peer.bars.reindex(tail_dates).eq(78).all()
    nvda = old.reindex(calendar)
    nvda.loc[tail_dates] = peer_close.reindex(tail_dates).to_numpy(float)
    assert nvda.loc["2011-01-03":"2026-10-07"].notna().all()
    provider = root / "data/qlib/stage1_nvda_sector_relative_lead_20261023"
    source_frame = pd.concat([
        pd.DataFrame({"datetime": calendar, "instrument": "SOXX", "close": soxx.to_numpy(float)}),
        pd.DataFrame({"datetime": calendar, "instrument": "NVDA", "close": nvda.to_numpy(float)}),
    ], ignore_index=True).dropna(subset=["close"])
    manifest = build_qlib_store_from_frame(source_frame, provider_uri=provider, freq="day", region="us",
        field_map={"$close": "close"}, as_of="2026-10-09",
        source="old_NVDA_Qlib_manifest=" + source["old_provider_manifest_sha256"] +
               ";tail_peer_NAS=" + source["NAS_verified_snapshot_source_id"] +
               ";SOXX_Stage1_manifest=" + source["Stage1_provider_manifest_sha256"])
    qlib.init(provider_uri=str(provider.resolve()), region="us", kernels=1)
    native = D.features(["NVDA", "SOXX"], ["$close"], start_time="2010-03-11",
                        end_time="2026-10-09", freq="day")["$close"].unstack("instrument").reindex(calendar)
    assert native.SOXX.notna().all()
    np.testing.assert_allclose(native.SOXX.to_numpy(float), soxx.to_numpy(float), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(native.NVDA.reindex(tail_dates).to_numpy(float),
                               peer_close.reindex(tail_dates).to_numpy(float), rtol=1e-6, atol=1e-5)
    lead = (np.log(native.NVDA.div(native.NVDA.shift(20))) -
            np.log(native.SOXX.div(native.SOXX.shift(20)))).rename("NVDA_minus_SOXX_log_momentum20")
    parent_card = root / source["original39_card"]
    parent_summary = json.loads((parent_card / "summary.json").read_text())
    assert parent_summary["recorder_id"] == source["original39_recorder_id"]
    with R.uri_context(str((parent_card / "qlib_recorder").resolve())):
        features = R.get_recorder(experiment_id=source["original39_experiment_id"],
            recorder_id=source["original39_recorder_id"]).load_object("inputs/features")
    base = list(parent_summary["input_audit"]["original_features"])
    assert len(base) == 39 and set(base) <= set(features.columns)
    pd.testing.assert_index_equal(features.index, calendar[calendar >= pd.Timestamp("2011-01-03")], check_names=False)
    coordinate = lead.reindex(features.index)
    train = features[base].loc["2011-01-03":"2022-12-30"].join(coordinate)
    held = coordinate.loc["2023-01-03":"2026-10-07"]
    max_corr = float(train.corr(method="spearman")[coordinate.name].loc[base].abs().max())
    gates = {"train_support": int(train[coordinate.name].notna().sum()) >= plan["gates"]["train_2011_2022_known_min"],
             "held_support": int(held.notna().sum()) >= plan["gates"]["held_2023_2026_known_min"],
             "seam_parity": seam_error <= .0001 and seam_ret_error <= .000001,
             "no_tail_split": bool(tail_actions.split_ratio.isna().all()),
             "novel_original39": max_corr < plan["gates"]["train_max_abs_spearman_original39_below"],
             "NAS_peer_hash": True, "Qlib_calendar_parity": True}
    decision = "conditional_source_novelty_pass_first_public_pending" if all(gates.values()) else "closed_prefit_NVDA_lead_source_or_novelty_failed"
    result = {"experiment": card.name, "decision": decision,
              "plan_sha256": _sha(card / "plan.json"),
              "old_provider_manifest_sha256": source["old_provider_manifest_sha256"],
              "peer_raw_sha256": source["peer_raw_NVDA_5min_sha256"],
              "peer_NAS_source_id": source["NAS_verified_snapshot_source_id"],
              "combined_provider_manifest_sha256": _sha(provider / "qlib_store_manifest.json"),
              "seam_common_sessions": len(seam), "seam_close_max_abs_difference": seam_error,
              "seam_return_max_abs_difference": seam_ret_error,
              "tail_sessions": len(tail_dates), "tail_corporate_action_rows": len(tail_actions),
              "tail_splits": int(tail_actions.split_ratio.notna().sum()),
              "train_rows": len(train), "train_known": int(train[coordinate.name].notna().sum()),
              "held_rows": len(held), "held_known": int(held.notna().sum()),
              "train_max_abs_spearman_original39": max_corr, "gates": gates,
              "historical_first_public_qualified": False,
              "future_return_reads": 0, "new_model_fits": 0,
              "new_native_accounts": 0, "new_market_requests": 0}
    with R.start(experiment_name=card.name, uri=str((card / "qlib_recorder").resolve())):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, provider_manifest=manifest)
        recorder.save_objects(artifact_path="analysis", lead=lead.reindex(features.index),
                              seam=pd.DataFrame({"old_Qlib": old.reindex(seam),
                                                 "peer_raw": peer_close.reindex(seam)}),
                              result=result)
        R.log_params(feature="NVDA_minus_SOXX_log_momentum20", parent="f696_A0", news_data=False)
        R.log_metrics(train_known=result["train_known"], held_known=result["held_known"],
                      max_abs_spearman_original39=max_corr)
        rid, eid = recorder.id, str(recorder.experiment_id)
    result.update(recorder_id=rid, experiment_id=eid)
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    binding = json.loads((root / source["peer_card"] / "nas_binding.json").read_text())
    binding.pop("evidence", None)
    (card / "nas_binding.json").write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n")
    archived = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
                                    card=card, recorder_id=rid, experiment_id=eid)
    remote_archive = _ssh(settings["ssh"], "sha256sum " + shlex.quote(archived["nas_path"])).split()[0]
    assert remote_archive == archived["sha256"]
    summary = {"experiment": card.name, "decision": decision, "objective_achieved": False,
               "plan_sha256": result["plan_sha256"], "result": "result.json",
               "recorder_id": rid, "experiment_id": eid,
               "NAS_evidence": archived, "NAS_readback": {"verified": True, "sha256": remote_archive},
               "new_model_fits": 0, "new_native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "analysis/lead", "analysis/seam"]},
               "limitations": ["NVDA price leadership is not its SOXX portfolio contribution without historical weights.",
                               "Yahoo-derived older Qlib prices and retrospective peer Futu history do not prove original first-public vintages.",
                               "Source and novelty checks do not establish H5 predictive value or SOXX excess."]}
    (card / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nentrypoint: " + card.name + ".ipynb\n"
        "provider_uri: data/qlib/stage1_nvda_sector_relative_lead_20261023\nphase: source_only_no_future_label\n")
    (card / "conclusion.md").write_text(
        "# NVDA相对SOXX：单一龙头领导性来源与去重\n\n"
        "只读旧Qlib NVDA日线截至2026-07-02，以及Yanxuan NAS经哈希验证的NVDA原价5分钟末bar"
        "补齐至f696末端信号2026-10-07。重叠13个会话不拟合换算比率；"
        f"最大收盘绝对差 {seam_error:.8f}、收益差 {seam_ret_error:.9f}。"
        "后段无拆分，现金分红按价格收益口径保留，不冒充总收益。\n\n"
        f"过去20会话NVDA减SOXX对数动量在训练 {result['train_known']}/{len(train)}、"
        f"复用评价 {result['held_known']}/{len(held)} 已知；原39项最高训练绝对秩相关 {max_corr:.4f}。"
        f"七项来源/去重门 {sum(gates.values())}/7，决定 `{decision}`。"
        "这不是NVDA对SOXX实际贡献，也没有未来标签、拟合或账户。\n\n"
        "旧Yahoo/Qlib及当前Futu历史最初公开版本未认证，通过只允许另冻SOXL相对H5信息门；"
        "任何经济路径仍须完整f696自身状态0bp Tiered账户及SOXX v2所有门。"
        f"Recorder `{rid}`，NAS SHA `{remote_archive}`。\n")
    return summary
