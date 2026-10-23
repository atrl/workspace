"""Futu SOXX daily turnover-rate source intake and native Qlib coverage audit."""

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
    card = root / "research/stage1_futu_daily_turnover_source_20261023"
    plan = json.loads((card / "plan.json").read_text())
    assert plan["frozen"]
    assert not (card / "summary.json").exists()
    digest = _sha(card / "plan.json")
    raw_root = root / "data/raw/external/stage1_futu_turnover_20261023"
    request = json.loads((raw_root / "request_receipt.json").read_text())
    nas = json.loads((card / "nas_raw_archive.json").read_text())
    raw_path = raw_root / "SOXX_NONE_day.csv"
    assert request["status"] == "captured" and nas["status"] == "verified"
    assert request["SDK_version"] and request["request"]["autype"] == "AuType.NONE"
    assert _sha(raw_path) == request["data_sha256"]
    assert len(request["pages"]) <= plan["budget"]["Futu_pages_max"]
    assert sum(page["rows"] for page in request["pages"]) == request["rows"]
    raw = pd.read_csv(raw_path)
    assert len(raw) == request["rows"]
    assert raw.code.eq("US.SOXX").all()
    date = pd.to_datetime(raw.time_key).dt.normalize()
    assert date.is_unique and date.is_monotonic_increasing
    if not (np.isfinite(raw.turnover_rate.to_numpy(float)).all() and
            raw.turnover_rate.ge(0).all()):
        raise ValueError("Futu turnover_rate has missing or negative values")
    provider = root / "data/qlib/stage1_futu_turnover_20261023"
    source_frame = pd.DataFrame({"datetime": date, "instrument": "SOXX",
                                 "turnover_rate": raw.turnover_rate.to_numpy(float)})
    manifest = build_qlib_store_from_frame(
        source_frame, provider_uri=provider, freq="day", region="us",
        field_map={"$turnover_rate": "turnover_rate"}, as_of=plan["request"]["end"],
        source="Futu request_history_kline US.SOXX K_DAY AuType.NONE; raw NAS SHA256=" + nas["NAS_archive_sha256"],
    )
    qlib.init(provider_uri=str(provider.resolve()), region="us", kernels=1)
    source_calendar = pd.DatetimeIndex(D.calendar(
        start_time="2011-01-03", end_time=plan["request"]["end"], freq="day"))
    source_values = D.features(["SOXX"], ["$turnover_rate"],
        start_time="2011-01-03", end_time=plan["request"]["end"], freq="day")
    native = source_values["$turnover_rate"].droplevel("instrument").sort_index()
    assert native.index.equals(source_calendar)
    np.testing.assert_allclose(native.to_numpy(float),
        source_frame.set_index("datetime").turnover_rate.reindex(native.index).to_numpy(float),
        rtol=1e-6, atol=1e-8)
    qlib.init(provider_uri=str((root / "data/qlib/minute_information_pit_allocator_20261012/daily_prices").resolve()),
              region="us", kernels=1)
    parent_calendar = pd.DatetimeIndex(D.calendar(
        start_time="2011-01-03", end_time=plan["request"]["end"], freq="day"))
    known = native.reindex(parent_calendar)
    train = known.loc["2011-01-03":"2022-12-30"]
    held = known.loc["2023-01-03":plan["request"]["end"]]
    gates = {"train_support": int(train.notna().sum()) >= plan["numerical_gate"]["training_2011_2022_known_min"],
             "held_support": int(held.notna().sum()) >= plan["numerical_gate"]["reused_held_2023_2026_known_min"],
             "no_duplicate_dates": bool(date.is_unique),
             "finite_nonnegative_turnover_rate": bool(np.isfinite(raw.turnover_rate.to_numpy(float)).all() and raw.turnover_rate.ge(0).all())}
    decision = ("conditional_source_coverage_pass_first_public_vintage_unknown" if all(gates.values())
                else "needs_source_coverage_no_factor_fit")
    result = {"experiment": card.name, "decision": decision, "plan_sha256": digest,
              "Futu_SDK_version": request["SDK_version"], "Futu_request_receipt": "data/raw/external/stage1_futu_turnover_20261023/request_receipt.json",
              "raw_sha256": request["data_sha256"], "raw_bytes": request["data_bytes"],
              "raw_NAS_archive_sha256": nas["NAS_archive_sha256"],
              "Qlib_provider_uri": str(provider), "Qlib_provider_manifest_sha256": _sha(provider / "qlib_store_manifest.json"),
              "Qlib_source_calendar_rows": len(source_calendar), "Stage1_parent_calendar_rows": len(parent_calendar),
              "train_calendar": len(train), "train_known": int(train.notna().sum()),
              "held_calendar": len(held), "held_known": int(held.notna().sum()),
              "missing_Stage1_dates": [str(x.date()) for x in known.index[known.isna()]],
              "turnover_rate_min": float(native.min()), "turnover_rate_median": float(native.median()),
              "turnover_rate_max": float(native.max()), "gates": gates,
              "new_market_requests": len(request["pages"]), "new_model_fits": 0,
              "new_native_accounts": 0, "future_return_reads": 0,
              "historical_first_public_qualified": False}
    R.set_uri(str((card / "qlib_recorder").resolve()))
    with R.start(experiment_name=card.name):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, request=request,
                              nas_raw_archive=nas, Qlib_provider_manifest=manifest)
        recorder.save_objects(artifact_path="analysis", native_turnover_rate=native,
                              Stage1_calendar_coverage=known, result=result)
        R.log_params(source="Futu_US_SOXX_K_DAY_AuType_NONE", news_data=False)
        R.log_metrics(train_known=result["train_known"], held_known=result["held_known"])
        rid, eid = recorder.id, str(recorder.experiment_id)
    result.update(recorder_id=rid, experiment_id=eid)
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    binding = json.loads((root / "research/minute_information_pit_allocator_20261012/nas_binding.json").read_text())
    binding.pop("evidence", None)
    (card / "nas_binding.json").write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n")
    archived = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
                                    card=card, recorder_id=rid, experiment_id=eid)
    remote = _ssh(_settings(root)["ssh"], "sha256sum " + shlex.quote(archived["nas_path"])).split()[0]
    assert remote == archived["sha256"]
    summary = {"experiment": card.name, "decision": decision, "objective_achieved": False,
               "plan_sha256": digest, "result": "result.json", "recorder_id": rid,
               "experiment_id": eid, "NAS_evidence": archived,
               "NAS_readback": {"verified": True, "sha256": remote},
               "new_model_fits": 0, "new_native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "analysis/native_turnover_rate",
                                                               "analysis/Stage1_calendar_coverage"]},
               "limitations": ["Futu current-history turnover_rate is not proven to be the value first public on each historical day.",
                               "Field is vendor-reported ETF turnover, not broker order flow or a predictive signal.",
                               "Reused 2023-2026 history is development feedback, not independent forward."]}
    (card / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nplan: plan.json\n"
        "provider_uri: data/qlib/stage1_futu_turnover_20261023\nphase: source_only_no_factor_fit\n")
    (card / "conclusion.md").write_text(
        "# Stage1 Futu SOXX 日换手率：原始来源与 Qlib 覆盖门\n\n"
        "以只读 Futu OpenAPI `US.SOXX` 日 K 线、明确 `AuType.NONE` 和五页固定预算取得当日已完成换手率。"
        "原响应 CSV 与请求收据按哈希备份 NAS，限定恢复逐件一致；随后建成独立原生 Qlib 日频来源。"
        "没有新闻数据、未来 SOXX 结果、模型拟合或资金账户。\n\n"
        f"Stage1 原日历训练 {result['train_known']}/{result['train_calendar']} 日有值，"
        f"复用历史评价 {result['held_known']}/{result['held_calendar']} 日有值；"
        f"缺日 {result['missing_Stage1_dates']}。数值范围 {result['turnover_rate_min']:.5f}–"
        f"{result['turnover_rate_max']:.5f}。四项来源门 {sum(gates.values())}/4，决定 `{decision}`。\n\n"
        "历史每日首次公开数值与供应商修订版本未核证；来源覆盖不证明特征增益或费用后 SOXX 超额。"
        "若继续，须另冻原39项受控信息检验，并在完整 Stage1-parent 自身状态0bp账户之前解决时点来源边界。"
        f"源 NAS SHA `{nas['NAS_archive_sha256']}`；Qlib Recorder `{rid}` NAS SHA `{remote}`。\n")
    return summary
