"""Build and record source-consistent Qlib price views for native accounting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.native_execution_marks import build_coherent_daily_view, build_execution_mark_view, build_repaired_intraday_view, build_uniform_intraday_view


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_native_price_views(*, root: Path, card_dir: Path) -> dict:
    import qlib
    from qlib.data import D
    from qlib.workflow import R

    root, card_dir = Path(root).resolve(), Path(card_dir).resolve()
    plan_path = card_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    for source in plan["sources"].values():
        manifest = root / source["provider_uri"] / "qlib_store_manifest.json"
        if _sha(manifest) != source["manifest_sha256"]:
            raise ValueError("blocked: source price manifest changed after plan freeze")
    daily = root / plan["sources"]["daily"]["provider_uri"]
    intraday = root / plan["sources"]["intraday"]["provider_uri"]
    execution = root / plan["outputs"]["execution_provider"]
    coherent = root / plan["outputs"]["daily_provider"]
    start, end, as_of = plan["window"]["start"], plan["window"]["end"], plan["window"]["as_of"]
    repair = plan.get("observed_repair")
    if repair:
        repair_manifest = root / repair["manifest"]
        if _sha(repair_manifest) != repair["manifest_sha256"]:
            raise ValueError("blocked: observed repair manifest changed after plan freeze")
        intraday = root / repair["provider_uri"]
        build_repaired_intraday_view(cache_path=root / repair["cache_path"], repair_manifest=repair_manifest,
                                     provider_uri=intraday, start=repair["start"], as_of=as_of)
    uniform = plan.get("uniform_vintage")
    if uniform:
        manifest = root / uniform["source_manifest"]
        if _sha(manifest) != uniform["source_manifest_sha256"]:
            raise ValueError("blocked: uniform source manifest changed after plan freeze")
        intraday = root / uniform["provider_uri"]
        build_uniform_intraday_view(source_manifest=manifest, trading_calendar=root / uniform["trading_calendar"],
                                    provider_uri=intraday, start=uniform["start"], as_of=as_of)
    qlib.init(provider_uri=str(daily), region="us", kernels=1)
    old = D.features(["SOXL", "SOXX"], ["$open"], start_time=start,
                     end_time=pd.Timestamp(end).normalize(), freq="day").reset_index()
    old["datetime"] = pd.to_datetime(old.datetime).dt.normalize()
    execution_manifest = build_execution_mark_view(source_provider=intraday, provider_uri=execution,
                                                    start=start, end=end, as_of=as_of)
    daily_manifest = build_coherent_daily_view(daily_provider=daily, intraday_provider=intraday,
                                              provider_uri=coherent, start=start, end=end, as_of=as_of)
    new = D.features(["SOXL", "SOXX"], ["$open"], start_time=start,
                     end_time=pd.Timestamp(end).normalize(), freq="day").reset_index()
    new["datetime"] = pd.to_datetime(new.datetime).dt.normalize()
    paired = old.merge(new, on=["instrument", "datetime"], suffixes=("_original", "_coherent"), validate="one_to_one")
    if len(paired) != len(old) or not np.isfinite(paired[["$open_original", "$open_coherent"]]).all().all():
        raise ValueError("blocked: coherent daily calendar or observed opening price missing")
    paired["relative_open_change"] = paired["$open_coherent"] / paired["$open_original"] - 1.0
    aggregates = []
    for symbol, block in paired.groupby("instrument"):
        delta = block.relative_open_change.abs()
        aggregates.append({"instrument": symbol, "sessions": len(block),
                           "maximum_relative_open_change": float(delta.max()),
                           "median_relative_open_change": float(delta.median()),
                           "over_one_bp_sessions": int((delta > 1e-4).sum())})
    recorder_uri = card_dir / "qlib_recorder"
    with R.start(experiment_name=plan["experiment"], recorder_name="price_source_validation", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(plan_sha256=_sha(plan_path), price_change_scope="observed price source only", models_trained=0)
        recorder.save_objects(**{"price_source_comparison.pkl": paired,
                                 "provider_manifests.pkl": {"execution": execution_manifest, "daily": daily_manifest}})
        recorder.log_artifact(str(plan_path), artifact_path="plan")
        identity = {"recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
                    "recorder_uri": str(recorder_uri), "recorder_local_dir": str(recorder_uri / str(recorder.experiment_id) / recorder.id)}
    summary = {"experiment": plan["experiment"], "decision": "coherent_price_views_built_not_strategy_validated",
               "promotion_eligible": False, "objective_achieved": False, "plan_sha256": _sha(plan_path),
               "provider_uri": str(coherent), "intraday_provider_uri": str(execution), "region": "us",
               "price_source_comparison": aggregates, "source_ohlcv_unchanged_in_execution_view": True,
               "intraday_price_source_from": daily_manifest["intraday_price_source_from"], **identity,
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["price_source_comparison.pkl", "provider_manifests.pkl"]},
               "qlib_docs_reference": ["https://qlib.readthedocs.io/en/latest/component/data.html",
                                       "https://qlib.readthedocs.io/en/latest/component/strategy.html"],
               "limitations": ["Only source consistency is verified; no model/policy improvement is inferred.",
                               "Historical opening and first-hour-close marks do not establish executable 09:31/10:31 fills."]}
    (card_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    pd.DataFrame(aggregates).to_csv(card_dir / "price_source_comparison.csv", index=False)
    import yaml
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump({"experiment": plan["experiment"],
        "plan_sha256": summary["plan_sha256"], "provider_uri": str(coherent), "intraday_provider_uri": str(execution),
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    conclusion = ("# 同源价格与原生账户验证准备（中文结论）\n\n"
                  "通过 Qlib Data API 读取真实日线和 15 分钟 OHLCV，构建有来源哈希的价格视图。\n"
                  "执行视图逐值保留原始 OHLCV，开盘与首小时收盘用真实观察价；日线前缀保持早期来源，"
                  "自实际日内数据起点起使用日内真实 OHLCV 聚合，避免跨频账户边界价格不一致。\n\n"
                  f"原始日线开盘价与新视图差异：{aggregates}。\n\n"
                  "决定：不晋升。本卡仅证明价格来源和视图读回，不证明策略收益、控制器正确性或真实成交。"
                  "下一步用冻结动作与费用一致候选分别重放，在 Qlib 原生账户中核现金、股数、费用和逐日结算。\n")
    (card_dir / "conclusion.md").write_text(conclusion)
    return summary
