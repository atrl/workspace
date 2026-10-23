"""One fixed SOXX prior-session overnight/intraday split source screen."""

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

    root = Path(root).resolve()
    card = root / "research/stage1_soxx_prior_session_tug_source_20261023"
    plan = json.loads((card / "plan.json").read_text())
    source = plan["source"]
    if (card / "summary.json").exists():
        raise ValueError("fixed source screen already complete")
    assert plan["frozen"]
    assert _sha(root / source["provider_uri"] / "qlib_store_manifest.json") == source["provider_manifest_sha256"]
    parent_card = root / source["parent_card"]
    assert _sha(parent_card / "summary.json") == source["parent_summary_sha256"]
    parent = json.loads((parent_card / "summary.json").read_text())
    assert parent["recorder_id"] == source["parent_recorder_id"]
    assert parent["nas_evidence"]["sha256"] == source["parent_NAS_archive_sha256"]
    assert json.loads((parent_card / "nas_binding.json").read_text())["evidence"]["sha256"] == source["parent_NAS_archive_sha256"]
    qlib.init(provider_uri=str((root / source["provider_uri"]).resolve()), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(start_time="2010-03-11", end_time="2026-10-09", freq="day"))
    bars = D.features(["SOXX"], ["$open", "$close"],
                      start_time="2010-03-11", end_time="2026-10-09", freq="day")
    open_ = bars["$open"].droplevel("instrument").reindex(calendar)
    close = bars["$close"].droplevel("instrument").reindex(calendar)
    assert open_.notna().all() and close.notna().all()
    assert np.isfinite(open_.to_numpy(float)).all() and np.isfinite(close.to_numpy(float)).all()
    assert open_.gt(0).all() and close.gt(0).all()
    overnight = np.log(open_.div(close.shift(1)))
    intraday = np.log(close.div(open_))
    feature = (intraday - overnight).rename(plan["fixed_feature"]["name"])
    total = np.log(close.div(close.shift(1))).rename("SOXX_total_log_return1")
    with R.uri_context(str((parent_card / "qlib_recorder").resolve())):
        features = R.get_recorder(experiment_id=source["parent_experiment_id"],
            recorder_id=source["parent_recorder_id"]).load_object("inputs/features")
    base = list(parent["input_audit"]["original_features"])
    assert len(base) == 39 and set(base) <= set(features.columns)
    pd.testing.assert_index_equal(features.index, calendar[calendar >= pd.Timestamp("2011-01-03")], check_names=False)
    feature = feature.reindex(features.index)
    total = total.reindex(features.index)
    joined = features[base].join(feature).join(total)
    train = joined.loc["2011-01-03":"2022-12-30"]
    held = joined.loc["2023-01-03":"2026-10-09"]
    corr = train.corr(method="spearman")[feature.name]
    max_abs_base = float(corr.loc[base].abs().max())
    abs_total = float(abs(corr.loc[total.name]))
    gates = {"train_support": int(train[feature.name].notna().sum()) >= plan["source_novelty_gates"]["train_2011_2022_known_min"],
             "held_support": int(held[feature.name].notna().sum()) >= plan["source_novelty_gates"]["reused_held_2023_2026_known_min"],
             "positive_finite_open_close": True,
             "novel_original39": max_abs_base < plan["source_novelty_gates"]["max_abs_training_Spearman_original39_below"],
             "novel_total_return1": abs_total < plan["source_novelty_gates"]["max_abs_training_Spearman_total_SOXX_return1_below"]}
    decision = "source_novelty_pass_needs_separate_cash_H5_information" if all(gates.values()) else "closed_prefit_source_or_novelty_failed"
    result = {"experiment": card.name, "decision": decision,
              "plan_sha256": _sha(card / "plan.json"),
              "provider_manifest_sha256": source["provider_manifest_sha256"],
              "parent_recorder_id": source["parent_recorder_id"],
              "QLib_calendar_rows": len(calendar), "train_rows": len(train),
              "train_known": int(train[feature.name].notna().sum()),
              "held_rows": len(held), "held_known": int(held[feature.name].notna().sum()),
              "train_max_abs_spearman_original39": max_abs_base,
              "train_abs_spearman_total_return1": abs_total,
              "feature_min": float(feature.min()), "feature_median": float(feature.median()),
              "feature_max": float(feature.max()),
              "gates": gates, "future_return_reads": 0, "new_fits": 0,
              "new_accounts": 0, "new_market_requests": 0}
    with R.start(experiment_name=card.name, uri=str((card / "qlib_recorder").resolve())):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan)
        recorder.save_objects(artifact_path="analysis", feature=feature,
                              overnight=overnight.reindex(features.index),
                              intraday=intraday.reindex(features.index),
                              total=total, result=result)
        R.log_params(source="SOXX_QLib_split_continuous_daily_open_close", no_future_labels=True)
        R.log_metrics(train_known=result["train_known"], held_known=result["held_known"],
                      max_abs_spearman_original39=max_abs_base)
        rid, eid = recorder.id, str(recorder.experiment_id)
    result.update(recorder_id=rid, experiment_id=eid)
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    binding = json.loads((parent_card / "nas_binding.json").read_text())
    binding.pop("evidence", None)
    (card / "nas_binding.json").write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n")
    archived = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
                                    card=card, recorder_id=rid, experiment_id=eid)
    remote = _ssh(_settings(root)["ssh"], "sha256sum " + shlex.quote(archived["nas_path"])).split()[0]
    assert remote == archived["sha256"]
    summary = {"experiment": card.name, "decision": decision, "objective_achieved": False,
               "plan_sha256": result["plan_sha256"], "result": "result.json",
               "recorder_id": rid, "experiment_id": eid,
               "NAS_evidence": archived, "NAS_readback": {"verified": True, "sha256": remote},
               "new_model_fits": 0, "new_native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "analysis/feature",
                                                               "analysis/overnight", "analysis/intraday",
                                                               "analysis/total"]},
               "limitations": ["Source is a completed prior-session price decomposition, not an independently profitable factor.",
                               "Adjusted Qlib daily prices remain tied to the pinned provider snapshot.",
                               "No future label, full Tiered account or independent forward evidence."]}
    (card / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nentrypoint: " + card.name + ".ipynb\n"
        "provider_uri: " + source["provider_uri"] + "\nphase: source_only_no_future_label\n")
    (card / "conclusion.md").write_text(
        "# SOXX 前一完成日的日内与隔夜拉锯：Stage1 来源检验\n\n"
        "从原 Stage1 钉住的 Qlib 拆分连续 SOXX 日频开收盘构造当天日内对数收益减隔夜对数收益。"
        "只在下一交易日09:29配置时使用完整前一交易日信息；不触碰同日09:45分钟条。"
        "与原39项在相同2011–2022训练日做无结果去重。\n\n"
        f"训练 {result['train_known']}/{len(train)}、复用评价 {result['held_known']}/{len(held)} 有值。"
        f"原39项最大绝对训练秩相关 {max_abs_base:.4f}，与SOXX单日总对数收益绝对相关 {abs_total:.4f}。"
        f"来源/去重门 {sum(gates.values())}/{len(gates)}，决定 `{decision}`。\n\n"
        "这是来源和表示检验，不读未来收益，不拟合、不跑账户；通过也只允许另冻现金优势H5信息门。"
        f"Qlib Recorder `{rid}` 与NAS SHA `{remote}` 已校验。\n")
    return summary
