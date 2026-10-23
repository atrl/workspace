"""Native Qlib source and novelty preflight for SOXX turnover surprise."""

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

    from src.data.minute_intake import atomic_json
    from src.data.nas_market_data import _settings, _ssh, archive_nas_recorder
    from src.factor.turnover_surprise import prior_median_log_turnover_surprise

    root = Path(root).resolve()
    card = root / "research/stage1_futu_turnover_surprise_preflight_20261023"
    plan = json.loads((card / "plan.json").read_text())
    source = plan["sources"]
    assert plan["frozen"] and not (card / "summary.json").exists()
    digest = _sha(card / "plan.json")
    turnover_card, parent_card = root / source["turnover_card"], root / source["parent_card"]
    assert _sha(turnover_card / "summary.json") == source["turnover_summary_sha256"]
    assert _sha(parent_card / "summary.json") == source["parent_summary_sha256"]
    ts = json.loads((turnover_card / "summary.json").read_text())
    ps = json.loads((parent_card / "summary.json").read_text())
    assert ts["recorder_id"] == source["turnover_recorder_id"]
    assert ts["NAS_evidence"]["sha256"] == source["turnover_NAS_sha256"]
    assert ps["recorder_id"] == source["parent_recorder_id"]
    assert ps["nas_evidence"]["sha256"] == source["parent_NAS_sha256"]
    assert _sha(root / "data/qlib/stage1_futu_turnover_20261023/qlib_store_manifest.json") == source["turnover_provider_manifest_sha256"]
    qlib.init(provider_uri=str((root / "data/qlib/stage1_futu_turnover_20261023").resolve()), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(start_time="2011-01-03", end_time="2026-10-09", freq="day"))
    turnover = D.features(["SOXX"], ["$turnover_rate"],
        start_time="2010-03-11", end_time="2026-10-09", freq="day")["$turnover_rate"].droplevel("instrument").sort_index()
    coordinate = prior_median_log_turnover_surprise(turnover, lookback=20).reindex(calendar)
    qlib.init(provider_uri=str((root / "data/qlib/minute_information_pit_allocator_20261012/daily_prices").resolve()),
              region="us", kernels=1)
    close = D.features(["SOXX"], ["$close"],
        start_time="2010-03-11", end_time="2026-10-09", freq="day")["$close"].droplevel("instrument").sort_index()
    return1 = close.pct_change(fill_method=None).rename("SOXX_return1").reindex(calendar)
    with R.uri_context(str((parent_card / "qlib_recorder").resolve())):
        features = R.get_recorder(experiment_name=parent_card.name,
            recorder_id=source["parent_recorder_id"]).load_object("inputs/features")
    pd.testing.assert_index_equal(features.index, calendar, check_names=False)
    base = list(ps["input_audit"]["original_features"])
    assert len(base) == 39 and set(base) <= set(features.columns)
    frame = features[base].join(return1).join(coordinate)
    train = frame.loc[plan["cohorts"]["train"][0]:plan["cohorts"]["train"][1]]
    held = frame.loc[plan["cohorts"]["reused_held"][0]:plan["cohorts"]["reused_held"][1]]
    corr = train.corr(method="spearman")[coordinate.name]
    abs_max_base = float(corr.loc[base].abs().max())
    abs_return1 = float(abs(corr.loc["SOXX_return1"]))
    protective = coordinate.loc[plan["cohorts"]["motivating_protective_interval"][0]:
                                plan["cohorts"]["motivating_protective_interval"][1]]
    cash_loss = coordinate.loc[plan["cohorts"]["motivating_cash_loss_interval"][0]:
                               plan["cohorts"]["motivating_cash_loss_interval"][1]]
    annual = coordinate.groupby(coordinate.index.year).agg(["count", "mean", "median"])
    quarterly = coordinate.groupby(coordinate.index.to_period("Q")).agg(["count", "mean", "median"])
    gates = {"train_support": int(train[coordinate.name].notna().sum()) >= plan["cohorts"]["minimum_train_known"],
             "held_support": int(held[coordinate.name].notna().sum()) >= plan["cohorts"]["minimum_held_known"],
             "not_redundant_to_original39": abs_max_base < .80,
             "not_redundant_to_return1": abs_return1 < .80,
             "protective_interval_higher_shock": float(protective.mean()) > float(cash_loss.mean())}
    decision = ("conditional_feature_novelty_pass_before_information_fit" if all(gates.values())
                else "closed_before_fit_fixed_turnover_surprise_mechanism_or_novelty_failed")
    result = {"experiment": card.name, "decision": decision, "plan_sha256": digest,
              "turnover_source_recorder_id": source["turnover_recorder_id"],
              "parent_recorder_id": source["parent_recorder_id"],
              "calendar_rows": len(calendar), "train_rows": len(train),
              "train_known": int(train[coordinate.name].notna().sum()),
              "held_rows": len(held), "held_known": int(held[coordinate.name].notna().sum()),
              "train_max_abs_spearman_original39": abs_max_base,
              "train_abs_spearman_return1": abs_return1,
              "protective_2024Q3_mean": float(protective.mean()),
              "cash_loss_2025_mean": float(cash_loss.mean()),
              "gates": gates, "Qlib_D_calendar_calls": 1, "Qlib_D_feature_calls": 2,
              "parent_Recorder_object_loads": 1, "future_return_reads": 0,
              "model_fits": 0, "native_accounts": 0, "new_market_requests": 0,
              "historical_first_public_qualified": False}
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    annual.reset_index(names="year").to_csv(card / "annual_source_summary.csv", index=False)
    quarterly.reset_index(names="quarter").to_csv(card / "quarterly_source_summary.csv", index=False)
    R.set_uri(str((card / "qlib_recorder").resolve()))
    with R.start(experiment_name=card.name):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, source_identities=source)
        recorder.save_objects(artifact_path="analysis", coordinate=coordinate,
                              annual=annual, quarterly=quarterly, result=result)
        R.log_params(coordinate="SOXX_prior20_log_turnover_median_surprise", no_future_labels=True)
        R.log_metrics(train_known=result["train_known"], held_known=result["held_known"],
                      train_max_abs_spearman_original39=abs_max_base)
        rid, eid = recorder.id, str(recorder.experiment_id)
    result.update(recorder_id=rid, experiment_id=eid)
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    binding = json.loads((turnover_card / "nas_binding.json").read_text())
    binding.pop("evidence", None)
    atomic_json(binding, card / "nas_binding.json")
    nas = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
                               card=card, recorder_id=rid, experiment_id=eid)
    remote = _ssh(_settings(root)["ssh"], "sha256sum " + shlex.quote(nas["nas_path"])).split()[0]
    assert remote == nas["sha256"]
    summary = {"experiment": card.name, "decision": decision,
               "objective_achieved": False, "plan_sha256": digest,
               "result": "result.json", "recorder_id": rid, "experiment_id": eid,
               "new_fits": 0, "new_native_accounts": 0,
               "NAS_evidence": nas, "NAS_readback": {"verified": True, "sha256": remote},
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "analysis/coordinate",
                                                               "analysis/annual", "analysis/quarterly"]},
               "limitations": ["Historical Futu turnover-rate first-public/revision vintages remain unqualified.",
                               "Named regime contrasts use previously reused market history; no switch is selected.",
                               "No future return, 0bp account, 2/5bp or independent forward evidence."]}
    atomic_json(summary, card / "summary.json")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nplan: plan.json\n"
        "source: research/stage1_futu_daily_turnover_source_20261023\nphase: feature_source_only_no_fit\n")
    (card / "conclusion.md").write_text(
        "# SOXX 换手率异常：Stage1 决策前表示与区间对照\n\n"
        "原生 Qlib 同日换手率减去严格前20个交易日对数中位数，作为前一信号日已完成信息供次日09:29配置。"
        "原39项、单日收益均只作同源去重对照；不读取未来收益、拟合或账户。\n\n"
        f"训练 {result['train_known']}/{result['train_rows']}、复用历史 {result['held_known']}/{result['held_rows']} 有值。"
        f"对原39项最大绝对 Spearman {abs_max_base:.4f}，对单日SOXX收益 {abs_return1:.4f}。"
        f"预定2024Q3与2025原值均值 {result['protective_2024Q3_mean']:.6f}/"
        f"{result['cash_loss_2025_mean']:.6f}；五项门 {sum(gates.values())}/5，决定 `{decision}`。\n\n"
        "这些已知区间只是机制方向的必要诊断，不产生开关或独立样本。历史Futu原始换手率首次公开版本未知；"
        "来源/去重通过也只允许另冻信息检验，不是0bp费用后SOXX超额。"
        f"Recorder `{rid}`，NAS SHA `{remote}`。\n")
    return summary
