"""Exact original-filing TSM FY2024/FY2025 revenue-tag source bridge."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pandas as pd

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.inventory_sales_original_value import _comparable_pairs, _monthly_source_support, _original_fact
from src.workflow.inventory_sales_source_qualification import normalized_notebook

NAME = "inventory_sales_tsm_revenue_bridge_20261022"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_originals(root, plan):
    source = plan["original_archive"]
    archive_path = Path(source["local_archive"])
    require(sha(root / source["manifest"]) == source["manifest_sha256"], "archive manifest changed")
    require(sha(archive_path) == source["archive_sha256"], "archive bytes changed")
    parent_plan = json.loads((root / plan["source"]["original_plan"]).read_text())
    specs = {y: next(x for x in parent_plan["issuer_years"] if x["ticker"] == "TSM" and x["fiscal_year"] == y)
             for y in (2024, 2025)}
    content = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for year in (2024, 2025):
            item = source["members"][str(year)]
            member = archive.getmember(item["member"])
            require(member.isfile() and member.size == item["bytes"], "original member size differs")
            stream = archive.extractfile(member)
            require(stream is not None, "original member absent")
            content[year] = stream.read()
            require(hashlib.sha256(content[year]).hexdigest() == item["sha256"], "original member hash differs")
    rule = plan["frozen_rule"]
    old = [_original_fact(content[2024], specs[2024], tag, "revenue") for tag in rule["FY2024_tags"]]
    require(all(values == {rule["FY2024_equal_value"]} for values, _ in old)
            and old[0][1] == old[1][1] and len(old[0][1]) == 1, "FY2024 original alias differs")
    revenue, contexts = _original_fact(content[2025], specs[2025], rule["FY2025_revenue_tag"], "revenue")
    inventory, _ = _original_fact(content[2025], specs[2025], rule["FY2025_inventory_tag"], "inventory")
    require(revenue == {rule["FY2025_revenue_value"]} and len(contexts) == 1
            and inventory == {rule["FY2025_inventory_value"]}, "FY2025 original facts differ")
    return sorted(contexts)


def run(root):
    import qlib
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R

    root = Path(root).resolve()
    card = root / "research" / NAME
    plan_path = card / "plan.json"
    plan = json.loads(plan_path.read_text())
    require(plan["frozen"] and plan["execution_authorized"], "frozen plan absent")
    require(not (card / "run_state.json").exists(), "attempt already started")
    require(sha(Path(__file__)) == plan["implementation"]["module_sha256"], "module bytes differ")
    notebook = card / (NAME + ".ipynb")
    require(hashlib.sha256(normalized_notebook(notebook)).hexdigest()
            == plan["implementation"]["notebook_sha256"], "notebook source differs")
    source = plan["source"]
    old = root / source["card"]
    require(sha(old / "summary.json") == source["summary_sha256"]
            and sha(old / "verification_manifest.json") == source["verification_manifest_sha256"]
            and sha(root / source["original_plan"]) == source["original_plan_sha256"],
            "parent source identity differs")
    for item in plan["evidence"].values():
        require(sha(root / item["path"]) == item["sha256"], "attribution identity differs")
    bridge = json.loads((root / plan["evidence"]["FY2024_exact_bridge"]["path"]).read_text())
    require(bridge["same_context_same_value"], "original alias bridge absent")
    revenue_contexts = checked_originals(root, plan)
    (card / "run_state.json").write_text(json.dumps({"status": "started", "plan_sha256": sha(plan_path)}) + "\n")
    old_plan = json.loads((root / source["original_plan"]).read_text())
    qlib.init(provider_uri=str(root / old_plan["calendar"]["provider_uri"]), region="us", kernels=1,
              expression_cache=None, dataset_cache=None)
    objects, audit = load_verified_recorder_objects(
        old, expected_recorder_id=source["recorder_id"], artifacts=list(source["artifacts"]),
        expected_artifact_sha256=source["artifacts"],
        expected_manifest_sha256=source["verification_manifest_sha256"])
    annual_old = objects["analysis/original_annual_values"]
    pairs_old = objects["analysis/original_comparable_pairs"]
    support_old = objects["analysis/four_name_monthly_support"]
    rule = plan["frozen_rule"]
    target = annual_old.ticker.eq("TSM") & annual_old.fiscal_year.eq(2025)
    require(target.sum() == 1 and annual_old.loc[target, "status"].iloc[0] == "missing_or_conflicting_original_fact",
            "old TSM missing row differs")
    annual = annual_old.copy(deep=True)
    idx = annual.index[target][0]
    annual.at[idx, "inventory"] = rule["FY2025_inventory_value"]
    annual.at[idx, "revenue"] = rule["FY2025_revenue_value"]
    annual.at[idx, "status"] = "original_own_period_qualified"
    annual.at[idx, "revenue_contexts"] = {rule["FY2025_revenue_tag"]: revenue_contexts}
    pd.testing.assert_frame_equal(annual_old.loc[~target].reset_index(drop=True),
                                  annual.loc[~target].reset_index(drop=True))
    pairs = _comparable_pairs(annual, old_plan)
    old_keys = {(x.ticker, x.prior_fiscal_end, x.fiscal_end) for x in pairs_old.itertuples()}
    new_keys = {(x.ticker, x.prior_fiscal_end, x.fiscal_end) for x in pairs.itertuples()}
    expected = {("TSM", pd.Timestamp(rule["expected_added_pair"][0]),
                 pd.Timestamp(rule["expected_added_pair"][1]))}
    require(new_keys - old_keys == expected and old_keys <= new_keys and len(pairs) == len(pairs_old) + 1,
            "unexpected original annual pair change")
    support = _monthly_source_support(support_old, pairs, old_plan["source_rules"]["max_pair_age_days"])
    pd.testing.assert_frame_equal(support_old.loc[support_old.ticker.ne("TSM")].reset_index(drop=True),
                                  support.loc[support.ticker.ne("TSM")].reset_index(drop=True),
                                  check_dtype=False)
    before = support_old.loc[support_old.ticker.eq("TSM"), ["datetime", "source_supported_weight_pct"]]
    after = support.loc[support.ticker.eq("TSM"), ["datetime", "source_supported_weight_pct"]]
    matched = before.merge(after, on="datetime", suffixes=("_old", "_new"), validate="one_to_one")
    require((matched.source_supported_weight_pct_new >= matched.source_supported_weight_pct_old).all(),
            "TSM monthly source support degraded")
    recovered = matched.loc[matched.source_supported_weight_pct_old.eq(0)
                            & matched.source_supported_weight_pct_new.gt(0)]
    panel = support[["datetime", "instrument", "weight_pct", "source_supported_weight_pct"]]
    panel = panel.set_index(["datetime", "instrument"]).sort_index()
    dataset = DatasetH(handler=DataHandlerLP.from_df(pd.concat({"meta": panel}, axis=1)),
                       segments={"all": (None, None)})
    dataset.config(dump_all=True, recursive=True)
    pd.testing.assert_frame_equal(dataset.prepare("all", col_set="meta"), panel)
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=NAME, uri=str(tracking)):
        rec = R.get_recorder()
        rec.save_objects(artifact_path="inputs", plan=plan, source_audit=audit,
                         original_bridge_contexts=revenue_contexts)
        rec.save_objects(artifact_path="analysis", bridged_annual_values=annual,
                         bridged_comparable_pairs=pairs, four_name_monthly_support=support,
                         recovered_TSM_months=recovered)
        rec.save_objects(artifact_path="metadata", bridge_dataset=dataset)
        rec.save_objects(artifact_path="sources", module=Path(__file__).read_bytes(), notebook=notebook.read_bytes())
        pd.testing.assert_frame_equal(rec.load_object("metadata/bridge_dataset").prepare("all", col_set="meta"),
                                      panel)
        rec.log_metrics(added_original_annual_pairs=1, recovered_TSM_months=len(recovered),
                        historical_PIT_rows=0, model_fits=0, native_accounts=0)
        summary = {"experiment": NAME, "status": "completed_source_only_not_promoted",
                   "decision": "exact_TSM_original_revenue_alias_repaired_only",
                   "original_issuer_years": len(annual),
                   "own_original_value_years": int(annual.status.eq("original_own_period_qualified").sum()),
                   "original_annual_pairs": len(pairs), "annual_pairs_added": 1,
                   "four_name_issuer_months": len(support),
                   "four_name_months_with_original_pair": int(support.source_supported_weight_pct.gt(0).sum()),
                   "TSM_conditionally_recovered_months": len(recovered),
                   "old_pair_keys_preserved": True, "non_TSM_support_unchanged": True,
                   "strict_PIT_qualified_rows": 0, "model_input_qualified": False,
                   "factor_effectiveness_tested": False, "SOXX_excess_established": False,
                   "D_calendar_calls": 0, "market_price_or_label_reads": 0,
                   "source_recorder_object_loads": len(source["artifacts"]),
                   "original_source_member_reads": 2, "model_fits": 0, "native_accounts": 0,
                   "source_recorder_id": source["recorder_id"], "plan_sha256": sha(plan_path),
                   "module_sha256": sha(Path(__file__)), "recorder_id": rec.id,
                   "experiment_id": str(rec.experiment_id), "recorder_uri": str(tracking),
                   "recorder_local_dir": str(tracking / str(rec.experiment_id) / rec.id),
                   "artifact_policy": plan["artifact_policy"]}
        rec.save_objects(artifact_path="analysis", summary=summary)
    artifact_root = Path(summary["recorder_local_dir"]) / "artifacts"
    manifest = {"recorder_id": summary["recorder_id"], "entries": [
        {"path": str(path.relative_to(card)), "sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(artifact_root.rglob("*")) if path.is_file()]}
    (card / "verification_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (card / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (card / "run_state.json").write_text(json.dumps({"status": "completed", "recorder_id": rec.id,
                                                     "plan_sha256": summary["plan_sha256"]}) + "\n")
    (card / "conclusion.md").write_text(
        "# TSM 原始收入标签桥：来源限定结果\n\n"
        f"Qlib Recorder：`{rec.id}`。FY2024 原始同上下文/同额的两个收入标签支持精确语义桥；"
        f"只补入 FY2025 原始申报的同概念收入，使四家原始数值年度达 {summary['own_original_value_years']}/"
        f"{summary['original_issuer_years']}、可比年度配对达 {len(pairs)}，"
        f"四家披露证券有 {summary['four_name_months_with_original_pair']}/{len(support)} 个发行人月份"
        f"具备未超龄的原始配对，其中 TSM 恢复 {len(recovered)} 个月。"
        "既有配对和其他发行人不变，篮子权重不重新归一化。\n\n"
        "这仍是来源配对修复；首次公开版本、完整篮子 PIT 模型输入和因子预测性未认证。"
        "本卡没有计算收益、费用账户或 SOXX 超额，不能晋升。\n")
    return summary
