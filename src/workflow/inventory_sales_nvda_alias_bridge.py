"""Frozen source-only NVDA revenue-alias bridge for the three-asset campaign."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pandas as pd

from src.core.recorder_sources import load_verified_recorder_objects
from src.workflow.inventory_sales_source_qualification import annual_pairs, normalized_notebook


NAME = "inventory_sales_nvda_alias_bridge_20261022"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"invalid JSON object: {path}")
    return value


def _original_snapshot(root: Path, plan: dict) -> dict:
    spec = plan["original_snapshot"]
    manifest_path = root / spec["manifest"]
    archive_path = root / spec["archive"]
    _require(_sha(manifest_path) == spec["manifest_sha256"], "original snapshot manifest changed")
    _require(_sha(archive_path) == spec["archive_sha256"], "original filing archive changed")
    manifest = _json(manifest_path)
    _require(len(manifest["members"]) == spec["members"], "original filing inventory changed")
    with tarfile.open(archive_path, "r:gz") as archive:
        actual = {member.name: member for member in archive.getmembers()}
        _require(set(actual) == {item["member"] for item in manifest["members"]}, "archive members differ")
        for item in manifest["members"]:
            member = actual[item["member"]]
            _require(member.isfile() and member.size == item["bytes"], "archive member size differs")
            source = archive.extractfile(member)
            _require(source is not None, "archive member unreadable")
            _require(hashlib.sha256(source.read()).hexdigest() == item["sha256"], "original byte hash differs")
    return manifest


def bridge_first(first: pd.DataFrame, extraction: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """Apply only the exact original-value alias bridge; retain FY2018/FY2019 break."""
    rule = plan["frozen_bridge"]
    end = pd.Timestamp(rule["period_end"])
    start = pd.Timestamp(rule["period_start"])
    filed = pd.Timestamp(rule["filed"])
    base = extraction.loc[
        extraction.cik.eq(rule["cik"])
        & extraction.taxonomy.eq(rule["taxonomy"])
        & extraction.unit.eq(rule["unit"])
        & extraction.form.eq(rule["form"])
        & extraction.accn.eq(rule["original_accession"])
        & extraction.end.eq(end)
        & extraction.filed.eq(filed)
        & extraction.mechanical_eligible.eq(True)
    ]
    revenue = base.loc[base.field.eq("revenue") & base.start.eq(start)]
    _require(set(revenue.tag) == set(rule["revenue_tags"])
             and all(revenue.loc[revenue.tag.eq(tag), ["numeric", "accn", "form", "start", "end"]]
                     .drop_duplicates().shape[0] == 1 for tag in rule["revenue_tags"]),
             "FY2022 earliest original revenue alias inventory differs")
    _require(revenue.numeric.nunique() == 1 and int(revenue.numeric.iloc[0]) == rule["revenue_value_usd"],
             "FY2022 original revenue aliases differ in value")
    _require(first.loc[first.cik.eq(rule["cik"]) & first.field.eq("revenue") & first.end.eq(end)].empty,
             "frozen FY2022 revenue conflict is not absent from first-row ledger")
    inventory = first.loc[
        first.cik.eq(rule["cik"]) & first.field.eq("inventory")
        & first.end.eq(end) & first.accn.eq(rule["original_accession"])
        & first.unit.eq(rule["unit"]) & first.tag.eq(rule["inventory_tag"])
    ]
    _require(len(inventory) == 1 and int(inventory.numeric.iloc[0]) == rule["inventory_value_usd"],
             "FY2022 original inventory differs")
    alias_rows = [revenue.loc[revenue.tag.eq(tag)].iloc[0][first.columns].copy()
                  for tag in rule["revenue_tags"]]
    bridged = pd.concat([first.copy(), pd.DataFrame(alias_rows)], ignore_index=True)
    current = bridged.loc[bridged.cik.eq(rule["cik"]) & bridged.field.eq("revenue")
                          & bridged.end.eq(end)]
    _require(len(current) == 2 and set(current.tag) == set(rule["revenue_tags"]),
             "FY2022 original dual-tag bridge differs")
    prior = bridged.loc[
        bridged.cik.eq(rule["cik"]) & bridged.field.eq("revenue")
        & bridged.end.eq(pd.Timestamp(rule["accounting_break_excluded_pair"][0]))
    ]
    following = bridged.loc[
        bridged.cik.eq(rule["cik"]) & bridged.field.eq("revenue")
        & bridged.end.eq(pd.Timestamp(rule["accounting_break_excluded_pair"][1]))
    ]
    _require(len(prior) == len(following) == 1 and prior.tag.iloc[0] != following.tag.iloc[0],
             "pre-606 accounting boundary was crossed")
    return bridged


def _project_nvda_support(support: pd.DataFrame, pairs: pd.DataFrame, rule: dict, max_age: int) -> pd.DataFrame:
    rows = support.loc[support.ticker.eq("NVDA")].copy()
    _require(len(rows) > 0 and rows.datetime.is_unique, "NVDA monthly source rows missing or duplicated")
    candidate = pairs.loc[pairs.cik.eq(rule["cik"])].copy()
    output = []
    for row in rows.itertuples(index=False):
        date = pd.Timestamp(row.datetime)
        eligible = candidate.loc[candidate.conditional_available_session.le(date)]
        chosen = eligible.loc[eligible.fiscal_end.eq(eligible.fiscal_end.max())] if len(eligible) else eligible
        reason = ""
        if chosen.empty:
            reason = "no_conditionally_available_annual_pair"
        elif len(chosen) != 1:
            reason = "ambiguous_latest_fiscal_pair"
        elif not 0 <= (date - chosen.fiscal_end.iloc[0]).days <= max_age:
            reason = "latest_pair_stale_or_future_period"
        selected = chosen.iloc[0] if not reason else None
        output.append({"datetime": date, "ticker": row.ticker,
                       "weight_pct": float(row.weight_pct),
                       "old_reason": row.uncovered_reason,
                       "new_reason": reason,
                       "old_pair_accession": row.pair_accession,
                       "new_pair_accession": None if selected is None else selected.accession,
                       "old_conditional_weight_pct": float(row.weight_pct) if row.uncovered_reason == "" else 0.0,
                       "new_conditional_weight_pct": float(row.weight_pct) if selected is not None else 0.0,
                       "historical_identity_qualified": False,
                       "historical_PIT_qualified": False})
    result = pd.DataFrame(output)
    _require((result.new_conditional_weight_pct >= result.old_conditional_weight_pct).all(),
             "alias bridge degraded existing NVDA monthly support")
    return result


def run(root: Path, *, recovery: bool = False) -> dict:
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R

    root = Path(root).resolve()
    card = root / "research" / NAME
    plan_path = card / ("recovery_plan.json" if recovery else "plan.json")
    plan = _json(plan_path)
    _require(plan["frozen"] and plan["execution_authorized"], "frozen source plan absent")
    state_path = card / ("attempt2_run_state.json" if recovery else "run_state.json")
    _require(not state_path.exists(), "source bridge attempt already started")
    if recovery:
        previous = _json(card / "run_state.json")
        _require(previous.get("status") == "execution_failed" and previous.get("attempt") == 1,
                 "prior failed attempt state differs")
        _require(plan["recovery"]["attempt"] == plan["recovery"]["maximum_attempts"] == 2,
                 "recovery attempt cap differs")
        for name, key in (("plan.json", "attempt1_plan_sha256"),
                          ("attempt1_failure.json", "attempt1_failure_sha256"),
                          ("attempt1_attribution.json", "attempt1_attribution_sha256")):
            _require(_sha(card / name) == plan["recovery"][key], f"recovery binding differs: {name}")
    _require(_sha(Path(__file__)) == plan["implementation"]["module_sha256"], "module bytes differ")
    notebook_path = card / f"{NAME}.ipynb"
    _require(hashlib.sha256(normalized_notebook(notebook_path)).hexdigest()
             == plan["implementation"]["notebook_sha256"], "notebook code differs")
    source = plan["source"]
    old = root / source["card"]
    _require(_sha(old / "summary.json") == source["summary_sha256"], "source summary changed")
    _require(_sha(old / "verification_manifest.json") == source["verification_manifest_sha256"],
             "source verification manifest changed")
    _require(_sha(root / source["cause_ledger"]) == source["cause_ledger_sha256"], "cause ledger changed")
    _require(_sha(root / plan["calendar"]["path"]) == plan["calendar"]["sha256"], "Qlib calendar changed")
    snapshot = _original_snapshot(root, plan)
    state_path.write_text(json.dumps({"status": "started", "attempt": 2 if recovery else 1,
                                      "plan_sha256": _sha(plan_path)}) + "\n")
    qlib.init(provider_uri=str(root / plan["calendar"]["provider_uri"]), region="us", kernels=1,
              expression_cache=None, dataset_cache=None)
    objects, audit = load_verified_recorder_objects(
        old, expected_recorder_id=source["recorder_id"],
        artifacts=list(source["artifacts"]), expected_artifact_sha256=source["artifacts"],
        expected_manifest_sha256=source["verification_manifest_sha256"])
    calendar = pd.DatetimeIndex(D.calendar(start_time=plan["calendar"]["bounds"][0],
                                           end_time=plan["calendar"]["bounds"][1], freq="day"))
    _require(len(calendar) > 0 and _sha(root / plan["calendar"]["path"]) == plan["calendar"]["sha256"],
             "native calendar identity differs")
    first = bridge_first(objects["analysis/earliest_non_amended_candidates"],
                         objects["analysis/extraction"], plan)
    old_plan = _json(old / "plan.json")
    annual, pairs = annual_pairs(first, calendar, old_plan)
    old_pairs = objects["analysis/annual_pairs"]
    old_other = old_pairs.loc[old_pairs.cik.ne(plan["frozen_bridge"]["cik"])].reset_index(drop=True)
    new_other = pairs.loc[pairs.cik.ne(plan["frozen_bridge"]["cik"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_other, new_other, check_dtype=False)
    nvda_old = old_pairs.loc[old_pairs.cik.eq(plan["frozen_bridge"]["cik"])]
    nvda_new = pairs.loc[pairs.cik.eq(plan["frozen_bridge"]["cik"])]
    old_keys = set(zip(nvda_old.prior_fiscal_end, nvda_old.fiscal_end))
    new_keys = set(zip(nvda_new.prior_fiscal_end, nvda_new.fiscal_end))
    added = new_keys - old_keys
    expected = {(pd.Timestamp(start), pd.Timestamp(end))
                for start, end in plan["frozen_bridge"]["expected_added_pairs"]}
    _require(added == expected and old_keys <= new_keys and len(nvda_new) == len(nvda_old) + 2,
             "NVDA bridge recovered unexpected annual pairs")
    _require((pd.Timestamp("2018-01-28"), pd.Timestamp("2019-01-27")) not in new_keys,
             "FY2018/FY2019 accounting break crossed")
    old_support = objects["analysis/issuer_month_support"]
    baseline = _project_nvda_support(old_support, old_pairs,
                                     plan["frozen_bridge"], plan["pair_rule"]["max_pair_age_days"])
    _require((baseline.old_reason == baseline.new_reason).all()
             and (baseline.old_pair_accession.fillna("") == baseline.new_pair_accession.fillna("")).all(),
             "saved NVDA source support cannot be reproduced from frozen pairs")
    support = _project_nvda_support(old_support, pairs,
                                    plan["frozen_bridge"], plan["pair_rule"]["max_pair_age_days"])
    old_supported = support.old_conditional_weight_pct.gt(0)
    new_supported = support.new_conditional_weight_pct.gt(0)
    recovered = support.loc[~old_supported & new_supported]
    panel = support[["datetime", "old_conditional_weight_pct", "new_conditional_weight_pct"]].copy()
    panel["delta_pct"] = panel.new_conditional_weight_pct - panel.old_conditional_weight_pct
    panel["instrument"] = "NVDA_IN_DISCLOSED_SOXX_PROXY"
    panel = panel.set_index(["datetime", "instrument"]).sort_index()
    dataset = DatasetH(handler=DataHandlerLP.from_df(pd.concat({"meta": panel}, axis=1)),
                       segments={"all": (None, None)})
    dataset.config(dump_all=True, recursive=True)
    pd.testing.assert_frame_equal(dataset.prepare("all", col_set="meta"), panel)
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=NAME, uri=str(tracking)):
        rec = R.get_recorder()
        rec.save_objects(artifact_path="inputs", source_audit=audit, original_snapshot_manifest=snapshot,
                         plan=plan)
        rec.save_objects(artifact_path="analysis", bridged_first=first, bridged_annual=annual,
                         bridged_pairs=pairs, nvda_month_support=support, recovered_months=recovered)
        rec.save_objects(artifact_path="metadata", bridge_dataset=dataset)
        rec.save_objects(artifact_path="sources", module=Path(__file__).read_bytes(),
                         notebook=(card / f"{NAME}.ipynb").read_bytes())
        pd.testing.assert_frame_equal(rec.load_object("metadata/bridge_dataset").prepare("all", col_set="meta"),
                                      panel)
        rec.log_metrics(recovered_nvda_annual_pairs=2, recovered_nvda_months=len(recovered),
                        qualified_PIT_rows=0, model_fits=0, native_accounts=0)
        summary = {"experiment": NAME, "status": "completed_source_only_not_promoted",
                   "decision": "conditional_NVDA_source_coverage_repaired_only",
                   "original_annual_pair_rows": len(old_pairs), "bridged_annual_pair_rows": len(pairs),
                   "NVDA_annual_pairs_added": 2, "NVDA_source_months": len(support),
                   "NVDA_conditionally_recovered_months": len(recovered),
                   "NVDA_conditionally_recovered_weight_pct_sum": float(recovered.weight_pct.sum()),
                   "non_NVDA_pairs_unchanged": True, "FY2018_FY2019_accounting_break_excluded": True,
                   "strict_PIT_qualified_rows": 0, "model_input_qualified": False,
                   "factor_effectiveness_tested": False, "SOXX_excess_established": False,
                   "D_calendar_calls": 1, "D_features_calls": 0, "market_price_or_label_reads": 0,
                   "model_fits": 0, "native_accounts": 0,
                   "source_recorder_object_loads": len(source["artifacts"]),
                   "cumulative_source_recorder_object_loads":
                       (plan["recovery"]["previous_consumption"]["source_recorder_object_loads"]
                        + plan["recovery"]["attribution_consumption"]["source_recorder_object_loads"]
                        + len(source["artifacts"])) if recovery else len(source["artifacts"]),
                   "attempt": 2 if recovery else 1,
                   "prior_failure_receipt": "attempt1_failure.json" if recovery else None,
                   "source_recorder_id": source["recorder_id"], "plan_sha256": _sha(plan_path),
                   "module_sha256": _sha(Path(__file__)),
                   "recorder_id": rec.id, "experiment_id": str(rec.experiment_id),
                   "recorder_uri": str(tracking),
                   "recorder_local_dir": str(tracking / str(rec.experiment_id) / rec.id),
                   "artifact_policy": plan["artifact_policy"]}
        rec.save_objects(artifact_path="analysis", summary=summary)
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    manifest = {"recorder_id": summary["recorder_id"], "entries": [
        {"path": str(path.relative_to(card)), "sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(artifacts.rglob("*")) if path.is_file()]}
    (card / "verification_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (card / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    state_path.write_text(json.dumps({"status": "completed", "attempt": 2 if recovery else 1,
                                      "recorder_id": rec.id,
                                      "plan_sha256": summary["plan_sha256"]}) + "\n")
    (card / "conclusion.md").write_text(
        f"# NVDA 原始收入标签桥：来源限定结果\n\n"
        f"Qlib Recorder：`{rec.id}`。在不跨 FY2018→FY2019 收入准则断点、不改变其他发行人年度配对的前提下，"
        f"FY2022 同一原始上下文/同值收入双标签使 NVDA 多出 2 个条件机械年度配对；"
        f"披露篮子中恢复 {len(recovered)} 个月的条件支持。原始证券权重不重新归一化。\n\n"
        "首次实现因统一改写多年标签而破坏既有 FY2024→FY2025 配对，在新 Recorder 写入前停止；"
        "保留失败和一次只读归因，第二次仅追加 FY2022 原始双标签并保持周边年份标签不变。\n\n"
        "这里只修补来源配对，不计算库存/销售因子、市场收益、费用账户或 SOXX 超额。"
        "历史证券身份、原始数值首次公开与版本时钟仍未合格；严格历史 PIT 和模型输入均为 false，"
        "不得晋升。下一步核查身份与可用时点，再冻结一次信息检验。\n")
    return summary
