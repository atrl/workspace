"""Read-only cash and SOXL selection attribution on common failing windows."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group(frame: pd.DataFrame, name: str, kind: str) -> dict:
    result = {"kind": kind, "group": name, "windows": len(frame)}
    for family in ("composition", "original_A0"):
        cash = frame[f"{family}_cash_log"]
        selection = frame[f"{family}_selection_log"]
        result.update({
            f"{family}_cash_negative": int(cash.lt(0).sum()),
            f"{family}_selection_negative": int(selection.lt(0).sum()),
            f"{family}_both_negative": int((cash.lt(0) & selection.lt(0)).sum()),
            f"{family}_cash_more_negative": int((cash.lt(selection) & cash.lt(0)).sum()),
            f"{family}_selection_more_negative": int((selection.lt(cash) & selection.lt(0)).sum()),
            f"{family}_cash_mean": float(cash.mean()),
            f"{family}_cash_median": float(cash.median()),
            f"{family}_selection_mean": float(selection.mean()),
            f"{family}_selection_median": float(selection.median()),
        })
    result["composition_relative_log_mean"] = float(frame.composition_relative_log.mean())
    return result


def run(root: Path) -> dict:
    import qlib
    from qlib.workflow import R

    from src.data.minute_intake import atomic_json
    from src.data.nas_market_data import _settings, _ssh, archive_nas_recorder

    root = Path(root).resolve()
    card = root / "research/stage1_common_failure_component_map_20261023"
    plan = json.loads((card / "plan.json").read_text())
    assert plan["frozen"] and not (card / "summary.json").exists()
    digest = _sha(card / "plan.json")
    qlib.init(provider_uri=str((root / "data/qlib/stage1_domain_heads_20261018").resolve()),
              region="us", kernels=1)
    sources = {}
    for name, source in plan["sources"].items():
        source_card = root / source["card"]
        assert _sha(source_card / "summary.json") == source["summary_sha256"]
        assert json.loads((source_card / "summary.json").read_text())["recorder_id"] == source["recorder_id"]
        assert json.loads((source_card / "nas_binding.json").read_text())["evidence"]["sha256"] == source["NAS_archive_sha256"]
        with R.uri_context(str((source_card / "qlib_recorder").resolve())):
            recorder = R.get_recorder(experiment_name=source_card.name, recorder_id=source["recorder_id"])
            sources[name] = recorder.load_object(source["artifact"])
    assert len(sources) == plan["budget"]["source_Recorder_object_loads_max"] == 3
    overlap, comp, orig = sources["overlap"], sources["composition_components"], sources["original_components"]
    assert len(overlap) == len(comp) == len(orig) == plan["groups"]["all_windows"]
    frame = overlap[["first_trade", "last_exit", "composition_relative_log",
                     "composition_positive", "strong9097_positive", "stage1_A0_positive",
                     "composition_and_strong_negative", "all_three_negative"]].copy()
    for family, source, fee in (("composition", comp, "candidate_direct_fees_log"),
                                ("original_A0", orig, "Stage1_direct_fees_log")):
        for key in ("first_trade", "last_exit"):
            np.testing.assert_array_equal(pd.to_datetime(source[key]).to_numpy(),
                                          pd.to_datetime(frame[key]).to_numpy())
        closure = (source.cash_opportunity_log + source.SOXL_selection_log +
                   source[fee] + source.timing_and_benchmark_residual_log)
        np.testing.assert_allclose(closure.to_numpy(float), source.relative_log_excess.to_numpy(float),
                                   rtol=0, atol=1e-10)
        reference = (overlap.composition_relative_log if family == "composition"
                     else overlap.stage1_A0_relative_log)
        np.testing.assert_allclose(source.relative_log_excess.to_numpy(float),
                                   reference.to_numpy(float), rtol=0, atol=1e-10)
        frame[f"{family}_cash_log"] = source.cash_opportunity_log.to_numpy(float)
        frame[f"{family}_selection_log"] = source.SOXL_selection_log.to_numpy(float)
        frame[f"{family}_relative_log"] = source.relative_log_excess.to_numpy(float)
        frame[f"{family}_cash_weight"] = source.mean_post_open_cash_weight.to_numpy(float)
        frame[f"{family}_risk_ge2"] = source.risk_state_ge2_fraction.to_numpy(float)
    assert frame.composition_and_strong_negative.sum() == plan["groups"]["common_composition_strong_negative"]
    assert frame.composition_positive.sum() == plan["groups"]["composition_positive"]
    assert frame.all_three_negative.sum() == plan["groups"]["all_three_negative"]
    groups = [_group(frame, "all", "all"),
              _group(frame.loc[frame.composition_and_strong_negative], "composition_and_strong_negative", "fixed"),
              _group(frame.loc[frame.composition_positive], "composition_positive", "fixed"),
              _group(frame.loc[frame.all_three_negative], "all_three_negative", "fixed")]
    labels = pd.DatetimeIndex(frame.first_trade)
    for kind, periods in (("start_year", labels.to_period("Y")),
                          ("start_quarter", labels.to_period("Q"))):
        for period in periods.unique():
            section = frame.loc[periods == period]
            groups.append(_group(section, str(period), kind))
    groups = pd.DataFrame(groups)
    common = groups.loc[groups.group.eq("composition_and_strong_negative")].iloc[0].to_dict()
    result = {"decision": "descriptive_common_window_cash_selection_attribution_no_policy",
              "plan_sha256": digest, "windows": len(frame),
              "common_failures": int(frame.composition_and_strong_negative.sum()),
              "composition_positives": int(frame.composition_positive.sum()),
              "common_failure_components": common,
              "source_Recorder_object_loads": 3,
              "new_market_requests": 0, "new_fits": 0, "new_accounts": 0, "new_bootstraps": 0,
              "limitation": "819 overlapping windows are not independent trials; cash and selection are descriptive physical account components, not causal policy effects."}
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    groups.loc[groups.kind.isin(["all", "fixed"])].to_csv(card / "component_group_summary.csv", index=False)
    R.set_uri(str((card / "qlib_recorder").resolve()))
    with R.start(experiment_name=card.name):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, source_identities=plan["sources"])
        recorder.save_objects(artifact_path="analysis", all_windows=frame, all_groups=groups, result=result)
        R.log_params(method="fixed_common_failure_component_signs", new_fits=0, new_accounts=0)
        R.log_metrics(common_failures=result["common_failures"],
                      composition_cash_negative=int(common["composition_cash_negative"]))
        rid, eid = recorder.id, str(recorder.experiment_id)
    result.update(recorder_id=rid, experiment_id=eid)
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    binding = json.loads((root / plan["sources"]["overlap"]["card"] / "nas_binding.json").read_text())
    binding.pop("evidence", None)
    atomic_json(binding, card / "nas_binding.json")
    nas = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
                               card=card, recorder_id=rid, experiment_id=eid)
    remote = _ssh(_settings(root)["ssh"], "sha256sum " + shlex.quote(nas["nas_path"])).split()[0]
    assert remote == nas["sha256"]
    summary = {"experiment": card.name, "decision": result["decision"],
               "objective_achieved": False, "plan_sha256": digest,
               "result": "result.json", "recorder_id": rid, "experiment_id": eid,
               "new_fits": 0, "new_native_accounts": 0,
               "NAS_evidence": nas, "NAS_readback": {"verified": True, "sha256": remote},
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "analysis/all_windows", "analysis/all_groups"]},
               "limitations": [result["limitation"],
                               "Reused 2023-2026 market outcomes are development attribution, not independent forward evidence."]}
    atomic_json(summary, card / "summary.json")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nplan: plan.json\n"
        "sources: frozen Qlib rolling-component Recorders\nnew_fits: 0\nnew_native_accounts: 0\n")
    (card / "conclusion.md").write_text(
        "# Stage1 组成与强参考共同失败窗口：现金与选择分量\n\n"
        "只读同日819个原生Qlib 0bp相对SOXX窗口；固定组成和强参考共同负向327窗，原Stage1 A_0分量并列。"
        "来源日期和四分量相对对数超额逐窗闭合。0拟合、0新账户、0行情。\n\n"
        f"共同失败327窗中，固定组成现金项为负 {common['composition_cash_negative']} 窗、"
        f"SOXL选择项为负 {common['composition_selection_negative']} 窗、两者同负 "
        f"{common['composition_both_negative']} 窗；现金更负 {common['composition_cash_more_negative']} 窗、"
        f"选择更负 {common['composition_selection_more_negative']} 窗。"
        f"现金/选择平均对数分量 {common['composition_cash_mean']:.6f}/"
        f"{common['composition_selection_mean']:.6f}。年度季度全部与窗口原件在Recorder。\n\n"
        "此为重叠窗口的描述性账户分解，不选开关，也不将现金/选择项当因果动作优势。"
        "下一机制必须在决策前区分共同失败窗口的主导分量，并另跑完整Stage1自身状态0bp Tiered账户。"
        f"Recorder `{rid}`，NAS SHA `{remote}`。\n")
    return summary
