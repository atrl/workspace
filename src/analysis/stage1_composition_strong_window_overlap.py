"""Read-only overlap of closed 0bp SOXX-relative Qlib window evidence."""

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
    from qlib.workflow import R

    from src.data.minute_intake import atomic_json
    from src.data.nas_market_data import _settings, _ssh, archive_nas_recorder

    root = Path(root).resolve()
    card = root / "research/stage1_composition_strong_window_overlap_20261023"
    plan = json.loads((card / "plan.json").read_text())
    assert plan["frozen"] and not (card / "summary.json").exists()
    digest = _sha(card / "plan.json")
    qlib.init(provider_uri=str((root / "data/qlib/stage1_domain_heads_20261018").resolve()),
              region="us", kernels=1)
    values = {}
    reads = 0
    for role in ("composition", "reference"):
        source = plan["sources"][role]
        source_card = root / source["card"]
        assert _sha(source_card / "summary.json") == source["summary_sha256"]
        assert json.loads((source_card / "summary.json").read_text())["recorder_id"] == source["recorder_id"]
        assert _sha(source_card / "verification_manifest.json") == source["manifest_sha256"]
        binding = json.loads((source_card / "nas_binding.json").read_text())
        assert binding["evidence"]["sha256"] == source["NAS_archive_sha256"]
        with R.uri_context(str((source_card / "qlib_recorder").resolve())):
            recorder = R.get_recorder(experiment_name=source_card.name, recorder_id=source["recorder_id"])
            names = ([source["rolling_artifact"]] if role == "composition"
                     else [source["strong_rolling_artifact"], source["stage1_rolling_artifact"]])
            for name in names:
                values[name] = recorder.load_object(name)
                reads += 1
    assert reads == 3 and reads <= plan["budget"]["source_Recorder_object_loads_max"]
    composition = values[plan["sources"]["composition"]["rolling_artifact"]]
    strong = values[plan["sources"]["reference"]["strong_rolling_artifact"]]
    stage1 = values[plan["sources"]["reference"]["stage1_rolling_artifact"]]
    windows = pd.DataFrame({"first_trade": composition.first_trade,
                            "last_exit": composition.last_exit})
    for name, source in (("composition", composition), ("strong9097", strong), ("stage1_A0", stage1)):
        assert len(source) == plan["scope"]["expected_windows"]
        pd.testing.assert_frame_equal(source[["first_trade", "last_exit"]].reset_index(drop=True),
                                      windows[["first_trade", "last_exit"]].reset_index(drop=True))
        np.testing.assert_allclose(source.relative_return_pct.to_numpy(float),
                                   100 * np.expm1(source.relative_log.to_numpy(float)), rtol=1e-12, atol=1e-10)
        windows[name + "_relative_log"] = source.relative_log.to_numpy(float)
        windows[name + "_positive"] = source.relative_log.gt(0).to_numpy(bool)
    a, b, c = (windows[name + "_positive"] for name in ("composition", "strong9097", "stage1_A0"))
    windows["composition_or_strong_positive"] = a | b
    windows["composition_or_strong_or_A_positive"] = a | b | c
    windows["composition_only_positive"] = a & ~b
    windows["strong_only_positive"] = b & ~a
    windows["composition_and_strong_negative"] = ~a & ~b
    windows["all_three_negative"] = ~a & ~b & ~c
    count = len(windows)
    result = {"decision": "read_only_window_coverage_overlap_no_strategy",
              "plan_sha256": digest, "windows": count,
              "required_positive_windows": plan["scope"]["required_positive_windows"],
              "counts": {name: int(windows[name].sum()) for name in
                         ("composition_positive", "strong9097_positive", "stage1_A0_positive",
                          "composition_or_strong_positive", "composition_or_strong_or_A_positive",
                          "composition_only_positive", "strong_only_positive",
                          "composition_and_strong_negative", "all_three_negative")},
              "source_Recorder_object_loads": reads,
              "new_market_requests": 0, "new_fits": 0, "new_accounts": 0, "new_bootstraps": 0,
              "scope_limit": "Union selects whole historical windows with hindsight; it is not a feasible daily switching policy or a universal upper bound on combined-account returns."}
    if result["counts"]["composition_or_strong_positive"] != (
            result["counts"]["composition_positive"] +
            result["counts"]["strong_only_positive"]):
        raise ValueError("window union arithmetic does not close")
    groups = []
    for kind, labels in (("start_year", pd.DatetimeIndex(windows.first_trade).to_period("Y")),
                         ("start_quarter", pd.DatetimeIndex(windows.first_trade).to_period("Q"))):
        for period in labels.unique():
            section = windows.loc[labels == period]
            groups.append({"kind": kind, "period": str(period), "windows": len(section),
                           "composition_positive": int(section.composition_positive.sum()),
                           "strong_positive": int(section.strong9097_positive.sum()),
                           "stage1_positive": int(section.stage1_A0_positive.sum()),
                           "composition_or_strong_positive": int(section.composition_or_strong_positive.sum()),
                           "both_negative": int(section.composition_and_strong_negative.sum())})
    groups = pd.DataFrame(groups)
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    groups.to_csv(card / "period_summary.csv", index=False)
    R.set_uri(str((card / "qlib_recorder").resolve()))
    with R.start(experiment_name=card.name):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, source_identities=plan["sources"])
        recorder.save_objects(artifact_path="analysis", all_windows=windows, period_groups=groups, result=result)
        R.log_params(analysis="fixed_three_path_window_overlap", new_fits=0, new_accounts=0)
        R.log_metrics(composition_or_strong_positive=result["counts"]["composition_or_strong_positive"])
        rid, eid = recorder.id, str(recorder.experiment_id)
    result.update(recorder_id=rid, experiment_id=eid)
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    source_card = root / plan["sources"]["reference"]["card"]
    binding = json.loads((source_card / "nas_binding.json").read_text())
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
                                   "recorder_only_artifacts": ["inputs/*", "analysis/all_windows", "analysis/period_groups"]},
               "limitations": [result["scope_limit"],
                               "2023-2026 is reused development history, not independent forward evidence."]}
    atomic_json(summary, card / "summary.json")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nplan: plan.json\n"
        "sources: frozen Qlib rolling-window Recorders\nnew_fits: 0\nnew_native_accounts: 0\n")
    counts = result["counts"]
    (card / "conclusion.md").write_text(
        "# Stage1 固定组成与强参考：0bp 滚动窗口覆盖重叠\n\n"
        "只读三个封存原生Qlib账户的同日819个连续126日SOXX相对窗口：原Stage1 A_0、固定beta组成及强9097参考。"
        "0拟合、0新账户、0行情请求；全窗口及起点年度/季度在Recorder。\n\n"
        f"正向窗口：A_0 {counts['stage1_A0_positive']}，组成 {counts['composition_positive']}，"
        f"强参考 {counts['strong9097_positive']}；组成或强参考历史逐窗择优的并集 "
        f"{counts['composition_or_strong_positive']}/{count}，要求 {plan['scope']['required_positive_windows']}。"
        f"两者同时为负 {counts['composition_and_strong_negative']} 个窗口。"
        "这个逐窗事后并集不能交易，也不是日频组合策略的通用收益上界；"
        "只用于判断两条已关闭路径的失败是否互补，不生成切换规则。"
        f"Recorder `{rid}`，NAS SHA `{remote}`。\n")
    return summary
