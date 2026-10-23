"""Read-only parent-proposal identity audit for closed Stage1-labelled cards."""

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
    card = root / "research/stage1_f696_parent_identity_audit_20261023"
    plan = json.loads((card / "plan.json").read_text())
    assert plan["frozen"] and not (card / "summary.json").exists()
    digest = _sha(card / "plan.json")
    qlib.init(provider_uri=str((root / "data/qlib/stage1_domain_heads_20261018").resolve()),
              region="us", kernels=1)
    objects = {}
    reads = 0
    for role, source in plan["sources"].items():
        source_card = root / source["card"]
        assert _sha(source_card / "summary.json") == source["summary_sha256"]
        assert json.loads((source_card / "summary.json").read_text())["recorder_id"] == source["recorder_id"]
        assert json.loads((source_card / "nas_binding.json").read_text())["evidence"]["sha256"] == source["NAS_archive_sha256"]
        with R.uri_context(str((source_card / "qlib_recorder").resolve())):
            recorder = R.get_recorder(experiment_name=source_card.name, recorder_id=source["recorder_id"])
            objects[role] = recorder.load_object(source["artifacts"][0])
            reads += 1
    assert reads == plan["budget"]["source_Recorder_object_loads_max"] == 4
    raw = objects["raw"].reset_index(drop=True)
    A = objects["f696_A"].reset_index(drop=True)
    target_source = objects["action"]["frozen_current"]
    targets = target_source.reset_index(drop=True)
    audit = objects["composition"]
    dates = ["signal_date", "trade_date", "exit_date"]
    cols = ["soxl_weight", "soxx_weight", "cash_weight", "tail_probability", "alpha_probability"]
    assert len(raw) == len(A) == len(targets) == 944
    pd.testing.assert_frame_equal(raw[dates], A[dates])
    pd.testing.assert_index_equal(pd.DatetimeIndex(target_source.index), pd.DatetimeIndex(A.signal_date),
                                  check_names=False)
    np.testing.assert_allclose(A[cols].to_numpy(float), targets[cols].to_numpy(float), atol=1e-12, rtol=0)
    delta = raw[cols].to_numpy(float) - A[cols].to_numpy(float)
    differences = pd.DataFrame(delta, columns=cols)
    differences.insert(0, "trade_date", raw.trade_date)
    mismatched = np.abs(delta).max(axis=1) > 1e-8
    assert int(mismatched.sum()) == 944
    assert audit.loc[audit.slot.eq("open"), "trade_date"].is_unique
    assert len(audit.loc[audit.slot.eq("open")]) == 919
    raw_plan = json.loads((root / plan["sources"]["raw"]["card"] / "plan.json").read_text())
    source_contract = raw_plan["frozen_signal_contract"]
    if "Reuse27baa" not in source_contract:
        raise ValueError("raw cashbook source role no longer identifies 27baa")
    code = {name: _sha(root / name) for name in plan["code_paths"]}
    helper_text = (root / plan["code_paths"][0]).read_text()
    static_text = (root / plan["code_paths"][1]).read_text()
    tail_text = (root / plan["code_paths"][2]).read_text()
    bypass_text = (root / plan["code_paths"][3]).read_text()
    A_text = (root / plan["code_paths"][4]).read_text()
    assert '"frame": frame' in helper_text and '"source_targets": action["inputs/source_targets"]["frozen_current"]' in helper_text
    assert 'account["frame"].copy' in static_text
    assert "frame=account['frame'].copy" in tail_text
    assert "account['frame'].copy" in bypass_text
    assert 'frame[name] = targets[name]' in A_text
    max_i = int(np.abs(delta).max(axis=1).argmax())
    result = {"decision": "invalid_f696_parent_identity_for_three_closed_accounts",
              "plan_sha256": digest, "calendar_rows": len(raw),
              "raw_cashbook_source_contract": source_contract,
              "raw_frame_vs_f696_A_mismatch_rows_gt1e8": int(mismatched.sum()),
              "max_abs_output_differences": dict(zip(cols, np.abs(delta).max(axis=0).tolist())),
              "mean_raw_cash": float(raw.cash_weight.mean()),
              "mean_f696_A_cash": float(A.cash_weight.mean()),
              "A_vs_frozen_current_max_abs_error": float(np.max(np.abs(A[cols].to_numpy(float) - targets[cols].to_numpy(float)))),
              "largest_difference_trade_date": str(raw.trade_date.iloc[max_i].date()),
              "largest_difference_raw_outputs": raw.loc[max_i, cols].to_dict(),
              "largest_difference_f696_A_outputs": A.loc[max_i, cols].to_dict(),
              "composition_open_transform_rows": 919,
              "affected_cards": plan["affected_cards"],
              "code_sha256": code,
              "source_Recorder_object_loads": reads,
              "new_market_requests": 0, "new_fits": 0, "new_accounts": 0,
              "market_outcome_reads": 0,
              "interpretation": "Raw cashbook frame is the 27baa proposal; the exact frozen f696 A proposal is source_targets.frozen_current. Three later cards replayed helper account[frame] without substituting those five outputs. Their numerical accounts remain real but are not interventions on f696 A_0. Corrected f696 variants need separate frozen own-state 0bp accounts."}
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    R.set_uri(str((card / "qlib_recorder").resolve()))
    with R.start(experiment_name=card.name):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, source_identities=plan["sources"])
        recorder.save_objects(artifact_path="analysis", output_differences=differences, result=result)
        R.log_params(scope="source_parent_identity_only", new_fits=0, new_accounts=0)
        R.log_metrics(mismatched_parent_rows=result["raw_frame_vs_f696_A_mismatch_rows_gt1e8"])
        rid, eid = recorder.id, str(recorder.experiment_id)
    result.update(recorder_id=rid, experiment_id=eid)
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    binding = json.loads((root / plan["sources"]["f696_A"]["card"] / "nas_binding.json").read_text())
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
                                   "recorder_only_artifacts": ["inputs/*", "analysis/output_differences"]},
               "limitations": ["Account metrics of affected cards are preserved, but their parent description is invalid.",
                               "No repaired strategy or SOXX economic result is created by this audit."]}
    atomic_json(summary, card / "summary.json")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nplan: plan.json\n"
        "sources: frozen Qlib Recorder proposal frames\nnew_fits: 0\nnew_native_accounts: 0\n")
    (card / "conclusion.md").write_text(
        "# f696 原 Stage1 与原始现金账本输入：父本身份审计\n\n"
        "冻结 A_0 研究重放显式把五项输出替换成 `source_targets.frozen_current`；"
        "另一份原始现金账本 `inputs/frame` 的计划明确说明复用27baa提议。"
        f"944/944 同日五输出至少一项不同；现金均值原始账本 {result['mean_raw_cash']:.2%}、"
        f"f696 A {result['mean_f696_A_cash']:.2%}，最大单字段差 "
        f"{max(result['max_abs_output_differences'].values()):.6f}。"
        "受影响的固定组成、tail条件组成、动态风险旁路三卡使用 helper 原始 frame 而未覆盖五项 f696 输出。\n\n"
        "三卡数值账户与费用仍保留，但不能称为原 f696 A_0 衍生干预或据此声称 Stage1 优化。"
        "本卡0收益读取、0拟合、0账户。下一步若重测组成等机制，必须另冻使用 exact frozen_current 的同源0bp自身状态账户，"
        "失败固定版本不在原目录改参数或重放。"
        f"Recorder `{rid}`，NAS SHA `{remote}`。\n")
    return summary
