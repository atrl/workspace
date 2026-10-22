"""One fixed funded composition intervention on the frozen Stage1 0bp chain."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.minute_intake import atomic_json
from src.model.nominal_beta_composition import AuditedComposition


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*, root: Path, card: Path) -> dict:
    """Spend the one frozen replay/native/bootstrap budget; never silently retry."""
    from qlib.workflow import R
    from src.backtest.native_raw_corporate import native_raw_validate
    from src.backtest.stage1_domain_heads_raw import replay_stage1_raw
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
    from src.workflow.stage1_nvda_guidance_receiver import prepare_account_sources
    from src.workflow.stage1_rolling126_shortfall import _analyze_account
    from src.data.nas_market_data import archive_nas_recorder

    root, card = Path(root).resolve(), Path(card).resolve()
    plan_path = card / "plan.json"
    plan = json.loads(plan_path.read_text())
    if not plan.get("frozen") or not plan.get("execution_authorized"):
        raise ValueError("blocked: fixed Stage1 composition execution not authorized")
    if (card / "run_state.json").exists():
        raise ValueError("blocked: one-shot Stage1 composition attempt already started")
    for relative, digest in plan["identities"].items():
        if _sha(root / relative) != digest:
            raise ValueError("blocked: frozen source/code identity differs: " + relative)

    helper_card = root / plan["account_source_helper"]
    before = _sha(helper_card / "account_source_preflight.json")
    if before != plan["account_source_receipt_sha256"]:
        raise ValueError("blocked: frozen Stage1 source binding differs")
    account = prepare_account_sources(root=root, card=helper_card)
    if _sha(helper_card / "account_source_preflight.json") != before:
        raise ValueError("blocked: source helper changed its sealed receipt")
    if (len(account["frame"]) != 944 or len(account["points"]) != 3778
            or account["references"]["Stage1_A"]["audit"]["status"] != "passed"
            or account["references"]["SOXX"]["audit"]["status"] != "passed"):
        raise ValueError("blocked: matched Stage1/SOXX native source differs")
    provider = root / plan["provider_uri"]
    if _sha(provider / "qlib_store_manifest.json") != plan["provider_manifest_sha256"]:
        raise ValueError("blocked: pinned native price provider differs")

    state = {"status": "account_attempted", "plan_sha256": _sha(plan_path),
             "new_model_fits": 0, "native_account_attempts": 1, "bootstrap_attempts": 0}
    atomic_json(state, card / "run_state.json")
    recovery = IntradayRecoveryDirectAllocationSpec(root=root, **account["recovery_spec"])
    action = Stage1OpenActionValueSpec(root=root, **account["action_spec"])
    execution = {"initial_cash": plan["initial_cash_usd"],
                 "tariff": account["parent"]["tariff"]}
    composition = AuditedComposition()
    path = replay_stage1_raw(
        account["frame"].copy(deep=True), point_prices=account["points"],
        schedule=account["schedule"], plan=execution,
        controller=account["controller"], recovery_spec=recovery,
        action_spec=action, selection=account["selection"], slippage_bps=0,
        target_transform=composition,
    )
    if len(path["replay"]) != 944 or len(path["points"]) != 3778:
        raise ValueError("blocked: transformed Stage1 calendar differs")
    if not composition.rows or any(abs(row["beta_error"]) > 1e-12
                                   or abs(row["simplex_error"]) > 1e-12
                                   for row in composition.rows):
        raise ValueError("blocked: funded composition invariant differs")
    book = native_raw_validate(
        path, provider_uri=provider, point_prices=account["points"],
        schedule=account["schedule"], tariff=execution["tariff"], cost=0,
        initial_cash=execution["initial_cash"],
    )
    if book["audit"]["status"] != "passed":
        raise ValueError("blocked: transformed account failed native Qlib readback")
    state.update(status="native_complete", native_account_completed=1,
                 transform_calls=len(composition.rows))
    atomic_json(state, card / "run_state.json")

    dates = ["signal_date", "trade_date", "exit_date"]
    for role in ("Stage1_A", "SOXX"):
        pd.testing.assert_frame_equal(
            book["authority"][dates].reset_index(drop=True),
            account["references"][role]["authority"][dates].reset_index(drop=True),
        )
    state["bootstrap_attempts"] = 1
    atomic_json(state, card / "run_state.json")
    result = _analyze_account(
        account=book["authority"],
        reference_soxx=account["references"]["SOXX"]["authority"],
        reference_stage1=account["references"]["Stage1_A"]["authority"],
        sampled_drawdown=-100 * float(path["replay"].monitored_drawdown.max()),
        plan=plan,
    )
    audit = pd.DataFrame(composition.rows)
    audit_summary = {"transform_calls": len(audit),
                     "open_calls": int(audit.slot.eq("open").sum()),
                     "recovery_calls": int(audit.slot.eq("recovery").sum()),
                     "max_abs_beta_error": float(audit.beta_error.abs().max()),
                     "max_abs_simplex_error": float(audit.simplex_error.abs().max()),
                     "mean_before_cash": float(audit.before_cash.mean()),
                     "mean_after_cash": float(audit.after_cash.mean())}
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=plan["experiment"], uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan,
                              source_receipt=account["receipt"], audit_summary=audit_summary)
        recorder.save_objects(artifact_path="paths", candidate_0=path)
        recorder.save_objects(artifact_path="native", candidate_0=book)
        recorder.save_objects(artifact_path="analysis", metrics=result["SOXX_metrics"],
                              Stage1_increment=result["Stage1_increment_metrics"],
                              annual=result["annual"], quarter=result["quarter"],
                              rolling_summary=result["rolling_summary"],
                              uncertainty=result["uncertainty"], gates=result["gates"],
                              composition_audit=audit)
        recorder.save_objects(artifact_path="analysis/rolling", all_windows=result["rolling"])
        recorder.log_metrics(new_model_fits=0, native_accounts=1,
                             SOXX_relative_wealth=result["SOXX_metrics"]["relative_wealth"],
                             historical_0bp_pass=int(result["historical_0bp_gate_pass"]))
        recorder_id, experiment_id = recorder.id, str(recorder.experiment_id)

    recorder_dir = tracking / experiment_id / recorder_id
    artifacts = recorder_dir / "artifacts"
    atomic_json({"recorder_id": recorder_id,
                 "entries": [{"path": str(p.relative_to(card)), "sha256": _sha(p),
                              "bytes": p.stat().st_size}
                             for p in sorted(artifacts.rglob("*")) if p.is_file()]},
                card / "verification_manifest.json")
    for kind in ("annual", "quarter"):
        result[kind].to_csv(card / f"{kind}_metrics.csv", index=False)
    summary = {"experiment": plan["experiment"],
               "decision": ("historical_0bp_pass_needs_independent_review" if result["historical_0bp_gate_pass"]
                            else "not_promoted_fixed_Stage1_beta_composition_0bp"),
               "objective_achieved": False, "plan_sha256": _sha(plan_path),
               "new_model_fits": 0, "new_native_accounts": 1,
               "qualified_forward_days": 0, "gates": result["gates"],
               "SOXX_metrics": result["SOXX_metrics"],
               "Stage1_increment_metrics": result["Stage1_increment_metrics"],
               "rolling_summary": result["rolling_summary"],
               "uncertainty": result["uncertainty"],
               "composition_audit": audit_summary,
               "recorder_id": recorder_id, "experiment_id": experiment_id,
               "recorder_uri": str(tracking), "recorder_local_dir": str(recorder_dir),
               "artifact_policy": {"policy": "slim_card_recorder_first",
                                   "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "paths/*", "native/*",
                                                               "analysis/rolling/*"]}}
    atomic_json(summary, card / "summary.json")
    original_binding = json.loads((root / "research/stage1_domain_heads_20261018/nas_binding.json").read_text())
    original_binding.pop("evidence", None)
    original_binding.update(source_id=plan["experiment"], source_plan="plan.json", as_of="2026-10-09")
    atomic_json(original_binding, card / "nas_binding.json")
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
                                                   card=card, recorder_id=recorder_id,
                                                   experiment_id=experiment_id)
    atomic_json(summary, card / "summary.json")
    state.update(status="completed", recorder_id=recorder_id,
                 experiment_id=experiment_id, historical_0bp_pass=result["historical_0bp_gate_pass"])
    atomic_json(state, card / "run_state.json")
    metric = result["SOXX_metrics"]
    complete_years = result["annual"].loc[result["annual"].complete_period]
    positive_years = int(complete_years.relative_wealth.gt(1 + 1e-8).sum())
    (card / "conclusion.md").write_text(
        "# 原Stage1固定名义beta组成：0bp完整账户\n\n"
        f"一项冻结的后决策组成映射，0拟合、1本独立0bp原生Qlib账户；"
        f"0bp仍扣IBKR Tiered费用。SOXX相对期末财富{metric['relative_wealth']:.6f}，"
        f"可用点最大回撤{100*path['replay'].monitored_drawdown.max():.4f}%，"
        f"126日正超额窗口{100*result['rolling_summary']['positive_fraction']:.2f}%，"
        f"SOXX区块下界{result['uncertainty']['daily_lower_bound']:.9f}。"
        f"原Stage1相对财富增量{result['Stage1_increment_metrics']['relative_wealth']:.6f}。\n\n"
        f"开盘映射{audit_summary['open_calls']}次，恢复映射{audit_summary['recovery_calls']}次；"
        f"{len(complete_years)}个完整年度中{positive_years}个跑赢SOXX；"
        "名义beta守恒，但实际风险、整股和状态反馈不守恒。全部完整/部分年度、16季度、"
        "819个连续126日窗口在Recorder和小表保留。\n\n"
        f"决定：`{summary['decision']}`。来源历史已复用，现有价格点并非全分钟风险认证；"
        "一旦本固定0bp版本失败，不改变转换比例、风险阈值或窗口挽救。"
        "即使历史通过也须独立结果复核、2/5bp和合格前向证据。"
        f"Recorder `{recorder_id}`，NAS SHA `{summary['nas_evidence']['sha256']}`。\n"
    )
    return summary
