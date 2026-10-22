"""Engineering v4 of the frozen 2023 source-target account diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

import numpy as np
from qlib.data import D
from src.data import build_qlib_store_from_frame
from src.analysis.integrated_allocation import _path_metrics
from src.backtest.native_raw_corporate import native_raw_validate
from src.backtest.risk_projection_reachability import source_target_path
from src.core.recorder_sources import load_verified_recorder_objects
from src.workflow.inventory_sales_source_qualification import normalized_notebook

NAME = "risk_projection_reachability_native_v4_20261022"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import qlib
    from qlib.workflow import R

    root = Path(root).resolve()
    card = root / "research" / NAME
    plan_path = card / "plan.json"
    plan = json.loads(plan_path.read_text())
    require(plan["frozen"] and plan["execution_authorized"], "engineering v4 frozen plan absent")
    require(not (card / "run_state.json").exists(), "engineering v4 already attempted")
    require(sha(Path(__file__)) == plan["implementation"]["module_sha256"], "engineering module changed")
    notebook = card / (NAME + ".ipynb")
    require(hashlib.sha256(normalized_notebook(notebook)).hexdigest()
            == plan["implementation"]["notebook_sha256"], "notebook code changed")
    original = root / plan["original_v1_card"]
    for name, digest in plan["original_v1_identity"].items():
        require(sha(original / name) == digest, f"prior attempt history changed: {name}")
    require(sha(root / plan["source_target_path_module"]["path"])
            == plan["source_target_path_module"]["sha256"], "prior source-path implementation changed")
    numeric = json.loads((original / "attempt2_numeric_audit.json").read_text())
    derived = plan["existing_derived_provider"]
    require(sha(original / "attempt2_numeric_audit.json") == derived["numeric_audit_sha256"]
            and numeric["status"] == derived["numeric_audit_status_required"]
            and numeric["numeric_array_exact_equal"] and numeric["max_absolute_difference"] == 0,
            "old Qlib/source point numerical identity is unproven")
    old_provider = root / derived["provider_uri"]
    for item in derived["files"].values():
        path = root / item["path"]
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                "existing exact native provider changed")
    for filename, digest in plan["v2_failure_identity"].items():
        require(sha(root / "research/risk_projection_reachability_native_v2_20261022" / filename) == digest,
                "v2 failure identity changed")
    for filename, digest in plan["v3_failure_identity"].items():
        require(sha(root / "research/risk_projection_reachability_native_v3_20261022" / filename) == digest,
                "v3 failure identity changed")
    provider = root / plan["new_derived_provider_uri"]
    require(not provider.exists(), "new guarded provider already exists; no overwrite")
    parent = plan["source"]
    p3_plan_path = root / parent["P3_plan_path"]
    require(sha(p3_plan_path) == parent["P3_plan_sha256"], "original cost plan changed")
    p3_plan = json.loads(p3_plan_path.read_text())
    for key in ("P3", "raw"):
        source = parent[key]
        folder = root / source["card"]
        require(sha(folder / "summary.json") == source["summary_sha256"]
                and sha(folder / "verification_manifest.json") == source["verification_manifest_sha256"],
                "source Recorder metadata changed")
    (card / "run_state.json").write_text(json.dumps({"status": "started", "plan_sha256": sha(plan_path)}) + "\n")
    qlib.init(provider_uri=str(old_provider), region="us", kernels=1)
    loaded, audits = {}, {}
    for key in ("P3", "raw"):
        source = parent[key]
        objects, audit = load_verified_recorder_objects(
            root / source["card"], expected_recorder_id=source["recorder_id"],
            artifacts=list(source["artifacts"]), expected_artifact_sha256=source["artifacts"],
            expected_manifest_sha256=source["verification_manifest_sha256"])
        loaded[key], audits[key] = objects, audit
    old_path = loaded["P3"]["paths/B_path_control_2"]
    old_native = loaded["P3"]["native/B_path_control_2"]
    trace = old_path["trace"]
    source_trace = trace.loc[pd.to_datetime(trace.trade_date).dt.year.eq(2023)].copy().reset_index(drop=True)
    replay = old_path["replay"]
    source_reference = replay.loc[pd.to_datetime(replay.trade_date).dt.year.eq(2023)].copy().reset_index(drop=True)
    require(len(source_trace) == len(source_reference) == plan["development_window"]["funded_sessions"],
            "2023 funded source dates differ")
    tariff = p3_plan["costs_and_budgets"]["tariff"]
    counterfactual, points = source_target_path(
        source_trace, source_reference, loaded["raw"]["inputs/raw_points"],
        loaded["raw"]["inputs/schedule"], tariff=tariff,
        cost=plan["costs"]["slippage_bps"], initial_cash=plan["costs"]["initial_cash_usd"])
    require(len(points) == numeric["rows"] and points.index[0] == pd.Timestamp("2023-01-04 09:45")
            and points.index[-1] == pd.Timestamp("2024-01-02 09:47"),
            "counterfactual point grid differs from prior exact numeric audit")
    guard_stamp = pd.Timestamp(plan["provider_only_guard_stamp"])
    raw_points = loaded["raw"]["inputs/raw_points"]
    require(guard_stamp > points.index[-1] and guard_stamp in raw_points.index,
            "frozen provider-only following quote absent")
    guard = raw_points.loc[[guard_stamp], list(points.columns)].astype(float)
    combined = pd.concat([points, guard]).sort_index()
    require(len(combined) == len(points) + 1 and combined.index.is_unique
            and combined.index[-1] == guard_stamp and np.isfinite(combined.to_numpy()).all(),
            "guarded provider point grid differs")
    fields = ["open", "high", "low", "close", "execution_price", "raw_execution_price"]
    rows = [{"datetime": stamp, "instrument": symbol,
             **{field: float(value) for field in fields}}
            for symbol in points.columns for stamp, value in combined[symbol].items()]
    manifest = build_qlib_store_from_frame(pd.DataFrame(rows), provider_uri=provider,
        freq="1min", region="us", as_of="2024-01-02",
        field_map={field: field for field in fields},
        source="Exact P3 raw Recorder marks; 2024-01-02 10:32 quote is Qlib calendar guard only, excluded from account and NAV")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    observed = D.features(list(points.columns), ["$raw_execution_price"],
        start_time=combined.index[0], end_time=combined.index[-1], freq="1min")
    check = observed["$raw_execution_price"].unstack("instrument").reindex(combined.index)[list(points.columns)]
    pd.testing.assert_index_equal(check.index, combined.index, check_names=False)
    np.testing.assert_array_equal(check.to_numpy(), combined.to_numpy())
    require(manifest["calendar_end"] == str(guard_stamp), "guarded provider terminal differs")
    (card / "provider_guard_audit.json").write_text(json.dumps({
        "status": "numeric_exact", "guard_stamp": str(guard_stamp),
        "guard_excluded_from_policy_and_NAV": guard_stamp not in points.index,
        "policy_point_rows": len(points), "provider_point_rows": len(combined),
        "manifest": manifest}, indent=2) + "\n")
    alternate = native_raw_validate(
        counterfactual, provider_uri=provider, point_prices=points,
        schedule=loaded["raw"]["inputs/schedule"], tariff=tariff,
        cost=plan["costs"]["slippage_bps"], initial_cash=plan["costs"]["initial_cash_usd"])
    require(alternate["audit"]["status"] == "passed"
            and len(alternate["authority"]) == len(source_trace), "native source-target account failed")
    baseline_full = old_native["authority"]
    baseline = baseline_full.loc[pd.to_datetime(baseline_full.trade_date).dt.year.eq(2023)].copy()
    require(len(baseline) == len(source_trace)
            and pd.DatetimeIndex(alternate["authority"].trade_date).equals(pd.DatetimeIndex(baseline.trade_date)),
            "baseline and counterfactual account dates differ")
    alt_metrics = _path_metrics(alternate["authority"])
    baseline_metrics = _path_metrics(baseline)
    yearly = loaded["P3"]["analysis/results"]["annual_metrics"]
    soxx = yearly.loc[yearly.family.eq("SOXX") & yearly.slippage_bps.eq(2)
                      & yearly.kind.eq("year") & yearly.period.eq("2023")]
    require(len(soxx) == 1, "same-cost native SOXX reference absent")
    soxx_metrics = {name: float(soxx.iloc[0][name]) for name in
                    ("return_pct", "max_drawdown_pct", "cvar5_pct")}
    comparison = {"source_target": alt_metrics, "risk_projected_B2": baseline_metrics,
                  "SOXX_2bp_reference": soxx_metrics,
                  "source_to_projected_terminal_wealth_ratio":
                      float(alternate["authority"].equity.iloc[-1] / baseline.equity.iloc[-1]),
                  "source_to_SOXX_2023_relative_wealth":
                      float((1 + alt_metrics["return_pct"] / 100) / (1 + soxx_metrics["return_pct"] / 100)),
                  "projected_to_SOXX_2023_relative_wealth":
                      float((1 + baseline_metrics["return_pct"] / 100) / (1 + soxx_metrics["return_pct"] / 100))}
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=NAME, uri=str(tracking)):
        rec = R.get_recorder()
        rec.save_objects(artifact_path="inputs", plan=plan, source_audits=audits,
                         reused_provider_identity=derived["files"], bounded_raw_points=points)
        rec.save_objects(artifact_path="native", source_target_account=alternate)
        rec.save_objects(artifact_path="analysis", matched_2023_comparison=comparison,
                         source_target_path=counterfactual)
        rec.save_objects(artifact_path="sources", module=Path(__file__).read_bytes(),
                         notebook=notebook.read_bytes())
        rec.log_metrics(source_to_projected_wealth_ratio=comparison["source_to_projected_terminal_wealth_ratio"],
                        alternate_native_sessions=len(source_trace), new_native_accounts=1, model_fits=0)
        summary = {"experiment": NAME, "status": "completed_development_counterfactual_no_promotion",
                   "decision": "same_scientific_counterfactual_engineering_v4_diagnostic_only",
                   "funded_sessions": len(source_trace), "cost_bps": 2,
                   "source_to_projected_terminal_wealth_ratio": comparison["source_to_projected_terminal_wealth_ratio"],
                   "source_to_SOXX_2023_relative_wealth": comparison["source_to_SOXX_2023_relative_wealth"],
                   "projected_to_SOXX_2023_relative_wealth": comparison["projected_to_SOXX_2023_relative_wealth"],
                   "source_target_2023_return_pct": alt_metrics["return_pct"],
                   "projected_B2_2023_return_pct": baseline_metrics["return_pct"],
                   "SOXX_2bp_2023_return_pct": soxx_metrics["return_pct"],
                   "source_target_max_drawdown_pct": alt_metrics["max_drawdown_pct"],
                   "projected_B2_max_drawdown_pct": baseline_metrics["max_drawdown_pct"],
                   "source_target_CVaR5_pct": alt_metrics["cvar5_pct"],
                   "projected_B2_CVaR5_pct": baseline_metrics["cvar5_pct"],
                   "source_target_mean_cash_weight": float(alternate["authority"].ending_cash_weight.mean()),
                   "projected_B2_mean_cash_weight": float(baseline.ending_cash_weight.mean()),
                   "source_target_order_fees_usd": float(alternate["orders"].total.sum()),
                   "projected_B2_order_fees_usd": float(old_native["orders"].loc[
                       pd.to_datetime(old_native["orders"].trade_date).dt.year.eq(2023), "total"].sum()),
                   "native_reconciliation_status": alternate["audit"]["status"],
                   "previous_engineering_failures": 4,
                   "source_history_independent": False, "held_candidate_evaluation_performed": False,
                   "qualified_forward_days": 0, "SOXX_v2_goal_achieved": False,
                   "trading_or_promotion_permission": False,
                   "new_source_recorder_object_loads": 5,
                   "cumulative_prior_source_recorder_object_loads": plan["budget"]["cumulative_prior_source_object_loads"],
                   "new_D_features_calls": 1, "new_provider_builds": 1,
                   "new_native_accounts": 1, "new_model_fits": 0, "new_bootstraps": 0,
                   "plan_sha256": sha(plan_path), "module_sha256": sha(Path(__file__)),
                   "recorder_id": rec.id, "experiment_id": str(rec.experiment_id),
                   "recorder_uri": str(tracking),
                   "recorder_local_dir": str(tracking / str(rec.experiment_id) / rec.id),
                   "artifact_policy": plan["artifact_policy"]}
        rec.save_objects(artifact_path="analysis", summary=summary)
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    verification = {"recorder_id": summary["recorder_id"], "entries": [
        {"path": str(path.relative_to(card)), "sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(artifacts.rglob("*")) if path.is_file()]}
    (card / "verification_manifest.json").write_text(json.dumps(verification, indent=2) + "\n")
    (card / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (card / "run_state.json").write_text(json.dumps({"status": "completed", "recorder_id": rec.id,
                                                     "plan_sha256": summary["plan_sha256"]}) + "\n")
    (card / "conclusion.md").write_text(
        "# 风控投影与原始目标：2023 路径反事实（工程 v4）\n\n"
        f"Qlib Recorder：`{rec.id}`。原始目标独立账户与既有 P3 B2 在相同 2023 年 249 个资金日、"
        "2bp、原始股数/现金/分红与 IBKR Tiered 费用下比较，且经 Qlib 原生逐点对账。"
        f"原始目标/风控投影的期末净财富比为 {comparison['source_to_projected_terminal_wealth_ratio']:.6f}；"
        f"两条账户的最大回撤分别为 {alt_metrics['max_drawdown_pct']:.2f}% 与 "
        f"{baseline_metrics['max_drawdown_pct']:.2f}%。\n\n"
        "此前两次工程尝试在原生账户前停止，v2 在 Qlib 终点日历推进时停止。原卡曾以数据类型敏感断言"
        "拒绝数值逐位相等的 Qlib 点价。失败和读取均保留，v3 仅给派生 provider 加一个账户终点后的日历点，"
        "没有更改科学问题、目标、日期或费用。\n\n"
        "这是已复用 2023 历史的诊断，不是可交易恢复信号、完整 0/2/5bp 对照或独立前向证据。"
        "不能下调 P3 冻结的 0.825 恢复阈值来救援，也不能声称稳定跑赢 SOXX。\n")
    return summary
