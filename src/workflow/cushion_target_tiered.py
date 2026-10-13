"""Fixed signal-free wealth-cushion target and matched native controls."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import paired_metrics, annual_paired_metrics, stationary_bootstrap_excess
from src.backtest.cushion_target import CushionTarget
from src.backtest.full_path_tiered import replay_full_path
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
from src.workflow.minute_action_value import _source


def run_cushion_target(*, root, card, plan):
    import qlib
    import yaml
    from qlib.workflow import R
    if (card / "summary.json").exists(): return json.loads((card / "summary.json").read_text())
    provider = root / "data/qlib/full_path_tiered_allocation_20261013"
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    source, audit = _source(root, plan["source"], ["inputs/frame"])
    frame = source["inputs/frame"]
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=plan["experiment"], recorder_name="fixed_cushion_target", uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.log_params(plan_sha256=hashlib.sha256((card / "plan.json").read_bytes()).hexdigest(), model_fits=0, maximum_native_cases=18)
        recorder.save_objects(artifact_path="inputs", frame=frame, source_audit=audit, plan=plan)
        recorder.save_objects(artifact_path="sources", policy=(root / "src/backtest/cushion_target.py").read_text(),
            workflow=(root / "src/workflow/cushion_target_tiered.py").read_text(), replay=(root / "src/backtest/full_path_tiered.py").read_text(),
            native=(root / "src/backtest/stage1_ibkr.py").read_text(), position=(root / "src/backtest/qlib_continuous_position.py").read_text())
    books = {}
    for window in plan["windows"]:
        data = frame.loc[frame.signal_date.between(window["start"], window["end"])].copy()
        for slip in plan["slippage_bps"]:
            for family in plan["families"]:
                policy = CushionTarget(floor_fraction=plan["floor_fraction"], multiplier=plan["multiplier"],
                    minimum_cash=plan["minimum_cash_weight"], maximum_soxl=plan["maximum_soxl_weight"], constant=family == "constant_beta")
                replay = replay_full_path(data, models=[policy], plan=plan, slippage_bps=slip, benchmark=family == "soxx")
                book = native_tiered_stage1(data, replay, provider_uri=provider, tariff=plan["tariff"], slippage_bps=slip,
                    initial_cash=plan["initial_cash"], freq="1min", opening_offset=pd.Timedelta(hours=9, minutes=47), position_type=QlibContinuousPosition)
                book["targets"] = replay["targets"]
                key = (window["name"], family, slip); books[key] = book
                recorder.save_objects(artifact_path="analysis/native", **{"_".join(map(str, key)): book})
                recorder.log_metrics(completed_native_cases=len(books))
                atomic_json({"recorder_id": recorder.id, "native_cases": len(books), "last_case": list(key)}, card / "progress.json")
                print("native",len(books),key,flush=True)
    rows, annual, windows, uncertainty = [], [], [], []
    for (window, family, slip), book in books.items():
        path, benchmark = book["authority"], books[(window, "soxx", slip)]["authority"]
        rows.append({"window": window, "family": family, "slippage_bps": slip, **paired_metrics(path, benchmark),
            "monitored_drawdown_pct": -100 * float(path.monitored_drawdown.max()),
            "average_cash_pct": 100 * float(path.ending_cash_weight.mean()), "orders": len(book["orders"]),
            "broker_fees_usd": float(path.broker_and_external_fees_usd.sum()), "slippage_usd": float(path.slippage_reserve_usd.sum())})
        years = annual_paired_metrics(path, benchmark); years["window"], years["family"], years["slippage_bps"] = window, family, slip
        annual.append(years)
        difference = np.log1p(path.strategy_return) - np.log1p(benchmark.strategy_return)
        rolling = difference.rolling(126).sum().dropna()
        windows.append({"window": window, "family": family, "slippage_bps": slip, "positive_126d_fraction": float(rolling.gt(1e-8).mean())})
        if family == "cushion_target":
            uncertainty.append({"window": window, "slippage_bps": slip,
                **stationary_bootstrap_excess(path, benchmark, mean_block=20, resamples=2000, alpha=.025, seed=20261013)})
    table, annual = pd.DataFrame(rows), pd.concat(annual, ignore_index=True)
    windows, uncertainty = pd.DataFrame(windows), pd.DataFrame(uncertainty)
    selected = table.loc[table.family.eq("cushion_target")]
    gates = {"net_excess_all_costs_and_starts": bool(selected.relative_wealth.gt(1 + 1e-8).all()),
        "drawdown_30pct": bool(selected.max_drawdown_pct.ge(-30).all() and selected.monitored_drawdown_pct.ge(-30).all()),
        "cvar_no_worse_than_soxx": bool(selected.cvar5_pct.ge(selected.benchmark_cvar5_pct).all()),
        "rolling_positive_70pct": bool(windows.loc[windows.family.eq("cushion_target"), "positive_126d_fraction"].ge(.70).all()),
        "bootstrap_positive": bool(uncertainty.lower_bound_positive.all()), "qualified_forward": False}
    historical = all(value for key, value in gates.items() if key != "qualified_forward")
    summary = {"experiment": plan["experiment"], "plan_sha256": hashlib.sha256((card / "plan.json").read_bytes()).hexdigest(),
        "decision": "needs_independent_forward" if historical else "not_promoted_cushion_target_gates_failed", "objective_achieved": False,
        "goal_remains_active": True, "model_fits": 0, "native_cases": len(books), "gates": gates,
        "provider_uri": str(provider), "as_of": plan["as_of"], "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
        "recorder_uri": str(tracking), "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "compact_metrics": rows, "production_enabled": False, "historical_windows_reused": True,
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml", "recorder_only_artifacts": ["inputs/*", "sources/*", "analysis/native/*"]}}
    recorder.save_objects(artifact_path="analysis", comparison=table, annual=annual, windows=windows, uncertainty=uncertainty, summary=summary)
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"; recorder.log_metrics(registered_artifact_count=len([p for p in artifacts.rglob("*") if p.is_file()]))
    for name, value in [("backtest_benchmark_comparison", table), ("regime_metrics", annual), ("window_metrics", windows), ("uncertainty", uncertainty)]: value.to_csv(card / (name + ".csv"), index=False)
    atomic_json({"recorder_id": recorder.id, "entries": [{"path": str(p.relative_to(card)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in sorted(artifacts.rglob("*")) if p.is_file()]}, card / "verification_manifest.json")
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": plan["experiment"], "provider_uri": str(provider), "recorder_uri": str(tracking), "as_of": plan["as_of"], "freq": "1min", "calendar_role": "sparse observed event marks", "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card, recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
    atomic_json(summary, card / "summary.json")
    text = "# 直接净值缓冲目标检验结论\n\n固定0.70峰值参考线、4倍缓冲，0次拟合、18个原生账户。目标暴露与恒定1.2beta、SOXX比较；长期连续与2023现金起点都保留。\n\n|窗口|策略|滑点bp|净收益%|监测回撤%|相对SOXX财富|\n|---|---|---:|---:|---:|---:|\n"
    for row in table.loc[table.slippage_bps.eq(2)].itertuples():text += f"|{row.window}|{row.family}|{row.slippage_bps}|{row.return_pct:.2f}|{row.monitored_drawdown_pct:.2f}|{row.relative_wealth:.4f}|\n"
    text += f"\n决定：`{summary['decision']}`。门槛：{json.dumps(gates,ensure_ascii=False)}。年度和滚动差异完整保留。\n\n"
    text += "这是直接分配预算的无预测控制，不是旧预测仓位的缩小上限。离散跳空、交易费、ETF相对3倍SOXX跟踪偏差与整数股数都可侵蚀参考线，不能声称未来回撤保证。原始价格/分红现金流与真实成交边界继承来源；历史复用不能代替独立前向。\n\n失败即结束该固定系数/映射，不另挑倍率或起点。原模拟盘未切换，目标继续active。\n"
    (card / "conclusion.md").write_text(text)
    return summary
