"""Matched native accounts for frozen versus monthly-updated price forecasts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import paired_metrics, annual_paired_metrics, stationary_bootstrap_excess
from src.backtest.cushion_target import CushionTarget
from src.backtest.forecast_vertices import ForecastVertices
from src.backtest.full_path_tiered import replay_full_path
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
from src.workflow.minute_action_value import _source


class ConstantWeights:
    def __init__(self, weights):
        self.weights = weights

    def target(self, row, q, cash, known_prices, peak):
        return self.weights


def run_monthly_forecast_account(*, root, card, plan):
    import qlib
    import yaml
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R
    digest = hashlib.sha256((card / "plan.json").read_bytes()).hexdigest()
    if (card / "summary.json").exists():
        previous = json.loads((card / "summary.json").read_text())
        if previous["plan_sha256"] != digest: raise ValueError("fixed account plan changed")
        return previous
    provider = root / "data/qlib/full_path_tiered_allocation_20261013"
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    months = pd.period_range("2021-01", "2026-10", freq="M")
    model_keys = ["models/monthly_" + str(month).replace("-", "_") for month in months]
    forecast, forecast_audit = _source(root, plan["forecast_source"], ["inputs/grouped", "analysis/updated_predictions", *model_keys])
    anchor_keys = ["analysis/native/reused_outer_anchor_frozen_" + str(slip) for slip in plan["slippage_bps"]]
    prices, price_audit = _source(root, plan["price_source"], ["inputs/frame", "inputs/plan", *anchor_keys])
    control_keys = ["analysis/native/cash_start_2023_" + family + "_" + str(slip) for family in ["soxx", "cushion_target"] for slip in plan["slippage_bps"]]
    controls, control_audit = _source(root, plan["control_source"], ["inputs/plan", *control_keys])
    for source_plan in [prices["inputs/plan"], controls["inputs/plan"]]:
        if source_plan["tariff"] != plan["tariff"] or source_plan["initial_cash"] != plan["initial_cash"]:
            raise ValueError("reused control fee/capital differs")
    release = root / "production/stage1_futu_sim_v1/runtime_release.json"
    if hashlib.sha256(release.read_bytes()).hexdigest() != prices["inputs/plan"]["release_sha256"]:
        raise ValueError("current reference release changed")
    frame = prices["inputs/frame"]
    data = frame.loc[frame.signal_date.between(plan["windows"][0]["start"], plan["as_of"])].copy()
    grouped = forecast["inputs/grouped"]
    handler = DataHandlerLP.from_df(grouped)
    predictions = {"frozen": [], "monthly": []}
    for month in months:
        dates = data.index[data.index.to_period("M") == month]
        if not len(dates): continue
        dataset = DatasetH(handler=handler, segments={"test": (dates.min(), dates.max())})
        for family in predictions:
            chosen = str(month) if family == "monthly" else str(min(month.year, 2023)) + "-01"
            model = forecast["models/monthly_" + chosen.replace("-", "_")]
            if model.cutoff > dates.min() or pd.Timestamp(model.audit["latest_label_available_at"]) >= model.cutoff:
                raise ValueError("model not historically available")
            predictions[family].append(model.predict(dataset).xs("SOXX", level="instrument"))
    predictions = {name: pd.concat(parts).sort_index() for name, parts in predictions.items()}
    original_monthly = forecast["analysis/updated_predictions"].xs("SOXX", level="instrument")
    common = original_monthly.index.intersection(predictions["monthly"].index)
    np.testing.assert_allclose(predictions["monthly"].loc[common], original_monthly.loc[common], atol=1e-12, rtol=1e-12)
    for prediction in predictions.values():
        if not prediction.index.equals(data.index): raise ValueError("account forecast calendar differs")
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=plan["experiment"], recorder_name="monthly_update_matched_accounts", uri=str(tracking)):
        recorder = R.get_recorder(); recorder.log_params(plan_sha256=digest, model_fits=0, maximum_new_native_cases=24)
        recorder.save_objects(artifact_path="inputs", plan=plan, frame=data, predictions=predictions,
            source_audits={"forecast": forecast_audit, "prices": price_audit, "controls": control_audit})
        recorder.save_objects(artifact_path="sources", policy=(root / "src/backtest/forecast_vertices.py").read_text(),
            replay=(root / "src/backtest/full_path_tiered.py").read_text(), workflow=Path(__file__).read_text())
    books, new_count, reused_count, mean_controls = {}, 0, 0, []
    for window in plan["windows"]:
        window_data = data.loc[data.signal_date.between(window["start"], window["end"])]
        for slip in plan["slippage_bps"]:
            for family in plan["families"]:
                reused = window["name"] == "cash_start_2023" and family in {"soxx", "cushion_target"}
                if reused:
                    book = controls["analysis/native/cash_start_2023_" + family + "_" + str(slip)]
                    reused_count += 1
                else:
                    if new_count >= plan["maximum_new_native_cases"]: raise ValueError("native account budget exhausted")
                    if family in {"frozen", "monthly"}:
                        policy = ForecastVertices(predictions[family], plan=plan, slippage_bps=slip)
                    elif family == "matched_mean_beta":
                        path = books[(window["name"], "monthly", slip)]["authority"]
                        beta = float((3*path.ending_soxl_weight+path.ending_soxx_weight).mean())
                        xl = max(0., (beta-(1-plan["minimum_cash_weight"]))/2)
                        weights = np.array([xl, beta-3*xl, 1-beta+2*xl])
                        if xl > plan["maximum_soxl_weight"]+1e-10: raise ValueError("mean exposure control exceeds target family")
                        policy = ConstantWeights(weights)
                        mean_controls.append({"window": window["name"], "slippage_bps": slip, "ex_post_mean_beta": beta,
                            "soxl_weight": weights[0], "soxx_weight": weights[1], "cash_weight": weights[2]})
                    else:
                        policy = CushionTarget(floor_fraction=plan["floor_fraction"], multiplier=plan["multiplier"],
                            minimum_cash=plan["minimum_cash_weight"], maximum_soxl=plan["maximum_soxl_weight"])
                    replay = replay_full_path(window_data, models=[policy], plan=plan, slippage_bps=slip, benchmark=family == "soxx")
                    book = native_tiered_stage1(window_data, replay, provider_uri=provider, tariff=plan["tariff"], slippage_bps=slip,
                        initial_cash=plan["initial_cash"], freq="1min", opening_offset=pd.Timedelta(hours=9, minutes=47), position_type=QlibContinuousPosition)
                    book["targets"] = replay["targets"]; new_count += 1
                if not pd.to_datetime(book["authority"].trade_date).reset_index(drop=True).equals(window_data.trade_date.reset_index(drop=True)):
                    raise ValueError("reused/native account dates differ")
                key = (window["name"], family, slip); books[key] = book
                recorder.save_objects(artifact_path="analysis/native", **{"_".join(map(str, key)): book})
                recorder.log_metrics(new_native_cases=new_count, reused_cases=reused_count)
                print("account", new_count, reused_count, key, flush=True)
            if window["name"] == "cash_start_2023":
                book = prices["analysis/native/reused_outer_anchor_frozen_" + str(slip)]
                if not pd.to_datetime(book["authority"].trade_date).reset_index(drop=True).equals(window_data.trade_date.reset_index(drop=True)):
                    raise ValueError("current anchor dates differ")
                key = (window["name"], "current_anchor", slip); books[key] = book; reused_count += 1
                recorder.save_objects(artifact_path="analysis/native", **{"_".join(map(str, key)): book})
    if new_count != plan["maximum_new_native_cases"] or reused_count != plan["expected_reused_cases"]:
        raise ValueError("native case count differs from fixed plan")
    rows, annuals, rolling, uncertainty, matched = [], [], [], [], []
    for (window, family, slip), book in books.items():
        path, benchmark = book["authority"], books[(window, "soxx", slip)]["authority"]
        rows.append({"window": window, "family": family, "slippage_bps": slip, **paired_metrics(path, benchmark),
            "monitored_drawdown_pct": -100*float(path.monitored_drawdown.max()), "average_cash_pct": 100*float(path.ending_cash_weight.mean()),
            "orders": len(book["orders"]), "broker_fees_usd": float(path.broker_and_external_fees_usd.sum()), "slippage_usd": float(path.slippage_reserve_usd.sum())})
        annuals.append(annual_paired_metrics(path, benchmark).assign(window=window, family=family, slippage_bps=slip))
        relative = np.log1p(path.strategy_return)-np.log1p(benchmark.strategy_return)
        rolling.append({"window": window, "family": family, "slippage_bps": slip, "positive_126d_fraction": float(relative.rolling(126).sum().dropna().gt(0).mean())})
        if family == "monthly":
            uncertainty.append({"window": window, "slippage_bps": slip, **stationary_bootstrap_excess(path, benchmark, mean_block=63, resamples=2000, alpha=.025, seed=20261014)})
            for control in ["frozen", "cushion_target", "matched_mean_beta", *(["current_anchor"] if window == "cash_start_2023" else [])]:
                matched.append({"window": window, "control": control, "slippage_bps": slip, **paired_metrics(path, books[(window, control, slip)]["authority"])})
    table, annual, rolls, uncertain, paired = pd.DataFrame(rows), pd.concat(annuals, ignore_index=True), pd.DataFrame(rolling), pd.DataFrame(uncertainty), pd.DataFrame(matched)
    candidate = table.loc[table.family.eq("monthly")]
    gates = {"net_excess_all_costs_starts": bool(candidate.relative_wealth.gt(1+1e-8).all()),
        "drawdown_30pct": bool(candidate.max_drawdown_pct.ge(-30).all() and candidate.monitored_drawdown_pct.ge(-30).all()),
        "cvar_no_worse_soxx": bool(candidate.cvar5_pct.ge(candidate.benchmark_cvar5_pct).all()),
        "positive_rolling_70pct": bool(rolls.loc[rolls.family.eq("monthly"), "positive_126d_fraction"].ge(.70).all()),
        "positive_years_70pct": bool(annual.loc[annual.family.eq("monthly")].groupby(["window", "slippage_bps"]).relative_wealth.apply(lambda values: values.gt(1).mean()).ge(.70).all()),
        "positive_bootstrap": bool(uncertain.lower_bound_positive.all()), "matched_controls": bool(paired.relative_wealth.gt(1+1e-8).all()), "qualified_forward": False}
    historical = all(value for key, value in gates.items() if key != "qualified_forward")
    summary = {"experiment": plan["experiment"], "plan_sha256": digest, "decision": "needs_independent_forward" if historical else "not_promoted_monthly_account_gates_failed",
        "gates": gates, "new_model_fits": 0, "new_native_cases": new_count, "reused_cases": reused_count, "objective_achieved": False,
        "goal_remains_active": True, "production_enabled": False, "historical_windows_reused": True, "risk_monitoring": plan["risk_monitoring"],
        "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id), "recorder_uri": str(tracking), "recorder_local_dir": str(tracking/str(recorder.experiment_id)/recorder.id),
        "provider_uri": str(provider), "as_of": plan["as_of"], "qlib_docs": "https://qlib.readthedocs.io/en/latest/component/strategy.html",
        "compact_metrics": [{key: row[key] for key in ["window", "family", "slippage_bps", "return_pct", "max_drawdown_pct", "monitored_drawdown_pct", "relative_wealth", "cvar5_pct", "orders", "broker_fees_usd", "slippage_usd", "average_cash_pct"]} for row in rows],
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml", "recorder_only_artifacts": ["inputs/*", "sources/*", "analysis/native/*"]}}
    recorder.save_objects(artifact_path="analysis", comparison=table, annual=annual, rolling=rolls, uncertainty=uncertain, matched=paired, mean_controls=mean_controls, summary=summary)
    artifacts = Path(summary["recorder_local_dir"])/"artifacts"; recorder.log_metrics(registered_artifact_count=len([p for p in artifacts.rglob("*") if p.is_file()]))
    atomic_json({"recorder_id": recorder.id, "entries": [{"path": str(p.relative_to(card)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in sorted(artifacts.rglob("*")) if p.is_file()]}, card/"verification_manifest.json")
    for name, value in [("backtest_benchmark_comparison", table), ("regime_metrics", annual), ("window_metrics", rolls), ("uncertainty", uncertain), ("matched_controls", paired), ("mean_exposure_controls", pd.DataFrame(mean_controls))]: value.to_csv(card/(name+".csv"), index=False)
    (card/"workflow.yaml").write_text(yaml.safe_dump({"experiment": plan["experiment"], "provider_uri": str(provider), "recorder_uri": str(tracking), "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    summary["nas_evidence"] = archive_nas_recorder(card/"nas_binding.json", workspace_root=root, card=card, recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
    atomic_json(summary, card/"summary.json")
    text = "# 月度更新完整账户检验\n\n0新增拟合，24个新增原生账户、9个已验证对照复用。初始10万美元，IBKR Pro Tiered，分别运行0/2/5bp。\n\n|窗口|策略|净收益%|监测回撤%|相对SOXX财富|\n|---|---|---:|---:|---:|\n"
    for row in table.loc[table.slippage_bps.eq(2)].itertuples(): text += f"|{row.window}|{row.family}|{row.return_pct:.2f}|{row.monitored_drawdown_pct:.2f}|{row.relative_wealth:.4f}|\n"
    text += f"\n决定：{summary['decision']}。门槛：{json.dumps(gates,ensure_ascii=False)}。\n\n"
    text += "冻结与月度模型使用同一目标映射、风险预算、费用估算和执行时点。HOLD确实不下单；预估成本按已知价格和第一费率档，实际账户按当月累计股数结算。恒定平均暴露对照为事后诊断，不可部署；current_anchor仅为同执行时钟的历史策略参考，不等于实际模拟盘成交。\n\n"
    text += "回撤依据日度原生收益与已知/收盘净值点，不宣称全分钟最大回撤或未来保证；价格/分红现金流及真实成交资格继承来源限制。校准改善不能替代收益和风险门。原生产SIM未改，目标未完成；失败则封存固定映射与更新规则，不改频率、风险倍率或成本假设寻找胜出。\n"
    (card/"conclusion.md").write_text(text)
    return summary
