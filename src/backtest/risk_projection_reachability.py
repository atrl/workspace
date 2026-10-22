"""One development-year, independently settled source-target counterfactual."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import TieredFees
from src.backtest.native_raw_corporate import native_raw_validate
from src.backtest.raw_corporate_policy import IncomeLedger, settle_raw
from src.core.recorder_sources import load_verified_recorder_objects
from src.workflow.inventory_sales_source_qualification import normalized_notebook

NAME = "risk_projection_reachability_20261022"
SYMBOLS = ("SOXL", "SOXX")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def price_grid(raw, funded_days, terminal):
    require(isinstance(raw, pd.DataFrame) and list(raw.columns) == list(SYMBOLS),
            "source raw point schema differs")
    needed = []
    for day in funded_days:
        hits = raw.loc[raw.index.normalize() == day]
        times = [x.strftime("%H:%M") for x in hits.index]
        require(len(hits) == 4 and times[:3] == ["09:45", "09:47", "10:32"]
                and times[3] in ("13:00", "16:00"), "funded original point clock differs")
        needed += list(hits.index)
    guard = [terminal + pd.Timedelta(hours=9, minutes=45),
             terminal + pd.Timedelta(hours=9, minutes=47)]
    require(all(stamp in raw.index for stamp in guard), "following terminal raw guard absent")
    needed += guard
    values = raw.loc[pd.DatetimeIndex(needed), list(SYMBOLS)].astype(float)
    require(values.index.is_unique and values.index.is_monotonic_increasing
            and np.isfinite(values.to_numpy()).all() and (values.to_numpy() > 0).all(),
            "physical raw point grid invalid")
    return values


def source_target_path(trace, baseline_replay, prices, schedule, *, tariff, cost, initial_cash):
    trace = trace.copy()
    trace.trade_date = pd.to_datetime(trace.trade_date)
    funded = pd.DatetimeIndex(trace.trade_date)
    require(funded.is_unique and funded.is_monotonic_increasing, "source target date grid differs")
    reference = baseline_replay.copy()
    reference.trade_date = pd.to_datetime(reference.trade_date)
    reference.exit_date = pd.to_datetime(reference.exit_date)
    require(pd.DatetimeIndex(reference.trade_date).equals(funded), "baseline and source target days differ")
    terminal = pd.Timestamp(reference.exit_date.iloc[-1])
    require((reference.exit_date.iloc[:-1].reset_index(drop=True)
             == pd.Series(funded[1:])).all(), "baseline exit clock differs")
    grid = price_grid(prices, funded, terminal)
    income = IncomeLedger()
    fees = TieredFees(tariff, slippage_bps=cost)
    qty = np.zeros(2, dtype=float)
    cash = float(initial_cash)
    peak = float(initial_cash)
    pending = None
    records, orders, points, targets = [], [], [], []

    def mark(stamp):
        p = grid.loc[stamp, list(SYMBOLS)].to_numpy(float)
        nav = float(qty @ p + cash + income.value)
        points.append({"stamp": stamp, "q_soxl": float(qty[0]), "q_soxx": float(qty[1]),
                       "cash": cash, "receivable": income.value, "nav": nav})
        return nav

    for index, day in enumerate([*funded, terminal]):
        qty, cash, _ = income.apply(day, qty, cash, schedule)
        known_at = day + pd.Timedelta(hours=9, minutes=45)
        trade_at = day + pd.Timedelta(hours=9, minutes=47)
        known_nav = mark(known_at)
        pretrade = float(qty @ grid.loc[trade_at, list(SYMBOLS)].to_numpy(float) + cash + income.value)
        if pending is not None:
            pending.update(strategy_return=pretrade / pending.pop("start_nav") - 1.0,
                           equity=pretrade / initial_cash, native_expected_nav=pretrade,
                           ending_soxl_quantity=float(qty[0]), ending_soxx_quantity=float(qty[1]),
                           ending_cash=cash + income.value, ending_spendable_cash=cash,
                           ending_receivable=income.value,
                           ending_soxl_weight=float(qty[0] * grid.loc[trade_at, "SOXL"] / pretrade),
                           ending_soxx_weight=float(qty[1] * grid.loc[trade_at, "SOXX"] / pretrade),
                           ending_cash_weight=float((cash + income.value) / pretrade))
            records.append(pending)
            pending = None
        if index == len(funded):
            mark(trade_at)
            break
        row = trace.iloc[index]
        desired = np.array([row.source_target_soxl_weight, row.source_target_soxx_weight,
                            row.source_target_cash_weight], dtype=float)
        known_drawdown = max(0.0, 1.0 - known_nav / peak)
        require(np.isfinite(desired).all() and desired.min() >= 0
                and np.isclose(desired.sum(), 1.0), "source target not legal long-only weights")
        targets.append({"signal_date": reference.signal_date.iloc[index], "trade_date": day,
                        "decision_at": day + pd.Timedelta(hours=9, minutes=46),
                        "known_equity": known_nav, "known_drawdown": known_drawdown,
                        "execute_open_target": True, "soxl_weight": desired[0],
                        "soxx_weight": desired[1], "cash_weight": desired[2]})
        qty, cash, todays = settle_raw(qty, cash, income.value, desired=desired,
                                      prices=grid.loc[trade_at, list(SYMBOLS)].to_numpy(float),
                                      stamp=trade_at, fees=fees)
        todays = [{**order, "slot": "open", "trade_date": day} for order in todays]
        orders.extend(todays)
        mark(trade_at)
        mark(day + pd.Timedelta(hours=10, minutes=32))
        closing = grid.index[grid.index.normalize() == day][-1]
        close_nav = mark(closing)
        peak = max(peak, close_nav)
        total_fees = sum(order["total"] for order in todays)
        slippage = sum(order["slippage"] for order in todays)
        pending = {"signal_date": reference.signal_date.iloc[index], "trade_date": day,
                   "exit_date": reference.exit_date.iloc[index], "start_nav": pretrade,
                   "turnover": sum(order["notional"] for order in todays) / pretrade,
                   "fees": total_fees / pretrade,
                   "broker_and_external_fees_usd": total_fees - slippage,
                   "slippage_reserve_usd": slippage,
                   "portfolio_drawdown": known_drawdown,
                   "monitored_drawdown": max(0.0, 1.0 - close_nav / peak),
                   "risk_state": 0, "risk_clock": 0, "current_beta_cap": 1.2,
                   "execute_open_target": True, "recovery_action": False}
    path = {"replay": pd.DataFrame(records), "orders": pd.DataFrame(orders),
            "points": pd.DataFrame(points), "targets": pd.DataFrame(targets),
            "corporate_receipts": pd.DataFrame(income.receipts),
            "monthly_fee_shares": fees.monthly_shares}
    require(len(path["replay"]) == len(funded)
            and len(path["points"]) == len(grid), "independent source-target book incomplete")
    return path, grid


def run(root, *, recovery=False):
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    from src.data import build_qlib_store_from_frame
    from src.analysis.integrated_allocation import _path_metrics

    root = Path(root).resolve()
    card = root / "research" / NAME
    plan_path = card / ("recovery_plan.json" if recovery else "plan.json")
    plan = json.loads(plan_path.read_text())
    require(plan["frozen"] and plan["execution_authorized"], "frozen counterfactual plan absent")
    state_path = card / ("attempt2_run_state.json" if recovery else "run_state.json")
    require(not state_path.exists(), "counterfactual attempt already exists")
    if recovery:
        previous = json.loads((card / "run_state.json").read_text())
        require(previous.get("status") == "execution_failed" and previous.get("attempt") == 1,
                "prior failed attempt state differs")
        require(plan["recovery"]["attempt"] == plan["recovery"]["maximum_attempts"] == 2,
                "recovery attempt cap differs")
        for name, key in (("plan.json", "attempt1_plan_sha256"),
                          ("attempt1_failure.json", "attempt1_failure_sha256"),
                          ("attempt1_attribution.json", "attempt1_attribution_sha256")):
            require(sha(card / name) == plan["recovery"][key], f"recovery binding differs: {name}")
    require(sha(Path(__file__)) == plan["implementation"]["module_sha256"], "module bytes differ")
    notebook = card / (NAME + ".ipynb")
    require(hashlib.sha256(normalized_notebook(notebook)).hexdigest()
            == plan["implementation"]["notebook_sha256"], "notebook code differs")
    p3_plan_path = root / plan["source"]["P3_plan_path"]
    require(sha(p3_plan_path) == plan["source"]["P3_plan_sha256"], "P3 tariff plan changed")
    for field in ("feasibility", "state_diagnostic"):
        require(sha(root / plan["source"][field + "_path"])
                == plan["source"][field + "_sha256"], "source feasibility identity changed")
    source_plan = json.loads(p3_plan_path.read_text())
    require(source_plan["costs_and_budgets"]["initial_cash_usd"] == plan["price_and_account"]["initial_cash_usd"]
            and plan["price_and_account"]["slippage_bps"] == 2, "cost identity differs")
    for item in (plan["source"]["P3"], plan["source"]["raw"]):
        source = root / item["card"]
        require(sha(source / "summary.json") == item["summary_sha256"]
                and sha(source / "verification_manifest.json") == item["verification_manifest_sha256"],
                "source Recorder identity changed")
    provider = root / plan["price_and_account"]["provider_output"]
    require(not provider.exists(), "derived native provider already exists")
    state_path.write_text(json.dumps({"status": "started", "attempt": 2 if recovery else 1,
                                      "plan_sha256": sha(plan_path)}) + "\n")
    qlib.init(provider_uri=str(root / "data/qlib/stage1_uniform_daily_marks_20261009"), region="us")
    loaded = {}
    audits = {}
    for key in ("P3", "raw"):
        item = plan["source"][key]
        objects, audit = load_verified_recorder_objects(root / item["card"],
            expected_recorder_id=item["recorder_id"], artifacts=list(item["artifacts"]),
            expected_artifact_sha256=item["artifacts"],
            expected_manifest_sha256=item["verification_manifest_sha256"])
        loaded[key], audits[key] = objects, audit
    p3 = loaded["P3"]
    trace = p3["paths/B_path_control_2"]["trace"]
    reference = p3["paths/B_path_control_2"]["replay"]
    baseline = p3["native/B_path_control_2"]
    funded = pd.to_datetime(trace.trade_date)
    mask = funded.dt.year.eq(2023)
    source_trace = trace.loc[mask].copy().reset_index(drop=True)
    source_reference = reference.loc[pd.to_datetime(reference.trade_date).dt.year.eq(2023)].copy().reset_index(drop=True)
    require(len(source_trace) == len(source_reference) == plan["development_window"]["funded_sessions"],
            "bounded 2023 funded date count differs")
    source_path, point_prices = source_target_path(source_trace, source_reference,
        loaded["raw"]["inputs/raw_points"], loaded["raw"]["inputs/schedule"],
        tariff=source_plan["costs_and_budgets"]["tariff"],
        cost=plan["price_and_account"]["slippage_bps"],
        initial_cash=plan["price_and_account"]["initial_cash_usd"])
    fields = ["open", "high", "low", "close", "execution_price", "raw_execution_price"]
    native_rows = [{"datetime": stamp, "instrument": symbol,
                    **{field: float(value) for field in fields}}
                   for symbol in SYMBOLS for stamp, value in point_prices[symbol].items()]
    manifest = build_qlib_store_from_frame(pd.DataFrame(native_rows), provider_uri=provider,
        freq="1min", region="us", as_of="2024-01-02",
        field_map={name: name for name in fields},
        source="P3 exact hash-verified raw Recorder points; 2023 funded dates and 2024-01-02 terminal guard only")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    observed = D.features(list(SYMBOLS), ["$raw_execution_price"],
        start_time=point_prices.index[0], end_time=point_prices.index[-1], freq="1min")
    price_readback = observed["$raw_execution_price"].unstack("instrument").reindex(point_prices.index)[list(SYMBOLS)]
    pd.testing.assert_frame_equal(price_readback, point_prices, check_names=False)
    alternate = native_raw_validate(source_path, provider_uri=provider, point_prices=point_prices,
        schedule=loaded["raw"]["inputs/schedule"], tariff=source_plan["costs_and_budgets"]["tariff"],
        cost=plan["price_and_account"]["slippage_bps"],
        initial_cash=plan["price_and_account"]["initial_cash_usd"])
    require(alternate["audit"]["status"] == "passed" and len(alternate["authority"]) == 249,
            "new native source-target account failed")
    baseline_authority = baseline["authority"].loc[baseline["authority"].trade_date.dt.year.eq(2023)].copy()
    require(len(baseline_authority) == 249
            and pd.DatetimeIndex(alternate["authority"].trade_date).equals(pd.DatetimeIndex(baseline_authority.trade_date)),
            "baseline and alternate date grid differs")
    alt_metrics = _path_metrics(alternate["authority"])
    baseline_metrics = _path_metrics(baseline_authority)
    yearly = p3["analysis/results"]["annual_metrics"]
    soxx = yearly.loc[yearly.family.eq("SOXX") & yearly.slippage_bps.eq(2)
                      & yearly.kind.eq("year") & yearly.period.eq("2023")]
    require(len(soxx) == 1, "same-cost SOXX 2023 native reference absent")
    soxx_metrics = {name: float(soxx.iloc[0][name]) for name in
                    ("return_pct", "max_drawdown_pct", "cvar5_pct")}
    comparison = {"source_target": alt_metrics, "risk_projected_B2": baseline_metrics,
                  "SOXX_2bp_reference": soxx_metrics,
                  "source_to_projected_terminal_wealth_ratio": float(alternate["authority"].equity.iloc[-1] /
                                                                       baseline_authority.equity.iloc[-1]),
                  "source_to_SOXX_2023_relative_wealth": float((1 + alt_metrics["return_pct"] / 100) /
                                                                (1 + soxx_metrics["return_pct"] / 100)),
                  "projected_to_SOXX_2023_relative_wealth": float((1 + baseline_metrics["return_pct"] / 100) /
                                                                   (1 + soxx_metrics["return_pct"] / 100))}
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=NAME, uri=str(tracking)):
        rec = R.get_recorder()
        rec.save_objects(artifact_path="inputs", plan=plan, source_audits=audits,
                         derived_provider_manifest=manifest, bounded_raw_points=point_prices)
        rec.save_objects(artifact_path="native", source_target_account=alternate)
        rec.save_objects(artifact_path="analysis", matched_2023_comparison=comparison,
                         source_target_path=source_path)
        rec.save_objects(artifact_path="sources", module=Path(__file__).read_bytes(),
                         notebook=notebook.read_bytes())
        rec.log_metrics(source_to_projected_wealth_ratio=comparison["source_to_projected_terminal_wealth_ratio"],
                        alternate_native_sessions=249, new_native_accounts=1, model_fits=0)
        summary = {"experiment": NAME, "status": "completed_development_counterfactual_no_promotion",
                   "decision": "training_2023_projection_tradeoff_diagnostic_only",
                   "cost_bps": 2, "funded_sessions": 249,
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
                   "projected_B2_mean_cash_weight": float(baseline_authority.ending_cash_weight.mean()),
                   "source_target_order_fees_usd": float(alternate["orders"].total.sum()),
                   "projected_B2_order_fees_usd": float(baseline["orders"].loc[
                       pd.to_datetime(baseline["orders"].trade_date).dt.year.eq(2023), "total"].sum()),
                   "native_reconciliation_status": alternate["audit"]["status"],
                   "source_history_independent": False, "candidate_held_evaluation_performed": False,
                   "qualified_forward_days": 0, "SOXX_v2_goal_achieved": False,
                   "trading_or_promotion_permission": False,
                   "source_recorder_object_loads": 5, "D_features_readbacks": 1,
                   "cumulative_source_recorder_object_loads":
                       (plan["recovery"]["previous_source_recorder_object_loads"]
                        + plan["recovery"]["attribution_source_recorder_object_loads"] + 5)
                       if recovery else 5,
                   "attempt": 2 if recovery else 1,
                   "prior_failure_receipt": "attempt1_failure.json" if recovery else None,
                   "model_fits": 0, "native_accounts": 1, "bootstrap_runs": 0,
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
    state_path.write_text(json.dumps({"status": "completed", "attempt": 2 if recovery else 1,
                                      "recorder_id": rec.id,
                                      "plan_sha256": summary["plan_sha256"]}) + "\n")
    (card / "conclusion.md").write_text(
        "# 风控投影与原始目标：2023 训练期独立账户诊断\n\n"
        f"Qlib Recorder：`{rec.id}`。原始目标独立账户与既有 P3 B2 在同一 249 个资金日、2bp 滑点、"
        "原始股数/现金/分红和现行 IBKR Tiered 费用下比较；新账户经原生 Qlib 逐点对账。"
        f"原始目标相对风控投影的期末净财富比 {comparison['source_to_projected_terminal_wealth_ratio']:.6f}；"
        f"两账户的最大回撤分别 {alt_metrics['max_drawdown_pct']:.2f}% 与 "
        f"{baseline_metrics['max_drawdown_pct']:.2f}%。\n\n"
        "首次路径构造因错误套用旧 SOXL 30% 上限而在原生账户和派生 provider 之前停止；"
        "保留失败收据与一次只读归因。第二次只移除该非用户硬门，逐字执行保存的只做多、无融资、权重和为一原始目标。\n\n"
        "这是 2023 已复用历史上的路径对照，不是恢复动作模型、完整 0/2/5bp 费用实验、"
        "独立前向证据或稳定超过 SOXX 的策略。不能通过下调旧 0.825 阈值救援 P3。\n")
    return summary
