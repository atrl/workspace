"""Audit observed SIM fills and valuations without inventing missed model trades."""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd

from src.data.qlib_native import build_qlib_store_from_frame


SYMBOLS = ("SOXL", "SOXX")
DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
    "https://qlib.readthedocs.io/en/latest/reference/api.html#qlib.contrib.evaluate.risk_analysis",
]


def _read(path: Path):
    return json.loads(path.read_text())


def build_recovery_provider(source_dir: Path, provider_uri: Path, *, as_of: str) -> dict:
    """Only the existing src/data builder may ingest raw daily market exports."""
    frames = []
    for symbol in SYMBOLS:
        frame = pd.DataFrame(_read(source_dir / f"{symbol}_daily_unadjusted.json"))
        frame["instrument"] = symbol
        frame["datetime"] = pd.to_datetime(frame["time_key"])
        frame["factor"] = 1.0  # Original, unadjusted dollar prices for actual share counts.
        if not (frame["close"].round(2) == frame["close"]).all():
            raise ValueError("Unexpected sub-cent close; revise price precision explicitly")
        if ((frame["low"] > frame[["open", "close"]].min(axis=1))
                | (frame["high"] < frame[["open", "close"]].max(axis=1))).any():
            raise ValueError("Invalid raw OHLC relationships")
        frames.append(frame)
    return build_qlib_store_from_frame(
        pd.concat(frames, ignore_index=True), provider_uri=provider_uri,
        field_map={k: k for k in ["open", "close", "high", "low", "volume", "factor"]},
        as_of=as_of, source="Futu request_history_kline K_DAY AuType.NONE; observed SIM audit",
    )


def reconcile_observed_fills(orders: list[dict], positions: list[dict], balance: dict,
                             history: dict) -> tuple[pd.DataFrame, dict]:
    """Reconcile shares and cash; an unexplained net adjustment is never a fee."""
    rows, holdings, cash = [], dict.fromkeys(SYMBOLS, Decimal(0)), Decimal(str(history["initial_equity"]))
    seen = set()
    for order in sorted(orders, key=lambda x: (x["create_time"], x["order_id"])):
        symbol = order["code"].removeprefix("US.")
        if (order["order_status"] != "FILLED_ALL" or symbol not in SYMBOLS
                or order["trd_side"] not in ("BUY", "SELL")
                or not order["remark"].startswith(history["remark_prefix"])):
            raise ValueError("Unexpected or unconfirmed order in cycle export")
        if order["order_id"] in seen:
            raise ValueError("Duplicate broker order")
        seen.add(order["order_id"])
        quantity, price = Decimal(str(order["dealt_qty"])), Decimal(str(order["dealt_avg_price"]))
        if quantity <= 0 or price <= 0 or quantity != Decimal(str(order["qty"])):
            raise ValueError("Invalid or partial fill")
        signed = quantity if order["trd_side"] == "BUY" else -quantity
        holdings[symbol] += signed
        cash -= signed * price
        if min(holdings.values()) < 0:
            raise ValueError("Unexpected short position")
        rows.append({"order_id": order["order_id"], "time": order["create_time"],
                     "date": order["create_time"][:10], "symbol": symbol,
                     "side": order["trd_side"], "quantity": float(quantity),
                     "price": float(price), "signed_quantity": float(signed),
                     "cash_flow": float(-signed * price), "remark": order["remark"]})
    broker_holdings = {p["code"].removeprefix("US."): Decimal(str(p["qty"])) for p in positions}
    if holdings != broker_holdings:
        raise ValueError("Filled quantities do not reconcile to broker positions")
    by_id = {row["order_id"]: row for row in rows}
    for old in history["fills"]:
        row = by_id.get(old["order_id"])
        if row is None or any(row[k] != old[k] for k in ("symbol", "side", "quantity", "price")):
            raise ValueError("Historical observed fill changed or disappeared")
    broker_cash = Decimal(str(balance["cash"]))
    marked = broker_cash + sum(Decimal(str(p["market_val"])) for p in positions)
    if abs(marked - Decimal(str(balance["total_assets"]))) > Decimal("0.001"):
        raise ValueError("Broker account equity fails cash plus holdings reconciliation")
    fills = pd.DataFrame(rows)
    audit = {
        "initial_equity": float(history["initial_equity"]), "inception_date": history["inception_date"],
        "fill_count": len(rows), "historical_fill_count_matched": len(history["fills"]),
        "last_fill_time_new_york": rows[-1]["time"],
        "positions": {k: float(v) for k, v in holdings.items()},
        "buy_value": float(-fills.loc[fills.side == "BUY", "cash_flow"].sum()),
        "sell_value": float(fills.loc[fills.side == "SELL", "cash_flow"].sum()),
        "trade_only_cash": float(cash), "broker_cash": float(broker_cash),
        "unclassified_net_cash_adjustment": float(broker_cash - cash),
        "cash_adjustment_interpretation": "Aggregate residual; timing/components unavailable. Not actual per-fill fees.",
        "positions_reconciled": True, "account_equity_reconciled": True,
    }
    return fills, audit


def run_recovery_audit(*, source_dir: Path, historical_dir: Path, provider_uri: Path,
                       card_dir: Path, as_of: str, account_observed_at: str,
                       cash_confirmation_path: Path | None = None,
                       anchor_source_dir: Path | None = None) -> dict:
    from qlib.contrib.evaluate import risk_analysis
    from qlib.data import D
    from qlib.workflow import R

    previous_summary = _read(card_dir / "summary.json") if (card_dir / "summary.json").exists() else None
    history = _read(historical_dir / "paper_ledger_v2.json")
    positions = _read(source_dir / "STOCK_AND_OPTION_positions.json")
    balance = _read(source_dir / "STOCK_AND_OPTION_balance.json")[0]
    fills, audit = reconcile_observed_fills(
        _read(source_dir / "STOCK_AND_OPTION_orders.json"), positions, balance, history,
    )
    confirmation = _read(cash_confirmation_path) if cash_confirmation_path else None
    if confirmation:
        user = confirmation["user_confirmation"]
        if (user.get("external_cash_flows") is not False or user.get("account_reset") is not False
                or user.get("scope_start") != audit["inception_date"]
                or confirmation["initial_equity"] != audit["initial_equity"]
                or abs(confirmation["anchor_net_cash_adjustment"] - audit["unclassified_net_cash_adjustment"]) > 1e-6):
            raise ValueError("User cash confirmation does not apply to this audit snapshot")
        anchor_inputs = {}
        for key, filename in (("balance", "STOCK_AND_OPTION_balance.json"), ("positions", "STOCK_AND_OPTION_positions.json"),
                              ("orders", "STOCK_AND_OPTION_orders.json")):
            anchor_path = (anchor_source_dir / filename) if anchor_source_dir else source_dir / filename
            if hashlib.sha256(anchor_path.read_bytes()).hexdigest() != confirmation["sources"][key]["source_sha256"]:
                raise ValueError("Confirmed snapshot hash differs from immutable anchor")
            anchor_inputs[key] = _read(anchor_path)
        anchor_fills, anchor_audit = reconcile_observed_fills(
            anchor_inputs["orders"], anchor_inputs["positions"], anchor_inputs["balance"][0], history,
        )
        pd.testing.assert_frame_equal(fills, anchor_fills)
        if audit["broker_cash"] != anchor_audit["broker_cash"] or audit["positions"] != anchor_audit["positions"]:
            raise ValueError("Cash or positions changed after the user-confirmed snapshot")
        audit["external_cash_flows"] = "excluded_by_scoped_user_confirmation"
        audit["cash_adjustment_classification"] = "unallocated_net_investment_cash_flow"
        audit["observation_lineage"] = "Same cash, share quantities and all filled-order fields as confirmed anchor; later equity changes are market marks only."
    observation = _read(source_dir / "observation_manifest.json") if (source_dir / "observation_manifest.json").exists() else None
    start = audit["inception_date"]
    calendar = pd.DatetimeIndex(D.calendar(start_time=start, end_time=as_of, freq="day"))
    features = D.features(list(SYMBOLS), ["$close"], start_time=start, end_time=as_of, freq="day")
    # The source gate established cent precision; undo only float32 storage roundoff.
    close = features["$close"].unstack("instrument").astype(float).round(2).reindex(calendar)
    if close.isna().any().any() or str(calendar[-1].date()) != as_of:
        raise ValueError("Incomplete qlib daily closes")
    if not set(pd.to_datetime(fills.date)).issubset(set(calendar)):
        raise ValueError("Fill outside the audited qlib calendar")
    initial, cash, shares, rows = audit["initial_equity"], audit["initial_equity"], dict.fromkeys(SYMBOLS, 0.0), []
    for date, prices in close.iterrows():
        today = fills.loc[fills.date == str(date.date())]
        cash += float(today.cash_flow.sum())
        for fill in today.itertuples():
            shares[fill.symbol] += fill.signed_quantity
        rows.append({"date": date, "trade_only_cash": cash,
                     **{f"{s}_shares": shares[s] for s in SYMBOLS},
                     **{f"{s}_close": prices[s] for s in SYMBOLS},
                     "equity_before_unclassified_cash_adjustments": cash + sum(shares[s] * prices[s] for s in SYMBOLS),
                     "fill_count": len(today)})
    report = pd.DataFrame(rows).set_index("date")
    nav = report.equity_before_unclassified_cash_adjustments / initial
    report["return"] = nav.pct_change()
    report.loc[report.index[0], "return"] = nav.iloc[0] - 1
    risk = risk_analysis(report["return"], N=252, freq=None, mode="product")
    close_prices = close.iloc[-1].to_dict()
    quotes = {q["code"].removeprefix("US."): q for q in _read(source_dir / "quotes.json")}
    if any(close_prices[s] != quotes[s]["last_price"] for s in SYMBOLS):
        raise ValueError("Snapshot last_price is not the audited latest regular close")
    distributions = []
    for symbol in SYMBOLS:
        for event in _read(source_dir / f"{symbol}_corporate_actions.json"):
            if not start <= event["ex_div_date"] <= as_of:
                continue
            if event.get("split_ratio") not in (None, 0, 1):
                raise ValueError("Unmodeled share-changing corporate action in audit interval")
            prior_shares = float(fills.loc[(fills.symbol == symbol) & (fills.date < event["ex_div_date"]), "signed_quantity"].sum())
            dividend = float(event.get("per_cash_div") or 0)
            distributions.append({"symbol": symbol, "ex_div_date": event["ex_div_date"],
                                  "prior_shares": prior_shares, "per_cash_div": dividend,
                                  "expected_gross_entitlement": prior_shares * dividend})
    gross_distributions = sum(x["expected_gross_entitlement"] for x in distributions)
    audit["corporate_action_evidence"] = distributions
    audit["expected_gross_cash_distributions"] = gross_distributions
    audit["gross_distributions_less_net_adjustment"] = gross_distributions - audit["unclassified_net_cash_adjustment"]
    audit["distribution_limit"] = "Corporate action entitlement only; payment timing, actual SIM credits, withholding and fees not individually verified."
    audit["cash_flow_api_limitation"] = "Futu SIM does not support order_fee_query or account cash-flow query."
    audit["cash_flow_api_reference"] = "https://openapi.futunn.com/futu-api-doc/en/trade/get-acc-cash-flow.html"
    current_cash_close_mark = audit["broker_cash"] + sum(shares[s] * close_prices[s] for s in SYMBOLS)
    broker_equity = float(balance["total_assets"])
    benchmark = []
    for symbol in SYMBOLS:
        first_price = float(fills.loc[(fills.symbol == symbol) & (fills.side == "BUY"), "price"].iloc[0])
        benchmark.append({"benchmark": symbol, "initial_price": first_price,
                          "close_price": close_prices[symbol], "return_pct": (close_prices[symbol] / first_price - 1) * 100,
                          "basis": "Hypothetical fractional buy-and-hold at first observed 09:31 fill price; price only, no fees/distributions"})
    metrics = {
        "latest_close_revaluation_equity": current_cash_close_mark,
        "latest_close_revaluation_profit": current_cash_close_mark - initial,
        "latest_close_revaluation_return_pct": (current_cash_close_mark / initial - 1) * 100,
        "trade_only_latest_close_equity": float(report.equity_before_unclassified_cash_adjustments.iloc[-1]),
        "trade_only_latest_close_return_pct": (float(nav.iloc[-1]) - 1) * 100,
        "account_snapshot_equity": broker_equity, "account_snapshot_profit": broker_equity - initial,
        "account_snapshot_return_pct": (broker_equity / initial - 1) * 100,
        "trade_only_max_drawdown_pct": float(risk.loc["max_drawdown", "risk"]) * 100,
    }
    manifest = {str(p): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                for directory in (source_dir, historical_dir) for p in sorted(directory.glob("*.json"))}
    card_dir.mkdir(parents=True, exist_ok=True)
    experiment = "stage1_recovery_20260928"
    # Absolute paths avoid Qlib's file:// experiment-lock path stripping bug.
    with R.start(experiment_name=experiment, uri=str((card_dir / "qlib_recorder").resolve())):
        recorder = R.get_recorder()
        R.log_params(provider_uri=str(provider_uri), region="us", as_of=as_of, inception_date=start,
                     analysis="Observed broker fills; no model replay", cash_adjustment_timing="unknown")
        R.log_metrics(**metrics)
        R.save_objects(artifact_path="broker_audit", **{
            "observed_fills.pkl": fills, "observed_holdings_valuation.pkl": report,
            "risk_analysis.pkl": risk, "benchmark.pkl": pd.DataFrame(benchmark),
            "source_manifest.pkl": manifest, "reconciliation.pkl": audit,
        })
        for folder, directory in (("broker", source_dir), ("historical", historical_dir)):
            for path in sorted(directory.glob("*.json")):
                recorder.log_artifact(str(path), artifact_path=f"source/{folder}")
        recorder.log_artifact(str(provider_uri / "qlib_store_manifest.json"), artifact_path="source")
        if cash_confirmation_path:
            recorder.log_artifact(str(cash_confirmation_path), artifact_path="source/confirmation")
        if anchor_source_dir:
            for key in ("balance", "positions", "orders"):
                anchor_path = anchor_source_dir / Path(confirmation["sources"][key]["source_ref"]).name
                recorder.log_artifact(str(anchor_path), artifact_path="source/anchor")
        # Read the primary report back through qlib Recorder before accepting results.
        restored = recorder.load_object("broker_audit/observed_holdings_valuation.pkl")
        pd.testing.assert_frame_equal(report, restored)
        recorder_id, recorder_dir = recorder.id, recorder.get_local_dir()
        experiment_id = recorder.experiment_id
    summary = {
        "card_type": "observed_sim_account_recovery_audit", "status": "audited_with_cash_timing_limitation",
        "execution_environment": "SIMULATE", "promotion_decision": "not_applicable_no_model_change",
        "provider_uri": str(provider_uri), "region": "us", "market": "all", "as_of": as_of,
        "data_range": [start, as_of], "trading_days": len(report),
        "account_observed_at": account_observed_at,
        "account_observed_at_precision": "recorded_query_start_and_end" if observation else "approximate acquisition window",
        "close_prices": close_prices, "account_nominal_prices": {p["code"]: p["nominal_price"] for p in positions},
        "valuation_basis": "Close revaluation uses the 2026-09-28 observed cash balance; it is not an independently confirmed 2026-09-25 statement.",
        "performance_scope": "Actual observed fills through 2026-08-12, then unchanged shares; not a continuously executed model return.",
        "daily_curve_basis": "Confirmed fills and qlib closes; unclassified cash adjustments are excluded because their timing is unknown.",
        "quote_timestamp_policy": "Snapshot update_time is not the timestamp of last_price; latest regular close was verified against unadjusted daily K bars.",
        "reconciliation": audit, "metrics": metrics, "benchmark": benchmark,
        "model_selection": {"performed": False}, "experiment_name": experiment,
        "experiment_id": experiment_id, "workflow_config": "workflow.yaml",
        "recorder_id": recorder_id, "recorder_local_dir": recorder_dir,
        "recorder_uri": str((card_dir / "qlib_recorder").resolve()),
        "qlib_native_api": ["qlib.init", "qlib.data.D.calendar", "qlib.data.D.features",
                            "qlib.workflow.R.start", "qlib.workflow.Recorder.load_object",
                            "qlib.contrib.evaluate.risk_analysis"],
        "qlib_docs_reference": DOCS,
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "primary_artifact": f"{experiment}.ipynb", "machine_entrypoint": "summary.json",
                            "prediction_artifacts": "not_applicable_observed_account_audit_no_model",
                            "recorder_only_artifacts": ["broker_audit/observed_fills.pkl", "broker_audit/observed_holdings_valuation.pkl",
                                                        "broker_audit/risk_analysis.pkl", "broker_audit/benchmark.pkl", "source/"]},
        "sidecars": ["summary.json", "conclusion.md", "risk_analysis.csv", "backtest_benchmark_comparison.csv"],
    }
    if confirmation:
        summary["cash_flow_confirmation"] = {
            "user_confirmation": confirmation["user_confirmation"],
            "evidence_recorded_at": confirmation["evidence_recorded_at"],
            "recorder_artifact": "source/confirmation/net_cash_adjustment_confirmation.json",
            "classification": "unallocated_net_investment_cash_flow", "total_fees": None,
        }
    if observation:
        summary["account_observation_window"] = {
            "started_at": observation["query_started_at"], "completed_at": observation["query_completed_at"],
            "api_calls": observation["calls"],
        }
    if previous_summary:
        summary["previous_observations"] = [*previous_summary.get("previous_observations", []), {
            "account_observed_at": previous_summary["account_observed_at"],
            "recorder_id": previous_summary["recorder_id"],
            "recorder_local_dir": previous_summary["recorder_local_dir"],
            "account_snapshot_equity": previous_summary["metrics"]["account_snapshot_equity"],
            "account_snapshot_return_pct": previous_summary["metrics"]["account_snapshot_return_pct"],
        }]
    observation_label = (
        f"{observation['query_started_at']} 至 {observation['query_completed_at']} 查询窗口"
        if observation else f"{account_observed_at} 约时段"
    )
    conclusion = (
        "# 富途 Stage1 模拟盘实际收益审计\n\n"
        f"从 {start} 的 $100,000 起点核对 {audit['fill_count']} 笔已成交订单，历史15笔全部匹配，当前持仓 SOXL 200 股、SOXX 98 股。最后成交为纽约时间 {audit['last_fill_time_new_york']}；此后收益来自未再调仓的持仓，不代表模型持续运行。\n\n"
        f"按 {as_of} 正式收盘 SOXL ${close_prices['SOXL']:.2f}、SOXX ${close_prices['SOXX']:.2f}，使用查询时现金重估权益 ${current_cash_close_mark:,.3f}，收益 {metrics['latest_close_revaluation_return_pct']:.6f}%。"
        f"{observation_label}账户权益 ${broker_equity:,.3f}，收益 {metrics['account_snapshot_return_pct']:.6f}%；两者是不同价格时点。行情快照 update_time 不能当作 last_price 的盘前成交时间。\n\n"
        f"买入 ${audit['buy_value']:,.3f}、卖出 ${audit['sell_value']:,.3f}，成交现金流推算现金 ${audit['trade_only_cash']:,.3f}，实际现金 ${audit['broker_cash']:,.3f}，差额 +${audit['unclassified_net_cash_adjustment']:,.3f} 为未分类净现金调整；未取得明细，不能称为逐笔手续费或直接认定分红。"
        f"企业行动资料显示本周期理论税前现金分配合计 ${gross_distributions:.4f}（SOXL 9月22日 $0.0929×200，SOXX 9月15日 $0.32505×98）；减去净现金调整后差额 ${audit['gross_distributions_less_net_adjustment']:.4f}，与费用等扣款相容，但不是已核实实际手续费。SIM 费用查询及资金流水接口不支持，支付日期、实入金额与扣款无法逐笔确认。"
        f"因此收盘重估不是已核实的9月25日历史结单。未分配该调整的成交现金流口径收盘权益 ${metrics['trade_only_latest_close_equity']:,.3f}，收益 {metrics['trade_only_latest_close_return_pct']:.6f}%。\n\n"
        f"qlib Data API 读取 {len(report)} 个交易日；日度曲线排除时间未知现金调整，qlib 几何风险分析最大回撤 {metrics['trade_only_max_drawdown_pct']:.4f}%。风险数只适用于此重建口径。基准是同起始成交价的假设买入持有价格收益，未计费用及分配款。\n\n"
        "本卡只审计实际模拟账户，不训练、不晋升、不补造遗漏交易、不修改运行账本。下一步补齐现金流水，再与修复后模型反事实回放分开比较。\n\n"
        f"Recorder: `{recorder_id}`。明细、原始快照、哈希及风险分析保存在 qlib recorder；专属 provider 来源为不复权日K。\n"
    )
    if confirmation:
        conclusion += cash_confirmation_conclusion(
            confirmation, audit["unclassified_net_cash_adjustment"],
            account_return_pct=metrics["account_snapshot_return_pct"],
            close_return_pct=metrics["latest_close_revaluation_return_pct"],
        )
    (card_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (card_dir / "conclusion.md").write_text(conclusion)
    risk.to_csv(card_dir / "risk_analysis.csv")
    pd.DataFrame(benchmark).to_csv(card_dir / "backtest_benchmark_comparison.csv", index=False)
    for name in ("code_diff.txt", "code_status.txt", "code_commit.txt"):
        (Path(recorder_dir) / "artifacts" / name).unlink(missing_ok=True)
    return summary


def cash_confirmation_conclusion(confirmation: dict, adjustment: float, *, account_return_pct: float, close_return_pct: float) -> str:
    """A dated evidence addendum leaves the original measured numbers unchanged."""
    return (
        f"\n补充确认（记录于 {confirmation['evidence_recorded_at']}）：用户原文“{confirmation['user_confirmation']['text']}”。"
        f"作用范围为8月3日初始10万美元至原始 {confirmation['snapshot_observed_at']} 账户快照，因此可以排除该区间外部加减资金及重置，"
        f"原始确认后已核对现金、持仓及全部成交不变，查询时权益相对初始本金的{account_return_pct:.6f}%可明确为本周期账户总收益。"
        f"现金差额 +${adjustment:.3f} 归为未分配的净投资现金流；股息、费用与其他投资现金项目仍未逐笔分解，"
        f"总手续费未知，历史到账日期不分配。最近收盘重估{close_return_pct:.6f}%的价格/现金时点限制不变。\n"
    )
