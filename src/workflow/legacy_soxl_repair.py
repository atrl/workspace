"""Recorder-first evaluation of a repaired legacy rule, using native Qlib fills.

The independent rule engine freezes next-open orders. Qlib Exchange, Account,
Position and daily reports own portfolio evidence. No parameters are selected
from the results, and annual tables slice continuous paths without restarting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT = "legacy_soxl_repair_20261005"
DOCS = ["https://qlib.readthedocs.io/en/latest/component/strategy.html",
        "https://qlib.readthedocs.io/en/latest/component/recorder.html"]
VARIANTS = ("legacy_repaired", "soxx_buy_hold", "soxl_buy_hold", "soxl_fixed_50", "soxl_fixed_90")
SOURCE_FILES = (
    "src/workflow/legacy_soxl_repair.py", "src/backtest/legacy_soxl_replay.py",
    "src/backtest/qlib_continuous_position.py", "src/data/legacy_vix_source.py", "src/data/legacy_execution_view.py",
    "src/data/qlib_native.py", "src/data/price_basis.py", "src/data/constrained_net_excess.py",
    "soxl_backtest/config/backtest_config.py", "soxl_backtest/indicators/indicator_calculator.py",
    "soxl_backtest/position/position_manager.py", "soxl_backtest/strategy/base_strategy.py",
    "soxl_backtest/strategy/buy_strategy.py", "soxl_backtest/strategy/sell_strategy.py",
    "soxl_backtest/indicators/panic_calculators/base.py", "soxl_backtest/indicators/panic_calculators/sigmoid.py",
)


def _source_hashes(root):
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def _authorization(root, card, plan, source_audit):
    path = card / "run_authorization.json"
    if not path.is_file():
        raise ValueError("blocked: root run authorization is absent")
    authorization = json.loads(path.read_text())
    plan_bytes = (card / "plan.json").read_bytes()
    if (authorization.get("authorized") is not True
            or authorization.get("plan_file_sha256") != hashlib.sha256(plan_bytes).hexdigest()
            or json.loads(plan_bytes) != plan
            or authorization.get("source_sha256") != _source_hashes(root)
            or authorization.get("vix_snapshot_manifest_sha256") != source_audit["vix"]["snapshot_manifest_sha256"]
            or authorization.get("execution_manifest_sha256") != source_audit["execution"]["execution_manifest_sha256"]
            or authorization.get("deadline_utc") != plan["deadline_utc"]):
        raise ValueError("blocked: plan, source, VIX or deadline authorization seal differs")
    return authorization


def _deadline(plan):
    if pd.Timestamp.now(tz="UTC") >= pd.Timestamp(plan["deadline_utc"]):
        raise ValueError("blocked: registered research deadline reached")


def _verify_cached_recorder(summary, recorder):
    if recorder.status != "FINISHED" or recorder.load_object("audit/summary.pkl") != summary:
        raise ValueError("blocked: cached Recorder is not FINISHED or differs from local summary")
    manifest = recorder.load_object("audit/artifact_manifest.pkl")
    if manifest != summary["recorder_artifact_sha256"]:
        raise ValueError("blocked: cached artifact manifest differs")
    directory = Path(recorder.get_local_dir()) / "artifacts"
    for name, digest in manifest.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"blocked: cached Recorder artifact hash changed: {name}")



def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def _dates(frame):
    result = frame.copy()
    if "date" in result:
        result = result.set_index("date")
    result.index = pd.DatetimeIndex(result.index).normalize()
    if result.empty or result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError("blocked: daily evidence must have unique ordered sessions")
    return result


def native_order_backtest(*, dates, cost_bps, initial_cash, trades=None, control=None):
    """Use the currently initialized provider; never reinitialize active Recorder.

    Orders use integer units of the qualified adjusted price basis. They are not
    raw broker shares. Missing $factor disables native rounding; requested and
    filled quantities are checked explicitly, so this is not concealed.
    """
    from qlib.backtest import backtest
    from qlib.backtest.decision import Order, TradeDecisionWO
    from qlib.strategy.base import BaseStrategy
    from src.backtest.qlib_continuous_position import QlibContinuousPosition

    dates = pd.DatetimeIndex(dates)
    if (dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing
            or not np.isfinite([cost_bps, initial_cash]).all() or cost_bps < 0 or initial_cash <= 0):
        raise ValueError("blocked: invalid native replay sessions or assumptions")
    if (trades is None) == (control is None):
        raise ValueError("blocked: supply exactly one frozen order schedule or fixed control")
    schedule = pd.DataFrame() if trades is None else trades.copy()
    if not schedule.empty:
        required = {"date", "signal_date", "symbol", "amount", "price", "fee"}
        if required - set(schedule):
            raise ValueError("blocked: frozen orders lack execution evidence")
        schedule["date"] = pd.to_datetime(schedule.date).dt.normalize()
        schedule["signal_date"] = pd.to_datetime(schedule.signal_date).dt.normalize()
        amounts = schedule.amount.to_numpy(float)
        if (not schedule.date.isin(dates).all() or not schedule.signal_date.lt(schedule.date).all()
                or not schedule.symbol.eq("SOXL").all() or not np.isfinite(amounts).all()
                or not np.equal(amounts, np.trunc(amounts)).all() or (amounts == 0).any()):
            raise ValueError("blocked: orders must be prior-signal integer SOXL quantities")
    if control is not None and control not in VARIANTS[1:]:
        raise ValueError("blocked: unregistered fixed control")
    execution_rows, submitted = [], []
    cost = cost_bps / 10_000.

    class FrozenOrders(BaseStrategy):
        def generate_trade_decision(self, execute_result=None):
            start, end = self.trade_calendar.get_step_time(self.trade_calendar.get_trade_step())
            day = pd.Timestamp(start).normalize()
            if day not in dates:
                raise ValueError("blocked: Qlib introduced an unexpected session")
            orders = []
            if control is None:
                rows = schedule.loc[schedule.date.eq(day)] if not schedule.empty else schedule
                for row in rows.itertuples():
                    quantity = float(row.amount)
                    order = Order(row.symbol, np.float64(abs(quantity)), Order.BUY if quantity > 0 else Order.SELL, start, end)
                    price = float(self.trade_exchange.get_deal_price(row.symbol, start, end, direction=order.direction))
                    if not np.isclose(price, row.price, rtol=0, atol=1e-9):
                        raise ValueError("blocked: engine fill price differs from Qlib open")
                    orders.append(order)
                    submitted.append((order, row))
            else:
                symbol = "SOXX" if control == "soxx_buy_hold" else "SOXL"
                held = float(self.trade_position.get_stock_amount_dict().get(symbol, 0.))
                cash = float(self.trade_position.get_cash())
                price = float(self.trade_exchange.get_deal_price(symbol, start, end, direction=Order.BUY))
                if not np.isfinite(price) or price <= 0:
                    raise ValueError("blocked: control has no native open")
                if control.endswith("buy_hold"):
                    amount = np.floor(cash / (price * (1 + cost))) if day == dates[0] else 0.
                else:
                    weight = .5 if control.endswith("50") else .9
                    nav = cash + held * price
                    # Exact proportional-cost target before integer rounding.
                    if held * price / nav <= weight:
                        delta = (weight * nav - held * price) / (price * (1 + weight * cost))
                        amount = min(np.floor(max(delta, 0)), np.floor(cash / (price * (1 + cost))))
                    else:
                        delta = (held * price - weight * nav) / (price * (1 - weight * cost))
                        amount = -min(np.floor(max(delta, 0)), held)
                if amount:
                    order = Order(symbol, np.float64(abs(amount)), Order.BUY if amount > 0 else Order.SELL, start, end)
                    orders.append(order)
                    submitted.append((order, None))
            return TradeDecisionWO(orders, self)

    native, indicators = backtest(
        start_time=dates[0], end_time=dates[-1], strategy=FrozenOrders(),
        executor={"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                  "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}},
        account=float(initial_cash), benchmark="SOXX", pos_type=QlibContinuousPosition,
        exchange_kwargs={"codes": ["SOXL", "SOXX"], "freq": "day", "deal_price": "open",
                         "limit_threshold": None, "trade_unit": None, "open_cost": cost,
                         "close_cost": cost, "min_cost": 0., "impact_cost": 0.},
    )
    report, positions = native["1day"]
    report = report.copy()
    report.index = pd.DatetimeIndex(report.index).normalize()
    if not report.index.equals(dates):
        raise ValueError("blocked: native Qlib report coverage differs")
    for order, reference in submitted:
        if not np.isclose(order.deal_amount, order.amount, rtol=0, atol=1e-8):
            raise ValueError("blocked: Qlib clipped a preregistered integer order")
        direction = 1 if order.direction == Order.BUY else -1
        execution_rows.append({"date": pd.Timestamp(order.start_time).normalize(), "symbol": order.stock_id,
            "amount": direction * float(order.deal_amount), "requested_amount": direction * float(order.amount),
            "signal_date": None if reference is None else reference.signal_date})
    rows = []
    previous = float(initial_cash)
    for date in dates:
        position = positions[date]
        if not isinstance(position, QlibContinuousPosition):
            raise ValueError("blocked: native Position implementation differs")
        nav = float(report.loc[date, "account"])
        cash = float(position.get_cash())
        shares = {symbol: float(position.get_stock_amount_dict().get(symbol, 0.)) for symbol in ("SOXL", "SOXX")}
        net_return = float(report.loc[date, "return"] - report.loc[date, "cost"])
        if not np.isclose(nav / previous - 1, net_return, rtol=0, atol=1e-9):
            raise ValueError("blocked: native account and net report returns disagree")
        rows.append({"date": date, "account": nav, "cash": cash, "soxl_shares": shares["SOXL"],
            "soxx_shares": shares["SOXX"], "net_return": net_return,
            "fee": float(report.loc[date, "cost"]) * previous,
            "turnover": float(report.loc[date, "turnover"]), "exposure": 1 - cash / nav})
        previous = nav
    return {"report": report, "positions": positions, "indicators": indicators,
            "daily": pd.DataFrame(rows).set_index("date"), "executed_orders": pd.DataFrame(execution_rows)}


def reconcile_engine(engine_daily, native):
    """Fail closed on any native fill/account discrepancy over the full path."""
    expected, actual = _dates(engine_daily), native["daily"]
    if not expected.index.equals(actual.index):
        raise ValueError("blocked: engine/native session coverage differs")
    errors = {}
    for source, target in (("account", "account"), ("cash", "cash"), ("shares", "soxl_shares"),
                           ("fee", "fee"), ("net_return", "net_return")):
        left, right = expected[source].to_numpy(float), actual[target].to_numpy(float)
        errors[source] = float(np.max(np.abs(left - right)))
        tolerance = 1e-7 if source == "net_return" else 1e-5
        if not np.allclose(left, right, rtol=1e-10, atol=tolerance):
            raise ValueError(f"blocked: native {source} disagrees with independent legacy ledger: {errors[source]}")
    return {"status": "passed", "sessions": len(actual), "maximum_absolute_errors": errors,
        "native_api": "qlib.backtest.backtest + SimulatorExecutor + Exchange + Account + Position",
        "position_type": "QlibContinuousPosition (positive share residual preservation only)",
        "scope": "frozen orders independently filled/accounted; rule signals tested separately",
        "mark": "first open cash-funded entry; thereafter close-to-close NAV; no terminal liquidation",
        "units": "integer qualified adjusted-price research shares, not raw broker shares"}


def path_metrics(daily):
    from qlib.contrib.evaluate import risk_analysis
    daily = _dates(daily)
    returns = daily.net_return.to_numpy(float)
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("blocked: invalid native portfolio returns")
    logs = np.log1p(returns)
    native_risk = risk_analysis(pd.Series(returns, index=daily.index), N=252, freq=None, mode="product")
    wealth = np.r_[1., np.exp(np.cumsum(logs))]
    ordered = np.sort(returns)
    mass = len(ordered) * .05
    whole = int(np.floor(mass))
    fraction = mass - whole
    cvar = (ordered[:whole].sum() + (fraction * ordered[whole] if fraction else 0)) / mass
    return {"sessions": len(daily), "return_pct": float(100 * np.expm1(logs.sum())),
        "cagr_pct": float(100 * native_risk.loc["annualized_return", "risk"]),
        "max_drawdown_pct": float(100 * (wealth / np.maximum.accumulate(wealth) - 1).min()),
        "cvar5_pct": float(100 * cvar), "worst_day_pct": float(100 * returns.min()),
        "annualized_volatility_pct": float(100 * np.std(returns, ddof=1) * np.sqrt(252)),
        "mean_exposure_pct": float(100 * daily.exposure.mean()), "fee_cash": float(daily.fee.sum()),
        "turnover": float(daily.turnover.sum()), "terminal_account": float(daily.account.iloc[-1])}


def bootstrap_excess(candidate, benchmark, *, mean_block, resamples=2000, seed=20261005):
    if not candidate.index.equals(benchmark.index) or len(candidate) < 2 * mean_block:
        raise ValueError("blocked: bootstrap requires complete common-date paths")
    values = np.log1p(candidate.net_return.to_numpy()) - np.log1p(benchmark.net_return.to_numpy())
    random = np.random.default_rng(seed)
    indices = random.integers(len(values), size=resamples)
    totals = values[indices].copy()
    for _ in range(1, len(values)):
        restart = random.random(resamples) < 1 / mean_block
        indices = (indices + 1) % len(values)
        indices[restart] = random.integers(len(values), size=int(restart.sum()))
        totals += values[indices]
    lower, median, upper = (float(x) for x in np.quantile(totals / len(values), [.025, .5, .975]))
    return {"mean_block": mean_block, "resamples": resamples, "seed": seed,
            "daily_mean_log_excess": float(values.mean()), "daily_lower_2_5pct": lower,
            "daily_median": median, "daily_upper_97_5pct": upper,
            "annualized_lower_log_excess": 252 * lower, "annualized_median_log_excess": 252 * median,
            "annualized_upper_log_excess": 252 * upper, "lower_bound_positive": lower > 0,
            "_draws": totals / len(values) * 252,
            "method": "paired stationary bootstrap percentile 2.5/50/97.5",
            "limitation": "reused historical sample; stationarity assumption; excludes past research selection"}


def analyze_paths(paths, windows, costs, *, bootstrap_resamples=2000):
    comparison, annual, bootstrap, gates, risks = [], [], [], [], []
    draws = {}
    from qlib.contrib.evaluate import risk_analysis
    for window in windows:
        for cost in costs:
            benchmark = paths[(window, cost, "soxx_buy_hold")]["daily"]
            for variant in VARIANTS:
                daily = paths[(window, cost, variant)]["daily"]
                if not daily.index.equals(benchmark.index):
                    raise ValueError("blocked: portfolio/benchmark coverage differs")
                metrics = path_metrics(daily)
                excess = np.log1p(daily.net_return) - np.log1p(benchmark.net_return)
                comparison.append({"window": window, "cost_bps": cost, "variant": variant,
                    **metrics, "log_excess_vs_soxx": float(excess.sum()),
                    "relative_return_vs_soxx_pct": float(100 * np.expm1(excess.sum()))})
                for year in sorted(daily.index.year.unique()):
                    mask = daily.index.year == year
                    annual.append({"window": window, "cost_bps": cost, "variant": variant,
                        "year": int(year), **path_metrics(daily.loc[mask]),
                        "log_excess_vs_soxx": float(excess.loc[mask].sum()),
                        "accounting": "slice of continuous portfolio; no annual reset"})
                risk = risk_analysis(daily.net_return, N=252, freq=None, mode="product").reset_index()
                risks.append(risk.assign(window=window, cost_bps=cost, variant=variant,
                    drawdown_convention="Native Qlib excludes initial capital mark; comparison and gates include it."))
            if window in {"development", "reused_diagnostic"} and cost in {10, 25}:
                candidate = paths[(window, cost, "legacy_repaired")]["daily"]
                candidate_metrics, benchmark_metrics = path_metrics(candidate), path_metrics(benchmark)
                excess = np.log1p(candidate.net_return) - np.log1p(benchmark.net_return)
                annual_excess = excess.groupby(excess.index.year).sum()
                checks = {"positive_net_log_excess": float(excess.sum()) > 0,
                    "max_drawdown_not_worse_soxx": candidate_metrics["max_drawdown_pct"] >= benchmark_metrics["max_drawdown_pct"] - 1e-10,
                    "cvar5_not_worse_soxx": candidate_metrics["cvar5_pct"] >= benchmark_metrics["cvar5_pct"] - 1e-10}
                if window == "development":
                    checks["at_least_6_of_7_positive_years"] = len(annual_excess) == 7 and int((annual_excess > 0).sum()) >= 6
                    checks["median_annual_log_excess_positive"] = float(annual_excess.median()) > 0
                for block in (20, 60):
                    row = bootstrap_excess(candidate, benchmark, mean_block=block, resamples=bootstrap_resamples)
                    draws[f"{window}/{cost}/{block}"] = row.pop("_draws")
                    bootstrap.append({"window": window, "cost_bps": cost, **row})
                    checks[f"block_{block}_positive_lower_bound"] = row["lower_bound_positive"]
                gates.append({"window": window, "cost_bps": cost, "positive_years": int((annual_excess > 0).sum()),
                              **checks, "all_pass": bool(all(checks.values()))})
    return {"backtest_benchmark_comparison.csv": pd.DataFrame(comparison), "annual_metrics.csv": pd.DataFrame(annual),
            "bootstrap.csv": pd.DataFrame(bootstrap), "development_gates.csv": pd.DataFrame(gates),
            "risk_analysis.csv": pd.concat(risks, ignore_index=True), "_bootstrap_draws": draws}



def load_legacy_inputs(root, plan):
    """Join separately qualified native price/VIX observations without filling."""
    from src.data.constrained_net_excess import _native_readback
    from src.data.legacy_vix_source import load_legacy_vix
    from src.data.legacy_execution_view import load_legacy_execution_view
    root = Path(root).resolve()
    vix, vix_audit = load_legacy_vix(root)
    raw, calendar, price_audit = _native_readback(root, plan, plan["source_audit"])
    if not calendar.equals(vix.index):
        raise ValueError("blocked: VIX and price Qlib calendars differ")
    soxl = raw.xs("SOXL", level="instrument").reindex(calendar)
    frame = pd.DataFrame({"date": calendar, "open": soxl["$open"].to_numpy(float),
        "close": soxl["$close"].to_numpy(float), "vix": vix.to_numpy(float)})
    if not np.isfinite(frame[["open", "close", "vix"]].to_numpy()).all():
        raise ValueError("blocked: incomplete native legacy input; no filling permitted")
    price_audit["provider_uri"] = plan["source"]["provider_uri"]
    execution_provider, execution_audit = load_legacy_execution_view(root)
    if execution_provider.resolve() != (root / plan["execution_provider_uri"]).resolve():
        raise ValueError("blocked: execution provider differs from plan")
    if execution_audit["native_frame_sha256"] != price_audit["native_frame_sha256"]:
        raise ValueError("blocked: execution SOXL/SOXX readback differs from canonical prices")
    return frame, {"price": price_audit, "vix": vix_audit, "execution": execution_audit,
        "decision_cutoff": "17:00 America/New_York, historical VIX availability assumed",
        "source_api": "qlib.data.D.calendar + D.instruments + D.features",
        "warmup_start": str(calendar[0].date()), "as_of": str(calendar[-1].date())}


def run_legacy_repair(*, root, card_dir, provider_uri, frame, source_audit, plan):
    """Exactly one registered attempt; a completed identical card is read back."""
    import qlib
    from qlib.workflow import R
    from src.backtest.legacy_soxl_replay import run_legacy_repaired
    root, card = Path(root).resolve(), Path(card_dir).resolve()
    card.mkdir(parents=True, exist_ok=True)
    provider = Path(provider_uri).resolve()
    if not provider.is_dir() or provider != (root / plan["execution_provider_uri"]).resolve():
        raise ValueError("blocked: native execution provider missing or differs from plan")
    hashes = _source_hashes(root)
    authorization = _authorization(root, card, plan, source_audit)
    from soxl_backtest.config.backtest_config import BacktestConfig
    if BacktestConfig().params != plan["frozen_legacy_parameters"]:
        raise ValueError("blocked: legacy defaults differ from registered parameters")
    input_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    identity = _digest({"plan": plan, "sources": hashes, "data": source_audit, "frame_sha256": input_hash, "authorization": authorization})
    if sorted(plan["costs_bps"]) != [0, 10, 25] or len(plan["costs_bps"]) != 3:
        raise ValueError("blocked: required fixed cost scenarios are 0/10/25 bp")
    attempt_path = card / "attempt.json"
    if attempt_path.exists():
        attempt = json.loads(attempt_path.read_text())
        if attempt.get("identity") != identity or attempt.get("status") != "complete":
            raise ValueError("blocked: an existing experiment attempt differs or is incomplete; no silent rerun")
        summary = json.loads((card / "summary.json").read_text())
        with R.uri_context(summary["recorder_uri"]):
            recorder = R.get_recorder(experiment_name=EXPERIMENT, recorder_id=summary["recorder_id"])
            if recorder.load_object("audit/identity.pkl") != identity:
                raise ValueError("blocked: Recorder cache identity differs")
            _verify_cached_recorder(summary, recorder)
        return summary
    _deadline(plan)
    with attempt_path.open("x") as handle:
        json.dump({"identity": identity, "status": "running"}, handle)
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    uri = str(Path(plan["recorder_uri"]).resolve())
    R.set_uri(uri)
    costs, windows = plan["costs_bps"], plan["books"]
    initial_cash = float(plan.get("initial_cash", 100000))
    paths, audits, ledgers = {}, {}, {}
    try:
        with R.start(experiment_name=EXPERIMENT):
            recorder = R.get_recorder()
            _json(attempt_path, {"identity": identity, "status": "running", "recorder_id": recorder.id, "recorder_uri": uri})
            R.log_params(plan_sha256=_digest(plan), identity=identity, rule="unchanged legacy defaults; repaired execution")
            R.save_objects(artifact_path="audit", **{"identity.pkl": identity, "plan.pkl": plan,
                           "source_audit.pkl": source_audit, "source_hashes.pkl": hashes, "input_hash.pkl": input_hash})
            R.save_objects(artifact_path="inputs", **{"market.pkl": frame})
            for window, bounds in windows.items():
                for cost in costs:
                    _deadline(plan)
                    result = run_legacy_repaired(frame, cost_bps=cost, start=bounds[0], end=bounds[1], initial_cash=initial_cash)
                    daily = _dates(result.daily)
                    native = native_order_backtest(dates=daily.index, cost_bps=cost, initial_cash=initial_cash, trades=result.trades)
                    audit = reconcile_engine(daily, native)
                    key = (window, cost, "legacy_repaired")
                    paths[key], audits[f"{window}/{cost}"], ledgers[(window, cost)] = native, audit, result
                    for variant in VARIANTS[1:]:
                        _deadline(plan)
                        paths[(window, cost, variant)] = native_order_backtest(dates=daily.index, cost_bps=cost,
                            initial_cash=initial_cash, control=variant)
                    R.save_objects(artifact_path="portfolio_analysis", **{"native_reports.pkl": paths})
                    R.save_objects(artifact_path="analysis", **{"ledgers.pkl": ledgers})
                    R.save_objects(artifact_path="audit", **{"progress.pkl": {"last_book": window, "last_cost": cost}})
                    print(f"Native legacy comparison complete: {window}, {cost} bp, {len(daily)} sessions", flush=True)
            _authorization(root, card, plan, source_audit)
            _deadline(plan)
            # Development/reused metrics carry the same full continuous holdings.
            analysis_paths = {}
            for window, spec in plan["windows"].items():
                for cost in costs:
                    for variant in VARIANTS:
                        native = paths[(spec["book"], cost, variant)]
                        daily = native["daily"].loc[spec["start"]:spec["end"]]
                        if daily.empty:
                            raise ValueError("blocked: analysis window has no native sessions")
                        analysis_paths[(window, cost, variant)] = {"daily": daily}
            tables = analyze_paths(analysis_paths, plan["windows"], costs,
                                   bootstrap_resamples=plan.get("bootstrap_resamples", 2000))
            bootstrap_draws = tables.pop("_bootstrap_draws")
            R.save_objects(artifact_path="portfolio_analysis", **{"native_reports.pkl": paths})
            R.save_objects(artifact_path="analysis", **{"ledgers.pkl": ledgers,
                "orders.pkl": {key: value.trades for key, value in ledgers.items()}, "bootstrap_draws.pkl": bootstrap_draws})
            R.save_objects(artifact_path="analysis", **{"tables.pkl": tables, "reconciliation_audits.pkl": audits})
            if recorder.load_object("audit/identity.pkl") != identity:
                raise ValueError("blocked: native Recorder readback failed")
            passed = bool(tables["development_gates.csv"].loc[lambda x: (x.window == "development") & x.cost_bps.isin([10, 25]), "all_pass"].all())
            R.log_metrics(historical_stable_excess_pass=int(passed), native_reconciliation_pass=1)
            recdir = Path(recorder.get_local_dir()).resolve()
        for filename, table in tables.items():
            table.to_csv(card / filename, index=False)
        comparison = tables["backtest_benchmark_comparison.csv"]
        baseline = comparison.loc[(comparison.window == "development") & (comparison.cost_bps == 10)]
        candidate = baseline.loc[baseline.variant == "legacy_repaired"].iloc[0]
        benchmark = baseline.loc[baseline.variant == "soxx_buy_hold"].iloc[0]
        conclusion = (f"# 旧 SOXL 规则修复与原生 Qlib 复核\n\n"
            f"固定旧参数，仅修复因果时序、成交/费用与状态记账；先前收盘信息生成指令，下一交易日开盘成交。"
            f"2016 年至样本末为同一连续财富路径，开发期与 2023 年后指标仅作切片；另设 2026-08-03 现金启动诊断。年度指标不重置组合。0/10/25 bp 每侧成本均独立执行。\n\n"
            f"2016–2022 开发期 10 bp：规则累计收益 {candidate.return_pct:.2f}%，SOXX {benchmark.return_pct:.2f}%；"
            f"规则年化 {candidate.cagr_pct:.2f}%，最大回撤 {candidate.max_drawdown_pct:.2f}%，CVaR5 {candidate.cvar5_pct:.2f}%。"
            f"稳定超额门禁{'通过' if passed else '未通过'}，详见 development_gates.csv、年度与 bootstrap 表。\n\n"
            f"2023 年以来和 2026-08-03 以来仅为已复用历史诊断，不是新样本；50%/90% SOXL 为固定配置参照，并非精确风险匹配；它们帮助判断降低暴露的贡献。"
            f"每个窗口与成本的订单、现金、持仓、费用、净收益均经 Qlib 原生 Exchange/Position/report 全路径对账。\n\n"
            f"结论：不晋升，不修改生产。历史价格含回溯公司行动调整，不是当时可获得价格档案；VIX 17:00 可用性为假设。历史开盘价不证明真实成交、价差、冲击或部分成交；整数复权价格单位不等于实际券商股数。"
            f"下一步应以新连续样本验证，不能根据本次结果修改旧参数后重复宣称验证成功。\n\n"
            f"Recorder：`{recorder.id}`；完整账本、订单、报告和持仓仅在 Recorder 中。\n")
        (card / "conclusion.md").write_text(conclusion)
        summary = {"experiment": EXPERIMENT, "identity": identity, "provider_uri": plan["execution_provider_uri"],
            "canonical_provider_uri": plan["source"]["provider_uri"],
            "market": "all", "region": "us", "as_of": plan["data_as_of"], "source_audit": source_audit,
            "source_sha256": hashes, "plan_sha256": _digest(plan), "windows": plan["windows"], "books": windows, "costs_bps": costs,
            "recorder_id": recorder.id, "recorder_uri": uri, "recorder_local_dir": str(recdir),
            "native_qlib_audit": audits, "qlib_docs_reference": DOCS,
            "benchmark": "native same-cost SOXX buy-and-hold; SOXL buy-hold and fixed daily 50%/90% SOXL controls",
            "model_selection": {"performed": False, "rule_parameters": "legacy defaults; no parameter search"},
            "historical_stable_excess_pass": passed,
            "reused_diagnostic_tests_pass": bool(tables["development_gates.csv"].loc[lambda x: x.window == "reused_diagnostic", "all_pass"].all()),
            "promotion_decision": "not_promoted",
            "new_forward_evidence": False, "production_change": False, "real_broker_orders": False, "objective_achieved": False,
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": ["inputs/market.pkl", "portfolio_analysis/native_reports.pkl",
                    "analysis/ledgers.pkl", "analysis/orders.pkl", "analysis/bootstrap_draws.pkl",
                    "analysis/tables.pkl", "analysis/reconciliation_audits.pkl"]},
            "sidecars": list(tables) + ["plan.json", "attempt.json", "conclusion.md"],
            "limitations": list(plan["limitations"]) + ["Previously reviewed history is not a fresh holdout.",
                "Daily open fills omit spread, impact and partial-fill evidence.",
                "Integer adjusted-price research units are not broker raw share quantities.",
                "Fixed 50/90 allocation controls are not exact risk-matched portfolios; weights fixed before open, integer sizing uses observed open.",
                "VIX 17:00 New York availability is assumed; historical publication vintages are not verified.",
                "Qlib independently accounts frozen orders; rule correctness has separate causal unit tests.",
                "Cash earns zero and no terminal liquidation fee is charged for any strategy."]}
        _authorization(root, card, plan, source_audit)
        final_frame, final_audit = load_legacy_inputs(root, plan)
        final_hash = hashlib.sha256(pd.util.hash_pandas_object(final_frame, index=True).values.tobytes()).hexdigest()
        if final_hash != input_hash or final_audit != source_audit:
            raise ValueError("blocked: native price/VIX evidence changed during execution")
        _deadline(plan)
        artifacts = recdir / "artifacts"
        manifest = {str(path.relative_to(artifacts)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(artifacts.rglob("*")) if path.is_file()}
        summary["recorder_artifact_sha256"] = manifest
        summary["run_authorization_sha256"] = hashlib.sha256((card / "run_authorization.json").read_bytes()).hexdigest()
        # Bind final metadata directly to the one completed Recorder. No new
        # workflow context, resume, or second experiment identity is created.
        recorder.save_objects(artifact_path="audit", **{"summary.pkl": summary, "artifact_manifest.pkl": manifest})
        _authorization(root, card, plan, source_audit)
        _deadline(plan)
        _json(card / "summary.json", summary)
        _json(attempt_path, {"identity": identity, "status": "complete", "recorder_id": recorder.id})
        return summary
    except Exception as exc:
        failure = json.loads(attempt_path.read_text())
        failure.update({"status": "failed", "error": str(exc)})
        _json(attempt_path, failure)
        raise
