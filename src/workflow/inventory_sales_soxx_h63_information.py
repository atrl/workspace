"""One frozen SOXX H63 information screen for original inventory-sales mismatch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.core.recorder_sources import load_verified_recorder_objects
from src.workflow.inventory_sales_source_qualification import normalized_notebook

NAME = "inventory_sales_soxx_h63_information_20261022"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stationary_bootstrap_lower(values, *, mean_block, draws, seed, quantile):
    values = np.asarray(values, dtype=float)
    require(len(values) > 0 and np.isfinite(values).all(), "paired losses invalid")
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(draws, dtype=float)
    restart = 1.0 / mean_block
    for draw in range(draws):
        index = int(rng.integers(n))
        sample = np.empty(n, dtype=float)
        for pos in range(n):
            if pos and rng.random() < restart:
                index = int(rng.integers(n))
            elif pos:
                index = (index + 1) % n
            sample[pos] = values[index]
        means[draw] = sample.mean()
    return float(np.quantile(means, quantile))


def monthly_signal(support: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = pairs.set_index(["ticker", "accession"], verify_integrity=True)
    eligible_rows = []
    excluded = []
    for date, group in support.groupby("datetime", sort=True):
        date = pd.Timestamp(date)
        if (group.source_supported_weight_pct.le(0)).any():
            excluded.append({"datetime": date, "reason": "active_disclosed_member_missing_original_pair",
                             "tickers": sorted(group.loc[group.source_supported_weight_pct.le(0), "ticker"].tolist())})
            continue
        signal = 0.0
        weight = 0.0
        for row in group.itertuples(index=False):
            lookup = (row.ticker, row.original_pair_accession)
            require(lookup in key.index, "source support pair identity missing")
            pair = key.loc[lookup]
            require(pd.Timestamp(pair.conditional_available_session) <= date,
                    "original filing pair used before source availability")
            amounts = np.array([pair.inventory, pair.prior_inventory, pair.revenue, pair.prior_revenue], dtype=float)
            require(np.isfinite(amounts).all() and (amounts > 0).all(), "original annual amounts invalid")
            term = np.log(amounts[0] / amounts[1]) - np.log(amounts[2] / amounts[3])
            raw_weight = float(row.weight_pct) / 100.0
            require(raw_weight >= 0 and np.isfinite(raw_weight), "disclosed basket weight invalid")
            signal += raw_weight * term
            weight += raw_weight
        require(np.isfinite(signal) and weight > 0, "partial basket signal invalid")
        eligible_rows.append({"datetime": date, "instrument": "SOXX_PARTIAL_BASKET",
                              "inventory_sales_mismatch": signal, "raw_four_name_weight": weight,
                              "active_issuer_rows": len(group)})
    columns = ["datetime", "instrument", "inventory_sales_mismatch", "raw_four_name_weight",
               "active_issuer_rows"]
    return pd.DataFrame(eligible_rows, columns=columns).sort_values("datetime"), pd.DataFrame(excluded)


def label_and_controls(monthly: pd.DataFrame, calendar: pd.DatetimeIndex, opens: pd.Series,
                       horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    require(opens.index.equals(calendar) and np.isfinite(opens.to_numpy()).all()
            and (opens.to_numpy() > 0).all(), "complete positive native SOXX open plane required")
    log_open = np.log(opens.astype(float))
    daily = log_open.diff()
    rows = []
    excluded = []
    for row in monthly.itertuples(index=False):
        date = pd.Timestamp(row.datetime)
        if date not in calendar:
            excluded.append({"datetime": date, "reason": "decision_not_in_price_calendar"})
            continue
        t = calendar.get_loc(date)
        if not isinstance(t, (int, np.integer)) or t < horizon or t + horizon + 1 >= len(calendar):
            excluded.append({"datetime": date, "reason": "trailing_or_future_open_window_missing"})
            continue
        trail = float(log_open.iloc[t] - log_open.iloc[t - horizon])
        volatility = float(daily.iloc[t - 19:t + 1].std(ddof=1))
        target = float(log_open.iloc[t + horizon + 1] - log_open.iloc[t + 1])
        if not np.isfinite([trail, volatility, target]).all():
            excluded.append({"datetime": date, "reason": "nonfinite_native_price_or_label"})
            continue
        rows.append({"datetime": date, "instrument": row.instrument,
                     "inventory_sales_mismatch": row.inventory_sales_mismatch,
                     "raw_four_name_weight": row.raw_four_name_weight,
                     "trailing_h63_open_return": trail, "trailing_open_vol20": volatility,
                     "SOXX_future_h63_open_return": target,
                     "entry_session": calendar[t + 1], "label_maturity_session": calendar[t + horizon + 1],
                     "active_issuer_rows": row.active_issuer_rows})
    columns = ["datetime", "instrument", "inventory_sales_mismatch", "raw_four_name_weight",
               "trailing_h63_open_return", "trailing_open_vol20", "SOXX_future_h63_open_return",
               "entry_session", "label_maturity_session", "active_issuer_rows"]
    return pd.DataFrame(rows, columns=columns).sort_values("datetime"), pd.DataFrame(excluded)


def run(root):
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R

    root = Path(root).resolve()
    card = root / "research" / NAME
    plan_path = card / "plan.json"
    plan = json.loads(plan_path.read_text())
    require(plan["frozen"] and plan["execution_authorized"], "frozen information plan absent")
    require(not (card / "run_state.json").exists(), "fixed information attempt already started")
    require(sha(Path(__file__)) == plan["implementation"]["module_sha256"], "module bytes differ")
    notebook = card / (NAME + ".ipynb")
    require(hashlib.sha256(normalized_notebook(notebook)).hexdigest()
            == plan["implementation"]["notebook_sha256"], "notebook code differs")
    source = plan["source_recorder"]
    source_card = root / source["card"]
    require(sha(source_card / "summary.json") == source["summary_sha256"]
            and sha(source_card / "verification_manifest.json") == source["verification_manifest_sha256"],
            "source Recorder metadata changed")
    price_spec = plan["price_source"]
    correction = plan["pre_market_source_correction"]
    require(sha(card / correction["prior_plan"]) == correction["prior_plan_sha256"]
            and correction["market_value_reads_before_change"] == 0
            and correction["fit_attempts_before_change"] == 0,
            "pre-market split-source correction identity differs")
    qualification = price_spec["qualification_summary"]
    require(sha(root / qualification["path"]) == qualification["sha256"],
            "split-adjusted SOXX qualification summary changed")
    for item in price_spec["files"].values():
        path = root / item["path"]
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                "Qlib SOXX price source changed")
    (card / "run_state.json").write_text(json.dumps({"status": "started", "plan_sha256": sha(plan_path)}) + "\n")
    qlib.init(provider_uri=str(root / price_spec["provider_uri"]), region="us", kernels=1,
              expression_cache=None, dataset_cache=None)
    objects, audit = load_verified_recorder_objects(
        source_card, expected_recorder_id=source["recorder_id"], artifacts=list(source["artifacts"]),
        expected_artifact_sha256=source["artifacts"],
        expected_manifest_sha256=source["verification_manifest_sha256"])
    pairs = objects["analysis/bridged_comparable_pairs"]
    support = objects["analysis/four_name_monthly_support"]
    require(len(pairs) == source["expected_pairs"] and len(support) == source["expected_issuer_months"]
            and support.source_supported_weight_pct.gt(0).sum() == source["expected_supported_issuer_months"],
            "original source panel counts differ")
    calendar = pd.DatetimeIndex(D.calendar(start_time=price_spec["bounds"][0],
                                           end_time=price_spec["bounds"][1], freq="day"))
    require(len(calendar) > 0 and sha(root / price_spec["files"]["calendar"]["path"])
            == price_spec["files"]["calendar"]["sha256"], "native SOXX calendar identity differs")
    native = D.features([price_spec["instrument"]], [price_spec["field"]],
                        start_time=calendar[0], end_time=calendar[-1], freq="day")
    require(price_spec["field"] in native and native.index.names == ["instrument", "datetime"],
            "native SOXX open feature schema differs")
    opens = native[price_spec["field"]].unstack("instrument").reindex(calendar)[price_spec["instrument"]]
    partial, source_exclusions = monthly_signal(support, pairs)
    frame, price_exclusions = label_and_controls(partial, calendar, opens, plan["target"]["horizon_sessions"])
    train_bounds = plan["target"]["training_decision_bounds"]
    held_bounds = plan["target"]["held_decision_bounds"]
    held_start = calendar[calendar.searchsorted(pd.Timestamp(held_bounds[0]))]
    train = frame.loc[frame.datetime.between(*train_bounds)
                      & frame.label_maturity_session.lt(held_start)].copy()
    held = frame.loc[frame.datetime.between(*held_bounds)].copy()
    partial_2026 = frame.loc[frame.datetime.dt.year.eq(2026)].copy()
    gate = plan["information_gate"]
    require(len(train) >= gate["minimum_train_months"] and len(held) >= gate["minimum_held_months"],
            "insufficient mature fixed train or held monthly rows")
    require(train.label_maturity_session.max() < held.datetime.min(), "training label crosses held boundary")
    baseline_columns = ["trailing_h63_open_return", "trailing_open_vol20", "raw_four_name_weight"]
    candidate_columns = baseline_columns + ["inventory_sales_mismatch"]
    feature_cols = candidate_columns
    indexed = frame.set_index(["datetime", "instrument"])
    native_panel = pd.concat({"feature": indexed[feature_cols],
                              "label": indexed[["SOXX_future_h63_open_return"]]}, axis=1)
    dataset = DatasetH(handler=DataHandlerLP.from_df(native_panel), segments={"all": (None, None)})
    dataset.config(dump_all=True, recursive=True)
    prepared = dataset.prepare("all", col_set=["feature", "label"])
    pd.testing.assert_frame_equal(prepared, native_panel)
    train_index = pd.MultiIndex.from_frame(train[["datetime", "instrument"]])
    held_index = pd.MultiIndex.from_frame(held[["datetime", "instrument"]])
    train_target = prepared["label"].loc[train_index, "SOXX_future_h63_open_return"].to_numpy(dtype=float)
    predictions = {}
    models = {}
    for name, columns in (("baseline", baseline_columns), ("candidate", candidate_columns)):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(prepared["feature"].loc[train_index, columns].to_numpy(dtype=float))
        x_held = scaler.transform(prepared["feature"].loc[held_index, columns].to_numpy(dtype=float))
        require(np.isfinite(x_train).all() and np.isfinite(x_held).all()
                and (scaler.var_ > 0).all(), "fixed train scaler invalid")
        model = Ridge(alpha=float(plan["model"]["ridge_alpha"]), fit_intercept=True)
        model.fit(x_train, train_target)
        predictions[name] = model.predict(x_held)
        models[name] = {"scaler": scaler, "model": model, "features": columns}
    coefficient = float(models["candidate"]["model"].coef_[-1])
    held = held[["datetime", "instrument", "SOXX_future_h63_open_return", "entry_session",
                 "label_maturity_session", "inventory_sales_mismatch", "raw_four_name_weight"]].copy()
    held["baseline_prediction"] = predictions["baseline"]
    held["candidate_prediction"] = predictions["candidate"]
    held["baseline_squared_error"] = (held.SOXX_future_h63_open_return - held.baseline_prediction) ** 2
    held["candidate_squared_error"] = (held.SOXX_future_h63_open_return - held.candidate_prediction) ** 2
    held["paired_improvement"] = held.baseline_squared_error - held.candidate_squared_error
    mean_improvement = float(held.paired_improvement.mean())
    lower = stationary_bootstrap_lower(held.paired_improvement.to_numpy(),
                                       mean_block=gate["bootstrap_mean_block_observations"],
                                       draws=gate["bootstrap_draws"], seed=gate["bootstrap_seed"],
                                       quantile=gate["bootstrap_lower_quantile"])
    passed = bool(coefficient < 0 and mean_improvement > 0 and lower > 0)
    yearly = {str(year): {"months": len(group), "mean_paired_MSE_improvement": float(group.paired_improvement.mean())}
              for year, group in held.groupby(held.datetime.dt.year)}
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=NAME, uri=str(tracking)):
        rec = R.get_recorder()
        rec.save_objects(artifact_path="inputs", plan=plan, source_audit=audit,
                         price_source_identity=price_spec["files"])
        rec.save_objects(artifact_path="analysis", source_monthly_signal=partial,
                         source_exclusions=source_exclusions, price_exclusions=price_exclusions,
                         fixed_train=train, fixed_held=held, partial_2026_diagnostic=partial_2026,
                         baseline_model=models["baseline"], candidate_model=models["candidate"])
        rec.save_objects(artifact_path="metadata", information_dataset=dataset)
        rec.save_objects(artifact_path="sources", module=Path(__file__).read_bytes(), notebook=notebook.read_bytes())
        pd.testing.assert_frame_equal(rec.load_object("metadata/information_dataset").prepare("all", col_set=["feature", "label"]), native_panel)
        rec.log_metrics(train_months=len(train), held_months=len(held), candidate_signal_coefficient=coefficient,
                        held_mean_paired_MSE_improvement=mean_improvement, bootstrap_lower=lower,
                        information_gate_pass=int(passed), new_fits=2, native_accounts=0)
        summary = {"experiment": NAME, "status": "completed_fixed_information_screen",
                   "decision": "information_gate_passed_development_only" if passed else "not_promoted_fixed_information_gate_failed",
                   "hypothesis_sign": "negative", "train_months": len(train), "held_months": len(held),
                   "held_years": yearly, "partial_2026_diagnostic_months": len(partial_2026),
                   "source_eligible_months": len(partial), "source_excluded_months": len(source_exclusions),
                   "price_or_label_excluded_months": len(price_exclusions),
                   "candidate_signal_coefficient_standardized": coefficient,
                   "baseline_held_MSE": float(held.baseline_squared_error.mean()),
                   "candidate_held_MSE": float(held.candidate_squared_error.mean()),
                   "held_mean_paired_MSE_improvement": mean_improvement,
                   "stationary_bootstrap_lower_alpha_0p025_div_15": lower,
                   "fixed_information_gate_pass": passed,
                   "source_history_independent": False, "forward_qualified_days": 0,
                   "fee_net_SOXX_excess_established": False, "trading_or_promotion_permission": False,
                   "D_calendar_calls": 1, "D_features_calls": 1, "model_fits": 2,
                   "bootstrap_runs": 1, "native_accounts": 0,
                   "source_recorder_object_loads": len(source["artifacts"]),
                   "source_recorder_id": source["recorder_id"], "plan_sha256": sha(plan_path),
                   "module_sha256": sha(Path(__file__)), "recorder_id": rec.id,
                   "experiment_id": str(rec.experiment_id), "recorder_uri": str(tracking),
                   "recorder_local_dir": str(tracking / str(rec.experiment_id) / rec.id),
                   "artifact_policy": plan["artifact_policy"]}
        rec.save_objects(artifact_path="analysis", summary=summary)
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    manifest = {"recorder_id": summary["recorder_id"], "entries": [
        {"path": str(path.relative_to(card)), "sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(artifacts.rglob("*")) if path.is_file()]}
    (card / "verification_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (card / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (card / "run_state.json").write_text(json.dumps({"status": "completed", "recorder_id": rec.id,
                                                     "plan_sha256": summary["plan_sha256"]}) + "\n")
    (card / "conclusion.md").write_text(
        "# 原始库存/销售相对 SOXX H63：固定信息检验\n\n"
        f"Qlib Recorder：`{rec.id}`。训练月 {len(train)}、固定评价月 {len(held)}；"
        f"新增原始篮子信号的训练标准化系数 {coefficient:.6g}，评价期配对 MSE 改善 "
        f"{mean_improvement:.8g}，预登记区块下界 {lower:.8g}。"
        f"固定信息门：{'通过，仅开发候选' if passed else '失败，不晋升'}。\n\n"
        "原论文是个股/制造业证据，四只披露证券原始权重未重新归一化，"
        "目标只是 SOXX 价格开盘到开盘 63 会话收益。重叠月度标签、已复用历史、"
        "缺分红/费用及独立前向证据限制解释。没有账户、风控或费用后 SOXX 财富门，"
        "不能声称策略已稳定跑赢 SOXX。\n")
    return summary
