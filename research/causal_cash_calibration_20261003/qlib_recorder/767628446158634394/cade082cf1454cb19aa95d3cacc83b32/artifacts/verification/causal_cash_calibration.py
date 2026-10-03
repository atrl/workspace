"""One registered, prior-only affine calibration of frozen Qlib forecasts.

No market head is refitted and no portfolio is constructed. A finished Recorder
is read back on rerun; a partial run blocks automatic repetition.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.frozen_cash_information import (
    _checked_rows, bootstrap_window, event_metrics, point_metrics,
)
from src.data.integrated_allocation import build_native_dataset, frame_sha256
from src.workflow.frozen_cash_information import (
    CANONICAL_ROOT, _artifact_identity, _native_flat, _no_active_recorder, _require, _sha,
    _source_hashes as frozen_source_hashes, _times, _write_json,
    load_frozen_information_inputs,
)

EXPERIMENT = "causal_cash_calibration_20261003"
COMPARATORS = {"raw": "cash_advantage_prediction", "prior_mean": "prior_mean_prediction"}
POOLED = ("development_2016_2022", "reused_2023_asof")


def deadline_check(plan, *, now=None):
    now = datetime.now(timezone.utc) if now is None else now
    cutoff = datetime.fromisoformat(plan["deadline_utc"].replace("Z", "+00:00"))
    _require(now.tzinfo is not None and now < cutoff, "one-hour research deadline reached; abandon route")


def read_plan(card):
    card = Path(card)
    plan = json.loads((card / "plan.json").read_text())
    registration = json.loads((card / "registration.json").read_text())
    identity = {"plan_sha256": _sha(card / "plan.json"),
                "registration_sha256": _sha(card / "registration.json")}
    _require(plan["experiment"] == EXPERIMENT and registration["plan_sha256"] == identity["plan_sha256"],
             "registered calibration plan identity differs")
    _require(plan["registered_before_calibrated_predictions"] is True
             and registration["before_calibrated_prediction_computation"] is True,
             "calibration was not registered before outcomes")
    _require(plan["fit"]["years"] == list(range(2016, 2027))
             and plan["fit"]["earliest_oof_signal"] == "2013-01-01"
             and plan["fit"]["minimum_rows"] == 252,
             "fixed annual calibration protocol differs")
    _require(plan["uncertainty"]["blocks"] == [20, 60]
             and plan["uncertainty"]["resamples"] == 2000
             and plan["uncertainty"]["seed"] == 20261003
             and plan["fees"]["cost_bps"] == [10, 25], "fixed uncertainty or costs differ")
    return plan, identity


def source_hashes(root):
    result = frozen_source_hashes(root)
    for name in ("src/workflow/causal_cash_calibration.py", "src/objective/relative_wealth.py"):
        result[name] = _sha(Path(root) / name)
    return result


def load_inputs(root, plan):
    """Native readback plus exact parent artifact parity; no fitted calibration."""
    from qlib.workflow import R
    deadline_check(plan)
    _no_active_recorder()
    root = Path(root)
    parent = plan["parent"]
    parent_card = root / "research" / parent["experiment"]
    manifest_path = parent_card / "artifact_manifest.json"
    _require(_sha(manifest_path) == parent["artifact_manifest_sha256"], "parent manifest changed")
    manifest = json.loads(manifest_path.read_text())
    artifact = "inputs/diagnostic_rows.pkl"
    suffix = f"/{parent['recorder_id']}/artifacts/{artifact}"
    matches = [entry for entry in manifest["entries"] if entry["path"].endswith(suffix)]
    _require(len(matches) == 1, "parent diagnostic artifact binding missing")
    with R.uri_context(parent["recorder_uri"]):
        recorder = R.get_recorder(experiment_name=parent["experiment"], recorder_id=parent["recorder_id"])
        _require(recorder.client.get_run(recorder.id).info.status == "FINISHED", "parent Recorder not FINISHED")
        binding = _artifact_identity(recorder, artifact)
        _require(binding["sha256"] == matches[0]["sha256"], "parent diagnostic artifact hash differs")
        saved = recorder.load_object(artifact)
        _require(_artifact_identity(recorder, artifact) == binding, "parent changed during read")
    bundle = load_frozen_information_inputs(root, plan["source_plan"])
    pd.testing.assert_frame_equal(bundle["diagnostic_rows"], saved, check_exact=True, check_freq=False)
    _checked_rows(bundle["diagnostic_rows"], plan["source_plan"])
    bundle["calibration_input_audit"] = {
        "parent": parent, "parent_artifact": binding,
        "diagnostic_frame_sha256": frame_sha256(saved),
        "native_parent_exact_parity": True, "predictive_metrics_computed": False,
        "source_audit": bundle["information_source_audit"],
    }
    deadline_check(plan)
    return bundle


def calibration_mask(rows, year, plan):
    cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    mask = rows.forecast_available & rows.score_available
    mask &= (rows.index >= pd.Timestamp(plan["fit"]["earliest_oof_signal"])) & (rows.index.year < year)
    for name in ("feature_available_at", "forecast_model_available_at", "return_available_at"):
        mask &= _times(rows[name], name, missing=True).lt(cutoff)
    return mask, cutoff


def native_calibration_rows(rows, plan):
    """The calibration's own native DatasetH view, with exact frozen parity."""
    rows = _checked_rows(rows, plan["source_plan"])
    features = ["cash_advantage_prediction", "prior_mean_prediction"]
    labels = ["cash_advantage_1d"]
    metadata = [name for name in rows.columns if name not in features + labels]
    dataset = build_native_dataset(rows, feature_columns=features, label_columns=labels,
        metadata_columns=metadata, as_of=plan["source"]["as_of"])
    prepared = _native_flat(dataset).loc[:, rows.columns]
    pd.testing.assert_frame_equal(prepared, rows, check_exact=True, check_freq=False)
    return dataset, prepared


def fit_calibrations(rows, plan):
    deadline_check(plan)
    _, rows = native_calibration_rows(rows, plan)
    result = rows.copy()
    result["calibrated_prediction"] = np.nan
    fits, audit = {}, []
    for year in plan["fit"]["years"]:
        deadline_check(plan)
        eligible, cutoff = calibration_mask(rows, year, plan)
        train = rows.loc[eligible]
        _require(len(train) >= plan["fit"]["minimum_rows"], f"insufficient prior OOF rows for {year}")
        p = train.cash_advantage_prediction.to_numpy(float)
        y = train.cash_advantage_1d.to_numpy(float)
        _require(np.isfinite(p).all() and np.isfinite(y).all(), "invalid calibration rows")
        X = np.column_stack([np.ones(len(p)), p])
        coefficients, _, rank, singular = np.linalg.lstsq(X, y, rcond=None)
        _require(rank == 2 and np.isfinite(coefficients).all(), "rank-deficient calibration")
        use = (rows.index.year == year) & rows.forecast_available
        result.loc[use, "calibrated_prediction"] = coefficients[0] + coefficients[1] * rows.loc[use, "cash_advantage_prediction"]
        fits[year] = {"intercept": float(coefficients[0]), "slope": float(coefficients[1]),
                      "training_dates": train.index.copy(), "singular_values": singular,
                      "cutoff": cutoff.isoformat(), "rank": int(rank)}
        item = {"model_year": year, "training_rows": len(train),
                "intercept": float(coefficients[0]), "slope": float(coefficients[1]),
                "cutoff": cutoff.isoformat(), "first_training_signal": str(train.index.min().date()),
                "last_training_signal": str(train.index.max().date()),
                "training_dates_sha256": hashlib.sha256(train.index.asi8.tobytes()).hexdigest(),
                "training_frame_sha256": frame_sha256(train), "design_rank": int(rank),
                "predicted_rows": int(use.sum())}
        for name in ("feature_available_at", "forecast_model_available_at", "return_available_at"):
            item[name + "_max"] = _times(train[name], name).max().isoformat()
        audit.append(item)
    scored = rows.score_available & rows.index.year.isin(plan["fit"]["years"])
    _require(np.isfinite(result.loc[scored, "calibrated_prediction"]).all(), "scored rows lost during calibration")
    return result, fits, pd.DataFrame(audit)


def window_definitions(rows, plan):
    dev = plan["windows"]["development_signal_years"]
    reused = plan["windows"]["reused_diagnostic_signal_years"]
    for year in [*dev, *reused]:
        yield f"signal_{year}", "annual", year, rows.index.year == year
    yield POOLED[0], "pooled", None, rows.index.year.isin(dev)
    yield POOLED[1], "pooled", None, rows.index.year.isin(reused)
    user = plan["windows"]["user_window"]
    yield "user_trade_2026_08_03", "user_window", None, (
        rows.trade_date.ge(user["trade_start"]) & rows.exit_date.le(user["exit_end"]))


def analyze(rows, plan):
    deadline_check(plan)
    points, events, bootstrap, raw = [], [], [], {}
    for name, kind, year, mask in window_definitions(rows, plan):
        selected = rows.loc[np.asarray(mask) & rows.score_available.to_numpy()]
        _require(len(selected) > 0, f"empty registered window {name}")
        y = selected.cash_advantage_1d.to_numpy(float)
        p = selected.calibrated_prediction.to_numpy(float)
        years = selected.index.year.to_numpy()
        identity = {"window": name, "window_type": kind, "signal_year": year}
        if kind == "user_window":
            _require(len(selected) == 38, "registered user open window must contain 38 mature observations")
        for comparator, column in COMPARATORS.items():
            b = selected[column].to_numpy(float)
            points.append({**identity, "comparator": comparator, **point_metrics(y, p, b, years),
                           "source_window_rows": int(np.sum(mask)),
                           "unscored_window_rows": int(np.sum(mask)) - len(selected),
                           "first_signal": str(selected.index.min().date()),
                           "last_signal": str(selected.index.max().date())})
        for role, column in {"calibrated": "calibrated_prediction", **COMPARATORS}.items():
            for cost in plan["fees"]["cost_bps"]:
                events.append({**identity, "forecast_role": role, "rows": len(y),
                               **event_metrics(y, selected[column], cost_bps=cost)})
        if kind == "pooled":
            for block in plan["uncertainty"]["blocks"]:
                paired_hash = None
                for comparator, column in COMPARATORS.items():
                    deadline_check(plan)
                    result = bootstrap_window(y, p, selected[column].to_numpy(float), years,
                        mean_block=block, resamples=plan["uncertainty"]["resamples"],
                        seed=plan["uncertainty"]["seed"])
                    if paired_hash is not None:
                        _require(result["indices_sha256"] == paired_hash, "comparators bootstrap rows differ")
                    paired_hash = result["indices_sha256"]
                    summary = result.pop("summary")
                    summary.insert(0, "comparator", comparator)
                    summary.insert(0, "mean_block", block)
                    summary.insert(0, "window", name)
                    bootstrap.append(summary)
                    raw[(name, block, comparator)] = result
    tables = {"point_metrics": pd.DataFrame(points), "event_metrics": pd.DataFrame(events),
              "bootstrap_summary": pd.concat(bootstrap, ignore_index=True)}
    tables["screening_gates"] = screen(tables, plan)
    return {"tables": tables, "bootstrap_draws": raw}


def screen(tables, plan):
    points, events, boot = (tables[name] for name in ("point_metrics", "event_metrics", "bootstrap_summary"))
    gates = []
    def add(name, value, required, passed, **extra):
        gates.append({"gate": name, "observed": value, "required": required, "passed": bool(passed), **extra})
    for comparator in COMPARATORS:
        annual = points.loc[points.signal_year.isin(plan["windows"]["development_signal_years"]) & points.comparator.eq(comparator)]
        count = int(annual.paired_MSE_improvement_baseline_minus_prediction.gt(0).sum())
        add("annual_MSE", count, ">=6 of 7", len(annual) == 7 and count >= 6, comparator=comparator)
        for window in POOLED:
            selected = points.loc[points.window.eq(window) & points.comparator.eq(comparator)]
            _require(len(selected) == 1, "pooled comparison inventory differs")
            value = float(selected.iloc[0].paired_MSE_improvement_baseline_minus_prediction)
            add("pooled_MSE", value, ">0", value > 0, window=window, comparator=comparator)
            for block in plan["uncertainty"]["blocks"]:
                selected = boot.loc[boot.window.eq(window) & boot.mean_block.eq(block)
                    & boot.comparator.eq(comparator) & boot.statistic.eq("paired_MSE_improvement")]
                _require(len(selected) == 1, "MSE interval inventory differs")
                row = selected.iloc[0]
                add("pooled_MSE_lower95", row.lower_2_5, ">0, all draws defined",
                    row.official_interval_defined and pd.notna(row.lower_2_5) and row.lower_2_5 > 0,
                    window=window, comparator=comparator, mean_block=block)
    for window in POOLED:
        for cost in plan["fees"]["cost_bps"]:
            selected = events.loc[events.window.eq(window) & events.forecast_role.eq("calibrated") & events.cost_bps.eq(cost)]
            _require(len(selected) == 1, "event inventory differs")
            value = selected.iloc[0].mean_event_net_log_bp
            add("event_mean_net_bp", value, ">0", pd.notna(value) and value > 0, window=window, cost_bps=cost)
            for block in plan["uncertainty"]["blocks"]:
                # Same calibrated event series and identical draws under both comparators.
                selected = boot.loc[boot.window.eq(window) & boot.mean_block.eq(block)
                    & boot.comparator.eq("raw") & boot.statistic.eq(f"conditional_event_net_mean_bp_{cost}")]
                _require(len(selected) == 1, "event interval inventory differs")
                row = selected.iloc[0]
                add("event_mean_net_bp_lower95", row.lower_2_5, ">0, all draws defined",
                    row.official_interval_defined and pd.notna(row.lower_2_5) and row.lower_2_5 > 0,
                    window=window, cost_bps=cost, mean_block=block)
    return pd.DataFrame(gates)


def _cached(card, identity, hashes):
    from qlib.workflow import R
    path = card / "recorder_reference.json"
    if not path.exists():
        return None
    reference = json.loads(path.read_text())
    _require(reference.get("complete") is True, "partial calibration blocks automatic repeat")
    for key, value in {**identity, "source_sha256": hashes}.items():
        _require(reference.get(key) == value, f"cached calibration {key} differs")
    uri = str(CANONICAL_ROOT / "research" / EXPERIMENT / "qlib_recorder")
    _require(reference["recorder_uri"] == uri, "cached calibration URI differs")
    with R.uri_context(uri):
        recorder = R.get_recorder(experiment_name=EXPERIMENT, recorder_id=reference["recorder_id"])
        _require(recorder.client.get_run(recorder.id).info.status == "FINISHED", "cached calibration not FINISHED")
        for name, binding in reference["artifacts"].items():
            _require(_artifact_identity(recorder, name) == binding, "cached calibration artifact changed")
        summary = recorder.load_object("analysis/summary.pkl")
    _require(json.loads((card / "summary.json").read_text()) == summary, "card summary differs from Recorder")
    return summary


def run(root, card):
    from qlib.workflow import R
    root, card = Path(root).resolve(), Path(card).resolve()
    plan, identity = read_plan(card)
    hashes = source_hashes(root)
    _no_active_recorder()
    cached = _cached(card, identity, hashes)
    if cached is not None:
        return cached
    deadline_check(plan)
    bundle = load_inputs(root, plan)
    uri = str(CANONICAL_ROOT / "research" / EXPERIMENT / "qlib_recorder")
    with R.start(experiment_name=EXPERIMENT, uri=uri):
        recorder = R.get_recorder()
        reference = {**identity, "source_sha256": hashes, "recorder_id": recorder.id,
                     "recorder_uri": uri, "complete": False}
        _write_json(card / "recorder_reference.json", reference)
        for name in ("plan.json", "registration.json"):
            recorder.log_artifact(str(card / name), artifact_path="inputs")
        R.save_objects(artifact_path="inputs", **{
            "diagnostic_rows.pkl": bundle["diagnostic_rows"],
            "source_audit.pkl": bundle["calibration_input_audit"],
            "head_readback_audit.pkl": bundle["head_readback_audit"], "source_hashes.pkl": hashes})
        R.log_params(plan_sha256=identity["plan_sha256"], model="prior_only_affine_OLS", annual_fits=11,
                     frozen_market_head_refits=0, portfolio_paths=0, bootstrap_resamples=2000)
        rows, fits, fit_audit = fit_calibrations(bundle["diagnostic_rows"], plan)
        R.save_objects(artifact_path="models", **{"calibrations.pkl": fits, "fit_audit.pkl": fit_audit})
        R.save_objects(artifact_path="analysis", **{"calibrated_predictions.pkl": rows})
        results = analyze(rows, plan)
        tables = {**results["tables"], "calibration_fits": fit_audit}
        for name, table in tables.items():
            table.to_csv(card / (name + ".csv"), index=False)
        passed = bool(tables["screening_gates"].passed.all())
        R.log_metrics(screening_passed=float(passed), failed_gates=float((~tables["screening_gates"].passed).sum()))
        finished_at = datetime.now(timezone.utc).isoformat()
        deadline_check(plan)
        summary = {"experiment": EXPERIMENT, "record_type": "bounded_causal_cash_calibration_screen",
            **identity, "source_sha256": hashes, "recorder_id": recorder.id, "recorder_uri": uri,
            "recorder_local_dir": recorder.get_local_dir(), "provider_uri": plan["source"]["provider_uri"],
            "region": "us", "market": "all", "as_of": plan["source"]["as_of"], "source": plan["source"],
            "parent": plan["parent"], "input_audit": bundle["calibration_input_audit"],
            "screening_passed": passed, "failed_gate_count": int((~tables["screening_gates"].passed).sum()),
            "decision": "screen_passed_further_consideration_only" if passed else "abandon_cash_calibration_route",
            "objective_achieved": False, "strategy_candidate_promotable": False, "new_forward_qualified": False,
            "historical_years_reused": True, "calibration_fits": len(fits), "forecast_model_refits": 0,
            "market_backtests": 0, "new_portfolio_paths": 0, "production_change": False, "real_broker_orders": False,
            "completed_at_utc": finished_at, "deadline_utc": plan["deadline_utc"],
            "sidecars": [name + ".csv" for name in tables], "artifact_policy": plan["artifact_policy"],
            "qlib_docs_reference": plan["source_plan"]["qlib_docs"],
            "benchmark": ["raw_frozen_cash_prediction", "saved_original_prior_mean"],
            "model_selection": "one registered recipe; no selection or search",
            "limitations": [plan["source"]["limitations"], plan["fees"]["limitations"],
                            "Historical windows reused; conditional-event intervals can be undefined; no independent forward evidence."]}
        _require(source_hashes(root) == hashes and read_plan(card)[1] == identity, "source or plan changed during run")
        R.save_objects(artifact_path="analysis", **{"tables.pkl": tables,
            "bootstrap_draws.pkl": results["bootstrap_draws"], "summary.pkl": summary})
        _write_json(card / "summary.json", summary)
        artifacts = {}
        for path in sorted((Path(recorder.get_local_dir()) / "artifacts").rglob("*")):
            name = path.relative_to(Path(recorder.get_local_dir()) / "artifacts").as_posix()
            if path.is_file() and name.split("/")[0] in {"inputs", "models", "analysis"}:
                artifacts[name] = _artifact_identity(recorder, name)
        reference["artifacts"] = artifacts
    with R.uri_context(uri):
        recorder = R.get_recorder(experiment_name=EXPERIMENT, recorder_id=reference["recorder_id"])
        _require(recorder.client.get_run(recorder.id).info.status == "FINISHED", "child Recorder did not finish")
    reference["complete"] = True
    _write_json(card / "recorder_reference.json", reference)
    return summary


def conclusion(card):
    """Render the notebook's Chinese conclusion from finished Recorder exports."""
    card = Path(card)
    summary = json.loads((card / "summary.json").read_text())
    points, events, fits, gates = (pd.read_csv(card / name) for name in
        ("point_metrics.csv", "event_metrics.csv", "calibration_fits.csv", "screening_gates.csv"))
    lines = ["# 因果现金预测校准：限时单次筛选", "",
        "本次只检验一个预登记方案：每年 1 月 1 日，用自 2013 年以来严格在此前成熟的 OOF 现金预测与 H1 标签，拟合 y=a+b×p；全年冻结 a、b。2016–2026 共 11 次两参数 OLS，原 14 个预测头未重训。", "",
        f"结论：{'筛选通过，仅可进一步考虑，不能晋升。' if summary['screening_passed'] else '筛选未通过，立即放弃这条现金校准路线。'}失败门禁 {summary['failed_gate_count']} 项；不启动其他配方、参数网格或组合回测。", ""]
    for window in POOLED:
        selected = points.loc[points.window.eq(window)]
        lines.append(f"{window}：{int(selected.iloc[0].rows)} 个已成熟样本；校准 MSE={selected.iloc[0].MSE_prediction:.9g}。")
        for row in selected.itertuples():
            lines.append(f"- 对 {row.comparator}：基准 MSE={row.MSE_prior_baseline:.9g}，基准减校准的 MSE 改善={row.paired_MSE_improvement_baseline_minus_prediction:.9g}。")
        for row in events.loc[events.window.eq(window) & events.forecast_role.eq("calibrated")].itertuples():
            value = "未定义（无事件）" if pd.isna(row.mean_event_net_log_bp) else f"{row.mean_event_net_log_bp:.4f} bp"
            lines.append(f"- 单边成本 {row.cost_bps:g} bp：孤立事件 {row.event_count} 次，条件平均净相对 log 收益 {value}。")
        lines.append("")
    for row in gates.loc[gates.gate.eq("annual_MSE")].itertuples():
        lines.append(f"开发期七年中，对 {row.comparator} 的 MSE 改善年数为 {int(row.observed)}/7（要求至少 6/7）。")
    user = points.loc[points.window.eq("user_trade_2026_08_03")].iloc[0]
    lines.extend(["", f"用户窗口按 2026-08-03 交易开盘至 2026-09-25 最后已观察开盘限定：{int(user.rows)} 个成熟观测，校准 MSE={user.MSE_prediction:.9g}；年表与双窗口差异见 point_metrics.csv，事件与区间见 event_metrics.csv / bootstrap_summary.csv。未新增行情状态分层。",
        f"年度校准斜率范围 {fits.slope.min():.6g} 至 {fits.slope.max():.6g}；每次训练日期、成熟时间上界、系数和散列见 calibration_fits.csv。",
        "", "风险：全部历史窗口已被复用，不属于新的独立样本；原数据的 published affine 与 serverQFQ 差异及 2012 年 1 月次级报价不确定性保留。现金事件只是 SOXX→现金→SOXX 的一期开盘费用测试，不能连乘或当成组合收益；不存在新前向证据，不晋升、不改生产。", "",
        "下一步：本轮到此结束，保留失败或通过的完整证据；不把筛选结果声明为稳定超额收益。",
        f"Qlib Recorder：{summary['recorder_id']}；数据源 {summary['provider_uri']}，as_of={summary['as_of']}；完成于 {summary['completed_at_utc']}（截止 {summary['deadline_utc']}）。", ""])
    text = "\n".join(lines)
    (card / "conclusion.md").write_text(text)
    return text
