"""Registered semiconductor breadth information test on native Qlib inputs.

The four sector measurements are continuous, prior-publication basket features.
There is no portfolio, regime classifier, feature search, or promotion workflow.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.frozen_cash_information import correlation, event_metrics
from src.workflow.soxs_cash5_revalidation import (
    BASE_FEATURES, TARGET, check_deadline, json_ready, paired_bootstrap as _paired_bootstrap,
    paired_metrics as _paired_metrics, sha256_file, write_json,
)

EXPERIMENT = "semiconductor_breadth_20261005"
SECTOR_FEATURES = (
    "breadth_up_fraction", "equal_minus_disclosed_weight_return",
    "cross_section_return_std", "mean_abs_correlation_60",
)
ALL_FEATURES = (*BASE_FEATURES, *SECTOR_FEATURES)
FEATURE_FAMILIES = {"base": BASE_FEATURES, "base_plus_breadth": ALL_FEATURES}
FIT_YEARS = (2024, 2025, 2026)
TRAIN_START = "2022-05-27"
FEATURE_END = "2026-07-02"
DEADLINE_UTC = "2026-10-05T04:53:21+00:00"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("blocked: " + message)


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def build_base_features_and_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Exact BASE5 recipe; form H5 on the full canonical calendar before joining."""
    required = ["soxl_close", "soxx_close", "soxx_open"]
    _require(isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is None,
             "canonical prices require a naive Qlib session index")
    _require(prices.index.is_monotonic_increasing and not prices.index.has_duplicates,
             "canonical sessions must be ordered and unique")
    _require(set(required).issubset(prices), "missing canonical price field")
    _require(np.isfinite(prices[required].to_numpy()).all() and (prices[required] > 0).all().all(),
             "canonical full-history prices must be finite and positive")
    frame = prices[required].copy()
    returns = frame.soxx_close.pct_change(fill_method=None)
    frame["soxx_trend20"] = frame.soxx_close / frame.soxx_close.rolling(20).mean() - 1
    frame["soxx_trend60"] = frame.soxx_close / frame.soxx_close.rolling(60).mean() - 1
    frame["soxx_vol20"] = returns.rolling(20).std(ddof=1)
    frame["soxx_drawdown60"] = frame.soxx_close / frame.soxx_close.rolling(60).max() - 1
    frame["relative_momentum20"] = frame.soxl_close.pct_change(20, fill_method=None) - frame.soxx_close.pct_change(20, fill_method=None)
    dates = frame.index.to_series()
    frame["trade_date"] = dates.shift(-1)
    frame["label_mature_date"] = dates.shift(-6)
    frame[TARGET] = -np.log(frame.soxx_open.shift(-6) / frame.soxx_open.shift(-1))
    frame["feature_available_at"] = frame.index.tz_localize("America/New_York") + pd.Timedelta(hours=17)
    frame["label_available_at"] = pd.DatetimeIndex(frame.label_mature_date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
    frame.index.name = "signal_date"
    return frame


def load_native_inputs(root: Path, plan: dict[str, Any]):
    """Use native D readbacks for both price labels and full-basket features."""
    import qlib
    from src.data.constrained_net_excess import _native_readback
    from src.data.semiconductor_breadth import load_sector_breadth_features

    raw, calendar, canonical_audit = _native_readback(root, plan, plan["canonical_source_audit"])
    prices = pd.DataFrame(index=calendar)
    for symbol, field, name in (("SOXL", "$close", "soxl_close"), ("SOXX", "$close", "soxx_close"), ("SOXX", "$open", "soxx_open")):
        prices[name] = raw.xs(symbol, level="instrument")[field].reindex(calendar)
    base = build_base_features_and_labels(prices)
    source = plan["sector_source"]
    sector, eligibility, sector_audit = load_sector_breadth_features(
        provider_uri=root / source["provider_uri"],
        supplemental_provider_uri=root / source["supplemental_provider_uri"],
        start=source["start"], end=source["end"],
    )
    expected = calendar[(calendar >= TRAIN_START) & (calendar <= FEATURE_END)]
    _require(sector.index.equals(expected), "sector features must cover every registered canonical session without gaps")
    _require(list(sector.columns) == list(SECTOR_FEATURES), "sector feature contract changed")
    _require(np.isfinite(sector.to_numpy()).all(), "needs data: incomplete full-basket feature history; no row dropping")
    frame = base.loc[expected].join(sector)
    _require(np.isfinite(frame[[*ALL_FEATURES, TARGET]].to_numpy()).all(),
             "needs data: paired BASE5/sector/label rows are incomplete")
    qlib.init(provider_uri=str((root / plan["source"]["provider_uri"]).resolve()), region="us", kernels=1)
    audit = {
        "provider_uri": str((root / plan["source"]["provider_uri"]).resolve()),
        "canonical_readback": canonical_audit, "sector_readback": sector_audit,
        "as_of": plan["source"]["as_of"], "feature_start": str(expected.min().date()),
        "feature_end": str(expected.max().date()), "paired_rows": len(frame),
        "native_input_sha256": frame_hash(frame), "sector_eligibility_sha256": frame_hash(eligibility),
        "label_calendar_scope": "full canonical Qlib calendar through 2026-09-25, before sector-window join",
        "membership_scope": "full previous-session-published SEC SOXX holdings basket; not official daily index membership",
        "price_vintage_limitation": "retrospective adjustment vintage; not archived PIT adjusted prices",
        "no_member_drop_or_fill": True, "no_august_2026_coverage": True,
    }
    return frame, eligibility, audit


def build_dataset(frame: pd.DataFrame):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    _require(isinstance(frame.index, pd.DatetimeIndex) and frame.index.is_monotonic_increasing
             and not frame.index.has_duplicates, "ordered unique dataset sessions required")
    _require(not frame.empty and np.isfinite(frame[[*ALL_FEATURES, TARGET]].to_numpy()).all(),
             "interior missing paired feature or label rows may not be removed")
    static = pd.concat({"feature": frame[list(ALL_FEATURES)], "label": frame[[TARGET]]}, axis=1)
    static.index = pd.MultiIndex.from_arrays([frame.index, np.repeat("SOXX_CASH5", len(frame))], names=["datetime", "instrument"])
    handler = DataHandlerLP(instruments=None, start_time=frame.index.min(), end_time=frame.index.max(), data_loader=StaticDataLoader(config=static))
    dataset = DatasetH(handler=handler, segments={"all": (frame.index.min(), frame.index.max())})
    return dataset, {"api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"], "source": "native Qlib D readback", "rows": len(frame), "feature_count": len(ALL_FEATURES), "training_start": TRAIN_START}


def prepare_dataset(dataset) -> pd.DataFrame:
    from qlib.data.dataset import DatasetH
    _require(isinstance(dataset, DatasetH), "native DatasetH is mandatory")
    prepared = dataset.prepare("all", col_set=["feature", "label"]).droplevel("instrument").sort_index()
    prepared.columns = prepared.columns.get_level_values(-1)
    return prepared


def training_rows(prepared: pd.DataFrame, market: pd.DataFrame, year: int):
    boundary = pd.Timestamp(f"{year}-01-01")
    cutoff = boundary.tz_localize("America/New_York")
    prior = prepared.loc[(prepared.index >= TRAIN_START) & (prepared.index < boundary)]
    _require(len(prior) > 20, "insufficient prior sessions for 20-row purge")
    purged = prior.iloc[:-20]
    _require(np.isfinite(purged[[*ALL_FEATURES, TARGET]].to_numpy()).all(), "invalid interior training values")
    timing = market.loc[purged.index, ["feature_available_at", "label_available_at"]]
    _require(timing.notna().all().all(), "missing actual training availability timestamp")
    mature = timing.feature_available_at.lt(cutoff) & timing.label_available_at.lt(cutoff)
    train = purged.loc[mature]
    _require(len(train) >= 252, "needs data: fewer than 252 mature training rows")
    return train, {
        "fit_year": year, "train_rows": len(train), "train_start": train.index.min(), "train_end": train.index.max(),
        "purged_signal_rows": 20, "first_purged_signal_date": prior.index[-20], "last_purged_signal_date": prior.index[-1],
        "maturity_excluded_rows": int((~mature).sum()), "cutoff_at": cutoff,
        "maximum_label_mature_date": market.loc[train.index, "label_mature_date"].max(),
        "maximum_label_available_at": timing.loc[train.index, "label_available_at"].max(),
        "maximum_feature_available_at": timing.loc[train.index, "feature_available_at"].max(),
        "actual_maturity_guard": True, "dataset_prepare_is_fit_source": True,
    }


def fit_predict(dataset, market: pd.DataFrame, *, deadline: str = DEADLINE_UTC):
    prepared = prepare_dataset(dataset)
    outputs, audits, models = [], [], {}
    # Validate all six planned fits before spending any of the fit budget.
    boundaries = {year: training_rows(prepared, market, year) for year in FIT_YEARS}
    for year in FIT_YEARS:
        check_deadline(deadline)
        train, audit = boundaries[year]
        test = prepared.loc[prepared.index.year == year]
        _require(len(test) >= 3 and np.isfinite(test[[*ALL_FEATURES, TARGET]].to_numpy()).all(), "invalid registered evaluation year")
        out = market.loc[test.index, ["trade_date", "label_mature_date", TARGET]].copy()
        out["prior_mean"] = float(train[TARGET].mean())
        out["fit_year"] = year
        for family, features in FEATURE_FAMILIES.items():
            check_deadline(deadline)
            model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
            model.fit(train.loc[:, list(features)], train[TARGET])
            out[family] = model.predict(test.loc[:, list(features)])
            models[f"{year}_{family}"] = model
            audits.append({**audit, "family": family, "feature_count": len(features), "alpha": 10.0, "test_rows": len(test), "test_start": test.index.min(), "test_end": test.index.max()})
        outputs.append(out)
    _require(len(models) == 6, "registered fit budget must equal exactly six")
    return pd.concat(outputs).sort_index(), pd.DataFrame(audits), models


def _rename_metric_names(value):
    if isinstance(value, dict):
        return {key.replace("base_plus_soxs", "base_plus_breadth"): val for key, val in value.items()}
    return value.rename(columns=lambda column: column.replace("base_plus_soxs", "base_plus_breadth"))


def paired_metrics(sample: pd.DataFrame) -> dict[str, Any]:
    return _rename_metric_names(_paired_metrics(sample.rename(columns={"base_plus_breadth": "base_plus_soxs"})))


def paired_bootstrap(sample: pd.DataFrame, *, paths=2000, block_length=20, seed=20261005):
    """Reuse the reviewed paired-index implementation without modifying it."""
    stats, draws = _paired_bootstrap(sample.rename(columns={"base_plus_breadth": "base_plus_soxs"}), paths=paths, block_length=block_length, seed=seed)
    return _rename_metric_names(stats), _rename_metric_names(draws)


def evaluation_slices(predictions: pd.DataFrame):
    return {"pooled_2024_2026_partial": predictions, **{f"year_{year}": predictions.loc[predictions.index.year == year] for year in FIT_YEARS}}


def signal_analysis(slices):
    """Single-asset time-series equivalent analyzer, saved in native Recorder.

    Cross-sectional daily SigAnaRecord IC is undefined for one synthetic asset.
    """
    rows = []
    for name, sample in slices.items():
        for family in FEATURE_FAMILIES:
            pearson, p_reason = correlation(sample[family], sample[TARGET])
            spearman, s_reason = correlation(sample[family], sample[TARGET], rank=True)
            rows.append({"slice": name, "family": family, "pearson_ic": pearson, "spearman_ic": spearman,
                         "pearson_undefined_reason": p_reason, "spearman_undefined_reason": s_reason,
                         "api": "src.analysis.frozen_cash_information.correlation",
                         "scope": "single_asset_time_series_equivalent_analyzer_in_qlib_recorder"})
    return pd.DataFrame(rows)


def research_gates(points: pd.DataFrame, bootstrap: pd.DataFrame):
    annual = points.loc[points["slice"].isin([f"year_{year}" for year in FIT_YEARS])]
    pooled = bootstrap.loc[bootstrap["slice"].eq("pooled_2024_2026_partial")]
    _require(len(annual) == 3 and set(pooled.block_length) == {20, 60} and len(pooled) == 2,
             "registered gates require three annual slices and both pooled block lengths")
    positive_ic = int((annual.delta_ic_full_minus_base > 0).sum())
    positive_mse = int((annual.mse_reduction_base_minus_full > 0).sum())
    same_year = int(((annual.delta_ic_full_minus_base > 0) & (annual.mse_reduction_base_minus_full > 0)).sum())
    def lower_positive(metric):
        values = pooled[f"{metric}_q0.025"]
        return bool(values.notna().all() and (values > 0).all())
    positive_full = int((annual.base_plus_breadth_ic > 0).sum())
    information = positive_full >= 2 and lower_positive("base_plus_breadth_ic")
    stability = positive_ic >= 2 and positive_mse >= 2
    incremental = stability and lower_positive("delta_ic_full_minus_base") and lower_positive("mse_reduction_base_minus_full")
    economic = all(lower_positive(f"full_conditional_net_event_mean_bp{fee}") for fee in (10, 25))
    passed = bool(information and incremental and economic)
    return {"positive_delta_ic_years": positive_ic, "positive_mse_reduction_years": positive_mse,
            "joint_positive_increment_years": same_year, "positive_full_ic_years": positive_full, "evaluated_years": 3,
            "information_pass": information, "incremental_pass": incremental, "annual_stability_pass": stability,
            "economic_pass": economic, "all_incremental_information_gates_pass": passed,
            "decision": "further_research_only_no_promotion" if passed else "end_route_no_further_search",
            "promotion_decision": "not_promoted", "fresh_holdout": False,
            "economic_scope": "isolated_H5_event_screen_only", "stable_excess_gate_evaluated": False}


def verify_registered_plan(plan: dict[str, Any]) -> None:
    _require(all((
        plan.get("experiment") == EXPERIMENT,
        plan.get("registered_before_forecast_fits") is True,
        plan["features"]["base"] == list(BASE_FEATURES),
        plan["features"]["base_plus_breadth"] == list(ALL_FEATURES),
        plan["training"]["start"] == TRAIN_START,
        plan["training"]["cutoff_years"] == list(FIT_YEARS),
        plan["training"]["purge_rows"] == 20, plan["training"]["minimum_rows"] == 252,
        plan["sector_source"]["start"] == TRAIN_START, plan["sector_source"]["end"] == FEATURE_END,
        plan["uncertainty"]["resamples"] == 2000, plan["uncertainty"]["blocks"] == [20, 60],
        plan["uncertainty"]["seed"] == 20261005, plan["fees"]["cost_bps"] == [10, 25],
    )), "experiment differs from the registered fixed protocol")


def smoke_inputs(root: Path | str):
    root = Path(root).resolve()
    plan = json.loads((root / "research" / EXPERIMENT / "plan.json").read_text())
    verify_registered_plan(plan)
    frame, eligibility, audit = load_native_inputs(root, plan)
    dataset, dataset_audit = build_dataset(frame)
    prepared = prepare_dataset(dataset)
    return json_ready({"data_audit": audit, "native_input_sha256": frame_hash(frame),
                       "qlib_dataset": dataset_audit,
                       "fit_boundaries": [training_rows(prepared, frame, year)[1] for year in FIT_YEARS],
                       "fit_count": 0, "financial_metrics_computed": False})


def verify_execution_seal(root: Path, card: Path, authorization: dict[str, Any]):
    fixed = {"plan_sha256": card / "plan.json", "workflow_sha256": card / "workflow.yaml",
             "implementation_sha256": Path(__file__), "data_implementation_sha256": root / "src/data/semiconductor_breadth.py"}
    if authorization.get("approved") is not True or any(authorization.get(key) != sha256_file(path) for key, path in fixed.items()):
        raise PermissionError("blocked: root GO execution seal differs from registered code or plan")
    files = authorization.get("source_files", {})
    if not files or any(sha256_file(root / name) != digest for name, digest in files.items()):
        raise PermissionError("blocked: root GO source seal differs from reviewed files")


def make_conclusion(summary):
    if summary["status"] == "needs_data":
        return ("# 最终中文结论\n\n本轮登记了半导体广度与相关性的增量信息检验，但完整历史输入未通过门禁。"
                f"原因：{summary['reason']}。模型拟合 0 次，没有计算金融指标或构造投资组合。"
                "\n\n不晋升；不得用缺员篮子、前向填充或缩短窗口替代登记的数据合同。"
                f"Qlib Recorder：`{summary['recorder_id']}`。下一步是补齐并独立核验数据，另行登记；本轮停止。\n")
    gates = summary["gates"]
    rows = {row["slice"]: row for row in summary["key_point_metrics"]}
    pooled = rows["pooled_2024_2026_partial"]
    def number(value):
        return "未定义" if value is None else f"{value:.6f}"
    annual = "；".join(f"{year}年 ΔIC={number(rows[f'year_{year}']['delta_ic_full_minus_base'])}、MSE改善={number(rows[f'year_{year}']['mse_reduction_base_minus_full'])}" for year in FIT_YEARS)
    return f"""# 最终中文结论

本轮使用固定 BASE5 ETF 特征，对比增加四个连续半导体特征：上涨比例、等权减已披露持仓权重收益、横截面收益标准差与 60 日平均绝对成对相关性。篮子由上一交易日以前已公开 SEC 持仓构成，每日要求完整 30 个成员及 61 个收盘价；不把它称为官方每日指数成分。训练起于 {TRAIN_START}，2024/2025/2026 各年仅拟合两个 StandardScaler + Ridge(alpha=10)，共 {summary['fit_count']} 次，均经 Qlib D → DatasetH.prepare → Recorder；20 行剔除并要求 H5 标签在年界以前成熟。

评价期为 2024—2025 全年与 2026 截至 {FEATURE_END}，共 {pooled['rows']} 行。BASE5 / 加广度的 Spearman IC 为 {number(pooled['base_ic'])} / {number(pooled['base_plus_breadth_ic'])}，ΔIC={number(pooled['delta_ic_full_minus_base'])}，MSE 改善（base-full）={number(pooled['mse_reduction_base_minus_full'])}。{annual}。未做事后 regime 分组；年间差异见 `point_metrics.csv`。

两个块长 20/60、各 2,000 次相同索引配对自助法结果见 `bootstrap_summary.csv`。信息门 / 增量门 / 经济门为 {gates['information_pass']} / {gates['incremental_pass']} / {gates['economic_pass']}；加广度 IC 为正 {gates['positive_full_ic_years']}/3 年，IC 增量为正 {gates['positive_delta_ic_years']}/3 年、MSE 改善 {gates['positive_mse_reduction_years']}/3 年，结论 `{gates['decision']}`。10/25bp 仅是重叠独立 H5 现金事件诊断，使用 log((1+c)/(1-c)) 资金门槛，不复利、不年化，不是投组收益或稳定超额验证。

不晋升。全部评价历史已被使用过，样本范围较短，2026 年不完整且没有 8 月窗口；价格复权是回溯版本，不是归档的历史时点复权价。定期披露篮子存在披露滞后与持仓变动风险，旧 WOLF 补充来源及身份限制见 `source_audit.json`。本轮没有 HMM、人工状态配置、参数搜索、投资组合或真实订单。Qlib Recorder：`{summary['recorder_id']}`。

下一步：{'只允许另立独立验证计划，本轮不继续搜索。' if gates['all_incremental_information_gates_pass'] else '结束该路线并保留失败证据，本轮不增加特征或修改门槛。'}
"""


def run_experiment(*, root: Path | str):
    """One authorized attempt; reopening a completed recorder never refits."""
    from qlib.workflow import R
    root = Path(root).resolve()
    card = root / "research" / EXPERIMENT
    plan = json.loads((card / "plan.json").read_text())
    verify_registered_plan(plan)
    authorization_path = card / "run_authorization.json"
    if not authorization_path.exists():
        raise PermissionError("blocked: independent review and root GO required before financial metrics")
    authorization = json.loads(authorization_path.read_text())
    verify_execution_seal(root, card, authorization)
    recorder_uri = Path(plan["recorder_uri"]).resolve()
    summary_path, reference_path = card / "summary.json", card / "recorder_reference.json"
    if summary_path.exists():
        cached = json.loads(summary_path.read_text())
        if cached.get("status") in ("finished", "needs_data"):
            reference = json.loads(reference_path.read_text())
            _require(reference.get("complete") is True and reference.get("recorder_id") == cached["recorder_id"], "completed recorder reference differs")
            with R.uri_context(recorder_uri.as_uri()):
                recorder = R.get_recorder(recorder_id=cached["recorder_id"], experiment_name=EXPERIMENT)
            _require(recorder.status == "FINISHED", "cached native recorder not FINISHED")
            artifact_root = Path(recorder.get_local_dir()) / "artifacts"
            _require(bool(reference.get("artifact_sha256")) and all(sha256_file(artifact_root / name) == digest for name, digest in reference["artifact_sha256"].items()), "recorded evidence hashes differ")
            _require(json_ready(recorder.load_object("card/summary.pkl")) == cached, "summary differs from recorder")
            return cached
    if reference_path.exists():
        raise RuntimeError("a native Recorder attempt already exists; automatic refit forbidden")
    check_deadline(plan["deadline_utc"])
    frame, eligibility, data_audit = load_native_inputs(root, plan)
    if authorization.get("native_input_sha256") != frame_hash(frame):
        raise PermissionError("blocked: native input differs from root-reviewed smoke")
    dataset, dataset_audit = build_dataset(frame)
    prepared = prepare_dataset(dataset)
    for year in FIT_YEARS:
        training_rows(prepared, frame, year)
    started = datetime.now(timezone.utc).isoformat()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    with R.start(experiment_name=EXPERIMENT, uri=recorder_uri.as_uri()):
        recorder = R.get_recorder()
        reference = {"complete": False, "recorder_id": recorder.id, "recorder_uri": str(recorder_uri), "plan_sha256": sha256_file(card / "plan.json"), "started_at_utc": started}
        write_json(reference_path, reference)
        R.log_params(plan_sha256=reference["plan_sha256"], native_input_sha256=frame_hash(frame), fit_budget=6, ridge_alpha=10, all_history_reused=True, target=TARGET)
        recorder.save_objects(artifact_path="inputs", **{"features_labels.pkl": frame, "eligibility.pkl": eligibility, "plan.pkl": plan, "data_audit.pkl": data_audit, "authorization.pkl": authorization})
        predictions, fits, models = fit_predict(dataset, frame, deadline=plan["deadline_utc"])
        slices = evaluation_slices(predictions)
        points = pd.DataFrame([{"slice": name, **paired_metrics(sample)} for name, sample in slices.items()])
        events = pd.DataFrame([{"slice": name, "family": family, **event_metrics(sample[TARGET], sample[family], cost_bps=fee)} for name, sample in slices.items() for family in (*FEATURE_FAMILIES, "prior_mean") for fee in (10, 25)])
        analyzer = signal_analysis(slices)
        stats, distributions = [], {}
        for name in plan["uncertainty"].get("slices", ["pooled_2024_2026_partial"]):
            for block in (20, 60):
                check_deadline(plan["deadline_utc"])
                row, draws = paired_bootstrap(slices[name], paths=2000, block_length=block, seed=20261005)
                stats.append({"slice": name, **row})
                distributions[f"{name}_block{block}"] = draws
        bootstrap = pd.DataFrame(stats)
        gates = research_gates(points, bootstrap)
        verify_execution_seal(root, card, authorization)
        check_deadline(plan["deadline_utc"])
        R.log_metrics(fit_count=len(models), joint_positive_increment_years=gates["joint_positive_increment_years"], all_incremental_information_gates_pass=int(gates["all_incremental_information_gates_pass"]))
        recorder.save_objects(artifact_path="models", **{"fitted_models.pkl": models})
        recorder.save_objects(artifact_path="analysis", **{"predictions.pkl": predictions, "labels.pkl": frame[[TARGET, "trade_date", "label_mature_date"]], "bootstrap_draws.pkl": distributions, "equivalent_signal_analysis.pkl": analyzer})
        for name, table in {"point_metrics": points, "event_metrics": events, "bootstrap_summary": bootstrap, "fit_audit": fits, "signal_analysis": analyzer}.items():
            table.to_csv(card / f"{name}.csv", index=False)
        write_json(card / "source_audit.json", data_audit)
        summary = {
            "experiment": EXPERIMENT, "status": "finished", "card_type": "registered_reused_history_incremental_information_diagnostic",
            "primary_notebook": f"{EXPERIMENT}.ipynb", "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": reference["plan_sha256"], "implementation_sha256": authorization["implementation_sha256"], "native_input_sha256": frame_hash(frame),
            "fit_count": len(models), "provider_uri": data_audit["provider_uri"], "as_of": plan["source"]["as_of"], "region": "us", "market": "all",
            "data_audit": data_audit, "qlib_dataset": dataset_audit, "recorder_id": recorder.id, "recorder_uri": str(recorder_uri),
            "recorder_local_dir": str(Path(recorder.get_local_dir()).resolve()), "recorder_artifact_uri": recorder.get_artifact_uri(), "qlib_docs": QLIB_DOCS,
            "analysis_api": "src.analysis.frozen_cash_information equivalent time-series analyzer persisted in native Qlib Recorder",
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml", "recorder_only_artifacts": ["inputs/features_labels.pkl", "inputs/eligibility.pkl", "models/fitted_models.pkl", "analysis/predictions.pkl", "analysis/labels.pkl", "analysis/bootstrap_draws.pkl", "analysis/equivalent_signal_analysis.pkl"]},
            "sidecars": ["plan.json", "run_authorization.json", "workflow.yaml", "point_metrics.csv", "event_metrics.csv", "bootstrap_summary.csv", "fit_audit.csv", "signal_analysis.csv", "source_audit.json", "conclusion.md"],
            "gates": gates, "promotion_decision": "not_promoted", "selection_uses_reused_history": True, "fresh_holdout": False,
            "model_selection": {"performed": False, "families": list(FEATURE_FAMILIES), "fixed_model": "StandardScaler + Ridge(alpha=10)"},
            "benchmark": {"predictive": ["base5", "training_prior_mean"], "event": "SOXX on identical open[t+1] through open[t+6]"},
            "key_point_metrics": json_ready(points.to_dict("records")), "no_portfolio_backtest": True, "stable_excess_gate_evaluated": False,
            "price_vintage_is_archived_pit": False, "august_2026_evaluated": False,
        }
        (card / "conclusion.md").write_text(make_conclusion(summary), encoding="utf-8")
        for name in summary["sidecars"]:
            recorder.log_artifact(str(card / name), artifact_path="card")
        recorder.save_objects(artifact_path="card", **{"summary.pkl": summary})
    with R.uri_context(recorder_uri.as_uri()):
        finished = R.get_recorder(recorder_id=summary["recorder_id"], experiment_name=EXPERIMENT)
    _require(finished.status == "FINISHED", "native Recorder failed to finish")
    verify_execution_seal(root, card, authorization)
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    for name in ("code_status.txt", "code_diff.txt", "code_cached.txt"):
        (artifacts / name).unlink(missing_ok=True)
    reference.update(complete=True, artifact_sha256={str(path.relative_to(artifacts)): sha256_file(path) for path in sorted(artifacts.rglob("*")) if path.is_file()})
    write_json(summary_path, summary)
    write_json(reference_path, reference)
    return json_ready(summary)
