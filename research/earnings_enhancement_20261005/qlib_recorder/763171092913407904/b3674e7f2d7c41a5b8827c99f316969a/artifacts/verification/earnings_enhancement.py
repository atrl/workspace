"""Fixed, recorder-first dual-head EPS-surprise information diagnostic.

Vendor historical EPS estimates are not archived contemporaneous vintages. This
module cannot promote a model, run a portfolio, or change broker/paper state.
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

from src.analysis.frozen_cash_information import correlation, cash_hurdle
from src.objective.relative_wealth import stationary_block_bootstrap_indices
from src.workflow.semiconductor_breadth import build_base_features_and_labels as _base_recipe
from src.workflow.soxs_cash5_revalidation import (
    BASE_FEATURES, TARGET as OLD_CASH_TARGET, _row_rank_correlation,
    check_deadline, json_ready, sha256_file, write_json,
)

EXPERIMENT = "earnings_enhancement_20261005"
EARNINGS_FEATURES = ("earnings_surprise_nvda", "earnings_surprise_amd", "earnings_surprise_intc")
ALL_FEATURES = (*BASE_FEATURES, *EARNINGS_FEATURES)
FEATURE_FAMILIES = {"base": BASE_FEATURES, "base_plus_earnings": ALL_FEATURES}
TARGETS = ("cash5", "leverage5")
FIT_YEARS = tuple(range(2016, 2023))
TRAIN_START = "2011-01-03"
FEATURE_END = "2022-12-30"
DEADLINE_UTC = "2026-10-05T07:05:04Z"
POOLED = "development_2016_2022"
REQUIRED_SEAL_SOURCES = {
    "src/workflow/earnings_enhancement.py", "src/data/earnings_surprise.py",
    "src/workflow/semiconductor_breadth.py", "src/workflow/soxs_cash5_revalidation.py",
    "src/analysis/frozen_cash_information.py", "src/objective/relative_wealth.py",
    "src/data/constrained_net_excess.py", "src/data/qlib_native.py", "src/data/price_basis.py",
}
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("blocked: " + message)


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def build_features_and_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Derive both H5 labels on the entire canonical calendar before any join."""
    _require("soxl_open" in prices, "SOXL open is required for leverage5")
    _require(np.isfinite(prices.soxl_open).all() and prices.soxl_open.gt(0).all(), "invalid SOXL opens")
    result = _base_recipe(prices).rename(columns={OLD_CASH_TARGET: "cash5"})
    result["leverage5"] = (np.log(prices.soxl_open.shift(-6) / prices.soxl_open.shift(-1))
                           - np.log(prices.soxx_open.shift(-6) / prices.soxx_open.shift(-1)))
    return result


def load_native_inputs(root: Path, plan: dict[str, Any]):
    """Read prices and EPS-state features through Qlib; no financial fit."""
    import qlib
    from src.data.constrained_net_excess import _native_readback
    from src.data.earnings_surprise import load_earnings_surprise_features

    raw, calendar, canonical_audit = _native_readback(root, plan, plan["canonical_source_audit"])
    prices = pd.DataFrame(index=calendar)
    for symbol in ("SOXL", "SOXX"):
        for field in ("open", "close"):
            prices[f"{symbol.lower()}_{field}"] = raw.xs(symbol, level="instrument")[f"${field}"].reindex(calendar)
    full = build_features_and_labels(prices)
    immature = full.loc[full[list(TARGETS)].isna().any(axis=1)]
    _require(immature.index.equals(calendar[-6:]), "only final six canonical signal dates may have immature H5 labels")
    source = plan["earnings_source"]
    features, earnings_audit = load_earnings_surprise_features(
        root / source["provider_uri"], start=source["start"], end=source["end"])
    _require(list(features.columns) == list(EARNINGS_FEATURES), "earnings feature schema/order differs")
    _require(isinstance(features.index, pd.DatetimeIndex) and features.index.tz is None
             and features.index.is_monotonic_increasing and not features.index.has_duplicates,
             "earnings features require unique naive canonical session dates")
    expected = calendar[(calendar >= TRAIN_START) & (calendar <= FEATURE_END)]
    _require(expected.isin(features.index).all(), "needs data: missing registered 2011-2022 earnings sessions")
    selected = features.loc[expected]
    _require(np.isfinite(selected.to_numpy()).all(), "needs data: every registered earnings state must be known; no filling/dropping")
    frame = full.loc[expected].join(selected)
    _require(np.isfinite(frame[[*ALL_FEATURES, *TARGETS]].to_numpy()).all(), "interior paired features or labels are missing")
    qlib.init(provider_uri=str((root / plan["source"]["provider_uri"]).resolve()), region="us", kernels=1)
    audit = {
        "provider_uri": str((root / plan["source"]["provider_uri"]).resolve()),
        "canonical_readback": canonical_audit, "earnings_readback": earnings_audit,
        "source_deployment_gate": False, "earnings_vintage_is_archived_pit": False,
        "source_limitation": "Vendor historical EPS/estimate snapshot may contain revisions; no contemporaneous estimate vintage archive.",
        "as_of": plan["source"]["as_of"], "paired_rows": len(frame),
        "feature_start": str(expected.min().date()), "feature_end": str(expected.max().date()),
        "native_input_sha256": frame_hash(frame), "earnings_features_sha256": frame_hash(selected),
        "explicit_unmatured_full_calendar_dates": [str(d.date()) for d in immature.index],
        "label_calendar_scope": "Entire canonical calendar before feature window join; 2022 last H5 can mature in early2023.",
        "no_post2022_signal_predictions_or_evaluation": True, "no_feature_row_dropping": True,
        "maximum_study_label_available_at": frame.label_available_at.max().isoformat(),
        "price_vintage_is_archived_pit": False,
    }
    return frame, selected, audit


def build_dataset(frame: pd.DataFrame):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    _require(isinstance(frame.index, pd.DatetimeIndex) and frame.index.is_monotonic_increasing
             and not frame.index.has_duplicates, "ordered unique dataset sessions required")
    _require(not frame.empty and np.isfinite(frame[[*ALL_FEATURES, *TARGETS]].to_numpy()).all(),
             "interior missing paired feature or label rows may not be removed")
    static = pd.concat({"feature": frame[list(ALL_FEATURES)], "label": frame[list(TARGETS)]}, axis=1)
    static.index = pd.MultiIndex.from_arrays([frame.index, np.repeat("SOXX_ENHANCEMENT", len(frame))], names=["datetime", "instrument"])
    handler = DataHandlerLP(instruments=None, start_time=frame.index.min(), end_time=frame.index.max(), data_loader=StaticDataLoader(config=static))
    dataset = DatasetH(handler=handler, segments={"all": (frame.index.min(), frame.index.max())})
    return dataset, {"api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"], "source": "native Qlib D readbacks", "rows": len(frame), "feature_count": 8, "targets": list(TARGETS)}


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
    _require(len(prior) > 20, "insufficient prior sessions for the registered 20-row purge")
    purged = prior.iloc[:-20]
    _require(np.isfinite(purged[[*ALL_FEATURES, *TARGETS]].to_numpy()).all(), "invalid interior training values")
    timing = market.loc[purged.index, ["feature_available_at", "label_available_at"]]
    _require(timing.notna().all().all(), "missing actual feature/label availability timestamp")
    mature = timing.feature_available_at.lt(cutoff) & timing.label_available_at.lt(cutoff)
    train = purged.loc[mature]
    _require(len(train) >= 252, "needs data: fewer than 252 mature training rows")
    return train, {"fit_year": year, "train_rows": len(train), "train_start": train.index.min(), "train_end": train.index.max(),
                   "purged_signal_rows": 20, "first_purged_signal_date": prior.index[-20], "last_purged_signal_date": prior.index[-1],
                   "maturity_excluded_rows": int((~mature).sum()), "cutoff_at": cutoff,
                   "maximum_label_available_at": timing.loc[train.index, "label_available_at"].max(),
                   "maximum_feature_available_at": timing.loc[train.index, "feature_available_at"].max(),
                   "actual_maturity_guard": True, "dataset_prepare_is_fit_source": True}


def fit_predict(dataset, market: pd.DataFrame, *, deadline: str = DEADLINE_UTC):
    """Exactly 7 years x 2 families; each Ridge jointly fits both fixed heads."""
    prepared = prepare_dataset(dataset)
    _require(prepared.index.max() <= pd.Timestamp(FEATURE_END), "post2022 rows must not enter the fitting dataset")
    boundaries = {year: training_rows(prepared, market, year) for year in FIT_YEARS}
    outputs, audits, models = [], [], {}
    for year in FIT_YEARS:
        check_deadline(deadline)
        train, audit = boundaries[year]
        test = prepared.loc[prepared.index.year == year]
        _require(len(test) >= 3, "registered evaluation year has insufficient rows")
        out = market.loc[test.index, ["trade_date", "label_mature_date", *TARGETS]].copy()
        out["fit_year"] = year
        for head in TARGETS:
            out[f"{head}_prior_mean"] = float(train[head].mean())
        for family, columns in FEATURE_FAMILIES.items():
            check_deadline(deadline)
            model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.))])
            model.fit(train[list(columns)], train[list(TARGETS)])
            prediction = model.predict(test[list(columns)])
            _require(prediction.shape == (len(test), 2) and np.isfinite(prediction).all(), "invalid joint Ridge predictions")
            for number, head in enumerate(TARGETS):
                out[f"{head}_{family}"] = prediction[:, number]
            models[f"{year}_{family}"] = model
            audits.append({**audit, "family": family, "feature_count": len(columns), "ridge_alpha": 10.,
                           "target_heads": "cash5,leverage5", "test_rows": len(test), "test_start": test.index.min(), "test_end": test.index.max()})
        outputs.append(out)
    _require(len(models) == 14, "registered fit budget must equal exactly fourteen")
    return pd.concat(outputs).sort_index(), pd.DataFrame(audits), models


def head_sample(predictions: pd.DataFrame, head: str) -> pd.DataFrame:
    _require(head in TARGETS, "unknown target head")
    return predictions[["trade_date", "label_mature_date", head,
                        *[f"{head}_{family}" for family in (*FEATURE_FAMILIES, "prior_mean")]]].rename(
        columns={head: "target", **{f"{head}_{family}": family for family in (*FEATURE_FAMILIES, "prior_mean")}})


def event_hurdle(head: str, cost_bps: float) -> float:
    _require(head in TARGETS, "unknown event head")
    return cash_hurdle(cost_bps) * (1 if head == "cash5" else 2)


def event_metrics(target, prediction, *, head: str, cost_bps: float):
    y, p = np.asarray(target, float), np.asarray(prediction, float)
    _require(y.ndim == 1 and y.shape == p.shape and len(y) > 0 and np.isfinite(np.r_[y, p]).all(), "invalid paired event vectors")
    hurdle = event_hurdle(head, cost_bps)
    mask = p > hurdle
    count = int(mask.sum())
    net = y - hurdle
    return {"head": head, "cost_bps": cost_bps, "log_hurdle": hurdle, "event_count": count,
            "coverage": count / len(y), "mean_event_gross_log_bp": float(10000 * y[mask].mean()) if count else None,
            "mean_event_net_log_bp": float(10000 * net[mask].mean()) if count else None,
            "conditional_mean_undefined_reason": None if count else "no_events",
            "event_scope": "isolated_overlapping_H5_SOXX_roundtrip_not_portfolio_not_compounded_not_annualized"}


def paired_metrics(sample: pd.DataFrame):
    _require(len(sample) >= 3 and np.isfinite(sample[["target", "base", "base_plus_earnings", "prior_mean"]].to_numpy()).all(), "invalid paired metric inputs")
    y, base, full, prior = (sample[name].to_numpy(float) for name in ("target", "base", "base_plus_earnings", "prior_mean"))
    bic, br = correlation(base, y, rank=True)
    fic, fr = correlation(full, y, rank=True)
    bmse, fmse = float(np.mean((base-y)**2)), float(np.mean((full-y)**2))
    return {"rows": len(sample), "signal_start": sample.index.min(), "signal_end": sample.index.max(),
            "trade_start": sample.trade_date.min(), "label_mature_end": sample.label_mature_date.max(),
            "base_ic": bic, "full_ic": fic, "base_ic_undefined_reason": br, "full_ic_undefined_reason": fr,
            "delta_ic_full_minus_base": fic-bic if fic is not None and bic is not None else None,
            "base_mse": bmse, "full_mse": fmse, "mse_reduction_base_minus_full": bmse-fmse,
            "prior_mean_mse": float(np.mean((prior-y)**2)), "mse_reduction_prior_minus_full": float(np.mean((prior-y)**2))-fmse}


def paired_bootstrap(predictions: pd.DataFrame, *, paths=2000, block_length=20, seed=20261005):
    """Share the exact same stationary indices across both heads and all fees."""
    indices = stationary_block_bootstrap_indices(observation_count=len(predictions), path_length=len(predictions), path_count=paths, expected_block_length=block_length, seed=seed)
    summaries, distributions = [], {}
    for head in TARGETS:
        sample = head_sample(predictions, head)
        paired_metrics(sample)
        y, base, full, prior = (sample[name].to_numpy(float) for name in ("target", "base", "base_plus_earnings", "prior_mean"))
        pieces = []
        for start in range(0, paths, 100):
            selected = indices[start:start+100]
            sy, sb, sf, sp = y[selected], base[selected], full[selected], prior[selected]
            bic, fic = _row_rank_correlation(sy, sb), _row_rank_correlation(sy, sf)
            bmse, fmse = np.mean((sb-sy)**2, axis=1), np.mean((sf-sy)**2, axis=1)
            part = pd.DataFrame({"base_ic": bic, "full_ic": fic, "delta_ic_full_minus_base": fic-bic,
                                 "mse_reduction_base_minus_full": bmse-fmse,
                                 "mse_reduction_prior_minus_full": np.mean((sp-sy)**2, axis=1)-fmse})
            for family, forecasts in (("base", sb), ("full", sf), ("prior_mean", sp)):
                for cost in (10, 25):
                    hurdle = event_hurdle(head, cost)
                    mask = forecasts > hurdle
                    count = mask.sum(axis=1)
                    numerator = np.where(mask, 10000*(sy-hurdle), 0.).sum(axis=1)
                    part[f"{family}_conditional_net_event_mean_bp{cost}"] = np.divide(numerator, count, out=np.full(len(count), np.nan), where=count > 0)
            pieces.append(part)
        distribution = pd.concat(pieces, ignore_index=True)
        row = {"slice": POOLED, "head": head, "block_length": block_length, "paths": paths, "seed": seed,
               "rows": len(sample), "paired_same_indices": True, "shared_heads_same_indices": True,
               "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(), "indices_shape": list(indices.shape),
               "head_family_size": 2, "official_gate_lower_quantile": .0125}
        for name in distribution:
            values = distribution[name].to_numpy()
            valid = bool(np.isfinite(values).all())
            row[f"{name}_official_interval_defined"] = valid
            row[f"{name}_undefined_draws"] = int((~np.isfinite(values)).sum())
            for quantile in (.0125, .025, .5, .975, .9875):
                row[f"{name}_q{quantile:g}"] = float(np.quantile(values, quantile)) if valid else np.nan
        summaries.append(row)
        distributions[head] = distribution
    return pd.DataFrame(summaries), distributions, indices


def signal_analysis(predictions: pd.DataFrame):
    points, events, analyzer = [], [], []
    slices = {POOLED: predictions, **{f"year_{year}": predictions.loc[predictions.index.year == year] for year in FIT_YEARS}}
    for slice_name, values in slices.items():
        for head in TARGETS:
            sample = head_sample(values, head)
            points.append({"slice": slice_name, "head": head, **paired_metrics(sample)})
            for family in (*FEATURE_FAMILIES, "prior_mean"):
                for cost in (10, 25):
                    events.append({"slice": slice_name, "family": family, **event_metrics(sample.target, sample[family], head=head, cost_bps=cost)})
                pearson, preason = correlation(sample[family], sample.target)
                spearman, sreason = correlation(sample[family], sample.target, rank=True)
                analyzer.append({"slice": slice_name, "head": head, "family": family, "pearson_ic": pearson, "spearman_ic": spearman,
                                 "pearson_undefined_reason": preason, "spearman_undefined_reason": sreason,
                                 "scope": "single_asset_time_series_equivalent_analyzer_in_native_Recorder"})
    return pd.DataFrame(points), pd.DataFrame(events), pd.DataFrame(analyzer)


def research_gates(points: pd.DataFrame, bootstrap: pd.DataFrame):
    heads = {}
    for head in TARGETS:
        annual = points.loc[points["head"].eq(head) & points["slice"].isin([f"year_{year}" for year in FIT_YEARS])]
        pooled = bootstrap.loc[bootstrap["head"].eq(head) & bootstrap["slice"].eq(POOLED)]
        _require(len(annual) == 7 and annual["slice"].nunique() == 7 and len(pooled) == 2
                 and set(pooled.block_length) == {20, 60} and pooled.paths.eq(2000).all(),
                 "registered gates need all seven years, both blocks, and 2000 draws")
        def positive_lower(name):
            value = pooled[f"{name}_q0.0125"]
            return bool(pooled[f"{name}_official_interval_defined"].eq(True).all() and value.notna().all() and value.gt(0).all())
        counts = {"positive_full_ic_years": int(annual.full_ic.gt(0).sum()),
                  "positive_delta_ic_years": int(annual.delta_ic_full_minus_base.gt(0).sum()),
                  "positive_mse_reduction_years": int(annual.mse_reduction_base_minus_full.gt(0).sum())}
        information = counts["positive_full_ic_years"] >= 5 and positive_lower("full_ic")
        incremental = counts["positive_delta_ic_years"] >= 5 and counts["positive_mse_reduction_years"] >= 5 and positive_lower("delta_ic_full_minus_base") and positive_lower("mse_reduction_base_minus_full")
        economic = all(positive_lower(f"full_conditional_net_event_mean_bp{cost}") for cost in (10, 25))
        heads[head] = {**counts, "evaluated_years": 7, "information_pass": information, "incremental_pass": incremental,
                       "economic_pass": economic, "all_head_gates_pass": bool(information and incremental and economic)}
    all_pass = all(value["all_head_gates_pass"] for value in heads.values())
    return {"heads": heads, "both_heads_information_and_economic_pass": all_pass,
            "source_deployment_gate": False, "promotion_decision": "not_promoted",
            "decision": "needs_archived_pit_estimates_before_any_portfolio" if all_pass else "end_fixed_recipe_no_further_search",
            "fresh_holdout": False, "stable_excess_gate_evaluated": False,
            "economic_scope": "isolated_overlapping_H5_events_not_a_portfolio"}


def verify_registered_plan(plan):
    _require(all((plan.get("experiment") == EXPERIMENT, plan.get("registered_before_forecast_fits") is True,
        plan["features"]["base"] == list(BASE_FEATURES), plan["features"]["base_plus_earnings"] == list(ALL_FEATURES),
        plan["training"]["start"] == TRAIN_START, plan["training"]["end"] == FEATURE_END,
        plan["training"]["cutoff_years"] == list(FIT_YEARS), plan["training"]["fit_budget"] == 14,
        plan["training"]["ridge_alpha"] == 10., plan["training"]["purge_rows"] == 20,
        plan["training"]["minimum_rows"] == 252, plan["labels"]["horizon_sessions"] == 5,
        plan["uncertainty"]["resamples"] == 2000, plan["uncertainty"]["blocks"] == [20, 60],
        plan["uncertainty"]["seed"] == 20261005, plan["uncertainty"]["head_family_size"] == 2,
        plan["uncertainty"]["percentiles"] == [.0125, .025, .5, .975, .9875],
        plan["fees"]["cost_bps"] == [10, 25], plan["gates"]["minimum_positive_years"] == 5,
        plan["gates"]["source_deployment_gate"] is False,
        plan["earnings_source"]["source_deployment_gate"] is False,
        plan["windows"]["post_2022_predictions"] is False)), "registered fixed protocol changed")


def smoke_inputs(root: Path | str):
    root = Path(root).resolve()
    plan = json.loads((root / "research" / EXPERIMENT / "plan.json").read_text())
    verify_registered_plan(plan)
    frame, _, audit = load_native_inputs(root, plan)
    dataset, dataset_audit = build_dataset(frame)
    prepared = prepare_dataset(dataset)
    return json_ready({"data_audit": audit, "native_input_sha256": frame_hash(frame), "qlib_dataset": dataset_audit,
                       "fit_boundaries": [training_rows(prepared, frame, year)[1] for year in FIT_YEARS],
                       "fit_count": 0, "financial_metrics_computed": False})


def verify_execution_seal(root: Path, card: Path, authorization: dict):
    fixed = {"plan_sha256": card / "plan.json", "workflow_sha256": card / "workflow.yaml",
             "implementation_sha256": Path(__file__), "data_implementation_sha256": root / "src/data/earnings_surprise.py"}
    if authorization.get("approved") is not True or any(authorization.get(key) != sha256_file(path) for key, path in fixed.items()):
        raise PermissionError("blocked: root GO execution seal differs from registered code or plan")
    files = authorization.get("source_files", {})
    if not REQUIRED_SEAL_SOURCES.issubset(files) or any(Path(name).is_absolute() or ".." in Path(name).parts or sha256_file(root/name) != digest for name, digest in files.items()):
        raise PermissionError("blocked: root GO source seal differs from reviewed files")


def make_conclusion(summary):
    lines = ["# 财报预期差：SOXX/SOXL/cash 指增信息层诊断", "",
        "本轮仅检验财报信息是否增加现有指增预测能力，没有运行仓位策略或真实订单。固定 NVDA/AMD/INTC 三个历史 EPS surprise 特征，与同样本 BASE5 对照；每年两次双输出 StandardScaler + Ridge(alpha=10)，共14次拟合。训练自2011年起，评价2016–2022七年，20行剔除与真实H5成熟时间门。2023及以后信号日未产生预测或评价；2022末信号的H5结果允许在2023年初成熟，完整保留。", "",
        "**数据源不满足部署门：历史供应商 EPS/一致预期快照缺少当时版本归档，可能有事后修订。保守延迟报告可见时间不能消除这个缺口。三家公司也是固定代理，不代表完整历史半导体成分。NVDA早期EPS约0.004–0.007且仅三位小数，回溯拆股调整与舍入可能改变surprise；本轮保留登记值与公式，不修数或换公司。无论下列指标如何，本轮不晋升。**", ""]
    for head in TARGETS:
        point = next(row for row in summary["key_point_metrics"] if row["slice"] == POOLED and row["head"] == head)
        gate = summary["gates"]["heads"][head]
        def number(value):
            return "未定义" if value is None else f"{value:.6f}"
        lines += [f"- {head}：{point['rows']}个开发期观察，BASE5/加财报 IC={number(point['base_ic'])}/{number(point['full_ic'])}，ΔIC={number(point['delta_ic_full_minus_base'])}，MSE改善(base-full)={number(point['mse_reduction_base_minus_full'])}。全特征IC正/IC增量正/MSE改善年数为{gate['positive_full_ic_years']}/{gate['positive_delta_ic_years']}/{gate['positive_mse_reduction_years']}（各7年）；信息/增量/费用事件门={gate['information_pass']}/{gate['incremental_pass']}/{gate['economic_pass']}。"]
    lines += ["", "cash5预测SOXX转现金的五日相对对数收益；leverage5预测SOXL对SOXX的五日相对对数收益。两头使用同一组20/60日期块、各2000次配对抽样；两头调整后的下1.25%界用于门禁，未定义抽样直接失败；该调整仅覆盖本次两个固定头，不覆盖此前全部研究次数。全部市场历史均已查看，不是新的留出样本。逐年差异见point_metrics.csv。季度状态持续多日，日数不是独立财报事件数，bootstrap亦不能消除历史复用和供应商修订风险。", "",
        "费用事件从已经持有SOXX出发，cash5门槛为h=log((1+c)/(1-c))；leverage5四条股票交易腿为2h。10/25bp均计算base/full/prior。事件互相重叠，只作独立五日费用诊断，不复利、不年化，不是投资组合或稳定超额证明。", "",
        f"结论：`{summary['gates']['decision']}`。source_deployment_gate=False，稳定超额目标未达成。下一步只允许先取得可信的历史时点预期版本，再另立验证；本轮不换符号、期限、特征或门槛继续筛选。即使信息门通过，也须另行接入精确费用结算、SOXX和同平均仓位静态对照及风险门，不能直接部署。", "",
        f"Qlib Recorder：`{summary['recorder_id']}`。大型预测、模型和抽样仅存Recorder；本卡保留审计入口。"]
    return "\n".join(lines) + "\n"


def run_experiment(*, root: Path | str):
    """One root-authorized financial attempt; never automatically refit."""
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
    uri = Path(plan["recorder_uri"]).resolve()
    reference_path, summary_path = card / "recorder_reference.json", card / "summary.json"
    if summary_path.exists():
        cached = json.loads(summary_path.read_text())
        if cached.get("status") == "finished":
            reference = json.loads(reference_path.read_text())
            _require(reference.get("complete") is True and reference["recorder_id"] == cached["recorder_id"], "cached reference mismatch")
            with R.uri_context(uri.as_uri()):
                recorder = R.get_recorder(experiment_name=EXPERIMENT, recorder_id=cached["recorder_id"])
            artifacts = Path(recorder.get_local_dir()) / "artifacts"
            _require(recorder.status == "FINISHED" and bool(reference.get("artifact_sha256")), "cached Recorder incomplete")
            _require(all(sha256_file(artifacts/name) == digest for name, digest in reference["artifact_sha256"].items()), "cached artifact hash mismatch")
            _require(json_ready(recorder.load_object("card/summary.pkl")) == cached, "cached summary differs from Recorder")
            return cached
    if reference_path.exists():
        raise RuntimeError("native Recorder attempt already exists; automatic refit forbidden")
    check_deadline(plan["deadline_utc"])
    frame, earnings, data_audit = load_native_inputs(root, plan)
    if authorization.get("native_input_sha256") != frame_hash(frame):
        raise PermissionError("blocked: native input differs from root-reviewed smoke")
    dataset, dataset_audit = build_dataset(frame)
    prepared = prepare_dataset(dataset)
    for year in FIT_YEARS:
        training_rows(prepared, frame, year)
    started = datetime.now(timezone.utc).isoformat()
    uri.mkdir(parents=True, exist_ok=True)
    with R.start(experiment_name=EXPERIMENT, uri=uri.as_uri()):
        recorder = R.get_recorder()
        reference = {"complete": False, "recorder_id": recorder.id, "recorder_uri": str(uri), "plan_sha256": sha256_file(card/"plan.json"), "started_at_utc": started}
        write_json(reference_path, reference)
        R.log_params(plan_sha256=reference["plan_sha256"], native_input_sha256=frame_hash(frame), fit_budget=14, ridge_alpha=10, source_deployment_gate=False, all_history_reused=True)
        recorder.save_objects(artifact_path="inputs", **{"features_labels.pkl": frame, "earnings_features.pkl": earnings, "plan.pkl": plan, "data_audit.pkl": data_audit, "authorization.pkl": authorization})
        predictions, fits, models = fit_predict(dataset, frame, deadline=plan["deadline_utc"])
        points, events, analyzer = signal_analysis(predictions)
        rows, distributions, indices = [], {}, {}
        for block in (20, 60):
            check_deadline(plan["deadline_utc"])
            stats, draws, sampled = paired_bootstrap(predictions, paths=2000, block_length=block, seed=20261005)
            rows.append(stats)
            distributions[f"block{block}"] = draws
            indices[f"block{block}"] = {"seed": 20261005, "block_length": block, "shape": list(sampled.shape),
                "indices_sha256": hashlib.sha256(sampled.tobytes()).hexdigest(),
                "sampler_source": "src/objective/relative_wealth.py",
                "sampler_source_sha256": sha256_file(root/"src/objective/relative_wealth.py")}
        bootstrap = pd.concat(rows, ignore_index=True)
        gates = research_gates(points, bootstrap)
        verify_execution_seal(root, card, authorization)
        check_deadline(plan["deadline_utc"])
        R.log_metrics(fit_count=len(models), both_heads_information_and_economic_pass=int(gates["both_heads_information_and_economic_pass"]), source_deployment_gate=0)
        recorder.save_objects(artifact_path="models", **{"fitted_models.pkl": models})
        recorder.save_objects(artifact_path="analysis", **{"predictions.pkl": predictions, "labels.pkl": frame[[*TARGETS, "trade_date", "label_mature_date"]], "bootstrap_draws.pkl": distributions, "bootstrap_indices_spec.pkl": indices, "equivalent_signal_analysis.pkl": analyzer})
        for name, table in {"point_metrics": points, "event_metrics": events, "bootstrap_summary": bootstrap, "fit_audit": fits, "signal_analysis": analyzer}.items():
            table.to_csv(card/f"{name}.csv", index=False)
        write_json(card/"source_audit.json", data_audit)
        summary = {"experiment": EXPERIMENT, "status": "finished", "card_type": "vendor_historical_dual_head_information_diagnostic",
            "primary_notebook": f"{EXPERIMENT}.ipynb", "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": reference["plan_sha256"], "implementation_sha256": authorization["implementation_sha256"], "native_input_sha256": frame_hash(frame),
            "fit_count": len(models), "provider_uri": data_audit["provider_uri"], "as_of": plan["source"]["as_of"], "region": "us", "market": "all",
            "data_audit": data_audit, "qlib_dataset": dataset_audit, "recorder_id": recorder.id, "recorder_uri": str(uri),
            "recorder_local_dir": str(Path(recorder.get_local_dir()).resolve()), "recorder_artifact_uri": recorder.get_artifact_uri(), "qlib_docs": QLIB_DOCS,
            "artifact_policy": plan["artifact_policy"], "gates": gates, "source_deployment_gate": False, "promotion_decision": "not_promoted",
            "sidecars": ["plan.json", "workflow.yaml", "run_authorization.json", "source_audit.json", "point_metrics.csv", "event_metrics.csv", "bootstrap_summary.csv", "fit_audit.csv", "signal_analysis.csv", "conclusion.md"],
            "key_point_metrics": json_ready(points.to_dict("records")), "no_portfolio_backtest": True, "stable_excess_gate_evaluated": False,
            "fresh_holdout": False, "post2022_signal_evaluated": False, "august_2026_evaluated": False,
            "earnings_vintage_is_archived_pit": False, "price_vintage_is_archived_pit": False,
            "model_selection": {"performed": False, "families": list(FEATURE_FAMILIES), "fixed_model": "StandardScaler + multioutput Ridge(alpha=10)"},
            "benchmark": {"predictive": ["base5", "training_prior_mean"], "event": "existing SOXX holding on identical next-open H5 intervals"},
            "analysis_api": "single_asset_time_series_equivalent_analyzer_saved_in_native_Qlib_Recorder"}
        (card/"conclusion.md").write_text(make_conclusion(summary), encoding="utf-8")
        for name in summary["sidecars"]:
            recorder.log_artifact(str(card/name), artifact_path="card")
        recorder.save_objects(artifact_path="card", **{"summary.pkl": summary})
    with R.uri_context(uri.as_uri()):
        recorder = R.get_recorder(experiment_name=EXPERIMENT, recorder_id=summary["recorder_id"])
    _require(recorder.status == "FINISHED", "native Recorder did not finish")
    verify_execution_seal(root, card, authorization)
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    for name in ("code_status.txt", "code_diff.txt", "code_cached.txt"):
        (artifacts/name).unlink(missing_ok=True)
    reference.update(complete=True, artifact_sha256={str(path.relative_to(artifacts)): sha256_file(path) for path in sorted(artifacts.rglob("*")) if path.is_file()})
    write_json(summary_path, summary)
    write_json(reference_path, reference)
    return json_ready(summary)
