"""One registered, bounded cash-advantage revalidation on qualified price inputs.

No portfolio, parameter search, promotion, or fresh-out-of-sample claim is made.
All fits consume DatasetH.prepare; all detailed evidence lives in Qlib Recorder.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.objective.relative_wealth import stationary_block_bootstrap_indices
from src.analysis.frozen_cash_information import cash_hurdle, correlation, event_metrics

EXPERIMENT = "soxs_cash5_revalidation_20261003"
TARGET = "cash_advantage_5d"
BASE_FEATURES = (
    "soxx_trend20", "soxx_trend60", "soxx_vol20", "soxx_drawdown60", "relative_momentum20",
)
SOXS_FEATURES = (
    "soxs_trend5", "soxs_trend20", "inverse_tracking_gap1", "inverse_tracking_gap20", "soxl_soxs_path20",
)
ALL_FEATURES = (*BASE_FEATURES, *SOXS_FEATURES)
FEATURE_FAMILIES = {"base": BASE_FEATURES, "base_plus_soxs": ALL_FEATURES}
FIT_YEARS = tuple(range(2016, 2024))
DEADLINE_UTC = "2026-10-03T12:24:00+00:00"

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_deadline(deadline: str = DEADLINE_UTC) -> None:
    if datetime.now(timezone.utc) >= datetime.fromisoformat(deadline):
        raise TimeoutError("bounded revalidation deadline reached; no more computation authorized")


def build_features_and_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Prices must retain every canonical Qlib trading day, including missing values."""
    required = ["soxl_close", "soxx_close", "soxs_close", "soxx_open"]
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_monotonic_increasing:
        raise ValueError("canonical calendar must be a sorted DatetimeIndex")
    if prices.index.has_duplicates or not set(required).issubset(prices):
        raise ValueError("canonical rows must be unique and contain the four required prices")
    values = prices[required].to_numpy(dtype=float)
    if np.any(np.isfinite(values) & (values <= 0)):
        raise ValueError("qualified published prices must be strictly positive")
    frame = prices[required].copy()
    for symbol in ("soxl", "soxx", "soxs"):
        frame[f"{symbol}_close_return"] = frame[f"{symbol}_close"].pct_change(fill_method=None)
    frame["soxx_trend20"] = frame.soxx_close / frame.soxx_close.rolling(20).mean() - 1.0
    frame["soxx_trend60"] = frame.soxx_close / frame.soxx_close.rolling(60).mean() - 1.0
    frame["soxx_vol20"] = frame.soxx_close_return.rolling(20).std(ddof=1)
    frame["soxx_drawdown60"] = frame.soxx_close / frame.soxx_close.rolling(60).max() - 1.0
    frame["relative_momentum20"] = frame.soxl_close.pct_change(20, fill_method=None) - frame.soxx_close.pct_change(20, fill_method=None)
    frame["soxs_trend5"] = frame.soxs_close / frame.soxs_close.rolling(5).mean() - 1.0
    frame["soxs_trend20"] = frame.soxs_close / frame.soxs_close.rolling(20).mean() - 1.0
    frame["inverse_tracking_gap1"] = frame.soxs_close_return + 3.0 * frame.soxx_close_return
    frame["inverse_tracking_gap20"] = frame.inverse_tracking_gap1.rolling(20).std(ddof=1)
    frame["soxl_soxs_path20"] = (
        np.log1p(frame.soxl_close_return.clip(lower=-0.99))
        + np.log1p(frame.soxs_close_return.clip(lower=-0.99))
    ).rolling(20).sum()
    dates = frame.index.to_series()
    frame["trade_date"] = dates.shift(-1)
    frame["label_mature_date"] = dates.shift(-6)
    frame[TARGET] = -np.log(frame.soxx_open.shift(-6) / frame.soxx_open.shift(-1))
    frame["feature_available_at"] = pd.DatetimeIndex(frame.index).tz_localize("America/New_York") + pd.Timedelta(hours=16)
    frame["label_available_at"] = pd.DatetimeIndex(frame.label_mature_date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
    frame.index.name = "signal_date"
    return frame


def load_native_inputs(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    import qlib
    from qlib.data import D
    from src.data.soxs_published_close import load_qualified_soxs_close

    provider = root / "data/qlib/published_futu_affine_20260925_v1"
    manifest_path = provider / "qlib_store_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("blocked: canonical Qlib provider manifest missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "ok" or set(manifest.get("instruments", [])) != {"SOXL", "SOXX"}:
        raise ValueError("blocked: canonical Qlib provider not qualified")
    from src.data.constrained_net_excess import _native_readback
    source_plan = json.loads((root / "research" / EXPERIMENT / "plan.json").read_text())
    raw, calendar, canonical_audit = _native_readback(root, source_plan, {
        "provider_manifest_sha256": "cbbdeeca2e1c7dc0bce0f9cdf3495f50d3219bddd992e1b893f8362a5e51bc65",
        "parent_summary_sha256": "0e444d39dd4e09e5bdcac2d44dd77fa3db356ce735ecdfc80b8f937cf0bf362f",
        "parent_recorder_id": "bd754e6362b241f194351a1e7aebff6f",
        "native_frame_sha256": "b7a59f39feed5aed6e61fec2b4239092c474e453c91d4d133c73f9dcc571a03f",
    })
    frame = pd.DataFrame(index=calendar)
    for symbol, field, name in (("SOXL", "$close", "soxl_close"), ("SOXX", "$close", "soxx_close"), ("SOXX", "$open", "soxx_open")):
        frame[name] = raw.xs(symbol, level="instrument")[field].reindex(calendar)
    close, supplement = load_qualified_soxs_close(root)
    if supplement.get("qualified") is not True:
        raise ValueError("blocked: SOXS supplemental source not qualified")
    if not close.index.equals(calendar):
        raise ValueError("blocked: supplemental SOXS calendar must exactly equal canonical calendar")
    frame["soxs_close"] = close
    if len(calendar) != 4162 or frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("blocked: complete full canonical price history required")
    if (frame <= 0).any().any():
        raise ValueError("blocked: nonpositive qualified prices")
    # Restore the canonical provider as the active Qlib data plane for the notebook.
    qlib.init(provider_uri=str(provider.resolve()), region="us", kernels=1)
    audit = {
        "provider_uri": str(provider.resolve()), "provider_manifest_sha256": sha256_file(manifest_path),
        "region": "us", "market": "all", "as_of": "2026-09-25", "calendar_start": str(calendar.min().date()),
        "calendar_end": str(calendar.max().date()), "calendar_rows": len(calendar),
        "source_api": ["qlib.init", "qlib.data.D.calendar", "qlib.data.D.features"],
        "soxs_supplement": supplement, "canonical_readback": canonical_audit,
        "uniform_price_basis_scope": "published split/distribution affine basis; source limitations remain explicit",
        "price_vintage_limitation": "Retrospective 2026-09-25 adjustment vintage, not archived PIT prices. Additive future cash adjustment B can change historical ratios; fit slicing does not establish complete PIT validation.",
    }
    featured = build_features_and_labels(frame)
    audit["original_clip_bound_rows"] = int(((featured.soxl_close_return < -0.99) | (featured.soxs_close_return < -0.99)).sum())
    if audit["original_clip_bound_rows"]:
        raise ValueError("blocked: original log1p clipping bound; scientific eligibility invalid")
    return featured, audit


def build_dataset(frame: pd.DataFrame):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    eligible = frame.loc["2011-01-01":]
    if not np.isfinite(eligible[list(ALL_FEATURES)].to_numpy()).all():
        raise ValueError("blocked: interior missing feature rows may not be silently removed")
    static = pd.concat({"feature": eligible[list(ALL_FEATURES)], "label": eligible[[TARGET]]}, axis=1)
    static.index = pd.MultiIndex.from_arrays([eligible.index, np.repeat("SOXX_CASH5", len(eligible))], names=["datetime", "instrument"])
    handler = DataHandlerLP(instruments=None, start_time=eligible.index.min(), end_time=eligible.index.max(), data_loader=StaticDataLoader(config=static))
    segments = {"all": (eligible.index.min(), eligible.index.max())}
    dataset = DatasetH(handler=handler, segments=segments)
    return dataset, {"api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"], "source": "native D feature readback", "rows": len(eligible), "train_start": "2011-01-03"}


def prepare_dataset(dataset) -> pd.DataFrame:
    prepared = dataset.prepare("all", col_set=["feature", "label"]).droplevel("instrument").sort_index()
    prepared.columns = prepared.columns.get_level_values(-1)
    return prepared


def training_rows(prepared: pd.DataFrame, market: pd.DataFrame, year: int, *, purge_rows: int = 20, minimum_rows: int = 252) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Purge feature-eligible signal rows, then require actual H5 label maturity."""
    boundary = pd.Timestamp(f"{year}-01-01")
    prior = prepared.loc[(prepared.index >= "2011-01-01") & (prepared.index < boundary)]
    if purge_rows <= 0 or len(prior) <= purge_rows:
        raise ValueError("insufficient rows for the registered purge")
    purged = prior.iloc[:-purge_rows]
    mature = market.loc[purged.index, "label_available_at"]
    features_at = market.loc[purged.index, "feature_available_at"]
    cutoff = boundary.tz_localize("America/New_York")
    eligible = mature.notna() & mature.lt(cutoff) & features_at.lt(cutoff)
    train = purged.loc[eligible]
    if not np.isfinite(train.loc[:, [*ALL_FEATURES, TARGET]].to_numpy()).all():
        raise ValueError("blocked: invalid interior training values")
    if len(train) < minimum_rows:
        raise ValueError("insufficient mature training rows")
    audit = {
        "fit_year": year, "train_rows": len(train), "train_start": train.index.min(), "train_end": train.index.max(),
        "maximum_label_mature_date": market.loc[train.index, "label_mature_date"].max(),
        "maximum_label_available_at": market.loc[train.index, "label_available_at"].max(),
        "maximum_feature_available_at": market.loc[train.index, "feature_available_at"].max(),
        "test_boundary": boundary, "cutoff_at": cutoff, "price_timing": "features at NY 16:00 close; H5 labels at NY 09:30 open", "purged_signal_rows": purge_rows,
        "first_purged_signal_date": prior.index[-purge_rows], "last_purged_signal_date": prior.index[-1],
        "actual_maturity_guard": True, "dataset_prepare_is_fit_source": True,
    }
    return train, audit


def fit_predict(dataset, market: pd.DataFrame, *, deadline: str = DEADLINE_UTC):
    prepared = prepare_dataset(dataset)
    outputs, fit_audits, models = [], [], {}
    for year in FIT_YEARS:
        check_deadline(deadline)
        train, audit = training_rows(prepared, market, year)
        test = prepared.loc[(prepared.index.year == year) if year < 2023 else (prepared.index.year >= 2023)]
        expected_mature = market.loc[test.index, "label_mature_date"].notna()
        test = test.loc[expected_mature]
        if not np.isfinite(test.loc[:, [*ALL_FEATURES, TARGET]].to_numpy()).all():
            raise ValueError("blocked: invalid interior evaluation values")
        if test.empty:
            raise ValueError(f"empty registered test slice: {year}")
        out = market.loc[test.index, ["trade_date", "label_mature_date", TARGET]].copy()
        out["prior_mean"] = float(train[TARGET].mean())
        out["fit_year"] = year
        out["segment"] = "development" if year < 2023 else "reused_2023_plus"
        for family, features in FEATURE_FAMILIES.items():
            check_deadline(deadline)
            model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
            model.fit(train.loc[:, features], train[TARGET])
            out[family] = model.predict(test.loc[:, features])
            models[f"{year}_{family}"] = model
            fit_audits.append({**audit, "family": family, "feature_count": len(features), "alpha": 10.0, "test_rows": len(test)})
        outputs.append(out)
    if len(models) != 16:
        raise AssertionError("registered experiment must perform exactly 16 fits")
    return pd.concat(outputs).sort_index(), pd.DataFrame(fit_audits), models


def safe_ic(actual: np.ndarray, prediction: np.ndarray) -> float:
    if len(actual) < 3 or np.ptp(actual) == 0 or np.ptp(prediction) == 0:
        return float("nan")
    value, _ = correlation(prediction, actual, rank=True)
    return float("nan") if value is None else value


def paired_metrics(sample: pd.DataFrame) -> dict[str, Any]:
    required = sample[[TARGET, "base", "base_plus_soxs"]]
    if len(sample) < 3 or not np.isfinite(required.to_numpy()).all():
        raise ValueError("paired evaluation requires finite, aligned observations")
    y, base, full = (required[column].to_numpy() for column in required)
    base_ic, full_ic = safe_ic(y, base), safe_ic(y, full)
    base_mse, full_mse = float(np.mean((base - y) ** 2)), float(np.mean((full - y) ** 2))
    return {
        "rows": len(sample), "signal_start": sample.index.min(), "signal_end": sample.index.max(),
        "trade_start": sample.trade_date.min(), "label_mature_end": sample.label_mature_date.max(),
        "base_ic": base_ic, "base_plus_soxs_ic": full_ic, "delta_ic_full_minus_base": full_ic - base_ic,
        "base_mse": base_mse, "base_plus_soxs_mse": full_mse, "delta_mse_full_minus_base": full_mse - base_mse,
        "mse_reduction_base_minus_full": base_mse - full_mse,
        "prior_mean_mse": float(np.mean((sample.prior_mean.to_numpy() - y) ** 2)),
        "mse_reduction_prior_mean_minus_full": float(np.mean((sample.prior_mean.to_numpy() - y) ** 2)) - full_mse,
    }


def evaluation_slices(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    slices = {
        "development": predictions.loc[predictions.segment.eq("development")],
        "reused_2023_plus": predictions.loc[predictions.segment.eq("reused_2023_plus")],
        "user_window": predictions.loc[predictions.trade_date.ge("2026-08-03") & predictions.label_mature_date.le("2026-09-25")],
    }
    for year, sample in predictions.groupby(predictions.index.year):
        slices[f"year_{year}"] = sample
    return slices


def _row_rank_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = rankdata(left, axis=1)
    right = rankdata(right, axis=1)
    left -= left.mean(axis=1, keepdims=True)
    right -= right.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.square(left).sum(axis=1) * np.square(right).sum(axis=1))
    return np.divide((left * right).sum(axis=1), denominator, out=np.full(len(left), np.nan), where=denominator > 0)


def paired_bootstrap(sample: pd.DataFrame, *, paths: int = 2000, block_length: int = 20, seed: int = 20261003) -> tuple[dict[str, Any], pd.DataFrame]:
    """Both families/baseline/events use identical chronological stationary draws."""
    paired_metrics(sample)
    y, base, full, prior = (sample[column].to_numpy(dtype=float) for column in (TARGET, "base", "base_plus_soxs", "prior_mean"))
    indices = stationary_block_bootstrap_indices(observation_count=len(sample), path_length=len(sample), path_count=paths, expected_block_length=block_length, seed=seed)
    draws = []
    for start in range(0, paths, 100):
        selected = indices[start:start + 100]
        sy, sb, sf, sp = y[selected], base[selected], full[selected], prior[selected]
        bic, fic = _row_rank_correlation(sy, sb), _row_rank_correlation(sy, sf)
        bmse, fmse = np.mean((sb - sy) ** 2, axis=1), np.mean((sf - sy) ** 2, axis=1)
        block = pd.DataFrame({"base_ic": bic, "base_plus_soxs_ic": fic, "delta_ic_full_minus_base": fic - bic, "delta_mse_full_minus_base": fmse - bmse, "mse_reduction_base_minus_full": bmse - fmse, "mse_reduction_prior_mean_minus_full": np.mean((sp - sy) ** 2, axis=1) - fmse})
        for cost in (10, 25):
            hurdle = cash_hurdle(cost)
            event = sf > hurdle
            count = event.sum(axis=1)
            numerator = np.where(event, 10000.0 * (sy - hurdle), 0.0).sum(axis=1)
            block[f"full_conditional_net_event_mean_bp{cost}"] = np.divide(numerator, count, out=np.full(len(count), np.nan), where=count > 0)
        draws.append(block)
    distribution = pd.concat(draws, ignore_index=True)
    summary: dict[str, Any] = {"block_length": block_length, "paths": paths, "seed": seed, "rows": len(sample), "paired_same_indices": True, "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(), "indices_shape": list(indices.shape), "degenerate_draws": int(distribution.isna().any(axis=1).sum())}
    for column in distribution:
        values = distribution[column].to_numpy()
        defined = bool(np.isfinite(values).all())
        summary[f"{column}_official_interval_defined"] = defined
        summary[f"{column}_undefined_draws"] = int((~np.isfinite(values)).sum())
        for quantile in (0.025, 0.05, 0.50, 0.95, 0.975):
            summary[f"{column}_q{quantile:g}"] = float(np.quantile(values, quantile)) if defined else np.nan
    return summary, distribution


def cost_event_diagnostics(sample: pd.DataFrame, fee_bps: int) -> list[dict[str, Any]]:
    """Independent overlapping H5 cash events; explicitly not portfolio returns."""
    return [{"family": family, **event_metrics(sample[TARGET], sample[family], cost_bps=fee_bps)} for family in (*FEATURE_FAMILIES, "prior_mean")]


def signal_analysis(slices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equivalent single-asset time-series analyzer; persisted in native Recorder.

    Qlib daily cross-sectional SigAnaRecord IC is undefined with one instrument.
    The established project analyzer provides explicit constant-vector handling.
    """
    rows = []
    for name, sample in slices.items():
        for family in FEATURE_FAMILIES:
            pearson, pearson_reason = correlation(sample[family], sample[TARGET])
            spearman, spearman_reason = correlation(sample[family], sample[TARGET], rank=True)
            rows.append({"slice": name, "family": family, "pearson_ic": pearson, "spearman_ic": spearman, "pearson_undefined_reason": pearson_reason, "spearman_undefined_reason": spearman_reason, "api": "src.analysis.frozen_cash_information.correlation", "scope": "single_asset_time_series_equivalent_analyzer_in_qlib_recorder"})
    return pd.DataFrame(rows)


def smoke_inputs(root: Path) -> dict[str, Any]:
    """Real-input and Dataset checks only; deliberately no R.start and no fitting."""
    frame, data_audit = load_native_inputs(root)
    dataset, dataset_audit = build_dataset(frame)
    prepared = prepare_dataset(dataset)
    audits = [training_rows(prepared, frame, year)[1] for year in FIT_YEARS]
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    return json_ready({"data_audit": data_audit, "native_input_sha256": fingerprint, "qlib_dataset": dataset_audit, "fit_boundaries": audits, "fit_count": 0})


def verify_registered_plan(plan: dict[str, Any]) -> None:
    checks = [
        plan.get("experiment") == EXPERIMENT,
        plan.get("registered_before_forecast_fits") is True,
        plan["features"]["base"] == list(BASE_FEATURES),
        plan["features"]["base_plus_soxs"] == list(ALL_FEATURES),
        plan["training"]["cutoff_years"] == list(FIT_YEARS),
        plan["training"]["purge_rows"] == 20,
        plan["training"]["minimum_rows"] == 252,
        plan["uncertainty"]["resamples"] == 2000,
        plan["uncertainty"]["blocks"] == [20, 60],
        plan["uncertainty"]["seed"] == 20261003,
        plan["fees"]["cost_bps"] == [10, 25],
    ]
    if not all(checks):
        raise ValueError("blocked: experiment differs from registered bounded plan")


def research_gates(points: pd.DataFrame, bootstrap: pd.DataFrame) -> dict[str, Any]:
    dev_years = points.loc[points["slice"].isin([f"year_{y}" for y in range(2016, 2023)])]
    positive = int((dev_years.base_plus_soxs_ic > 0).sum())
    if len(dev_years) != 7 or len(bootstrap) != 4:
        raise ValueError("registered gates require seven development years and four bootstrap cases")

    def all_lower_positive(metric: str) -> bool:
        key = f"{metric}_q0.025"
        return bool(bootstrap[key].notna().all() and (bootstrap[key] > 0).all())

    legacy_row = bootstrap.loc[(bootstrap["slice"] == "development") & (bootstrap.block_length == 20)].iloc[0]
    information = positive >= 6 and all_lower_positive("base_plus_soxs_ic")
    incremental = all_lower_positive("delta_ic_full_minus_base") and all_lower_positive("mse_reduction_base_minus_full")
    economic = all(all_lower_positive(f"full_conditional_net_event_mean_bp{cost}") for cost in (10, 25))
    passed = bool(information and incremental and economic)
    return {
        "positive_development_years": positive, "development_years": 7,
        "legacy_compatibility_diagnostic": bool(positive >= 5 and legacy_row["base_plus_soxs_ic_q0.05"] > 0),
        "legacy_scope": "different bootstrap budget and seed; not exact old bootstrap reproduction",
        "information_pass": bool(information), "incremental_pass": bool(incremental), "economic_pass": bool(economic),
        "all_research_gates_pass": passed,
        "decision": "further_research_only_no_promotion" if passed else "end_route_no_further_search",
        "promotion_decision": "not_promoted", "fresh_holdout": False,
    }


def make_conclusion(summary: dict[str, Any]) -> str:
    gates = summary["gates"]
    points = {row["slice"]: row for row in summary["key_point_metrics"]}
    dev, reused, user = (points[name] for name in ("development", "reused_2023_plus", "user_window"))
    def fmt(value):
        return "未定义" if value is None else f"{value:.6f}"
    return f"""# 最终中文结论

本轮只复验此前看过的 `cash_advantage_5d` 假设，使用统一说明的复权口径、固定 5/10 特征与 StandardScaler + Ridge(alpha=10)，共 {summary['fit_count']} 次拟合；没有搜索新参数或运行投资组合。数据截至 2026-09-25，先经 Qlib D 读取，再经 DatasetH.prepare 进入训练，预测与不确定性明细在 Qlib Recorder `{summary['recorder_id']}`。

开发期 2016—2022 共 {dev['rows']} 行，base / 加 SOXS 的 IC 为 {fmt(dev['base_ic'])} / {fmt(dev['base_plus_soxs_ic'])}，增量 IC {fmt(dev['delta_ic_full_minus_base'])}；加 SOXS 有 {gates['positive_development_years']}/7 年 IC 为正。2023 年冻结模型在已复用历史区间共 {reused['rows']} 行，base / 加 SOXS IC 为 {fmt(reused['base_ic'])} / {fmt(reused['base_plus_soxs_ic'])}，增量 IC {fmt(reused['delta_ic_full_minus_base'])}。用户窗口按入场不早于 2026-08-03、H5 退出不晚于 2026-09-25 开盘计算，共 {user['rows']} 行，增量 IC {fmt(user['delta_ic_full_minus_base'])}；这是 H5 样本，不能沿用 H1 的行数。年间差异见 `point_metrics.csv`，本轮没有另做 regime 搜索。

预注册的信息 / 增量 / 经济门分别为 {gates['information_pass']} / {gates['incremental_pass']} / {gates['economic_pass']}；总判定 `{gates['decision']}`。旧 5/7 正年与 p05 规则兼容诊断为 {gates['legacy_compatibility_diagnostic']}，仅作兼容比较，不承担晋升。双窗口、20/60 日块、各 2,000 次同索引配对自助结果见 `bootstrap_summary.csv`；任一未定义抽样导致该区间未定义并失败。10/25bp 成本诊断采用卖出系数 1-c、买入资金系数 1/(1+c)，仅评价相互重叠的独立 SOXX→现金→SOXX 事件，不复利、不年化、不视作投组收益。

不晋升：全部历史均已观察，选择该假设本身受旧阳性结果影响，不能当作新样本外证据。2023 年模型修复了旧外层训练的年末标签穿越，使用同样 20 行剔除及 H5 成熟日期严格早于年界；因此不声称精确重现旧外层数值。三序列是回溯至 2026-09-25 的复权版本，未来分红的加法 B 可改变历史比率；训练切片和标签门不能证明完整的历史时点复权可得性。SOXS 早期 709 行依赖第二来源，并补充 Futu 缺失的早期事件；详见 `source_audit.json`，不等同于 SDK QFQ 或生产级权威数据。

下一步：{('只保留复验结果，任何后续研究需另立预注册；本轮不延长搜索。' if gates['all_research_gates_pass'] else '结束该路线，保留失败证据；本轮不增加特征、调整门槛或继续搜索。')}
"""


def verify_execution_seal(root: Path, card: Path, authorization: dict[str, Any]) -> None:
    checks = {
        "plan_sha256": card / "plan.json",
        "implementation_sha256": Path(__file__),
        "workflow_sha256": card / "workflow.yaml",
        "source_manifest_sha256": root / "data/qlib/soxs_published_close_20260925_v1/soxs_close_manifest.json",
        "data_implementation_sha256": root / "src/data/soxs_published_close.py",
    }
    if authorization.get("approved") is not True or any(authorization.get(key) != sha256_file(path) for key, path in checks.items()):
        raise PermissionError("blocked: execution seal no longer matches plan, workflow, implementation, or source")


def run_revalidation(*, root: Path | str) -> dict[str, Any]:
    """Execute once after independent review, or reopen the completed Recorder."""
    from qlib.workflow import R

    root = Path(root).resolve()
    card = root / "research" / EXPERIMENT
    plan_path = card / "plan.json"
    plan = json.loads(plan_path.read_text())
    verify_registered_plan(plan)
    plan_hash = sha256_file(plan_path)
    summary_path = card / "summary.json"
    if summary_path.exists():
        cached = json.loads(summary_path.read_text())
        if cached.get("status") == "finished":
            if cached["plan_sha256"] != plan_hash:
                raise ValueError("completed run plan has changed; automatic rerun forbidden")
            reference = json.loads((card / "recorder_reference.json").read_text())
            if reference.get("complete") is not True or reference.get("recorder_id") != cached["recorder_id"]:
                raise ValueError("completed summary lacks a complete matching recorder reference")
            with R.uri_context(Path(plan["recorder_uri"]).as_uri()):
                recorder = R.get_recorder(recorder_id=cached["recorder_id"], experiment_name=EXPERIMENT)
            if recorder.status != "FINISHED":
                raise ValueError("cached summary has no finished native recorder")
            artifact_root = Path(recorder.get_local_dir()) / "artifacts"
            if not reference.get("artifact_sha256") or any(sha256_file(artifact_root / name) != digest for name, digest in reference["artifact_sha256"].items()):
                raise ValueError("cached Recorder evidence differs from its completion hashes")
            if json_ready(recorder.load_object("card/summary.pkl")) != cached:
                raise ValueError("local summary differs from bound Recorder summary")
            verify_execution_seal(root, card, json.loads((card / "run_authorization.json").read_text()))
            return cached
    if (card / "recorder_reference.json").exists():
        raise RuntimeError("a Recorder attempt already exists; automatic fit rerun is forbidden")
    authorization_path = card / "run_authorization.json"
    if not authorization_path.exists():
        raise PermissionError("blocked: independent source/code review and root GO required before fitting")
    authorization = json.loads(authorization_path.read_text())
    verify_execution_seal(root, card, authorization)
    check_deadline(plan["deadline_utc"])
    started = datetime.now(timezone.utc).isoformat()
    frame, data_audit = load_native_inputs(root)
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    if authorization.get("native_input_sha256") != fingerprint:
        raise PermissionError("blocked: real native input differs from independently reviewed smoke")
    data_audit["native_input_sha256"] = fingerprint
    dataset, dataset_audit = build_dataset(frame)
    # Fail closed before R.start on any missing source, malformed rows, or train boundary.
    prepared = prepare_dataset(dataset)
    for year in FIT_YEARS:
        training_rows(prepared, frame, year)
    check_deadline(plan["deadline_utc"])
    recorder_uri = Path(plan["recorder_uri"]).resolve()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    with R.start(experiment_name=EXPERIMENT, uri=recorder_uri.as_uri()):
        recorder = R.get_recorder()
        reference = {"complete": False, "recorder_id": recorder.id, "recorder_uri": str(recorder_uri), "plan_sha256": plan_hash, "started_at_utc": started}
        write_json(card / "recorder_reference.json", reference)
        R.log_params(plan_sha256=plan_hash, implementation_sha256=authorization["implementation_sha256"], target=TARGET, fit_budget=16, ridge_alpha=10, all_history_reused=True, source_as_of="2026-09-25", native_input_sha256=fingerprint)
        recorder.save_objects(artifact_path="inputs", **{"features_labels.pkl": frame, "plan.pkl": plan, "data_audit.pkl": data_audit, "authorization.pkl": authorization})
        predictions, fits, models = fit_predict(dataset, frame, deadline=plan["deadline_utc"])
        slices = evaluation_slices(predictions)
        points = pd.DataFrame([{"slice": name, **paired_metrics(sample)} for name, sample in slices.items()])
        events = pd.DataFrame([{"slice": name, **row} for name, sample in slices.items() for cost in (10, 25) for row in cost_event_diagnostics(sample, cost)])
        analyzer = signal_analysis(slices)
        bootstrap_rows, distributions = [], {}
        for name in ("development", "reused_2023_plus"):
            for block in plan["uncertainty"]["blocks"]:
                check_deadline(plan["deadline_utc"])
                stats, draws = paired_bootstrap(slices[name], paths=plan["uncertainty"]["resamples"], block_length=block, seed=plan["uncertainty"]["seed"])
                bootstrap_rows.append({"slice": name, **stats})
                distributions[f"{name}_block{block}"] = draws
        bootstrap = pd.DataFrame(bootstrap_rows)
        gates = research_gates(points, bootstrap)
        verify_execution_seal(root, card, authorization)
        check_deadline(plan["deadline_utc"])
        R.log_metrics(fit_count=len(models), development_full_ic=float(points.loc[points["slice"] == "development", "base_plus_soxs_ic"].iloc[0]), positive_development_years=gates["positive_development_years"], all_research_gates_pass=int(gates["all_research_gates_pass"]))
        recorder.save_objects(artifact_path="models", **{"fitted_models.pkl": models})
        recorder.save_objects(artifact_path="analysis", **{"predictions.pkl": predictions, "labels.pkl": frame[[TARGET, "trade_date", "label_mature_date"]], "bootstrap_draws.pkl": distributions, "equivalent_signal_analysis.pkl": analyzer})
        for name, table in {"point_metrics": points, "event_metrics": events, "bootstrap_summary": bootstrap, "fit_audit": fits, "signal_analysis": analyzer}.items():
            table.to_csv(card / f"{name}.csv", index=False)
        write_json(card / "source_audit.json", data_audit)
        summary = {
            "experiment": EXPERIMENT, "status": "finished", "card_type": "registered_reused_history_revalidation",
            "primary_notebook": f"{EXPERIMENT}.ipynb", "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": plan_hash, "implementation_sha256": authorization["implementation_sha256"], "native_input_sha256": fingerprint,
            "fit_count": len(models), "provider_uri": data_audit["provider_uri"], "as_of": "2026-09-25", "region": "us", "market": "all",
            "data_audit": data_audit, "qlib_dataset": dataset_audit, "recorder_id": recorder.id, "recorder_uri": str(recorder_uri),
            "recorder_local_dir": str(Path(recorder.get_local_dir()).resolve()), "recorder_artifact_uri": recorder.get_artifact_uri(),
            "qlib_docs": ["https://qlib.readthedocs.io/en/latest/component/data.html", "https://qlib.readthedocs.io/en/latest/component/recorder.html"],
            "analysis_api": "src.analysis.frozen_cash_information equivalent single-asset time-series analyzer recorded in Qlib",
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml", "recorder_only_artifacts": ["inputs/features_labels.pkl", "models/fitted_models.pkl", "analysis/predictions.pkl", "analysis/labels.pkl", "analysis/bootstrap_draws.pkl", "analysis/equivalent_signal_analysis.pkl"]},
            "sidecars": ["plan.json", "run_authorization.json", "workflow.yaml", "point_metrics.csv", "event_metrics.csv", "bootstrap_summary.csv", "fit_audit.csv", "signal_analysis.csv", "source_audit.json", "conclusion.md"],
            "gates": gates, "promotion_decision": "not_promoted", "selection_uses_reused_history": True, "fresh_holdout": False,
            "model_selection": {"performed": False, "families": list(FEATURE_FAMILIES), "fixed_model": "StandardScaler + Ridge(alpha=10)"},
            "benchmark": {"predictive": ["base5", "training_prior_mean"], "event": "SOXX buy-and-hold over identical open[t+1]..open[t+6] event"},
            "key_point_metrics": json_ready(points.loc[points["slice"].isin(["development", "reused_2023_plus", "user_window"])].to_dict("records")),
            "no_portfolio_backtest": True, "price_vintage_is_archived_pit": False,
        }
        conclusion = make_conclusion(summary)
        (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
        for name in summary["sidecars"]:
            recorder.log_artifact(str(card / name), artifact_path="card")
        recorder.save_objects(artifact_path="card", **{"summary.pkl": summary})
    # Completion is only published after native R ended; partial attempts never refit.
    with R.uri_context(recorder_uri.as_uri()):
        finished_recorder = R.get_recorder(recorder_id=summary["recorder_id"], experiment_name=EXPERIMENT)
    if finished_recorder.status != "FINISHED":
        raise RuntimeError("native Recorder failed to finish")
    verify_execution_seal(root, card, authorization)
    check_deadline(plan["deadline_utc"])
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    for name in ("code_status.txt", "code_diff.txt", "code_cached.txt"):
        (artifacts / name).unlink(missing_ok=True)
    reference.update(complete=True, artifact_sha256={str(path.relative_to(artifacts)): sha256_file(path) for path in sorted(artifacts.rglob("*")) if path.is_file()})
    write_json(summary_path, summary)
    write_json(card / "recorder_reference.json", reference)
    return json_ready(summary)
