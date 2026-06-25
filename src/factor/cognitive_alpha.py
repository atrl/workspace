from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COGALPHA_AGENT_LEVELS: tuple[dict[str, Any], ...] = (
    {
        "level": "market_structure_cycle",
        "agents": ("AgentMarketCycle", "AgentVolatilityRegime"),
        "theme": "large-scale trend, market phase, and volatility state",
    },
    {
        "level": "extreme_risk_fragility",
        "agents": ("AgentTailRisk", "AgentCrashPredictor"),
        "theme": "tail-risk exposure, crash precursors, and fragility accumulation",
    },
    {
        "level": "price_volume_dynamics",
        "agents": ("AgentLiquidity", "AgentOrderImbalance", "AgentPriceVolumeCoherence", "AgentVolumeStructure"),
        "theme": "price-volume pressure, liquidity, and order-imbalance proxies",
    },
    {
        "level": "price_volatility_behavior",
        "agents": ("AgentDailyTrend", "AgentReversal", "AgentRangeVol", "AgentLagResponse", "AgentVolAsymmetry"),
        "theme": "trend persistence, reversal, range volatility, and asymmetric price behavior",
    },
    {
        "level": "multi_scale_complexity",
        "agents": ("AgentDrawdown", "AgentFractal"),
        "theme": "drawdown-recovery geometry, long memory, and roughness",
    },
    {
        "level": "stability_regime_gating",
        "agents": ("AgentRegimeGating", "AgentStability"),
        "theme": "temporal stability and adaptive regime gating",
    },
    {
        "level": "geometric_fusion",
        "agents": ("AgentBarShape", "AgentCreative", "AgentComposite", "AgentHerding"),
        "theme": "bar geometry, factor fusion, and composite signal interaction",
    },
)

_FUTURE_PATTERNS = (
    re.compile(r"\.shift\(\s*-\d+"),
    re.compile(r"Ref\([^)]*,\s*-\d+"),
)


@dataclass(frozen=True)
class CognitiveAlphaCandidate:
    name: str
    level: str
    agent: str
    theme: str
    required_columns: tuple[str, ...]
    lookback: int
    function: Callable[[pd.DataFrame], pd.Series]
    economic_rationale: str
    code: str = ""
    declared_lookahead: int = 0


@dataclass(frozen=True)
class CognitiveAlphaReport:
    metrics: pd.DataFrame
    correlation: pd.DataFrame
    feature_frame: pd.DataFrame
    selected_feature_names: list[str]
    summary: dict[str, Any]

    def write_sidecars(self, directory: Path, *, prefix: str = "cognitive_alpha") -> dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "metrics": directory / f"{prefix}_metrics.csv",
            "correlation": directory / f"{prefix}_correlation.csv",
            "features": directory / f"{prefix}_selected_features.csv",
            "summary": directory / f"{prefix}_summary.json",
        }
        self.metrics.to_csv(paths["metrics"], index=False)
        self.correlation.to_csv(paths["correlation"])
        self.feature_frame[self.selected_feature_names].to_csv(paths["features"], index=True)
        paths["summary"].write_text(json.dumps(self.summary, indent=2, sort_keys=True), encoding="utf-8")
        return paths


def _safe_divide(numerator: pd.Series, denominator: pd.Series | float, eps: float = 1e-12) -> pd.Series:
    return numerator.astype("float64") / (pd.Series(denominator, index=numerator.index).astype("float64") + eps)


def _rolling_zscore(series: pd.Series, window: int, *, min_periods: int | None = None) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    min_obs = min_periods or max(5, min(window, 20))
    mean = numeric.rolling(window, min_periods=min_obs).mean()
    std = numeric.rolling(window, min_periods=min_obs).std(ddof=0)
    return (numeric - mean) / std.replace(0.0, np.nan)


def _factor_pressure_drawdown_fisher_10d(frame: pd.DataFrame) -> pd.Series:
    close = frame["close"].astype("float64")
    pressure = (close - frame["open"].astype("float64")).abs() * np.log1p(frame["volume"].astype("float64"))
    pressure_med10 = pressure.rolling(10, min_periods=10).median()
    rolling_max = close.rolling(252, min_periods=20).max()
    drawdown = -((close - rolling_max) / (rolling_max + 1e-12))
    decayed_drawdown = drawdown.ewm(span=8, adjust=False).mean()
    log_return = np.log(close + 1e-12).diff()
    log_volume = np.log(frame["volume"].astype("float64") + 1e-12).diff()
    corr20 = log_return.rolling(20, min_periods=20).corr(log_volume)
    fisher_corr = np.arctanh(corr20.clip(-0.999, 0.999))
    out = pressure_med10 * fisher_corr * np.tanh(decayed_drawdown)
    return out.rename("factor_pressure_drawdown_fisher_10d")


def _factor_volume_confirmed_trend_20d(frame: pd.DataFrame) -> pd.Series:
    close = frame["close"].astype("float64")
    trend20 = close / close.shift(20) - 1.0
    volume_ratio = frame["volume"].astype("float64") / frame["volume"].rolling(20, min_periods=10).mean() - 1.0
    out = trend20 * np.tanh(volume_ratio)
    return out.rename("factor_volume_confirmed_trend_20d")


def _factor_range_volume_liquidity_stress_20d(frame: pd.DataFrame) -> pd.Series:
    high = frame["high"].astype("float64")
    low = frame["low"].astype("float64")
    close = frame["close"].astype("float64")
    range_pct = (high - low) / (close + 1e-12)
    volume_z = _rolling_zscore(np.log1p(frame["volume"].astype("float64")), 20, min_periods=10)
    out = range_pct.rolling(20, min_periods=10).mean() * (-volume_z)
    return out.rename("factor_range_volume_liquidity_stress_20d")


def _factor_downside_vol_asymmetry_20d(frame: pd.DataFrame) -> pd.Series:
    close = frame["close"].astype("float64")
    returns = close.pct_change()
    downside = returns.where(returns < 0.0, 0.0).rolling(20, min_periods=10).std(ddof=0)
    upside = returns.where(returns > 0.0, 0.0).rolling(20, min_periods=10).std(ddof=0)
    out = downside - upside
    return out.rename("factor_downside_vol_asymmetry_20d")


def _factor_failed_breakout_pressure_20d(frame: pd.DataFrame) -> pd.Series:
    high = frame["high"].astype("float64")
    low = frame["low"].astype("float64")
    close = frame["close"].astype("float64")
    rolling_high = high.rolling(20, min_periods=10).max()
    breakout_attempt = (high / rolling_high.shift(1) - 1.0).clip(lower=0.0)
    close_location = (close - low) / (high - low + 1e-12)
    out = breakout_attempt * (0.5 - close_location)
    return out.rename("factor_failed_breakout_pressure_20d")


def _factor_drawdown_recovery_velocity_20d(frame: pd.DataFrame) -> pd.Series:
    close = frame["close"].astype("float64")
    drawdown20 = close / close.rolling(20, min_periods=10).max() - 1.0
    recovery5 = close / close.shift(5) - 1.0
    out = recovery5 / (drawdown20.abs() + 0.01)
    return out.rename("factor_drawdown_recovery_velocity_20d")


def _factor_bar_shape_reversal_5d(frame: pd.DataFrame) -> pd.Series:
    high = frame["high"].astype("float64")
    low = frame["low"].astype("float64")
    close = frame["close"].astype("float64")
    open_ = frame["open"].astype("float64")
    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low
    body = (close - open_) / (high - low + 1e-12)
    out = (lower_shadow - upper_shadow) / (high - low + 1e-12) - body.rolling(5, min_periods=3).mean()
    return out.rename("factor_bar_shape_reversal_5d")


def _factor_volatility_regime_compression_20_60d(frame: pd.DataFrame) -> pd.Series:
    returns = frame["close"].astype("float64").pct_change()
    hv20 = returns.rolling(20, min_periods=10).std(ddof=0)
    hv60 = returns.rolling(60, min_periods=20).std(ddof=0)
    out = 1.0 - hv20 / (hv60 + 1e-12)
    return out.rename("factor_volatility_regime_compression_20_60d")


def build_cognitive_alpha_seed_candidates() -> list[CognitiveAlphaCandidate]:
    required = ("open", "high", "low", "close", "volume")
    return [
        CognitiveAlphaCandidate(
            name="factor_pressure_drawdown_fisher_10d",
            level="multi_scale_complexity",
            agent="AgentDrawdown",
            theme="pressure drawdown and return-volume coupling",
            required_columns=required,
            lookback=252,
            code="pressure_med10 * fisher_corr20(log_return, log_volume) * tanh(drawdown_ema8)",
            economic_rationale="Measures whether price-volume pressure is amplified by drawdown fragility.",
            function=_factor_pressure_drawdown_fisher_10d,
        ),
        CognitiveAlphaCandidate(
            name="factor_volume_confirmed_trend_20d",
            level="price_volume_dynamics",
            agent="AgentPriceVolumeCoherence",
            theme="volume-confirmed trend persistence",
            required_columns=required,
            lookback=20,
            code="ret20 * tanh(volume / mean(volume,20) - 1)",
            economic_rationale="Trend is more credible when participation confirms the price move.",
            function=_factor_volume_confirmed_trend_20d,
        ),
        CognitiveAlphaCandidate(
            name="factor_range_volume_liquidity_stress_20d",
            level="price_volume_dynamics",
            agent="AgentLiquidity",
            theme="range per liquidity stress",
            required_columns=required,
            lookback=20,
            code="mean((high-low)/close,20) * -zscore(log(volume),20)",
            economic_rationale="Wide ranges on weak volume can flag unstable liquidity conditions.",
            function=_factor_range_volume_liquidity_stress_20d,
        ),
        CognitiveAlphaCandidate(
            name="factor_downside_vol_asymmetry_20d",
            level="price_volatility_behavior",
            agent="AgentVolAsymmetry",
            theme="downside versus upside volatility asymmetry",
            required_columns=required,
            lookback=20,
            code="std(negative_returns,20) - std(positive_returns,20)",
            economic_rationale="Downside volatility dominance can mark unstable regimes for leveraged ETFs.",
            function=_factor_downside_vol_asymmetry_20d,
        ),
        CognitiveAlphaCandidate(
            name="factor_failed_breakout_pressure_20d",
            level="price_volatility_behavior",
            agent="AgentReversal",
            theme="failed breakout reversal pressure",
            required_columns=required,
            lookback=20,
            code="positive(high / ref(rolling_high20,1) - 1) * (0.5 - close_location)",
            economic_rationale="A breakout attempt that closes weakly can indicate exhaustion.",
            function=_factor_failed_breakout_pressure_20d,
        ),
        CognitiveAlphaCandidate(
            name="factor_drawdown_recovery_velocity_20d",
            level="multi_scale_complexity",
            agent="AgentDrawdown",
            theme="drawdown recovery velocity",
            required_columns=required,
            lookback=20,
            code="ret5 / (abs(drawdown20) + 0.01)",
            economic_rationale="Recovery speed relative to drawdown depth separates resilient from fragile rallies.",
            function=_factor_drawdown_recovery_velocity_20d,
        ),
        CognitiveAlphaCandidate(
            name="factor_bar_shape_reversal_5d",
            level="geometric_fusion",
            agent="AgentBarShape",
            theme="candlestick shadow reversal geometry",
            required_columns=required,
            lookback=5,
            code="(lower_shadow-upper_shadow)/range - mean(body,5)",
            economic_rationale="Body and shadow geometry summarizes intraday rejection pressure.",
            function=_factor_bar_shape_reversal_5d,
        ),
        CognitiveAlphaCandidate(
            name="factor_volatility_regime_compression_20_60d",
            level="stability_regime_gating",
            agent="AgentRegimeGating",
            theme="volatility compression versus long regime",
            required_columns=required,
            lookback=60,
            code="1 - hv20 / hv60",
            economic_rationale="Compression can precede trend expansion and helps gate exposure.",
            function=_factor_volatility_regime_compression_20_60d,
        ),
    ]


def build_column_alpha_candidates(
    columns: list[str],
    *,
    level: str,
    agent: str,
    lookback: int = 1,
) -> list[CognitiveAlphaCandidate]:
    candidates: list[CognitiveAlphaCandidate] = []
    for column in columns:
        candidates.append(
            CognitiveAlphaCandidate(
                name=str(column),
                level=level,
                agent=agent,
                theme="existing column alpha",
                required_columns=(str(column),),
                lookback=lookback,
                code=f"df[{column!r}]",
                economic_rationale=f"Existing qlib feature column {column} evaluated through CogAlpha metrics.",
                function=lambda frame, col=str(column): frame[col],
            )
        )
    return candidates


def materialize_cognitive_alpha_features(
    frame: pd.DataFrame,
    candidates: list[CognitiveAlphaCandidate],
    *,
    selected_names: list[str] | None = None,
) -> pd.DataFrame:
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    names = selected_names or [candidate.name for candidate in candidates]
    missing = [name for name in names if name not in candidate_by_name]
    if missing:
        raise ValueError(f"selected cognitive alpha candidates are missing: {missing}")
    values: dict[str, pd.Series] = {}
    for name in names:
        candidate = candidate_by_name[name]
        reason = _reject_reason(candidate, frame)
        if reason is not None:
            raise ValueError(f"cannot materialize rejected candidate {name}: {reason}")
        series = candidate.function(frame)
        if not isinstance(series, pd.Series):
            raise ValueError(f"candidate {name} did not return a pandas Series")
        values[name] = pd.to_numeric(series.reindex(frame.index), errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(values, index=frame.index)


def _reject_reason(candidate: CognitiveAlphaCandidate, frame: pd.DataFrame) -> str | None:
    if candidate.declared_lookahead > 0:
        return "future_lookahead_declared"
    if any(pattern.search(candidate.code or "") for pattern in _FUTURE_PATTERNS):
        return "future_lookahead_code_pattern"
    missing = [column for column in candidate.required_columns if column not in frame.columns]
    if missing:
        return "missing_required_columns:" + ",".join(missing)
    return None


def _compute_candidate_feature(
    frame: pd.DataFrame,
    candidate: CognitiveAlphaCandidate,
    *,
    max_missing_ratio: float,
    min_distinct_values: int,
) -> tuple[pd.Series | None, str, str]:
    reason = _reject_reason(candidate, frame)
    if reason is not None:
        return None, "reject", reason
    try:
        series = candidate.function(frame)
    except Exception as exc:  # pragma: no cover - exact candidate exceptions are data dependent
        return None, "reject", f"runtime_error:{type(exc).__name__}"
    if not isinstance(series, pd.Series):
        return None, "reject", "output_not_series"
    series = pd.to_numeric(series.reindex(frame.index), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if float(series.isna().mean()) > max_missing_ratio:
        return series, "reject", "too_many_missing_values"
    if int(series.dropna().nunique()) < min_distinct_values:
        return series, "reject", "too_few_distinct_values"
    return series.rename(candidate.name), "pass", "pass"


def _aligned(feature: pd.Series, label: pd.Series) -> pd.DataFrame:
    frame = pd.concat(
        [
            pd.to_numeric(feature, errors="coerce").rename("feature"),
            pd.to_numeric(label, errors="coerce").rename("label"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    return frame.dropna()


def _corr_metrics(feature: pd.Series, label: pd.Series, *, min_observations: int) -> dict[str, float]:
    clean = _aligned(feature, label)
    if len(clean) < min_observations or clean["feature"].nunique() <= 1 or clean["label"].nunique() <= 1:
        return {"ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0, "mutual_information": 0.0}
    ic = float(clean["feature"].corr(clean["label"], method="pearson"))
    rank_ic = float(clean["feature"].corr(clean["label"], method="spearman"))
    rolling_window = max(20, min(63, int(len(clean) // 3)))
    rolling_ic = clean["feature"].rolling(rolling_window, min_periods=max(10, rolling_window // 2)).corr(clean["label"])
    rolling_rank_ic = (
        clean["feature"]
        .rank(method="average")
        .rolling(rolling_window, min_periods=max(10, rolling_window // 2))
        .corr(clean["label"].rank(method="average"))
    )
    icir = _information_ratio(rolling_ic.dropna())
    rank_icir = _information_ratio(rolling_rank_ic.dropna())
    return {
        "ic": round(float(ic), 10),
        "rank_ic": round(float(rank_ic), 10),
        "icir": round(float(icir), 10),
        "rank_icir": round(float(rank_icir), 10),
        "mutual_information": round(float(_mutual_information(clean["feature"], clean["label"])), 10),
    }


def _information_ratio(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        return math.copysign(1e6, mean) if abs(mean) > 1e-12 else 0.0
    return mean / std


def _mutual_information(feature: pd.Series, label: pd.Series) -> float:
    try:
        from sklearn.feature_selection import mutual_info_regression

        values = mutual_info_regression(
            feature.to_numpy(dtype="float64").reshape(-1, 1),
            label.to_numpy(dtype="float64"),
            random_state=42,
        )
        return max(float(values[0]), 0.0)
    except Exception:
        corr = feature.corr(label, method="spearman")
        return 0.0 if pd.isna(corr) else abs(float(corr))


def _percentile_rank(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    return values.rank(method="average", pct=True)


def _apply_percentile_tiers(
    metrics: pd.DataFrame,
    *,
    qualified_percentile: float,
    elite_percentile: float,
) -> pd.DataFrame:
    out = metrics.copy()
    metric_cols = ["abs_ic", "abs_rank_ic", "abs_icir", "abs_rank_icir", "mutual_information"]
    pass_mask = out["quality_status"] == "pass"
    for column in metric_cols:
        percentile_col = f"{column}_percentile"
        out[percentile_col] = 0.0
        out.loc[pass_mask, percentile_col] = _percentile_rank(out.loc[pass_mask, column]).astype("float64")
    percentile_cols = [f"{column}_percentile" for column in metric_cols]
    out["min_metric_percentile"] = out[percentile_cols].min(axis=1)
    out["selection_score"] = out[percentile_cols].mean(axis=1)
    out["selection_tier"] = "drop"
    out.loc[pass_mask & (out["min_metric_percentile"] >= qualified_percentile), "selection_tier"] = "qualified"
    out.loc[pass_mask & (out["min_metric_percentile"] >= elite_percentile), "selection_tier"] = "elite"
    return out


def _apply_redundancy(metrics: pd.DataFrame, correlation: pd.DataFrame, *, max_abs_correlation: float) -> pd.DataFrame:
    out = metrics.copy()
    out["redundancy_status"] = "not_selected"
    ordered = out.loc[out["selection_tier"].isin(["elite", "qualified"])].sort_values(
        ["selection_score", "min_metric_percentile"],
        ascending=[False, False],
    )
    kept: list[str] = []
    for idx, row in ordered.iterrows():
        factor = str(row["factor"])
        max_corr = 0.0
        for existing in kept:
            max_corr = max(max_corr, abs(float(correlation.loc[factor, existing])))
        if max_corr > max_abs_correlation:
            out.loc[idx, "redundancy_status"] = "drop_high_correlation"
        else:
            out.loc[idx, "redundancy_status"] = "selected"
            kept.append(factor)
    return out


def evaluate_cognitive_alpha_candidates(
    frame: pd.DataFrame,
    *,
    candidates: list[CognitiveAlphaCandidate],
    label_col: str,
    generation: int = 0,
    min_observations: int = 60,
    max_missing_ratio: float = 0.45,
    min_distinct_values: int = 5,
    qualified_percentile: float = 0.65,
    elite_percentile: float = 0.80,
    max_abs_correlation: float = 0.95,
    fallback_top_n: int = 0,
) -> CognitiveAlphaReport:
    if label_col not in frame.columns:
        raise ValueError(f"missing label column: {label_col}")
    if not candidates:
        raise ValueError("at least one cognitive alpha candidate is required")

    data = frame.copy()
    label = pd.to_numeric(data[label_col], errors="coerce")
    feature_values: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        series, quality_status, quality_reason = _compute_candidate_feature(
            data,
            candidate,
            max_missing_ratio=max_missing_ratio,
            min_distinct_values=min_distinct_values,
        )
        metrics = (
            _corr_metrics(series, label, min_observations=min_observations)
            if series is not None and quality_status == "pass"
            else {"ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0, "mutual_information": 0.0}
        )
        if series is not None:
            feature_values[candidate.name] = series
        rows.append(
            {
                "factor": candidate.name,
                "generation": int(generation),
                "level": candidate.level,
                "agent": candidate.agent,
                "theme": candidate.theme,
                "lookback": int(candidate.lookback),
                "quality_status": quality_status,
                "quality_reason": quality_reason,
                "missing_ratio": float(series.isna().mean()) if series is not None else 1.0,
                "distinct_values": int(series.dropna().nunique()) if series is not None else 0,
                **metrics,
            }
        )

    metrics_frame = pd.DataFrame(rows)
    metrics_frame["abs_ic"] = metrics_frame["ic"].abs()
    metrics_frame["abs_rank_ic"] = metrics_frame["rank_ic"].abs()
    metrics_frame["abs_icir"] = metrics_frame["icir"].abs()
    metrics_frame["abs_rank_icir"] = metrics_frame["rank_icir"].abs()
    metrics_frame = _apply_percentile_tiers(
        metrics_frame,
        qualified_percentile=qualified_percentile,
        elite_percentile=elite_percentile,
    )
    feature_frame = pd.DataFrame(feature_values, index=data.index)
    if feature_frame.empty:
        correlation = pd.DataFrame()
    else:
        correlation = feature_frame.corr(method="spearman").fillna(0.0).round(10)
    metrics_frame = _apply_redundancy(metrics_frame, correlation, max_abs_correlation=max_abs_correlation)
    if fallback_top_n > 0 and not (metrics_frame["redundancy_status"] == "selected").any():
        top_indices = metrics_frame.loc[metrics_frame["quality_status"] == "pass"].sort_values(
            ["selection_score", "min_metric_percentile"],
            ascending=[False, False],
        ).head(fallback_top_n).index
        metrics_frame.loc[top_indices, "redundancy_status"] = "selected_fallback"
    selected = metrics_frame.loc[
        metrics_frame["redundancy_status"].isin(["selected", "selected_fallback"]),
        "factor",
    ].astype(str).tolist()

    summary = {
        "workflow": "cognitive_alpha_mining",
        "paper_reference": "arxiv:2511.18850",
        "source": "codex_code_based_candidate_pool",
        "generation": int(generation),
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(selected)),
        "selected_feature_names": selected,
        "decision_counts": {
            "elite": int((metrics_frame["selection_tier"] == "elite").sum()),
            "qualified": int((metrics_frame["selection_tier"] == "qualified").sum()),
            "drop": int((metrics_frame["selection_tier"] == "drop").sum()),
        },
        "quality_counts": {
            "pass": int((metrics_frame["quality_status"] == "pass").sum()),
            "reject": int((metrics_frame["quality_status"] == "reject").sum()),
        },
        "thresholds": {
            "min_observations": int(min_observations),
            "max_missing_ratio": float(max_missing_ratio),
            "qualified_percentile": float(qualified_percentile),
            "elite_percentile": float(elite_percentile),
            "max_abs_correlation": float(max_abs_correlation),
            "fallback_top_n": int(fallback_top_n),
        },
        "agent_levels": [level["level"] for level in COGALPHA_AGENT_LEVELS],
        "metrics": ["IC", "ICIR", "RankIC", "RankICIR", "MutualInformation"],
    }
    return CognitiveAlphaReport(
        metrics=metrics_frame.drop(columns=["abs_ic", "abs_rank_ic", "abs_icir", "abs_rank_icir"]),
        correlation=correlation,
        feature_frame=feature_frame,
        selected_feature_names=selected,
        summary=summary,
    )


def cognitive_alpha_codex_prompt_seed() -> str:
    levels = "\n".join(
        f"- {item['level']}: {', '.join(item['agents'])} -> {item['theme']}" for item in COGALPHA_AGENT_LEVELS
    )
    return (
        "Generate concise, vectorized alpha factor functions for a pandas OHLCV DataFrame.\n"
        "Each factor must represent one economic idea, use no future data, avoid nested loops, "
        "and stay within five logical steps.\n\n"
        f"CogAlpha hierarchy:\n{levels}\n\n"
        "Return CognitiveAlphaCandidate-compatible functions with required_columns, lookback, "
        "economic_rationale, and code metadata."
    )
