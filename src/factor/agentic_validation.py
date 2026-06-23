from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.workflow.agentic_alpha import AGENTIC_ALPHA_DECISION_LABELS, AGENTIC_ALPHA_STAGE_ORDER


@dataclass(frozen=True)
class AgenticAlphaFactorValidationReport:
    source: str
    workflow: str
    metrics: pd.DataFrame
    correlation: pd.DataFrame
    summary: dict[str, Any]

    def write_sidecars(self, directory: Path) -> dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "metrics": directory / "agentic_alpha_factor_validation.csv",
            "correlation": directory / "agentic_alpha_factor_correlation.csv",
            "summary": directory / "agentic_alpha_factor_summary.json",
        }
        self.metrics.to_csv(paths["metrics"], index=False)
        self.correlation.to_csv(paths["correlation"])
        paths["summary"].write_text(json.dumps(self.summary, indent=2, sort_keys=True), encoding="utf-8")
        return paths


def _datetime_level(index: pd.Index) -> str | int | None:
    if not isinstance(index, pd.MultiIndex):
        return None
    for name in ("datetime", "date"):
        if name in index.names:
            return name
    return index.names[-1] if index.names else -1


def _daily_rank_ic(frame: pd.DataFrame, *, factor_col: str, label_col: str, min_cross_section: int) -> pd.Series:
    clean = frame[[factor_col, label_col]].dropna()
    if clean.empty:
        return pd.Series(dtype="float64")

    def rank_corr(group: pd.DataFrame) -> float:
        if len(group) < min_cross_section:
            return np.nan
        if group[factor_col].nunique() <= 1 or group[label_col].nunique() <= 1:
            return np.nan
        return float(group[factor_col].rank(method="average").corr(group[label_col].rank(method="average")))

    level = _datetime_level(clean.index)
    if level is None:
        value = rank_corr(clean)
        return pd.Series([value]).dropna()
    return clean.groupby(level=level, sort=True).apply(rank_corr).dropna().astype("float64")


def _ic_metrics(daily_ic: pd.Series) -> dict[str, float]:
    if daily_ic.empty:
        return {
            "n_days": 0,
            "rank_ic_mean": 0.0,
            "rank_ic_std": 0.0,
            "rank_icir": 0.0,
            "rank_ic_t": 0.0,
            "p_value": 1.0,
        }
    mean = float(daily_ic.mean())
    std = float(daily_ic.std(ddof=0))
    n_days = int(len(daily_ic))
    if std <= 1e-12:
        t_stat = math.copysign(1e6, mean) if abs(mean) > 0 else 0.0
        icir = math.copysign(1e6, mean) if abs(mean) > 0 else 0.0
    else:
        t_stat = mean / (std / math.sqrt(n_days))
        icir = mean / std
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return {
        "n_days": n_days,
        "rank_ic_mean": round(mean, 10),
        "rank_ic_std": round(std, 10),
        "rank_icir": round(float(icir), 10),
        "rank_ic_t": round(float(t_stat), 10),
        "p_value": round(float(p_value), 10),
    }


def _stability_metrics(daily_ic: pd.Series) -> dict[str, float]:
    if daily_ic.empty:
        return {"stability_score": 0.0, "ic_recent_vs_full": 0.0, "ic_positive_pct": 0.0}
    mean = float(daily_ic.mean())
    target_sign = 1.0 if mean >= 0 else -1.0
    aligned = daily_ic * target_sign
    positive_pct = float((aligned > 0).mean())
    recent_n = max(1, int(math.ceil(len(daily_ic) * 0.2)))
    recent_mean = float(daily_ic.iloc[-recent_n:].mean())
    recent_vs_full = recent_mean / mean if abs(mean) > 1e-12 else 0.0
    std = float(daily_ic.std(ddof=0))
    stability_score = np.mean(
        [
            positive_pct,
            max(0.0, 1.0 - std / (abs(mean) + 1e-12)),
            min(max(recent_vs_full, 0.0), 2.0) / 2.0,
        ]
    )
    return {
        "stability_score": round(float(stability_score), 10),
        "ic_recent_vs_full": round(float(recent_vs_full), 10),
        "ic_positive_pct": round(float(positive_pct), 10),
    }


def _bh_fdr(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype="float64")
    order = np.argsort(values)
    adjusted = np.ones(len(values), dtype="float64")
    for rank, idx in enumerate(order, start=1):
        adjusted[idx] = values[idx] * len(values) / rank
    ordered_adjusted = adjusted[order]
    for index in range(len(ordered_adjusted) - 2, -1, -1):
        ordered_adjusted[index] = min(ordered_adjusted[index], ordered_adjusted[index + 1])
    adjusted[order] = np.minimum(ordered_adjusted, 1.0)
    return [round(float(value), 10) for value in adjusted]


def _apply_redundancy(metrics: pd.DataFrame, correlation: pd.DataFrame, *, max_abs_correlation: float) -> pd.DataFrame:
    ordered = metrics.assign(_original_order=range(len(metrics))).sort_values(
        ["abs_rank_icir", "_original_order"],
        ascending=[False, True],
    )
    kept: list[str] = []
    redundancy: dict[str, tuple[str, bool, float]] = {}
    for _, row in ordered.iterrows():
        factor = str(row["factor"])
        max_corr = 0.0
        duplicate = False
        for existing in kept:
            corr = abs(float(correlation.loc[factor, existing]))
            max_corr = max(max_corr, corr)
            if corr > max_abs_correlation:
                duplicate = True
        if duplicate:
            redundancy[factor] = ("high_similarity", False, max_corr)
        else:
            kept.append(factor)
            redundancy[factor] = ("pass", True, max_corr)

    out = metrics.copy()
    out["redundancy_flag"] = [redundancy[str(factor)][0] for factor in out["factor"]]
    out["orthogonality_pass"] = [bool(redundancy[str(factor)][1]) for factor in out["factor"]]
    out["max_abs_correlation"] = [round(float(redundancy[str(factor)][2]), 10) for factor in out["factor"]]
    return out


def _decision(
    row: pd.Series,
    *,
    min_abs_rank_icir: float,
    max_fdr_p: float,
    min_stability_score: float,
) -> tuple[str, str]:
    if not bool(row["orthogonality_pass"]):
        return "Drop", "high_similarity"
    if abs(float(row["rank_icir"])) < min_abs_rank_icir:
        return "Drop", "weak_rank_icir"
    if float(row["fdr_p"]) > max_fdr_p:
        return "Drop", "fdr_gate"
    if float(row["stability_score"]) < min_stability_score:
        return "Drop", "stability_gate"
    return "Promote", "passed_agentic_alpha_sfa_gates"


def validate_agentic_alpha_factors(
    dataset,
    *,
    segment: str,
    factor_cols: list[str],
    label_cols: dict[str, str],
    primary_horizon: str = "1d",
    min_cross_section: int = 30,
    min_abs_rank_icir: float = 0.10,
    max_fdr_p: float = 0.01,
    max_abs_correlation: float = 0.50,
    min_decay_ratio: float = 0.50,
    min_stability_score: float = 0.30,
) -> AgenticAlphaFactorValidationReport:
    if primary_horizon not in label_cols:
        raise ValueError("primary_horizon must be present in label_cols")
    frame = dataset.prepare(segment).copy()
    missing = [column for column in [*factor_cols, *label_cols.values()] if column not in frame.columns]
    if missing:
        raise ValueError(f"missing qlib dataset columns: {missing}")

    rows: list[dict[str, Any]] = []
    primary_label = label_cols[primary_horizon]
    for factor in factor_cols:
        daily_ic = _daily_rank_ic(frame, factor_col=factor, label_col=primary_label, min_cross_section=min_cross_section)
        row = {"factor": factor, **_ic_metrics(daily_ic), **_stability_metrics(daily_ic)}
        for horizon, label_col in label_cols.items():
            horizon_ic = _daily_rank_ic(frame, factor_col=factor, label_col=label_col, min_cross_section=min_cross_section)
            row[f"rank_icir_{horizon}"] = _ic_metrics(horizon_ic)["rank_icir"]
        primary_icir = float(row[f"rank_icir_{primary_horizon}"])
        five_day_icir = float(row.get("rank_icir_5d", 0.0))
        decay_ratio = abs(five_day_icir) / abs(primary_icir) if abs(primary_icir) > 1e-12 else 0.0
        row["decay_5d_1d"] = round(float(decay_ratio), 10)
        row["weekly_suitable"] = bool(decay_ratio >= min_decay_ratio)
        row["abs_rank_icir"] = abs(float(row["rank_icir"]))
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics["fdr_p"] = _bh_fdr(metrics["p_value"].astype(float).tolist())
    correlation = frame[factor_cols].corr(method="spearman").fillna(0.0).round(10)
    metrics = _apply_redundancy(metrics, correlation, max_abs_correlation=max_abs_correlation)
    decisions = [
        _decision(
            row,
            min_abs_rank_icir=min_abs_rank_icir,
            max_fdr_p=max_fdr_p,
            min_stability_score=min_stability_score,
        )
        for _, row in metrics.iterrows()
    ]
    metrics["decision"] = [decision for decision, _ in decisions]
    metrics["decision_basis"] = [basis for _, basis in decisions]
    metrics["weekly_suitable"] = metrics["weekly_suitable"].astype(object)
    metrics["orthogonality_pass"] = metrics["orthogonality_pass"].astype(object)

    decision_counts = {label: int((metrics["decision"] == label).sum()) for label in AGENTIC_ALPHA_DECISION_LABELS}
    summary = {
        "source": "qlib_dataset",
        "workflow": "agentic_alpha_sfa",
        "segment": segment,
        "stage_order": AGENTIC_ALPHA_STAGE_ORDER,
        "decision_counts": decision_counts,
        "factor_count": int(len(factor_cols)),
        "label_cols": label_cols,
        "primary_horizon": primary_horizon,
        "recorder_first": True,
        "thresholds": {
            "min_abs_rank_icir": min_abs_rank_icir,
            "max_fdr_p": max_fdr_p,
            "max_abs_correlation": max_abs_correlation,
            "min_decay_ratio": min_decay_ratio,
            "min_stability_score": min_stability_score,
        },
    }
    return AgenticAlphaFactorValidationReport(
        source="qlib_dataset",
        workflow="agentic_alpha_sfa",
        metrics=metrics.drop(columns=["abs_rank_icir"]),
        correlation=correlation,
        summary=summary,
    )
