"""Qlib-native, factor-only SOXL path-state risk experiment."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, build_qlib_dataset, build_qlib_store_from_frame, load_qlib_feature_frame
from src.factor.soxl_path_state_risk import (
    PATH_STATE_BASELINE_FEATURES,
    PATH_STATE_FEATURES,
    PATH_STATE_FORWARD_METRICS,
    PATH_STATE_LABELS,
    build_soxl_path_state_frame,
    structural_state_age,
)
from src.workflow.multi_3x_price_volume_transfer import _diagnostic_metrics
from src.workflow.soxl_exhaustion_structure_fragility import conditional_incremental_lift


EXPERIMENT = "soxl_path_state_risk_v2"
CARD_TYPE = "soxl_factor_only_path_state_risk_research_card"
QLIB_DOCS = {
    "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
    "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
    "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
}

SEGMENTS = {
    "development": ("2011-01-03", "2022-12-30"),
    "validation": ("2023-01-03", "2024-12-31"),
    "test": ("2025-01-02", "2026-07-02"),
}

FAST_FORWARD_COLUMNS = ["fast_tail_5d", "forward_terminal_5", "forward_mae_5", "forward_mfe_5"]
STRUCTURAL_FORWARD_COLUMNS = [
    "structural_break_20d",
    "forward_terminal_20",
    "forward_mae_20",
    "forward_mfe_20",
]


@dataclass(frozen=True)
class SoxlPathStateRiskSpec:
    root: Path
    experiment: str = EXPERIMENT
    start: str = "2011-01-03"
    as_of: str = "2026-07-02"
    normalization_window: int = 252
    min_periods: int = 60

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / self.experiment

    @property
    def source_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"


def split_and_purge_segments(
    frame: pd.DataFrame,
    *,
    development: tuple[str, str],
    validation: tuple[str, str],
    test: tuple[str, str],
) -> dict[str, pd.DataFrame]:
    """Split fixed calendar windows and blank labels whose paths could cross a boundary."""

    dates = pd.to_datetime(frame["datetime"])
    result: dict[str, pd.DataFrame] = {}
    for name, (start, end) in {
        "development": development,
        "validation": validation,
        "test": test,
    }.items():
        segment = frame.loc[dates.between(pd.Timestamp(start), pd.Timestamp(end))].copy().reset_index(drop=True)
        segment["segment"] = name
        if len(segment):
            segment.loc[segment.index[-min(6, len(segment)) :], FAST_FORWARD_COLUMNS] = np.nan
            segment.loc[segment.index[-min(21, len(segment)) :], STRUCTURAL_FORWARD_COLUMNS] = np.nan
        result[name] = segment
    return result


def add_segment_episode_ids(frame: pd.DataFrame, *, event_column: str) -> pd.DataFrame:
    """Assign episode IDs without joining episodes across evaluation boundaries."""

    out = frame.copy()
    active = out[event_column].fillna(False).astype(bool)
    segment_changed = out["segment"].astype(str).ne(out["segment"].astype(str).shift())
    starts = active & (~active.shift(fill_value=False) | segment_changed)
    ids = starts.cumsum().astype("Int64")
    out["episode_id"] = ids.where(active)
    return out


def evaluate_path_state_decision(kind: str, metrics_by_segment: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Fail closed on sparse support, reversed direction, or absent conditional lift."""

    if kind not in {"fast", "structural"}:
        raise ValueError("kind must be fast or structural")
    missing = sorted(set(SEGMENTS).difference(metrics_by_segment))
    if missing:
        raise ValueError(f"missing segment metrics: {missing}")
    development = metrics_by_segment["development"]
    validation = metrics_by_segment["validation"]
    test = metrics_by_segment["test"]
    min_observations = 10 if kind == "structural" else 15
    min_episodes = 3
    sparse = any(
        int(metrics_by_segment[name].get("observations", 0)) < min_observations
        or int(metrics_by_segment[name].get("episodes", 0)) < min_episodes
        for name in ("validation", "test")
    )
    gates = {
        "outer_support": {"passed": not sparse, "min_observations": min_observations, "min_episodes": min_episodes},
        "development_direction": {
            "passed": float(development.get("auc", np.nan)) > 0.50
            and float(development.get("quintile_lift", np.nan)) > 0.0
        },
        "validation_direction": {
            "passed": float(validation.get("auc", np.nan)) > 0.50
            and float(validation.get("quintile_lift", np.nan)) > 0.0
        },
        "test_direction": {
            "passed": float(test.get("auc", np.nan)) > 0.50 and float(test.get("quintile_lift", np.nan)) > 0.0
        },
        "outer_conditional_increment": {
            "passed": float(validation.get("conditional_lift", np.nan)) > 0.0
            and float(test.get("conditional_lift", np.nan)) > 0.0
        },
    }
    if sparse:
        decision = "diagnostic_rare_event_support"
    elif any(not gates[name]["passed"] for name in ("development_direction", "validation_direction", "test_direction")):
        decision = "not_retained_direction_instability"
    elif not gates["outer_conditional_increment"]["passed"]:
        decision = "not_retained_no_conditional_increment"
    else:
        decision = "retained_incremental_factor_evidence"
    return {"decision": decision, "all_passed": all(gate["passed"] for gate in gates.values()), "gates": gates}


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _segment_metrics(
    frame: pd.DataFrame,
    *,
    kind: str,
    label_column: str,
    score_column: str,
    event_column: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    sample = frame[["segment", "datetime", label_column, score_column, event_column, "price_only_score"]].copy()
    usable = sample[[label_column, score_column]].dropna()
    metrics = {
        "score_observations": int(len(usable)),
        "label_events": int(usable[label_column].sum()) if len(usable) else 0,
    } | _tie_safe_diagnostic_metrics(usable[label_column], usable[score_column])
    conditional = conditional_incremental_lift(
        sample[label_column], sample["price_only_score"], sample[score_column], min_bin_observations=10
    )
    episodes = add_segment_episode_ids(sample, event_column=event_column)
    active = episodes.loc[episodes[event_column].eq(1.0)]
    metrics.update(
        observations=int(len(active)),
        episodes=int(active["episode_id"].nunique()),
        conditional_lift=float(conditional["conditional_event_rate_lift"]),
        conditional_bins=int(conditional["eligible_price_bins"]),
        kind=kind,
    )
    return metrics, episodes


def _tie_safe_diagnostic_metrics(labels: pd.Series, scores: pd.Series) -> dict[str, Any]:
    """Keep AUC/AP but avoid assigning tied sparse scores to quintiles by date order."""

    work = pd.DataFrame({"label": labels.astype(float), "score": scores.astype(float)}).dropna()
    metrics: dict[str, Any] = _diagnostic_metrics(work["label"], work["score"])
    if len(work) < 5 or work["score"].nunique() < 2:
        return metrics | {
            "quintile_1_rate": np.nan,
            "quintile_5_rate": np.nan,
            "quintile_lift": np.nan,
            "quintile_monotonicity": np.nan,
            "tail_metric_method": "insufficient_unique_scores",
        }
    low_threshold = float(work["score"].quantile(0.20))
    high_threshold = float(work["score"].quantile(0.80))
    if low_threshold < high_threshold:
        low = work.loc[work["score"].le(low_threshold), "label"]
        high = work.loc[work["score"].ge(high_threshold), "label"]
        method = "actual_score_q20_q80_with_ties"
    else:
        minimum = float(work["score"].min())
        low = work.loc[work["score"].eq(minimum), "label"]
        high = work.loc[work["score"].gt(minimum), "label"]
        method = "minimum_vs_above_minimum_due_tied_q20_q80"
    try:
        bins = pd.qcut(work["score"], q=5, duplicates="drop")
        rates = work.assign(score_bin=bins).groupby("score_bin", observed=True)["label"].mean()
        monotonicity = float(rates.corr(pd.Series(range(len(rates)), index=rates.index), method="spearman")) if len(rates) >= 2 else np.nan
    except ValueError:
        monotonicity = np.nan
    return metrics | {
        "quintile_1_rate": float(low.mean()) if len(low) else np.nan,
        "quintile_5_rate": float(high.mean()) if len(high) else np.nan,
        "quintile_lift": float(high.mean() - low.mean()) if len(low) and len(high) else np.nan,
        "quintile_monotonicity": monotonicity,
        "tail_metric_method": method,
        "tail_low_observations": int(len(low)),
        "tail_high_observations": int(len(high)),
    }


def _threshold_stability(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment, piece in frame.groupby("segment", sort=False):
        for efficiency in (0.15, 0.25, 0.35):
            for volatility in (0.67, 0.75, 0.80):
                event = piece["signed_efficiency_20"].ge(efficiency) & piece["rv_rank_20_252"].ge(volatility)
                sample = piece.loc[event, "fast_tail_5d"].dropna()
                rows.append(
                    {
                        "kind": "fast",
                        "segment": segment,
                        "efficiency_threshold": efficiency,
                        "volatility_threshold": volatility,
                        "age_threshold": 1,
                        "event_days": int(len(sample)),
                        "event_rate": float(sample.mean()) if len(sample) else np.nan,
                    }
                )
        for efficiency in (0.15, 0.20):
            for volatility in (0.60, 0.67, 0.75):
                state = (
                    piece["range_pos_252"].ge(0.70)
                    & piece["signed_efficiency_20"].abs().le(efficiency)
                    & piece["rv_rank_20_252"].ge(volatility)
                )
                age = structural_state_age(state)
                event = age.ge(4)
                sample = piece.loc[event, "structural_break_20d"].dropna()
                rows.append(
                    {
                        "kind": "structural",
                        "segment": segment,
                        "efficiency_threshold": efficiency,
                        "volatility_threshold": volatility,
                        "age_threshold": 4,
                        "event_days": int(len(sample)),
                        "event_rate": float(sample.mean()) if len(sample) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "path-state-title", "metadata": {}, "source": [f"# {EXPERIMENT}\n\n", conclusion]},
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "path-state-run",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\nfrom pathlib import Path\nimport qlib\nfrom qlib.data import D\nfrom qlib.workflow import R\n",
                    "ROOT = Path.cwd()\nwhile not (ROOT / 'src').exists():\n    ROOT = ROOT.parent\nsys.path.insert(0, ROOT.as_posix())\n",
                    "from src.workflow.soxl_path_state_risk import run_soxl_path_state_risk_card\n",
                    "qlib.init(provider_uri=(ROOT / 'data' / 'qlib' / 'soxl').as_posix(), region='us', kernels=1)\n",
                    "_ = D.features(['SOXL'], ['$close'], start_time='2011-01-03', freq='day')\nassert callable(R.start)\n",
                    "summary = run_soxl_path_state_risk_card(root=ROOT)\nsummary\n",
                ],
            },
            {"cell_type": "markdown", "id": "path-state-final", "metadata": {}, "source": [conclusion]},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (card / f"{EXPERIMENT}.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def run_soxl_path_state_risk_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = SoxlPathStateRiskSpec(root=_resolve_root(root))
    source = load_qlib_feature_frame(
        QlibProviderConfig(provider_uri=spec.source_provider_uri, market="all", region="us", kernels=1),
        instruments=["SOXL"],
        fields=["$close"],
        start=spec.start,
        end=spec.as_of,
    ).reset_index().rename(columns={"$close": "close"})
    feature_frame = build_soxl_path_state_frame(
        source[["instrument", "datetime", "close"]],
        normalization_window=spec.normalization_window,
        min_periods=spec.min_periods,
    )
    segments = split_and_purge_segments(feature_frame, **SEGMENTS)
    purged = pd.concat(segments.values(), ignore_index=True)
    store_columns = [
        "instrument", "datetime", "signed_efficiency_20", "rv_20", "rv_rank_20_252",
        "structural_chop_age", "fast_turbulence_event", "structural_chop_event",
        *PATH_STATE_BASELINE_FEATURES, *PATH_STATE_FEATURES, *PATH_STATE_LABELS, *PATH_STATE_FORWARD_METRICS,
    ]
    provider_manifest = build_qlib_store_from_frame(
        purged[store_columns], provider_uri=spec.provider_uri, as_of=spec.as_of,
        source="qlib D.features SOXL daily close with segment-purged forward paths",
    )
    fields = [f"${column}" for column in store_columns if column not in {"instrument", "datetime"}]
    readback = load_qlib_feature_frame(
        QlibProviderConfig(provider_uri=spec.provider_uri, market="all", region="us", kernels=1),
        instruments=["SOXL"], fields=fields, start=spec.start, end=spec.as_of,
    ).reset_index().rename(columns={field: field.removeprefix("$") for field in fields})
    readback["datetime"] = pd.to_datetime(readback["datetime"]).dt.normalize()
    readback_segments = split_and_purge_segments(readback, **SEGMENTS)
    scored = pd.concat(readback_segments.values(), ignore_index=True)

    metric_rows: list[dict[str, Any]] = []
    episode_frames: list[pd.DataFrame] = []
    decision_inputs: dict[str, dict[str, dict[str, Any]]] = {"fast": {}, "structural": {}}
    contracts = {
        "fast": ("fast_tail_5d", "fast_turbulence_risk", "fast_turbulence_event"),
        "structural": ("structural_break_20d", "structural_chop_risk", "structural_chop_event"),
    }
    for kind, (label, score, event) in contracts.items():
        for segment, piece in scored.groupby("segment", sort=False):
            metrics, episodes = _segment_metrics(
                piece, kind=kind, label_column=label, score_column=score, event_column=event
            )
            decision_inputs[kind][str(segment)] = metrics
            metric_rows.append({"kind": kind, "segment": segment, "score": score, **metrics})
            episode_frames.append(episodes.assign(kind=kind, score_column=score))
    segment_metrics = pd.DataFrame(metric_rows)
    decisions = {kind: evaluate_path_state_decision(kind, values) for kind, values in decision_inputs.items()}
    threshold_stability = _threshold_stability(scored)
    annual_rows: list[dict[str, Any]] = []
    for kind, (label, score, _) in contracts.items():
        for year, piece in scored.groupby(pd.to_datetime(scored["datetime"]).dt.year):
            sample = piece[[label, score]].dropna()
            annual_rows.append(
                {
                    "kind": kind,
                    "year": int(year),
                    "observations": int(len(sample)),
                    "events": int(sample[label].sum()),
                }
                | _tie_safe_diagnostic_metrics(sample[label], sample[score])
            )
    annual_metrics = pd.DataFrame(annual_rows)
    episode_rows = pd.concat(episode_frames, ignore_index=True)

    dataset_evidence: dict[str, Any] = {}
    for kind, (label, _, _) in contracts.items():
        dataset = build_qlib_dataset(
            QlibProviderConfig(provider_uri=spec.provider_uri, market="all", region="us", kernels=1),
            instruments=["SOXL"], features=[f"${feature}" for feature in PATH_STATE_FEATURES], label=f"${label}",
            train=SEGMENTS["development"], valid=SEGMENTS["validation"], test=SEGMENTS["test"],
        )
        dataset_evidence[kind] = {
            "api": dataset.qlib_dataset_api,
            "segments": dataset.qlib_segments,
            "prepared_rows": {name: int(len(dataset.prepare(name))) for name in ("train", "valid", "test")},
        }

    return _write_card(
        spec, source=source, scored=scored, episode_rows=episode_rows, segment_metrics=segment_metrics,
        annual_metrics=annual_metrics, threshold_stability=threshold_stability, decisions=decisions,
        provider_manifest=provider_manifest, dataset_evidence=dataset_evidence,
    )


def _write_card(spec: SoxlPathStateRiskSpec, **artifacts: Any) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    artifacts["segment_metrics"].to_csv(card / "segment_metrics.csv", index=False)
    artifacts["annual_metrics"].to_csv(card / "annual_metrics.csv", index=False)
    artifacts["threshold_stability"].to_csv(card / "threshold_stability.csv", index=False)
    factor_summary = pd.DataFrame(
        [{"factor": "trend_permission", "role": "positive trend permission", "direction": "higher is safer"},
         {"factor": "fast_turbulence_risk", "role": "fast 5d tail risk", "direction": "higher is riskier"},
         {"factor": "structural_chop_risk", "role": "persistent 20d high-level chop", "direction": "higher is riskier"}]
    )
    factor_summary.to_csv(card / "factor_summary.csv", index=False)
    (card / "workflow.yaml").write_text(
        f"experiment: {spec.experiment}\nprovider_uri: {spec.provider_uri.resolve().as_posix()}\n"
        "method: transparent_factor_only_path_state\nfeature_timing: close_t\nlabel_timing: strictly_after_close_t\n"
        "segments:\n  development: [2011-01-03, 2022-12-30]\n  validation: [2023-01-03, 2024-12-31]\n  test: [2025-01-02, 2026-07-02]\n"
        "qlib_api:\n  - qlib.data.D.features\n  - DatasetH\n  - qlib.workflow.R\n  - Recorder\n"
        "artifact_policy: slim_card_recorder_first\n",
        encoding="utf-8",
    )
    rows = artifacts["segment_metrics"].set_index(["kind", "segment"])
    latest = artifacts["scored"].sort_values("datetime").iloc[-1]
    lines: list[str] = []
    for kind in ("fast", "structural"):
        for segment in ("development", "validation", "test"):
            row = rows.loc[(kind, segment)]
            lines.append(
                f"- {kind}/{segment}: score样本={int(row['score_observations'])}, 事件日={int(row['observations'])}, "
                f"episode={int(row['episodes'])}, AUC={row['auc']:.3f}, Q5-Q1={row['quintile_lift']:.1%}, "
                f"price分层条件增量={row['conditional_lift']:.1%}。"
            )
    decision_lines = [f"- {kind}: {decision['decision']}" for kind, decision in artifacts["decisions"].items()]
    conclusion = (
        "# 结论\n\n"
        "- 执行内容：只用SOXL收盘前可知的价格路径，区分趋势许可、快速湍流和持续高位震荡；不用预测模型。\n"
        f"- 最新状态（{pd.Timestamp(latest['datetime']).date().isoformat()}）：252日区间位置={float(latest['range_pos_252']):.3f}，"
        f"20日有向效率={float(latest['signed_efficiency_20']):.3f}，波动率排名={float(latest['rv_rank_20_252']):.3f}，"
        f"趋势许可={float(latest['trend_permission']):.3f}，快速湍流风险={float(latest['fast_turbulence_risk']):.3f}，"
        f"持续震荡年龄={int(latest['structural_chop_age'])}。\n"
        + "\n".join(lines) + "\n"
        + "\n".join(decision_lines) + "\n"
        "- 晋升决定：这是因子诊断卡，不是交易策略晋升；验证与测试必须同向、事件支持充分，并在price-only分层内都有正增量才保留。\n"
        "- 风险：日线无法观察盘中路径；5日和20日标签重叠；末端样本已按窗口purge；事件episode比逐日样本更接近独立证据。\n"
        "- 下一步：只把保留因子接入分层降仓回测；失败因子停止调模型，改查盘中/事件数据。\n"
    )
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    (card / "failure_modes.md").write_text(
        "# Failure Modes\n\n- overlapping_labels: 逐日标签重叠，决策同时检查episode。\n"
        "- daily_resolution: 日线无法区分同日先涨后跌与先跌后涨。\n"
        "- fixed_thresholds: 门槛在运行前固定；稳定性网格仅诊断，不据此事后选参。\n"
        "- factor_only: 风险辨识不等于含成本策略收益。\n",
        encoding="utf-8",
    )

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    from qlib.workflow import R

    R.set_uri(recorder_uri.resolve().as_posix())
    recorder_only = ["scored_rows.csv", "episode_rows.csv"]
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as temp_dir:
        temp = Path(temp_dir)
        artifacts["scored"].to_csv(temp / recorder_only[0], index=False)
        artifacts["episode_rows"].to_csv(temp / recorder_only[1], index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(method="transparent_factor_only_path_state", feature_timing="close_t", label_timing="strictly_after_close_t")
            for path in temp.iterdir():
                recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            summary = {
                "experiment": spec.experiment,
                "card_type": CARD_TYPE,
                "status": "completed",
                "as_of": spec.as_of,
                "method": "transparent_factor_only_no_predictive_model",
                "qlib_contract": {
                    "docs": QLIB_DOCS,
                    "data_api": ["qlib.init", "qlib.data.D.features"],
                    "dataset_api": ["DatasetH", "DataHandlerLP", "QlibDataLoader"],
                    "workflow_api": ["qlib.workflow.R", "Recorder"],
                    "dataset_evidence": artifacts["dataset_evidence"],
                },
                "data_lineage": {"source_provider_uri": spec.source_provider_uri.resolve().as_posix(), "provider_manifest": artifacts["provider_manifest"]},
                "data_quality": {
                    "source_rows": int(len(artifacts["source"])),
                    "duplicate_instrument_datetime_rows": int(
                        artifacts["source"].duplicated(["instrument", "datetime"]).sum()
                    ),
                    "required_close_null_rows": int(artifacts["source"]["close"].isna().sum()),
                    "min_datetime": pd.Timestamp(artifacts["source"]["datetime"].min()).date().isoformat(),
                    "max_datetime": pd.Timestamp(artifacts["source"]["datetime"].max()).date().isoformat(),
                },
                "factor_contract": {
                    "features": PATH_STATE_FEATURES,
                    "labels": {"fast": "5d MAE<=-10% and terminal<=-5%", "structural": "20d MAE<=-20% and terminal<=-10%"},
                    "direction": "risk factors higher means more downside path risk; trend_permission higher means safer",
                    "timing": "close_t_features_predict_strictly_after_close_t",
                    "segment_purge_rows": {"fast": 6, "structural": 21},
                },
                "factor_decisions": artifacts["decisions"],
                "latest_state": {
                    "datetime": pd.Timestamp(latest["datetime"]).date().isoformat(),
                    "range_pos_252": float(latest["range_pos_252"]),
                    "signed_efficiency_20": float(latest["signed_efficiency_20"]),
                    "rv_rank_20_252": float(latest["rv_rank_20_252"]),
                    "trend_permission": float(latest["trend_permission"]),
                    "fast_turbulence_risk": float(latest["fast_turbulence_risk"]),
                    "structural_chop_age": int(latest["structural_chop_age"]),
                    "structural_chop_risk": float(latest["structural_chop_risk"]),
                },
                "promotion_decision": "not_promoted_factor_diagnostic_only",
                "recorder": {"experiment": spec.experiment, "recorder_id": recorder.id, "recorder_uri": recorder_uri.resolve().as_posix(), "local_dir": Path(recorder.get_local_dir()).resolve().as_posix()},
                "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml", "recorder_only_artifacts": recorder_only},
                "compact_sidecars": ["factor_summary.csv", "segment_metrics.csv", "annual_metrics.csv", "threshold_stability.csv", "failure_modes.md"],
            }
            (card / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
            for name in summary["compact_sidecars"] + ["workflow.yaml", "summary.json", "conclusion.md"]:
                recorder.log_artifact((card / name).as_posix(), artifact_path=spec.experiment)
    _write_notebook(card, summary)
    return summary
