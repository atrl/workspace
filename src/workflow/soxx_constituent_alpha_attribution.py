from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis import (
    build_constituent_attribution_panel,
    evaluate_constituent_attribution,
    summarize_execution_censoring,
)
from src.data import QlibProviderConfig, init_qlib_provider, load_qlib_feature_frame


SOXX_ATTRIBUTION_EXPERIMENT = "soxx_constituent_alpha_attribution_v1"
SOXX_ATTRIBUTION_CARD_TYPE = "diagnostic_soxx_constituent_alpha_attribution"
SOURCE_EXECUTION_EXPERIMENT = "hardware_cycle_cogalpha_mining_v1"
SOURCE_EXECUTION_ARTIFACT = "hardware_cycle_cogalpha_mining/positions.csv"


@dataclass(frozen=True)
class SoxxConstituentAlphaAttributionSpec:
    root: Path
    experiment: str = SOXX_ATTRIBUTION_EXPERIMENT
    train: tuple[str, str] = ("2019-01-02", "2024-12-31")
    validation: tuple[str, str] = ("2025-01-02", "2025-12-31")
    test_start: str = "2026-01-02"
    lookback: int = 20
    lag_days: int = 1
    horizon: int = 5

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def soxl_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def constituent_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxx_constituents"


def run_soxx_constituent_alpha_attribution_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = SoxxConstituentAlphaAttributionSpec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: SoxxConstituentAlphaAttributionSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    soxl_manifest = _read_json(spec.soxl_provider_uri / "qlib_store_manifest.json")
    constituent_manifest = _read_json(spec.constituent_provider_uri / "qlib_store_manifest.json")
    as_of = min(str(soxl_manifest["calendar_end"]), str(constituent_manifest["calendar_end"]))
    segments = {
        "train": spec.train,
        "validation": spec.validation,
        "test": (spec.test_start, as_of),
    }

    soxl_config = QlibProviderConfig(
        provider_uri=spec.soxl_provider_uri,
        region=str(soxl_manifest["region"]),
        market=str(soxl_manifest["market"]),
        kernels=1,
    )
    provider_meta = init_qlib_provider(soxl_config)
    index_qlib = load_qlib_feature_frame(
        soxl_config,
        instruments=["SOXL", "SOXX"],
        fields=["$open"],
        start=spec.train[0],
        end=as_of,
    ).reset_index()
    index_frame = (
        index_qlib.pivot(index="datetime", columns="instrument", values="$open")
        .rename(columns={"SOXL": "soxl_open", "SOXX": "soxx_open"})
        .reset_index()
    )

    constituent_config = QlibProviderConfig(
        provider_uri=spec.constituent_provider_uri,
        region=str(constituent_manifest["region"]),
        market=str(constituent_manifest["market"]),
        kernels=1,
    )
    constituent_qlib = load_qlib_feature_frame(
        constituent_config,
        instruments=list(constituent_manifest["instruments"]),
        fields=["$adj_close", "$weight_pct"],
        start=spec.train[0],
        end=as_of,
    ).reset_index()
    constituent_frame = constituent_qlib.rename(
        columns={"$adj_close": "adj_close", "$weight_pct": "weight_pct"}
    )[["datetime", "instrument", "adj_close", "weight_pct"]]

    panel = build_constituent_attribution_panel(
        constituent_frame,
        index_frame,
        train=spec.train,
        lookback=spec.lookback,
        lag_days=spec.lag_days,
        horizon=spec.horizon,
    )
    metrics = evaluate_constituent_attribution(panel, segments=segments)
    source_summary = _read_json(spec.root / "research" / SOURCE_EXECUTION_EXPERIMENT / "summary.json")
    source_positions = _load_source_positions(source_summary)
    execution_metrics = summarize_execution_censoring(source_positions, panel.daily[["datetime", "soxl_5d"]])

    sidecars = {
        "individual_factor_metrics": "individual_factor_metrics.csv",
        "aggregate_factor_metrics": "aggregate_factor_metrics.csv",
        "aggregate_stability": "aggregate_stability.csv",
        "nonoverlap_factor_metrics": "nonoverlap_factor_metrics.csv",
        "leverage_mapping_metrics": "leverage_mapping_metrics.csv",
        "execution_censoring_metrics": "execution_censoring_metrics.csv",
        "risk_analysis": "risk_analysis.csv",
    }
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    import qlib
    from qlib.workflow import R

    _ = qlib
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix="soxx_constituent_alpha_attribution_") as tmp:
        tmp_dir = Path(tmp)
        daily_path = tmp_dir / "daily_attribution_panel.csv"
        selected_positions_path = tmp_dir / "source_selected_positions.csv"
        panel.daily.to_csv(daily_path, index=False)
        selected_positions = source_positions.loc[_selected_mask(source_positions)].copy()
        selected_positions.to_csv(selected_positions_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=provider_meta["provider_uri"],
                constituent_provider_uri=spec.constituent_provider_uri.resolve().as_posix(),
                train_start=spec.train[0],
                train_end=spec.train[1],
                validation_start=spec.validation[0],
                validation_end=spec.validation[1],
                test_start=spec.test_start,
                test_end=as_of,
                lookback=spec.lookback,
                lag_days=spec.lag_days,
                horizon=spec.horizon,
                constituent_count=len(constituent_manifest["instruments"]),
                source_execution_recorder_id=source_summary["recorder_id"],
            )
            summary = _summary_payload(
                spec=spec,
                provider_meta=provider_meta,
                constituent_manifest=constituent_manifest,
                as_of=as_of,
                segments=segments,
                metrics=metrics,
                execution_metrics=execution_metrics,
                source_summary=source_summary,
                sidecars=sidecars,
                recorder_uri=recorder_uri,
                recorder_id=recorder.id,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
            )
            risk_analysis = _risk_analysis(summary)
            _write_card(
                card,
                summary=summary,
                metrics=metrics,
                execution_metrics=execution_metrics,
                risk_analysis=risk_analysis,
                sidecars=sidecars,
            )
            recorder.log_artifact(daily_path.as_posix(), artifact_path="soxx_constituent_alpha_attribution")
            recorder.log_artifact(selected_positions_path.as_posix(), artifact_path="soxx_constituent_alpha_attribution")
            for rel_path in sidecars.values():
                recorder.log_artifact((card / rel_path).as_posix(), artifact_path=spec.experiment)
    return summary


def _load_source_positions(source_summary: dict[str, Any]) -> pd.DataFrame:
    from qlib.workflow import R

    R.set_uri(Path(source_summary["recorder_uri"]).resolve().as_posix())
    recorder = R.get_recorder(
        recorder_id=str(source_summary["recorder_id"]),
        experiment_name=str(source_summary["experiment"]),
    )
    with tempfile.TemporaryDirectory(prefix="source_execution_positions_") as tmp:
        downloaded = Path(recorder.download_artifact(SOURCE_EXECUTION_ARTIFACT, dst_path=tmp))
        return pd.read_csv(downloaded)


def _selected_mask(frame: pd.DataFrame) -> pd.Series:
    if "selected" not in frame:
        return pd.Series(True, index=frame.index)
    selected = frame["selected"]
    return selected if selected.dtype == bool else selected.astype(str).str.lower().eq("true")


def _summary_payload(
    *,
    spec: SoxxConstituentAlphaAttributionSpec,
    provider_meta: dict[str, Any],
    constituent_manifest: dict[str, Any],
    as_of: str,
    segments: dict[str, tuple[str, str]],
    metrics: dict[str, pd.DataFrame],
    execution_metrics: pd.DataFrame,
    source_summary: dict[str, Any],
    sidecars: dict[str, str],
    recorder_uri: Path,
    recorder_id: str,
    recorder_local_dir: Path,
) -> dict[str, Any]:
    aggregate = metrics["aggregate"]
    test_soxx = aggregate.loc[aggregate["target"].eq("soxx_5d") & aggregate["segment"].eq("test")]
    best_test = (
        test_soxx.sort_values(["rank_correlation", "top_bottom_spread_pct"], ascending=False).iloc[0].to_dict()
        if not test_soxx.empty
        else {}
    )
    best_nonoverlap = metrics["nonoverlap"].loc[
        metrics["nonoverlap"]["signal"].eq(best_test.get("signal"))
        & metrics["nonoverlap"]["target"].eq("soxx_5d")
        & metrics["nonoverlap"]["segment"].eq("test")
    ]
    offset_correlations = pd.to_numeric(best_nonoverlap["rank_correlation"], errors="coerce").dropna()
    nonoverlap_payload = {
        "signal": best_test.get("signal"),
        "target": "soxx_5d",
        "segment": "test",
        "horizon_offsets": int(spec.horizon),
        "finite_offset_count": int(len(offset_correlations)),
        "positive_offset_ratio": float(offset_correlations.gt(0.0).mean()) if len(offset_correlations) else None,
        "rank_correlation_min": float(offset_correlations.min()) if len(offset_correlations) else None,
        "rank_correlation_median": float(offset_correlations.median()) if len(offset_correlations) else None,
        "rank_correlation_max": float(offset_correlations.max()) if len(offset_correlations) else None,
        "observations_per_offset_min": int(best_nonoverlap["observations"].min()) if not best_nonoverlap.empty else 0,
        "observations_per_offset_max": int(best_nonoverlap["observations"].max()) if not best_nonoverlap.empty else 0,
    }
    stable_count = int(metrics["stability"]["all_segments_same_sign"].map(bool).sum())
    individual = metrics["individual"]
    sign_pivot = individual.pivot(index="instrument", columns="segment", values="raw_rank_correlation")
    sign_columns = [column for column in ["train", "validation", "test"] if column in sign_pivot]
    same_sign_ratio = (
        float(sign_pivot[sign_columns].apply(lambda row: len(set(row.dropna().map(lambda x: 1 if x >= 0 else -1))) == 1, axis=1).mean())
        if sign_columns
        else 0.0
    )
    execution_test = execution_metrics.loc[execution_metrics["segment"].eq("test")]
    execution_test_payload = execution_test.iloc[0].to_dict() if not execution_test.empty else {}
    return {
        "experiment": spec.experiment,
        "card_type": SOXX_ATTRIBUTION_CARD_TYPE,
        "status": "complete",
        "promotion_decision": "not_promoted_missing_pit_holdings",
        "as_of": as_of,
        "provider_uri": provider_meta["provider_uri"],
        "recorder_uri": recorder_uri.resolve().as_posix(),
        "recorder_id": recorder_id,
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "segments": {name: list(window) for name, window in segments.items()},
        "analysis_layers": [
            "constituent_attribution",
            "soxx_aggregation",
            "daily_reset_leverage_mapping",
            "execution_censoring",
        ],
        "constituent_data": {
            "source": "qlib.data.D.features",
            "provider_uri": spec.constituent_provider_uri.resolve().as_posix(),
            "instrument_count": len(constituent_manifest["instruments"]),
            "point_in_time_status": "blocked_current_holdings_backfill",
            "limitations": list(constituent_manifest.get("limitations", [])),
        },
        "factor_diagnostic": {
            "lookback": spec.lookback,
            "lag_days": spec.lag_days,
            "horizon": spec.horizon,
            "train_only_orientation": True,
            "aggregate_stable_signal_count": stable_count,
            "single_name_all_segment_same_sign_ratio": same_sign_ratio,
            "best_test_soxx_signal": _json_safe(best_test),
            "best_test_nonoverlap": _json_safe(nonoverlap_payload),
        },
        "leverage_mapping": {
            "formula": "product(1 + 3 * future_daily_soxx_open_to_open_return) - 1",
            "metrics": _records(metrics["leverage"]),
        },
        "execution_censoring": {
            "source": "qlib_recorder_artifact",
            "source_experiment": source_summary["experiment"],
            "source_recorder_id": source_summary["recorder_id"],
            "test": _json_safe(execution_test_payload),
        },
        "qlib_native_api": [
            "qlib.init",
            "qlib.data.D.features",
            "qlib.workflow.R.start",
            "qlib.workflow.Recorder.download_artifact",
        ],
        "qlib_docs": {
            "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
            "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
        },
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "machine_entrypoint": "summary.json",
            "primary_artifact": f"{spec.experiment}.ipynb",
            "recorder_only_artifacts": ["daily_attribution_panel.csv", "source_selected_positions.csv"],
        },
        "spec": _json_safe(asdict(spec)),
    }


def _risk_analysis(summary: dict[str, Any]) -> pd.DataFrame:
    diagnostic = summary["factor_diagnostic"]
    execution = summary["execution_censoring"]["test"]
    nonoverlap = diagnostic["best_test_nonoverlap"]
    return pd.DataFrame(
        [
            {
                "check": "point_in_time_holdings",
                "status": "fail",
                "value": summary["constituent_data"]["point_in_time_status"],
                "detail": "current holdings are backfilled through history",
            },
            {
                "check": "aggregate_signal_sign_stability",
                "status": "pass" if diagnostic["aggregate_stable_signal_count"] > 0 else "fail",
                "value": diagnostic["aggregate_stable_signal_count"],
                "detail": "count of aggregate signals with the same oriented sign in train/validation/test",
            },
            {
                "check": "execution_signal_censoring",
                "status": "fail" if float(execution.get("position_floor_ratio", 0.0) or 0.0) >= 0.95 else "pass",
                "value": execution.get("position_floor_ratio"),
                "detail": "test share fixed at the selected position floor",
            },
            {
                "check": "nonoverlap_offset_stability",
                "status": "pass"
                if float(nonoverlap.get("positive_offset_ratio", 0.0) or 0.0) >= 1.0
                else "fail",
                "value": nonoverlap.get("positive_offset_ratio"),
                "detail": "all five non-overlapping test offsets must have positive oriented rank correlation",
            },
            {
                "check": "promotion_boundary",
                "status": "fail",
                "value": summary["promotion_decision"],
                "detail": "diagnostic evidence cannot promote without PIT holdings",
            },
        ]
    )


def _write_card(
    card: Path,
    *,
    summary: dict[str, Any],
    metrics: dict[str, pd.DataFrame],
    execution_metrics: pd.DataFrame,
    risk_analysis: pd.DataFrame,
    sidecars: dict[str, str],
) -> None:
    metrics["individual"].to_csv(card / sidecars["individual_factor_metrics"], index=False)
    metrics["aggregate"].to_csv(card / sidecars["aggregate_factor_metrics"], index=False)
    metrics["stability"].to_csv(card / sidecars["aggregate_stability"], index=False)
    metrics["nonoverlap"].to_csv(card / sidecars["nonoverlap_factor_metrics"], index=False)
    metrics["leverage"].to_csv(card / sidecars["leverage_mapping_metrics"], index=False)
    execution_metrics.to_csv(card / sidecars["execution_censoring_metrics"], index=False)
    risk_analysis.to_csv(card / sidecars["risk_analysis"], index=False)
    (card / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")
    _write_notebook(card, summary)


def _workflow_yaml(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"experiment: {summary['experiment']}",
            f"card_type: {summary['card_type']}",
            f"provider_uri: {summary['provider_uri']}",
            f"constituent_provider_uri: {summary['constituent_data']['provider_uri']}",
            f"recorder_uri: {summary['recorder_uri']}",
            f"recorder_id: {summary['recorder_id']}",
            f"as_of: {summary['as_of']}",
            f"promotion_decision: {summary['promotion_decision']}",
            "analysis_layers:",
            *[f"  - {layer}" for layer in summary["analysis_layers"]],
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            "  machine_entrypoint: summary.json",
            "",
        ]
    )


def _conclusion(summary: dict[str, Any]) -> str:
    factor = summary["factor_diagnostic"]
    best = factor["best_test_soxx_signal"]
    execution = summary["execution_censoring"]["test"]
    nonoverlap = factor["best_test_nonoverlap"]
    leverage_test = next(
        (row for row in summary["leverage_mapping"]["metrics"] if row.get("segment") == "test"),
        {},
    )
    return (
        "# 结论与反思\n\n"
        f"- what_ran: 使用 qlib SOXX 成分股和 SOXL/SOXX open 数据，分离单股归因、SOXX 聚合、3x 日复位映射与执行截断。\n"
        f"- data: as_of={summary['as_of']}，constituents={summary['constituent_data']['instrument_count']}，"
        f"PIT={summary['constituent_data']['point_in_time_status']}。\n"
        f"- factor: stable_aggregate_count={factor['aggregate_stable_signal_count']}，"
        f"single_name_same_sign_ratio={float(factor['single_name_all_segment_same_sign_ratio']):.4f}，"
        f"best_test_signal={best.get('signal')}，test_rank_corr={float(best.get('rank_correlation', 0.0) or 0.0):.4f}，"
        f"test_spread={float(best.get('top_bottom_spread_pct', 0.0) or 0.0):.4f}%。\n"
        f"- uncertainty: nonoverlap_positive_offset_ratio={float(nonoverlap.get('positive_offset_ratio', 0.0) or 0.0):.4f}，"
        f"offset_rank_corr_range=[{float(nonoverlap.get('rank_correlation_min', 0.0) or 0.0):.4f}, "
        f"{float(nonoverlap.get('rank_correlation_max', 0.0) or 0.0):.4f}]，"
        f"observations_per_offset={int(nonoverlap.get('observations_per_offset_min', 0) or 0)}-"
        f"{int(nonoverlap.get('observations_per_offset_max', 0) or 0)}。\n"
        f"- leverage: test_tracking_rmse={float(leverage_test.get('tracking_rmse_pct', 0.0) or 0.0):.4f}%，"
        f"rank_corr={float(leverage_test.get('synthetic_actual_rank_correlation', 0.0) or 0.0):.4f}。\n"
        f"- execution: raw_zero_ratio={float(execution.get('raw_zero_ratio', 0.0) or 0.0):.4f}，"
        f"floor_ratio={float(execution.get('position_floor_ratio', 0.0) or 0.0):.4f}，"
        f"probability_rank_corr={float(execution.get('probability_rank_correlation', 0.0) or 0.0):.4f}。\n"
        f"- promotion_decision: {summary['promotion_decision']}。\n"
        "- 反思: 过去把单股代理、SOXX 传导、SOXL 杠杆路径和仓位策略压进同一模型，导致因子失效与执行截断无法区分。"
        "本实验只允许形成研究诊断；在补齐 PIT holdings、purged walk-forward 和稳定 SOXX 样本外增量前，不应放宽交易约束。\n"
        "- next_action: 先补 point-in-time SOXX holdings，再对每只成分股做横截面预测和 leave-one-name-out 归因；"
        "通过后才把预测映射为 SOXX core + SOXL overlay。\n"
        "- risk: 当前权重存在 survivorship bias，五日标签重叠，结果不构成买卖建议。\n"
        f"- recorder: {summary['recorder_id']}。\n"
    )


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "soxx-attribution-title",
                "metadata": {},
                "source": [f"# {summary['experiment']}\n", "\n", _conclusion(summary)],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "soxx-attribution-run",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "import qlib\n",
                    "from qlib.data import D\n",
                    "from qlib.workflow import R\n",
                    "\n",
                    "ROOT = Path.cwd()\n",
                    "while not (ROOT / 'src').exists():\n",
                    "    if ROOT == ROOT.parent:\n",
                    "        raise RuntimeError('could not locate repository root')\n",
                    "    ROOT = ROOT.parent\n",
                    "sys.path.insert(0, ROOT.as_posix())\n",
                    "from src.workflow.soxx_constituent_alpha_attribution import run_soxx_constituent_alpha_attribution_card\n",
                    "\n",
                    "qlib.init(provider_uri=(ROOT / 'data' / 'qlib' / 'soxl').as_posix(), region='us', kernels=1)\n",
                    "_ = D.features(['SOXL', 'SOXX'], ['$open'], start_time='2019-01-02', freq='day')\n",
                    "assert callable(R.start)\n",
                    "summary = run_soxx_constituent_alpha_attribution_card(root=ROOT)\n",
                    "assert R.get_uri() == summary['recorder_uri']\n",
                    "summary\n",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "soxx-attribution-final",
                "metadata": {},
                "source": [_conclusion(summary)],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (card / f"{summary['experiment']}.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(row) for row in frame.to_dict(orient="records")]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path.as_posix())
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")
