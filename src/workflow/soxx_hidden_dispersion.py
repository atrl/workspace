from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import build_hidden_dispersion_panel, evaluate_hidden_dispersion
from src.data import QlibProviderConfig, init_qlib_provider, load_qlib_feature_frame


SOXX_HIDDEN_DISPERSION_EXPERIMENT = "soxx_hidden_dispersion_v1"
SOXX_HIDDEN_DISPERSION_CARD_TYPE = "diagnostic_soxx_hidden_dispersion"


@dataclass(frozen=True)
class SoxxHiddenDispersionSpec:
    root: Path
    experiment: str = SOXX_HIDDEN_DISPERSION_EXPERIMENT
    train: tuple[str, str] = ("2019-01-02", "2024-12-31")
    validation: tuple[str, str] = ("2025-01-02", "2025-12-31")
    test_start: str = "2026-01-02"
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


def run_soxx_hidden_dispersion_card(*, root: Path | str | None = None) -> dict[str, Any]:
    return _run_card(SoxxHiddenDispersionSpec(root=_resolve_root(root)))


def _run_card(spec: SoxxHiddenDispersionSpec) -> dict[str, Any]:
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
        fields=["$open", "$vix"],
        start=spec.train[0],
        end=as_of,
    ).reset_index()
    open_frame = index_qlib.pivot(index="datetime", columns="instrument", values="$open").rename(
        columns={"SOXL": "soxl_open", "SOXX": "soxx_open"}
    )
    vix = index_qlib.sort_values(["datetime", "instrument"]).groupby("datetime")["$vix"].first().rename("vix")
    index_frame = open_frame.join(vix, how="left").reset_index()
    required_index = {"datetime", "soxl_open", "soxx_open", "vix"}
    if missing := sorted(required_index - set(index_frame.columns)):
        raise ValueError(f"qlib SOXL/SOXX hidden dispersion fields missing: {missing}")

    constituent_config = QlibProviderConfig(
        provider_uri=spec.constituent_provider_uri,
        region=str(constituent_manifest["region"]),
        market=str(constituent_manifest["market"]),
        kernels=1,
    )
    constituent_qlib = load_qlib_feature_frame(
        constituent_config,
        instruments=list(constituent_manifest["instruments"]),
        fields=["$open", "$high", "$low", "$adj_close", "$volume", "$weight_pct"],
        start=spec.train[0],
        end=as_of,
    ).reset_index()
    constituent_frame = constituent_qlib.rename(
        columns={
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$adj_close": "adj_close",
            "$volume": "volume",
            "$weight_pct": "weight_pct",
        }
    )
    panel = build_hidden_dispersion_panel(
        constituent_frame,
        index_frame,
        train=spec.train,
        lag_days=spec.lag_days,
        horizon=spec.horizon,
    )
    metrics = evaluate_hidden_dispersion(panel, segments=segments)

    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    import qlib
    from qlib.workflow import R

    _ = qlib
    R.set_uri(recorder_uri.resolve().as_posix())
    sidecars = {
        "feature_metrics": "feature_metrics.csv",
        "nonoverlap_feature_metrics": "nonoverlap_feature_metrics.csv",
        "model_comparison": "model_comparison.csv",
        "risk_analysis": "risk_analysis.csv",
    }
    with tempfile.TemporaryDirectory(prefix="soxx_hidden_dispersion_") as tmp:
        daily_path = Path(tmp) / "daily_hidden_dispersion_panel.csv"
        panel.daily.to_csv(daily_path, index=False)
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
                lag_days=spec.lag_days,
                horizon=spec.horizon,
                constituent_count=len(constituent_manifest["instruments"]),
            )
            summary = _summary_payload(
                spec=spec,
                provider_meta=provider_meta,
                constituent_manifest=constituent_manifest,
                as_of=as_of,
                segments=segments,
                metrics=metrics,
                sidecars=sidecars,
                recorder_uri=recorder_uri,
                recorder_id=recorder.id,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
            )
            risk_analysis = _risk_analysis(summary)
            _write_card(card, summary=summary, metrics=metrics, risk_analysis=risk_analysis)
            recorder.log_artifact(daily_path.as_posix(), artifact_path="soxx_hidden_dispersion")
            for rel_path in sidecars.values():
                recorder.log_artifact((card / rel_path).as_posix(), artifact_path=spec.experiment)
    return summary


def _summary_payload(
    *,
    spec: SoxxHiddenDispersionSpec,
    provider_meta: dict[str, Any],
    constituent_manifest: dict[str, Any],
    as_of: str,
    segments: dict[str, tuple[str, str]],
    metrics: dict[str, pd.DataFrame],
    sidecars: dict[str, str],
    recorder_uri: Path,
    recorder_id: str,
    recorder_local_dir: Path,
) -> dict[str, Any]:
    comparison = metrics["candidate_comparison"]
    baseline = _select_metric(comparison, "weighted_residual_baseline", "soxx_5d", "test")
    combined = _select_metric(comparison, "baseline_plus_interaction", "soxx_5d", "test")
    combined_risk = _select_metric(comparison, "baseline_plus_interaction", "soxx_abs_5d", "test")
    nonoverlap = metrics["nonoverlap"].loc[
        metrics["nonoverlap"]["candidate"].eq("baseline_plus_interaction")
        & metrics["nonoverlap"]["target"].eq("soxx_5d")
        & metrics["nonoverlap"]["segment"].eq("test")
    ]
    offset_corr = pd.to_numeric(nonoverlap["rank_correlation"], errors="coerce").dropna()
    stable_candidate_count = int(
        metrics["stability"]
        .loc[
            metrics["stability"]["target"].eq("soxx_5d")
            & metrics["stability"]["segment_pair"].eq("validation_test"),
            "same_sign",
        ]
        .map(bool)
        .sum()
    )
    feature_test = metrics["feature_metrics"].loc[
        metrics["feature_metrics"]["target"].eq("soxx_5d")
        & metrics["feature_metrics"]["segment"].eq("test")
    ]
    best_feature = (
        feature_test.sort_values(["rank_correlation", "top_bottom_spread_pct"], ascending=False).iloc[0].to_dict()
        if not feature_test.empty
        else {}
    )
    baseline_corr = _finite_or_none(baseline.get("rank_correlation"))
    combined_corr = _finite_or_none(combined.get("rank_correlation"))
    incremental = (
        float(combined_corr - baseline_corr) if baseline_corr is not None and combined_corr is not None else None
    )
    return {
        "experiment": spec.experiment,
        "card_type": SOXX_HIDDEN_DISPERSION_CARD_TYPE,
        "status": "complete",
        "promotion_decision": "not_promoted_missing_pit_holdings",
        "as_of": as_of,
        "provider_uri": provider_meta["provider_uri"],
        "recorder_uri": recorder_uri.resolve().as_posix(),
        "recorder_id": recorder_id,
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "segments": {name: list(window) for name, window in segments.items()},
        "lag_policy": {
            "lag_days": spec.lag_days,
            "label": "future_open_to_open",
            "train_only_beta_and_orientation": True,
        },
        "constituent_data": {
            "source": "qlib.data.D.features",
            "provider_uri": spec.constituent_provider_uri.resolve().as_posix(),
            "instrument_count": len(constituent_manifest["instruments"]),
            "point_in_time_status": "blocked_current_holdings_backfill",
            "limitations": list(constituent_manifest.get("limitations", [])),
        },
        "feature_set": list(metrics["feature_metrics"]["feature"].drop_duplicates()),
        "candidate_diagnostic": {
            "baseline_test_soxx_rank_correlation": baseline_corr,
            "combined_test_soxx_rank_correlation": combined_corr,
            "incremental_test_soxx_rank_correlation": incremental,
            "combined_test_soxx_abs_rank_correlation": _finite_or_none(combined_risk.get("rank_correlation")),
            "nonoverlap_positive_offset_ratio": float(offset_corr.gt(0.0).mean()) if len(offset_corr) else None,
            "nonoverlap_rank_correlation_min": float(offset_corr.min()) if len(offset_corr) else None,
            "nonoverlap_rank_correlation_median": float(offset_corr.median()) if len(offset_corr) else None,
            "nonoverlap_rank_correlation_max": float(offset_corr.max()) if len(offset_corr) else None,
            "stable_candidate_count": stable_candidate_count,
            "best_test_feature": _json_safe(best_feature),
        },
        "qlib_native_api": [
            "qlib.init",
            "qlib.data.D.features",
            "qlib.workflow.R.start",
            "qlib.workflow.Recorder.log_artifact",
        ],
        "qlib_docs": {
            "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
            "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
        },
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "machine_entrypoint": "summary.json",
            "primary_artifact": f"{spec.experiment}.ipynb",
            "recorder_only_artifacts": ["daily_hidden_dispersion_panel.csv"],
        },
        "spec": _json_safe(asdict(spec)),
    }


def _select_metric(frame: pd.DataFrame, candidate: str, target: str, segment: str) -> dict[str, Any]:
    rows = frame.loc[
        frame["candidate"].eq(candidate) & frame["target"].eq(target) & frame["segment"].eq(segment)
    ]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _risk_analysis(summary: dict[str, Any]) -> pd.DataFrame:
    diagnostic = summary["candidate_diagnostic"]
    incremental = diagnostic.get("incremental_test_soxx_rank_correlation")
    offset_ratio = diagnostic.get("nonoverlap_positive_offset_ratio")
    return pd.DataFrame(
        [
            {
                "check": "point_in_time_holdings",
                "status": "fail",
                "value": summary["constituent_data"]["point_in_time_status"],
                "detail": "current holdings are backfilled through history",
            },
            {
                "check": "incremental_directional_evidence",
                "status": "pass" if incremental is not None and float(incremental) > 0.0 else "fail",
                "value": incremental,
                "detail": "test SOXX rank-correlation increment over weighted residual baseline",
            },
            {
                "check": "nonoverlap_offset_stability",
                "status": "pass" if offset_ratio is not None and float(offset_ratio) >= 0.8 else "fail",
                "value": offset_ratio,
                "detail": "positive share across five non-overlapping test offsets",
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
    risk_analysis: pd.DataFrame,
) -> None:
    metrics["feature_metrics"].to_csv(card / summary["sidecars"]["feature_metrics"], index=False)
    metrics["nonoverlap"].to_csv(card / summary["sidecars"]["nonoverlap_feature_metrics"], index=False)
    metrics["candidate_comparison"].to_csv(card / summary["sidecars"]["model_comparison"], index=False)
    risk_analysis.to_csv(card / summary["sidecars"]["risk_analysis"], index=False)
    (card / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
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
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            "  machine_entrypoint: summary.json",
            "",
        ]
    )


def _conclusion(summary: dict[str, Any]) -> str:
    diagnostic = summary["candidate_diagnostic"]
    best = diagnostic.get("best_test_feature", {})
    return (
        "# 结论与反思\n\n"
        f"- what_ran: 使用 qlib SOXX 成分股 OHLCV、SOXX/SOXL open 与 VIX，构建内部离散度、抵消、相关性破裂、领导轮换和成交量 breadth 特征。\n"
        f"- data: as_of={summary['as_of']}，constituents={summary['constituent_data']['instrument_count']}，"
        f"PIT={summary['constituent_data']['point_in_time_status']}。\n"
        f"- direction: baseline_test_rank_corr={_number(diagnostic.get('baseline_test_soxx_rank_correlation')):.4f}，"
        f"combined_test_rank_corr={_number(diagnostic.get('combined_test_soxx_rank_correlation')):.4f}，"
        f"increment={_number(diagnostic.get('incremental_test_soxx_rank_correlation')):.4f}。\n"
        f"- risk: combined_test_abs_return_rank_corr={_number(diagnostic.get('combined_test_soxx_abs_rank_correlation')):.4f}。\n"
        f"- nonoverlap: positive_offset_ratio={_number(diagnostic.get('nonoverlap_positive_offset_ratio')):.4f}，"
        f"rank_corr_range=[{_number(diagnostic.get('nonoverlap_rank_correlation_min')):.4f}, "
        f"{_number(diagnostic.get('nonoverlap_rank_correlation_max')):.4f}]。\n"
        f"- best_feature: {best.get('feature')}，test_rank_corr={_number(best.get('rank_correlation')):.4f}。\n"
        f"- promotion_decision: {summary['promotion_decision']}。\n"
        "- 反思: 成分股高振动而 VIX 平静首先是风险状态，不自动给出方向。只有与残差动量交互后仍有样本外增量，才适合作为方向 meta-label。\n"
        "- next_action: 补齐 point-in-time SOXX holdings 后重跑 validated card；在此之前仅保留诊断，不放宽仓位约束。\n"
        "- risk_notes: 当前成分回填存在 survivorship bias；VIX 是 SPX 隐含波动率，不是半导体板块隐含波动率。\n"
        f"- recorder: {summary['recorder_id']}。\n"
    )


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "hidden-dispersion-title",
                "metadata": {},
                "source": [f"# {summary['experiment']}\n", "\n", _conclusion(summary)],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "hidden-dispersion-run",
                "metadata": {},
                "outputs": [],
                "source": [
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
                    "from src.workflow.soxx_hidden_dispersion import run_soxx_hidden_dispersion_card\n",
                    "\n",
                    "qlib.init(provider_uri=(ROOT / 'data' / 'qlib' / 'soxl').as_posix(), region='us', kernels=1)\n",
                    "_ = D.features(['SOXL', 'SOXX'], ['$open', '$vix'], start_time='2019-01-02', freq='day')\n",
                    "assert callable(R.start)\n",
                    "summary = run_soxx_hidden_dispersion_card(root=ROOT)\n",
                    "assert R.get_uri() == summary['recorder_uri']\n",
                    "summary\n",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "hidden-dispersion-final",
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
    (card / f"{summary['experiment']}.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"qlib store manifest missing: {path.as_posix()}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_root(root: Path | str | None) -> Path:
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _number(value: Any) -> float:
    numeric = _finite_or_none(value)
    return 0.0 if numeric is None else numeric


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value
