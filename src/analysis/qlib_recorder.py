from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .exports import AnalysisReport


QLIB_RECORDER_SCHEMA_VERSION = "analysis.qlib_recorder.v1"

DEFAULT_QLIB_ARTIFACT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "qlib_portfolio_report": (
        "portfolio_analysis/report_normal_1day.pkl",
        "report_normal_1day.pkl",
        "report_normal_1day.csv",
        "report.pkl",
        "report.csv",
    ),
    "qlib_risk_analysis": (
        "portfolio_analysis/risk_analysis_1day.pkl",
        "risk_analysis_1day.pkl",
        "risk_analysis_1day.csv",
        "risk_analysis.pkl",
        "risk_analysis.csv",
    ),
    "qlib_positions": (
        "portfolio_analysis/positions_normal_1day.pkl",
        "positions_normal_1day.pkl",
        "positions_normal_1day.csv",
        "positions.pkl",
        "positions.csv",
    ),
    "qlib_orders": (
        "portfolio_analysis/orders_normal_1day.pkl",
        "orders_normal_1day.pkl",
        "orders_normal_1day.csv",
        "orders.pkl",
        "orders.csv",
    ),
    "qlib_indicator_analysis": (
        "portfolio_analysis/indicator_analysis_1day.pkl",
        "indicator_analysis_1day.pkl",
        "indicator_analysis_1day.csv",
        "indicator_analysis.pkl",
        "indicator_analysis.csv",
    ),
}


@dataclass(frozen=True)
class QlibRecorderAnalysis:
    tables: Mapping[str, pd.DataFrame]
    summary: dict[str, Any]


def _is_recorder_like(source: object) -> bool:
    return callable(getattr(source, "load_object", None))


def _candidate_map(artifact_candidates: Mapping[str, Sequence[str]] | None) -> dict[str, tuple[str, ...]]:
    result = {key: tuple(values) for key, values in DEFAULT_QLIB_ARTIFACT_CANDIDATES.items()}
    for key, values in (artifact_candidates or {}).items():
        result[str(key)] = tuple(str(value) for value in values)
    return result


def _load_from_recorder(recorder: object, candidates: Sequence[str]) -> tuple[object, str] | None:
    for name in candidates:
        try:
            return recorder.load_object(name), name
        except Exception:
            continue
    return None


def _read_local_artifact(path: Path) -> object:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported qlib artifact file type: {path}")


def _load_from_artifact_dir(artifact_dir: Path, candidates: Sequence[str]) -> tuple[object, str] | None:
    for name in candidates:
        path = artifact_dir / name
        if path.exists():
            return _read_local_artifact(path), name
    return None


def _frame_from_object(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.rename("value").to_frame()
    if isinstance(value, Mapping):
        return pd.DataFrame([dict(value)])
    return pd.DataFrame(value)


def _date_column_name(frame: pd.DataFrame) -> str | None:
    for column in ["date", "datetime", "time"]:
        if column in frame.columns:
            return column
    return None


def _normalise_portfolio_report(value: object, *, model_name: str, source_artifact: str) -> pd.DataFrame:
    frame = _frame_from_object(value)
    if _date_column_name(frame) is None and frame.index.name is not None:
        frame = frame.reset_index()
    elif _date_column_name(frame) is None:
        frame = frame.reset_index().rename(columns={"index": "date"})

    date_column = _date_column_name(frame) or "date"
    result = pd.DataFrame()
    result["date"] = pd.to_datetime(frame[date_column])
    result["model"] = model_name

    strategy_source = "return" if "return" in frame.columns else "strategy_return"
    benchmark_source = "bench" if "bench" in frame.columns else "benchmark_return"
    result["strategy_return"] = frame[strategy_source].astype(float) if strategy_source in frame.columns else 0.0
    result["benchmark_return"] = frame[benchmark_source].astype(float) if benchmark_source in frame.columns else 0.0
    result["excess_return"] = result["strategy_return"] - result["benchmark_return"]
    result["nav"] = (1.0 + result["strategy_return"].astype(float)).cumprod()
    result["benchmark_nav"] = (1.0 + result["benchmark_return"].astype(float)).cumprod()
    if "cost" in frame.columns:
        result["cost"] = frame["cost"].astype(float)
    result["source_artifact"] = source_artifact
    return result.sort_values("date").reset_index(drop=True)


def _flatten_risk_dict(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, item in value.items():
        if isinstance(item, Mapping):
            for subkey, subvalue in item.items():
                rows.append({"metric": str(subkey), "value": subvalue, "return_kind": str(key)})
        else:
            rows.append({"metric": str(key), "value": item})
    return rows


def _normalise_risk_analysis(value: object, *, model_name: str, source_artifact: str) -> pd.DataFrame:
    if isinstance(value, Mapping):
        frame = pd.DataFrame(_flatten_risk_dict(value))
    else:
        raw = _frame_from_object(value)
        if {"metric", "value"}.issubset(raw.columns):
            frame = raw.loc[:, [column for column in raw.columns if column in {"metric", "value", "return_kind"}]].copy()
        elif len(raw.index) > 0 and len(raw.columns) == 1:
            frame = raw.reset_index()
            frame.columns = ["metric", "value"]
        elif len(raw) == 1:
            frame = raw.melt(var_name="metric", value_name="value")
        else:
            frame = raw.reset_index().melt(id_vars=["index"], var_name="metric", value_name="value")
            frame["metric"] = frame["index"].astype(str) + "::" + frame["metric"].astype(str)
            frame = frame.drop(columns=["index"])

    frame["model"] = model_name
    frame["source_artifact"] = source_artifact
    if "return_kind" not in frame.columns:
        frame["return_kind"] = "qlib_risk"
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.loc[:, ["model", "metric", "value", "return_kind", "source_artifact"]].reset_index(drop=True)


def _normalise_generic_table(value: object, *, model_name: str, source_artifact: str) -> pd.DataFrame:
    frame = _frame_from_object(value)
    if _date_column_name(frame) is None and frame.index.name is not None:
        frame = frame.reset_index()
    frame["model"] = model_name
    frame["source_artifact"] = source_artifact
    return frame.reset_index(drop=True)


def _normalise_artifact(key: str, value: object, *, model_name: str, source_artifact: str) -> pd.DataFrame:
    if key == "qlib_portfolio_report":
        return _normalise_portfolio_report(value, model_name=model_name, source_artifact=source_artifact)
    if key == "qlib_risk_analysis":
        return _normalise_risk_analysis(value, model_name=model_name, source_artifact=source_artifact)
    return _normalise_generic_table(value, model_name=model_name, source_artifact=source_artifact)


def load_qlib_recorder_analysis(
    source: object,
    *,
    model_name: str = "qlib_recorder",
    artifact_candidates: Mapping[str, Sequence[str]] | None = None,
) -> QlibRecorderAnalysis:
    candidates = _candidate_map(artifact_candidates)
    source_type = "recorder" if _is_recorder_like(source) else "artifact_dir"
    artifact_dir = None if source_type == "recorder" else Path(source)

    tables: dict[str, pd.DataFrame] = {}
    loaded: dict[str, str] = {}
    missing: list[str] = []
    for key, names in candidates.items():
        loaded_object = (
            _load_from_recorder(source, names)
            if source_type == "recorder"
            else _load_from_artifact_dir(artifact_dir or Path(), names)
        )
        if loaded_object is None:
            missing.append(key)
            continue
        value, artifact_name = loaded_object
        tables[key] = _normalise_artifact(key, value, model_name=model_name, source_artifact=artifact_name)
        loaded[key] = artifact_name

    required = {"qlib_portfolio_report", "qlib_risk_analysis"}
    missing_required = sorted(required - set(tables))
    if missing_required:
        raise ValueError(f"missing required qlib recorder artifacts: {missing_required}")

    summary = {
        "schema_version": QLIB_RECORDER_SCHEMA_VERSION,
        "source_type": source_type,
        "model": model_name,
        "loaded_artifacts": loaded,
        "missing_artifact_keys": missing_required,
        "missing_optional_artifact_keys": sorted(set(missing) - required),
        "table_counts": {key: int(len(table)) for key, table in sorted(tables.items())},
        "notes": "Qlib recorder artifacts are normalized into stable src/analysis tables before dashboard/notebook consumption.",
    }
    return QlibRecorderAnalysis(tables=tables, summary=summary)


def build_qlib_recorder_analysis_report(
    *,
    artifact_id: str,
    recorder_analysis: QlibRecorderAnalysis,
    summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AnalysisReport:
    summary_payload = dict(summary or {})
    summary_payload.setdefault("status", "qlib_recorder_loaded")
    summary_payload["qlib_recorder_summary"] = recorder_analysis.summary
    summary_payload["analysis_layers"] = list(recorder_analysis.tables.keys())

    metadata_payload = {
        "source_type": recorder_analysis.summary["source_type"],
        "loaded_artifacts": recorder_analysis.summary["loaded_artifacts"],
        **dict(metadata or {}),
    }
    return AnalysisReport(
        artifact_id=artifact_id,
        summary=summary_payload,
        tables=recorder_analysis.tables,
        metadata=metadata_payload,
    )
