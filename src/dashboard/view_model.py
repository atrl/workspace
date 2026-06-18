from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class DashboardViewFilters:
    models: tuple[str, ...] = ()
    base_models: tuple[str, ...] = ()
    feature_sets: tuple[str, ...] = ()
    regimes: tuple[str, ...] = ()
    windows: tuple[str, ...] = ()
    paper_fidelities: tuple[str, ...] = ()
    attribution_features: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    review_signals: tuple[str, ...] = ()
    decision_hints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "models": list(self.models),
            "base_models": list(self.base_models),
            "feature_sets": list(self.feature_sets),
            "regimes": list(self.regimes),
            "windows": list(self.windows),
            "paper_fidelities": list(self.paper_fidelities),
            "attribution_features": list(self.attribution_features),
            "failure_modes": list(self.failure_modes),
            "review_signals": list(self.review_signals),
            "decision_hints": list(self.decision_hints),
        }


def _as_list(value: Any) -> list[dict[str, Any]]:
    return list(value) if isinstance(value, list) else []


def _string_set(values: Iterable[Any]) -> set[str]:
    return {str(value) for value in values if value is not None and str(value) != ""}


def _sorted_strings(values: Iterable[Any]) -> list[str]:
    return sorted(_string_set(values))


def _base_model(row: dict[str, Any]) -> str:
    if row.get("base_model"):
        return str(row["base_model"])
    return str(row.get("model", "")).split("__", 1)[0]


def _feature_set(row: dict[str, Any]) -> str:
    if row.get("feature_set"):
        return str(row["feature_set"])
    parts = str(row.get("model", "")).split("__", 1)
    return parts[1] if len(parts) == 2 else "all_features"


def _capability_by_model(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("model", "")): row for row in _as_list(payload.get("model_capabilities")) if row.get("model")}


def _ledger_by_model(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("model", "")): row for row in _as_list(payload.get("model_run_ledger")) if row.get("model")}


def _run_record_for_model(ledger: dict[str, dict[str, Any]], *, model: str, base_model: str) -> dict[str, Any]:
    if model in ledger:
        return dict(ledger[model])
    if base_model in ledger:
        return dict(ledger[base_model])
    return {}


def _run_status(record: dict[str, Any]) -> str:
    if not record:
        return "unknown"
    required = [
        "trained",
        "predicted",
        "backtested",
        "objective_evaluated",
        "promotion_evaluated",
        "analysis_recorded",
        "manifested",
    ]
    return "complete" if all(bool(record.get(column, False)) for column in required) else "incomplete"


def _fidelity_for(base_model: str, capabilities: dict[str, dict[str, Any]]) -> str:
    row = capabilities.get(base_model, {})
    value = row.get("paper_fidelity")
    return str(value) if value else ""


def _proxy_models(payload: dict[str, Any]) -> set[str]:
    model_fidelity = payload.get("model_fidelity")
    if isinstance(model_fidelity, dict):
        return _string_set(model_fidelity.get("proxy_models", []))
    return set()


def _rows_for_model(
    rows: Sequence[dict[str, Any]],
    model: str,
    *,
    base_model: str,
    regimes: set[str],
    windows: set[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        row_model = str(row.get("model", ""))
        if row_model not in {model, base_model}:
            continue
        row_regime = str(row.get("regime", "")) if row.get("regime") is not None else ""
        row_window = str(row.get("window", "")) if row.get("window") is not None else ""
        if regimes and row_regime not in regimes:
            continue
        if windows and row_window not in windows:
            continue
        result.append(dict(row))
    return result


def _diagnostics_for_model(
    rows: Sequence[dict[str, Any]],
    model: str,
    *,
    regimes: set[str],
    windows: set[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("model", "")) != model:
            continue
        row_regime = str(row.get("regime", "")) if row.get("regime") is not None else ""
        row_window = str(row.get("window", "")) if row.get("window") is not None else ""
        if regimes and row_regime and row_regime not in regimes:
            continue
        if windows and row_window and row_window not in windows:
            continue
        result.append(dict(row))
    return result


def _segment_filter_match(row: dict[str, Any], *, regimes: set[str], windows: set[str]) -> bool:
    segment = str(row.get("segment", ""))
    row_regime = str(row.get("regime", "")) if row.get("regime") is not None else ""
    row_window = str(row.get("window", "")) if row.get("window") is not None else ""
    if regimes:
        if row_regime:
            if row_regime not in regimes:
                return False
        elif not any(regime in segment for regime in regimes):
            return False
    if windows:
        if row_window:
            if row_window not in windows:
                return False
        elif not any(segment == window or segment.startswith(f"{window}::") for window in windows):
            return False
    return True


def _failure_modes_for_model(
    rows: Sequence[dict[str, Any]],
    model: str,
    *,
    base_model: str,
    regimes: set[str],
    windows: set[str],
    failure_modes: set[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        row_model = str(row.get("model", ""))
        if row_model not in {model, base_model}:
            continue
        row_failure_mode = str(row.get("failure_mode", "")) if row.get("failure_mode") is not None else ""
        if failure_modes and row_failure_mode not in failure_modes:
            continue
        if not _segment_filter_match(row, regimes=regimes, windows=windows):
            continue
        result.append(dict(row))
    return sorted(result, key=lambda item: float(item.get("severity_score", 0.0) or 0.0), reverse=True)


def _model_review_for_model(
    rows: Sequence[dict[str, Any]],
    model: str,
    *,
    base_model: str,
    regimes: set[str],
    windows: set[str],
    review_signals: set[str],
    decision_hints: set[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        row_model = str(row.get("model", ""))
        if row_model not in {model, base_model}:
            continue
        row_signal = str(row.get("review_signal", "")) if row.get("review_signal") is not None else ""
        row_hint = str(row.get("decision_hint", "")) if row.get("decision_hint") is not None else ""
        if review_signals and row_signal not in review_signals:
            continue
        if decision_hints and row_hint not in decision_hints:
            continue
        if not _segment_filter_match(row, regimes=regimes, windows=windows):
            continue
        result.append(dict(row))
    return sorted(result, key=lambda item: float(item.get("priority_rank", 999999) or 999999))


def _attribution_by_model(payload: dict[str, Any], features: set[str]) -> dict[str, list[dict[str, Any]]]:
    attribution = payload.get("attribution") if isinstance(payload.get("attribution"), dict) else {}
    rows = _as_list(attribution.get("top_features_by_model") if isinstance(attribution, dict) else [])
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        feature = str(row.get("feature", ""))
        if features and feature not in features:
            continue
        result.setdefault(str(row.get("model", "")), []).append(dict(row))
    return result


def _controls(payload: dict[str, Any], capabilities: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    top_models = _as_list(payload.get("top_models"))
    attribution = payload.get("attribution") if isinstance(payload.get("attribution"), dict) else {}
    attribution_rows = _as_list(attribution.get("top_features_by_model") if isinstance(attribution, dict) else [])
    ledger_rows = _as_list(payload.get("model_run_ledger"))
    return {
        "models": _sorted_strings(row.get("model") for row in top_models),
        "base_models": _sorted_strings(_base_model(row) for row in top_models),
        "feature_sets": _sorted_strings(_feature_set(row) for row in top_models),
        "regimes": _sorted_strings(row.get("regime") for row in _as_list(payload.get("regime_metrics"))),
        "windows": _sorted_strings(row.get("window") for row in _as_list(payload.get("window_metrics"))),
        "paper_fidelities": _sorted_strings(row.get("paper_fidelity") for row in capabilities.values()),
        "adapter_statuses": _sorted_strings(row.get("adapter_status") for row in ledger_rows),
        "run_statuses": _sorted_strings(_run_status(dict(row)) for row in ledger_rows),
        "attribution_features": _sorted_strings(row.get("feature") for row in attribution_rows),
        "failure_modes": _sorted_strings(row.get("failure_mode") for row in _as_list(payload.get("failure_modes"))),
        "review_signals": _sorted_strings(row.get("review_signal") for row in _as_list(payload.get("model_review_briefing"))),
        "decision_hints": _sorted_strings(row.get("decision_hint") for row in _as_list(payload.get("model_review_briefing"))),
    }


def _matches_card(
    *,
    model: str,
    base_model: str,
    feature_set: str,
    paper_fidelity: str,
    top_attribution_features: list[dict[str, Any]],
    failure_modes: list[dict[str, Any]],
    model_review_briefing: list[dict[str, Any]],
    filters: DashboardViewFilters,
) -> bool:
    if filters.models and model not in filters.models:
        return False
    if filters.base_models and base_model not in filters.base_models:
        return False
    if filters.feature_sets and feature_set not in filters.feature_sets:
        return False
    if filters.paper_fidelities and paper_fidelity not in filters.paper_fidelities:
        return False
    if filters.attribution_features:
        card_features = _string_set(row.get("feature") for row in top_attribution_features)
        if not card_features.intersection(filters.attribution_features):
            return False
    if filters.failure_modes:
        card_failure_modes = _string_set(row.get("failure_mode") for row in failure_modes)
        if not card_failure_modes.intersection(filters.failure_modes):
            return False
    if filters.review_signals:
        card_review_signals = _string_set(row.get("review_signal") for row in model_review_briefing)
        if not card_review_signals.intersection(filters.review_signals):
            return False
    if filters.decision_hints:
        card_decision_hints = _string_set(row.get("decision_hint") for row in model_review_briefing)
        if not card_decision_hints.intersection(filters.decision_hints):
            return False
    return True


def build_dashboard_view_model(
    payload: dict[str, Any],
    *,
    filters: DashboardViewFilters | None = None,
) -> dict[str, Any]:
    active_filters = filters or DashboardViewFilters()
    regimes = set(active_filters.regimes)
    windows = set(active_filters.windows)
    attribution_features = set(active_filters.attribution_features)
    selected_failure_modes = set(active_filters.failure_modes)
    selected_review_signals = set(active_filters.review_signals)
    selected_decision_hints = set(active_filters.decision_hints)
    capabilities = _capability_by_model(payload)
    model_run_ledger = _ledger_by_model(payload)
    proxy_models = _proxy_models(payload)
    attribution = _attribution_by_model(payload, attribution_features)
    top_models = _as_list(payload.get("top_models"))
    regime_metrics = _as_list(payload.get("regime_metrics"))
    window_regime_metrics = _as_list(payload.get("window_regime_metrics"))
    diagnostics = _as_list(payload.get("model_effect_deltas"))
    failure_modes = _as_list(payload.get("failure_modes"))
    model_review_briefing = _as_list(payload.get("model_review_briefing"))

    cards: list[dict[str, Any]] = []
    for row in top_models:
        model = str(row.get("model", ""))
        base_model = _base_model(row)
        feature_set = _feature_set(row)
        paper_fidelity = _fidelity_for(base_model, capabilities)
        run_record = _run_record_for_model(model_run_ledger, model=model, base_model=base_model)
        run_status = _run_status(run_record)
        top_attribution_features = attribution.get(base_model, [])
        card_failure_modes = _failure_modes_for_model(
            failure_modes,
            model,
            base_model=base_model,
            regimes=regimes,
            windows=windows,
            failure_modes=selected_failure_modes,
        )
        card_model_review_briefing = _model_review_for_model(
            model_review_briefing,
            model,
            base_model=base_model,
            regimes=regimes,
            windows=windows,
            review_signals=selected_review_signals,
            decision_hints=selected_decision_hints,
        )
        if not _matches_card(
            model=model,
            base_model=base_model,
            feature_set=feature_set,
            paper_fidelity=paper_fidelity,
            top_attribution_features=top_attribution_features,
            failure_modes=card_failure_modes,
            model_review_briefing=card_model_review_briefing,
            filters=active_filters,
        ):
            continue
        cards.append(
            {
                **dict(row),
                "model": model,
                "base_model": base_model,
                "feature_set": feature_set,
                "paper_fidelity": paper_fidelity,
                "proxy_model": base_model in proxy_models,
                "promotion_blocker": str(capabilities.get(base_model, {}).get("promotion_blocker", "")),
                "run_status": run_status,
                "run_record": run_record,
                "top_attribution_features": top_attribution_features,
                "regime_metrics": _rows_for_model(
                    regime_metrics,
                    model,
                    base_model=base_model,
                    regimes=regimes,
                    windows=set(),
                ),
                "window_regime_metrics": _rows_for_model(
                    window_regime_metrics,
                    model,
                    base_model=base_model,
                    regimes=regimes,
                    windows=windows,
                ),
                "diagnostics": _diagnostics_for_model(diagnostics, model, regimes=regimes, windows=windows),
                "failure_modes": card_failure_modes,
                "model_review_briefing": card_model_review_briefing,
            }
        )

    ordered_failure_modes = sorted(
        [dict(row) for row in failure_modes],
        key=lambda item: float(item.get("severity_score", 0.0) or 0.0),
        reverse=True,
    )
    ordered_review_briefing = sorted(
        [dict(row) for row in model_review_briefing],
        key=lambda item: float(item.get("priority_rank", 999999) or 999999),
    )
    return {
        "schema_version": "dashboard.view_model.v1",
        "source_schema_version": str(payload.get("schema_version", "")),
        "source_artifact_id": str(payload.get("artifact_id", "")),
        "filters_applied": active_filters.to_dict(),
        "controls": _controls(payload, capabilities),
        "summary": dict(payload.get("summary", {})) if isinstance(payload.get("summary"), dict) else {},
        "analysis_overview": dict(payload.get("analysis_overview", {}))
        if isinstance(payload.get("analysis_overview"), dict)
        else {},
        "model_fidelity": dict(payload.get("model_fidelity", {})) if isinstance(payload.get("model_fidelity"), dict) else {},
        "model_run_coverage": dict(payload.get("model_run_coverage", {}))
        if isinstance(payload.get("model_run_coverage"), dict)
        else {},
        "comparison_table": [dict(row) for row in top_models],
        "failure_mode_summary": {
            "count": int(len(failure_modes)),
            "top_failure_mode": str(ordered_failure_modes[0].get("failure_mode", "")) if ordered_failure_modes else "",
            "top_model": str(ordered_failure_modes[0].get("model", "")) if ordered_failure_modes else "",
            "top_segment": str(ordered_failure_modes[0].get("segment", "")) if ordered_failure_modes else "",
            "top_severity_score": float(ordered_failure_modes[0].get("severity_score", 0.0) or 0.0)
            if ordered_failure_modes
            else 0.0,
        },
        "review_briefing_summary": {
            "count": int(len(model_review_briefing)),
            "top_review_signal": str(ordered_review_briefing[0].get("review_signal", ""))
            if ordered_review_briefing
            else "",
            "top_decision_hint": str(ordered_review_briefing[0].get("decision_hint", ""))
            if ordered_review_briefing
            else "",
            "top_model": str(ordered_review_briefing[0].get("model", "")) if ordered_review_briefing else "",
            "top_segment": str(ordered_review_briefing[0].get("segment", "")) if ordered_review_briefing else "",
            "top_priority_rank": int(float(ordered_review_briefing[0].get("priority_rank", 0) or 0))
            if ordered_review_briefing
            else 0,
        },
        "model_cards": cards,
    }


def load_dashboard_view_model(
    dashboard_export_path: str | Path,
    *,
    filters: DashboardViewFilters | None = None,
) -> dict[str, Any]:
    path = Path(dashboard_export_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    view = build_dashboard_view_model(payload, filters=filters)
    view["source_path"] = str(path)
    return view
