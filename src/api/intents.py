from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from src.core import (
    build_dashboard_manifest_source_contract,
    build_dashboard_readiness_overview,
    build_objective_loss_readiness_gate,
)

from .contracts import BrokerAdapter, assert_paper_sim_mode
from .paper import AccountSnapshot


@dataclass(frozen=True)
class ExecutionIntent:
    instrument: str
    target_position: float
    source_model: str
    source_artifact: str
    mode: str = "paper"
    schema_version: str = "api.execution_intent.v1"
    guardrail: str = "v1 只允许 paper/sim，不连接真实 broker，不触发真实下单。"
    metadata: dict[str, Any] | None = None

    def with_mode(self, mode: str) -> "ExecutionIntent":
        return replace(self, mode=mode)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata or {})
        return payload


@dataclass(frozen=True)
class PaperExecutionResult:
    intent: ExecutionIntent
    snapshot: AccountSnapshot
    broker_name: str
    status: str = "paper_executed"
    schema_version: str = "api.paper_execution_result.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "broker_name": self.broker_name,
            "intent": self.intent.to_dict(),
            "snapshot": asdict(self.snapshot),
        }


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(value), float(upper)))


def _clean_metadata_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _int_field(row: dict[str, Any], key: str) -> int:
    try:
        return int(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None and str(item) != ""]


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _top_promotion_for_model(dashboard_payload: dict[str, Any], model: str) -> dict[str, Any]:
    objective = dashboard_payload.get("objective")
    if not isinstance(objective, dict):
        return {}
    rows = objective.get("promotion_scores")
    if not isinstance(rows, list):
        return {}
    candidates = [row for row in rows if isinstance(row, dict) and str(row.get("model", "")) == model]
    if not candidates:
        return {}
    return sorted(
        candidates,
        key=lambda row: (
            int(float(row.get("rank", 999999) or 999999)),
            -float(row.get("promotion_score", 0.0) or 0.0),
        ),
    )[0]


def _analysis_context(dashboard_payload: dict[str, Any]) -> dict[str, Any]:
    overview = dashboard_payload.get("analysis_overview")
    if not isinstance(overview, dict):
        return {}
    weakness = overview.get("top_weakness")
    top_weakness = weakness if isinstance(weakness, dict) else {}
    gaps = overview.get("largest_relative_return_spread_segments")
    top_gap = gaps[0] if isinstance(gaps, list) and gaps and isinstance(gaps[0], dict) else {}
    return {
        "analysis_overview_schema_version": str(overview.get("schema_version", "")),
        "analysis_top_weakness_model": str(top_weakness.get("model", "")),
        "analysis_top_weakness_segment": str(top_weakness.get("segment", "")),
        "analysis_top_weakness_decision_hint": str(top_weakness.get("decision_hint", "")),
        "analysis_top_gap_segment": str(top_gap.get("segment", "")),
        "analysis_top_gap_best_model": str(top_gap.get("best_relative_return_model", "")),
        "analysis_top_gap_worst_model": str(top_gap.get("worst_relative_return_model", "")),
        "analysis_top_gap_spread": float(top_gap.get("relative_return_pct_spread", 0.0) or 0.0),
    }


def _model_difference_context(dashboard_payload: dict[str, Any], model: str) -> dict[str, Any]:
    rows = dashboard_payload.get("model_difference_explanations")
    if not isinstance(rows, list) or not rows:
        return {}
    base_model = model.split("__", 1)[0]
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        compared_models = {
            str(row.get("baseline_model", "")),
            str(row.get("winner_model", "")),
            str(row.get("loser_model", "")),
        }
        if model in compared_models or base_model in compared_models:
            candidates.append(row)
    if not candidates:
        return {}
    ordered = sorted(
        candidates,
        key=lambda row: (
            int(float(row.get("priority_rank", 999999) or 999999)),
            -float(row.get("relative_return_pct_spread", 0.0) or 0.0),
        ),
    )
    top = ordered[0]
    loser_model = str(top.get("loser_model", ""))
    decision_hint = str(top.get("decision_hint", ""))
    return {
        "model_difference_schema_version": str(top.get("schema_version", "")),
        "model_difference_top_signal": str(top.get("explanation_signal", "")),
        "model_difference_top_decision_hint": decision_hint,
        "model_difference_top_segment": str(top.get("segment", "")),
        "model_difference_top_baseline_model": str(top.get("baseline_model", "")),
        "model_difference_top_winner_model": str(top.get("winner_model", "")),
        "model_difference_top_loser_model": loser_model,
        "model_difference_top_spread": float(top.get("relative_return_pct_spread", 0.0) or 0.0),
        "model_difference_top_priority_rank": int(float(top.get("priority_rank", 0) or 0)),
        "model_difference_marks_source_loser": bool(
            decision_hint == "block_or_rework_loser" and loser_model in {model, base_model}
        ),
    }


def _backtest_context(dashboard_payload: dict[str, Any]) -> dict[str, Any]:
    backtest = dashboard_payload.get("backtest")
    if not isinstance(backtest, dict):
        return {}
    config = backtest.get("config")
    if not isinstance(config, dict):
        return {}
    cost_model = config.get("cost_model")
    if not isinstance(cost_model, dict):
        cost_model = {}
    splits = backtest.get("walk_forward_splits")
    return {
        "backtest_schema_version": str(config.get("schema_version", "")),
        "backtest_walk_forward_fold_count": int(len(splits) if isinstance(splits, list) else 0),
        "backtest_open_cost_bps": float(cost_model.get("open_cost_bps", 0.0) or 0.0),
        "backtest_close_cost_bps": float(cost_model.get("close_cost_bps", 0.0) or 0.0),
        "backtest_slippage_bps": float(cost_model.get("slippage_bps", 0.0) or 0.0),
    }


def _run_record_for_model(dashboard_payload: dict[str, Any], model: str) -> dict[str, Any]:
    rows = dashboard_payload.get("model_run_ledger")
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and str(row.get("model", "")) == model:
            return dict(row)
    base_model = model.split("__", 1)[0]
    for row in rows:
        if isinstance(row, dict) and str(row.get("model", "")) == base_model:
            return dict(row)
    return {}


def _run_status(record: dict[str, Any]) -> str:
    if not record:
        return "unknown"
    required = (
        "trained",
        "predicted",
        "backtested",
        "objective_evaluated",
        "promotion_evaluated",
        "analysis_recorded",
        "manifested",
    )
    return "complete" if all(bool(record.get(column, False)) for column in required) else "incomplete"


def _model_run_context(dashboard_payload: dict[str, Any], model: str) -> dict[str, Any]:
    coverage = dashboard_payload.get("model_run_coverage")
    coverage_payload = coverage if isinstance(coverage, dict) else {}
    missing = coverage_payload.get("missing_models")
    missing_models = [str(value) for value in missing] if isinstance(missing, list) else []
    record = _run_record_for_model(dashboard_payload, model)
    status = _run_status(record)
    return {
        "run_status": status,
        "run_trained": bool(record.get("trained", False)) if record else False,
        "run_predicted": bool(record.get("predicted", False)) if record else False,
        "run_backtested": bool(record.get("backtested", False)) if record else False,
        "run_objective_evaluated": bool(record.get("objective_evaluated", False)) if record else False,
        "run_promotion_evaluated": bool(record.get("promotion_evaluated", False)) if record else False,
        "run_analysis_recorded": bool(record.get("analysis_recorded", False)) if record else False,
        "run_manifested": bool(record.get("manifested", False)) if record else False,
        "model_run_schema_version": str(coverage_payload.get("schema_version", "")),
        "model_run_all_completed": bool(coverage_payload.get("all_models_completed", False)),
        "model_run_missing_models": ",".join(missing_models),
    }


def _requested_model_context(dashboard_payload: dict[str, Any]) -> dict[str, Any]:
    contract = dashboard_payload.get("requested_model_contract")
    if not isinstance(contract, dict) or not contract:
        return {}
    missing_models = _dedupe(
        _string_list(contract.get("missing_from_registry")) + _string_list(contract.get("missing_from_ledger"))
    )
    incomplete_models = _string_list(contract.get("incomplete_requested_models"))
    return {
        "requested_model_schema_version": str(contract.get("schema_version", "")),
        "requested_model_count": _int_field(contract, "requested_model_count"),
        "requested_model_registered_count": _int_field(contract, "registered_model_count"),
        "requested_model_ledger_count": _int_field(contract, "ledger_model_count"),
        "requested_model_completed_count": _int_field(contract, "completed_requested_model_count"),
        "requested_model_all_completed": bool(contract.get("all_requested_models_completed", False)),
        "requested_model_missing_count": int(len(missing_models)),
        "requested_model_incomplete_count": int(len(incomplete_models)),
        "requested_model_missing_models": ",".join(missing_models),
        "requested_model_incomplete_models": ",".join(incomplete_models),
    }


def _dashboard_manifest_context(dashboard_payload: dict[str, Any]) -> dict[str, Any]:
    contract = build_dashboard_manifest_source_contract(dashboard_payload.get("manifest_source"))
    return {
        "dashboard_manifest_required": True,
        "dashboard_manifest_status": str(contract.get("status", "")),
        "dashboard_manifest_kind": str(contract.get("kind", "")),
        "dashboard_manifest_path": str(contract.get("path", "")),
        "dashboard_manifest_hash": str(contract.get("manifest_hash", "")),
        "dashboard_manifest_producer": str(contract.get("producer", "")),
        "dashboard_manifest_allowed_kinds": ",".join(str(kind) for kind in contract.get("allowed_kinds", [])),
        "dashboard_manifest_error": str(contract.get("error", "")),
    }


def _readiness_context(dashboard_payload: dict[str, Any]) -> dict[str, Any]:
    overview = build_dashboard_readiness_overview(dashboard_payload)
    gates = overview.get("gates")
    gate_rows = gates if isinstance(gates, list) else []
    if not gate_rows:
        return {}
    gate_names = [str(gate.get("gate", "")) for gate in gate_rows if isinstance(gate, dict)]
    review_gates = [
        str(gate.get("gate", ""))
        for gate in gate_rows
        if isinstance(gate, dict) and str(gate.get("status", "")) == "review"
    ]
    block_gates = [
        str(gate.get("gate", ""))
        for gate in gate_rows
        if isinstance(gate, dict) and str(gate.get("status", "")) == "block"
    ]
    return {
        "readiness_overview_schema_version": str(overview.get("schema_version", "")),
        "readiness_overview_status": str(overview.get("status", "")),
        "readiness_overview_gate_count": _int_field(overview, "gate_count"),
        "readiness_overview_pass_count": _int_field(overview, "pass_count"),
        "readiness_overview_review_count": _int_field(overview, "review_count"),
        "readiness_overview_block_count": _int_field(overview, "block_count"),
        "readiness_overview_gates": ",".join(gate_names),
        "readiness_overview_review_gates": ",".join(review_gates),
        "readiness_overview_block_gates": ",".join(block_gates),
    }


def _objective_loss_context(dashboard_payload: dict[str, Any]) -> dict[str, Any]:
    objective = dashboard_payload.get("objective")
    objective_payload = objective if isinstance(objective, dict) else {}
    production_loss = objective_payload.get("production_loss")
    losses = production_loss if isinstance(production_loss, dict) and production_loss else objective_payload.get("losses")
    gate = build_objective_loss_readiness_gate(losses)
    if not gate:
        return {}
    return {
        "objective_loss_schema_version": str(gate.get("source_schema_version", "")),
        "objective_loss_metric_source_schema_version": str(gate.get("metric_source_schema_version", "")),
        "objective_loss_status": str(gate.get("status", "")),
        "objective_loss_detail": str(gate.get("detail", "")),
        "objective_loss_optimization_direction": str(gate.get("optimization_direction", "")),
        "objective_loss_reward_surface": str(gate.get("reward_surface", "")),
        "objective_loss_row_count": _int_field(gate, "row_count"),
        "objective_loss_rank_group_count": _int_field(gate, "rank_group_count"),
        "objective_loss_component_count": _int_field(gate, "loss_component_count"),
        "objective_loss_top_component": str(gate.get("top_component", "")),
        "objective_loss_top_value": float(gate.get("top_value", 0.0) or 0.0),
        "objective_loss_readiness_scope": str(gate.get("readiness_scope", "")),
        "objective_loss_model": str(gate.get("model", "")),
    }


def build_target_position_intent_from_dashboard(
    dashboard_payload: dict[str, Any],
    *,
    instrument: str,
    min_position: float = 0.0,
    max_position: float = 1.0,
    mode: str = "paper",
) -> ExecutionIntent:
    mode = assert_paper_sim_mode(mode, surface="v1 execution intent")
    top_models = dashboard_payload.get("top_models") or []
    if not top_models:
        raise ValueError("dashboard payload has no top_models")
    top = top_models[0]
    raw_avg_position = float(top.get("avg_position", 0.0))
    source_model = str(top.get("model", ""))
    promotion = _top_promotion_for_model(dashboard_payload, source_model)
    analysis = _analysis_context(dashboard_payload)
    model_difference = _model_difference_context(dashboard_payload, source_model)
    run_context = _model_run_context(dashboard_payload, source_model)
    requested_context = _requested_model_context(dashboard_payload)
    dashboard_manifest_context = _dashboard_manifest_context(dashboard_payload)
    readiness_context = _readiness_context(dashboard_payload)
    objective_loss_context = _objective_loss_context(dashboard_payload)
    blocked_by_promotion = str(promotion.get("promotion_decision", "")) == "blocked"
    blocked_by_review = (
        analysis.get("analysis_top_weakness_model") == source_model
        and analysis.get("analysis_top_weakness_decision_hint") == "block_or_rework"
    )
    blocked_by_incomplete_run = run_context.get("run_status") == "incomplete"
    blocked_by_readiness = int(readiness_context.get("readiness_overview_block_count", 0) or 0) > 0
    blocked_by_model_difference = bool(model_difference.get("model_difference_marks_source_loser", False))
    execution_gate = (
        "blocked_to_min_position"
        if (
            blocked_by_promotion
            or blocked_by_review
            or blocked_by_incomplete_run
            or blocked_by_readiness
            or blocked_by_model_difference
        )
        else "paper_review"
    )
    target_position = float(min_position) if execution_gate == "blocked_to_min_position" else _clamp(
        raw_avg_position,
        min_position,
        max_position,
    )
    return ExecutionIntent(
        instrument=instrument,
        target_position=target_position,
        source_model=source_model,
        source_artifact=str(dashboard_payload.get("artifact_id", "")),
        mode=mode,
        metadata={
            "execution_gate": execution_gate,
            "raw_avg_position": raw_avg_position,
            "min_position": float(min_position),
            "max_position": float(max_position),
            "base_model": str(top.get("base_model", "")),
            "feature_set": str(top.get("feature_set", "")),
            "stability_score": float(top.get("stability_score", 0.0) or 0.0),
            "max_drawdown_pct": float(top.get("max_drawdown_pct", 0.0) or 0.0),
            "promotion_decision": str(promotion.get("promotion_decision", "")),
            "promotion_score": float(promotion.get("promotion_score", 0.0) or 0.0),
            "promotion_blockers": _clean_metadata_string(promotion.get("promotion_blockers", "")),
            "dashboard_schema_version": str(dashboard_payload.get("schema_version", "")),
            **dashboard_manifest_context,
            **objective_loss_context,
            **run_context,
            **requested_context,
            **readiness_context,
            **analysis,
            **model_difference,
            **_backtest_context(dashboard_payload),
        },
    )


def execute_paper_intent(
    intent: ExecutionIntent,
    *,
    broker: BrokerAdapter,
    price: float,
) -> PaperExecutionResult:
    assert_paper_sim_mode(intent.mode, surface="v1 API adapter")
    ok, detail = broker.healthcheck()
    if not ok:
        raise RuntimeError(detail)
    snapshot = broker.place_target_position(target_position=intent.target_position, price=price)
    return PaperExecutionResult(intent=intent, snapshot=snapshot, broker_name=broker.broker_name)


def write_intent_ledger(
    path: str | Path,
    *,
    intent: ExecutionIntent,
    result: PaperExecutionResult | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "api.intent_ledger.v1",
        "intent": intent.to_dict(),
        "execution_result": result.to_dict() if result is not None else None,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
