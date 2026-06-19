from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

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


def _readiness_gates(dashboard_payload: dict[str, Any]) -> list[dict[str, str]]:
    gates: list[dict[str, str]] = []
    production_health = dashboard_payload.get("production_health")
    if isinstance(production_health, dict) and production_health:
        missing_count = len(_string_list(production_health.get("missing_required_paths")))
        healthy = bool(production_health.get("is_healthy", False)) and missing_count == 0
        gates.append({"gate": "production_health", "status": "pass" if healthy else "block"})

    notebook_readiness = dashboard_payload.get("notebook_card_readiness")
    if isinstance(notebook_readiness, dict) and notebook_readiness:
        summary = notebook_readiness.get("summary")
        notebook_summary = summary if isinstance(summary, dict) else {}
        published_count = _int_field(notebook_summary, "production_published_count")
        template_count = _int_field(notebook_summary, "production_template_count")
        missing_count = _int_field(notebook_summary, "production_manifest_missing_count")
        mismatch_count = _int_field(notebook_summary, "production_registry_mismatch_count")
        if missing_count > 0 or mismatch_count > 0 or published_count == 0:
            status = "block"
        elif template_count > 0:
            status = "review"
        else:
            status = "pass"
        gates.append({"gate": "notebook_cards", "status": status})

    requested_contract = dashboard_payload.get("requested_model_contract")
    if isinstance(requested_contract, dict) and requested_contract:
        completed_count = _int_field(requested_contract, "completed_requested_model_count")
        all_completed = bool(requested_contract.get("all_requested_models_completed", False))
        status = "pass" if all_completed else "review" if completed_count > 0 else "block"
        gates.append({"gate": "requested_models", "status": status})

    model_run_coverage = dashboard_payload.get("model_run_coverage")
    if isinstance(model_run_coverage, dict) and model_run_coverage:
        all_completed = bool(model_run_coverage.get("all_models_completed", False))
        gates.append({"gate": "model_runs", "status": "pass" if all_completed else "review"})

    return gates


def _readiness_context(dashboard_payload: dict[str, Any]) -> dict[str, Any]:
    gates = _readiness_gates(dashboard_payload)
    if not gates:
        return {}
    pass_gates = [gate["gate"] for gate in gates if gate["status"] == "pass"]
    review_gates = [gate["gate"] for gate in gates if gate["status"] == "review"]
    block_gates = [gate["gate"] for gate in gates if gate["status"] == "block"]
    status = "blocked" if block_gates else "review" if review_gates else "ready"
    return {
        "readiness_overview_schema_version": "dashboard.readiness_overview.v1",
        "readiness_overview_status": status,
        "readiness_overview_gate_count": int(len(gates)),
        "readiness_overview_pass_count": int(len(pass_gates)),
        "readiness_overview_review_count": int(len(review_gates)),
        "readiness_overview_block_count": int(len(block_gates)),
        "readiness_overview_gates": ",".join(gate["gate"] for gate in gates),
        "readiness_overview_review_gates": ",".join(review_gates),
        "readiness_overview_block_gates": ",".join(block_gates),
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
    run_context = _model_run_context(dashboard_payload, source_model)
    requested_context = _requested_model_context(dashboard_payload)
    readiness_context = _readiness_context(dashboard_payload)
    blocked_by_promotion = str(promotion.get("promotion_decision", "")) == "blocked"
    blocked_by_review = (
        analysis.get("analysis_top_weakness_model") == source_model
        and analysis.get("analysis_top_weakness_decision_hint") == "block_or_rework"
    )
    blocked_by_incomplete_run = run_context.get("run_status") == "incomplete"
    blocked_by_readiness = int(readiness_context.get("readiness_overview_block_count", 0) or 0) > 0
    execution_gate = (
        "blocked_to_min_position"
        if blocked_by_promotion or blocked_by_review or blocked_by_incomplete_run or blocked_by_readiness
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
            **run_context,
            **requested_context,
            **readiness_context,
            **analysis,
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
