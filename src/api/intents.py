from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .paper import AccountSnapshot, PaperBrokerAdapter


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


def build_target_position_intent_from_dashboard(
    dashboard_payload: dict[str, Any],
    *,
    instrument: str,
    min_position: float = 0.0,
    max_position: float = 1.0,
    mode: str = "paper",
) -> ExecutionIntent:
    if mode not in {"paper", "sim"}:
        raise ValueError("v1 only supports paper/sim execution intent mode")
    top_models = dashboard_payload.get("top_models") or []
    if not top_models:
        raise ValueError("dashboard payload has no top_models")
    top = top_models[0]
    raw_avg_position = float(top.get("avg_position", 0.0))
    source_model = str(top.get("model", ""))
    promotion = _top_promotion_for_model(dashboard_payload, source_model)
    analysis = _analysis_context(dashboard_payload)
    blocked_by_promotion = str(promotion.get("promotion_decision", "")) == "blocked"
    blocked_by_review = (
        analysis.get("analysis_top_weakness_model") == source_model
        and analysis.get("analysis_top_weakness_decision_hint") == "block_or_rework"
    )
    execution_gate = "blocked_to_min_position" if blocked_by_promotion or blocked_by_review else "paper_review"
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
            **analysis,
            **_backtest_context(dashboard_payload),
        },
    )


def execute_paper_intent(
    intent: ExecutionIntent,
    *,
    broker: PaperBrokerAdapter,
    price: float,
) -> PaperExecutionResult:
    if intent.mode not in {"paper", "sim"}:
        raise ValueError("v1 API adapter only supports paper/sim execution")
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
