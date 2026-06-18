from __future__ import annotations

import json
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
    target_position = _clamp(raw_avg_position, min_position, max_position)
    return ExecutionIntent(
        instrument=instrument,
        target_position=target_position,
        source_model=str(top.get("model", "")),
        source_artifact=str(dashboard_payload.get("artifact_id", "")),
        mode=mode,
        metadata={
            "raw_avg_position": raw_avg_position,
            "min_position": float(min_position),
            "max_position": float(max_position),
            "base_model": str(top.get("base_model", "")),
            "feature_set": str(top.get("feature_set", "")),
            "stability_score": float(top.get("stability_score", 0.0) or 0.0),
            "max_drawdown_pct": float(top.get("max_drawdown_pct", 0.0) or 0.0),
            "dashboard_schema_version": str(dashboard_payload.get("schema_version", "")),
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
