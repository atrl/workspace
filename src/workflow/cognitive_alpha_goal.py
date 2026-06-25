from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.api import PaperBroker, TargetPosition
from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.schedule import BarkNotifier, TradeNotification, trade_notification_from_order_intents


COGNITIVE_ALPHA_GOAL_STAGE_ORDER = [
    "data_gate",
    "candidate_generation",
    "quality_checker",
    "fitness_evaluation",
    "evolution_loop",
    "model_and_position",
    "research_card",
    "signal_push",
]
COGNITIVE_ALPHA_PAPER_REFERENCE = "arxiv:2511.18850"


@dataclass(frozen=True)
class CognitiveAlphaGoalRunnerSpec:
    run_id: str
    objective: str
    state_path: Path
    manifest_path: Path
    required_max_datetime: str | None = None
    resume: bool = False
    paper_reference: str = COGNITIVE_ALPHA_PAPER_REFERENCE


class CognitiveAlphaGoalRunner:
    def __init__(
        self,
        spec: CognitiveAlphaGoalRunnerSpec,
        *,
        data_gate: Callable[[], list[dict[str, Any]]],
        evolution: Callable[..., dict[str, Any]],
        position: Callable[..., dict[str, Any]],
        research_card: Callable[..., dict[str, Any]],
        signal_push: Callable[..., list[dict[str, Any]]],
    ) -> None:
        self.spec = spec
        self.data_gate = data_gate
        self.evolution = evolution
        self.position = position
        self.research_card = research_card
        self.signal_push = signal_push

    def run(self) -> dict[str, Any]:
        state = self._load_state()
        completed = list(state.get("completed_stages", []))
        outputs = dict(state.get("stage_outputs", {}))

        if "data_gate" not in completed:
            manifests = self.data_gate()
            self._validate_data_manifests(manifests)
            outputs["data_gate"] = {"manifests": _jsonable(manifests)}
            completed = _append_stages(completed, ["data_gate"])
            self._write_state(status="running", completed_stages=completed, stage_outputs=outputs)
        else:
            manifests = list(outputs.get("data_gate", {}).get("manifests", []))
            self._validate_data_manifests(manifests)

        evolution_stages = ["candidate_generation", "quality_checker", "fitness_evaluation", "evolution_loop"]
        if not set(evolution_stages).issubset(completed):
            evolution_result = _jsonable(self.evolution(manifests))
            outputs["evolution"] = evolution_result
            completed = _append_stages(completed, evolution_stages)
            self._write_state(status="running", completed_stages=completed, stage_outputs=outputs)
        else:
            evolution_result = dict(outputs.get("evolution", {}))

        if "model_and_position" not in completed:
            position_result = _jsonable(self.position(manifests, evolution_result))
            outputs["position"] = position_result
            completed = _append_stages(completed, ["model_and_position"])
            self._write_state(status="running", completed_stages=completed, stage_outputs=outputs)
        else:
            position_result = dict(outputs.get("position", {}))

        if "research_card" not in completed:
            research_card_result = _jsonable(self.research_card(manifests, evolution_result, position_result))
            outputs["research_card"] = research_card_result
            completed = _append_stages(completed, ["research_card"])
            self._write_state(status="running", completed_stages=completed, stage_outputs=outputs)
        else:
            research_card_result = dict(outputs.get("research_card", {}))

        if "signal_push" not in completed:
            notifications = _jsonable(self.signal_push(manifests, [position_result]))
            outputs["signal_push"] = {"notifications": notifications}
            completed = _append_stages(completed, ["signal_push"])

        manifest = self._manifest(
            manifests=manifests,
            evolution_result=evolution_result,
            position_result=position_result,
            research_card_result=research_card_result,
            notifications=list(outputs.get("signal_push", {}).get("notifications", [])),
            completed_stages=completed,
            stage_outputs=outputs,
        )
        self._write_manifest(manifest)
        self._write_state(status="complete", completed_stages=COGNITIVE_ALPHA_GOAL_STAGE_ORDER, stage_outputs=outputs)
        return manifest

    def _load_state(self) -> dict[str, Any]:
        if not self.spec.resume or not self.spec.state_path.exists():
            return {"status": "new", "completed_stages": [], "stage_outputs": {}}
        return json.loads(self.spec.state_path.read_text(encoding="utf-8"))

    def _validate_data_manifests(self, manifests: list[dict[str, Any]]) -> None:
        if not manifests:
            raise ValueError("cognitive alpha goal runner requires qlib-derived data manifests")
        for manifest in manifests:
            if not str(manifest.get("source", "")).startswith("qlib"):
                raise ValueError("cognitive alpha goal runner requires qlib-derived data manifests")
        max_datetime = max(str(manifest.get("max_datetime", "")) for manifest in manifests)
        required = self.spec.required_max_datetime
        if required is not None and max_datetime < required:
            raise ValueError(
                f"stale qlib data max_datetime {max_datetime} before required_max_datetime {required}"
            )

    def _manifest(
        self,
        *,
        manifests: list[dict[str, Any]],
        evolution_result: dict[str, Any],
        position_result: dict[str, Any],
        research_card_result: dict[str, Any],
        notifications: list[dict[str, Any]],
        completed_stages: list[str],
        stage_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        data_gate_summary = _data_gate_summary(manifests, required_max_datetime=self.spec.required_max_datetime)
        return {
            "run_id": self.spec.run_id,
            "objective": self.spec.objective,
            "paper_reference": self.spec.paper_reference,
            "status": "complete",
            "stage_order": list(COGNITIVE_ALPHA_GOAL_STAGE_ORDER),
            "completed_stages": list(COGNITIVE_ALPHA_GOAL_STAGE_ORDER),
            "data_gate": data_gate_summary,
            "candidate_generation": _candidate_generation_summary(evolution_result),
            "quality_checker": _quality_checker_summary(evolution_result),
            "fitness_evaluation": _fitness_evaluation_summary(evolution_result),
            "evolution_loop": _evolution_loop_summary(evolution_result),
            "model_and_position": position_result,
            "research_card": research_card_result,
            "signal_push": {"notifications": notifications},
            "stage_outputs": stage_outputs,
        }

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.spec.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.spec.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def _write_state(
        self,
        *,
        status: str,
        completed_stages: list[str],
        stage_outputs: dict[str, Any],
    ) -> None:
        self.spec.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.spec.run_id,
            "status": status,
            "completed_stages": completed_stages,
            "stage_outputs": stage_outputs,
        }
        self.spec.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class CognitiveAlphaSignalRefreshSpec:
    card_dir: Path
    provider_uri: Path
    output_dir: Path
    paper_cash: float
    symbol: str = "SOXL"
    market: str = "all"
    region: str = "us"
    freq: str = "day"
    required_max_datetime: str | None = None


class CognitiveAlphaSignalRefreshJob:
    def __init__(
        self,
        spec: CognitiveAlphaSignalRefreshSpec,
        *,
        notifier: BarkNotifier | None = None,
    ) -> None:
        self.spec = spec
        self.notifier = notifier

    def run(self, data_manifests: list[dict[str, Any]], model_results: list[dict[str, Any]] | None = None) -> list[dict]:
        summary = _load_card_summary(self.spec.card_dir)
        signal = _latest_signal(summary, self.spec.card_dir)
        data_gate = _validate_signal_freshness(
            signal,
            data_manifests=data_manifests,
            required_max_datetime=self.spec.required_max_datetime,
        )
        price = _latest_qlib_close(self.spec, as_of=str(signal["as_of"]))
        notification = _notification_from_signal(summary, signal, spec=self.spec, price=price)
        payload = _notification_payload(
            notification,
            summary=summary,
            signal=signal,
            data_gate=data_gate,
            model_results=model_results or [],
        )
        if self.notifier is not None:
            payload["push_result"] = self.notifier.send_trade_notification(notification)
        else:
            payload["push_result"] = {"status": "not_configured"}
        self._write_manifest(payload)
        return [payload]

    def _write_manifest(self, payload: dict[str, Any]) -> Path:
        self.spec.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.spec.output_dir / "latest_signal_push.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


def _load_card_summary(card_dir: Path) -> dict[str, Any]:
    path = card_dir / "summary.json"
    if not path.exists():
        raise ValueError(f"missing cognitive alpha card summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_signal(summary: dict[str, Any], card_dir: Path) -> dict[str, Any]:
    if "latest_signal" in summary:
        return dict(summary["latest_signal"])
    sidecar = summary.get("sidecars", {}).get("latest_signal", "latest_signal.json")
    path = card_dir / str(sidecar)
    if not path.exists():
        raise ValueError(f"missing cognitive alpha latest signal: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_signal_freshness(
    signal: dict[str, Any],
    *,
    data_manifests: list[dict[str, Any]],
    required_max_datetime: str | None,
) -> dict[str, Any]:
    if not data_manifests:
        raise ValueError("cognitive alpha signal refresh requires qlib data manifests")
    max_datetime = max(str(manifest.get("max_datetime", "")) for manifest in data_manifests)
    signal_as_of = str(signal.get("as_of", ""))
    required = required_max_datetime or max_datetime
    if max_datetime < required:
        raise ValueError(
            f"stale qlib data max_datetime {max_datetime} before required_max_datetime {required}"
        )
    if signal_as_of < max_datetime:
        raise ValueError(f"stale cognitive alpha signal as_of {signal_as_of} before qlib data {max_datetime}")
    if signal_as_of > max_datetime:
        raise ValueError(f"cognitive alpha signal as_of {signal_as_of} exceeds qlib data {max_datetime}")
    return {
        "status": "ok",
        "max_datetime": max_datetime,
        "required_max_datetime": required,
        "signal_as_of": signal_as_of,
        "source": "qlib.data.D.features",
    }


def _latest_qlib_close(spec: CognitiveAlphaSignalRefreshSpec, *, as_of: str) -> float:
    config = QlibProviderConfig(
        provider_uri=spec.provider_uri,
        region=spec.region,
        market=spec.market,
        freq=spec.freq,
    )
    frame = load_qlib_feature_frame(config, instruments=[spec.symbol], fields=["$close"], start=as_of, end=as_of)
    normalized = frame.reset_index()
    if "$close" not in normalized.columns:
        raise ValueError("qlib latest signal price lookup missing $close")
    closes = normalized.loc[normalized["instrument"] == spec.symbol, "$close"].dropna()
    if closes.empty:
        raise ValueError(f"qlib latest signal price lookup has no close for {spec.symbol} at {as_of}")
    return float(closes.iloc[-1])


def _notification_from_signal(
    summary: dict[str, Any],
    signal: dict[str, Any],
    *,
    spec: CognitiveAlphaSignalRefreshSpec,
    price: float,
) -> TradeNotification:
    broker = PaperBroker(cash=spec.paper_cash)
    current = broker.create_order_intent(
        TargetPosition(
            symbol=spec.symbol,
            target_percent=float(signal["target_position"]),
            price=price,
            as_of=str(signal["as_of"]),
            recorder_id=str(summary["recorder_id"]),
        )
    )
    previous_target = TargetPosition(
        symbol=spec.symbol,
        target_percent=float(signal.get("previous_target_position", 0.0)),
        price=price,
        as_of=str(signal["as_of"]),
        recorder_id=str(summary["recorder_id"]),
    )
    previous = PaperBroker(cash=spec.paper_cash).create_order_intent(previous_target)
    notification = trade_notification_from_order_intents(current=current, previous=previous)
    if signal.get("action") in {"BUY", "SELL", "HOLD"} and notification.action != signal["action"]:
        return TradeNotification(
            action=str(signal["action"]),
            symbol=notification.symbol,
            target_percent=notification.target_percent,
            previous_target_percent=notification.previous_target_percent,
            target_delta_percent=notification.target_delta_percent,
            quantity=notification.quantity,
            previous_quantity=notification.previous_quantity,
            quantity_delta=notification.quantity_delta,
            price=notification.price,
            as_of=notification.as_of,
            recorder_id=notification.recorder_id,
            execution_mode=notification.execution_mode,
            real_broker_order=notification.real_broker_order,
        )
    return notification


def _notification_payload(
    notification: TradeNotification,
    *,
    summary: dict[str, Any],
    signal: dict[str, Any],
    data_gate: dict[str, Any],
    model_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source": "qlib_recorder.latest_signal",
        "experiment": str(summary.get("experiment", "")),
        "card_type": str(summary.get("card_type", "")),
        "symbol": notification.symbol,
        "action": notification.action,
        "as_of": notification.as_of,
        "target_percent": notification.target_percent,
        "previous_target_percent": notification.previous_target_percent,
        "target_delta_percent": notification.target_delta_percent,
        "quantity": notification.quantity,
        "previous_quantity": notification.previous_quantity,
        "quantity_delta": notification.quantity_delta,
        "price": notification.price,
        "recorder_id": notification.recorder_id,
        "real_broker_order": notification.real_broker_order,
        "execution_mode": notification.execution_mode,
        "signal": signal,
        "data_gate": data_gate,
        "model_results": model_results,
        "notification": asdict(notification),
        "disclaimer": "research risk-control paper/sim notification; not investment advice",
    }


def _append_stages(existing: list[str], new_stages: list[str]) -> list[str]:
    out = list(existing)
    for stage in new_stages:
        if stage not in out:
            out.append(stage)
    return out


def _data_gate_summary(manifests: list[dict[str, Any]], *, required_max_datetime: str | None) -> dict[str, Any]:
    max_datetime = max(str(manifest.get("max_datetime", "")) for manifest in manifests)
    return {
        "status": "ok",
        "source": "qlib.data.D.features",
        "max_datetime": max_datetime,
        "required_max_datetime": required_max_datetime or max_datetime,
        "manifests": _jsonable(manifests),
    }


def _candidate_generation_summary(evolution_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "candidate_provider": evolution_result.get("candidate_provider", {}),
        "initial_candidate_count": int(evolution_result.get("initial_candidate_count", 0) or 0),
        "checkpoints": _checkpoint_subset(evolution_result, ["candidate_pool"]),
    }


def _quality_checker_summary(evolution_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "repair_attempt_count": int(evolution_result.get("repair_attempt_count", 0) or 0),
        "thresholds": evolution_result.get("thresholds", {}),
        "checkpoints": _checkpoint_subset(evolution_result, ["quality", "rejected"]),
    }


def _fitness_evaluation_summary(evolution_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "candidate_metric_rows": int(evolution_result.get("candidate_metric_rows", 0) or 0),
        "selected_count": int(evolution_result.get("selected_count", 0) or 0),
        "selected_feature_names": list(evolution_result.get("selected_feature_names", [])),
        "checkpoints": _checkpoint_subset(evolution_result, ["fitness"]),
    }


def _evolution_loop_summary(evolution_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "completed_generations": int(evolution_result.get("completed_generations", 0) or 0),
        "parent_pool_size": int(evolution_result.get("parent_pool_size", 0) or 0),
        "children_per_generation": int(evolution_result.get("children_per_generation", 0) or 0),
        "evolution_ops": list(evolution_result.get("evolution_ops", [])),
        "checkpoints": _checkpoint_subset(evolution_result, ["lineage", "state"]),
    }


def _checkpoint_subset(payload: dict[str, Any], names: list[str]) -> dict[str, Any]:
    checkpoints = payload.get("checkpoints", {})
    return {name: checkpoints[name] for name in names if name in checkpoints}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value
