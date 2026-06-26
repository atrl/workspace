from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
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
    "direct_factor_backtest",
    "research_card",
    "signal_push",
]
COGNITIVE_ALPHA_PAPER_REFERENCE = "arxiv:2511.18850"


@dataclass(frozen=True)
class CognitiveAlphaGoalGateSpec:
    min_initial_candidate_count: int = 0
    min_completed_generations: int = 0
    min_selected_count: int = 0
    min_repair_attempt_count: int | None = None
    required_candidate_provider_ids: tuple[str, ...] = ()
    required_candidate_provider_kinds: tuple[str, ...] = ()
    require_candidate_provider_no_openai_api_key: bool = False
    required_metric_names: tuple[str, ...] = ()
    required_evolution_ops: tuple[str, ...] = ()
    required_position_model_families: tuple[str, ...] = ()
    required_benchmark_names: tuple[str, ...] = ()
    required_direct_factor_benchmark_names: tuple[str, ...] = ()
    require_direct_factor_beats_constrained_weight_reference: bool = False
    require_latest_signal_matches_data: bool = True
    min_target_position: float = 0.0
    max_target_position: float = 1.0


@dataclass(frozen=True)
class CognitiveAlphaGoalRunnerSpec:
    run_id: str
    objective: str
    state_path: Path
    manifest_path: Path
    required_max_datetime: str | None = None
    resume: bool = False
    paper_reference: str = COGNITIVE_ALPHA_PAPER_REFERENCE
    gates: CognitiveAlphaGoalGateSpec = field(default_factory=CognitiveAlphaGoalGateSpec)


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
        direct_factor_backtest: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.spec = spec
        self.data_gate = data_gate
        self.evolution = evolution
        self.position = position
        self.direct_factor_backtest = direct_factor_backtest
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
            _validate_evolution_result(evolution_result, self.spec.gates)
            outputs["evolution"] = evolution_result
            completed = _append_stages(completed, evolution_stages)
            self._write_state(status="running", completed_stages=completed, stage_outputs=outputs)
        else:
            evolution_result = dict(outputs.get("evolution", {}))
            _validate_evolution_result(evolution_result, self.spec.gates)

        if "model_and_position" not in completed:
            position_result = _jsonable(self.position(manifests, evolution_result))
            _validate_position_result(position_result, manifests=manifests, gates=self.spec.gates)
            outputs["position"] = position_result
            completed = _append_stages(completed, ["model_and_position"])
            self._write_state(status="running", completed_stages=completed, stage_outputs=outputs)
        else:
            position_result = dict(outputs.get("position", {}))
            _validate_position_result(position_result, manifests=manifests, gates=self.spec.gates)

        if "direct_factor_backtest" not in completed:
            direct_factor_result = _jsonable(self._run_direct_factor_backtest(manifests, evolution_result, position_result))
            _validate_direct_factor_result(direct_factor_result, gates=self.spec.gates)
            outputs["direct_factor_backtest"] = direct_factor_result
            completed = _append_stages(completed, ["direct_factor_backtest"])
            self._write_state(status="running", completed_stages=completed, stage_outputs=outputs)
        else:
            direct_factor_result = dict(outputs.get("direct_factor_backtest", {}))
            _validate_direct_factor_result(direct_factor_result, gates=self.spec.gates)

        if "research_card" not in completed:
            research_card_result = _jsonable(self.research_card(manifests, evolution_result, position_result))
            _validate_research_card_result(research_card_result, direct_factor_result=direct_factor_result)
            outputs["research_card"] = research_card_result
            completed = _append_stages(completed, ["research_card"])
            self._write_state(status="running", completed_stages=completed, stage_outputs=outputs)
        else:
            research_card_result = dict(outputs.get("research_card", {}))
            _validate_research_card_result(research_card_result, direct_factor_result=direct_factor_result)

        if "signal_push" not in completed:
            notifications = _jsonable(self.signal_push(manifests, [position_result, direct_factor_result]))
            outputs["signal_push"] = {"notifications": notifications}
            completed = _append_stages(completed, ["signal_push"])

        manifest = self._manifest(
            manifests=manifests,
            evolution_result=evolution_result,
            position_result=position_result,
            direct_factor_result=direct_factor_result,
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

    def _run_direct_factor_backtest(
        self,
        manifests: list[dict[str, Any]],
        evolution_result: dict[str, Any],
        position_result: dict[str, Any],
    ) -> dict[str, Any]:
        if self.direct_factor_backtest is None:
            return {
                "workflow": "not_configured",
                "status": "skipped",
                "comparison": {},
                "selected_candidate": {},
            }
        return self.direct_factor_backtest(manifests, evolution_result, position_result)

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
        direct_factor_result: dict[str, Any],
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
            "validation_gates": _jsonable(asdict(self.spec.gates)),
            "data_gate": data_gate_summary,
            "candidate_generation": _candidate_generation_summary(evolution_result),
            "quality_checker": _quality_checker_summary(evolution_result),
            "fitness_evaluation": _fitness_evaluation_summary(evolution_result),
            "evolution_loop": _evolution_loop_summary(evolution_result),
            "model_and_position": position_result,
            "direct_factor_backtest": direct_factor_result,
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


def _validate_evolution_result(evolution_result: dict[str, Any], gates: CognitiveAlphaGoalGateSpec) -> None:
    candidate_provider = evolution_result.get("candidate_provider", {})
    if not isinstance(candidate_provider, dict):
        raise ValueError("cognitive alpha goal runner requires candidate_provider summary")
    if gates.required_candidate_provider_ids:
        provider_id = str(candidate_provider.get("provider_id", ""))
        if provider_id not in gates.required_candidate_provider_ids:
            raise ValueError(
                "cognitive alpha candidate provider id "
                f"{provider_id!r} not in required {list(gates.required_candidate_provider_ids)}"
            )
    if gates.required_candidate_provider_kinds:
        provider_kind = str(candidate_provider.get("provider_kind", ""))
        if provider_kind not in gates.required_candidate_provider_kinds:
            raise ValueError(
                "cognitive alpha candidate provider kind "
                f"{provider_kind!r} not in required {list(gates.required_candidate_provider_kinds)}"
            )
    if gates.require_candidate_provider_no_openai_api_key and candidate_provider.get("openai_api_key_required") is not False:
        raise ValueError("cognitive alpha candidate provider must explicitly not require OPENAI_API_KEY")

    initial_count = int(evolution_result.get("initial_candidate_count", 0) or 0)
    if initial_count < gates.min_initial_candidate_count:
        raise ValueError(
            "cognitive alpha goal runner initial candidate count "
            f"{initial_count} below required {gates.min_initial_candidate_count}"
        )
    completed_generations = int(evolution_result.get("completed_generations", 0) or 0)
    if completed_generations < gates.min_completed_generations:
        raise ValueError(
            "cognitive alpha goal runner completed generations "
            f"{completed_generations} below required {gates.min_completed_generations}"
        )
    selected_count = int(evolution_result.get("selected_count", 0) or 0)
    if selected_count < gates.min_selected_count:
        raise ValueError(
            f"cognitive alpha goal runner selected count {selected_count} below required {gates.min_selected_count}"
        )
    if gates.min_repair_attempt_count is not None:
        repair_attempt_count = int(evolution_result.get("repair_attempt_count", 0) or 0)
        if repair_attempt_count < gates.min_repair_attempt_count:
            raise ValueError(
                "cognitive alpha goal runner repair attempt count "
                f"{repair_attempt_count} below required {gates.min_repair_attempt_count}"
            )
    missing_metrics = _missing_required_values(
        available=evolution_result.get("metric_names", []),
        required=gates.required_metric_names,
    )
    if missing_metrics:
        raise ValueError(f"missing required fitness metrics: {missing_metrics}")
    missing_ops = _missing_required_values(
        available=evolution_result.get("evolution_ops", []),
        required=gates.required_evolution_ops,
    )
    if missing_ops:
        raise ValueError(f"missing required evolution ops: {missing_ops}")


def _validate_position_result(
    position_result: dict[str, Any],
    *,
    manifests: list[dict[str, Any]],
    gates: CognitiveAlphaGoalGateSpec,
) -> None:
    missing_models = _missing_required_values(
        available=position_result.get("model_families", []),
        required=gates.required_position_model_families,
    )
    if missing_models:
        raise ValueError(f"missing required position model families: {missing_models}")
    comparison = position_result.get("comparison", {})
    missing_benchmarks = [
        name
        for name in gates.required_benchmark_names
        if f"beats_{name}" not in comparison and f"selected_vs_{name}_return_gap_pct" not in comparison
    ]
    if missing_benchmarks:
        raise ValueError(f"missing required benchmark comparisons: {missing_benchmarks}")
    signal = position_result.get("latest_signal")
    if not isinstance(signal, dict):
        raise ValueError("cognitive alpha goal runner requires latest_signal in position result")
    target_position = float(signal.get("target_position", 0.0))
    if target_position < gates.min_target_position or target_position > gates.max_target_position:
        raise ValueError(
            "cognitive alpha latest_signal target_position "
            f"{target_position} outside [{gates.min_target_position}, {gates.max_target_position}]"
        )
    if gates.require_latest_signal_matches_data:
        max_datetime = max(str(manifest.get("max_datetime", "")) for manifest in manifests)
        signal_as_of = str(signal.get("as_of", ""))
        if signal_as_of != max_datetime:
            raise ValueError(f"latest_signal as_of {signal_as_of} does not match qlib data {max_datetime}")


def _validate_direct_factor_result(direct_factor_result: dict[str, Any], *, gates: CognitiveAlphaGoalGateSpec) -> None:
    comparison = direct_factor_result.get("comparison", {})
    missing_benchmarks = [
        name
        for name in gates.required_direct_factor_benchmark_names
        if f"beats_{name}" not in comparison and f"selected_vs_{name}_return_gap_pct" not in comparison
    ]
    if missing_benchmarks:
        raise ValueError(f"missing required direct factor benchmark comparisons: {missing_benchmarks}")
    if (
        gates.require_direct_factor_beats_constrained_weight_reference
        and comparison.get("beats_constrained_weight_reference") is not True
    ):
        raise ValueError("direct factor backtest must beat constrained_weight_reference")

    selected = direct_factor_result.get("selected_candidate", {})
    if not isinstance(selected, dict):
        raise ValueError("cognitive alpha direct factor result requires selected_candidate")
    if "max_position" in selected and float(selected["max_position"]) > gates.max_target_position + 1e-9:
        raise ValueError("direct factor selected candidate exceeds max_target_position")
    if "min_position" in selected and float(selected["min_position"]) < gates.min_target_position - 1e-9:
        raise ValueError("direct factor selected candidate is below min_target_position")
    if int(selected.get("days_above_100", 0) or 0) > 0:
        raise ValueError("direct factor selected candidate has days_above_100")


def _validate_research_card_result(
    research_card_result: dict[str, Any],
    *,
    direct_factor_result: dict[str, Any],
) -> None:
    promotion_decision = str(research_card_result.get("promotion_decision", ""))
    direct_workflow_ran = direct_factor_result.get("workflow") == "cognitive_alpha_weighted_factor_search"
    if direct_workflow_ran and promotion_decision == "candidate_for_validated_research_direct_factor":
        raise ValueError("direct factor backtest cannot be the final promotion gate")


def _missing_required_values(*, available: Any, required: tuple[str, ...]) -> list[str]:
    available_set = {str(value) for value in (available or [])}
    return [value for value in required if value not in available_set]


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
        "metric_names": list(evolution_result.get("metric_names", [])),
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
