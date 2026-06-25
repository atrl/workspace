from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.api import PaperBroker, TargetPosition
from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.schedule import BarkNotifier, TradeNotification, trade_notification_from_order_intents


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
