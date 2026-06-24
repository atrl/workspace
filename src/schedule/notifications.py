from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class TradeNotification:
    action: str
    symbol: str
    target_percent: float
    previous_target_percent: float
    target_delta_percent: float
    quantity: int
    previous_quantity: int
    quantity_delta: int
    price: float
    as_of: str
    recorder_id: str
    execution_mode: str
    real_broker_order: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BarkPushConfig:
    server_url: str
    device_key: str
    timeout_seconds: float = 15.0
    dry_run: bool = True

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        server_var: str = "BARK_SERVER_URL",
        device_key_var: str = "BARK_DEVICE_KEY",
        dry_run: bool = True,
    ) -> "BarkPushConfig":
        values = os.environ if env is None else env
        server_url = values.get(server_var, "").strip()
        device_key = values.get(device_key_var, "").strip()
        if not server_url or not device_key:
            raise ValueError(f"{server_var} and {device_key_var} are required for Bark push")
        return cls(server_url=server_url, device_key=device_key, dry_run=dry_run)


class BarkNotifier:
    def __init__(
        self,
        config: BarkPushConfig,
        *,
        sender: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.sender = sender or _send_json

    def send_trade_notification(self, notification: TradeNotification) -> dict[str, Any]:
        endpoint = self.config.server_url.rstrip("/") + "/push"
        payload = bark_payload_for_trade(notification, device_key=self.config.device_key)
        if self.config.dry_run:
            return {
                "status": "dry_run",
                "endpoint": endpoint,
                "payload": payload | {"device_key": "<redacted>"},
            }
        response = self.sender(endpoint, payload, self.config.timeout_seconds)
        status_code = int(response.get("status", 0))
        if 200 <= status_code < 300:
            return {"status": "sent", "response_status": status_code}
        raise RuntimeError(f"Bark push failed with status {status_code}: {response.get('body', '')}")


def trade_notification_from_order_intents(
    *,
    current: Any,
    previous: Any | None = None,
    min_target_delta: float = 0.000001,
) -> TradeNotification:
    current_payload = _intent_payload(current)
    previous_payload = _intent_payload(previous) if previous is not None else {}
    if bool(current_payload.get("real_broker_order", False)):
        raise ValueError("Bark trade notifications only support paper/sim order intents in v1")

    target_percent = float(current_payload["target_percent"])
    previous_target_percent = float(previous_payload.get("target_percent", 0.0))
    quantity = int(current_payload.get("quantity", 0))
    previous_quantity = int(previous_payload.get("quantity", 0))
    target_delta = target_percent - previous_target_percent
    quantity_delta = quantity - previous_quantity
    action = _trade_action(target_delta, quantity_delta, min_target_delta=min_target_delta)
    return TradeNotification(
        action=action,
        symbol=str(current_payload["symbol"]),
        target_percent=target_percent,
        previous_target_percent=previous_target_percent,
        target_delta_percent=target_delta,
        quantity=quantity,
        previous_quantity=previous_quantity,
        quantity_delta=quantity_delta,
        price=float(current_payload["price"]),
        as_of=str(current_payload["as_of"]),
        recorder_id=str(current_payload["recorder_id"]),
        execution_mode=str(current_payload.get("execution_mode", "paper_sim")),
        real_broker_order=bool(current_payload.get("real_broker_order", False)),
    )


def bark_payload_for_trade(notification: TradeNotification, *, device_key: str) -> dict[str, str]:
    return {
        "device_key": device_key,
        "title": f"{notification.symbol} paper/sim {notification.action}",
        "body": format_trade_notification(notification),
        "group": "SOXL qlib schedule",
    }


def format_trade_notification(notification: TradeNotification) -> str:
    return (
        f"{notification.symbol} {notification.action} paper/sim target {notification.target_percent:.2%} "
        f"(delta {notification.target_delta_percent:+.2%}), "
        f"qty {notification.quantity} (delta {notification.quantity_delta:+d}), "
        f"price {notification.price:.2f}, as_of {notification.as_of}, "
        f"recorder {notification.recorder_id}, real_broker_order={notification.real_broker_order}"
    )


def _intent_payload(intent: Any) -> dict[str, Any]:
    if hasattr(intent, "to_dict"):
        return dict(intent.to_dict())
    if isinstance(intent, Mapping):
        return dict(intent)
    raise TypeError("order intent must be an OrderIntent or mapping")


def _trade_action(target_delta: float, quantity_delta: int, *, min_target_delta: float) -> str:
    if abs(target_delta) <= min_target_delta and quantity_delta == 0:
        return "HOLD"
    if target_delta > min_target_delta or quantity_delta > 0:
        return "BUY"
    return "SELL"


def _send_json(endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {"status": response.status, "body": response.read().decode("utf-8", errors="replace")}
    except urllib.error.HTTPError as error:
        return {"status": error.code, "body": error.read().decode("utf-8", errors="replace")}
