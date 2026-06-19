from __future__ import annotations

from typing import Protocol, runtime_checkable

from .market_data import MarketQuote
from .paper import AccountSnapshot


PAPER_SIM_MODES = frozenset({"paper", "sim"})


@runtime_checkable
class BrokerAdapter(Protocol):
    broker_name: str

    def healthcheck(self) -> tuple[bool, str]:
        ...

    def snapshot(self, *, price: float) -> AccountSnapshot:
        ...

    def place_target_position(self, *, target_position: float, price: float) -> AccountSnapshot:
        ...


@runtime_checkable
class MarketDataAdapter(Protocol):
    adapter_name: str
    data_mode: str

    def latest_quote(self, instrument: str) -> MarketQuote:
        ...


def assert_paper_sim_mode(mode: str, *, surface: str = "api") -> str:
    value = str(mode)
    if value not in PAPER_SIM_MODES:
        raise ValueError(f"{surface} only supports paper/sim mode")
    return value
