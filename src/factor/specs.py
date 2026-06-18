from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorSpec:
    name: str
    family: str
    inputs: tuple[str, ...]
    expression: str
    horizon: int = 1
