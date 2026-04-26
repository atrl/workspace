from __future__ import annotations

"""Universe registry helpers for leveraged ETF experiments."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
UNIVERSE_DIR = ROOT / "runtime" / "universes"
DEFAULT_UNIVERSE = "3x_etf_core"


def load_universe(name: str = DEFAULT_UNIVERSE) -> dict[str, Any]:
    path = UNIVERSE_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Universe config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assets = payload.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError(f"Universe {name} has no assets")
    return payload


def load_etf_universe(name: str = DEFAULT_UNIVERSE) -> dict[str, dict[str, float | str]]:
    payload = load_universe(name)
    return {
        str(asset).upper(): {
            "underlying": str(meta["underlying"]).upper(),
            "leverage": float(meta.get("leverage", 1.0)),
            **({"sector": str(meta["sector"])} if "sector" in meta else {}),
        }
        for asset, meta in payload["assets"].items()
    }


__all__ = ["DEFAULT_UNIVERSE", "UNIVERSE_DIR", "load_etf_universe", "load_universe"]
