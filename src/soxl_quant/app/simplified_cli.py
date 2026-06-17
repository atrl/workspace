from __future__ import annotations

from typing import Iterable

from .simplified_architecture import main as simplified_architecture_main


def main(argv: Iterable[str] | None = None) -> int:
    """Default non-RL package CLI for the simplified SOXL architecture."""
    return simplified_architecture_main(argv)


__all__ = ["main"]
