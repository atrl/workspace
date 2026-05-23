from __future__ import annotations

"""Compatibility import for the repo-owned IBKR broker adapter.

New code should import from `src.soxl_quant.app.ibkr`.
"""

from src.soxl_quant.app.ibkr import IBConnectionConfig, IBBroker

__all__ = ["IBBroker", "IBConnectionConfig"]
