from __future__ import annotations

"""Compatibility import for the repo-owned IBKR client wrapper.

New code should import from `src.soxl_quant.app.ibkr`.
"""

from src.soxl_quant.app.ibkr import IBClient, IBConnectionConfig

__all__ = ["IBClient", "IBConnectionConfig"]
