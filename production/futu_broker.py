#!/usr/bin/env python3
from __future__ import annotations

"""Compatibility import for the repo-owned Futu broker adapter.

New code should import `FutuBroker` from `src.soxl_quant.app.futu`.
"""

from src.soxl_quant.app.futu import FutuBroker

__all__ = ["FutuBroker"]
