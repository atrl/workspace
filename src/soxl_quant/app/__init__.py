"""Application orchestration entrypoints for the SOXL harness."""

from .cli import main
from .sources import build_multi_asset_sources

__all__ = ["build_multi_asset_sources", "main"]
