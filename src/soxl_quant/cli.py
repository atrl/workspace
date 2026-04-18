from __future__ import annotations

"""Compatibility facade for the app-layer CLI split.

Train/evaluate/promote orchestration now lives under `soxl_quant.app`.
Keep this module import-compatible while preventing new logic from drifting
back into a monolithic file.
"""

from .app.cli import main, parse_args, run_walk_forward_isolated, run_worker_profile
from .app.evaluate import compact_result, describe_metrics, slice_frame, slice_metrics, validation_score
from .app.promote import (
    build_promotion_report,
    fixed_promotion_report,
    persist_promotion_report,
    walk_forward_promotion_report,
)
from .app.sources import build_multi_asset_sources
from .app.train import run_profile, set_global_seed
from .app.selection import train_with_selection

__all__ = [
    "build_multi_asset_sources",
    "build_promotion_report",
    "compact_result",
    "describe_metrics",
    "fixed_promotion_report",
    "main",
    "parse_args",
    "persist_promotion_report",
    "run_profile",
    "run_walk_forward_isolated",
    "run_worker_profile",
    "set_global_seed",
    "slice_frame",
    "slice_metrics",
    "train_with_selection",
    "validation_score",
    "walk_forward_promotion_report",
]

if __name__ == "__main__":
    main()
