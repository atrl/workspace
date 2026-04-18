from __future__ import annotations

"""Compatibility entrypoint for the new app-layer CLI."""

from ..cli import main, parse_args, run_walk_forward_isolated, run_worker_profile

__all__ = ["main", "parse_args", "run_walk_forward_isolated", "run_worker_profile"]
