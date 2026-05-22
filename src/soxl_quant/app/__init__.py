"""Application orchestration entrypoints for the SOXL harness."""

from .sources import build_multi_asset_sources


def main(*args, **kwargs):
    from .cli import main as cli_main

    return cli_main(*args, **kwargs)

__all__ = ["build_multi_asset_sources", "main"]
