"""Application orchestration entrypoints for the SOXL harness."""

from .sources import build_multi_asset_sources


def main(argv=None):
    from .simplified_cli import main as app_main

    return app_main(argv)

__all__ = ["build_multi_asset_sources", "main"]
