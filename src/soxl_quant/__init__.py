"""SOXL-only qlib supervised research harness."""


def main(argv=None):
    from .app.simplified_cli import main as app_main

    return app_main(argv)

__all__ = ["main"]
