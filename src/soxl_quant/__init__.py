"""SOXL-only qlib reinforcement learning package."""

def main(*args, **kwargs):
    from .app.cli import main as app_main

    return app_main(*args, **kwargs)

__all__ = ["main"]
