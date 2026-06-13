#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


UNIFIED_DASHBOARD_ENTRYPOINT = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="0; url=/web/dashboard-v2/dist/index.html">
    <link rel="icon" href="data:,">
    <title>SOXL Dashboard</title>
    <style>
      :root {
        color: #d1d5db;
        background: #0b1220;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      body {
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
      }
      main {
        display: grid;
        gap: 10px;
        width: min(520px, calc(100vw - 32px));
      }
      h1 {
        margin: 0;
        color: #f9fafb;
        font-size: 22px;
        letter-spacing: 0;
      }
      p {
        margin: 0;
        color: #9ca3af;
      }
      a {
        color: #38bdf8;
      }
      code {
        color: #e5e7eb;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>SOXL Dashboard</h1>
      <p>Loading the unified dashboard bundle.</p>
      <p><a href="/web/dashboard-v2/dist/index.html">Open dashboard</a></p>
      <p><code>scripts/dashboard_server.py</code> is required for interactive reruns.</p>
    </main>
    <script>
      window.location.replace('/web/dashboard-v2/dist/index.html');
    </script>
  </body>
</html>
"""


def _run(cmd: list[str], description: str) -> None:
    print(f"[dashboard-bundle] {description}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def write_unified_dashboard_entrypoints(root: Path | None = None) -> None:
    target_root = root or ROOT
    html = UNIFIED_DASHBOARD_ENTRYPOINT
    for name in ("dashboard.html", "dashboard_v2.html"):
        (target_root / name).write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local SOXL dashboard artifacts.")
    parser.add_argument(
        "--skip-v2-check",
        action="store_true",
        help="Build v2 data but skip the v2 contract checker.",
    )
    parser.add_argument(
        "--skip-retention-gate",
        action="store_true",
        help="Skip refreshing the model retention gate sidecars before dashboard v2 data build.",
    )
    args = parser.parse_args()

    python = sys.executable
    _run([python, "scripts/build_backtest_dashboard.py"], "dashboard.html build")
    if not args.skip_retention_gate:
        _run([python, "scripts/review_model_retention_gates.py"], "model retention gate review")
        _run([python, "scripts/review_model_retention_frontier.py"], "model retention frontier review")
    _run([python, "scripts/build_dashboard_v2_data.py", "--detect-launchd"], "dashboard v2 data build")
    _run(["npm", "--prefix", "web/dashboard-v2", "run", "build"], "dashboard v2 frontend build")
    write_unified_dashboard_entrypoints()
    if not args.skip_v2_check:
        _run([python, "scripts/check_dashboard_v2_data.py"], "dashboard v2 data check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
