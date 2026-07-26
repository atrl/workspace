#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h:h}"
cd "$repo_root"
exec qlib_venv/bin/python -m src.dashboard.stage1_app --host 127.0.0.1 --port 8765
