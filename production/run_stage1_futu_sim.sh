#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${SOXL_STAGE1_PYTHON:-${ROOT}/qlib_venv/bin/python}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
exec "${PYTHON}" -m src.schedule.stage1_futu "$@"
