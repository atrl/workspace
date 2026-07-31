#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${SOXL_STAGE1_PYTHON:-${ROOT}/qlib_venv/bin/python}"
BARK_ENV_FILE="${BARK_ENV_FILE:-${HOME}/.config/bark/remote.env}"

if [[ -r "${BARK_ENV_FILE}" ]]; then
  set -a
  source "${BARK_ENV_FILE}"
  set +a
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
exec "${PYTHON}" -m src.schedule.stage1_futu "$@"
