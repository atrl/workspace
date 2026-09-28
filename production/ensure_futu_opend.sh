#!/bin/bash
set -euo pipefail
umask 077

# Reuse the existing local gateway. Account credentials remain in OpenD's config.
if /usr/sbin/lsof -nP -iTCP:11111 -sTCP:LISTEN >/dev/null 2>&1; then
  exit 0
fi
OPEND_DIR="${SOXL_OPEND_DIR:-${HOME}/futu_opend/Futu_OpenD_10.2.6208_Mac/Futu_OpenD_10.2.6208_Mac}"
test -r "${OPEND_DIR}/FutuOpenD.xml"
cd "${OPEND_DIR}"
exec "${OPEND_DIR}/FutuOpenD.app/Contents/MacOS/FutuOpenD" \
  "-cfg_file=${OPEND_DIR}/FutuOpenD.xml" -api_ip=127.0.0.1 -api_port=11111 -console=0
