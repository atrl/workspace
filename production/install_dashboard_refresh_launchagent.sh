#!/bin/bash
# Install or reload the local dashboard refresh LaunchAgent.

set -euo pipefail

RUN_NOW=0
if [[ "${1:-}" == "--run-now" ]]; then
  RUN_NOW=1
elif [[ "${1:-}" != "" ]]; then
  echo "usage: $0 [--run-now]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="${SOXL_DASHBOARD_ROOT:-${DEFAULT_ROOT}}"
LABEL="com.soxl.dashboard.refresh"
LAUNCHAGENTS_DIR="${SOXL_LAUNCHAGENTS_DIR:-${HOME}/Library/LaunchAgents}"
TARGET="${LAUNCHAGENTS_DIR}/${LABEL}.plist"
RUNNER="${ROOT}/production/run_dashboard_refresh.sh"
LOG_DIR="${ROOT}/runtime/soxl/paper_trading"
UID_VALUE="$(id -u)"

if [[ ! -x "${RUNNER}" ]]; then
  echo "dashboard refresh runner is missing or not executable: ${RUNNER}" >&2
  exit 1
fi

mkdir -p "${LAUNCHAGENTS_DIR}" "${LOG_DIR}"

cat > "${TARGET}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${RUNNER}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${ROOT}</string>
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Hour</key><integer>6</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>2</integer></dict>
        <dict><key>Hour</key><integer>6</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>3</integer></dict>
        <dict><key>Hour</key><integer>6</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>4</integer></dict>
        <dict><key>Hour</key><integer>6</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>5</integer></dict>
        <dict><key>Hour</key><integer>6</integer><key>Minute</key><integer>30</integer><key>Weekday</key><integer>6</integer></dict>
    </array>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/dashboard_refresh.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/dashboard_refresh.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>SOXL_DASHBOARD_ROOT</key>
        <string>${ROOT}</string>
        <key>SOXL_SCHEDULE_TRIGGER</key>
        <string>launchagent_dashboard_refresh</string>
    </dict>
</dict>
</plist>
PLIST

if ! command -v launchctl >/dev/null 2>&1; then
  echo "wrote ${TARGET}; launchctl is unavailable, load it manually on macOS"
  exit 0
fi

launchctl bootout "gui/${UID_VALUE}" "${TARGET}" >/dev/null 2>&1 || true
launchctl bootstrap "gui/${UID_VALUE}" "${TARGET}"
launchctl enable "gui/${UID_VALUE}/${LABEL}" >/dev/null 2>&1 || true

if [[ "${RUN_NOW}" == "1" ]]; then
  launchctl kickstart -k "gui/${UID_VALUE}/${LABEL}"
fi

echo "installed ${LABEL} at ${TARGET}"
launchctl print "gui/${UID_VALUE}/${LABEL}" >/dev/null 2>&1 && echo "loaded ${LABEL}"
