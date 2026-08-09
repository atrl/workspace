#!/bin/zsh
set -euo pipefail

config_path="${SOXL_STAGE1_CLOUDFLARED_CONFIG:-${HOME}/.cloudflared/soxl-stage1-dashboard.yml}"
cloudflared_bin="${SOXL_STAGE1_CLOUDFLARED_BIN:-/opt/homebrew/bin/cloudflared}"

if [[ ! -x "$cloudflared_bin" ]]; then
  echo "blocked: cloudflared is unavailable at $cloudflared_bin" >&2
  exit 1
fi
if [[ ! -r "$config_path" ]]; then
  echo "blocked: Stage1 cloudflared config is unavailable at $config_path" >&2
  exit 1
fi

exec "$cloudflared_bin" tunnel --config "$config_path" run
