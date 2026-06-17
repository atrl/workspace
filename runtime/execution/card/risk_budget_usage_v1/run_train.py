#!/usr/bin/env python3
"""Legacy PPO training wrapper retained only as historical runtime context."""

from __future__ import annotations

MESSAGE = (
    "risk_budget_usage_v1 was produced by the retired PPO training CLI. "
    "The active architecture now uses the simplified Alpha158 supervised "
    "cadence surface; see runtime/harness/card/"
    "simplified_non_rl_architecture_audit_v1/summary.json."
)

if __name__ == '__main__':
    raise SystemExit(MESSAGE)
