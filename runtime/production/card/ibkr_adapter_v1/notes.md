# ibkr_adapter_v1

Date: 2026-05-23

Goal:

- Install the Python-side IBKR dependency and bring the existing IBKR manual
  path into the current production broker boundary.
- Keep IBKR opt-in and paper-first. Do not change the scheduled default broker
  from `paper`.

External reference:

- IBKR TWS API socket model: local TWS / IB Gateway must be running, logged in,
  and configured to accept socket clients.
- Default ports: TWS live `7496`, TWS paper `7497`, IB Gateway live `4001`,
  IB Gateway paper `4002`.

Changes:

- Installed `ib-insync==0.9.86` into `qlib_venv`; added
  `ib-insync==0.9.86` and transitive `eventkit==1.0.3` to
  `requirements/qlib.txt`.
- Added `src/soxl_quant/app/ibkr.py` as the repo-owned IBKR adapter boundary.
- Converted `production/ib_broker.py` and `production/ib_client.py` to thin
  compatibility imports.
- Added `production/check_ibkr_gateway.py` for fast TWS / IB Gateway readiness
  checks.
- Added `--broker ibkr` plus host/port/client/account/timeout args to
  `production/soxl_paper_trading.py`, `production/soxl_daily_scheduler.py`, and
  `production/composed_session.py`.
- Updated `production/ib_manual_session.py` to import from
  `src.soxl_quant.app.ibkr`.
- Updated `production/IB_SEMI_AUTO.md`.
- Pinned the duplicate-skip scheduler-health fixture to a postclose-due health
  time so the runtime-proof regression is repeatable across weekdays.
- Kept IBKR order proof conservative: an IBKR `Submitted`/pending status does
  not assume the portfolio has reached the target position until a filled status
  is observed.

TDD evidence:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_ibkr_adapter.py -q
# red before implementation: 8 failed
# green after implementation: 8 passed

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_ibkr_adapter.py::test_ibkr_submitted_order_does_not_assume_resulting_position_changed -q
# red before fix: resulting_position was 0.5 for an IBKR Submitted status
# green after fix: 1 passed
```

Verification:

```bash
qlib_venv/bin/python - <<'PY'
import inspect
import ib_insync
from ib_insync import IB
print('ib_insync', getattr(ib_insync, '__version__', 'unknown'))
print(inspect.signature(IB.connect))
PY
# ib_insync 0.9.86
# (self, host: str = '127.0.0.1', port: int = 7497, clientId: int = 1,
#  timeout: float = 4, readonly: bool = False, account: str = '')

PYTHONPATH=. qlib_venv/bin/python production/check_ibkr_gateway.py --port 1
# FAIL: IBKR TWS/Gateway is NOT reachable at 127.0.0.1:1

PYTHONPATH=. qlib_venv/bin/python production/check_ibkr_gateway.py --port 7497 --timeout 1
# FAIL: IBKR TWS/Gateway is NOT reachable at 127.0.0.1:7497

brew info --cask trader-workstation
# trader-workstation 10.47.1b is available but not installed

HOMEBREW_NO_AUTO_UPDATE=1 brew install --cask trader-workstation
# dmg downloaded, installer started, then install4j process did not exit
# after writing a partial /Applications/Trader Workstation directory.
# Terminating the installer made Homebrew purge the cask; the unusable partial
# application directory was removed.

PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --broker ibkr --help
# exposes --broker {paper,futu,ibkr} and --ibkr-* args

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --broker ibkr --help
# exposes --broker {paper,futu,ibkr} and --ibkr-* args

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_ibkr_adapter.py tests/test_futu_adapter.py tests/test_daily_scheduler.py \
  tests/test_paper_trading.py tests/test_production_schedule.py \
  tests/test_scheduler_health.py::test_scheduler_health_accepts_selected_profile_duplicate_skip_as_runtime_proof -q
# 64 passed

MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# [summary] passed=203 failures=0
```

Decision:

- Python-side IBKR installation and repo integration are complete.
- Runtime IBKR broker readiness is externally blocked until a reliable local
  TWS / IB Gateway install is running, logged in, and socket API is enabled on
  the selected port.
- Homebrew cask installation of TWS is not reliable in this environment because
  the quiet install4j process hangs; use the local GUI installer or IB Gateway
  install path before expecting `check_ibkr_gateway.py` to return OK.
- Keep scheduled production on `--broker paper`; use `--broker ibkr` manually
  after `production/check_ibkr_gateway.py` returns OK.
