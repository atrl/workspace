#!/usr/bin/env python3
"""SOXL daily unified scheduler.

Runs the full daily pipeline in order:
    1. Market data update
    2. Paper-trading signal + execution
    3. Dashboard rebuild
    4. Completion notification

Usage:
    PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py
    PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --broker futu
    PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run
    PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --smoke
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from production.notify import get_notifier


TRANSIENT_DATA_SOURCE_SNIPPETS = (
    "Too Many Requests",
    "Rate limited",
    "No overlapping dates for SOXL/SOXX/VIX",
    "DNS/network unavailable",
    "Data update completed but data is stale",
    "Cached data is stale for non-dry-run execution",
)


class SchedulerStepError(RuntimeError):
    def __init__(self, description: str, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.description = description
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f"{description} failed with exit code {returncode}")


def _is_transient_data_source_failure(exc: BaseException) -> bool:
    text = f"{exc}\n{getattr(exc, 'stdout', '')}\n{getattr(exc, 'stderr', '')}".lower()
    return any(snippet.lower() in text for snippet in TRANSIENT_DATA_SOURCE_SNIPPETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOXL daily unified scheduler")
    parser.add_argument("--broker", default="paper", choices=["paper", "futu"])
    parser.add_argument("--futu-host", default="127.0.0.1")
    parser.add_argument("--futu-port", type=int, default=11111)
    parser.add_argument("--unlock-password", default="")
    parser.add_argument("--profile-family", default="balanced")
    parser.add_argument("--profile", default="")
    parser.add_argument("--runtime-composer", default="none", choices=["none", "budget_usage"])
    parser.add_argument("--composer-profile", default="")
    parser.add_argument("--slow-profile", default="")
    parser.add_argument("--fast-profile", default="")
    parser.add_argument("--bull-participation-guard", action="store_true")
    parser.add_argument("--bull-ma-window", type=int, default=50)
    parser.add_argument("--bull-momentum-window", type=int, default=20)
    parser.add_argument("--bull-high-window", type=int, default=50)
    parser.add_argument("--bull-min-periods", type=int, default=20)
    parser.add_argument("--bull-min-momentum", type=float, default=0.03)
    parser.add_argument("--bull-max-drawdown", type=float, default=0.08)
    parser.add_argument("--data-provider", default="auto", choices=["auto", "yfinance", "yahoo_api", "polygon"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--discord", action="store_true", help="Also send notifications to Discord")
    parser.add_argument("--skip-data-update", action="store_true")
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument(
        "--vercel-deploy",
        action="store_true",
        help="After rebuilding the dashboard, try to deploy the static bundle to Vercel.",
    )
    parser.add_argument(
        "--strict-vercel-deploy",
        action="store_true",
        help="Fail the scheduler when Vercel deployment is blocked by missing credentials.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run cached-data inference plus dashboard rebuild without placing orders or notifying",
    )
    args = parser.parse_args()
    if args.smoke:
        args.dry_run = True
        args.no_notify = True
        args.skip_data_update = True
    return args


def _run(cmd: list[str], description: str, allowed_returncodes: set[int] | None = None) -> int:
    print(f"[scheduler] {description}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        if allowed_returncodes and result.returncode in allowed_returncodes:
            return result.returncode
        print(f"[scheduler] ERROR in {description}:", file=sys.stderr)
        raise SchedulerStepError(
            description=description,
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )
    return result.returncode


def _run_vercel_deploy(cmd: list[str], *, strict: bool) -> None:
    allowed = set() if strict else {2}
    returncode = _run(cmd, "vercel dashboard deploy", allowed_returncodes=allowed)
    if not strict and returncode == 2:
        print(
            "[scheduler] Vercel deploy blocked by missing credentials; "
            "continuing after writing deployment evidence"
        )


def main() -> None:
    args = parse_args()
    use_notify = not args.no_notify
    notifier = get_notifier(prefer_bark=True, include_discord=args.discord) if use_notify else None

    venv_python = str(ROOT / "qlib_venv" / "bin" / "python")
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_marker = (
        "[scheduler] run started at "
        f"{started_at} broker={args.broker} dry_run={args.dry_run} "
        f"skip_data_update={args.skip_data_update} data_provider={args.data_provider}"
    )
    if args.runtime_composer != "none":
        run_marker += (
            f" runtime_composer={args.runtime_composer}"
            f" composer_profile={args.composer_profile or 'auto'}"
            f" slow_profile={args.slow_profile or 'auto'}"
            f" fast_profile={args.fast_profile or 'auto'}"
            f" bull_guard={args.bull_participation_guard}"
        )
    print(run_marker)

    try:
        # 1. Paper trading runner (includes data update unless skipped)
        paper_cmd = [
            venv_python, "production/soxl_paper_trading.py",
            "--broker", args.broker,
            "--profile-family", args.profile_family,
            "--data-provider", args.data_provider,
        ]
        if args.profile:
            paper_cmd.extend(["--profile", args.profile])
        if args.runtime_composer != "none":
            paper_cmd.extend(["--runtime-composer", args.runtime_composer])
            if args.composer_profile:
                paper_cmd.extend(["--composer-profile", args.composer_profile])
            if args.slow_profile:
                paper_cmd.extend(["--slow-profile", args.slow_profile])
            if args.fast_profile:
                paper_cmd.extend(["--fast-profile", args.fast_profile])
            if args.bull_participation_guard:
                paper_cmd.append("--bull-participation-guard")
            paper_cmd.extend(
                [
                    "--bull-ma-window", str(args.bull_ma_window),
                    "--bull-momentum-window", str(args.bull_momentum_window),
                    "--bull-high-window", str(args.bull_high_window),
                    "--bull-min-periods", str(args.bull_min_periods),
                    "--bull-min-momentum", str(args.bull_min_momentum),
                    "--bull-max-drawdown", str(args.bull_max_drawdown),
                ]
            )
        if args.dry_run:
            paper_cmd.append("--dry-run")
        if args.skip_data_update:
            paper_cmd.append("--skip-data-update")
        if args.no_notify:
            paper_cmd.append("--no-notify")
        if args.broker == "futu":
            paper_cmd.extend(["--futu-host", args.futu_host, "--futu-port", str(args.futu_port)])
            if args.unlock_password:
                paper_cmd.extend(["--unlock-password", args.unlock_password])

        _run(paper_cmd, "paper trading")

        # 2. Dashboard rebuild
        if not args.skip_dashboard:
            dash_cmd = [venv_python, "scripts/build_backtest_dashboard.py"]
            _run(dash_cmd, "dashboard build")
            if args.vercel_deploy:
                vercel_cmd = [venv_python, "scripts/deploy_vercel_dashboard.py", "--deploy"]
                _run_vercel_deploy(vercel_cmd, strict=args.strict_vercel_deploy)

        if use_notify:
            notifier.send(
                title="SOXL Daily Pipeline OK",
                body=f"Paper trading + dashboard completed. Broker={args.broker} dry_run={args.dry_run}",
                level="info",
            )
        print("[scheduler] all done")

    except Exception as exc:
        msg = str(exc)
        print(f"[scheduler] FAILED: {msg}", file=sys.stderr)
        # If data update failed due to transient source issues, retry with
        # existing data only as a dry-run so stale bars cannot move broker state.
        if _is_transient_data_source_failure(exc):
            print("[scheduler] Fresh data unavailable or stale. Retrying cached data as dry-run...", file=sys.stderr)
            try:
                retry_cmd = list(paper_cmd)
                if "--skip-data-update" not in retry_cmd:
                    retry_cmd.append("--skip-data-update")
                if "--dry-run" not in retry_cmd:
                    retry_cmd.append("--dry-run")
                _run(retry_cmd, "paper trading (cached dry-run)")
                if not args.skip_dashboard:
                    dash_cmd = [venv_python, "scripts/build_backtest_dashboard.py"]
                    _run(dash_cmd, "dashboard build")
                if use_notify:
                    notifier.send(
                        title="SOXL Daily Pipeline OK (cached dry-run)",
                        body="Fresh data unavailable or stale after update. Used cached data for dry-run only; no order was placed.",
                        level="warning",
                    )
                print("[scheduler] all done (with cached dry-run)")
                return
            except Exception as exc2:
                msg2 = str(exc2)
                print(f"[scheduler] RETRY FAILED: {msg2}", file=sys.stderr)
                if use_notify:
                    notifier.send(title="SOXL Daily Pipeline FAILED", body=msg2, level="error")
                sys.exit(1)
        if use_notify:
            notifier.send(title="SOXL Daily Pipeline FAILED", body=msg, level="error")
        sys.exit(1)


if __name__ == "__main__":
    main()
