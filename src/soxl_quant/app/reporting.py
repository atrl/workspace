from __future__ import annotations

from pathlib import Path

from ..backtest import save_results
from ..settings import RESULT_PNG, RESULT_TXT
from .dashboard import rebuild_dashboard


def persist_and_print_results(results, walk_forward_results) -> None:
    save_results(results)
    dashboard_ok = rebuild_dashboard()
    print(f"\n[done] chart={RESULT_PNG.name} summary={RESULT_TXT.name}")
    print(f"  - dashboard: {'refreshed' if dashboard_ok else 'refresh_failed'}")
    for name, result in results.items():
        print(
            f"  - {name}: {result['return']:.1f}% | Sharpe {result['sharpe']:.2f} | "
            f"MaxDD {result['max_dd']:.1f}% | trades {result['trades']}"
        )
        for wf_name, wf_result in walk_forward_results.get(name, {}).items():
            print(
                f"    {wf_name}: {wf_result['return']:.1f}% | Sharpe {wf_result['sharpe']:.2f} | "
                f"MaxDD {wf_result['max_dd']:.1f}% | trades {wf_result['trades']}"
            )
        promotion = result.get("promotion", {})
        if promotion:
            status = "eligible" if promotion.get("eligible") else "blocked"
            print(f"    promotion: {status}")


def print_profile_result(profile_name: str, bt_result: dict) -> None:
    print(
        f"[test:{profile_name}] return={bt_result['return']:.1f}% "
        f"sharpe={bt_result['sharpe']:.2f} maxdd={bt_result['max_dd']:.1f}% trades={bt_result['trades']} "
        f"model={Path(bt_result['model_path']).name}"
    )


__all__ = ["persist_and_print_results", "print_profile_result"]
