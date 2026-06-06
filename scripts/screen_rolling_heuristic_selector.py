#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.heuristic_strategy_search import (
    DEFAULT_HEURISTIC_WINDOWS,
    build_heuristic_position_path,
    default_heuristic_config_grid,
    rolling_select_candidate_path,
    screen_heuristic_strategies,
)
from src.soxl_quant.app.quantbench_gates import compute_backtest_metrics, slice_window


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen a rolling selector over heuristic SOXL strategies")
    parser.add_argument("--panel", default="data/processed/panels/soxl_soxx_vix_full.csv")
    parser.add_argument("--output-dir", default="runtime/model_selection/card/rolling_heuristic_selector_v1")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--rebalance-days", type=int, default=21)
    parser.add_argument("--min-history-days", type=int, default=252)
    parser.add_argument("--initial-candidate-id", type=int, default=0)
    parser.add_argument("--max-drawdown-pct", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    panel = pd.read_csv(ROOT / args.panel)
    configs = default_heuristic_config_grid()
    screen = screen_heuristic_strategies(panel, configs, max_drawdown_pct=args.max_drawdown_pct)
    selected_config_ids = [int(v) for v in screen.head(max(int(args.top_n), 1))["config_id"].tolist()]
    selected_configs = [configs[idx] for idx in selected_config_ids]
    candidate_paths = [build_heuristic_position_path(panel, config) for config in selected_configs]
    daily = rolling_select_candidate_path(
        panel,
        candidate_paths,
        lookback_days=args.lookback_days,
        rebalance_days=args.rebalance_days,
        min_history_days=args.min_history_days,
        max_drawdown_pct=args.max_drawdown_pct,
        initial_candidate_id=args.initial_candidate_id,
    )
    rows = []
    gate_pass = True
    for window_name, (start, end) in DEFAULT_HEURISTIC_WINDOWS.items():
        metric = compute_backtest_metrics(
            slice_window(daily, start, end),
            window=window_name,
            profile="rolling_heuristic_selector",
            max_drawdown_pct=args.max_drawdown_pct,
        )
        payload = asdict(metric)
        payload["failures"] = list(metric.failures)
        rows.append(payload)
        gate_pass = gate_pass and metric.gate_pass

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    screen.to_csv(output_dir / "heuristic_strategy_screen.csv", index=False)
    daily.to_csv(output_dir / "rolling_selector_daily.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "rolling_selector_metrics.csv", index=False)
    config_table = pd.DataFrame(
        [{"selector_candidate_id": i, "source_config_id": selected_config_ids[i], **asdict(config)} for i, config in enumerate(selected_configs)]
    )
    config_table.to_csv(output_dir / "rolling_selector_candidates.csv", index=False)
    summary = {
        "experiment": "rolling_heuristic_selector_v1",
        "decision": "hard_gate_pass_found" if gate_pass else "hard_gate_no_rolling_selector_passes",
        "promotion_allowed": bool(gate_pass),
        "training_allowed_from_this_card": False,
        "panel": args.panel,
        "configs_screened": int(len(screen)),
        "top_n": int(args.top_n),
        "selected_config_ids": selected_config_ids,
        "lookback_days": int(args.lookback_days),
        "rebalance_days": int(args.rebalance_days),
        "min_history_days": int(args.min_history_days),
        "initial_candidate_id": int(args.initial_candidate_id),
        "max_drawdown_pct": float(args.max_drawdown_pct),
        "window_metrics": rows,
        "next_action": (
            "The rolling selector passed 2022 but failed 2023-2024 because the static heuristic family has no "
            "positive-excess 2023-2024 candidate; next branch should use model replay/BC candidate paths with "
            "cached lookback scoring or train a stronger reentry learner."
        ),
        "sidecars": [
            "summary.json",
            "heuristic_strategy_screen.csv",
            "rolling_selector_candidates.csv",
            "rolling_selector_daily.csv",
            "rolling_selector_metrics.csv",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "rolling_heuristic_selector "
        f"gate={summary['decision']} top_n={args.top_n} output={args.output_dir}"
    )
    return summary


if __name__ == "__main__":
    main()
