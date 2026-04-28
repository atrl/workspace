#!/usr/bin/env python3
"""Analyze the learned router's mode distribution on test data vs rule router."""
import sys
sys.path.insert(0, "/Users/rziiiii/.openclaw/workspace")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.soxl_quant.data import build_dataset, feature_columns_for_packs, init_qlib
from src.soxl_quant.policy.environment import TrancheLedger, build_price_path
from src.soxl_quant.policy.modes import classify_rule_mode, route_mode, ModeState, ModeRouterConfig
from src.soxl_quant.policy.models.mode_classifier import LearnedModeRouter
from src.soxl_quant.policy.observation import make_observation
from src.soxl_quant.settings import LOOKBACK, QLIB_DATA_PATH, TEST_SEG, load_profile


def analyze():
    init_qlib(QLIB_DATA_PATH)
    profile = load_profile("balanced_rule_residual_learned_mode_v1")
    dataset = build_dataset(feature_packs=tuple(profile.feature.packs), profile_name=profile.name)
    df = dataset.handler.feature_frame.droplevel("instrument").sort_index()

    slices = {
        "test_proxy": TEST_SEG,
        "test_2020": ("2020-01-01", "2020-12-31"),
        "test_2022": ("2022-01-01", "2022-12-31"),
        "test_2023_2024": ("2023-01-01", "2024-12-31"),
    }

    learned_router = LearnedModeRouter("runtime/model/card/mode_classifier/mode_classifier.pt")
    rule_cfg = profile.mode_router
    learned_cfg = ModeRouterConfig(
        enabled=True,
        use_learned=True,
        learned_model_path="runtime/model/card/mode_classifier/mode_classifier.pt",
        trend_hold_min_dwell_days=rule_cfg.trend_hold_min_dwell_days,
        defensive_hold_min_dwell_days=rule_cfg.defensive_hold_min_dwell_days,
        crash_rebuild_min_dwell_days=rule_cfg.crash_rebuild_min_dwell_days,
        flat_wait_min_dwell_days=rule_cfg.flat_wait_min_dwell_days,
    )

    results = {}
    for sname, (s, e) in slices.items():
        seg = df.loc[s:e].copy()
        if len(seg) <= 120:
            continue
        price_path = build_price_path(seg)

        rule_modes = []
        learned_modes = []
        regime_risks = []
        cum_drops = []

        pos = 0.0
        cash = 100_000.0
        peak = cash
        ledger = TrancheLedger()
        mode_state_rule = ModeState()
        mode_state_learned = ModeState()

        for idx in range(LOOKBACK, len(seg) - 1):
            obs = make_observation(seg, idx, pos, cash, peak, ledger, price_path,
                                   observation_window=profile.model.observation_window,
                                   feature_columns=feature_columns_for_packs(profile.feature.packs, profile.name))
            mode_state_rule = route_mode(obs, mode_state_rule, rule_cfg)
            mode_state_learned = route_mode(obs, mode_state_learned, learned_cfg)
            rule_modes.append(mode_state_rule.name)
            learned_modes.append(mode_state_learned.name)
            regime_risks.append(float(seg.iloc[idx]["regime_risk"]))
            cum_drops.append(float(seg.iloc[idx]["cum_drop_30"]))

        from collections import Counter
        rule_counts = Counter(rule_modes)
        learned_counts = Counter(learned_modes)

        # Compute agreement
        agreement = sum(1 for r, l in zip(rule_modes, learned_modes) if r == l) / len(rule_modes)

        # Correlation: learned flat_wait vs regime_risk
        flat_wait_mask = [1 if m == "flat_wait" else 0 for m in learned_modes]
        avg_rr_when_flat = np.mean([regime_risks[i] for i in range(len(learned_modes)) if learned_modes[i] == "flat_wait"]) if any(flat_wait_mask) else 0

        results[sname] = {
            "agreement": agreement,
            "rule_distribution": dict(rule_counts),
            "learned_distribution": dict(learned_counts),
            "avg_regime_risk_when_flat_wait": avg_rr_when_flat,
            "learned_flat_wait_fraction": learned_counts.get("flat_wait", 0) / len(learned_modes),
            "rule_flat_wait_fraction": rule_counts.get("flat_wait", 0) / len(rule_modes),
        }

    print("\n" + "="*80)
    print(f"{'Slice':<16} {'Agreement':>10} {'Rule flat%':>12} {'Learned flat%':>15} {'Avg RR@flat':>12}")
    print("="*80)
    for sname, r in results.items():
        print(f"{sname:<16} {r['agreement']:>9.1%} {r['rule_flat_wait_fraction']:>11.1%} {r['learned_flat_wait_fraction']:>14.1%} {r['avg_regime_risk_when_flat_wait']:>11.2f}")
    print("="*80)

    print("\nDetailed distributions:")
    for sname, r in results.items():
        print(f"\n{sname}:")
        print(f"  Rule:    {r['rule_distribution']}")
        print(f"  Learned: {r['learned_distribution']}")

    out = Path("runtime/model/learned_router_runtime_analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    analyze()
