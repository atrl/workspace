from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.official_candidate_aggregation import (
    OfficialReplayCandidateConfig,
    normalise_official_replay_candidate_frame,
    score_official_replay_candidate_pool,
)


def test_normalise_official_replay_candidates_keeps_source_identity() -> None:
    frame = pd.DataFrame(
        [
            {
                "variant": "near_miss",
                "fold": "wf_2022",
                "official_warm_relative_return": 83.5,
                "official_warm_max_dd": 8.7,
                "official_warm_avg_pos": 0.06,
                "full_gate": True,
            },
            {
                "variant": "near_miss",
                "fold": "wf_2023_2024",
                "official_warm_relative_return": 29.8,
                "official_warm_max_dd": 15.1,
                "official_warm_avg_pos": 0.21,
                "full_gate": False,
            },
        ]
    )

    normalised = normalise_official_replay_candidate_frame(
        frame,
        source_label="sell_guard",
        mechanism="action_cap_overlay",
        candidate_col="variant",
    )

    assert normalised["candidate_id"].tolist() == [
        "sell_guard::near_miss",
        "sell_guard::near_miss",
    ]
    assert normalised["source_label"].unique().tolist() == ["sell_guard"]
    assert normalised["mechanism"].unique().tolist() == ["action_cap_overlay"]
    assert normalised["relative_gap"].round(3).tolist() == [0.0, 0.2]
    assert normalised["max_dd_gap"].round(3).tolist() == [0.0, 0.1]


def test_score_official_replay_candidate_pool_flags_near_miss_not_robust() -> None:
    pool = pd.DataFrame(
        [
            {
                "source_label": "sell_guard",
                "mechanism": "action_cap_overlay",
                "candidate": "cap05_hit2",
                "candidate_id": "sell_guard::cap05_hit2",
                "window": "wf_2022",
                "relative_return": 83.56,
                "max_dd": 8.75,
                "avg_pos": 0.06,
                "window_gate_pass": True,
                "relative_gap": 0.0,
                "max_dd_gap": 0.0,
            },
            {
                "source_label": "sell_guard",
                "mechanism": "action_cap_overlay",
                "candidate": "cap05_hit2",
                "candidate_id": "sell_guard::cap05_hit2",
                "window": "wf_2023_2024",
                "relative_return": 29.82,
                "max_dd": 15.11,
                "avg_pos": 0.21,
                "window_gate_pass": False,
                "relative_gap": 0.18,
                "max_dd_gap": 0.11,
            },
            {
                "source_label": "target_cap",
                "mechanism": "hard_target_cap",
                "candidate": "volstress_cap010",
                "candidate_id": "target_cap::volstress_cap010",
                "window": "wf_2022",
                "relative_return": 80.67,
                "max_dd": 11.95,
                "avg_pos": 0.08,
                "window_gate_pass": True,
                "relative_gap": 0.0,
                "max_dd_gap": 0.0,
            },
            {
                "source_label": "target_cap",
                "mechanism": "hard_target_cap",
                "candidate": "volstress_cap010",
                "candidate_id": "target_cap::volstress_cap010",
                "window": "wf_2023_2024",
                "relative_return": 2.0,
                "max_dd": 15.11,
                "avg_pos": 0.15,
                "window_gate_pass": False,
                "relative_gap": 28.0,
                "max_dd_gap": 0.11,
            },
        ]
    )

    scored = score_official_replay_candidate_pool(
        pool,
        config=OfficialReplayCandidateConfig(
            robust_relative_margin_pp=2.0,
            robust_max_dd_margin_pct=1.0,
            near_miss_relative_gap_pp=1.0,
            near_miss_max_dd_gap_pct=0.5,
        ),
    )

    top = scored.iloc[0]
    assert top["candidate_id"] == "sell_guard::cap05_hit2"
    assert bool(top["full_gate"]) is False
    assert bool(top["near_miss"]) is True
    assert bool(top["robust_full_gate"]) is False
    assert top["failure_profile"] == "near_miss"
    assert scored.iloc[1]["failure_profile"] == "trend_capture"
