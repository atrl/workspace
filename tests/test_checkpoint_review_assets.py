from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_balanced_checkpoint_review_profile_loads():
    path = ROOT / "configs" / "soxl" / "balanced_checkpoint_review.yaml"
    text = path.read_text(encoding="utf-8")
    assert "name: balanced_checkpoint_review" in text
    assert "staged_training: true" in text
    assert "stage_iters: 20" in text
    assert "bear_min_cap: 0.30" in text


def test_position_cap_overlay_variants_are_non_empty():
    path = ROOT / "configs" / "soxl" / "position_cap_overlay_variants.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "base_bundle" in payload
    assert "bear_floor25_regime055" in payload
    assert payload["bear_floor25_regime055"]["bear_min_cap"] == 0.25
    assert payload["bull70_bear35_regime060"]["bull_min_cap"] == 0.70


def test_balanced_stress_select_v2_profile_exists():
    path = ROOT / "configs" / "soxl" / "balanced_stress_select_v2.yaml"
    text = path.read_text(encoding="utf-8")
    assert "name: balanced_stress_select_v2" in text
    assert "stress_avg_pos_penalty: 25.0" in text
    assert "staged_training: true" in text
