from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_balanced_checkpoint_review_card_in_runtime():
    path = ROOT / "runtime" / "model" / "cards" / "balanced_checkpoint_review" / "card.md"
    text = path.read_text(encoding="utf-8")
    assert "balanced_checkpoint_review" in text


def test_position_cap_overlay_variants_are_non_empty():
    path = ROOT / "runtime" / "configs" / "position_cap_overlay_variants.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "base_bundle" in payload
    assert "bear_floor25_regime055" in payload
    assert payload["bear_floor25_regime055"]["bear_min_cap"] == 0.25
    assert payload["bull70_bear35_regime060"]["bull_min_cap"] == 0.70


# Legacy profile existence tests removed for dropped profiles.
# Archived/dropped yaml files were cleaned up from configs/soxl/ root.
# Profiles are now loaded from experiments/ first, then legacy fallback.
