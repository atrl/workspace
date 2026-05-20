from __future__ import annotations

from scripts.review_selective_release_promotion import build_promotion_review
from src.soxl_quant.policy.execution_overlays import POST_RELEASE_CAP_VARIANT


def test_post_release_cap_review_requires_walkforward_improvement_and_fixed_noop():
    walk_rows = [
        {
            "actor_window": "wf_2022",
            "variant": "baseline",
            "return": -34.1052,
            "max_dd": 38.2746,
            "changed_days": 0,
        },
        {
            "actor_window": "wf_2022",
            "variant": POST_RELEASE_CAP_VARIANT,
            "return": -34.1052,
            "max_dd": 38.2746,
            "changed_days": 0,
        },
        {
            "actor_window": "wf_2023_2024",
            "variant": "baseline",
            "return": -20.2056,
            "max_dd": 38.5425,
            "changed_days": 0,
        },
        {
            "actor_window": "wf_2023_2024",
            "variant": POST_RELEASE_CAP_VARIANT,
            "return": -11.3675,
            "max_dd": 27.6610,
            "changed_days": 185,
        },
    ]
    fixed_rows = [
        {"window": "test_proxy", "variant": "baseline", "return": 1511.9729, "max_dd": 56.4821, "trades": 174},
        {
            "window": "test_proxy",
            "variant": POST_RELEASE_CAP_VARIANT,
            "return": 1511.9729,
            "max_dd": 56.4821,
            "trades": 174,
        },
        {"window": "test_2022", "variant": "baseline", "return": -30.9725, "max_dd": 40.6407, "trades": 18},
        {
            "window": "test_2022",
            "variant": POST_RELEASE_CAP_VARIANT,
            "return": -30.9725,
            "max_dd": 40.6407,
            "trades": 18,
        },
    ]

    review = build_promotion_review(walk_rows, fixed_rows, variant=POST_RELEASE_CAP_VARIANT)

    assert review["decision"] == "promotion_review_candidate"
    assert review["runtime_auto_select_allowed"] is False
    assert review["checks"]["wf_2022_unchanged"]["passed"] is True
    assert review["checks"]["wf_2023_2024_return_improved"]["passed"] is True
    assert review["checks"]["wf_2023_2024_maxdd_improved"]["passed"] is True
    assert review["checks"]["fixed_actor_noop"]["passed"] is True
