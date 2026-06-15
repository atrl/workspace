from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_skill_records_dashboard_v2_operational_contract():
    text = (ROOT / ".codex" / "skills" / "dashboard" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    normalized = re.sub(r"\s+", " ", text)

    required_fragments = [
        "dashboard_v2.html",
        "web/dashboard-v2/dist/index.html",
        "query/hash",
        "Chart start/end",
        "Run start/end",
        "Use chart range",
        "window.__SOXL_DASHBOARD_CHART_DEBUG__",
        "Kimi WebBridge",
        "evidence-only models",
        "must not be offered as runnable inference",
    ]

    for fragment in required_fragments:
        assert fragment in normalized
