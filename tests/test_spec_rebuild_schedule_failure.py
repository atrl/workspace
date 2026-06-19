from __future__ import annotations

import pytest

from src.schedule import RefreshPlan, run_refresh_plan


def test_refresh_plan_notifies_failure_summary_and_releases_lock(tmp_path):
    notifications: list[dict] = []
    plan = RefreshPlan(name="failing_refresh", lock_path=tmp_path / "locks" / "refresh.lock", attempts=2, delay_seconds=0.0)

    def data_update():
        raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        run_refresh_plan(
            plan,
            data_update=data_update,
            model_refresh=lambda: None,
            dashboard_export=lambda manifests, comparison: {},
            notify=notifications.append,
        )

    assert notifications == [
        {
            "name": "failing_refresh",
            "status": "failed",
            "attempts": 2,
            "error": "provider unavailable",
        }
    ]
    assert not plan.lock_path.exists()
