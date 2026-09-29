from datetime import datetime
from zoneinfo import ZoneInfo

from src.schedule.stage1_economic_shadow import in_open_window, run_shadow_schedule


def test_open_window_is_new_york_local_across_dst():
    utc = ZoneInfo("UTC")
    assert in_open_window(datetime(2026, 9, 29, 13, 31, tzinfo=utc))
    assert in_open_window(datetime(2026, 1, 6, 14, 31, tzinfo=utc))
    assert not in_open_window(datetime(2026, 1, 6, 13, 31, tzinfo=utc))
    assert not in_open_window(datetime(2026, 9, 29, 13, 37, tzinfo=utc))
    assert not in_open_window(datetime(2026, 9, 26, 13, 31, tzinfo=utc))


def test_waiting_scheduler_does_not_connect_or_require_broker_configuration(tmp_path):
    def forbidden(**kwargs):
        raise AssertionError("off-session quote connection attempted")
    result = run_shadow_schedule(root=tmp_path, config_path=tmp_path / "absent.json",
        now=datetime(2026, 9, 29, 3, 0, tzinfo=ZoneInfo("America/New_York")), quote_reader=forbidden)
    assert result["status"] == "waiting"
    assert result["orders_submitted"] == 0
    assert (tmp_path / "runtime/soxl/stage1_economic_shadow/latest_status.json").exists()
