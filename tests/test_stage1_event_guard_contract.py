import numpy as np
import pandas as pd

from src.analysis.stage1_event_guard_labels import _features
from src.workflow.stage1_event_guard_information import _first_passage


def test_event_features_never_consume_later_minutes():
    day = pd.Timestamp("2021-06-01")
    stamps = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=35, freq="1min")
    first = pd.DataFrame({"SOXL_close": np.linspace(10., 10.5, len(stamps)),
        "SOXX_close": np.linspace(100., 101., len(stamps))}, index=stamps)
    second = first.copy()
    second.loc[stamps[22]:, "SOXL_close"] *= .8
    common = dict(q=np.array([100., 10.]), cash=1000., known_drawdown=.05,
                  cap=.8, alpha=.5, tail=.2,
                  close_at=day + pd.Timedelta(hours=16))
    left = _features(day, first, **common)
    right = _features(day, second, **common)
    cutoff = stamps[21]
    pd.testing.assert_frame_equal(left.loc[left.bar_end_at.le(cutoff)].reset_index(drop=True),
                                  right.loc[right.bar_end_at.le(cutoff)].reset_index(drop=True))


def test_first_passage_uses_earliest_positive_score_once_per_day():
    day = pd.Timestamp("2021-06-01")
    rows = pd.DataFrame({"trade_date": [day] * 3,
        "decision_at": [day + pd.Timedelta(hours=10, minutes=x) for x in (1, 2, 3)],
        "local_advantage": [.02, -.01, .03]})
    chosen = _first_passage(rows, np.array([-.1, .01, .2]))
    assert len(chosen) == 1
    assert chosen.decision_at.iloc[0] == rows.decision_at.iloc[1]
    assert chosen.local_advantage.iloc[0] == -.01
