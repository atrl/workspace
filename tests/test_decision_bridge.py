import numpy as np
import pandas as pd
import pytest

from src.analysis.decision_bridge import build_decision_bridge


def test_native_grid_missing_stage_and_actual_holding_duration():
    calendar = pd.bdate_range("2020-12-01", "2022-02-01")
    dates = calendar[(calendar >= "2021-01-04") & (calendar <= "2021-08-02")]
    trace = pd.DataFrame({"trade_date": dates, "source_a": .2, "source_b": .6, "source_c": .2,
                          "proj_a": .1, "proj_b": .5, "proj_c": .4,
                          "orders": [1, 0, 0, np.nan, 0, 1] + [0] * (len(dates) - 6)})
    replay = pd.DataFrame({"trade_date": dates, "strategy_return": .001})
    kwargs = dict(native_audit={"status": "passed"}, source_identity={"recorder_id": "synthetic"},
                  fields={"source_target": ["source_a", "source_b", "source_c"],
                          "projected_target": ["proj_a", "proj_b", "proj_c"],
                          "order_count": "orders"}, native_calendar=calendar)
    result = build_decision_bridge(trace, replay, **kwargs)
    assert len(result["daily"]) == len(dates)
    assert len(result["rolling_126"]) == len(dates) - 125
    assert set(result["summary"]["unknown_stages"]) == {"postfill_weights", "fees_usd"}
    assert result["summary"]["fees_usd_sum"] is None
    assert result["daily"].projected_target_to_postfill_weights_l1.isna().all()
    assert result["daily"].sessions_since_last_filled_order.iloc[:3].tolist() == [0, 1, 2]
    assert result["daily"].sessions_since_last_filled_order.iloc[3:5].isna().all()
    assert not result["summary"]["promotion_eligible"]
    assert not result["periods"].query("kind == 'year'").complete_period.any()
    with pytest.raises(ValueError, match="same nonempty date grid"):
        build_decision_bridge(trace.iloc[1:], replay, **kwargs)
    with pytest.raises(ValueError, match="omit native sessions"):
        build_decision_bridge(trace.drop(index=5), replay.drop(index=5), **kwargs)
