import numpy as np
import pandas as pd

from src.data.guidance_information import guidance_information


def test_late_weekend_document_waits_for_native_decision_and_unknown_resets_state():
    calendar = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
    review = pd.DataFrame([
        {"symbol": "AMD", "report_date": "2024-01-04", "newsletter_date": "2024-01-05", "status": "numeric_pair", "numeric_pair_present": True, "direction": 1., "reported_midpoint_gap": .1},
        {"symbol": "AMD", "report_date": "2024-01-08", "newsletter_date": "2024-01-09", "status": "source_missing", "numeric_pair_present": False, "direction": np.nan, "reported_midpoint_gap": np.nan}])
    manifest = {"entries": [{"symbol": "AMD", "report_date": "2024-01-04", "created_at": "2024-01-05T07:00:00-05:00", "modified_at": "2024-01-06T12:00:00-05:00"}]}
    state, _ = guidance_information(review, manifest, calendar)
    assert state.amd_direction_known.tolist() == [0., 0., 1., 0.]
    assert state.amd_gap.iloc[2] == .1 and np.isnan(state.amd_gap.iloc[3])
