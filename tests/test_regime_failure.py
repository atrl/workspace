import pandas as pd
import pytest
from src.analysis.regime_failure import regime_features


def test_regime_rejects_information_at_or_after_decision():
    frame=pd.DataFrame({'soxx_ret5':[-.02],'soxx_ret20':[-.1],
        'feature_available_at':[pd.Timestamp('2026-10-12 16:01')],
        'decision_at':[pd.Timestamp('2026-10-13 09:46')]})
    result=regime_features(frame)
    frame['close_soxx']=9999.0  # Same-day outcome must not become a conditioning input.
    pd.testing.assert_frame_equal(result,regime_features(frame))
    frame['feature_available_at']=frame.decision_at
    with pytest.raises(ValueError,match='known before decision'):
        regime_features(frame)
