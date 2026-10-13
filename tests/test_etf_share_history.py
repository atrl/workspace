import numpy as np
import pandas as pd
import pytest

from src.data.etf_share_history import parse_ishares_history, share_information


def test_history_schema_lag_missingness_and_share_scale():
    dates = pd.bdate_range("2024-02-20", periods=15)
    values = [100 + index for index in range(15)]
    rows = [["As Of", "NAV per Share", "Ex-Dividends", "Shares Outstanding"]]
    rows += [[str(date.date()), "20", "0", str(value)] for date, value in zip(dates, values)]
    xml = '<ss:Worksheet ss:Name="Historical"><ss:Table>'
    xml += ''.join('<ss:Row>' + ''.join('<ss:Cell><ss:Data>' + item + '</ss:Data></ss:Cell>' for item in row) + '</ss:Row>' for row in rows)
    xml += '</ss:Table></ss:Worksheet>'
    history = parse_ishares_history(xml.encode())
    state = share_information(history, dates, 2)
    assert state.loc[dates[1], "known"] == 0
    assert state.loc[dates[7], "share_change_5d"] == pytest.approx(105 / 100 - 1)
    assert state.loc[dates[7], "source_date"] == dates[5]
    scaled = history.copy(); scaled["shares"] *= 3; scaled["nav"] /= 3
    pd.testing.assert_frame_equal(state, share_information(scaled, dates, 2))
    history.loc[dates[5], "shares"] = np.nan
    missing = share_information(history, dates, 2)
    assert missing.loc[dates[7:13], "share_change_5d"].isna().all()
    with pytest.raises(ValueError, match="schema"):
        parse_ishares_history(xml.replace("Shares Outstanding", "Shares").encode())
    with pytest.raises(ValueError, match="same-session"):
        share_information(history, dates, 0)
