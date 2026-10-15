import pandas as pd
import pytest
from src.data.cpi_release_calendar import parse_header


def test_actual_release_date_not_observation_month_and_clock_validation():
    raw=b'<pre>Transmission of material in this release is embargoed until 8:30 a.m. (EST) January 11, 2019 USDL-19-0010 CONSUMER PRICE INDEX - DECEMBER 2018</pre>'
    event=parse_header(raw,'https://www.bls.gov/news.release/archives/cpi_01112019.htm')
    assert event['release_date']=='2019-01-11' and event['observation_month']=='2018-12'
    assert pd.Timestamp(event['release_at']).hour==8
    with pytest.raises(ValueError,match='filename'):parse_header(raw,'https://www.bls.gov/news.release/archives/cpi_01102019.htm')
    with pytest.raises(ValueError,match='offset'):parse_header(raw.replace(b'(EST)',b'(EDT)'),'https://www.bls.gov/news.release/archives/cpi_01112019.htm')
    with pytest.raises(ValueError,match='clock'):parse_header(raw.replace(b'8:30',b'10:00'),'https://www.bls.gov/news.release/archives/cpi_01112019.htm')
