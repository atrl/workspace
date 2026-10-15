import pytest
from src.workflow.cpi_official_text_intake import parse_rendered_header


def test_utf8_renderer_separator_and_line_numbers_preserve_complete_header():
    capture='L177: Transmission of material in this release is embargoed until\nL178: 8:30 a.m. (EST) January 11, 2019 USDL-19-0018\nL179: \nL183: CONSUMER PRICE INDEX �DECEMBER 2018'
    event=parse_rendered_header(capture,'https://www.bls.gov/news.release/archives/cpi_01112019.htm')
    assert event['release_date']=='2019-01-11' and event['observation_month']=='2018-12'
    with pytest.raises(ValueError,match='filename'):parse_rendered_header(capture,'https://www.bls.gov/news.release/archives/cpi_01102019.htm')
