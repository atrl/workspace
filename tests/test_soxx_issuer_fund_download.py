from pathlib import Path

import pytest

from src.data.soxx_issuer_fund_download import read_historical


def test_issuer_spreadsheetml_keeps_original_bytes_and_rejects_duplicate_dates(tmp_path: Path):
    header = ["As Of", "NAV per Share", "Ex-Dividends", "Shares Outstanding"]

    def row(values):
        return "<ss:Row>" + "".join(
            f'<ss:Cell><ss:Data ss:Type="String">{value}</ss:Data></ss:Cell>'
            for value in values) + "</ss:Row>"

    prefix = ('<ss:Workbook xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">'
              '<ss:Worksheet ss:Name="Disclaimers"><ss:Table>'
              '<ss:Row><ss:Cell ss:HRef="https://example.org/?a=1&b=2"/></ss:Row>'
              '</ss:Table></ss:Worksheet><ss:Worksheet ss:Name="Historical"><ss:Table>')
    suffix = '</ss:Table></ss:Worksheet></ss:Workbook>'
    source = tmp_path / "source.xls"
    original = prefix + row(header) + row(["Jan 03, 2011", "18.6", "--", "1200000"]) + suffix
    source.write_text(original)
    frame, audit = read_historical(source)
    assert frame.loc["2011-01-03", "shares_outstanding"] == 1200000
    assert audit["invalid_xml_ampersands_escaped_in_memory"] == 1
    assert source.read_text() == original

    source.write_bytes(("\ufeff\ufeff" + prefix +
        row(["As Of", "Index Level", "NAV per Share", "Ex-Dividends", "Shares Outstanding"]) +
        row(["Jan 03, 2011", "100", "18.6", "--", "1200000"]) + suffix).encode())
    indexed, indexed_audit = read_historical(source)
    assert indexed.loc["2011-01-03", "nav_per_share"] == 18.6
    assert indexed_audit["index_level_column_present"] is True

    source.write_text(original.replace(suffix,
        row(["Jan 03, 2011", "18.7", "--", "1200000"]) + suffix))
    with pytest.raises(ValueError, match="dates or positive numeric"):
        read_historical(source)
