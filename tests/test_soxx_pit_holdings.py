from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


ATOM_XML = """
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <content type="text/xml">
      <accession-number>0000000000-20-000002</accession-number>
      <filing-date>2020-06-01</filing-date>
      <filing-type>NPORT-P</filing-type>
    </content>
  </entry>
  <entry>
    <content type="text/xml">
      <accession-number>0000000000-19-000001</accession-number>
      <filing-date>2019-11-25</filing-date>
      <filing-type>NPORT-P</filing-type>
    </content>
  </entry>
</feed>
"""


def _primary_xml(
    *,
    report_date: str = "2019-09-30",
    series_id: str = "S000004354",
    amd_weight: float = 60.0,
    nvda_weight: float = 40.0,
) -> str:
    return f"""
<edgarSubmission xmlns="http://www.sec.gov/edgar/nport">
  <headerData>
    <filerInfo>
      <filer>
        <seriesClassInfo>
          <seriesId>{series_id}</seriesId>
          <classId>C000012084</classId>
        </seriesClassInfo>
      </filer>
    </filerInfo>
  </headerData>
  <formData>
    <genInfo><repPdDate>{report_date}</repPdDate></genInfo>
    <invstOrSecs>
      <invstOrSec>
        <name>Advanced Micro Devices, Inc.</name>
        <cusip>007903107</cusip>
        <identifiers><isin value="US0079031078"/></identifiers>
        <valUSD>600</valUSD><pctVal>{amd_weight}</pctVal>
        <assetCat>EC</assetCat><invCountry>US</invCountry>
      </invstOrSec>
      <invstOrSec>
        <name>NVIDIA Corp.</name>
        <cusip>67066G104</cusip>
        <identifiers><isin value="US67066G1040"/></identifiers>
        <valUSD>400</valUSD><pctVal>{nvda_weight}</pctVal>
        <assetCat>EC</assetCat><invCountry>US</invCountry>
      </invstOrSec>
      <invstOrSec>
        <name>Cash</name><cusip>NONE</cusip>
        <valUSD>1</valUSD><pctVal>0.1</pctVal><assetCat>STIV</assetCat>
      </invstOrSec>
    </invstOrSecs>
  </formData>
</edgarSubmission>
"""


def test_parse_sec_soxx_nport_atom_sorts_public_filings() -> None:
    from src.data.soxx_pit_holdings import parse_sec_soxx_nport_atom

    filings = parse_sec_soxx_nport_atom(ATOM_XML)

    assert [item.filing_date for item in filings] == [
        "2019-11-25",
        "2020-06-01",
    ]
    assert filings[0].primary_xml_url.endswith(
        "/000000000019000001/primary_doc.xml"
    )


def test_parse_sec_soxx_nport_primary_xml_maps_cusip_to_ticker() -> None:
    from src.data.soxx_pit_holdings import (
        SecNportFiling,
        parse_sec_soxx_nport_primary_xml,
    )

    filing = SecNportFiling(
        accession="0000000000-19-000001",
        filing_date="2019-11-25",
        form="NPORT-P",
    )
    frame = parse_sec_soxx_nport_primary_xml(
        _primary_xml(),
        filing=filing,
    )

    assert frame["ticker"].tolist() == ["AMD", "NVDA"]
    assert frame["weight_pct"].tolist() == [60.0, 40.0]
    assert set(frame["report_date"]) == {"2019-09-30"}
    assert set(frame["filing_date"]) == {"2019-11-25"}


def test_parse_sec_soxx_nport_primary_xml_fails_closed_on_wrong_series() -> None:
    from src.data.soxx_pit_holdings import (
        SecNportFiling,
        parse_sec_soxx_nport_primary_xml,
    )

    with pytest.raises(ValueError, match="series mismatch"):
        parse_sec_soxx_nport_primary_xml(
            _primary_xml(series_id="S000000000"),
            filing=SecNportFiling(
                accession="0000000000-19-000001",
                filing_date="2019-11-25",
                form="NPORT-P",
            ),
        )


def test_build_pit_soxx_weight_panel_never_backdates_report_weight() -> None:
    from src.data.soxx_pit_holdings import (
        SecNportFiling,
        build_pit_soxx_weight_panel,
        parse_sec_soxx_nport_primary_xml,
    )

    first = parse_sec_soxx_nport_primary_xml(
        _primary_xml(report_date="2019-09-30"),
        filing=SecNportFiling(
            accession="0000000000-19-000001",
            filing_date="2019-11-25",
            form="NPORT-P",
        ),
    )
    second = parse_sec_soxx_nport_primary_xml(
        _primary_xml(
            report_date="2019-12-31",
            amd_weight=45.0,
            nvda_weight=55.0,
        ),
        filing=SecNportFiling(
            accession="0000000000-20-000002",
            filing_date="2020-02-27",
            form="NPORT-P",
        ),
    )
    calendar = pd.to_datetime(
        [
            "2019-09-30",
            "2019-11-22",
            "2019-11-25",
            "2020-02-26",
            "2020-02-27",
        ]
    )

    panel = build_pit_soxx_weight_panel(
        pd.concat([first, second], ignore_index=True),
        calendar=calendar,
    )

    assert panel["datetime"].min() == pd.Timestamp("2019-11-25")
    amd = panel.loc[panel["instrument"].eq("AMD")].set_index("datetime")
    assert amd.loc[pd.Timestamp("2019-11-25"), "weight_pct"] == 60.0
    assert amd.loc[pd.Timestamp("2020-02-26"), "weight_pct"] == 60.0
    assert amd.loc[pd.Timestamp("2020-02-27"), "weight_pct"] == 45.0
    assert (
        pd.to_datetime(panel["datetime"])
        >= pd.to_datetime(panel["filing_date"])
    ).all()


def test_fetch_and_store_sec_soxx_nport_history_writes_audit_manifest(
    tmp_path: Path,
) -> None:
    from src.data.soxx_pit_holdings import (
        SoxxPitHoldingsPaths,
        fetch_and_store_sec_soxx_nport_history,
    )

    class Response:
        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self) -> None:
            return None

    def request_get(url: str, **_: object) -> Response:
        if "browse-edgar" in url:
            return Response(ATOM_XML.encode())
        if "2019000001" in url:
            return Response(_primary_xml().encode())
        return Response(
            _primary_xml(
                report_date="2020-03-31",
                amd_weight=45.0,
                nvda_weight=55.0,
            ).encode()
        )

    paths = SoxxPitHoldingsPaths(root=tmp_path / "soxx")
    manifest = fetch_and_store_sec_soxx_nport_history(
        paths=paths,
        request_get=request_get,
        sleep_seconds=0.0,
    )
    saved = pd.read_csv(paths.history_csv)

    assert manifest["snapshot_count"] == 2
    assert manifest["instrument_count"] == 2
    assert manifest["availability_policy"].startswith(
        "holdings become model-eligible"
    )
    assert len(saved) == 4
    assert paths.manifest_json.exists()


def test_fetch_and_store_soxx_pit_prices_reports_membership_weight_coverage(
    tmp_path: Path,
) -> None:
    from src.data.soxx_pit_holdings import (
        SecNportFiling,
        SoxxPitHoldingsPaths,
        fetch_and_store_soxx_pit_price_history,
        parse_sec_soxx_nport_primary_xml,
    )

    paths = SoxxPitHoldingsPaths(root=tmp_path / "soxx")
    paths.root.mkdir(parents=True)
    history = parse_sec_soxx_nport_primary_xml(
        _primary_xml(),
        filing=SecNportFiling(
            accession="0000000000-19-000001",
            filing_date="2019-11-25",
            form="NPORT-P",
        ),
    )
    history.to_csv(paths.history_csv, index=False)

    def price_fetcher(symbol: str) -> pd.DataFrame:
        if symbol == "NVDA":
            raise RuntimeError("delisted-provider-example")
        return pd.DataFrame(
            {
                "date": ["2019-09-30", "2019-10-01"],
                "open": [1.0, 1.1],
                "high": [1.1, 1.2],
                "low": [0.9, 1.0],
                "close": [1.05, 1.15],
                "adj_close": [1.05, 1.15],
                "volume": [100, 110],
            }
        )

    manifest = fetch_and_store_soxx_pit_price_history(
        paths=paths,
        start="2019-09-30",
        end="2019-10-01",
        sleep_seconds=0.0,
        reuse_existing=False,
        price_fetcher=price_fetcher,
    )

    assert manifest["requested_instruments"] == 2
    assert manifest["usable_instruments"] == 1
    assert manifest["snapshot_usable_weight_pct_min"] == pytest.approx(60.0)
    coverage = pd.read_csv(paths.price_coverage_csv).set_index("symbol")
    assert coverage.loc["AMD", "status"] == "ok"
    assert coverage.loc["NVDA", "status"] == "error"
