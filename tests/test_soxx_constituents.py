from __future__ import annotations

import json

import pandas as pd


WORKBOOK_XML = """
<ss:Workbook xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">
<ss:Row>
  <ss:Cell><ss:Data ss:Type="String">Fund Holdings as of</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Jul 02, 2026</ss:Data></ss:Cell>
</ss:Row>
<ss:Row>
  <ss:Cell><ss:Data ss:Type="String">Number of Securities (excluding cash and derivatives)</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">30.00</ss:Data></ss:Cell>
</ss:Row>
<ss:Row>
  <ss:Cell><ss:Data ss:Type="String">Ticker</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Name</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Sector</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Asset Class</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Market Value</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Weight (%)</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Notional Value</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Quantity</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Price</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Location</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Exchange</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Currency</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">FX Rate</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Accrual Date</ss:Data></ss:Cell>
</ss:Row>
<ss:Row>
  <ss:Cell><ss:Data ss:Type="String">MU</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">MICRON TECHNOLOGY INC</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Information Technology</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Equity</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">3439671397.08</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">8.16407</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">3439671397.08</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">3525843</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">975.56</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">United States</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">NASDAQ</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">USD</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">1</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">--</ss:Data></ss:Cell>
</ss:Row>
<ss:Row>
  <ss:Cell><ss:Data ss:Type="String">AMD</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">ADVANCED MICRO DEVICES INC</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Information Technology</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Equity</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">3435016448.02</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">8.15302</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">3435016448.02</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">6633611</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">517.82</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">United States</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">NASDAQ</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">USD</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">1</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">--</ss:Data></ss:Cell>
</ss:Row>
<ss:Row>
  <ss:Cell><ss:Data ss:Type="String">USD</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">USD CASH</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Cash and/or Derivatives</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">Cash</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">-100</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">-0.1</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">-100</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">0</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">--</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">--</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">--</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">USD</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="Number">1</ss:Data></ss:Cell>
  <ss:Cell><ss:Data ss:Type="String">--</ss:Data></ss:Cell>
</ss:Row>
</ss:Workbook>
"""


def _price_frame(symbol: str) -> pd.DataFrame:
    base = 100.0 if symbol == "MU" else 200.0
    return pd.DataFrame(
        {
            "date": ["2026-07-01", "2026-07-02"],
            "open": [base, base + 1.0],
            "high": [base + 2.0, base + 3.0],
            "low": [base - 2.0, base - 1.0],
            "close": [base + 0.5, base + 1.5],
            "adj_close": [base + 0.4, base + 1.4],
            "volume": [1000, 1200],
        }
    )


def test_parse_soxx_holdings_workbook_extracts_equity_universe():
    from src.data.soxx_constituents import parse_soxx_holdings_workbook

    snapshot = parse_soxx_holdings_workbook(WORKBOOK_XML, source_url="https://example.test/workbook.xls")

    assert snapshot.as_of == "2026-07-02"
    assert snapshot.number_of_securities == 30
    assert len(snapshot.rows) == 3
    assert snapshot.equity_rows["ticker"].tolist() == ["MU", "AMD"]
    assert snapshot.equity_rows["weight_pct"].round(5).tolist() == [8.16407, 8.15302]


def test_fetch_and_store_soxx_constituent_history_writes_manifest_and_coverage(tmp_path):
    from src.data.soxx_constituents import (
        SoxxConstituentPaths,
        fetch_and_store_soxx_constituent_history,
        parse_soxx_holdings_workbook,
    )

    snapshot = parse_soxx_holdings_workbook(WORKBOOK_XML, source_url="https://example.test/workbook.xls")
    paths = SoxxConstituentPaths(root=tmp_path / "soxx_constituents")

    manifest = fetch_and_store_soxx_constituent_history(
        start="2026-07-01",
        end="2026-07-02",
        paths=paths,
        sleep_seconds=0.0,
        holdings_snapshot=snapshot,
        price_fetcher=_price_frame,
    )

    coverage = pd.read_csv(paths.coverage_csv)
    saved_manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    mu = pd.read_csv(paths.daily_dir / "MU.csv")

    assert set(coverage["symbol"]) == {"MU", "AMD"}
    assert set(coverage["status"]) == {"ok"}
    assert paths.equity_universe_csv.exists()
    assert mu["adj_close"].tolist() == [100.4, 101.4]
    assert manifest["equity_constituents"] == 2
    assert saved_manifest["successful_symbols"] == ["MU", "AMD"]
    assert saved_manifest["limitations"][0].startswith("Current SOXX constituents backfilled")


def test_normalize_yahoo_ohlcv_payload_keeps_adjusted_close():
    from src.data.soxx_constituents import normalize_yahoo_ohlcv_payload

    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1783008000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [10.0],
                                "high": [11.0],
                                "low": [9.5],
                                "close": [10.5],
                                "volume": [123],
                            }
                        ],
                        "adjclose": [{"adjclose": [10.25]}],
                    },
                }
            ],
            "error": None,
        }
    }

    frame = normalize_yahoo_ohlcv_payload("MU", payload)

    assert list(frame.columns) == ["date", "open", "high", "low", "close", "adj_close", "volume"]
    assert frame.loc[0, "adj_close"] == 10.25
