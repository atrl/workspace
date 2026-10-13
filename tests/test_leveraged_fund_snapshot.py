import pandas as pd
from src.data.leveraged_fund_snapshot import parse_holdings


def test_signed_swap_exposure_is_not_nav_and_midnight_trade_date_is_not_receipt():
    raw=b'Fund\nSOXS\nShares Outstanding:10000000\n\n\n"TradeDate","AccountTicker","StockTicker","SecurityDescription","Shares","Price","MarketValue","Cusip","HoldingsPercent"\n"10/13/2026 12:00:00 AM","SOXS","","ICESEMI SWAP ASSET LEG","-10","100","-1000","SWAP","-300"\n'
    table,meta=parse_holdings(raw,symbol='SOXS',observed_at_utc='2026-10-13T20:48:18Z')
    assert table.MarketValue.iloc[0]==-1000 and meta['reported_holdings_percent_sum']==-300
    assert meta['shares_outstanding_exact']==10000000 and meta['market_value_sum_is_net_assets'] is False
    assert meta['intraday_net_creation_flow'] is None
    assert pd.Timestamp(meta['cache_available_at_ny']).tz_convert('UTC')==pd.Timestamp('2026-10-13T20:49:00Z')
