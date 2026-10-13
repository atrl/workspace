from src.data.issuer_result_tables import parse_quarterly_results


def test_actuals_keep_bases_precision_and_do_not_backfill_missing_current():
    table = '''<table>
    <tr><td></td><td colspan="3">GAAP</td><td>Non-GAAP</td></tr>
    <tr><td></td><td>Q4 2021</td><td>Q4 2020</td><td>vs Q4 2020</td><td>Q4 2021</td></tr>
    <tr><td>Revenue ($B)</td><td>$20.5</td><td>$20.0</td><td>up3%</td><td>$19.5</td></tr>
    <tr><td>Earnings per share</td><td>$1.13</td><td>$1.42</td><td>down21%</td><td>($0.09)</td></tr>
    </table>'''
    outlook = '<p>Business Outlook</p>' + table.replace('Q4 2021', 'Q1 2022').replace('$20.5', '$18.3')
    result = parse_quarterly_results((outlook + '<p>Q4 2021 Financial Highlights</p>' + table).encode())['basis']
    assert result['gaap']['quarter_label'] == 'Q4 2021'
    assert result['gaap']['revenue_million_usd'] == 20500
    assert result['gaap']['revenue_rounding_half_million'] == 50
    assert result['non_gaap']['revenue_million_usd'] == 19500
    assert result['non_gaap']['eps_usd'] == -.09
    separate = '''<p>Other non-GAAP results are discussed below.</p><table>
    <tr><td colspan="2">GAAP</td></tr><tr><td></td><td>Q1 FY26</td></tr>
    <tr><td>Revenue ($M)</td><td>$100</td></tr>
    <tr><td>Diluted earnings per share</td><td>$0.81</td></tr>
    <tr><td>Diluted earnings per share excluding charge</td><td>$0.96</td></tr></table>'''
    result = parse_quarterly_results(separate.encode())['basis']
    assert result['gaap']['eps_usd'] == .81
    assert result['gaap']['eps_alternatives'][0]['eps_usd'] == .96
    plain = separate[separate.index('<table>'):].replace('<tr><td colspan="2">GAAP</td></tr>', '')
    two = '<p>GAAP Quarterly Financial Results</p>' + plain + '<p><b>Non-GAAP(*)</b> <b>Quarterly Financial Results</b></p>' + plain
    assert set(parse_quarterly_results(two.encode())['basis']) == {'gaap', 'non_gaap'}
    missing = parse_quarterly_results(table.replace('$20.5', '--').encode())['basis']
    assert missing['gaap']['revenue_million_usd'] is None
    assert missing['gaap']['eps_usd'] == 1.13 and 'non_gaap' in missing
    footnoted = table.replace('Q4 2021', "Q4'21<sup>(1)</sup>").replace('$20.5', '$20.5^').replace('($0.09)', '$(0.09)')
    result = parse_quarterly_results(footnoted.encode())['basis']
    assert result['gaap']['quarter_label'] == 'Q4 21'
    assert result['non_gaap']['eps_usd'] == -.09
