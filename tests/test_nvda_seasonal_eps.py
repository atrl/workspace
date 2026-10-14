import pytest
from src.data.nvda_seasonal_eps import extract_q2_eps,split_bridge


def test_gaap_table_and_split_rounding_do_not_create_false_negative_growth():
    html='<h1>NVIDIA Announces Financial Results for Second Quarter Fiscal 2025</h1><div class="article-date">August 28, 2024</div><p>GAAP earnings per diluted share was $0.67. Non-GAAP earnings per diluted share was $0.68.</p><table><tr><td>GAAP Q2 FY25 Q1 FY25 Q2 FY24</td></tr><tr><td>Diluted earnings per share</td><td>$0.67</td><td>$0.60</td><td>$0.25</td></tr></table><table><tr><td>Non-GAAP Q2 FY25 Q2 FY24</td></tr><tr><td>Diluted earnings per share $0.68 $0.61 $0.27</td></tr></table>'
    current=extract_q2_eps(html,fiscal_year=2025);result=split_bridge({'current_eps':'2.48','fiscal_quarter':'FY2024Q2','accounting_basis':'GAAP','share_basis':'diluted'},current,split_ratio=10)
    assert result['naive_eps_change']=='-1.81' and result['split_normalized_change']=='0.422' and result['comparative_rounding_intervals_overlap']
    with pytest.raises(ValueError,match='mismatch'):extract_q2_eps(html.replace('was $0.67','was $0.66'),fiscal_year=2025)

    with pytest.raises(ValueError,match='same-quarter'):
        split_bridge({'current_eps':'2.48','fiscal_quarter':'FY2023Q2','accounting_basis':'GAAP','share_basis':'diluted'},current,split_ratio=10)
