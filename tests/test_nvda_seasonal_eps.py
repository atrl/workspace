import pytest
from src.data.nvda_seasonal_eps import extract_q2_eps,split_bridge


def test_gaap_table_and_split_rounding_do_not_create_false_negative_growth():
    html='<h1>NVIDIA Announces Financial Results for Second Quarter Fiscal 2025</h1><div class="article-date">August 28, 2024</div><p>GAAP earnings per diluted share was $0.67. Non-GAAP earnings per diluted share was $0.68.</p><table><tr><td>GAAP Q2 FY25 Q1 FY25 Q2 FY24</td></tr><tr><td>Diluted earnings per share</td><td>$0.67</td><td>$0.60</td><td>$0.25</td></tr></table><table><tr><td>Non-GAAP Q2 FY25 Q2 FY24</td></tr><tr><td>Diluted earnings per share $0.68 $0.61 $0.27</td></tr></table>'
    current=extract_q2_eps(html,fiscal_year=2025);result=split_bridge({'current_eps':'2.48','fiscal_quarter':'FY2024Q2','accounting_basis':'GAAP','share_basis':'diluted'},current,split_ratio=10)
    assert result['naive_eps_change']=='-1.81' and result['split_normalized_change']=='0.422' and result['comparative_rounding_intervals_overlap']
    with pytest.raises(ValueError,match='mismatch'):extract_q2_eps(html.replace('was $0.67','was $0.66'),fiscal_year=2025)

    with pytest.raises(ValueError,match='same-quarter'):
        split_bridge({'current_eps':'2.48','fiscal_quarter':'FY2023Q2','accounting_basis':'GAAP','share_basis':'diluted'},current,split_ratio=10)


def test_fourth_quarter_excludes_annual_headline_and_annual_table():
    from src.data.nvda_seasonal_eps import extract_quarter_eps
    html='<h1>NVIDIA Announces Financial Results for Fourth Quarter and Fiscal Year 2025</h1><div class="article-date">February 26, 2025</div><p>For fiscal 2025, GAAP earnings per diluted share were $2.94.</p><p>For the quarter, GAAP earnings per diluted share were $0.89.</p><table><tr><td>GAAP Q4 FY25 Q3 FY25 Q4 FY24</td></tr><tr><td>Diluted earnings per share $0.89 $0.78 $0.49</td></tr></table><table><tr><td>GAAP FY25 FY24</td></tr><tr><td>Diluted earnings per share $2.94 $1.19</td></tr></table>'
    result=extract_quarter_eps(html,fiscal_year=2025,quarter=4)
    assert result['current_eps']=='0.89' and result['headline_crosschecked']
    with pytest.raises(ValueError,match='column order'):extract_quarter_eps(html.replace('Q4 FY25 Q3 FY25 Q4 FY24','Q4 FY24 Q3 FY25 Q4 FY25'),fiscal_year=2025,quarter=4)


def test_combined_gaap_nongaap_sections_keep_explicit_quarter_identity():
    from src.data.nvda_seasonal_eps import extract_quarter_eps
    html='<h1>NVIDIA Announces Financial Results for Third Quarter Fiscal 2021</h1><div class="article-date">November 18, 2020</div><p>GAAP earnings per diluted share were $2.12.</p><table><tr><td>GAAP Q 3 FY2 1 Q2 FY21 Q 3 FY20</td></tr><tr><td>Earnings per share $2.12 $0.99 $1.45</td></tr><tr><td>Non-GAAP Q3 FY21 Q2 FY21 Q3 FY20</td></tr><tr><td>Diluted earnings per share $2.91 $2.18 $1.78</td></tr></table>'
    result=extract_quarter_eps(html,fiscal_year=2021,quarter=3)
    assert result['current_eps']=='2.12' and result['prior_year_comparative_eps']=='1.45'


def test_split_knowledge_is_as_of_and_restatement_is_not_silently_repaired():
    from src.data.nvda_seasonal_eps import split_ratio_between_reports
    old={'fiscal_quarter':'FY2021Q3','release_date':'2020-11-18','current_eps':'2.12','accounting_basis':'GAAP','share_basis':'diluted'}
    new={'fiscal_quarter':'FY2022Q3','release_date':'2021-11-17','current_eps':'0.97','prior_year_comparative_eps':'0.53','accounting_basis':'GAAP','share_basis':'diluted'}
    event={'event_id':'split2021','effective_date':'2021-07-19','known_by_date':'2021-08-18','event_type':'stock_split','new_shares_per_old_share':4}
    ratio=split_ratio_between_reports(old,new,[event]);assert ratio==4
    assert split_bridge(old,new,split_ratio=ratio)['rounding_relation']=='overlap'
    changed={**new,'prior_year_comparative_eps':'0.60'}
    assert split_bridge(old,changed,split_ratio=ratio)['rounding_relation']=='disjoint'
    with pytest.raises(ValueError,match='unavailable'):split_ratio_between_reports(old,{**new,'release_date':'2021-08-01'},[event])
    with pytest.raises(ValueError,match='duplicate'):split_ratio_between_reports(old,new,[event,event])
