import pytest
from src.data.ibkr_interest_archive import parse_archived_rates


def test_rolling_archive_dates_are_not_truncated_to_heading_month():
    html='<h2>Benchmark Rates for December 2023</h2><table><tr><th>Date</th><th>USD</th><th>USD(LIBOR)</th></tr><tr><td>20231130</td><td>5.330%</td><td>9.0%</td></tr><tr><td>20231201</td><td>5.330*%</td><td>9.0%</td></tr></table>'
    f=parse_archived_rates(html,original_url='https://www.interactivebrokers.com/monthlyInterestRates.php',as_of='2023-12-05')
    assert len(f)==2 and f.benchmark_rate.tolist()==[.0533,.0533] and f.prior_day_flag.tolist()==[0,1]
    with pytest.raises(ValueError,match='requested month'):parse_archived_rates(html,original_url='https://www.interactivebrokers.com/monthlyInterestRates.php?date=202312',as_of='2023-12-05')
