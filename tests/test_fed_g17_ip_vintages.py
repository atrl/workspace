import pytest

from src.data.fed_g17_ip_vintages import parse_ip_supplement


def _document(*, gap="", code="3344", latest="125.9"):
    return (
        "January 14, 2011\n9:15 a.m. Eastern\n"
        "Table 1 (continued)\n"
        "INDUSTRIAL PRODUCTION: INDUSTRY SUBTOTALS AND INDIVIDUAL SERIES\n"
        "Index, 2007 = 100\n"
        "| | | Seasonally adjusted | Not seasonally adjusted |\n"
        "| NAICS | Proportion | July[r] Aug.[r] Sept.[r] Oct.[r] Nov.[r] Dec.[p] | six NSA months |\n"
        "Semiconductor and other electronic | | | |\n"
        + gap +
        f"component | {code} | 2.02 | 124.4 125.3 124.9 123.6 124.1 {latest} | 116.3 123.5 146.2 123.8 124.7 126.0\n"
    ).encode()


@pytest.mark.parametrize("gap", ["", "\n"])
def test_original_table1_same_vintage_growth_and_blank_line(gap):
    result = parse_ip_supplement(_document(gap=gap), expected_date="2011-01-14",
                                 observation_month="2010-12-01")
    assert result["index_base_year"] == 2007
    assert result["published_months"][-2:] == ["2010-11-01", "2010-12-01"]
    assert result["ip_mom_pct_same_vintage"] == pytest.approx(100 * (125.9 / 124.1 - 1))
    assert result["layout_blank_lines"] == (1 if gap else 0)


def test_wrong_scope_and_month_cannot_enter_original_ip_series():
    with pytest.raises(ValueError, match="missing or ambiguous"):
        parse_ip_supplement(_document(code="334412-9"), expected_date="2011-01-14",
                            observation_month="2010-12-01")
    with pytest.raises(ValueError, match="six seasonally adjusted IP months"):
        parse_ip_supplement(_document(), expected_date="2011-01-14",
                            observation_month="2010-11-01")


def test_seven_month_release_requires_explicit_source_bound_exception():
    source = (
        "December 23, 2025\n9:15 a.m. Eastern\nTable 1D\n"
        "INDUSTRIAL PRODUCTION: INDUSTRY SUBTOTALS AND INDIVIDUAL SERIES\n"
        "Index, 2017 = 100\n"
        "| | | Seasonally adjusted | Not seasonally adjusted |\n"
        "| NAICS | Proportion | May[r] June[r] July[r] Aug.[r] Sept.[r] Oct.[r] Nov.[p] | NSA |\n"
        "Semiconductor and other electronic | | | |\n"
        "component | 3344 | 1.5 | 101 102 103 104 105 106 107 | 100 101 102 103 104 105 106\n"
    ).encode()
    with pytest.raises(ValueError, match="six seasonally adjusted IP months"):
        parse_ip_supplement(source, expected_date="2025-12-23",
                            observation_month="2025-11-01")
    result = parse_ip_supplement(source, expected_date="2025-12-23",
                                 observation_month="2025-11-01", monthly_columns=7)
    assert result["published_months"] == [
        "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01",
        "2025-09-01", "2025-10-01", "2025-11-01",
    ]
    assert result["ip_mom_pct_same_vintage"] == pytest.approx(100 * (107 / 106 - 1))
