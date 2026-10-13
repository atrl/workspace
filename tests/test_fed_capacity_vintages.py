import copy

import pandas as pd
import pytest

from src.data.fed_capacity_vintages import capacity_information_state, parse_capacity_supplement


def publication(date="January 17, 2018", value="67.6"):
    return (f"FEDERAL RESERVE STATISTICAL RELEASE\nFor release at 9:15 a.m. (EST)\n{date}\n"
        "Table 2\nCapacity Utilization: Manufacturing, Mining, and Utilities\nPercent of capacity, seasonally adjusted\n"
        " | | 2016 | 1972-2016 | 2016 | 2017\n"
        "Item | NAICS | Proportion | Ave. Low High | Dec. | July[r] Aug.[r] Sept.[r] Oct.[r] Nov.[r] Dec.[p]\n"
        "------------------------------------------------------------\n"
        f"Electronic components | 3344 | 1.56 | 78.6 62.8 91.8 | 69.8 | 66.3 65.8 65.6 66.7 68.2 {value}\n"
        "Table 3\nIndustrial Capacity\n").encode()


def test_original_publication_months_and_units_are_not_release_month_or_production_index():
    parsed = parse_capacity_supplement(publication(), expected_date="2018-01-17")
    assert parsed["observation_month"] == "2017-12-01"
    assert parsed["capacity_level"] == 67.6
    assert parsed["capacity_change_1m"] == pytest.approx(-.6)
    assert parsed["capacity_gap_to_average"] == -11.
    page_control = publication().replace(b"Table 2", b"1Table 2").replace(b"Table 3", b"1Table 3")
    assert parse_capacity_supplement(page_control)["capacity_level"] == 67.6
    legacy = parse_capacity_supplement(publication().replace(b"| 3344 |", b"| 334412-9 |"))
    assert legacy["source_naics"] == "334412-9" and not legacy["source_scope_qualified"]
    with pytest.raises(ValueError, match="publication disagree"):
        parse_capacity_supplement(publication(), expected_date="2018-01-18")
    with pytest.raises(ValueError, match="not seasonally adjusted utilization"):
        parse_capacity_supplement(publication().replace(b"Percent of capacity, seasonally adjusted", b"Production index, 2012=100"))


def test_vintage_revision_enters_only_after_release_and_unknown_latest_invalidates_state():
    first = parse_capacity_supplement(publication())
    revised = parse_capacity_supplement(publication("January 19, 2018", "73.4"))
    manifest = {"entries": [{"parsed": first}, {"parsed": revised}], "failures": [{"url": "https://www.federalreserve.gov/releases/g17/20180123/g17_sup.txt"}]}
    calendar = pd.to_datetime(["2018-01-16", "2018-01-17", "2018-01-18", "2018-01-19", "2018-01-22", "2018-01-23", "2018-01-24"])
    state, _ = capacity_information_state(manifest, calendar)
    assert state.known.tolist() == [0., 0., 1., 1., 1., 1., 0.]
    assert state.capacity_level.iloc[2] == 67.6
    assert state.capacity_level.iloc[4] == 73.4
    assert pd.isna(state.capacity_level.iloc[-1])
    changed = copy.deepcopy(manifest); changed["entries"][1]["parsed"]["capacity_level"] = 99.
    other, _ = capacity_information_state(changed, calendar)
    pd.testing.assert_frame_equal(state.iloc[:4], other.iloc[:4])
    complete = copy.deepcopy(manifest); complete["failures"] = []
    complete_state, _ = capacity_information_state(complete, calendar)
    assert complete_state.known.iloc[2:].eq(1).all()
