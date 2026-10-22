import copy

import pandas as pd
import pytest

from src.workflow.inventory_sales_nvda_alias_bridge import bridge_first
from src.workflow.inventory_sales_source_qualification import annual_pairs


RULE = {
    "frozen_bridge": {
        "cik": 1045810,
        "taxonomy": "us-gaap",
        "unit": "USD",
        "form": "10-K",
        "original_accession": "0001045810-22-000036",
        "period_start": "2021-02-01",
        "period_end": "2022-01-30",
        "filed": "2022-03-18",
        "revenue_tags": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"],
        "revenue_value_usd": 26914000000,
        "inventory_tag": "InventoryNet",
        "inventory_value_usd": 2605000000,
        "accounting_break_excluded_pair": ["2018-01-28", "2019-01-27"],
        "expected_added_pairs": [["2021-01-31", "2022-01-30"], ["2022-01-30", "2023-01-29"]],
    }
}


def _sources():
    rows = []
    periods = {2018: ("2017-01-30", "2018-01-28"), 2019: ("2018-01-29", "2019-01-27"),
               2020: ("2019-01-28", "2020-01-26"), 2021: ("2020-01-27", "2021-01-31"),
               2023: ("2022-01-31", "2023-01-29"), 2024: ("2023-01-30", "2024-01-28"),
               2025: ("2024-01-29", "2025-01-26")}
    for year, tag in [(2018, "Revenues"), (2019, "RevenueFromContractWithCustomerExcludingAssessedTax"),
                      (2020, "RevenueFromContractWithCustomerExcludingAssessedTax"),
                      (2021, "RevenueFromContractWithCustomerExcludingAssessedTax"),
                      (2023, "Revenues"), (2024, "Revenues"), (2025, "Revenues")]:
        rows.append({"cik": 1045810, "taxonomy": "us-gaap", "field": "revenue", "tag": tag,
                     "unit": "USD", "form": "10-K", "accn": str(year),
                     "start": pd.Timestamp(periods[year][0]), "end": pd.Timestamp(periods[year][1]),
                     "filed": pd.Timestamp(f"{year}-03-18"), "numeric": 1})
    for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        rows.append({"cik": 1045810, "taxonomy": "us-gaap", "field": "inventory", "tag": "InventoryNet",
                     "unit": "USD", "form": "10-K",
                     "accn": RULE["frozen_bridge"]["original_accession"] if year == 2022 else str(year),
                     "start": pd.NaT,
                     "end": pd.Timestamp("2022-01-30") if year == 2022 else pd.Timestamp(periods[year][1]),
                     "filed": pd.Timestamp(f"{year}-03-18"),
                     "numeric": 2605000000 if year == 2022 else 100})
    first = pd.DataFrame(rows)
    extracted = pd.DataFrame([{**rows[0], "field": "revenue", "tag": tag,
                               "accn": RULE["frozen_bridge"]["original_accession"],
                               "start": pd.Timestamp("2021-02-01"), "end": pd.Timestamp("2022-01-30"),
                               "filed": pd.Timestamp("2022-03-18"), "numeric": 26914000000,
                               "mechanical_eligible": True}
                              for tag in RULE["frozen_bridge"]["revenue_tags"]])
    return first, extracted


def test_exact_equal_alias_preserves_neighboring_pairs():
    first, extraction = _sources()
    result = bridge_first(first, extraction, RULE)
    revenue = result.loc[result.field.eq("revenue")]
    assert set(revenue.loc[revenue.end.dt.year.eq(2022), "tag"]) == set(RULE["frozen_bridge"]["revenue_tags"])
    assert revenue.loc[revenue.end.dt.year.eq(2018), "tag"].iloc[0] == "Revenues"
    assert revenue.loc[revenue.end.dt.year.eq(2019), "tag"].iloc[0] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert revenue.loc[revenue.end.dt.year.eq(2025), "tag"].iloc[0] == "Revenues"
    _, pairs = annual_pairs(result, pd.bdate_range("2018-01-01", "2025-12-31"), RULE)
    assert set(pairs.fiscal_end.dt.year) == {2020, 2021, 2022, 2023, 2024, 2025}


def test_mismatched_alias_values_fail_closed():
    first, extraction = _sources()
    bad = extraction.copy()
    bad.loc[bad.tag.eq("Revenues"), "numeric"] = 26914000001
    with pytest.raises(ValueError, match="aliases differ in value"):
        bridge_first(first, bad, copy.deepcopy(RULE))
