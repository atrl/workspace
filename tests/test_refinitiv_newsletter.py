from datetime import datetime, timezone
from types import SimpleNamespace

from src.data import refinitiv_newsletter


def test_newsletter_does_not_swap_smart_consensus_or_invent_publication(monkeypatch):
    text = """KEY RESULTS
Company Name* Quarter ET EPS Estimates** Year Ago Rev Estimates (mln) Smart Estimates
NVIDIA Q4 AMC $1.23 $1.22 $0.78 $7,419.74
I/B/E/S EPS and revenue estimates, and StarMine Smart Estimates, provided by Refinitiv.
"""
    date = datetime(2022, 2, 15, 21, 24, tzinfo=timezone.utc)
    metadata = SimpleNamespace(creation_date=date, modification_date=date)
    monkeypatch.setattr(refinitiv_newsletter, "PdfReader", lambda _: SimpleNamespace(
        pages=[SimpleNamespace(extract_text=lambda: text)], metadata=metadata))
    value = refinitiv_newsletter.parse_newsletter(b"sample", "NVDA", "2022-02-16")
    assert value["consensus_eps"] == 1.22 and value["smart_eps"] == 1.23
    assert value["consensus_revenue_million_usd"] == 7419.74
    assert value["document_clock_before_report_day"]
    assert not value["historical_publication_clock_verified"]
    metadata.modification_date = datetime(2022, 2, 17, tzinfo=timezone.utc)
    assert not refinitiv_newsletter.parse_newsletter(b"sample", "NVDA", "2022-02-16")["document_clock_before_report_day"]
    morning = "COMPANIES REPORTING RESULTS\nAdvanced Micro Devices Inc: Expected Q2 earnings of $1.03 per share\n(All analysts' estimates are according to I/B/E/S Refinitiv data)"
    monkeypatch.setattr(refinitiv_newsletter, "PdfReader", lambda _: SimpleNamespace(
        pages=[SimpleNamespace(extract_text=lambda: morning)], metadata=metadata))
    value = refinitiv_newsletter.parse_newsletter(b"sample", "AMD", "2022-08-02", "morning_call")
    assert value["consensus_eps"] == 1.03
    assert value["smart_eps"] is None and value["consensus_revenue_million_usd"] is None
    morning = morning.replace("Advanced Micro Devices Inc", "Intel Corp").replace("earnings of $1.03", "loss of 15 cents")
    assert refinitiv_newsletter.parse_newsletter(b"sample", "INTC", "2023-04-27", "morning_call")["consensus_eps"] == -0.15
