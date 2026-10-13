from types import SimpleNamespace

import pytest

from src.data import sia_sales_pdf


def test_gsr_reconciles_amounts_without_inventing_missing_timestamps(monkeypatch):
    rows = "\n".join(name + " 10.00 11.00 10.0%" for name in
        ["Americas", "Europe", "Japan", "China", "Asia Pacific/All Other"]) + "\nTotal 50.00 55.00 10.0%\n"
    text = "July 2022 Billions\n" + "\n".join(heading + "\n" + rows for heading in
        ["Month-to-Month Sales", "Year-to-Year Sales", "Three-Month-Moving Average Sales"])
    def read(value):
        monkeypatch.setattr(sia_sales_pdf, "PdfReader", lambda _: SimpleNamespace(
            pages=[SimpleNamespace(extract_text=lambda: value)],
            metadata=SimpleNamespace(creation_date=None, modification_date=None)))
        return sia_sales_pdf.parse_sales_pdf(b"test")
    observed = read(text)
    assert observed["sales_billion_usd"] == 55
    assert observed["created_at"] is None and observed["modified_at"] is None
    alternate = text.replace("Billions", "000's omitted").replace("10.00", "10,000,000").replace("11.00", "11,000,000").replace("50.00", "50,000,000").replace("55.00", "55,000,000")
    assert read(alternate.replace("Average Sales", "Average\nSales"))["sales_billion_usd"] == 55
    with pytest.raises(ValueError, match="percentage inconsistent"):
        read(text.replace("10.0%", "99.0%", 1))
    with pytest.raises(ValueError, match="total reconciliation"):
        read(text.replace("Americas 10.00 11.00 10.0%", "Americas 10.00 12.00 20.0%", 1))
