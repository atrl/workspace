from datetime import datetime, timezone
from types import SimpleNamespace

from src.data import guidance_newsletters


def test_legacy_ticker_paragraph_excludes_analyst_targets_and_does_not_infer_values(monkeypatch):
    text = "STOCKS TO WATCH • Nvidia Corp (NVDA). Forecast current-quarter revenue above estimates. ANALYSTS' RECOMMENDATION • Nvidia Corp (NVDA). Broker raises target to $500."
    metadata = SimpleNamespace(creation_date=datetime(2023, 5, 25, 11, tzinfo=timezone.utc), modification_date=None)
    monkeypatch.setattr(guidance_newsletters, "PdfReader", lambda _: SimpleNamespace(
        pages=[SimpleNamespace(extract_text=lambda: text)], metadata=metadata))
    result = guidance_newsletters.guidance_paragraphs(b"sample", "NVDA")
    assert len(result["paragraphs"]) == 1 and "$500" not in result["paragraphs"][0]
    assert result["guidance_word_present"] and result["quarter_word_present"]
    assert not result["numeric_pair_qualified"] and not result["clock_timezone_known"]
