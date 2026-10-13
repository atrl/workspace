import hashlib

import numpy as np
import pandas as pd
import pytest

from src.data.guidance_review import validate_guidance_reviews


def test_explicit_range_grounding_and_no_inferred_consensus():
    text = "AMD forecast next-quarter revenue of $12 billion to $14 billion, versus estimates of $13.5 billion."
    entry = {"symbol": "AMD", "report_date": "2024-01-30", "newsletter_date": "2024-01-31",
        "target_fiscal_quarter": "FY2024Q1", "sha256": "a" * 64, "url": "https://example.invalid", "paragraphs": [text]}
    review = {"source_index": 0, "paragraph_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "guidance_billion": 13., "consensus_billion": 13.5, "direction": -1,
        "guidance_range_billion": [12., 14.], "scope_caveat": False}
    issuer = pd.DataFrame({"next_quarter_numeric": pd.Series(dtype=bool), "release_date": pd.Series(dtype="datetime64[ns]")})
    manifest = {"entries": [entry], "failures": []}
    result = validate_guidance_reviews(manifest, [review], issuer)
    assert result.status.iloc[0] == "numeric_pair"
    with pytest.raises(ValueError, match="consensus amount absent"):
        validate_guidance_reviews(manifest, [{**review, "consensus_billion": 13.7}], issuer)
    result = validate_guidance_reviews(manifest, [{**review, "consensus_billion": None}], issuer)
    assert result.status.iloc[0] == "direction_only" and np.isnan(result.reported_midpoint_gap.iloc[0])
