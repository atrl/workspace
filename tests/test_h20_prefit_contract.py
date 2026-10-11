"""Frozen input identities must block before any Qlib read or Recorder creation."""
import json
from pathlib import Path

import pytest

from src.analysis import h20_prefit_contract as prefit


def test_changed_provider_blocks_before_qlib_read(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / 'research/h20_matched_forecast_execution_20261010/preregister_plan.json').read_text())
    provider = Path(plan['source_binding']['price_features_provider']).name
    plan['source_binding']['provider_content_sha256'][provider] = '0' * 64
    (tmp_path / 'preregister_plan.json').write_text(json.dumps(plan))

    def unexpected_read(*args, **kwargs):
        pytest.fail('A mismatched frozen provider must fail before Qlib reads')

    monkeypatch.setattr(prefit, 'actual_calendar_prices', unexpected_read)
    with pytest.raises(ValueError, match='frozen provider differs'):
        prefit.run_prefit_contract(root=root, card_dir=tmp_path)
    assert not (tmp_path / 'qlib_recorder').exists()
    assert not (tmp_path / 'summary.json').exists()
