"""Explicit Qlib supplement resolution; never overwrite an observed base price."""
import hashlib
import json
from pathlib import Path
import numpy as np
from src.data.qlib_native import QlibProviderConfig, load_qlib_feature_frame


def apply_constituent_supplements(prices, *, root, registry):
    root = Path(root).resolve()
    def verified(relative, digest):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('supplement source path/hash differs: ' + relative)
        return path
    result = prices.copy()
    audit = []
    for entry in registry['sources']:
        for relative, digest in entry['required_hashes'].items():
            verified(relative, digest)
        provider = root / entry['provider_uri']
        native = load_qlib_feature_frame(QlibProviderConfig(provider),
            instruments=[entry['instrument']], fields=[entry['field']],
            start=entry['start'], end=entry['end'])
        series = native.xs(entry['instrument'], level='instrument')[entry['field']].reindex(result.index)
        for alias in entry['historical_aliases']:
            if alias not in result:
                result[alias] = np.nan
            overlap = result[alias].notna() & series.notna()
            if not np.allclose(result.loc[overlap, alias], series.loc[overlap], rtol=2e-7, atol=1e-5):
                raise ValueError('conflicting supplement overlap: ' + alias)
            fill = result[alias].isna() & series.notna()
            result.loc[fill, alias] = series.loc[fill]
            audit.append({'source_id':entry['source_id'],'alias':alias,
                          'filled_price_rows':int(fill.sum()),'unchanged_overlap_rows':int(overlap.sum())})
    return result, audit
