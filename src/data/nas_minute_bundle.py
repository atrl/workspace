"""Read a pinned NAS core/context source without changing live cache contents."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.full_path_prices import _source_frames
from src.data.minute_price_basis import load_source_bundle


def load_nas_minute_bundle(binding, *, workspace_root):
    root = Path(workspace_root).resolve()
    objects = {entry['local_path']: entry for entry in binding['objects']}

    def verified(relative):
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root):
            raise ValueError('blocked: NAS source path escapes workspace')
        if relative not in objects or hashlib.sha256(path.read_bytes()).hexdigest() != objects[relative]['sha256']:
            raise ValueError('blocked: NAS source object hash differs: ' + relative)
        return path

    source = verified(binding['source_manifest_path'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != binding['source_id']:
        raise ValueError('blocked: pinned source identity differs')
    receipt = json.loads(source.read_text())
    if receipt.get('schema_version') == 'semis_minute_source_v1':
        return load_source_bundle(source)
    if receipt.get('schema_version') != 'canonical_none_refresh_v1':
        raise ValueError('blocked: unsupported NAS minute schema')
    core, events = _source_frames(source, receipt, as_of=binding['as_of'], workspace_root=root)
    context = binding.get('metadata', {})
    context_source = verified(context['context_manifest_path'])
    context_receipt = json.loads(context_source.read_text())
    pinned_context = context_receipt['context']
    for key in ('context_as_of', 'context_raw_cache', 'context_rehab_paths'):
        if context[key] != pinned_context[key]:
            raise ValueError('blocked: context pointer differs from its source receipt')
    context_path = verified(context['context_raw_cache'])
    if hashlib.sha256(context_path.read_bytes()).hexdigest() != context_receipt['context_raw_sha256']:
        raise ValueError('blocked: context source digest differs')
    peer = pd.read_parquet(context_path)
    symbols = {'QQQ', 'SMH', 'NVDA', 'AMD'}
    if set(peer.symbol) != symbols or not peer.interval.eq('5m').all():
        raise ValueError('blocked: context instrument/interval differs')
    if peer.date.gt(context['context_as_of']).any():
        raise ValueError('blocked: context extends beyond its as_of')
    for symbol in symbols:
        events[symbol] = pd.read_parquet(verified(context['context_rehab_paths'][symbol]))
    parts = [*core.values(), peer]
    for part in parts:
        if not isinstance(part.timestamp.dtype, pd.DatetimeTZDtype):
            raise ValueError('blocked: timezone-aware source required')
        part['timestamp'] = part.timestamp.dt.tz_convert('America/New_York')
        if (not part.adjustment_type.eq('NONE').all()
                or part.date.gt(binding['as_of']).any()
                or not part.date.eq(part.timestamp.dt.strftime('%Y-%m-%d')).all()
                or not part.is_final.fillna(False).all()):
            raise ValueError('blocked: source basis/date/finality differs')
    raw = pd.concat(parts, ignore_index=True)
    if raw.duplicated(['symbol', 'timestamp']).any():
        raise ValueError('blocked: duplicate NAS minute observations')
    return raw, events
