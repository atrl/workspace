import hashlib
import json

import pandas as pd
import pytest

from src.data import nas_minute_bundle as module


def test_joined_source_preserves_clocks_and_rejects_context_tampering(tmp_path, monkeypatch):
    objects = []
    def register(name):
        objects.append({'local_path': name, 'sha256': hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()})
    stamp = pd.Timestamp('2026-10-13 16:00', tz='America/New_York')
    common = {'timestamp': stamp, 'date': '2026-10-13', 'adjustment_type': 'NONE', 'is_final': True,
              'received_at': pd.Timestamp('2026-10-14', tz='UTC')}
    peers = pd.DataFrame([{**common, 'symbol': s, 'interval': '5m'} for s in ['QQQ','SMH','NVDA','AMD']])
    peers.to_parquet(tmp_path/'peer.parquet'); register('peer.parquet')
    pd.DataFrame({'event': []}).to_parquet(tmp_path/'events.parquet'); register('events.parquet')
    context = {'context_as_of':'2026-10-13','context_raw_cache':'peer.parquet',
               'context_rehab_paths':{s:'events.parquet' for s in peers.symbol}, 'context_manifest_path':'context.json'}
    (tmp_path/'context.json').write_text(json.dumps({'context': context, 'context_raw_sha256': objects[0]['sha256']})); register('context.json')
    (tmp_path/'core.json').write_text(json.dumps({'schema_version':'canonical_none_refresh_v1'})); register('core.json')
    core = {s:pd.DataFrame([{**common, 'timestamp':stamp.tz_convert('UTC'),'symbol':s,'interval':'1m'}]) for s in ['SOXL','SOXX']}
    monkeypatch.setattr(module,'_source_frames',lambda *a,**k:(core,{}))
    binding = {'objects':objects,'source_manifest_path':'core.json','source_id':objects[-1]['sha256'],'as_of':'2026-10-13','metadata':context}
    raw,_ = module.load_nas_minute_bundle(binding,workspace_root=tmp_path)
    assert len(raw)==6 and str(raw.timestamp.dt.tz)=='America/New_York'
    assert raw.timestamp.eq(stamp).all() and raw.received_at.eq(common['received_at']).all()
    peers.loc[0,'date']='2026-10-14';peers.to_parquet(tmp_path/'peer.parquet')
    with pytest.raises(ValueError,match='hash differs'):
        module.load_nas_minute_bundle(binding,workspace_root=tmp_path)
