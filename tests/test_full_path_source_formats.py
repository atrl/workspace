import hashlib,json
from pathlib import Path
import pandas as pd
import pytest
from src.data.full_path_prices import build_full_path_prices
from src.data.minute_price_basis import normalize_none_history


def test_formats_match_and_missing_real_halfday_close_cannot_be_substituted(tmp_path):
    source=tmp_path/'source';source.mkdir();entries=[];rehab=[];normalized=[]
    for symbol in ['SOXL','SOXX']:
        rows=[]
        for day,close in [('2025-11-26','16:00'),('2025-11-28','13:00')]:
            for i,clock in enumerate(['09:45','09:48','10:33',close]):
                rows.append(dict(code='US.'+symbol,time_key=day+' '+clock+':00',open=100.+i,close=100.+i,high=101.+i,low=99.+i,volume=1000.,turnover=(100.+i)*1000))
        raw=pd.DataFrame(rows);file=source/(symbol+'.parquet');raw.to_parquet(file,index=False)
        stamp='2025-11-29T00:00:00Z';entries.append(dict(symbol=symbol,path=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),fetched_at=stamp))
        event=source/(symbol+'_rehab.parquet');pd.DataFrame(columns=["ex_div_date","forward_adj_factorA","forward_adj_factorB"]).to_parquet(event,index=False);rehab.append(dict(symbol=symbol,path=event.name,sha256=hashlib.sha256(event.read_bytes()).hexdigest()))
        normalized.append(normalize_none_history(raw,symbol=symbol,interval='1m',fetched_at=stamp))
    legacy={'schema_version':'semis_minute_source_v1','complete':True,'adjustment':'NONE','session':'RTH','as_of':'2025-11-28','entries':entries,'rehab':rehab}
    canonical_raw=tmp_path/'canonical.parquet';canonical_frame=pd.concat(normalized,ignore_index=True)
    for column in ['timestamp','bar_start_at','bar_end_at']:canonical_frame[column]=canonical_frame[column].dt.tz_convert('UTC')
    canonical_frame.to_parquet(canonical_raw,index=False)
    canonical={'schema_version':'canonical_none_refresh_v1','source_adjustment':'NONE','session':'RTH','as_of':'2025-11-28','raw_cache':'canonical.parquet','raw_sha256':hashlib.sha256(canonical_raw.read_bytes()).hexdigest(),'rehab_sha256':{r['symbol']:r['sha256'] for r in rehab}}
    def run(receipt,name):
        file=source/(name+'.json');file.write_text(json.dumps(receipt))
        return build_full_path_prices(source_manifest=file,source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),provider_uri=tmp_path/(name+'_provider'),start='2025-11-26',as_of='2025-11-28',workspace_root=tmp_path)[0]
    left,right=run(legacy,'legacy'),run(canonical,'canonical');pd.testing.assert_frame_equal(left,right)
    bad=pd.concat(normalized,ignore_index=True);mask=bad.timestamp.dt.strftime('%Y-%m-%d %H:%M').eq('2025-11-28 13:00');bad.loc[mask,'timestamp']-=pd.Timedelta(hours=1);bad.loc[mask,'bar_start_at']-=pd.Timedelta(hours=1);bad.loc[mask,'bar_end_at']-=pd.Timedelta(hours=1);bad.to_parquet(canonical_raw,index=False)
    with pytest.raises(ValueError,match='bytes changed'):run(canonical,'tampered')
    canonical['raw_sha256']=hashlib.sha256(canonical_raw.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='session-close'):run(canonical,'partial')
