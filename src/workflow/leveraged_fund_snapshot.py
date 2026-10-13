"""Immutable current issuer snapshot; retrospective flow-model gate stays closed."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.leveraged_fund_snapshot import parse_holdings
from src.data.qlib_native import build_qlib_store_from_frame,QlibProviderConfig,load_qlib_feature_frame
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():
        old=json.loads((card/'summary.json').read_text())
        if old['plan_sha256']!=digest:raise ValueError('fixed snapshot plan changed')
        return old
    tables={};metadata=[];raws={};snapshot=root/plan['raw_snapshot']
    for item in plan['source_metadata']:
        symbol=item['symbol'];raw=(snapshot/(symbol+'.csv')).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('issuer snapshot changed')
        frame,meta=parse_holdings(raw,symbol=symbol,observed_at_utc=item['observed_at_utc']);tables[symbol]=frame;metadata.append(meta);raws[symbol]=raw
    minute_path=root/plan['minute_source_manifest']
    if hashlib.sha256(minute_path.read_bytes()).hexdigest()!=plan['minute_source_manifest_sha256']:raise ValueError('minute receipt identity differs')
    minute_manifest=json.loads(minute_path.read_text());entry=next(e for e in minute_manifest['entries'] if e['symbol']=='SOXL');minute_file=minute_path.parent/entry['path']
    if hashlib.sha256(minute_file.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('minute raw object changed')
    bars=pd.read_parquet(minute_file,columns=['volume','turnover_rate']);rate=pd.to_numeric(bars.turnover_rate,errors='coerce')
    minute_audit={'source_manifest_sha256':plan['minute_source_manifest_sha256'],'object_sha256':entry['sha256'],'rows':len(bars),'zero_rate_rows':int(rate.eq(0).sum()),'positive_rate_rows':int(rate.gt(0).sum()),'positive_volume_without_positive_rate':int((bars.volume.gt(0)&~rate.gt(0)).sum()),'shares_can_be_inferred_from_this_field':False}
    nav=json.loads((card/'reviewed_nav.json').read_text());nav={r['symbol']:r for r in nav}
    for row in metadata:
        row['nav_as_of']=nav[row['symbol']]['nav_as_of'];row['reported_nav']=nav[row['symbol']]['nav'];row['nav_date_matches_holdings_date']=row['nav_as_of']==row['reported_trade_date']
    observations=pd.DataFrame([{'instrument':r['symbol'],'datetime':pd.Timestamp(r['cache_available_at_ny']).tz_localize(None),'shares_million':r['shares_outstanding_exact']/1e6,'reported_holdings_percent_sum':r['reported_holdings_percent_sum']} for r in metadata])
    provider=root/'data/qlib'/plan['experiment'];store=build_qlib_store_from_frame(observations,provider_uri=provider,freq='1min',as_of=plan['as_of'],field_map={'shares_million':'shares_million','reported_holdings_percent_sum':'reported_holdings_percent_sum'},source='issuer observations at our raw receipt rounded UP to minute; not historical publication time')
    native=load_qlib_feature_frame(QlibProviderConfig(provider,freq='1min'),instruments=['SOXL','SOXS'],fields=['$shares_million','$reported_holdings_percent_sum'],start='2026-10-13',end=plan['as_of']+' 23:59:00')
    for row in metadata:
        value=native.xs(row['symbol'],level='instrument').dropna().iloc[0]
        np.testing.assert_allclose(value['$shares_million'],row['shares_outstanding_exact']/1e6,rtol=1e-7)
        np.testing.assert_allclose(value['$reported_holdings_percent_sum'],row['reported_holdings_percent_sum'],rtol=1e-7)
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='issuer_first_raw_receipt',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_backtests=0)
        rec.save_objects(artifact_path='sources',raw_csv=raws,receipt_metadata=plan['source_metadata'],reviewed_nav=nav)
        rec.save_objects(artifact_path='inputs',holdings=tables,metadata=metadata,minute_turnover_audit=minute_audit,native=native,provider_manifest=store,provider_files={str(p.relative_to(provider)):p.read_bytes() for p in provider.rglob('*') if p.is_file()})
        rec.save_objects(artifact_path='code',parser=(root/'src/data/leveraged_fund_snapshot.py').read_text(),workflow=Path(__file__).read_text())
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'needs_historical_flow_and_publication_data','objective_achieved':False,'new_fits':0,'new_backtests':0,'funds':2,'distinct_reported_trade_dates':len({r['reported_trade_date'] for r in metadata}),'historical_training_eligible':False,'intraday_flow_identified':False,'net_assets_from_holdings_sum_allowed':False,'exact_shares_authority':'original CSV header and Recorder metadata; native shares_million is approximate float32 cache, not a share ledger','snapshot_summary':metadata,'minute_turnover_audit':minute_audit,'provider_uri':str(provider),'region':'us','as_of':plan['as_of'],'research_market_as_of':plan['research_market_as_of'],'qlib_docs':'https://qlib.readthedocs.io/en/latest/component/data.html','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['sources/*','inputs/*','code/*']}}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    pd.DataFrame(metadata).to_csv(card/'snapshot_summary.csv',index=False)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':str(provider),'recorder_uri':str(tracking),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary
