"""Notebook-facing source qualification and an immutable native Qlib snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from src.data.price_basis import (FIELDS, INSTRUMENTS, PriceBasisError, build_qualified_price_store,
    load_qlib_price_calendar, load_qlib_feature_frame, price_frame_sha256, QlibProviderConfig, validate_price_basis)
from src.data.published_price_basis import build_published_basis
from src.workflow.stable_excess_integration import CANONICAL_ROOT, json_write, sha

EXPERIMENT = 'canonical_price_basis_20261002'
PROVIDER = 'data/qlib/published_futu_affine_20260925_v1'
REFERENCE = 'data/qlib/stable_excess_integration_20261002'


def verify_futu_snapshot(directory, *, adjustment, start, as_of):
    """Verify the actual decoded SDK pages, not just a combined export."""
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    expected = {'requested_start':start,'requested_end':as_of,'adjustment':adjustment,'session':'RTH','extended_time':False,'read_only':True}
    if any(manifest.get(k)!=v for k,v in expected.items()):
        raise PriceBasisError('intake request differs from the registered source contract')
    originals, artifacts = {}, [(directory/'manifest.json','manifest.json')]
    receipts = manifest['raw_responses']
    if len({r['name'] for r in receipts})!=len(receipts):
        raise PriceBasisError('duplicate SDK receipt name')
    for receipt in receipts:
        path = directory/Path(receipt['response_path']).name
        if sha(path)!=receipt['response_sha256'] or receipt['ret']!=0:
            raise PriceBasisError('raw SDK response identity or return status failed')
        before, after = pd.Timestamp(receipt['requested_at_utc']),pd.Timestamp(receipt['received_at_utc'])
        if before.tzinfo is None or after.tzinfo is None or before>after:
            raise PriceBasisError('raw SDK receipt timestamps invalid')
        with path.open('rb') as file:originals[receipt['name']]=pickle.load(file)
        if originals[receipt['name']][0]!=0:
            raise PriceBasisError('original SDK tuple reports failure')
        artifacts.append((path,path.name))
    frames, rehabilitation, evidence = [], {}, {}
    for symbol in INSTRUMENTS:
        pages = sorted([r for r in receipts if r['name'].startswith(symbol+'_history_page')],key=lambda r:r['params']['page_number'])
        if [r['params']['page_number'] for r in pages]!=list(range(1,len(pages)+1)) or not pages:
            raise PriceBasisError('missing or duplicated SDK pagination receipt')
        chunks=[]
        previous_received=None
        for index,receipt in enumerate(pages):
            params=receipt['params']
            expected_page={'code':'US.'+symbol,'start':start,'end':as_of,'ktype':'K_DAY',
                'autype':{'NONE':'None','QFQ':'qfq'}[adjustment],'max_count':1000,'extended_time':False,'session':'RTH'}
            if any(params.get(k)!=v for k,v in expected_page.items()):
                raise PriceBasisError('SDK page request differs from declared source convention')
            if previous_received is not None and pd.Timestamp(receipt['requested_at_utc'])<previous_received:
                raise PriceBasisError('SDK page receipt order is not sequential')
            previous_received=pd.Timestamp(receipt['received_at_utc'])
            response=originals[receipt['name']]
            if len(response)!=3 or not isinstance(response[1],pd.DataFrame) or response[1].empty:
                raise PriceBasisError('SDK historical response lacks observed candles')
            data=response[1]
            if (not data.code.eq('US.'+symbol).all() or len(data)!=receipt['rows']
                    or bool(response[2])!=(index<len(pages)-1)
                    or receipt['params']['has_prior_page_key']!=(index>0)):
                raise PriceBasisError('SDK page symbol, count or continuation differs')
            dates=pd.to_datetime(data.time_key)
            if dates.min()<pd.Timestamp(start) or dates.max()>pd.Timestamp(as_of):
                raise PriceBasisError('SDK market rows outside requested date bounds')
            chunks.append(data)
        frame=pd.concat(chunks,ignore_index=True)
        if frame.time_key.duplicated().any() or not pd.to_datetime(frame.time_key).is_monotonic_increasing:
            raise PriceBasisError('SDK pages overlap or change chronological order')
        frames.append(frame)
        pre=next(r for r in receipts if r['name'].startswith(symbol+'_rehab_before'))
        post=next(r for r in receipts if r['name'].startswith(symbol+'_rehab_after'))
        if pre['params']!={'code':'US.'+symbol} or post['params']!={'code':'US.'+symbol}:
            raise PriceBasisError('rehabilitation request instrument differs')
        pre_data,post_data=originals[pre['name']][1],originals[post['name']][1]
        pd.testing.assert_frame_equal(pre_data,post_data)
        if pd.Timestamp(pre['received_at_utc'])>pd.Timestamp(pages[0]['requested_at_utc']) or pd.Timestamp(pages[-1]['received_at_utc'])>pd.Timestamp(post['requested_at_utc']):
            raise PriceBasisError('rehabilitation evidence does not bracket the history pages')
        rehabilitation[symbol]=pre_data
        evidence[symbol]={'before_sha256':pre['response_sha256'],'after_sha256':post['response_sha256'],
            'before_observed_at_utc':pre['received_at_utc'],'after_observed_at_utc':post['received_at_utc']}
    identity={'source':'Futu','adjustment':adjustment,'session':'RTH','extended_time':False,'timezone':'America/New_York',
        'volume_unit':'shares','volume_basis':'raw_shares' if adjustment=='NONE' else 'split_adjusted_shares',
        'retrieval_id':directory.name,'retrieved_at_utc':max(r['received_at_utc'] for r in receipts),
        'requested_start':start,'requested_end':as_of,'sdk_version':manifest['sdk_version'],
        'raw_artifacts':[{'name':name,'sha256':sha(path)} for path,name in artifacts], 'rehabilitation':evidence}
    return {'frame':pd.concat(frames,ignore_index=True),'rehabilitation':rehabilitation,'identity':identity,
        'manifest':manifest,'artifacts':artifacts}


def yahoo_missing_bars(directory, primary, calendar, *, as_of, rehabilitation, expected_secondary_dates):
    directory=Path(directory);receipt=json.loads((directory/'manifest.json').read_text());raw=directory/'original_response.json'
    if receipt['status_code']!=200 or sha(raw)!=receipt['sha256']:
        raise PriceBasisError('Yahoo raw response hash or HTTP status failed')
    payload=json.loads(raw.read_text())
    if payload['chart'].get('error') is not None or len(payload['chart']['result'])!=1:
        raise PriceBasisError('Yahoo chart does not contain one successful source result')
    result=payload['chart']['result'][0]
    if result['meta']['symbol']!='SOXL' or result['meta']['exchangeTimezoneName']!='America/New_York' or result['meta']['currency']!='USD':
        raise PriceBasisError('Yahoo source instrument, timezone or currency differs')
    address=urlparse(receipt['url'])
    if (address.scheme!='https' or address.hostname not in {'query1.finance.yahoo.com','query2.finance.yahoo.com'}
            or address.path!='/v8/finance/chart/SOXL' or address.username is not None or address.password is not None
            or address.port not in (None,443)):
        raise PriceBasisError('Yahoo source URL is not the declared quote endpoint')
    requested_at,received_at=pd.Timestamp(receipt['requested_at_utc']),pd.Timestamp(receipt['received_at_utc'])
    if requested_at.tzinfo is None or received_at.tzinfo is None or requested_at>received_at:
        raise PriceBasisError('Yahoo receipt timestamps invalid')
    query=parse_qs(address.query)
    if query.get('interval')!=['1d']:
        raise PriceBasisError('Yahoo source request is not daily')
    dates=pd.to_datetime(result['timestamp'],unit='s',utc=True).tz_convert('America/New_York').normalize().tz_localize(None)
    try:
        lower,upper=int(query['period1'][0]),int(query['period2'][0])
    except (KeyError,ValueError,IndexError) as error:
        raise PriceBasisError('Yahoo request epoch bounds missing') from error
    stamps=np.asarray(result['timestamp'],dtype=np.int64)
    if lower>=upper or len(stamps)==0 or (stamps<lower).any() or (stamps>=upper).any():
        raise PriceBasisError('Yahoo quote timestamps exceed requested epoch bounds')
    prices=result['indicators']['quote'][0]
    frame=pd.DataFrame({name:prices[name] for name in FIELDS})
    if len(frame)!=len(dates):raise PriceBasisError('Yahoo timestamp and quote arrays differ')
    frame['date']=dates;frame['instrument']='SOXL';frame['original_payload_ordinal']=np.arange(len(frame))
    frame['original_payload_sha256']=receipt['sha256']
    if dates.has_duplicates or not dates.is_monotonic_increasing or dates.max()>pd.Timestamp(as_of):
        raise PriceBasisError('Yahoo source dates duplicate, reorder or exceed as_of')
    desired=set(pd.to_datetime(calendar['dates']))
    missing={symbol:desired-set(pd.to_datetime(primary.loc[primary.code.eq('US.'+symbol),'time_key'])) for symbol in INSTRUMENTS}
    if missing['SOXX'] or not missing['SOXL'].issubset(set(dates)):
        raise PriceBasisError('secondary source does not cover the actual primary gaps')
    secondary=frame.loc[frame.date.isin(missing['SOXL'])].copy().reset_index(drop=True)
    if set(secondary.date)!=set(pd.to_datetime(expected_secondary_dates)):
        raise PriceBasisError('secondary coverage differs from the registered observations')
    identity={'source':'Yahoo','instrument':'SOXL','adjustment':'SPLIT_ONLY','session':'RTH','extended_time':False,
        'timezone':'America/New_York','volume_unit':'shares','volume_basis':'split_adjusted_shares',
        'retrieval_id':directory.name+'_'+receipt['received_at_utc'],'retrieved_at_utc':receipt['received_at_utc'],
        'requested_start':dates.min().date().isoformat(),'requested_end':dates.max().date().isoformat(),
        'raw_artifacts':[{'name':'original_response.json','sha256':sha(raw)},{'name':'manifest.json','sha256':sha(directory/'manifest.json')}],
        'rehabilitation':rehabilitation,'receipt':receipt,'quote_uncertainty':'Neighbor Jan2012 quotations differ from Futu; no coefficient fitted to resolve quotations.',
        'metadata_scope':'Only historical timestamp/quote arrays are observations; current quote metadata is archived but never used for prices/features.'}
    return {'frame':secondary,'full_observed_frame':frame,'identity':identity,'receipt':receipt,
        'selection_audit':{'observed_yahoo_rows':len(frame),'selected_primary_gap_rows':len(secondary),
            'policy':'only registered primary-source gaps; full competing source frame is retained, not used to replace available Futu rows'},
        'artifacts':[(raw,'original_response.json'),(directory/'manifest.json','manifest.json')]}


def _existing_snapshot(frame, provider, qualification):
    manifest=json.loads((provider/'price_basis_manifest.json').read_text())
    marker=json.loads((provider/'price_basis_status.json').read_text())
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
    if marker.get('status')!='qualified' or marker.get('manifest_sha256')!=digest:
        raise PriceBasisError('existing snapshot lacks a matching qualified marker')
    if manifest['qualification']['frame_sha256']!=qualification['frame_sha256'] or not manifest['qualification']['qualification_pass']:
        raise PriceBasisError('existing immutable price snapshot has a different qualified identity')
    loaded=load_qlib_feature_frame(QlibProviderConfig(provider_uri=provider),instruments=list(INSTRUMENTS),
        fields=['$'+f for f in FIELDS],start=qualification['required_full_start'],end=qualification['as_of']).sort_index()
    from qlib.data import D
    dates=pd.DatetimeIndex(D.calendar(freq='day')).strftime('%Y-%m-%d').tolist()
    listed=sorted(D.list_instruments(D.instruments('all'),as_list=True))
    if dates!=manifest['reference_calendar']['dates'] or listed!=sorted(INSTRUMENTS):
        raise PriceBasisError('existing snapshot full calendar or instrument universe changed')
    expected=frame.set_index(['instrument','datetime'])[list(FIELDS)].sort_index().astype(np.float32)
    expected.columns=['$'+f for f in FIELDS]
    if not loaded.index.equals(expected.index) or not np.array_equal(loaded.to_numpy(),expected.to_numpy()):
        raise PriceBasisError('existing Qlib snapshot differs from the qualified source')
    return manifest


def run_data_qualification(*, root, none_directory=None, qfq_directory=None, yahoo_directory=None):
    import qlib
    from qlib.workflow import R
    from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts
    root=Path(root).resolve();card=root/'research'/EXPERIMENT
    plan=json.loads((card/'data_plan.json').read_text());plan_hash=sha(card/'data_plan.json')
    qlib.init(provider_uri=str(root/REFERENCE),region='us',kernels=1)
    source_paths=('src/data/price_basis.py','src/data/published_price_basis.py','src/workflow/canonical_price_basis.py','src/data/qlib_native.py')
    source_hashes={p:sha(root/p) for p in source_paths}
    artifact_copies=[]
    if (card/'summary.json').exists():
        previous=json.loads((card/'summary.json').read_text())
        if previous['plan_sha256']!=plan_hash:raise PriceBasisError('retained data plan changed')
        with R.uri_context(previous['recorder_uri']):
            old=R.get_recorder(experiment_name=EXPERIMENT,recorder_id=previous['recorder_id'])
            input_path=Path(old.get_local_dir())/'artifacts/inputs/verified_raw_inputs.pkl'
            if sha(input_path)!=previous.get('verified_raw_inputs_sha256'):
                raise PriceBasisError('retained verified input bundle identity changed')
            inputs=old.load_object('inputs/verified_raw_inputs.pkl')
            for entry in inputs['archive_index']:
                path=Path(old.get_local_dir())/'artifacts'/entry['artifact_path']
                if sha(path)!=entry['sha256']:raise PriceBasisError('retained raw source archive hash changed')
                artifact_copies.append((path,entry['artifact_path']))
        source_mode='immutable_recorder_reexecution'
    else:
        if any(p is None for p in (none_directory,qfq_directory,yahoo_directory)):
            raise PriceBasisError('first execution requires the actual bounded source receipts')
        calendar=load_qlib_price_calendar(root/REFERENCE,plan['original_calendar_start'],plan['data_as_of'])
        none=verify_futu_snapshot(none_directory,adjustment='NONE',start=plan['original_calendar_start'],as_of=plan['data_as_of'])
        qfq=verify_futu_snapshot(qfq_directory,adjustment='QFQ',start=plan['original_calendar_start'],as_of=plan['data_as_of'])
        for symbol in INSTRUMENTS:pd.testing.assert_frame_equal(none['rehabilitation'][symbol],qfq['rehabilitation'][symbol])
        yahoo=yahoo_missing_bars(yahoo_directory,none['frame'],calendar,as_of=plan['data_as_of'],rehabilitation=none['identity']['rehabilitation'],expected_secondary_dates=plan['registered_secondary_dates'])
        for group,data in [('futu_none',none),('futu_qfq',qfq),('yahoo',yahoo)]:
            artifact_copies.extend((path,f'sources/{group}/{name}') for path,name in data.pop('artifacts'))
        inputs={'calendar':calendar,'none':none,'qfq':qfq,'yahoo':yahoo,
            'archive_index':[{'artifact_path':name,'sha256':sha(path)} for path,name in artifact_copies]}
        source_mode='verified_original_sdk_and_http_responses'
    frame,transform=build_published_basis(inputs['none']['frame'],inputs['yahoo']['frame'],inputs['none']['rehabilitation'],
        inputs['none']['identity'],inputs['yahoo']['identity'],plan['data_as_of'])
    qualification=validate_price_basis(frame,inputs['calendar'],plan['data_as_of'],plan['required_model_signal_start'])
    if not qualification['qualification_pass']:
        json_write(card/'qualification_failure.json',qualification)
        raise PriceBasisError('source qualification failed; no provider built')
    compare=[];details=[]
    qfq=inputs['qfq']['frame'].copy();qfq['instrument']=qfq.code.str.removeprefix('US.');qfq['datetime']=pd.to_datetime(qfq.time_key)
    base=frame.set_index(['instrument','datetime'])
    for symbol in INSTRUMENTS:
        server=qfq.loc[qfq.instrument.eq(symbol)].set_index('datetime');current=base.xs(symbol).loc[server.index]
        for field in FIELDS:
            difference=current[field].to_numpy()/server[field].to_numpy()-1
            compare.append({'instrument':symbol,'field':field,'paired_rows':len(server),'maximum_absolute_relative_difference':float(abs(difference).max()),
                'worst_date':str(server.index[int(np.argmax(abs(difference)))].date())})
            details.append(pd.DataFrame({'instrument':symbol,'datetime':server.index,'field':field,'published_value':current[field].to_numpy(),
                'server_qfq_value':server[field].to_numpy(),'relative_difference':difference}))
    provider=root/PROVIDER
    source_manifest={'calendar':inputs['calendar'],'qualification':qualification,'source_identities':frame.attrs['source_identities'],
        'transformation':transform,'verified_raw_archives':inputs['archive_index'],'data_plan_sha256':plan_hash}
    manifest=_existing_snapshot(frame,provider,qualification) if provider.exists() else build_qualified_price_store(frame,provider,plan['data_as_of'],source_manifest)
    if sha(card/'data_plan.json')!=plan_hash or any(sha(root/p)!=v for p,v in source_hashes.items()):
        raise PriceBasisError('data plan or implementation changed during execution')
    uri=CANONICAL_ROOT/'research'/EXPERIMENT/'qlib_recorder';uri.mkdir(parents=True,exist_ok=True);R.set_uri(str(uri))
    with R.start(experiment_name=EXPERIMENT):
        recorder=R.get_recorder();R.log_params(plan_sha256=plan_hash,market_fits_or_pnl=False,basis='published_futu_affine_v1',source_as_of=plan['data_as_of'])
        R.log_metrics(source_qualification_pass=1.,rows=len(frame),secondary_whole_bars=transform['secondary_whole_bar_rows'])
        R.save_objects(artifact_path='inputs',**{'verified_raw_inputs.pkl':inputs,'source_manifest.pkl':source_manifest})
        R.save_objects(artifact_path='data',**{'qualified_frame.pkl':frame,'qualification.pkl':qualification,'transformation.pkl':transform,'provider_manifest.pkl':manifest})
        R.save_objects(artifact_path='analysis',**{'server_comparison.pkl':pd.DataFrame(compare),'server_comparison_detail.pkl':pd.concat(details,ignore_index=True)})
        for path,name in artifact_copies:recorder.log_artifact(str(path),artifact_path=str(Path(name).parent))
        recorder.log_artifact(str(card/'data_plan.json'),artifact_path='inputs');recdir=Path(recorder.get_local_dir())
        pd.testing.assert_frame_equal(frame,recorder.load_object('data/qualified_frame.pkl'))
    _remove_qlib_code_snapshot_artifacts(recdir)
    pd.DataFrame(compare).to_csv(card/'server_qfq_comparison.csv',index=False)
    json_write(card/'qualification.json',qualification);json_write(card/'transformation.json',transform)
    summary={'experiment':EXPERIMENT,'campaign_status':'active_research','objective_achieved':False,'data_as_of':plan['data_as_of'],
        'decision':'qualified_research_price_basis_with_disclosed_secondary_quotes','basis':'published_futu_affine_v1',
        'provider_uri':PROVIDER,'source_mode':source_mode,'recorder_id':recorder.id,'recorder_uri':str(uri),'recorder_local_dir':str(recdir),
        'recorder_relative_dir':str(recdir.relative_to(CANONICAL_ROOT/'research'/EXPERIMENT)),
        'plan_sha256':plan_hash,'source_sha256':source_hashes,
        'verified_raw_inputs_sha256':sha(recdir/'artifacts/inputs/verified_raw_inputs.pkl'),
        'qlib_docs_reference':['https://qlib.readthedocs.io/en/latest/component/data.html','https://qlib.readthedocs.io/en/latest/component/workflow.html'],
        'rows':len(frame),'sessions_per_instrument':len(inputs['calendar']['dates']),
        'secondary_whole_bars':transform['secondary_whole_bar_rows'],'source_qualification_pass':True,'native_readback':manifest['native_readback'],
        'model_fits_or_portfolio_replays_performed':False,'research_promotion':False,'production_change':False,'real_broker_orders':False,
        'source_audit_parent':plan['source_audit_recorder_id'],'source_archive_files':len(inputs['archive_index']),
        'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml',
            'recorder_only_artifacts':['sources/*','inputs/*.pkl','data/*.pkl','analysis/*detail.pkl']},
        'limitations':['Explicitly defined published affine basis, not exact SDKQFQ.','16observedYahoo whole bars fill Futu gaps; Jan2012cross-vendor quote uncertainty remains.',
            'Source qualification verifies lineage, declared transformations, physical bounds and coverage; it does not establish exchange ground truth.',
            'Historical source updates do not create unseen out-of-sample evidence. No stable excess or deployment qualification is claimed.']}
    json_write(card/'summary.json',summary)
    import yaml
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':EXPERIMENT,'provider_uri':PROVIDER,'region':'us','market':'all',
        'as_of':plan['data_as_of'],'source_plan':'data_plan.json','recorder_uri':str(uri),'artifact_policy':summary['artifact_policy']},sort_keys=False))
    lines=['# 统一行情口径验收结论','',f'完成{len(frame)}根日线、每个ETF{summary["sessions_per_instrument"]}个交易日，覆盖2010-03-11至2026-09-25；2011起模型期没有删减。',
        '',f'主要来源为同批Futu NONE/RTH；{transform["secondary_whole_bar_rows"]}根缺失SOXL整根日线由有原始响应的Yahoo报价补齐，来源逐行保留。每根OHLC使用同一套公开A/B变换，成交量按精确拆合股比例处理；原生Qlib日历、标的和float32读回验证通过。',
        '', '这套口径称published_futu_affine_v1，不是服务器QFQ的同义词。全历史价格相对服务器QFQ的差异仍保留；Jan2012跨供应商报价分歧尚不代表交易所原始成交真值已核实。',
        '', '数据验收范围是来源、转换、覆盖与价格关系。本卡没有训练模型或回放组合，也没有完成稳定超额目标。下一步在此固定数据上预登记条件波动比较，保留原有收益、费用、跨年及尾部验收。任何通过候选仍需数据敏感性和新前向证据。',
        '',f'Qlib Recorder：{recorder.id}；provider：{PROVIDER}。']
    (card/'conclusion.md').write_text('\n'.join(lines)+'\n')
    return summary
