"""Prospective IO v2: explicit source planes, strict receipts, isolated Qlib builds."""
from pathlib import Path
from hashlib import sha256
import json,os,time
import numpy as np
import pandas as pd
from src.data.diagnostic_forward_inputs import canonical,digest,ordered,utc
from src.data.minute_intake import atomic_json
from src.data.diagnostic_forward_probe import clock
NAME='diagnostic_forward_io_v2_20261016'
PLAN_SHA='a8eb75331f4a0071bfe00f1f207711f540ff8c3c41cd4806a2a62d5e07a4438b'
ETFS=('SOXL','SOXX','SOXS','TQQQ','UPRO','SPXL','TECL','SPY','QQQ','XLK')
FIELDS=('open','high','low','close','volume')
PAIRS={'SOXL':'SOXX','TQQQ':'QQQ','UPRO':'SPY','SPXL':'SPY','TECL':'XLK'}

def native_dates(start,end):
    import exchange_calendars as xc
    return pd.DatetimeIndex(xc.get_calendar('XNYS',start=start,end=end).sessions).tz_localize(None).rename('datetime')

def strict_grid(frame,expected,*,label):
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:raise ValueError(label+': duplicate/unordered dates')
    pd.testing.assert_index_equal(frame.index.rename('datetime'),pd.DatetimeIndex(expected).rename('datetime'))

def finite_prices(frame,*,volume=True):
    x=frame[list(FIELDS if volume else FIELDS[:4])].to_numpy(float)
    if not np.isfinite(x).all() or (x[:,:4]<=0).any() or (volume and (x[:,4]<0).any()):raise ValueError('nonfinite/nonpositive OHLC or negative volume')
    if not frame.high.ge(frame[['open','close']].max(axis=1)).all() or not frame.low.le(frame[['open','close']].min(axis=1)).all():raise ValueError('OHLC range violation')

def yahoo_frame(payload,symbol,*,expected):
    chart=payload.get('chart',{})
    if chart.get('error') or len(chart.get('result') or [])!=1:raise ValueError('Yahoo chart result/error')
    data=chart['result'][0];meta=data['meta']
    if meta.get('symbol')!=symbol or meta.get('exchangeTimezoneName')!='America/New_York':raise ValueError('Yahoo symbol/timezone')
    if symbol!='^VIX' and meta.get('currency')!='USD':raise ValueError('Yahoo currency')
    dates=pd.to_datetime(data['timestamp'],unit='s',utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
    quote=data['indicators']['quote'][0]
    fields=list(FIELDS) if symbol!='^VIX' else ['close']
    if any(len(quote.get(k,[]))!=len(dates) for k in fields):raise ValueError('Yahoo array lengths')
    raw=pd.DataFrame({k:quote[k] for k in fields},index=pd.DatetimeIndex(dates,name='datetime')).astype(float)
    strict_grid(raw,expected,label=symbol)
    if symbol=='^VIX':
        if not np.isfinite(raw.close).all() or not raw.close.gt(0).all():raise ValueError('VIX close')
        return raw,raw.copy(),{'meta':meta,'events':data.get('events',{}),'basis':'Yahoo quote.close index'}
    finite_prices(raw)
    adj=data['indicators']['adjclose'][0]['adjclose']
    if len(adj)!=len(raw):raise ValueError('adjclose length')
    ratio=pd.Series(adj,index=raw.index,dtype=float)/raw.close
    if not np.isfinite(ratio).all() or not ratio.gt(0).all():raise ValueError('invalid adjustment ratio')
    for kind,events in data.get('events',{}).items():
        if kind not in ('dividends','splits'):raise ValueError('unsupported corporate action kind')
        for event in events.values():
            date=pd.Timestamp(event['date'],unit='s',tz='UTC').tz_convert('America/New_York').normalize().tz_localize(None)
            if date not in raw.index:raise ValueError('action outside exact source calendar')
            if kind=='dividends':
                if not np.isfinite(float(event['amount'])) or float(event['amount'])<=0:raise ValueError('cash event amount')
            else:
                n=float(event['numerator']);d=float(event['denominator'])
                if not np.isfinite([n,d]).all() or min(n,d)<=0:raise ValueError('split ratio')
                if event.get('splitRatio'):
                    pieces=event['splitRatio'].split(':')
                    if len(pieces)!=2 or not np.isclose(float(pieces[0])/float(pieces[1]),n/d,rtol=0,atol=1e-12):raise ValueError('ambiguous split ratio')
    adjusted=raw.copy();adjusted[list(FIELDS[:4])]=raw[list(FIELDS[:4])].mul(ratio,axis=0)
    return raw,adjusted,{'meta':meta,'events':data.get('events',{}),'ratio':ratio,'basis':'OHLC quote*adjclose/close; volume verbatim vendor-reported shares, split vintage vendor-defined'}

def nasdaq_frame(payload,*,expected):
    rows=((payload.get('data') or {}).get('tradesTable') or {}).get('rows')
    if not isinstance(rows,list) or not rows:raise ValueError('Nasdaq empty rows')
    raw=pd.DataFrame(rows);dates=pd.to_datetime(raw['date'],errors='raise').dt.normalize()
    if dates.duplicated().any():raise ValueError('Nasdaq duplicate date')
    frame=pd.DataFrame(index=pd.DatetimeIndex(dates,name='datetime'))
    for key in FIELDS[:4]:frame[key]=pd.to_numeric(raw[key].astype(str).str.replace('$','',regex=False).str.replace(',','',regex=False),errors='raise').to_numpy()
    frame=frame.sort_index();strict_grid(frame,expected,label='Nasdaq');finite_prices(frame,volume=False);return frame

class HttpLedger:
    """Every request is registered before send; one transport attempt per fixed key."""
    def __init__(self,directory,*,permit):
        if permit.get('phase')!='pilot' or not permit.get('network_authorized') or permit.get('budgets',{}).get('http')!=17:raise PermissionError('exact v2 pilot HTTP permit required')
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.permit=permit;self.records={p.name.removesuffix('.receipt.json'):json.loads(p.read_text()) for p in self.directory.glob('*.receipt.json')}
        for key,record in self.records.items():self._check_request(key,record['url'],record['params'])
        if len(self.records)>17:raise ValueError('persisted HTTP budget exceeded')
    def _check_request(self,key,url,params):
        from urllib.parse import quote
        urls={**{'daily_yahoo_'+s.replace('^',''):'https://query2.finance.yahoo.com/v8/finance/chart/'+quote(s,safe='') for s in (*ETFS,'^VIX')},**{'daily_nasdaq_'+s:'https://api.nasdaq.com/api/quote/'+s+'/historical' for s in ('SOXL','SOXX','SOXS')},**{'live_open_'+s:'https://query2.finance.yahoo.com/v8/finance/chart/'+s for s in ('SOXL','SOXX')},'SEC_status':'https://data.sec.gov/submissions/CIK0001045810.json'}
        if key not in urls or url!=urls[key]:raise ValueError('unregistered HTTP key/endpoint')
        if 'session' not in self.permit:raise ValueError('HTTP session not registered')
        dates=native_dates('2010-03-11',self.permit['session']);previous=dates[-2];session=self.permit['session']
        if key.startswith('daily_yahoo'):expected=chart_params('2010-03-11',str(previous.date()))
        elif key.startswith('daily_nasdaq'):expected={'assetclass':'etf','fromdate':str((previous-pd.Timedelta(days=120)).date()),'todate':str(previous.date()),'limit':10000}
        elif key.startswith('live_open'):expected=chart_params(str(previous.date()),session)
        else:expected={}
        if params!=expected:raise ValueError('HTTP fixed phase/window differs')
    def get(self,key,url,params,*,deadline,headers=None):
        import requests
        self._check_request(key,url,params)
        if key in self.records or (self.directory/(key+'.receipt.json')).exists():raise ValueError('request already attempted; no retry')
        if len(self.records)>=17:raise ValueError('HTTP budget exceeded')
        start=clock()
        day=pd.Timestamp(self.permit['session']).tz_localize('America/New_York');fixed_deadline=day+pd.Timedelta(hours=9,minutes=46 if key.startswith('live_open') else 20)
        earliest=day+pd.Timedelta(hours=9,minutes=45) if key.startswith('live_open') else day+pd.Timedelta(hours=8)
        if utc(deadline)!=fixed_deadline or utc(start['utc'])<earliest:raise ValueError('HTTP phase clock')
        if utc(start['utc'])>=utc(deadline):raise ValueError('request missed deadline')
        receipt={'key':key,'url':url,'params':params,'request_started':start,'completed':False};self.records[key]=receipt;atomic_json(receipt,self.directory/(key+'.receipt.json'))
        try:
            token=_HTTP_LEDGER_TOKEN.set(True)
            try:response=requests.get(url,params=params,headers=headers or {'User-Agent':'Mozilla/5.0 diagnostic-qlib-research'},timeout=30,allow_redirects=False)
            finally:_HTTP_LEDGER_TOKEN.reset(token)
            received=clock();body=response.content;(self.directory/(key+'.body')).write_bytes(body);receipt.update(received=received,status=int(response.status_code),resolved_url=response.url,payload_sha256=sha256(body).hexdigest(),completed=True)
            atomic_json(receipt,self.directory/(key+'.receipt.json'));ordered(start,received)
            if utc(received['utc'])>=utc(deadline):raise ValueError('late HTTP payload')
            if response.status_code!=200 or response.is_redirect:raise ValueError('HTTP non200/redirect')
            return json.loads(body),receipt
        except Exception as error:
            receipt.update(completed=True,error_type=type(error).__name__);atomic_json(receipt,self.directory/(key+'.receipt.json'));raise

def chart_params(start,end):
    return {'period1':int(pd.Timestamp(start,tz='America/New_York').timestamp()),'period2':int((pd.Timestamp(end,tz='America/New_York')+pd.Timedelta(days=1)).timestamp()),'interval':'1d','includeAdjustedClose':'true','events':'div,splits'}

def collect_daily(http,*,session):
    day=pd.Timestamp(session);dates=native_dates('2010-03-11',session);previous=dates[-2];expected=dates[:-1];deadline=day.tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=20)
    yahoo={};nasdaq={};receipts={};audits={};raw={}
    for symbol in (*ETFS,'^VIX'):
        from urllib.parse import quote
        payload,receipt=http.get('daily_yahoo_'+symbol.replace('^',''),f'https://query2.finance.yahoo.com/v8/finance/chart/{quote(symbol,safe="")}',chart_params('2010-03-11',str(previous.date())),deadline=deadline)
        # SOXS inception can be after requested history start; declared Yahoo firstTradeDate fixes its leading domain.
        first=pd.Timestamp(payload['chart']['result'][0]['meta']['firstTradeDate'],unit='s',tz='UTC').tz_convert('America/New_York').tz_localize(None).normalize()
        domain=expected[expected>=first]
        raw[symbol],yahoo[symbol],audits[symbol]=yahoo_frame(payload,symbol,expected=domain);receipts['Yahoo_'+symbol]=receipt
    start=previous-pd.Timedelta(days=120);overlap=native_dates(str(start.date()),str(previous.date()))
    for symbol in ('SOXL','SOXX','SOXS'):
        payload,receipt=http.get('daily_nasdaq_'+symbol,f'https://api.nasdaq.com/api/quote/{symbol}/historical',{'assetclass':'etf','fromdate':str(start.date()),'todate':str(previous.date()),'limit':10000},deadline=deadline,headers={'User-Agent':'Mozilla/5.0','Accept':'application/json','Origin':'https://www.nasdaq.com','Referer':'https://www.nasdaq.com/'})
        nasdaq[symbol]=nasdaq_frame(payload,expected=overlap);receipts['Nasdaq_'+symbol]=receipt
    return {'session':session,'previous_session':str(previous.date()),'yahoo':yahoo,'yahoo_raw':raw,'nasdaq':nasdaq,'audit':audits,'receipts':receipts,'collected_at':clock()}

def build_features(root,isolated_root,bundle):
    """Original formulas run on new isolated providers; no active Recorder allowed."""
    import qlib
    from qlib.data import D
    from src.data.qlib_native import build_qlib_store_from_frame
    from src.workflow.multi_3x_etf_normalized_nested_wf import _attach_normalized_features,_build_normalized_provider,Multi3xEtfNormalizedNestedWfSpec
    from src.workflow.direct_continuous_soxx_allocation import _load_frame,DirectContinuousAllocationSpec
    from src.workflow.pit_constituent_auxiliary_training_allocation import _soxx_common_training_frame,PitAuxiliaryAllocationSpec,COMMON_FEATURES
    root=Path(root);isolated_root=Path(isolated_root);end=bundle['previous_session'];yahoo=bundle['yahoo'];manifests={}
    raw=[]
    for symbol in ETFS:
        frame=yahoo[symbol].copy();frame['vix']=yahoo['^VIX'].close.reindex(frame.index)
        if frame.vix.isna().any():raise ValueError('VIX calendar mismatch')
        frame['instrument']=symbol;raw.append(frame.reset_index())
    daily=pd.concat(raw,ignore_index=True);manifests['daily']=build_qlib_store_from_frame(daily,provider_uri=isolated_root/'data/qlib/soxl',as_of=end,source='Prospective Yahoo quote OHLC*ratio, vendor volume')
    # Original execution prefix is read from canonical provider, never overwritten.
    import pickle
    binding=json.loads((root/'research'/NAME/'bootstrap_prefix_binding.json').read_text());prefix_path=root/binding['object_path']
    if sha256(prefix_path.read_bytes()).hexdigest()!=binding['object_sha256']:raise ValueError('immutable execution prefix changed')
    prefix=pickle.loads(prefix_path.read_bytes());prefix=prefix.loc[prefix.index.get_level_values('datetime')<=pd.Timestamp(end)]
    execution=[]
    for symbol in ('SOXL','SOXX','SOXS'):
        old=prefix.xs(symbol,level='instrument').dropna(how='all');fresh=bundle['nasdaq'][symbol].mul(bundle['audit'][symbol]['ratio'].reindex(bundle['nasdaq'][symbol].index),axis=0)
        if fresh.isna().any().any():raise ValueError('execution adjustment ratio gap')
        overlap=fresh.index[fresh.index<=old.index.max()]
        np.testing.assert_allclose(old.loc[overlap].to_numpy(float),fresh.loc[overlap].to_numpy(float),rtol=1e-6,atol=1e-8)
        combined=pd.concat([old,fresh[fresh.index>old.index.max()]]);strict_grid(combined,native_dates(str(combined.index.min().date()),end),label='execution_'+symbol)
        if not np.isfinite(combined[['open','close']].to_numpy(float)).all() or not combined[['open','close']].gt(0).all().all():raise ValueError('execution used open/close invalid')
        combined['instrument']=symbol;execution.append(combined.reset_index())
    manifests['execution']=build_qlib_store_from_frame(pd.concat(execution,ignore_index=True),provider_uri=isolated_root/'data/qlib/soxl_soxx_soxs',as_of=end,source='Pinned canonical execution prefix + strict NasdaqOHLC/Yahoo ratio overlap')
    path=isolated_root/'data/raw/soxs_defensive_v1';path.mkdir(parents=True,exist_ok=True);atomic_json({'source':'IOv2 strict roles','as_of':end,'relationship_audit':{'kind':'descriptive_only','close_return_correlation':pd.concat(execution).pivot(index='datetime',columns='instrument',values='close').pct_change(fill_method=None).corr().to_dict()}},path/'source_manifest.json')
    pairframes=[]
    for etf,underlying in PAIRS.items():
        a=yahoo[etf].copy();b=yahoo[underlying];strict_grid(b.reindex(a.index),a.index,label='pair calendar')
        if b.reindex(a.index).isna().any().any():raise ValueError('pair missing rows')
        a=a.join(b.add_prefix('underlying_'));a['vix']=yahoo['^VIX'].close.reindex(a.index);a['instrument']=etf;pairframes.append(a.reset_index())
    normalized=_attach_normalized_features(pd.concat(pairframes,ignore_index=True));manifests['normalized']=_build_normalized_provider(Multi3xEtfNormalizedNestedWfSpec(root=isolated_root,experiment='dynamic_soxx_soxl_cash_allocation_wf_v1'),normalized,as_of=end)
    frame,audit,columns=_load_frame(DirectContinuousAllocationSpec(root=isolated_root),include_incomplete_latest=True)
    common=_soxx_common_training_frame(PitAuxiliaryAllocationSpec(root=isolated_root),end=end)
    date=pd.Timestamp(end);features=frame.loc[[date],columns];common=common.loc[[date],list(COMMON_FEATURES)]
    if len(columns)!=39 or common.shape[1]!=16 or not np.isfinite(features.to_numpy(float)).all() or not np.isfinite(common.to_numpy(float)).all():raise ValueError('exact finite39/16 feature gate failed')
    return {'features':features,'common':common,'market_row':frame.loc[date],'provider_manifests':manifests,'builder_audit':audit,'bundle_hash':digest({'source_receipts':bundle['receipts'],'features':features,'common':common}),'source_bundle':bundle}

def require_phase_v2(card,phase,*,at=None):
    import jsonschema
    from src.schedule.diagnostic_forward_study import registered_slots
    from src.backtest.diagnostic_forward_state import FAMILIES,COSTS
    card=Path(card).resolve();root=card.parents[1]
    if phase!='pilot' or sha256((card/'plan.json').read_bytes()).hexdigest()!=PLAN_SHA:raise PermissionError('v2 plan/phase changed')
    schemas=json.loads((card/'registration_schema.json').read_text());permit=json.loads((card/'pilot_activation.json').read_text());jsonschema.validate(permit,schemas['permit_schema'])
    regpath=card/permit['registration_file']
    if sha256(regpath.read_bytes()).hexdigest()!=permit['registration_sha256']:raise PermissionError('v2 registration hash')
    registration=json.loads(regpath.read_text());jsonschema.validate(registration,schemas['registration_schema'])
    required_modules={'src/data/diagnostic_forward_io_v2.py','src/schedule/diagnostic_forward_transport_v2.py','src/backtest/diagnostic_forward_native_v2.py','src/data/diagnostic_forward_inputs.py','src/backtest/diagnostic_forward_state.py','src/schedule/diagnostic_forward_study.py'}
    current_plan=json.loads((card/'plan.json').read_text());parent_path=root/current_plan['parent_contract']['path']
    if sha256(parent_path.read_bytes()).hexdigest()!=current_plan['parent_contract']['sha256']:raise PermissionError('parent contract hash')
    parent=json.loads(parent_path.read_text());required_modules.update(parent['unchanged_code_sha256'])
    required_modules.update({'src/data/qlib_native.py','src/data/minute_intake.py','src/data/recovery_first_hour.py','src/workflow/direct_continuous_soxx_allocation.py','src/workflow/multi_3x_etf_normalized_nested_wf.py','src/workflow/soxs_defensive_hurdle.py','src/factor/high_chop_ml.py','src/model/minute_pit_allocator.py','src/model/tangent_allocator.py'})
    if not required_modules.issubset(registration['code_sha256']):raise PermissionError('missing required module/dependency code pins')
    for relative,expected in parent['unchanged_code_sha256'].items():
        if sha256((root/relative).read_bytes()).hexdigest()!=expected:raise PermissionError('frozen dependency changed')
    for relative,value in registration['code_sha256'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or sha256(path.read_bytes()).hexdigest()!=value:raise PermissionError('v2 code hash')
    for file,key in [(root/'runtime/soxl/stage1_regime_paper/config.json','endpoint_config_sha256'),(card/'event_input_binding.json','event_binding_sha256'),(card/'bootstrap_prefix_binding.json','prefix_binding_sha256')]:
        if sha256(file.read_bytes()).hexdigest()!=registration[key]:raise PermissionError('v2 bound input hash')
    bootstrap=json.loads((card/'bootstrap_prefix_binding.json').read_text());prefix_path=(root/bootstrap['object_path']).resolve()
    if not prefix_path.is_relative_to(root) or sha256(prefix_path.read_bytes()).hexdigest()!=bootstrap['object_sha256']:raise PermissionError('execution prefix artifact hash')
    slots=registered_slots(registration['session']);start=pd.Timestamp(registration['session']).tz_localize('America/New_York')+pd.Timedelta(hours=8);stop=pd.Timestamp(slots['terminal_session']).tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=48)
    if registration['session']<'2026-10-19' or slots['terminal_session']>='2027-01-01' or registration['previous_session']!=slots['previous_session'] or registration['terminal_session']!=slots['terminal_session']:raise PermissionError('v2 session identity')
    if set(registration['ledger_ids'])!={f'{f}_{c}' for f in FAMILIES for c in COSTS}:raise PermissionError('ledger identity')
    now=utc(at or pd.Timestamp.now(tz='UTC'))
    if not utc(registration['sealed_at'])<start or not utc(permit['issued_at'])<start or not start<=now<=stop:raise PermissionError('v2 permit clock')
    for item in (registration,permit):
        if utc(item['not_before'])!=start or utc(item['expires_at'])!=stop:raise PermissionError('v2 registered time window')
    if permit['budgets']!=registration['budgets']:raise PermissionError('v2 budget identity')
    return {**permit,'session':registration['session']}

def validate_daily_bundle_v2(bundle,*,seal,role_approval,expected_dates,required_roles,session):
    if role_approval.get('status')!='APPROVED_PROSPECTIVE_IO_V2' or role_approval.get('plan_sha256')!=PLAN_SHA:raise ValueError('v2 role identity')
    day=pd.Timestamp(session).tz_localize('America/New_York')
    if utc(seal['utc'])>=day+pd.Timedelta(hours=9,minutes=29):raise ValueError('proposal missed09:29')
    for receipt in bundle['source_bundle']['receipts'].values():
        ordered(receipt['request_started'],receipt['received']);ordered(receipt['received'],seal,strict=True)
        if utc(receipt['received']['utc'])>=day+pd.Timedelta(hours=9,minutes=20):raise ValueError('source receipt missed09:20')
    if set(bundle['source_bundle']['yahoo'])!=set((*ETFS,'^VIX')) or set(bundle['source_bundle']['nasdaq'])!={'SOXL','SOXX','SOXS'}:raise ValueError('incomplete source bundle')
    expected_previous=native_dates('2010-03-11',session)[-2]
    if bundle['source_bundle']['previous_session']!=str(expected_previous.date()) or bundle['source_bundle']['session']!=session:raise ValueError('bundle session identity')
    for name,count in [('features',39),('common',16)]:
        if not bundle[name].index.equals(pd.DatetimeIndex([expected_previous],name=bundle[name].index.name)):raise ValueError('bundle feature date')
        values=bundle[name].to_numpy(float)
        if values.shape!=(1,count) or not np.isfinite(values).all():raise ValueError('invalid frozen feature vector')
    if bundle.get('known_unknown_fundamental',True):raise ValueError('reference refresh unresolved')
    return digest(bundle)

def _end_as_start(frame):
    transformed=frame.copy();times=pd.to_datetime(frame.time_key)
    transformed['time_key']=times-pd.Timedelta(minutes=1)
    return transformed

def validate_bars_v2(frame,receipt,*,session,cutoff,seal,mapping_evidence):
    from src.data.diagnostic_forward_inputs import validate_bars
    if mapping_evidence.get('edge')!='end' or mapping_evidence.get('plan_sha256')!=PLAN_SHA:raise ValueError('v2 end mapping absent')
    result=validate_bars(_end_as_start(frame),receipt,session=session,cutoff=cutoff,seal=seal,mapping_evidence={**mapping_evidence,'edge':'start'})
    result['original_interval_end']=result.bar_start+pd.Timedelta(minutes=1)
    return result

def validate_close_bars_v2(frame,receipt,*,session,session_close,commit_clock,mapping_evidence):
    from src.data.diagnostic_forward_inputs import validate_close_bars
    if mapping_evidence.get('edge')!='end' or mapping_evidence.get('plan_sha256')!=PLAN_SHA:raise ValueError('v2 end mapping absent')
    return validate_close_bars(_end_as_start(frame),receipt,session=session,session_close=session_close,commit_clock=commit_clock,mapping_evidence={**mapping_evidence,'edge':'start'})

def capture_execution_prefix(root):
    """One immutable local bootstrap capture, not a new market observation."""
    import pickle,qlib
    from qlib.data import D
    from qlib.workflow import R
    root=Path(root).resolve();card=root/'research'/NAME;path=card/'bootstrap_prefix_binding.json'
    if path.exists():return json.loads(path.read_text())
    os.environ['MLFLOW_DISABLE_TELEMETRY']='true';os.environ['MLFLOW_ALLOW_FILE_STORE']='true';os.environ['DO_NOT_TRACK']='true'
    provider=root/'data/qlib/soxl_soxx_soxs';meta=json.loads((provider/'qlib_store_manifest.json').read_text())
    qlib.init(provider_uri=str(provider),region='us',kernels=1)
    frame=D.features(['SOXL','SOXX','SOXS'],['$open','$high','$low','$close'],start_time=meta['calendar_start'],end_time=meta['calendar_end'],freq='day').rename(columns=lambda x:x.removeprefix('$'))
    with R.start(experiment_name=NAME+'_offline',uri=str(card/'qlib_recorder')):
        rec=R.get_recorder();rec.save_objects(artifact_path='bootstrap',execution_prefix=frame,provider_manifest=meta);rec.log_metrics(bootstrap_rows=len(frame))
    artifact=card/'qlib_recorder'/str(rec.experiment_id)/rec.id/'artifacts/bootstrap/execution_prefix'
    binding={'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'source':'existing Qlib D.features immutable local bootstrap; not fresh source receipt','captured_at':clock(),'object_path':str(artifact.relative_to(root)),'object_sha256':sha256(artifact.read_bytes()).hexdigest(),'array_digest':digest(frame),'source_provider_manifest_sha256':sha256((provider/'qlib_store_manifest.json').read_bytes()).hexdigest(),'calendar_start':meta['calendar_start'],'calendar_end':meta['calendar_end'],'rows':len(frame)}
    atomic_json(binding,path);return binding

# Only a validated, pre-registered HttpLedger call may enter requests transport.
from contextlib import contextmanager
from contextvars import ContextVar
_HTTP_LEDGER_TOKEN=ContextVar('diagnostic_io_v2_HTTP_ledger',default=False)
@contextmanager
def http_fence():
    import requests
    from unittest.mock import patch
    original=requests.sessions.Session.request;blocked=[]
    def guarded(session,method,url,*args,**kwargs):
        if not _HTTP_LEDGER_TOKEN.get():
            blocked.append({'kind':'unregistered_HTTP_before_send','method':str(method)});raise PermissionError('HTTP outside fixed17 ledger')
        return original(session,method,url,*args,**kwargs)
    with patch.object(requests.sessions.Session,'request',guarded):yield blocked
