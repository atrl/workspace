"""Exactly four optional read-only Futu calls, gated by an exact-code root permit."""
from contextlib import contextmanager
from pathlib import Path
from hashlib import sha256
import importlib.metadata
import json,os,socket,time
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.data.diagnostic_forward_inputs import PLAN_SHA,require_phase,sanitize_global_state,canonical,digest,utc,ordered
from src.data.minute_intake import atomic_json
NAME='diagnostic_forward_probe_20261016'
CONFIG='runtime/soxl/stage1_regime_paper/config.json'
SDK_FILES=['quote/open_quote_context.py','common/open_context_base.py','common/constant.py','common/ft_logger.py']

def clock():return {'utc':pd.Timestamp.now(tz='UTC').isoformat(),'monotonic_ns':time.monotonic_ns()}

def sdk_identity(root):
    base=Path(root)/'qlib_venv/lib/python3.12/site-packages/futu'
    return {'distribution_version':importlib.metadata.version('futu-api'),'files':{str((base/f).relative_to(root)):sha256((base/f).read_bytes()).hexdigest() for f in SDK_FILES},'history_signature':'request_history_kline(code,start,end,ktype,autype,fields,max_count,page_req_key,extended_time,session)','history_arguments':{'ktype':'KLType.K_1M','autype':'AuType.NONE','fields':['KL_FIELD.ALL'],'max_count':1000,'page_req_key':None,'extended_time':False,'session':'Session.RTH'},'implicit_transport':'One InitConnect handshake, optional KeepAlive only. Initial subscription registry empty; automatic reconnect disabled before first connection attempt. No account or subscription queries.','docs':'Official pages direct-open timed out in this turn; previously read official documentation plus official cached search excerpts support API semantics. Installed SDK source hashes bind actual signature/behavior.'}

def completed_session(at):
    import exchange_calendars as xc
    at=utc(at);day=at.tz_convert('America/New_York').tz_localize(None).normalize();cal=xc.get_calendar('XNYS',start=day-pd.Timedelta(days=14),end=day+pd.Timedelta(days=1))
    eligible=[session for session in cal.sessions if cal.session_close(session)+pd.Timedelta(seconds=120)<=at]
    if not eligible:raise ValueError('no completed native session')
    session=eligible[-1];opening=cal.session_open(session).tz_convert('America/New_York');closing=cal.session_close(session).tz_convert('America/New_York')
    return str(pd.Timestamp(session).date()),pd.date_range(opening,closing-pd.Timedelta(minutes=1),freq='min')

def global_allowlist(raw):
    safe={k:v for k,v in raw.items() if k in ['market_us','qot_logined','server_ver','open_d_version'] and isinstance(v,(str,int,float,bool))}
    if 'qot_logined' in safe:
        value=str(safe['qot_logined']).lower()
        if value not in ['0','1','false','true']:safe.pop('qot_logined')
        else:safe['qot_logined']=value in ['1','true']
    return sanitize_global_state(safe)

def validate_history(frame,symbol,expected):
    if not isinstance(frame,pd.DataFrame) or not {'code','time_key','open','high','low','close','volume'}<=set(frame):raise ValueError('history_schema')
    if set(frame.code)!={symbol}:raise ValueError('history_symbol')
    times=pd.to_datetime(frame.time_key);times=times.dt.tz_localize('America/New_York') if times.dt.tz is None else times.dt.tz_convert('America/New_York')
    if not pd.DatetimeIndex(times).equals(expected):raise ValueError('history_exact_RTH_slot_coverage')
    values=frame[['open','high','low','close','volume']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:,:4]<=0).any() or (values[:,4]<0).any():raise ValueError('history_invalid_OHLCV')
    if not frame.high.ge(frame[['open','close']].max(axis=1)).all() or not frame.low.le(frame[['open','close']].min(axis=1)).all():raise ValueError('history_OHLC_range')
    return {'symbol':symbol,'rows':len(frame),'first_raw_label':str(times.iloc[0]),'last_raw_label':str(times.iloc[-1]),'empirical_edge':'start','normalized_first_end':str(times.iloc[0]+pd.Timedelta(minutes=1)),'normalized_last_end':str(times.iloc[-1]+pd.Timedelta(minutes=1)),'as_received_closed_interval':True,'immutable_vendor_finality_certified':False}

def snapshot_clocks(frame,received):
    if not isinstance(frame,pd.DataFrame) or not {'code','last_price','update_time'}<=set(frame) or set(frame.code)!={'US.SOXL','US.SOXX'} or len(frame)!=2:raise ValueError('snapshot_pair_coverage')
    if not np.isfinite(frame.last_price.astype(float)).all() or not frame.last_price.gt(0).all():raise ValueError('snapshot_prices')
    rows=[]
    for item in frame.to_dict('records'):
        stamp=pd.Timestamp(item['update_time'])
        if pd.isna(stamp):raise ValueError('snapshot_clock_missing')
        stamp=stamp.tz_localize('America/New_York') if stamp.tzinfo is None else stamp.tz_convert('America/New_York')
        if stamp>utc(received['utc']):raise ValueError('snapshot_clock_future')
        rows.append({'symbol':item['code'],'quote_at':stamp.isoformat(),'received_at':received['utc'],'lag_seconds':float((utc(received['utc'])-stamp).total_seconds()),'live_fill_freshness_qualified':False})
    return rows

class SingleQueryGuard:
    """Guard BEFORE the SDK processor can pack/send an automatic next page."""
    def __init__(self,session,ret_ok,ret_error,emit):
        self.session=session;self.ret_ok=ret_ok;self.ret_error=ret_error;self.emit=emit;self.admitted={};self.blocked=[];self.first_pages={}
    def wrap(self,pack_func,delegate):
        kind=pack_func.__qualname__.split('.')[0]
        def guarded(**kwargs):
            if kind=='GlobalStateQuery':key='global_state'
            elif kind=='MarketSnapshotQuery':
                if set(kwargs.get('stock_list',[]))!={'US.SOXL','US.SOXX'}:return self.ret_error,'unexpected snapshot universe',None
                key='snapshot_pair'
            elif kind=='RequestHistoryKlineQuery':
                code=kwargs.get('code');key='history_'+str(code)
                if code not in ['US.SOXL','US.SOXX'] or kwargs.get('start_date')!=self.session+' 00:00:00' or kwargs.get('end_date')!=self.session+' 23:59:59':return self.ret_error,'unexpected history domain',None
            else:return self.ret_error,'unapproved data processor',None
            if key in self.admitted or len(self.admitted)>=4 or (kind=='RequestHistoryKlineQuery' and kwargs.get('next_req_key') is not None):
                record={'operation':key,'at':clock(),'blocked_before_send':True};self.blocked.append(record);self.emit('protocol_blocked',key,record,None);return self.ret_error,'fixed single-query budget blocked paging',None
            start=clock();self.admitted[key]={'started':start};self.emit('protocol_started',key,{'operation':key,'started':start},None)
            result=delegate(**kwargs);received=clock();ordered(start,received);self.admitted[key].update(received=received,ret_code=int(result[0]))
            preserved=None
            if kind=='RequestHistoryKlineQuery' and result[0]==self.ret_ok:
                data,has_next,page=result[2];preserved={'rows':data,'has_next':bool(has_next),'page_key_present':page is not None,'received':received};self.first_pages[kwargs['code']]=preserved
            # Never persist raw global state, request kwargs/user_id, or raw error strings.
            self.emit('protocol_completed',key,{'operation':key,**self.admitted[key]},preserved)
            return result
        return guarded

def collect_four(client,session,expected,constants,emit):
    """Emit a durable attempt BEFORE each call; rejected decoded data is retained."""
    receipts=[];checks=[];snap=[]
    operations=[('global_state',lambda:client.get_global_state()),('snapshot_pair',lambda:client.get_market_snapshot(['US.SOXL','US.SOXX']))]
    for symbol in ['US.SOXL','US.SOXX']:
        operations.append(('history_'+symbol,lambda symbol=symbol:client.request_history_kline(symbol,start=session,end=session,ktype=constants['ktype'],autype=constants['autype'],fields=constants['fields'],max_count=1000,page_req_key=None,extended_time=False,session=constants['session'])))
    for key,call in operations:
        receipt={'operation':key,'attempt':1,'started':clock(),'completed':False};emit('started',key,dict(receipt),None);safe=None
        try:
            result=call();received=clock();ret=int(result[0]);receipt.update(received=received,ret_code=ret)
            if ret!=constants['RET_OK'] and key.startswith('history_'):
                first=getattr(client,'probe_guard',None)
                first=first.first_pages.get(key.removeprefix('history_')) if first is not None else None
                if first is not None:safe=pd.DataFrame(first['rows']);receipt['recovered_first_processor_page']=True
            if ret==constants['RET_OK']:
                safe=global_allowlist(result[1]) if key=='global_state' and isinstance(result[1],dict) else (None if key=='global_state' else result[1])
            page=result[2] if key.startswith('history_') and ret==constants['RET_OK'] else None
            del result # raw global discarded before any emit/hash/log.
            ordered(receipt['started'],received)
            if ret!=constants['RET_OK']:receipt['outcome']='SDK_return_not_OK'
            elif key=='global_state':receipt['outcome']='observed' if safe is not None and 'market_us' in safe else 'global_schema_failed'
            elif key.startswith('history_'):
                if page is not None:raise ValueError('pagination_not_permitted')
                checks.append(validate_history(safe,key.removeprefix('history_'),expected));receipt['outcome']='observed'
            else:snap=snapshot_clocks(safe,received);receipt['outcome']='observed'
        except Exception as error:receipt.update(received=clock(),outcome='failed',error_type=type(error).__name__)
        receipt.update(completed=True,payload_sha256=digest(safe),payload_kind='sanitized_global_allowlist' if key=='global_state' else 'decoded_SDK_response_not_wire_bytes')
        receipts.append(receipt);emit('completed',key,dict(receipt),safe)
    passed=len(checks)==2 and all(r['outcome']=='observed' for r in receipts)
    return {'application_calls':len(receipts),'source_gate_passed':passed,'bar_checks':checks,'snapshot_clocks':snap,'receipts':receipts,'qualified_forward_days':0,'live_fill_clock_qualified':False,'pilot_ready':False}

@contextmanager
def endpoint_only(host,port):
    import requests
    original_connect=socket.socket.connect;original_ex=socket.socket.connect_ex;denied=[]
    def allowed(address):return isinstance(address,tuple) and address[0]==host and address[1]==port
    def connect(sock,address):
        if not allowed(address):denied.append('nonpermitted_socket');raise PermissionError('endpoint fence')
        return original_connect(sock,address)
    def connect_ex(sock,address):
        if not allowed(address):denied.append('nonpermitted_socket');raise PermissionError('endpoint fence')
        return original_ex(sock,address)
    def no_http(*args,**kwargs):denied.append('nonpermitted_HTTP');raise PermissionError('probe has zero HTTP budget')
    with patch.object(socket.socket,'connect',connect),patch.object(socket.socket,'connect_ex',connect_ex),patch.object(requests.sessions.Session,'request',no_http):yield denied

def make_client(host,port,session,emit):
    from futu import OpenQuoteContext,RET_OK,RET_ERROR,KLType,AuType,KL_FIELD,Session
    from futu.common.constant import ContextStatus
    from futu.common.ft_logger import logger
    # Disable both destinations before any connection or response can be logged.
    logger.file_logger.disabled=True;logger.console_logger.disabled=True
    logger.file_level=100;logger.console_level=100
    class BoundedContext(OpenQuoteContext):
        probe_connections=0;probe_handshakes=0;probe_heartbeats=0
        def __init__(self,*args,**kwargs):
            self.probe_guard=SingleQueryGuard(session,RET_OK,RET_ERROR,emit)
            try:super().__init__(*args,**kwargs)
            except Exception:
                try:self.close()
                except Exception:pass
                raise
        def _get_sync_query_processor(self,pack_func,unpack_func):
            return self.probe_guard.wrap(pack_func,super()._get_sync_query_processor(pack_func,unpack_func))
        def _init_connect_sync(self):
            self._auto_reconnect=False;self._query_timeout=12;self._sync_query_connect_timeout=0
            return super()._init_connect_sync()
        def _connect_sync(self):
            self.probe_connections+=1
            if self.probe_connections>1:return RET_ERROR
            return super()._connect_sync()
        def _wait_reconnect(self,*args,**kwargs):self._auto_reconnect=False;return None
        def _send_init_connect_sync(self):
            self.probe_handshakes+=1
            if self.probe_handshakes>1:return RET_ERROR,'bounded handshake'
            return super()._send_init_connect_sync()
        def on_api_socket_reconnected(self):
            if self._sub_record.get_sub_list():return RET_ERROR,'subscription prohibited'
            return RET_OK,''
        def keep_alive(self):self.probe_heartbeats+=1;return super().keep_alive()
    client=BoundedContext(host=host,port=port,is_async_connect=False)
    constants={'RET_OK':RET_OK,'ktype':KLType.K_1M,'autype':AuType.NONE,'fields':[KL_FIELD.ALL],'session':Session.RTH}
    return client,constants,client._status==ContextStatus.READY

def native_readback(root,card,rec,checks):
    import qlib
    from qlib.data import D
    from src.data.qlib_native import build_qlib_store_from_frame
    rows=[];symbols=[]
    for check in checks:
        code=check['symbol'];symbol=code.removeprefix('US.');symbols.append(symbol)
        frame=rec.load_object('receipts/history_'+code+'_completed')['payload'].copy()
        times=pd.to_datetime(frame.time_key)
        if times.dt.tz is not None:times=times.dt.tz_convert('America/New_York').dt.tz_localize(None)
        part=frame[['open','high','low','close','volume']].copy();part['datetime']=times+pd.Timedelta(minutes=1);part['instrument']=symbol;rows.append(part)
    if not rows:return {'status':'not_run_no_valid_bars','symbols':[]}
    source=pd.concat(rows,ignore_index=True);provider=root/'data/qlib'/NAME
    manifest=build_qlib_store_from_frame(source,provider_uri=provider,freq='1min',region='us',as_of=str(source.datetime.max().date()),field_map={x:x for x in ['open','high','low','close','volume']},source='Observed Futu NONE RTH prepilot probe; rawlabels start, normalizedends; not live-decision or immutable-finality qualification')
    qlib.init(provider_uri=str(provider),region='us',kernels=1);expected=pd.DatetimeIndex(sorted(source.datetime.unique()))
    actual_calendar=pd.DatetimeIndex(D.calendar(start_time=expected[0],end_time=expected[-1],freq='1min'));pd.testing.assert_index_equal(actual_calendar.rename(None),expected.rename(None))
    fields=['open','high','low','close','volume'];actual=D.features(symbols,['$'+x for x in fields],start_time=expected[0],end_time=expected[-1],freq='1min');errors=[]
    for symbol in symbols:
        wanted=source[source.instrument.eq(symbol)].set_index('datetime')[fields].to_numpy(float);got=actual.xs(symbol,level='instrument').to_numpy(float)
        np.testing.assert_allclose(got,wanted,rtol=1e-7,atol=1e-7);errors.append({'symbol':symbol,'rows':len(wanted),'max_absolute_quantization':float(np.max(np.abs(got-wanted)))})
    rec.save_objects(artifact_path='native_data',provider_manifest=manifest,source_rows=source,calendar=actual_calendar,readback=actual,quantization=pd.DataFrame(errors));atomic_json(manifest,card/'provider_manifest.json')
    return {'status':'passed','symbols':symbols,'rows':len(actual),'calendar_rows':len(actual_calendar),'quantization':errors}

def run(root):
    root=Path(root).resolve();study=root/'research/diagnostic_forward_study_20261016';card=root/'research'/NAME
    permit=require_phase(study,'probe',at=pd.Timestamp.now(tz='UTC')) # before any context/connection
    manifest=json.loads((card/'probe_manifest.json').read_text());config_path=root/CONFIG
    if sha256(config_path.read_bytes()).hexdigest()!=manifest['endpoint_config_sha256']:raise ValueError('endpoint config changed')
    if sdk_identity(root)!=manifest['sdk_identity']:raise ValueError('installed SDK identity changed')
    if sha256(Path(__file__).read_bytes()).hexdigest()!=manifest['probe_code_sha256']:raise ValueError('probe code changed')
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    if (card/'probe_progress.json').exists():raise ValueError('one context attempt already registered; no silent retry')
    config=json.loads(config_path.read_text());host=config['host'];port=int(config['port']);session,expected=completed_session(pd.Timestamp.now(tz='UTC'))
    os.environ['MLFLOW_DISABLE_TELEMETRY']='true';os.environ['DO_NOT_TRACK']='true';os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR']='false';os.environ['MLFLOW_ALLOW_FILE_STORE']='true'
    import qlib
    from qlib.workflow import R
    qlib.init(provider_uri=str(root/'data/qlib/soxl'),region='us',kernels=1)
    progress={'phase':'probe','status':'starting','selected_completed_session':session,'context_attempts':1,'application_attempts':[],'terminal_outcome':None};atomic_json(progress,card/'probe_progress.json')
    tracking=card/'qlib_recorder';client=None;result=None
    with endpoint_only(host,port) as denied:
        with R.start(experiment_name=NAME,uri=str(tracking)):
            rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,card/'probe_progress.json')
            rec.save_objects(artifact_path='inputs',permit=permit,manifest=manifest,selected_session=session,expected_raw_labels=expected);rec.save_objects(artifact_path='sources',probe=Path(__file__).read_text())
            def emit(stage,key,receipt,payload):
                if stage=='started':
                    if len(progress['application_attempts'])>=4:raise ValueError('application budget exceeded')
                    progress['application_attempts'].append(key)
                rec.save_objects(artifact_path='receipts',**{key+'_'+stage:{'receipt':receipt,'payload':payload}});atomic_json(progress,card/'probe_progress.json')
            try:
                client,constants,ready=make_client(host,port,session,emit)
                if not ready:result={'source_gate_passed':False,'application_calls':0,'terminal_outcome':'connection_not_ready','bar_checks':[]}
                else:result=collect_four(client,session,expected,constants,emit);result['terminal_outcome']='completed_four_calls' if result['source_gate_passed'] else 'completed_with_source_gaps'
            except Exception as error:result={'source_gate_passed':False,'application_calls':len(progress['application_attempts']),'terminal_outcome':'context_or_capture_failed','error_type':type(error).__name__,'bar_checks':[]}
            finally:
                transport={'context_instances':1,'connection_attempts':getattr(client,'probe_connections',None),'InitConnect_handshakes':getattr(client,'probe_handshakes',None),'KeepAlive_messages':getattr(client,'probe_heartbeats',None),'protocol_data_queries_admitted':len(client.probe_guard.admitted) if client is not None else None,'protocol_queries_with_success_response':sum(x.get('ret_code')==0 for x in client.probe_guard.admitted.values()) if client is not None else None,'blocked_queries_before_send':len(client.probe_guard.blocked) if client is not None else None,'query_count_semantics':'Admitted processor calls are registered before pack/send; success responses prove sent queries. Failed admitted calls may fail before wire send.'}
                if client is not None:client.close()
            progress.update(status='finished',terminal_outcome=result['terminal_outcome'],finished_at=clock(),transport=transport);atomic_json(progress,card/'probe_progress.json')
            summary={'experiment':NAME,'parent_plan_sha256':PLAN_SHA,**result,'transport':transport,'application_budget':4,'blocked_nonpermitted_network_attempts':len(denied),'new_fits':0,'model_inference_calls':0,'qualified_forward_days':0,'pilot_launches':0,'orders':0,'pilot_ready':False,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','receipts/*','analysis/*']}}
            rec.save_objects(artifact_path='analysis',summary=summary);rec.log_metrics(application_calls=summary['application_calls'],registered_artifact_count=sum(x.is_file() for x in (Path(summary['recorder_local_dir'])/'artifacts').rglob('*')))
    with endpoint_only(host,port) as readback_denied:
        try:summary['Qlib_native_readback']=native_readback(root,card,rec,result['bar_checks'])
        except Exception as error:
            summary['Qlib_native_readback']={'status':'failed','error_type':type(error).__name__};summary['source_gate_passed']=False;summary['terminal_outcome']='native_readback_failed'
        summary['blocked_nonpermitted_network_attempts']+=len(readback_denied)
        rec.save_objects(artifact_path='analysis',summary=summary)
    progress['terminal_outcome']=summary['terminal_outcome'];atomic_json(progress,card/'probe_progress.json')
    artifacts=Path(summary['recorder_local_dir'])/'artifacts';atomic_json({'recorder_id':rec.id,'entries':[{'path':str(x.relative_to(card)),'sha256':sha256(x.read_bytes()).hexdigest(),'bytes':x.stat().st_size} for x in sorted(artifacts.rglob('*')) if x.is_file()]},card/'verification_manifest.json');atomic_json(summary,card/'summary.json');return summary
