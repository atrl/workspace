"""Prospective IO v2 finite transport and atomic durable state runner.
"""
from copy import deepcopy
import hashlib,json,os,pickle,time
os.environ["MLFLOW_DISABLE_TELEMETRY"]="true"
os.environ["DO_NOT_TRACK"]="true"
os.environ["MLFLOW_ALLOW_FILE_STORE"]="true"
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data.diagnostic_forward_inputs import network_denied,digest,canonical,utc,validate_live_open,validate_quote
from src.data.diagnostic_forward_io_v2 import PLAN_SHA,NAME
from src.data.diagnostic_forward_io_v2 import validate_daily_bundle_v2 as validate_daily_bundle, validate_bars_v2 as validate_bars, validate_close_bars_v2 as validate_close_bars, require_phase_v2 as require_phase
from src.backtest.diagnostic_forward_state import ForwardPaperState,FAMILIES,COSTS
from src.data.recovery_first_hour import aggregate_first_hour,FIELDS
from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json
NAME='diagnostic_forward_io_v2_20261016'
MODULES=['src/data/diagnostic_forward_io_v2.py','src/schedule/diagnostic_forward_transport_v2.py','src/backtest/diagnostic_forward_native_v2.py']

def persist_checkpoint(directory,cursor,record,payload):
    """Commit marker is written last; a quote may consume only a committed seal."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    event=json.dumps(canonical(record),sort_keys=True,allow_nan=False).encode();state=pickle.dumps(payload,protocol=pickle.HIGHEST_PROTOCOL)
    names={f'event_{cursor:03d}.json':event,f'state_{cursor:03d}.pkl':state}
    hashes={name:hashlib.sha256(body).hexdigest() for name,body in names.items()}
    names[f'commit_{cursor:03d}.json']=json.dumps({'cursor':cursor,'files':hashes,'plan_sha256':PLAN_SHA},sort_keys=True).encode()
    for name,body in names.items():
        path=directory/name
        if path.exists():
            if path.read_bytes()!=body:raise ValueError('conflicting durable checkpoint')
            continue
        with path.open('xb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
    descriptor=os.open(directory,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return hashes

def load_checkpoint(directory):
    directory=Path(directory);commits=sorted(directory.glob('commit_*.json'))
    if not commits:return None
    meta=json.loads(commits[-1].read_text())
    if meta['plan_sha256']!=PLAN_SHA:raise ValueError('checkpoint plan changed')
    for name,expected in meta['files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:raise ValueError('checkpoint byte mismatch')
    return pickle.loads((directory/f"state_{meta['cursor']:03d}.pkl").read_bytes())

def registered_slots(session):
    import exchange_calendars as xc
    day=pd.Timestamp(session).normalize().tz_localize(None)
    calendar=xc.get_calendar('XNYS',start=day-pd.Timedelta(days=10),end=day+pd.Timedelta(days=10))
    dates=pd.DatetimeIndex(calendar.sessions).tz_localize(None);position=dates.get_loc(day)
    previous=dates[position-1];terminal=dates[position+1]
    at=lambda date,h,m:date.tz_localize('America/New_York')+pd.Timedelta(hours=h,minutes=m)
    return {'session':str(day.date()),'previous_session':str(previous.date()),'terminal_session':str(terminal.date()),'known_cutoff':at(day,9,45),'open_fill':at(day,9,47),'recovery_cutoff':at(day,10,30),'recovery_fill':at(day,10,32),'close':calendar.session_close(day).tz_convert('America/New_York'),'terminal_fill':at(terminal,9,47)}

def assert_fixed_time(value,expected):
    if utc(value)!=utc(expected):raise ValueError('event differs from registered fixed slot')

class FiniteStudyRunner:
    SLOTS=('daily','known','open_intent','open_quote','recovery_data','recovery_intent','recovery_quote','close','terminal')
    def __init__(self,card,*,session,previous_session,expected_dates,tariff,recovery_spec,action_spec,controller,selection,role_approval,bar_mapping,clock_now):
        self.permit=require_phase(card,'pilot',at=clock_now)
        if self.permit.get('session')!=session:raise PermissionError('pilot session differs from permit')
        self.journal_dir=Path(card).resolve().parents[1]/'runtime/research'/NAME/self.permit['registration_sha256'][:16]
        if role_approval.get('status')!='APPROVED_PROSPECTIVE_IO_V2':raise ValueError('source mapping remains pending')
        self.slots=registered_slots(session)
        if previous_session!=self.slots['previous_session']:raise ValueError('previous date is not prior XNYS session')
        self.session=session;self.previous_session=previous_session;self.expected_dates=expected_dates;self.role_approval=role_approval;self.bar_mapping=bar_mapping
        self.state=ForwardPaperState(tariff);self.spec=recovery_spec;self.action_spec=action_spec;self.controller=controller;self.selection=selection
        self.price_events=[];self.native_points={k:[] for k in self.state.books};self.native_orders={k:[] for k in self.state.books}
        self.cursor=0;self.journal=[];self.rows=None;self.bundle_hash=None;self.pending_targets=None;self.recovery_calls=0;self.durable_intents=set()
        saved=load_checkpoint(self.journal_dir)
        if saved is not None:
            for key,value in saved.items():setattr(self,key,value)
    def dispatch(self,event):
        slot=event['slot']
        if self.cursor>=len(self.SLOTS) or slot!=self.SLOTS[self.cursor]:raise ValueError('finite slot order violation')
        state_before=deepcopy(self.state)
        record={'slot':slot,'input_sha256':digest(event),'state_before':self.state.state_digest()}
        try:
            if slot=='daily':
                self.bundle_hash=validate_daily_bundle(event['bundle'],seal=event['seal'],role_approval=self.role_approval,expected_dates=self.expected_dates,required_roles=self.role_approval['roles'],session=self.session)
                self.rows=deepcopy(event['model_rows'])
                if set(self.rows)!=set(FAMILIES[:-1]) or event['proposal_bundle_hash']!=self.bundle_hash:raise ValueError('proposals not bound to daily bundle')
            elif slot=='known':
                assert_fixed_time(event['cutoff'],self.slots['known_cutoff'])
                bars=validate_bars(event['bars'],event['receipt'],session=self.session,cutoff=event['cutoff'],seal=event['seal'],mapping_evidence=self.bar_mapping)
                first=bars.groupby('symbol').open.first().to_dict();returns=validate_live_open(event['open_receipt'],session=self.session,previous_session=self.previous_session,decision_seal=event['seal'],futu_opens=first)
                proposals={f:np.array([self.rows[f][c] for c in ['soxl_weight','soxx_weight','cash_weight']]) for f in FAMILIES[:-1]};smoothed=self.state.smooth(proposals,returns)
                for f in smoothed:
                    for column,value in zip(['soxl_weight','soxx_weight','cash_weight'],smoothed[f]):self.rows[f][column]=value
                self.state.apply_actions(self.session,event['schedule']);prices=bars.groupby('symbol').close.last().reindex(['SOXL','SOXX']).to_numpy()
                self.pending_targets=self.state.open_targets(self.rows,prices,recovery_spec=self.spec,action_spec=self.action_spec)
            elif slot in ['open_intent','recovery_intent']:
                assert_fixed_time(event['fill_slot_at'],self.slots['open_fill' if slot=='open_intent' else 'recovery_fill'])
                if not self.state.trading_disabled:self.state.seal(slot,self.pending_targets,event['seal'],slot_at=event['fill_slot_at'],bindings={'feature_bundle':self.bundle_hash,'plan':PLAN_SHA})
            elif slot=='recovery_data':
                assert_fixed_time(event['cutoff'],self.slots['recovery_cutoff'])
                if not self.state.trading_disabled:
                    used=validate_bars(event['bars'],event['receipt'],session=self.session,cutoff=event['cutoff'],seal=event['seal'],mapping_evidence=self.bar_mapping)
                    points=used.copy();points['timestamp']=points.bar_start+pd.Timedelta(minutes=1);points['date']=pd.Timestamp(self.session);points['price_valid']=True;points['is_final']=True
                    aggregated=aggregate_first_hour(points).loc[pd.Timestamp(self.session)]
                    if aggregated.first_hour_ready!=1:raise ValueError('incomplete recovery aggregation')
                    fields={name:float(aggregated[name]) for name in FIELDS}
                    rows={f:pd.Series({**dict(self.rows[f]),**fields}) for f in FAMILIES[:-1]}
                    record['recovery_fields_from_validated_bars']=fields
                    self.pending_targets,votes,calls=self.state.recovery_targets(rows,recovery_spec=self.spec,controller=self.controller,selection=self.selection);self.recovery_calls+=calls
                    if self.recovery_calls>9:raise ValueError('recovery call budget exceeded')
                    record['votes']=votes
            elif slot in ['open_quote','recovery_quote']:
                intent='open_intent' if slot=='open_quote' else 'recovery_intent'
                fixed=self.slots['open_fill' if slot=='open_quote' else 'recovery_fill']
                if 'slot_at' in event:assert_fixed_time(event['slot_at'],fixed)
                if not self.state.trading_disabled:
                    if intent not in self.durable_intents:raise ValueError('intent not durably committed before quote')
                    record['execution']=self.state.execute_slot(intent,event['receipt'])
                else:
                    prices=validate_quote(event['receipt'],seal=event['prior_seal'],slot_at=fixed);record['passive_nav']=self.state.mark(prices,update_peak=False)
            elif slot=='close':
                assert_fixed_time(event['session_close'],self.slots['close'])
                used=validate_close_bars(event['bars'],event['receipt'],session=self.session,session_close=event['session_close'],commit_clock=event['commit_clock'],mapping_evidence=self.bar_mapping)
                prices=used.groupby('symbol').close.last().reindex(['SOXL','SOXX']).to_numpy();record['nav']=self.state.mark(prices,update_peak=True)
            elif slot=='terminal':
                if event['session']!=self.slots['terminal_session']:raise ValueError('terminal session differs from registration')
                assert_fixed_time(event['slot_at'],self.slots['terminal_fill'])
                self.state.apply_actions(event['session'],event['schedule']);prices=validate_quote(event['receipt'],seal=event['prior_seal'],slot_at=event['slot_at']);record['nav']=self.state.mark(prices,update_peak=False)
            record['status']='committed'
        except Exception as error:
            self.state=state_before
            self.state.disable(type(error).__name__+': '+str(error));record.update(status='unqualified_slot',error=str(error))
        if record['status']=='committed' and slot in ('known','open_quote','recovery_quote','close','terminal'):
            logical={'known':'known_cutoff','open_quote':'open_fill','recovery_quote':'recovery_fill','close':'close','terminal':'terminal_fill'}[slot]
            if slot in ('known','close'):
                observed=bars if slot=='known' else used
                prices=observed.groupby('symbol').close.last().reindex(['SOXL','SOXX']).to_numpy(float)
                price_times={s:str(observed.loc[observed.symbol.eq(s),'bar_start'].iloc[-1]+pd.Timedelta(minutes=1)) for s in ('SOXL','SOXX')}
            else:
                prices=np.array([event['receipt']['quotes'][s]['price'] for s in ('SOXL','SOXX')],float);price_times={s:event['receipt']['quotes'][s]['quote_at'] for s in ('SOXL','SOXX')}
            mark={'event_id':slot,'logical_stamp':self.slots[logical].tz_localize(None),'session':self.slots['terminal_session'] if slot=='terminal' else self.session,'price_soxl':prices[0],'price_soxx':prices[1],'ca_due':slot in ('known','terminal'),'endpoint_role':'start' if slot=='open_quote' else ('terminal' if slot=='terminal' else 'none'),'receipt':deepcopy(event['receipt']),'received_at':event['receipt']['received']['utc'],'price_times':price_times}
            self.price_events.append(mark)
            for key,book in self.state.books.items():
                self.native_points[key].append({'event_id':slot,'q_soxl':book.q[0],'q_soxx':book.q[1],'cash':book.cash,'receivable':book.income.value,'nav':book.nav(prices)})
                for order in record.get('execution',{}).get('reports',{}).get(key,{}).get('orders',[]):self.native_orders[key].append({'event_id':slot,**order,'originalstamp':order['stamp']})
        record['books_after']=self.state.snapshot();record['state_after']=self.state.state_digest();self.journal.append(record);self.cursor+=1
        if slot in ['open_intent','recovery_intent'] and record['status']=='committed':self.durable_intents.add(slot)
        payload={key:getattr(self,key) for key in ['state','cursor','journal','rows','bundle_hash','pending_targets','recovery_calls','durable_intents','price_events','native_points','native_orders']}
        try:persist_checkpoint(self.journal_dir,self.cursor,{'event':event,'result':record},payload)
        except Exception:
            self.durable_intents.discard(slot);self.state.disable('durable commit failed');raise
        return deepcopy(record)
    def run(self,events):
        for event in events:self.dispatch(event)
        if self.cursor!=len(self.SLOTS):raise ValueError('incomplete finite pilot stream')
        return {'journal':self.journal,'state':self.state,'qualified_interval':not self.state.trading_disabled,'recovery_calls':self.recovery_calls}


from src.data.diagnostic_forward_io_v2 import HttpLedger,collect_daily,build_features,chart_params
from src.data.diagnostic_forward_probe import clock,global_allowlist,snapshot_clocks
from src.data.diagnostic_forward_inputs import ordered

class PilotQuoteClient:
    """One context, fixed15 application operations; processor guard forbids hidden paging."""
    def __init__(self,host,port,directory):
        from futu import OpenQuoteContext,RET_OK,RET_ERROR,KLType,AuType,KL_FIELD,Session
        from futu.common.ft_logger import logger
        logger.file_logger.disabled=True;logger.console_logger.disabled=True;logger.file_level=100;logger.console_level=100
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.calls={};self.queries={};self.active=None;self.first_pages={}
        owner=self
        class Bounded(OpenQuoteContext):
            connections=0;handshakes=0
            def __init__(self,*a,**kw):
                try:super().__init__(*a,**kw)
                except Exception:
                    try:self.close()
                    except Exception:pass
                    raise
            def _init_connect_sync(self):self._auto_reconnect=False;self._query_timeout=12;self._sync_query_connect_timeout=0;return super()._init_connect_sync()
            def _connect_sync(self):
                self.connections+=1
                return RET_ERROR if self.connections>1 else super()._connect_sync()
            def _send_init_connect_sync(self):
                self.handshakes+=1
                return (RET_ERROR,'bounded handshake') if self.handshakes>1 else super()._send_init_connect_sync()
            def _wait_reconnect(self,*a,**kw):self._auto_reconnect=False;return None
            def on_api_socket_reconnected(self):
                if self._sub_record.get_sub_list():return RET_ERROR,'subscriptions forbidden'
                return RET_OK,''
            def _get_sync_query_processor(self,pack,unpack):
                delegate=super()._get_sync_query_processor(pack,unpack);kind=pack.__qualname__.split('.')[0]
                def guarded(**kwargs):
                    key=owner.active
                    allowed={'global':'GlobalStateQuery','rehab':'RequestRehab','history':'RequestHistoryKlineQuery','snapshot':'MarketSnapshotQuery'}
                    if key is None or kind!=allowed[key.split('_')[0]] or key in owner.queries or kwargs.get('next_req_key') is not None:return RET_ERROR,'fixed query budget',None
                    owner.queries[key]={'started':clock()};atomic_json(owner.queries,owner.directory/'protocol_queries.json')
                    result=delegate(**kwargs);owner.queries[key].update(received=clock(),ret_code=int(result[0]));atomic_json(owner.queries,owner.directory/'protocol_queries.json')
                    if kind=='RequestHistoryKlineQuery' and result[0]==RET_OK:
                        owner.first_pages[key]=result[2];(owner.directory/(key+'.first_page.pkl')).write_bytes(pickle.dumps(result[2]))
                    return result
                return guarded
        self.context=Bounded(host=host,port=port,is_async_connect=False);self.constants=(RET_OK,KLType.K_1M,AuType.NONE,[KL_FIELD.ALL],Session.RTH)
    def call(self,key,*,session=None,symbol=None):
        allowed={'global_open','global_terminal','rehab_open_SOXL','rehab_open_SOXX','rehab_terminal_SOXL','rehab_terminal_SOXX','history_known_SOXL','history_known_SOXX','history_recovery_SOXL','history_recovery_SOXX','history_close_SOXL','history_close_SOXX','snapshot_open','snapshot_recovery','snapshot_terminal'}
        if key not in allowed or key in self.calls or (self.directory/(key+'.receipt.json')).exists():raise ValueError('unregistered/duplicate Futu call')
        self.active=key;receipt={'operation':key,'request_started':clock()};self.calls[key]=receipt;atomic_json(receipt,self.directory/(key+'.receipt.json'))
        safe=None
        try:
            if key.startswith('global'):result=self.context.get_global_state()
            elif key.startswith('rehab'):result=self.context.get_rehab('US.'+symbol)
            elif key.startswith('snapshot'):result=self.context.get_market_snapshot(['US.SOXL','US.SOXX'])
            else:
                _,ktype,autype,fields,rth=self.constants
                result=self.context.request_history_kline('US.'+symbol,start=session,end=session,ktype=ktype,autype=autype,fields=fields,max_count=1000,page_req_key=None,extended_time=False,session=rth)
            receipt.update(received=clock(),ret_code=int(result[0]));ordered(receipt['request_started'],receipt['received'])
            if result[0]!=self.constants[0]:raise ValueError('SDK return not OK')
            safe=global_allowlist(result[1]) if key.startswith('global') else result[1]
            if key.startswith('history') and result[2] is not None:raise ValueError('pagination incomplete')
            receipt['payload_sha256']=digest(safe);receipt['source_identity']='Futu OpenQuoteContext';receipt['completed']=True
            return safe,receipt
        except Exception as error:receipt.update(completed=True,error_type=type(error).__name__);raise
        finally:
            self.active=None
            if safe is not None:(self.directory/(key+'.payload.pkl')).write_bytes(pickle.dumps(safe))
            atomic_json(receipt,self.directory/(key+'.receipt.json'))
    def close(self):self.context.close()

def await_time(target):
    target=utc(target)
    while utc(pd.Timestamp.now(tz='UTC'))<target:time.sleep(max(0.,min(1.,(target-utc(pd.Timestamp.now(tz='UTC'))).total_seconds())))

def paired_bars(client,phase,session,cutoff,mapping):
    frames=[];receipts=[]
    for symbol in ('SOXL','SOXX'):
        frame,receipt=client.call('history_'+phase+'_'+symbol,session=session,symbol=symbol)
        if not isinstance(frame,pd.DataFrame) or set(frame['code'])!={'US.'+symbol}:raise ValueError('history returned wrong symbol')
        part=frame.copy();part['symbol']=symbol;frames.append(part);receipts.append(receipt)
    full=pd.concat(frames,ignore_index=True);times=pd.to_datetime(full.time_key).dt.tz_localize('America/New_York');used=full[times<=utc(cutoff)].copy()
    receipt={'request_started':receipts[0]['request_started'],'received':receipts[-1]['received'],'payload_sha256':digest(full),'source_identity':'Futu paired history','children':receipts,**{k:mapping[k] for k in ['sdk_sha256','endpoint','query_mode']}}
    return used,receipt,full

def paired_quote(client,phase):
    frame,receipt=client.call('snapshot_'+phase);snapshot_clocks(frame,receipt['received'])
    receipt['quotes']={str(r.code).removeprefix('US.'):{'price':float(r.last_price),'quote_at':pd.Timestamp(r.update_time).tz_localize('America/New_York').isoformat()} for r in frame.itertuples()}
    return receipt

def validate_live_open_child(row,receipt):
    received=utc(receipt['received']['utc']);market=utc(row['regular_market_time'])
    if not 0<=(received-market).total_seconds()<=120:raise ValueError('child own-receipt market time stale/future')
    if any(utc(row[k])>received for k in ('current_bar_timestamp','previous_bar_timestamp')):raise ValueError('child own-receipt future bar timestamp')

def live_open(http,session,previous):
    start=clock();symbols={};receipts=[];deadline=pd.Timestamp(session).tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=46)
    for symbol in ('SOXL','SOXX'):
        payload,receipt=http.get('live_open_'+symbol,'https://query2.finance.yahoo.com/v8/finance/chart/'+symbol,chart_params(previous,session),deadline=deadline)
        data=payload['chart']['result'][0];meta=data['meta']
        if meta.get('symbol')!=symbol or meta.get('currency')!='USD' or meta.get('exchangeTimezoneName')!='America/New_York':raise ValueError('live-open metadata')
        stamps=pd.to_datetime(data['timestamp'],unit='s',utc=True);dates=stamps.tz_convert('America/New_York').date.astype(str);quote=data['indicators']['quote'][0];adj=data['indicators']['adjclose'][0]['adjclose'];row={'regular_market_time':pd.Timestamp(meta['regularMarketTime'],unit='s',tz='UTC').isoformat()}
        if list(dates)!=[previous,session]:raise ValueError('live-open exact previous/current dates')
        for i,prefix in enumerate(('previous','current')):
            row.update({prefix+'_date':dates[i],prefix+'_bar_timestamp':stamps[i].isoformat(),prefix+'_open':quote['open'][i],prefix+'_close':quote['close'][i],prefix+'_adjclose':adj[i]})
        validate_live_open_child(row,receipt)
        symbols[symbol]=row;receipts.append(receipt)
    return {'request_started':start,'received':receipts[-1]['received'],'payload_sha256':digest(symbols),'source_identity':'Yahoo query2 LIVE_OPENv2','symbols':symbols,'children':receipts}

def load_frozen(root,plan):
    from src.schedule.diagnostic_forward_study import _local_source
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
    binding_path=Path(root)/plan['bindings']['controller_file']
    if hashlib.sha256(binding_path.read_bytes()).hexdigest()!=plan['bindings']['controller_sha256']:raise ValueError('controller binding identity')
    bindings=json.loads(binding_path.read_text())
    controller,_=_local_source(root,bindings['controller'],list(bindings['controller']['artifacts']));action,_=_local_source(root,bindings['action_and_recovery_spec'],list(bindings['action_and_recovery_spec']['artifacts']));raw,_=_local_source(root,bindings['raw_inheritance'],list(bindings['raw_inheritance']['artifacts']))
    models={}
    for label,source in plan['bindings']['models'].items():
        if label not in ('reference27baa','index_candidate_and_control'):continue
        arms=['candidate'] if label=='reference27baa' else ['candidate','control'];keys=[f'models/{arm}_{seed}' for arm in arms for seed in (71,172,273)];loaded,_=_local_source(root,source,keys)
        for arm in arms:
            family='reference27baa' if label=='reference27baa' else 'index_'+arm
            models[family]=[loaded[f'models/{arm}_{seed}'] for seed in (71,172,273)]
    return {'models':models,'controller':controller['inputs/controller'],'selection':action['controller/source_recovery_selection'],'recovery_spec':IntradayRecoveryDirectAllocationSpec(root=root,**action['inputs/plan']['recovery_spec']),'action_spec':Stage1OpenActionValueSpec(root=root,**action['inputs/plan']['action_spec']),'tariff':raw['inputs/plan']['tariff'],'schedule':raw['inputs/schedule']}

def sealed_predict(bundle,models,extra,directory,*,session):
    from src.schedule.diagnostic_forward_study import _dataset
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);rows={};calls=[]
    for family,cohort in models.items():
        predictions=[];features=bundle['features'].join(extra) if family=='reference27baa' else bundle['features']
        dataset=_dataset(features,bundle['common'])
        for model in cohort:
            key=family+'_'+str(model.seed)
            if (directory/(key+'.attempt.json')).exists():raise ValueError('model already attempted; no retry')
            stamp=clock()
            if utc(stamp['utc'])>=pd.Timestamp(session).tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=29):raise ValueError('prediction deadline')
            atomic_json({'key':key,'started':stamp,'feature_hash':digest(features),'common_hash':digest(bundle['common'])},directory/(key+'.attempt.json'));calls.append(key)
            prediction=model.predict(dataset,segment='predict');(directory/(key+'.prediction.pkl')).write_bytes(pickle.dumps(prediction));predictions.append(prediction)
        ensemble=sum(predictions)/3;weights=['soxl_weight','soxx_weight','cash_weight'];ensemble[weights]=ensemble[weights].div(ensemble[weights].sum(axis=1),axis=0)
        rows[family]=pd.Series({**bundle['market_row'].to_dict(),**ensemble.iloc[0].to_dict()})
    if len(calls)!=9:raise ValueError('exact frozen model cohort')
    return rows


def qualified_mapping(root,endpoint):
    from src.data.diagnostic_forward_probe import sdk_identity
    probe=json.loads((Path(root)/'research/diagnostic_forward_probe_20261016/probe_manifest.json').read_text())
    if sdk_identity(root)!=probe['sdk_identity']:raise ValueError('SDK differs from qualified historical probe')
    plan=json.loads((Path(root)/'research'/NAME/'plan.json').read_text());bound=plan['bindings'];mapping_card=Path(root)/bound['end_mapping_card'];manifest_path=mapping_card/'verification_manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=bound['end_mapping_manifest_sha256']:raise ValueError('end mapping manifest identity')
    manifest=json.loads(manifest_path.read_text())
    if manifest['recorder_id']!=bound['end_mapping_recorder']:raise ValueError('end mapping Recorder identity')
    # Bind the actual qualified repair result, not an editable card status string.
    entry=next(x for x in manifest['entries'] if x['path'].endswith('native_repair_1/result'));result_path=mapping_card/entry['path']
    if hashlib.sha256(result_path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('end mapping result bytes')
    result=pickle.loads(result_path.read_bytes())
    if not result['passed'] or result['rows']!=780:raise ValueError('end mapping saved result failed')
    mapping=json.loads((mapping_card/'summary.json').read_text())
    if mapping.get('qualified_gate')!='bounded_saved_completed_history_end_label_and_Qlib':raise ValueError('saved end mapping unqualified')
    return {'passed':True,'edge':'end','scope':'observed_probe','plan_sha256':PLAN_SHA,'sdk_sha256':digest(probe['sdk_identity']),'endpoint':digest({'host':endpoint['host'],'port':endpoint['port']}),'query_mode':'K_1M_NONE_RTH_single_query'}

def validate_recorded_native(runner,directory,*,schedule,tariff):
    from src.backtest.diagnostic_forward_native_v2 import build_event_provider,validate_event_book
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if len(runner.price_events)!=5:raise ValueError('incomplete actual mark map; no fabricated native points')
    manifest=build_event_provider(runner.price_events,directory/'provider');results={}
    for key in runner.state.books:
        path=directory/(key+'.attempt.json')
        if path.exists():raise ValueError('native book already attempted; no silent retry')
        atomic_json({'book':key,'started':clock(),'mode':'recorded_event_accounting_only'},path)
        result=validate_event_book(pd.DataFrame(runner.native_points[key]),pd.DataFrame(runner.native_orders[key]),runner.price_events,directory/'provider',schedule,tariff,runner.state.books[key].cost,100000.)
        (directory/(key+'.result.pkl')).write_bytes(pickle.dumps(result));results[key]=result
    return {'provider':manifest,'books':results,'all12_passed':len(results)==12 and all(x['status']=='passed' for x in results.values())}

def relevant_sec_form(form):
    return str(form).strip().upper().removesuffix('/A') in ('8-K','10-Q','10-K')

def _run_pilot(root):
    """Real finite IO chain. Only an exact root v2 activation record can enter it."""
    import qlib
    from qlib.workflow import R
    from src.data.nvda_own_guidance_realization import align_realizations
    root=Path(root).resolve();card=root/'research'/NAME;permit=require_phase(card,'pilot',at=clock()['utc']);plan=json.loads((card/'plan.json').read_text());registration=json.loads((card/permit['registration_file']).read_text())
    session=permit['session'];slots=registered_slots(session);directory=root/'runtime/research'/NAME/permit['registration_sha256'][:16]
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'launch.json').exists():raise ValueError('finite pilot already attempted; no automatic restart')
    atomic_json({'started':clock(),'permit':permit,'registered_session':session},directory/'launch.json')
    client=None;http=None;runner=None;frozen=None;status={'qualified_interval':False,'forward_days':0}
    try:
        # Setup failures after launch registration also produce a terminal receipt.
        inbox_path=card/'event_input_binding.json'
        if hashlib.sha256(inbox_path.read_bytes()).hexdigest()!=registration['event_binding_sha256']:raise ValueError('event inbox identity')
        inbox=json.loads(inbox_path.read_text());config_path=root/'runtime/soxl/stage1_regime_paper/config.json'
        if hashlib.sha256(config_path.read_bytes()).hexdigest()!=registration['endpoint_config_sha256']:raise ValueError('endpoint identity')
        endpoint=json.loads(config_path.read_text());mapping=qualified_mapping(root,endpoint)
        qlib.init(provider_uri=str(root/'data/qlib/soxl'),region='us',kernels=1)
        frozen=load_frozen(root,plan);http=HttpLedger(directory/'http',permit=permit)
        await_time(pd.Timestamp(session).tz_localize('America/New_York')+pd.Timedelta(hours=8))
        daily=collect_daily(http,session=session)
        sec,sec_receipt=http.get('SEC_status','https://data.sec.gov/submissions/CIK0001045810.json',{},deadline=pd.Timestamp(session).tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=20),headers={'User-Agent':'soxl-qlib-research/1.0'})
        recent=sec['filings']['recent'];cutoff=str(pd.Timestamp(inbox['reference']['latest_classified_available_at']).date())
        lengths=[len(recent.get(k,[])) for k in ('accessionNumber','filingDate','form')]
        if not lengths[0] or len(set(lengths))!=1 or min(recent['filingDate'])>cutoff:raise ValueError('SEC recent coverage does not reach classified boundary')
        new=[a for a,d,f in zip(recent['accessionNumber'],recent['filingDate'],recent['form']) if d>=cutoff and relevant_sec_form(f) and a not in inbox['reference']['acknowledged_accessions']]
        if new:raise ValueError('new unclassified SEC filing: reference unavailable')
        bundle=build_features(root,directory/'isolated',daily);bundle['known_unknown_fundamental']=False
        reference=inbox['reference'];source_summary=json.loads((root/reference['card']/'summary.json').read_text());pair_path=Path(source_summary['recorder_local_dir'])/'artifacts'/reference['pairs']['artifact']
        if hashlib.sha256(pair_path.read_bytes()).hexdigest()!=reference['pairs']['sha256']:raise ValueError('reference pairs identity')
        pairs=pickle.loads(pair_path.read_bytes()).copy()
        received=pd.to_datetime(inbox['study_ingested_at_utc'],utc=True);available=pd.to_datetime(pairs.actual_report_available_at,utc=True)
        # At study intake only latest realized state is carried; guide-only NaT rows never overwrite it.
        observed=pairs.loc[available.notna() & available.le(received)].copy()
        latest=available.loc[observed.index].idxmax();pairs=observed.loc[[latest]].copy();pairs['actual_report_available_at']=received
        decisions=pd.Series([pd.Timestamp(session).tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=29)],index=bundle['features'].index);extra,trace,_=align_realizations(pairs,decisions)
        bundle['fundamental_trace']=trace;bundle['SEC_receipt']=sec_receipt
        role={'status':'APPROVED_PROSPECTIVE_IO_V2','plan_sha256':PLAN_SHA,'roles':['execution','daily','common','multi_pair']}
        rows=sealed_predict(bundle,frozen['models'],extra,directory/'predictions',session=session)
        runner=FiniteStudyRunner(card,session=session,previous_session=slots['previous_session'],expected_dates=bundle['features'].index,tariff=frozen['tariff'],recovery_spec=frozen['recovery_spec'],action_spec=frozen['action_spec'],controller=frozen['controller'],selection=frozen['selection'],role_approval=role,bar_mapping=mapping,clock_now=clock()['utc'])
        seal=clock();runner.dispatch({'slot':'daily','bundle':bundle,'seal':seal,'model_rows':rows,'proposal_bundle_hash':digest(bundle)})
        if runner.state.trading_disabled:raise ValueError('daily state rejected')
        atomic_json({'attempt':1,'started':clock(),'reconnect_allowed':False},directory/'context_attempt.json')
        client=PilotQuoteClient(endpoint['host'],int(endpoint['port']),directory/'Futu')
        schedule=frozen['schedule']
        def event_check(phase):
            global_state,_=client.call('global_'+phase)
            if not global_state.get('qot_logined',False):raise ValueError('quote session unavailable')
            for symbol in ('SOXL','SOXX'):
                rehab,receipt=client.call('rehab_'+phase+'_'+symbol,symbol=symbol)
                dates=pd.to_datetime(rehab['ex_div_date'],errors='raise').dt.normalize()
                active=rehab.loc[dates.between(pd.Timestamp(session),pd.Timestamp(slots['terminal_session']))]
                if not active.empty:raise ValueError('unclassified effective corporate-action interval '+symbol)
        def stage(slot,producer):
            try:event=producer()
            except Exception as error:
                runner.state.disable('transport '+slot+': '+type(error).__name__+': '+str(error));event={'slot':slot}
            return runner.dispatch(event)
        try:event_check('open')
        except Exception as error:runner.state.disable('initial event gate: '+str(error))
        await_time(slots['known_cutoff']+pd.Timedelta(seconds=5))
        def known():
            bars,receipt,_=paired_bars(client,'known',session,slots['known_cutoff'],mapping);opening=live_open(http,session,slots['previous_session'])
            return {'slot':'known','bars':bars,'receipt':receipt,'cutoff':slots['known_cutoff'],'seal':clock(),'open_receipt':opening,'schedule':schedule}
        stage('known',known);seal=clock();stage('open_intent',lambda:{'slot':'open_intent','seal':seal,'fill_slot_at':slots['open_fill']})
        await_time(slots['open_fill']);stage('open_quote',lambda:{'slot':'open_quote','receipt':paired_quote(client,'open'),'slot_at':slots['open_fill'],'prior_seal':seal})
        await_time(slots['recovery_cutoff']+pd.Timedelta(seconds=5))
        def recovery():
            bars,receipt,_=paired_bars(client,'recovery',session,slots['recovery_cutoff'],mapping)
            return {'slot':'recovery_data','bars':bars,'receipt':receipt,'cutoff':slots['recovery_cutoff'],'seal':clock()}
        stage('recovery_data',recovery);seal=clock();stage('recovery_intent',lambda:{'slot':'recovery_intent','seal':seal,'fill_slot_at':slots['recovery_fill']})
        await_time(slots['recovery_fill']);stage('recovery_quote',lambda:{'slot':'recovery_quote','receipt':paired_quote(client,'recovery'),'slot_at':slots['recovery_fill'],'prior_seal':seal})
        await_time(slots['close']+pd.Timedelta(seconds=5))
        def closing():
            bars,receipt,_=paired_bars(client,'close',session,slots['close'],mapping)
            return {'slot':'close','bars':bars,'receipt':receipt,'session_close':slots['close'],'commit_clock':clock()}
        stage('close',closing);await_time(slots['terminal_fill']-pd.Timedelta(minutes=2))
        try:event_check('terminal')
        except Exception as error:runner.state.disable('terminal event gate: '+str(error))
        seal=clock();await_time(slots['terminal_fill']);stage('terminal',lambda:{'slot':'terminal','session':slots['terminal_session'],'slot_at':slots['terminal_fill'],'receipt':paired_quote(client,'terminal'),'prior_seal':seal,'schedule':schedule})
        status.update(qualified_interval=not runner.state.trading_disabled,forward_days=int(not runner.state.trading_disabled),journal=runner.journal,recovery_calls=runner.recovery_calls)
    except Exception as error:
        status.update(error_type=type(error).__name__,error=str(error))
        if runner is not None:runner.state.disable(str(error));status['state']=runner.state.snapshot();status['journal']=runner.journal
    finally:
        if client is not None:client.close()
        status.update(context_attempts=int((directory/'context_attempt.json').exists()),connection_attempts=None if client is None and (directory/'context_attempt.json').exists() else (0 if client is None else client.context.connections),handshakes=None if client is None and (directory/'context_attempt.json').exists() else (0 if client is None else client.context.handshakes),http_attempts=0 if http is None else len(http.records),futu_calls=0 if client is None else len(client.calls),finished_at=clock(),pilot_identity='prospective_IO_v2_not_historical_equivalence')
        atomic_json(status,directory/'terminal_status.json')
    if frozen is None:
        status['terminal_outcome']='initialization_failed_before_sources';atomic_json(status,directory/'terminal_status.json');return status
    native=None
    if runner is not None:
        try:
            native=validate_recorded_native(runner,directory/'native',schedule=frozen['schedule'],tariff=frozen['tariff']);status['native12_passed']=native['all12_passed']
        except Exception as error:status.update(native12_passed=False,native_error=type(error).__name__+': '+str(error))
    if not status.get('native12_passed',False):status['forward_days']=0;status['qualified_interval']=False
    atomic_json(status,directory/'terminal_status.json')
    # Qlib lifecycle: all feature-provider builds above have finished before Recorder context.
    qlib.init(provider_uri=str(root/'data/qlib/soxl'),region='us',kernels=1)
    with R.start(experiment_name=NAME+'_pilot',uri=str(card/'qlib_recorder')):
        rec=R.get_recorder();rec.save_objects(artifact_path='pilot',status=status,registration=registration,permit=permit,native=native)
        rec.save_objects(artifact_path='transport',durable_files={str(p.relative_to(directory)):p.read_bytes() for p in directory.rglob('*') if p.is_file() and 'isolated' not in p.parts})
        if runner is not None:rec.save_objects(artifact_path='pilot',state=runner.state,journal=runner.journal)
        rec.save_objects(artifact_path='sources',**{Path(f).stem:(root/f).read_text() for f in MODULES});rec.log_metrics(http_attempts=status['http_attempts'],futu_calls=status['futu_calls'],qualified_intervals=int(status['qualified_interval']))
    return status

def run_pilot(root):
    from src.data.diagnostic_forward_io_v2 import http_fence
    with http_fence() as blocked:
        try:result=_run_pilot(root)
        finally:
            # A separate receipt also survives activation/setup exceptions.
            atomic_json({'blocked_unregistered_HTTP':blocked,'actual_ledger_calls_only':True},Path(root)/'research'/NAME/'http_fence_receipt.json')
    result['blocked_unregistered_HTTP']=len(blocked)
    return result

def offline_fixture(root,output):
    """One local full chain: real saved parser + explicit synthetic transport/models."""
    import qlib
    from qlib.data import D
    from unittest.mock import patch
    from types import SimpleNamespace
    from src.data.diagnostic_forward_io_v2 import yahoo_frame,native_dates,capture_execution_prefix
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True);session='2026-10-12';previous='2026-10-09';slots=registered_slots(session)
    binding=capture_execution_prefix(root)
    # A genuine saved Yahoo quote/adjclose/actions parser fixture, without downloading.
    sourcecard=root/'research/equal_weight_market_context_information_20261015';source_summary=json.loads((sourcecard/'summary.json').read_text());vm=json.loads((sourcecard/'verification_manifest.json').read_text());entry=next(x for x in vm['entries'] if x['path'].endswith('source_attempts/SPY_query2'));path=sourcecard/entry['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'];saved=pickle.loads(path.read_bytes());payload=json.loads(saved['raw_response_bytes']);real_raw,real_adjusted,real_audit=yahoo_frame(payload,'SPY',expected=native_dates('2010-03-11','2022-12-30'))
    original=pickle.loads((root/binding['object_path']).read_bytes());source_frames={};source_manifests={}
    for provider,symbols in [('soxl',['SOXL','SOXX']),('tqqq',['TQQQ','QQQ']),('upro',['UPRO','SPY']),('spxl',['SPXL']),('tecl',['TECL','XLK'])]:
        uri=root/'data/qlib'/provider;qlib.init(provider_uri=str(uri),region='us',kernels=1)
        loaded=D.features(symbols,['$open','$high','$low','$close','$volume','$vix'],start_time='2010-03-11',end_time=previous,freq='day').rename(columns=lambda x:x.removeprefix('$'))
        source_manifests[provider]=hashlib.sha256((uri/'qlib_store_manifest.json').read_bytes()).hexdigest()
        for symbol in symbols:source_frames[symbol]=loaded.xs(symbol,level='instrument')
    soxs=original.xs('SOXS',level='instrument').loc[:previous].copy();soxs['volume']=100.;source_frames['SOXS']=soxs
    source_frames['^VIX']=source_frames['SOXL'][['vix']].rename(columns={'vix':'close'})
    origin=pd.Timestamp(session+' 08:00',tz='America/New_York')
    def fixture_clock(at):
        stamp=pd.Timestamp(at)
        if stamp.tzinfo is None:stamp=stamp.tz_localize('America/New_York')
        return {'utc':stamp.isoformat(),'monotonic_ns':int((stamp-origin).total_seconds()*1e9)+1}
    class SavedTransport:
        def __init__(self):self.keys=[]
        def get(self,key,url,params,*,deadline,headers=None):
            if key in self.keys:raise AssertionError('duplicate fixture request')
            self.keys.append(key);symbol=key.rsplit('_',1)[-1];symbol='^VIX' if symbol=='VIX' else symbol
            if key.startswith('daily_yahoo'):
                frame=source_frames[symbol].copy()
                if symbol!='^VIX':
                    # Explicit synthetic candle envelope; not certification of old mixed-provider OHLC.
                    frame['high']=frame[['open','high','close']].max(axis=1);frame['low']=frame[['open','low','close']].min(axis=1)
                index=frame.index;quote={c:frame[c].tolist() for c in ['open','high','low','close','volume'] if c in frame}
                stamps=[int(d.tz_localize('America/New_York').replace(hour=9,minute=30).timestamp()) for d in index];data={'meta':{'symbol':symbol,'currency':'USD','exchangeTimezoneName':'America/New_York','firstTradeDate':stamps[0]},'timestamp':stamps,'indicators':{'quote':[quote],'adjclose':[{'adjclose':frame.close.tolist()}]},'events':{}};body={'chart':{'result':[data],'error':None}}
            else:
                frame=original.xs(symbol,level='instrument').loc[params['fromdate']:params['todate']];body={'data':{'tradesTable':{'rows':[{'date':str(d.date()),**{c:str(r[c]) for c in ['open','high','low','close']}} for d,r in frame.iterrows()]}}}
            start=fixture_clock(session+' 08:30:00');end=fixture_clock(session+' 08:30:01');return body,{'request_started':start,'received':end,'payload_sha256':digest(body),'source_identity':'SYNTHETIC_TRANSPORT_FROM_SAVED_QLIB','key':key}
    transport=SavedTransport();daily=collect_daily(transport,session=session);bundle=build_features(root,output/'isolated',daily);bundle['known_unknown_fundamental']=False
    families=FAMILIES[:-1];models={f:[SimpleNamespace(seed=s) for s in (71,172,273)] for f in families};mock_calls=[]
    for family,cohort in models.items():
        for model in cohort:
            def predict(dataset,segment='predict',family=family,seed=model.seed):
                mock_calls.append((family,seed));index=dataset.prepare(segment,col_set=['feature']).index
                return pd.DataFrame([[.2,.5,.3,.1,.6]],index=index,columns=['soxl_weight','soxx_weight','cash_weight','tail_probability','alpha_probability'])
            model.predict=predict
    with patch(__name__+'.clock',lambda:fixture_clock(session+' 09:28:00')):
        rows=sealed_predict(bundle,models,pd.DataFrame(index=bundle['features'].index),output/'mock_predictions',session=session)
    card=output/'research'/NAME;card.mkdir(parents=True,exist_ok=True)
    tariff=json.loads((root/'research/frozen_policy_raw_cashbook_20261015/plan.json').read_text())['tariff'];permit={'session':session,'registration_sha256':'f'*64}
    mapping={'passed':True,'edge':'end','scope':'observed_probe','plan_sha256':PLAN_SHA,'sdk_sha256':'synthetic','endpoint':'synthetic','query_mode':'synthetic'}
    with patch(__name__+'.require_phase',return_value=permit):
        runner=FiniteStudyRunner(card,session=session,previous_session=previous,expected_dates=bundle['features'].index,tariff=tariff,recovery_spec=IntradayRecoveryDirectAllocationSpec(root=root),action_spec=Stage1OpenActionValueSpec(root=root),controller={'frozen_recovery_model':None},selection={'action_beta':.6,'minimum_source_alpha_probability':.25,'probability_threshold':.825},role_approval={'status':'APPROVED_PROSPECTIVE_IO_V2','plan_sha256':PLAN_SHA,'roles':[]},bar_mapping=mapping,clock_now=origin)
    def bars(cutoff):
        times=pd.date_range(pd.Timestamp(session+' 09:31',tz='America/New_York'),cutoff,freq='min');return pd.DataFrame([{'symbol':symbol,'time_key':t,'open':price,'high':price,'low':price,'close':price,'volume':100.} for symbol,price in [('SOXL',100.),('SOXX',200.)] for t in times])
    def receipt(start,end):return {'request_started':fixture_clock(start),'received':fixture_clock(end),'payload_sha256':'a'*64,'source_identity':'synthetic','sdk_sha256':'synthetic','endpoint':'synthetic','query_mode':'synthetic'}
    def quote(day,time):
        start=pd.Timestamp(day+' '+time,tz='America/New_York');r=receipt(start,start+pd.Timedelta(seconds=2));r['quotes']={s:{'price':p,'quote_at':(start+pd.Timedelta(seconds=1)).isoformat()} for s,p in [('SOXL',100.),('SOXX',200.)]};return r
    r=runner.dispatch({'slot':'daily','bundle':bundle,'model_rows':rows,'seal':fixture_clock(session+' 09:28:10'),'proposal_bundle_hash':digest(bundle)});assert r['status']=='committed',r
    opened=receipt(session+' 09:45:10',session+' 09:45:15');opened['symbols']={s:{'current_date':session,'previous_date':previous,'regular_market_time':pd.Timestamp(session+' 09:45:14',tz='America/New_York').isoformat(),'current_bar_timestamp':pd.Timestamp(session+' 09:30',tz='America/New_York').isoformat(),'previous_bar_timestamp':pd.Timestamp(previous+' 09:30',tz='America/New_York').isoformat(),'current_open':p,'current_close':p,'current_adjclose':p,'previous_open':p,'previous_close':p,'previous_adjclose':p} for s,p in [('SOXL',100.),('SOXX',200.)]}
    events=[{'slot':'known','bars':bars(slots['known_cutoff']),'receipt':receipt(session+' 09:45:05',session+' 09:45:10'),'cutoff':slots['known_cutoff'],'seal':fixture_clock(session+' 09:45:20'),'open_receipt':opened,'schedule':{}},{'slot':'open_intent','seal':fixture_clock(session+' 09:46:00'),'fill_slot_at':slots['open_fill']},{'slot':'open_quote','receipt':quote(session,'09:47:00'),'slot_at':slots['open_fill']},{'slot':'recovery_data','bars':bars(slots['recovery_cutoff']),'receipt':receipt(session+' 10:30:05',session+' 10:30:10'),'cutoff':slots['recovery_cutoff'],'seal':fixture_clock(session+' 10:31:00')},{'slot':'recovery_intent','seal':fixture_clock(session+' 10:31:10'),'fill_slot_at':slots['recovery_fill']},{'slot':'recovery_quote','receipt':quote(session,'10:32:00'),'slot_at':slots['recovery_fill'],'prior_seal':fixture_clock(session+' 10:31:10')},{'slot':'close','bars':bars(slots['close']),'receipt':receipt(slots['close']+pd.Timedelta(seconds=5),slots['close']+pd.Timedelta(seconds=10)),'session_close':slots['close'],'commit_clock':fixture_clock(slots['close']+pd.Timedelta(seconds=11))},{'slot':'terminal','session':slots['terminal_session'],'slot_at':slots['terminal_fill'],'receipt':quote(slots['terminal_session'],'09:47:00'),'prior_seal':fixture_clock(slots['terminal_fill']-pd.Timedelta(minutes=1)),'schedule':{}}]
    for event in events:
        record=runner.dispatch(event)
        assert record['status']=='committed',record
    assert runner.cursor==9 and len(runner.state.books)==12 and not runner.state.trading_disabled
    native=validate_recorded_native(runner,output/'native',schedule={},tariff=tariff);assert native['all12_passed']
    result={'real_saved_Yahoo_SPY_rows':len(real_raw),'real_saved_parser_SHA':entry['sha256'],'synthetic_daily_requests':len(transport.keys),'mock_model_calls':len(mock_calls),'actual_model_calls':0,'feature_shape':list(bundle['features'].shape),'common_shape':list(bundle['common'].shape),'committed_slots':runner.cursor,'books':len(runner.state.books),'native12_passed':native['all12_passed'],'source_provider_manifest_sha256':source_manifests,'forward_days':0,'live_ready':False,'explicit_coverage':'One real saved Yahoo SPY parser; other instrument HTTP structure from saved Qlib synthetic responses with synthetic high/low envelopes, mockmodels and synthetic clock/bar/quote fullslot stream. No live transport or model equivalence claim.'}
    (output/'fixture_payload.pkl').write_bytes(pickle.dumps({'result':result,'native':native,'bundle':bundle,'journal':runner.journal,'state':runner.state,'price_events':runner.price_events,'native_points':runner.native_points,'native_orders':runner.native_orders,'real_SPY_raw':real_raw,'real_SPY_adjusted':real_adjusted}));atomic_json(result,output/'fixture_result.json');return result
