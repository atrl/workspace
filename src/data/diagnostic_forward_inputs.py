"""Pure input/permission checks for the isolated diagnostic study; no collectors run here."""
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import socket
from unittest.mock import patch
import numpy as np
import pandas as pd

PLAN_SHA='b27c525b21f230a06bc0463a05fddda13a7c87eaf4b8c33e34bb00ede67927c7'

def canonical(value):
    if isinstance(value,pd.DataFrame):return {'columns':list(value.columns),'index':canonical(list(value.index)),'data':canonical(value.to_numpy().tolist())}
    if isinstance(value,(np.ndarray,pd.Series,pd.Index)):return canonical(value.tolist())
    if isinstance(value,dict):return {str(k):canonical(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [canonical(v) for v in value]
    if isinstance(value,(np.integer,np.bool_)):return value.item()
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else {'nonfinite':str(value)}
    if value is None or isinstance(value,(str,int,bool)):return value
    return str(value)

def digest(value):
    return sha256(json.dumps(canonical(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def utc(value):
    value=pd.Timestamp(value)
    if value.tzinfo is None:raise ValueError('timezone required')
    return value.tz_convert('UTC')

def check_clock(clock):
    stamp=utc(clock['utc']);mono=int(clock['monotonic_ns'])
    if mono<0:raise ValueError('negative monotonic clock')
    return stamp,mono

def ordered(before,after,*,strict=False):
    a,ma=check_clock(before);b,mb=check_clock(after)
    if b<a or mb<ma or (strict and (b<=a or mb<=ma)):raise ValueError('receipt/seal clock order')
    if abs((b-a).total_seconds()-(mb-ma)/1e9)>1:raise ValueError('wall/monotonic elapsed drift')

def require_phase(card,phase,*,at=None):
    card=Path(card).resolve();root=card.parents[1];actual=sha256((card/'plan.json').read_bytes()).hexdigest()
    if actual!=PLAN_SHA:raise PermissionError('unapproved study plan')
    filename={'implementation':'implementation_approval.json','probe':'probe_activation.json','pilot':'pilot_activation.json'}.get(phase)
    if filename is None or not (card/filename).is_file():raise PermissionError('missing explicit '+str(phase)+' permit')
    permit=json.loads((card/filename).read_text())
    if permit.get('phase')!=phase or permit.get('plan_sha256')!=PLAN_SHA:raise PermissionError('phase/plan permit mismatch')
    if phase=='implementation':
        if permit.get('decision')!='APPROVED_IMPLEMENTATION_ONLY' or permit.get('network_authorized') is not False:raise PermissionError('invalid implementation scope')
        return permit
    if not permit.get('network_authorized') or not permit.get(phase+'_authorized'):raise PermissionError('network phase not authorized')
    files=permit.get('code_sha256',{})
    if not files:raise PermissionError('missing code identity')
    for relative,expected in files.items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path.read_bytes()).hexdigest()!=expected:raise PermissionError('code identity mismatch')
    regpath=(card/permit.get('registration_file','')).resolve()
    if not regpath.is_relative_to(card) or not regpath.is_file() or sha256(regpath.read_bytes()).hexdigest()!=permit.get('registration_sha256'):raise PermissionError('registration identity mismatch')
    reg=json.loads(regpath.read_text())
    if reg.get('plan_sha256')!=PLAN_SHA or reg.get('old_days_credited')!=0:raise PermissionError('wrong registration')
    now=utc(at or pd.Timestamp.now(tz='UTC'));start=utc(permit['not_before']);stop=utc(permit['expires_at'])
    if not start<=now<=stop or utc(reg['sealed_at'])>=start:raise PermissionError('phase clock/registration invalid')
    caps={'http':0,'futu_api':4} if phase=='probe' else {'http':24,'futu_api':15,'allocator_calls':9,'recovery_calls':9,'ledgers':12}
    if permit.get('budgets')!=caps:raise PermissionError('phase budget differs from fixed plan')
    if phase=='pilot' and reg.get('source_role_mapping_status')!='APPROVED_ROLE_MAPPING':raise PermissionError('source role mapping pending')
    return permit

@contextmanager
def network_denied():
    attempts=[]
    def deny(*args,**kwargs):
        attempts.append('network_attempt_blocked');raise PermissionError('implementation is offline: network disabled')
    with patch.object(socket.socket,'connect',deny),patch.object(socket.socket,'connect_ex',deny),patch('requests.sessions.Session.request',deny):
        yield attempts

def sanitize_global_state(payload):
    allowed={key:payload[key] for key in ['market_us','qot_logined','server_ver','open_d_version'] if key in payload}
    if 'qot_logined' in allowed:allowed['qot_logined']=bool(allowed['qot_logined'])
    return allowed

def validate_receipt(receipt):
    ordered(receipt['request_started'],receipt['received'])
    if len(receipt.get('payload_sha256',''))!=64:raise ValueError('missing payload identity')
    if not receipt.get('source_identity'):raise ValueError('missing source identity')

def validate_quote(receipt,*,seal,slot_at):
    validate_receipt(receipt);ordered(seal,receipt['request_started'],strict=True)
    start=utc(slot_at);received,_=check_clock(receipt['received']);requested,_=check_clock(receipt['request_started'])
    if requested<start or received>start+pd.Timedelta(seconds=15):raise ValueError('quote outside frozen slot')
    if set(receipt['quotes'])!={'SOXL','SOXX'}:raise ValueError('incomplete quote pair')
    prices=[]
    for symbol in ['SOXL','SOXX']:
        row=receipt['quotes'][symbol];price=float(row['price']);stamp=utc(row['quote_at'])
        if not np.isfinite(price) or price<=0 or stamp<start or stamp>received:raise ValueError('invalid/stale/future quote')
        prices.append(price)
    return np.array(prices,float)

def validate_bars(frame,receipt,*,session,cutoff,seal,mapping_evidence,synthetic=False):
    validate_receipt(receipt);ordered(receipt['received'],seal,strict=True)
    if not mapping_evidence.get('passed') or mapping_evidence.get('edge')!='start':raise ValueError('bar edge mapping unqualified')
    if not synthetic and mapping_evidence.get('scope')!='observed_probe':raise ValueError('fixture is not live bar qualification')
    for key in ['sdk_sha256','endpoint','query_mode']:
        if mapping_evidence.get(key)!=receipt.get(key) or not receipt.get(key):raise ValueError('bar endpoint/SDK/mode identity mismatch')
    end=utc(cutoff).tz_convert('America/New_York')
    if end.date()!=pd.Timestamp(session).date():raise ValueError('bar cutoff belongs to another session')
    received=utc(receipt['received']['utc']);decision=utc(seal['utc'])
    if end.hour==9 and end.minute==45:earliest=end+pd.Timedelta(seconds=5);deadline=end+pd.Timedelta(minutes=1)
    elif end.hour==10 and end.minute==30:earliest=end+pd.Timedelta(seconds=5);deadline=end+pd.Timedelta(seconds=90)
    else:raise ValueError('unsupported decision bar cutoff')
    if utc(receipt['request_started']['utc'])<earliest:raise ValueError('bar request started before closed interval delay')
    if not earliest<=received<deadline or not received<decision<=deadline:raise ValueError('bar receipt/decision deadline missed')
    opening=pd.Timestamp(session).tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=30)
    expected=pd.date_range(opening,end-pd.Timedelta(minutes=1),freq='min')
    used=frame.copy();times=pd.to_datetime(used.time_key)
    times=times.dt.tz_localize('America/New_York') if times.dt.tz is None else times.dt.tz_convert('America/New_York')
    used['bar_start']=times;used=used[(times>=opening)&(times+pd.Timedelta(minutes=1)<=end)].copy()
    if set(used.symbol)!= {'SOXL','SOXX'}:raise ValueError('missing bars symbol')
    for _,group in used.groupby('symbol'):
        if not pd.DatetimeIndex(group.bar_start).equals(expected):raise ValueError('bar gap/duplicate/order mismatch')
    v=used[['open','high','low','close','volume']].to_numpy(float)
    if not np.isfinite(v).all() or (v[:,:4]<=0).any() or (v[:,4]<0).any():raise ValueError('invalid OHLCV')
    if not (used.high>=used[['open','close']].max(axis=1)).all() or not (used.low<=used[['open','close']].min(axis=1)).all():raise ValueError('invalid OHLC range')
    if (used.bar_start+pd.Timedelta(minutes=1)).max()>utc(receipt['received']['utc']):raise ValueError('forming bar used')
    return used

def validate_live_open(receipt,*,session,previous_session,decision_seal,futu_opens):
    validate_receipt(receipt);ordered(receipt['received'],decision_seal,strict=True);received=utc(receipt['received']['utc']);returns=[]
    deadline=pd.Timestamp(session).tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=46)
    if received>=deadline or utc(decision_seal['utc'])>deadline:raise ValueError('live open missed09:46')
    if set(receipt['symbols'])!={'SOXL','SOXX'}:raise ValueError('incomplete daily-open pair')
    for symbol in ['SOXL','SOXX']:
        row=receipt['symbols'][symbol];stamp=utc(row['regular_market_time'])
        if not 0<=(received-stamp).total_seconds()<=120:raise ValueError('stale partial-day source')
        if pd.Timestamp(row['current_date']).date()!=pd.Timestamp(session).date():raise ValueError('wrong current open date')
        if pd.Timestamp(row['previous_date']).date()!=pd.Timestamp(previous_session).date():raise ValueError('previous open not previous XNYS session')
        if utc(row['current_bar_timestamp'])>received or utc(row['previous_bar_timestamp'])>received:raise ValueError('future raw bar timestamp')
        values=np.array([row[k] for k in ['current_open','current_close','current_adjclose','previous_open','previous_close','previous_adjclose']],float)
        if not np.isfinite(values).all() or (values<=0).any():raise ValueError('invalid same-payload opening basis')
        if abs(values[0]-float(futu_opens[symbol]))>.01:raise ValueError('opening sources disagree')
        current=values[0]*values[2]/values[1];previous=values[3]*values[5]/values[4];returns.append(current/previous-1)
    return np.array(returns)

def validate_source_roles(role_approval,*,required_roles,soxx_volume,expected_dates):
    if role_approval.get('status')!='APPROVED_ROLE_MAPPING' or role_approval.get('plan_sha256')!=PLAN_SHA:raise ValueError('source role/volume mapping pending')
    if not set(required_roles)<=set(role_approval.get('roles',{})):raise ValueError('unqualified input role')
    if not role_approval.get('SOXX_volume_basis'):raise ValueError('SOXX volume basis unqualified')
    if not soxx_volume.index.equals(expected_dates) or not np.isfinite(soxx_volume).all() or (soxx_volume<0).any():raise ValueError('missing actual SOXX volume')
    # Zero-volume observations are accepted only as actual received values, never generated fills.
    if role_approval.get('volume_filled',True):raise ValueError('fabricated/fill volume forbidden')

def validate_daily_bundle(bundle,*,seal,role_approval,expected_dates,required_roles,session):
    if role_approval.get('status')!='APPROVED_ROLE_MAPPING' or role_approval.get('plan_sha256')!=PLAN_SHA:raise ValueError('daily source mapping pending')
    roles=bundle.get('roles',{})
    if set(roles)!=set(required_roles) or not set(required_roles)<=set(role_approval.get('roles',{})):raise ValueError('missing actual role frames')
    day=pd.Timestamp(session).tz_localize('America/New_York');deadline=day+pd.Timedelta(hours=9,minutes=20);proposal_deadline=day+pd.Timedelta(hours=9,minutes=29)
    if utc(seal['utc'])>proposal_deadline:raise ValueError('raw proposal missed09:29')
    for role in required_roles:
        item=roles[role];receipt=item['receipt'];validate_receipt(receipt);ordered(receipt['received'],seal,strict=True)
        if utc(receipt['received']['utc'])>deadline:raise ValueError('daily receipt missed09:20')
        mapping=role_approval['roles'][role]
        if receipt['source_identity']!=mapping['source_identity'] or item['field_basis']!=mapping['field_basis']:raise ValueError('unapproved field source/basis')
        frame=item['frame']
        if not isinstance(frame,pd.DataFrame) or not frame.index.equals(expected_dates):raise ValueError('actual source calendar mismatch')
        required=mapping['required_fields']
        if not set(required)<=set(frame) or not np.isfinite(frame[required].to_numpy(float)).all():raise ValueError('missing source field values')
        if item.get('filled_rows',0):raise ValueError('filled input not qualified')
        if item.get('used_rows_sha256')!=digest(frame[required]):raise ValueError('used-row hash mismatch')
        if role=='SOXX_daily_volume':validate_source_roles(role_approval,required_roles=required_roles,soxx_volume=frame[mapping['volume_field']],expected_dates=expected_dates)
    for key,count in [('primary',39),('common',16)]:
        values=np.asarray(bundle[key],float)
        if values.shape!=(count,) or not np.isfinite(values).all():raise ValueError('invalid '+key+' features')
    if bundle.get('known_unknown_fundamental',False):raise ValueError('reference update pending; no stale carry')
    return digest(bundle)


def validate_close_bars(frame,receipt,*,session,session_close,commit_clock,mapping_evidence):
    validate_receipt(receipt);ordered(receipt['received'],commit_clock,strict=True)
    end=utc(session_close).tz_convert('America/New_York');day=pd.Timestamp(session).date()
    if end.date()!=day or (end.hour,end.minute) not in [(16,0),(13,0)]:raise ValueError('invalid declared XNYS close')
    if not mapping_evidence.get('passed') or mapping_evidence.get('edge')!='start' or mapping_evidence.get('scope')!='observed_probe':raise ValueError('bar mapping unqualified')
    for key in ['sdk_sha256','endpoint','query_mode']:
        if not receipt.get(key) or receipt[key]!=mapping_evidence.get(key):raise ValueError('close bar endpoint mismatch')
    received=utc(receipt['received']['utc'])
    if utc(receipt['request_started']['utc'])<end+pd.Timedelta(seconds=5):raise ValueError('close request started before closure delay')
    if not end+pd.Timedelta(seconds=5)<=received<=end+pd.Timedelta(seconds=120):raise ValueError('close receipt deadline')
    start=pd.Timestamp(session).tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=30);expected=pd.date_range(start,end-pd.Timedelta(minutes=1),freq='min')
    used=frame.copy();times=pd.to_datetime(used.time_key);used['bar_start']=times.dt.tz_localize('America/New_York') if times.dt.tz is None else times.dt.tz_convert('America/New_York')
    if set(used.symbol)!= {'SOXL','SOXX'}:raise ValueError('close pair absent')
    for _,part in used.groupby('symbol'):
        if not pd.DatetimeIndex(part.bar_start).equals(expected):raise ValueError('close session coverage')
    values=used[['open','high','low','close','volume']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:,:4]<=0).any() or (values[:,4]<0).any():raise ValueError('invalid close OHLCV')
    if not used.high.ge(used[['open','close']].max(axis=1)).all() or not used.low.le(used[['open','close']].min(axis=1)).all():raise ValueError('close OHLC range')
    return used
