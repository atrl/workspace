"""One-shot read-only flow receipts; never backdate or replace missed decisions."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import pandas as pd
from src.data.minute_intake import atomic_json, atomic_parquet


def capture_flow(context, *, symbol, slot, output_dir, clock=None):
    clock = clock or (lambda: datetime.now(timezone.utc))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {'symbol': symbol, 'slot': slot}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    receipt_path = output_dir / (slot['name'] + '_' + symbol + '.json')
    data_path = receipt_path.with_suffix('.parquet')
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if saved['request_sha256'] != digest:
            raise ValueError('immutable slot identity changed')
        if saved.get('sha256') and hashlib.sha256(data_path.read_bytes()).hexdigest() != saved['sha256']:
            raise ValueError('immutable snapshot bytes changed')
        return saved
    now = pd.Timestamp(clock()).tz_convert('UTC')
    planned = pd.Timestamp(slot['request_at_utc'])
    deadline = pd.Timestamp(slot['deadline_utc'])
    if now < planned:
        raise ValueError('slot not due; caller must wait')
    receipt = dict(identity, request_sha256=digest, started_at_utc=now.isoformat(),
                   received_at_utc=None, qualified=False, read_only=True,
                   provider_timezone_assumption='America/New_York',
                   qualification_scope='receipt and timestamp checks only; not closed-bar, prediction or production certification')
    if now >= deadline:
        receipt['status'] = 'missed_not_requested'
        atomic_json(receipt, receipt_path)
        return receipt
    try:
        ret, data = context.get_capital_flow('US.' + symbol, period_type='INTRADAY')
    except Exception as exc:
        receipt.update(received_at_utc=pd.Timestamp(clock()).tz_convert('UTC').isoformat(),
                       status='transport_exception', error_type=type(exc).__name__)
        atomic_json(receipt, receipt_path)
        return receipt
    received = pd.Timestamp(clock()).tz_convert('UTC')
    receipt.update(ret_code=int(ret), received_at_utc=received.isoformat(),
                   elapsed_seconds=(received-now).total_seconds())
    reasons = []
    if received < now:
        reasons.append('clock_reversed')
    if received >= deadline:
        reasons.append('late_response')
    if ret != 0:
        receipt['error'] = str(data)[:500]
        reasons.append('api_error')
    else:
        atomic_parquet(data, data_path)
        receipt.update(path=data_path.name, sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(), rows=len(data))
        if data.empty:
            reasons.append('empty')
        else:
            stamps = pd.to_datetime(data.capital_flow_item_time, errors='coerce')
            valid = pd.to_datetime(data.last_valid_time, errors='coerce')
            local_received = received.tz_convert('America/New_York').tz_localize(None)
            expected = pd.Timestamp(slot['session_date'])
            receipt.update(data_first=str(stamps.min()), data_last=str(stamps.max()),
                           provider_last_valid_time=str(valid.max()))
            if stamps.isna().any() or valid.isna().any():reasons.append('unknown_provider_clock')
            if not stamps.dt.normalize().eq(expected).all():reasons.append('wrong_session')
            if stamps.duplicated().any():reasons.append('duplicate_source_times')
            if stamps.gt(local_received).any() or valid.gt(local_received).any():reasons.append('future_provider_clock')
            if not valid.dt.normalize().eq(expected).all() or valid.lt(expected+pd.Timedelta(hours=9,minutes=30)).any():reasons.append('provider_not_current_session')
    receipt.update(status='received_current_before_deadline' if not reasons else '|'.join(reasons), qualified=not reasons)
    atomic_json(receipt, receipt_path)
    return receipt
