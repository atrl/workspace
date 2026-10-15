"""Fixed saved-source end-label qualification; no market transport."""
from pathlib import Path
from hashlib import sha256
import json,os,pickle
import numpy as np
import pandas as pd
from src.data.minute_intake import atomic_json
from src.data.diagnostic_forward_inputs import network_denied
from src.data.qlib_native import build_qlib_store_from_frame
NAME='diagnostic_forward_bar_mapping_20261016'
PLAN_SHA='00b8b8a366046e9765e2c0950cccc255af78b27e66a486d285d3b00c9339ffcf'
FIELDS=['open','high','low','close','volume']
def frame_at(path):
    frame=pd.read_parquet(path)
    times=pd.to_datetime(frame['time_key'])
    if times.dt.tz is not None:times=times.dt.tz_convert('America/New_York').dt.tz_localize(None)
    return frame.set_index(pd.DatetimeIndex(times,name='datetime'))
def run(root):
    root=Path(root).resolve();card=root/'research'/NAME;plan=json.loads((card/'plan.json').read_text())
    assert sha256((card/'plan.json').read_bytes()).hexdigest()==PLAN_SHA
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    if (card/'execution_attempt.json').exists():raise RuntimeError('no silent qualification retry')
    for item in plan['old_files']+plan['new_payload_objects']:
        assert sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    assert sha256((root/plan['old_lineage_manifest']).read_bytes()).hexdigest()==plan['old_lineage_manifest_sha256']
    for path,expected in plan['source_code_sha256'].items():assert sha256((root/path).read_bytes()).hexdigest()==expected
    os.environ['MLFLOW_ALLOW_FILE_STORE']='true';os.environ['MLFLOW_DISABLE_TELEMETRY']='true';os.environ['DO_NOT_TRACK']='true'
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    qlib.init(provider_uri=str(root/'data/qlib/soxl'),region='us',kernels=1)
    atomic_json({'status':'started','plan_sha256':PLAN_SHA,'network':0},card/'execution_attempt.json')
    with network_denied() as denied:
        with R.start(experiment_name=NAME,uri=str(card/'qlib_recorder')):
            rec=R.get_recorder();rec.save_objects(artifact_path='sources',plan=plan,workflow=Path(__file__).read_text())
            checks=[];parts=[];failure=None
            try:
                import exchange_calendars as xc
                cal=xc.get_calendar('XNYS',start='2026-07-29',end='2026-10-14')
                def grid(day,freq):
                    start=cal.session_open(day).tz_convert('America/New_York').tz_localize(None);end=cal.session_close(day).tz_convert('America/New_York').tz_localize(None)
                    return pd.date_range(start+pd.Timedelta(freq),end,freq=freq,name='datetime')
                oldbase=root/'data/raw/external/stage1_intraday_gap_repair_20261009_v1'
                for symbol in ['SOXL','SOXX']:
                    minute=frame_at(oldbase/(symbol+'_2026-07-29_1m.parquet'));direct=frame_at(oldbase/(symbol+'_2026-07-29_15m.parquet'))
                    pd.testing.assert_index_equal(minute.index,grid('2026-07-29','1min'));pd.testing.assert_index_equal(direct.index,grid('2026-07-29','15min'))
                    grouped=minute[FIELDS].resample('15min',closed='right',label='right').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
                    pd.testing.assert_index_equal(grouped.index,direct.index)
                    delta=grouped-direct[FIELDS];ok=np.isclose(grouped.to_numpy(float),direct[FIELDS].to_numpy(float),rtol=0,atol=1e-7)
                    rec.save_objects(artifact_path='old_'+symbol,minute=minute,direct=direct,aggregate=grouped,difference=delta,passed=pd.DataFrame(ok,index=direct.index,columns=FIELDS))
                    checks.append({'symbol':symbol,'windows':len(direct),'passed':bool(ok.all()),'max_absolute_difference':{k:float(delta[k].abs().max()) for k in FIELDS}})
                if not all(x['passed'] for x in checks):raise ValueError('fixed_aggregation_mismatch')
                for item in plan['new_payload_objects']:
                    obj=pickle.loads((root/item['path']).read_bytes());frame=obj['payload'].copy();symbol=str(frame.code.iloc[0]).removeprefix('US.');end=pd.DatetimeIndex(pd.to_datetime(frame.time_key),name='datetime')
                    pd.testing.assert_index_equal(end,grid('2026-10-14','1min'));values=frame[FIELDS].to_numpy(float)
                    assert np.isfinite(values).all() and (values[:,:4]>0).all() and (values[:,4]>=0).all()
                    assert frame.high.ge(frame[['open','close']].max(axis=1)).all() and frame.low.le(frame[['open','close']].min(axis=1)).all()
                    rec.save_objects(artifact_path='new_'+symbol,original=obj,interval_start=end-pd.Timedelta(minutes=1),interval_end=end)
                    part=frame[FIELDS].copy();part['datetime']=end;part['instrument']=symbol;parts.append(part)
                source=pd.concat(parts,ignore_index=True);assert len(source)==780
                provider=root/'data/qlib'/NAME
                manifest=build_qlib_store_from_frame(source,provider_uri=provider,freq='1min',region='us',as_of='2026-10-14',field_map={x:x for x in FIELDS},source='Saved Futu NONE payload; declared end-label bounded corroboration, no live finality qualification')
                qlib.init(provider_uri=str(provider),region='us',kernels=1);calendar=pd.DatetimeIndex(D.calendar(start_time='2026-10-14 09:31:00',end_time='2026-10-14 16:00:00',freq='1min'),name='datetime');pd.testing.assert_index_equal(calendar,grid('2026-10-14','1min'))
                readback=D.features(['SOXL','SOXX'],['$'+x for x in FIELDS],start_time=calendar[0],end_time=calendar[-1],freq='1min');quant=[]
                for symbol in ['SOXL','SOXX']:
                    raw=source[source.instrument.eq(symbol)].set_index('datetime')[FIELDS];got=readback.xs(symbol,level='instrument');pd.testing.assert_index_equal(got.index.rename('datetime'),raw.index)
                    np.testing.assert_array_equal(got.to_numpy(float),raw.to_numpy(dtype=np.float32).astype(float));quant.append({'symbol':symbol,'max_absolute_float32_quantization':float(np.abs(got.to_numpy(float)-raw.to_numpy(float)).max())})
                rec.save_objects(artifact_path='native',source=source,readback=readback,calendar=calendar,manifest=manifest,quantization=quant);atomic_json(manifest,card/'provider_manifest.json')
            except Exception as error:failure={'type':type(error).__name__,'detail':str(error)}
            summary={'experiment':NAME,'plan_sha256':PLAN_SHA,'passed':failure is None,'failure':failure,'comparison':checks,'network_attempts':len(denied),'model_calls':0,'fits':0,'strategy_replays':0,'forward_days':0,'pilot_ready':False,'qualified_gate':'bounded_saved_completed_history_end_label_and_Qlib' if failure is None else None,'rows':780 if failure is None else None,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_local_dir':str(card/'qlib_recorder'/str(rec.experiment_id)/rec.id),'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['sources/*','old_*/*','new_*/*','native/*','analysis/*']}}
            rec.save_objects(artifact_path='analysis',summary=summary);rec.log_metrics(mapping_passed=int(failure is None),registered_artifact_count=1)
    artifacts=Path(summary['recorder_local_dir'])/'artifacts';atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json');atomic_json(summary,card/'summary.json');atomic_json({'status':'finished','passed':summary['passed'],'recorder_id':rec.id},card/'execution_attempt.json');return summary
