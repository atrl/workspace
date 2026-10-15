from pathlib import Path
import json,pickle,hashlib
import pytest
from src.data.diagnostic_forward_inputs import network_denied
from src.data.diagnostic_forward_io_v2 import HttpLedger,chart_params
from src.schedule.diagnostic_forward_transport_v2 import offline_fixture

def test_saved_parser_synthetic_full_io_and_durable_budget(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    with network_denied() as blocked:
        card=root/'research/diagnostic_forward_io_v2_20261016'
        if (card/'summary.json').exists():
            summary=json.loads((card/'summary.json').read_text());manifest=json.loads((card/'verification_manifest.json').read_text());item=next(x for x in manifest['entries'] if x['path'].endswith('offline_v2/fixture'));source=card/item['path'];assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256'];result=pickle.loads(source.read_bytes())['result']
        else:result=offline_fixture(root,tmp_path/'chain')
        assert result['committed_slots']==9 and result['books']==12
        assert result['actual_model_calls']==0 and result['mock_model_calls']==9
        assert result['synthetic_daily_requests']==14 and result['real_saved_Yahoo_SPY_rows']==3226
        ledger_dir=tmp_path/'http';ledger_dir.mkdir();params=chart_params('2010-03-11','2026-10-09');url='https://query2.finance.yahoo.com/v8/finance/chart/SOXL'
        (ledger_dir/'daily_yahoo_SOXL.receipt.json').write_text(json.dumps({'url':url,'params':params}))
        ledger=HttpLedger(ledger_dir,permit={'phase':'pilot','session':'2026-10-12','network_authorized':True,'budgets':{'http':17}})
        assert len(ledger.records)==1
        with pytest.raises(ValueError,match='attempted'):ledger.get('daily_yahoo_SOXL',url,params,deadline='2026-10-12T09:20:00-04:00')
        with pytest.raises(ValueError,match='unregistered'):ledger.get('new_source',url,params,deadline='2026-10-12T09:20:00-04:00')
        from src.data.diagnostic_forward_io_v2 import http_fence
        import requests
        with http_fence() as extra_http:
            with pytest.raises(PermissionError,match='fixed17'):requests.get('https://example.invalid/unregistered')
        assert len(extra_http)==1
        from src.schedule.diagnostic_forward_transport_v2 import validate_live_open_child,relevant_sec_form
        assert all(relevant_sec_form(f) for f in ('8-K','8-K/A','10-Q','10-Q/A','10-K','10-K/A'))
        assert not relevant_sec_form('4/A')
        own={'received':{'utc':'2026-10-19T13:45:10Z'}}
        row={'regular_market_time':'2026-10-19T13:45:11Z','current_bar_timestamp':'2026-10-19T13:30:00Z','previous_bar_timestamp':'2026-10-16T13:30:00Z'}
        with pytest.raises(ValueError,match='own-receipt'):validate_live_open_child(row,own)
        row['regular_market_time']='2026-10-19T13:45:09Z';row['current_bar_timestamp']='2026-10-19T13:45:11Z'
        with pytest.raises(ValueError,match='own-receipt'):validate_live_open_child(row,own)
        # A first-symbol future timestamp must not be rescued by the second response at13:45:20.
        row['current_bar_timestamp']='2026-10-19T13:30:00Z';validate_live_open_child(row,own)
        import src.schedule.diagnostic_forward_transport_v2 as module
        import qlib
        fake=tmp_path/'failed_setup';fc=fake/'research'/module.NAME;fc.mkdir(parents=True);(fc/'plan.json').write_text('{}');inbox=b'{}';(fc/'event_input_binding.json').write_bytes(inbox)
        config=fake/'runtime/soxl/stage1_regime_paper/config.json';config.parent.mkdir(parents=True);config.write_text('{"host":"127.0.0.1","port":11111}')
        registration={'event_binding_sha256':hashlib.sha256(inbox).hexdigest(),'endpoint_config_sha256':hashlib.sha256(config.read_bytes()).hexdigest()};(fc/'pilot_registration.json').write_text(json.dumps(registration))
        monkeypatch.setattr(module,'require_phase',lambda *a,**kw:{'session':'2026-10-19','registration_file':'pilot_registration.json','registration_sha256':'f'*64})
        initialized=[];monkeypatch.setattr(qlib,'init',lambda **kw:initialized.append(True))
        monkeypatch.setattr(module,'qualified_mapping',lambda *args:{'passed':True})
        def fail_load(*args):
            assert initialized==[True]
            raise FileNotFoundError('fixed local source absent')
        monkeypatch.setattr(module,'load_frozen',fail_load)
        terminal=module.run_pilot(fake)
        assert terminal['terminal_outcome']=='initialization_failed_before_sources' and terminal['context_attempts']==0 and terminal['http_attempts']==0
        assert (fake/'runtime/research'/module.NAME/('f'*16)/'terminal_status.json').exists()
    assert not blocked
