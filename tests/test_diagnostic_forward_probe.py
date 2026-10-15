import json
import pytest
import pandas as pd
from src.data.diagnostic_forward_probe import collect_four,completed_session,global_allowlist,snapshot_clocks

def test_exact_four_calls_no_retry_pagination_and_global_sanitization():
    session,expected=completed_session('2026-10-15 17:53:00+00:00');assert session=='2026-10-14'
    events=[]
    class Fake:
        def __init__(self):self.calls=[]
        def get_global_state(self):self.calls.append('global');return 0,{'market_us':'AFTERNOON','qot_logined':'0','user_id':'SECRET','token':'SECRET'}
        def get_market_snapshot(self,codes):
            self.calls.append(('snapshot',codes));return 0,pd.DataFrame({'code':codes,'last_price':[70.,400.],'update_time':['2026-10-15 13:53:00']*2})
        def request_history_kline(self,code,**kwargs):
            self.calls.append(('history',code,kwargs));assert kwargs['start']==kwargs['end']==session and kwargs['session']=='RTH' and kwargs['page_req_key'] is None and kwargs['max_count']==1000
            frame=pd.DataFrame({'code':code,'time_key':expected.tz_localize(None).astype(str),'open':100.,'high':101.,'low':99.,'close':100.,'volume':10.})
            # The second series reports a page: fail qualification, do NOT request it.
            return 0,frame,(b'no-follow' if code=='US.SOXX' else None)
    fake=Fake();constants={'RET_OK':0,'ktype':'K_1M','autype':'NONE','fields':['ALL'],'session':'RTH'}
    out=collect_four(fake,session,expected,constants,lambda stage,key,receipt,payload:events.append((stage,key,receipt,payload)))
    assert len(fake.calls)==out['application_calls']==4 and not out['source_gate_passed'] and len(out['bar_checks'])==1
    assert [x[0] for x in events]==['started','completed']*4
    global_event=events[1];assert global_event[3]=={'market_us':'AFTERNOON','qot_logined':False}
    assert 'SECRET' not in json.dumps(global_event[:3]) and 'SECRET' not in json.dumps(global_event[3])
    assert isinstance(events[-1][3],pd.DataFrame) # retain failed history evidence
    assert global_allowlist({'qot_logined':'1','account_id':123})=={'qot_logined':True}

    future=pd.DataFrame({"code":["US.SOXL","US.SOXX"],"last_price":[1.,2.],"update_time":["2026-10-15 14:01:00"]*2})
    with pytest.raises(ValueError,match="future"):
        snapshot_clocks(future,{"utc":"2026-10-15T18:00:00+00:00","monotonic_ns":1})

    # Exercise the REAL SDK public while-loop, with a fake processor and no context.
    from futu import OpenQuoteContext,RET_OK,RET_ERROR,KLType,AuType,KL_FIELD,Session
    from src.data.diagnostic_forward_probe import SingleQueryGuard
    sent=[];protocol=[];guard=SingleQueryGuard(session,RET_OK,RET_ERROR,lambda *item:protocol.append(item))
    class SDKShim:
        request_history_kline=OpenQuoteContext.request_history_kline
        def get_sync_conn_id(self):return 0
        def _get_sync_query_processor(self,pack,unpack):
            def first_page(**kwargs):
                sent.append(kwargs['code']);return RET_OK,'',([{'code':kwargs['code'],'time_key':str(expected[0].tz_localize(None)),'open':100.,'high':101.,'low':99.,'close':100.,'volume':10.}],True,b'cursor')
            return guard.wrap(pack,first_page)
    status,message,page=SDKShim().request_history_kline('US.SOXL',start=session,end=session,ktype=KLType.K_1M,autype=AuType.NONE,fields=[KL_FIELD.ALL],max_count=1000,page_req_key=None,extended_time=False,session=Session.RTH)
    assert status==RET_ERROR and sent==['US.SOXL'] and len(guard.admitted)==1 and len(guard.blocked)==1
    assert len(guard.first_pages['US.SOXL']['rows'])==1
