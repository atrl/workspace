from datetime import datetime
import pandas as pd
from src.data.flow_receipts import capture_flow


def test_receipt_rejects_missed_late_and_prior_session_and_never_replaces(tmp_path):
    slot={'name':'open_a','session_date':'2026-10-14','request_at_utc':'2026-10-14T13:45:10Z','deadline_utc':'2026-10-14T13:46:00Z'}
    class Context:
        calls=0
        def get_capital_flow(self,*args,**kwargs):
            self.calls+=1
            return 0,pd.DataFrame({'capital_flow_item_time':['2026-10-14 09:45:00'],'last_valid_time':['2026-10-14 09:45:11'],'in_flow':[1.]})
    def clocks(*values):
        it=iter(pd.Timestamp(v).to_pydatetime() for v in values)
        return lambda:next(it)
    c=Context()
    missed=capture_flow(c,symbol='SOXX',slot=slot,output_dir=tmp_path/'missed',clock=clocks('2026-10-14T13:46:00Z'))
    assert missed['status']=='missed_not_requested' and c.calls==0
    late=capture_flow(c,symbol='SOXX',slot=slot,output_dir=tmp_path/'late',clock=clocks('2026-10-14T13:45:10Z','2026-10-14T13:46:01Z'))
    assert not late['qualified'] and 'late_response' in late['status']
    timely=capture_flow(c,symbol='SOXX',slot=slot,output_dir=tmp_path/'timely',clock=clocks('2026-10-14T13:45:10Z','2026-10-14T13:45:12Z'))
    assert timely['qualified']
    before=c.calls
    assert capture_flow(c,symbol='SOXX',slot=slot,output_dir=tmp_path/'timely',clock=lambda:None)==timely and c.calls==before
    class Stale(Context):
        def get_capital_flow(self,*args,**kwargs):
            return 0,pd.DataFrame({'capital_flow_item_time':['2026-10-13 16:00:00'],'last_valid_time':['2026-10-14 09:45:11'],'in_flow':[1.]})
    stale=capture_flow(Stale(),symbol='SOXX',slot=slot,output_dir=tmp_path/'stale',clock=clocks('2026-10-14T13:45:10Z','2026-10-14T13:45:12Z'))
    assert not stale['qualified'] and 'wrong_session' in stale['status']
