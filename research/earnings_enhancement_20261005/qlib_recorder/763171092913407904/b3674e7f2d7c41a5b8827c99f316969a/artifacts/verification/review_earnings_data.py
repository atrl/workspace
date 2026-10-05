from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,qlib
from qlib.data import D
root=Path.cwd(); card=root/'research/earnings_enhancement_20261005'
source=root/'data/snapshots/earnings_surprise_20261005_v1'
provider=root/'data/qlib/earnings_surprise_20261005_v1'
qlib.init(provider_uri=str(provider),region='us',kernels=1)
cal=pd.DatetimeIndex(D.calendar(freq='day'))
fields=['$earnings_surprise_'+x.lower() for x in ('NVDA','AMD','INTC')]
raw=D.features(['SOXX'],fields,start_time=cal[0],end_time=cal[-1],freq='day').xs('SOXX',level='instrument').reindex(cal)
checks=[]
for sym,field in zip(('NVDA','AMD','INTC'),fields):
 path=source/f'alpha_{sym.lower()}_earnings.json'
 records=sorted(json.loads(path.read_text())['quarterlyEarnings'],key=lambda x:x['reportedDate'])
 expected=[];prior_report=[]
 for day in cal:
  eligible=[x for x in records if pd.Timestamp(x['reportedDate']) < day]
  row=eligible[-1]
  a,b=float(row['reportedEPS']),float(row['estimatedEPS'])
  value=0. if a==b==0 else (a-b)/(abs(a)+abs(b))
  expected.append(value);prior_report.append(row['reportedDate'])
 expected=np.array(expected,dtype=np.float32)
 actual=raw[field].to_numpy()
 assert np.array_equal(actual,expected,equal_nan=True),sym
 checks.append({'symbol':sym,'rows':len(cal),'all_cells_equal_independent_raw_reconstruction':True,'max_absolute_error':float(np.max(np.abs(actual-expected))),'reports_used':len(set(prior_report)),'all_report_dates_strictly_before_signal':all(pd.Timestamp(x)<y for x,y in zip(prior_report,cal)),'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
result={'scope':'Independent raw JSON to latest strictly prior report per Qlib session; no model fits or financial metrics.','calendar_start':str(cal[0].date()),'calendar_end':str(cal[-1].date()),'checks':checks,'strict_pit_claim':False,'passed':True}
(card/'independent_data_review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
