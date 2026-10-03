from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import qlib
from qlib.workflow import R

root=Path('/Users/rziiiii/dev/workspace-worktrees/frozen-cash-information-20261003')
card=root/'research/frozen_cash_information_20261003'
ref=json.loads((card/'recorder_reference.json').read_text())
assert ref['complete'] is True and ref['recorder_id']=='ef3da36b5e6a4f4297c2bf76a6a19afd'
assert ref['plan_sha256']==hashlib.sha256((card/'diagnostic_plan.json').read_bytes()).hexdigest()
qlib.init(provider_uri='/Users/rziiiii/dev/workspace/data/qlib/published_futu_affine_20260925_v1',region='us',kernels=1)
with R.uri_context(ref['recorder_uri']):
    rec=R.get_recorder(experiment_name='frozen_cash_information_20261003',recorder_id=ref['recorder_id'])
    assert rec.client.get_run(rec.id).info.status=='FINISHED'
    def load(name):
        path=Path(rec.get_local_dir())/'artifacts'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['artifacts'][name]['sha256']
        return rec.load_object(name)
    rows=load('inputs/diagnostic_rows.pkl')
    draw_bundle=load('analysis/bootstrap_draws.pkl')[('development_2016_2022',20)]
    tables=load('analysis/tables.pkl')
sel=rows.loc[rows.index.year.isin(range(2016,2023)) & rows.score_available]
n=len(sel)
assert draw_bundle['mean_block']==20 and draw_bundle['resamples']==2000 and draw_bundle['seed']==20261003
# Reconstruct only draw0 while preserving the generator's registered 2000-stream RNG consumption.
rng=np.random.default_rng(20261003)
idx=np.empty(n,dtype=np.int32);idx[0]=rng.integers(0,n,size=2000)[0]
for t in range(1,n):
    restart=rng.random(2000)[0] < 1/20
    fresh=rng.integers(0,n,size=2000)[0]
    idx[t]=fresh if restart else (idx[t-1]+1)%n
p=sel.cash_advantage_prediction.to_numpy(dtype=float)[idx]
b=sel.prior_mean_prediction.to_numpy(dtype=float)[idx]
y=sel.cash_advantage_1d.to_numpy(dtype=float)[idx]
c=25/10000;hurdle=np.log1p(c)-np.log1p(-c);mask=p>hurdle;net=y-hurdle
expected={'paired_MSE_improvement':float(np.mean((b-y)**2-(p-y)**2)),
 'conditional_event_net_mean_bp_25':float(10000*np.mean(net[mask])) if mask.any() else np.nan,
 'masked_event_net_mean_bp_per_observation_25':float(10000*np.mean(np.where(mask,net,0.)))}
errors={}
for key,val in expected.items():
    stored=float(draw_bundle['statistics'].iloc[0][key])
    assert np.isclose(val,stored,rtol=1e-12,atol=1e-14,equal_nan=True),(key,val,stored)
    errors[key]=None if np.isnan(val) else abs(val-stored)
# Only the same three registered CSV summary rows, compared to their saved Recorder table.
csv=pd.read_csv(card/'bootstrap_summary.csv',float_precision='round_trip')
saved=tables['bootstrap_summary.csv']
filter_rows=lambda table:table.loc[(table.window=='development_2016_2022') & table.mean_block.eq(20) & table.statistic.isin(expected)].set_index('statistic').sort_index()
a,b=filter_rows(csv),filter_rows(saved)
assert len(a)==len(b)==3
pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-14)
report={'status':'PASS','recorder_id':ref['recorder_id'],'scope':'ONE fixed development/block20/draw0 statistical spotcheck; three corresponding registered CSV summary rows matched Recorder. No new bootstrap statistics or portfolio paths.','rows':n,'seed':20261003,'mean_block':20,'draw':0,'absolute_errors':errors,'csv_rows_matched':3,'artifact_hashes_verified':3,'source_plan_unchanged':True,'financial_qualification_claim':False}
Path('/tmp/frozen-cash-final-spotcheck.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
