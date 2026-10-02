import os
os.environ['MLFLOW_ALLOW_FILE_STORE']='true'
os.environ['MLFLOW_DISABLE_AGENT_HINT']='1'
from pathlib import Path
import json, hashlib
import numpy as np,pandas as pd,qlib
from qlib.workflow import R
from src.model.convex_allocation import DPPConvexAllocationSolver
from src.workflow.stage1_economic_action import fee_affordable_target
from src.backtest.position_accounting import settle_target_session
root=Path('/Users/rziiiii/dev/workspace-worktrees/conditional-variance-20261002')
s=json.loads((root/'research/conditional_variance_20261002/summary.json').read_text())
assert s['recorder_id']=='9a447c33dc934baeb1d4c3f2b9bf2122'
qlib.init(provider_uri=str(root/s['provider_uri']),region='us');R.set_uri(s['recorder_uri'])
r=R.get_recorder(experiment_id='607243708770492389',recorder_id=s['recorder_id'])
path=r.load_object('analysis/portfolio_replays.pkl');daily=r.load_object('analysis/daily_solver_audits.pkl');development=r.load_object('policy/development_inputs.pkl');outer=r.load_object('policy/diagnostic_inputs.pkl')
all_rows=[];targets=[];failures=[]
def score(w,h,mu,cost):
 ex=fee_affordable_target(w,pretrade_weights=h,cost_bps=cost)
 settled=settle_target_session(h,open_target_weights=ex,asset_returns=[0.,0.,0.],cost_bps=cost)
 fee=settled['fees'];n=1-fee;g=float(w[:2]@mu)
 return {'target':w.tolist(),'gross_expected_h5_growth':g,'fee':fee,'fee_terminal_wealth_drag':fee*(1+g),'net_expected_h5_growth':n*(1+g)-1,'net_invested_expected_growth_component':n*g,'turnover':settled['turnover']}
def compare(chosen,alt):
 # Flat extrapolation diagnostic only: mu(h)=h/5*mu_H5, fixed weights and one upfront fee.
 gross=chosen['gross_expected_h5_growth']-alt['gross_expected_h5_growth'];fee=alt['fee_terminal_wealth_drag']-chosen['fee_terminal_wealth_drag'];net=chosen['net_expected_h5_growth']-alt['net_expected_h5_growth']
 assert abs(gross+fee-net)<1e-12
 extra_fee=alt['fee']-chosen['fee'];growth_adv=alt['net_invested_expected_growth_component']-chosen['net_invested_expected_growth_component']
 horizon=5*extra_fee/growth_adv if extra_fee>0 and growth_adv>0 else None
 return {'chosen_minus_alternative_gross_h5_bps':gross*1e4,'chosen_fee_advantage_terminal_bps':fee*1e4,'chosen_minus_alternative_net_h5_bps':net*1e4,'alternative_extra_upfront_fee_bps':extra_fee*1e4,'alternative_growth_component_advantage_h5_bps':growth_adv*1e4,'linear_frozen_moment_break_even_sessions':horizon}
for variant in development.keys():
 repeat,name=variant
 if repeat!=1:continue
 inp=development[variant]
 for year in range(2016,2023):
  i=np.flatnonzero(inp['signal_dates'].year==year)[0]
  for cost in [10,25]:targets.append((name,'development',cost,inp,i,(1,name,cost)))
name=s['selected']['variant']+'_annual_refit';inp=outer[(1,'annual_refit')]
i=np.flatnonzero(inp['signal_dates'].year==2023)[0]
for cost in [10,25]:targets.append((name,'reused_diagnostic',cost,inp,i,('outer',1,'annual_refit',cost)))
for name,segment,cost,inp,i,dkey in targets:
 date=inp['signal_dates'][i];row=path[(path.repeat_id==1)&(path.variant==name)&(path.cost_bps==cost)&(path.signal_date==date)].iloc[0];details=daily[dkey][i]
 h=row[['pretrade_soxl_weight','pretrade_soxx_weight','pretrade_cash_weight']].to_numpy(float);w=row[['soxl_weight','soxx_weight','cash_weight']].to_numpy(float);mu=inp['means'][i];sigma=inp['covariance'][i];B=float(inp['budget'][i]);q=min(1,np.sqrt(B/sigma[1,1]));null=np.array([0.,q,1-q])
 chosen=score(w,h,mu,cost);scaled_soxx=score(null,h,mu,cost)
 assert abs(1+chosen['net_expected_h5_growth']-row.objective)<1e-12
 assert abs(1+scaled_soxx['net_expected_h5_growth']-details['nulls']['SCALED_SOXX']['objective'])<1e-12
 zero=DPPConvexAllocationSolver(cost_bps=0,budget_mode='external_positive').solve(h,mu,sigma,risk_budget_variance=B,robust_radius=0.,budget_provenance=inp['budget_provenance'][i])
 assert zero['optimality_verified'] and zero['status']=='solved'
 gross_optimal=score(zero['target_net_weights'],h,mu,cost)
 assert gross_optimal['gross_expected_h5_growth']+1e-8>=max(chosen['gross_expected_h5_growth'],scaled_soxx['gross_expected_h5_growth'])
 all_rows.append({'variant':name,'segment':segment,'signal_date':str(date.date()),'cost_bps':cost,'actual_holdings':h.tolist(),'means_H5':mu.tolist(),'covariance_daily':sigma.tolist(),'B':B,'soxx_gross_mu_over_sigma':float(mu[1]/np.sqrt(sigma[1,1])),'soxl_gross_mu_over_sigma':float(mu[0]/np.sqrt(sigma[0,0])),'chosen':chosen,'scaled_soxx':scaled_soxx,'cost_zero_gross_optimal_evaluated_at_actual_cost':gross_optimal,'chosen_vs_scaled_soxx':compare(chosen,scaled_soxx),'chosen_vs_cost_zero_gross_optimal':compare(chosen,gross_optimal),'cost_zero_certificate':zero['solver_audit']['numerical_certificate'],'saved_solver_certificate':details['solver']['numerical_certificate']})
result={'source_recorder':s['recorder_id'],'scope':'Same saved one-step holdings, frozen mu_H5, daily covariance and B. No new return paths; no realized returns used. Cost-zero solve is gross-objective diagnosis only.','fee_accounting':'True sum of absolute risky traded NAV fractions, cash-funded. Net expected wealth=(1-fee)*(1+w_risky dot mu_H5).','break_even_scope':'Algebraically extrapolate the same mu_H5 linearly with h/5, hold both compared targets and their upfront fees fixed, ignore all future trades and changes in risk/moments. Not a forecast, not an optimized horizon, not evidence of long-run superiority.','source_hashes':s['source_sha256'],'decisions':all_rows,'selected_points':len(all_rows)}
Path('/tmp/conditional-variance-one-step-fee-diagnosis.json').write_text(json.dumps(result,indent=2))
summary=[]
for a in all_rows:
 b=a['chosen_vs_scaled_soxx'];z=a['chosen_vs_cost_zero_gross_optimal'];summary.append({'variant':a['variant'],'signal_date':a['signal_date'],'cost_bps':a['cost_bps'],'held_SOXL':a['actual_holdings'][0],'chosen_SOXL':a['chosen']['target'][0],'chosen_SOXX':a['chosen']['target'][1],'scaled_SOXX':a['scaled_soxx']['target'][1],'zero_SOXL':a['cost_zero_gross_optimal_evaluated_at_actual_cost']['target'][0],'zero_SOXX':a['cost_zero_gross_optimal_evaluated_at_actual_cost']['target'][1],**b,'vs_zero_gross_h5_bps':z['chosen_minus_alternative_gross_h5_bps'],'vs_zero_net_h5_bps':z['chosen_minus_alternative_net_h5_bps'],'vs_zero_break_even_sessions':z['linear_frozen_moment_break_even_sessions']})
df=pd.DataFrame(summary);df.to_csv('/tmp/conditional-variance-one-step-fee-diagnosis.csv',index=False)
print('Audited fixed decisions:',len(df));print(df[df.variant.str.startswith('expanding')].to_string(index=False));print('\nCOUNTS:',json.dumps({'scaled_soxx_gross_better':int(df.chosen_minus_alternative_gross_h5_bps.lt(-1e-5).sum()),'scaled_soxx_net_worse':int(df.chosen_minus_alternative_net_h5_bps.gt(1e-5).sum()),'gross_optimum_gross_better':int(df.vs_zero_gross_h5_bps.lt(-1e-5).sum()),'gross_optimum_net_worse':int(df.vs_zero_net_h5_bps.gt(1e-5).sum())}))
