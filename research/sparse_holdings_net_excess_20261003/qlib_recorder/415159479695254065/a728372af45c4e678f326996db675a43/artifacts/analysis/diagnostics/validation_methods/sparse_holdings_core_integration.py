"""Independent synthetic core/workflow integration; no fitting, market data, or source edits."""
from pathlib import Path
import copy,hashlib,json,sys
import numpy as np,pandas as pd,torch
sys.path.insert(0,str(Path.cwd()/'tests'))
from test_benchmark_anchor_workflow import history as history_fixture
from src.model.sparse_holdings_net_excess import SparseHoldingsPolicy,sparsemax,sparsemax_numpy,embed_parent_policy,policy_to_dict,policy_from_dict,numpy_rollout
from src.model.benchmark_anchor_utility import BenchmarkAnchorPolicy,numpy_rollout as parent_rollout
from src.data.benchmark_anchor_utility import build_anchor_dataset
from src.workflow import sparse_holdings_net_excess as wf

torch.set_num_threads(1);DT=torch.float64;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=Path('src/model/sparse_holdings_net_excess.py');before=sha(source)
rng=np.random.default_rng(20261003)
# Projection and strict-support Jacobian, including onehot vertices.
projection_errors=[];jac_errors=[];fd_errors=[]
for row in rng.normal(size=(40,4))*2:
 x=torch.tensor(row,dtype=DT,requires_grad=True);q=sparsemax(x).detach().numpy();qn=sparsemax_numpy(row)
 projection_errors.append(float(np.max(abs(q-qn))))
 active=(q>0).astype(float);J=np.diag(active)-np.outer(active,active)/active.sum()
 actual=torch.autograd.functional.jacobian(sparsemax,x).numpy();jac_errors.append(float(np.max(abs(actual-J))))
 eps=1e-6;fd=np.column_stack([(sparsemax_numpy(row+eps*np.eye(4)[i])-sparsemax_numpy(row-eps*np.eye(4)[i]))/(2*eps) for i in range(4)])
 fd_errors.append(float(np.max(abs(fd-J))))
 assert (q>=0).all() and abs(q.sum()-1)<1e-12
 if np.count_nonzero(q)==1:assert np.array_equal(q,np.eye(4)[np.argmax(q)])
assert max(projection_errors)<1e-14 and max(jac_errors)<1e-14 and max(fd_errors)<1e-8
# All legal parent classes, including boundary vertices, embed pointwise.
x=rng.normal(size=(30,27));h=rng.dirichlet(np.ones(3),size=30);embedding_errors=[]
for i in range(20):
 old=BenchmarkAnchorPolicy(seed=(17,41,71)[i%3],anchor=True)
 theta=rng.random((3,9));theta*=rng.uniform(.05,1.)/theta.sum()
 if i<4:
  theta[:]=0
  if i:theta[i-1,0]=1
 with torch.no_grad():old.theta.copy_(torch.tensor(theta,dtype=DT))
 new=embed_parent_policy(old)
 assert torch.equal(new.hidden.weight,old.hidden.weight) and torch.equal(new.hidden.bias,old.hidden.bias)
 oldz=old(torch.tensor(x,dtype=DT),torch.tensor(h,dtype=DT)).detach().numpy()
 newz=new(torch.tensor(x,dtype=DT),torch.tensor(h,dtype=DT)).detach().numpy()
 embedding_errors.append(float(np.max(abs(oldz-newz))))
 assert embedding_errors[-1]<1e-14
# Fresh three-seed initial complete paths, exact encoder, bothcosts.
r=np.c_[.005+.02*np.sin(np.arange(90)),.001+.007*np.sin(np.arange(90))];features=rng.normal(size=(90,27));initial=[]
for seed in (17,41,71):
 old=BenchmarkAnchorPolicy(seed=seed);new=SparseHoldingsPolicy(seed=seed)
 assert sum(p.numel() for p in new.parameters())==272
 assert torch.equal(new.hidden.weight,old.hidden.weight) and torch.equal(new.hidden.bias,old.hidden.bias)
 for cost in (10,25):
  left=parent_rollout(old,features,r,cost_bps=cost);right=numpy_rollout(new,features,r,cost_bps=cost)
  errs={key:float(np.max(abs(left[key]-right[key]))) for key in ('target_weights','pretrade_weights','ending_weights','net_equity','fees','turnover')}
  assert max(errs.values())<1e-12;initial.append({'seed':seed,'cost_bps':cost,'errors':errs})
# True implementation in workflow; no APIstub/model fitter.
plan=json.loads(Path('research/sparse_holdings_net_excess_20261003/selection_plan.json').read_text());plan['training']['minimum_rows']=63
hist=history_fixture.__wrapped__();folds=wf.prepare_folds(build_anchor_dataset(hist,as_of='2017-01-10'),['price_holdings','price_vol_holdings'],[2015,2016],plan)
model=SparseHoldingsPolicy()
with torch.no_grad():model.A.zero_();model.K.zero_();model.a.copy_(torch.tensor([1.,0.,0.,0.],dtype=DT));model.K[3,0]=2.
payload=policy_to_dict(model)
def record(variant,year):
 f=folds[(variant,year)]
 return {'variant':variant,'parent_market_variant':f['parent_market_variant'],'model_year':year,'scaler':f['scaler'],
 'scaler_sha256':f['scaler']['scaler_sha256'],'seed_models':{s:copy.deepcopy(payload) for s in wf.SEEDS},
 'ensemble_models':[copy.deepcopy(payload) for _ in wf.SEEDS],'ensemble_audit':{'anchor_selected':False},
 'invalid_training':False,'numerical_failure':False}
frame=hist.loc['2015-12-29':'2016-01-07'];annual={year:record('price_holdings',year) for year in (2015,2016)};own=[]
for cost in (10,25):
 path=wf.replay_models(frame,annual,variant='price_holdings',cost_bps=cost,plan=plan)
 assert path.exact_hold_support.tolist()==[False]+[True]*(len(path)-1)
 assert path.fees.iloc[1:].eq(0).all() and path.turnover.iloc[1:].eq(0).all()
 assert np.array_equal(path[[f'pretrade_{a}_weight' for a in wf.ASSETS]].iloc[1:].to_numpy(),path[[f'ending_{a}_weight' for a in wf.ASSETS]].iloc[:-1].to_numpy())
 for seed in wf.SEEDS:
  pd.testing.assert_frame_equal(path.assign(ensemble_anchor_selected=False),wf.replay_models(frame,annual,variant='price_holdings',seed=seed,cost_bps=cost,plan=plan))
 own.append({'cost_bps':cost,**wf.independent_path_audit(frame,path,cost_bps=cost)})
# A bad future volume cannot rewrite earlier holdings; next decision sees forcedS book.
frame=hist.loc['2016-01-01':'2016-01-08'].copy();annual={2016:record('price_vol_holdings',2016)}
good=wf.replay_models(frame,annual,variant='price_vol_holdings',cost_bps=25,plan=plan)
frame.loc[frame.index[3],'future_vol_prediction']=np.nan
bad=wf.replay_models(frame,annual,variant='price_vol_holdings',cost_bps=25,plan=plan)
pd.testing.assert_frame_equal(good.iloc[:3],bad.iloc[:3]);assert bad.observed_forecast_fallback.tolist()==[False,False,False,True,False,False]
assert bad.soxx_weight.iloc[3]==1 and abs(bad.pretrade_soxx_weight.iloc[4]-1)<1e-14 and bad.soxl_weight.iloc[4]==1
assert wf.independent_path_audit(frame,bad,cost_bps=25)['passed']
# H-dependent ensemble members share the ensemble book, not separate seed books.
models=[]
for bias in (.4,.7,1.1):
 m=SparseHoldingsPolicy()
 with torch.no_grad():m.A.zero_();m.K.zero_();m.a.copy_(torch.tensor([bias,0.,0.,.1],dtype=DT));m.K[3,0]=1.3;m.K[2,1]=.4
 models.append(m)
ens=numpy_rollout(models,features,r,cost_bps=25)
manual_errors=[]
for i,held in enumerate(ens['pretrade_weights']):
 target=np.mean([m(torch.tensor(features[i],dtype=DT),torch.tensor(held,dtype=DT)).detach().numpy() for m in models],axis=0)
 manual_errors.append(float(np.max(abs(target-ens['target_weights'][i]))))
assert max(manual_errors)<1e-14
# Exact null bypass retains the strict singleton projection zero Jacobian.
null=SparseHoldingsPolicy(anchor=True)
huge=(torch.sign(null.hidden.weight[0]).detach()*1e308).requires_grad_()
owned=torch.tensor([.2,.6,.2],dtype=DT,requires_grad=True)
null_target=null(huge,owned)
assert torch.equal(null_target,torch.tensor([0.,1.,0.],dtype=DT))
null_grads=torch.autograd.grad(null_target@torch.tensor([.7,1.3,.2],dtype=DT),list(null.parameters())+[huge,owned])
assert all(torch.isfinite(g).all() and torch.count_nonzero(g)==0 for g in null_grads)
# Fitted tensor identity is failclosed under tampering.
restored=policy_from_dict(payload);assert policy_to_dict(restored)['tensor_sha256']==payload['tensor_sha256']
wrong=copy.deepcopy(payload);wrong['state_dict']['K'][0,0]+=.01
try:policy_from_dict(wrong)
except ValueError:tamper_rejected=True
else:raise AssertionError('changed tensor accepted')
report={'scope':'Real new policy and workflow API, synthetic data only; no model fitting, market data, fake model stub, monkeypatch or repo edits.',
 'source_sha256_before':before,'source_sha256_after':sha(source),'projection_cases':40,'parent_embedding_cases':20,
 'max_torch_numpy_projection_error':max(projection_errors),'max_support_Jacobian_error':max(jac_errors),'max_projection_FD_error':max(fd_errors),
 'max_parent_embedding_error':max(embedding_errors),'fresh_initial_paths':initial,'real_workflow_annual_owned_state_and_HOLD':own,
 'current_date_bad_vol_forced_SOXX_and_future_owned_state':True,'ensemble_shared_book_max_error':max(manual_errors),
 'serialized_tensor_tamper_rejected':tamper_rejected,'huge_finite_exact_anchor_forward_and_zero_Jacobian':True,'trainer_review_pending':True,'failures':[]}
Path('/tmp/sparse-holdings-validation/core-integration.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
