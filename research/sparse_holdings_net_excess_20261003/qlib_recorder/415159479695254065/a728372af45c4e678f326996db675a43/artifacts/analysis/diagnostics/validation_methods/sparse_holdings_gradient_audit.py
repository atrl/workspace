"""Synthetic attached-state gradient and actual trainer/workflow API audit."""
from pathlib import Path
import copy,hashlib,json
import numpy as np,torch
from src.model.sparse_holdings_net_excess import SparseHoldingsPolicy,ConstrainedNetExcessSpec,evaluate_policy,truncated_utility_gradient,policy_to_dict,evaluate_ensemble
from src.model.constrained_net_excess import gradient_proxy
from src.model.post_fee_settlement import settle_torch
from src.workflow.sparse_holdings_net_excess import _fit_job,_validate_retained

torch.set_num_threads(1);DT=torch.float64;rng=np.random.default_rng(20261003)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();source=Path('src/model/sparse_holdings_net_excess.py');before=sha(source)
def panel(n):
 t=np.arange(n);x=rng.normal(size=(n,27))*.4;s=.002+.01*np.sin(t*.51)+.003*np.cos(t*.173)
 return x,np.c_[2.8*s+.001*np.sin(t*.34),s]
def make_model():
 m=SparseHoldingsPolicy(seed=17)
 with torch.no_grad():m.K[0,1]=.2;m.K[2,0]=-.15;m.K[3,2]=.1
 return m
def gdict(model):return {name:p.grad.detach().clone() for name,p in model.named_parameters()}
def diff(a,b):return max(float((a[k]-b[k]).abs().max()) for k in a)
def explicit_chunks(model,x,r,proxy,spec,*,detach_inner=False):
 model.zero_grad(set_to_none=True);loss=torch.tensor(0.,dtype=DT);n=len(x)
 for cost in spec.cost_bps:
  full=proxy['paths'][cost];w=proxy['components'][cost]['daily_weights']*proxy['cost_weights'][cost]
  for start in range(0,n,spec.gradient_chunk_length):
   context=max(0,start-spec.context_sessions);end=min(n,start+spec.gradient_chunk_length)
   held=torch.tensor(full['pretrade_weights'][context].copy(),dtype=DT)
   for i in range(context,end):
    target=model(torch.tensor(x[i],dtype=DT),held)
    out=settle_torch(held,target,torch.tensor(np.r_[r[i],0.],dtype=DT),cost/10000)
    if i>=start:loss=loss-w[i]*torch.log(out['net_equity'])
    held=out['ending_weights'].detach() if detach_inner else out['ending_weights']
 loss.backward();return gdict(model),float(loss.detach())
def full_loss_torch(model,x,r,current,dual):
 n=len(x);A=[];risks=[]
 for cost in (10.,25.):
  held=torch.tensor([0.,0.,1.],dtype=DT);logs=[]
  for i in range(n):
   out=settle_torch(held,model(torch.tensor(x[i],dtype=DT),held),torch.tensor(np.r_[r[i],0.],dtype=DT),cost/10000)
   logs.append(torch.log(out['net_equity']));held=out['ending_weights']
  log=torch.stack(logs)
  # Same fixed reference SOXX: entry fee then asset returns, derived independently.
  blog=torch.log1p(torch.tensor(r[:,1],dtype=DT));blog=blog-torch.nn.functional.pad(torch.tensor([np.log1p(cost/10000)],dtype=DT),(0,n-1))
  A.append((log-blog).sum()*63/n)
  lv=torch.cat((torch.zeros(1,dtype=DT),log.cumsum(0)));D=(1-torch.exp(lv-torch.cummax(lv,0).values)).max()
  net=torch.expm1(log);ordered=torch.sort(net).values;mass=n*.05;whole=int(np.floor(mass));fraction=mass-whole
  C=-(ordered[:whole].sum()+fraction*ordered[whole])/mass
  B=current['constraints_by_cost'][cost]['normalizers'];risks.extend([(D-B[0])/B[0],(C-B[1])/B[1]])
 return -torch.stack(A).min()+torch.tensor(dual,dtype=DT)@torch.stack(risks)
results=[]
for n in (63,137):
 x,r=panel(n);spec=ConstrainedNetExcessSpec(updates=0,minimum_rows=63,gradient_chunk_length=63,context_sessions=21)
 model=make_model();dual=np.array([.015,.02,.01,.025]);current=evaluate_policy(model,x,r,spec=spec);proxy,_=gradient_proxy(current,dual,spec=spec)
 audit=truncated_utility_gradient(model,x,r,proxy,spec=spec);actual=gdict(model)
 explicit,explicit_loss=explicit_chunks(copy.deepcopy(model),x,r,proxy,spec)
 detached,_=explicit_chunks(copy.deepcopy(model),x,r,proxy,spec,detach_inner=True)
 error=diff(actual,explicit);Kdetach=float((actual['K']-detached['K']).abs().max())
 assert audit['scored_rows']==n and audit['support_mismatch_count']==0 and audit['forward_parity_max_error']<1e-10
 assert error<2e-11 and Kdetach>1e-7
 row={'rows':n,'vectorized_vs_explicit_context_gradient_error':error,'K_gradient_norm':float(actual['K'].norm()),
 'K_gradient_difference_if_state_wrongly_detached':Kdetach,'forward_parity_max_error':audit['forward_parity_max_error'],'scored_rows':audit['scored_rows'],'chunk_boundaries':audit['chunk_boundaries']}
 if n==63:
  ref=copy.deepcopy(model);ref.zero_grad(set_to_none=True);direct=full_loss_torch(ref,x,r,current,dual);direct.backward();der=diff(actual,gdict(ref));assert der<2e-10
  row['whole_risk_loss_autograd_error']=der;row['whole_risk_loss_value']=float(direct.detach());row['finite_difference_directions']=[]
  for group in ('K','head','encoder'):
   direction={name:torch.tensor(rng.normal(size=tuple(p.shape)),dtype=DT) if (name=='K' if group=='K' else name in ('A','a') if group=='head' else name.startswith('hidden.')) else torch.zeros_like(p) for name,p in model.named_parameters()}
   scale=torch.sqrt(sum(v.square().sum() for v in direction.values()));direction={k:v/scale for k,v in direction.items()}
   expected=sum(float((actual[k]*d).sum()) for k,d in direction.items())
   vals=[]
   for sign in (-1,1):
    probe=copy.deepcopy(model)
    with torch.no_grad():
     for name,p in probe.named_parameters():p.add_(direction[name]*sign*1e-6)
    e=evaluate_policy(probe,x,r,spec=spec);vals.append(-e['F']+dual@e['normalized_g'])
   fd=(vals[1]-vals[0])/2e-6;err=abs(fd-expected);assert err<2e-7
   row['finite_difference_directions'].append({'group':group,'autograd':expected,'finite_difference':float(fd),'absolute_error':float(err)})
 results.append(row)
# Actual worker -> actual metadata gate -> actual three-policy ensemble. 2 updates are a synthetic API fixture only.
x,r=panel(80);spec=ConstrainedNetExcessSpec(updates=2,minimum_rows=63);plan=json.loads(Path('research/sparse_holdings_net_excess_20261003/selection_plan.json').read_text());plan['training']['updates']=2;plan['training']['minimum_rows']=63
models=[];smoke=[]
for seed in (17,41,71):
 name,year,returned_seed,payload,metadata=_fit_job(('price_holdings',2016,seed,x,r,spec));models.extend(_validate_retained([payload],metadata,plan))
 assert (name,year,returned_seed)==('price_holdings',2016,seed)
 assert metadata['failure'] is None and metadata['optimizer_diagnostics']['updates_completed']==2
 assert metadata['training_feature_sha256']==hashlib.sha256(x.astype(np.float64).tobytes()).hexdigest()
 assert metadata['training_return_sha256']==hashlib.sha256(r.astype(np.float64).tobytes()).hexdigest()
 smoke.append({'seed':seed,'epochs':len(metadata['epochs']),'selected_epoch':metadata['selected_epoch'],'F':metadata['F'],'failure':metadata['failure'],'initial_path_parity':metadata['initial_path_parity']})
accepted,ensemble=evaluate_ensemble(models,x,r,spec=spec);_validate_retained([policy_to_dict(m) for m in accepted],ensemble,plan)
assert ensemble['failure'] is None and ensemble['training_feasibility_passed']
report={'scope':'Synthetic mathematical and actual worker/retention/ensemble API checks only; no market data or market fitting, no outcomes-based parameter search, no source edits. Three 2-update fits are explicitly a small synthetic API fixture, not the registered120-update known-information gate.',
 'source_sha256_before':before,'source_sha256_after':sha(source),'gradient_cases':results,'worker_smoke':smoke,
 'ensemble_smoke':{'failure':ensemble['failure'],'training_feasibility_passed':ensemble['training_feasibility_passed'],'anchor_selected':ensemble['anchor_selected'],'component_count':ensemble['component_count']},'failures':[]}
Path('/tmp/sparse-holdings-validation/gradient-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
