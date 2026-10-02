"""Independent synthetic-only methodology probe; no project source writes or market reads."""
from pathlib import Path
import json,time
import numpy as np
import torch
from src.model.integrated_allocation import _settle
from src.workflow.stage1_economic_action import fee_affordable_target
from src.backtest.position_accounting import settle_target_session
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
rng=np.random.default_rng(17);N=130
features=torch.tensor(rng.normal(size=(N,27)))
market=np.c_[rng.normal(.0004,.023,N),rng.normal(.0002,.008,N),np.zeros(N)]
returns=torch.tensor(market)
W=torch.tensor(rng.normal(0,.15,(8,27)),requires_grad=True)
bias=torch.zeros(8,requires_grad=True);theta=torch.full((3,9),1/54,requires_grad=True);params=(W,bias,theta)
S=torch.tensor([0.,1.,0.]);L=torch.tensor([1.,0.,0.]);C=torch.tensor([0.,0.,1.])
def groups(n):
 out=[(i,i+63) for i in range(0,n-62,63)]
 if out[-1][1]!=n:out.append((n-63,n))
 return out
def utility(x):
 q=torch.stack([x[a:b].sum() for a,b in groups(len(x))]);return 63/len(x)*x.sum()-torch.sqrt(torch.relu(-q).square().mean()+1e-12)
def target(t,h,null=False):
 if null:return S
 b=torch.cat((torch.ones(1),torch.sigmoid(W@features[t]+bias)));alpha=theta@b
 return (1-alpha.sum())*S+alpha[0]*L+alpha[1]*C+alpha[2]*h
def path(cost,begin=0,end=N,initial=None,null=False):
 h=C.clone() if initial is None else initial;logs=[];states=[];fees=[];targets=[]
 for t in range(begin,end):
  states.append(h);z=target(t,h,null=null);r=_settle(h,z,returns[t],cost/1e4)
  logs.append(torch.log(r['net_equity']));fees.append(r['fees']);targets.append(z);h=r['ending_weights']
 return torch.stack(logs),torch.stack(states),torch.stack(fees),torch.stack(targets)
def objective():
 paths=[path(c) for c in (10,25)];bench=[path(c,null=True)[0] for c in (10,25)]
 components=torch.stack([utility(p[0]-b)+1e-6 for p,b in zip(paths,bench)])
 return torch.amin(components),paths,bench,components
start=time.perf_counter();J,actual,benchmark,U=objective();full_grad=torch.cat([g.flatten() for g in torch.autograd.grad(J,params)])
x=torch.stack([(p[0]-b).detach() for p,b in zip(actual,benchmark)]).requires_grad_();w=torch.autograd.grad(torch.amin(torch.stack([utility(v)+1e-6 for v in x])),x)[0].detach()
def chunk_grad(context):
 accumulated=[torch.zeros_like(p) for p in params];maximum_forward_error=0.;entry_scored=False
 for cindex,cost in enumerate((10,25)):
  for s in range(0,N,63):
   e=min(s+63,N);a=max(0,s-context)
   log,_,_,_=path(cost,a,e,actual[cindex][1][a].detach().clone())
   maximum_forward_error=max(maximum_forward_error,float((log.detach()-actual[cindex][0][a:e].detach()).abs().max()))
   loss=(w[cindex,s:e]*log[s-a:]).sum()
   g=torch.autograd.grad(loss,params)
   accumulated=[old+new for old,new in zip(accumulated,g)]
   if s==0:entry_scored=True
 return torch.cat([g.flatten() for g in accumulated]),maximum_forward_error,entry_scored
exact_context,forward_error,_=chunk_grad(N);truncated,_,entry_scored=chunk_grad(21)
# Independent NumPy settlement uses the same candidate target on its own h.
np_errors=[];initial_fees=[]
for i,cost in enumerate((10,25)):
 h=np.array([0.,0.,1.]);errors=[]
 for t in range(N):
  b=np.r_[1.,1/(1+np.exp(-(W.detach().numpy()@features[t].numpy()+bias.detach().numpy())))]
  a=theta.detach().numpy()@b;z=(1-a.sum())*np.array([0.,1.,0.])+a[0]*np.array([1.,0.,0.])+a[1]*np.array([0.,0.,1.])+a[2]*h
  executable=fee_affordable_target(z,pretrade_weights=h,cost_bps=cost)
  r=settle_target_session(h,open_target_weights=executable,asset_returns=market[t],cost_bps=cost)
  errors.append(abs(np.log(r['net_equity'])-float(actual[i][0][t].detach())))
  if t==0:initial_fees.append(r['fees'])
  h=r['ending_weights']
 np_errors.append(max(errors))
# The analytic global derivative includes overlapping final-window memberships.
weight_errors={};group_shapes={}
for n in (63,64,126,127,130,252,253):
 values=torch.tensor(rng.normal(-.0003,.002,n),requires_grad=True);q=np.array([float(values[a:b].sum().detach()) for a,b in groups(n)])
 actual_weight=torch.autograd.grad(utility(values),values)[0].detach().numpy();expected=np.full(n,63/n);rms=np.sqrt(np.mean(np.maximum(-q,0)**2)+1e-12)
 for value,(a,b) in zip(q,groups(n)):expected[a:b]+=max(-value,0)/(len(q)*rms)
 weight_errors[n]=float(np.max(np.abs(actual_weight-expected)));group_shapes[n]=groups(n)
# Central finite differences at the strictly interior test point.
finite_differences=[]
for p,index in ((theta,(0,0)),(W,(0,0)),(bias,(0,))):
 original=float(p[index].detach());epsilon=1e-6
 with torch.no_grad():p[index]=original+epsilon
 plus=float(objective()[0].detach())
 with torch.no_grad():p[index]=original-epsilon
 minus=float(objective()[0].detach())
 with torch.no_grad():p[index]=original
 numerical=(plus-minus)/(2*epsilon)
 direct=torch.autograd.grad(objective()[0],params)[params.index(p)] if False else None
 Jnew=objective()[0];grad=torch.autograd.grad(Jnew,p)[0];finite_differences.append({'parameter_shape':list(p.shape),'index':list(index),'autograd':float(grad[index]),'central_difference':numerical,'absolute_error':abs(float(grad[index])-numerical)})
# At anchor the global cost tie must share its adjoint symmetrically.
nullx=torch.zeros((2,N),requires_grad=True);nullJ=torch.amin(torch.stack([utility(z)+1e-6 for z in nullx]));nullw=torch.autograd.grad(nullJ,nullx)[0]
report={'scope':'synthetic-only independent prototype; no market data, model fits or repo source edits',
 'N':N,'theta_total':float(theta.sum().detach()),'centered_cost_utilities':U.detach().tolist(),'J':float(J.detach()),
 'entry_fees_both_costs':initial_fees,'entry_in_first_scored_chunk':entry_scored,'independent_numpy_forward_log_errors':np_errors,
 'global_weight_max_errors':weight_errors,'risk_group_examples':group_shapes,'chunk_reconstruction_max_log_error':forward_error,
 'full_context_vs_full_BPTT_max_gradient_error':float((exact_context-full_grad).abs().max()),
 'context21_vs_full_BPTT_relative_L2_error':float(torch.linalg.vector_norm(truncated-full_grad)/torch.linalg.vector_norm(full_grad)),
 'context21_vs_full_BPTT_gradient_cosine':float(torch.dot(truncated,full_grad)/(torch.linalg.vector_norm(truncated)*torch.linalg.vector_norm(full_grad))),
 'finite_difference_checks':finite_differences,'anchor_centered_J':float(nullJ.detach()),'anchor_tie_cost_weights_equal':bool(torch.equal(nullw[0],nullw[1])),
 'anchor_daily_cost_adjoint':float(nullw[0,0]),'expected_anchor_daily_cost_adjoint':.5*63/N,'elapsed_seconds':time.perf_counter()-start}
Path('/tmp/benchmark_anchor_synthetic_review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
# A near-HOLD point exposes genuinely long state gradients; the registered
# interior start contracts holdings too quickly to stress truncation alone.
with torch.no_grad():
 theta.zero_();theta[2,0]=.97;theta[0,0]=.015;theta[1,0]=.005
J,actual,benchmark,U=objective();full_grad=torch.cat([g.flatten() for g in torch.autograd.grad(J,params)])
x=torch.stack([(p[0]-b).detach() for p,b in zip(actual,benchmark)]).requires_grad_();w=torch.autograd.grad(torch.amin(torch.stack([utility(v)+1e-6 for v in x])),x)[0].detach()
long_exact,err,_=chunk_grad(N);long_truncated,_,_=chunk_grad(21)
report['near_HOLD_synthetic_stress']={'theta_mass':float(theta.sum().detach()),'hold_coefficient':.97,'forward_chunk_error':err,
 'full_context_gradient_max_error':float((long_exact-full_grad).abs().max()),
 'context21_gradient_relative_L2_error':float(torch.linalg.vector_norm(long_truncated-full_grad)/torch.linalg.vector_norm(full_grad)),
 'context21_gradient_cosine':float(torch.dot(long_truncated,full_grad)/(torch.linalg.vector_norm(long_truncated)*torch.linalg.vector_norm(full_grad))),
 'interpretation':'Exact forward accounting is preserved; bounded-context gradient is deliberately approximate at high inertia. This is not a market fit.'}
Path('/tmp/benchmark_anchor_synthetic_review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['near_HOLD_synthetic_stress'],indent=2))
