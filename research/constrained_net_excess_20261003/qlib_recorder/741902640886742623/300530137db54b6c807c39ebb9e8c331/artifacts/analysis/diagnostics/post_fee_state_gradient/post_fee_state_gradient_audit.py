"""Synthetic derivative audit; never edits, patches or fits project code."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from src.model.integrated_allocation import _settle
from src.workflow.stage1_economic_action import fee_affordable_target
from src.backtest.position_accounting import settle_target_session

torch.set_num_threads(1);DT=torch.float64
RET=np.array([.03,.01,0.])

def algebraic_prototype(h,z,r,c):
    """Same sign-root selection; algebraic post-fee values/state, no STE."""
    if z is None:
        z=h;n=torch.ones_like(h[...,0]);turn=torch.zeros_like(n);fee=turn;execution=h
    else:
        signs=h.new_tensor(((-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)))
        hr,zr=h[...,None,:2],z[...,None,:2]
        candidates=(1+c*(signs*hr).sum(-1))/(1+c*(signs*zr).sum(-1))
        residual=(candidates+c*(candidates[...,None]*zr-hr).abs().sum(-1)-1).abs()
        selected=residual.detach().argmin(-1,keepdim=True)  # Identical existing branch selection.
        n=candidates.gather(-1,selected).squeeze(-1)
        risky=n[...,None]*z[...,:2];turn=(risky-h[...,:2]).abs().sum(-1);fee=c*turn
        execution=torch.cat((risky,(1-risky.sum(-1))[...,None]),dim=-1)
    opening=n[...,None]*z
    marked=z*(1+r);base=marked.sum(-1)
    ending=opening*(1+r);net=ending.sum(-1)
    return {'ending_weights':marked/base[...,None], 'net_equity':net,'fees':fee,'turnover':turn,
            'ending_values':ending,'open_values':opening,'execution_target_weights':execution,'net_open':n}

def economic(h,z,c,ret=RET):
    execution=None if z is None else fee_affordable_target(z,pretrade_weights=h,cost_bps=c*10000)
    return settle_target_session(h,open_target_weights=execution,asset_returns=ret,cost_bps=c*10000)

def analytic_state(z,dz,ret=RET):
    a=1+ret;G=float(z@a)
    return dz*a/G-z*a*(dz@a)/(G*G)

def fd(function,eps,side):
    return ((-3*function(0.)+4*function(eps)-function(2*eps))/(2*eps)
            if side=='right' else (function(eps)-function(-eps))/(2*eps))

def probe(name,h,z,dh,dz,cost,*,side='central',none_target=False):
    h,z,dh,dz=map(lambda v:np.asarray(v,dtype=float),(h,z,dh,dz));c=cost/10000
    assert abs(dh.sum())<1e-14 and abs(dz.sum())<1e-14
    ht,zt,dht,dzt,rt=[torch.tensor(v,dtype=DT) for v in (h,z,dh,dz,RET)]
    def call(kernel,t):return kernel(ht+t*dht,None if none_target else zt+t*dzt,rt,c)
    zero=torch.tensor(0.,dtype=DT,requires_grad=True)
    old=call(_settle,zero);new=call(algebraic_prototype,zero)
    old_j=torch.autograd.functional.jacobian(lambda t:call(_settle,t)['ending_weights'],zero).detach().numpy()
    new_j=torch.autograd.functional.jacobian(lambda t:call(algebraic_prototype,t)['ending_weights'],zero).detach().numpy()
    true_j=analytic_state(h if none_target else z,dh if none_target else dz)
    def numpy_call(t):return economic(h+t*dh,None if none_target else z+t*dz,c)
    fd_j=fd(lambda t:numpy_call(t)['ending_weights'],1e-6,side)
    old_log=float(torch.autograd.grad(torch.log(old['net_equity']),zero,retain_graph=True)[0])
    new_log=float(torch.autograd.grad(torch.log(new['net_equity']),zero,retain_graph=True)[0])
    old_fee=float(torch.autograd.grad(old['fees']+zero*0,zero,retain_graph=True)[0])
    new_fee=float(torch.autograd.grad(new['fees']+zero*0,zero,retain_graph=True)[0])
    fd_log=float(fd(lambda t:np.log(numpy_call(t)['net_equity']),1e-6,side))
    fd_fee=float(fd(lambda t:numpy_call(t)['fees'],1e-6,side))
    def left_fd(function,eps=1e-6):return float((3*function(0.)-4*function(-eps)+function(-2*eps))/(2*eps))
    right_log=float(fd(lambda t:np.log(numpy_call(t)['net_equity']),1e-6,'right'))
    right_fee=float(fd(lambda t:numpy_call(t)['fees'],1e-6,'right'))
    left_log=None if side=='right' else left_fd(lambda t:np.log(numpy_call(t)['net_equity']))
    left_fee=None if side=='right' else left_fd(lambda t:numpy_call(t)['fees'])
    numeric=numpy_call(0.)
    forward={key:abs(float(old[key].detach())-float(new[key].detach())) for key in ('net_equity','fees','turnover')}
    forward['ending_weights']=float(np.max(abs(old['ending_weights'].detach().numpy()-new['ending_weights'].detach().numpy())))
    forward['independent_numpy_ending_weights']=float(np.max(abs(numeric['ending_weights']-new['ending_weights'].detach().numpy())))
    forward['independent_numpy_net_equity']=abs(numeric['net_equity']-float(new['net_equity'].detach()))
    forward['cash_debit_identity']=abs(float(new['open_values'][2].detach())-float((new['execution_target_weights'][2]-new['fees']).detach()))
    forward['funding_identity']=abs(float((new['net_open']+new['fees']-1).detach()))
    return {'case':name,'cost_bps':cost,'no_trade_at_point':none_target or np.array_equal(h,z),'finite_difference':side,'held':h.tolist(),'target':None if none_target else z.tolist(),'dh':dh.tolist(),'dz':dz.tolist(),
        'legacy_state_jvp':old_j.tolist(),'algebraic_state_jvp':new_j.tolist(),'analytic_state_jvp':true_j.tolist(),'independent_numpy_fd_state_jvp':fd_j.tolist(),
        'legacy_state_max_error':float(np.max(abs(old_j-true_j))),'algebraic_state_max_error':float(np.max(abs(new_j-true_j))),'finite_difference_state_max_error':float(np.max(abs(fd_j-true_j))),
        'scalar_log_NAV_derivatives':{'legacy_autograd':old_log,'algebraic_autograd':new_log,'independent_numpy_fd':fd_log,'right_fd':right_log,'left_fd':left_log},
        'scalar_fee_derivatives':{'legacy_autograd':old_fee,'algebraic_autograd':new_fee,'independent_numpy_fd':fd_fee,'right_fd':right_fee,'left_fd':left_fee},'forward_errors':forward}

source=Path('src/model/integrated_allocation.py');before=hashlib.sha256(source.read_bytes()).hexdigest();cases=[];mixed=np.array([.2,.6,.2]);zero=[0.,0.,0.]
directions={'S_to_cash':[0.,-1.,1.],'L_to_cash':[-1.,0.,1.],'S_to_L':[1.,-1.,0.],'cash_to_S':[0.,1.,-1.]}
for cost in (0.,10.,25.):
    cases.append(probe('S_to_cash_target_at_theta0',[0,1,0],[0,1,0],zero,directions['S_to_cash'],cost,side='right'))
    cases.append(probe('S_to_L_scalar_boundary_control',[0,1,0],[0,1,0],zero,directions['S_to_L'],cost,side='right'))
    for name,d in directions.items():
        cases.append(probe('mixed_no_trade_target_'+name,mixed,mixed,zero,d,cost))
        cases.append(probe('mixed_no_trade_pretrade_only_'+name,mixed,mixed,d,zero,cost))
        cases.append(probe('away_from_kink_target_'+name,mixed,[.3,.4,.3],zero,d,cost))
    d=np.asarray(directions['S_to_cash'])
    for alpha in (.7,1.):cases.append(probe('mixed_coupled_HOLD_'+str(alpha),mixed,mixed,d,alpha*d,cost))
    cases.append(probe('explicit_None_HOLD_control',mixed,mixed,d,d,cost,none_target=True))
# The return-coordinate Jacobian is a separate smooth control at no-trade.
return_controls=[]
for cost in (0.,10.,25.):
    h=torch.tensor(mixed,dtype=DT);z=h.clone();r=torch.tensor(RET,dtype=DT,requires_grad=True);a=1+RET;G=mixed@a
    expected=np.diag(mixed)/G-np.outer(mixed*a,mixed)/(G*G)
    old=torch.autograd.functional.jacobian(lambda v:_settle(h,z,v,cost/10000)['ending_weights'],r).numpy()
    new=torch.autograd.functional.jacobian(lambda v:algebraic_prototype(h,z,v,cost/10000)['ending_weights'],r).numpy()
    return_controls.append({'cost_bps':cost,'legacy_max_error':float(np.max(abs(old-expected))),'algebraic_max_error':float(np.max(abs(new-expected)))})
report={'scope':'Purely synthetic derivative/value verification; no market data, model training, parameter search, repo edits, monkeypatch or old-result changes.',
 'source_file':str(source),'source_sha256_before':before,'source_sha256_after':hashlib.sha256(source.read_bytes()).hexdigest(),'torch_version':torch.__version__,'dtype':'float64','returns':RET.tolist(),
 'state_identity':'H(z)=z*(1+r)/sum(z*(1+r)); on simplex equals z*(1+r)/(1+z dot r), independent of cost and of h except through z.',
 'prototype':'Same existing detached sign-root index and n expression; open_values=n*z, ending_weights=target-only normalized marked z. fees remain c*actual risky L1 turnover. No custom backward, straight-through estimator or detached substitute Jacobian.',
 'cases':cases,'return_jacobian_controls':return_controls,
 'maximum_legacy_state_error_no_trade':max(c['legacy_state_max_error'] for c in cases if c['no_trade_at_point']),
 'maximum_legacy_state_error_away_from_kink':max(c['legacy_state_max_error'] for c in cases if not c['no_trade_at_point']),
 'maximum_algebraic_state_error':max(c['algebraic_state_max_error'] for c in cases),
 'maximum_independent_fd_state_error':max(c['finite_difference_state_max_error'] for c in cases),
 'maximum_forward_error':max(v for c in cases for v in c['forward_errors'].values()),
 'scalar_boundary_scope':'Central scalar finite differences at no-trade are symmetric secants, not certified derivatives; explicit left/right secants are also retained where both directions are feasible. State smoothness does not certify scalar fee or net-NAV derivatives at no-trade. Each specified feasible curve has a definite one-sided derivative; chosen sign-root/abs subgradients need not give that derivative. S-to-L remains a counterexample after the proposed state fix.',
 'autograd_reference':'https://docs.pytorch.org/docs/2.14/notes/autograd.html#gradients-for-non-differentiable-functions'}
assert report['source_sha256_before']==report['source_sha256_after']
assert report['maximum_forward_error']<1e-12
assert report['maximum_algebraic_state_error']<1e-12
assert report['maximum_independent_fd_state_error']<2e-8
assert report['maximum_legacy_state_error_no_trade']>1e-4
assert report['maximum_legacy_state_error_away_from_kink']<1e-12
Path('/tmp/post_fee_state_gradient_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('cases','return_jacobian_controls')},indent=2))
print('S_to_cash examples',json.dumps([c for c in cases if c['case']=='S_to_cash_target_at_theta0' and c['cost_bps']>0],indent=2))
