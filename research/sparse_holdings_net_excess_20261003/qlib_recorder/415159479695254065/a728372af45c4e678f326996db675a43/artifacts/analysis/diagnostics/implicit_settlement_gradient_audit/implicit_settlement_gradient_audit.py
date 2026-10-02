"""Read-only synthetic audit of the separately versioned implicit settlement kernel.
No market data, model fitting, search, source mutation or monkeypatch.
"""
from pathlib import Path
import hashlib,itertools,json
import numpy as np
import torch
from src.model.post_fee_settlement import settle_torch, scalar_settle
from src.model.integrated_allocation import _settle as legacy_settle

torch.set_num_threads(1)
DT=torch.float64
SOURCE=Path('src/model/post_fee_settlement.py')
LEGACY=Path('src/model/integrated_allocation.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source_before=sha(SOURCE);legacy_before=sha(LEGACY)
RET=np.array([.03,.01,0.])

def implicit_reference(h,z,c):
    """Independent monotone bisection, not four-root enumeration."""
    lo,hi=0.,1.
    for _ in range(90):
        mid=(lo+hi)/2
        if mid+c*np.abs(mid*z[:2]-h[:2]).sum()>1: hi=mid
        else:lo=mid
    return (lo+hi)/2

def numpy_reference(h,z,r,c):
    n=implicit_reference(h,z,c);marked=z*(1+r);g=marked.sum()
    return np.r_[n,c*np.abs(n*z[:2]-h[:2]).sum(),n*g,marked/g]

def output_vector(o):
    return torch.cat((torch.stack((o['open_net_equity'],o['fees'],o['net_equity'])),o['ending_weights']))

def state_jvp(z,dz,r):
    a=1+r;g=z@a
    return dz*a/g-z*a*(dz@a)/(g*g)

def branch_mix(n,z,signs,c):
    zero=np.flatnonzero(signs==0);variants=[]
    for bits in itertools.product((-1.,1.),repeat=len(zero)):
        s=signs.copy();s[zero]=bits
        den=1+c*(s@z[:2]);v=np.r_[c*s/den,-c*n*s/den]
        variants.append((s,den,v))
    d0=1+c*(signs@z[:2]);weights=np.array([den/(len(variants)*d0) for _,den,_ in variants])
    average=sum(weight*v for weight,(_,_,v) in zip(weights,variants))
    chosen=np.r_[c*signs/d0,-c*n*signs/d0]
    return {'zero_legs':zero.tolist(),'weights':weights.tolist(),'minimum_weight':float(weights.min()),
            'weight_sum_error':float(abs(weights.sum()-1)),'gradient_mixture_error':float(np.max(abs(average-chosen))),
            'chosen_gradient_h_then_z':chosen.tolist(),'branch_gradients':[v.tolist() for _,_,v in variants],
            'sign_patterns':[s.tolist() for s,_,_ in variants], 'denominator':float(d0)}

def probe(name,h,z,dh,dz,bps,*,smooth=False,one_sided=False):
    h,z,dh,dz=map(lambda v:np.asarray(v,dtype=float),(h,z,dh,dz));c=bps/10000
    ht,zt,rt,dht,dzt=map(lambda v:torch.tensor(v,dtype=DT),(h,z,RET,dh,dz))
    def out(t):return settle_torch(ht+t*dht,zt+t*dzt,rt,c)
    t=torch.tensor(0.,dtype=DT,requires_grad=True);o=out(t)
    values=output_vector(o).detach().numpy()
    jac=torch.autograd.functional.jacobian(lambda x:output_vector(out(x)),t).numpy()
    n=values[0];s=np.sign(n*z[:2]-h[:2]);D=1+c*s@z[:2]
    dn=c*s@(dh[:2]-n*dz[:2])/D
    expected=np.r_[dn,-dn,dn*(z@(1+RET))+n*(dz@(1+RET)),state_jvp(z,dz,RET)]
    eps=1e-6;f=lambda v:numpy_reference(h+v*dh,z+v*dz,RET,c)
    left=(3*f(0)-4*f(-eps)+f(-2*eps))/(2*eps)
    right=(-3*f(0)+4*f(eps)-f(2*eps))/(2*eps)
    center=(f(eps)-f(-eps))/(2*eps)
    ref=f(0);old=legacy_settle(ht,zt,rt,c)
    scalar=scalar_settle(*h,*z,RET[0],RET[1],c)
    forward_errors={'bisection':float(np.max(abs(values-ref))),
      'legacy_nav':abs(float(o['net_equity'])-float(old['net_equity'])),
      'legacy_fee':abs(float(o['fees'])-float(old['fees'])),
      'legacy_state':float(torch.max(torch.abs(o['ending_weights']-old['ending_weights']))),
      'scalar_state':float(np.max(abs(np.array(scalar[0])-values[3:]))),
      'scalar_nav':abs(scalar[1]-values[2]),'scalar_fee':abs(scalar[2]-values[1]),
      'funding':abs(values[0]+values[1]-1.)}
    funding_grad=float(torch.autograd.grad(o['open_net_equity']+o['fees'],t,retain_graph=True)[0])
    cash_error=o['execution_target_weights'][2]-o['fees']-o['open_net_equity']*(zt+t*dzt)[2]
    cash_grad=float(torch.autograd.grad(cash_error,t,retain_graph=True)[0])
    mixture=branch_mix(n,z,s,c)
    result={'case':name,'cost_bps':bps,'smooth':smooth,'h':h.tolist(),'z':z.tolist(),'dh':dh.tolist(),'dz':dz.tolist(),
      'actual_trade':(n*z[:2]-h[:2]).tolist(),'actual_signs':s.tolist(),
      'autograd_vector_n_fee_nav_state':jac.tolist(),'analytic_selected_vector':expected.tolist(),
      'left_fd_vector':None if one_sided else left.tolist(),'right_fd_vector':right.tolist(),
      'center_fd_vector':None if one_sided else center.tolist(),
      'implicit_formula_error':float(np.max(abs(jac-expected))),
      'state_identity_error':float(np.max(abs(jac[3:]-expected[3:]))),
      'state_fd_error':float(np.max(abs((right if one_sided else center)[3:]-expected[3:]))),
      'smooth_fd_error':float(np.max(abs(center-jac))) if smooth else None,
      'funding_identity_gradient_error':abs(funding_grad),'cash_debit_gradient_error':abs(cash_grad),
      'forward_errors':forward_errors,'ambient_Clarke_convex_combination':mixture}
    assert result['implicit_formula_error']<2e-14
    assert result['state_identity_error']<2e-14
    assert result['state_fd_error']<2e-8
    assert result['funding_identity_gradient_error']<2e-14
    assert result['cash_debit_gradient_error']<2e-14
    assert max(forward_errors.values())<2e-12
    assert mixture['minimum_weight']>0 and mixture['weight_sum_error']<1e-14 and mixture['gradient_mixture_error']<1e-14
    if smooth: assert result['smooth_fd_error']<2e-8
    return result

cases=[];zero=[0.,0.,0.];mixed=np.array([.2,.6,.2]);dirs={'S_to_C':[0,-1,1],'S_to_L':[1,-1,0],'L_to_C':[-1,0,1]}
for bps in (0.,10.,25.):
    for name,d in dirs.items():
        cases.append(probe('mixed_zero_trade_target_'+name,mixed,mixed,zero,d,bps))
        cases.append(probe('mixed_zero_trade_held_'+name,mixed,mixed,d,zero,bps))
        cases.append(probe('away_kink_target_'+name,mixed,[.3,.4,.3],zero,d,bps,smooth=True))
        cases.append(probe('away_kink_held_'+name,mixed,[.3,.4,.3],d,zero,bps,smooth=True))
    for name,d in list(dirs.items())[:2]:
        cases.append(probe('pure_SOXX_boundary_'+name,[0,1,0],[0,1,0],zero,d,bps,one_sided=True))
    cases.append(probe('partial_zero_absent_L_buy',[0,.6,.4],[0,.4,.6],zero,[1,0,-1],bps,one_sided=True))
    cases.append(probe('coupled_exact_HOLD',mixed,mixed,[0,-1,1],[0,-1,1],bps))
# Exactly zero non-absent partial leg in float64 at25bp, while the other leg sells.
c=.0025;z=np.array([.25,.5,.25]);n=1-2**-14;q=2**-14/c;h=np.r_[n*z[0],n*z[1]+q,0.];h[2]=1-h[:2].sum()
cases.append(probe('partial_zero_interior_non_absent_L',h,z,zero,[1,0,-1],25))
assert cases[-1]['actual_signs']==[0.,-1.]
# Random value parity is not derivative certification; fixed seed, no optimization.
rng=np.random.default_rng(20261003);random_errors=[]
for i in range(300):
    h=rng.dirichlet(np.ones(3));z=rng.dirichlet(np.ones(3));r=np.r_[rng.uniform(-.7,.7,2),0.];c=(.001,.0025)[i%2]
    tensors=[torch.tensor(x,dtype=DT) for x in (h,z,r)]
    out=settle_torch(*tensors,c);old=legacy_settle(*tensors,c)
    v=output_vector(out).numpy();ref=numpy_reference(h,z,r,c)
    random_errors.append(max(float(np.max(abs(v-ref))),float(torch.max(torch.abs(out['ending_weights']-old['ending_weights']))),abs(float(out['net_equity']-old['net_equity']))))
assert max(random_errors)<2e-12
# Bad values must not silently produce an output.
invalid=[]
for name,h,z,r,c in [
 ('nan_held',[float('nan'),.5,.5],mixed,RET,.001),('inf_target',mixed,[float('inf'),0,0],RET,.001),
 ('negative_target',mixed,[-.1,.5,.6],RET,.001),('bad_sum',mixed,[.2,.3,.4],RET,.001),
 ('invalid_physical_return',mixed,mixed,[-1,.01,0],.001),('nan_return',mixed,mixed,[float('nan'),.01,0],.001),
 ('negative_cost',mixed,mixed,RET,-.001),('unit_cost',mixed,mixed,RET,1.),('infinite_cost',mixed,mixed,RET,float('inf'))]:
    try:settle_torch(*[torch.tensor(v,dtype=DT) for v in (h,z,r)],c)
    except (ValueError,RuntimeError) as exc:invalid.append({'case':name,'exception':type(exc).__name__,'message':str(exc)})
    else:raise AssertionError('invalid input accepted: '+name)
# One-step HOLD is smooth even though scalar fee has a kink before composition.
for row in cases:
    if row['case']=='coupled_exact_HOLD':
        assert abs(row['autograd_vector_n_fee_nav_state'][0])<1e-15
        assert abs(row['autograd_vector_n_fee_nav_state'][1])<1e-15
# Explicit feasible-side counterexample: ambient sign-zero is not a directional derivative.
partial=next(row for row in cases if row['case']=='partial_zero_absent_L_buy' and row['cost_bps']==25.)
assert abs(partial['autograd_vector_n_fee_nav_state'][0])<1e-15
assert partial['right_fd_vector'][0]<-.002
report={'scope':'Read-only pure synthetic audit of actual separately versioned kernel; no market data, fits, parameter selection, source edits, monkeypatch or changed historical results.',
 'kernel_source':str(SOURCE),'source_sha256_before':source_before,'source_sha256_after':sha(SOURCE),
 'legacy_source_sha256_before':legacy_before,'legacy_source_sha256_after':sha(LEGACY),'torch_version':torch.__version__,'dtype':'float64',
 'case_count':len(cases),'random_forward_cases':300,'invalid_input_cases':len(invalid),
 'maximum_forward_error':max(max(row['forward_errors'].values()) for row in cases),
 'maximum_random_forward_error':max(random_errors),'maximum_implicit_formula_error':max(row['implicit_formula_error'] for row in cases),
 'maximum_smooth_fd_error':max(row['smooth_fd_error'] for row in cases if row['smooth_fd_error'] is not None),
 'maximum_state_identity_error':max(row['state_identity_error'] for row in cases),'maximum_state_fd_error':max(row['state_fd_error'] for row in cases),
 'maximum_funding_gradient_error':max(row['funding_identity_gradient_error'] for row in cases),
 'maximum_cash_debit_gradient_error':max(row['cash_debit_gradient_error'] for row in cases),
 'maximum_Clarke_convex_combination_error':max(row['ambient_Clarke_convex_combination']['gradient_mixture_error'] for row in cases),
 'cases':cases,'invalid_inputs':invalid,'failures':[],
 'proof':'For zero-trade indices Z, hold other signs fixed and enumerate sigma_Z in{-1,+1}. D_sigma=1+c*sigma dot z; D_0=1+c*s_0 dot z. The probabilities mu_sigma=D_sigma/(2^|Z|*D_0) are positive and sum1, and their weighted gradients c/D_sigma*(sigma,-n*sigma) average to c/D_0*(s_0,-n*s_0). Adjacent ambient branches are reached by perturbing trades t, setting n=1-c*sum|t| and h=n*z-t. The implicit funding function is strictly increasing in n when c*||z_r||_1<1, hence locally unique and Lipschitz. This certifies the chosen ambient Clarke gradient of n, not every feasible one-sided derivative or the Clarke gradient of an arbitrary multi-step network composition.',
 'constraint_scope':'At boundaries or after restriction to a fixed-held/policy curve, some adjacent ambient branches can be inaccessible. The partial-zero absent-L example has chosen dn0 but a negative feasible buy-L right derivative. No unique scalar derivative is claimed at kinks. No minimum-norm claim is made for partial-zero sign0.',
 'numerical_scope':'Exact floating-point trade zero alone selects sign0. Roundoff-sized nonzero trades retain their actual signs; there is no near-zero tolerance override. First-order backward only; higher derivatives not certified.',
 'definition_reference':'https://jmlr.org/papers/volume23/20-899/20-899.pdf (Definition5: generalized gradient as convex hull of limiting gradients)'}
assert source_before==report['source_sha256_after'] and legacy_before==report['legacy_source_sha256_after']
Path('/tmp/implicit_settlement_gradient_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('cases','invalid_inputs')},indent=2))
print('PARTIAL_ZERO_FEASIBLE_SIDE_COUNTEREXAMPLE',json.dumps(partial,indent=2))
