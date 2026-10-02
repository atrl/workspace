"""Synthetic accounting/gradient contracts; no market fits or policy selection."""
from dataclasses import replace

import numpy as np
import pytest
import torch

import src.model.benchmark_anchor_utility as m
from src.model.integrated_allocation import _settle
from src.workflow.stage1_economic_action import fee_affordable_target
from src.backtest.position_accounting import settle_target_session


def fixture(n=127):
    rng = np.random.default_rng(246)
    x = rng.normal(size=(n, 27))
    underlying = rng.normal(.0002, .009, n)
    r = np.c_[2.8 * underlying + rng.normal(0, .001, n), underlying]
    return x, r


def independent_path(alpha, returns, cost, initial=None):
    h = np.array([0., 0., 1.]) if initial is None else initial.copy()
    rows = []
    for a, r in zip(alpha, returns):
        z = (1-a.sum()) * np.array([0., 1., 0.]) + a[0] * np.array([1., 0., 0.]) + a[1] * np.array([0., 0., 1.]) + a[2] * h
        executable = fee_affordable_target(z, pretrade_weights=h, cost_bps=cost)
        settled = settle_target_session(h, open_target_weights=executable, asset_returns=np.r_[r, 0.], cost_bps=cost)
        rows.append({"target_weights": z, "pretrade_weights": h.copy(), "execution_target_weights": executable,
                     **{k: settled[k] for k in ["net_equity", "fees", "turnover", "ending_weights", "open_net_weights"]}})
        h = settled["ending_weights"]
    return {name: np.asarray([row[name] for row in rows]) for name in rows[0]}


@pytest.mark.parametrize("cost", [0., 10., 25.])
@pytest.mark.parametrize("expert", [None, 0, 1, 2])
def test_exact_anchor_experts_and_independent_accounting(cost, expert):
    x, r = fixture()
    model = m.BenchmarkAnchorPolicy(anchor=True)
    if expert is not None:
        with torch.no_grad(): model.theta[expert, 0] = 1.
    alpha = m.numpy_alphas(model, x)
    actual = m.numpy_rollout(model, x, r, cost_bps=cost)
    expected = independent_path(alpha, r, cost)
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, atol=1e-11, rtol=0)
    if expert is None:
        buy_hold = np.cumprod(1 + r[:, 1]) / (1 + cost / 10000)
        np.testing.assert_allclose(actual["equity"], buy_hold, atol=1e-11, rtol=0)
        assert np.max(actual["fees"][1:]) < 1e-14
    if expert in [1, 2]:
        assert np.array_equal(actual["fees"], np.zeros(len(r)))
        assert np.array_equal(actual["net_equity"], np.ones(len(r)))


def test_random_mixtures_true_fees_and_own_state():
    _, r = fixture(320)
    rng = np.random.default_rng(500)
    alpha = rng.dirichlet(np.ones(4), size=len(r))[:, :3]
    initial = np.array([.53, .31, .16])
    actual = m.rollout_alphas(alpha, r, cost_bps=25., initial_weights=initial)
    expected = independent_path(alpha, r, 25., initial)
    for name in expected: np.testing.assert_allclose(actual[name], expected[name], atol=1e-11, rtol=0)
    np.testing.assert_allclose(actual["fees"], .0025 * actual["turnover"], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(actual["pretrade_weights"][1:], actual["ending_weights"][:-1])


def test_scalar_torch_alpha_parity_serialization_and_seed():
    x, _ = fixture(20)
    first = m.BenchmarkAnchorPolicy(seed=71)
    second = m.BenchmarkAnchorPolicy(seed=71)
    assert m.policy_to_dict(first)["tensor_sha256"] == m.policy_to_dict(second)["tensor_sha256"]
    assert sum(p.numel() for p in first.parameters()) == 251
    assert float(first.theta.sum()) == pytest.approx(.5)
    torch_alpha = first.alphas(torch.from_numpy(x)).detach().numpy()
    np.testing.assert_allclose(m.numpy_alphas(first, x), torch_alpha, atol=2e-16, rtol=0)
    restored = m.policy_from_dict(m.policy_to_dict(first))
    np.testing.assert_array_equal(m.numpy_alphas(first, x), m.numpy_alphas(restored, x))
    damaged = m.policy_to_dict(first)
    damaged["state_dict"]["hidden.bias"][0] += .1
    with pytest.raises(ValueError, match="hash mismatch"): m.policy_from_dict(damaged)


def test_equal_target_ensemble_uses_same_held_book():
    models = [m.BenchmarkAnchorPolicy(seed=s) for s in [17, 41, 71]]
    x, _ = fixture(1)
    held = np.array([.8, .03, .17])
    expected = np.mean([model(torch.from_numpy(x[0]), torch.from_numpy(held)).detach().numpy() for model in models], axis=0)
    np.testing.assert_allclose(m.policy_target(models, x[0], held), expected, atol=2e-16, rtol=0)


def test_causal_suffix_and_contemporaneous_forced_anchor():
    x, r = fixture(100)
    model = m.BenchmarkAnchorPolicy()
    base = m.numpy_rollout(model, x, r, cost_bps=10)
    altered = x.copy(); altered[40:] += 100
    future_r = r.copy(); future_r[40:] *= -1
    changed = m.numpy_rollout(model, altered, future_r, cost_bps=10)
    for name in base: np.testing.assert_array_equal(base[name][:40], changed[name][:40])
    mask = np.zeros(100, dtype=bool); mask[60] = True
    forced = m.numpy_rollout(model, x, r, cost_bps=10, force_anchor_mask=mask)
    np.testing.assert_array_equal(base["net_equity"][:60], forced["net_equity"][:60])
    np.testing.assert_array_equal(forced["target_weights"][60], [0., 1., 0.])
    np.testing.assert_array_equal(forced["pretrade_weights"][61], forced["ending_weights"][60])


@pytest.mark.parametrize("n,expected", [(63, [(0,63)]), (64, [(0,63),(1,64)]), (126,[(0,63),(63,126)]),
    (127,[(0,63),(63,126),(64,127)]), (252,[(0,63),(63,126),(126,189),(189,252)])])
def test_risk_groups_no_duplicate_exact_final_group(n, expected):
    assert m.risk_groups(n) == expected


def test_global_objective_and_daily_adjoints_match_autograd():
    spec = m.AnchorUtilitySpec(minimum_rows=63)
    rng = np.random.default_rng(8)
    for n in [63,64,126,127,252,253]:
        arr = rng.normal(-.0002, .007, n)
        x = torch.tensor(arr, requires_grad=True)
        sums = torch.stack([x[a:b].sum() for a,b in m.risk_groups(n)])
        utility = 63/n*x.sum() - torch.sqrt(torch.relu(-sums).square().mean()+1e-12)
        utility.backward()
        actual = m.utility_and_daily_weights(arr, spec=spec)
        assert actual["utility"] == pytest.approx(float(utility), abs=1e-14)
        np.testing.assert_allclose(actual["daily_weights"], x.grad.numpy(), atol=1e-14, rtol=0)
        assert actual["global_mean_observations"] == n


def full_torch_objective(model, x, r, spec, anchors):
    utilities = []
    for cost in spec.cost_bps:
        held = torch.tensor([0., 0., 1.], dtype=torch.float64)
        multipliers = []
        for f, ret in zip(torch.from_numpy(x), torch.from_numpy(r)):
            target = model(f, held)
            result = _settle(held, target, torch.cat((ret, torch.zeros(1,dtype=torch.float64))), cost/10000)
            multipliers.append(result["net_equity"])
            held = result["ending_weights"]
        relative = torch.log(torch.stack(multipliers)) - torch.log(torch.from_numpy(anchors[cost]["net_equity"]))
        sums = torch.stack([relative[a:b].sum() for a,b in m.risk_groups(len(r),spec.risk_group_length)])
        utility = spec.risk_group_length/len(r)*relative.sum()-torch.sqrt(torch.relu(-sums).square().mean()+spec.rms_epsilon)
        utilities.append(utility + np.sqrt(spec.rms_epsilon))
    return torch.stack(utilities).min()


def test_short_full_context_gradient_matches_full_bptt_and_hidden_route():
    x, r = fixture(13)
    spec = m.AnchorUtilitySpec(risk_group_length=4, gradient_chunk_length=4, context_sessions=20,minimum_rows=4,updates=1)
    model = m.BenchmarkAnchorPolicy(seed=17)
    anchors = {c:m.rollout_alphas(np.zeros((len(r),3)),r,cost_bps=c) for c in spec.cost_bps}
    evaluated = m.evaluate_policy(model,x,r,spec=spec,anchors=anchors)
    diagnostics = m.truncated_utility_gradient(model,x,r,evaluated,spec=spec)
    chunk_gradients = [p.grad.clone() for p in model.parameters()]
    assert diagnostics["scored_rows"] == len(r)
    assert diagnostics["forward_parity_max_error"] < 1e-12
    assert diagnostics["hidden_gradient_norm"] > 0
    model.zero_grad(set_to_none=True)
    loss = -full_torch_objective(model,x,r,spec,anchors)
    loss.backward()
    for actual,p in zip(chunk_gradients,model.parameters()): torch.testing.assert_close(actual,p.grad,atol=1e-12,rtol=1e-9)


@pytest.mark.parametrize("cost", [10.,25.])
def test_documented_no_trade_boundary_autograd_is_not_right_derivative(cost):
    theta = torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    held = torch.tensor([0.,1.,0.],dtype=torch.float64)
    target = torch.stack((theta,1-theta,theta*0))
    result = _settle(held,target,torch.zeros(3,dtype=torch.float64),cost/10000)
    torch.log(result["net_equity"]).backward()
    assert float(theta.grad) == 0
    step=1e-7
    zero=m.rollout_alphas(np.array([[0.,0.,0.]]),np.zeros((1,2)),cost_bps=cost,initial_weights=held.numpy())
    right=m.rollout_alphas(np.array([[step,0.,0.]]),np.zeros((1,2)),cost_bps=cost,initial_weights=held.numpy())
    derivative=(np.log(right['net_equity'][0])-np.log(zero['net_equity'][0]))/step
    assert derivative == pytest.approx(-2*(cost/10000)/(1-cost/10000),abs=2e-9)


def test_projection_kkt_feasibility_and_idempotence():
    v=torch.tensor([-.3,.2,1.1,.7],dtype=torch.float64)
    p=m.project_theta(v)
    assert bool((p>=0).all()) and float(p.sum()) <= 1
    active=p>0
    torch.testing.assert_close((v-p)[active],torch.full_like(v[active],(v-p)[active][0]))
    assert bool((v[~active] <= (v-p)[active][0]).all())
    torch.testing.assert_close(m.project_theta(p),p)
    torch.testing.assert_close(m.project_theta(torch.tensor([-.2,.3])),torch.tensor([0.,.3]))


def test_fit_incumbent_audits_all_updates_and_repeats_exactly():
    x,r=fixture(64)
    spec=m.AnchorUtilitySpec(updates=2,minimum_rows=63)
    a,info=m.fit_policy(x,r,seed=41,spec=spec)
    b,other=m.fit_policy(x,r,seed=41,spec=spec)
    assert len(info['epochs'])==3
    assert info['failure'] is None
    assert info['centered_worst_utility'] >= 0
    assert min(info['centered_utility_by_cost'].values()) >= 0
    assert info['centered_worst_utility'] == max([0.]+[h['J'] for h in info['epochs']])
    assert m.policy_to_dict(a)['tensor_sha256']==m.policy_to_dict(b)['tensor_sha256']
    assert info['selected_epoch']==other['selected_epoch']
    if info['selected_epoch']!='anchor':
        assert m.policy_to_dict(a)['tensor_sha256']==info['epochs'][info['selected_epoch']]['checkpoint']['tensor_sha256']


def test_null_remains_available_and_ensemble_uses_own_full_path():
    x=np.zeros((64,27));r=np.tile([-.002,.002],(64,1));spec=m.AnchorUtilitySpec(updates=1,minimum_rows=63)
    model,audit=m.fit_policy(x,r,seed=17,spec=spec)
    assert audit['selected_epoch']=='anchor' and audit['null_selected']
    models=[m.BenchmarkAnchorPolicy(seed=s,anchor=True) for s in [17,41,71]]
    for candidate in models:
        with torch.no_grad():candidate.theta[1,0]=1
    selected,ensemble=m.evaluate_ensemble(models,x,r,spec=spec)
    assert len(selected)==1 and ensemble['anchor_selected']
    assert ensemble['ensemble_rejected_for_training_utility']
    assert ensemble['centered_worst_utility']==0
    assert ensemble['component_count']==3


def test_numerical_failure_is_disqualified_not_silently_selected_null(monkeypatch):
    x,r=fixture(64);spec=m.AnchorUtilitySpec(updates=1,minimum_rows=63)
    def fail(*args,**kwargs):raise m.NumericalFitFailure('synthetic bad gradient')
    monkeypatch.setattr(m,'truncated_utility_gradient',fail)
    model,info=m.fit_policy(x,r,seed=17,spec=spec)
    assert info['failure']['epoch']==1
    assert not info['qualification_eligible']
    assert info['null_selected'] and torch.count_nonzero(model.theta)==0
    def bug(*args,**kwargs):raise RuntimeError('programming error')
    monkeypatch.setattr(m,'truncated_utility_gradient',bug)
    with pytest.raises(RuntimeError,match='programming error'):m.fit_policy(x,r,seed=17,spec=spec)


def test_invalid_source_and_policy_fail_closed():
    x,r=fixture(64);x[4,0]=np.nan
    with pytest.raises(ValueError,match='finite'):m.fit_policy(x,r,seed=17,spec=m.AnchorUtilitySpec(minimum_rows=63))
    with pytest.raises(ValueError):m.rollout_alphas(np.zeros((64,3)),np.full((64,2),-1.),cost_bps=10)
    with pytest.raises(ValueError):m.rollout_alphas(np.ones((64,3)),r,cost_bps=10)
    with pytest.raises(ValueError):m.risk_groups(62)
