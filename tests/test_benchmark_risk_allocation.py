from dataclasses import replace
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from src.backtest.position_accounting import settle_target_session
from src.workflow.stage1_economic_action import fee_affordable_target
from src.model.integrated_allocation import AllocationSpec, IntegratedAllocationPolicy, differentiable_settlement
from src.model.benchmark_risk_allocation import (
    BenchmarkRiskAllocationPolicy, fit_policy, parameterize_risk_budget,
    policy_target, replay_policy, risk_budget_components,
)


def _data(rows=40, base_features=4):
    rng = np.random.default_rng(26)
    covariance = np.tile([.0009, .00028, .0001], (rows, 1))
    x = np.column_stack((rng.normal(size=(rows, base_features)), covariance))
    r = rng.normal(0, .01, size=(rows, 2))
    return x, r


def _model(base_features=4, seed=7, market_conditioning=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return BenchmarkRiskAllocationPolicy(base_features, 8, common_feature_indices=tuple(range(base_features)),
            market_conditioning=market_conditioning).double()


def test_feasible_risk_fraction_handles_correlated_psd_covariances():
    rng = np.random.default_rng(124)
    matrices = rng.normal(size=(200, 2, 2)) * .02
    sigma = matrices @ np.swapaxes(matrices, -1, -2)
    cov = np.column_stack((sigma[:, 0, 0], sigma[:, 0, 1], sigma[:, 1, 1]))
    raw = rng.dirichlet(np.ones(3), size=200)
    result = risk_budget_components(torch.tensor(raw), torch.tensor(cov))
    weights = result["weights"].numpy()
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-14, rtol=0)
    assert (weights[:, :2] <= raw[:, :2] + 1e-14).all()
    variance = np.einsum("ni,nij,nj->n", weights[:, :2], sigma, weights[:, :2])
    assert (variance <= cov[:, 2] + 1e-14).all()
    assert (variance <= raw[:, :2].sum(axis=1)**2 * cov[:, 2] + 1e-14).all()


def test_cash_logit_retains_nonzero_gradient_when_composition_budget_binds():
    logits = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    covariance = torch.tensor([.0009, .0003, .0001], dtype=torch.float64)
    result = risk_budget_components(torch.softmax(logits, dim=-1), covariance)
    assert bool(result["composition_budget_binds"])
    torch.testing.assert_close(result["weights"], torch.tensor([1/6, 1/6, 2/3], dtype=torch.float64))
    result["weights"][2].backward()
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[2]) > .1


def test_no_forced_risk_increase_and_zero_variance_gradients_are_finite():
    covariances = torch.tensor([[0., 0., 0.], [.001, 0., 0.], [.0001, -.0001, .0001]], dtype=torch.float64)
    logits = torch.tensor([[.2, .4, -.1], [.2, .4, -.1], [.2, .2, -.1]], dtype=torch.float64, requires_grad=True)
    raw = torch.softmax(logits, dim=-1)
    target = parameterize_risk_budget(raw, covariances)
    assert torch.isfinite(target).all()
    torch.testing.assert_close(target[0], raw[0])
    torch.testing.assert_close(target[1], torch.tensor([0., 0., 1.], dtype=torch.float64))
    assert torch.all(target[:, :2] <= raw[:, :2] + 1e-14)
    target[:, 2].sum().backward()
    assert torch.isfinite(logits.grad).all()
    all_cash = torch.tensor([0., 0., 1.], dtype=torch.float64)
    torch.testing.assert_close(parameterize_risk_budget(all_cash, covariances[1]), all_cash)


def test_risk_mapping_and_cash_funding_have_finite_correct_gradients():
    covariance = torch.tensor([[.0009, .00028, .0001]], dtype=torch.float64)
    current = torch.tensor([[.1, .5, .4]], dtype=torch.float64)
    returns = torch.tensor([[.02, .007, 0.]], dtype=torch.float64)
    logits = torch.tensor([[.2, -.1, .3]], dtype=torch.float64, requires_grad=True)

    def objective(value):
        desired = parameterize_risk_budget(torch.softmax(value, dim=-1), covariance)
        return differentiable_settlement(current, desired, returns, cost_bps=25.)["net_equity"]

    assert torch.autograd.gradcheck(objective, (logits,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_base_initialization_is_unchanged_and_covariance_is_not_a_network_input():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        plain = IntegratedAllocationPolicy(4, 8, common_feature_indices=(0, 1, 2, 3), auxiliary_target_count=2).double()
    wrapped = _model()
    assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in wrapped.parameters())
    for key, tensor in plain.state_dict().items():
        torch.testing.assert_close(tensor, wrapped.base_policy.state_dict()[key], atol=0, rtol=0)
    seen = []
    handle = wrapped.base_policy.allocator.network[0].register_forward_pre_hook(lambda module, values: seen.append(values[0].detach().clone()))
    x, _ = _data(1)
    changed = x.copy(); changed[:, -3:] *= 4
    wrapped(torch.tensor(x), torch.tensor([[0., 0., 1.]], dtype=torch.float64))
    wrapped(torch.tensor(changed), torch.tensor([[0., 0., 1.]], dtype=torch.float64))
    handle.remove()
    assert seen[0].shape[-1] == 7
    torch.testing.assert_close(seen[0], seen[1], atol=0, rtol=0)


def test_training_replay_and_numpy_funding_match_at_both_costs():
    x, returns = _data(12)
    model = _model()
    for cost in (10., 25.):
        actual = replay_policy(model, x, returns, cost_bps=cost)
        weights = np.array([0., 0., 1.]); equity = 1.
        for index, row in actual.iterrows():
            target = policy_target(model, x[index], weights)
            execution = fee_affordable_target(target, pretrade_weights=weights, cost_bps=cost)
            settled = settle_target_session(weights, open_target_weights=execution,
                asset_returns=np.r_[returns[index], 0.], cost_bps=cost)
            equity *= settled["net_equity"]
            np.testing.assert_allclose([row.strategy_return, row.turnover, row.fees, row.equity],
                [settled["net_return"], settled["turnover"], settled["fees"], equity], atol=1e-10, rtol=0)
            np.testing.assert_allclose(settled["open_net_weights"], target, atol=1e-10, rtol=0)
            weights = settled["ending_weights"]


def test_cost_stress_recomputes_the_own_path_and_future_suffix_cannot_change_prior_targets():
    x, returns = _data(15)
    model = _model()
    normal = replay_policy(model, x, returns, cost_bps=10)
    stress = replay_policy(model, x, returns, cost_bps=25)
    assert stress.fees.sum() > normal.fees.sum()
    assert not np.array_equal(normal.equity.to_numpy(), stress.equity.to_numpy())
    changed = x.copy(); changed[8:, :4] = 100; changed[8:, -3:] *= 3
    later_returns = returns.copy(); later_returns[7:] = [.2, -.1]
    altered = replay_policy(model, changed, later_returns, cost_bps=10)
    columns = ["soxl_weight", "soxx_weight", "cash_weight"]
    pd.testing.assert_frame_equal(normal.loc[:7, columns], altered.loc[:7, columns])


def test_no_signal_control_inputs_are_literal_zeros_but_covariance_still_matters():
    model = _model(market_conditioning=False)
    model.base_policy.feature_mean.fill_(40)
    seen = []
    handle = model.base_policy.allocator.network[0].register_forward_pre_hook(lambda module, values: seen.append(values[0].detach().clone()))
    x, _ = _data(1)
    first = policy_target(model, x[0], np.array([0., 0., 1.]))
    changed = x[0].copy(); changed[:4] = -500
    second = policy_target(model, changed, np.array([.4, .3, .3]))
    np.testing.assert_array_equal(first, second)
    changed[-3:] = [.0001, .0001, .0001]
    third = policy_target(model, changed, np.array([.4, .3, .3]))
    handle.remove()
    assert all(torch.count_nonzero(value) == 0 for value in seen)
    assert not np.allclose(first, third)


def test_repeat_seed_empty_pit_and_pickle_replay_are_deterministic():
    x, returns = _data()
    spec = AllocationSpec(epochs=2, block_length=10, warmup_sessions=3, batch_blocks=2)
    first, audit = fit_policy(x, returns, spec=spec, seed=17)
    second, _ = fit_policy(x, returns, spec=spec, seed=17)
    empty, _ = fit_policy(x, returns, spec=replace(spec, auxiliary_coefficient=.05), seed=17)
    for key, tensor in first.state_dict().items():
        torch.testing.assert_close(tensor, second.state_dict()[key], atol=0, rtol=0)
        torch.testing.assert_close(tensor, empty.state_dict()[key], atol=0, rtol=0)
    assert audit["network_market_feature_count"] == 4 and audit["feature_count"] == 7
    assert not audit["covariance_enters_network"]
    pd.testing.assert_frame_equal(replay_policy(first, x, returns), replay_policy(pickle.loads(pickle.dumps(first)), x, returns))


def test_invalid_covariance_and_auxiliary_control_fail_closed():
    raw = torch.tensor([.2, .5, .3], dtype=torch.float64)
    with pytest.raises(ValueError, match="positive semidefinite"):
        parameterize_risk_budget(raw, torch.tensor([.0001, .0002, .0001], dtype=torch.float64))
    with pytest.raises(ValueError, match="negative"):
        parameterize_risk_budget(raw, torch.tensor([-.0001, 0., .0001], dtype=torch.float64))
    with pytest.raises(ValueError, match="auxiliary learning"):
        fit_policy(*_data(), spec=AllocationSpec(epochs=1, block_length=10, warmup_sessions=3, auxiliary_coefficient=.05),
                   seed=17, market_conditioning=False)
