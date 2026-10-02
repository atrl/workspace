import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from src.backtest.position_accounting import settle_target_session
from src.workflow.stage1_economic_action import fee_affordable_target
from src.model.integrated_allocation import (
    AllocationSpec, IntegratedAllocationPolicy, differentiable_settlement,
    fit_policy, policy_target, portfolio_objective, replay_policy, rollout_policy,
)


def _data(rows=40, features=4):
    rng = np.random.default_rng(174)
    return rng.normal(size=(rows, features)), rng.normal(0, .01, size=(rows, 2))


def _model(features=4, seed=7):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return IntegratedAllocationPolicy(features, 8, common_feature_indices=tuple(range(features)), auxiliary_target_count=2).double()


@pytest.mark.parametrize("cost", [0., 10., 25., 9000.])
def test_exact_fee_funded_batch_matches_numpy_reference(cost):
    rng = np.random.default_rng(27)
    before = rng.dirichlet(np.ones(3), size=60)
    target = rng.dirichlet(np.ones(3), size=60)
    returns = np.column_stack((rng.uniform(-.4, .4, size=(60, 2)), np.zeros(60)))
    before[:3] = np.eye(3)
    target[:3] = np.roll(np.eye(3), 1, axis=0)
    actual = differentiable_settlement(torch.tensor(before), torch.tensor(target), torch.tensor(returns), cost_bps=cost)
    for index in range(len(before)):
        executable = fee_affordable_target(target[index], pretrade_weights=before[index], cost_bps=cost)
        expected = settle_target_session(before[index], open_target_weights=executable, asset_returns=returns[index], cost_bps=cost)
        for name in ("net_equity", "net_return", "ending_weights", "turnover", "fees", "open_net_weights", "open_net_equity"):
            np.testing.assert_allclose(actual[name][index].numpy(), expected[name], atol=1e-10, rtol=0)


def test_hold_retains_quantities_and_true_risky_turnover_counts_both_legs():
    before = torch.tensor([.6, .3, .1], dtype=torch.float64)
    returns = torch.tensor([.1, -.02, .0], dtype=torch.float64)
    held = differentiable_settlement(before, None, returns, cost_bps=25)
    repeated = differentiable_settlement(before, before, returns, cost_bps=25)
    assert held["fees"] == 0 and held["turnover"] == 0
    np.testing.assert_array_equal(held["ending_values"].numpy(), (before * (1 + returns)).numpy())
    np.testing.assert_allclose(held["ending_values"], repeated["ending_values"], atol=1e-14)
    switch = differentiable_settlement(torch.tensor([1., 0., 0.], dtype=torch.float64),
        torch.tensor([0., 1., 0.], dtype=torch.float64), torch.zeros(3, dtype=torch.float64), cost_bps=25)
    expected_net = (1 - .0025) / (1 + .0025)
    assert float(switch["turnover"]) == pytest.approx(1 + expected_net)
    assert float(switch["fees"]) == pytest.approx(.0025 * (1 + expected_net))
    assert float(switch["ending_weights"][2]) == pytest.approx(0., abs=1e-12)


def test_piecewise_cost_has_finite_correct_gradients():
    pretrade = torch.tensor([[.2, .5, .3], [.4, .1, .5]], dtype=torch.float64)
    returns = torch.tensor([[.03, -.02, 0], [-.01, .01, 0]], dtype=torch.float64)
    logits = torch.tensor([[.4, -.2, .1], [-.3, .7, .2]], dtype=torch.float64, requires_grad=True)

    def objective(value):
        target = torch.softmax(value, dim=-1)
        return differentiable_settlement(pretrade, target, returns, cost_bps=25)["net_equity"]

    assert torch.autograd.gradcheck(objective, (logits,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_rollout_target_cannot_see_current_return_or_future_suffix():
    x, r = _data(12)
    model = _model()
    first = replay_policy(model, x, r, cost_bps=10)
    x2, r2 = x.copy(), r.copy()
    x2[7:] = 20
    r2[6:] = [.3, -.2]
    second = replay_policy(model, x2, r2, cost_bps=10)
    columns = ["soxl_weight", "soxx_weight", "cash_weight"]
    pd.testing.assert_frame_equal(first.loc[:6, columns], second.loc[:6, columns])
    pd.testing.assert_frame_equal(first.iloc[:6], second.iloc[:6])


def test_replay_capital_scale_and_serialized_model_identity():
    x, r = _data(8)
    model = _model()
    first = replay_policy(model, x, r, cost_bps=10, startcash=1)
    second = replay_policy(pickle.loads(pickle.dumps(model)), x, r, cost_bps=10, startcash=100000)
    np.testing.assert_array_equal(first.strategy_return, second.strategy_return)
    np.testing.assert_array_equal(first.turnover, second.turnover)
    np.testing.assert_allclose(second.equity, first.equity * 100000, rtol=0, atol=1e-10)


def test_ensemble_uses_equal_targets_with_same_actual_holdings():
    x, _ = _data(1)
    first, second = _model(seed=1), _model(seed=2)
    weights = np.array([.2, .7, .1])
    target = policy_target([first, second], x[0], weights)
    expected = .5 * (first.predict_target(x[0], weights) + second.predict_target(x[0], weights))
    np.testing.assert_allclose(target, expected, atol=1e-15, rtol=0)
    assert target.sum() == pytest.approx(1., abs=1e-15)


def test_objective_excludes_warmup_and_uses_registered_block_rms():
    net = torch.tensor([[.9, 1.10, 1.02], [1.2, .96, 1.01]], dtype=torch.float64)
    benchmark = torch.tensor([[.05, .01, .01], [.03, .02, -.01]], dtype=torch.float64)
    rollout = {"mask": torch.ones_like(net, dtype=torch.bool), "net_equity": net, "turnover": torch.zeros_like(net)}
    loss, _ = portfolio_objective(rollout, benchmark, downside_coefficient=1., cost_bps=25., warmup=1)
    excess = (np.log(net.numpy()[:, 1:]) - np.log1p(benchmark.numpy()[:, 1:])).sum(axis=1)
    expected = -excess.mean() + np.sqrt(np.maximum(-excess, 0).dot(np.maximum(-excess, 0)) / 2 + 1e-12)
    assert float(loss) == pytest.approx(expected, abs=1e-15)


def test_seed_repeat_and_pre_pit_empty_auxiliary_equal_primary_only():
    x, r = _data()
    spec = AllocationSpec(epochs=2, block_length=10, warmup_sessions=3, batch_blocks=2)
    first, audit = fit_policy(x, r, spec=spec, seed=17)
    repeat, _ = fit_policy(x, r, spec=spec, seed=17)
    empty_aux = {"features": np.empty((0, 4)), "targets": np.empty((0, 2)), "common_feature_indices": [0, 1, 2, 3]}
    empty, empty_audit = fit_policy(x, r, spec=replace(spec, auxiliary_coefficient=.05), seed=17, auxiliary=empty_aux)
    for key, tensor in first.state_dict().items():
        torch.testing.assert_close(tensor, repeat.state_dict()[key], atol=0, rtol=0)
        torch.testing.assert_close(tensor, empty.state_dict()[key], atol=0, rtol=0)
    assert audit["training_block_start_rows"][-1] + 13 == len(x)
    assert empty_audit["auxiliary_empty_fallback"] and not empty_audit["auxiliary_active"]


def test_auxiliary_uses_only_explicit_common_columns_and_trains_shared_projection():
    x, r = _data()
    spec = AllocationSpec(epochs=2, block_length=10, warmup_sessions=3, auxiliary_coefficient=.05)
    auxiliary = {"features": x[:, :2], "targets": np.column_stack((r[:, 0], np.abs(r[:, 1]))),
                 "common_feature_indices": [0, 1], "sample_weight": np.ones(len(x))}
    model, audit = fit_policy(x, r, spec=spec, seed=17, auxiliary=auxiliary)
    assert audit["auxiliary_active"] and audit["auxiliary_target_count"] == 2
    model.zero_grad()
    model.auxiliary_prediction(torch.tensor(x[:, :2])).sum().backward()
    gradient = model.allocator.network[0].weight.grad
    assert bool(torch.isfinite(gradient).all())
    assert gradient[:, :2].abs().sum() > 0
    assert gradient[:, 2:].abs().sum() == 0


def test_auxiliary_target_scaler_respects_per_date_sample_mass():
    x, r = _data()
    targets = np.array([[1., 2.], [3., 4.], [8., 10.]])
    weights = np.array([.5, .5, 1.])
    auxiliary = {"features": x[:3, :2], "targets": targets,
                 "common_feature_indices": [0, 1], "sample_weight": weights}
    spec = AllocationSpec(epochs=1, block_length=10, warmup_sessions=3, auxiliary_coefficient=.05)
    model, audit = fit_policy(x, r, spec=spec, seed=17, auxiliary=auxiliary)
    expected = (targets * weights[:, None]).sum(0) / weights.sum()
    np.testing.assert_allclose(model.auxiliary_target_mean.numpy(), expected, atol=1e-15, rtol=0)
    assert audit["auxiliary_target_sha256"] is not None


def test_missing_data_and_invalid_weight_paths_fail_closed():
    x, r = _data(5)
    x[2, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        replay_policy(_model(), x, r)
    with pytest.raises(ValueError, match="short"):
        differentiable_settlement(torch.tensor([1.1, -.1, 0.]), None, torch.zeros(3), cost_bps=10)
    with pytest.raises(ValueError, match="complete training block"):
        fit_policy(*_data(5), spec=AllocationSpec(), seed=17)
