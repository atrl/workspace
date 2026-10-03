"""Neutral execution reuse, never an alternative policy or cost assumption."""
import numpy as np
import pytest
import torch

from src.model.paired_sparse_paths import rollout_costs
from src.model.sparse_holdings_net_excess import SparseHoldingsPolicy, numpy_rollout


@pytest.mark.parametrize('count', [1, 3])
@pytest.mark.parametrize('vertices', [False, True])
def test_every_field_matches_independent_full_owned_books(count, vertices):
    rng = np.random.default_rng(159)
    x = rng.normal(size=(193, 27)); r = rng.uniform(-.08, .09, size=(193, 2))
    models = [SparseHoldingsPolicy(seed=s) for s in (17, 41, 71)[:count]]
    for model in models:
        with torch.no_grad():
            model.K.copy_(torch.from_numpy(rng.normal(0, .15, size=(4, 3))))
            if vertices:
                model.A.copy_(torch.from_numpy(rng.normal(0, 3, size=(4, 8))))
                model.a.copy_(torch.from_numpy(rng.normal(0, 1, size=4)))
    force = np.zeros(len(x), bool); force[[4, 51, 87]] = True
    initial = np.array([.3, .4, .3])
    costs = (10., 25., 0.)
    actual = rollout_costs(models, x, r, costs=costs, initial_weights=initial, force_soxx_mask=force)
    for cost in costs:
        expected = numpy_rollout(models, x, r, cost_bps=cost, initial_weights=initial, force_soxx_mask=force)
        assert set(actual[cost]) == set(expected)
        for field in expected:
            if expected[field].dtype == bool:
                np.testing.assert_array_equal(actual[cost][field], expected[field])
            else:
                np.testing.assert_allclose(actual[cost][field], expected[field], atol=2e-13, rtol=0)
        np.testing.assert_array_equal(actual[cost]['expert_weights'], expected['expert_weights'])
        np.testing.assert_array_equal(actual[cost]['ending_weights'], expected['ending_weights'])
    assert not np.shares_memory(actual[10.]['pretrade_weights'], actual[25.]['pretrade_weights'])
    assert np.any(actual[10.]['fees'] != actual[25.]['fees'])


@pytest.mark.parametrize('expert', [0, 1, 2, 3])
def test_exact_vertices_hold_and_cash_funding(expert):
    model = SparseHoldingsPolicy()
    with torch.no_grad():
        model.A.zero_(); model.a.zero_(); model.K.zero_(); model.a[expert] = 2
    x = np.zeros((81, 27)); r = np.tile([.02, -.01], (81, 1))
    paths = rollout_costs(model, x, r, costs=(25., 10.), initial_weights=[.2, .6, .2])
    for cost, path in paths.items():
        expected = numpy_rollout(model, x, r, cost_bps=cost, initial_weights=[.2, .6, .2])
        for field in expected:
            np.testing.assert_array_equal(path[field], expected[field])
        if expert == 3:
            assert path['exact_hold'].all() and np.count_nonzero(path['fees']) == 0
            np.testing.assert_array_equal(path['pretrade_weights'], path['target_weights'])


@pytest.mark.parametrize('costs', [(), (10,10), (float('nan'),25), (-1,25), (10,10000)])
def test_bad_cost_inventory_fails_closed(costs):
    with pytest.raises(ValueError):
        rollout_costs(SparseHoldingsPolicy(), np.zeros((1,27)), np.zeros((1,2)), costs=costs)
