from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.model.online_expert_allocation import FixedShareAllocator, FixedShareConfig


UTC = timezone.utc
START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _config(**overrides):
    config = FixedShareConfig(
        expert_names=("benchmark", "leveraged", "defensive"),
        expert_weights=((0., 1., 0.), (.2, .8, 0.), (0., .8, .2)),
        anchor_weights=(0., 1., 0.), prior=(1 / 3, 1 / 3, 1 / 3),
        active_fraction=.5, eta=1., share=1 / 252,
    )
    return replace(config, **overrides)


def _update(allocator, index=0, returns=(.01, .02, -.01), benchmark=.01):
    available = START + timedelta(days=index)
    return allocator.update(str(index), available, available + timedelta(seconds=1), returns, benchmark)


def test_first_update_matches_net_relative_wealth_and_anchor_blend():
    config = _config(prior=(.2, .3, .5), eta=2., share=.1)
    allocator = FixedShareAllocator(config)
    initial = allocator.target(START)
    np.testing.assert_allclose(initial, [.03, .92, .05], atol=1e-15)
    updated = _update(allocator)
    growth = np.array(config.prior) * (np.array([1.01, 1.02, .99]) / 1.01) ** 2
    expected = .9 * growth / growth.sum() + .1 * np.array(config.prior)
    np.testing.assert_allclose(updated, expected, atol=1e-15)
    np.testing.assert_allclose(allocator.target(START + timedelta(seconds=1)),
                               .5 * np.array(config.anchor_weights)
                               + .5 * expected @ np.array(config.expert_weights), atol=1e-15)
    assert allocator.observation_count == 1
    assert allocator.last_available_at == START
    assert allocator.last_decision_at == START + timedelta(seconds=1)


def test_equal_net_returns_leave_prior_unchanged_and_benchmark_is_common_reference():
    equal = FixedShareAllocator(_config(prior=(.2, .3, .5)))
    for index in range(20):
        _update(equal, index, returns=(.01, .01, .01))
    np.testing.assert_allclose(equal.posterior, [.2, .3, .5], atol=1e-15)
    # A common benchmark subtracts the same log return from every expert.
    first, second = FixedShareAllocator(_config()), FixedShareAllocator(_config())
    for index in range(10):
        _update(first, index, benchmark=.02)
        _update(second, index, benchmark=-.4)
    np.testing.assert_allclose(first.posterior, second.posterior, atol=1e-15)


@pytest.mark.parametrize("overrides", [{"active_fraction": 0.},
    {"expert_weights": ((0., 1., 0.),) * 3}])
def test_benchmark_null_allocation_is_exact(overrides):
    allocator = FixedShareAllocator(_config(**overrides))
    for index in range(10):
        _update(allocator, index)
        np.testing.assert_allclose(allocator.target(START + timedelta(days=index, seconds=1)),
                                   [0., 1., 0.], atol=1e-15)


def test_share_one_resets_prior_and_share_floor_preserves_normalized_simplex():
    reset = FixedShareAllocator(_config(share=1., prior=(.1, .2, .7)))
    np.testing.assert_allclose(_update(reset), [.1, .2, .7], atol=1e-15)
    allocator = FixedShareAllocator(_config(eta=1e6))
    for index in range(30):
        weights = _update(allocator, index)
        assert np.isfinite(weights).all()
        assert (weights >= np.array(allocator.config.prior) * allocator.config.share - 1e-15).all()
        assert weights.sum() == pytest.approx(1., abs=1e-15)
        target = allocator.target(START + timedelta(days=index, seconds=1))
        assert target.sum() == pytest.approx(1., abs=1e-15)
        assert (target >= 0).all()


def test_extreme_legal_returns_and_log_state_do_not_underflow_away_recovering_expert():
    allocator = FixedShareAllocator(_config(share=0., eta=1.))
    low = np.nextafter(-1., 0.)
    high = np.finfo(float).max
    for index in range(3):
        _update(allocator, index, returns=(high, low, 0.), benchmark=low)
        assert np.isfinite(allocator.posterior).all()
    assert allocator.posterior[1] == 0.  # Display underflows, stored log mass does not.
    for index in range(3, 6):
        _update(allocator, index, returns=(low, high, 0.), benchmark=high)
    np.testing.assert_allclose(allocator.posterior[:2], [.5, .5], atol=1e-12)
    assert allocator.posterior.sum() == pytest.approx(1.)
    huge_eta = FixedShareAllocator(_config(eta=high))
    assert np.isfinite(_update(huge_eta, returns=(high, low, 0.))).all()


def test_future_suffix_perturbation_cannot_change_any_earlier_target():
    returns = np.array([[.01, .03, -.005], [-.02, -.03, -.01], [.02, .04, .01],
                        [-.01, -.02, -.005], [.01, .02, 0.], [.02, .03, .01]])
    changed = returns.copy()
    changed[3:] = [4., -.99, .2]

    def replay(events):
        allocator = FixedShareAllocator(_config())
        targets = []
        for index, row in enumerate(events):
            decision = START + timedelta(days=index)
            if index:
                allocator.update(str(index - 1), decision - timedelta(seconds=1), decision,
                                 events[index - 1], events[index - 1, 0])
            targets.append(allocator.target(decision))
        return np.array(targets)

    first, second = replay(returns), replay(changed)
    np.testing.assert_array_equal(first[:4], second[:4])
    assert not np.array_equal(first[4:], second[4:])


def test_mature_events_can_share_availability_and_decision_and_timezone_is_absolute():
    allocator = FixedShareAllocator(_config())
    eastern = timezone(timedelta(hours=-5))
    available = START.astimezone(eastern)
    decision = START + timedelta(seconds=1)
    allocator.update("a", available, decision, [.01, .02, 0.], .01)
    allocator.update("b", available, decision, [.01, .02, 0.], .01)
    assert allocator.observation_count == 2
    assert allocator.last_available_at == START
    assert allocator.last_available_at.tzinfo == UTC


@pytest.mark.parametrize("failure", ["duplicate", "availability_order", "decision_order",
    "equal_time", "future", "naive_available", "naive_decision", "empty_id"])
def test_timing_and_identity_rejections_leave_state_unchanged(failure):
    allocator = FixedShareAllocator(_config())
    _update(allocator)
    prior = allocator.posterior.copy()
    identity, available, decision = "next", START + timedelta(days=1), START + timedelta(days=2)
    if failure == "duplicate":
        identity = "0"
    elif failure == "availability_order":
        available = START - timedelta(seconds=1)
    elif failure == "decision_order":
        decision = START
    elif failure == "equal_time":
        available = decision
    elif failure == "future":
        available = decision + timedelta(seconds=1)
    elif failure == "naive_available":
        available = available.replace(tzinfo=None)
    elif failure == "naive_decision":
        decision = decision.replace(tzinfo=None)
    elif failure == "empty_id":
        identity = " "
    with pytest.raises(ValueError, match="blocked:"):
        allocator.update(identity, available, decision, [.01, .02, 0.], .01)
    np.testing.assert_array_equal(allocator.posterior, prior)
    assert allocator.observation_count == 1
    assert allocator.last_available_at == START
    assert allocator.last_decision_at == START + timedelta(seconds=1)


@pytest.mark.parametrize("returns,benchmark", [([.01, .02], .01),
    ([[.01, .02, .03]], .01), ([np.nan, .01, .02], .01),
    ([np.inf, .01, .02], .01), ([-1., .01, .02], .01),
    ([-2., .01, .02], .01), ([.01, .02, .03], np.nan),
    ([.01, .02, .03], np.inf), ([.01, .02, .03], -1.),
    ([.01, .02, .03], [.01])])
def test_bad_returns_fail_closed_and_rejected_id_can_be_retried(returns, benchmark):
    allocator = FixedShareAllocator(_config())
    with pytest.raises(ValueError, match="blocked:"):
        _update(allocator, returns=returns, benchmark=benchmark)
    assert allocator.observation_count == 0
    assert allocator.last_available_at is None
    _update(allocator)
    assert allocator.observation_count == 1


def test_target_rejects_naive_and_backdated_queries_and_returned_arrays_are_isolated():
    allocator = FixedShareAllocator(_config())
    before = allocator.posterior
    before[:] = 0.
    assert allocator.posterior.sum() == pytest.approx(1.)
    target = allocator.target(START)
    target[:] = 0.
    assert allocator.target(START).sum() == pytest.approx(1.)
    with pytest.raises(ValueError, match="timezone-aware"):
        allocator.target(START.replace(tzinfo=None))
    with pytest.raises(ValueError, match="backwards"):
        allocator.target(START - timedelta(seconds=1))


@pytest.mark.parametrize("overrides", [{"share": -.1}, {"share": 1.1},
    {"eta": -1.}, {"eta": np.inf}, {"active_fraction": np.nan},
    {"active_fraction": 1.1}, {"prior": (.5, .5)}, {"prior": (.2, .3, .4)},
    {"anchor_weights": (0., 1.1, -.1)}, {"expert_names": ("a", "a", "b")},
    {"expert_weights": ((0., 1., 0.),)}, {"expert_weights": ((0., 1., 0.), (.2, .7, 0.), (0., 1., 0.))}])
def test_invalid_config_fails_closed(overrides):
    with pytest.raises(ValueError, match="blocked:"):
        _config(**overrides)


def test_config_is_immutable_and_zero_prior_mass_stays_disabled():
    allocator = FixedShareAllocator(_config(prior=(.5, .5, 0.)))
    with pytest.raises(FrozenInstanceError):
        allocator.config.eta = 2.
    _update(allocator, returns=(0., 0., 100.))
    np.testing.assert_allclose(allocator.posterior, [.5, .5, 0.], atol=1e-15)
    no_learning = FixedShareAllocator(_config(eta=0.))
    np.testing.assert_allclose(_update(no_learning), [1 / 3] * 3, atol=1e-15)
