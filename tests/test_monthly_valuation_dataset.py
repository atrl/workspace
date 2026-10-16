import pandas as pd
import pytest

from src.data.monthly_valuation_dataset import align_monthly_valuations


def fixture():
    dates = pd.date_range("2020-01-31", periods=5, freq="ME", tz="America/New_York") + pd.Timedelta(hours=16)
    events = pd.DataFrame({"valuation_at": dates, "available_at": dates,
        "observation_id": [str(i) for i in range(5)], "source_id": [str(i) for i in range(5)],
        "method_id": "signed_basket_ep", "valuation_earnings_yield": [-.02, .01, .02, .03, .04],
        "earnings_change": .1, "loss_weight": .2, "coverage_weight": .97})
    return events, pd.DataFrame({"signal_at": dates+pd.Timedelta(minutes=15)})


def test_future_revision_does_not_rewrite_rank_and_losses_are_retained():
    events, decisions = fixture()
    base = align_monthly_valuations(events, decisions, minimum_snapshots=2)
    revision = events.iloc[[0]].copy()
    # A later filing of an OLD period must not rewind the current basket, even
    # when available before the final decision (observed in real NPORT data).
    revision["available_at"] = decisions.signal_at.iloc[-1]-pd.Timedelta(minutes=5)
    revision["source_id"] = "restatement"
    revision["valuation_earnings_yield"] = 100.
    actual = align_monthly_valuations(pd.concat([events, revision]), decisions, minimum_snapshots=2)
    pd.testing.assert_frame_equal(base, actual)
    assert actual.iloc[0].valuation_earnings_yield < 0
    assert actual.iloc[2].earnings_yield_percentile == 1
    assert actual.iloc[2].unknown_earnings_weight == pytest.approx(.03)
    incomplete = events.copy(); incomplete.loc[2, "coverage_weight"] = .8
    assert not align_monthly_valuations(incomplete, decisions, minimum_snapshots=2).iloc[2].eligible


def test_repeated_snapshot_method_change_and_naive_publication_are_not_history():
    events, decisions = fixture()
    repeated = align_monthly_valuations(events.iloc[:1], decisions, minimum_snapshots=2, maximum_age_days=365)
    assert not repeated.eligible.any()
    assert repeated.rank_prior_snapshots.max() == 0
    events.loc[2:, "method_id"] = "changed_loss_treatment"
    changed = align_monthly_valuations(events, decisions, minimum_snapshots=2)
    assert not changed.iloc[2].eligible
    assert changed.iloc[4].eligible
    events["available_at"] = events.available_at.dt.tz_localize(None)
    with pytest.raises(ValueError, match="explicit timezone"):
        align_monthly_valuations(events, decisions, minimum_snapshots=2)
