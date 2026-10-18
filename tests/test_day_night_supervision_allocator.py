"""One concentrated synthetic contract. No real Dataset, model, price or fit."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.day_night_labels import CLOCKS, TARGETS, original_target_digest
from src.model.day_night_supervision_allocator import (
    DayNightSupervisionAllocator, append_supervision_loss, clip_once,
    fit_supervision_scalers, initialization_contract, primary_forward_with_state,
    transform_supervision_target,
)
from src.model.minute_pit_allocator import MinutePitAllocator
from src.workflow.day_night_supervision import append_supervision
from src.workflow.pit_constituent_auxiliary_training_allocation import (
    COMMON_FEATURES, PitAuxiliaryAllocationSpec,
)


def test_paired_supervision_alignment_scale_initialization_inference_and_global_clip(monkeypatch):
    # Real schema, fabricated values: original PG85, source original8+derived11.
    sessions = pd.bdate_range("2019-01-02", periods=11)
    dates = pd.DatetimeIndex(sessions[:5].to_numpy(), name="signal_date")
    mapping = pd.DataFrame({f"d{j}": sessions[1+j:6+j] for j in range(6)}, index=dates)
    meta = pd.DataFrame({name: pd.DatetimeIndex(mapping[day]).tz_localize("America/New_York")
        + pd.Timedelta(hours=9, minutes=minute)
        for name, day, minute in zip(CLOCKS, ("d0", "d1", "d5"), (29, 31, 31))})
    labels = pd.DataFrame({name: np.arange(5, dtype=float) / (100 + n)
                           for n, name in enumerate(TARGETS)})
    base_features = [f"x{i}" for i in range(39)]
    features = pd.DataFrame(np.arange(5 * 61, dtype=float).reshape(5, 61) / 100.,
                            columns=base_features + [f"unused{i}" for i in range(22)])
    common = pd.DataFrame(np.linspace(-1., 1., 5 * 16).reshape(5, 16), columns=COMMON_FEATURES)
    primary = pd.concat({"feature": features, "label": labels, "common": common, "meta": meta}, axis=1)
    primary.index = pd.MultiIndex.from_arrays([dates, ["SOXL_SOXX_CASH"] * 5], names=["datetime", "instrument"])
    reference = pd.concat([meta, labels], axis=1); reference.index = dates
    derived_columns = ["D_SOXL", "N_SOXL", "endpoint_SOXL", "D_SOXX", "N_SOXX", "endpoint_SOXX",
                       "D_relative", "N_relative", "S", "Z", "S_endpoint"]
    derived = pd.DataFrame(np.zeros((5, 11), dtype=np.float64), index=dates, columns=derived_columns)
    derived["S"] = np.array([-2., -1., 0., 1., 2.])
    derived["Z"] = np.array([-4., 3., 0., -3., 4.])
    source = pd.concat({"original": reference, "fixed_day_night_label": derived}, axis=1)
    source.index = pd.MultiIndex.from_arrays([dates, ["SOXL_SOXX"] * 5], names=["datetime", "instrument"])
    assert primary.shape == (5, 85) and source.shape == (5, 19)
    augmented, raw, audit = append_supervision(primary, source, reference, mapping,
        expected_rows=5, expected_original_columns=85, expected_digest=original_target_digest(reference))
    pd.testing.assert_frame_equal(augmented.iloc[:, :85], primary, check_exact=True)
    pd.testing.assert_index_equal(augmented.index, primary.index, exact=True)
    assert list(augmented.columns[85:]) == [("supervision", "S"), ("supervision", "Z")]
    assert audit["datetime_one_to_one_exact_order"] and not audit["join_drop_fill_or_index_rename"]
    wrong_instrument = source.copy()
    wrong_instrument.index = primary.index
    for invalid in (wrong_instrument, source.iloc[:-1], source.iloc[::-1]):
        with pytest.raises((ValueError, AssertionError)):
            append_supervision(primary, invalid, reference, mapping,
                               expected_rows=5, expected_original_columns=85)

    package = fit_supervision_scalers(raw)
    assert package["stats"]["S"]["mean"] == 0.
    assert package["stats"]["S"]["std"] == np.sqrt(2.5)
    assert package["stats"]["Z"]["std"] == np.sqrt(12.5)
    for name in ("S", "Z"):
        got = transform_supervision_target(raw[name], package["stats"][name])
        pd.testing.assert_series_equal(got, package["normalized_targets"][name], check_exact=True)
        assert got.dtype == np.float32
    for invalid in (raw.assign(S=0.), raw.assign(S=np.nan), raw.assign(Z=np.inf), raw.iloc[::-1]):
        with pytest.raises(ValueError):
            fit_supervision_scalers(invalid)
    changed = raw["S"].copy(); changed.iloc[0] += 1.
    with pytest.raises(ValueError):
        transform_supervision_target(changed, package["stats"]["S"])
    corrupted_scale = deepcopy(package["stats"]["S"]); corrupted_scale["std"] *= 2
    with pytest.raises(ValueError):
        transform_supervision_target(raw["S"], corrupted_scale)

    inactive_masks = {"unused_minute_core": "minute_core_ready", "unused_minute_peer": "minute_peer_ready"}
    args = dict(base_features=base_features, extra_features=[], feature_masks=inactive_masks,
        spec=asdict(PitAuxiliaryAllocationSpec(root=Path("."))), seed=71,
        cutoff="2023-01-01T00:00:00-05:00", train_end="2022-12-30", minimum_observations=1)
    original = MinutePitAllocator(**args)
    B = DayNightSupervisionAllocator(target_name="S", target_scaler=package["stats"]["S"], **args)
    C = DayNightSupervisionAllocator(target_name="Z", target_scaler=package["stats"]["Z"], **args)
    assert all(model.feature_masks == inactive_masks and not model.extra_features for model in (original, B, C))
    initial = initialization_contract(original, B, C)
    assert initial["rng_identical"] and initial["original_parameter_order_preserved"]
    base = original._initialize(); base_rng = torch.get_rng_state().clone()
    B.network = B._initialize(); C.network = C._initialize()
    assert torch.equal(base_rng, torch.get_rng_state())
    names = [name for name, _ in base.named_parameters()]
    assert [name for name, _ in B.network.named_parameters()] == names + ["supervision_head.weight", "supervision_head.bias"]
    assert sum(p.numel() for p in list(B.network.parameters())[-2:]) == 9

    # Neither target nor the new head is required/used by inherited prediction.
    class PredictionOnly:
        def __init__(self, frame):
            self.frame = frame
        def prepare(self, segment, col_set):
            assert segment == "predict" and col_set == ["feature", "common"]
            return self.frame.loc[:, col_set]

    original.network = base
    for model in (original, B, C):
        model.primary_scaler = {key: dict.fromkeys(base_features, value)
                                for key, value in [("median", 0.), ("mean", 0.), ("scale", 1.)]}
        model.common_scaler = {key: dict.fromkeys(COMMON_FEATURES, value)
                               for key, value in [("median", 0.), ("mean", 0.), ("scale", 1.)]}
        model.extra_scaler = {}
    only_inputs = primary.loc[:, ["feature", "common"]]
    expected = original.predict(PredictionOnly(only_inputs))
    poisoned = augmented.copy()
    poisoned[("supervision", "S")] = np.nan
    poisoned[("supervision", "Z")] = np.inf
    with torch.no_grad():
        B.network.supervision_head.weight.fill_(123.)
        B.network.supervision_head.bias.fill_(-99.)
    pd.testing.assert_frame_equal(B.predict(PredictionOnly(only_inputs)), expected, check_exact=True)
    pd.testing.assert_frame_equal(B.predict(PredictionOnly(poisoned)), expected, check_exact=True)
    pd.testing.assert_frame_equal(C.predict(PredictionOnly(only_inputs)), expected, check_exact=True)

    network = B._initialize()
    primary_x = torch.linspace(-1., 1., 5 * 39).reshape(5, 39)
    common_x = torch.linspace(-.5, .5, 5 * 16).reshape(5, 16)
    outputs, primary_state = primary_forward_with_state(network, primary_x, common_x)
    assert not network.tail_encoder._forward_hooks
    aux_state = network.tail_encoder(common_x + .75)
    assert primary_state is not aux_state
    assert torch.equal(primary_state, network.tail_encoder(common_x))
    assert torch.equal(outputs[1], network.tail_head(primary_state).squeeze(1))
    with pytest.raises(RuntimeError):
        primary_forward_with_state(network, primary_x[:, :2], common_x)
    assert not network.tail_encoder._forward_hooks

    # An extreme fabricated residual activates clipping and exposes coupling.
    prediction = network.supervision_head(primary_state).squeeze(1)
    target = torch.full((5,), 10000.)
    private = list(network.allocation_head.parameters())
    original_loss = sum(p.sum() for p in private)
    combined, half_mse = append_supervision_loss(original_loss, prediction, target)
    assert torch.equal(combined, original_loss + .02 * .5 * ((prediction - target) ** 2).mean())
    direct = torch.autograd.grad(half_mse, [*network.tail_encoder.parameters(), *private],
                                 retain_graph=True, allow_unused=True)
    assert all(g is not None and torch.count_nonzero(g) == 0 for g in direct[:2])
    assert all(g is None for g in direct[2:])
    combined.backward()
    parameters = list(network.parameters())
    preclip = [None if p.grad is None else p.grad.clone() for p in parameters]
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in private)
    original_private_norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in private]))
    calls = []
    original_clip = torch.nn.utils.clip_grad_norm_
    def counted_clip(params, max_norm):
        params = list(params)
        calls.append(([id(p) for p in params], max_norm))
        return original_clip(params, max_norm=max_norm)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", counted_clip)
    norm = clip_once(network.parameters())
    assert calls == [([id(p) for p in parameters], 2.)]
    assert norm > original_private_norm > 2.
    coefficient = torch.clamp(2. / (norm + 1e-6), max=1.)
    for p, before in zip(parameters, preclip):
        if before is not None:
            assert torch.equal(p.grad, before * coefficient)
    assert all(not torch.equal(p.grad, torch.ones_like(p)) for p in private)
    # No optimizer step, model.fit, real source, HTTP or account was executed.
