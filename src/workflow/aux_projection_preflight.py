"""Frozen 24-cell gradient mechanics only: no optimizer, fitting or accounts."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json
from src.model.minute_pit_allocator import TARGETS, _flat_index, _scaled
from src.workflow.minute_pit_allocator import _dataset, _auxiliary_dataset
from src.workflow.pit_constituent_auxiliary_training_allocation import (
    COMMON_FEATURES, _allocation_loss, _project_legal_weights,
)

NAME = "stage1_aux_projection_20261018"
DRAFT_SHA = "42928b9d3cebf84063d9860bed0d8b474df567a5dc74042e940d3063361e0872"
REVIEW_SHA = "2f407bfa6b0075b45dd03c6edf5aa78fea7bcb4d63b9a1ca39c7ffc71ece1332"
SHARED = ("tail_encoder.0.weight", "tail_encoder.0.bias", "tail_head.weight",
          "tail_head.bias", "alpha_head.weight", "alpha_head.bias")
PRIVATE = ("allocation_head.0.weight", "allocation_head.0.bias",
           "allocation_head.2.weight", "allocation_head.2.bias")
CODE = ("src/workflow/aux_projection_preflight.py", "src/model/minute_pit_allocator.py",
        "src/workflow/pit_constituent_auxiliary_training_allocation.py",
        "src/workflow/minute_pit_allocator.py", "src/core/recorder_sources.py",
        "src/data/minute_intake.py", "tests/test_aux_projection_preflight.py")
ARTIFACTS = ("inputs/daily_frame", "inputs/features", "inputs/common", "inputs/timing",
             "inputs/auxiliary", "models/base_outer_71", "models/base_outer_172",
             "models/base_outer_273")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _now():
    return datetime.now(timezone.utc).isoformat()


def state_sha(network):
    h = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _software():
    import qlib
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "qlib": qlib.__version__,
            "device": "cpu", "original_gradient_dtype": "torch.float32",
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "default_dtype": str(torch.get_default_dtype()), "torch_config": torch.__config__.show()}


def freeze(*, root, card):
    """Freeze identities and the complete grid before loading label/model objects."""
    root, card = Path(root).resolve(), Path(card).resolve()
    _require(sha(card / "mechanical_preflight_draft.json") == DRAFT_SHA, "approved draft differs")
    _require(sha(card / "mechanical_preflight_review.json") == REVIEW_SHA, "approved review differs")
    review = json.loads((card / "mechanical_preflight_review.json").read_text())
    _require(review["verdict"] == "APPROVE_MECHANICAL_PREFLIGHT_ONLY", "mechanical approval missing")
    source = json.loads((root / "research/stage1_domain_heads_20261018/plan.json").read_text())["sources"]["training"]
    source = {**source, "artifacts": list(ARTIFACTS),
              "artifact_sha256": {name: source["artifact_sha256"][name] for name in ARTIFACTS}}
    summary = json.loads((root / source["card"] / "summary.json").read_text())
    binding = json.loads((root / source["card"] / "nas_binding.json").read_text())
    _require(binding["evidence"]["sha256"] == source["archive"], "source NAS identity differs")
    _require(sha(root / source["card"] / "verification_manifest.json") == source["inventory_sha256"], "source inventory differs")
    artifact_root = Path(summary["recorder_local_dir"]) / "artifacts"
    for name, digest in source["artifact_sha256"].items():
        _require(sha(artifact_root / name) == digest, "source object hash differs: " + name)
    torch.set_num_threads(1)
    grid = [{"probe_id": f"s{seed}_{anchor}_draw{epoch}", "seed": seed, "anchor": anchor,
             "auxiliary_epoch_index": epoch, "auxiliary_rng_seed": seed * 1000 + epoch}
            for seed in [71, 172, 273]
            for anchor in ["original_initialization", "saved_base_outer_final"]
            for epoch in [0, 99, 199, 299]]
    plan = {"experiment": NAME, "scope": "approved mechanical preflight only", "review_sha256": REVIEW_SHA,
            "draft_sha256": DRAFT_SHA, "source": source, "probes": grid,
            "shared": list(SHARED), "private": list(PRIVATE), "primary_rows": 3014, "auxiliary_rows": 20741,
            "cutoff": "2023-01-01T00:00:00-05:00", "train_end": "2022-12-30",
            "provider_uri": summary["provider_uri"], "region": "us", "as_of": "2022-12-30",
            "provider_role": "Qlib initialization only; no market feature/calendar reads. Inputs are pinned Recorder training slices.",
            "code_sha256": {name: sha(root / name) for name in CODE}, "software": _software(),
            "execution_limits": {"optimizer_steps": 0, "fits": 0, "accounts": 0,
                "new_market_http": 0, "evaluation_label_use": 0, "model_selection": 0},
            "numeric_contract": json.loads((card / "mechanical_preflight_draft.json").read_text())["numerical_screen_proposal"],
            "frozen_at_utc": _now()}
    _require(not (card / "plan.json").exists(), "plan already exists; do not overwrite a frozen plan")
    atomic_json(plan, card / "plan.json")
    identity = {"plan_sha256": sha(card / "plan.json"), "review_sha256": REVIEW_SHA,
                "draft_sha256": DRAFT_SHA, "code_sha256": plan["code_sha256"],
                "source_artifact_sha256": source["artifact_sha256"], "probe_count": 24,
                "software": plan["software"], "frozen_before_source_object_loading_and_probes": True,
                "owner_authorization": "Explicit mechanical-only execution handoff in this task; no fit/account authorization."}
    atomic_json(identity, card / "execution_identity.json")
    (card / "workflow.yaml").write_text(f"experiment: {NAME}\nentrypoint: src.workflow.aux_projection_preflight.run\nphase: frozen_mechanical_only\nprovider_uri: {plan['provider_uri']}\nrecorder_uri: {card / 'qlib_recorder'}\nartifact_policy: slim_card_recorder_first\n")
    return identity


def _read_training_source(root, plan):
    """Hash full objects, then immediately cut by date before inspecting labels."""
    source = plan["source"]
    loaded, audits = {}, {}
    for name in source["artifacts"]:
        objects, audit = load_verified_recorder_objects(root / source["card"],
            expected_recorder_id=source["recorder_id"], artifacts=[name],
            expected_manifest_sha256=source["inventory_sha256"],
            expected_artifact_sha256={name: source["artifact_sha256"][name]})
        value = objects.pop(name)
        if name == "inputs/auxiliary":
            # The key 'outer' denotes the 2022 training fold, not evaluation rows.
            frame = value["outer"]
            value = frame.loc[pd.to_datetime(frame["datetime"]).le(pd.Timestamp(plan["train_end"]))].copy()
            del frame
        elif name.startswith("inputs/"):
            value = value.loc[value.index <= pd.Timestamp(plan["train_end"])].copy()
        loaded[name], audits[name] = value, audit
        del value, objects
    return loaded, audits


def _prepare_training(objects, plan):
    frame = objects.pop("inputs/daily_frame")
    features, common, timing = [objects.pop("inputs/" + name) for name in ["features", "common", "timing"]]
    _require(frame.index.equals(features.index) and frame.index.equals(common.index)
             and frame.index.equals(timing.index), "training source calendars differ")
    maturity = pd.to_datetime(timing["label_available_at"])
    eligible = maturity.notna() & maturity.lt(pd.Timestamp(plan["cutoff"]))
    eligible &= np.isfinite(frame.loc[:, list(TARGETS)].to_numpy(float)).all(axis=1)
    eligible &= frame.index >= pd.Timestamp("2011-01-03")
    frame, features, common, timing = [item.loc[eligible].copy() for item in [frame, features, common, timing]]
    dataset = _dataset(features, frame.loc[:, list(TARGETS)], common, timing,
                       {"train": ("2011-01-03", plan["train_end"])})
    auxiliary = _auxiliary_dataset(objects.pop("inputs/auxiliary"), plan["train_end"])
    train = _flat_index(dataset.prepare("train", col_set=["feature", "label", "common", "meta"]))
    aux = auxiliary.prepare("train", col_set=["feature", "label", "meta"])
    _require(len(train) == 3014 and len(aux) == 20741, "approved training counts differ")
    _require(train.index.max() <= pd.Timestamp(plan["train_end"]), "future primary signal")
    _require(pd.to_datetime(train[("meta", "label_available_at")]).max() < pd.Timestamp(plan["cutoff"]), "immature primary label")
    _require(aux.index.get_level_values("datetime").max() <= pd.Timestamp(plan["train_end"]), "future auxiliary signal")
    original = objects["models/base_outer_71"]
    for seed in [71, 172, 273]:
        model = objects[f"models/base_outer_{seed}"]
        _require(not model.extra_features and len(model.base_features) == 39 and model.seed == seed, "not original model")
        _require(model.primary_scaler == original.primary_scaler and model.common_scaler == original.common_scaler, "source scaler differs across seeds")
        _require(str(model.cutoff) == str(pd.Timestamp(plan["cutoff"])) and model.train_end == pd.Timestamp(plan["train_end"]), "model cutoff differs")
        _require(model.spec.primary_tail_loss_weight == .02 and model.candidate.auxiliary_weight == .05
                 and model.spec.auxiliary_batch_size == 2048 and model.spec.training_cost_bps == 10
                 and model.spec.turnover_loss_weight == .006, "approved objective differs")
    tensors = {
        "primary": _scaled(train["feature"], original.base_features, original.primary_scaler),
        "common": _scaled(train["common"], list(COMMON_FEATURES), original.common_scaler),
        "auxiliary": _scaled(aux["feature"], list(COMMON_FEATURES), original.common_scaler),
        "returns": torch.tensor(train["label"].loc[:, list(TARGETS[:2])].to_numpy(np.float32)),
        "tail": torch.tensor(train[("label", "tail_event_5d")].to_numpy(np.float32)),
        "alpha": torch.tensor(train[("label", "relative_alpha_5d")].gt(0).to_numpy(np.float32)),
        "cash": torch.tensor(train[("label", "tail_cash_target_5d")].to_numpy(np.float32)),
        "years": torch.tensor(train.index.year.to_numpy(), dtype=torch.int64),
        "aux_tail": torch.tensor(aux[("label", "synthetic_2x_crash_label")].to_numpy(np.float32)),
        "aux_alpha": torch.tensor(aux[("label", "synthetic_2x_upside_label")].to_numpy(np.float32)),
        "aux_weight": torch.tensor(aux[("label", "training_weight")].to_numpy(np.float32)),
    }
    tensors["aux_weight"] /= tensors["aux_weight"].mean()
    _require(all(torch.isfinite(x).all() for x in tensors.values()), "nonfinite training tensor")
    audit = {"primary_rows": len(train), "auxiliary_rows": len(aux),
             "first_primary_signal": str(train.index.min()), "last_primary_signal": str(train.index.max()),
             "last_primary_maturity": str(pd.to_datetime(train[("meta", "label_available_at")]).max()),
             "last_auxiliary_signal": str(aux.index.get_level_values("datetime").max()),
             "auxiliary_observations_end": plan["train_end"],
             "auxiliary_maturity_basis": "Inherited pinned outer training fold and original eligible mask; no new label construction.",
             "dataset_segments": ["train"], "evaluation_rows_in_dataset": 0,
             "source_use": "Full source byte hashes, immediate date clipping, then maturity/eligible filtering before label inspection or Dataset construction.",
             "forbidden_artifact_outer_predictions_loaded": False,
             "source_api": ["Recorder.load_object", "DataHandlerLP.from_df", "DatasetH.prepare"],
             "tensor_sha256": {name: hashlib.sha256(t.numpy().tobytes()).hexdigest() for name, t in tensors.items()}}
    return dataset, auxiliary, tensors, audit


def asymmetric_projection(gp, ga):
    """Strict mathematical branch; audit tolerances never change the algorithm."""
    _require(gp.dtype == ga.dtype == torch.float32 and gp.shape == ga.shape, "unsupported gradient dtype/shape")
    _require(torch.isfinite(gp).all() and torch.isfinite(ga).all(), "nonfinite projection input")
    p, a = gp.double(), ga.double()
    dot, norm_squared = torch.dot(p, a), torch.dot(p, p)
    if norm_squared == 0 or dot >= 0:
        return ga.clone(), a.clone(), False
    projected = a - dot / norm_squared * p
    return projected.to(ga.dtype), projected, True


def _norm(vector):
    return float(torch.linalg.vector_norm(vector.double()))


def _dot_screen(left, right, r):
    n = left.numel()
    unit = np.finfo(np.float64).eps / 2
    gamma = (2 * n * unit) / (1 - 2 * n * unit)
    value = float(torch.dot(left.double(), right.double()))
    bound = (gamma + 2 * r + r * r) * _norm(left) * _norm(right)
    return value, bound


def vector_audit(gp, ga, combined, *, shared_count=154):
    """Both local and global parity; preserve original-dtype addition."""
    _require(gp.dtype == ga.dtype == combined.dtype == torch.float32, "unsupported gradient precision")
    _require(all(torch.isfinite(x).all() for x in [gp, ga, combined]), "nonfinite gradient")
    split = gp + ga
    r = 32 * torch.finfo(gp.dtype).eps
    result = {"dtype": str(gp.dtype), "eps": torch.finfo(gp.dtype).eps, "r": r}
    for label, sl in [("full", slice(None)), ("shared", slice(0, shared_count))]:
        scale = _norm(gp[sl]) + _norm(ga[sl])
        error = _norm(combined[sl].double() - split[sl].double())
        result.update({f"{label}_parity_error": error, f"{label}_parity_tolerance": r * scale,
                       f"{label}_parity_pass": error <= r * scale if scale else error == 0,
                       f"{label}_parity_relative": error / scale if scale else 0.})
    a_projected, a64, triggered = asymmetric_projection(gp[:shared_count], ga[:shared_count])
    candidate = torch.cat([gp[:shared_count] + a_projected, split[shared_count:].clone()])
    dot, tau_dot = _dot_screen(gp[:shared_count], ga[:shared_count], r)
    delta = _norm(a_projected.double() - ga[:shared_count].double())
    tau_projection = r * (_norm(gp[:shared_count]) + _norm(ga[:shared_count]))
    private_equal = torch.equal(candidate[shared_count:], split[shared_count:])
    parity = result["full_parity_pass"] and result["shared_parity_pass"]
    resolvable = parity and private_equal and dot < -tau_dot and delta > tau_projection
    denom = _norm(gp[:shared_count]) * _norm(ga[:shared_count])
    result.update({"shared_primary_norm": _norm(gp[:shared_count]), "shared_auxiliary_norm": _norm(ga[:shared_count]),
                   "shared_primary_auxiliary_dot": dot, "shared_cosine": dot / denom if denom else None,
                   "tau_dot": tau_dot, "strict_projection_triggered": triggered,
                   "projection_displacement_before_cast": _norm(a64 - ga[:shared_count].double()),
                   "projection_displacement_after_cast": delta, "tau_projection": tau_projection,
                   "projection_cast_error": _norm(a64 - a_projected.double()),
                   "zero_primary_norm": _norm(gp[:shared_count]) == 0,
                   "private_preclip_unchanged": private_equal,
                   "classification": "resolvable" if resolvable else "needs_numerical_review" if not parity or not private_equal else "identity" if delta == 0 else "unresolved"})
    return result, split, candidate


def _flat(grads, parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g.detach()).reshape(-1)
                      for g, p in zip(grads, parameters)]).clone()


def _clip(vector, parameters):
    offset = 0
    for p in parameters:
        p.grad = vector[offset:offset + p.numel()].reshape_as(p).clone()
        offset += p.numel()
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=2.)
    coefficient = float(torch.clamp(2. / (norm + 1e-6), max=1.))
    return _flat([p.grad for p in parameters], parameters), float(norm), coefficient


def _probe(network, original, tensors, cell):
    before = state_sha(network)
    parameters = dict(network.named_parameters())
    _require(tuple(parameters) == SHARED + PRIVATE, "parameter inventory/order differs")
    ordered = [parameters[name] for name in SHARED + PRIVATE]
    _require(sum(p.numel() for p in ordered[:6]) == 154 and sum(p.numel() for p in ordered[6:]) == 819, "parameter counts differ")
    _require(all(p.dtype == torch.float32 and p.device.type == "cpu" for p in ordered), "original CPU float32 required")
    network.zero_grad(set_to_none=True)
    logits, tail_logits, alpha_logits = network(tensors["primary"], tensors["common"])
    weights = _project_legal_weights(logits, maximum_soxl_weight=original.spec.maximum_soxl_weight,
                                    minimum_cash_weight=original.spec.minimum_cash_weight)
    allocation, _, _, _ = _allocation_loss(weights=weights, returns=tensors["returns"], years=tensors["years"],
        cash_target=tensors["cash"], tail_event=tensors["tail"], candidate=original.candidate,
        drawdown_target_pct=-30., worst_5d_target_pct=original.spec.research_worst_5d_target_pct,
        minimum_cash_weight=original.spec.minimum_cash_weight, cost_bps=original.spec.training_cost_bps,
        turnover_weight=original.spec.turnover_loss_weight)
    primary_bce = F.binary_cross_entropy_with_logits(tail_logits, tensors["tail"])
    primary_bce = primary_bce + F.binary_cross_entropy_with_logits(alpha_logits, tensors["alpha"])
    primary = allocation + original.spec.primary_tail_loss_weight * primary_bce
    generator = torch.Generator().manual_seed(cell["auxiliary_rng_seed"])
    indices = torch.randint(len(tensors["auxiliary"]), (2048,), generator=generator)
    encoded = network.tail_encoder(tensors["auxiliary"][indices])
    aux_tail = network.tail_head(encoded).squeeze(1)
    aux_alpha = network.alpha_head(encoded).squeeze(1)
    auxiliary = ((F.binary_cross_entropy_with_logits(aux_tail, tensors["aux_tail"][indices], reduction="none")
                  + F.binary_cross_entropy_with_logits(aux_alpha, tensors["aux_alpha"][indices], reduction="none"))
                 * tensors["aux_weight"][indices]).mean() * original.candidate.auxiliary_weight
    _require(torch.isfinite(primary + auxiliary), "nonfinite original loss")
    pg = torch.autograd.grad(primary, ordered, retain_graph=True, allow_unused=True)
    ag = torch.autograd.grad(auxiliary, ordered, retain_graph=True, allow_unused=True)
    lg = torch.autograd.grad(allocation, ordered, retain_graph=True, allow_unused=True)
    _require(all(g is not None for g in pg[:6] + ag[:6]), "missing shared gradient")
    _require(all(g is None or torch.count_nonzero(g) == 0 for g in ag[6:]), "auxiliary private gradient nonzero")
    gp, ga, gl = [_flat(g, ordered) for g in [pg, ag, lg]]
    gp_before = gp.clone()
    (primary + auxiliary).backward()
    combined = _flat([p.grad for p in ordered], ordered)
    metrics, split, candidate = vector_audit(gp, ga, combined)
    directions, clipped = [], {}
    for arm, vector in [("control", combined), ("candidate", candidate)]:
        clipped[arm], total_norm, coefficient = _clip(vector, ordered)
        metrics[arm + "_preclip_norm"] = total_norm
        metrics[arm + "_clip_coefficient"] = coefficient
        for stage, actual in [("preclip", vector), ("postclip", clipped[arm])]:
            for objective, reference in [("P", gp), ("allocation", gl)]:
                for scope, sl in [("full", slice(None)), ("shared", slice(0, 154))]:
                    dot, tolerance = _dot_screen(reference[sl], actual[sl], metrics["r"])
                    directions.append({**cell, "arm": arm, "stage": stage, "objective": objective,
                                       "scope": scope, "dot": dot, "tau_dot": tolerance,
                                       "screen_sign": "positive" if dot > tolerance else "negative" if dot < -tolerance else "unresolved"})
    network.zero_grad(set_to_none=True)
    after = state_sha(network)
    _require(before == after and torch.equal(gp, gp_before), "parameters or primary gradient changed")
    metrics.update({**cell, "status": "completed", "parameters_before_sha256": before,
                    "parameters_after_sha256": after, "parameters_unchanged": True,
                    "primary_gradient_unchanged": True, "allocation_loss": float(allocation.detach()),
                    "primary_loss": float(primary.detach()), "auxiliary_loss": float(auxiliary.detach()),
                    "sample_indices_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
                    "sample_count": len(indices), "sample_unique_count": len(torch.unique(indices)),
                    "actual_adamw_step_protection": "unknown_not_measured"})
    vectors = {name: value.detach().numpy() for name, value in {
        "gP": gp, "gA": ga, "g_allocation": gl, "combined_backward": combined,
        "split_original_dtype": split, "candidate": candidate,
        "control_clipped": clipped["control"], "candidate_clipped": clipped["candidate"],
        "auxiliary_indices": indices}.items()}
    return metrics, directions, vectors


def run(*, root, card):
    """Execute every frozen cell once, including retaining numerical failures."""
    import qlib
    from qlib.workflow import R
    from src.data.nas_market_data import archive_nas_recorder

    root, card = Path(root).resolve(), Path(card).resolve()
    identity = json.loads((card / "execution_identity.json").read_text())
    plan = json.loads((card / "plan.json").read_text())
    _require(sha(card / "plan.json") == identity["plan_sha256"], "frozen plan changed")
    _require(sha(card / "mechanical_preflight_review.json") == REVIEW_SHA and sha(card / "mechanical_preflight_draft.json") == DRAFT_SHA, "approval identity differs")
    _require({name: sha(root / name) for name in CODE} == plan["code_sha256"], "frozen code changed")
    torch.set_num_threads(1)
    _require(_software() == plan["software"], "execution software/backend changed")
    _require(not (card / "run_state.json").exists(), "existing run handle must be inspected, not restarted")
    state = {"phase": "loading_training_only", "plan_sha256": identity["plan_sha256"],
             "started_at_utc": _now(), "completed_probes": [], "attempted_probes": [],
             "optimizer_steps": 0, "fits": 0, "accounts": 0}
    atomic_json(state, card / "run_state.json")
    qlib.init(provider_uri=plan["provider_uri"], region="us", kernels=1)
    objects, source_audits = _read_training_source(root, plan)
    dataset, auxiliary_dataset, tensors, training_audit = _prepare_training(objects, plan)
    rows, directions = [], []
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=NAME, uri=str(tracking)):
        recorder = R.get_recorder()
        state.update(phase="probing", recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
        atomic_json(state, card / "run_state.json")
        print("RECORDER", recorder.id, "PROBES", len(plan["probes"]), flush=True)
        recorder.save_objects(artifact_path="inputs", plan=plan, identity=identity,
            review=json.loads((card / "mechanical_preflight_review.json").read_text()),
            dataset=dataset, auxiliary_dataset=auxiliary_dataset, training_audit=training_audit,
            tensors=tensors, source_audits=source_audits)
        recorder.save_objects(artifact_path="sources", **{name.replace("/", "__"): (root / name).read_text() for name in CODE})
        anchor_cache = {}
        for cell in plan["probes"]:
            original = objects[f"models/base_outer_{cell['seed']}"]
            key = (cell["seed"], cell["anchor"])
            if key not in anchor_cache:
                network = original._initialize() if cell["anchor"] == "original_initialization" else deepcopy(original.network)
                anchor_cache[key] = network
                recorder.save_objects(artifact_path="models", **{f"s{key[0]}_{key[1]}": network.state_dict()})
            state["attempted_probes"].append(cell["probe_id"])
            atomic_json(state, card / "run_state.json")
            try:
                result, directional, vectors = _probe(anchor_cache[key], original, tensors, cell)
                directions.extend(directional)
                recorder.save_objects(artifact_path="probes/" + cell["probe_id"], vectors=vectors, result=result, directions=directional)
            except Exception as error:
                result = {**cell, "status": "failed", "classification": "needs_numerical_review",
                          "exception_type": type(error).__name__, "exception": str(error)}
                recorder.save_objects(artifact_path="probes/" + cell["probe_id"], failure=result)
            rows.append(result)
            state["completed_probes"].append(cell["probe_id"])
            atomic_json(state, card / "run_state.json")
            recorder.log_metrics(completed_probes=len(rows))
            print("PROBE", len(rows), cell["probe_id"], result["classification"], flush=True)
        results, direction_frame = pd.DataFrame(rows), pd.DataFrame(directions)
        failures = int(results.classification.eq("needs_numerical_review").sum())
        resolvable = int(results.classification.eq("resolvable").sum())
        decision = "needs_numerical_review" if failures else "resolvable_intervention_requires_separate_training_design" if resolvable else "no_resolvable_intervention_at_prespecified_anchors"
        _require({name: sha(root / name) for name in CODE} == plan["code_sha256"], "source code changed during probes")
        summary = {"experiment": NAME, "decision": decision, "plan_sha256": identity["plan_sha256"],
            "probe_count": len(rows), "resolvable_probes": resolvable, "failed_probes": failures,
            "classification_counts": results.classification.value_counts().to_dict(),
            "primary_rows": 3014, "auxiliary_rows": 20741, "new_fits": 0, "optimizer_steps": 0,
            "new_accounts": 0, "evaluation_label_use": 0, "new_market_http": 0,
            "all_parameters_unchanged": bool(results.get("parameters_unchanged", pd.Series(False, index=results.index)).fillna(False).all()),
            "real_step_protection": "unknown_not_measured_no_saved_optimizer_state_reconstruction",
            "formal_training_authorized": False, "production_changed": False, "objective_achieved": False,
            "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
            "recorder_uri": str(tracking), "recorder_local_dir": str(Path(recorder.get_local_dir())),
            "provider_uri": plan["provider_uri"], "region": "us", "as_of": plan["as_of"],
            "data_lineage": "Pinned original Recorder objects immediately restricted to mature 2011-2022 training domain, then Qlib DatasetH/DataHandlerLP.",
            "qlib_docs_reference": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                "recorder_only_artifacts": ["inputs/*", "sources/*", "models/*", "probes/*"]},
            "limitations": ["24 frozen-anchor draws are not a training trajectory", "dtype screens do not bound true upstream differentiation error",
                            "Composite P protection is not allocation-only or AdamW-step protection", "No economic/forward/promotion evidence"]}
        recorder.save_objects(artifact_path="analysis", summary=summary, probe_results=results, direction_metrics=direction_frame)
        results.to_csv(card / "probe_results.csv", index=False)
        direction_frame.to_csv(card / "direction_metrics.csv", index=False)
        atomic_json(training_audit, card / "training_domain_audit.json")
        recorder.log_metrics(registered_artifact_count=len(recorder.list_artifacts()), resolvable_probes=resolvable, failed_probes=failures)
    artifact_root = Path(summary["recorder_local_dir"]) / "artifacts"
    for name in ["code_diff.txt", "code_status.txt", "code_cached.txt"]:
        (artifact_root / name).unlink(missing_ok=True)
    atomic_json({"recorder_id": summary["recorder_id"], "entries": [
        {"path": str(p.relative_to(card)), "sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(artifact_root.rglob("*")) if p.is_file()]}, card / "verification_manifest.json")
    binding = json.loads((root / plan["source"]["card"] / "nas_binding.json").read_text())
    binding.pop("evidence", None)
    atomic_json(binding, card / "nas_binding.json")
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card,
        recorder_id=summary["recorder_id"], experiment_id=summary["experiment_id"])
    atomic_json(summary, card / "summary.json")
    conclusion = (f"# Stage1辅助梯度投影：固定机械预检\n\n完成原3个seed、初始化/已保存终态、4个辅助抽样序号共{len(rows)}个固定probe；"
        f"可辨投影{resolvable}个，数值/机械失败{failures}个。决定：**{decision}**。\n\n"
        "只使用原3014条成熟主训练样本和20741条辅助样本，2011–2022训练域；未读取评价收益用于计算。"
        "0优化器步、0拟合、0账户、0新市场HTTP；未重建AdamW状态，实际步保护未知。\n\n"
        "这24个锚点/抽样不是训练轨迹，不能推断冲突频率、负迁移、泛化或经济收益。"
        "原始合成P方向不等于配置单项方向，工程精度界不认证真实梯度误差；全部方向及失败记录保留。\n\n"
        "不改变现役release，不晋升，稳定费用后SOXX超额目标未实现。"
        + ("下一步仅可提交另一份完整训练与账户设计，不能自动拟合。" if resolvable and not failures else
           "本固定机械版本停止；不得改变抽样、容差或锚点挽救，需按具体失败交独立评审。")
        + f"\n\nRecorder：`{summary['recorder_id']}`；见probe_results.csv、direction_metrics.csv、summary.json及NAS身份。\n")
    (card / "conclusion.md").write_text(conclusion)
    state.update(phase="completed", completed_at_utc=_now(), decision=decision)
    atomic_json(state, card / "run_state.json")
    return summary
