"""Rescore frozen Qlib annual candidates on a benchmark-relative response.

Only the prior validation year's strictly mature labels enter the score. The
caller owns the candidate inventory, tie order, and any subsequent outer refit.
Changing the response coordinate of a fixed linear Ridge fit alone would not
change its predictions; this module changes which saved fit is selected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.integrated_allocation import frame_sha256
from src.model.generic_annual_ridge_selection import qualify_annual_horizon_inputs
from src.model.generic_horizon_estimator import predict_horizon


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def rescore_frozen_annual_candidates(dataset, frame, parent_selection, *, outer_year,
                                    feature_columns, label_columns, availability_column,
                                    candidate_names, tie_priority, horizon_sessions=5,
                                    minimum_training_rows=252, minimum_validation_rows=126,
                                    hac_lags=5, train_start="2011-01-03",
                                    eligibility_column=None):
    """Compare absolute-leg and relative-response losses without fitting.

The original parent selection is reproduced before accepting an alternative.
Inner model years, training bytes, feature schemas, validation cohorts, and
selection cutoffs must agree. No portfolio outcomes enter either loss.
"""
    names, priority = tuple(candidate_names), tuple(tie_priority)
    _require(len(names) == len(set(names)) == len(priority)
             and set(names) == set(priority), "candidate inventory or tie priority differs")
    _require(len(label_columns) == 2, "relative selection requires ordered asset and benchmark labels")
    qualified = qualify_annual_horizon_inputs(
        dataset, frame, outer_year=outer_year, horizon_sessions=horizon_sessions,
        feature_columns=feature_columns, label_columns=label_columns,
        availability_column=availability_column, eligibility_column=eligibility_column,
        minimum_training_rows=minimum_training_rows, minimum_validation_rows=minimum_validation_rows,
        hac_lags=hac_lags, train_start=train_start)
    audit = qualified["audit"]
    for key in ("inner_identity", "outer_identity", "validation_identity", "validation_frame_sha256"):
        _require(parent_selection["audit"][key] == audit[key], "parent cohort differs: " + key)
    models = parent_selection["inner_models"]
    _require(set(models) == set(names), "frozen inner model inventory differs")
    prepared, dates = qualified["prepared"], qualified["dates"]
    train = prepared.loc[qualified["inner_mask"]]
    actual = prepared.loc[qualified["validation_mask"], "label"].to_numpy(float)
    query = frame.loc[dates[qualified["validation_mask"]]]
    training_labels = train["label"].to_numpy(float)
    spread_train = training_labels[:, 0] - training_labels[:, 1]
    spread_variance = float(spread_train.var(ddof=0))
    _require(np.isfinite(spread_variance) and spread_variance > 0, "degenerate inner relative-label variance")
    training_hash = frame_sha256(train)
    parent_scores = parent_selection["candidate_scores"].set_index("candidate_name")
    _require(parent_scores.index.is_unique and set(parent_scores.index) == set(names), "parent scores differ")
    rows = []
    for name in names:
        model = models[name]
        _require(model["model_year"] == outer_year - 1
                 and model["model_available_at"] == audit["inner_cutoff"], "frozen inner model clock differs")
        _require(tuple(model["feature_columns"]) == tuple(feature_columns)
                 and tuple(model["label_columns"]) == tuple(label_columns), "frozen inner model schema differs")
        _require(model["audit"]["training_frame_sha256"] == training_hash, "frozen inner training bytes differ")
        predicted = predict_horizon(model, query, feature_columns=feature_columns,
                                    horizon_sessions=horizon_sessions)["ridge"]
        errors = predicted - actual
        leg_mse = np.square(errors).mean(axis=0)
        joint_loss = float((leg_mse / qualified["variance"]).mean())
        _require(np.isclose(joint_loss, parent_scores.loc[name, "normalized_mse"], rtol=1e-12, atol=1e-14),
                 "parent validation loss cannot be reproduced")
        relative_mse = float(np.square(errors[:, 0] - errors[:, 1]).mean())
        relative_loss = relative_mse / spread_variance
        _require(np.isfinite(relative_loss), "nonfinite relative validation loss")
        rows.append(dict(year=int(outer_year), candidate_name=name, absolute_loss=joint_loss,
                         relative_loss=relative_loss, relative_mse=relative_mse,
                         relative_training_variance=spread_variance, validation_rows=len(actual),
                         exact_tie_priority=priority.index(name)))
    absolute = min(rows, key=lambda r: (r["absolute_loss"], r["exact_tie_priority"]))["candidate_name"]
    relative = min(rows, key=lambda r: (r["relative_loss"], r["exact_tie_priority"]))["candidate_name"]
    _require(absolute == parent_selection["selected_name"], "parent absolute-loss winner differs")
    outer_hash = frame_sha256(prepared.loc[qualified["outer_mask"]])
    _require(parent_selection["selected_fit"]["audit"]["training_frame_sha256"] == outer_hash,
             "parent outer fit training bytes differ")
    for row in rows:
        row.update(absolute_selected=row["candidate_name"] == absolute,
                   relative_selected=row["candidate_name"] == relative)
    return dict(selected_name=relative, parent_selected_name=absolute, requires_refit=relative != absolute,
                candidate_scores=pd.DataFrame(rows), outer_training_frame_sha256=outer_hash,
                audit=dict(audit, selection_loss="H5 asset-minus-benchmark MSE / inner-training spread variance",
                           selected_name=relative, parent_selected_name=absolute,
                           exact_tie_order=list(priority), model_fit_calls=0,
                           portfolio_outcomes_used_for_selection=False))
