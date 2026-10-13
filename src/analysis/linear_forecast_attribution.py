"""Exact fitted-Ridge prediction decomposition, not causal feature importance."""
import numpy as np
import pandas as pd


def decompose_forecast(model, grouped, recorded):
    predictions = recorded["prediction"]
    truth = recorded["truth"]
    score = recorded["label_mature"]
    x = grouped.loc[predictions.index, "feature"][model.columns].fillna(model.impute_mean)
    z = model.x_scaler.transform(x)
    maturity = pd.to_datetime(grouped[("meta", "label_available_at")])
    train_mask = maturity.notna() & maturity.lt(model.cutoff) & (grouped.index.get_level_values("datetime") < model.cutoff)
    train = grouped.loc[train_mask, "feature"][model.columns].fillna(model.impute_mean)
    if len(train) != model.audit["training_rows"]:
        raise ValueError("training population differs from fitted model")
    np.testing.assert_allclose(train.to_numpy(dtype=np.float64).mean(axis=0), model.x_scaler.mean_, rtol=1e-10, atol=1e-10)
    train_min, train_max = train.min().to_numpy(), train.max().to_numpy()
    summaries, features, daily = [], [], {}
    valid = score.to_numpy(bool)
    for j, target in enumerate(model.targets):
        contributions = z * model.estimator.coef_[j] * model.y_scaler.scale_[j]
        intercept = model.y_scaler.mean_[j] + model.y_scaler.scale_[j] * model.estimator.intercept_[j]
        reconstructed = contributions.sum(axis=1) + intercept
        error = float(np.max(np.abs(reconstructed - predictions[target].to_numpy())))
        if error > 1e-10:
            raise ValueError("recorded prediction does not reconcile with fitted coefficients")
        actual_mean = float(truth.loc[score, target].mean())
        predicted_mean = float(predictions.loc[score, target].mean())
        mean_contribution = contributions[valid].mean(axis=0)
        summaries.append({"target": target, "scored_rows": int(valid.sum()), "training_rows": len(train),
            "prediction_mean": predicted_mean, "actual_mean": actual_mean, "bias": predicted_mean - actual_mean,
            "training_target_mean": float(model.y_scaler.mean_[j]), "intercept": float(intercept),
            "sum_feature_mean_contributions": float(mean_contribution.sum()),
            "intercept_minus_actual_mean": float(intercept - actual_mean),
            "sum_absolute_mean_contributions": float(np.abs(mean_contribution).sum()),
            "reconstruction_max_abs_error": error,
            "forecast_outside_training_target_range_fraction": float(((predictions.loc[score, target] < grouped.loc[train_mask, ("label", target)].min()) | (predictions.loc[score, target] > grouped.loc[train_mask, ("label", target)].max())).mean())})
        for k, column in enumerate(model.columns):
            observed = x.iloc[valid, k].to_numpy()
            features.append({"target": target, "feature": column, "mean_forecast_contribution": float(mean_contribution[k]),
                "mean_absolute_daily_contribution": float(np.abs(contributions[valid, k]).mean()),
                "mean_standardized_feature": float(z[valid, k].mean()), "maximum_absolute_standardized_feature": float(np.abs(z[valid, k]).max()),
                "outside_training_minmax_fraction": float(((observed < train_min[k]) | (observed > train_max[k])).mean()),
                "training_minimum": float(train_min[k]), "training_maximum": float(train_max[k]),
                "scored_minimum": float(observed.min()), "scored_maximum": float(observed.max())})
        daily[target] = pd.DataFrame(contributions, index=predictions.index, columns=model.columns)
    return {"summary": pd.DataFrame(summaries), "features": pd.DataFrame(features), "daily_contributions": daily,
        "training_correlation": train.corr(), "scored_correlation": x.loc[score].corr(),
        "scope": "algebraic decomposition of frozen fitted model; correlated feature contributions are not independent causal effects"}
