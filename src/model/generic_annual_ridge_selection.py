"""Native nested annual prior/Ridge selection with a parameterized return horizon."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.data.integrated_allocation import frame_sha256
from src.model.generic_horizon_estimator import (fit_horizon_estimator, fit_horizon_prior,
    predict_horizon, validate_native_horizon_dataset, _strict_year, _strict_count)


def _require(condition, message):
    if not condition:
        raise ValueError('blocked: ' + message)


def select_annual_horizon(dataset, frame, *, horizon_sessions=5, feature_columns,
                          label_columns, availability_column, outer_year,
                          minimum_training_rows=252, minimum_validation_rows=126,
                          hac_lags=5, train_start='2011-01-03'):
    """Fit four mature inner candidates, select on prior-year error, then refit once.

    The portfolio never enters model selection. The old H5 entry remains unchanged;
    this returns the selected fit and all four inner fits without an extra portfolio
    or an unused outer Ridge control.
    """
    outer_year = _strict_year(outer_year)
    minimum_validation_rows = _strict_count(minimum_validation_rows, 'minimum_validation_rows')
    validate_native_horizon_dataset(dataset, horizon_sessions=horizon_sessions,
        feature_columns=feature_columns, label_columns=label_columns, availability_column=availability_column)
    prepared = dataset.prepare('all', col_set=['feature', 'label', 'metadata'])
    _require(not prepared.empty and not prepared.columns.has_duplicates, 'ambiguous native source')
    dates = pd.DatetimeIndex(prepared.index.get_level_values('datetime'))
    calendar = pd.DatetimeIndex(dataset.calendar)
    positions = calendar.get_indexer(dates)
    _require(np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)), 'interior native session missing')
    flat = prepared.copy()
    flat.columns = flat.columns.get_level_values(1)
    flat.index = dates
    columns = list(flat.columns)
    _require(frame.index.equals(dates) and not frame.columns.has_duplicates, 'inference dates or schema differ')
    try:
        pd.testing.assert_frame_equal(flat, frame[columns], check_exact=True, check_names=False, check_freq=False)
    except AssertionError as error:
        raise ValueError('blocked: DatasetH and inference frame values differ') from error
    signal = pd.DatetimeIndex(flat.signal_date)
    _require(signal.equals(dates), 'signal metadata differs from native dates')
    for name in ('feature_available_at', availability_column):
        _require(all(pd.isna(v) or pd.Timestamp(v).tzinfo is not None for v in flat[name]),
                 name + ' must carry explicit timezone-aware availability')
    available = pd.to_datetime(flat[availability_column], utc=True)
    feature_at = pd.to_datetime(flat.feature_available_at, utc=True)
    _require(feature_at.notna().all(), 'missing feature availability cannot be dropped')
    inner_year = outer_year - 1
    inner_cut = pd.Timestamp(f'{inner_year}-01-01', tz='America/New_York')
    outer_cut = pd.Timestamp(f'{outer_year}-01-01', tz='America/New_York')
    start = dates >= pd.Timestamp(train_start)
    inner_mask = (start & (feature_at < inner_cut) & (available < inner_cut)).to_numpy()
    outer_mask = (start & (feature_at < outer_cut) & (available < outer_cut)).to_numpy()
    year_mask = dates.year == inner_year
    prefix = ((feature_at < outer_cut) & (available < outer_cut)).to_numpy()[year_mask]
    n = int(prefix.sum())
    _require(n >= minimum_validation_rows and prefix[:n].all() and not prefix[n:].any(),
             'validation needs a complete mature prefix')
    validation_mask = year_mask.copy()
    validation_mask[year_mask] = prefix
    for name, mask in (('inner', inner_mask), ('outer', outer_mask), ('validation', validation_mask)):
        part = prepared.loc[mask]
        x, y = part['feature'].to_numpy(float), part['label'].to_numpy(float)
        _require(np.isfinite(x).all() and np.isfinite(y).all() and (y > -1).all(),
                 name + ' contains invalid values before any fit')
    train = prepared.loc[inner_mask]
    variance = train['label'].to_numpy(float).var(axis=0, ddof=0)
    _require(np.isfinite(variance).all() and (variance > 0).all(), 'inner label variance is degenerate')
    common = dict(horizon_sessions=horizon_sessions, feature_columns=feature_columns,
                  label_columns=label_columns, availability_column=availability_column,
                  minimum_training_rows=minimum_training_rows, hac_lags=hac_lags, train_start=train_start)
    names = ('prior', 'ridge_10', 'ridge_100', 'ridge_1000')
    priority = ('prior', 'ridge_1000', 'ridge_100', 'ridge_10')
    inner_models, rows = {}, []
    expected_hash = frame_sha256(train)
    actual = prepared.loc[validation_mask, 'label'].to_numpy(float)
    query = frame.loc[dates[validation_mask]]
    for name in names:
        model = (fit_horizon_prior(dataset, year=inner_year, **common) if name == 'prior'
                 else fit_horizon_estimator(dataset, year=inner_year, ridge_alpha=float(name.split('_')[1]), **common))
        _require(model['audit']['training_frame_sha256'] == expected_hash, 'candidate training sources differ')
        predicted = predict_horizon(model, query, feature_columns=feature_columns,
                                    horizon_sessions=horizon_sessions)['ridge']
        mse = np.square(predicted - actual).mean(axis=0)
        score = float((mse / variance).mean())
        _require(np.isfinite(score), 'nonfinite validation score')
        inner_models[name] = model
        rows.append(dict(candidate_name=name, normalized_mse=score, mse_soxl=float(mse[0]), mse_soxx=float(mse[1]),
                         inner_training_frame_sha256=expected_hash, validation_rows=n, exact_tie_priority=priority.index(name)))
    selected = min(rows, key=lambda r: (r['normalized_mse'], r['exact_tie_priority']))['candidate_name']
    fitted = (fit_horizon_prior(dataset, year=outer_year, **common) if selected == 'prior'
              else fit_horizon_estimator(dataset, year=outer_year, ridge_alpha=float(selected.split('_')[1]), **common))
    _require(fitted['audit']['training_frame_sha256'] == frame_sha256(prepared.loc[outer_mask]),
             'outer refit source differs')
    for row in rows:
        row['selected'] = row['candidate_name'] == selected
    return dict(selected_name=selected, selected_fit=fitted, inner_models=inner_models,
        candidate_scores=pd.DataFrame(rows), audit=dict(outer_year=outer_year, inner_year=inner_year,
            horizon_sessions=horizon_sessions, inner_training_rows=int(inner_mask.sum()),
            validation_rows=n, validation_unmatured_suffix_rows=int(year_mask.sum()) - n,
            outer_training_rows=int(outer_mask.sum()), validation_frame_sha256=frame_sha256(prepared.loc[validation_mask]),
            inner_cutoff=inner_cut.isoformat(), outer_cutoff=outer_cut.isoformat(),
            exact_tie_order=list(priority), selected_name=selected, model_fit_calls=5,
            outer_outcomes_used_for_selection=False, invalid_rows_dropped=0))
