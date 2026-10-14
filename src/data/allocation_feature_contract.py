"""Research-only semantic aliases; never rewrite a frozen network's input schema."""
import numpy as np
import pandas as pd

RETURN_ALIASES = {f'soxx_ret{horizon}': f'mom{horizon}' for horizon in (5, 20, 60)}


def canonical_return_inputs(primary, common):
    if not primary.index.equals(common.index):
        raise ValueError('feature calendars differ')
    audit = []
    for canonical, alias in RETURN_ALIASES.items():
        left, right = primary[canonical], common[alias]
        if not left.isna().equals(right.isna()):
            raise ValueError('alias missing masks differ: ' + canonical)
        valid = left.notna()
        if not np.isfinite(left.loc[valid]).all() or not np.isfinite(right.loc[valid]).all():
            raise ValueError('nonfinite alias: ' + canonical)
        # Native Qlib float32 roundoff tolerance, unrelated to returns/performance.
        if not np.allclose(left.loc[valid], right.loc[valid], rtol=1e-6, atol=1e-7):
            raise ValueError('alias values differ: ' + canonical)
        audit.append({'canonical': 'primary:' + canonical, 'alias': 'common:' + alias,
                      'observed_rows': int(valid.sum()),
                      'maximum_raw_difference': float((left.loc[valid]-right.loc[valid]).abs().max())})
    view = pd.concat([primary.add_prefix('primary:'),
                      common.drop(columns=list(RETURN_ALIASES.values())).add_prefix('common:')], axis=1)
    return view, pd.DataFrame(audit)


MOMENTUM_HORIZONS = (3, 5, 10, 20, 60)


def disjoint_momentum_inputs(canonical):
    """Lossless fixed-window log-return coordinates on a complete return chain.

    Research view only: all source horizons are preserved, no outcome-based
    selection or learned projection. Missing short horizons cannot be silently
    reconstructed from longer observed horizons.
    """
    result = canonical.copy()
    mapping = []
    for asset in ('soxl', 'soxx'):
        previous = pd.Series(0., index=result.index)
        previous_horizon = 0
        for horizon in MOMENTUM_HORIZONS:
            source = f'primary:{asset}_ret{horizon}'
            value = canonical[source]
            if np.isinf(value).any() or value.dropna().le(-1).any():
                raise ValueError('log-return source outside domain: ' + source)
            if (value.notna() & previous.isna()).any():
                raise ValueError('non-nested momentum availability: ' + source)
            current = np.log1p(value)
            target = f'primary:{asset}_log_return_lags_{previous_horizon + 1}_{horizon}'
            result[target] = current - previous
            result = result.drop(columns=source)
            mapping.append({'source':source,'target':target,'lag_start':previous_horizon+1,
                            'lag_end':horizon,'inverse':'expm1(cumulative sum of asset segments)'})
            previous, previous_horizon = current, horizon
    return result, pd.DataFrame(mapping)


def reconstruct_momentum(disjoint):
    restored = pd.DataFrame(index=disjoint.index)
    for asset in ('soxl', 'soxx'):
        total = pd.Series(0., index=disjoint.index)
        previous = 0
        for horizon in MOMENTUM_HORIZONS:
            total = total + disjoint[f'primary:{asset}_log_return_lags_{previous + 1}_{horizon}']
            restored[f'primary:{asset}_ret{horizon}'] = np.expm1(total)
            previous = horizon
    return restored
