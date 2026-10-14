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
