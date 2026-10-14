"""Conservative measurement bounds, not confidence intervals or trading signals."""
import numpy as np
import pandas as pd


def float32_rounding_cell(values):
    center = np.asarray(values, dtype=np.float64)
    stored = center.astype(np.float32)
    if not np.isfinite(center).all() or not (center > 0).all() or not np.array_equal(stored.astype(np.float64), center):
        raise ValueError('positive exact float32 native prices required')
    below = np.nextafter(stored, np.float32(-np.inf)).astype(np.float64)
    above = np.nextafter(stored, np.float32(np.inf)).astype(np.float64)
    return (below + center) / 2, (center + above) / 2


def rotation_measurement_bounds(prices: pd.DataFrame, *, opening_halfwidth: float):
    """Bound 2 min(up-cross mass, down-cross mass) over a price uncertainty box.

    The same opening value determines both endpoint signs for each member.
    Endpoint uncertainty is only native float32 rounding. Opening uncertainty
    additionally includes a user-specified measurement stress. The upper bound
    may be loose when the same member can switch in either direction.
    """
    if not np.isfinite(opening_halfwidth) or opening_halfwidth < 0:
        raise ValueError('nonnegative finite opening halfwidth required')
    w = prices['weight'].to_numpy(dtype=float)
    if not len(w) or not np.isfinite(w).all() or not (w > 0).all() or not np.isclose(w.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError('complete normalized positive basket required')
    o, a, b = [prices[f].to_numpy(dtype=float) for f in ['open', 'close_0945', 'close_1030']]
    ol, oh = float32_rounding_cell(o)
    al, ah = float32_rounding_cell(a)
    bl, bh = float32_rounding_cell(b)
    if opening_halfwidth:
        ol = np.nextafter(ol - opening_halfwidth, -np.inf)
        oh = np.nextafter(oh + opening_halfwidth, np.inf)
    if (ol <= 0).any():
        raise ValueError('stress includes nonpositive opening price')
    certain_up = (ol > ah) & (oh < bl)
    possible_up = (oh > al) & (ol < bh) & (bh > al)
    certain_down = (ol > bh) & (oh < al)
    possible_down = (oh > bl) & (ol < ah) & (ah > bl)
    up = float(w[(a < o) & (b > o)].sum())
    down = float(w[(b < o) & (a > o)].sum())
    lower = 2 * min(w[certain_up].sum(), w[certain_down].sum())
    upper = min(2 * min(w[possible_up].sum(), w[possible_down].sum()), w[possible_up | possible_down].sum())
    point = 2 * min(up, down)
    if not lower <= point + 1e-12 or not point <= upper + 1e-12:
        raise AssertionError('native point outside conservative measurement bounds')
    return dict(point=point, lower=float(lower), upper=float(upper), width=float(upper-lower),
                uncertain_member_weight=float(w[(possible_up & ~certain_up) | (possible_down & ~certain_down)].sum()))
