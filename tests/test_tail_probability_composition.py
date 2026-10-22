import numpy as np
import pandas as pd
import unittest

from src.model.nominal_beta_composition import cancel_leverage_cash
from src.model.tail_probability_composition import (
    AuditedTailProbabilityComposition, tail_probability_composition,
)


class TailProbabilityCompositionTest(unittest.TestCase):
    def test_probability_endpoints_and_funded_beta_identity(self):
        before = np.array([0.3, 0.3, 0.4])
        at_zero, _, _ = tail_probability_composition(before, 0)
        at_one, _, _ = tail_probability_composition(before, 1)
        np.testing.assert_allclose(at_zero, cancel_leverage_cash(before))
        np.testing.assert_allclose(at_one, [0, 0.6, 0.4])
        for p in (0.0, 0.33, 1.0):
            after, tail_shift, _ = tail_probability_composition(before, p)
            self.assertTrue(np.isfinite(after).all() and after.min() >= -1e-12)
            self.assertLess(abs(after.sum() - 1), 1e-12)
            self.assertLess(abs(after @ [3, 1, 0]
                                - (before @ [3, 1, 0] - 2 * tail_shift)), 1e-12)

    def test_audit_uses_only_matching_trade_date_probability(self):
        frame = pd.DataFrame({"trade_date": pd.to_datetime(["2026-07-06", "2026-07-27"]),
                              "tail_probability": [0.334, 0.312]})
        transform = AuditedTailProbabilityComposition(frame)
        before = np.array([0.3, 0.3, 0.4])
        first = transform(before, slot="open", trade_date=pd.Timestamp("2026-07-06"))
        second = transform(before, slot="recovery", trade_date=pd.Timestamp("2026-07-27"))
        self.assertEqual(transform.rows[0]["tail_probability"], 0.334)
        self.assertEqual(transform.rows[1]["tail_probability"], 0.312)
        self.assertLess(first[0], second[0])
        with self.assertRaisesRegex(ValueError, "outside frozen predictions"):
            transform(before, slot="open", trade_date=pd.Timestamp("2026-07-28"))


if __name__ == "__main__":
    unittest.main()
