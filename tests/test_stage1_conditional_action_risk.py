import numpy as np

from src.model.stage1_conditional_action_risk import fit_conditional_action_risk
from src.model.stage1_on_policy_critics import WeightedStateScaler


def test_conditional_risk_zero_action_and_received_floor():
    rng = np.random.default_rng(14)
    z = rng.normal(size=(45, 10))
    u = rng.uniform(-0.2, 0.2, size=(45, 2))
    w = np.ones(45)
    scaler = WeightedStateScaler.fit(z, w)
    base = np.full(45, 0.15)
    y = np.clip(base + 0.01 * u[:, 0] - 0.004 * u[:, 1], 0, 1)
    model = fit_conditional_action_risk(z, u, y, base, w, saved_scaler=scaler)
    assert model.solver_status == "optimal"
    np.testing.assert_allclose(model.correction(z, np.zeros_like(u)), 0, atol=1e-12)
    predicted = model.predict(z, u, base, np.full(45, 0.16))
    assert np.all(predicted >= 0.16)
