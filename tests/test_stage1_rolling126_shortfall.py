import torch

from src.model.stage1_rolling126_shortfall import WINDOW, rolling_soxx_shortfall


def test_rolling_shortfall_penalizes_cash_drag_and_has_training_gradient():
    returns = torch.tensor([[0.02, 0.01]] * (WINDOW + 4), dtype=torch.float64)
    benchmark = torch.tensor([[0.0, 1.0, 0.0]] * len(returns), dtype=torch.float64)
    assert rolling_soxx_shortfall(benchmark, returns, cost_bps=0).item() == 0.0

    cash_heavy = torch.tensor([[0.0, 0.5, 0.5]] * len(returns), dtype=torch.float64,
                              requires_grad=True)
    penalty = rolling_soxx_shortfall(cash_heavy, returns, cost_bps=0)
    assert penalty.item() > 0.0
    penalty.backward()
    assert cash_heavy.grad is not None
    assert torch.isfinite(cash_heavy.grad).all()
    assert cash_heavy.grad.abs().sum().item() > 0.0
