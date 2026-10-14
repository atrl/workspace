import torch
from src.analysis.frozen_input_attribution import integrated_inputs


def test_integral_reconciles_nonlinear_score_and_zero_change_input():
    x=torch.tensor([[2.,3.],[1.,4.]],dtype=torch.float64)
    baseline=torch.tensor([[0.,3.],[0.,1.]],dtype=torch.float64)
    def score(v):return v[:,0]**3+v[:,0]*v[:,1]
    contribution,residual=integrated_inputs(score,x,baseline,nodes=8)
    assert residual.abs().max()<1e-10
    assert contribution[0,1]==0
    expected=torch.tensor([[14.,0.],[3.5,1.5]],dtype=torch.float64)
    torch.testing.assert_close(contribution,expected)
