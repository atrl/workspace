import copy
import pytest
import torch
from torch import nn
from src.model.tangent_allocator import head_anchor,tangent_head,audit_zero_extended_heads
from src.workflow.pit_constituent_auxiliary_training_allocation import PitAuxiliaryAllocationNet


def test_extended_head_preserves_old_function_and_rejects_live_new_weights():
    torch.manual_seed(71)
    base=PitAuxiliaryAllocationNet(primary_feature_count=5,common_feature_count=3,hidden_units=8,tail_hidden_units=2)
    base.allocation_head=tangent_head(base.allocation_head,head_anchor(base));other=copy.deepcopy(base)
    other.allocation_head=nn.Linear(base.allocation_head.in_features+2,3)
    with torch.no_grad():
        other.allocation_head.weight.zero_();other.allocation_head.bias.copy_(base.allocation_head.bias)
        other.allocation_head.weight[:,:5].copy_(base.allocation_head.weight[:,:5]);other.allocation_head.weight[:,7:].copy_(base.allocation_head.weight[:,5:])
    assert audit_zero_extended_heads(base,other,shared_primary_count=5,extra_count=2)['new_channels_zero_initialized']
    with torch.no_grad():other.allocation_head.weight[0,5]=.01
    with pytest.raises(ValueError,match='zero initialized'):audit_zero_extended_heads(base,other,shared_primary_count=5,extra_count=2)
