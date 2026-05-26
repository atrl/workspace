# Gate-Gap Failure Attribution

## External References

- [Qlib](https://github.com/microsoft/qlib): reproducible quant workflow and artifact discipline
- [Qlib arXiv](https://arxiv.org/abs/2009.11189): AI-oriented quantitative investment platform
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL): DRL trading workflow with train/validate/test separation
- [DQfD](https://arxiv.org/abs/1704.03732): demonstration learning still needs controlled policy improvement
- [Risk-sensitive RL](https://arxiv.org/abs/1312.4077): risk-sensitive sequential decision objective basis
- [PBO](https://doi.org/10.21314/JCF.2016.322): repeated trials require OOS and walk-forward discipline

## Profile Summary

| profile | windows | failures | dominant type | worst window | severity | next mechanism |
| --- | ---: | ---: | --- | --- | ---: | --- |
| base_multi_obj_plain | 2 | 2 | drawdown_control | wf_2022 | 213.32 | drawdown_controller_or_sequence_risk_feature |
| short_volshock_weighted_obj025_cb099 | 2 | 2 | drawdown_control | wf_2022 | 97.46 | drawdown_controller_or_sequence_risk_feature |
| short_volshock_plain | 2 | 2 | drawdown_control | wf_2022 | 72.62 | drawdown_controller_or_sequence_risk_feature |

## Window Attribution

| profile | window | rel return | max dd | avg pos | type | path |
| --- | --- | ---: | ---: | ---: | --- | --- |
| base_multi_obj_plain | wf_2022 | 51.94pp | 36.33% | 0.181 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| short_volshock_weighted_obj025_cb099 | wf_2022 | 65.87pp | 24.75% | 0.078 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| short_volshock_plain | wf_2022 | 67.62pp | 22.26% | 0.097 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| base_multi_obj_plain | wf_2023_2024 | 33.76pp | 20.91% | 0.253 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| short_volshock_plain | wf_2023_2024 | 5.02pp | 16.69% | 0.183 | joint_tradeoff | walk_forward_weighted_multi_objective_selection |
| short_volshock_weighted_obj025_cb099 | wf_2023_2024 | 8.82pp | 12.80% | 0.144 | selection_bias | checkpoint_or_model_selection |

## Readout

- Treat joint trade-off rows as evidence against another single-coefficient reward sweep.
- Use near-miss drawdown rows for cap calibration or execution friction checks before retraining.
- Use underparticipation or trend-capture rows to justify participation-aware objectives or bull guards.
