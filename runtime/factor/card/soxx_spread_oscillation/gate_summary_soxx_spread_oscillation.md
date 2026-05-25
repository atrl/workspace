# Factor Gate Summary

## External References

- [Qlib](https://github.com/microsoft/qlib): workflow and reproducibility boundary
- [Qlib arXiv](https://arxiv.org/abs/2009.11189): AI-oriented quantitative investment workflow
- [Avellaneda Zhang LETF path dependence](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1404708): LETF path dependence and realized variance basis
- [Cheng Madhavan LETF dynamics](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1393995): daily rebalancing and microstructure basis
- [Backtest overfitting](https://www.davidhbailey.com/dhbpapers/backtest-overfitting.pdf): multiple-testing guard for factor screens

## Gate Table

| feature | IC | IC+ | PF | Acc | max rho | full | failure |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SOXL_SOXX_SPREAD_VOL_Z_21 | +0.0156 | 0.488 | 1.178 | 0.544 | 0.917 | False | correlation_redundant |
| SOXL_SOXX_RATIO_Z_252 | -0.0049 | 0.551 | 1.161 | 0.543 | 0.885 | False | correlation_redundant |
| SOXL_SOXX_RATIO_WIDEN_5 | -0.0028 | 0.491 | 1.145 | 0.535 | 0.922 | False | correlation_redundant |
| SOXL_SOXX_CHOP_DIVERGENCE_21 | -0.0016 | 0.425 | 1.157 | 0.537 | 0.081 | False | weak_direction_and_execution |

## Readout

- Full-gate factors can move to a profile or gated-composite proposal.
- Correlation failures should not be promoted as core RL features without a distinct non-linear use case.
- Weak-direction failures need IC or regime-slice improvement before any training branch.
