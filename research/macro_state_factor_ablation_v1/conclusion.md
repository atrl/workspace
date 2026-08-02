# 结论

- what_ran: 执行 macro_state_factor_ablation_v1，对实际利率/通胀预期、广义美元、利率波动及全量组合做 qlib 消融。
- selection: 只按验证集选择，selected_group=base_cogalpha_soxx，validation_score=42.528345；外层测试未参与选组。
- key_metrics: selected_test_return=7.1073%，selected_maxDD=-4.0717%，baseline_test_return=7.1073%。
- walk_forward: risk_all_pass=True，return_all_pass=True，min_return=3.2041%。
- promotion_decision: not_promoted_validation_selected_baseline。
- risks: 5Y5Y 通胀项是财政部名义/实际曲线推导 proxy，并非 FRED T5YIFR；发布映射是保守可用性假设。
- next_action: 仅当非基线组同时通过外层测试和 walk-forward 门时，才进入下一层 validated research；recorder=c4a9e243a0aa447ab58bb5503e875997。
