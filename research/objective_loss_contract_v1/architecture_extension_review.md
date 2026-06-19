# 架构扩展决策 Review

## 中文结论

- 决策 ID：src-objective-loss-contract-v1。
- 目标 surface：src/objective。
- 扩展原因：AGENTS.md 明确 src/objective 负责 loss、IC/ranking objective 和收益风险 utility；在 rank_ic_objective research evidence 完成后，需要 compute_objective_losses 作为稳定 loss contract，供模型训练、analysis 和 promotion 复用，同时继续保持 reward surface retired。
- 影响范围：src/objective, tests/test_objective_contract.py, research/objective_loss_contract_v1, research/rank_ic_objective_validation_v1。
- 证据 notebook：research/objective_loss_contract_v1/objective_loss_contract_v1.ipynb。
- 验证命令：PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_objective_contract.py -q; PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests -q。
- 是否晋升：否。本记录只说明稳定 src surface 的扩展理由，不代表策略或模型晋升。
- 风险：没有 evidence notebook、聚焦验证和中文结论时，不应继续扩展稳定 src 契约。
- 下一步：在 registry `research/objective_loss_contract_v1/architecture_extension_registry.json` 中追踪后续架构扩展记录。
