# 架构扩展决策 Review

## 中文结论

- 决策 ID：paper-execution-objective-loss-gate-v1。
- 目标 surface：src/core,src/api,src/schedule。
- 扩展原因：需要把 objective.loss.v1 从 dashboard 证据扩展到 paper 执行前 gate，防止 reward surface 或损失契约异常时继续按原仓位执行。
- 影响范围：src/core/contracts.py, src/api/intents.py, src/schedule/pipelines.py, research/paper_execution_objective_loss_gate_v1。
- 证据 notebook：research/paper_execution_objective_loss_gate_v1/paper_execution_objective_loss_gate_v1.ipynb。
- 验证命令：PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_platform_contracts.py tests/test_api_paper_intents.py tests/test_schedule_paper_execution_pipeline.py -q; PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests -q。
- 是否晋升：否。本记录只说明稳定 src surface 的扩展理由，不代表策略或模型晋升。
- 风险：没有 evidence notebook、聚焦验证和中文结论时，不应继续扩展稳定 src 契约。
- 下一步：在 registry `/Users/rziiiii/dev/workspace/research/paper_execution_objective_loss_gate_v1/architecture_extension_registry.json` 中追踪后续架构扩展记录。
