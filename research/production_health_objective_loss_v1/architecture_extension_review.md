# 架构扩展决策 Review

## 中文结论

- 决策 ID：production-health-objective-loss-v1。
- 目标 surface：src/schedule,src/dashboard。
- 扩展原因：需要让 production health 从 paper execution evidence 读取 objective.loss.v1 gate，统一展示执行前 review/block 原因。
- 影响范围：src/schedule/pipelines.py, src/dashboard/page.py, research/production_health_objective_loss_v1。
- 证据 notebook：research/production_health_objective_loss_v1/production_health_objective_loss_v1.ipynb。
- 验证命令：PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_schedule_production_health.py tests/test_dashboard_page_renderer.py -q; PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests -q。
- 是否晋升：否。本记录只说明稳定 src surface 的扩展理由，不代表策略或模型晋升。
- 风险：没有 evidence notebook、聚焦验证和中文结论时，不应继续扩展稳定 src 契约。
- 下一步：在 registry `/Users/rziiiii/dev/workspace/research/production_health_objective_loss_v1/architecture_extension_registry.json` 中追踪后续架构扩展记录。
