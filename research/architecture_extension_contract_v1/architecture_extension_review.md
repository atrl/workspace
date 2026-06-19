# 架构扩展决策 Review

## 中文结论

- 决策 ID：src-core-architecture-extension-contract-v1。
- 目标 surface：src/core。
- 扩展原因：原始架构要求稳定 src 后继续扩展必须说明原因；需要把原因、影响范围、证据 notebook 和验证命令变成可审阅契约。
- 影响范围：src/core, tests/test_core_architecture_decisions.py, research/architecture_extension_contract_v1。
- 证据 notebook：research/architecture_extension_contract_v1/architecture_extension_contract_v1.ipynb。
- 验证命令：PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_core_architecture_decisions.py -q; PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests -q。
- 是否晋升：否。本记录只说明稳定 src surface 的扩展理由，不代表策略或模型晋升。
- 风险：没有 evidence notebook、聚焦验证和中文结论时，不应继续扩展稳定 src 契约。
- 下一步：在 registry `research/architecture_extension_contract_v1/architecture_extension_registry.json` 中追踪后续架构扩展记录。
