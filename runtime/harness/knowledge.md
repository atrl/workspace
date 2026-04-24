# HARNESS Knowledge

> 最后更新: 2026-04-24
> 该方向负责仓库结构、工具链、验证脚本和 agent 工作流。

## 验证索引

| 检查 | 状态 | 证据 | 卡片 |
|------|------|------|------|
| Harness Validation | ✅ Passing | 59 passed, 0 failures | [card](cards/harness_validation/card.md) |

## 核心结论

1. **验证入口统一**: `scripts/check_harness.py` 是单一验证入口。
2. **目录边界清晰**: runtime/<direction>/ 替代了 memory/。
3. **Skill 自管理**: `.codex/skills/_index.json` 追踪版本和方向。
4. **Artifact 治理**: canonical/balanced_family/smoke_family/risk_budget_family/other。

## 有效方法 ✅

| 方法 | 验证结果 |
|------|---------|
| check_harness.py | 59 tests, 0 failures |
| runtime index | `scripts/build_runtime_index.py` |

## 待验证方向

- [ ] 自动化 skill 过期检测
- [ ] 实验卡片完整性检查
- [ ] 回归测试覆盖 promotion gate
