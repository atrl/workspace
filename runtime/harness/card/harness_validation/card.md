# Harness Validation

**方向**: harness
**类别**: validation
**状态**: ✅ Passing

## 假设

仓库结构和验证流程必须机械可执行，不能依赖人工记忆。

## 方法/配置

- `scripts/check_harness.py`: 单一验证入口
- `scripts/check_source_boundaries.py`: 目录边界检查
- `scripts/check_skills.py`: skill 索引同步检查
- `runtime/index.json`: 运行时索引

## 过程

每次提交前运行 `check_harness.py`，验证：
1. 测试通过
2. py_compile 通过
3. 目录边界检查通过
4. task router 有效

## 结果

| 检查 | 状态 | 证据 |
|------|------|------|
| check_harness.py | ✅ | 59 passed, 0 failures |
| check_source_boundaries.py | ✅ | runtime dirs validated |
| check_skills.py | ✅ | skills index synced |
| retention_manifest.json | ✅ | artifact families organized |
| profile_registry.yaml | ✅ | all profiles registered |

## 结论

验证流程稳定。所有检查均通过。目录结构从 memory/ 迁移到 runtime/<direction>/ 后边界清晰。

## 与基线/相关实验的差异

这是 harness 自身的验证，不涉及与其他实验的对比。

## 下一步

- [ ] 自动化 skill 过期检测
- [ ] 实验卡片完整性检查
- [ ] 回归测试覆盖 promotion gate
