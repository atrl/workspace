# balanced_curriculum — 课程学习采样实验

## 实验设计

- **Changed layer**: data / sampling
- **Baseline**: `balanced_rule_residual_v19`
- **Role**: diagnose
- **Hypothesis**: 通过 regime_stratified + phase_balanced 采样，增加 stress/bear market episode 的曝光频率，改善 bear market generalization

## 采样配置变更

| 参数 | v19 baseline | curriculum |
|------|-------------|------------|
| `regime_stratified` | false | **true** |
| `phase_balanced` | false | **true** |
| `severe_repeat` | 3 | **4** |
| `rebound_repeat` | 2 | **3** |

## 理论机制

1. **regime_stratified**: 对 high-risk episode（risk_score > severe_risk_threshold）增加采样重复次数
2. **phase_balanced**: 对 rebound phase episode 增加采样重复次数，确保代理见过足够的 recovery 场景
3. 综合效果：训练数据中 bear market 和 rebound 的占比提升，减少 bull market 过采样

## 通过标准

- MUST: `wf_2022 Sharpe > -0.80`
- MUST: `wf_2024 Sharpe > -0.23`
- NICE: `test_proxy Sharpe > 0.90`

## 失败标准

- `wf_2022 Sharpe < -1.00` → 采样调整无效
- Valid score 显著低于 v19 → 过拟合到 stress episode

## Smoke Test 结果 (2026-04-30)

| 指标 | 结果 |
|------|------|
| Valid best score | **86.36** (iter 40) |
| test_proxy Return | 702.5% |
| test_proxy Sharpe | **0.97** |
| test_proxy MaxDD | 49.4% |
| test_2022 Return | **-28.9%** |
| test_2022 Sharpe | -1.08 |
| test_2022 MaxDD | 30.2% |
| test_2022 avg_pos | **0.283** ✅ |
| test_2023_2024 Return | 44.9% |
| test_2023_2024 Sharpe | 0.69 |
| trades | 314 |
| Fixed split gate | **PASSED** |

## 关键发现

1. **Bear market 防御显著改善**：test_2022 return -28.9% vs v19 baseline -36.7%（+7.8pp），avg_pos 0.283 vs 0.328（更防御）
2. **test_proxy Sharpe 提升**：0.97 vs v19 0.89
3. **Recovery 期表现下降**：test_2023_2024 return 44.9% vs v19 62.4%（-17.5pp）
4. **训练稳定性**：valid score 在 iter 40 达到峰值 86.36，之后下降，最佳 checkpoint 在 iter 40

## 实现状态

- [x] Profile YAML 创建
- [x] Smoke test（快速训练验证）
- [ ] Walk-forward 验证
