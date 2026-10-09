# 夜盘信号卡：完整性更正（INVALID FOR PROMOTION）

> **状态：`invalid_for_promotion_look_ahead`。本卡原结论与指标全部作废，不得用于晋升、实盘或任何可用策略声明。**
> 原指标保留为 **invalid historical result**，仅用于审计；原 qlib Recorder 保留不改。

## 缺陷（机器可验证）

1. **前视（决定性）**：隔夜信号的窗口用了 `index.time < 09:30`，但 Yahoo NQ 的 `09:00` 一小时 bar 覆盖 `09:00–10:00`，其 close 是 **10:00 价格**，在 09:30 现金开盘之后。已用 `interval=1h` 与 `interval=5m` 对齐核实 2026-10-05/06/07/08：`09:00` 一小时 bar 的 OHLCV 与 `09:00≤t<10:00` 的 5m 聚合**完全一致**。
   - 不安全 cutoff（09:30）：IC(SOXL) = **+0.2200**，IC(超额) = +0.2244
   - 安全 cutoff（09:00）：IC(SOXL) = **−0.0119**，IC(超额) = −0.0072
   - 即：+0.22 的“信号”完全来自前视 bar。
2. **回报口径不一致**：headline 是 open→close，native 对账用 open→次日 open。
3. **成本漏记**：换手按相邻日 target 差计，日内 round trip 未计足。
4. **基准误标**：所谓 “SOXX buyhold” 实为日内 session return。
5. **诊断内重选阈值**：阈值在各评估窗内被重新选择。
6. **Recorder 不完整**：只保存 summary，却声称有 panels。

## 更正内容（最小隔离，不重做有泄漏的 pipeline）

- `summary.json`：`decision` 改为 `invalid_for_promotion_look_ahead`，`promotion_eligible=false`，`objective_achieved=false`，并写入 `integrity` 区块；原指标保留在 `integrity.superseded_metrics_preserved`。
- `integrity_correction.json`：机器可验证的快照（bar span 审计 + 两个 cutoff 的 IC 复算）。
- `src/workflow/night_session_signal.py`：**无条件阻断**——`run_night_session_signal` 对任意目标目录（包括全新空目录）都抛 `blocked_known_integrity_defects`，不依赖是否已有 invalid 标记；legacy writer 同样拒绝写入任何 `promotion_eligible`/`objective_achieved` 或 `stable_*` 卡片。
- `src/analysis/night_session_integrity.py`：bar-end 可用性 helper（`bar_available_at` / `selectable_at`），供未来跨界 bar 未来扰动回归。

## 说明

- 两个 cutoff 的 IC 只是**源时间戳敏感性取证**，不是可交易结果，也不是新的 Qlib 研究；其余错误口径未修复，只隔离了时点。
- 后续任何修正研究必须使用**新的卡片身份**，不得在本卡上重跑或宣称成功。

## qlib Recorder（保留，不改）

- `6719efdbc2734e999feb282bcf7cd42e`（原始，invalid historical result）

> **再次声明：本卡 INVALID FOR PROMOTION，不得 eligible 或 objective achieved。**
