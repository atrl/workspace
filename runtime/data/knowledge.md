# DATA Knowledge

> 最后更新: 2026-05-14
> 该方向负责数据管道、qlib store、外部数据接入和数据质量监控。

## 数据配置索引

| 配置 | 类别 | 状态 | 关键指标 | 卡片 |
|------|------|------|----------|------|
| SOXL Data Pipeline | data_pipeline | ✅ Operational | 2010-2025, 78 features | [card](cards/soxl_data_pipeline/card.md) |

## 核心结论

1. **SOXL canonical 数据覆盖**: `2010-03-11` 至 `2026-04-27`，包含 OHLCV + VIX。
2. **外部数据**: VIX3M, VIX9D, SKEW, VVIX 已接入（qlib store）。
3. **Feature 稳定性**: 50 core features + 14 rulebase + 7 academic + 7 transition。
4. **VIX-free 替代**: regime_risk_no_vix 在 VIX 缺失时可用。

## Canonical Panel Sync（2026-05-09）

### 问题

仓库内已经存在最新的 canonical panel `data/processed/panels/soxl_soxx_vix_full.csv`，但下游 `raw/` 面和部分历史认知还停留在更旧的 `2026-04-01` 状态，容易让后续 agent 误判为“数据没更新”。

### 这次修复

- `scripts/update_market_data.py` 新增 `--local-only`
- canonical panel 现在会同步镜像到：
  - `data/raw/symbols/SOXL.csv`
  - `data/raw/symbols/SOXX.csv`
  - `data/raw/external/VIX.csv`
- `scripts/update_data_yahoo_api.py` 收口为 canonical updater 的兼容入口，避免两套刷新逻辑漂移

### 当前环境限制

本自动化运行环境对 Yahoo 端点存在 **DNS / 网络不可达** 问题。这里的 live refresh 失败根因不是 `yfinance` 速率限制，而是当前 run 无法解析外部域名。

### 2026-05-10 Revalidation

- `data/processed/panels/soxl_soxx_vix_full.csv`、`data/raw/symbols/SOXL.csv`、`data/raw/symbols/SOXX.csv`、`data/raw/external/VIX.csv`、`data/qlib/soxl/calendars/day.txt` 仍然全部对齐到 `2026-04-27`
- 直接请求 `https://query1.finance.yahoo.com/v8/finance/chart/SOXL` 仍然报 `NameResolutionError`
- 结论：当前 automation 环境下“数据更新”已收敛为环境问题，不是仓库内部数据面不一致

### 2026-05-14 Revalidation

- `PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --local-only` 成功同步 canonical panel 到 raw mirrors 与 `data/qlib/soxl/`。
- `PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py` 成功刷新 `data/datasets.yaml`，`generated_at=2026-05-14T02:02:18Z`。
- Canonical panel、raw SOXL/SOXX/VIX、qlib calendar 仍然全部对齐到 `2026-04-27`，没有更新的本地源可继续推进。
- 直接请求 Yahoo chart endpoint 仍然报 `NameResolutionError`，因此 live refresh 依然是当前 automation 环境网络限制，不是数据管线逻辑问题。

### 操作指引

- 在**无网络**自动化环境里，用：
  - `PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --local-only`
- 在**可联网**环境里，再用：
  - `PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py`
  - 或 `PYTHONPATH=. qlib_venv/bin/python scripts/update_data_yahoo_api.py`
- 完成后始终刷新 registry：
  - `PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py`

### 2026-05-15 Revalidation

- `PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --local-only` 成功同步 canonical panel 到 raw mirrors，并重建 `data/qlib/soxl`。
- `PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py` 成功刷新 registry。
- Canonical panel、raw SOXL/SOXX/VIX、qlib calendar 继续全部对齐到 `2026-04-27`。
- 直接请求 Yahoo chart endpoint 仍然报 `NameResolutionError`，当前 automation 运行仍无法 live refresh。

## 50 Core Features 主题分类

| 主题 | 特征数 | 关键驱动 |
|------|--------|----------|
| Trend / Anchor | 6 | 长期趋势锚定 |
| Momentum | 7 | 短期方向 |
| Volatility | 6 | 风险体制 |
| Drawdown | 7 | 压力深度与持续时间 |
| Structure | 4 | 结构性弱势 |
| Downside | 4 | 下行不对称性 |
| VIX | 4 | 市场恐惧指标 |
| Tracking Error | 4 | LETF 特有摩擦 |
| Velocity | 5 | 体制变化速率 |
| Proxy | 4 | 衍生绩效指标 |
| Composite | 3 | 多信号 stress 指数 |

## 数据泄露修复（2026-04-27）

### 问题
`engineer_feature_frame()` 中的 `regime_risk`、`vol_stress_score` 等核心特征使用了**全序列** `.rank(pct=True)`。由于 feature frame 跨越 train+valid+test（`start_time=train_seg[0], end_time=test_seg[1]`），训练数据获得了未来信息。

### 修复
- 8 处 `.rank(pct=True)` → `_rolling_percentile(..., 252)`
- `jump_variation_5` 的 full-series std fallback → `expanding().std()`

### 影响评估
- `balanced` test_proxy Return 下降 ~12%（1958.9% → 1729.3%）
- 泄露确实存在但非灾难性
- 修复后 MaxDD 反而改善（-52.9% → -50.1%）

### 教训
- 所有 percentile/rank 操作必须用滚动窗口
- 已有 `_rolling_percentile()` 安全函数，但未全面应用
- 未来添加新特征时必须强制审查是否有 look-ahead

## 待验证方向

- [ ] 多资产联合数据管道（TQQQ/UPRO/SPXL/TECL）
- [ ] 实时数据延迟模拟
- [ ] 修复泄露后重新训练子模型
