# DATA Knowledge

> 最后更新: 2026-05-23
> 该方向负责数据管道、qlib store、外部数据接入和数据质量监控。

## 数据配置索引

| 配置 | 类别 | 状态 | 关键指标 | 卡片 |
|------|------|------|----------|------|
| SOXL Data Pipeline | data_pipeline | ✅ Operational | 2010-2025, 78 features | [card](cards/soxl_data_pipeline/card.md) |

## 核心结论

1. **SOXL canonical 数据覆盖**: `2010-03-11` 至 `2026-05-22`，包含 SOXL/SOXX OHLCV + VIX。
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

### 2026-05-21 Registry Auto-Refresh

恢复 SOXX/model 优化前发现一个 freshness surface 漂移：

- canonical panel、raw SOXL、raw SOXX、raw VIX、qlib calendar 均为
  `2026-05-20` / `4074` rows。
- `data/datasets.yaml` 仍显示 `2026-05-19` / `4073` rows。

根因是 scheduled paper runner 和 `scripts/update_market_data.py --local-only`
只刷新 master/raw/qlib，不刷新 registry；`update_data_registry.py --check`
也没有对照当前文件元数据。

现在：

- `scripts/update_market_data.py` 在 raw/qlib 后自动刷新
  `data/datasets.yaml`。
- `production/soxl_paper_trading.py` 的 scheduled data update 路径也刷新
  registry。
- `scripts/update_data_registry.py --check` 会重建当前元数据并比较
  stored rows/end/path，stale registry 会失败。
- `scripts/check_harness.py` 默认运行 `scripts/update_data_registry.py
  --check`。
- 当前 `data/datasets.yaml` 已刷新到 `SOXL=2026-05-20`,
  `SOXX=2026-05-20`, `soxl_soxx_vix_full=2026-05-20`。

证据卡：

- `runtime/data/card/data_registry_autorefresh_v1/notes.md`

## Tushare Optional Intake（2026-05-23）

- 新增 `src/soxl_quant/core/tushare.py` 和 `scripts/fetch_tushare_data.py`，用于把 Tushare Pro `us_daily_adj` 缓存到 `data/raw/external/tushare_us_daily_adj_<SYMBOL>.csv`。
- 新增 feature pack `tushare_us_daily_adj`，当前字段为 SOXX/SOXL turnover、amount、turnover spread、SOXX adj factor delta，全部 T+1 滞后后进入特征。
- 官方 Tushare 数据需要 token，且美股日线/复权行情是独立权限面；token 不写入 repo，只在运行时通过 `TUSHARE_TOKEN` 环境变量传入。
- 2026-05-23 使用新的运行时 token 成功拉取 `SOXX`/`SOXL` cache，覆盖到 `2026-05-22`：`SOXX=3833` rows，`SOXL=3860` rows。
- 训练前必须先刷新本地 Tushare cache 并跑因子 gate；本轮原始 pack 未过 IC/positive-ratio gate，所以仍不启动 Tushare feature retrain。

## Futu Small Order Probe（2026-05-23）

- Futu OpenAPI 是当前最接近 SOXL/SOXX 小单资金流的数据源，目标接口为 `get_capital_flow` / `get_capital_distribution`。
- 本机 `futu-api=10.02.6218` 可用，OpenD 安装路径存在，启动后 `127.0.0.1:11111` 端口可达。
- 初次 probe 被 OpenD 图形验证阻断；完成本地 phone/UI 验证后，`OpenQuoteContext` 可用并成功落地非 secret cache。
- 新增 Futu cache：
  - `futu_capital_flow_day_US_SOXL.csv` / `futu_capital_flow_day_US_SOXX.csv`：各 `251` rows，`2025-05-23` 至 `2026-05-22`。
  - `futu_capital_flow_intraday_US_SOXL.csv` / `futu_capital_flow_intraday_US_SOXX.csv`：各 `391` rows，覆盖 `2026-05-22`。
  - `futu_capital_distribution_US_SOXL.csv` / `futu_capital_distribution_US_SOXX.csv`：各 `1` row，更新到 `2026-05-22`。
  - `futu_snapshot_US_SOXL.csv` / `futu_snapshot_US_SOXX.csv`：各 `1` row，更新到 `2026-05-22`。
- 2026-05-22 latest daily small-flow readout：SOXL `sml_in_flow=+189.8M`，SOXX `sml_in_flow=-93.5M`。这是候选因子信号，不是训练许可；下一步必须跑 lagged factor gate。

## Futu Full History Check（2026-05-23）

- Futu `request_history_kline` 可分页拉取长历史 K 线，已新增：
  - `futu_history_kline_US_SOXL.csv`：`4060` rows，`2010-03-31` 至 `2026-05-22`。
  - `futu_history_kline_US_SOXX.csv`：`4076` rows，`2010-03-11` 至 `2026-05-22`。
- Futu `get_capital_flow(..., PeriodType.DAY)` 不是全历史小单资金流源。本轮显式测试 `2010-2011`、`2016`、`2020`、`2024` 窗口均返回 `0` rows；全范围 `2010-03-11` 至 `2026-05-22` 也只返回最近 `251` rows（`2025-05-23` 至 `2026-05-22`）。
- 结论：Futu K 线可作为全量 OHLCV 对照源；Futu 小单流只能作为短历史/live overlay 候选。除非找到能覆盖训练全历史的小单流来源，否则不要用 Futu 小单流开全历史训练。

证据卡：

- `runtime/data/card/futu_full_history_probe_v1/notes.md`

## Futu SOXL/SOXX 1m Intraday Cache（2026-06-06）

- `scripts/fetch_intraday_data.py` 现在给 Futu SDK 配置 repo 内可写
  `runtime/futu_sdk_home`，并对 OpenD 连接和 `request_history_kline` 调用加
  timeout，避免分钟级分页卡住时无限等待。
- 使用本机 Futu OpenD 成功拉取并落盘 SOXL/SOXX 真实 1m regular-session K 线：
  - `data/raw/intraday/futu/soxl_1m.parquet`：`236040` rows。
  - `data/raw/intraday/futu/soxx_1m.parquet`：`236040` rows。
  - `data/processed/intraday/soxl_soxx_1m.parquet`：`472080` rows，`2024-01-02 09:31 America/New_York` 至 `2026-06-04 16:00 America/New_York`，`608` trading days。
  - `data/raw/external/intraday_features_soxl_1m.csv`：`608` T+1 daily-safe rows，`2024-01-03` 至 `2026-06-05`。
- Futu 1m timestamp 表现为 minute bar close time，regular session 每日 `390`
  bars，首根通常为 `09:31`，不应误判为缺少 `09:30`。
- 这批分钟数据足够开始 2024-2026 的分钟微结构因子筛选/模型辅助实验，但还不是
  SOXL 上市以来全历史分钟覆盖；训练前仍需按切片验证并保持 T+1 滞后。

证据卡：

- `runtime/data/card/quantbench_intraday_soxl_v1/quantbench_intraday_soxl_v1.ipynb`
- `runtime/data/card/quantbench_intraday_soxl_v1/summary.json`

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
