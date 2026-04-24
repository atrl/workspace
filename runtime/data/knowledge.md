# DATA Knowledge

> 最后更新: 2026-04-24  
> 该方向负责数据管道、qlib store、外部数据接入和数据质量监控。

## 核心结论

1. **qlib store 已构建 5 只 3x ETF**: SOXL, TQQQ, UPRO, SPXL, TECL。
2. **外部数据源注册在 data/datasets.yaml**: VIX、宏观指标通过统一 registry 管理。
3. **训练/验证/测试切分**: 2010-2018 train / 2019 valid / 2020-2024 test（固定切分）。
4. **Walk-forward 窗口**: 2020, 2022, 2023-2024 三个独立测试窗口用于验证泛化。

## 数据集清单

| 数据集 | 路径 | 说明 |
|--------|------|------|
| SOXL qlib store | `data/qlib/soxl/` | 主训练标的，日频 |
| TQQQ/UPRO/SPXL/TECL | `data/qlib/{symbol}/` | 泛化测试标的 |
| VIX 外部数据 | `data/raw/external/vix.csv` | 日频 VIX spot |
| datasets.yaml | `data/datasets.yaml` | 统一数据注册表 |

## 特征工程管道

```
raw CSV → qlib init → DataHandlerLP → engineer_feature_frame()
                                    ↓
                          CORE_FEATURE_COLUMNS (~50 features)
                                    ↓
                          feature_packs (transition, kelly, etc.)
```

## 有效方法 ✅

| 方法 | 说明 |
|------|------|
| qlib binary store | 标准化存储，加速训练数据加载 |
| datasets.yaml 注册 | 统一管理 raw/processed/qlib 三级数据 |
| 外部数据注入 | VIX 等外部特征通过 external data frame 合并 |

## 待验证方向

- [ ] 分钟频数据接入（当前仅日频）
- [ ] 宏观因子（利率、信用利差）与 LETF 的相关性面板
- [ ] 数据刷新自动化（当前需手动触发）

---

## 实验记录（参考）

见 memory archive 原始记录，按日期排序。
