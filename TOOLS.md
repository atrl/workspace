# TOOLS.md - Qlib 使用规范

## 核心原则

**Qlib 是唯一量化框架**。所有策略、因子、回测必须基于 Qlib 实现。

---

## 环境配置

### 安装
```bash
# 正确的包名
pip install pyqlib  # 不是 "qlib"！

# 验证安装
python -c "import qlib; print(qlib.__version__)"
```

### 初始化
```python
import qlib

# 方式1: 使用本地数据
qlib.init(provider_uri='qlib_data/bin')

# 方式2: 使用 Qlib 云数据 (需网络)
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')
```

### 虚拟环境
```bash
# 项目专用 venv
./qlib_venv/bin/python3 script.py
```

---

## Use Case 1: 数据加载

### 基础数据获取
```python
from qlib.data import D

# 获取单只股票的收盘价
df = D.features(
    instruments=['SOXL'],
    fields=['$close', '$volume'],
    start_time='2020-01-01',
    end_time='2026-03-21'
)
```

### QlibDataLoader (推荐)
```python
from qlib.data.dataset.loader import QlibDataLoader

# 批量加载多个特征
loader = QlibDataLoader(config={
    'feature': [
        '$close',
        '$volume',
        'Mean($close, 20)',
        'Std($close, 20)',
        'RSI($close, 14)'
    ],
    'label': ['Ref($close, -5) / $close - 1']  # 未来5日收益 (用于训练)
})

data = loader.load(instruments=['SOXL'], start_time='2020-01-01')
```

### CSV 回退策略 (当 Qlib 数据不可用时)
```python
import pandas as pd

# 仅在 Qlib Provider 失败时使用
df = pd.read_csv('data/symbols/SOXL.csv')
df['date'] = pd.to_datetime(df['date'])

# 手动计算 Qlib 风格特征
df['Mean($close, 20)'] = df['close'].rolling(20).mean()
df['Std($close, 20)'] = df['close'].rolling(20).std()
```

---

## Use Case 2: 因子计算 (Qlib Expression)

### 核心表达式
| 表达式 | 含义 | 示例 |
|--------|------|------|
| `$close` | 当日收盘价 | - |
| `Ref($close, N)` | N日前数据 | `Ref($close, 5)` = 5日前收盘 |
| `Mean($close, N)` | N日均值 | `Mean($close, 20)` = MA20 |
| `Std($close, N)` | N日标准差 | `Std($close, 20)` |
| `Sum($volume, N)` | N日求和 | - |
| `Rank(X)` | 横截面排名 | - |
| `Corr(X, Y, N)` | N日相关性 | `Corr($close, $volume, 20)` |
| `If(cond, a, b)` | 条件判断 | `If($close > MA20, 1, 0)` |
| `Max(X, N)` / `Min(X, N)` | N日最大/最小 | - |

### 复合因子示例
```python
# 趋势得分 (4个时间框架)
factors = [
    'If($close > Mean($close, 5), 0.2, 0)',   # 短期
    'If($close > Mean($close, 20), 0.3, 0)',  # 中期
    'If($close > Mean($close, 60), 0.3, 0)',  # 长期
    'If(Mean($close, 5) > Mean($close, 20), 0.2, 0)',  # 均线交叉
]

# 波动率
'Std(Ref($close, 1) / $close - 1, 20) * Sqrt(252)'  # 年化波动率

# 反转信号
'-(Ref($close, 5) / $close - 1)'  # 5日反转
```

---

## Use Case 3: 策略实现

### 继承 BaseStrategy
```python
from qlib.contrib.strategy import BaseStrategy
from qlib.backtest.decision import TradeDecision, Order

class MyStrategy(BaseStrategy):
    def __init__(self, max_position=0.75, **kwargs):
        super().__init__(**kwargs)
        self.max_position = max_position

    def generate_trade_decision(self, execute_result=None):
        """核心接口 - 每个交易日调用"""
        trade_date = self.trade_calendar.get_trade_step()
        current_data = self.get_current_data()

        # 计算信号
        signal = self._calculate_signal(current_data)

        # 生成订单
        orders = []
        if signal > 0.5:
            orders.append(Order(
                stock_id='SOXL',
                amount=signal * self.max_position,
                direction=Order.BUY
            ))

        return TradeDecision(orders)

    def _calculate_signal(self, data):
        """自定义信号逻辑"""
        # 使用预计算的因子
        return data['trend_score'].iloc[-1]
```

---

## Use Case 4: 回测

### 标准回测配置
```python
from qlib.contrib.backtest import backtest
from qlib.contrib.evaluate import risk_analysis

# 回测参数
config = {
    'start_time': '2020-01-01',
    'end_time': '2026-03-21',
    'account': 100000,
    'benchmark': 'SOXL',
    'exchange_kwargs': {
        'limit_threshold': 0.095,  # 涨跌停限制
        'deal_price': 'close',     # 以收盘价成交
        'open_cost': 0.001,        # 开仓成本 0.1%
        'close_cost': 0.001,       # 平仓成本 0.1%
        'min_cost': 5,             # 最低佣金
    }
}

# 运行回测
portfolio = backtest(strategy=my_strategy, **config)

# 分析结果
metrics = risk_analysis(portfolio)
print(f"Sharpe: {metrics['sharpe']:.2f}")
print(f"Max DD: {metrics['max_drawdown']:.2%}")
```

### 回测结果分析
```python
from qlib.contrib.report import analysis_position

# IC 分析 (预测能力)
ic = analysis_position.score_ic(pred, label)

# 风险分析
risk = analysis_position.risk_analysis(position)

# 可视化报告
analysis_position.report(position, label)
```

---

## Use Case 5: 模型训练

### 使用 Qlib 内置模型
```python
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH

# 准备数据集
dataset = DatasetH(
    handler={
        'class': 'Alpha158',  # 158维因子
        'module_path': 'qlib.contrib.data.handler'
    },
    segments={
        'train': ('2018-01-01', '2022-12-31'),
        'valid': ('2023-01-01', '2024-12-31'),
        'test': ('2025-01-01', '2026-03-21')
    }
)

# 训练模型
model = LGBModel(loss='mse', colsample_bytree=0.8)
model.fit(dataset)

# 预测
predictions = model.predict(dataset)
```

---

## ⚠️ 量化问题 Checklist

### 1. 数据问题 (Data Issues)

| 问题 | 症状 | 检查方法 | 解决方案 |
|------|------|---------|---------|
| **Look-ahead Bias** | Sharpe 异常高 (>2) | 用 CSV 对比验证 | 只用 `data/symbols/*.csv` |
| **Survivorship Bias** | 只包含存活股票 | 检查退市股票 | 包含已退市股票数据 |
| **Point-in-Time** | 财务数据延迟 | 检查公告日期 | 使用 PIT 数据库 |
| **数据缺失** | NaN 值 | `df.isnull().sum()` | 前值填充或剔除 |
| **价格异常** | 价格跳变 | 检查拆股/分红 | 复权处理 |
| **时区问题** | 日期对不上 | 检查 UTC vs Local | 统一用 UTC |

**验证规则**: Sharpe > 1.5 必须通过 CSV 验证

---

### 2. 因子问题 (Factor Issues)

| 问题 | 症状 | 检查方法 | 解决方案 |
|------|------|---------|---------|
| **IC 不稳定** | IC 时正时负 | 计算 IC 时间序列 | 增加 smoothing 或放弃 |
| **IC 衰减快** | IC(lag1) ≈ 0 | IC Decay 分析 | 缩短持仓周期 |
| **因子相关性高** | 多因子冗余 | 计算相关矩阵 | 正交化或剔除 |
| **过拟合** | IS 好 OOS 差 | Walk-forward 测试 | 减少参数/正则化 |
| **非线性** | 线性模型失效 | 残差分析 | 尝试非线性模型 |

**验证规则**: |IC| < 0.02 的因子直接弃用

---

### 3. 回测问题 (Backtest Issues)

| 问题 | 症状 | 检查方法 | 解决方案 |
|------|------|---------|---------|
| **交易成本低估** | 回测远好于实盘 | 检查 cost 参数 | 设置 0.1%+ 双边成本 |
| **滑点忽略** | 成交价理想化 | 检查 deal_price | 加入滑点模型 |
| **流动性不足** | 大单无冲击 | 检查成交量 | 限制仓位比例 |
| **涨跌停** | 无法成交 | 检查 limit_threshold | 设置涨跌停过滤 |
| **未来函数** | 使用当日收盘买入 | 检查信号时点 | 用次日开盘价 |
| **过度交易** | 换手率极高 | 计算 turnover | 加入持仓平滑 |

**验证规则**: 年换手率 > 20x 需重新评估成本

---

### 4. 风控问题 (Risk Control Issues)

| 问题 | 症状 | 检查方法 | 解决方案 |
|------|------|---------|---------|
| **回撤过大** | Max DD > -30% | 计算 max_drawdown | 多级回撤保护 |
| **集中度风险** | 单票 > 50% | 检查持仓分布 | 设置权重上限 |
| **杠杆失控** | 杠杆 > 2x | 检查 gross exposure | 硬性杠杆限制 |
| **波动率过高** | 年化 vol > 40% | 计算波动率 | 波动率目标管理 |
| **尾部风险** | 极端日损失大 | 检查 VaR/CVaR | 压力测试 |
| **相关性崩塌** | 危机时分散失效 | 检查危机期相关性 | 加入对冲资产 |

**验证规则**: Max DD < -20% 才能上线

---

### 5. 模型问题 (Model Issues)

| 问题 | 症状 | 检查方法 | 解决方案 |
|------|------|---------|---------|
| **过拟合** | Train >> Test | 对比训练/测试指标 | 减少特征/正则化 |
| **样本外失效** | OOS 表现差 | Walk-forward | 滚动训练 |
| **参数敏感** | 参数微调结果巨变 | 参数敏感性分析 | 稳健参数区间 |
| **特征重要性** | 模型依赖单一特征 | 检查 feature importance | 特征多样性 |
| **预测偏差** | 系统性高估/低估 | 检查预测分布 | 偏差校正 |

**验证规则**: OOS Sharpe < IS Sharpe * 0.8 需警惕

---

### 6. 执行问题 (Execution Issues)

| 问题 | 症状 | 检查方法 | 解决方案 |
|------|------|---------|---------|
| **延迟** | 信号到成交有延迟 | 检查执行时间戳 | 降低延迟要求 |
| **市场冲击** | 大单推高价格 | 检查成交均价 vs VWAP | TWAP/VWAP 算法 |
| **部分成交** | 订单未完全成交 | 检查成交率 | 分单/限价单 |
| **系统故障** | 订单丢失 | 检查系统日志 | 冗余/监控 |
| **市场关闭** | 非交易时段信号 | 检查交易时段 | 交易时段过滤 |

**验证规则**: 实盘 vs 回测偏差 > 20% 需排查

---

### 7. Qlib 特有问题 (Qlib-Specific Issues)

| 问题 | 症状 | 检查方法 | 解决方案 |
|------|------|---------|---------|
| **包名错误** | `import qlib` 失败 | 检查安装包名 | `pip install pyqlib` |
| **Provider 数据问题** | 数据含未来信息 | 用 CSV 对比 | CSV-First Policy |
| **初始化失败** | `qlib.init()` 报错 | 检查 provider_uri | 确保 `qlib_data/bin` 存在 |
| **回测超时** | 长时间无响应 | 检查数据量/因子数 | 缩短时间范围 |
| **内存溢出** | OOM | 检查数据加载方式 | 分批加载 |

**验证规则**: 所有 Qlib 回测必须 CSV 二次验证

---

## 🔍 策略上线前 Checklist

### 必查项 (Must Check)

- [ ] **Sharpe < 1.5** - 超过需 CSV 验证
- [ ] **Max DD > -20%** - 风控底线
- [ ] **OOS Sharpe > 0.5** - 样本外有效
- [ ] **年换手率 < 20x** - 成本可控
- [ ] **IC |mean| > 0.02** - 因子有效
- [ ] **Walk-forward 验证** - 非过拟合
- [ ] **CSV 二次验证** - 数据无误
- [ ] **压力测试通过** - 极端场景 OK

### 建议检查 (Recommended)

- [ ] 参数敏感性分析
- [ ] 特征重要性检查
- [ ] 持仓集中度 < 30%
- [ ] 年化 vol < 30%
- [ ] 连续亏损保护
- [ ] 交易时段过滤

---

## 踩坑记录

### 1. 包名错误
```bash
# ❌ 错误
pip install qlib  # 这是另一个包！

# ✅ 正确
pip install pyqlib  # Microsoft Qlib
```

### 2. Look-ahead Bias
```python
# ❌ 错误 - Qlib Provider 可能包含未来数据
df = D.features(instruments=['SOXL'], fields=['$close'])

# ✅ 正确 - 使用本地 CSV 验证
csv_df = pd.read_csv('data/symbols/SOXL.csv')
# 对比验证

# 规则: 任何 Sharpe > 1.5 必须通过 CSV 验证
```

### 3. 数据初始化失败
```python
# 如果 qlib_data/bin 不存在
qlib.init(provider_uri='qlib_data/bin')  # 会失败

# 解决方案: 使用 CSV 回退
if not Path('qlib_data/bin').exists():
    df = pd.read_csv('data/symbols/SOXL.csv')
```

### 4. 回测超时
```python
# 长时间回测可能超时
# 解决: 缩短时间范围或简化因子
config['end_time'] = '2025-12-31'  # 缩短到1年
```

---

## 文件结构

```
workspace/
├── qlib_data/           # Qlib 本地数据
│   └── bin/            # 二进制数据文件
├── data/
│   └── symbols/        # CSV 数据 (真值源)
│       ├── SOXL.csv
│       └── ...
├── production/         # 生产策略
├── research/           # 研究代码
└── scripts/
    └── alpha/          # 因子挖掘脚本
```

---

## 代码模板

### 最小可运行示例
```python
#!/usr/bin/env python3
import qlib
from qlib.data import D
import pandas as pd

# 初始化
qlib.init(provider_uri='qlib_data/bin')

# 加载数据
df = D.features(
    instruments=['SOXL'],
    fields=['$close', 'Mean($close, 20)', 'Std($close, 20)'],
    start_time='2020-01-01',
    end_time='2026-03-21'
)

# 计算
df['returns'] = df['$close'].pct_change()
df['vol'] = df['Std($close, 20)'] * (252**0.5)

print(df.tail())
```

---

## 参考资料

- **官方文档**: https://qlib.readthedocs.io/
- **GitHub**: https://github.com/microsoft/qlib

---

**版本**: v2.1
**更新日期**: 2026-04-15
