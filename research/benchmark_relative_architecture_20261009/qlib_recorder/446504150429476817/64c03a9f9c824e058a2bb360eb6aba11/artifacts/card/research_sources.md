# 外部机制与证据边界

本卡测试一个新机制：用已成熟的专家净收益反馈更新 fixed-share 权重，并与使用相同专家及初始权重的固定混合对照比较。专家、成本、更新时点、权重约束和评价窗口以 [plan.json](plan.json) 为唯一参数来源。本文不记录运行结果，也不把论文结果当作 SOXL/SOXX 的超额证据。

## 八项一手来源

| 来源 | 支持的机制 | 对本卡的约束 |
| --- | --- | --- |
| [Herbster、Warmuth：Tracking the Best Expert，1998](https://mwarmuth.bitbucket.io/pubs/J39.pdf) | 收到结果后更新专家损失，再分享少量权重，使失势专家能够恢复；避免长期锁定过去赢家 | 本卡只借用在线更新结构。论文的损失条件和 regret 界不直接保证含交易成本的主账户跑赢 SOXX；不得用未成熟收益更新 |
| [Boyd 等：Multi-Period Trading via Convex Optimization，2017](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf) | 将预期收益、主动风险、成本和预测误差分离；收益预测不确定性可对应基准偏离的 L1 惩罚 | 优化方法不创造收益预测优势。下一独立实验可研究 uncertainty-aware allocator，须先取得可信的过去 OOS 校准证据 |
| [Gârleanu、Pedersen：Dynamic Trading with Predictable Returns and Transaction Costs，2013](https://www.nber.org/papers/w15205) | 交易成本与信号衰减共同决定目标和调整速度 | 已有费用结算不能被称为新增机制。部分调整仅在预测持续性得到验证后独立测试；平滑也可能延迟必要退出 |
| [Moreira、Muir：Volatility-Managed Portfolios，2017](https://www.nber.org/papers/w22208) | 高波动时降低风险在其因子样本中改善风险调整收益 | 支持研究假设，不支持把 SOXL 波动上限等同于稳定超额；不得套用全样本尺度系数 |
| [Cederburg 等：On the Performance of Volatility-Managed Portfolios，2020](https://www.lehigh.edu/~xuy219/research/COWY.pdf) | 广泛股票策略的实时比较显示，波动管理并不系统性占优；参数结构变化会削弱样本内优势 | 与前项共同阅读。风险下降、回归 alpha、可实施净财富改善必须分别核验 |
| [Bailey、López de Prado：The Deflated Sharpe Ratio，2014](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) | 多重试验和非正态收益会夸大样本表现；需要记录失败试验与搜索规模 | 本轮一个候选不代表全项目只有一次试验。历史试验统计不完整时，不伪造独立试验数或声称 DSR 已解决选择偏差 |
| [Bailey 等：The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) | 根据完整候选结果诊断选择过程的过拟合，提供 CSCV/PBO 框架 | 历史重采样不能恢复 fresh holdout，也不能替代严格先后时序的验证。本卡不因引用而声称已运行 PBO |
| [Direxion：SOXL/SOXS 官方产品说明](https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs) | SOXL 的杠杆目标为其标的指数的单日收益，长期结果受每日复利路径影响 | SOXL 不是 SOXX 长期收益的三倍；只使用实际 SOXL 价格回放，不能用三倍 SOXX 收益合成长历史 |

## 最小可证伪问题

在专家集合、初始组合、成本、现金收益和交易时点相同的条件下，成熟反馈驱动的在线更新，能否改善固定混合的成本后相对财富及分期稳定性？固定混合隔离在线更新的增量；显式 SOXX 基准检验整个主动组合是否值得承担。杠杆与防御专家的独立结果用于解释收益来源，不从中事后挑选赢家替代主候选。

专家自身净收益的加权平均不等于主账户净收益。主账户必须按自己的实际持仓、现金、权重漂移和交易费用独立结算，并用原生 Qlib 回放核验。在线权重不使用当时尚不可得的开盘价、未来收益或后验 regime 标签。

## 研究边界

- 既有历史已经被多轮研究使用。即使本卡按时间执行、通过未来扰动测试，也只能提供复用历史上的机制诊断。
- 本轮现金收益设为零；这不代表现实现金工具无收益，也没有计入特定现金工具的费用、税务或可交易性。引入有来源的现金收益应另建数据版本与实验。
- 初始状态、权重约束及 fixed-share 的恢复权重可能持续分配给弱专家；候选允许输给固定混合或 SOXX，失败也完整归档。
- purging 解决标签重叠和成熟边界；冻结候选、记录所有尝试及连续前向证据解决另外的选择问题，彼此不能替代。
- qlib API 行为以[官方 Workflow 文档](https://qlib.readthedocs.io/en/latest/component/workflow.html)为准；研究原理不能作为绕过 Qlib 数据、Recorder 或原生回测的理由。
