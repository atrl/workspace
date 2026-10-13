# 本轮机制与边界

本轮固定检验“成本与自有账户状态进入连续持仓路径训练”，而不是继续给旧一步恢复回归加参数。目标仍是稳定净 SOXX 超额和回撤控制；27 次模型训练、66 个原生账户只是本轮预算。

公开机制参考：

- [Boyd 等，Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html)：持仓决策需要同时考虑回报、风险、交易成本与持有状态。本轮采用监督式历史路径优化，不声称复现该论文的凸多期规划。
- [Nystrup 等，Multi-Period Portfolio Selection with Drawdown Control](https://web.stanford.edu/~boyd/papers/multiperiod_portfolio_drawdown.html)：回撤状态和后续调整应在同一决策系统里评价。本轮的软回撤处罚没有未来概率保证，不将论文结果迁移成 SOXL 保证。
- [Busseti 等，Risk-Constrained Kelly Gambling](https://web.stanford.edu/~boyd/papers/kelly.html)：长期增长与回撤风险必须一起明确。该文依赖的概率模型及风险界不由当前 NN 自动获得；本轮只借鉴目标一致性，不声称具有其数学证书。
- [AQR，Portfolio Implementation](https://www.aqr.com/-/media/AQR/Documents/Insights/Working-Papers/AQR--Craftsmanship-Alpha.pdf)：理论信号可能被交易和风险管理侵蚀，实际持仓/费用链路需独立验证。本轮整数股数、逐笔最低佣金、月度阶梯及费率上限前向计算与独立账本核对；整数梯度为近似。

已有连续回撤预算、CPPI、波动率上限、趋势与 H5 不确定性负路线保留原结论，不重新挑其参数。新模型直接优化监督式连续路径的净 log 超额、负区块超额与超过 30% 的回撤违约；没有探索、critic、环境奖励或 RL。

公平对照为同结构 no_market / daily / minute，分别无外生市场通道、39 项日级输入加可用性标记、再加 12 项真实前一完整交易日分钟输入；三者均读取自己的持仓和已实现回撤。所有 seed、最终 epoch 与成本场景均保留，不能选最好种子、最低滑点或外窗阈值。

时点协议：前一完整交易日特征在次日决策前已知；09:45 原始分钟 close 假设 09:46 可用；09:46 作分配，09:47 下一观察到的开盘价成交。未来成交价仅用于结算，不是当前 NN 输入。日内成交价、已知估值价、真实价格佣金股数与公开复权 NAV 单位分开。半日市保留，所有真实交易日核对 XNYS 日历，原始坏分钟不补造，压力日账户保留。

原 f696/b472 模型及参数作为 outer 锚点，按本轮统一时点和已知净值监测协议运行；它与原 09:30/旧峰值协议的账户存在时间及估值差异，不能把两种协议的收益差当作新模型 alpha。开发期不使用 2022 年最终锚点冒充当年可用模型。原模拟盘 release、真实交易行为和固定父 Recorder 均不改动。

训练片段长度为 63 天状态形成加 126 天计分，边界重置现金，末段可能重叠；完整评估账户连续持仓。训练回撤/峰值与评估监测使用已知上午价格和当日收盘，不能解释为每一秒或完整分钟内回撤保证。最终需明确比较费用后收益、日级及监测回撤、CVaR、年度/滚动稳定、全部 seed 与区块区间。

所有历史区间均为复用诊断。历史门通过后也必须冻结身份、取得独立且完整配对的前向证据和实际信息/报价/佣金实现性；不把历史重放称作前向。失败则结束本固定机制，不在同窗口继续搜索参数；目标保持 active，下一限定机制先做独立工业容量利用率的原始发布/vintage 资格。
