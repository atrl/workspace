# 现金信号：未见可靠的SOXX防守排序信息

0拟合、0回测。两组冻结输出：f696旧单模型与已保存base三seed重训ensemble。各取现金权重及tail评分，统一映射到09:47实际研究成交时钟后的SOXX H1/H5结果。与旧Ridge现金预测卡对象不同，不重复训练或校准旧路线。

944个交易区间全部保留，H1成熟944，H5成熟940，末端4行H5未知。H5观测下跌超过5%的事件有177行，因窗口重叠不能视为177个独立事件。

现金优势标签为负的SOXX持有期对数收益（现金利息假设0），正值意味着该区间现金优于SOXX；这是**毛信息标签**，未扣交易成本，不是可交易组合收益。H5 downside取5个区间的收盘/退出09:47价格相对最初入场价的最大下跌，非峰谷MaxDD、非全分钟极值。tail用其是否至少下跌5%定义。所有标签在最终退出09:49以后才可用；预测仍为此前冻结输入计算，末端不补造。

|评分|标签|指标|点估计|95%区块区间|
|---|---|---|---:|---:|
|frozen_current_cash_weight|cash_advantage_h1|Spearman|-0.0093|[-0.0640, 0.0419]|
|frozen_current_cash_weight|cash_advantage_h5|Spearman|-0.0182|[-0.1644, 0.0956]|
|frozen_current_cash_weight|downside_h5|Spearman|-0.0592|[-0.1808, 0.0448]|
|frozen_current_cash_weight|tail_h5|AUC|0.4656|[0.3907, 0.5349]|
|frozen_current_tail_probability|cash_advantage_h1|Spearman|-0.0457|[-0.1115, 0.0121]|
|frozen_current_tail_probability|cash_advantage_h5|Spearman|-0.0684|[-0.1758, 0.0365]|
|frozen_current_tail_probability|downside_h5|Spearman|0.0780|[-0.0565, 0.2073]|
|frozen_current_tail_probability|tail_h5|AUC|0.5710|[0.4640, 0.6667]|
|base_cash_weight|cash_advantage_h1|Spearman|0.0176|[-0.0247, 0.0616]|
|base_cash_weight|cash_advantage_h5|Spearman|0.0045|[-0.1003, 0.0965]|
|base_cash_weight|downside_h5|Spearman|-0.0030|[-0.1025, 0.0885]|
|base_cash_weight|tail_h5|AUC|0.5019|[0.4369, 0.5692]|
|base_tail_probability|cash_advantage_h1|Spearman|-0.0417|[-0.1141, 0.0194]|
|base_tail_probability|cash_advantage_h5|Spearman|-0.0722|[-0.1917, 0.0419]|
|base_tail_probability|downside_h5|Spearman|0.0650|[-0.0782, 0.2026]|
|base_tail_probability|tail_h5|AUC|0.5602|[0.4498, 0.6622]|

所有16项区间均包含各自无信息水平（相关0或AUC 0.5）。使用共同交易日的63日循环移动区块，固定seed、2000次重采样；保留H5重叠和末端缺失的日历结构。这些为未作多重研究校正的描述区间，不是独立确认性检验。

旧模型现金H1排序仅2024为正，H5为2024/2025正、2023/2026负；tail评分AUC点值约0.571，但区间[0.464,0.667]跨0.5。重训ensemble现金H1/H5排序约0.0176/0.0045，也没有可靠改善证据。不能把负点估计翻号后当成alpha。

本结果只覆盖预定的H1/H5 SOXX现金优势和该下跌定义，不能证明所有期限、市场状态或组合内部条件信息都不存在。现金权重由联合配置目标生成，不是校准概率；tail评分原训练标签并非本卡SOXX 5%事件，所以本卡只检验排序，不能拿它直接做概率校准误差。

决定：**diagnostic_only_no_promotion**。现有证据未支持“高现金能可靠预告SOXX接下来下跌”，但不因此把仓位机械改满，也不反转评分。结合完整账户未跑赢SOXX的证据，继续改现金阈值、换手惩罚或已有风险周期，缺乏信息依据。稳定净SOXX超额目标未完成，生产SIM未改。

下一步必须针对尚未被检验的新预测信息或明确的模型条件机制，而不是把当前弱排序改名或重复搜索失败参数；先对照已有信息路线和时间可用性，再决定是否值得新增拟合。需要独立验证，而不是从该复用窗口挑出正相关的年份。

验证：H5时点/末端未知/下跌标签合同测试通过，来源Recorder逐对象哈希核验，结果归档NAS并回读。[逐年统计](signal_statistics.csv)、[区块区间](block_intervals.csv)、[原始对象清单](verification_manifest.json)。原始每日预测/标签仅保存在Recorder。复用历史及原价格/真实成交资格限制继承父证据。
