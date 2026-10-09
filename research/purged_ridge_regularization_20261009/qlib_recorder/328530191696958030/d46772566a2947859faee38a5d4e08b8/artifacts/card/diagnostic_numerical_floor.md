# 数值超额门诊断（原结果不改写）

来源Recorder d46772566a2947859faee38a5d4e08b8；只读保存的原生策略return。selected开发2/7正年为真实2017/2022增益，不是浮点噪声。fixed10开发raw3正年中有1个机器正值；252日滚动raw651/1511正，超过1e-8累计log阈值的只有415。selected25开发467个raw正滚动窗中48个仅是机器噪声；有意义正窗419，胜率从30.91%变27.73%。selected后段完全SOXX，但raw正滚动窗10/25bp仍有58/228个；超过1e-8后均为0。所有此类近零窗最大绝对累计log差约3.1e-15。

当前aggregate、median、leave-one-year均明显负，整体失败不受影响。旧收益/表格/计划不能重写；新计划可以预先固定numerical_log_excess_floor=1e-8，约0.0001bp财富变化。aggregate、annual、LOO、rolling的单位都是所比较区间的累计log差，均以>floor计正；abs<=floor明确为数值不可区分。bootstrap lower仍保留日均单位，门判断需N*lower_daily>floor，避免跨单位混用。该阈值不替代效应大小或金融意义门，更不改变MaxDD/CVaR要求。

新对账测试宜包括不同capital的完全SOXX路径、长纯SOXX后缀、极小正负噪声和刚跨阈值的确定性序列，同时保留原始return而不round到零。详细逐年/逐窗计数在本目录numeric_annual_excess.csv、numeric_rolling_excess.csv与numeric_gate_effect.csv。
