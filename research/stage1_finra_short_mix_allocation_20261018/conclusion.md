# Stage1 FINRA 场外卖空标记占比差：固定研究

完成B_FINRAcontext与C_FINRA_short_mix各3个seed，共6次固定300轮拟合。主3014条、辅助20741条；原39主输入、16共享输入、损失、样本与Stage1控制链不变。仅主allocation接6项新坐标，B/C只有q信息不同。

新增B/C各0/2/5bp共6个原生账户；严格复用A、原shared B、SOXX、9097、27baa共15账户；每账户944区间。各账户保留原policy与独立成本状态，原始股数及分红应收经Qlib核对。

|方案|滑点bp|累计净收益%|收盘最大回撤%|CVaR5%|
|---|---:|---:|---:|---:|
|A_active_research_replay|0|229.686|-29.487|-4.389|
|A_active_research_replay|2|280.650|-27.060|-4.211|
|A_active_research_replay|5|41.521|-30.517|-4.026|
|B_shared|0|280.383|-25.709|-4.215|
|B_shared|2|270.223|-25.766|-4.215|
|B_shared|5|265.858|-25.819|-4.204|
|B_FINRAcontext|0|45.061|-28.815|-3.595|
|B_FINRAcontext|2|21.495|-30.411|-3.641|
|B_FINRAcontext|5|44.405|-30.340|-3.682|
|SOXX|0|380.588|-40.863|-4.905|
|SOXX|2|380.568|-40.867|-4.905|
|SOXX|5|380.538|-40.872|-4.906|
|strong9097|0|441.681|-26.812|-4.399|
|strong9097|2|434.269|-26.881|-4.400|
|strong9097|5|423.730|-26.956|-4.401|
|C_FINRA_short_mix|0|101.030|-26.641|-3.754|
|C_FINRA_short_mix|2|83.291|-26.688|-3.668|
|C_FINRA_short_mix|5|48.015|-30.434|-3.563|
|additional27baa|0|485.498|-27.171|-4.454|
|additional27baa|2|477.826|-27.223|-4.455|
|additional27baa|5|466.073|-27.317|-4.457|

决定：**not_promoted_fixed_FINRA_short_mix_v1**。
失败或未取得的门：net_relative_wealth_all_costs, sampled_drawdown_30pct, complete_annual_70pct, rolling126_70pct, paired_block_lower_positive, fresh_fullminute_risk_qualified, qualified_independent_forward。

所有年度、季度、126交易日窗口及每个seed分类诊断均保留；不择窗、择seed或按分类分数取消账户。全窗财富和配对CI对5个required比较，年度/rolling门相对SOXX。首尾partial年度单列。

FINRA ShortVolume包含ShortExemptVolume，CNMS仅场外RTH子集；q不是shortinterest、情绪、净库存或作者operationalshorting复现。18:00仅历史可用性假设，当前字节修订与首次接收未认证。结果不能授予PIT或前向资格。

采样回撤不认证全分钟风险，重复市场历史不独立。生产未改，稳定超额目标未完成。

关闭固定q_v1，不反号、平滑、择窗口、调损失或控制器救援。

Qlib Recorder：`8e87af53d57a4022bcf19b5dbc690e33`；来源与NAS身份见summary.json、verification_manifest.json、nas_binding.json。

完整年度及区间差异（2bp）：
2024年C净收益1.171%，SOXX为17.428%；相对财富0.861554。
2025年C净收益14.046%，SOXX为43.691%；相对财富0.793690。
C相对SOXX的126日滚动正超额比例在0/2/5bp下仅29.18%/16.97%/6.11%；15个预定配对置信下界均不大于0。q相对同背景B的全窗财富在三个成本均有改善，但背景B显著落后于原shared B，不能把此弱对照改善认定为稳定超额。
5bp下C收盘最大回撤−30.434%、采样点−31.308%，30%风险门失败；CVaR相对SOXX门通过。成本改变独立账户状态路径，收益差并非单纯费用相减。
独立结果评价已接受关闭固定版本：ACCEPT_NOT_PROMOTED_FIXED_EXPERIMENT；执行者未自签晋升，整体Goal未完成。

一次固定失败归因（仅本轮Recorder）：2bp下，C在全部4个年度（2完整、2partial）均胜背景B，但全部年度均不及原shared B及SOXX；16季度分别胜10/16、4/16、4/16，完整14季度与partial2季度另列。所有3成本的180项年/季度配对保存在failure_period_comparison.csv。

C平滑提议平均名义beta为0.866，低于背景B的0.905；2bp实际持仓beta反而为0.597，高于B的0.506，风险非零日641对761。同成本B/C风险状态227日不同。原shared B实际beta为0.849、现金46.43%，新背景B为0.506/62.94%、C为0.597/58.71%。这些是不同决策/持仓时点及内生路径的描述，不能相加为精确cash drag分解，也不能归因于单一controller。

C在0对2、0对5、2对5bp之间分别有200、524、552日风险状态不同，费用与自身状态反馈共同影响结果。6个新账户各3778个点的原生核对最大相对误差均为0；没有发现可以解释失败的账务实现误差。原shared B的raw/smoothed及逐日riskstate未复制进本轮Recorder，本次未额外读旧源或倒造反事实。详见failure_attribution.json和小表。
独立交付读回：NAS冷恢复102个产物的大小及哈希全部一致，原生summary/gates、主/辅助/来源Dataset直接prepare及6个已存原生账本通过。0重拟合、0账户重放。详见independent_evaluation.json与readback_verification.json。
