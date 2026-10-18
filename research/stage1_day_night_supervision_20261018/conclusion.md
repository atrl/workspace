# 固定 S/Z 监督：实现与唯一合成合同完成

新workflow保留原PG3014×85主训练和20741×20辅助，只在严格证明C来源instrument/完整日期顺序/原8字段与digest后追加独立S/Z监督组到87列。原39/16输入和全部原目标不变；来源artifact名与实际fixed_day_night_label列组明确分开，原闲置feature_masks字典保留但extra_features仍为空。

模型保持原973参数子集，在同一次primary共享8状态后加9参数零head；B监督S、C监督Z。原combined loss之后才加固定0.02×half-MSE，原aux采样、一次global clip2及AdamW保留；新head参与全局范数，不能声称private更新受保护。推理不读监督目标，训练图不封存。

唯一集中synthetic最终句柄55347 exit0，1 passed / 2.13s。保留首19539仅fixture频率metadata失败、52338前版通过，以及原闲置mask兼容修正后的55347最终通过履历。实际19列来源schema、PG85→87、原index/参数/RNG、尺度、推理隔离及globalclip耦合都纳入同一合同；没有调用真实model.fit或optimizer.step。

当前真实source对象/标签/模型读取=0、真实梯度=0、无拟合预检=0、fit/account=0。六模型SHA封存后才允许原946行评价Dataset与6成本独立路径/native、15参考及15CI；0新市场来源查询不等于否认已授权账户阶段对固定raw价及派生Qlib execution provider的读取。

无市场分窗或经济结果，不晋升。历史PIT/独立前向资格仍false。下一步仅为完整实现独评与单独train-only无拟合GO；预检通过后正式训练/账户仍需最终评审和owner GO。代码测试通过不证明S/Z预测性或SOXX稳定超额。
