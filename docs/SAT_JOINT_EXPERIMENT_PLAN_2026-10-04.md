# SAT 视觉与文本联合适配：组合空间与首批实验

目标仍是提高SAT图像与文本的匹配，视觉backbone永久冻结。本次联合适配配置从同一官方权重重新初始化；不直接resume已有adapter或LoRA的best来改变训练范围。

最新状态（2026-10-04）：首批四组联合训练报告已审阅，均完整完成1710step，best均为终点。
Adapter+全文本LoRA验证loss最低（0.902339），后续完整检索也已审阅：两数据集mean Recall为10.884%/7.044%，当前九组最高，但RSICD图→文R@1低于adapter-only。
最新结论见[联合完整检索审阅](SAT_JOINT_3EPOCH_RETRIEVAL_ANALYSIS_2026-10-04.md)；训练与历史tmux检索命令见[联合三轮训练审阅](SAT_JOINT_3EPOCH_ANALYSIS_2026-10-04.md)。下文保留实验设计、启动与打包命令作为复现记录。

## 组合空间

计数先固定文本全量更新为末两层+ln_final、LoRA为rank8覆盖全部24层；不把学习率、rank、层数、batch和训练日程算作新的模块组合。

视觉侧有三种当前可用的非空组合：A=输出adapter、H=原有两层视觉head全量更新、A+H。
文本侧有五种当前可用的非空组合：P=projection全量更新；C=末两层+ln_final；C+P；L=全24层LoRA且projection冻结；L+Pᵣ=全24层与projection均用LoRA。
因此目前已有实现支持3×5=15种两侧联合训练方案：

| 视觉侧 | P | C | C+P | L（projection冻结） | L+Pᵣ |
| --- | --- | --- | --- | --- | --- |
| A | A+P：首批 | A+C | A+C+P | A+L | A+L+Pᵣ：首批 |
| H | H+P：首批 | H+C | H+C+P | H+L | H+L+Pᵣ：首批 |
| A+H | A+H+P | A+H+C | A+H+C+P | A+H+L | A+H+L+Pᵣ |

冻结整侧是单侧对照，不算这15种联合组合。也不能把“全层LoRA”与“末两层全量更新”直接并列打开，当作现有代码已经支持的组合：当前配置显式拒绝混合全量文本更新与LoRA。

扩展视觉head的LoRA后，视觉侧还可加入Hᵣ、A+Hᵣ，共五种视觉选择，与上述文本五列构成25种。如果再支持“文本block LoRA + projection全量更新”，文本可增至六列，对应30种。但这两个扩展尚未实现；attention-only、projection-only LoRA、不同text_last_k、rank和学习率又会进一步扩大空间。不存在脱离粒度定义的唯一总数。

## 为什么输出adapter可能如此有效

当前代码在最终归一化2048维图像向量v后计算：

\[
v'=\operatorname{normalize}\left[v+W_u\operatorname{GELU}(W_d\operatorname{LN}(v)+b_d)+b_u\right].
\]

其中W_d把2048维压到256维，W_u再映射回2048维。这个模块约105万参数，有非线性，并非只有一个缩放系数。结构见 `src/dinotxt_rs/models/embedding_adapter.py`。

以下是符合结构和结果的机制假设，尚不是已经完成消融验证的因果结论：

1. **更新位置与目标直接对应。** 对比目标就在最终图文向量上计算，adapter可以直接改变向量方向。视觉head需要先改变token级表示、再经池化进入最终向量，对同一目标的更新路径不同。
2. **保留已有视觉特征。** 不必重新学习遥感视觉结构，只需利用已有高维表示修正匹配空间。低step0检索说明SAT backbone与通用head/text组合不兼容，不说明SAT视觉表征没有信息。
3. **零残差起点。** up.weight和up.bias置零，初始化输出保持原映射；学习从逐渐添加修正开始。该结构不同于从随机head重训，也不等于此前head从随机初始化开始——head同样加载了预训练权重。
4. **更新受到瓶颈约束。** 归一化前的残差由256维系数经同一个up矩阵生成，限制了修正的自由度，同时保留输入相关的非线性。这可能让当前数据上的优化更容易、更新更稳定。

参数少并不意味着没有足够表达能力；参数更多也不自动意味着在指定学习率和预算下更容易优化。我们目前没有看到head validation loss反弹，因此不能直接解释为head过拟合；head与adapter的学习率也不同。已有检索只能证明本轮adapter配置更有效，不能证明视觉head上限更低。

参数高效适配的通用动机见[Houlsby等的adapter论文](https://arxiv.org/abs/1902.00751)和[LoRA论文](https://arxiv.org/abs/2106.09685)。这些论文不直接证明本项目输出adapter为何领先。

## 首批四组：两个视觉选择 × 两个文本选择

| 配置名中间部分 | 实际更新范围 | 各组LR | 可训练参数 |
| --- | --- | --- | ---: |
| adapter_textlora | A + 全层L + projection LoRA | A/L/Pᵣ均1e-4 | 5,013,760 |
| visionhead_textlora | H + 全层L + projection LoRA | H=1e-5，L/Pᵣ=1e-4 | 29,285,120 |
| adapter_textproj | A + P | A=1e-4，P=1e-5 | 3,676,416 |
| visionhead_textproj | H + P | H/P均1e-5 | 27,947,776 |

每组都有 `configs/skyscript_sat_<中间部分>_3epoch_seed11.toml` 和同名outputs目录。
已有A-only、H-only、P-only、L+Pᵣ-only三轮结果构成单侧参照，因而四组均可与现有单侧结果比较。
这回答“是否互补”及“最终projection的轻量联合调整是否已经足够”，不把全部15种都当作必须先跑的实验。

共同设置保持1710step、warmup171、物理batch16×累积4、BF16、queue0、验证分组16、无增强。
Head更新组显式关闭drop_path；logit_scale冻结。每200step及step0/终点验证，保留step0/best/latest。
暂不同时扩大候选池，也不改变学习率搜索范围，以便理解组合的影响。学习率沿用单侧首轮设置，不宣称各组合已经最优。

从官方权重重新训练是联合适配主实验。先训adapter、再加载其best训练文本属于分阶段适配，是另一个变量；当前严格resume机制不能用改过的配置冒充同一次续训。
新增adapter会改变后续LoRA随机初始化的RNG消费；同seed保证每个配置可复现，不保证与LoRA-only的内部随机A矩阵逐项相同。零残差使step0函数起点一致，最终稳健结论仍需多seed。

## 运行与审阅

同步新增配置和脚本到服务器后，在空闲GPU运行：

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
tmux new -s sat-joint-3epoch
bash scripts/run_sat_joint_3epoch_seed11.sh
```

顺序为adapter+LoRA、head+LoRA、adapter+projection、head+projection。脚本已完成则跳过，中断则从latest恢复，失败则停止；不自动运行检索。Ctrl-b再d脱离，回来用 `tmux attach -t sat-joint-3epoch`。

完成后打包训练报告：

```bash
tar --exclude='*.pt' --exclude='*.part' -czf sat_joint_3epoch_reports.tar.gz \
  outputs/skyscript_sat_adapter_textlora_3epoch_seed11 \
  outputs/skyscript_sat_visionhead_textlora_3epoch_seed11 \
  outputs/skyscript_sat_adapter_textproj_3epoch_seed11 \
  outputs/skyscript_sat_visionhead_textproj_3epoch_seed11
```

随后评测各组step0/best的SkyScript-val与RSICD-val，分别与相应单侧模型比较。联合训练不保证收益相加；两侧共同移动也可能使优化更困难。判断依据是完整双向检索、训练曲线与成本，不能只看loss或同某一个弱基线比较。

若要解释adapter优势，可另做去掉GELU的低秩线性残差对照、不同bottleneck容量对照、视觉head学习率对照，并检查残差幅度及检索正例rank。这些机制诊断独立于当前四组联合实验，不是启动它们的前置条件。
