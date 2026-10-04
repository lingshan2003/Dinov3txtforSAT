# SAT 四组联合实验：完整检索审阅（2026-10-04）

**Adapter + 全文本LoRA（含projection）在本轮九个配置中，两项数据集的mean Recall均最高。** 相对adapter-only，SkyScript从9.684%升至10.884%（+1.200个百分点），RSICD从6.679%升至7.044%（+0.366个百分点）。此前验证loss下降9.37%确实转化为整体检索改善，支持将其作为当前主候选；仍是单seed开发结果。

收益有方向差异：RSICD图→文R@1由2.102%降至1.737%，尽管其余五个Recall子指标均提升。不能表述为所有方向全面领先。Head + LoRA是当前mean Recall最好的无adapter方案，也有同样的RSICD图→文R@1退化。Adapter + projection没有额外mean Recall收益。

## 1. 报告核验与证据范围

新来源：`/Users/wangyue/Downloads/sat_joint_3epoch_retrieval_reports.tar.gz`，16份JSON，四组 × SkyScript/RSICD × step0/best。单侧参照来自此前`sat_all_3epoch_retrieval_reports.tar.gz`的20份报告；联合训练范围及选模见[联合训练审阅](SAT_JOINT_3EPOCH_ANALYSIS_2026-10-04.md)。

- 16份预期报告完整，全部split=val。Step0均记录step0，best均为1710，与四组训练summary一致。
- 同组跨数据集引用的checkpoint路径、已存身份和run_identity一致；评测身份检查全部match，无blocking/advisory mismatch。
- 新报告使用同一训练/评测项目提交`316b6c0c381ccf3e0fc63b1c2dce82a99789c2a0`及相同上游版本。资产身份、候选池、排序规则与旧报告一致；项目提交差异保留在历史报告，未发现原有训练路径在本次提交改变。
- SkyScript为4055图像/4055文本，一一对应正例；RSICD为1094图像/5470描述，代码按图像ID构造多描述正例，图→文取最佳正例rank，文→图匹配对应图像。
- 两项数据集各自九个配置的step0全部指标逐项相同。共同step0 mean Recall为SkyScript 0.115084%、RSICD 0.469226%。
- 所有mean Recall经重新计算，等于图→文与文→图R@1/5/10六项的算术平均。Recall合法且有限；统一tie policy为相似度降序、相等时候选索引升序。
- RTX3090、Torch2.7.1+cu128、CUDA12.8、encoding batch64、retrieval chunk256及冻结logit_scale100与旧评测一致。

本次解析报告并比较已存来源身份，没有重算SHA256、下载权重或重新运行GPU。报告核验支持评测一致性，但没有直接复算实际embedding、逐查询rank或冻结参数差分；无法做配对bootstrap显著性检验。聚合命中数量变化也不能说明具体哪些查询新增命中或退化。

SkyScript-val已参与loss选模，RSICD-val已用于开发观察。这些不是未使用的最终test结果。两个数据集候选池和正例定义不同，只在各数据集内部比较配置，不按其绝对分数高低推断域难度。

## 2. 九组总体比较

以下mean Recall均为百分比，按SkyScript得分排列；所有best均为step1710。

| 配置 | SkyScript mean R (%) | RSICD mean R (%) |
| --- | ---: | ---: |
| Adapter + 全文本LoRA | 10.884 | 7.044 |
| Adapter-only | 9.684 | 6.679 |
| Adapter + projection | 9.679 | 6.639 |
| 视觉head + 全文本LoRA | 6.663 | 5.098 |
| 全文本LoRA-only | 4.624 | 3.876 |
| 视觉head + projection | 3.440 | 3.918 |
| 视觉head-only | 3.042 | 3.659 |
| 文本末两层 + ln_final | 0.925 | 0.948 |
| Projection-only | 0.744 | 0.966 |

### 联合配置相对视觉单侧的增量

| 联合配置 | SkyScript增量 (pp) | RSICD增量 (pp) |
| --- | ---: | ---: |
| Adapter + 全文本LoRA | +1.200 | +0.366 |
| 视觉head + 全文本LoRA | +3.621 | +1.438 |
| Adapter + projection | -0.004 | -0.040 |
| 视觉head + projection | +0.399 | +0.259 |

Adapter + LoRA相对adapter-only的mean Recall相对提升为SkyScript 12.39%、RSICD 5.47%；这与+1.200/+0.366个百分点是不同计量，不能混用。RSICD的整体收益较小，且非所有子指标一致改善，需要多seed和逐查询分析确认稳健性。

## 3. 双向Recall细节

仅列四组联合与主要单侧参照；全部数值为百分比。

### SkyScript-val

| 配置 | 图→文R@1 | R@5 | R@10 | 文→图R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adapter + 全文本LoRA | 2.762 | 11.122 | 17.115 | 3.502 | 11.985 | 18.816 |
| Adapter-only | 1.899 | 9.371 | 15.882 | 2.935 | 10.777 | 17.238 |
| Adapter + projection | 1.924 | 9.273 | 15.758 | 2.984 | 10.875 | 17.263 |
| 视觉head + 全文本LoRA | 1.874 | 6.979 | 12.010 | 1.578 | 6.190 | 11.344 |
| 全文本LoRA-only | 1.258 | 5.080 | 8.952 | 0.764 | 4.094 | 7.596 |
| 视觉head + projection | 0.691 | 3.132 | 5.771 | 0.740 | 3.600 | 6.708 |
| 视觉head-only | 0.543 | 2.688 | 5.031 | 0.838 | 3.206 | 5.943 |

### RSICD-val

| 配置 | 图→文R@1 | R@5 | R@10 | 文→图R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adapter + 全文本LoRA | 1.737 | 5.667 | 8.867 | 1.664 | 8.720 | 15.612 |
| Adapter-only | 2.102 | 5.119 | 8.684 | 1.188 | 7.623 | 15.356 |
| Adapter + projection | 1.920 | 5.119 | 8.592 | 1.188 | 7.751 | 15.265 |
| 视觉head + 全文本LoRA | 0.640 | 3.839 | 7.495 | 1.225 | 5.795 | 11.590 |
| 全文本LoRA-only | 0.731 | 2.834 | 4.845 | 0.859 | 5.137 | 8.848 |
| 视觉head + projection | 1.005 | 3.839 | 5.119 | 1.005 | 4.589 | 7.952 |
| 视觉head-only | 1.097 | 3.473 | 5.210 | 1.005 | 3.821 | 7.349 |

### Adapter + LoRA：提升真实，但Top1与整体排序仍有差异

SkyScript六个Recall均高于adapter-only。图→文R@1命中由77/4055增至112/4055（+35），文→图由119/4055增至142/4055（+23）。图→文/文→图正例中位rank由74/77改善至62/63，平均rank由192.96/193.39改善至165.03/168.51，支持整体排序也有改善。

RSICD五个Recall提高，图→文R@1却由23/1094降至19/1094（净少4个命中）；文→图R@1由65/5470增至91/5470（净多26个命中）。图→文/文→图中位rank由116/56改善至108/48；图→文平均rank却从281.61略升至284.56，文→图平均rank从127.97降至118.38。该分布并非所有查询都一致向前移动。

跨数据集的mean Recall没有整体退化，但这些Top1及尾部rank结果不支持“通用能力全面保持”的说法。若目标要求图→文第一名准确性，应单独考虑该方向的权衡。

最强配置在SkyScript的图→文/文→图R@1仍仅2.76%/3.50%，R@10为17.11%/18.82%。相对近随机step0有大幅改善，绝对匹配能力仍弱，不能称对齐问题已解决。

### Head + LoRA：有效的无adapter联合路线

SkyScript mean Recall为6.663%，超过head-only的3.042%和LoRA-only的4.624%，六个Recall均超过这两个参照；RSICD为5.098%，超过3.659%和3.876%。但RSICD图→文R@1为0.640%（7/1094），低于head-only的1.097%（12/1094）和LoRA-only的0.731%（8/1094），其余五项均更高。

这支持两侧共同适配对当前配置的整体检索目标有互补收益，也确认原架构上不添加视觉adapter可以学习有效对齐。它仍落后于adapter-only的两项mean Recall，更未达到adapter + LoRA的整体水平；“最高mean Recall的无adapter配置”不等于每个指标都最高。

### Projection的联合价值依赖视觉更新路径

Adapter + projection的SkyScript/RSICD mean Recall分别比adapter-only低0.004/0.040pp，属于本轮几乎持平、子指标有涨有跌的结果，不能称稳定退化。此前loss虽略低0.06%，没有转化为额外mean Recall收益。此例也说明细小loss差异不足以替代检索评测。

Head + projection比head-only提高0.399/0.259pp，但并非所有Recall均提升；其RSICD mean Recall 3.918%还略高于LoRA-only的3.876%，而SkyScript明显低于后者。这不意味着projection普遍优于LoRA，二者训练范围、学习率和视觉侧条件不同。

## 4. 当前研究判断与后续优先级

**候选定位：adapter + 全文本LoRA作为当前主候选，adapter-only保留为稳定参照，head + 全文本LoRA保留为无adapter的重要对照。** 不依据这次单seed结果直接宣布最终胜者。Projection联合组已经提供必要信息，无需因理论上可组合就继续穷举全部15种。

**对adapter优势的解释仍是机制假设。** 两侧都适配后adapter路线仍领先，说明其最终表示残差修正在当前协议中有价值；不单独证明瓶颈、GELU、零初始化或更新位置中的哪一个因素造成优势。Head与adapter学习率不同，LoRA也同时覆盖24层与projection，当前比较不是各方案最优超参数的公平上限比较。

后续优先考虑：

1. 对上述主候选及两个重要参照增加独立seed，在相同预算/选模规则下检查mean Recall和各方向R@1；尤其确认RSICD图→文R@1的取舍是否复现。保存逐查询rank可做配对重采样，并结合多seed波动判断稳健性。
2. 导出主候选与adapter-only的检索Top10/正例rank作定性诊断，检查视觉相似但语义不等价、近重复文本及可能的正例定义局限。当前聚合报告不证明这些因素就是原因。
3. 若继续改训练目标，将物理batch与梯度累积区分。当前每次对比只有16个候选，累积4并不提供64候选。可另做物理batch32×累积2、保持optimizer有效batch64的对照，在设备允许时单独测试候选池因素；此为后续提议，本次没有改协议或启动新训练。
4. 候选和协议固定后再进行留出的最终test及同预算Web参照，保留开发验证与最终报告的区别。

本轮最明确的结论是：充分训练后，全文本LoRA与视觉适配联合有额外整体检索收益；adapter + LoRA目前mean Recall最高；小projection联合更新的收益依赖视觉侧，且loss的微小改进不保证检索改善。
