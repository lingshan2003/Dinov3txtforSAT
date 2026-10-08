# 原生 Web DINOv3.txt 一图片 epoch 结果审阅

日期：2026-10-06。来源：用户提供的`web_native_seed11_reports.tar.gz`。
主结论：adapter+全层文本LoRA明显提高SkyScript同源验证检索；在当前相同full-maskpos协议下，终点绝对指标高于SAT组合。
RSICD相对本次原生基线小幅下降，不能宣称跨遥感数据源全面改善。当前仍为seed11开发验证。

## 1. 训练与报告完整性

24份文件：数据audit、训练config/metrics/optimizer/provenance/log/summary/validation、9份微调检索JSON、3份官方JSON及日志/汇总。
本包未含模型权重、图像、逐查询rank或完整manifests；以下为配置/日志/报告记录的核对，没有独立重跑GPU或重新计算SHA256。
机器可读审阅汇总保存于`outputs/web_native_seed11_review.json`（忽略目录）。

- 完成5,269optimizer step、21,075microbatch、337,199次图片曝光，完整1图片epoch；不同图与36,495caption组覆盖100%。
- BF16、batch16/accum4、mask_same_caption、queue0、noaug、seed11；末批及不足累计窗口保留。
- 未恢复运行，无记录到跳步、NaN/Inf；527条训练日志步数递增，28次验证包含0、200…5200、5269。
- 验证4055样本、loss batch16、forward batch64；best=latest=5269，best及latest三池指标完全相同。
- 可训练5,013,760参数：adapter1,054,976；全24个文本block LoRA3,932,160；projection LoRA26,624。rank8/alpha16/dropout0，共97个插入点。
- SAT/Web视觉backbone、原文本权重与vision head均按配方冻结，记录中optimizer仅含adapter/文本LoRA三个组。未提供权重张量，无法独立逐参数验证冻结状态。
- 三份官方报告无checkpoint，trainable=0；官方与项目step0在三池的全部检索指标完全一致。9份checkpoint身份检查均为match，无blocking/advisory mismatch。
- 共同summary中的训练及检索字段与单独JSON一致；MR算术、候选数、正例定义、tie policy核验通过。
- RTX3090，峰值allocated约5.68GiB；训练日志区间耗时求和约78.99分钟，不包含独立官方/训练后全局检索的完整总耗时。

## 2. Loss下降是否真实转化为检索

| loss口径 | 原生/起点 | 终点 |
| --- | ---: | ---: |
| 固定unique-val，16候选对比loss | 1.600889 | 0.590825 |
| 第1/最后1个optimizer更新的训练loss（summary） | 1.770747 | 0.400583 |
| 首/末日志窗口训练loss（metrics） | 1.499184（前10更新） | 0.380279（末9更新） |

summary训练loss是单次optimizer更新内microbatch均值，metrics是日志窗口多个更新的均值；两者数值不同是明确的统计口径差异，不是记录错误。
验证总体下降，但并非严格单调，600→800、3200→3400、4800→5000有小幅回升，最低值仍在终点。
gradient norm为clip之前的范数，最后7.7038不意味着max_grad_norm=1失效；各优化器组有梯度记录，未出现非有限值。

| 全局mean Recall | 原生官方 | Web微调终点 | 适配增量 |
| --- | ---: | ---: | ---: |
| SkyScript unique-val | 8.6313% | **17.6860%** | **+9.0547pp** |
| SkyScript多图val，按图 | 6.6031% | **17.0671%** | **+10.4640pp** |
| SkyScript多图val，按caption组均衡 | 7.3741% | **15.8317%** | **+8.4577pp** |
| RSICD-val | 14.4759% | **13.9305%** | **−0.5454pp** |

unique MR约为原生2.049倍；多图按组结果也大幅提高，说明提升不只体现于图片较多的caption组。
但单seed下没有显著性估计，不能把开发验证的全部收益外推到独立测试或地理区域泛化。

## 3. 每个方向发生了什么

下表均为百分数；MR是两方向R@1/5/10六项均值。多图任务每query只需top-K出现任一个标注正例，不是找到所有正例。

| 数据/方向 | 原生R@1/5/10 | 微调R@1/5/10 | 原生→微调 median rank |
| --- | --- | --- | --- |
| unique图→文 | 2.466 / 7.891 / 12.750 | **5.401 / 18.594 / 29.199** | 180→30 |
| unique文→图 | 3.107 / 10.160 / 15.413 | **5.869 / 18.594 / 28.459** | 119→32 |
| 多图图→文（按图） | 1.591 / 6.015 / 10.336 | **5.659 / 21.004 / 33.891** | 193→21 |
| 多图文→图 | 2.959 / 7.867 / 10.851 | **5.894 / 14.797 / 21.159** | 473→92 |
| RSICD图→文 | 4.662 / 10.603 / 17.276 | **3.748 / 12.797 / 20.750** | 56→43 |
| RSICD文→图 | 5.430 / 18.574 / 30.311 | **3.729 / 14.790 / 27.770** | 23→24 |

同源两池全部六项Recall提高，并有排名分布中位数明显改善，loss下降确实对应全局排序收益。
RSICD图→文R@5/10提高，但R@1下降0.9141pp；文→图三项分别下降1.7002/3.7843/2.5411pp。
因此RSICD MR小降掩盖了方向性权衡，不能简单称所有指标基本持平，也不能描述为所有指标都下降。
原unique双向R@1仍只有5.40%/5.87%，在4055候选上绝对精确匹配能力仍有改进空间。

候选池：unique4055图/4055文；多图33118图/4055文；RSICD1094图/5470文。
多图按组均衡先在caption组内平均图→文命中率，再对4055组等权平均；文→图已有每组一个查询。

## 4. 与SAT full-maskpos的一轮比较

直接读取此前用户SAT报告中的配置/报告后，核对到两组仅四字段不同：experiment.name/output_dir、model.backbone_domain/backbone_weights。
训练/验证manifest、BPE、dino.txt权重的已有来源记录一致；三个检索manifest、候选数、正例定义及tie policy一致。
所有527个日志窗口的caption-group采样诊断完全相同；两组均同预算、全图片覆盖、同优化器模块/学习率日程。
项目commit分别为SAT5187abb4、Weba446e975；两commit间src/dinotxt_rs仅修改两个评测CLI，新增official加载选项，训练器/采样器/目标/检索算法未改变。

| 指标 | SAT终点 | Web终点 | Web−SAT |
| --- | ---: | ---: | ---: |
| unique-val loss | 0.847802 | **0.590825** | 相对低30.31% |
| unique MR | 11.1467% | **17.6860%** | +6.5393pp |
| 多图按图MR | 11.3392% | **17.0671%** | +5.7279pp |
| 多图按组MR | 9.8190% | **15.8317%** | +6.0127pp |
| RSICD MR | 7.0658% | **13.9305%** | +6.8647pp |

此处以两组完整覆盖latest比较；SAT best5200与latest5269有小差异，Web best就是latest。
比较支持“原生配套组合在当前成本与配方下有更高绝对检索能力”；domain还控制对应预处理，且SAT接入通用head有兼容性混杂。
不能据此写成SAT骨干纯视觉特征较差，也不能用SAT从近零起点涨幅更大宣称它更优。
两组训练成本和可训练参数近似相同，本轮收益差异主要体现在适配结果，不是Web训练时间大幅更少。

## 5. 当前可以写进论文与下一步

可支持：冻结原生Web DINOv3视觉backbone，通过约501万adapter+文本LoRA参数，完整一图片epoch后获得清晰的同源held-out检索提升；存在RSICD迁移方向性权衡。
仍不能支持：LoRA优于adapter-only、maskpos优于其他目标、全部参数范围的最优组合、跨seed稳定、独立最终test改善、通用自然图像能力保持。

下一步优先补同数据/预算的Web adapter-only，判断LoRA是否带来额外收益、是否加重RSICD退化。
2026-10-08更新：用户已提供adapter-only完成报告，联合LoRA在同源和RSICD MR均高于adapter-only，见[对照审阅](WEB_ADAPTER_CONTROL_SEED11_ANALYSIS_2026-10-08.md)。本节其余结论为仅有主组时的边界与计划。
随后对主候选补seed23/47，报告均值、离散度及各方向R@1；不优先扩大3epoch/head全量/强文本参数矩阵。
head+LoRA无adapter可作为原架构更新的后续对照；更丰富caption是独立轴，待当前候选/消融关系明确后再推进。
若RSICD保持是核心目标，再单独比较较低文本LoRA LR、只更新部分层/不更新projection或保持约束；本轮不能把迁移代价定位到某个模块。
现有首轮loss虽低，16候选对比目标与4055/33118候选全局排序仍不同；不以loss接近0.4/0.6宣称模型上限。

历史旧Web500step资料引用RSICD原生MR15.8775%，本次同池重测为14.4759%，因此本轮退化必须以14.4759%作基线。
旧M4阶段的并列排序在2026-10-03修复：旧rank仅计分数严格大于最优正例者，当前按score降序/候选index升序处理ties。
这构成历史指标不可直接混用的已知协议差异；没有旧M4原生逐查询结果，不能精确把1.4016pp差异全部归因于该修复。
后续论文最终表格应按同一当前协议重评需引用的模型，不能用旧15.8775%对新13.9305%估计本次退化。

协议与启动器见[原生适配计划](WEB_NATIVE_ADAPTATION_PLAN_2026-10-06.md)，SAT对照来源见[全图片一轮审阅](SAT_FULL_IMAGE_EPOCH_ANALYSIS_2026-10-05.md)。
