# Web 三seed配对复现结果审阅

审阅日期：2026-10-08。来源：用户提供的`/Users/wangyue/Downloads/web_paired_seeds_reports.tar.gz`。
读取111个文件，包括六组训练元数据/日志、57份检索报告和三seed汇总；未重跑GPU或重新计算SHA256。
源报告的已有来源记录用于一致性核验；未包含checkpoint张量、图片、完整manifest或逐查询rank。

第一阶段结论：在当前固定SkyScript划分及一图片epoch预算下，视觉adapter与全文本LoRA联合适配，相比adapter-only，在三个配对训练seed上均提高同域全局检索，并减小RSICD mean Recall损失。
联合方案可作为有跨seed重复证据的阶段主候选；两种适配方案的RSICD MR均仍低于原生。
这里的稳定是已观察到的训练随机性重复一致，不是独立test、跨数据集全面提升或统计显著性结论。

## 1. 配对设计与完成性

训练seed为11/23/47，数据split不变，每seed均有adapter-only与adapter＋LoRA。
六组均使用337199训练图、36495caption组、完整1图片epoch、5269optimizer step及21075microbatch。
六组completed=true、无resume或跳过更新，loss/gradient均有限；各有28次验证，步骤恰为0、200至5200及5269。
同seed两方法527条日志的采样诊断、图片覆盖/曝光及epoch进度逐条一致，三seed共1581组窗口核对无差异。
共同配置为batch16/accum4、mask_same_caption、queue0、无增强、527warmup/cosine、BF16；每200step及step0/终点验证。
梯度累计不扩大单个对比损失的物理候选batch。
视觉骨干与原生head冻结，控制组仅更新视觉embedding adapter的1054976参数；联合组增加24个文本block及projection LoRA，共5013760参数。

已有seed11两组训练/检索/官方报告与上一控制包的35个对应文件逐字节一致。
seed11主组训练commit为a446e975，控制组5c74e5ca，新增四组e0e09536；a446e975至e0e09536的`src/dinotxt_rs`没有差异。
因此主组结果复用及调度入口扩展没有引入核心训练/评测实现差异。
比较表的全部latest/best单seed值、均值、样本标准差、配对差值及相对原生增量均已从单独报告重算一致。
54份checkpoint报告身份全部match，无blocking/advisory mismatch，run_identity与各自provenance记录一致。
57份报告的对应候选池、manifest、模型资产及评测协议一致；18份step0的指标和计数与官方精确相同。
六份归档配置与本地对应配置逐字节相同。核验仅比较已有来源记录，没有重算SHA256或读取实际权重张量。
三seed共享同一验证池，unique与多图设置也共享held-out caption组，不应当作独立测试样本池计数。

## 2. 主要结果

主比较为完整覆盖latest。数值为mean Recall百分比，±为三个训练seed的样本标准差（ddof=1），不是置信区间。
MR为双向R@1/5/10共六项平均。多图组均衡指标对caption组等权，区别于每张图等权的统计。

| 验证指标 | 原生 | adapter-only | adapter＋LoRA | 配对LoRA增益 |
| --- | ---: | ---: | ---: | ---: |
| SkyScript unique | 8.6313% | 14.6486 ± 0.1305% | 17.7764 ± 0.1711% | +3.1278 ± 0.1368pp |
| SkyScript多图按图 | 6.6031% | 13.5031 ± 0.0199% | 16.9132 ± 0.1641% | +3.4101 ± 0.1509pp |
| SkyScript多图按caption组均衡 | 7.3741% | 12.8729 ± 0.0665% | 15.7638 ± 0.1666% | +2.8909 ± 0.2147pp |
| RSICD-val | 14.4759% | 12.2171 ± 0.0895% | 14.1367 ± 0.2459% | +1.9196 ± 0.3300pp |

配对增益在同seed内先计算联合减adapter-only，再汇总三个差值；不是两组标准差的简单相减或独立方差合成。

| seed | unique增益 | 多图按图增益 | 多图组均衡增益 | RSICD增益 |
| --- | ---: | ---: | ---: | ---: |
| 11 | +3.1854pp | +3.5414pp | +2.9201pp | +1.6728pp |
| 23 | +2.9716pp | +3.2453pp | +2.6631pp | +2.2943pp |
| 47 | +3.2265pp | +3.4435pp | +3.0896pp | +1.7916pp |

四项汇总指标在3/3配对seed内方向一致。
unique增益范围2.97–3.23pp，多图按图3.25–3.54pp，组均衡2.66–3.09pp；收益也出现在caption组等权设置，不能仅归因于大组权重。
同域两个检索设置的六项Recall在每个seed内全部提高，共36/36项方向一致。
联合unique MR相对原生平均+9.1451pp，adapter-only也平均+6.0173pp，表明adapter本身有效、联合文本更新仍有可重复的额外价值。

## 3. 外部迁移与R@1边界

RSICD相对adapter-only的MR收益为1.67–2.29pp，但相对原生仍有损失：

| 方案 | 平均MR | 相对原生平均变化 | 各seed相对原生变化 |
| --- | ---: | ---: | --- |
| adapter-only | 12.2171% | −2.2588pp | −2.2182 / −2.3614 / −2.1968pp |
| adapter＋LoRA | 14.1367% | −0.3392pp | −0.5454 / −0.0670 / −0.4052pp |

联合方案在三个seed均减小迁移损失；本轮不支持“额外文本更新必然加重RSICD退化”或“冻结文本就能保留最终检索能力”的推断。
这些是当前配方的行为证据，不能定位到某个模块或embedding几何机制。

| RSICD方向 | 原生R@1/5/10 | adapter-only三seed均值 | adapter＋LoRA三seed均值 |
| --- | --- | --- | --- |
| 图→文 | 4.662 / 10.603 / 17.276 | 3.931 / 10.390 / 16.484 | 4.083 / 12.797 / 20.293 |
| 文→图 | 5.430 / 18.574 / 30.311 | 3.790 / 13.949 / 24.759 | 3.961 / 15.765 / 27.922 |

联合相对adapter-only的六项Recall在三个seed中16项提高、2项相同（seed11/23图→文R@1），没有下降。
相对原生则每个seed都呈现同一方向：图→文R@5/10提高，图→文R@1及文→图三项下降。
不能用MR只差0.3392pp称所有检索能力都保持不变，更不能写成全面跨域泛化改善。
同源unique联合R@1均值为图→文5.1870%、文→图5.7131%，仍存在明显绝对性能改进空间。

## 4. Checkpoint选择与资源

| seed | 方法 | best step | 终点val loss | 日志训练区间累计 | 峰值allocated |
| --- | --- | ---: | ---: | ---: | ---: |
| 11 | adapter＋LoRA | 5269 | 0.590825 | 78.99分钟 | 5.68GiB |
| 11 | adapter-only | 5269 | 0.688097 | 57.12分钟 | 3.91GiB |
| 23 | adapter＋LoRA | 5000 | 0.578410 | 78.17分钟 | 5.68GiB |
| 23 | adapter-only | 5000 | 0.676711 | 57.18分钟 | 3.91GiB |
| 47 | adapter＋LoRA | 5200 | 0.586234 | 78.48分钟 | 5.68GiB |
| 47 | adapter-only | 5269 | 0.685148 | 57.32分钟 | 3.91GiB |

latest均为5269且完整覆盖；best依固定unique-val loss选取，不根据RSICD或检索最高分选择。
改用best时，配对unique/多图按图/按组/RSICD平均收益仍为+3.1292/+3.4037/+2.8694/+1.9317pp，所有seed方向保持。
结论不依赖选择latest或best。best与latest并非六组都相同，不能沿用只有seed11时的表述。
两组终点val loss三seed分别为0.683319±0.005909和0.585157±0.006277，与全局检索方向一致。
日志时间均值为adapter-only57.21分钟、联合78.54分钟；峰值allocated约增加1.77GiB。
时间是RTX3090日志区间求和，包含周期验证，不是含所有全局检索的端到端基准。
末更新训练loss受末尾batch影响明显，不以它代替全轮或窗口平均，也不据此判断模型上限。

## 5. 阶段报告可采用的结论

中文表述：

> 在冻结DINOv3视觉骨干的条件下，视觉adapter与文本encoder及projection的LoRA联合适配，在三个配对训练随机种子上稳定优于仅训练视觉adapter。固定SkyScript一图一文验证集的mean Recall由原生模型的8.63%提升至17.78±0.17%，相比adapter-only的14.65±0.13%获得3.13±0.14个百分点的配对增益。联合适配同时减小了RSICD上的检索退化，但尚未超过原生模型的外部迁移表现，表明领域适配收益与迁移保持仍需共同考察。

英文表述：

> Across three paired training seeds, joint adaptation of a visual embedding adapter and low-rank parameters in the text encoder and its projection consistently improved in-domain retrieval over adapter-only training while keeping the DINOv3 visual backbone frozen. On the fixed SkyScript validation set with one image per caption, mean Recall increased from 8.63% for the native model to 17.78 ± 0.17%, compared with 14.65 ± 0.13% for adapter-only training. The paired gain was 3.13 ± 0.14 percentage points, with dispersion reported as the sample standard deviation across seeds. Joint adaptation also reduced the loss in RSICD mean Recall relative to adapter-only training, although it remained below the native baseline.

在第一份阶段报告中，可将seed23/47配对复现由ongoing改为completed，将上表三seed结果作为主要实验依据。
仍需保留的边界：固定验证split、三训练seed、无独立最终test；不宣称统计显著性、SOTA、文本LoRA单独有效或模块贡献已拆分。
本轮仅检验在adapter存在时增加全文本LoRA的效应；联合组的adapter更新轨迹也改变，不能全部解释为文本encoder独立变强。
报告展望可聚焦Transformer/projection模块消融、adapter必要性、迁移保持和caption质量，尚未实施的内容明确写为future work。

启动协议见[三seed配对方案](WEB_PAIRED_SEEDS_PLAN_2026-10-08.md)，seed11单独结果见[此前对照审阅](WEB_ADAPTER_CONTROL_SEED11_ANALYSIS_2026-10-08.md)。
