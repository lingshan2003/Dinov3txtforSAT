# 毕业论文项目审阅：方法、数据与任务

审阅日期：2026-10-02。

总体判断：**冻结视觉 backbone 的方向合理，SkyScript 可以继续作为训练数据，双向图文检索适合作为主任务。但目前有效结果主要回答图像 embedding 适配，尚未充分回答用户提出的文本 encoder 微调问题。正式扩展训练前，优先修正实验归因与评测协议。**

本次读取科研计划、交接、架构文档、模型/训练/损失/评测实现、实验配置、本地 SkyScript 选择清单与审计、历史 M3 报告，并核对固定版本的 DINOv3 上游源码和 SkyScript 官方资料。未修改原训练代码或历史协议，未重跑训练，未重新计算 SHA-256。当前 M4/F1–F3 summary/checkpoint 未在本地发现；这些阶段的训练数值以下均按交接文档记录引用，不能称为本次独立复算结果。

## 1. 当前证据到底回答了什么

| 问题 | 当前判断 | 证据边界 |
| --- | --- | --- |
| 能否冻结视觉 backbone 做遥感图文适配？ | 合理，且冻结边界实现可靠 | 代码有参数、优化器、eval 模式与梯度检查 |
| 当前图像 adapter 能否学习？ | 文档记录支持短预算下稳定学习 | Web/SAT 各三 seed，500 step；不足一轮数据 |
| 文本 projection 是否增益？ | 两档低 LR、单 seed 下未持续超过 F0 | 是输出投影的有限筛查 |
| 文本 encoder 本体是否值得微调？ | 尚未得到干净对照 | SkyScript F3 的 `text_last_k=0`；旧 ChatEarthNet 联合改变多个变量 |
| SAT 的视觉表征是否弱于 Web？ | 不能由当前结果判定 | SAT 使用为 Web 路径训练的对齐头，含接口兼容性混杂 |
| 是否已经证明地理或跨数据源泛化？ | 尚未 | 同源验证、开发保持集与已观察历史外部评测的角色不同 |
| 是否支持定位或分割能力？ | 不支持 | 当前 adapter 仅改变全局 embedding，patch 输出不变 |

冻结大视觉模型、优化其后的对齐模块和文本侧有直接方法依据。DINOv3 的文本对齐方案采用冻结视觉 backbone、增加两层视觉 transformer head、训练文本表示的路线。但上游从头训练文本表示的结果，不等同于在当前小规模遥感子集上微调预训练文本 encoder 必然有效。[DINOv3 论文 §5.3.3](https://arxiv.org/html/2508.10104v1#S5.SS3.SSS3)

## 2. 方法设计中最应重新确认的事项

### 2.1 文本微调的研究目标被 adapter 成功结果替代了

当前 F0 的 1,054,976 个可训练参数全部属于 image adapter；文本 backbone 和 projection 均冻结。F3 只解冻最终 projection，并没有更新 transformer blocks。参见 [F3 配置](/Users/wangyue/Documents/ChatGPT/Dinov3txtforSAT/configs/skyscript_sat_f3_adapter256_textproj_lr1e5_500step_seed11.toml:14) 与 [冻结策略](/Users/wangyue/Documents/ChatGPT/Dinov3txtforSAT/src/dinotxt_rs/models/official_dinotxt.py:41)。

历史 ChatEarthNet 路线确实更新过最后四个文本 blocks，但同时改变 vision head、projection、logit scale、queue 与增强，不能把失败单独归因于文本微调。

**因此当前准确结论是：还缺少 SkyScript 固定协议下、隔离文本 encoder 收益的实验。** F0 很适合保留为强基线；若论文仍以文本微调为主题，不能把 F0 的成功直接当作主题已完成。

交接规定 projection 必须先成功，才允许最后一个 text block。这个规则可作为算力止损安排，却没有科学上的必然性：投影改变最终映射，encoder 改变词义和上下文表示，二者处理的问题不同。projection 无增益不能推出 encoder 无增益。[现有止损规则](/Users/wangyue/Documents/ChatGPT/Dinov3txtforSAT/docs/PREPARE_HANDOFF.md:278)

建议预注册一个小规模 encoder 候选，例如最后一个 block + final norm；是否包含 projection 要固定并说明。直接检验它相对冻结文本和 projection-only 的作用。先用现有实现支持的范围即可，不必立即增加 LoRA 工程或解冻整个文本塔。

### 2.2 SAT 初始化几乎没有图文对齐，首先存在视觉接口问题

固定上游实现默认使用 LVD1689M backbone，加载同一份 dino.txt vision-head/text-encoder 权重。本项目替换 SAT backbone 后仍沿用这些对齐权重。[固定版本加载代码](https://raw.githubusercontent.com/facebookresearch/dinov3/6876159a11b4df116f30f667f8c9888617df0751/dinov3/hub/dinotxt.py)

交接记录 SkyScript step0 mean recall：Web 0.0862721、SAT 0.0011097。4,055 个一对一候选的均匀随机排序期望 mean recall 为 `(1+5+10)/(3×4055)≈0.001315`，SAT 起点接近随机检索。此后 SAT 提升说明这条接口能够适配，较大的 delta 则不能单独证明 SAT 更优。

文档记录三 seed step500 Web/SAT mean recall 分别为 0.1400055/0.0638033。它支持“当前迁移头、adapter 与固定预算下 Web 路径更好”；不能识别视觉预训练域的净收益。Web 应称强参考基线，“参考上界”不是已证明的事实。

若要保留视觉域比较为主问题，建议增加一个接口诊断：直接从冻结 backbone 的 CLS + mean patch 特征接同构新线性/小 MLP 投影，两域采用相同初始化策略、文本空间和训练规则。比较它与经过通用 dino.txt head 的路径，可以检查差距是否主要由迁移头造成。SAT/Web 的预训练数据规模和训练历史仍不同，所以这种诊断也不是严格只改变数据域的预训练因果实验。

后置 adapter 只接收池化、归一化后的全局向量；若上游 head/池化丢失区分信息，它不能保证恢复。**这是值得检验的瓶颈假设，尚非已确定原因。** 不应据此无限增加模块。

### 2.3 F1/F2 同时开启随机深度，影响 head 微调的归因

本地训练器对 trainable vision head 调用 `train()`，F0/F3 的冻结 head 则处于 eval。[训练模式策略](/Users/wangyue/Documents/ChatGPT/Dinov3txtforSAT/src/dinotxt_rs/training/trainer.py:115)

固定上游 head 的两层 block 使用 `drop_path=0.3`。训练模式会对 attention 和 FFN 残差分支分别随机抽样；batch16 时每分支实际选择 11 个样本计算，再放大残差。[head 实现](https://raw.githubusercontent.com/facebookresearch/dinov3/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/text/vision_tower.py)、[随机分支实现](https://raw.githubusercontent.com/facebookresearch/dinov3/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/block.py)

所以已完成 F1/F2 检验的是“更新 head，同时采用上游随机深度”，不能仅归因于新增可训练参数。它们的实验结果仍然有效，但解释范围应收紧。

如果继续研究 head，优先补一次关闭 head 随机深度的诊断，并保留原结果。`eval()` 本身不会禁止梯度；可显式控制随机深度与可训练状态，分别记录。没有证据说关闭后一定改善。

### 2.4 短预算筛查不能全面排除慢更新方案

当前 physical batch16、累积4、500 step 共32,000次样本曝光，约0.877 epoch；对比候选仍只有16。adapter 约1.05M参数，vision head约25.33M，预训练模块 LR 又低10–20倍。固定 step 比较适合回答有限预算效率，不等同于每种方法已经充分优化。

统一3 epochs、1,710 optimizer steps 可作为有限资源下的正式日程；它也不是收敛保证。最终保留少数机制明确的候选，观察完整曲线，再用共同规则决定是否延长。不能只把500-step胜者延长，然后声称其他微调范围普遍无效。

## 3. 数据集选择：可以继续，但应先补数据证据

### 3.1 SkyScript 的适用性与真实监督来源

SkyScript 将 OSM 语义与多来源遥感影像关联，适合设施/地物短语的图文对齐。当前短句和去模板化有助于建立可控基线；换回长生成描述没有充分理由。其监督是弱监督，需要审查可见性。[SkyScript 论文](https://arxiv.org/abs/2312.12856)

当前使用的是 CLIP-laion-RS 过滤的 language-polished top30 子集。`title_raw` 是这份 polished CSV 的内容句，不能解释成未经 LLM 改写的原始 OSM 文本。官方明确说明 polished 版本由 ChatGPT 将 OSM tags 转成自然描述，top30 由 CLIP-laion-RS 相似度筛选。[官方数据说明](https://github.com/wangzhecheng/SkyScript#download-captions)

因此当前文本对照首先研究“内容句、模板和视角前缀”，不能直接声称研究原始标签与自然语言的差别。top30 筛选也使评价偏向筛选模型认为可对齐的样本；这是选择偏差风险，不等于标签泄漏。与 CLIP/SkyCLIP 比较时须披露筛选模型。

### 3.2 已确认的元数据错误与来源偏斜

[prepare_skyscript.py](/Users/wangyue/Documents/ChatGPT/Dinov3txtforSAT/tools/prepare_skyscript.py:26) 将文件名末尾字段解释成 country/zoom；实际是 image-source alias/拍摄年份。比如 `_CH_19` 的19代表2019年，CH代表SWISSIMAGE来源。S2、P3、L8等含数字alias还会被当前正则列为unknown。[官方文件名与 metadata 说明](https://github.com/wangzhecheng/SkyScript#download-image-files)

这不影响当前训练图文读取，却会误导来源与尺度分析。应改成 source/year；实际分辨率应读取来源配置/metadata。

本地40,550条选择数据的官方前缀对应39,083条aerial、1,467条satellite，约96.4%/3.6%。应写成以航空影像为主的多来源遥感子集；SAT是backbone预训练身份，不能把它当作当前训练影像来源。若论文特别研究Sentinel-2式中低分辨率卫星影像，当前训练分布代表性不足；若研究广义遥感RGB图文对齐，这个构成可以接受，但应报告来源分层结果。

另有未来模板实验的工具风险：[caption 处理](/Users/wangyue/Documents/ChatGPT/Dinov3txtforSAT/tools/prepare_skyscript.py:83) 的auto/contextual-summary遇到`It shows:`后一律恢复aerial前缀，会改写satellite样本。当前title_raw/full不受影响；RQ2开跑前应纠正或明确使用统一中性前缀。

### 3.3 unique-caption 简化了任务，也改变了数据分布

370,317条images2/3候选压缩为40,550条唯一文本，一组一图保留约10.95%的候选。它有效减少精确同文假负例，但把常见caption与稀有caption的权重拉平，舍弃了合法多图正例。**现在研究的是unique-caption子集上的适配，不能直接代表原始SkyScript分布。**

同义句和相似对象仍可能互为假负例。比如表面不同的道路描述，未必需要在视觉上区别；单正例InfoNCE可能要求模型分辨图像里看不到的属性。下一轮数据扩展应优先考虑caption/对象group与多正例，而不是直接恢复大queue。

caption精确互斥也不是所有视觉泛化任务都必须满足的条件。“Greenhouse”同时出现在不同地点的train和test，并不自动构成泄漏。当前split更多考查未见内容句的匹配，应与未见地理区域的泛化区分。

### 3.4 地理隔离可以进一步做

本地manifest检查没有发现跨split的同一OSM object ID；已有审计记录文件/像素精确重叠为零。这是积极证据，但不同对象可能位于同一视野、相邻区域或不同时间版本。

上游提供meta2–meta7，包含每张图的经纬度bbox、时间、focus/surrounding tags。当前CSV没有坐标，不等于上游没有坐标。[官方 metadata](https://github.com/wangzhecheng/SkyScript#download-meta-files)

建议保留现有split作为开发协议，取得所选图像对应metadata后审核bbox相交和近邻区域，建立独立地理held-out评测。新的划分与现有pilot分开命名，重新建立基线；不要覆盖旧结果。

### 3.5 可见性抽查比继续盲试 LR 更有价值

本地样本包含Restaurant、University、fuel transportation、stop sign等用途或微小属性。本次查看了Restaurant与Bunker对应图片：用途属性不能仅凭这些俯视图可靠确认。这两例只说明审核必要性，不能估计全数据错误率。

建议先审核200–300个按来源、文本长度、常见/稀有语义分层抽样的pair，标注“直接可见／有视觉依据可推断／无法确认／不匹配”，报告原始计数和抽样规则。区分对象类别与用途属性；不要把LLM流畅表达当作新增真值。

## 4. 任务和评测：主任务正确，结论标准需修整

### 4.1 以全局双向检索为主，分类为辅助

双向遥感图文检索直接测量当前图文空间，适合作为论文主任务。RSICD已有一图多描述，评价代码正确地把同图多caption作为I→T正例。EuroSAT可以诊断场景语义迁移，但不能替代文本检索或证明文本encoder改善。当前没有局部监督和局部评测，定位/分割不宜并入最低交付。

SkyScript检索使用完整4,055候选池，query chunk只影响计算吞吐；训练的16候选loss与这个全局检索不是同一难度。RSICD的候选池与多正例规则不同，两个数据集的mean recall不能直接横比。正式表格应写明images/captions、候选池、正例定义与双向R@1/5/10。

### 4.2 并列排名存在已复现的边界缺陷

[retrieval.py:151](/Users/wangyue/Documents/ChatGPT/Dinov3txtforSAT/src/dinotxt_rs/evaluation/retrieval.py:151) 使用`count(scores > best_positive)+1`，把所有与正例同分的候选视为排在它后面。

本次CPU反例：三个图像和三个文本embedding完全相同，当前函数仍返回双向R@1=1、mean recall=1。这确认了乐观tie处理的边界缺陷；**没有证据说明现有M4结果有大量ties或发生坍缩，不能因此撤销历史结果。**

正式评测前应固定并声明tie规则，监控并列率/表示坍缩，采用确定性排序或随机tie的期望指标；必要时同时给乐观/悲观排名。修正后的指标版本应同时重评baseline和候选，不能只改某个模型。

### 4.3 Gate 与文本 drift 都不是能力证明

现有Gate很适合决定是否继续耗费算力，但不等同于统计显著性。F3单seed增益相对F0跨seed标准差的倍数，不能证明效应真实或不存在。[F3接受规则](/Users/wangyue/Documents/ChatGPT/Dinov3txtforSAT/tools/summarize_skyscript_m4_f3_sat.py:202)

文档将F3 LR5e-6的RSICD +0.00521称为“真实信号”，宜收紧为“单seed正向信号，待配对复现”。最终比较报告每个seed的候选减基线差值；若给样本不确定性，按图像/空间group做配对bootstrap，多caption不能当作独立图像。

两个遥感caption集合上的cosine drift很小，只能说明这些输入的文本输出改动小。小漂移不保证排序、通用语言空间或图文能力保持。正式文本微调应增加固定的外部能力检查；若要主张通用能力保持，需要通用域任务，而不只遥感caption距离。

RSICD的六项平均保持可能掩盖T→I下降，文档已认识到这一点。保持旧门槛的历史解释；后续新实验若更看重文本检索，应事先增加方向指标并说明目的，不能事后重判旧实验。

### 4.4 独立最终评测仍需冻结

RSICD-val已用于开发决策，RSICD-test/全量EuroSAT已被历史实验观察。这些结果仍可报告，但不能称完全未触碰测试。应选择一个未用于本项目开发的外部评测，或者构建新的地理held-out，在冻结数据、prompt、候选与checkpoint选择规则后统一评测。

领域模型比较可选官方dino.txt、CLIP与RemoteCLIP。它们主要回答实用性能位置，训练数据和预训练预算不同，不能全部当作方法因果对照。RemoteCLIP的检索评测覆盖RSICD/RSITMD等，使用前需核对其训练来源与目标评测重叠。[RemoteCLIP 官方仓库](https://github.com/ChenDelong1999/RemoteCLIP)

## 5. 建议的最小正式实验

若保留“冻结视觉 backbone、检验文本微调”这个毕业论文目标，建议核心矩阵如下。vision head和logit scale固定，文本更新范围在C/D完全一致：

| 条件 | 图像 adapter | 文本侧 | 要回答的问题 |
| --- | --- | --- | --- |
| A | 无更新 | 全冻结 | 当前迁移接口起点 |
| B | 训练 | 全冻结 | F0图像适配基线 |
| C | 无更新 | 最后1 block + final norm；projection是否更新固定 | 文本侧单独能否适配 |
| D | 训练 | 与C相同 | 文本更新相对B是否提供增量 |

已有F3可作为projection-only诊断，帮助区分最终映射调整与encoder内部更新。如果资源只够验证文本的互补作用，最低保留B、F3与新增encoder候选；不要将它表述为已完整回答文本单独适配。

先以SAT完成固定候选筛查，随后对B和保留候选运行统一3 epochs、三个配对seed；Web复制冻结后的关键配置作为对照。SAT调出的参数复制到Web适合测同一配方的迁移，不能声称两域各自最优。随机深度、有效候选数、数据曝光和选择规则都进入协议。

不要在这轮同时增加queue、改caption、扩数据或开随机裁剪。若研究视觉预训练域，另补直接投影的接口诊断；若研究文本微调，则以B→D的配对差异作为主要方法问题。两种研究问题都可成立，但应有明确主次。

最终只依据预设开发指标选择一次候选，再运行冻结的外部评测。统一报告训练耗时、显存、参数量、适配收益与保持代价。负结果同样可写成在明确数据与预算下的适用条件，不能扩写成所有文本微调无效。

## 6. 建议执行顺序与文档修正

1. **先修评测与元数据解释。** 固定tie规则；source/year代替country/zoom；模板处理避免无意替换视角。旧实验保留原指标版本。
2. **补数据审核与独立评测协议。** 做可见性抽查，检查metadata可用性，规定地理/外部held-out用途。
3. **保留一个真正的文本encoder实验。** 取消“projection必须先赢才可检验encoder”的科研前提；限定候选数量、范围与预算。
4. **若继续head分支，先隔离随机深度。** 原F1/F2可作为既有配方结果，不当作head微调的全面否定。
5. **统一充分曝光的少量配对实验。** 最终报告稳定性和双向权衡；不要只延长短预算赢家。
6. **最后按主问题扩展。** 多正例/候选数优先于规模律或局部任务；不为模块数量而增加模块。

交接第8.2节仍写优化器只有全局LR、分组是待做任务，但[分组实现](/Users/wangyue/Documents/ChatGPT/Dinov3txtforSAT/src/dinotxt_rs/models/official_dinotxt.py:120)已经存在，需更新状态。README和科研计划相对最新交接也滞后；论文问题、实际主线和下一步入口应统一，避免继续沿用旧止损规则替代当前研究目标。

可以考虑将主问题表述为：**在冻结遥感视觉主干的条件下，图像接口适配与文本侧有限微调，分别如何影响遥感跨模态检索及迁移能力？** 若题目必须保持文本微调，则图像adapter与视觉接口诊断服务于建立可信基线，文本encoder的独立增量必须得到实际检验。成熟adapter结构本身不宜直接宣称算法创新；贡献应落在明确问题、可识别对照、数据/评测质量及由证据支持的机制分析。
