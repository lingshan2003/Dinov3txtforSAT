# 最新交接：M4 短预算机制证据完成，进入 SAT-centric 微调研究

更新时间：2026-09-10

## 1. 一页结论

项目当前已经得到一个可信的稳定基线：

> SkyScript images2+images3 unique-caption 数据、`title_raw` 文本、冻结官方 dino.txt、仅训练
> 2048→256→2048 image embedding residual adapter，在 Web DINOv3 ViT-L/16 上能够跨
> seed 稳定改善同源全局图文检索，并在不足一轮数据的 500-step 短日程中持续改善而不发生历史路线
> 的短程反转。

Gate S0、S1、S2、M4-A 和 M4-B 均已通过，但这些 Gate 证明的是短预算下的工程正确性、可适配性和
跨 seed 方向，不是充分训练后的最终域结论。500 optimizer step 只消费 32,000 个样本，相当于当前
36,495 条训练集的 0.877 轮。因此 M4 应定位为 **Web/SAT 短预算 matched pilot**。

项目的研究主角仍是 SAT DINOv3：目标是在永久冻结 SAT backbone 的前提下，利用 SkyScript 微调
其后的对齐模块和文本侧，使只有视觉预训练权重的 SAT backbone 接入现有通用 dino.txt vision
head/text encoder。Web 是天然兼容该对齐头的参照组和性能参考，不是替代 SAT 的后续研究主线。
下一步进入第 8 节的 SAT-centric 微调研究，先把不同模块的作用与安全 learning rate 搞清楚，再
设计覆盖完整 epoch 的正式长训练。

本轮同时冻结以下研究决策：

1. 当前固定训练数据使用完整的 36,495 条 train，不再为了沿用旧的“10k / 50k”命名而人为切出
   10k 主实验。
2. 当前不单列紧接着必须完成的 scale 阶段。规模研究推迟到扩展更多分片、恢复多图同语义关系，
   或建立 multi-positive/false-negative 处理后再做。
3. 原计划中的统一文本模板实验不再是进入 M4 的前置 Gate；它保留为 RQ2 可选实验。
4. unique-caption 不只服务 negative queue。它已经减少当前 in-batch InfoNCE 的精确同文假负例，
   同时让未来 queue 更安全；但它并没有自动解决同义文本、近重复地点和 queue embedding 过时。
5. queue 与“扩大可训练参数范围”是两个独立实验轴，不能在同一个 run 中同时打开。
6. `visual_model.backbone` 在所有后续实验中永久冻结，这是不可突破的研究协议；可训练范围只允许
   位于其后的 dino.txt 对齐头、adapter、文本 encoder/projection 和 `logit_scale`，并按模块使用
   分组 learning rate，不能把当前 adapter 的 `1e-4` 直接施加到全部预训练模块。
7. 后续按“SAT 机制研究 → 冻结候选和协议 → 一次性 Web matched control”的顺序执行；开发期间不在
   每个候选上来回切换 backbone。只有 SAT 侧的提升情况、训练范围、LR 和选择规则确定后，才统一
   替换为 Web backbone，批量完成同构对照矩阵。

研究背景见[科研背景](../DINOv3_Remote_Sensing_Domain_Text_Alignment_Research_Plan.md)，稳定工程规则见
[开发架构](DEVELOPMENT_ARCHITECTURE.md)。本文是当前状态、具体协议和下一步操作的唯一主文档。

## 2. 已完成证据

### 2.1 Gate S0：单 seed 候选成立

seed11 的 100-step adapter-only 实验通过：

- 训练与 validation 精确文件 hash overlap=0；
- 解码 RGB 像素 hash overlap=0；
- adapter step0 与无 adapter 官方输出 parity 通过；
- SkyScript held-out 全局 retrieval 改善；
- RSICD-val mean recall 满足绝对退化不超过 0.01 的保持门槛；
- checkpoint、validation、best、provenance 和 queue_size=0 均核验通过。

### 2.2 Gate S1：跨 seed 复现通过

主报告：

```text
outputs/skyscript_gate_s1/summary.json
```

seed11/23/47 的所有预设门槛均通过：

| seed | step100 validation loss | SkyScript mean recall | 相对 step0 | RSICD 相对官方 |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 0.9726497 | 0.1138512 | +0.0275791 | +0.0021938 |
| 23 | 0.9662531 | 0.1161118 | +0.0298397 | +0.0012492 |
| 47 | 0.9638958 | 0.1169338 | +0.0306617 | +0.0068251 |

三 seed 汇总：

- step100 validation loss：`0.9675995 ± 0.0045296`；
- SkyScript mean recall：`0.1156323 ± 0.0015963`；
- SkyScript 相对共同 step0 的绝对增量：`0.0293602 ± 0.0015963`，相对约 34%；
- RSICD-val 相对官方的 mean recall 增量：`0.0034227 ± 0.0029842`。

这里的离散度是 sample standard deviation，ddof=1。跨 seed 稳定结论只覆盖这三个预先指定 seed、
当前数据和 100-step 协议。

### 2.3 Gate S2：500-step 稳定性通过

训练目录：

```text
outputs/skyscript_images23_top30raw_imageadapter256_36495_500step_seed11/
```

主报告：

```text
outputs/skyscript_gate_s2_seed11/summary.json
```

S2 使用新的 500-step scheduler，从官方初始化开始，而不是续跑任何 100-step run。训练依次在
step100、250、500 停止；每个阶段通过产物、resume、parity、overlap、SkyScript retrieval 和
RSICD-val 保持门槛后才进入下一段。

| step | validation loss | SkyScript mean recall | 相对 step0 | RSICD mean recall | 相对官方 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / official | 1.5994003 | 0.0862721 | 0 | 0.1587751 | 0 |
| 100 | 1.0224082 | 0.1127826 | +0.0265105 | 0.1539610 | -0.0048141 |
| 250 | 0.8698717 | 0.1283189 | +0.0420469 | 0.1542657 | -0.0045094 |
| 500 | **0.7617921** | **0.1375668** | **+0.0512947** | **0.1542962** | **-0.0044790** |

结论：

- validation 每 50 step 严格下降，step500 为 best；相对 step0 下降 52.37%；
- SkyScript retrieval 在 100→250→500 持续提高，step500 相对 step0 提高 59.46%；
- 没有出现历史 ChatEarthNet 路线中“短期改善、长训练反转”的现象；
- RSICD mean recall 在三个阶段都低于官方，但退化约 0.45–0.48 个百分点，满足预先规定的
  退化不超过 1 个百分点门槛；
- step500 是当前协议内的首选 checkpoint，但不能据此声称外部能力全面改善。

### 2.4 外部保持的方向性限制

S1 和 S2 都显示 RSICD 方向不对称：

- image→text 多数 recall 和 rank 改善；
- text→image recall 系统性下降；
- mean recall 门槛通过会掩盖这个方向性权衡。

因此当前允许的表述是“RSICD-val 平均检索能力在预设门槛内得到保持”。不能写成“RSICD 双向能力
全面提高”或“跨数据源泛化已经解决”。后续继续报告两个方向的全部 R@1/R@5/R@10、median rank
和 mean rank，但不根据已经观察到的 RSICD-val 结果事后修改主门槛。

### 2.5 M4-A：Web–SAT matched seed11 短预算 pilot 通过

M4-A 归档报告：

```text
outputs/skyscript_m4_rq1_seed11/summary.json
```

报告状态为 `pass`，Web/SAT 配置差异、模型与预处理域、权重 hash、manifest、训练报告和每个
checkpoint 的身份均已核验。SAT run 在 step100、250、500 的预设 Gate 全部通过。

| 数据集 | 指标 | Web step0/official | Web step500 | Web delta | SAT step0/official | SAT step500 | SAT delta | SAT−Web step500 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SkyScript-val | mean recall | 0.0862721 | 0.1375668 | +0.0512947 | 0.0011097 | 0.0659679 | +0.0648582 | -0.0715988 |
| RSICD-val | mean recall | 0.1587751 | 0.1542962 | -0.0044790 | 0.0046313 | 0.0642901 | +0.0596587 | -0.0900061 |

validation loss 同样显示持续适配：Web 从 1.5994003 降至 0.7617921，SAT 从 3.8645770 降至
1.1673670，二者的 best 都是 step500。

允许的结论：

- SAT backbone 与通用 dino.txt 对齐头在初始化时存在严重表示错配；两个数据集的绝对检索能力都
  远低于 Web 路径。
- adapter 能够持续、显著地修复 SAT 路径；到 step500，SAT 在 SkyScript 与 RSICD 上相对自身
  初始化的 mean-recall 增量均大于 Web 路径。
- 这种较大的增量主要来自更低的起点。适配后的 SAT 绝对性能仍明显落后 Web，因此 M4-A 不支持
  “SAT 视觉初始化优于 Web”的结论。
- 该结果研究的是“通用 dino.txt 对齐头迁移到不同视觉初始化后的适配行为”，不是纯粹比较两个
  backbone 的视觉表征质量。
- seed11 只是 pilot。任何稳定性或 RQ1 总结必须等待 M4-B 的 matched 三 seed 结果。

### 2.6 M4-B：Web–SAT matched 三 seed 短预算对照通过

M4-B step500 归档报告：

```text
outputs/skyscript_m4_rq1_m4_b_stage500/summary.json
```

Web/SAT × seed11/23/47 六个 run 在 stage100、250、500 全部通过预注册 Gate。以下离散度均为
sample standard deviation（ddof=1），SAT−Web 为同 seed 配对差异后再汇总。

| 数据集/指标 | Web step500 | Web 同域 delta | SAT step500 | SAT 同域 delta | SAT−Web step500 | SAT−Web delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation loss | 0.7634067 ± 0.0042998 | -0.8369860 ± 0.0041016 | 1.1684883 ± 0.0038230 | -2.6957862 ± 0.0038978 | +0.4050816 ± 0.0077754 | -1.8588002 ± 0.0077946 |
| SkyScript mean recall | 0.1400055 ± 0.0034379 | +0.0537060 ± 0.0034234 | 0.0638033 ± 0.0020638 | +0.0626661 ± 0.0020854 | -0.0762022 ± 0.0053554 | +0.0089601 ± 0.0053554 |
| RSICD mean recall | 0.1554845 ± 0.0039894 | -0.0030875 ± 0.0040383 | 0.0624924 ± 0.0032465 | +0.0578204 ± 0.0032635 | -0.0929921 ± 0.0027745 | +0.0609080 ± 0.0029723 |

三 seed 方向完全一致：

- SkyScript step500 的 Web 绝对 mean recall 在每个 seed 都高于 SAT；Web−SAT 优势约
  0.0716–0.0821。
- SAT 在每个 seed 的 SkyScript 同域增量都略大于 Web，但这只是从极低错配起点追赶，未转化为
  更高绝对性能。
- RSICD step500 的 Web 绝对 mean recall 在每个 seed 都高于 SAT；Web 路径相对官方初始化的
  三 seed delta 为 -0.00448、-0.00625、+0.00146，均满足预设的 -0.01 保持门槛。
- SAT 在 RSICD 上也从近乎失效的初始化明显恢复，但 step500 绝对性能仍远低于 Web。
- 两域 validation loss 都稳定下降且 step500 为 best；SAT 的更大下降反映严重初始错配被部分
  修复，不能解读成最终模型更好。

M4 的 adapter 阶段至此完成，并已具备 Web/SAT × 三 seed 的 RQ1 配对矩阵证据。短预算结论是：
在“冻结通用 dino.txt 对齐头 + 仅训练相同 image adapter”的协议下，Web DINOv3
初始化在500 step时提供跨 seed 稳定的更高绝对图文检索性能；SAT DINOv3 则表现出更大的同域增量，
证明通用对齐能力可以通过微调迁移到 SAT 路径，但单个输出 adapter 尚不足以弥合初始化错配。

这不是“SAT 应被 Web 替代”的结论。恰恰相反，它给出了继续研究 SAT 的依据：SAT 路径不是不可训练，
瓶颈更可能位于 SAT backbone 与通用 dino.txt vision head 的接口，以及被冻结的对齐模块容量。后续
主线应以 SAT 为研究对象，Web 仅作为天然兼容对照。该结论不能外推为 SAT backbone 的纯视觉表征
质量劣于 Web backbone；n=3 的均值与 sample std 只是短预算重复实验的描述性证据。

### 2.7 SAT F1/F2：vision head 微调机制筛查完成

seed11 的四个500-step候选已经完成，汇总状态为 `complete`，并核验
`visual_backbone_permanently_frozen=true`。F0 直接复用 M4-A 的 SAT adapter-only 归档结果，没有
重训。step500 结果如下；validation loss 越低越好，其余两列越高越好：

| 候选 | 可训练范围 | validation loss | SkyScript mean recall | RSICD mean recall |
| --- | --- | ---: | ---: | ---: |
| F0 | adapter | **1.1673670** | **0.0659679** | 0.0642901 |
| F1, head LR `1e-5` | vision head | 2.5149034 | 0.0066995 | 0.0085314 |
| F1, head LR `5e-6` | vision head | 2.8703583 | 0.0036580 | 0.0071603 |
| F2, head LR `1e-5` | adapter + vision head | 1.2193511 | 0.0577065 | 0.0590798 |
| F2, head LR `5e-6` | adapter + vision head | 1.2396846 | 0.0564324 | **0.0649299** |

新旧代码运行的 step0 存在极小数值/排序差异：validation loss 约0.00045，SkyScript mean recall
约0.000041，RSICD mean recall约0.000061；远小于主要 step500 候选差距，不改变结论。F2 `5e-6`
在 RSICD 上仅高于 F0 的0.00064，因此该微小优势尤其不能脱离三 seed 波动单独解读。

结论：

- 四个候选相对各自接近失效的 step0 都有提升，证明梯度、冻结策略和训练链路有效；但这不足以说明
  新机制优于已经完成的 F0。
- F1 在两个 LR 下都远弱于 F0。vision head 有约25.33M可训练参数，仅靠它在当前500-step日程中
  不是有效替代；`1e-5` 全程优于 `5e-6`，说明更低 LR 主要让学习更慢，不能据此宣称 head 无容量。
- F2 明显依赖 adapter 才能接近 F0，但两个 LR 的 SkyScript 都低于 F0：分别低0.00826和0.00954；
  validation loss 也分别高0.05198和0.07232。因此当前没有证据表明 vision head 与 adapter 在同域
  主指标上互补。
- F2 `5e-6` 的 RSICD 比 F0 高0.00064，但幅度远小于 M4-B SAT F0 的三 seed 标准差0.00325，且伴随
  SkyScript/validation退化，只能记录为可能的域保持权衡，不能选为主方案。
- F2 `1e-5` 比 `5e-6` 更适合同域 SkyScript 和 validation；`5e-6` 更保留 RSICD，呈现清晰的
  domain-adaptation trade-off。四个候选均未满足同时超过 F0 的预设条件，`eligible_candidates=[]`。

决策：F1/F2 不进入三 seed，也暂不复制到 Web。图像侧主方案继续使用 F0 adapter-only；下一项
机制筛查以 F0 为锚点，只新增 text projection。若 text projection 仍提供一致收益，再研究最后1个
text block/final norm，并报告相对冻结参考文本模型的 embedding drift。

## 3. 证据身份与核验范围

| 范围 | 身份 |
| --- | --- |
| seed11 初始训练代码 | `4c564d25f3946531055e26dea6890fb892fed2fc` |
| Gate S0 代码 | `9202423` |
| seed23/47 与 Gate S1 | `bda81a2d27c0ded8529937c80d02ecbfccdcb6ae` |
| Gate S2 训练与编排 | `78930a4ff820ccc7afcda4ed79e2bd0f456875d6` |
| M4-A SAT 初始训练 | `864b61de0910f159d7e29567928e19bc8a928254` |
| M4-A SAT 后续恢复 | `e862995556d3a5deb048b100df362d04ff35af1e` |
| DINOv3 上游 | `6876159a11b4df116f30f667f8c9888617df0751` |
| S1 summary SHA-256 | `e306ee2a4404b3e371fac9ab9e63d302514c34bc21ed46e01e45f658eb18c1cf` |
| S2 summary SHA-256 | `3620403b3653c8955177e612e8da0d23d766fe369187f2fb367b4d32f071ebca` |
| M4-A SAT summary SHA-256 | `7723b4cd6a4851e2c766ebd18aae0a97102b5cfee37f465df2d9309ddf55f2be` |
| M4-A Web–SAT summary SHA-256 | `69bef90195afca5d19bd79a110f2261c0ece9a281fe6b71672aaa3d93ffaa9dc` |
| M4-B step500 summary SHA-256 | `925ec542e5e2276fe903930f2295b6db78be1c4798dfb3429f0c185d548da24d` |
| M4-B 编排与报告代码 | `97f5b6ab3f672772bb15158d8c05653f1130e2b6` |
| SAT F1/F2 summary SHA-256 | `66a3aff1aced2b7a3bc4388bc7b9c0b7bba0efac5970a0df5c408135310910b7` |

本地还读取了用户从服务器下载的 SAT F1/F2 `summary.json` 副本。本会话没有直接登录服务器重新
读取 checkpoint；服务器中的训练目录、parity、optimizer groups、逐 checkpoint retrieval 和
resume history 仍是最底层完整证据来源。

seed11 与 seed23/47 的训练 project commit 不同，但两者之间没有改变训练器、adapter 或模型训练
实现；新增的是评测、编排、配置和文档。正式报告仍须披露 provenance，不把不同 commit 写成同一
运行身份。

## 4. 冻结的数据协议

### 4.1 数据与 split

当前使用 SkyScript language-polished top30 的 images2+images3：

- 370,317 条候选记录；
- 按规范化 `title_raw` 分组，seed11 每组确定性选择一张图；
- 得到 40,550 个唯一 caption-image pair；
- seed23 按 caption 确定性拆分为 36,495 train 和 4,055 validation；
- train/validation 规范化 caption overlap=0；
- 精确文件 overlap=0，解码像素 overlap=0。

固定输入：

| 文件 | 样本数 | SHA-256 |
| --- | ---: | --- |
| `skyscript_images23_train_raw_unique36495_seed11_global77.jsonl` | 36,495 | `4264d884d564631816baf3e339e7ca12be2958966d99a60f6a75ae3d5d4dbaec` |
| `skyscript_images23_val_raw_unique4055_seed23_global77.jsonl` | 4,055 | `062238716b8fddad890b4f97354588ad7524f1099667d3c0baa932002845b7f6` |

manifest 文件名中的 seed11 是代表图片选择身份，seed23 是 split 身份；实验 seed 改变时不得改名。

文本固定使用 `title_raw` 和真实 dino.txt tokenizer 的 77-token global 协议。统一视角模板不再是
M4 前置任务，未来 RQ2 若执行，必须使用完全相同图片与 split 做 matched-image 对照。

### 4.2 为什么当前不再单独切 scale

当前 500 optimizer steps、physical batch16、gradient accumulation4 共消费 32,000 个样本，约为
36,495 条 train 的 87.7%，还不到完整一轮数据。sampler 每轮有 `floor(36,495/16)=2,280` 个
physical batch，4 次累积更新一次，所以一轮恰好对应 570 optimizer step；每轮末尾因 `drop_last`
固定丢弃15条。M4 的500 step属于短预算机制筛查，不能称为充分利用完整训练集的正式长训练。

此时把 train 人为缩为 10k 会带来两种混杂：

- 固定 steps 时，10k 数据会被重复看更多轮；
- 固定 epochs 时，优化步数和 scheduler 预算又不同。

images2+images3 只有 40,550 个唯一 caption-image pair；完整 polished CSV 的规范化唯一文本统计上限
约 52,923。当前可形成的纯 unique-caption 规模跨度不足以支撑有解释力的 scale law。为了回答 RQ1，
直接在已经验证成功的完整 36,495 train 上控制 Web/SAT 变量，比退回任意 10k 更合理。

因此具体里程碑调整为：

- M4 使用固定 36,495/4,055 协议完成短预算视觉域对照和可适配性判断；
- 模块范围确定后的正式实验改用 epoch-aligned 预算，至少覆盖完整数据轮次；
- 原 M5 scale 暂停，不删除架构中的抽象研究问题；
- 只有扩展更多分片，或利用 multi-positive 正确恢复“同 caption 多图片”后，才重新设计嵌套规模实验；
- 如果未来确实研究 data efficiency，必须提前冻结“固定样本曝光、固定 steps 或固定 epochs”中的一种
  比较语义，不能事后选择有利口径。

### 4.3 仍未解决的数据风险

- exact caption 唯一不等于语义唯一；同义描述仍可能成为假负例；
- 精确文件/像素 hash 无法排除同一区域的不同裁剪、尺度或时间版本；
- 当前 CSV 没有直接可用于 geography-group split 的可靠坐标；
- 每个 caption 只保留一张图降低了歧义，但也丢失了合法多正例和真实频率信息；
- SkyScript validation 与 train 来自同一生成和筛选流程，不是独立跨数据源测试。

## 5. 冻结的方法基线

### 5.1 模型与可训练范围

当前 baseline：

```text
official normalized image embedding (2048)
  → LayerNorm
  → Linear(2048, 256)
  → GELU
  → zero-initialized Linear(256, 2048)
  → residual add
  → L2 normalize
```

可训练参数为 1,054,976，仅属于 image adapter。冻结：

- DINOv3 backbone；
- dino.txt vision head；
- text backbone；
- text projection；
- logit scale。

adapter 只处理全局 image embedding，不处理 patch tokens，因此不能支持 local alignment 结论。

### 5.2 当前 500-step 协议

| 项目 | 设置 |
| --- | --- |
| seed | M4 的 Web/SAT 500-step 对照均为 11/23/47 |
| physical batch | 16 |
| gradient accumulation | 4 |
| optimizer effective samples/step | 64 |
| contrastive candidates/forward | 16（queue 关闭时） |
| precision | BF16 |
| optimizer | AdamW |
| adapter learning rate | `1e-4` |
| weight decay | 0.01 |
| max grad norm | 1.0 |
| total steps | 500 |
| 等效训练轮数 | 0.877 epoch（32,000 / 36,495） |
| warmup | 50 |
| scheduler | linear warmup + cosine decay |
| validation/checkpoint | 每 50 step |
| augmentation | 关闭 |
| negative queue | 关闭 |

梯度累积不会把对比候选从16扩为64；它只扩大一次 optimizer update 消费的样本数。后续“完整一轮”
应写成570 optimizer step，而不是沿用500这个便于 staged 检查的整数。

## 6. unique-caption 与 negative queue 的关系

### 6.1 unique-caption 现在已经发挥作用

使用 unique-caption 的目的不只是在未来打开 queue：

1. 当前每次 forward 的 16 个样本互为负例。即使 queue_size=0，重复 caption 也会在普通 in-batch
   InfoNCE 中制造一对多关系却只给一个 target 的假负例。
2. caption 互斥使 train/validation 文本泄漏更容易定义和审计。
3. 每个文本只选一张图，简化了当前“一图一文、单正例”retrieval 和 loss 语义。
4. 它为 future queue 去掉了最明确的一类 exact-caption collision，使 queue 实验更可控。

所以当前 adapter-only 成功与 unique-caption 清洗是同一基线的组成部分，不能说“因为没有 queue，
去重没有用”。

### 6.2 queue 关闭不是因为 adapter 太小

queue 决定 softmax 能看到多少负样本，与有多少可训练参数没有直接对应关系。小 adapter 同样可能从
更丰富的负例中获益。当前关闭 queue 的真实原因是先隔离最小因果变量，并避免现有实现的风险：

- `EmbeddingQueue` 保存 detached image/text embeddings，不保存 sample ID、caption group 或地点组；
- loss 对每个 query 只承认一个正例，无法 mask 已知同义或近重复样本；
- text tower 冻结时 queued text embedding 是稳定的，但 image adapter 持续变化，queued image
  embedding 会逐渐 stale；
- queue 同时改变两个方向的候选集合，可能放大当前 RSICD 的方向性权衡；
- 历史 4096 queue 与其他变量同时变化，不能作为当前 queue 有效性的证据。

unique-caption 是安全使用 queue 的重要准备，但不是充分条件。

### 6.3 未来 queue 实验的前置工程

在打开 queue 前按顺序完成：

1. 为训练记录保留稳定 sample ID、normalized-caption group，并尽可能增加近重复/地理 group；
2. queue 同步保存这些 identity，loss 能 mask 已知正例或采用 multi-positive target；
3. 明确 symmetric queue 中 stale image embedding 的策略；可以先研究只给 image→text 使用冻结文本
   negative 的非对称诊断，但不能与 symmetric baseline 混称；
4. 增加 queue occupancy、age、masked-negative count 和 in-batch-only loss 监控；
5. 先做单变量小容量 pilot，例如 256，再决定是否到 1024；不直接恢复历史 4096；
6. 数据、augmentation、trainable scope、LR 和 schedule 全部保持不变。

queue 只有在自己的同源 retrieval 改善且外部保持不恶化时才接受。失败也要保留，因为它能说明
“唯一 caption 足以清除精确冲突，但不足以保证大负样本集合安全”。

## 7. 已完成阶段：M4 Web–SAT 短预算严格对照

### 7.1 要回答的问题

在相同 SkyScript 数据、文本、adapter、优化、训练预算和评测下，更换视觉预训练域会如何影响：

- 各自初始化时的图文能力；
- 适配带来的相对增益；
- 适配后的绝对性能；
- 外部保持与方向性权衡；
- 长日程稳定性。

SAT 路径使用 SAT DINOv3 backbone 加通用 dino.txt vision/text head。这不是官方联合训练的
SAT-dino.txt，存在表示兼容性混杂。结果应解释为“通用对齐头迁移到不同视觉初始化后的适配行为”，
不能简化为纯视觉预训练域优劣。

### 7.2 M4-A：matched seed11 staged pilot

状态：已完成并通过。结果和解释见 2.5 节。

Web 对照直接使用已经通过的 S2 run。新增 SAT 配置只允许改变：

- `experiment.name`；
- `experiment.output_dir`；
- `model.backbone_domain = "sat"`；
- `model.backbone_weights = assets/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`。

其余字段与 `configs/skyscript_web_adapter_500step_seed11.toml` 完全相同。SAT 也必须从自己的官方
初始化开始，不能加载 Web adapter 或历史 SAT checkpoint。

执行仍为 0→100→250→500。每个阶段对每个 backbone 分别检查：

- 当前 validation loss 低于自身 step0，best 不回到 step0；
- SkyScript retrieval 高于自身 step0；
- RSICD-val mean recall 相对自身 official 初始化退化不超过 0.01；
- step0 parity、train/val overlap、train/RSICD overlap、checkpoint 和 resume 全部通过。

比较报告必须同时给出：

1. Web/SAT 各自 step0 的绝对值；
2. 各自适配前后的 delta；
3. 同 step 的绝对差；
4. 双向 retrieval 全指标；
5. trainable 参数、预处理域、权重 hash、代码和 manifest 身份。

任一 SAT stage 失败即停止 SAT 长训练，但失败不是工程失败时仍属于有效研究结果。不能因为 SAT
初始化绝对值低，就降低其保持门槛或更换外部集。

### 7.3 M4-B：跨 seed 的 RQ1 证据

状态：已完成并通过。六个 domain×seed run 均通过 stage100、250、500；结果和解释见2.6节。
M4 到此停止追加 seed 或事后更改 Gate，但完整 RQ1 尚未结束：当前只回答 adapter-only、0.877 epoch
下的可适配性和方向，下一阶段需要在 SAT 路径上研究更合理的微调范围并进行 epoch-aligned 训练。

M4-A 是 matched-seed pilot，不足以单独完成视觉域结论。预先规定下一层复现：

1. 为 Web 500-step schedule 和 SAT 500-step schedule 分别创建 seed23/47 配置；
2. 先让两域所有 seed 运行到 stage100，形成真正相同 warmup50/total500 scheduler 的三 seed 对照；
   不能用 S1 的 warmup10/total100 run 冒充这一对照；
3. 两域的某个 seed 若在 stage100 失败，保留并汇报，不能只延长通过的 seed 后报告最好结果；
4. 两域三个 seed 均通过 stage100 后，再按相同规则扩展到 250/500；
5. 最终按 domain 报告逐 seed、mean、sample std 和 Web–SAT difference，不只报告最好 checkpoint。

M4-B 允许对“500-step adapter-only 现象”做跨 seed 稳定性结论，但不能把这个有限结论升级为
充分训练后的最终域优劣。

### 7.4 M4 归档身份

M4 短预算对照已完成，以下产物只读保留，不作为后续实验的可续跑目录：

```text
configs/skyscript_sat_adapter_500step_seed11.toml
outputs/skyscript_images23_top30raw_sat_imageadapter256_36495_500step_seed11/
outputs/skyscript_gate_m4_sat_seed11/
outputs/skyscript_m4_rq1_seed11/summary.json
outputs/skyscript_gate_m4_b_{web,sat}_seed{23,47}/
outputs/skyscript_m4_rq1_m4_b_stage{100,250,500}/summary.json
```

## 8. 下一阶段：SAT backbone 如何接入通用 dino.txt

### 8.1 研究目标与角色

后续主角固定为 SAT DINOv3 backbone。Web backbone 的作用是：

- 提供通用 dino.txt 对齐头天然兼容时的参考上界；
- 帮助判断 SAT 的问题来自训练器本身，还是来自 backbone/head 接口错配；
- 在 SAT 阶段结束后，对冻结的关键候选一次性做 matched control，而不是替代 SAT 主线或为每个
  开发尝试即时重复训练。

SAT 路径要回答的是：在**始终冻结 SAT 视觉 backbone**、完整保留遥感视觉表征的前提下，能否微调
其后的对齐层和文本侧，把它映射到通用 dino.txt 文本空间。M4 已证明仅在最终2048维 image
embedding 后增加 adapter 可以部分做到，但仍留下明显绝对差距；下一步先分辨 dino.txt vision
head 与 adapter 各自的作用，再逐步研究 text projection 和 text encoder。这里的 dino.txt vision
head 虽然内部可以包含 transformer blocks，但它是附加在冻结 backbone 之后的对齐头，绝不能写成
“SAT backbone block”或据此解冻 `visual_model.backbone`。这与
[DINOv2 Meets Text](https://openaccess.thecvf.com/content/CVPR2025/html/Jose_DINOv2_Meets_Text_A_Unified_Framework_for_Image-_and_Pixel-Level_CVPR_2025_paper.html)
冻结预训练视觉 backbone、训练其后新增对齐模块及文本侧的做法保持一致。

### 8.2 开跑前必须完成的优化器能力

当前代码虽然能解冻 `vision_head`、`text_projection`、最后若干 text block 和 `logit_scale`，AdamW
仍只有一个全局 LR。直接打开开关会把 adapter 的 `1e-4` 施加到所有预训练模块，实验无法区分
“模块没有帮助”和“学习率过大导致破坏”。第一项代码任务是实现显式 parameter groups：

| 参数组 | 初始用途 | 必须独立配置/记录 |
| --- | --- | --- |
| image adapter | 保留快速适配能力 | LR、weight decay、参数名/数量 |
| dino.txt vision head | 修复 SAT backbone→对齐头接口 | 较低 LR、weight decay、参数名/数量 |
| text projection | 后续调整文本空间最终映射 | 独立 LR、参数名/数量 |
| text last-k blocks + final norm | 研究语言域适配 | last-k、LR、层身份、参数名/数量 |
| logit scale | 只研究校准 | 独立 LR，默认无 weight decay |

实现必须保证：参数组互斥且覆盖全部 trainable 参数；零参数的未启用组不能静默出现；每组初始 LR、
当前 LR、weight decay、参数名和数量进入 provenance、checkpoint、训练日志和 verification；resume
必须逐组严格恢复；scheduler 对各组保持固定 LR 比例。

同时新增不可关闭的视觉冻结断言，而不是新增任何 `vision_last_k` 配置：

- `visual_model.backbone.*` 的 `requires_grad` 必须全部为 `False`；
- optimizer parameter groups 不得包含该前缀下的任何参数；
- 冻结 backbone 必须始终处于 eval mode，反向后其梯度必须全部为 `None`；
- provenance、checkpoint 和 verification 必须记录并复核上述不变量；若违反则拒绝训练；
- trainable-only checkpoint 不得包含该前缀下的权重。

### 8.3 微调机制 ladder

所有候选都使用永久冻结的 SAT backbone、相同数据/split/loss/augmentation-off/queue-off，并从
同一官方初始化重新开始；checkpoint 不能从上一个候选续跑：

| 阶段 | 可训练范围 | 状态/要检验的机制 |
| --- | --- | --- |
| F0 | adapter only | 已完成；当前图像侧锚点 |
| F1 | dino.txt vision head only | 已完成；明显弱于 F0，不扩 seed |
| F2 | adapter + dino.txt vision head | 已完成；未在主指标上超过 F0，不扩 seed |
| F3 | F0 adapter + text projection | 下一步；允许文本空间最终映射轻量适配 |
| F4 | F3 + 最后1个 text block/final norm | 研究有限文本 encoder 微调是否继续改善 |
| F5 | 仅在 F4 有效后扩到最后2或4个 text blocks | 检验更多文本侧容量，不能默认执行 |
| F6 | logit scale | 单独研究温度校准，不与新增表示层同时开启 |

F1/F2 已经分辨出现有 dino.txt vision head、adapter 及二者组合的作用：当前短预算下 adapter 是
主要有效模块，微调 vision head 没有增加同域收益。因此 F3 不携带 vision-head 微调，而以 F0 为
锚点只增加 text projection。无论结果如何都不能把解冻视觉 backbone 当作候选。文本侧实验还需报告
相对冻结参考模型的 text embedding cosine drift，避免把“适配遥感文本”与“遗忘通用文本空间”
混为一谈。

### 8.4 先搞明白微调，再做长训练

第一轮 seed11 mechanism screen 已完成，归档协议是：

1. F0 已由 M4-A 的 SAT adapter-only seed11 run 完成，不得重训；直接复用其 step100/250/500
   归档指标作为 matched baseline。为保持 scheduler 和样本曝光严格可比，F1/F2 也运行500 step；
2. F1 只训练 vision head；F2 同时训练 adapter 与 vision head。vision-head LR 只预注册两个低量级
   候选 `1e-5` 与 `5e-6`，F2 的 adapter LR 保持 `1e-4`，共四个候选，不做连续试参；
3. 在 step0/100/250/500 评估 validation、SkyScript 和 RSICD 双向 retrieval，并记录各参数组梯度范数；
4. F1/F2 的主判断不是“相对自身极低 SAT step0 是否上涨”，而是同 step 是否超过 matched F0，
   同时比较 F1 与 F2 以判断 adapter 是否仍有独立价值；
5. 若 image-side 候选失败，先诊断接口、归一化、梯度和参数组，不得解冻视觉 backbone；是否进入
   text projection/text encoder 候选应依据预注册规则和诊断结果，而不是把视觉解冻作为补救；
6. mechanism screen 使用同一个 seed 只负责方向和稳定性检查，不产生最终论文性能结论。

结果见第2.7节。没有候选同时超过 F0，故当前不追加 F1/F2 的三 seed 或 Web run；这不是运行失败，
而是一次有效的负向机制筛查。

建议的选择规则需在代码配置前进一步固化：SkyScript mean recall 为主要指标；validation loss、
双向 rank/recall 和 RSICD 为约束；不得只按最好 checkpoint 或单方向 R@K 选择。SAT 的 official
RSICD 起点接近零，旧的“相对 official 不下降0.01”门槛在这一阶段过于宽松，F1/F2 应直接与 matched
F0 的同 step 绝对结果比较。

### 8.5 正式长训练的预算定义

只有微调范围和 LR 选择完成后才冻结正式实验。正式预算不再用任意整数500表达，而使用完整 epoch：

- 1 epoch = 2,280 physical batches = 570 optimizer steps = 36,480 次样本曝光；
- 候选正式比较建议先预注册3 epochs，即1,710 optimizer steps；
- checkpoint/retrieval 至少在570、1,140、1,710 step执行，观察是否仍改善或开始过拟合；
- F0 与胜出的微调候选必须使用相同 total steps、warmup 语义和 seed11/23/47，从头训练；
- 如果3 epochs时 validation 与 retrieval 仍一致改善，再在看见结果前预注册是否延长到5 epochs
  （2,850 step），不能只延长表现最好的单个 seed；
- physical InfoNCE candidates 仍是16。增大候选集合/queue 是另一个实验轴，不能在微调范围研究中
  同时改变。

这套顺序把两个问题分开：短 screen 回答“应该微调哪些模块、用什么量级的 LR”，epoch-aligned
三 seed实验才回答“选定 SAT 微调方案在充分数据曝光下是否稳定优于 adapter-only”。

### 8.6 Web backbone 对照统一后置

后续不为 F1、F2、F3 等每一个 SAT 开发结果立刻追加 Web run。先完成 SAT 侧机制筛查和正式训练，
确认保留候选在预注册指标上确有稳定提升，并冻结需要进入最终对照表的配置集合。然后统一复制这些
配置，一次性运行 Web matched control；除以下四项外不得产生差异：

- `experiment.name`；
- `experiment.output_dir`；
- `model.backbone_domain`；
- `model.backbone_weights`。

Web 批次至少覆盖 adapter-only 基线和最终 SAT 胜出方案；若多个 SAT 机制候选被保留进主结果矩阵，
则对应 Web 候选也在同一批次全部完成。Web 结果不回流用于修改 SAT 超参数、筛选 checkpoint 或重写
通过规则，只用于回答相同对齐方法在不同冻结视觉 backbone 上的效果差异。这样既避免开发阶段反复
切换 backbone，也保留 RQ1 所需的严格配对解释。

## 9. 可选研究分支

### 9.1 Negative queue / multi-positive

完成 M4 baseline 后按第 6.3 节开展。它是优先级较高的方法扩展，因为当前 contrastive candidates
只有 16，unique-caption 又已经提供了较干净的起点。但必须先补 identity-aware masking 或明确的
multi-positive 语义。

### 9.2 RQ2 文本表达

`title_raw` 与 `An aerial image. It shows: ...` 的 matched-image 对照保留，但不再阻塞 M4。只有在
研究问题需要解释视角模板、结构化或层级文本时执行；不得重新选择图片或 split。

### 9.3 Scale

暂缓。扩展到更多分片、更多唯一文本，或恢复多图同文本后，重新定义嵌套数据集和训练预算，再决定
是否恢复正式 M5 scale。

### 9.4 Local alignment

当前 adapter 不作用于 patch token。局部对齐需要独立模块、短语/区域关系和定位或分割评价，属于
后续 RQ3，不从全局 retrieval 结果外推。

## 10. 当前禁止事项

- 不修改或续跑已经完成的 S0/S1/S2 输出；
- 不从 Web checkpoint 初始化 SAT adapter；
- 不把 SAT+dino.txt 迁移头称为官方 SAT-dino.txt；
- 不为了让 SAT 通过而修改 RSICD 门槛、数据或 checkpoint 选择规则；
- 不使用 RSICD test 或全量 EuroSAT 循环调参；
- 不同时改变 backbone、trainable scope、queue、augmentation 或 caption；
- 不直接恢复 4096 queue；
- 不把 gradient accumulation=4 写成 64 个对比候选；
- 不把当前 36,495 样本称为已经完成 scale study；
- 不把500 optimizer step写成完整一轮或充分训练后的正式结果；
- 不因为 Web 在通用 dino.txt 头下绝对值更高，就把研究主线从 SAT 偷换为 Web；
- 不只以“相对 SAT 极低 step0 有提升”作为新增微调模块成功的证据，必须与 matched F0 比较；
- 任何时候都不解冻 `visual_model.backbone`，不允许它进入 optimizer 或 trainable-only checkpoint；
- 不把 dino.txt vision head 内的 transformer blocks 称为 SAT backbone blocks；
- 不在没有独立参数组和 LR 的情况下直接解冻 dino.txt 对齐头或 text 预训练模块；
- 不删除 step0、中间 stage、best、provenance、resume history 或失败报告；
- 不因 M4 短预算 Gate 通过就宣称充分训练的 RQ1、M5、RQ2、RQ3 或跨数据源全面泛化已经完成。

## 11. 路径与资产

### 11.1 环境

```text
本地：/Users/wangyue/Documents/ChatGPT/Dinov3txtforSAT
服务器：/root/autodl-tmp/Dinov3txtforSAT
虚拟环境：/root/autodl-tmp/Dinov3txtforSAT/.venv
```

服务器已验证：Python 3.12.3、PyTorch 2.7.1+cu128、CUDA 12.8、RTX 4090、BF16。

### 11.2 模型资产

| 文件 | SHA-256 |
| --- | --- |
| Web ViT-L/16 | `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035` |
| SAT ViT-L/16 | `eadcf0ffc02418b6c22a885ea1a7aaeeef84fbf0f5bb4d0b7d1d36e68a964f48` |
| dino.txt head/text encoder | `a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0` |
| BPE gzip | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |

`bpe_simple_vocab_16e6.txt.gz` 必须保持 gzip 压缩状态。

### 11.3 主要代码入口

| 任务 | 入口 |
| --- | --- |
| M4 SAT 短预算配置 | `configs/skyscript_sat_adapter_500step_seed11.toml` |
| M4 Web 对照配置 | `configs/skyscript_web_adapter_500step_seed11.toml` |
| SAT F1/F2 机制筛查 | `scripts/run_skyscript_m4_f1_f2_sat_seed11.sh` |
| SAT F1/F2 汇总 | `tools/summarize_skyscript_m4_f1_f2_sat.py` |
| 训练 | `src/dinotxt_rs/cli/train.py` |
| adapter 与冻结 | `src/dinotxt_rs/models/embedding_adapter.py`、`official_dinotxt.py` |
| loss 与 queue | `src/dinotxt_rs/losses/contrastive.py` |
| 训练产物核验 | `tools/verify_training_run.py` |
| step0 parity | `src/dinotxt_rs/cli/check_step0_parity.py` |
| SkyScript retrieval | `src/dinotxt_rs/cli/evaluate_skyscript.py` |
| RSICD retrieval | `src/dinotxt_rs/cli/evaluate_rsicd.py` |
| S1 编排/汇总 | `scripts/run_skyscript_gate_s1.sh`、`tools/summarize_skyscript_gate_s1.py` |
| S2 编排/汇总 | `scripts/run_skyscript_gate_s2.sh`、`tools/summarize_skyscript_gate_s2.py` |

代码检查：

```bash
ruff check .
pytest
python -m compileall -q src tools
```

## 12. 下一会话接手清单

1. 把 M4 定位为 adapter-only、0.877 epoch 的三 seed 短预算机制证据，不写成充分训练的最终实验。
2. 后续研究主线固定 SAT backbone；Web 只保留为通用 dino.txt 天然兼容参照。
3. 显式 optimizer parameter groups、独立 LR、视觉 backbone 永久冻结断言和完整
   provenance/checkpoint/resume 核验已经实现；不得新增 `vision_last_k`。
4. F1/F2 seed11 mechanism screen 已完成，四个候选均未同时超过 F0；不为它们补三 seed 或 Web
   对照。详细数值和结论见第2.7节。
5. 下一步以已归档 F0 为锚点，建立“adapter + text projection”的500-step seed11候选；只新增文本
   projection 这一变量，并加入冻结参考文本 embedding drift 评价。视觉 backbone 和 vision head 均冻结。
6. 微调范围确定后，才建立1,710-step（3 epochs）F0/胜出候选三 seed正式实验；是否延长到2,850
   step须提前冻结，不能根据单个 seed 的结果选择性续跑。
7. SAT 侧机制与正式结果完成后，冻结最终矩阵，再只替换 backbone domain/weights 和输出身份，
   一次性运行对应 Web matched controls；不得用 Web 结果反向调整 SAT 方案。

当前证据说明：SAT backbone 已包含有价值且不应被训练破坏的遥感视觉表征；在当前短预算下，最终
embedding adapter 比直接微调25.33M参数的 dino.txt vision head 更有效，二者联合也未超过
adapter-only。下一步不是放弃或微调 SAT backbone，而是在永久冻结 backbone 和 vision head 的前提
下，以 adapter 为锚点检验 text projection 和少量 text blocks 的可控微调，把“能涨点”的 M4 现象
推进为可解释、充分训练且可复现的 SAT 图文对齐方法。
