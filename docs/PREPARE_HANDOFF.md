# 最新交接：M4-A 单种子试验完成，进入 M4-B 跨种子对照

更新时间：2026-09-10

## 1. 一页结论

项目当前已经得到一个可信的稳定基线：

> SkyScript images2+images3 unique-caption 数据、`title_raw` 文本、冻结官方 dino.txt、仅训练
> 2048→256→2048 image embedding residual adapter，在 Web DINOv3 ViT-L/16 上能够跨
> seed 稳定改善同源全局图文检索，并在 500-step 日程中持续改善而不发生历史路线的长训练反转。

Gate S0、S1、S2 和 M4-A 均已通过。下一步是 **M4-B：Web 与 SAT 的 seed11/23/47
严格跨种子对照**，先完成全部新增 run 的 stage100，不能提前只延长其中一部分。

本轮同时冻结以下研究决策：

1. 当前正式训练数据使用完整的 36,495 条 train，不再为了沿用旧的“10k / 50k”命名而人为切出
   10k 主实验。
2. 当前不单列紧接着必须完成的 scale 阶段。规模研究推迟到扩展更多分片、恢复多图同语义关系，
   或建立 multi-positive/false-negative 处理后再做。
3. 原计划中的统一文本模板实验不再是进入 M4 的前置 Gate；它保留为 RQ2 可选实验。
4. unique-caption 不只服务 negative queue。它已经减少当前 in-batch InfoNCE 的精确同文假负例，
   同时让未来 queue 更安全；但它并没有自动解决同义文本、近重复地点和 queue embedding 过时。
5. queue 与“扩大可训练参数范围”是两个独立实验轴，不能在同一个 run 中同时打开。
6. 其他官方模块的微调按 image side → text projection → text blocks 的顺序逐步增加，并先实现分组
   learning rate；不能把当前 adapter 的 `1e-4` 直接施加到全部预训练模块。

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

### 2.5 M4-A：Web–SAT matched seed11 pilot 通过

正式对照报告：

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

本次读取的是用户从服务器下载的 S2 `summary.json` 副本。没有直接登录服务器逐个读取子报告；
服务器中的训练目录、stage summary、preflight、parity、overlap、逐 checkpoint retrieval 和 resume
history 是完整证据来源。

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
36,495 条 train 的 87.7%，还不到完整一轮数据。此时把 train 人为缩为 10k 会带来两种混杂：

- 固定 steps 时，10k 数据会被重复看更多轮；
- 固定 epochs 时，优化步数和 scheduler 预算又不同。

images2+images3 只有 40,550 个唯一 caption-image pair；完整 polished CSV 的规范化唯一文本统计上限
约 52,923。当前可形成的纯 unique-caption 规模跨度不足以支撑有解释力的 scale law。为了回答 RQ1，
直接在已经验证成功的完整 36,495 train 上控制 Web/SAT 变量，比退回任意 10k 更合理。

因此具体里程碑调整为：

- M4 使用固定 36,495/4,055 协议完成视觉域对照；
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
| seed | 11（S2）；跨 seed 100-step 证据为 11/23/47 |
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
| warmup | 50 |
| scheduler | linear warmup + cosine decay |
| validation/checkpoint | 每 50 step |
| augmentation | 关闭 |
| negative queue | 关闭 |

梯度累积不会把对比候选从 16 扩为 64；它只扩大一次 optimizer update 消费的样本数。

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

## 7. 当前步骤：M4 / RQ1 Web–SAT 严格对照

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

状态：即将执行。seed23/47 的 Web/SAT 配置与通用 staged runner 已准备。默认使用一键完整编排：
它先运行四个新增 run 到 stage100 并与 seed11 统一汇总，全部通过后自动以同样方式进入
stage250、stage500，不需要人工分三次续跑，也不会越过失败的全矩阵 Gate。

M4-A 是 matched-seed pilot，不足以单独完成视觉域结论。预先规定下一层复现：

1. 为 Web 500-step schedule 和 SAT 500-step schedule 分别创建 seed23/47 配置；
2. 先让两域所有 seed 运行到 stage100，形成真正相同 warmup50/total500 scheduler 的三 seed 对照；
   不能用 S1 的 warmup10/total100 run 冒充这一对照；
3. 两域的某个 seed 若在 stage100 失败，保留并汇报，不能只延长通过的 seed 后报告最好结果；
4. 两域三个 seed 均通过 stage100 后，再按相同规则扩展到 250/500；
5. 最终按 domain 报告逐 seed、mean、sample std 和 Web–SAT difference，不只报告最好 checkpoint。

只有 M4-B 给出匹配的跨 seed 证据后，才能对 RQ1 做稳定性结论。资源不足时，可以把 M4-A 写成
pilot，并明确 M4-B 未完成，而不能把单 seed 写成正式域优劣。

### 7.4 下一会话的具体代码任务

下一会话从这里开始，不先做 queue、模板或额外解冻：

1. 新建 SAT seed11 500-step 不可变配置；
2. 增加配置差异测试，确保除实验身份和 SAT backbone 两项外完全一致；
3. 抽取或复用 S2 staged runner 的公共核验逻辑，避免复制后出现 Web/SAT 门槛漂移；历史 S2 脚本和
   已有产物身份不能被改写；
4. 实现 M4-A 汇总工具，读取 Web S2 和 SAT staged reports，输出绝对值、within-domain delta 和
   between-domain difference；
5. 先只运行 SAT seed11 stage100；通过后再到 250/500；
6. M4-A 完成后再生成 M4-B seed23/47 配置与编排，不提前启动其他实验轴。

建议输出身份：

```text
configs/skyscript_sat_adapter_500step_seed11.toml
outputs/skyscript_images23_top30raw_sat_imageadapter256_36495_500step_seed11/
outputs/skyscript_gate_m4_sat_seed11/
outputs/skyscript_m4_rq1_seed11/summary.json
```

## 8. M4 之后：逐步扩大可训练范围

当前代码已经支持 `train_vision_head`、`train_text_projection`、`text_last_k` 和
`train_logit_scale`，但 optimizer 只有一个全局 learning rate。直接打开这些开关会把 adapter 的
`1e-4` 同时施加到预训练官方模块，无法区分“模块无效”和“LR 过大”。

因此先增加显式 parameter groups，并把每组 LR、参数名、数量写入 provenance/checkpoint 核验。
然后在选定 backbone 上按以下 ladder 单变量推进：

| 阶段 | 相对上一步唯一新增项 | 目的 |
| --- | --- | --- |
| E0 | adapter only（当前 baseline） | 最小稳定参照 |
| E1 | + vision head，使用较低独立 LR | 增加 image-side 对齐能力 |
| E2 | + text projection，使用较低独立 LR | 允许最终文本映射适配 |
| E3 | + 最后 1 个 text block 与 final norm | 测试轻量语言域适配 |
| E4 | last 2/4 text blocks，仅在 E3 通过后 | 测试更大语言容量 |
| E5 | logit scale 单独解冻 | 研究校准，不让它掩盖表示质量 |

规则：

- 每个候选都从官方初始化开始，不从前一个 ladder checkpoint 接着训练；
- 先 seed11 stage100 筛查，再逐段到 250/500；最终候选再做 seed23/47；
- 每次只增加一个可训练组件，queue 和 augmentation 保持关闭；
- image backbone blocks 当前没有细粒度解冻配置，不在没有明确 last-k、LR group 和保存测试前解冻；
- 若某一步外部保持失败，优先研究表示 anchoring/distillation，而不是继续增加层数赌恢复。

这个 ladder 回答“额外容量是否带来可复现收益”，不能与 Web/SAT 对照或 queue 实验混在同一个表里
当作单因素结论。

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
- 不删除 step0、中间 stage、best、provenance、resume history 或失败报告；
- 不因 S2 通过就宣称 M4、M5、RQ2、RQ3 或跨数据源全面泛化已经完成。

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
| 当前 Web S2 配置 | `configs/skyscript_web_adapter_500step_seed11.toml` |
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

1. 不重启 seed11，也不混入 queue、额外解冻或其他实验轴。
2. 在服务器运行 `scripts/run_skyscript_m4_b_full.sh`；脚本依次完成全矩阵 stage100→250→500。
3. 单个 run 数值 Gate 失败时，脚本仍完成同 stage 的其余预注册 run 并保留证据，但不会让任何 run
   进入下一 stage，避免选择性延长和选择性报告。
4. 每个 stage 都生成独立的跨种子报告；最终检查
   `outputs/skyscript_m4_rq1_m4_b_stage500/summary.json` 中的逐 seed、mean、sample std 和配对
   SAT−Web difference。
5. M4-B step500 报告生成后，把正式跨种子结论和证据 hash 写回本文，再决定后续实验轴。

本轮目标不是让 SAT 必须获胜，而是获得一个身份严格、可解释的视觉域对照。SAT 失败、只在同域改善
或外部保持不足，都可能是 RQ1 的有效答案。
