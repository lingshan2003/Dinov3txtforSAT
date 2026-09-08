# 最新交接：SkyScript Gate S0 通过与跨 seed 复现

更新时间：2026-09-08

当前状态：项目已经完成基础工程闭环，也完成了对旧 ChatEarthNet 路线的失败诊断。当前主线已切换为
**SkyScript unique-caption 数据 + 冻结官方 dino.txt + 小型 image embedding adapter**。首个 Web
100-step 实验已通过 Gate S0：4,055 条 held-out SkyScript validation 的全局 retrieval 明显提升，
RSICD-val mean recall 得到保持。当前应进入 Gate S1 跨 seed 复现，不直接延长到 500 step。

这份文档从当前状态重新编写。ChatEarthNet、历史 500-step pilot 和后续 A/B/C 诊断只作为历史背景
总结；它们不再代表当前数据协议、当前模型更新范围或下一轮训练入口。

本文是当前进度和操作协议的唯一主文档。研究动机见[科研背景](../DINOv3_Remote_Sensing_Domain_Text_Alignment_Research_Plan.md)，稳定开发规则见[开发架构](DEVELOPMENT_ARCHITECTURE.md)。

## 1. 已完成、当前状态与证据来源

| 范围 | 已有工作 | 当前接手点 |
| --- | --- | --- |
| 基础工程 | 模型加载、训练、validation/best、严格 resume、下游评测和审计工具 | adapter parity 与一一配对 retrieval 已完成服务器实测 |
| 历史路线 | ChatEarthNet pilot、外部退化与 A/B/C、A/C 诊断 | 已停止作为当前主训练协议，保留失败分析与原始产物 |
| 当前数据 | SkyScript images2+images3 唯一 caption 选择、抽取、36,495/4,055 拆分和 manifest | global77 train/validation 精确文件和解码像素 overlap 均为 0 |
| 当前方法 | adapter-only 100-step seed 11 | SkyScript mean recall 提升且 RSICD-val 平均保持，Gate S0 已通过 |
| 下一步 | Gate S1 | seed 23/47 不可变配置、执行脚本和三 seed 汇总工具已准备，待提交后在服务器运行 |

**本次文档核验范围**：训练方法 commit 为 `4c564d25f3946531055e26dea6890fb892fed2fc`；
Gate S0 评测与编排代码已进入 `9202423`。本次读取了用户从服务器下载的
`outputs/skyscript_gate_s0_seed11/summary.json` 副本，报告状态为 `pass` 且六项检查全部为 true。
本次未直接登录服务器读取各子报告；服务器上的 `preflight.txt`、parity、overlap、逐 checkpoint
retrieval 报告和训练原始产物仍是完整证据来源，下载的 summary 是本次结果复核依据。

“当前工作”指跨 seed 复现准备，不表示服务器此刻正在训练。阅读顺序：第 4–6 节了解候选与 S0
证据，第 8 节接手 S1；需要重建环境或数据时再看第 12 节。

## 2. 里程碑与当前定位

架构文档定义抽象阶段；以下是本项目当前沿用的具体科研协议，样本规模统一在本节维护：

1. `M0 assets`；
2. `M1 inference`；
3. `M2 data`；
4. `M3 MVP`：Web 10k 训练稳定、retrieval 指标可计算；
5. `M4 RQ1`：SAT 10k 严格对照；
6. `M5 scale`：50k/full；
7. `M6 RQ2`：自然、结构化、层级文本；
8. `M7 RQ3`：local alignment。

当前 SkyScript 实验使用 36,495 条训练数据，是一次方法候选验证，不符合 `M4 RQ1 = SAT 10k
严格对照` 的定义，也不能因为样本量接近 50k 就称为 M5。只有先完成泄漏、保持性、跨 seed 和更长
日程验证，才能冻结一个正式 Web/SAT 对照协议。

历史产物中带有 `m4`、`m5` 的目录名是早期误命名的实验身份。目录和 checkpoint 不应重命名；
在文字中一律称为“历史 A/B/C 诊断”和“历史 A/C 续跑诊断”。

## 3. 当前代码与运行环境

### 3.1 路径与版本

- GitHub 仓库：`lingshan2003/Dinov3txtforSAT`（私有）；
- AutoDL 项目：`/root/autodl-tmp/Dinov3txtforSAT`；
- 本地项目：`/Users/wangyue/Documents/ChatGPT/Dinov3txtforSAT`；
- 服务器虚拟环境：`/root/autodl-tmp/Dinov3txtforSAT/.venv`；
- 当前 SkyScript 实验项目 commit：`4c564d25f3946531055e26dea6890fb892fed2fc`；
- DINOv3 固定 commit：`6876159a11b4df116f30f667f8c9888617df0751`；
- 已验证服务器环境：Python 3.12.3、PyTorch `2.7.1+cu128`、CUDA 12.8、RTX 4090、BF16。

上述实验代码 commit（不是本次文档修订版本）引入了：

- SkyScript 定向 ZIP 抽取、unique-caption 选择和确定性拆分；
- SkyScript manifest 与 77-token 文本审计；
- 2048→256→2048 的图像 embedding adapter；
- `configs/skyscript_web_adapter_10step.toml`；
- `configs/skyscript_web_adapter_100step.toml`。

### 3.2 模型资产

| 文件 | SHA-256 |
| --- | --- |
| `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035` |
| `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | `eadcf0ffc02418b6c22a885ea1a7aaeeef84fbf0f5bb4d0b7d1d36e68a964f48` |
| `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth` | `a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0` |
| `bpe_simple_vocab_16e6.txt.gz` | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |

`bpe_simple_vocab_16e6.txt.gz` 必须保持 gzip 压缩状态。

## 4. 当前数据协议：SkyScript

### 4.1 为什么更换训练数据

旧 ChatEarthNet caption 较长、噪声和任务不匹配风险较高，第一句清洗还可能保留泛化描述而丢掉
后部主体语义。SkyScript polished top30 的 caption 更短、更接近可见地物或设施，并且提供了便于
去模板化的 `title_raw`。

更换数据后得到更好的曲线是重要证据，但不是单因素因果实验。当前方案同时改变了数据源、caption
分布、重复处理、可训练模块、queue 和增强，因此不能把改善全部归因于“数据质量”。

### 4.2 完整 CSV 的只读统计

`SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv`：

- 1,518,888 条记录，filepath 全部唯一；
- `title_raw` 有 52,923 个规范化后唯一文本；
- 约 96.52% 的记录与其他记录共享相同的精确语义文本；
- `title_raw` 空格分词长度 p50=4、p95=10、max=62；
- `title` 中 `It shows:` 后的正文规范化后 100% 等于 `title_raw`；
- `title` 的主要额外信息是视角模板：1,496,105 条使用 `An aerial image`，22,783 条使用
  `A satellite image`。

因此当前首轮使用 `title_raw`。带模板的 `title` 保留为未来 matched-image 文本消融，不与当前
基线混用。

### 4.3 images2+images3 快速路径

两个分片共有 370,317 条候选记录。以规范化 `title_raw` 为组、seed 11 确定性选择每组一张图，
得到 40,550 个唯一 caption-image pair：

| 分片 | 代表图片数 |
| --- | ---: |
| images2 | 20,391 |
| images3 | 20,159 |
| 合计 | 40,550 |

随后以 seed 23 对唯一 caption 做确定性 90/10 拆分：

| split | 样本数 | 规范化 caption 重叠 |
| --- | ---: | ---: |
| train | 36,495 | 0 |
| validation | 4,055 | 0 |

本地抽取时 40,550 张图片全部通过 ZIP CRC 和图片解码检查。上传到服务器的压缩包及拆分 CSV 身份为：

| 文件 | SHA-256 |
| --- | --- |
| `skyscript_images23_unique40550.zip` | `39368eb926edd3dd5e1b14a975b21c729d826b8c6081749dfe8b00433bf3ab3b` |
| `images23_unique36495_train.csv` | `7effc995620643a647b49a21f556b0baf7a799a500be234d918c6fd29dc512d2` |
| `images23_unique4055_val.csv` | `f2c27035cba000afce8620ed2ee419a224319c97d9e1ef176f0d4923cbd2bafe` |

服务器数据根目录：

```text
assets/data/raw/skyscript/images23_unique40550/
├── images2/
├── images3/
└── _metadata/selection.csv
```

### 4.4 当前训练 manifest

服务器重新生成 manifest，使 `image` 字段指向服务器绝对路径。当前 100-step run 的 provenance
记录为：

| split | manifest | 样本数 | SHA-256 |
| --- | --- | ---: | --- |
| train | `skyscript_images23_train_raw_unique36495_seed11_global77.jsonl` | 36,495 | `4264d884d564631816baf3e339e7ca12be2958966d99a60f6a75ae3d5d4dbaec` |
| validation | `skyscript_images23_val_raw_unique4055_seed23_global77.jsonl` | 4,055 | `062238716b8fddad890b4f97354588ad7524f1099667d3c0baa932002845b7f6` |

文本使用真实 dino.txt tokenizer，以 `complete_word_backoff_without_sentence_selection` 保持尽可能完整
的 `title_raw`，并确保不超过 77 tokens。这里没有复用 ChatEarthNet 的第一完整句清洗器。

### 4.5 已完成审计与剩余数据风险

- 服务器 global77 train/validation 已完成精确文件哈希和解码 RGB 像素哈希审计，两种 overlap
  count 均为 0；
- 精确哈希不能发现同一区域的不同裁剪、缩放或时间版本；
- 当前 CSV filepath 不直接提供适合分组拆分的经纬度，地理泄漏仍是限制；
- 每个精确 caption 只保留一张图会降低 false negative，但也改变了真实频率分布；
- validation 来自相同 SkyScript 过滤与生成流程，不是跨数据源泛化测试。

## 5. 当前方法：冻结官方模型，只训练 image adapter

### 5.1 Adapter 结构

当前 wrapper 在官方归一化 image embedding 后应用：

```text
image embedding (2048)
  → LayerNorm
  → Linear(2048, 256)
  → GELU
  → Linear(256, 2048), zero initialized
  → residual add
  → L2 normalize
```

可训练参数为 1,054,976。第二个 Linear 的权重和 bias 为零初始化，因此 step 0 的 image embedding
在理论上与官方输出相同。

冻结边界：

- Web DINOv3 backbone：冻结并保持 eval；
- dino.txt vision head：冻结并保持 eval；
- text encoder、text projection：冻结并保持 eval；
- logit scale：冻结，当前 validation 中始终为 100.0；
- image adapter：唯一可训练模块；
- patch tokens：不经过 adapter，因此当前实验只支持 global embedding alignment，不支持 local
  alignment 结论。

### 5.2 当前 100-step 配置

配置：`configs/skyscript_web_adapter_100step.toml`

| 项目 | 设置 |
| --- | --- |
| backbone | Web DINOv3 ViT-L/16 |
| image adapter bottleneck | 256 |
| trainable official layers | 无 |
| physical batch | 16 |
| gradient accumulation | 4 |
| effective samples / optimizer step | 64 |
| precision | BF16 |
| optimizer | AdamW |
| peak learning rate | `1e-4` |
| weight decay | 0.01 |
| max gradient norm | 1.0 |
| total steps | 100 |
| warmup | 10 steps |
| scheduler | linear warmup + cosine decay |
| random augmentation | 关闭 |
| negative queue | 关闭 |
| validation | step 0/25/50/75/100，全量 4,055 条 |
| validation loss group | 16 |
| validation forward batch | 64，仅用于加速前向 |

关闭增强和 queue 是当前基线的一部分，不是最终方案。这样可以先验证数据和小型 adapter 本身，避免
方向性 caption 被随机几何变换破坏，也避免同义 caption 或近重复地点被 queue 当作 false negative。

## 6. 当前实验结果

### 6.1 已完成 run

```text
outputs/skyscript_images23_top30raw_imageadapter256_36495_100step_seed11/
```

身份与完整性：

- steps=100，target_steps=100，completed=true；
- project commit=`4c564d25f3946531055e26dea6890fb892fed2fc`；
- DINOv3 commit=`6876159a11b4df116f30f667f8c9888617df0751`；
- 保存 step 0/25/50/75/100、`best.pt`、provenance、metrics、validation 和 summary；
- 所有 loss 与梯度有限；
- final queue size=0；
- 峰值 CUDA allocated memory=4,198,526,464 bytes，约 3.91 GiB；
- 本次 run 未发生 resume。

### 6.2 Validation 曲线

| step | validation loss | 相对 step 0 |
| ---: | ---: | ---: |
| 0 | 1.5994003 | 0.00% |
| 25 | 1.1477629 | -28.24% |
| 50 | 1.0803549 | -32.45% |
| 75 | 0.9872210 | -38.27% |
| 100 | **0.9726497** | **-39.19%** |

五个点严格单调下降，best step=100。step 75→100 仍下降 1.48%，但改善幅度已经明显缩小，可能
接近平台。

训练 loss：

- 第一个 10-step 窗口均值：1.5650553；
- 最后一个 10-step 窗口均值：0.9447836；
- 窗口均值下降 39.63%；
- 单步最小值 0.7713198，最后一步 1.0586457。训练 batch 随机变化，所以不要求逐步单调。

梯度范数前 10-step 均值为 29.9066，末 10-step 均值为 16.0512；记录值均有限，并在 optimizer
更新前按配置裁剪到最大范数 1.0。

### 6.3 对结果的解释

validation loss 使用固定的 16-sample InfoNCE group。随机匹配的参考值约为
`ln(16)=2.7726`。官方 step 0 已达到 1.5994，表明官方 dino.txt 对这些简洁 SkyScript 语义本来
就具有明显的图文对应能力；小型 adapter 进一步将其降至 0.9726。

这个结果强于固定 16 条 smoke：训练和 validation 的精确 caption 不重叠，下降发生在全部 4,055
条 held-out 样本上，因此不是简单重复记忆一个训练 batch。当前证据支持：

> SkyScript unique-caption + raw title + adapter-only 是值得继续核验的首个稳定候选。

### 6.4 Gate S0 结果

Gate S0 使用 `scripts/run_skyscript_gate_s0.sh` 在服务器完成，汇总报告为
`outputs/skyscript_gate_s0_seed11/summary.json`。训练产物、train/validation overlap、step-0
parity、train/RSICD-val overlap、SkyScript retrieval 改善和 RSICD-val 保持六项检查全部通过。

SkyScript 4,055 对一一配对全局 retrieval：

| step | mean recall | image→text median rank | text→image median rank |
| ---: | ---: | ---: | ---: |
| 0 | 0.0862721 | 181 | 118 |
| 25 | 0.1065351 | 94 | 83 |
| 50 | 0.1079326 | 89 | 75 |
| 75 | 0.1137279 | 72 | 68 |
| 100 | **0.1138512** | **70** | **67** |

step 100 相对 step 0 的 mean recall 绝对提升 0.0275791，即 2.76 个百分点、相对约 32%；双向
R@1/R@5/R@10 六项都高于 step 0。step 75→100 仅增加 0.0001233，说明当前日程末段已经接近
平台，但不能据此预测新的 500-step 日程。

RSICD-val 开发保持集：

| checkpoint | mean recall | 相对官方初始化 |
| --- | ---: | ---: |
| official / step 0 | 0.1587751 | 0.0000000 |
| step 25 | 0.1629494 | +0.0041743 |
| step 50 | 0.1603900 | +0.0016149 |
| step 75 | 0.1632541 | +0.0044790 |
| step 100 | **0.1609689** | **+0.0021938** |

step 100 没有发生 mean recall 遗忘，并比官方初始化高 0.22 个百分点。但方向并不一致：
image→text 的 R@1/R@5/R@10 均提高，text→image 的三项 recall 均低于官方初始化。因此当前证据
支持“外部平均检索能力得到保持”，不支持“RSICD 双向能力全面提升”。

Gate S0 后的结论升级为：

> seed 11 上，SkyScript unique-caption + raw title + adapter-only 不仅降低同源 validation loss，
> 也提高了全局 retrieval，并在预先规定的 RSICD-val mean recall 门槛下保持外部能力；该候选可以
> 进入跨 seed 复现。

### 6.5 当前不能得出的结论

- 不能把当前 loss 与历史 ChatEarthNet loss 直接比较；数据、caption、更新范围、queue、增强和
  scheduler 均不同；
- 不能断言改善完全来自 SkyScript 数据质量；当前不是正交消融；
- 不能把 RSICD-val mean recall 保持写成外部双向能力全面提升或跨数据源泛化已经成立；
- 不能断言结果跨 seed 稳定；当前完整证据只有 seed 11；
- 不能断言 100 step 后继续训练仍会改善；历史项目曾在更长训练中发生反转；
- 不能断言 Web backbone 优于 SAT；尚未做当前协议下的严格对照；
- 不能宣称 local alignment 改善，因为 adapter 不作用于 patch tokens。

## 7. 历史路线总结：ChatEarthNet（非当前协议）

### 7.1 历史工作完成了什么

旧路线建立了项目仍在使用的工程基础：

- 官方 Web/SAT DINOv3 与 dino.txt 加载；
- 对称 InfoNCE、queue、冻结策略和不同更新范围；
- 确定性 validation、固定 batch monitor、step-0 best 候选；
- checkpoint provenance、严格 resume、sampler/RNG/optimizer/scheduler 状态；
- EuroSAT 零样本分类、RSICD retrieval、step-0 parity；
- train/evaluation manifest 的文件和解码像素重叠审计。

因此旧路线不是“毫无价值的失败”。它证明了工程路径，并揭示了只看同域 loss 会误判模型质量。

### 7.2 历史数据与主要结果

ChatEarthNet 历史训练协议使用 10,000 条候选、移除占位图后实际 9,969 条训练样本，并使用较长
caption 的第一完整句与 word backoff。该数据和文本协议现已停止作为当前主训练集。

历史 Web/SAT 500-step pilot 在 ChatEarthNet validation 上都下降约 27.6%，但外部结果没有同步
改善：

| 历史同一 backbone 比较 | EuroSAT Top-1 | RSICD test mean recall |
| --- | ---: | ---: |
| Web 官方初始化 | 50.81% | 23.14% |
| Web 历史 best | 9.26% | 11.04% |
| SAT 官方初始化 | 7.09% | 0.46% |
| SAT 历史 best | 11.11% | 0.35% |

Web 历史 checkpoint 在 EuroSAT 上几乎坍缩为单一类别；SAT 的 11.11% 对应主要输出一个类别，
不能解释为有效提升。

随后历史 A/B/C 诊断比较了大范围更新与 vision-head-only、`5e-5` 与 `5e-6`。较低 LR 和冻结
文本塔只能延迟或减轻遗忘：A、C 在 step 100 短期改善，继续到 step 250 后 ChatEarthNet
validation 均高于初始化。C 的 RSICD-val 保持较好，但同域目标仍恶化；没有候选同时通过双门槛。

### 7.3 从历史路线保留的结论

1. 工程正确、loss 有限、checkpoint 可恢复，不代表训练方法正确。
2. 同域 validation 改善可能伴随严重外部遗忘。
3. 单 seed、100-step 的局部排名不能预测更长训练。
4. 冻结文本塔能减轻遗忘，但仅冻结文本塔不是充分条件。
5. 不能用已观察过的 RSICD test 或全量 EuroSAT 继续调参。
6. queue、augmentation、更新范围和数据质量必须分别消融，不能同时加入。

### 7.4 历史产物

以下目录仅用于追溯和复现，不作为当前训练续跑起点：

```text
outputs/gate_a_step0_parity/
outputs/m3_web_global77_formalschedule_500step_pilot_seed11/
outputs/m3_sat_global77_formalschedule_500step_pilot_fastval_seed11/
outputs/m3_downstream_500step_pilot_seed11/
outputs/m4_web_abc_100step_development_seed11/
outputs/m4_web_a_fullscope_lr5e6_100step_seed11/
outputs/m4_web_b_visionhead_lr5e5_100step_seed11/
outputs/m4_web_c_visionhead_lr5e6_100step_seed11/
outputs/m5_web_ac_250step_development_seed11/
```

不得重命名这些目录、修改历史 config/metrics/provenance，或从历史 checkpoint 接续当前 SkyScript
训练。

## 8. 下一步计划

### 8.1 Gate S0：已完成并通过

S0 的完整结果和解释见第 6.4 节。服务器证据至少包括：

```text
outputs/skyscript_images23_top30raw_imageadapter256_36495_100step_seed11/verification_report.json
outputs/skyscript_images23_top30raw_imageadapter256_36495_100step_seed11/skyscript_step0_parity.json
outputs/skyscript_s0_train_val_overlap.json
outputs/skyscript_gate_s0_seed11/preflight.txt
outputs/skyscript_gate_s0_seed11/train_vs_rsicd_val_overlap.json
outputs/skyscript_gate_s0_seed11/skyscript_val_step{000,025,050,075,100}.json
outputs/skyscript_gate_s0_seed11/rsicd_val_official.json
outputs/skyscript_gate_s0_seed11/rsicd_val_step{000,025,050,075,100}.json
outputs/skyscript_gate_s0_seed11/summary.json
```

这些报告与当前 100-step 训练目录都必须保留。S0 已结束，不需要重复运行来寻找更好的 checkpoint，
也不能在同一 RSICD-val 上继续调节门槛。

### 8.2 Gate S1：100-step 跨 seed 复现

S0 通过后，只改变 seed 和输出目录，以 seed 23、47 从官方初始化重复完全相同的 100-step 协议。
数据、caption、adapter 256、LR、warmup、queue 和 augmentation 都不改变。

每个 seed 都必须满足：

- step 100 validation 低于其自身 step 0，且 best 不退回 step 0；
- SkyScript retrieval 相对 step 0 改善；
- RSICD-val 保持性通过；
- 所有 loss、梯度、checkpoint 和 provenance 核验通过。

最终报告三个 seed 的均值、标准差和逐 seed 曲线。不能只汇报最好的一次。

S1 本地代码准备已经完成，但服务器实验尚未运行：

- `configs/skyscript_web_adapter_100step_seed23.toml` 和
  `configs/skyscript_web_adapter_100step_seed47.toml` 只改变实验名称、seed 和输出目录；manifest
  文件名中的 seed 11/23 仍表示数据选择与拆分身份；
- `scripts/run_skyscript_gate_s1.sh` 负责协议身份检查、seed 23/47 训练、逐 seed 完整 S0 等价门槛
  和最终三 seed 汇总；脚本可选择 `--seed 23`、`--seed 47` 或默认 `all`；
- `tools/summarize_skyscript_gate_s1.py` 核验三个 run 的实际 `config.toml`，只允许上述三个实验身份
  字段不同，并输出逐 seed 曲线以及 mean/sample standard deviation（ddof=1）；
- 若某个 seed 数值不通过，`all` 模式仍会完成另一个 seed 并生成 `fail` 汇总；身份、产物或协议错误
  则立即停止；
- 默认新训练从官方初始化开始。只有运行在 checkpoint 边界安全中断，且 metrics/validation 尾部与
  checkpoint 精确一致时，才允许显式 `--resume-seed23` 或 `--resume-seed47`；不能从 seed 11
  checkpoint 续跑。

提交并推送这些文件后，在服务器项目根目录建议进入 `tmux` 再执行：

```bash
git pull --ff-only
tmux new-session -s skyscript-s1 'bash scripts/run_skyscript_gate_s1.sh; exec bash'
```

脚本不能 `source`。完整通过后的主报告为 `outputs/skyscript_gate_s1/summary.json`；seed 23/47
逐 seed 报告分别在 `outputs/skyscript_gate_s1_seed23/` 和
`outputs/skyscript_gate_s1_seed47/`。在该汇总实际为 `pass` 前不得进入 S2。

### 8.3 Gate S2：新的 500-step 稳定性 pilot

S1 通过后创建一个新的、不可变的 500-step 配置。保持当前数据、`title_raw`、adapter、无增强和
无 queue。建议使用相同 peak LR `1e-4`、50-step warmup 和 cosine decay。

当前 100-step checkpoint **不能直接续跑**：当前 scheduler 的设计总长就是 100；把总长改成 500
会改变完整 LR 轨迹，也会违反严格 config identity。新 500-step run 必须从官方初始化开始。

500-step run 分段执行：

1. 0→100；
2. 100→250，严格 resume；
3. 250→500，严格 resume。

在 step 0/100/250/500 运行 SkyScript 全局 retrieval 和 RSICD-val；SkyScript loss 至少每 25 或
50 step 验证。进入下一段前要求当前 validation 低于 step 0、外部保持通过、best 与 resume 身份
完整。任一门槛失败立即停止，不以“继续训练可能恢复”为理由越过。

### 8.4 Gate S3：单因素消融

S2 通过后按顺序执行：

1. 在完全相同图片和 split 上比较 `title_raw` 与统一模板
   `An aerial image. It shows: ...`；
2. 比较 adapter 容量或 residual/embedding anchoring；
3. 单独测试随机裁剪，并先审查方向性与局部结构 caption；
4. 只有建立 multi-positive 或等价 false-negative 处理后，才考虑恢复 negative queue；
5. 不在同一个实验里同时解冻官方层、改变 caption、加入增强和 queue。

### 8.5 正式 M4 与 M5

S0–S2 通过后，按现有架构定义，从 SkyScript train 固定一个确定性 10k 子集，使用同一数据、文本、
adapter、优化和评测协议运行 Web/SAT 严格对照，才进入正式 M4。

如果决定把 M4 改为 36,495 条，必须先修订本文件第 2 节的具体协议并记录变更理由，不能静默
改变里程碑定义。架构仅维护抽象阶段目的。

M4 得到可复现且保持外部能力的结果后，再进入 M5，比较：

- 固定 SkyScript 10k；
- 当前 images2+images3 的 36,495 train；
- 扩展分片覆盖后的更大 unique-caption 训练子集：完整 polished CSV 统计上限为 52,923 个
  唯一文本，不代表扣除固定 validation、完成图像与地理审计后仍有同等训练数量；
- 更大重复样本规模只能在 multi-positive/false-negative 问题解决后考虑。

## 9. 当前冻结边界

- 不直接续跑当前 SkyScript step-100 checkpoint；
- 不把当前结果称为 M4、M5、跨 seed 稳定或 RSICD 双向能力全面提升；
- 不用 RSICD test 或全量 EuroSAT 循环调参；
- 不在 S1 通过前创建新的 500-step 长训练配置；
- 不恢复 4,096 negative queue；
- 不加入随机增强；
- 不解冻官方 backbone、vision head 或文本塔；
- 不删除原始 SkyScript CSV、images2/images3 大 ZIP 或旧数据集；它们仍用于复核和未来消融；
- 不因 `best.pt` 存在而删除 step 0、中间 step、final step 或 provenance。

## 10. 当前产物与保留要求

当前 SkyScript 输出必须保留：

```text
outputs/skyscript_images23_top30raw_imageadapter256_fixed16_10step_seed11/
outputs/skyscript_images23_top30raw_imageadapter256_36495_100step_seed11/
outputs/skyscript_gate_s0_seed11/
outputs/skyscript_s0_train_val_overlap.json
```

100-step run 至少保留：

```text
config.toml
provenance.json
metrics.jsonl
validation.jsonl
step_0000000.pt
step_0000025.pt
step_0000050.pt
step_0000075.pt
step_0000100.pt
best.pt
training_summary.json
smoke.log
train.log
verification_report.json
skyscript_step0_parity.json
```

`best.pt` 只是 step 100 的副本或链接语义，不能替代其他 checkpoint 的 optimizer、scheduler、
sampler 和 RNG 状态。任何清理前都应先生成只读盘点并完成异地备份。

## 11. 代码入口与更新要求

| 任务 | 入口 |
| --- | --- |
| 当前实验参数 | seed 11 基线 `configs/skyscript_web_adapter_100step.toml`；S1 复现 `configs/skyscript_web_adapter_100step_seed23.toml`、`configs/skyscript_web_adapter_100step_seed47.toml` |
| 定向抽取、确定性拆分 | `tools/extract_skyscript_subset.py`、`tools/split_skyscript_selection.py` |
| manifest 与文本预算 | `tools/prepare_skyscript.py`、`tools/prepare_global77_manifest.py`、`tools/audit_manifest_text.py` |
| 当前适配模块与冻结 | `src/dinotxt_rs/models/embedding_adapter.py`、`src/dinotxt_rs/models/official_dinotxt.py` |
| 产物与数据核验 | `tools/verify_training_run.py`、`tools/audit_manifest_image_overlap.py` |
| 初始化与下游评测 | `src/dinotxt_rs/evaluation/`、`src/dinotxt_rs/cli/evaluate_rsicd.py`、`src/dinotxt_rs/cli/evaluate_skyscript.py` |
| Gate S0 完整执行 | `scripts/run_skyscript_gate_s0.sh`（使用 `bash` 执行，不能 `source`） |
| Gate S1 训练、逐 seed 核验与汇总 | `scripts/run_skyscript_gate_s1.sh`（使用 `bash` 执行，不能 `source`） |
| Gate S1 三 seed 汇总 | `tools/summarize_skyscript_gate_s1.py` |
| 历史异常产物盘点 | `tools/inspect_training_artifacts.py` |

历史 `run_m4_*`、`run_m5_*` 和 ChatEarthNet pilot 脚本仅用于追溯，不能当作当前 SkyScript
入口。下一次更新本交接时，填写每个 Gate 的报告路径、代码/输入身份、结果与失败分支；未经验证的
计划保持待办状态。研究动机与架构规则不再复制到这里。

## 12. 操作附录

本附录合并原 `SKYSCRIPT_ADAPTER_PREPARATION.md` 的当前有效步骤。命令均在项目根目录运行。
已有数据和输出先核验复用；这些是环境或数据重建说明，不表示需要重新训练已完成的候选。
原始 CSV 和 ZIP 的路径必须替换为真实路径；不要用 `--allow-missing` 绕过缺图。

### 12.1 环境与模型资产

AutoDL 新环境使用：

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
bash scripts/bootstrap_autodl.sh
source .venv/bin/activate
bash scripts/download_assets.sh weights
```

现有环境直接激活并核对第 3 节版本与资产 hash；不要重复安装。bootstrap 面向 Python 3.12、
CUDA 12.8 的服务器环境，需核实驱动匹配。`download_assets.sh all/data-metadata/data-rgb`
仍属于 ChatEarthNet 下载路径，不负责获取 SkyScript。当前 SkyScript 原 CSV、大 ZIP 和已抽取包
按第 4 节身份移交；不能从 ZIP 文件名猜测其版本。tokenizer 词表保持 gzip 状态。

### 12.2 定向抽取与拆分

```bash
python tools/extract_skyscript_subset.py \
  --csv SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv \
  --archives /替换为真实路径/images2.zip /替换为真实路径/images3.zip \
  --output-root assets/data/raw/skyscript/images23_unique40550 \
  --selection-output assets/data/raw/skyscript/images23_unique40550.csv \
  --audit-output assets/data/raw/skyscript/images23_unique40550.extract.audit.json \
  --group-field title_raw \
  --selection-prefix images2 images3 \
  --include-prefix images2 images3 \
  --max-per-group 1 \
  --limit 40550 \
  --seed 11 \
  --bundle-output assets/data/raw/skyscript/skyscript_images23_unique40550.zip \
  --dry-run
```

确认 dry-run 的 `selected_images=40550`、`missing=0` 后删除 `--dry-run` 执行。脚本是在两包
并集上先去重，不是各自去重后拼接，所以共享的 caption 不会重复进入训练。

如果不能同时保留两个大 ZIP，仍使用相同的 `--selection-prefix images2 images3`，但第一次只传
`images2.zip` 并设置 `--include-prefix images2`；第二次只传 `images3.zip` 并设置
`--include-prefix images3`。两次使用同一个 output-root 和 selection-output，不同的 audit/bundle
文件名。第二次会验证并复用字节完全相同的 40,550 条全局选择清单。

完成两包抽取后，将 40,550 个唯一 caption 确定性拆为 36,495 train 和 4,055 validation：

```bash
python tools/split_skyscript_selection.py \
  --input assets/data/raw/skyscript/images23_unique40550.csv \
  --train-output assets/data/raw/skyscript/images23_unique36495_train.csv \
  --val-output assets/data/raw/skyscript/images23_unique4055_val.csv \
  --audit-output assets/data/raw/skyscript/images23_unique40550.split.audit.json \
  --group-field title_raw \
  --val-count 4055 \
  --seed 23
```

拆分工具要求输入 caption 已全局唯一；发现重复 caption 或 filepath 会失败。train/val 保持原 CSV
顺序，并在审计中记录输入、两个输出的 SHA-256。

其他模式只在未来扩展时使用：`--all-csv-rows` 按分片抽取 CSV 全部引用，默认 unique 模式
可设置全局选择上限。执行前看 `--help` 与 dry-run 报告，不再沿用旧文“首轮需要 50k”的安排。
抽取工具拒绝缺失或歧义成员、危险路径、冲突覆盖，所选文件经过 CRC 检查；原始大 ZIP 保留复核。

### 12.3 生成当前文本协议

快速路径已经生成拆分后的 CSV。主基线直接使用 `title_raw`，不再次抽样：

```bash
python tools/prepare_skyscript.py \
  --csv assets/data/raw/skyscript/images23_unique36495_train.csv \
  --images-root assets/data/raw/skyscript/images23_unique40550 \
  --split train \
  --caption-field title_raw \
  --caption-mode full \
  --output assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11.jsonl \
  --audit-output assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11.audit.json
```

对 `images23_unique4055_val.csv` 执行相同命令并把 `--split` 改成 `val`，输出
`skyscript_images23_val_raw_unique4055_seed23.jsonl`。

模板消融使用同一 selection CSV 和图片，只把 `caption-field` 改成 `title`、`caption-mode` 改成
`auto`，输出独立 manifest。此时 `auto` 会把 aerial/satellite 两种前缀统一为
`An aerial image. It shows: ...`，不引入人为的双前缀变量。

先用真实 dino.txt tokenizer 在保留完整语义的前提下做 complete-word backoff。即使所有 caption
已经能放入 77 tokens，这一步也会产生独立、可审计且内容不变的 global77 manifest。不要改用
旧的第一句截断器。

```bash
python tools/prepare_global77_manifest.py \
  --input assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11.jsonl \
  --output assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.jsonl \
  --audit-output assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.audit.json \
  --dinov3-repo external/dinov3 \
  --bpe-vocab assets/checkpoints/bpe_simple_vocab_16e6.txt.gz \
  --context-length 77 \
  --strategy complete-word-backoff
```

对快速路径的 validation manifest 执行相同转换并输出
`assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl`。随后运行只读
复核；任何 caption 仍超过 77 tokens 都会让命令返回非零：

```bash
python tools/audit_manifest_text.py \
  --manifest assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.jsonl \
  --dinov3-repo external/dinov3 \
  --bpe-vocab assets/checkpoints/bpe_simple_vocab_16e6.txt.gz \
  --output assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.text-audit.json \
  --require-fit
```

对 validation 同样运行 `audit_manifest_text.py --require-fit`。生成清单的 `image` 包含环境绝对
路径，因此本地 manifest hash 不应强求等于服务器 hash；原 run 的服务器清单身份以第 4 节为准。
文本 backoff 后仍需复核重复与 split 交集，清洗不能悄悄改变一一配对假设。

### 12.4 Gate S0 的产物核验与图像审计

以下在保有原始服务器产物的环境执行。核验器向标准输出写 JSON；仅成功时发布报告，避免覆盖
已有证据。首次保存前确认下列正式报告和临时文件都不存在；已有报告先读取核实。

推荐直接运行完整编排脚本：

```bash
bash scripts/run_skyscript_gate_s0.sh
```

不要使用 `source scripts/run_skyscript_gate_s0.sh`；脚本失败只应结束子进程，不应关闭当前终端。
脚本会校验并复用已有正式报告、保留失败日志，并在
`outputs/skyscript_gate_s0_seed11/summary.json` 汇总门槛结果。以下分步命令仅用于排查。

```bash
set -euo pipefail
run_dir=outputs/skyscript_images23_top30raw_imageadapter256_36495_100step_seed11
test ! -e "$run_dir/verification_report.json"
test ! -e "$run_dir/verification_report.json.part"
verification_tmp="$(mktemp outputs/.skyscript-verification.XXXXXX)"
python tools/verify_training_run.py \
  --output "$run_dir" \
  --expected-steps 100 --expected-target-steps 100 --require-completed \
  --expected-train-manifest-sha256 4264d884d564631816baf3e339e7ca12be2958966d99a60f6a75ae3d5d4dbaec \
  --expected-val-manifest-sha256 062238716b8fddad890b4f97354588ad7524f1099667d3c0baa932002845b7f6 \
  --expected-dinov3-commit 6876159a11b4df116f30f667f8c9888617df0751 \
  --expected-final-queue-size 0 \
  --required-checkpoint-step 0 --required-checkpoint-step 25 \
  --required-checkpoint-step 50 --required-checkpoint-step 75 --required-checkpoint-step 100 \
  --require-validation --validation-every 25 --require-best-checkpoint \
  --expected-validation-loss-batch-size 16 --expected-validation-forward-batch-size 64 \
  > "$verification_tmp"
mv "$verification_tmp" "$run_dir/verification_report.json"
```

临时报告必须放在训练目录外，否则核验器会把自己的 `*.part` 识别为未完成训练产物。如果运行失败，
先查看错误与临时报告，不能把部分输出登记为通过。

```bash
python tools/audit_manifest_image_overlap.py \
  --left-manifest assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.jsonl \
  --right-manifest assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl \
  --output outputs/skyscript_s0_train_val_overlap.json \
  --require-zero-overlap
```

审计输出若已存在则先读取，不覆盖。对训练集与 RSICD、EuroSAT 清单也应分别执行同类审计；
这些检查无法排除地理近重复。RSICD 开发保持评测入口为
`python -m dinotxt_rs.cli.evaluate_rsicd`，必须显式 `--split val`（默认是 test），
微调模型同时提供 `--checkpoint` 与 `--training-output`，配置使用该 run 保存的原始配置。

adapter parity 入口已经适配，使用固定的 16 条 SkyScript validation 输入运行：

```bash
python -m dinotxt_rs.cli.check_step0_parity \
  --config "$run_dir/config.toml" \
  --checkpoint "$run_dir/step_0000000.pt" \
  --training-output "$run_dir" \
  --input-manifest assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl \
  --batch-size 16 \
  --output "$run_dir/skyscript_step0_parity.json"
```

SkyScript 一一配对全局 retrieval 入口为 `python -m dinotxt_rs.cli.evaluate_skyscript`。对 step
0/25/50/75/100 分别传入对应的 `--checkpoint`、相同的 `--training-output "$run_dir"`、上面的
validation global77 manifest、`--split val` 和互不覆盖的 `--output`。输出明确记录 manifest
身份、候选数、正例定义、双向 R@1/R@5/R@10 与 mean recall。

当前 10-step/100-step 配置是已存在实验的复现身份，不能在原输出目录直接重跑。
从 100 改到 500 的新日程以及跨 seed 新实验按 Gate S1/S2 创建新配置；严格 resume 还会检查
项目 commit，文档提交后也不能放宽身份规则去恢复旧 run。
