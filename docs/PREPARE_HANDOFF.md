# DINOv3 Remote-Sensing Text Alignment：M3 后的方法反思阶段交接记录

更新时间：2026-09-07
当前阶段：M0（资产）、M1（推理）、M2（数据）和 M3（Web 10k MVP 工程闭环）已完成；项目暂停在 **M3 之后、正式 M4 RQ1 之前的方法诊断与反思阶段**。现有训练实现具备可复现、可恢复、可评测的工程闭环，但尚未得到一个能够同时改善 ChatEarthNet validation 并保持外部能力的稳定训练方法，因此不宣称进入后续正式里程碑。

本文取代此前版本的 `docs/PREPARE_HANDOFF.md`，集中保留已核验资产、数据协议、已完成实验、负向结果和当前认识边界。除明确标为“解释”或“未决问题”的内容外，所有数值均来自已保存的配置、输出报告或核验记录。本文不制定下一轮实验方案。

## 0. 阶段命名纠正

`docs/DEVELOPMENT_ARCHITECTURE.md` 第 12 节规定的唯一阶段顺序是：

1. `M0 assets`；
2. `M1 inference`；
3. `M2 data`；
4. `M3 MVP`：Web 10k 训练稳定、retrieval 指标可计算；
5. `M4 RQ1`：SAT 10k 严格对照；
6. `M5 scale`：50k/full；
7. `M6 RQ2`：自然/结构化/层级文本；
8. `M7 RQ3`：local alignment。

此前文档和产物中称为 “M4 Web A/B/C” 与 “M5 Web A/C” 的两轮工作并不符合上述含义。它们
都是历史 500-step 退化之后、M3 结项后的 Web 训练策略诊断，不能宣称完成了规范中的
`M4 RQ1` 或进入了 `M5 scale`。相关脚本、配置和输出目录已经参与实际运行，其名字作为不可变
实验标识保留；后文统一称为“100-step A/B/C 诊断”和“250-step A/C 续跑诊断”。规范中的
`M4`、`M5` 仍保持原定义。

## 1. 项目与运行环境

- GitHub 仓库：`lingshan2003/Dinov3txtforSAT`（私有）
- AutoDL 项目：`/root/autodl-tmp/Dinov3txtforSAT`
- 本地项目：`/Users/wangyue/Documents/ChatGPT/Dinov3txtforSAT`
- 服务器虚拟环境：`/root/autodl-tmp/Dinov3txtforSAT/.venv`
- DINOv3 固定源码：`/root/autodl-tmp/Dinov3txtforSAT/external/dinov3`
- DINOv3 固定 commit：`6876159a11b4df116f30f667f8c9888617df0751`
- 已验证服务器环境：Python 3.12.3、PyTorch `2.7.1+cu128`、CUDA runtime 12.8、RTX 4090、BF16 可用。

服务器重新连接后的起始命令：

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
source .venv/bin/activate
git pull
```

权重、原始数据、DINOv3 外部源码、`outputs/` 与 `.venv/` 均不进入 Git。不要以删除 checkpoint 节省空间的方式替代实验产物管理；详见第 9 节。

截至本文更新，代码中的关键提交为：

| 提交 | 内容 |
| --- | --- |
| `14916f8` | Web 500-step、正式 5,000-step 调度的受限 pilot 与含 step 0 的 best 选择 |
| `4569b14` | 缺失训练产物的只读恢复诊断 |
| `b2a68dd` | SAT 500-step pilot 与加速但等价的 validation forward |
| `1f58343` | 修复 validation 计时变量遮蔽；增加 EuroSAT/RSICD 下游评测闭环 |
| `a81e8e0` | Gate A step-0 parity 对应的交接状态 |
| `13b254e` | 历史误命名的 100-step Web A/B/C 诊断、RSICD-val 与泄漏审计闭环 |
| `cfd087a` | 历史误命名的 250-step A/C 续跑诊断与验收器 |

## 2. 已核验资产

### 2.1 模型与文本资源

下列资源位于服务器 `assets/checkpoints/`，文件身份通过 SHA-256 核验：

| 文件 | SHA-256 |
| --- | --- |
| `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035` |
| `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | `eadcf0ffc02418b6c22a885ea1a7aaeeef84fbf0f5bb4d0b7d1d36e68a964f48` |
| `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth` | `a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0` |
| `bpe_simple_vocab_16e6.txt.gz` | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |

`bpe_simple_vocab_16e6.txt.gz` 必须保持压缩状态。重新迁移或下载资产后，先运行：

```bash
python tools/verify_server_assets.py
```

### 2.2 数据集

| 数据集 | 已核验位置 | 已知规模 | 已用角色 |
| --- | --- | ---: | --- |
| ChatEarthNet | `assets/data/raw/chatearthnet` | 163,488 PNG | 图文对齐训练、同域 validation |
| EuroSAT | `assets/data/raw/eurosat` | 27,000 JPEG | 已观测的零样本诊断 |
| RSICD | `assets/data/raw/rsicd` | 10,921 JPG | 已观测的 test 检索诊断 |

RSICD 标注 `dataset_rsicd.json` 的 SHA-256 是：

```text
5e342037d469d074711676bdb9c02b6942a624530b1959d24d2734e68af9cede
```

### 2.3 ChatEarthNet 当前数据协议

训练不是“任意 ChatEarthNet 样本”，而是下列不可变协议：

1. 从 35-train 的 98,092 条记录以 seed 11 抽取 10,000 条候选。
2. 在 train/val/test 中都移除两张全黑/全白占位图；清洗后 split 间的 ID 和图像内容哈希交集均为零。
3. 文本使用 `first_complete_sentence_then_complete_word_backoff`，避免 dino.txt 77-token context length 的静默截断。
4. 不改变已有 manifest，不就地改写 caption；每个派生产物均有审计文件和 SHA-256。

有效输入如下：

| split | 文件 | 样本数 | SHA-256 |
| --- | --- | ---: | --- |
| train | `chatearthnet_35_train_10k_seed11_no_nodata_global77.jsonl` | 9,969 | `78abc613fbc8d98ea4617770473b30662d9eda31c0deb0dd06b51b1965d9fc0b` |
| val | `chatearthnet_35_val_no_nodata_global77.jsonl` | 16,277 | `1040ccf2ec07100ceb81ad665e28527d38b948cc9e1e547eb76e23a265c25f88` |
| test | `chatearthnet_35_test_no_nodata_global77.jsonl` | 48,860 | `51662bee9618051416bc7a7983d1fc658c2fe6ed825b86f9258205597ca77716` |

“10k”在任何论文或报告中都必须准确表述为：**10,000 条候选、9,969 条清洗后实际训练样本**。

## 3. 已完成的工程验证

以下项目已经被证实，不应被当前下游退化误解为“管线完全不能运行”：

- Web、SAT backbone 均可加载；dino.txt 图文前向输出有限值。
- 可训练参数为 Web 106,644,737 / 866,611,713（12.306%），SAT 106,644,737 / 866,613,761（12.306%）。
- 真实 ChatEarthNet batch 可读取为 `[16, 3, 224, 224]`，token 与前向正常。
- Web 的固定 16 样本、无 queue、无增强 10-step 验证 loss 从 3.70285 降到 1.61173，所有 loss、梯度均有限。这只验证计算图和局部优化能力，**不构成泛化性能结论**。
- checkpoint 含 config 快照、权重/manifest provenance、优化器、scheduler、RNG、sampler、queue 与 trainable state；resume 会校验运行身份。
- `best.pt` 现在在 step 0 与所有 validation step 中选择；因此它是“该次 ChatEarthNet validation 最优”，而不是泛化任务的最优模型。
- SAT 的 fast validation 使用 64 张前向后切回四个连续 16 张 loss group，故不改变已定义的 validation loss。旧 SAT pilot 的 `elapsed_seconds` 受变量遮蔽影响而失真；loss、checkpoint、resume 与模型权重不受影响。代码已修复并有回归测试。

## 4. 500-step 正式调度受限 pilot

两次 pilot 均使用未来正式实验的训练骨架：batch size 16、gradient accumulation 4、随机裁剪、shuffle、queue 4,096、最大学习率 `5e-5`、weight decay 0.01、warmup 250、设计总长度 5,000 optimizer steps。脚本仅运行到 500 step，因此它们是 pilot，不是正式 5,000-step 实验。

每 50 step 对固定的 16,277 条 ChatEarthNet val 做无增强、无 queue 的 16 样本 InfoNCE；step 0 也是合法 `best.pt` 候选。训练 loss 含 queue，不能与无 queue validation loss 直接等同。

### 4.1 Web pilot

- 输出：`outputs/m3_web_global77_formalschedule_500step_pilot_seed11`
- 同域 validation：step 0 为 3.8230855；最佳 step 300 为 2.7668279（相对下降 27.63%）；step 500 为 2.7683333。
- `best.pt` 指向 step 300，SHA-256 为 `62b15c393773a5a80078e3e5330a7f0858618718cde0632b1fe763f2131bb171`。
- 训练与固定监测均为有限值；固定 16 样本 monitor 从 3.7621241 降至 2.7710328。
- 该 run 曾真实从 step 250 恢复到 step 500；但用户在完成后删除 `step_0000250.pt`，所以恢复证据状态为 **degraded**。记录、step 0、step 500 与 best checkpoint 仍存在，不能据此伪造或宣称可重新验证 step-250 resume。

### 4.2 SAT pilot

- 输出：`outputs/m3_sat_global77_formalschedule_500step_pilot_fastval_seed11`
- 同域 validation：step 0 为 3.8234951；最佳且最终 step 500 为 2.7698112（相对下降约 27.56%）。
- `best.pt` 指向 step 500，SHA-256 为 `1b209963843b6e14fa1413377e09b52b7c2023e2d83ab2e82ddb38f40b3c630d`。
- 该 run 保留了要求的 checkpoint/resume 证据；恢复与资产身份已通过严格核验。

### 4.3 此阶段能与不能得出的结论

可以得出：当前训练设置在 ChatEarthNet validation 定义下改善了同域局部对比损失，且未发生 NaN、OOM 或明显的训练中断。

不能得出：它改善了通用遥感视觉语言能力、EuroSAT 零样本分类、RSICD 检索，或值得直接延长到 5,000 steps。step 500 时学习率仍约为 `4.97e-5`，接近峰值；这不是“已充分衰减、只需继续等待”的状态。

## 5. 已完成的下游闭环与结果

运行入口：

```bash
bash scripts/run_downstream_pilot_evaluation.sh
```

输出：`outputs/m3_downstream_500step_pilot_seed11`。综合核验 `verification_report.json` 的 `status` 为 `complete`。

评测是在预先固定的同一份数据、提示词与脚本上比较每个域的官方初始化和该域的 best checkpoint：

- EuroSAT：全量 27,000 图像，manifest SHA-256 `dad37240dd85a5fd3f25f8c40daa5d5c3f083a66e52338f4f4aeab345347c7a2`；固定两个零样本 prompt template。
- RSICD：官方 test split，1,093 图像、5,465 captions，manifest SHA-256 `4df6a9c8be0d3674ab5e7f9fa4c27ebdd352bf4d2f48746988ad535036187126`。
- fine-tuned 模型在加载前校验 config 文本、训练 provenance、输入文件哈希、checkpoint `run_identity` 与 trainable parameter 名称。外部数据未用于 `best.pt` 选择。

| 同一 backbone 域内的比较 | EuroSAT Top-1 | EuroSAT macro accuracy | RSICD mean recall |
| --- | ---: | ---: | ---: |
| Web 官方初始化 | 50.81% | 52.32% | 23.14% |
| Web step-300 best | 9.26% | 10.00% | 11.04% |
| SAT 官方初始化 | 7.09% | 7.11% | 0.46% |
| SAT step-500 best | 11.11% | 10.00% | 0.35% |

更细的信号：

- Web RSICD image-to-text 的 R@1/R@5/R@10 从 9.88% / 23.24% / 33.30% 降至 0.27% / 0.37% / 0.91%。text-to-image 同样下降，但幅度较小（R@10：37.47% → 34.42%）。
- Web EuroSAT 微调模型几乎总是输出 `PermanentCrop`：该类准确率为 100%，其他类接近 0。这是输出退化的强信号。
- SAT 微调模型几乎总是输出 `AnnualCrop`。11.11% Top-1 恰好等于 3,000 / 27,000，因此不是有意义的分类提升；其他类别准确率为 0。

这些绝对分数不是与外部论文直接可比的 SOTA 声明；它们首先是**在相同实现、相同数据与相同 backbone 内，初始化与微调 checkpoint 的受控差分**。在这个差分意义下，Web 发生了显著退化，SAT 也没有显示可靠改善。

## 6. M3 后、M4 前的训练策略诊断

### 6.1 Step-0 parity：排除保存、加载和评测路径错误

Gate A 已在 Web/SAT 上通过。它在固定输入上分别加载官方初始化与对应
`step_0000000.pt`，核对 checkpoint 身份与可训练参数指纹，并比较 token、图像/文本特征、
vision-head patch token、backbone patch token、logit scale、相似度矩阵和无 queue 对比损失。
两种 backbone 的全部最大绝对误差均为 `0.0`，综合状态为 `complete`。

这证明 step 0 checkpoint 与官方初始化在当前受检路径上完全一致。它不能证明训练方法正确，
但排除了“历史退化只是保存、加载或评测接错权重”这一解释。

### 6.2 开发保持集与泄漏审计

EuroSAT 全量和 RSICD test 已在历史下游诊断中被观察，不再用于选择学习率、更新范围或步数。
两轮后续诊断使用 RSICD 官方 val split：

- manifest：`assets/data/manifests/rsicd_val_retrieval_v1.jsonl`；
- SHA-256：`7e45a77872a488a487cea95657a24618f6008231305eee1b4fa1854b4e5cedc8`；
- 1,094 images / 5,470 captions；
- 官方 Web 初始化 mean recall：`0.15877513773739338`；
- 与 9,969 条 ChatEarthNet train 的文件 SHA-256 和解码 RGB 像素 SHA-256 精确交集均为零。

该审计不检测感知近重复，因此“零重叠”只限于字节完全一致或解码后像素完全一致。

### 6.3 100-step A/B/C 诊断（历史产物名含 `m4`）

共同设置为 Web backbone、seed 11、physical batch 16、gradient accumulation 4、queue 4,096、
设计总长 5,000 step、warmup 250，并由命令行限制在 step 100。训练使用 9,969 条
ChatEarthNet train；同域 validation 使用固定的 16,277 条 ChatEarthNet val。step 50 执行一次
严格 resume，保留 step 0/50/100、validation、fixed monitor、best 和恢复证据。

| 标签 | 可训练范围 | 最大 LR | 可训练参数 |
| --- | --- | ---: | ---: |
| A | vision head + text projection + text last-4 + logit scale | `5e-6` | 106,644,737 |
| B | vision head only | `5e-5` | 25,326,336 |
| C | vision head only | `5e-6` | 25,326,336 |

vision-head-only 时，整个 text tower、text projection、logit scale 和视觉 backbone 均冻结并保持
eval，只有 `visual_model.head` 处于 train。

| 标签 | ChatEarthNet val step 0 | step 50 | step 100 | RSICD-val step 50 | step 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 3.820279 | 3.728249 | 3.716718 | 0.158105 | 0.159476 |
| B | 3.820279 | 3.799968 | 4.652625 | 0.159263 | 0.155058 |
| C | 3.820279 | 3.792764 | 3.763962 | 0.159293 | 0.158227 |

预注册的 step-100 硬门槛只要求 RSICD-val mean recall 不低于官方初始化超过 `0.01` 绝对值，
因为 step 100 尚在 warmup 内，ChatEarthNet 是否改善当时只记录、不参与资格判断。因此 A、B、C
形式上全部通过，按 step-100 RSICD-val 排序为 A、C、B，A 被选为首位；鉴于 B 的同域 loss 已
明显反弹，实际只继续了 A 和 C。

这项规则忠实执行了预注册条件，但结果也暴露了它的局限：一个同域指标已经明显恶化的配置仍会
得到形式资格。step-100 时 A 的双指标领先也只是单 seed、warmup 中途的瞬时结果，不能支持
“更新文本塔优于冻结文本塔”的稳定结论。

综合输出位于 `outputs/m4_web_abc_100step_development_seed11/verification_report.json`。其中
`m4` 是历史误命名的实验标识，不代表规范中的 `M4 RQ1`。

### 6.4 250-step A/C 续跑诊断（历史产物名含 `m5`）

A、C 从各自经过验证的 `step_0000100.pt` 严格恢复到 warmup 终点 step 250；新增
step 150/200/250 checkpoint、ChatEarthNet validation 和 RSICD-val 评测。checkpoint 身份包含
产生 100-step run 的项目 commit `13b254e55910411bffc6be647022bcabc181b54d`。续跑没有放宽
身份校验，而是在临时 Git worktree 中执行该 commit 的训练代码，当前目录只复用原数据、权重和
输出。

step-250 的预注册资格要求同时满足：ChatEarthNet validation 严格低于 step 0；RSICD-val
mean recall 不低于官方初始化超过 `0.01` 绝对值（阈值 `0.14877513773739337`）。若有候选通过，
只有 step-250 两指标上的 Pareto 前沿唯一时才自动推荐；两组各有取舍时必须返回人工复核，不用
test 集打破平局。结果如下：

| 标签 | ChatEarthNet val step 0 | 100 | 150 | 200 | 250 | 同域通过 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 3.820279 | 3.716718 | 4.139433 | 5.615601 | 6.323068 | 否 |
| C | 3.820279 | 3.763962 | 3.816468 | 3.993479 | 4.285517 | 否 |

| 标签 | RSICD official | 100 | 150 | 200 | 250 | 保持性通过 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 0.158775 | 0.159476 | 0.157343 | 0.152559 | 0.136015 | 否 |
| C | 0.158775 | 0.158227 | 0.158836 | 0.158836 | 0.156886 | 是 |

A 到 step 250 的 ChatEarthNet loss 相对 step 0 增加 `2.502789`，RSICD-val 相对官方下降
`0.022761`，且从 step 100 起逐点下降。C 的 RSICD-val 只比官方低 `0.001889`，没有形成从
step 100 起的逐点下降，并通过保持性门槛，但 ChatEarthNet loss 相对 step 0 增加
`0.465238`。因此没有候选同时通过双门槛，
`eligible_for_500step=[]`、`recommended_for_500step=null`、决策状态为
`no_candidate_passed`。

综合输出位于 `outputs/m5_web_ac_250step_development_seed11/verification_report.json`。其中
`m5` 同样只是历史误命名的实验标识，不代表规范中的 `M5 scale`。

## 7. 当前证据、反思与未决问题

### 已由实验支持的事实

1. **工程路径正确不等于训练方法正确。** Step-0 parity、严格 resume、有限 loss/梯度和一致的
   输入身份均已通过，但训练后的能力仍然可以真实退化。
2. **历史 `5e-5` 配置不可接受。** 它虽降低 ChatEarthNet validation，却使 Web 的 EuroSAT 和
   RSICD test 严重退化；vision-head-only 的 100-step B 也出现同域反弹。
3. **把最大 LR 降至 `5e-6` 只延迟了退化，没有解决它。** A、C 在 step 100 暂时改善，继续到
   step 250 后同域 validation 均高于初始化。
4. **冻结文本塔显著减轻外部遗忘，但不是充分条件。** C 在 step 250 仍保持 RSICD-val，而 A
   明显下降；然而 C 的 ChatEarthNet validation 仍然恶化。
5. **100-step 的局部排名不能代表稳定策略。** 当时 A 同时领先 C，但延长到 250 step 后 A 的
   两项退化都远大于 C。单 seed、warmup 中途的微小差异不能支撑架构结论。
6. **ChatEarthNet validation 与外部能力必须同时观察。** 历史 500-step 结果证明同域 loss 改善
   可以伴随外部能力坍塌；250-step C 又证明外部保持也不等于同域目标改善。

### 只能视为解释、不能写成结论

当前 scheduler 在前 250 step 将 LR 线性升至峰值；`5e-6` 配置在 step 50/100/150/200/250
对应约 `1e-6/2e-6/3e-6/4e-6/5e-6`。A、C 都在升温前段改善、后段反转，这使学习率轨迹成为
重要嫌疑，但步数、累计更新和 LR 同时变化，现有实验没有将它们解耦，不能断言存在一个已知的
“安全 LR 阈值”。

同样尚未回答的问题包括：4,096 negative queue 与无 queue validation 的目标差异；9,969 条
单一数据源及 caption 分布是否足以支撑对齐；AdamW、weight decay、logit scale 和不同模块是否
需要不同优化强度；全局 InfoNCE 是否与研究目标匹配；以及外部指标的小幅变化有多少来自 seed
方差。现有 A/B/C 并不是完整的正交消融，不能从中把单一根因归到文本 encoder、视觉头或学习率。

当前应把“方法设计本身是否建立在可靠经验上”视为开放问题。继续增加自造配置之前，需要系统
回到相关领域已有工作的训练规模、冻结策略、参数分组、学习率、warmup、负样本构造、灾难性
遗忘控制和评测协议中寻找依据。本交接记录刻意不把这一反思提前写成下一步实验方案。

## 8. 当前冻结边界

- 不从任何现有 250/500-step checkpoint 继续到 5,000 step。
- 不宣称已经完成规范中的 `M4 RQ1` 或进入 `M5 scale`。
- 不把 ChatEarthNet validation 的短期下降表述为外部任务性能提升。
- 不用已观测的 RSICD test 或全量 EuroSAT 循环调参。
- 不因 `best.pt` 存在而删除 step 0、resume step、final step 或失败实验报告。
- 不更改已运行实验的 config、metrics、provenance、checkpoint 或历史输出目录名来修饰结果。
- 在完成方法层面的文献与经验审视之前，本文件不授权新的训练配置。

## 9. 产物保留、恢复与复现

每个 run 至少保留以下文件，直到完整核验报告和异地备份均确认：

```text
config.toml
provenance.json
metrics.jsonl
validation.jsonl
fixed_monitor.jsonl             # 若配置启用
resume_history.jsonl            # 若发生恢复
step_0000000.pt
每一个预定 resume checkpoint
最终 step checkpoint
best.pt
training_summary.json
verification_report.json
```

`best.pt` 只是被选中的某一步训练状态，不能恢复任意另一 step 的 optimizer、scheduler、queue、
sampler 或 RNG 状态。不得删除或覆盖下列诊断证据：

```text
outputs/gate_a_step0_parity/
outputs/m4_web_abc_100step_development_seed11/
outputs/m4_web_a_fullscope_lr5e6_100step_seed11/
outputs/m4_web_b_visionhead_lr5e5_100step_seed11/
outputs/m4_web_c_visionhead_lr5e6_100step_seed11/
outputs/m5_web_ac_250step_development_seed11/
outputs/m3_downstream_500step_pilot_seed11/
```

其中含 `m4`/`m5` 的路径是已经落盘的历史实验身份，只应在叙述中纠正含义，不应通过改目录或
改 checkpoint 内容追溯性“修正”。若发现产物缺失，先用 `tools/inspect_training_artifacts.py`
只读盘点，不伪造恢复证据。

## 10. 相关代码与文档

- `docs/DEVELOPMENT_ARCHITECTURE.md`：阶段命名、训练、checkpoint、验证和评测的约束；
- `scripts/run_step0_parity_checks.sh`：已完成的 step-0 parity 入口；
- `scripts/run_web_500step_formal_schedule_pilot.sh`：历史 Web 受限 pilot；
- `scripts/run_sat_500step_formal_schedule_pilot.sh`：历史 SAT 受限 pilot；
- `scripts/run_downstream_pilot_evaluation.sh`：已完成的下游闭环入口；
- `scripts/run_m4_web_abc_100step_development.sh`：历史误命名的 100-step 诊断复现入口；
- `scripts/run_m5_web_ac_250step_development.sh`：历史误命名的 250-step 续跑复现入口；
- `src/dinotxt_rs/evaluation/`：EuroSAT、RSICD 与 parity 的确定性评测实现；
- `tools/inspect_training_artifacts.py`：缺失产物的只读诊断。

本项目当前停留在 M3 后、M4 前的方法反思阶段。本文只记录已经发生的事实、失败模式与认识
边界，不给出下一步配置或实验排期。
