# DINOv3 Remote-Sensing Text Alignment：截至 M3 的交接记录

更新时间：2026-09-04
当前阶段：M0（资产与环境）、M1（模型加载）、M2（数据协议）和 M3（受限训练、恢复、下游评测闭环）已完成。**当前 500-step 配置不具备启动 5,000-step 正式训练的资格。** 这不是项目失败，而是一个有效的负向实验结果：它证明当前训练目标、可训练范围和超参数组合虽能降低 ChatEarthNet 验证损失，却损害了已观测的外部零样本与检索能力。

本文取代此前版本的 `docs/PREPARE_HANDOFF.md`，并保留已核验资产、数据协议、已完成实验、已知局限与下一阶段门槛。除明确标为“假设”的内容外，所有数值均来自已保存的配置、输出报告或核验记录。

## 1. 项目与运行环境

- GitHub 仓库：`lingshan2003/Dinov3txtforSAT`（私有）
- AutoDL 项目：`/root/autodl-tmp/Dinov3txtforSAT`
- 本地项目：`/Users/wangyue/Documents/ChatGPT/Dinov3txtforSAT`
- 服务器虚拟环境：`/root/autodl-tmp/Dinov3txtforSAT/.venv`（由 `uv` 管理）
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

## 6. 当前科学判断与待验证假设

### 已证实的事实

1. 当前策略可降低 ChatEarthNet 同域 validation loss。
2. 当前 best checkpoint 在已观测的 EuroSAT/RSICD 上没有展示可接受的泛化改善，Web 则明显恶化。
3. 内部 validation loss 不能单独作为正式训练的放行条件。
4. 目前的 500-step 结果不足以支持“继续到 5,000 steps 后自然恢复”的假设。

### 合理但尚未证明的解释

以下是要通过下一轮消融检验的假设，不应在论文中提前写成结论：

1. **训练数据与目标分布过窄。** 当前只有 9,969 条 ChatEarthNet `global77` caption，模型可能拟合了该数据的语言和图像统计，未保持更广泛的遥感语义几何。
2. **更新范围过大。** 一次更新约 1.066 亿参数，包括视觉对齐头、文本 projection、文本 Transformer 最后四层和 logit scale；对该数据规模而言，`5e-5` 可能过激。
3. **文本 encoder 更新是风险来源之一。** 这是重要假设，而非已证实的单一根因。必须与“只训练视觉头”“只训练 projection”等消融比较。
4. **优化/选择指标不充分。** 训练使用 4,096 negative queue，而 validation 使用连续 16 对的无 queue InfoNCE；后者可改善局部匹配，却未约束更大候选集合上的检索或零样本分类。
5. **checkpoint 加载路径尚需端到端 parity 证明。** 当前严格身份校验强烈降低了加载错权重的可能性，但仍未执行“官方初始化与 `step_0000000.pt` 在同一输入上数值一致”的专门实验。因此不能把代码风险宣称为零。

## 7. 下一步工作：按门槛推进

### Gate A：先排除 checkpoint/evaluation 路径问题（不训练）

下一项代码工作应实现并运行 **step-0 parity check**。它必须：

1. 用同一 config 分别加载官方初始化与该 run 的 `step_0000000.pt`。
2. 在固定、hash 记录的图像/文本输入上比较 image feature、text feature、logit scale 和最终指标。
3. 报告最大绝对差、相对差、输入/配置/checkpoint SHA-256；以 BF16/浮点容差而非文字“看起来一致”判定通过。
4. 只有 parity 通过，才能把第 5 节退化主要归因于训练策略；若不通过，停止新的训练，优先修复 checkpoint 保存/加载或评测路径。

此检查应先对 Web step 0 运行，再对 SAT step 0 运行。它不是为了证明模型表现好，而是为了把“训练真的改变了模型”与“加载实现出错”严格区分开。

### Gate B：建立不污染最终报告的开发协议

EuroSAT 全量和 RSICD test 的汇总数值现在已经被观察到；从现在起，**不得用它们反复挑选学习率、可训练层或训练步数**，否则它们会变成调参集。

在新训练前先写明并固定开发协议：

- 优先使用此前未参与本项目决策的开发数据（例如 RSICD 的非 test split，前提是先核验官方 split 和泄漏情况）。
- 若需要分类保持性指标，应使用一个与最终报告集明确分离的开发集；不要把已观测的全量 EuroSAT 再包装成“未见 test”。
- ChatEarthNet val 继续只用于同域 checkpoint 选择；ChatEarthNet test、RSICD test 与已观测 EuroSAT 结果在最终报告中如实标注为已观测评估证据，而非新的调参依据。
- 预先写下候选配置、选择指标、停止条件和最终一次性评测规则。

### Gate C：从最小的可解释消融开始

在 Gate A、B 完成之前，不运行 5,000-step 正式训练。随后以 Web 为优先（其初始化下游基线有较强可用信号），按一次只改变一个因素的顺序进行受限 pilot：

1. **视觉头-only、低学习率**：冻结文本 Transformer、文本 projection 与 logit scale，仅训练视觉对齐头；最大学习率先降低一个数量级（候选 `5e-6`），保持数据与可复现/恢复协议。
2. 若第一项未伤害预先指定的开发指标，再加入文本 projection；仍冻结文本 Transformer。
3. 只有前两项通过，才逐步解冻文本末层（例如先 1 层，再到当前的 4 层），并将每个配置单独报告。
4. 每个 pilot 保持 step 0 候选、全量同域 validation、resume、checkpoint 身份核验和参数数目记录；不得只保留 `best.pt`。

每次 pilot 的放行条件必须同时包括：数值/恢复完整、同域 validation 不恶化、预先指定的开发保持性指标不出现分类单类塌缩或检索显著退化。达到这些条件后，才考虑 5,000-step 的单一预注册配置。

### Gate D：正式实验与论文报告

只有 Gate A–C 均通过后，正式实验才按固定协议覆盖清洗后的 9,969、嵌套 50k、全量 ChatEarthNet 规模，并以固定 seed / 必要时多 seed 报告均值与方差。最终报告应同时呈现：

- 初始化基线、所有决定性超参数、可训练参数范围、数据规模、文本截断策略；
- ChatEarthNet 同域曲线与 checkpoint 选择规则；
- 预先冻结的外部评测指标、每个类别和检索方向；
- 正向结果、无明显变化以及负向结果；
- 对适用范围、数据单一性、caption 语义、灾难性遗忘和评测局限的讨论。

若经过公平的 parity、消融和正式实验后仍未优于初始化，研究报告应明确写出这一事实及证据。一个可复现、控制变量充分的负向结果仍是毕业论文中有价值的实验结论；不得只保留“最好看”的曲线或删除不利 checkpoint。

## 8. 当前禁止事项

- 不要从现有 Web/SAT 500-step `best.pt` 继续到 5,000 step。
- 不要把 ChatEarthNet validation loss 的下降表述为外部任务性能提升。
- 不要用已观测的 RSICD test 或全量 EuroSAT 结果循环调参。
- 不要因 `best.pt` 存在而删除 step 0、resume step、final step 或验证报告。
- 不要更改已运行实验输出中的 config、metrics、provenance 或 checkpoint 来“修复历史”；修复应进入新的不可变 run。

## 9. 产物保留、恢复与复现

每个 run 至少保留以下文件，直到该 run 的完整核验报告和数据备份都已确认：

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

`best.pt` 只是被选中的某一步训练状态，不能恢复任意另一 step 的优化器、queue、sampler 或 RNG 状态。若发现产物缺失，先用只读工具盘点，绝不伪造恢复：

```bash
OUTPUT=outputs/m3_web_global77_formalschedule_500step_pilot_seed11
python tools/inspect_training_artifacts.py \
  --output "$OUTPUT" \
  --expected-checkpoint-step 0 \
  --expected-checkpoint-step 250 \
  --expected-checkpoint-step 500 \
  --report "$OUTPUT/recovery_report.json"
```

下游结果已下载到本地的：

```text
outputs/m3_downstream_500step_pilot_seed11/
```

其中四份任务报告与 `verification_report.json` 都应和论文实验记录一同备份。

## 10. 相关代码与文档

- `docs/DEVELOPMENT_ARCHITECTURE.md`：训练、checkpoint、验证和评测的目标架构。
- `scripts/run_web_500step_formal_schedule_pilot.sh`：历史 Web 受限 pilot。
- `scripts/run_sat_500step_formal_schedule_pilot.sh`：历史 SAT 受限 pilot。
- `scripts/run_downstream_pilot_evaluation.sh`：已完成的下游闭环入口。
- `src/dinotxt_rs/evaluation/`：EuroSAT 与 RSICD 的确定性评测实现。
- `tools/inspect_training_artifacts.py`：缺失产物的只读诊断。

下一位执行者应从 **Gate A 的 step-0 parity check** 开始，而不是启动任何长训练。
