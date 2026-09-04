# DINOtxt-RS 开发架构与实验纪律

**状态**：v0.1，适用于 Model A / Minimum Viable Thesis。  
**原则**：任何代码变化都必须能回答“它改变了哪一个研究变量”，任何实验结果都必须能回答“它能否被同一配置和同一数据清单复现”。

## 1. 系统边界

首版系统只实现以下闭环：

```text
ChatEarthNet RGB + caption
        │
        ├── canonical JSONL manifest
        │
        ▼
frozen DINOv3 ViT-L/16 (Web 或 SAT)
        + trainable official dino.txt vision head
        + trainable text projection
        + optional last-K text blocks
        │
        ▼
symmetric image-text contrastive loss
        │
        ▼
retrieval / zero-shot / patch-text diagnostic
```

以下内容不进入 MVP 主分支：DINOv3 backbone 全量微调、7B backbone、多光谱输入、caption generation、pixel decoder、LoRA 与 last-K 同时启用、局部监督损失。它们只能在基线冻结并产生可复现结果后，以独立配置和独立消融加入。

## 2. 依赖方向

代码必须遵守单向依赖：

```text
cli → training → {models, data, losses, config}
tools → canonical data contract
models/data/losses 不依赖 cli 或 training
```

规则：

1. `cli/` 只解析参数和组装对象，不实现算法。
2. `models/` 不读取实验 TOML、不创建 DataLoader、不写 checkpoint。
3. `data/` 不知道模型结构，只输出 pixels、caption、sample id。
4. `losses/` 只接受 tensor，不读取全局状态和文件。
5. `training/` 是唯一允许同时依赖 model、data 和 loss 的层。
6. `tools/` 产生的最终结果必须符合 canonical manifest，训练代码不得为每个原始数据集增加 if/else。

## 3. 第三方代码规则

DINOv3 作为外部依赖放在 `external/dinov3`，不得复制其源码到本项目后静默修改。固定提交为：

```text
6876159a11b4df116f30f667f8c9888617df0751
```

如需升级：

1. 单独提交依赖升级；
2. 记录旧、新 commit；
3. 对 Web 与 SAT 两个 smoke test 重新验证；
4. 比较模型参数名、输出 shape、预处理统计量和 checkpoint 加载报告；
5. 不得在同一提交中同时改变训练方法。

本项目 checkpoint 只保存可训练参数，不复制官方 backbone 权重。发布时必须分别遵守 DINOv3 License 与数据集许可，不把上游权重重新打包进项目产物。

## 4. 配置是实验唯一事实源

每次训练必须由一个 TOML 配置完整定义。禁止把学习率、冻结层数、数据子集、随机种子或 normalization 写成脚本内的临时常量。

配置分四组：

- `experiment`：名字、seed、输出目录；
- `model`：上游 commit 对应目录、权重、视觉域、冻结策略；
- `data`：manifest 和加载参数；
- `train`：优化器相关超参数和 negative queue。

规则：

1. 已经开始训练的配置视为不可变；新实验复制并改名。
2. 输出目录保存配置原文 `config.toml`。
3. 实验名必须包含关键变量，例如 `sat_last4_10k_seed11`。
4. 路径可以因服务器变化而变化，但影响研究结论的字段不可隐藏在环境变量里。
5. 同一对照组除目标研究变量外必须逐字段相同。

## 5. Canonical 数据契约

训练只读取 UTF-8 JSONL，每行必须包含：

```json
{"id":"chatearthnet:4334_2404_patch00.png","image":"/absolute/path/4334_2404_patch00.png","caption":"...","split":"train","source":"ChatEarthNet"}
```

字段约束：

- `id`：跨数据集唯一且稳定，格式推荐 `<source>:<native-id>`；
- `image`：预处理时写入绝对路径，训练时不猜目录；
- `caption`：单个非空字符串；多 caption 样本在预处理阶段明确选择或展开；
- `split`：只能由数据准备阶段决定，训练阶段不得重新随机划分；
- `source`：保留来源，用于泄漏审计与分组分析。

数据准备必须满足：

1. 原始压缩包只读保留，转换产物写入新目录；
2. 下载文件验证官方 hash；
3. 子集抽样固定 seed，并把抽样后的具体样本固化为 manifest；
4. train/val/test 按 `id` 去重，并对图像内容 hash 做二次泄漏审计；
5. NWPU-Captions 与 NWPU-RESISC45 的同源关系必须在报告中标注，不能作为完全独立 zero-shot 证据；
6. caption 清洗前后分别保存 manifest，不覆盖原文；
7. 坏图、缺图和超长文本数量必须输出统计，禁止训练时静默跳过。

## 6. 模型契约与冻结策略

模型 forward 采用官方 dino.txt 返回值：

```text
image_features:          [B, 2048], L2 normalized
text_features:           [B, 2048], L2 normalized
logit_scale:             scalar, exponentiated
patch_tokens:            [B, N, 1024]（alignment head 后）
backbone_patch_tokens:   [B, N, 1024]（backbone 输出）
```

MVP 的参数状态必须由一个函数显式设置：

| 模块 | 默认状态 |
|---|---|
| DINOv3 backbone | frozen + eval |
| dino.txt vision head | train |
| text token/position embedding | frozen |
| text transformer last K blocks | train |
| text final norm | K > 0 时 train |
| text projection | train |
| logit scale | train，限制到 `[1, 100]` |

每次启动必须打印 total/trainable parameter count。不得仅依赖模块名字符串的模糊匹配来冻结参数。若参数命名因上游升级改变，应当失败而不是静默训练错误模块。

Web 与 SAT backbone 必须使用各自官方 normalization：

```text
Web mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
SAT mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)
```

SAT 实验加载通用 dino.txt vision head/text encoder 后替换 SAT backbone，属于 head transfer 初始化。报告必须准确命名为 `SAT backbone + generic dino.txt initialization + RS tuning`，不能称为官方 SAT-dino.txt。

## 7. Loss 与 batch 语义

主损失是 batch 内对称 InfoNCE。图文第 `i` 项必须是正对，任何数据增强都不得打乱配对顺序。

需要特别区分：

- physical batch：一次 forward 的样本数；
- optimizer effective batch：physical batch × gradient accumulation；
- contrastive batch：一次 softmax 可见的当前样本数。

梯度累积只增大 optimizer effective batch，不增大 contrastive batch。`queue_size > 0` 会加入 detached 历史负样本，但这些 embedding 是 stale negatives。因此：

1. queue on/off 必须单独做消融；
2. 不得把 queue 宣称为等价于大 batch；
3. 未来若加入 GradCache 或跨卡 all-gather，必须新增 loss 实现，不在现有函数内隐式改变语义；
4. logit scale 用 fp32 计算/约束，防止温度失控。

## 8. 训练可靠性

训练必须满足：

1. 优先 bf16；只有硬件不支持时才使用 fp16 scaler；
2. backbone 保持 `eval()`，alignment/text trainable 模块保持 `train()`；
3. 记录 seed、torch/CUDA/GPU、DINOv3 commit、权重 hash、manifest hash；
4. checkpoint 采用先写 `.part` 再原子替换；
5. checkpoint 只保存 trainable state，不重复保存冻结权重；但必须同时保存 optimizer、scheduler、scaler、queue、已消费 sampler 位置、DataLoader generator 与 RNG 状态、step、配置和运行身份；
6. validation 必须使用完整固定 manifest、无增强、`model.eval()`、无 queue；`best.pt` 必须在 step 0 初始模型和所有 validation step 中按 validation loss 全局选择，test 集不得参与选模；
7. 训练中断后不得从“看起来相近”的配置恢复；恢复前必须严格验证配置文本 hash、项目/DINOv3 commit 与所有上游权重、tokenizer、manifest hash；
8. `num_workers = 0` 的恢复必须重现 sampler/RNG 状态；多 worker 运行可恢复训练状态但不得声称跨进程的随机增强逐样本 bitwise 相同，除非另行保存 worker RNG 状态；
9. NaN/Inf、空 batch、图像读取失败必须立即失败并指出 sample id；
10. 首次长训练前依次通过 CPU 数据 smoke test、单 batch CUDA forward、10-step overfit、100-step loss trend、validation/best/resume 和等价 SAT 受限验证。

## 9. 实验协议

RQ1 的 2×2 对照必须拆成清晰命名的实验：

| Vision | Language/init | 目的 |
|---|---|---|
| Web | official generic dino.txt，不训练 | 官方 baseline |
| SAT | official generic head transfer，不训练 | vision-domain diagnostic |
| Web | RS text tuning | language-domain contribution |
| SAT | RS text tuning | domain-matched system |

最低报告规范：

1. 主结果至少 3 个 seed，prototype 可先 seed 11；
2. 10k、50k、full 的 manifest 固定并嵌套，不能各自重新随机抽样；
3. 选择 checkpoint 的指标与最终 test 指标分离；
4. retrieval 报 I→T/T→I Recall@1/5/10、median rank；
5. zero-shot 分类固定 prompt templates 并纳入版本控制；
6. Web/SAT 比较使用相同 resolution、batch/loss、训练步数和数据顺序；
7. 任何失败实验都保留配置和最后日志，失败原因写入实验登记表。

## 10. 代码质量门槛

每个合并到主分支的变化至少通过：

```bash
ruff check .
pytest
python -m compileall -q src tools
```

模型或训练变化还需通过：

```bash
dinotxt-rs-smoke --config configs/train_mvp_web.toml
dinotxt-rs-smoke --config configs/train_mvp_sat.toml
```

新增功能需要最小单元测试；修 bug 必须先添加可复现该 bug 的测试。测试不得下载权重或依赖外网。涉及真实权重的检查属于显式 integration/smoke test。

## 11. 大文件、凭据与服务器规则

1. checkpoint、数据、输出和 `.env` 永不进入 Git；
2. Hugging Face token、Meta 签名 URL、对象存储凭据不得写入配置、日志或 shell history；
3. 镜像只改变传输路径，不改变文件内容；下载后仍验证官方 hash；
4. 来源不明且 hash 不匹配的“网盘权重”不得用于报告实验；
5. AutoDL 关机前确认下载完成、hash 通过、关键日志已保存到数据盘；
6. 不删除原始数据或旧 checkpoint，除非已经确认有第二份可恢复副本。

## 12. 分支演进顺序

只有上一阶段达到验收条件才进入下一阶段：

1. `M0 assets`：环境、四个权重文件、ChatEarthNet metadata/RGB 下载并校验；
2. `M1 inference`：Web/SAT smoke、官方 dino.txt embedding 与 patch shape 正确；
3. `M2 data`：10k manifest、泄漏/坏图统计、DataLoader throughput；
4. `M3 MVP`：Web 10k 训练稳定、retrieval 指标可计算；
5. `M4 RQ1`：SAT 10k 严格对照；
6. `M5 scale`：50k/full；
7. `M6 RQ2`：自然/结构化/层级文本；
8. `M7 RQ3`：只有前述结论稳定后才加入 local alignment。

这套顺序的核心不是限制探索，而是确保每一项新增复杂度都有一个已经冻结、可比较的参照物。
