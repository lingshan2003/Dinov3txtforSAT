# DINOv3 Remote-Sensing Text Alignment：M0–M2 交接记录

更新时间：2026-09-03
当前状态：M0（资产与环境）、M1（模型加载）和 M2（数据准备与真实批次前向）均已完成。**尚未启动正式训练。** 下一阶段是受限的 Web 过拟合验证（10 step，再到 100 step），随后才补齐验证、checkpoint/resume 与下游评测闭环。

## 1. 服务器与仓库

- GitHub 仓库：`lingshan2003/Dinov3txtforSAT`（私有仓库）
- AutoDL 项目：`/root/autodl-tmp/Dinov3txtforSAT`
- 虚拟环境：`/root/autodl-tmp/Dinov3txtforSAT/.venv`
- DINOv3 源码：`/root/autodl-tmp/Dinov3txtforSAT/external/dinov3`
- 权重：`/root/autodl-tmp/Dinov3txtforSAT/assets/checkpoints`
- 原始数据：`/root/autodl-tmp/Dinov3txtforSAT/assets/data/raw`
- 派生 manifest：`/root/autodl-tmp/Dinov3txtforSAT/assets/data/manifests`
- 审计与实验输出：`/root/autodl-tmp/Dinov3txtforSAT/outputs`

数据、权重、外部 DINOv3 源码、输出目录和本地 `.venv` 都被 `.gitignore` 排除，均不应提交到 Git。

每次重新连接服务器：

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
source .venv/bin/activate
```

服务器已确认的运行环境：Python 3.12、PyTorch `2.7.1+cu128`、CUDA runtime 12.8、RTX 4090、BF16 可用。DINOv3 固定在提交 `6876159a11b4df116f30f667f8c9888617df0751`，项目以 editable 模式安装。

## 2. M0：已核验的资产

### 2.1 模型与文本资源

下列文件均位于 `assets/checkpoints/`，文件大小与 SHA-256 已在服务器核验：

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | 1,213,050,671 | `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035` |
| `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | 1,213,059,235 | `eadcf0ffc02418b6c22a885ea1a7aaeeef84fbf0f5bb4d0b7d1d36e68a964f48` |
| `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth` | 2,253,936,683 | `a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0` |
| `bpe_simple_vocab_16e6.txt.gz` | 1,356,917 | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |

`bpe_simple_vocab_16e6.txt.gz` 必须保持压缩状态。复核入口为 `python tools/verify_server_assets.py`。

### 2.2 原始数据

| 数据集 | 已核验的真实位置 | 核验结果 | 当前用途 |
| --- | --- | --- | --- |
| ChatEarthNet | `assets/data/raw/chatearthnet/s2_images` | 163,488 张 PNG，全部可读 | 图文对齐微调 |
| EuroSAT | `assets/data/raw/eurosat` | 27,000 张 JPEG，全部可读 | 零样本分类（待实现评测） |
| RSICD | `assets/data/raw/rsicd` | 10,921 张 JPG，全部可读，全部标注引用可解析 | 双向检索（待实现评测） |

ChatEarthNet 标注的真实位置是：

```text
assets/data/raw/chatearthnet/json_files/ChatEarthNet_caps_35_{train,val,test}.json
```

其 35 版本原生记录数分别为 train 98,092、val 16,348、test 49,048。RSICD 的 `dataset_rsicd.json` SHA-256 为 `5e342037d469d074711676bdb9c02b6942a624530b1959d24d2734e68af9cede`。EuroSAT 类别分布为：AnnualCrop、Forest、HerbaceousVegetation、Residential、SeaLake 各 3,000；Highway、Industrial、PermanentCrop、River 各 2,500；Pasture 2,000。

### 2.3 原始归档与再次核验约定

本文合并并取代旧的 `docs/SERVER_ASSETS.md`。当前事实以本交接记录为准；原始压缩包只作为可恢复的只读归档，不应替代已核验的解压数据。历史归档名称分别为 ChatEarthNet 的 `s2_rgb_images.zip`、EuroSAT 的 `EuroSAT.zip`、RSICD 的 `RSICD_images.zip` 与 `annotations_rsicd.rar`。若需要从归档恢复 RSICD 标注，服务器的 `p7zip 16.02` 不支持该 RAR 的压缩方法，应使用 `unrar` 或 `unar`。

资产迁移、恢复或重新下载后，先在项目根目录运行：

```bash
python tools/verify_server_assets.py
find assets/checkpoints -maxdepth 1 -type f -exec ls -lh {} \;
find assets/data/raw -maxdepth 4 -type d | sort
find assets/data/raw/chatearthnet -type f | wc -l
find assets/data/raw/eurosat -type f | wc -l
find assets/data/raw/rsicd -type f | wc -l
find assets/data/raw/rsicd -name 'dataset_rsicd.json' -print
du -sh assets/checkpoints assets/data/raw/*
```

不得因重新核验而删除原始归档、旧 checkpoint 或已有实验输出；先确认存在第二份可恢复副本。

## 3. M1：模型与管线冒烟结果

### 3.1 合成模型 smoke test

Web 和 SAT 主干均能加载，dino.txt 文本组件和视觉投影头可正常前向；特征、patch token 均为有限值。

| 域 | 总参数 | 可训练参数 | 图文特征 | patch token | logit scale | 峰值显存（约） |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| Web | 866,611,713 | 106,644,737 | `[2, 2048]` | `[2, 196, 1024]` | 100.0 | 3.76 GB |
| SAT | 866,613,761 | 106,644,737 | `[2, 2048]` | `[2, 196, 1024]` | 100.0 | 3.76 GB |

### 3.2 真实 ChatEarthNet 批次 smoke test

当前训练配置使用下面第 4 节的 `global77` 训练 manifest。真实 DataLoader 能稳定读取 `[16, 3, 224, 224]` 的图像批次，Web/SAT 单批前向、损失与显存均正常。

| 域 | 真实 batch | token shape | 最大非 padding token | smoke loss | 峰值 CUDA allocated |
| --- | --- | --- | ---: | ---: | ---: |
| Web | 16 | `[16, 77]` | 38 | 3.7621241 | 3,885,012,992 bytes |
| SAT | 16 | `[16, 77]` | 38 | 3.5476735 | 3,907,730,432 bytes |

这些 loss 仅证明计算图连通且数值有限，**不是**训练性能、基线或 Web/SAT 的比较结论。

已作出的实现修正：

- `tools/prepare_chatearthnet.py` 已适配实际 JSON 与图像目录，并输出可审计 JSONL manifest。
- 由于 caption 包含左上、右下等方向信息，图文训练数据不再使用未同步修改文本的水平/垂直翻转。
- `tools/verify_server_assets.py` 可核验权重、目录、图像和标注引用。
- `tools/filter_manifest_by_image_hash.py` 与 `tools/prepare_global77_manifest.py` 用于生成不可变的派生 manifest。
- `src/dinotxt_rs/cli/smoke_model.py` 已检查输出 shape、有限数值、可训练参数、logit scale 与 CUDA 峰值分配。

## 4. M2：ChatEarthNet 数据协议（当前有效输入）

### 4.1 10k 候选子集和 no-data 清洗

从 35-train 的 98,092 条记录，以 seed 11 抽取 10,000 条候选样本。原始抽样 manifest 的 SHA-256 是 `cc34af91b93303e77e2f69b6b8260a6a8b1f05036ff0366d883dbaf700a6e511`。

随后发现跨 split 的“泄漏”不是样本 ID 相同，而是两张占位图重复出现：一张全白和一张全黑。这两张图的 SHA-256 分别为：

- `13c0915d226521fb56bba264c92a227dbdebcf9a4add3a4957fd9c810582dbe8`（全白 RGB 255）
- `3b93f26267630963185ee67d8020972faee108a7bca80db725817e4680c9955c`（全黑 RGB 0）

它们不是有效遥感内容，因而从 train、val、test **全部**移除；不是只从训练集移除。清洗后任意两 split 的 ID overlap 和 image-content-SHA overlap 都为 0。

| split | no-data 前 | 移除 | no-data 后 | no-data 后 SHA-256 |
| --- | ---: | ---: | ---: | --- |
| 35 train 10k 候选 | 10,000 | 31 | 9,969 | `3d9d27380e3c7565b136033298934ab1cc9f5d93c3eb11f9e059470860a018fa` |
| 35 val | 16,348 | 71 | 16,277 | `3426eb87002854c594a2b60ddf1af105a4cbdb1f868651e423af8ef440b491cd` |
| 35 test | 49,048 | 188 | 48,860 | `8eb3d5d039688afce70c57accf35817ffe71de3e14e87fdbc2b7b3961d1a2024` |

所以“10k”准确说是 **10,000 条候选、9,969 条可用训练样本**。不要用未清洗 manifest 训练或报告结果。

### 4.2 文本长度与 `global77` 策略

dino.txt tokenizer 的 context length 是 77。原始 35 caption 的 BPE token 数 p50/p95 为 184/208，约 99.84% 会被模型静默截断；4v 训练集也有约 96.3% 会截断。因此不能把原始完整 caption 直接作为当前训练目标。

当前采用的、已固化的文本协议为 `first_complete_sentence_then_complete_word_backoff`：

1. 只取首个完整句子；句末仅认定为 `.`, `!`, `?` 后接空白或文本结尾，**逗号不截断**。
2. 若该完整首句仍超出 77 token 预算，按完整空白分词逐词从末尾回退，使 BPE 内容加 SOT/EOT 不超过 77。
3. 不丢弃样本，不原地改写输入 manifest；生成新的派生 JSONL 与审计信息。

| split | 当前 `global77` 文件 | 记录数 | SHA-256 | token p50/p95/max | 完整词回退数 |
| --- | --- | ---: | --- | --- | ---: |
| train | `chatearthnet_35_train_10k_seed11_no_nodata_global77.jsonl` | 9,969 | `78abc613fbc8d98ea4617770473b30662d9eda31c0deb0dd06b51b1965d9fc0b` | 16 / 35 / 76 | 1 |
| val | `chatearthnet_35_val_no_nodata_global77.jsonl` | 16,277 | `1040ccf2ec07100ceb81ad665e28527d38b948cc9e1e547eb76e23a265c25f88` | 15 / 35 / 62 | 0 |
| test | `chatearthnet_35_test_no_nodata_global77.jsonl` | 48,860 | `51662bee9618051416bc7a7983d1fc658c2fe6ed825b86f9258205597ca77716` | 15 / 35 / 76 | 0 |

两份现有 MVP 配置的有效训练/验证输入是：

```toml
# configs/train_mvp_web.toml 与 configs/train_mvp_sat.toml
train_manifest = "assets/data/manifests/chatearthnet_35_train_10k_seed11_no_nodata_global77.jsonl"
val_manifest = "assets/data/manifests/chatearthnet_35_val_no_nodata_global77.jsonl"
```

4v 的 manifest 已生成（train/val/test 为 6,000/1,000/3,000），但尚未按本协议生成 `no_nodata_global77` 版本；它们不是当前正式输入。

### 4.3 后续 50k 与全量实验规则

“10k、50k、全量”是用于观察微调量影响的嵌套训练规模，而不是重新按 7:2:1 随机切分数据。后续应：

1. 从同一 35-train 固定 seed/顺序产生前 50k 和全量候选；确保前一规模是后一规模的子集。
2. 应用同一对全黑/全白 SHA 的 no-data 过滤。
3. 应用相同的 `global77` 文本生成脚本与 context length。
4. 固定现有 35 val/test 的 no-data + global77 文件，不因训练规模改变验证或测试集。
5. 记录输入与输出 manifest SHA-256，实验报告以清洗后的实际样本数为准。

## 5. 已完成的本地与服务器验证

- 本地使用 `uv` 创建的 `.venv` 仅用于测试，不提交；当前已通过 `ruff check .`、`compileall`、`pytest`（21 passed）。
- 服务器已通过 `ruff check .`、测试、模型 synthetic smoke、真实 DataLoader smoke，以及上述 Web/SAT 的真实批次前向。
- 35 train/val/test 清洗后已完成 manifest 审计：无重复 ID、无缺图、无空 caption，且三对 split 均为零 ID/内容哈希交集。

## 6. 阶段门槛与下一步

现在**接近**正式实验，但尚不满足“正式多步训练”的闭环标准。下一步不是直接启动大规模或多 seed 训练，而是按下列顺序推进：

1. 在服务器从当前 `global77` 训练 manifest 派生固定 16 条训练输入，再运行 Web 受限 10-step 验证；验证逐步 loss、有限梯度、峰值显存与原子 checkpoint 写入，不将它作为正式结果：

   ```bash
   python tools/prepare_fixed_manifest.py \
     --input assets/data/manifests/chatearthnet_35_train_10k_seed11_no_nodata_global77.jsonl \
     --output assets/data/manifests/chatearthnet_35_train_10k_seed11_no_nodata_global77_fixed16.jsonl \
     --limit 16 \
     --audit-output assets/data/manifests/chatearthnet_35_train_10k_seed11_no_nodata_global77_fixed16.audit.json
   dinotxt-rs-train --config configs/verify_web_10step.toml
   ```

   该配置关闭随机裁剪、shuffle 和 negative queue，并以 `num_workers = 0` 重复同一物理 batch。验收产物为输出目录中的 `config.toml`、`provenance.json`、`metrics.jsonl`（恰好 10 条记录）、`step_0000010.pt` 和 `training_summary.json`；后者必须报告有限的首末 loss、梯度及峰值 CUDA allocated bytes。
2. 初版完整训练清单的 Web 100-step 已通过结构性检查（数值、queue、step 50/100 checkpoint 与资产身份），但其 `in_batch_loss` 来自不同的随机样本和随机裁剪，前后窗口不可直接比较。保留该输出作为稳定性证据，不以它作优化趋势结论，也不修改已运行的不可变配置。
3. 用固定监测 batch 重新运行独立的 Web 100-step 配置：

   ```bash
   bash scripts/run_web_100step_fixed_monitor.sh
   ```

   训练继续使用完整 9,969 条 `global77` 清单、随机裁剪、shuffle、`gradient_accumulation = 4` 和 `queue_size = 4096`。固定的 16 条 `global77` 样本仅在 step 0、10、…、100 时以 `model.eval()` 和无增强预处理计算不含 queue 的 loss。脚本会重新生成并 hash 校验此监测 manifest，且核验 `metrics.jsonl`、fixed-monitor 曲线、provenance、step 50/100 checkpoint 与原子写入结果。以 `verification_report.json` 的 `fixed_monitor_loss.mean_first_window` 与 `mean_last_window` 判断趋势；该字段的 `window_size` 为 3，分别比较 step 0/10/20 与 step 80/90/100。
4. Web 与 SAT 的 100-step validation / best / resume 已完成，且工程门槛通过：两者均完整运行、queue 到 4,096、step 50→100 resume 与资产身份均通过。然而，Web validation loss 从 3.8231（step 0）变为 4.0567（step 100），SAT 从 3.8224 变为 4.5138；两者都仍劣于初始模型。此前实现只在 post-training step 选择 `best.pt`，故报告的 step 100 不是含 baseline 的全局最佳。该选择逻辑现已修正：训练会保存 `step_0000000.pt`，并让 `best.pt` 在 step 0 和全部 validation step 中全局选择。

5. 下一次服务器运行不是正式 5,000-step，而是 Web 500-step 的正式调度受限 pilot：

   ```bash
   bash scripts/run_web_500step_formal_schedule_pilot.sh
   ```

   配置的 `max_steps = 5000`、`warmup_steps = 250` 与未来正式训练一致，但脚本仅运行至 step 500，并在 step 250 保存、重启、严格恢复到 step 500。全量 validation 在 step 0、50、…、500 以固定 batch、无增强、无 queue 的 in-batch InfoNCE 按样本加权平均写入 `validation.jsonl`；`best.pt` 可以合法地指向 `step_0000000.pt`。报告必须确认 `target_steps = 5000`、`completed = false`、step 0/250/500 checkpoint、step 250 resume 和全局 best 选择。若 validation 从未低于 step 0，停止并调整受限配置，**不得**继续到 5,000。

6. 只有第 5 步 Web pilot 出现优于 step 0 的 validation，才以同一受限协议为 SAT 编写/运行 500-step pilot；若 SAT 仍不改善，则把它作为 generic dino.txt initialization 的 domain-mismatch 诊断，而不是正式对照。
7. 实现 EuroSAT 零样本分类和 RSICD 双向检索，先保存未微调的 Web/SAT 基线。
8. 当以上闭环完成后，才按统一配置运行 9,969（原 10k 候选）、50k 和全量的 Web/SAT 正式对照实验，并固定随机种子和报告指标。

正式实验前需要留存的证据包括：配置快照、代码 commit、每个 manifest 的 SHA-256、随机种子、硬件/软件环境、训练/验证曲线、最佳 checkpoint、恢复训练结果以及 EuroSAT/RSICD 指标。

## 7. 相关文档

- `docs/DEVELOPMENT_ARCHITECTURE.md`：训练、检查点、评测与实验记录的目标架构。
- `docs/PREPARE_HANDOFF.md`（本文）：截至 M2 的实际执行事实和下一阶段门槛。
