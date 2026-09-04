# DINOtxt for Remote Sensing

本仓库是“面向遥感视觉理解的 DINOv3 领域文本对齐”毕业研究的首版工程骨架。当前实现聚焦可验证的 Minimum Viable Thesis：加载 Meta 官方 DINOv3/dino.txt，冻结视觉 backbone，训练视觉 alignment head、文本投影层与文本编码器最后 K 层，使用遥感 image-text pair 做对称 InfoNCE 对齐。

本项目不复刻官方 32 卡训练方案，也不从头预训练 DINOv3。首个目标是单张 RTX 4090 上跑通 ChatEarthNet 10k 子集，并建立 Web/SAT backbone 的受控比较。

## 今晚在 AutoDL 上完成什么

建议把项目放在数据盘（例如 `/root/autodl-tmp/Dinov3txtforSAT`），不要把 13 GB 数据和约数 GB 权重放在系统盘。

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
bash scripts/bootstrap_autodl.sh
source .venv/bin/activate
```

脚本使用 Python 3.12 `venv`，默认安装 PyTorch 2.7.1 + CUDA 12.8 wheel，并把 DINOv3 固定在提交 `6876159a11b4df116f30f667f8c9888617df0751`。如果实例驱动不支持 CUDA 12.8，先运行 `nvidia-smi`，再将 `PYTORCH_INDEX_URL` 改成匹配的官方 CUDA wheel 索引；不要混装不同 CUDA 后缀的 torch/torchvision。

下载权重、ChatEarthNet 元数据与 RGB 图像：

```bash
bash scripts/download_assets.sh all 2>&1 | tee download.log
```

可以开三个终端并行下载：

```bash
bash scripts/download_assets.sh weights
bash scripts/download_assets.sh data-metadata
bash scripts/download_assets.sh data-rgb
```

远程 SSH 可能断开时，建议放进 `tmux`，而不是只用普通前台终端：

```bash
tmux new -s dinotxt-download
bash scripts/download_assets.sh all 2>&1 | tee download.log
# 按 Ctrl-b，再按 d 可退出 tmux；之后用 tmux attach -t dinotxt-download 恢复查看。
```

脚本支持断点续传，并验证官方 checkpoint 的 SHA-256 文件名前缀以及 ChatEarthNet 的 MD5。若官方站点在服务器不可达，通过环境变量逐项替换 URL，不要修改脚本并提交临时镜像地址：

```bash
DINOV3_SAT_URL='你的可信镜像或 Meta 签名 URL' bash scripts/download_assets.sh weights
ZENODO_RECORD_BASE='你的 Zenodo 镜像记录目录' bash scripts/download_assets.sh data-rgb
```

下载完成后的关键文件应为：

```text
assets/checkpoints/
├── bpe_simple_vocab_16e6.txt.gz
├── dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth
├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
└── dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
assets/data/raw/chatearthnet/
├── json_files.zip
└── s2_rgb_images.zip
```

只下载不解压是今晚最低验收标准。磁盘充足时再解压：

```bash
mkdir -p assets/data/raw/chatearthnet/extracted
unzip -q assets/data/raw/chatearthnet/json_files.zip -d assets/data/raw/chatearthnet/extracted
unzip -q assets/data/raw/chatearthnet/s2_rgb_images.zip -d assets/data/raw/chatearthnet/extracted
find assets/data/raw/chatearthnet/extracted -type f | head
```

不要假定压缩包解压后的目录名。先用 `find` 找到实际 PNG 目录，再生成 manifest：

```bash
python tools/prepare_chatearthnet.py \
  --annotations assets/data/raw/chatearthnet/extracted/json_files/ChatEarthNet_caps_35_train.json \
  --images-root assets/data/raw/chatearthnet/extracted/s2_rgb_images \
  --split train \
  --limit 10000 \
  --seed 11 \
  --output assets/data/manifests/chatearthnet_train_10k.jsonl
```

验证模型权重、维度和冻结策略：

```bash
dinotxt-rs-smoke --config configs/train_mvp_web.toml
dinotxt-rs-smoke --config configs/train_mvp_sat.toml
```

M0–M2 已完成后，**不要直接启动** `train_mvp_web.toml` 的 5,000-step 正式训练。先从已清洗的 `global77` manifest 固化一个固定 batch，并执行受限 Web 10-step 验证：

```bash
python tools/prepare_fixed_manifest.py \
  --input assets/data/manifests/chatearthnet_35_train_10k_seed11_no_nodata_global77.jsonl \
  --output assets/data/manifests/chatearthnet_35_train_10k_seed11_no_nodata_global77_fixed16.jsonl \
  --limit 16 \
  --audit-output assets/data/manifests/chatearthnet_35_train_10k_seed11_no_nodata_global77_fixed16.audit.json
dinotxt-rs-train --config configs/verify_web_10step.toml
```

该配置重复同一批 16 条样本，关闭随机裁剪、shuffle 与 negative queue，只用于检查反向传播、数值稳定性、显存与 checkpoint 写入。验收 `metrics.jsonl` 的 10 条记录以及 `training_summary.json` 后，运行下列可复现实验脚本完成 Web 100-step 趋势验证：

```bash
bash scripts/run_web_100step_verification.sh
```

脚本使用完整 9,969 条训练清单、随机裁剪、shuffle、梯度累积和 4,096 条 queue，并在运行前拒绝未提交代码或既有输出目录。它会保存 smoke/train 日志、step 50/100 checkpoint，以及自动生成的 `verification_report.json`；完成 validation、best checkpoint 和 resume 后才可启动正式训练。

若该 100-step 运行的随机 `in_batch_loss` 无法给出可比趋势（不同 step 的样本和裁剪不同），不要修改已运行配置或直接延长训练。改用新的固定监测配置：它保持真实训练语义，但以同一批无增强的 16 条样本在 step 0、10、…、100 下的 eval loss 判断趋势。

```bash
bash scripts/run_web_100step_fixed_monitor.sh
```

该脚本使用新的输出目录，重新生成并校验监测 manifest，随后验证固定监测曲线、训练曲线、资产身份与 step 50/100 checkpoint。报告以固定监测 loss 的首 3 个点（step 0/10/20）与末 3 个点（step 80/90/100）均值判断趋势。

固定监测通过后，先执行新的 Web 验证 / best / resume 门槛；它以全量、不增强且不使用 queue 的 validation loss 在初始模型（step 0）和每个验证 step 中选取 `best.pt`，故不会把 test 集参与选模。脚本会在 step 50 正常结束，随后以相同 TOML 从该 checkpoint 恢复至 step 100；恢复前严格比对配置、项目与 DINOv3 commit、权重、tokenizer、train/val/fixed-monitor manifest 的 SHA-256。

```bash
bash scripts/run_web_100step_validation_resume.sh
```

验收 `validation.jsonl`（step 0/50/100）、`best.pt`、`resume_history.jsonl`、step 50/100 checkpoint 与 `verification_report.json`。验证 loss 是对全部 16,277 条样本、固定 batch 的无 queue InfoNCE 按样本加权平均；它是与训练目标同语义的确定性 validation 指标，而非 test 指标。该受限配置刻意采用 `num_workers = 0`，从而连同 sampler、queue、optimizer、scheduler 和 RNG 状态一起可精确恢复；它验证的是可靠性而非吞吐量。

仅当上述 Web 脚本通过后，运行等价的 SAT 受限验证。SAT 脚本会先重新核验 Web 产物，避免越过阶段门槛。

```bash
bash scripts/run_sat_100step_validation_resume.sh
```

若 100-step 的 validation 没有低于 step 0，不得直接启动正式 5,000-step。先运行 Web 的 500-step 受限 pilot：它采用未来正式训练的 `max_steps = 5000`、`warmup_steps = 250` 和 cosine schedule，却由命令行在 step 500 截止，并在 step 250 做严格恢复。因此它检查的是正式调度的前 500 step，不是新的 500-step 调度实验或正式训练。

```bash
bash scripts/run_web_500step_formal_schedule_pilot.sh
```

验收条件是：全量 validation 曲线至少出现低于 step 0 的点，`best.pt` 指向这一全局最优候选；否则停止并调整受限配置，不延长至 5,000 step。

在该脚本产生 `verification_report.json` 前，保留 `step_0000000.pt`、`step_0000250.pt`、`step_0000500.pt` 和 `best.pt`。`best.pt` 只能代表它自身 payload 所属的 step，不能代替已经删除的不同 step checkpoint。若发生误删，运行 `tools/inspect_training_artifacts.py` 生成 `recovery_report.json`，先据此判断是否仍能恢复某个同 step 的副本，以及 resume 证据是否已降级。

当前 Web pilot 已在全量 validation 上从 3.8231 降至 2.7668（best step 300），可进入等价的 SAT 500-step pilot。该配置保留 validation 的 16 样本 InfoNCE 指标定义，但让模型每次前向处理 64 个样本，并用 4 个仅用于确定性 validation 的 DataLoader workers 来缩短评估时间；训练仍是 `num_workers = 0`，不改变严格 resume 语义。

```bash
bash scripts/run_sat_500step_formal_schedule_pilot.sh
```

脚本会读取已审查的 Web `recovery_report.json`，明确记录其 step-250 resume 证据已降级；这只允许继续 SAT pilot，**不**构成启动正式 5,000-step 实验的许可。SAT 输出的 verifier 成功写出前，必须保留 step 0/250/500 与 `best.pt`。

训练启动时会计算当前配置、backbone、dino.txt 头、tokenizer 和 manifest 的 SHA-256，并将项目/DINOv3 commit、GPU/CUDA、Python 与 PyTorch 写入输出目录的 `provenance.json`。大权重哈希计算需要短暂等待，这是实验可复现性的必要成本。

## 当前代码结构

```text
configs/                     # 每次实验的不可变 TOML 配置
docs/                        # 架构规则与研究协议
scripts/                     # 环境和大文件下载入口
tools/                       # 一次性数据转换工具
src/dinotxt_rs/
├── cli/                     # 用户入口，不承载核心算法
├── data/                    # canonical manifest 与 transform
├── losses/                  # InfoNCE 与可选 negative queue
├── models/                  # 官方模型加载和显式冻结策略
└── training/                # 单卡训练、日志、轻量 checkpoint
```

详细的开发边界、数据契约、实验可复现规则与合并门槛见 [docs/DEVELOPMENT_ARCHITECTURE.md](docs/DEVELOPMENT_ARCHITECTURE.md)。研究动机与完整实验规划见 [DINOv3_Remote_Sensing_Domain_Text_Alignment_Research_Plan.md](DINOv3_Remote_Sensing_Domain_Text_Alignment_Research_Plan.md)。

## 已知边界

- 这是单卡 MVP，不是官方 FSDP 训练代码的缩小复刻。
- `gradient_accumulation` 只扩大优化器的有效 batch，不会让每个 InfoNCE softmax 看到更多当前批次负样本；`queue_size` 提供的是 detached、可能过时的额外负样本，必须单独消融。
- SAT 配置会加载通用 dino.txt alignment/text checkpoint，再替换为 SAT backbone；这正是需要测量的 domain mismatch 初始化，不应称为官方 SAT-dino.txt。
- 首版使用文本最后 K 层解冻。LoRA、multi-text 与 local alignment 要在 MVP 基线稳定后分支实现。
- ChatEarthNet HF 数据集预览存在 schema cast 问题，当前数据入口以作者提供的 Zenodo 压缩包为准。
