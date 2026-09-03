# DINOv3 Remote-Sensing Text Alignment：准备阶段交接记录

更新时间：2026-09-02  
阶段状态：服务器软件环境、训练/评测数据和模型权重已基本准备完毕；下一阶段进入资产核验、数据适配和模型冒烟测试。

## 1. 项目与服务器位置

- GitHub 仓库：`lingshan2003/Dinov3txtforSAT`（私有仓库）
- AutoDL 持久化目录：`/root/autodl-tmp`
- AutoDL 项目目录：`/root/autodl-tmp/Dinov3txtforSAT`
- Python 虚拟环境：`/root/autodl-tmp/Dinov3txtforSAT/.venv`
- DINOv3 源码：`/root/autodl-tmp/Dinov3txtforSAT/external/dinov3`
- 模型文件目录：`/root/autodl-tmp/Dinov3txtforSAT/assets/checkpoints`
- 原始数据目录：`/root/autodl-tmp/Dinov3txtforSAT/assets/data/raw`
- 后续 manifest 目录：`/root/autodl-tmp/Dinov3txtforSAT/assets/data/manifests`
- 实验输出目录：`/root/autodl-tmp/Dinov3txtforSAT/outputs`

数据、权重、DINOv3 外部源码和实验输出均已通过 `.gitignore` 排除，不应提交到 Git 仓库。

## 2. 已准备的数据

### 2.1 ChatEarthNet

- 已下载并上传：`s2_rgb_images.zip`
- 用途：遥感 image-text pair 微调训练数据
- 推荐解压位置：`assets/data/raw/chatearthnet`
- 下一阶段仍需核验：
  - 实际解压后的图像根目录；
  - 图像数量与损坏文件；
  - ChatEarthNet 文本标注文件的实际文件名、路径和格式；
  - 图像文件名与标注是否能逐条对应；
  - 官方 split 是否存在，若不存在则固定并记录自定义 split。

### 2.2 EuroSAT

- 已下载并上传：`EuroSAT.zip`
- 用途：零样本遥感场景分类评测
- 推荐解压位置：`assets/data/raw/eurosat`
- 下一阶段仍需核验：类别目录、类别名称、每类图像数和总图像数。

### 2.3 RSICD

- 已下载并上传图像：`RSICD_images.zip`
- 已下载并上传标注：`annotations_rsicd.rar`
- 标注压缩包包含：`dataset_rsicd.json`、`readme.txt`
- 用途：image-to-text 和 text-to-image 双向检索评测
- 推荐解压位置：`assets/data/raw/rsicd`
- 注意：服务器原有的 `p7zip 16.02` 不支持该 RAR 的压缩方法，应使用 `unrar` 或 `unar` 解压。
- 下一阶段仍需核验：图像目录、标注 JSON 路径、split 字段、caption 数量以及标注引用的所有图片是否存在。

## 3. 已准备的模型与文本资源

以下文件均已下载并上传服务器：

1. Web DINOv3 ViT-L/16 主干：
   `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
2. SAT DINOv3 ViT-L/16 主干：
   `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`
3. dino.txt 视觉对齐头与文本编码器：
   `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`
4. 文本 BPE 词表：
   `bpe_simple_vocab_16e6.txt.gz`

预期均位于：

```text
/root/autodl-tmp/Dinov3txtforSAT/assets/checkpoints/
```

`bpe_simple_vocab_16e6.txt.gz` 应保持压缩状态，不要手动解压。

下一阶段应记录上述四个文件的大小和 SHA-256，防止上传不完整或后续被替换。

## 4. 已配置并验证的软件环境

项目采用 Python `venv`，未使用 Conda。服务器已经确认：

- Python：3.12
- PyTorch：`2.7.1+cu128`
- PyTorch CUDA runtime：12.8
- GPU：NVIDIA GeForce RTX 4090
- `torch.cuda.is_available()`：`True`
- BF16：支持
- DINOv3：已经从官方公开仓库通过 HTTPS 克隆并能正常导入
- DINOv3 实际导入路径：
  `/root/autodl-tmp/Dinov3txtforSAT/external/dinov3/dinov3/__init__.py`
- 项目包 `dinotxt-rs==0.1.0`：已以 editable 模式安装
- 项目 Python 依赖：已安装

每次重新进入服务器终端后，应先执行：

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
source .venv/bin/activate
```

## 5. Git 和网络情况

- 本机仓库远端已经由 GitHub HTTPS 改为 GitHub SSH。
- GitHub 仓库是私有仓库；AutoDL 若要直接拉取，需要单独配置仓库访问权限。
- AutoDL 克隆公开的 DINOv3 仓库时，SSH 因 `Permission denied (publickey)` 失败。
- DINOv3 后来改用以下公开 HTTPS 地址并成功完成：
  `https://github.com/facebookresearch/dinov3.git`
- 服务器以后执行环境脚本时，不应再通过 `DINOV3_GIT_URL` 强制覆盖为 SSH 地址。
- 数据集和权重不经过 Git/GitHub 传输。

## 6. 当前代码状态

仓库已经包含：

- Web/SAT 两套 MVP 训练配置；
- ChatEarthNet 通用 manifest 转换工具；
- 图文数据读取器；
- 官方 dino.txt 组件加载与冻结策略；
- 对称 InfoNCE 损失和可选负样本队列；
- BF16、梯度累积、AdamW、warmup/cosine 训练器；
- checkpoint、运行环境记录和文件哈希记录；
- 模型 smoke-test 入口及基础单元测试；
- 开发架构规范 `docs/DEVELOPMENT_ARCHITECTURE.md`。

当前代码仍属于 MVP 骨架，尚未完成正式实验闭环。

## 7. 下一阶段应严格按顺序完成

### 第一批：只做核验，不开始正式训练

1. 拉取最新仓库并激活 `.venv`。
2. 自动核对四个模型文件是否存在、大小是否合理并计算 SHA-256。
3. 扫描三个数据集的真实解压目录，记录文件数、类别和标注路径。
4. 检查 ChatEarthNet 与 RSICD 标注引用的图片是否全部存在。
5. 运行 `pytest` 和 `ruff check .`。

### 第二批：打通数据与模型

6. 根据服务器真实目录适配 ChatEarthNet manifest 转换器。
7. 固定训练/验证划分及随机种子，生成 10k MVP manifest。
8. 修改方向敏感的数据增强：不要在不改 caption 的情况下随意水平/垂直翻转带有方位描述的样本。
9. 分别对 Web 和 SAT 模型执行加载与前向 smoke test。
10. 检查输出 shape、有限数值、显存占用和可训练参数数量。

### 第三批：建立实验闭环

11. 实现 EuroSAT 零样本分类数据加载、prompt 和指标。
12. 实现 RSICD 双向检索加载与 Recall@1/5/10、Mean Recall。
13. 先保存未经 ChatEarthNet 微调的 Web/SAT 基线结果。
14. 做单 batch 的 10～100 步过拟合测试。
15. 增加 validation、best-checkpoint 和 resume 后，再启动正式训练。
16. 在完全相同协议下运行 Web 与 SAT 对照实验。

## 8. 下一对话建议的开场指令

可将下面的话直接发给下一位助手：

> 请先完整阅读 `docs/PREPARE_HANDOFF.md`、`docs/SERVER_ASSETS.md` 和
> `docs/DEVELOPMENT_ARCHITECTURE.md`。准备阶段已经结束。请从“服务器资产自动核验与
> 数据真实目录适配”开始，不要直接启动正式训练；先给出服务器核验命令，根据结果完善
> 数据解析代码，然后依次运行测试和 Web/SAT smoke test。

