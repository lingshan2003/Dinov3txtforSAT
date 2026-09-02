# 面向遥感视觉理解的 DINOv3 领域文本对齐研究规划

> **暂定英文题目**  
> **What Should Be Aligned? Domain-Specific Vision-Language Alignment of DINOv3 for Remote Sensing**
>
> **暂定中文题目**  
> **面向遥感视觉理解的 DINOv3 领域文本对齐研究**
>
> **文档定位**：毕业研究报告选题与实验规划草案  
> **目标周期**：4–6 个月  
> **算力约束**：单卡或双卡，24–48 GB 显存  
> **研究目标**：以“问题定义 + 方法实现 + 系统验证 + 失败分析”为主要验收标准，不以达到 SOTA 或单一准确率提升为首要目标  
> **更新日期**：2026-08-31

---

## 0. Executive Summary

本项目以 **DINOv3** 为视觉基础模型，以 CVPR 2025 的 **DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment**（以下简称 **dino.txt**）为核心 anchor paper，研究一个比“把 DINOv2 换成 DINOv3”更有实际研究意义的问题：

> **当 DINOv3 已经在遥感影像上进行领域级视觉预训练后，如何进一步进行“领域匹配的文本对齐”，使其获得适合遥感场景的语言接口？视觉域匹配、语言域匹配、文本粒度和局部/全局对齐分别起到什么作用？**

该方向适合毕业研究报告，原因是：

1. **问题明确而非盲目堆模块**：DINOv3 已有 Web 预训练和 SAT-493M 卫星预训练版本，但官方公开 dino.txt 权重目前绑定通用 Web ViT-L backbone，因此存在自然的“视觉域 × 语言域”研究问题。
2. **Anchor paper 清晰**：dino.txt 提供可直接继承的 LiT 式冻结视觉 backbone + 文本对齐方案，并强调 global/dense unified alignment。
3. **工程量充分**：涉及 DINOv3 backbone、dino.txt text tower、parameter-efficient domain adaptation、遥感图文数据处理、zero-shot/retrieval/dense evaluation、可视化与消融。
4. **算力可控**：主模型使用 DINOv3 ViT-L/16（300M）并冻结 backbone；避免 7B 模型和完整 foundation-model 预训练。
5. **验收天然不依赖 SOTA**：核心结论可以是“什么对齐有效、什么无效、为何有效、在哪类任务有效”，即使最终 mIoU 或 retrieval 没有刷新榜单，也可以形成完整研究报告。
6. **现实意义明确**：遥感专业数据标注成本高，而开放词汇识别、文本检索、弱监督定位都具有实际用途。

---

# 1. 研究背景

## 1.1 从自监督视觉基础模型到语言接口

DINO 系列代表了一条不同于 CLIP 的视觉基础模型路线：

- CLIP：通过大规模 image-text pairs 获得视觉—语言联合语义空间。
- DINO/DINOv2/DINOv3：主要通过自监督视觉训练学习视觉表征，本身没有天然语言接口。
- 优势：DINO 类模型的 patch-level 特征具有较强的结构、局部和 dense information。
- 劣势：未经额外对齐时，不能直接使用文本 prompt 做 zero-shot/open-vocabulary 任务。

这形成一个重要研究问题：

> **能否在保留 DINO 强视觉表示的同时，以较低成本为其加入语言能力？**

CVPR 2025 的 **DINOv2 Meets Text** 正面回答了这个问题。

---

## 1.2 为什么 anchor paper 是 DINOv2 Meets Text

### Paper

**Cijo Jose et al.  
DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment.  
CVPR 2025.**

官方页面：  
https://openaccess.thecvf.com/content/CVPR2025/html/Jose_DINOv2_Meets_Text_A_Unified_Framework_for_Image-_and_Pixel-Level_CVPR_2025_paper.html

### 核心思想

该工作建立在 **Locked-image Text Tuning (LiT)** 思路之上：

- 冻结强视觉 backbone；
- 学习与之匹配的文本编码器；
- 通过 image-text contrastive learning 获得开放词汇能力。

但 dino.txt 进一步发现：

- 仅仅让文本对齐全局 CLS token 对 dense task 不够；
- DINO patch token 本身包含非常重要的局部语义；
- 因而使用类似：

\[
z_{\text{image}}
=
\left[
z_{\text{CLS}};
\operatorname{AvgPool}(z_{\text{patch}})
\right]
\]

进行 image-level + dense-feature-aware alignment。

### 为什么适合作为毕业研究 anchor

它留下了很自然的 extension：

1. DINOv2 → DINOv3；
2. 通用 Web domain → specialized domain；
3. 普通 caption → domain-aware / structured text；
4. global + pooled local alignment → 更有领域结构的 multi-granular alignment。

本项目并不需要完全重构其训练框架，而是可以：

> **在它已经验证有效的对齐机制上研究“domain alignment”问题。**

---

# 2. DINOv3 带来的新机会

## 2.1 DINOv3 基础事实

DINOv3 于 2025 年发布，是 Meta 推出的新一代 self-supervised vision foundation model。

官方信息：

- Web dataset：**LVD-1689M**
- Satellite dataset：**SAT-493M**
- 视觉模型：
  - ViT-S/16：21M
  - ViT-B/16：86M
  - ViT-L/16：300M
  - ViT-H+/16：840M
  - ViT-7B/16：约 6.7B
- SAT-493M 公开：
  - ViT-L/16：300M
  - ViT-7B/16：约 6.7B

官方仓库：  
https://github.com/facebookresearch/dinov3

官方模型卡：  
https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md

### SAT-493M

官方模型卡说明：

- 493M 个 satellite image crops
- 512×512
- 来源于 Maxar RGB ortho-rectified imagery
- 约 0.6 m spatial resolution

这意味着：

> DINOv3 不只是“一个通用视觉模型”，而是官方直接提供了 **domain-specific visual foundation model**。

这为本项目的 controlled study 提供了非常理想的实验条件。

---

## 2.2 DINOv3 已经拥有 dino.txt，但这里仍有 gap

DINOv3 官方公开了 dino.txt 代码和权重。

当前公开代码中的配置大致为：

- Vision backbone：DINOv3 ViT-L/16
- Vision backbone frozen
- Vision head blocks：2
- Use CLS token：Yes
- Use patch tokens：Yes
- Patch pooling：mean
- Text Transformer：
  - context length：77
  - dim：1280
  - heads：20
  - layers：24
- shared embedding dim：2048

代码位置：

https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/dinotxt.py

尤其关键的是，目前公开函数默认：

```text
backbone_weights = LVD1689M
DINOTxtWeights = LVTD2300M
```

即官方公开的 dino.txt 是围绕 **Web-pretrained DINOv3 ViT-L** 发布的。

与此同时：

- SAT-493M ViT-L 是单独公开的；
- 没有在官方 model table 中看到一个配套的 “SAT-specific dino.txt” 公开权重。

因此可以提出：

> **SAT-DINOv3 已经具有专业遥感视觉表示，但它是否拥有匹配的专业语言空间？**

这构成本项目最核心的研究缺口。

---

# 3. 核心 Research Gap

本项目不将 gap 表述为：

> “DINOv3 没做遥感。”

因为这是错误的——DINOv3 明确拥有 SAT-493M 模型，而且 2026 已经有工作将 DINOv3/dino.txt 用于遥感开放词汇分割。

真正的 gap 建议定义为：

> **Existing work has demonstrated strong remote-sensing visual representations in DINOv3 and the general language alignment capability of DINO.txt. However, the interaction between domain-specific visual pretraining and domain-specific language alignment remains underexplored. In particular, it is unclear whether a satellite-pretrained DINOv3 should be aligned with generic web language or remote-sensing-specific descriptions, and which level of textual and visual granularity is most effective under limited computational resources.**

中文：

> **现有研究分别验证了 DINOv3 的专业遥感视觉表征能力，以及 DINO.txt 为 DINO 引入开放词汇语言能力的有效性。然而，“领域视觉预训练”和“领域语言对齐”之间的关系仍缺乏系统研究。尤其对于 SAT-493M DINOv3，目前尚不清楚其应如何与遥感专业文本进行低成本匹配，以及文本粒度、视觉粒度和数据规模分别会产生什么影响。**

---

# 4. 推荐研究题目

## 4.1 首选标题

### 中文

**面向遥感视觉理解的 DINOv3 领域文本对齐研究**

### 英文

**Domain-Specific Vision-Language Alignment of DINOv3 for Remote Sensing**

---

## 4.2 更研究型的标题

**What Should Be Aligned? Visual-Domain and Language-Domain Adaptation of DINOv3 for Remote Sensing**

这一标题更能体现本项目的核心是分析：

- visual-domain match
- language-domain match
- granularity match

而不只是造新模块。

---

# 5. Research Questions

建议最终只保留 **3 个主要 Research Questions**。

## RQ1：视觉域匹配和语言域匹配分别有什么贡献？

> Does a remote-sensing-specific visual backbone still require remote-sensing-specific language alignment?

构造一个非常清晰的 2×2 实验：

| Vision Backbone | Language Alignment | 含义 |
|---|---|---|
| Web DINOv3 | Generic dino.txt | 官方通用 baseline |
| Web DINOv3 | RS text | 只改变 language domain |
| SAT DINOv3 | Generic text/head transfer | 只改变 vision domain |
| SAT DINOv3 | RS text | domain-matched system |

希望回答：

- SAT visual pretraining 是否是主要收益来源？
- domain text alignment 是否仍然重要？
- 两者是否有协同？
- 不同任务结论是否一致？

---

## RQ2：什么样的文本最适合专业领域对齐？

比较：

### Level 1 — Label Prompt

```text
a satellite image of dense residential area
```

### Level 2 — Natural Caption

```text
A dense residential area containing closely packed buildings,
roads and scattered vegetation.
```

### Level 3 — Structured Caption

```text
Scene: dense residential area.
Land cover: buildings dominant; sparse vegetation.
Structure: roads divide dense building clusters.
```

### Level 4 — Hierarchical Text

```text
built-up
→ residential
→ dense residential
```

研究：

- generic natural language 是否足够？
- structured domain description 是否更适合 dense/local task？
- 层级类别是否提升长尾/相似类别区分？
- 更长的文本一定更好吗？

---

## RQ3：全局语义和局部语义应该怎样对齐？

anchor paper 已经说明：

- CLS-only 不够；
- patch-aware alignment 对 dense prediction 很重要。

本项目进一步探索：

\[
L
=
L_{\text{global}}
+
\lambda L_{\text{local}}
\]

其中：

### Global

整图：

```text
image ↔ scene caption
```

### Local / Weak Local

patch / patch group：

```text
local region ↔ land-cover term
```

例如：

```text
water
building
road
forest
agriculture
```

不一定要求人工 pixel annotation。

可以利用：

- land-cover metadata；
- segmentation dataset label；
- caption 中的 entity extraction；
- pretrained model pseudo region；
- WorldCover/类似地理元数据。

因此该部分可以作为项目中最明显的 **method contribution**。

---

# 6. 预期 Contribution 设计

不要把 contribution 写成“提出三个全新模块”。

推荐按研究报告逻辑写。

## C1 — Domain-Specific DINOv3 Text Alignment Framework

实现一个：

> **frozen DINOv3 + lightweight visual alignment + parameter-efficient text adaptation**

框架，使 Web/SAT DINOv3 都可以在 remote-sensing image-text dataset 上重新进行语言对齐。

这是工程主体。

---

## C2 — Controlled Study of Vision-Domain × Language-Domain Matching

系统研究：

\[
\text{Vision Domain}
\times
\text{Language Domain}
\]

而不是只报一个最终模型。

这是论文最重要的研究贡献。

---

## C3 — Text Granularity Study

研究：

- class prompt
- generic caption
- domain caption
- structured caption
- hierarchical text

对：

- zero-shot classification
- retrieval
- dense localization

分别有什么影响。

即使没有新结构，这个实验本身也可以形成有价值结论。

---

## C4 — Metadata-Assisted Multi-Granular Alignment（推荐扩展）

如果时间允许：

\[
L = L_g + \lambda L_l
\]

利用 land-cover / region metadata 引导局部 patch-language alignment。

它是最适合成为“方法创新”的部分。

---

# 7. Model Architecture

## 7.1 Baseline Architecture

```text
                  Remote-Sensing Image
                           │
                           ▼
               Frozen DINOv3 ViT-L/16
                           │
           ┌───────────────┴───────────────┐
           │                               │
          CLS                          Patch Tokens
           │                               │
           └───────────────┬───────────────┘
                           │
                2 Trainable ViT Blocks
                           │
                           ▼
                    Image Projection
                           │
                           ▼
                     z_image (2048)
                           │
                     Contrastive Loss
                           │
                           ▲
                     z_text (2048)
                           │
                    Text Projection
                           │
                           ▲
         DINO.txt Text Transformer / Adapted Text Tower
                           ▲
                           │
              Remote-Sensing Description
```

---

## 7.2 推荐训练参数状态

| 模块 | 状态 |
|---|---|
| DINOv3 backbone | Frozen |
| 2 vision alignment blocks | Train |
| vision projection | Train |
| text backbone | Frozen / LoRA / last-K |
| text projection | Train |
| logit scale | Train |

避免 full fine-tuning。

---

## 7.3 为什么推荐 ViT-L 而不是 7B

ViT-L：

- 300M 参数；
- SAT 与 Web 两个版本都有；
- 视觉 backbone 可以冻结；
- 足以支持高质量 patch representations；
- controlled experiment 完整。

7B：

- 约 6.7B；
- 即使冻结也明显提高 inference / feature extraction 成本；
- 对 24–48 GB 单卡环境不够友好；
- 对毕业研究的问题验证没有必要。

因此：

> **ViT-L/16 应作为整个项目统一 backbone size。**

---

# 8. 三个候选模型版本

## Model A — Minimum Viable Thesis

### Domain-LiT DINOv3

只进行 domain image-text contrastive alignment。

```text
Frozen DINOv3
      +
2 alignment blocks
      +
Adapted dino.txt text encoder
      +
InfoNCE
```

目的：

验证 RQ1。

### 优点

- 最稳；
- 最容易完成；
- 足够作为毕业论文主体；
- 可在早期得到完整实验结果。

---

## Model B — Recommended Final Model

### Structured Domain DINO.txt

在 Model A 基础上加入 text granularity：

```text
Image
  ↔
Natural Caption
  +
Structured Remote-Sensing Caption
```

或者 multi-view text：

\[
L =
L(I,T_{\text{natural}})
+
\alpha L(I,T_{\text{structured}})
\]

主要验证 RQ2。

---

## Model C — Stretch Model

### Multi-Granular DINO.txt

同时：

```text
global image ↔ global caption

local patches ↔ local semantic terms
```

Loss：

\[
L =
L_{\text{global}}
+
\lambda L_{\text{local}}
\]

主要验证 RQ3。

---

# 9. 数据集规划

---

## 9.1 ChatEarthNet — 首选 Domain Alignment Dataset

### Paper

**ChatEarthNet: a global-scale image–text dataset empowering vision–language geo-foundation models**

Earth System Science Data, 2025.

https://essd.copernicus.org/articles/17/1245/2025/

### 数据量

- **163,488 Sentinel-2 images**
- 每张生成一条较长 caption
- 另有：
  - **10,000 images**
  - GPT-4V detailed captions

### 特点

caption 不只是普通 object description，而包含：

- land-cover classes；
- quantity；
- spatial distribution；
- geographic structure。

这使它特别适合：

> structured / multi-granular domain text alignment。

### 推荐用途

**主训练集。**

---

## 9.2 RS5M

### Paper

**RS5M and GeoRSCLIP: A Large-Scale Vision-Language Dataset and a Large Vision-Language Model for Remote Sensing**

IEEE TGRS 2024.

DOI：

https://doi.org/10.1109/TGRS.2024.3449154

代码：

https://github.com/om-ai-lab/RS5M

### 数据量

约：

**5,000,000 image-text pairs**

### 数据来源

包括：

- 过滤已有 image-text dataset；
- 对 label-only remote-sensing dataset 生成 caption。

### 本项目不建议一开始全部训练

推荐做 data-scale experiment：

```text
10k
50k
100k
500k
```

如果后期资源足够再扩大。

---

## 9.3 NWPU-Captions

### 数据量

- 31,500 images
- 45 scene classes
- 5 captions / image
- 157,500 sentences

### 特点

人工 caption，词汇比很多早期 RS caption dataset 更丰富。

### 推荐用途

- retrieval training/evaluation；
- auxiliary alignment；
- domain-generalization evaluation。

---

## 9.4 RSICD

### 数据量

- 10,921 images
- 30 scene classes
- 通常每张约 5 条 caption/reference sentences

### 推荐用途

- image-text retrieval evaluation；
- cross-dataset generalization。

---

## 9.5 RSITMD

### 数据量

- 4,743 images
- 每图 5 captions
- 还具有细粒度 keyword information

### 推荐用途

尤其适合：

> fine-grained text-image retrieval。

---

## 9.6 EuroSAT

用于：

**Zero-Shot Scene Classification**

优点：

- 小；
- 类别直观；
- remote-sensing VLM 中非常常见；
- 容易快速得到 diagnostic result。

---

## 9.7 AID

用于：

**Zero-Shot Classification**

主要特点：

- aerial scene；
- 类别粒度比 EuroSAT 更细。

可以检验：

> model 是否只学 land cover，还是获得更细场景语义。

---

## 9.8 NWPU-RESISC45

用于：

**Zero-Shot Classification**

- 31,500 images
- 45 classes

由于 NWPU-Captions 与其存在关联，需要特别避免 train/test leakage。

如果训练使用 NWPU-Captions，则不应直接把同源 NWPU-RESISC45 当完全独立 zero-shot test。

---

## 9.9 LoveDA

### 数据量

- **5,987 RGB images**
- 1024×1024
- 0.3 m spatial resolution
- urban + rural domains
- 7 classes：
  - background
  - building
  - road
  - water
  - barren/bare land
  - forest
  - agriculture

### 推荐用途

**dense/open-vocabulary segmentation evaluation**

尤其适合研究：

- urban vs rural generalization；
- patch-level semantic localization；
- text prompt segmentation。

---

# 10. Dataset Role Matrix

| Dataset | Train Alignment | Classification | Retrieval | Dense | Domain Shift |
|---|---:|---:|---:|---:|---:|
| ChatEarthNet | ★★★★★ |  |  | ★★ | ★★★ |
| RS5M | ★★★★ |  |  | ★★ | ★★★ |
| NWPU-Captions | ★★★ | ★★ | ★★★★★ |  | ★★★ |
| RSICD | ★★ |  | ★★★★★ |  | ★★★ |
| RSITMD | ★★ |  | ★★★★★ |  | ★★★ |
| EuroSAT |  | ★★★★★ |  |  | ★★★ |
| AID |  | ★★★★ |  |  | ★★★★ |
| NWPU-RESISC45 | 注意数据泄漏 | ★★★★ |  |  | ★★★ |
| LoveDA |  |  |  | ★★★★★ | ★★★★★ |

---

# 11. Baselines

## B0 — CLIP

基础通用 VLM baseline。

目的：

证明：

> remote sensing domain 中 general CLIP 的能力与局限。

---

## B1 — RemoteCLIP

### Paper

**RemoteCLIP: A Vision Language Foundation Model for Remote Sensing**

IEEE TGRS, 2024.

DOI：

https://doi.org/10.1109/TGRS.2024.3390838

Project：

https://remotefamily.github.io/RemoteCLIP.html

### 意义

这是最重要的遥感 VLM baseline 之一。

其核心问题与本项目相似：

> generic vision-language model 如何进入 remote sensing domain。

区别：

RemoteCLIP 的 backbone/范式属于 CLIP 系；

本项目研究的是：

> DINOv3 self-supervised visual space 如何获得 domain language interface。

---

## B2 — GeoRSCLIP

与 RS5M 配套。

意义：

- 大规模 remote-sensing VLM；
- 支持 ZSC、retrieval、semantic localization；
- 是 domain text alignment 的强 baseline。

---

## B3 — DINOv2 + dino.txt

### Anchor baseline

CVPR 2025。

作用：

直接回答：

> DINOv3 相较于 DINOv2 在 domain alignment 上有什么变化？

---

## B4 — Official DINOv3.txt

作为：

**最重要通用 DINO baseline。**

Web DINOv3 ViT-L + official dino.txt。

---

## B5 — SAT DINOv3 + Linear Probe（仅视觉 diagnostic）

这不是 VLM baseline，而是 diagnostic baseline。

作用：

验证：

> SAT-DINOv3 的视觉特征本身有多强？

防止把视觉 backbone 的收益误认为 language alignment 的收益。

---

# 12. 关键相关工作

---

## 12.1 Talking to DINO / Talk2DINO

### Paper

**Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation**

ICCV 2025.

https://openaccess.thecvf.com/content/ICCV2025/html/Barsellotti_Talking_to_DINO_Bridging_Self-Supervised_Vision_Backbones_with_Language_for_ICCV_2025_paper.html

### 方法

- Frozen DINOv2
- Frozen CLIP
- learned mapping
- CLIP text embeddings → DINO patch feature space
- 利用 DINO attention map 筛选 local patch alignment

### 与本项目关系

它提供另一种范式：

```text
不是训练新的语言塔与 DINO 对齐，
而是将现有 CLIP language space 映射到 DINO space。
```

可作为：

- 方法设计参考；
- local alignment 参考；
- lightweight baseline 思路。

---

## 12.2 DINO Soars / CAFe-DINO

### Paper

**DINO Soars: DINOv3 for Open-Vocabulary Semantic Segmentation of Remote Sensing Imagery**

CVPR Workshops 2026.

https://openaccess.thecvf.com/content/CVPR2026W/MORSE/html/Faulkenberry_DINO_Soars_DINOv3_for_Open-Vocabulary_Semantic_Segmentation_of_Remote_Sensing_CVPRW_2026_paper.html

### 方法重点

- DINOv3
- DINO.txt
- remote-sensing OVSS
- cost aggregation
- feature upsampling

### 非常重要的边界

这篇论文已经说明：

> “DINOv3 + dino.txt 做遥感 open-vocabulary segmentation”

本身已经不是新题目。

所以本项目不能把 contribution 写成：

> 首次把 DINOv3 用于遥感 OVS。

本项目与其区别应该明确：

### CAFe-DINO

重点：

> 如何用现成 DINOv3.txt 进行 RS dense prediction。

### 本项目

重点：

> 如何重新构建 **domain-matched DINOv3-language space**。

---

## 12.3 DinoSplat-OV

### Paper

**Standalone DINOv3 for Training-Free Open-Vocabulary Semantic Segmentation in Remote Sensing**

arXiv:2608.03023, 2026-08.

https://arxiv.org/abs/2608.03023

### 核心

training-free：

- text-aware Laplacian propagation；
- Gaussian splatting upsampling；
- sliding-window；
- dense remote-sensing imagery。

### 对本项目的启示

进一步确认：

> downstream DINOv3 OVSS 已经成为快速发展方向。

因此本项目更应该把研究位置放在：

**upstream/domain alignment**

而不是只做 segmentation 后处理。

---

## 12.4 PALADIN

### Paper

**PALADIN: Prompt-Aligned Localization and Anomaly Detection with DINOv3**

CVPR Workshops 2026.

https://openaccess.thecvf.com/content/CVPR2026W/VAND/html/Basaran_PALADIN_Prompt-Aligned_Localization_and_Anomaly_Detection_with_DINOv3_CVPRW_2026_paper.html

### 方法

- Frozen DINOv3
- Frozen CLIP text encoder
- lightweight Cross-Modal Alignment Adapters
- patch-level DINO feature ↔ CLIP text space
- industrial anomaly localization

### 为什么重要

它证明：

> lightweight alignment of DINOv3 with text is a viable applied research direction。

同时也意味着：

工业异常检测已经有非常直接的相邻工作，因此相比 industrial domain，remote sensing domain 更适合作为本毕业项目主方向。

---

# 13. 推荐 Loss Design

## 13.1 Global Image-Text Contrastive Loss

标准 symmetric InfoNCE：

\[
L_{i2t}
=
-\frac{1}{N}
\sum_i
\log
\frac
{\exp(s(v_i,t_i)/\tau)}
{\sum_j \exp(s(v_i,t_j)/\tau)}
\]

\[
L_{t2i}
=
-\frac{1}{N}
\sum_i
\log
\frac
{\exp(s(t_i,v_i)/\tau)}
{\sum_j \exp(s(t_i,v_j)/\tau)}
\]

\[
L_g
=
\frac{1}{2}
(L_{i2t}+L_{t2i})
\]

---

## 13.2 Multi-Text Loss

同一 image 同时拥有：

- natural caption
- structured caption
- class prompt

可以：

\[
L_{\text{text}}
=
L(I,T_n)
+
\alpha L(I,T_s)
+
\beta L(I,T_c)
\]

不建议一开始全部加入。

先分别训练/比较，再考虑联合。

---

## 13.3 Local Semantic Alignment

设 patch token：

\[
P =
\{p_1,\dots,p_M\}
\]

local concept embedding：

\[
E =
\{e_1,\dots,e_K\}
\]

可以计算：

\[
A_{mk}
=
\operatorname{cos}(p_m,e_k)
\]

如果有弱 region supervision：

\[
L_l
=
CE(A,Y_{\text{weak}})
\]

最终：

\[
L =
L_g
+
\lambda L_l
\]

---

# 14. Training Strategy

## Stage 0 — Reproduce Inference

目标：

跑通：

- official DINOv3
- official dino.txt
- SAT DINOv3

输出：

- embedding
- patch similarity heatmap
- zero-shot classification

这是 smoke test。

---

## Stage 1 — Reproduce Official DINO.txt Evaluation

不要求完全复现大规模训练。

只需要：

- 官方预训练权重；
- 跑标准 inference；
- 验证 pipeline 正确。

---

## Stage 2 — Domain Alignment Prototype

数据：

ChatEarthNet 10k subset。

模型：

- Frozen Web DINOv3 ViT-L
- Frozen/LoRA text encoder
- train vision head + projection

如果能显著改变 retrieval / similarity distribution，即证明 pipeline 可行。

---

## Stage 3 — Full Domain Alignment

ChatEarthNet：

```text
163k
```

或者：

```text
100k–160k
```

形成主模型。

---

## Stage 4 — SAT Backbone

将：

```text
LVD-1689M ViT-L
```

替换为：

```text
SAT-493M ViT-L
```

重新训练 alignment module。

得到 RQ1 主实验。

---

## Stage 5 — Text Granularity

进行：

- labels
- natural captions
- structured captions
- hierarchical captions

ablation。

---

## Stage 6 — Optional Local Alignment

只有前五个阶段完成后再进入。

---

# 15. Parameter-Efficient Adaptation

## 15.1 推荐优先级

### Option A

冻结 text transformer，只训练 projection。

优点：

- 最省算力。

缺点：

- language domain adaptation 能力有限。

---

### Option B — 推荐

LoRA text transformer。

例如：

- Q/V LoRA
- rank 8 / 16
- last 6–12 layers

这是推荐主方案。

---

### Option C

只解冻最后 K 层：

```text
K = 2 / 4 / 6
```

优点：

简单、容易解释。

---

### Option D

Full text tuning。

作为资源允许时的 upper bound。

不建议作为默认。

---

# 16. 算力规划

## 16.1 原则

绝对不建议：

- 从头训练 DINOv3；
- 重做 SAT-493M；
- 训练 7B backbone；
- 完全复制官方大规模 dino.txt pretraining。

研究问题并不需要这些。

---

## 16.2 推荐配置

### GPU

可接受：

- 1 × 24GB
- 1 × 48GB
- 2 × 24GB
- 2 × 48GB

### Backbone

ViT-L/16 300M。

### Precision

bf16。

### Backbone

Frozen。

### Training image resolution

初期：

```text
224 / 256
```

dense evaluation 后期再：

```text
448 / 512
```

---

## 16.3 降低算力的关键技巧

### 1. Precompute Frozen Visual Features

如果 backbone 完全冻结：

可以预先提取：

- CLS
- patch tokens

对于只训练 alignment/text module 的实验可以极大节省训练时间。

注意：

如果 visual alignment blocks 需要 backbone 的最后若干中间层，则预存对应 feature。

---

### 2. Gradient Accumulation

例如：

```text
physical batch = 16
accumulate = 8
effective batch = 128
```

contrastive learning 对 batch size 敏感，因此可：

- gradient accumulation；
- cross-GPU gather；
- memory queue。

---

### 3. LoRA

显著降低 optimizer states 和 gradient memory。

---

### 4. Stage-wise Dataset Scaling

先：

```text
10k
```

再：

```text
50k
```

确认趋势后再：

```text
160k
```

避免在错误方法上浪费数天 GPU。

---

# 17. Evaluation Design

本项目不要使用单一 benchmark 作为结论。

建议三层 evaluation。

---

## Level 1 — Global Semantic Understanding

### Zero-Shot Classification

datasets：

- EuroSAT
- AID
- NWPU-RESISC45（注意 leakage）

metrics：

- Top-1 Accuracy
- Mean per-class Accuracy

研究问题：

> domain alignment 是否真的改变了视觉—语言语义空间？

---

## Level 2 — Cross-Modal Alignment

### Image → Text Retrieval

metrics：

- R@1
- R@5
- R@10
- Mean Recall

### Text → Image Retrieval

同样：

- R@1
- R@5
- R@10

datasets：

- RSICD
- RSITMD
- NWPU-Captions

这是最直接的 language alignment evaluation。

---

## Level 3 — Dense Semantic Understanding

### Open-Vocabulary Semantic Segmentation

dataset：

LoveDA。

metrics：

- mIoU
- mean accuracy
- per-class IoU

但这里重点不必是击败 CAFe-DINO。

重点：

> domain text alignment 是否改变 patch-level semantics？

---

# 18. Interpretability / Diagnostic Evaluation

这一部分很适合研究报告。

## 18.1 Text-Patch Heatmap

输入：

```text
water
road
building
forest
dense residential area
```

计算：

\[
s_{ik}
=
cos(p_i,t_k)
\]

再 reshape 成 patch grid。

比较：

- official DINOv3.txt
- Web-domain aligned
- SAT-domain aligned
- structured-text aligned

---

## 18.2 Embedding Visualization

使用：

- PCA
- UMAP
- t-SNE（辅助）

观察：

```text
water / river / sea
building / residential
forest / agriculture
road / runway
```

的视觉和文本 embedding 是否靠近。

---

## 18.3 Confusion Analysis

重点分析：

### Easy

- water
- forest
- airport

### Similar

- dense residential
- medium residential
- commercial
- industrial

### Fine-grained

- road
- runway
- railway

真正解释：

> domain language alignment 改善的是哪种语义？

---

# 19. Ablation Matrix

这是论文主体之一。

---

## A1 — Backbone Domain

| ID | Backbone |
|---|---|
| A1 | Web DINOv3 |
| A2 | SAT DINOv3 |

---

## A2 — Language Domain

| ID | Training Text |
|---|---|
| B1 | official generic dino.txt |
| B2 | RS natural captions |
| B3 | RS structured captions |

---

## A3 — Text Adaptation

| ID | Text Encoder |
|---|---|
| C1 | frozen |
| C2 | projection only |
| C3 | LoRA |
| C4 | last-K |
| C5 | full tuning |

不需要全部跑。

主论文：

```text
frozen / LoRA / last-K
```

即可。

---

## A4 — Data Scale

```text
10k
50k
100k
160k
```

观察：

\[
Performance = f(N_{\text{pairs}})
\]

重点：

> domain alignment 是否 data-efficient？

---

## A5 — Text Granularity

```text
class name
natural caption
structured caption
hierarchical caption
```

---

## A6 — Image Representation

参考 dino.txt：

```text
CLS only
mean patch only
CLS + mean patch
```

这是必须做的 anchor-paper reproduction ablation。

---

## A7 — Local Alignment

```text
global only
global + weak local
```

如果时间允许。

---

# 20. 最小实验矩阵

不要一开始跑几十个组合。

推荐主表：

| Model | Vision | Text Train | Text Tune | Global | Local |
|---|---|---|---|---|---|
| M0 | Web | generic | official | ✓ | native |
| M1 | Web | RS caption | LoRA | ✓ | × |
| M2 | SAT | RS caption | LoRA | ✓ | × |
| M3 | SAT | structured | LoRA | ✓ | × |
| M4 | SAT | structured | LoRA | ✓ | ✓ |

这五个模型已经足以形成完整 story。

---

# 21. Acceptance Criteria

本项目不要预先设：

```text
必须超过 SOTA 3%
```

建议按以下验收。

---

## A. Engineering

必须实现：

- DINOv3 feature extraction
- dino.txt inference
- domain alignment training
- zero-shot evaluation
- retrieval evaluation
- dense visualization/evaluation
- checkpointing
- logging
- reproducible config

---

## B. Research

至少回答：

### Q1

SAT backbone vs Web backbone 的 domain alignment 差异。

### Q2

generic vs remote-sensing-specific language 的差异。

### Q3

不同 text granularity 的效果。

---

## C. Analysis

必须包含：

- ablation
- failure case
- qualitative heatmap
- compute cost
- data scale
- domain shift discussion

---

# 22. “论文成功”的定义

以下任意一种结果都可以成为有价值结论。

---

## 情况 A

SAT + RS text 全面最好。

结论：

> visual-domain and language-domain matching are complementary.

---

## 情况 B

SAT visual backbone 提升明显，但 RS text 收益很小。

也非常有价值：

> strong domain-specific self-supervised visual pretraining may reduce the dependence on domain-specific linguistic supervision.

---

## 情况 C

RS text 对 retrieval 很强，但 segmentation 无明显收益。

结论：

> global semantic alignment and dense semantic alignment are not equivalent.

---

## 情况 D

structured caption 对 dense task 好，但自然 caption 对 retrieval 好。

这是非常漂亮的结论：

> different downstream tasks require different language granularity.

---

## 情况 E

SAT backbone 在某些 aerial high-resolution dataset 不如 Web backbone。

同样很重要：

> domain pretraining itself may introduce a new domain bias due to differences in source imagery, resolution and data distribution.

所以项目并不依赖某个固定“提升”。

---

# 23. 风险分析

## Risk 1 — SAT backbone 与现有 dino.txt head 不兼容

这反而正是研究问题的一部分。

解决：

- 重新初始化 visual alignment blocks；
- 保留 text encoder；
- 在 domain dataset 重新 alignment。

---

## Risk 2 — Contrastive batch 太小

解决：

- gradient accumulation；
- distributed all-gather；
- feature queue；
- frozen feature precomputation。

---

## Risk 3 — Caption 质量噪声大

解决：

比较：

- raw caption
- filtered caption
- structured caption
- label-only prompt

caption 清洗本身可以成为 ablation。

---

## Risk 4 — Dense task 过于复杂

最低论文版本不依赖 dense SOTA。

可以：

- semantic localization；
- patch-text heatmap；
- lightweight segmentation evaluation；

作为补充。

---

## Risk 5 — Local alignment 做不出来

直接放弃 C4。

C1 + C2 + C3 已经可以构成完整毕业报告。

---

# 24. 4–6 个月时间计划

---

## Month 1 — Reproduction & Dataset

### Week 1–2

阅读：

- DINOv3
- DINOv2 Meets Text
- RemoteCLIP
- RS5M / GeoRSCLIP
- Talk2DINO

实现：

- DINOv3 inference
- dino.txt inference

### Week 3–4

完成：

- ChatEarthNet preprocessing
- RSICD evaluation loader
- EuroSAT evaluation
- feature extraction cache

### Milestone

得到：

```text
official DINOv3.txt
→ EuroSAT zero-shot
→ RSICD retrieval
```

---

## Month 2 — Domain Alignment MVP

完成：

```text
Web DINOv3
+
RS captions
```

training。

实验：

- 10k
- 50k

调通：

- loss
- retrieval
- zero-shot

### Milestone

**Minimum viable research pipeline 完成。**

---

## Month 3 — SAT Domain Study

完成：

```text
Web + RS text
vs
SAT + RS text
```

主实验。

加入：

- data scale
- LoRA / last-K comparison

### Milestone

RQ1 基本可以回答。

此时即使项目后续失败，也已经具备论文主体。

---

## Month 4 — Text Granularity

构造：

- label prompt
- natural caption
- structured caption
- hierarchical caption

完成 RQ2。

加入：

- visualization
- retrieval analysis
- class-wise analysis

---

## Month 5 — Dense / Local Alignment

如果进度良好：

实现：

```text
global + weak local
```

LoveDA evaluation。

同时比较：

- CLS
- patch
- CLS+patch

完成 RQ3。

---

## Month 6 — Consolidation

主要不再增加新模块。

完成：

- repeat seeds
- final ablation
- failure cases
- visualization
- compute table
- thesis writing
- slides

---

# 25. Minimum Viable Thesis

如果时间只有约 4 个月：

只做：

### 1.

Official DINOv3.txt baseline

### 2.

Remote-sensing domain text alignment

### 3.

Web vs SAT

### 4.

Generic vs domain caption

### 5.

Zero-shot classification

### 6.

Image-text retrieval

### 7.

Patch-text visualization

### 8.

Data scale ablation

已经足够形成：

> 一篇完整、有问题、有方法、有实验、有分析的毕业研究报告。

---

# 26. Full Thesis

5–6 个月版本再加入：

### 1.

structured captions

### 2.

hierarchical semantics

### 3.

weak local alignment

### 4.

LoveDA open-vocabulary segmentation

### 5.

domain shift experiments

### 6.

parameter efficiency analysis

---

# 27. 推荐论文叙事结构

最终论文不建议写成：

```text
我提出 XXX block
我又提出 YYY module
我又设计 ZZZ loss
所以提升 2%
```

更推荐：

---

## Introduction

问题：

DINOv3 has strong visual representations but domain-specific language alignment remains unclear.

---

## Observation

DINOv3 exists in:

```text
Web visual space
Satellite visual space
```

but available text alignment is primarily generic.

---

## Research Question

> What should be aligned when adapting a self-supervised vision foundation model to a specialized visual domain?

---

## Method

parameter-efficient domain text alignment。

---

## Experiments

### Study 1

vision domain

### Study 2

language domain

### Study 3

text granularity

### Study 4

local/global alignment

---

## Conclusion

给出：

> practical design guidelines for domain-specific language alignment of self-supervised vision foundation models.

这比只说“我们比 baseline 高 X%”研究味更强。

---

# 28. 论文 Related Work 推荐组织

## 2.1 Self-Supervised Vision Foundation Models

- DINO
- DINOv2
- DINOv3

---

## 2.2 Vision-Language Alignment

- CLIP
- LiT
- DINOv2 Meets Text
- Talk2DINO

---

## 2.3 Remote-Sensing Vision-Language Models

- RemoteCLIP
- RS5M
- GeoRSCLIP
- ChatEarthNet

---

## 2.4 Open-Vocabulary Remote-Sensing Understanding

- CAFe-DINO / DINO Soars
- DinoSplat-OV

---

# 29. 重点论文阅读顺序

## Priority A — 必须精读

### 1. DINOv3

理解：

- architecture
- Gram anchoring
- Web vs SAT pretraining
- dense features

Official project：

https://github.com/facebookresearch/dinov3

---

### 2. DINOv2 Meets Text

CVPR 2025。

重点：

- LiT baseline
- vision head
- CLS + patch average
- dataset curation
- global vs dense performance
- training objective

---

### 3. RS5M + GeoRSCLIP

重点：

- remote-sensing text data 如何构建；
- general VLM → domain VLM；
- PEFT；
- evaluation protocol。

---

### 4. RemoteCLIP

重点：

- remote-sensing language supervision；
- retrieval；
- zero-shot。

---

### 5. ChatEarthNet

重点：

- captions 如何产生；
- land-cover metadata；
- text structure；
- 是否可以复用生成流程构造 structured caption。

---

## Priority B — 强烈建议

### 6. Talk2DINO

重点：

local patch alignment。

### 7. DINO Soars

重点：

目前 DINOv3 在 remote sensing OVSS 的直接竞争/相关工作。

### 8. DinoSplat-OV

重点：

2026 最新 DINOv3 remote-sensing training-free dense adaptation。

---

## Priority C — 方法参考

### 9. PALADIN

重点：

轻量 DINOv3 ↔ text adapter。

---

# 30. 文献列表（当前核心版本）

## DINO 系列 / Vision-Language Alignment

1. Jose, C. et al. **DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment.** CVPR 2025.  
   https://openaccess.thecvf.com/content/CVPR2025/html/Jose_DINOv2_Meets_Text_A_Unified_Framework_for_Image-_and_Pixel-Level_CVPR_2025_paper.html

2. Siméoni, O. et al. **DINOv3.** Meta AI, 2025.  
   https://ai.meta.com/blog/dinov3-self-supervised-vision-model/

3. Barsellotti, L. et al. **Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation.** ICCV 2025.  
   https://openaccess.thecvf.com/content/ICCV2025/html/Barsellotti_Talking_to_DINO_Bridging_Self-Supervised_Vision_Backbones_with_Language_for_ICCV_2025_paper.html

---

## Remote-Sensing Vision-Language

4. Liu, F. et al. **RemoteCLIP: A Vision Language Foundation Model for Remote Sensing.** IEEE TGRS, 2024.  
   https://doi.org/10.1109/TGRS.2024.3390838

5. Zhang, Z. et al. **RS5M and GeoRSCLIP: A Large-Scale Vision-Language Dataset and a Large Vision-Language Model for Remote Sensing.** IEEE TGRS, 2024.  
   https://doi.org/10.1109/TGRS.2024.3449154

6. **ChatEarthNet: a global-scale image–text dataset empowering vision–language geo-foundation models.** ESSD, 2025.  
   https://essd.copernicus.org/articles/17/1245/2025/

---

## DINOv3 + Remote Sensing / Applied Alignment

7. Faulkenberry, R., Prasad, S. **DINO Soars: DINOv3 for Open-Vocabulary Semantic Segmentation of Remote Sensing Imagery.** CVPR Workshops 2026.  
   https://openaccess.thecvf.com/content/CVPR2026W/MORSE/html/Faulkenberry_DINO_Soars_DINOv3_for_Open-Vocabulary_Semantic_Segmentation_of_Remote_Sensing_CVPRW_2026_paper.html

8. Zhao, C. et al. **Standalone DINOv3 for Training-Free Open-Vocabulary Semantic Segmentation in Remote Sensing.** arXiv:2608.03023, 2026.  
   https://arxiv.org/abs/2608.03023

9. Basaran, A. **PALADIN: Prompt-Aligned Localization and Anomaly Detection with DINOv3.** CVPR Workshops 2026.  
   https://openaccess.thecvf.com/content/CVPR2026W/VAND/html/Basaran_PALADIN_Prompt-Aligned_Localization_and_Anomaly_Detection_with_DINOv3_CVPRW_2026_paper.html

---

# 31. 与课程 PPT 的关系

课程给出的 **AI Applications: State of the Art (2024–2026)** 中：

- Computer Vision & Image Analysis 被列为七大应用方向之一；
- 其中 open problems 明确包含：
  - long-tail objects
  - domain shift
  - annotation cost
- DINOv3 被作为 2025 vision foundation model 的 case study；
- PPT 强调其 frozen features 在 detection / segmentation 等任务的强泛化能力。

因此本选题很好地从课程给出的 broad direction 收敛到：

> **foundation vision model + domain shift + low-label/domain language adaptation**

不是脱离课程范围另起炉灶。

---

# 32. 当前最推荐的开题表述

如果现在需要和导师用 1–2 分钟解释，可以概括为：

> DINOv3 是一个非常强的纯视觉自监督基础模型，并且官方不仅提供通用 Web 版本，也提供了专门在卫星图像上训练的 SAT-493M 版本。另一方面，CVPR 2025 的 DINOv2 Meets Text 已经证明，可以冻结 DINO backbone 并通过低成本文本对齐，使 DINO 同时获得 zero-shot classification 和 dense open-vocabulary 能力。DINOv3 官方也延续了 dino.txt，但目前公开的语言对齐主要围绕 Web backbone，而并没有系统回答：一个已经在遥感视觉域预训练的 DINOv3 是否仍需要遥感专业文本重新对齐、什么样的文本最有效、全局和局部语义应该怎样对齐。我的项目计划以遥感为 specific domain，通过 Web/SAT visual backbone 与 generic/domain language 的 controlled study，研究低资源的 DINOv3 领域文本对齐，并进一步尝试 structured caption 或 metadata-assisted local alignment。项目主要以问题验证、系统实现、消融和失败分析作为验收，不要求重新预训练 DINOv3 或达到 SOTA。

---

# 33. 当前项目边界：明确不做什么

为了保证 4–6 个月完成：

## 不做

- DINOv3 7B pretraining
- SAT-493M reproduction
- 从零构造 million-scale foundation dataset
- 重新设计完整视觉 backbone
- 完整训练一个大语言模型
- 与所有遥感 SOTA 全面比较
- 同时做 classification / detection / segmentation / caption / VQA 五种任务

## 做

- domain alignment
- PEFT
- controlled experiments
- retrieval
- zero-shot
- dense semantics
- interpretability
- failure analysis

---

# 34. 下一步工作清单

## Step 1

精读：

```text
DINOv2 Meets Text
DINOv3 text module source code
```

把：

- model classes
- trainable parameters
- input/output shape
- loss
- training dataset

逐项还原。

---

## Step 2

确认：

```text
SAT-493M ViT-L
+
official dino.txt visual/text heads
```

直接替换 backbone 时会出现什么表现。

这可以成为第一组非常有价值的 diagnostic experiment。

---

## Step 3

下载：

- ChatEarthNet metadata/sample
- EuroSAT
- RSICD

先做 10k training prototype。

---

## Step 4

形成代码结构：

```text
project/
├── configs/
├── datasets/
├── models/
│   ├── dinov3_backbone.py
│   ├── text_encoder.py
│   └── domain_alignment.py
├── losses/
├── train/
├── eval/
│   ├── zero_shot.py
│   ├── retrieval.py
│   └── localization.py
├── visualization/
└── scripts/
```

---

## Step 5

先冻结论文 baseline：

```text
Official DINOv3.txt
RemoteCLIP
GeoRSCLIP
Web Domain-DINO
SAT Domain-DINO
```

之后不要无限增加 baseline。

---

# 35. 一句话总结

这个毕业项目最值得研究的不是：

> **“DINOv3 能不能识别遥感文本？”**

而是：

> **“当一个强自监督视觉基础模型已经进入专业视觉域后，我们还需要怎样的语言监督，才能以有限数据和算力建立真正匹配该领域的视觉—语言空间？”**

这个问题同时具备：

- **Feasibility**
- **Usefulness**
- **Technical Depth**
- **Innovation**

并且失败路径可控，非常适合作为 4–6 个月的研究生毕业研究报告。

---

# 36. 信息可信度说明

本文档将内容分为两类理解：

## 已由公开论文/官方资料支持的事实

包括但不限于：

- DINOv3 Web/SAT backbone 类型和规模；
- SAT-493M 基本信息；
- official dino.txt 代码结构；
- DINOv2 Meets Text 核心方法；
- RS5M 约 5M image-text pairs；
- ChatEarthNet 163,488 images + 10,000 GPT-4V detailed captions；
- NWPU-Captions / RSICD / LoveDA 等公开数据集规模；
- Talk2DINO、CAFe-DINO、DinoSplat-OV、PALADIN 的论文定位。

## 本项目提出的研究假设/建议

以下不是现有论文已经证明的结论，而是本毕业项目拟验证的内容：

- SAT visual backbone 与 RS-specific language 是否存在协同；
- structured captions 是否优于 natural captions；
- weak local alignment 是否值得加入；
- official generic text space 是否会限制 SAT-DINOv3；
- domain alignment 是否能以少量 image-text pairs 获得稳定收益。

这些内容应通过实验验证，而不能在最终论文中预先写成事实。
