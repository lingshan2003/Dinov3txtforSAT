# Full-maskpos 后续实验与容量、数据诊断

日期：2026-10-06。代码与配置已准备；本地未启动真实 GPU 训练。

后续用户提出优先适配原生DINOv3.txt；目前启动优先级见[原生适配精简方案](WEB_NATIVE_ADAPTATION_PLAN_2026-10-06.md)。本文SAT长矩阵保留为可选探索，不要求先执行。

本地验证：全量372项测试、Ruff、Bash语法、CLI帮助及diff检查通过；新增测试包括3epoch编排/报告路径/无权重打包/完成跳过，以及两种新模块组合的CPU真实参数更新与冻结检查。

## 1. 这轮实际运行什么

继续使用已审阅的 full-maskpos：冻结 SAT ViT-L/16、adapter256、同 caption 非对角正例屏蔽、queue0、无增强。
训练仍是 337,199 张图、36,495 个 caption 组，保持原 train/val 边界。每个图片 epoch 完整访问训练图一次。
下面每组都从共同官方权重重新开始；不接续上一轮 cosine 已降到零的终点权重。

| 名称 | 实际 batch × 梯度累计 | 图片 epoch | optimizer steps | 可训练范围 | 主要比较 |
| --- | ---: | ---: | ---: | --- | --- |
| 历史 full-maskpos | 16 × 4 | 1 | 5,269 | adapter + 全文本 LoRA（含 projection） | 已完成参照 |
| batch32_1epoch | 32 × 2 | 1 | 5,269 | 同上 | 与历史比较实际对比候选规模 |
| epochs3_batch16 | 16 × 4 | 3 | 15,807 | 同上 | 与历史比较更长完整图片训练 |
| head_lora_1epoch | 32 × 2 | 1 | 5,269 | vision head 全量 + adapter + 全文本 LoRA | 与 batch32_1epoch 比较开放 head |
| fulltext4_1epoch（可选） | 32 × 2 | 1 | 5,269 | vision head 全量 + adapter + 文本末4层/ln_final 全量 + projection 全量 | 探索更强更新范围 |

默认前三组；`--include-strong` 加入第四组。第四组同时改变文本更新范围和学习率，是容量探索，不能归因为某一个文本模块。
三 epoch 的 warmup 为1,581步，其他组527步，之后各自 cosine 到终点。它比较完整较长日程，不是只增加数据而保持所有学习率轨迹相同。
3 epoch 精确图片曝光1,011,597次；1 epoch精确337,199次，末批保留，不能用 steps×64当作精确曝光数。
梯度累计只扩大一次更新累计的样本量，16×4不会自动提供64个对比候选。batch32训练仍屏蔽精确同文正例，实际有效负例数由组重复决定。

各组 module LR：

| 组 | adapter | vision head | 文本 Transformer | projection |
| --- | ---: | ---: | ---: | ---: |
| batch32 / epochs3 | 1e-4 | 冻结 | LoRA 1e-4 | LoRA 1e-4 |
| head_lora | 1e-4 | 1e-5 | LoRA 1e-4 | LoRA 1e-4 |
| fulltext4 | 1e-4 | 1e-5 | 末4层及ln_final 5e-6 | 全量1e-5 |

可训练参数量按历史官方模块元数据估算：当前 adapter+LoRA约501万；开放 head后约3,034万；
第四组约1.077亿（含约7,870万末4层/ln_final和262万projection）。以实际服务器 optimizer 分组日志为最终值。
LoRA已经更新全部24个文本 block及projection；低秩约束不保证原有语义能力保持。
本轮仍冻结SAT视觉 backbone、文本embedding/位置编码和logit_scale；第四组还冻结前20个文本block。
暂不随机重置视觉head或开放SAT backbone，避免把更新范围和初始化同时改变。

## 2. 在服务器启动

先同步本轮代码和四份配置；沿用已安装的图片池、manifests、官方模型资产、RSICD-val，不需要重新上传原ZIP。

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
bash scripts/run_sat_maskpos_next.sh --preflight-only
tmux new -s sat-maskpos-next
bash scripts/run_sat_maskpos_next.sh
```

若希望同时执行更强文本配置，预检和正式运行都加同一个开关：

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
bash scripts/run_sat_maskpos_next.sh --preflight-only --include-strong
tmux new -s sat-maskpos-strong
bash scripts/run_sat_maskpos_next.sh --include-strong
```

这两种启动方式选其一，不要并发运行同一批输出目录。先跑默认三组，再用`--include-strong`重跑也可以；完整三组会通过身份/覆盖检查后跳过。
`Ctrl-b`然后`d`离开tmux；`tmux attach -t sat-maskpos-next`返回（强组会话对应sat-maskpos-strong）。

支持`--train-only`和`--evaluate-only`。默认顺序训练完各组后，评测每组step0/best/latest在原unique-val、caption-group-val、RSICD-val三池的全局检索。
默认27份检索报告，含强组36份。组验证同时输出按图与按caption组均衡指标；保留原unique-val loss选择best，并比较完整覆盖latest。
中断后在同一代码/配置下重跑同一命令，未完成组从latest恢复，完整组和通过身份校验的检索报告跳过；异常立即停止。
预检会检查资产、split边界、池审计与实际图片数预算，GPU启动前完成全部选中配置检查。
预检不执行GPU前向，因此不能证明batch32和更强配置显存足够；不要悄悄改batch后继续旧目录。

仍每200step验证并强制step0及训练终点验证；每个实验长期只保留以下三个权重：

```text
step_0000000.pt
best.pt
latest.pt
```

新增head和文本全量更新会增大单个checkpoint及AdamW状态，固定三份仅控制数量，不能保证沿用小LoRA实验的磁盘占用。
报告包不含权重和图片；原子写入checkpoint时还需要临时文件空间。

默认完成后下载：

```text
/root/autodl-tmp/Dinov3txtforSAT/outputs/sat_maskpos_next_seed11_reports.tar.gz
```

包内包含各组config、训练日志、metrics、provenance、training_summary、数据audit和检索报告；共同汇总位于outputs/sat_maskpos_next_seed11/summary.json。
这轮仍是seed11开发验证，未执行最终test。先看同候选池R@1/5/10、按组指标、RSICD，以及训练覆盖/参数分组，再决定跨seed复现。

## 3. 11% mean Recall 与0.9 loss如何理解

历史full-maskpos latest的原unique-val MR为11.1467%，按图多图MR11.3392%，按组均衡MR9.8190%，RSICD MR7.0658%。
unique-val双向R@1仅2.6387%/3.4032%；绝对对齐能力仍弱。MR是双向R@1、R@5、R@10六项的平均，不是分类准确率。
原池有4,055图/4,055文，多图池有33,118图/4,055文，候选数量和正例定义必须同时说明，不能与任意论文的Recall直接比较。

验证loss约0.8478是物理batch16的单正例对比损失，不是在全部4,055文本上计算的损失。
均匀预测在16候选下CE约ln(16)=2.7726（尾批略有不同）；所以它表明学到了区分，但无法据此断言全局检索已接近上限。
单正例CE下exp(-L)可理解为正确配对softmax概率的几何平均，L=0.9对应约0.407，既不是命中率也不是平均概率。
maskpos训练分母屏蔽同文正例，验证不屏蔽；训练loss和验证loss不直接等价。均匀多正例目标还存在ln(m)目标熵，不能拿其loss绝对值直接与maskpos比较。
扩大实际batch会改变对比任务难度及loss的基准，本轮各组验证batch仍固定16，训练loss不用于跨batch排名。
cosine训练末尾学习率近零，曲线变平也不能单独证明模型容量极限。

目前近零step0结果来自SAT骨干接入原通用dino.txt head/text的联合输出，不能推导出SAT骨干本身没有视觉识别能力。
潜在瓶颈包括跨骨干特征空间错配、小批有效负例不足、标签语义粒度、低秩更新容量以及训练日程；这轮分别给出可比较的实验，不承诺开放参数必然提高。

## 4. 更丰富描述的数据候选

caption需要增加图上可核实的对象、属性、数量、空间关系，而不是靠模板填满77 tokens。
当前恢复37万图仍只有40,550种完整caption，训练部分只有36,495种文本；增加实例不会自动增加文字信息量。
77是token上限，不是77个单词。新数据需要用现有BPE实测长度、截断比例和丢失的信息；不能静默延长位置编码。

优先级建议：同图SkyScript multi-objects对照或VRSBench小规模丰富描述，随后再探索RSTeller子集；ChatEarthNet作为土地覆盖分支。

| 候选 | 已核实规模和文本来源 | 对当前实验的用途与限制 |
| --- | --- | --- |
| SkyScript `title_multi_objects` | 官方定义为焦点对象加周围对象；现有openai top30 CSV有此列 | 可利用已有图片建立更丰富监督对照，但两top30筛选池不同，需在共同图像上固定split和图片才能分离文字收益 |
| VRSBench原RGB版 | 29,614图/29,614详细caption，GPT-4V初稿经人工核验；基于DOTA-v2/DIOR | 适合先检验对象、属性、空间关系监督；只取官方train caption任务，不混入VQA或测试标注 |
| RSTeller | 1,309,926 RGB航拍图/2,619,852图文对，NAIP 0.6m，LLM基于OSM生成详细描述 | RGB接入相对直接，可先选固定分片子集；需处理同图两caption正例关系及自动文本噪声 |
| ChatEarthNet | 163,488 GPT-3.5图文对，另10,000 GPT-4V图文对；Sentinel-2/WorldCover描述 | 更适合土地覆盖语义；与高分辨率航拍域不同，需要确认RGB构建、官方字段及长度，不能直接当成现成JPEG数据池 |

当前polished CSV只有filepath/title_raw/title，不含multi_objects；另一openai CSV有title_multi_objects。
两CSV在images2/3交集227,746图，直接换CSV同时改变图像选择和文本。重新标注还会改变caption group，必须重新审计正例定义与train/val标签边界，不能沿用旧group_id。
项目已有ChatEarthNet通用manifest转换工具，但本轮没有完成官方数据字段/RGB接入验证或新增数据训练配方。
这些候选尚未下载、清洗或加入本轮训练；丰富描述与更强更新范围应分开比较。

官方依据（2026-10-06查阅）：

- [SkyScript官方数据说明](https://github.com/wangzhecheng/SkyScript)
- [VRSBench官方仓库](https://github.com/lx709/VRSBench)与[数据下载](https://huggingface.co/datasets/xiang709/VRSBench)
- [RSTeller官方仓库](https://github.com/SlytherinGe/RSTeller)
- [ChatEarthNet官方仓库](https://github.com/zhu-xlab/ChatEarthNet)与[数据下载](https://doi.org/10.5281/zenodo.11003436)

历史具体结果见[SAT全图片一轮审阅](SAT_FULL_IMAGE_EPOCH_ANALYSIS_2026-10-05.md)。
