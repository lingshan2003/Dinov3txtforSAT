# SAT 文本侧三轮训练审阅（2026-10-04）

后续五组完整检索评测已完成，最新结论见[完整检索审阅](SAT_3EPOCH_RETRIEVAL_ANALYSIS_2026-10-04.md)。
本文保留当时仅有训练报告的审阅与评测复现命令。

三组训练正常完成，验证对比损失持续下降。当前文本侧最有进展的是全层 LoRA：验证 loss 降到 1.327784，低于末两层全量更新和 projection-only，也低于前一轮视觉 head-only 的 1.572853。Adapter-only 仍以 0.995599 的 loss 领先。上传包没有完整检索报告，因此这个排序只描述当前配置的验证 loss，不是检索性能排序。

## 1. 来源与完整性

本次来源：`/Users/wangyue/Downloads/sat_text_3epoch_reports.tar.gz`。三组目录各有 config、train.log、metrics.jsonl、validation.jsonl、optimizer_groups.json、provenance.json、training_summary.json，共21份报告文件，没有 checkpoint 或 retrieval JSON。

三组记录的项目提交均为 `d1f8869ffbe600e6fb78e6346219f27ff2aa253c`，上游提交为 `6876159a11b4df116f30f667f8c9888617df0751`，设备为 RTX 3090，Python3.12.3、Torch2.7.1+cu128、CUDA12.8。共享资产身份只比较 provenance 已存储的值，没有重新计算 SHA256。

- 三组均完成1710次更新、6840个微批次，completed=true，无恢复训练，无跳过更新，记录的 loss/梯度均有限。
- 每组171条训练记录，step10到1710、间隔10；10条验证记录，step0、200、400、600、800、1000、1200、1400、1600、1710。
- 训练清单36,495对样本、物理batch16、累积4、drop-last，共三个完整轮次；总曝光109,440对样本，每轮略去15对。
- 验证清单4055对样本，loss分组16、前向batch64；queue0、无训练增强、shuffle=true、seed11、warmup171，均符合配置。
- 三组 initial validation loss 完全相同：3.8641233303102993。记录中未出现初始化 loss 被新增 LoRA 改变的迹象；这不替代逐项 embedding/logits parity 核验。
- 三组 best 均为step1710，latest记录为终点；不需要再分别评测 best 与 latest 的同一步状态。
- 本次日志没有上一轮的 OMP_NUM_THREADS 无效警告。

没有上传权重，不能核验实际三份checkpoint的文件数量、硬链接、每层LoRA增量大小或冻结权重零漂移。优化器记录和运行断言支持训练范围符合设计，但不等于已经直接比较真实权重。

## 2. 实际训练范围

| 实验 | 优化器中的范围 | 参数张量 | 可训练参数 | 初始学习率 |
| --- | --- | ---: | ---: | ---: |
| Projection-only | text_model.head.linear_projection.weight | 1 | 2,621,440 | 1e-5 |
| Text-last2 | blocks22、23的attention/MLP/LayerNorm，以及ln_final | 24 | 39,349,760 | 1e-5 |
| 全层LoRA | 24层的QKV、attention输出、MLP两层，以及最终projection的A/B | 194 | 3,958,784 | 1e-4 |

三组没有image adapter，也没有视觉侧或logit_scale的可训练参数记录。Text-last2的projection冻结；projection-only的文本Transformer冻结。

LoRA元数据包含97个插入点：每层4个，共96个，加最终projection一个。192个文本block因子张量合计3,932,160参数，projection的2个因子张量合计26,624参数。所有可训练参数名均以lora_A/lora_B结尾，没有基础矩阵、embedding或LayerNorm参数。

LoRA训练日志中两个优化器组的梯度范数都有正值，loss持续下降，说明LoRA分支获得了训练信号，不像一个未接入优化器或被整体截断梯度的空分支。不过当前没有逐层梯度日志，不能声称已经验证每一个插入点都产生了同等程度的更新。

## 3. 曲线与五组对照

| step | Projection-only | Text-last2 | 全层LoRA |
| --- | ---: | ---: | ---: |
| 0 | 3.864123 | 3.864123 | 3.864123 |
| 200 | 3.336039 | 3.263383 | 2.368724 |
| 400 | 2.955000 | 2.842102 | 1.734568 |
| 600 | 2.738969 | 2.597709 | 1.544299 |
| 800 | 2.609135 | 2.443857 | 1.446573 |
| 1000 | 2.528347 | 2.346024 | 1.390167 |
| 1200 | 2.483536 | 2.290948 | 1.350017 |
| 1400 | 2.463246 | 2.265673 | 1.335048 |
| 1600 | 2.457468 | 2.258566 | 1.328128 |
| 1710 | 2.457213 | 2.258260 | 1.327784 |

所有记录的验证点逐次下降，未观察到验证loss反弹。Projection、last2、LoRA的验证loss相对step0分别下降36.41%、41.56%、65.64%。在600step（约1.05epoch）之后，又分别改善10.29%、13.07%、14.02%，进一步说明短预算会低估训练目标上的改善。

结合前一轮已上传的 `/Users/wangyue/Downloads/sat_3epoch_reports.tar.gz`：

| 当前配置 | 可训练参数（百万） | 学习率 | 终点验证loss | 相对step0下降 |
| --- | ---: | ---: | ---: | ---: |
| Image adapter-only | 1.05 | 1e-4 | **0.995599** | 74.23% |
| 全层文本LoRA（含projection） | 3.96 | 1e-4 | **1.327784** | 65.64% |
| 原有视觉head-only | 25.33 | 1e-5 | 1.572853 | 59.30% |
| 文本末两层 + ln_final | 39.35 | 1e-5 | 2.258260 | 41.56% |
| 文本projection-only | 2.62 | 1e-5 | 2.457213 | 36.41% |

两轮共享权重、tokenizer、train/val清单的存储身份一致，数据协议、训练预算与验证分组一致。项目提交不同；检查两次提交之间trainer与loss变化，没有发现针对原有非LoRA训练路径的优化算法改变，trainer新增的是LoRA元数据记录。

这支持在同一验证定义下比较这五个已运行配置，但不是所有方法的最优超参数比较：只有seed11；LoRA比末两层覆盖范围更广，还适配projection，学习率也高10倍。不能将优势单独归因为低秩参数化、跨层覆盖、projection或学习率中的任意一个因素。

## 4. 研究含义与成本

**文本侧整体适配值得保留。** 全层LoRA的loss改善明显，说明视觉侧固定时，文本侧也能适应该训练目标。此前短预算的“文本侧效果不好”不能作为排除整条路线的理由。

**当前不是参数越多越好。** 末两层全量更新约39.35M参数，但本次配置的loss落后于3.96M参数的全层LoRA。它们的更新范围、步长和参数化不同；这可以形成下一轮对照问题，不能直接写成“LoRA普遍优于全量微调”。

**Projection-only不是零效果。** 它有36.41%的loss下降，只是单独更新这一层、用当前学习率时落后于其他已运行配置。它与视觉head或文本block联合更新的价值没有被本次实验排除。

**后期趋平仍受调度影响。** 1400到1710step，projection/last2/LoRA又改善0.245%/0.327%/0.544%；末步学习率归零。没有依据宣称它们已达到不可改善的极限，也不应直接将同一已完成配置resume视为有效延长训练。

| 本轮实验 | 训练日志窗口累计墙钟 | Torch峰值allocated |
| --- | ---: | ---: |
| Projection-only | 18.40分钟 | 3.923GiB |
| Text-last2 | 19.30分钟 | 4.268GiB |
| 全层LoRA | 25.50分钟 | 5.667GiB |

LoRA比末两层全量微调少约10倍可训练参数，却用了约32.1%更多日志窗口墙钟和约32.8%更多Torch峰值allocated；反向需要经过全24层，因此少参数不意味着更少激活计算。当前RTX3090运行正常，这条路线的实际成本可接受。

上述时间包含窗口内验证、保存和IO，不含完整初始化/step0验证及末尾保存；不是纯GPU计算时间。Allocated是Torch累计峰值、可能包括验证，不是nvidia-smi总占用。两轮OMP环境情况也不同，不能将旧视觉实验与本轮的耗时差异单独归因于模块架构。梯度范数是裁剪前数值，不能仅因其大于1就认定梯度爆炸。

## 5. 下一步：统一完整检索评测

目前仍缺五组的step0/best完整检索结果。训练和验证损失都在16对候选的独立分组上计算，梯度累积4次不扩大单次对比集合；完整SkyScript retrieval需要每个查询对全部4055个候选排序。只有拿到双向R@1/5/10和mean recall，才能判断训练目标的改善是否转化为检索改善，并检查文本LoRA的迁移表现。

建议现在评测全部五组，而不是在当前loss排序上继续选新训练预算。先做SkyScript-val，再用已有RSICD-val作跨数据集开发比较；最终test用途仍应留给选定方案。

在服务器空闲GPU的tmux会话中，下面代码先评测五组step0/best（10次）。它使用各输出目录的原始config.toml，新代码可以加载旧adapter/head权重；项目commit差异只作来源告警，配置、资产和参数结构仍会严格核验。

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

(
  set -e
  for trial in adapter visionhead textproj textlast2 textlora; do
    run_dir="outputs/skyscript_sat_${trial}_3epoch_seed11"
    for tag in step_0000000 best; do
      report="$run_dir/skyscript_val_$tag.json"
      if [ -f "$report" ]; then
        echo "Existing report, skipped: $report"
        continue
      fi
      .venv/bin/python -u -m dinotxt_rs.cli.evaluate_skyscript \
        --config "$run_dir/config.toml" \
        --manifest assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl \
        --checkpoint "$run_dir/$tag.pt" \
        --training-output "$run_dir" --split val \
        --output "$report"
    done
  done
)
```

如果服务器有此前使用的RSICD validation清单和图像，再运行同样10次：

```bash
(
  set -e
  for trial in adapter visionhead textproj textlast2 textlora; do
    run_dir="outputs/skyscript_sat_${trial}_3epoch_seed11"
    for tag in step_0000000 best; do
      report="$run_dir/rsicd_val_$tag.json"
      if [ -f "$report" ]; then
        echo "Existing report, skipped: $report"
        continue
      fi
      .venv/bin/python -u -m dinotxt_rs.cli.evaluate_rsicd \
        --config "$run_dir/config.toml" \
        --manifest assets/data/manifests/rsicd_val_retrieval_v1.jsonl \
        --checkpoint "$run_dir/$tag.pt" \
        --training-output "$run_dir" --split val \
        --output "$report"
    done
  done
)
```

以上块遇到失败会停止，修复后可以重跑并跳过已有报告；已有文件的完整性仍需看评测输出确认，不能只凭文件名认定全部评测已完成。评测完成后只打包结果JSON，不必下载权重：

```bash
tar -czf sat_all_3epoch_retrieval_reports.tar.gz \
  outputs/skyscript_sat_adapter_3epoch_seed11/*_val_*.json \
  outputs/skyscript_sat_visionhead_3epoch_seed11/*_val_*.json \
  outputs/skyscript_sat_textproj_3epoch_seed11/*_val_*.json \
  outputs/skyscript_sat_textlast2_3epoch_seed11/*_val_*.json \
  outputs/skyscript_sat_textlora_3epoch_seed11/*_val_*.json
```

评测之后再决定下一轮变量。如果LoRA在完整检索上也保持优势，可以比较LoRA不更新projection与当前整体LoRA，或给末两层/projection做有限学习率对照。如果训练loss改善却检索持平，应优先检查小候选池目标与全局排序的差异、数据中的语义近重复及多正例关系。如果同域提升而RSICD下降，则要明确域适配与通用能力保持的取舍。尚未评测之前，不把这些分支预设成已经成立的结论。
