# SAT 三轮训练报告审阅（2026-10-03）

本次两组训练都正常完成，验证对比损失持续下降。Adapter 配置的最终验证 loss 更低；直接更新原有视觉 head 也取得了明显改善。结果支持继续研究原架构微调，但目前没有完整检索评测，不能据此宣称检索性能提升，或判定 head 微调路线失败。

## 证据与训练完整性

来源：用户上传的 `/Users/wangyue/Downloads/sat_3epoch_reports.tar.gz`。其中包含两组实验的 `config.toml`、`train.log`、`metrics.jsonl`、`validation.jsonl`、`provenance.json`、`optimizer_groups.json`、`training_summary.json`，没有权重或检索报告。仅比较 provenance 中已存储的身份字段，没有重新计算资产散列。

两组项目提交均为 `008fb331523345faf87c0646608c0cf8316e4886`，上游提交、运行环境、共享数据及权重身份记录一致，实际设备均为 RTX 3090。

- 均完成 1710 次参数更新、6840 个微批次，没有恢复训练、跳过更新或非有限 loss/梯度。
- 每组 171 条训练记录，step 为 10 到 1710，间隔 10；10 次验证，step 为 0、200、400、600、800、1000、1200、1400、1600、1710。训练 stdout 与 JSONL 记录相符。
- 训练清单 36,495 对样本，物理 batch 16、drop-last。每轮 2280 个微批次、570 次更新；本次确实完成三个完整轮次，每轮略去 15 对样本，总曝光 109,440 对样本。
- 两组 best_step 均为 1710，best_loss 等于终点 loss。日志符合 step0、best、latest 的滚动保存设计；没有上传权重，因此不能直接核验实际三份文件、硬链接或参数零漂移。
- 两组只训练视觉侧：adapter 仅训练其 6 个参数张量；head 仅训练其 32 个参数张量。优化器元数据没有 backbone、文本 encoder、文本 projection 或 logit scale。

## 验证曲线与结果

| 更新 step | Adapter 验证 loss | 原有视觉 head 验证 loss |
| --- | ---: | ---: |
| 0 | 3.864123 | 3.864123 |
| 200 | 1.499868 | 2.612982 |
| 400 | 1.227086 | 2.071004 |
| 600 | 1.140125 | 1.837655 |
| 800 | 1.097355 | 1.712297 |
| 1000 | 1.044100 | 1.636486 |
| 1200 | 1.013556 | 1.595811 |
| 1400 | 1.004201 | 1.577939 |
| 1600 | 0.995678 | 1.573012 |
| 1710 | 0.995599 | 1.572853 |

两个 step0 仅相差约 4e-8，起点基本一致。所有记录的验证 loss 均逐次下降，未观察到验证 loss 后期反弹。

| 项目 | Adapter | 原有视觉 head |
| --- | ---: | ---: |
| 可训练参数 | 1,054,976 | 25,326,336 |
| 初始学习率 | 1e-4 | 1e-5 |
| 验证 loss 相对 step0 下降 | 74.23% | 59.30% |
| best step | 1710 | 1710 |
| 训练日志窗口累计墙钟 | 27.59 分钟 | 28.82 分钟 |
| Torch 峰值 allocated | 3.910 GiB | 4.132 GiB |

这不是所有超参数下的最优方法比较。两组同 seed、同数据、同训练预算及调度形式，但训练位置、参数量和学习率不同；只有一个 seed。Head 配置关闭随机深度，adapter 的冻结 head 运行于 eval 模式。结果应表述为“本次两个配置中，adapter 的验证对比损失更低”。

## 对研究判断的影响

**短预算不足以否定方法。** 在同一条新曲线上，600 step（约 1.05 epoch）到 1710 step，adapter 验证 loss 又下降 12.68%，head 又下降 14.41%。这为“只看约一轮训练可能低估效果”的判断提供了证据。本次未在 step500 验证，不能把插值当作实测；也不能把与旧实验的全部差异归因于训练长度，因为调度、初始化可复现性以及 head 随机深度设置也发生过改变。

**Head 确实能够学习。** 其 loss 从 3.864 降到 1.573，不能称为“没有效果”。同时它当前落后于 adapter，并不能证明参数更多一定更好、head 无效或学习率一定过低。文本 projection、文本 block、LoRA，以及两侧联合更新的作用，本次都未检验。

**后期趋平不能证明充分收敛。** 1400 到 1710 step，adapter/head 还分别改善 0.86%/0.32%，但本次 cosine 学习率在终点归零。继续训练能否改善，需要新的调度实验来回答；不能直接把已完成的 latest 按同一配置 resume 当作有效延长训练。

**当前指标是小批候选池的对比损失。** 验证 forward batch64 只是加速前向，计算 loss 时仍切回 16 对样本，末组不足 16 对。它覆盖 4055 个验证样本，但不是每个查询在完整 4055 候选池内的检索。较低 loss 表明局部批内匹配目标改善；不等于 R@1、R@5、R@10 或跨数据集检索改善。

**梯度累积并未扩大对比候选池。** 训练 batch16、累积4次，参数更新汇总64对样本的梯度，但 loss 在各微批次单独计算，且 queue_size=0。每个样本每次仍只面对当前批次的其他15个候选。后续如要评估负样本数量，应把物理对比 batch、梯度累积和队列分开设计；不能将当前配置描述成 batch64 对比学习。

## 成本与运行警告

在这对 RTX 3090 实验中，head 可训练参数约为 adapter 的24倍，但日志窗口墙钟只多约4.46%，Torch allocated 峰值多约0.222 GiB。视觉 backbone 冻结时，直接训练这两层 head 在当前设置下成本可接受。这只是本次配置的观察，不是对其他分辨率、batch 或设备的保证。

墙钟合计包含窗口内的验证、保存和 IO，不包含完整初始化、step0 验证及最后一次日志之后的全部保存时间。后9次验证已经计入这些窗口，不能再把全部验证耗时加上。Torch allocated 峰值可能包括验证前向，不能等同于 nvidia-smi 总占用。日志梯度范数是裁剪前的值；其超过 max_grad_norm=1 不能单独证明梯度爆炸。

两份日志均出现 `libgomp: Invalid value for environment variable OMP_NUM_THREADS`。训练随后正常完成；下次运行前可以设置 `export OMP_NUM_THREADS=4`，清理这一环境配置问题。

## 下一步：先补完整检索评测

优先对两组的 step0 和 best 各做一次 SkyScript validation 完整检索，共4份报告。best 与 latest 的记录都对应 step1710，此时无需重复评测 latest。再在已有 RSICD validation 清单上做同样4次评测，观察跨数据集迁移。RSICD validation 用于开发比较；最终测试集应按最终选定方案统一评估。

在服务器现有 tmux 会话中运行下列代码。沿用已保存的配置；RSICD 清单路径来自仓库已有实验脚本，运行前仍需确认服务器存在该文件。评测程序拒绝覆盖已有报告；失败后需要查明原因再续跑，避免忽略报错。

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
source .venv/bin/activate
export OMP_NUM_THREADS=4

for trial in skyscript_sat_adapter_3epoch_seed11 skyscript_sat_visionhead_3epoch_seed11; do
  for tag in step_0000000 best; do
    python -u -m dinotxt_rs.cli.evaluate_skyscript \
      --config "outputs/$trial/config.toml" \
      --manifest assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl \
      --checkpoint "outputs/$trial/$tag.pt" \
      --training-output "outputs/$trial" \
      --split val \
      --output "outputs/$trial/skyscript_val_$tag.json" || break 2
  done
done
```

如果服务器有此前使用的 RSICD validation 数据和清单，再运行：

```bash
for trial in skyscript_sat_adapter_3epoch_seed11 skyscript_sat_visionhead_3epoch_seed11; do
  for tag in step_0000000 best; do
    python -u -m dinotxt_rs.cli.evaluate_rsicd \
      --config "outputs/$trial/config.toml" \
      --manifest assets/data/manifests/rsicd_val_retrieval_v1.jsonl \
      --checkpoint "outputs/$trial/$tag.pt" \
      --training-output "outputs/$trial" \
      --split val \
      --output "outputs/$trial/rsicd_val_$tag.json" || break 2
  done
done
```

打包上述评测 JSON；无需下载权重：

```bash
tar -czf sat_3epoch_retrieval_reports.tar.gz \
  outputs/skyscript_sat_adapter_3epoch_seed11/*_val_*.json \
  outputs/skyscript_sat_visionhead_3epoch_seed11/*_val_*.json
```

后续比较应分别展示两个方向的 Recall@1/5/10、mean recall 及相对各自 step0 的变化。若 SkyScript 有改善而 RSICD 下降，表明存在域适配与迁移之间的取舍；若 loss 显著下降而完整检索持平，应优先调查候选池、近似/重复语义负样本和评测目标的一致性。完成这些评测之后，再决定扩大训练预算、调 head 学习率、改变对比 batch 或加入文本侧微调。
