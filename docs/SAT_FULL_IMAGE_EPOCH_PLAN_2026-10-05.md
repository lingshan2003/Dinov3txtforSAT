# SAT 全图片池微调：独立数据规模系列

日期：2026-10-05。用户选择：首轮完整训练 **1 个图片 epoch**。

## 1. 研究目标与两条系列

用户明确希望探索大规模图片数据对微调效果的影响，不再要求所有新方案与旧 unique-caption 实验保持 1710step。

- 原 unique-caption 与 rotate 保留为组均衡系列：相同 caption 集合、1710step，对比固定代表图与组内轮换。
- multipos 与 maskpos 新增全图片系列：使用恢复后的整个训练图片池，每张训练图恰好一次，研究数据规模和训练覆盖。

总池 370,317 张图中，训练集 337,199 张、验证集 33,118 张。
保留已建立的完整 caption 组 train/val 边界，原 unique-val 和新多图 val 都留作评测。
这一系列扩展的是相同 36,495 个训练 caption 下的视觉实例，不增加不同文本的数量。
使用同一 held-out 数据可以比较实际效果；不同图片曝光、采样权重和日程意味着，不能将新旧差异解释为多正例 loss 的独立因果作用。

## 2. 采样器真正改变了什么

此前 `caption_sampling="caption_group"` 每个组每轮只取一至两张图，属于组均衡访问。
新增 `caption_sampling="image_epoch"`，每个图片 epoch 消费所有训练行一次：

1. 组内图片打乱，每组切成最多两张图的小块，剩余单图也保留。
2. 将整个池的小块统一打乱、展平为图片索引序列，再按物理 batch16 取样。
3. 最后不足 16 张的 batch 保留，不补图、不丢图，也不进入下一轮凑满累计窗口。

两图块用于保留真实的 batch 内多正例信号；完全随机打散所有图片时，同文图片同批出现会很少。
块可能跨 batch 边界，同一 caption 的多个块也可能同批出现，因此 `images_per_caption=2` 在这个协议中表示块大小，**不是每批最多两张同文图片**。
loss 按实际 group ID 正确识别所有同文正例，不假设正例数量固定为 2。

该采样方式是按图片覆盖：大组的总曝光更多，这与此前每组等权不同。
重复文本仍按当前方形配对矩阵编码，均匀目标与屏蔽目标保持不变；矩形唯一文本方案暂作为另一个目标设计轴。
日志继续记录不同 caption 数、每查询正例数、同组非对角正例数、图片覆盖率和图片轮次。

本地用报告中的组大小直方图构造 337,199 行合成映射验证：21075 个 batch 消费 337199 个不同索引，终点完整一轮、尾批15张。
该合成映射的平均每批不同 caption 约 9.06，每查询正例约 1.88；这仅验证机制，服务器真实行顺序的诊断以实际日志为准。
新旧系列的负例描述多样性也会不同，需要保留方向指标及按组均衡评测。

## 3. 首轮两组配置

两组仅对比目标，初始化、模型、图片顺序、验证集、训练预算都一致：

| 方法 | 配置 | 目标 |
| --- | --- | --- |
| multipos-full | `configs/skyscript_sat_adapter_textlora_multipos_fullimage1epoch_seed11.toml` | 同文正例集合的均匀交叉熵 |
| maskpos-full | `configs/skyscript_sat_adapter_textlora_maskpos_fullimage1epoch_seed11.toml` | 配对目标，屏蔽同文非对角候选 |

共同模型仍为 adapter256 + 全文本 LoRA（含 projection），冻结 SAT backbone 和官方 vision head；不随机重置原 head。
沿用 seed11、batch16、梯度累计4、BF16、LR1e-4、queue0、无增强、训练 workers0。
新的 cosine 日程从官方共同初始化开始，warmup527，完整预算5269step；不续接此前接近零 LR 的1710step run。

预算按图片计算：

```text
训练图片                  337199
图片 epoch                1
microbatch 数             ceil(337199 / 16) = 21075
optimizer step 数         ceil(21075 / 4)   = 5269
末尾 microbatch           15 张图
最后 optimizer 累计窗口   3 个 microbatch
总图片曝光                337199（精确，不是5269×64）
终点不同图覆盖            100%
```

训练 loop 按最后窗口真实的三个 microbatch 缩放梯度，避免仍除以四造成末次更新变小。
运行前校验 actual manifest 大小与预算是否一致；数据不同则拒绝沿用错误的步数。
验证仍每200step，强制step0和正常终点；checkpoint仍只保留step0/best/latest三个固定文件。
best 保持原 unique-val loss 选择，latest 记录完整图片覆盖终点。
两者都做全局检索，以免 best 在中途、只评 best 就遗漏完整覆盖结果。

依据此前 RTX3090 训练窗口耗时按step数线性粗估，每组训练约1.3小时；两组加完整检索另需时间，实际速度以服务器日志为准。

## 4. 服务器执行

先同步本次代码和新配置；图片无需再次上传或重建已有分组清单。
独立启动器不会调用旧系列，也不要求旧1710step权重作为训练前置条件。

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
bash scripts/run_sat_full_image_epoch.sh --preflight-only
tmux new -s sat-full-image
bash scripts/run_sat_full_image_epoch.sh
```

`--preflight-only` 检查两组输入、原代表图保留、分组审计数量、train/val边界、两组配方一致性、预算和已有输出，执行前不写输出、不加载模型、不启动GPU训练。
默认依次训练 multipos、maskpos，再评测两组各自 step0/best/latest 的原 SkyScript、新多图 SkyScript、RSICD-val，共18份检索报告。
验证和 checkpoint 保存不增加编号权重文件。

使用 `Ctrl+B` 后按 `D` 离开 tmux；返回：

```bash
tmux attach -t sat-full-image
```

启动器默认完成后同时打包：

```text
outputs/sat_full_image_epoch_seed11_reports.tar.gz
```

上传该报告包即可，包含训练配置、日志、coverage、来源、优化器、数据审计和检索指标，不包含实际 `.pt` 权重。
汇总为 `outputs/sat_full_image_epoch_seed11/summary.json`。

中断后在 tmux 中重跑同一命令，已完成组跳过、未完成组从 `latest.pt` 恢复；配置变化拒绝覆盖旧 run。
也支持：

```bash
bash scripts/run_sat_full_image_epoch.sh --train-only
bash scripts/run_sat_full_image_epoch.sh --evaluate-only
```

train-only 保留分组验证资产检查，不依赖 RSICD；evaluate-only 要求两组完整覆盖已完成。
进一步改变图片 epoch 数，应创建独立配置/输出目录，按实际 manifest 重新计算 max_steps、warmup；不要覆盖本轮run配置。

## 5. 工程验证与审阅边界

采样器测试覆盖每轮索引单次全覆盖、尾批保留、多组和大单组、预取不推进、跨轮精确恢复及配置/索引映射不一致拒绝。
CPU训练测试覆盖连续/中断恢复参数一致、冻结backbone、精确曝光数、100%覆盖、step0/终点验证、三个固定checkpoint及最后不足累计窗口的梯度缩放。
旧配置默认不启用 image_epoch，原1710step路线保持不变。
启动器另外验证完整训练/评测/打包、重复执行跳过、resume、预检失败先于GPU调用等行为。
真实服务器 GPU 训练和效果尚未由本地执行；本轮准备完成不代表已有性能结论。
本地全量359项测试通过，Ruff、Bash语法、CLI帮助及diff检查通过。
