# M4 Web A/B/C 100-step 开发协议

本阶段只筛选训练策略，不启动 5,000-step 正式训练。Gate A 必须在与本阶段相同的项目提交上通过，EuroSAT 全量与 RSICD test 不参与配置或 checkpoint 选择。

## 固定输入与共同设置

- backbone：Web DINOv3 ViT-L/16；
- 训练：ChatEarthNet 10,000 条候选清洗后的 9,969 样本；
- 同域 validation：固定的 16,277 条 ChatEarthNet val；
- 外部开发保持性：RSICD 官方 val split；
- seed：11；physical batch：16；gradient accumulation：4；queue：4,096；
- 设计 scheduler：5,000 optimizer steps、warmup 250；本阶段命令行限制为 100 step；
- step 50 中断并严格恢复，保留 step 0/50/100、best、validation 和恢复证据。

## 三组实验

| 标签 | 可训练范围 | 最大学习率 | 目的 |
| --- | --- | ---: | --- |
| A | vision head + text projection + text last-4 + logit scale | `5e-6` | 相对历史结果只降低学习率 |
| B | vision head only | `5e-5` | 相对历史结果只缩小更新范围 |
| C | vision head only | `5e-6` | 最保守的组合候选 |

vision-head-only 要求整个 text tower、text projection、logit scale 和视觉 backbone 均冻结并保持 eval，只有 `visual_model.head` 处于 train。

## 开发集与门槛

RSICD val manifest 由官方标注确定性生成。运行前对 ChatEarthNet train 与 RSICD val 做文件 SHA-256 和解码 RGB 像素 SHA-256 的零重叠审计。该审计不声称检测感知近重复。

预先固定的 100-step 放行条件：

1. 数值、checkpoint、step-50 resume、provenance 和全量 validation 核验通过；
2. step 100 RSICD val mean recall 不低于官方初始化超过 0.01 绝对值。

step 100 仍位于 250-step warmup 内，因此 ChatEarthNet validation 是否低于 step 0 只作为趋势记录，不作为此安全筛选的硬门槛。满足保持性条件的配置有资格进入 250-step 开发实验；若多个配置通过，先选择 step-100 RSICD val mean recall 较高者。到 250/500-step 门槛时，必须同时要求 ChatEarthNet validation 改善。该选择只授予 250-step 资格，不授予 5,000-step 正式训练资格。

## 运行入口

```bash
bash scripts/run_m4_web_abc_100step_development.sh
```

综合报告写入：

```text
outputs/m4_web_abc_100step_development_seed11/verification_report.json
```

不得用已观测的 EuroSAT 或 RSICD test 结果改变上述门槛，也不得删除失败组的配置、checkpoint 和报告。
