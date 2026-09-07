# M5 Web A/C 250-step 开发协议

M5 只延长 M4 中的低学习率 A、C 两组，不重跑前 100 step，也不继续已经出现同域
validation 反弹的高学习率 B。该阶段到达 5,000-step scheduler 的 warmup 终点，但仍不授予
正式 5,000-step 训练资格。

## 固定实验与恢复边界

- A：vision head + text projection + text last-4 + logit scale，最大学习率 `5e-6`；
- C：vision head only，最大学习率 `5e-6`；
- 从各自已验证的 `step_0000100.pt` 严格恢复至 step 250；
- scheduler、optimizer、queue、sampler、DataLoader generator 和全部 RNG 状态均恢复；
- ChatEarthNet 全量 validation 与 checkpoint 继续每 50 step 记录，因此新增 step 150/200/250；
- RSICD 官方 val 在 step 150/200/250 评估，并保留 M4 的 official/step-100 报告；
- EuroSAT 和 RSICD test 继续隔离，不参与选择。

checkpoint 身份包含产生 M4 的项目 commit。M5 脚本不会放宽这项校验，而是从两个 run 的
`provenance.json` 读取相同的历史 commit，在临时 Git worktree 中执行当时的训练代码。当前项目
目录只提供原始数据、权重和既有输出，临时 worktree 在脚本退出时移除。这样新的 M5 验收代码
不会改变 step 100 之后的训练实现。

## Step-250 双门槛与选择

每个候选必须同时满足：

1. step-250 ChatEarthNet validation loss 严格低于 step 0；
2. step-250 RSICD-val mean recall 不低于官方初始化超过 `0.01` 绝对值。

报告同时标记从 step 100 到 250 的 RSICD 指标是否逐点下降；它是风险提示，不替代上述预注册
门槛。通过双门槛的候选才有资格进入 500-step。

若一个候选在 step 250 的 ChatEarthNet loss 不高于另一个候选，同时 RSICD mean recall 不低于
另一个候选，并且至少一项严格更优，则称其支配另一个候选。只有 Pareto 前沿唯一时才自动给出
`recommended_for_500step`；若两组各有取舍，报告返回 `tradeoff_review_required`，不得用 test
集打破平局。

## 运行入口

M4 原始输出必须仍在默认路径且工作树必须干净：

```bash
bash scripts/run_m5_web_ac_250step_development.sh
```

综合报告写入：

```text
outputs/m5_web_ac_250step_development_seed11/verification_report.json
```

必须保留 A/C 的 step 0/50/100/150/200/250 checkpoint、`best.pt`、训练日志、恢复历史和所有
RSICD-val 报告。脚本中断后不得删除或覆盖部分产物，应先检查最后一个完整 checkpoint 再制定
恢复方案。
