# DINOtxt for Remote Sensing

研究如何以有限算力，让视觉基础模型更好地理解遥感领域文本，并在领域适配过程中保持已有能力。

当前主线是 **SkyScript `title_raw` + 冻结官方 dino.txt + 小型图像 embedding adapter**。ChatEarthNet 已转为历史诊断数据。具体进度、实验结果和下一步统一维护在交接文档中。

## 文档入口

| 需要了解什么 | 唯一主文档 | 内容边界 |
| --- | --- | --- |
| 为什么做、要回答什么科研问题 | [科研背景与研究设计](DINOv3_Remote_Sensing_Domain_Text_Alignment_Research_Plan.md) | 研究动机、问题、对照逻辑与结论标准 |
| 系统应该怎样组织、如何可靠开发 | [开发架构](docs/DEVELOPMENT_ARCHITECTURE.md) | 抽象流程、模块契约、实验与工程纪律 |
| 已经完成什么、当前卡在哪里、接下来怎么做 | [最新交接](docs/PREPARE_HANDOFF.md) | 当前数据与方法、结果证据、待办门槛和操作入口 |

阅读顺序：初次了解项目按表格从上到下；接手开发或服务器实验先读交接。

## 代码导航

```text
src/dinotxt_rs/
├── config.py       # 配置解析与约束
├── data/           # 统一图文数据接口
├── models/         # 上游模型加载、冻结策略与 adapter
├── losses/         # 对比目标及可选历史负样本队列
├── training/       # 训练、验证、状态恢复与来源记录
├── evaluation/     # 检索、零样本分类与初始化一致性检查
└── cli/            # 命令行组装入口
tools/              # 数据准备、审计、产物核验
scripts/            # 环境准备及可复现实验脚本（含历史路线）
configs/            # 具体实验的 TOML 配置（含历史配置）
tests/              # 单元与工程行为验证
```

运行环境与当前数据重建步骤见[交接附录](docs/PREPARE_HANDOFF.md#12-操作附录)。旧脚本或配置仍留存不代表它们是当前推荐入口；尤其 `download_assets.sh all` 包含历史 ChatEarthNet 下载，不能作为 SkyScript 的准备命令。

文档维护：研究问题变化更新科研文档，接口或开发规则变化更新架构，实验进度变化只更新交接。原 `SKYSCRIPT_ADAPTER_PREPARATION.md` 已并入交接，仅保留迁移链接；历史全文可通过 Git 追溯。

## 训练与 checkpoint 保留

新的正式 SkyScript SAT 训练默认每 200 step 验证一次，并在 step 0 和正常训练终点强制验证。验证指标写入输出目录的 `validation.jsonl`；训练指标写入 `metrics.jsonl`。当前 best 按验证 loss 选择。

滚动保留策略固定使用三份 checkpoint：

- `step_0000000.pt`：初始化权重与状态，用于比较训练前后表现。
- `best.pt`：验证 loss 最低的权重与恢复状态。
- `latest.pt`：最近一次 checkpoint，供中断后续训。

训练按 200 step 的 checkpoint 间隔更新 `latest.pt`，best 改善时覆盖 `best.pt`，不会持续生成每次验证对应的编号 checkpoint。正常终点会额外验证并保存最终状态；如果终点模型不是 best，`latest.pt` 保留终点状态，`best.pt` 仍指向验证 loss 最低的模型。

示例：

```bash
dinotxt-rs-train --config configs/skyscript_sat_adapter_3epoch_seed11.toml
```

不带 adapter 的文本侧三轮对照已准备好：只训练 projection、只训练最后两层文本
Transformer 和末端 LayerNorm，以及在全部文本层和 projection 上训练 LoRA。
三组视觉侧均冻结，从官方权重重新初始化。
在 tmux 会话中执行 `bash scripts/run_sat_text_3epoch_seed11.sh`，依次跑 projection、文本末两层和 LoRA；
已完成的实验会跳过，中断的实验从 `latest.pt` 续跑。脚本记录验证 loss，完整检索评测另行执行。
配置和启动说明见[交接中的当前实验协议](docs/PREPARE_HANDOFF.md#13-文本侧三轮训练协议与复现入口)。

视觉与文本联合适配首批四组已准备：adapter/head分别搭配projection/全层文本LoRA。
在tmux中运行 `bash scripts/run_sat_joint_3epoch_seed11.sh`；组合矩阵、训练范围与启动方式见
[联合实验计划](docs/SAT_JOINT_EXPERIMENT_PLAN_2026-10-04.md)。

从最近状态继续训练时，将 checkpoint 明确传给 `--resume`：

```bash
dinotxt-rs-train \
  --config configs/skyscript_sat_adapter_3epoch_seed11.toml \
  --resume outputs/skyscript_sat_adapter_3epoch_seed11/latest.pt
```

新配置使用 `checkpoint_policy = "rolling"`。历史阶段配置显式使用 `checkpoint_policy = "numbered"`，旧阶段脚本继续对应这些历史配置和编号 checkpoint。已有实验产物不会因新保留策略而删除。

配置会参与 checkpoint 的严格身份校验。历史模板现在显式声明 `checkpoint_policy = "numbered"`，因此它们与早期运行时保存的配置内容不同。续跑已有实验时，使用该输出目录内保存的原始 `config.toml` 副本匹配 checkpoint 身份；不要直接用更新后的模板代替。需要注意，当前代码会把旧副本中缺失的 `checkpoint_policy` 解释为 `rolling`，所以用当前代码续跑时保存策略会转为滚动保留，不能继续依赖旧阶段脚本要求的编号边界。要复现旧的编号阶段工作流，需使用旧版本代码。已有历史 checkpoint 和其他实验产物保持原样，不会自动迁移或删除。
