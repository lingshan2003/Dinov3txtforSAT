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
