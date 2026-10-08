# Web 两种适配方法三seed配对复现

2026-10-08完成更新：用户已提供六组完整训练/评测报告。联合方案在三个配对seed的同域与RSICD mean Recall均优于adapter-only；unique平均17.7764±0.1711%、配对收益3.1278±0.1368pp，详见[三seed结果审阅](WEB_PAIRED_SEEDS_ANALYSIS_2026-10-08.md)。以下保留启动前协议。

日期：2026-10-08。用户同意复现adapter-only与adapter＋文本LoRA；代码已准备，本地未启动服务器GPU。

本地验证：411项全量测试通过，Ruff、Bash语法、CLI帮助及diff检查通过。新增13项测试覆盖配方/seed校验、全组预检、缓存拒绝、配对统计和seed23真实共享编排的首次训练/恢复/跳过；原配置测试的rolling清单同步新增四份配方。

## 实验范围

入口：`scripts/run_web_paired_seeds.sh`，调用`tools/run_web_paired_seeds.py`。

| seed | adapter＋LoRA | adapter-only |
| --- | --- | --- |
| 11 | 复用已有完整训练与检索 | 复用已有完整训练与检索 |
| 23 | 新训练 | 新训练 |
| 47 | 新训练 | 新训练 |

默认按seed11→23→47，每seed内adapter＋LoRA→adapter-only顺序执行，单进程顺序使用GPU。
与seed11各自对应配置只变experiment.name/output_dir/seed三个字段；固定数据split，不按训练seed重新划分数据。
每组仍337199图/36495caption组完整1图片epoch、batch16×accum4、5269step、527warmup、BF16、mask_same_caption、queue0、无增强。
物理对比batch仍为16，梯度累计4次不等于64候选损失。
每200step及step0/终点验证，固定step_0000000/best/latest三份权重；主比较latest，另保留best。
参数范围、学习率、weight decay、冻结规则均沿用已审阅seed11，无新增方法变体。

新增四份配置：

```text
configs/skyscript_web_adapter_textlora_maskpos_fullimage1epoch_seed23.toml
configs/skyscript_web_adapteronly_maskpos_fullimage1epoch_seed23.toml
configs/skyscript_web_adapter_textlora_maskpos_fullimage1epoch_seed47.toml
configs/skyscript_web_adapteronly_maskpos_fullimage1epoch_seed47.toml
```

## 预检、缓存和恢复

先校验所有六组配置、数据池/预算、训练状态和已有目标报告，再启动任何GPU子进程。
要求seed11两组已经完整完成，且保留各自config/provenance/training_summary及三份checkpoint；否则直接停止，不重新训练seed11。
官方报告沿用`outputs/web_native_seed11/official`；seed11的18份检索从`outputs/web_adapter_control_seed11`校验后复制。
若检索缓存缺失，用既有权重补齐缺失报告。不会从报告包恢复模型权重。
新seed从共同官方模型初始化；不接续seed11权重。中断后自动从当前组latest恢复，完成组和有效报告会跳过。
不要同时启动多个本入口，或其他写相同训练输出目录的入口。

每组step0/best/latest评unique-val、多图val、RSICD-val，新增四组共36份检索JSON；最终包含原生3份和六组54份，共57份。
结果位于`outputs/web_paired_seeds/seed11|seed23|seed47/<method>/`，训练输出仍各自独立。
每个seed结束生成单独报告包，全部完成后再生成汇总报告包；不含.pt权重、图片或临时.part文件。
三份固定权重限制每组文件数；四个新组仍会增加磁盘使用，总共新增最多12份checkpoint。

## 服务器启动

同步本轮新代码及四份配置。保留现有模型资产、数据和seed11两组完整训练输出。

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
bash scripts/run_web_paired_seeds.sh --preflight-only
tmux new -s web-paired-seeds
bash scripts/run_web_paired_seeds.sh
```

Ctrl-b再d离开；`tmux attach -t web-paired-seeds`返回。
中断后在相同代码/配置下重新运行最后一条命令，自动恢复，不需要手动选择checkpoint。
沿用seed11日志估算，四组训练区间累计约4.5小时，另加三池全局检索，实际取决于服务器状态。

可选模式：

```bash
bash scripts/run_web_paired_seeds.sh --train-only
bash scripts/run_web_paired_seeds.sh --evaluate-only
```

`--train-only`不执行全局检索或生成跨seed统计，但仍进行完整输入预检，需要RSICD manifest/图片可用。
`--evaluate-only`要求六组都已完整训练，然后补缺失评测、汇总与打包。
`--preflight-only`只读检查，不写输出，不训练或评测。

## 汇总和下载

全部完成后下载：

```text
/root/autodl-tmp/Dinov3txtforSAT/outputs/web_paired_seeds_reports.tar.gz
```

表格：`outputs/web_paired_seeds/comparison.md`；完整值：同目录`comparison.json`。
包含三seed各组数值、四项MR、全部双向R@1/5/10、latest/best、相对原生变化、step0一致性与训练summary。
Recall的组间均值/标准差以百分比表示，差值以百分点表示；标准差为sample std、ddof=1。
配对差值先算每seed的adapter＋LoRA减adapter-only，再计算三个差值的均值与标准差，不用两组独立方差替代。
验证池保持固定；三seed重复用于考察训练随机性，不是三个独立数据集，不自动宣称统计显著性。

seed11已完成证据与边界见[对照审阅](WEB_ADAPTER_CONTROL_SEED11_ANALYSIS_2026-10-08.md)。
