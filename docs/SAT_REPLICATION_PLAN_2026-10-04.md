# SAT 三个候选的三seed复现（2026-10-04）

本轮复现adapter + 全文本LoRA（含projection）、adapter-only、官方vision head + 全文本LoRA（含projection）。已有seed11的训练与完整检索保留，新增seed23、47各三组，共六次新训练。尚未在服务器GPU启动。

当前seed11的SkyScript/RSICD mean Recall分别是10.884%/7.044%、9.684%/6.679%、6.663%/5.098%。重点检查adapter+LoRA的整体收益和RSICD图→文R@1下降是否跨seed重复；不要求每个新seed都超过历史best，也不按验证集最好的seed挑选论文结果。

## 固定协议

只改变experiment.seed、name和output_dir；其余字段逐项保持对应seed11配置。所有新run从官方权重开始，不加载seed11的best作为初始状态。

- SAT视觉backbone永久冻结；不随机初始化vision head。Head方案继续微调官方预训练head，不新增head架构。
- LoRA rank8、alpha16、dropout0，覆盖24个文本block的QKV、attention输出和MLP两层，以及最终projection。
- Adapter为2048→256→2048残差模块。Adapter/LoRA LR均1e-4，head LR为1e-5；logit_scale冻结，可训练head显式drop_path0，冻结head保持eval。
- 同一36,495对训练清单、4055对验证清单；**清单文件名中的seed11/seed23是原数据划分标识，新增训练seed不会重划分数据。** 三个方法使用相同seed编号，但因模块初始化的RNG消费不同，不承诺跨架构的随机矩阵或训练批序逐项一致。
- 1710 optimizer step、warmup171、物理batch16×累积4、BF16、queue0、无增强、shuffle=true。每次对比候选仍为16。
- Step0、每200step和正常终点验证，best按验证loss选择；固定step0/best/latest三份权重。每组独立输出目录。
- 各seed都评step0/best的SkyScript-val与RSICD-val；不改test用途。

## 新配置与顺序

| Seed | 运行顺序 | 配置文件 |
| --- | --- | --- |
| 23 | Adapter+LoRA | `configs/skyscript_sat_adapter_textlora_3epoch_seed23.toml` |
| 23 | Adapter-only | `configs/skyscript_sat_adapter_3epoch_seed23.toml` |
| 23 | Head+LoRA | `configs/skyscript_sat_visionhead_textlora_3epoch_seed23.toml` |
| 47 | Adapter+LoRA | `configs/skyscript_sat_adapter_textlora_3epoch_seed47.toml` |
| 47 | Adapter-only | `configs/skyscript_sat_adapter_3epoch_seed47.toml` |
| 47 | Head+LoRA | `configs/skyscript_sat_visionhead_textlora_3epoch_seed47.toml` |

Outputs目录为`outputs/<配置名，不含.toml>`，不会覆盖seed11或历史500step复现。

## tmux启动与断点恢复

将新增配置和脚本同步到服务器，确认没有另一项任务占用同一GPU后运行：

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
tmux new -s sat-replications-3epoch
bash scripts/run_sat_replications_3epoch.sh
```

默认流程：先顺序完成六组训练，然后完成24次检索（六组×两个数据集×step0/best），最后自动打包训练和检索报告。Ctrl-b再d脱离；返回用：

```bash
tmux attach -t sat-replications-3epoch
```

脚本预检全部配置、必需模型/数据路径及已有配置副本；检索模式额外要求RSICD-val清单存在，不重新计算额外SHA256。CLI保留已有的资产/checkpoint来源校验。

已完成训练且三份权重存在则跳过；中断且latest存在则从latest恢复。已有目录非空但没有latest时停止，避免覆盖实验；如果只有step0，需检查后用保存配置手动resume。训练/检索失败时停止，修复后重跑同一脚本。已有检索JSON会先检查task/split、候选数、manifest、checkpoint路径/步骤、模型配置路径、identity状态、tie规则和mean Recall后跳过；错误或不完整的报告会停止，不会静默覆盖。该检查不替代逐embedding重放或实际权重差分。

若此次只想训练，可执行：

```bash
bash scripts/run_sat_replications_3epoch.sh --train-only
```

之后无参数重跑即可补评测，六组已完成训练会跳过。自动归档会在成功完成当前模式时生成或刷新，不把中断时留下的旧归档当作当前完整结果。

## 下载与审阅

默认流程成功结束后下载：

```text
/root/autodl-tmp/Dinov3txtforSAT/outputs/sat_replications_3epoch_reports.tar.gz
```

归档包含六组config、provenance、optimizer_groups、训练日志、validation、summary、retrieval.log及24份检索JSON，排除`*.pt`和`*.part`。权重留在服务器。`--train-only`归档只有训练阶段产物，补评测后脚本会重新打包同名文件。

上传归档后，结合已经审阅过的seed11结果构成3方法×3seed。汇总每项mean Recall、双向R@1/5/10的均值及样本标准差（ddof=1），同时列逐seed结果和同seed方法差值；三个seed仍不足以支撑强显著性结论。特别区分RSICD图→文Top1与mean Recall取舍，并核验各seed step0起点。

本轮只复现已有方法，不启动随机head、扩大候选池或改变数据集。
