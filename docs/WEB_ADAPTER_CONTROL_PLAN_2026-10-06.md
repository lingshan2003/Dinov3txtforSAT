# Web adapter-only 同预算对照

2026-10-08状态：用户已提供完整结果，两组训练/评测完成。联合方案unique/多图按图/按组/RSICD MR分别比adapter-only高3.1854/3.5414/2.9201/1.6728pp，详细核验见[对照结果审阅](WEB_ADAPTER_CONTROL_SEED11_ANALYSIS_2026-10-08.md)。以下保留启动前协议。

日期：2026-10-06。用户同意补adapter-only，判断文本LoRA的额外收益与RSICD代价。代码已准备，尚未启动服务器实验。

本地验证：全量398项测试通过，Ruff、Bash语法、CLI帮助及diff检查通过。新增9项测试覆盖主组完成要求、各执行模式、缓存身份拒绝、对比百分点/协议校验及报告打包；未执行真实GPU训练。

## 1. 比较范围

已完成主组：`skyscript_web_adapter_textlora_maskpos_fullimage1epoch_seed11`，unique MR17.6860%、多图按图17.0671%/按组15.8317%、RSICD13.9305%。
新增训练仅为已有配置`skyscript_web_adapteronly_maskpos_fullimage1epoch_seed11.toml`。
它与主组只改变experiment name/output_dir及text_lora_rank/include_projection四字段；有意义的方法变化是关闭全部文本LoRA。

| 内容 | 两组共同协议 |
| --- | --- |
| 原生模型 | Web ViT-L/16 + 配套官方dino.txt head/text |
| 图片与caption | 固定337,199训练图、36,495caption组，保留原train/val边界 |
| 覆盖与预算 | 完整1图片epoch、5269optimizer step、batch16×accum4 |
| 优化 | 527warmup、cosine、BF16、adapter LR1e-4/WD0.01 |
| 目标与采样 | full-maskpos/image_epoch、queue0、noaug |
| 验证与权重 | 每200step及step0/终点；固定step_0000000.pt/best.pt/latest.pt |

adapter-only只更新1,054,976个adapter参数；视觉backbone/head、完整文本encoder/projection和logit_scale冻结。
主组约501万可训练参数。均独立从官方初始化开始，控制组不接续主组权重。
两组都有同构adapter；插入adapter在插入LoRA之前，因此关闭LoRA不会改变adapter构造前的初始化步骤。
实际step0三池指标是否与官方一致，会在比较JSON中记录；采样、来源和覆盖仍需结果报告核对。

## 2. 新入口的行为

入口：`scripts/run_web_adapter_control.sh`，调用`tools/run_web_adapter_control.py`。

1. 校验两份配置与固定配方；要求主组已完整完成并保留config/provenance及三份权重，主组未完成时直接报错。
2. 预检真实图片池、预算、split、模型资产、运行状态和已有目标报告；`--preflight-only`不写输出或调用GPU。
3. 复用原生官方三池报告与已完成主组九份检索JSON，经身份/候选池检查后复制到独立对照报告目录。
4. 仅训练adapter-only；若已中断则从它的latest恢复，完整完成则跳过。主组不会重新训练。
5. adapter-only的step0/best/latest在unique-val、多图val、RSICD-val评测，共新增9份检索JSON。
6. 自动生成原生/adapter-only/adapter+LoRA的对比JSON和Markdown，再打包两组日志与共同报告。

若主组某份检索缓存缺失，可用它既有权重补齐缺失评测，不重新训练主组。
原`outputs/web_native_seed11`的检索JSON及共同summary作为来源保留，新增汇总写入`outputs/web_adapter_control_seed11`。
默认最终包含21份检索JSON：原生3份＋两组各9份。报告包不含.pt权重或图片。
共有打包器新增收录报告目录中的.toml/.md，确保包含官方config快照与比较表，仍排除权重和临时.part文件。

## 3. 在服务器运行

同步这次代码即可，配置已经在上一轮提交中。沿用服务器现有模型权重、图片、manifests和Web主组输出目录。

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
bash scripts/run_web_adapter_control.sh --preflight-only
tmux new -s web-adapter-control
bash scripts/run_web_adapter_control.sh
```

Ctrl-b然后d离开tmux；`tmux attach -t web-adapter-control`返回。
中断后在相同代码/配置下重跑同一命令，使用adapter-only latest恢复；完整组和有效检索报告会跳过。
避免同时用旧`--include-controls`入口和本入口写同一个adapter-only训练目录。
若只训练，使用`--train-only`，不执行全局评测或生成完整对比表；可暂不安装RSICD。
训练完成后使用`--evaluate-only`补齐评测、比较与报告包。其预检要求两组训练均完整完成。

最后下载：

```text
/root/autodl-tmp/Dinov3txtforSAT/outputs/web_adapter_control_seed11_reports.tar.gz
```

自动表格：`outputs/web_adapter_control_seed11/comparison.md`；完整数值：同目录`comparison.json`。
主要比较使用完整覆盖的latest，另保留按unique-val loss选best的比较。正的`delta_lora_minus_adapter_only_pp`表示LoRA候选Recall更高，单位是百分点。
包含四项MR、双向R@1/5/10、两组相对原生变化、step0一致性、训练summary及候选数；不按其中最高指标自动宣称最终最优。

## 4. 结果要回答的问题

若LoRA同域更高而RSICD更低，支持在本配方下增加文本更新带来领域专化权衡；不能由此定位到某个Transformer层或projection。
若adapter-only两者都相近或更优，则优先用更小范围作为候选；若LoRA在多个指标更好，再对该候选补seed23/47。
当前仍是seed11开发验证，无独立test或显著性推断；原生RSICD基线用本轮14.4759%，不混用旧M4的15.8775%。

主组结果见[原生首轮审阅](WEB_NATIVE_SEED11_ANALYSIS_2026-10-06.md)，原生入口见[首轮协议](WEB_NATIVE_ADAPTATION_PLAN_2026-10-06.md)。
