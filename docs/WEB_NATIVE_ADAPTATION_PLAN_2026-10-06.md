# 原生 DINOv3.txt 遥感领域适配：精简首轮

日期：2026-10-06。用户提出优先适配原生DINOv3.txt，避免继续投入长SAT错配修复矩阵。
本轮准备独立Web入口；尚未在服务器启动评测或训练，保留全部SAT实验与未运行配置。

本地验证：全量389项测试通过，Ruff、Bash语法、CLI帮助及diff检查通过；新增原生baseline/缓存身份、默认与可选编排、四身份字段迁移、官方loader及CLI互斥参数测试。

## 1. 研究问题调整

新的主线候选是：在冻结原生Web DINOv3视觉backbone的条件下，遥感图文监督能否用低成本适配提高领域检索？
比较原生初始化、适配后同域检索及外部遥感迁移；检验参数范围、图像覆盖与caption粒度的作用。
原生组合 = LVD1689M Web ViT-L/16 + 官方配套dino.txt视觉head/文本encoder。
与SAT组合相比，不需要先修复更换视觉权重后产生的head/text接口错配。仍然属于已有模型的领域后训练，不是从零预训练文本encoder。

用户新方向优先于旧交接中“SAT必须作为唯一主角、Web统一后置”的安排。
旧SAT材料保留作初始化错配诊断与迁移实验；不能把不匹配head的SAT较低分数写成SAT视觉表征本身劣于Web。
有能力训练视觉自监督模型并不自动意味着拥有合适的图文监督数据；本项目此前的冻结领域骨干迁移问题仍有意义，但可以作为辅助研究。
不要求首轮重跑全部历史消融；方法有效性由新原生基线及同协议适配前后验证，而非从近零起点恢复的相对涨幅证明。

## 2. 已有证据与可比边界

旧500step、唯一caption数据、adapter-only三seed已有Web结果：

| 指标 | 原生step0 | Web适配后 | 说明 |
| --- | ---: | ---: | --- |
| SkyScript unique-val MR（seed11） | 8.6272% | 13.7567% | +5.1295个百分点 |
| SkyScript unique-val MR（三seed） | 各自初始化 | 14.0006% ± 0.3438pp | 平均适配增益5.3706pp |
| RSICD-val MR（seed11） | 15.8775% | 15.4296% | 小幅下降0.4479pp |
| RSICD-val MR（三seed） | 各自初始化 | 15.5485% ± 0.3989pp | 平均变化−0.3087pp |

标准差为sample std；数据来自历史M4归档及交接第2.5/2.6节。
这证明Web领域适配已有可重复的同源收益，并不证明跨遥感数据泛化已改善或本文方法超过其他模型。
这些旧结果不是当前全图片一轮实验，不能拿14%与SAT全图片11%直接计算纯backbone差异。
新首轮复用当前三个验证池，重新测官方原生模型；使用相同候选池/正例关系/预处理规则比较适配前后。
RSICD是遥感迁移开发验证，不能称为通用自然图像保持性；通用能力保持还需要另设评测，本轮未实现。

## 3. 只跑有用的首轮

入口：`scripts/run_web_native.sh` / `tools/run_web_native.py`。

默认流程：预检 → 原生官方三池全局检索 → 一组adapter+全部文本LoRA训练 → step0/best/latest三池检索 → 无权重报告包。
原生评测通过新增`--official`调用已有官方reference loader：不加入项目adapter或LoRA，所有参数冻结，报告checkpoint=null/trainable=0。
与其他代码默认的“根据训练配置构建零残差adapter模型”区别明确；既有checkpoint评测方式保持不变。
官方reference的三份报告在正式训练前生成；`--baseline-only`可先看起点，完全不训练。

| 变体 | 可训练范围 | 首轮是否默认 |
| --- | --- | --- |
| adapter_lora | adapter256 + 全24文本block及projection LoRA，rank8/alpha16 | 是，仅这一组 |
| adapter_only | adapter256，其余冻结 | `--include-controls`可选 |
| head_lora | 官方两层视觉head全量 + 全文本LoRA，无adapter | `--include-controls`可选 |

全部保持视觉backbone永久冻结、相同Web权重、同官方head/text与BPE；不加载任何SAT微调参数。
主组与已完成SAT full-maskpos一轮配方仅改变实验name/output_dir和backbone_domain/backbone_weights四项；domain同时选择原生Web预处理。
可选两组是同数据预算下模块消融；没有对应已完成的SAT full-image一轮结果，不声称它们构成完整matched Web/SAT矩阵。

共同协议：seed11、337,199训练图/36,495caption组、完整1图片epoch、batch16×accum4、5,269optimizer step、527warmup、cosine、BF16、queue0、noaug、mask_same_caption。
adapter/文本LoRA LR1e-4，head全量LR1e-5，WD0.01。独立从官方初始化重新开始。
该首轮与历史单组约78分钟的训练量相近，但Web实际耗时和显存以服务器日志为准。
默认训练预算从上一入口三组累计5个图片epoch缩为1个，仍需加官方及训练后全局检索时间。
没有用500step短预算替代完整一轮，也没有默认3epoch或batch32/强文本全量矩阵。

## 4. tmux启动与下载

同步代码和三份Web配置；需要既有Web权重`assets/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`。
图片、manifests、原生head/text权重及BPE全部复用，不需要再准备数据。

先只测原生基线，不训练：

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
bash scripts/run_web_native.sh --preflight-only
tmux new -s web-native
bash scripts/run_web_native.sh --baseline-only
```

完整首轮在同一个tmux会话内运行：

```bash
bash scripts/run_web_native.sh
```

已有正确官方报告会校验并跳过；不加`--baseline-only`也会先评官方再训练。
若决定补齐两个模块对照，则运行以下命令；已完成主组跳过：

```bash
bash scripts/run_web_native.sh --preflight-only --include-controls
bash scripts/run_web_native.sh --include-controls
```

默认3份官方+9份微调检索报告；含controls为3+27份。不要同时启动多个写同一输出目录的入口。
支持`--train-only`（不执行全局检索，可暂不安装RSICD）、`--evaluate-only`（必须训练完成）、预检、完成跳过和latest恢复。
仍每200step验证及step0/终点强制验证，每组只保留step_0000000.pt/best.pt/latest.pt。
官方baseline-only不保存训练权重，报告包不含任何.pt或图片。
Ctrl-b然后d离开tmux；`tmux attach -t web-native`返回。

统一报告包：

```text
/root/autodl-tmp/Dinov3txtforSAT/outputs/web_native_seed11_reports.tar.gz
```

官方基线明细为`outputs/web_native_seed11/official/{skyscript_unique,skyscript_group,rsicd}.json`，含summary和config快照；
训练后共同汇总为`outputs/web_native_seed11/summary.json`。包中包括这些报告与各训练日志/config/provenance/metrics/summary及数据audit。
仅跑baseline-only时不会创建训练汇总；若该目录历史已存在训练报告，打包会一起收录已有内容。

## 5. 如何决定继续

先确认原生起点、原生official与项目step0的功能等价，以及单轮适配增益。
看每个方向R@1/5/10、原unique MR、多图按图/按组均衡MR，以及RSICD是否下降；不能只盯loss或最大caption组频率带来的按图收益。
best仍按固定unique-val小batch loss选择，latest为完整覆盖终点；全局检索使用全部候选。
本轮单seed属于开发探索。主候选有正向且有用的收益后，再补adapter-only等消融和seed23/47，避免先把巨大矩阵跑完。
如果Web在同域提高、RSICD下降，应明确报告领域专化权衡，再考虑更丰富caption、数据混合或表示保持目标；本轮不把这些轴同时加入。
新路线无法消除SkyScript短caption、粗标签和同义假负例瓶颈。后续VRSBench/同图multi_objects可以继续用于RQ2，并保持图片与训练范围固定。
论文贡献应围绕领域适配证据、重复caption处理、数据规模/文本粒度和稳定性分析，不能把直接套LoRA包装为全新算法。

官方模型搭配依据：[Meta DINOv3 dino.txt加载入口](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/dinotxt.py)，默认Web LVD1689M backbone与配套视觉head/text权重。
历史结果见[交接第2.5/2.6节](PREPARE_HANDOFF.md#25-m4-awebsat-matched-seed11-短预算-pilot-通过)。
