# SAT 四组联合三轮训练审阅（2026-10-04）

后续16份联合检索报告已完成审阅，最新结论见[联合完整检索审阅](SAT_JOINT_3EPOCH_RETRIEVAL_ANALYSIS_2026-10-04.md)。本文保留仅收到训练报告时的结论与评测复现命令。

四组联合训练正常完成。当前验证对比损失最好的配置是adapter + 全文本LoRA（含projection），终点为0.902339，比adapter-only的0.995599低9.37%。视觉head + 全文本LoRA也优于其两个单侧参照。Projection联合更新的收益较小，尤其adapter + projection几乎与adapter-only重合。

**这些是验证loss结论；上传没有retrieval报告，尚不能声称联合适配提高了全局Recall，也不能宣布替换adapter检索基线。** 先用本报告末尾命令补齐四组step0/best的SkyScript-val与RSICD-val检索，再判断实际互补性。

## 1. 来源与训练完整性

来源：`/Users/wangyue/Downloads/sat_joint_3epoch_reports.tar.gz`。四组各包含config、train.log、metrics.jsonl、validation.jsonl、optimizer_groups.json、provenance.json和training_summary.json，共28份文件，没有checkpoint或检索JSON。

- 四份保存配置与本地同名配置一致。项目提交均为`316b6c0c381ccf3e0fc63b1c2dce82a99789c2a0`；该提交相对文本实验提交`d1f8869`没有`src/dinotxt_rs`变化。
- 四组及此前五组共享的backbone权重、dino.txt权重、BPE词表、train/val清单的已存来源身份一致。只比较已有记录，未重算SHA256。
- 上游提交`6876159a11b4df116f30f667f8c9888617df0751`，RTX3090、Python3.12.3、Torch2.7.1+cu128、CUDA12.8与此前一致。
- 全部completed=true，1710/1710次optimizer更新、6840微批次，无恢复训练，无跳过更新，loss/梯度有限；日志未发现异常或warning。
- 每组171条训练记录（step10至1710、间隔10），10条验证记录（step0、每200step及终点1710），样本数4055，均完整。
- 四组验证loss在所有记录点严格下降，summary与validation.jsonl的最小值/终点一致；best均为step1710，latest也记录为1710。
- Train为36,495对样本，batch16×累积4、三个drop-last轮次，共109,440次样本曝光。Seed11、warmup171、余弦调度、BF16、queue0、无增强、shuffle=true，logit_scale固定100，head drop_path为0。

Adapter两组step0 validation loss为3.864123371702822，head两组为3.8641233303102993，差约4.14e-8，与旧单侧对应组一致。这支持未改变初始损失的判断，不能替代逐项embedding/logits核验。两组LoRA均为rank8、alpha16、dropout0，覆盖24层四个线性模块及最终projection，共97处。

没有上传实际权重，因此不能直接核验三份checkpoint实体、逐层增量及冻结参数零漂移。Summary和优化器记录支持更新范围符合设计；各优化器组在全部训练日志点都有正梯度范数，但这不证明每个LoRA插入点都更新相同，也不能比较不同组梯度范数来分配因果贡献。

## 2. 参数范围与成本

| 联合配置 | 可训练参数 | 初始学习率 | 日志窗口累计时间 | Torch峰值allocated |
| --- | ---: | --- | ---: | ---: |
| Adapter + 全文本LoRA（含projection） | 5,013,760 | image_adapter=1e-04，text_projection=1e-04，text_backbone=1e-04 | 25.39分钟 | 5.683GiB |
| 视觉head + 全文本LoRA（含projection） | 29,285,120 | vision_head=1e-05，text_projection=1e-04，text_backbone=1e-04 | 25.85分钟 | 6.224GiB |
| Adapter + projection全量更新 | 3,676,416 | image_adapter=1e-04，text_projection=1e-05 | 18.64分钟 | 3.936GiB |
| 视觉head + projection全量更新 | 27,947,776 | vision_head=1e-05，text_projection=1e-05 | 19.61分钟 | 4.157GiB |

Adapter为6个张量、1,054,976参数；head为32个张量、25,326,336参数。LoRA block组为192个A/B张量、3,932,160参数，projection LoRA为2个A/B张量、26,624参数；projection全量更新为一个权重张量、2,621,440参数。视觉backbone及logit_scale不在优化器中。

日志窗口时间包含窗口内验证/保存/IO，不含完整初始化、step0验证及末尾保存，不是纯GPU耗时或完整任务墙钟。Allocated是Torch累计峰值，不是设备总显存。旧视觉实验有OMP环境差异，不能直接据旧/新时间认定联合训练反而加速。本轮LoRA联合组约25–26分钟、projection联合组约19–20分钟，实际成本在当前设备可接受。

## 3. 联合与单侧对照

| 配置 | 终点validation loss | 相对对应视觉单侧loss下降 |
| --- | ---: | ---: |
| Adapter + 全文本LoRA | **0.902339** | **9.37%** |
| Adapter + projection | 0.994999 | 0.06% |
| Adapter-only（旧参照） | 0.995599 | — |
| 视觉head + 全文本LoRA | 1.088776 | 30.78% |
| 全文本LoRA-only（旧参照） | 1.327784 | — |
| 视觉head + projection | 1.506328 | 4.23% |
| 视觉head-only（旧参照） | 1.572853 | — |
| 文本末两层 + ln_final（旧参照） | 2.258260 | — |
| Projection-only（旧参照） | 2.457213 | — |

上述百分比是loss相对下降，**不是Recall增加，也不是百分点**。旧五组指标可追溯至此前两份训练包及[完整检索审阅](SAT_3EPOCH_RETRIEVAL_ANALYSIS_2026-10-04.md)。

- **Adapter + 全文本LoRA是下一步最值得检索验证的组合。** 比adapter-only低0.093261，比LoRA-only低32.04%。这支持在本轮设置下两侧联合对训练目标有额外收益；不是两项收益简单相加的证明。
- **Head + 全文本LoRA有明显联合价值。** 比head-only低30.78%，比LoRA-only低18.00%，仍落后于adapter-only及两种adapter联合配置。没有adapter的原架构适配路线仍有研究价值，不能因其暂时没有领先就解释为完全无效。
- **Adapter + projection与adapter-only几乎重合。** 终点差仅0.000600，曲线也几乎重合。当前单seed/学习率下没有有力的额外loss收益，不能将0.06%写成稳定改进，也不能据此断言projection在其他组合中无用。它确实在优化器中且有正梯度，日志不支持“根本没有训练”。
- **Head + projection有较小收益。** 比head-only低4.23%，收益弱于联合全文本LoRA。这里LoRA覆盖范围、参数化及文本学习率都不同，不能把差别完全归因为LoRA技术本身。

### 验证曲线

| step | Adapter+LoRA | Head+LoRA | Adapter+projection | Head+projection |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3.864123 | 3.864123 | 3.864123 | 3.864123 |
| 200 | 1.436946 | 2.023844 | 1.500131 | 2.510169 |
| 400 | 1.149873 | 1.466355 | 1.226476 | 1.980128 |
| 600 | 1.061502 | 1.284265 | 1.139814 | 1.757250 |
| 800 | 1.004910 | 1.195159 | 1.097447 | 1.638179 |
| 1000 | 0.958390 | 1.144343 | 1.043415 | 1.566326 |
| 1200 | 0.924861 | 1.107179 | 1.013055 | 1.527741 |
| 1400 | 0.909033 | 1.094022 | 1.003561 | 1.511035 |
| 1600 | 0.902370 | 1.089078 | 0.995017 | 1.506502 |
| 1710 | 0.902339 | 1.088776 | 0.994999 | 1.506328 |

600到1710step，四组按表列顺序继续下降14.99%、15.22%、12.71%、14.28%，再次说明约一轮的短训练不能代表三轮结果。1400到1710的额外下降分别仅0.736%、0.480%、0.853%、0.312%，但末尾学习率趋零，不能据此确认模型达到能力极限。当前也没有观察到验证loss反弹，不能将某一路线落后直接归因于过拟合。

Summary的final_loss是最后一个optimizer step的训练损失；metrics.jsonl末条loss是最后10个step的平均。二者不同是日志定义，不能误判为不一致。上表全部使用4055条验证集的固定定义，未混入训练loss。

## 4. 对adapter机制的含义与判断边界

本轮验证loss进一步支持“最终视觉表示上直接做残差修正是一条有效适配路径”：联合强文本适配之后，adapter路线仍领先head路线。它不直接证明adapter为何领先；最终表示的更新位置、非线性瓶颈、零残差初始化、学习率和优化难度都尚未分离。

LoRA与adapter的额外收益也说明“只要视觉adapter就足够，文本侧没有价值”过于绝对。Projection几乎没有给adapter增加loss收益，而全层LoRA有，提示当前文本适配需要的变化可能超出这一个末端矩阵在当前学习率下能够提供的变化；这只是下一步消融的假设。

所有结果仍是单seed，新增adapter会影响后续LoRA随机A矩阵的RNG消费，不能声称LoRA-only与adapter+LoRA逐项随机初始化相同。训练及验证loss对比集合都是16对；累积4次只扩大optimizer有效batch至64，不扩大单次对比候选池。完整4055候选的排序可能改变结论。RSICD结果未包含在本次包中，也未验证迁移收益。

当前先补检索，不根据loss排序立即扩展训练组合。SkyScript-val已用于loss选模，RSICD-val用于开发观察；最终论文仍需要留出的test及选定候选的多seed检验。

## 5. 下一步：tmux内补齐16份检索报告

在服务器仓库执行，使用每组训练时保存的config.toml。四组各评step0/best × SkyScript-val/RSICD-val；best与latest本次都为终点，不必再评同一步的latest。以下不会启动训练。

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
tmux new -s sat-joint-retrieval
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

(
  set -e
  for trial in adapter_textlora visionhead_textlora adapter_textproj visionhead_textproj; do
    run_dir="outputs/skyscript_sat_${trial}_3epoch_seed11"
    for dataset in skyscript rsicd; do
      if [ "$dataset" = skyscript ]; then
        manifest="assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl"
      else
        manifest="assets/data/manifests/rsicd_val_retrieval_v1.jsonl"
      fi
      for tag in step_0000000 best; do
        report="$run_dir/${dataset}_val_${tag}.json"
        if [ -f "$report" ]; then
          echo "Existing report, skipped: $report"
          continue
        fi
        .venv/bin/python -u -m "dinotxt_rs.cli.evaluate_${dataset}" \
          --config "$run_dir/config.toml" \
          --manifest "$manifest" \
          --checkpoint "$run_dir/$tag.pt" \
          --training-output "$run_dir" --split val \
          --output "$report"
      done
    done
  done
)
```

Ctrl-b再d脱离，之后用`tmux attach -t sat-joint-retrieval`返回。遇到失败会停止，解决后重跑可跳过已有报告；存在文件名仍需最终检查内容，不能单凭文件存在宣称评测完整。评测CLI原子写JSON，并拒绝覆盖已有报告。

完成后只打包检索JSON：

```bash
tar -czf sat_joint_3epoch_retrieval_reports.tar.gz \
  outputs/skyscript_sat_adapter_textlora_3epoch_seed11/*_val_*.json \
  outputs/skyscript_sat_visionhead_textlora_3epoch_seed11/*_val_*.json \
  outputs/skyscript_sat_adapter_textproj_3epoch_seed11/*_val_*.json \
  outputs/skyscript_sat_visionhead_textproj_3epoch_seed11/*_val_*.json
```

上传`sat_joint_3epoch_retrieval_reports.tar.gz`后，先核验16份报告的step、来源身份、候选池及step0一致性，再比较双向R@1/5/10、mean Recall和rank。重点是adapter+LoRA是否超过adapter-only，以及head+LoRA是否在无adapter路线中同时改善同域与RSICD；若loss收益未转化为检索，应检查全局排序及近重复正例等因素。
