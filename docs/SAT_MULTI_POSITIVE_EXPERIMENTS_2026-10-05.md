# SAT 多正例实验：实施与启动（2026-10-05）

本轮继续使用 adapter256＋全文本 LoRA rank8，永久冻结 SAT backbone 和原视觉 head。训练集按完整 caption 组恢复图片；保留原 unique-val 的 loss 与 best 选择规则，新增同文多图检索。当前本地无完整 CSV、图像、官方权重或 CUDA GPU，以下为已实现的服务器运行流程，未实际启动 GPU 实验。

## 实验矩阵

| 代号 | 图片池与采样 | 训练目标 | 用途 |
| --- | --- | --- | --- |
| A，历史基线 | 原每组固定一张图 | 单正例交叉熵 | 复用已完成的 seed11 adapter＋LoRA，不重训 |
| B，rotate | 恢复训练组内所有可用图片，每组访问选一张，跨组 epoch 轮换 | 单正例交叉熵 | 观察真实图片覆盖增加的效果 |
| C，multipos | 同一恢复图片池，每组访问最多两张不同图片，singleton 保留 | 均匀多正例交叉熵 | 利用已知多匹配关系 |
| D，maskpos | 与 C 使用同一采样器、seed、batch | 单正例目标，但从分母屏蔽同组非对角线项 | 对照显式吸引所有正例与只移除假负例 |

B 与 C 的 batch 组数不同，因此 B→C 的差异包含 batch 构成和目标两方面。C 与 D 的图像顺序完全相同，更适合诊断目标差异。每组最多两图是每次访问的采样设置，不是把组内其它图删除。不同组仍可能有未标注的语义匹配；本轮没有据模型相似度自动合并。

三组新实验均从同一官方初始化开始，seed11、物理 batch16、梯度累积4、1710 optimizer steps、warmup171、原学习率与衰减，queue0、增强关闭、训练 workers0。每组共109,440次图像曝光，与历史 A 相同。

**1710step 不表示恢复后图片池的三个 image epoch。** 新采样器的 group epoch 是每个 caption 组访问一次，尾部不足完整 batch 的组本轮丢弃，下一轮重新洗牌；组内图像循环轮换。每个 batch 优先完整填满16条，单例组贡献一张图，其余组贡献至多两张；若只剩一个空位则该组本次只取一张，保证同组不跨 batch。充分覆盖新图片池的预算需依据恢复数量和本轮覆盖日志另行确定，1710不是后续训练上限。

## 1. 恢复图片清单

服务器先保留原两份 unique 清单及原图目录。使用**此前同一 top30 筛选版本的完整候选 CSV**，不要输入已压缩为40,550行的一图一文 selection CSV，也不要直接输入未经 top30 筛选的其它版本。工具读取 `filepath` 和 `title_raw`，默认仅 images2/images3；它不重新执行 top30 评分筛选。

```bash
cd /root/autodl-tmp/Dinov3txtforSAT

.venv/bin/python tools/prepare_skyscript_groups.py \
  --csv /实际路径/原top30候选.csv \
  --images-root /实际路径/包含images2和images3的目录 \
  --train-reference assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.jsonl \
  --val-reference assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl \
  --output-dir assets/data/manifests/skyscript_caption_groups_v1
```

`--images-root` 是 `filepath` 中 `images2/...`、`images3/...` 的共同父目录。原始 CSV 与这个目录的服务器绝对路径尚未从本地资产确认，因此以上两处需替换为实际值，其余参数固定。

输出 `train_grouped.jsonl`、`val_grouped.jsonl`、`audit.json`。所有验证 caption 的兄弟图片都保持 val 身份，原代表图片和 canonical caption 保留。组 ID 为完整 caption 的空白归一化与 casefold 字符串。审计记录组数、图片数、单例组、大小分布、缺图和排除记录，没有新增 SHA 校验。

默认任何可恢复图片缺失都会报错。若服务器只保留了部分额外图像，可显式传 `--allow-missing`，但所有原代表图片必须存在；审阅时需说明缺图数。工具不会覆盖已有输出目录；报错时不会发布半份数据。首轮应查看 audit 中 train/val 组数是否为36,495/4,055，是否有大量缺图，抽查高频组图片与完整 caption 的可见性。文本相同只定义已有弱监督关系，不保证每条 caption 人工无误。

## 2. tmux 启动

当前六组复现继续使用原脚本与数据。新流程应等 GPU 空闲后运行；也可先仅准备图片清单。

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
tmux new -s sat-multi-positive
bash scripts/run_sat_multi_positive.sh
```

按 Ctrl+B 再 D 脱离，`tmux attach -t sat-multi-positive` 返回。默认顺序 B→C→D，每组训练后保存 step0／best／latest 三份权重；验证每200step及step0／正常终点强制执行。重复运行跳过完整训练，未完成则从 latest 恢复；不得修改配置后继续 resume。

只训练：`bash scripts/run_sat_multi_positive.sh --train-only`。
三组均完成后只评测：`bash scripts/run_sat_multi_positive.sh --evaluate-only`。
只训练后也可直接再次执行默认命令，跳过训练补齐评测。

## 3. 评测与选择规则

训练中的 best 仍由原4,055对 unique-val 的固定16候选 loss 选择，不使用扩展 val 选择权重。新训练 loss 的数值不能同 A 直接比较。

默认完整检索在 A/B/C/D 各自的 step0、best 上执行三套固定任务，共24份报告：

1. 原 SkyScript unique-val：4,055图与4,055文本，一对一正例，历史对照。
2. 新 SkyScript grouped-val：原4,055个验证 caption 组的全部保留图片，文本去重。文本→图像命中组内任意已标注图片算正确；图像→文本对应唯一组文本。双向 R@1/5/10 以及按 caption 组均衡的图→文与 mean Recall 都进入报告。
3. 原 RSICD-val：1,094图与5,470文本，沿用每图多个 caption 的既有标注。

扩展任务上必须同时重评旧 A 与新模型，不能把新候选池 Recall 和旧池 Recall 直接相减。这里 Recall@K 是有至少一个已知正例进入 topK 的查询比例，不代表找回全部正例。原始 grouped-val 图片数量取决于真实恢复清单，代码不预设。

现有 A 权重目录 `outputs/skyscript_sat_adapter_textlora_3epoch_seed11` 必须保留，默认评测阶段需要其 config、summary、step0、best。不会重新训练 A。新评测报告集中存放于 `outputs/sat_multi_positive_seed11`，原 A 目录不写入新报告。

## 4. 观察日志与下载

新训练 `metrics.jsonl` 包含不同 caption 数、每个 query 正例数分布、非对角线正例数、singleton query 比例，以及累计见过的不同图片数和图片池覆盖率。`training_summary.json` 报告 caption 组规模、group epoch 和样本曝光；checkpoint 保留采样器计划与轮换游标，暂停后能够精确恢复。

默认流程结束生成：

```text
outputs/sat_multi_positive_seed11_reports.tar.gz
```

该包包含三组训练日志／配置／报告、新旧模型24份共同任务检索报告与数据 audit，不含 `.pt` 权重。请下载这个包供审阅。仅训练模式的包不含尚未执行的检索，之后默认或仅评测流程会刷新报告包。

## 本地验证范围

数据工具使用合成 CSV 与图片检查划分边界、缺图、重复与原子发布。采样器检查组内轮换、单例组、预取与严格恢复。loss 检查 float32/BF16 下单例组与原损失及梯度一致、多正例梯度以及屏蔽对照。CPU tiny model 检查 B/C/D 连续训练和中断恢复的最终参数一致、冻结backbone不变、原验证规则与滚动三份权重。完整测试308项通过，Ruff、Bash语法、CLI帮助和diff检查通过；这些检查不替代服务器真实数据审计、GPU吞吐或实验效果。
