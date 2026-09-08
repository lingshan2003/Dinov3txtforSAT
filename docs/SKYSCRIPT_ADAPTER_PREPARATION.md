# SkyScript 小型 Adapter 实验准备

状态：仅准备代码与配置，尚未授权或启动训练。数据下载完成前不要使用
`--allow-missing` 绕过缺图检查。

## 1. 文本选择

polished CSV 的常见字段是 `filepath,title_raw,title`。`title` 形如：

```text
An aerial image. It shows: Hospital providing healthcare services.
```

不得对此字段复用 ChatEarthNet 的“第一完整句”策略，否则大量 caption 会坍缩为相同的
`An aerial image.`。首轮使用 `auto`：若存在 `It shows:`，提取最后一个 marker 后的内容并统一
构造成上面的短句；若是当前 OpenAI-CLIP top30 CSV 的普通 `a satellite image of ...` title，
则完整保留。原 CSV 保持只读。

当前完整的 language-polished CSV 只读扫描结果是：

- 1,518,888 条记录、1,518,888 个不同 filepath；
- `title` 的 `It shows:` 正文规范化空白后 100% 等于 `title_raw`；
- 1,496,105 条使用 `An aerial image`，22,783 条使用 `A satellite image`；
- `title_raw` 有 52,923 种规范化文本，约 96.52% 的记录共享重复语义；
- `title_raw` 的空格分词长度 p50=4、p95=10、max=62。

因此首轮不是从百万条中普通随机抽样，而是每个精确 caption 最多保留一张图。当前快速路径只用
images2+images3 的 40,550 个唯一语义；未来全分片实验可从 52,923 个唯一文本中固定抽取 50k。
这些分布用于判断“数据与最小 adapter 能否工作”，不作为最终训练分布。

旧 OpenAI-CLIP top30 CSV 有 1,518,890 条记录，与新版有 936,351 张相同图片。旧文件保留用于在
相同图片上比较 OSM 风格 title、polished `title_raw` 和统一模板 title，不用完整未配对数据混淆
文本质量与过滤样本差异。

## 2. 从大 ZIP 定向抽取

ZIP 必须完整下载，但无需完整解压。`tools/extract_skyscript_subset.py` 有两种显式模式：默认按
规范化 `title_raw` 每组保留一张、固定选择全局 50k；`--all-csv-rows` 则抽取 CSV 引用的全部
约 151.9 万张图。它读完 ZIP 中央目录并确认每个目标只命中一次后，才会写图片；读取被选中的
成员时由 Python ZIP 实现校验 CRC，未选成员不会解压。

### images2 + images3 快速验证路径

当前 CSV 已确认：`images2` 和 `images3` 范围共有 370,317 条图片记录；在两包并集上按
规范化 `title_raw` 去重后是 40,550 个语义。seed 11 选择的代表图为 images2 20,391 张、
images3 20,159 张。两个 ZIP 都已下载时先 dry-run：

```bash
python tools/extract_skyscript_subset.py \
  --csv SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv \
  --archives /替换为真实路径/images2.zip /替换为真实路径/images3.zip \
  --output-root assets/data/raw/skyscript/images23_unique40550 \
  --selection-output assets/data/raw/skyscript/images23_unique40550.csv \
  --audit-output assets/data/raw/skyscript/images23_unique40550.extract.audit.json \
  --group-field title_raw \
  --selection-prefix images2 images3 \
  --include-prefix images2 images3 \
  --max-per-group 1 \
  --limit 40550 \
  --seed 11 \
  --bundle-output assets/data/raw/skyscript/skyscript_images23_unique40550.zip \
  --dry-run
```

确认 dry-run 的 `selected_images=40550`、`missing=0` 后删除 `--dry-run` 执行。脚本是在两包
并集上先去重，不是各自去重后拼接，所以共享的 caption 不会重复进入训练。

如果不能同时保留两个大 ZIP，仍使用相同的 `--selection-prefix images2 images3`，但第一次只传
`images2.zip` 并设置 `--include-prefix images2`；第二次只传 `images3.zip` 并设置
`--include-prefix images3`。两次使用同一个 output-root 和 selection-output，不同的 audit/bundle
文件名。第二次会验证并复用字节完全相同的 40,550 条全局选择清单。

完成两包抽取后，将 40,550 个唯一 caption 确定性拆为 36,495 train 和 4,055 validation：

```bash
python tools/split_skyscript_selection.py \
  --input assets/data/raw/skyscript/images23_unique40550.csv \
  --train-output assets/data/raw/skyscript/images23_unique36495_train.csv \
  --val-output assets/data/raw/skyscript/images23_unique4055_val.csv \
  --audit-output assets/data/raw/skyscript/images23_unique40550.split.audit.json \
  --group-field title_raw \
  --val-count 4055 \
  --seed 23
```

拆分工具要求输入 caption 已全局唯一；发现重复 caption 或 filepath 会失败。train/val 保持原 CSV
顺序，并在审计中记录输入、两个输出的 SHA-256。

### 其他抽取模式

首轮 adapter 只需要 50k。先运行只读 dry-run：

```bash
python tools/extract_skyscript_subset.py \
  --csv SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv \
  --archives /替换为真实路径/images1.zip \
  --output-root assets/data/raw/skyscript/selected_unique50k \
  --selection-output assets/data/raw/skyscript/selected_unique50k.csv \
  --audit-output assets/data/raw/skyscript/extract_images1.audit.json \
  --group-field title_raw \
  --max-per-group 1 \
  --limit 50000 \
  --seed 11 \
  --include-prefix images1 \
  --bundle-output assets/data/raw/skyscript/skyscript_unique50k_images1.zip \
  --dry-run
```

dry-run 报告 `selected_members` 非零且没有 missing/ambiguous 错误后，删掉最后的 `--dry-run`
再执行。处理 `images2.zip` 时把 archive、prefix、audit 和 bundle 文件名相应改成 `images2`；
`selection-output` 保持不变，脚本只会复用字节完全一致的全局选择 CSV，不会覆盖不同内容。

因此可以逐个循环：下载一个约 9GB 的原 ZIP，定向抽取该分片，生成小 ZIP 并上传，再处理下一个。
七个小 ZIP 在服务器上解压到同一个根目录。小 ZIP 使用 `ZIP_STORED`，因为 JPEG 已经压缩，再做
DEFLATE 通常只会消耗时间而几乎不减小体积。不要在确认小包上传成功且 SHA-256 有记录前删除原 ZIP。

若希望把完整 top30 的约 151.9 万张图也保存下来，对每个大 ZIP 使用 `--all-csv-rows`。此模式
要求指定分片前缀，并为每个分片使用不同的 selection/audit/bundle 文件名：

```bash
python tools/extract_skyscript_subset.py \
  --csv SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv \
  --archives /替换为真实路径/images1.zip \
  --output-root assets/data/raw/skyscript/top30_all \
  --selection-output assets/data/raw/skyscript/top30_images1.csv \
  --audit-output assets/data/raw/skyscript/top30_images1.audit.json \
  --include-prefix images1 \
  --all-csv-rows \
  --bundle-output assets/data/raw/skyscript/skyscript_top30_images1.zip \
  --dry-run
```

这里的 dry-run 会报告该分片真正需要解压的图片数量和未压缩字节数，可先据此决定磁盘和上传计划。
完整模式的分片 selection CSV 只包含对应 `imagesN` 的记录；50k 模式的 selection CSV 则始终是
同一个全局 50k 清单。

脚本具有以下失败边界：

- 目标缺失、在多个 archive 中重复命中、路径包含 `..` 或单文件异常超过 100 MiB时停止；
- 输出文件已存在且 CRC/大小不同则拒绝覆盖；
- 选择 CSV、bundle 和 audit 使用临时文件后原子发布；
- 空间检查按“抽取文件 + 可选小 ZIP”预留，`--dry-run` 不写任何产物。

## 3. 生成 manifest

先确认真实目录和 CSV 表头；不要照抄未经确认的压缩包内部路径：

```bash
find assets/data/raw/skyscript -maxdepth 3 -type f | head -50
python - <<'PY'
import csv
from pathlib import Path

path = Path("assets/data/raw/skyscript/替换为实际top30.csv")
with path.open(encoding="utf-8-sig", newline="") as handle:
    reader = csv.reader(handle)
    print(next(reader))
PY
```

快速路径已经生成拆分后的 CSV。主基线直接使用 `title_raw`，不再次抽样：

```bash
python tools/prepare_skyscript.py \
  --csv assets/data/raw/skyscript/images23_unique36495_train.csv \
  --images-root assets/data/raw/skyscript/images23_unique40550 \
  --split train \
  --caption-field title_raw \
  --caption-mode full \
  --output assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11.jsonl \
  --audit-output assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11.audit.json
```

对 `images23_unique4055_val.csv` 执行相同命令并把 `--split` 改成 `val`，输出
`skyscript_images23_val_raw_unique4055_seed23.jsonl`。

模板消融使用同一 selection CSV 和图片，只把 `caption-field` 改成 `title`、`caption-mode` 改成
`auto`，输出独立 manifest。此时 `auto` 会把 aerial/satellite 两种前缀统一为
`An aerial image. It shows: ...`，不引入人为的双前缀变量。

未来取得基本结果后，可再用官方 validation CSV 建立独立验证。它同样限制每个精确 caption
一张图，使一一配对 InfoNCE 指标不含已知的精确重复 false negative：

```bash
python tools/prepare_skyscript.py \
  --csv assets/data/raw/skyscript/替换为实际val.csv \
  --images-root assets/data/raw/skyscript/替换为实际图片根目录 \
  --split val \
  --caption-field title \
  --caption-mode auto \
  --max-per-caption 1 \
  --output assets/data/manifests/skyscript_val_auto_unique.jsonl \
  --audit-output assets/data/manifests/skyscript_val_auto_unique.audit.json
```

## 4. 训练前门槛

先用真实 dino.txt tokenizer 在保留完整语义的前提下做 complete-word backoff。即使所有 caption
已经能放入 77 tokens，这一步也会产生独立、可审计且内容不变的 global77 manifest。不要改用
旧的第一句截断器。

```bash
python tools/prepare_global77_manifest.py \
  --input assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11.jsonl \
  --output assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.jsonl \
  --audit-output assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.audit.json \
  --dinov3-repo external/dinov3 \
  --bpe-vocab assets/checkpoints/bpe_simple_vocab_16e6.txt.gz \
  --context-length 77 \
  --strategy complete-word-backoff
```

对快速路径的 validation manifest 执行相同转换并输出
`assets/data/manifests/skyscript_images23_val_raw_unique4055_seed23_global77.jsonl`。随后运行只读
复核；任何 caption 仍超过 77 tokens 都会让命令返回非零：

```bash
python tools/audit_manifest_text.py \
  --manifest assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.jsonl \
  --dinov3-repo external/dinov3 \
  --bpe-vocab assets/checkpoints/bpe_simple_vocab_16e6.txt.gz \
  --output assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.text-audit.json \
  --require-fit
```

还需运行现有图像 overlap 工具，分别对 EuroSAT、RSICD manifest 检查精确文件和解码像素重叠。
该工具不能发现同一地理区域的不同裁剪；若 CSV 提供坐标，地理重叠应另行审计。

## 5. 训练代码边界

新增的 `image_adapter_bottleneck = 256` 是放在官方归一化图像 embedding 后的残差 MLP：

- 官方 Web DINOv3 backbone、dino.txt vision head、文本塔、文本投影和 logit scale 全部冻结；
- 仅训练约 105 万参数的图像 adapter；
- 最后一层零初始化，因此 step 0 保持官方 embedding；
- 首轮关闭随机裁剪与 negative queue；
- patch tokens 不经过这个全局 adapter，不能据此宣称 local alignment 得到改善。

先从 36,495 条 train manifest 固化 16 条 smoke 数据：

```bash
python tools/prepare_fixed_manifest.py \
  --input assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77.jsonl \
  --output assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77_fixed16.jsonl \
  --limit 16 \
  --audit-output assets/data/manifests/skyscript_images23_train_raw_unique36495_seed11_global77_fixed16.audit.json
```

对应配置为：

- `configs/skyscript_web_adapter_10step.toml`：固定 16 条的数值与 checkpoint smoke；
- `configs/skyscript_web_adapter_100step.toml`：36,495 train + 4,055 held-out val 的受限开发诊断。

这两个配置都不会自动运行。10-step 通过也不表示可以直接扩大训练；还需核验 step-0 表示、
validation 曲线、RSICD-val 保持性和 checkpoint resume。
