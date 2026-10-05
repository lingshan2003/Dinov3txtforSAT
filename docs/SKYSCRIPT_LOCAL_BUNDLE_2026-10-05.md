# SkyScript 两个 top30 版本与本地上传包（2026-10-05）

已在本地根目录找到两份 CSV 和 images2.zip／images3.zip。服务器空间有限，因此本轮直接从 ZIP 读取选中图片，生成候选上传包，没有完整解压两份原档，也没有重新建立 train/val 划分。

## 两份 CSV 的实际区别

| 项目 | OpenAI top30 | LAION-RS language-polished top30 |
| --- | --- | --- |
| 文件 | `SkyScript_train_top30pct_filtered_by_CLIP_openai.csv` | `SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv` |
| 官方筛选模型 | 原始 OpenAI CLIP | 在遥感 LAION 子集上适配的 CLIP-laion-RS |
| 描述生成 | OSM 标签按规则生成 | OSM 标签经 ChatGPT 改写为自然语言 |
| 本地列 | filepath、title、title_multi_objects、similarity_CLIP_openai | filepath、title_raw、title |
| 本地总行数 | 1,518,890 | 1,518,888 |
| images2＋images3 行数 | 406,333 | 370,317 |
| 当前主要文本列 | 若使用此版，一般为 title；多对象列是另一个目标 | 已有实验使用 title_raw |
| 全文件主要文本规范化唯一数 | title：70,003 | title_raw：52,923 |

官方版本定义见[SkyScript 下载说明](https://github.com/wangzhecheng/SkyScript#download-captions)。本地数字通过 CSV 流式读取取得；分组仅按空白归一化及 casefold，不进行 NFKC、同义句合并或内容哈希验证。

两份 CSV 全部图片路径交集为936,351，images2＋images3路径交集227,746。因此不能把它们理解成完全相同图片只换了文字；直接切换同时改变筛选模型、图片池与文本目标。

OpenAI 版示例：`a satellite image of man made bridge`；polished 对应内容句可能是 `man-made bridge.`，带模板的 title 为 `An aerial image. It shows: man-made bridge.`。另一个共享图片的 OpenAI 文本为 `a satellite image of building, office of energy supplier`，polished title_raw 为 `building.`：改写有时简化细节，不能把 polished 自动视为信息更丰富或更准确。

**本轮继续用 polished 的 title_raw。** 自然描述对通用文本 encoder 的语言习惯可能更合适，遥感筛选模型也可能更好地判断遥感图文关系；这些是合理假设，不是已经验证的性能结论。OpenAI 版保留规则化属性／周边对象信息，对细粒度问题也可能有价值。这里最确定的选择依据是已有所有实验都使用 polished，本轮只改变多图覆盖与正例处理。若比较两种文本风格，应在共同图片上做独立控制，而不是直接拿不同候选池的 Recall 相减。

`title_raw` 在 polished 文件中表示去掉统一视角前缀的内容句，仍是语言改写后的文本，并非原始 OSM 标签。

## 已生成的本地文件

```text
outputs/skyscript_polished_top30_images23_candidates_v1.zip
outputs/skyscript_candidates_bundle.audit.json
outputs/skyscript_candidates_bundle.log
```

上传包实际5,937,046,977字节，约5.94GB／5.53GiB；两份原ZIP合计19,434,569,022字节，约19.43GB。包内保留370,317张图片及其原始字节，不裁剪、不缩放、不重新编码JPEG；图片直接存储，较容易压缩的候选CSV另行压缩。

候选数据包含40,550个规范化caption组，22,713个singleton组，最大组2,777张。包内有 `_metadata/candidates.csv`（filepath／title_raw）、`audit.json` 和安装说明。所有选中图片在读取ZIP时通过其自带CRC；另检查了输出ZIP目录的图片数、字节数与压缩设置，未重算SHA。没有逐图语义标注审核。

重建命令（不会覆盖已生成的包）：

```bash
.venv/bin/python tools/bundle_skyscript_candidates.py \
  --csv SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv \
  --archives images2.zip images3.zip \
  --output outputs/skyscript_polished_top30_images23_candidates_v1.zip
```

## 上传与服务器安装

先同步新增工具及本轮实验代码，上传上述单个ZIP至服务器：

```text
/root/autodl-tmp/Dinov3txtforSAT/assets/data/raw/skyscript/skyscript_polished_top30_images23_candidates_v1.zip
```

然后执行：

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
.venv/bin/python tools/install_skyscript_group_bundle.py \
  --bundle assets/data/raw/skyscript/skyscript_polished_top30_images23_candidates_v1.zip
```

本地目前没有服务器的固定train/val manifest，因此包内是候选池；最终split由服务器既有两份unique清单决定。安装器从这些清单推断原来的图片root，不需要手填CSV或图片根路径。先检查组关系、所有原代表图片是否在包内、已有图片大小与CRC，再复用符合的旧图并恢复其它图。保持原代表图片路径和caption，所有验证组的兄弟图片仍归val。候选CSV写到原root的 `_metadata/group_candidates_v1.csv`，分组清单写到配置要求的 `assets/data/manifests/skyscript_caption_groups_v1`。

工具拒绝覆盖不匹配图像、候选CSV或已存在的分组输出目录；文件复制中断后可以复用已正确恢复的图片。安装器只生成数据，不启动训练。它不自动删除上传ZIP。

候选图片本体约5.87GB，加上清单后图库约6GB。若上传包和解压数据同时保留，这部分数据占用接近12GB；已有代表图会复用，新增空间略小。安装成功并确认清单后，可删除服务器的上传包，只保留图库与清单，本地保留备份。

GPU空闲后按[多正例实验说明](SAT_MULTI_POSITIVE_EXPERIMENTS_2026-10-05.md)运行tmux启动脚本。整个过程不会重建既有验证划分或改变正在进行的复现。
