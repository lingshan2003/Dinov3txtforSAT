# 小规模短/详细caption 1:1混合试验

日期：2026-10-08。本地上传数据包已构建，服务器完整训练3轮、评价自身验证集与RSICD-val的代码和脚本已准备。实际数据检查结果见本文末尾；服务器实验尚未启动。

## 试验定义

- 模型：原生Web DINOv3.txt，冻结视觉backbone/head，训练adapter256及全部文本block/projection LoRA（rank8、alpha16）。
- 训练图片16,000张：8,000张使用polished `title_raw`短描述，8,000张使用OpenAI版 `title_multi_objects`详细描述。
- 验证图片1,600张：800张短描述、800张详细描述；另外在同一批1,600张图片上提供全短、全详细两套评价。
- 每张图片在本次训练中固定使用一种caption；比例每轮均为1:1。3轮完整遍历16,000张训练图，共48,000次图文曝光。这不是每张图片的短/长两种描述都训练3轮。
- batch16、accumulation4：每图片epoch1,000个microbatch，3轮共3,000个microbatch、750次optimizer更新，warmup75。
- 每200step验证，强制step0及终点750验证；仅保存step_0000000.pt、best.pt、latest.pt。
- queue0、无随机裁剪、训练seed11。默认仅训练这一组；未自动新增一组短描述-only训练。

先做这种固定caption、不同图片的比例混合，复用现有训练实现，避免首轮同时引入动态多caption选择与新loss。最终评价检查同一批val图片的三种文本条件。多视角正例训练仍保留在[后续方案](MIXED_CAPTION_TRAINING_PLAN_2026-10-08.md)。

## 数据选择与边界

从本地两份CSV共有的images2/images3图片中选择，每个规范化原短caption组只保留一张图片，并限制每个OSM对象只出现一次。不是随机抽取16,000个图片文件，也不是保留原37万池的图片频次分布。

原短caption组的train/val归属沿用历史seed23、最低4,055组归val的确定性拆分。生产端需要先用全部40,550个原短组重建该归属，不能在共同图片子集上重新算90/10。这里复用既有拆分优先级算法，不新增文件SHA验证。

服务器安装必须用两份原固定unique manifest核对每条选中短caption的split。若不匹配，停止而非重新分配。图片的两个caption版本一起归属于同一split。对出现在原train/val两侧的OSM对象作保守排除；这仍不等于已经完成邻接区域或近重复影像的地理隔离。

为减少首轮超长及粗描述重复，候选短句至多20个空白词，详细句12–50个空白词；这不是77 BPE tokens的等价检查。两套caption的规范化文本都要求跨所选图片及split无精确重复，避免已知同文假负例。筛选形成偏向信息较丰富且标签较独特的诊断子集，不代表全SkyScript的均衡随机样本。

服务器用真实tokenizer对两种文本适配77-token上下文，保留原文与实际输入、记录回退情况。若适配后产生caption碰撞，应停止并明确原因，不静默删除图片或改变1:1比例。

## 上传与准备流程

本地生产器从原images2.zip/images3.zip直接流式读取17,600张选中图片，只打包原始JPEG字节及短/长标注，避免完整解压。读取使用ZIP自带CRC。服务器优先复用已有图片，仅补缺图，不覆盖不匹配图片。

默认上传包：`outputs/skyscript_mixed_pilot_16k_1to1_v1.zip`。

将代码同步到服务器，并将该ZIP上传到项目的`assets/data/raw/skyscript/`。安装器输出到`assets/data/manifests/skyscript_mixed_pilot_v1/`，不覆盖已有实验数据。

```bash
cd /root/autodl-tmp/Dinov3txtforSAT
.venv/bin/python tools/install_skyscript_mixed_pilot.py \
  --bundle assets/data/raw/skyscript/skyscript_mixed_pilot_16k_1to1_v1.zip
```

安装器需要服务器已有的原train/val两份unique manifest、DINOv3源码及BPE词表。原图片root从旧manifest推断，不需要用户手填服务器图片路径。

安装结束先预检，再启动tmux：

```bash
bash scripts/run_web_mixed_pilot.sh --preflight-only
tmux new -s web-mixed-pilot
bash scripts/run_web_mixed_pilot.sh
```

按Ctrl+B再按D离开会话；重新查看：

```bash
tmux attach -t web-mixed-pilot
```

中断后重新运行同一脚本会使用latest恢复；已完成训练应跳过训练并继续未完成的评价。不要为了恢复修改原config。默认从官方权重初始化，不继承37万图片实验的adapter/LoRA。

## 评价与报告

原生模型、step0、best和latest都评价相同的四个候选池：

1. 本试验1:1混合val，1,600图/1,600文。
2. 同图片全短val，1,600图/1,600文。
3. 同图片全详细val，1,600图/1,600文。
4. 原固定RSICD-val，1,094图/5,470条caption记录。

保留每池双向R@1/5/10、MR及相对该池原生起点的变化，主观察为完整3轮latest。best按固定mixed-val loss保存，不根据RSICD挑选。RSICD仍是外部开发验证，不是首次触碰的最终test。

本pilot的1,600候选池与原4,055/33,118池不同，因此新MR不能直接和历史17.78%等数值相减来证明混合有效。首轮可以判断混合方案能否训练、在各池相对原生有什么变化；要识别混合相对只训练短句的净收益，需要补同图、同预算的S-only控制。

默认报告包为`outputs/web_mixed_pilot_seed11_reports.tar.gz`，包含配置、训练/验证日志、数据审计和检索报告，不含权重或图片。实验完成后下载该报告包用于审阅。

## 本地构建与检查结果

上传包已生成：`outputs/skyscript_mixed_pilot_16k_1to1_v1.zip`，261,656,598字节，约261.7 MB / 249.5 MiB。只包含17,600张选中图片及元数据，无须上传原始两个大ZIP。

本地独立核对通过：训练8,000短＋8,000详细、验证800短＋800详细；17,600个原短caption组、17,600个OSM对象，全部35,200条短/详细备选文本无规范化精确重复。选中图片的ZIP CRC、图像结构及完整像素解码全部通过。保留原始图片字节，图片尺寸不同，由原生模型预处理统一到输入尺寸。

筛选前合格候选组为train20,164、val2,188。选中短句平均7.30个空白词、详细句28.54个词；此统计为真实tokenizer处理前的原文。详细描述来自OpenAI CSV的`title_multi_objects`字段，仍是SkyScript元数据式描述，不能等同人工撰写的视觉真值。

本地检查记录为`outputs/skyscript_mixed_pilot_16k_local_checks.json`。服务器的原manifest边界核对、真实tokenizer检查、GPU训练和完整检索由上传后的流程执行。

代码检查：全量431项测试通过，包含真实producer→installer→runner的合成数据集成、截断文本、完整默认编排、恢复与完成跳过；新增代码Ruff、启动脚本Bash语法、CLI帮助及diff检查通过。
