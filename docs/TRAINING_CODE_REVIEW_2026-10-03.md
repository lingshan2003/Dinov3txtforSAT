# 训练代码审查与 checkpoint 精简建议

审查日期：2026-10-03。

修复进展（同日后续）：用户授权后已修正seed时机、终点验证、同分排名、滚动保存、异常写入、恢复日志与best核验、冻结温度修改和FP16溢出恢复；验证/monitor使用独立CPU RNG。新的正式配置每200 step验证与保存latest，强制step0/正常终点验证，保留step0/best/latest；历史配置显式保留numbered策略。原审查和复现结果保留下文，供了解变更依据。新策略与运行入口见[README](/Users/wangyue/Documents/ChatGPT/Dinov3txtforSAT/README.md)。本轮修复没有重跑真实GPU训练。

修复后的全量回归：162项测试通过；修改涉及的Python文件通过ruff检查，`git diff --check`通过。测试覆盖三文件保留、最佳状态不改善、非整除终点验证、latest恢复参数一致、best回滚日志归档、异常原子写入、初始化seed及检索同分规则。FP16更新跳过以CPU测试替身验证控制流，真实CUDA AMP仍需在训练机器上确认。

**结论：当前核心梯度累积、参数分组和视觉 backbone 冻结实现总体正确；存在需要修正的复现性、验证终点、恢复日志及文件保存问题。现有低分不能直接归因于“代码没有训练”，也不能在这些问题未控制时全面排除微调方案。只保留初始化与当前 best 可以显著限制 checkpoint 占用；保留滚动 latest 则能避免恢复时退回很早的最佳点。**

本轮只审查和复现，未修改训练源码、历史配置或 checkpoint，未启动真实模型训练。使用现有 `.venv/bin/python`（PyTorch 2.7.1）进行 CPU 小模型测试，不安装依赖，不重新校验真实模型权重 SHA-256。18 项 trainer/config/retrieval 测试及另一组 6 项模式/冻结/验证测试均通过，其中验证分组测试重复，合计 23 个不同测试。针对性测试中，合成输入的文件身份校验使用测试替身；因此这些通过结果不构成真实权重身份验证。另运行下述边界复现。

## 1. 已确认的问题

### 1.1 [P1] 实验 seed 在 adapter 构造后才设置

入口 `src/dinotxt_rs/cli/train.py:37` 加载模型，43 行构造 adapter；直到 `src/dinotxt_rs/training/trainer.py:346` 才调用 `seed_everything(config.experiment.seed)`。adapter 的 down 层默认随机初始化，因此 experiment seed 没有覆盖其初始化。

CPU 复现：构造前使用不同全局 RNG、之后均设 seed11，两份 adapter 的 down 权重不同。up 零初始化使 step0 输出完全相同，但首个 up 梯度已经不同，最大差值约 0.1596。**step0 parity 通过无法排除这个问题。** 它影响同 seed 复跑和不同方案的配对初始化，不能据此撤销所有历史改善结果。完整 checkpoint 恢复会覆盖可训练参数，不受这次重新构造的随机初始化影响。

修正：在官方模型与所有新增可训练模块构造之前设置实验 seed；后续训练入口保留必要的 RNG 策略并明确记录。未来随机重训 head、LoRA 初始化也需要覆盖。

### 1.2 [P2] 验证间隔不整除终点时，没有最后一步验证

`training/trainer.py:705` 只在 `global_step % validation_every == 0` 验证；退出训练后仅保存权重，没有补验证。summary 的 `validation.final_loss`（833 行）实际表示最近一次验证的 loss，可能不是终点状态。

CPU 小模型复现：训练 5 step、每 2 step 验证，验证记录只有 `[0, 2, 4]`，最终权重为 step5，但报告中的 final validation 来自 step4。现有 500/50 日程终点恰好整除，所以这个复现不说明旧 step500 验证缺失。未来 1710/200 日程会暴露此问题。

修正：正常训练完成时强制验证一次，已在终点验证则去重；报告显式记录 `last_validation_step`。执行中断是否验证应有独立规则，避免每次中断改变实验采样点。

### 1.3 [P2] 从较早 checkpoint 恢复会追加重复或失效日志

`training/trainer.py:484` 起恢复 checkpoint 状态，但不会处理已有 `metrics.jsonl`、`validation.jsonl`、monitor 等日志中高于恢复 step 的记录。

CPU 复现：完成 step5 后从 step4 的 best 恢复到 step5，训练日志为 `[1,2,3,4,5,5]`。这也对应崩溃前日志已经写入、checkpoint 尚未更新的情况。正常在已保存的停止点恢复，且日志没有超前记录时，不必发生此问题。

另外，恢复时对已有 best.pt 只检查文件存在（525–528 行），没有核对它的 payload step 与恢复状态中的 best_step；如果恢复较旧编号权重，而 best.pt 已被后续训练更新，可能出现 best 元数据与文件不一致。

修正：定义明确的回滚/分支恢复规则。回滚前保留原日志，重新建立截至恢复 step 的有效日志；核对 best 文件与恢复状态。或显式在新输出目录建立恢复分支。仅保留 best 的方案尤其需要这一处理。

### 1.4 [P2] 刷新 best 仍永久留下每个编号 checkpoint

`training/trainer.py:756–764`：每次验证改善先写 `step_*.pt`，再更新 best.pt。定期保存也写编号文件；旧文件没有自动清理。**即使把 checkpoint_every 调大，只要验证 loss 持续下降，仍会每次验证留下完整 checkpoint。**

CPU 复现：max_steps=5、validation_every=2、checkpoint_every=100，强制构造逐次改善的验证值，最终仍产生 step0、step2、step4、step5 和 best.pt。

best.pt 在支持硬链接的文件系统上通常与当前最佳编号文件共享 inode，因此一般不额外占一份空间；真正累积的是旧编号文件。无硬链接支持时 fallback 会复制。

修正：将验证频率、best 更新、滚动恢复点和历史快照保留分开配置。直接原子覆盖命名 best/latest，或验证保存成功后只清理本次策略管理的旧快照。不要用简单调大 checkpoint_every 替代保留策略。

### 1.5 [P2] 遗留 best.pt.part 硬链接可能导致旧 checkpoint 被改写

`training/checkpoint.py:57–68` 使用固定临时文件名。若上一次中断遗留的 best.pt.part 是旧 checkpoint 的硬链接，下一次 os.link 因文件已存在失败，代码进入复制 fallback，并以 `wb` 打开已有临时文件；这会截断、改写其关联的旧 inode。

标准库文件系统复现：旧 checkpoint、best.pt、best.pt.part 指向同一个 inode；更新新 best 后，旧 checkpoint 内容从 `OLD_CHECKPOINT` 变为 `NEW_CHECKPOINT`。因此异常中断后的更新并非始终保持旧文件安全。

修正：使用唯一临时文件，确保临时文件是新 inode；把“硬链接不支持”与“临时路径已存在”等异常分别处理。未来滚动保存策略应一起修正。

### 1.6 [P1，评测边界] 同分检索使用过于乐观的排名

`evaluation/retrieval.py:151` 使用 `count(scores > best_positive) + 1`。所有与正例同分的候选都不会降低正例排名。

CPU 复现：三组完全相同的图像/文本向量，一对一正例，双向 R@1 均为 1.0。在无区分能力的全同分情况下，这不代表实际检索成功。当前没有证据证明历史 M4 有大量同分，因此不能直接撤销历史数值。

修正：声明并实现一致的 tie 规则，例如稳定排序并按固定候选顺序打破同分，或同分情况下的期望指标；记录同分率。对基线与候选同时使用修正后的评测版本。若未来将 mean recall 用于 best 选择，先处理此问题。

## 2. 已确认的条件性限制

- **FP16 GradScaler 恢复路径：** `trainer.py:663–670` unscale 后先执行 finite-grad 检查，inf 会直接 raise，无法进入 scaler 的跳过更新和降低 scale 路径。CPU half 反向可以复现有限 loss、缩放后 inf 梯度。当前 BF16 配置不启用 scaler，不受这一溢出恢复限制影响。若支持 FP16，需要区分可恢复的缩放溢出，并保证 scheduler/global_step 只在真实更新后推进。
- **冻结 logit_scale 仍会被 clamp：** `trainer.py:674–675` 无条件原地限制参数。冻结参数为 5 时会变为 log(100)；若原值已在允许区间，不发生实际改变。应仅限制可训练 scale，或在初始化阶段定义统一的固定温度协议。
- **persistent validation loader 的 CPU RNG：** loader 未设置独立 generator（233–247 行）。原 run 已建立的 persistent iterator 重置与恢复后首次新建 iterator 对全局 CPU RNG 的消费不同。CPU随机层/随机增强可能无法精确恢复。此问题不能直接推断当前 CUDA head 随机深度发生偏移：CUDA randperm 使用 CUDA RNG，当前关闭随机增强。建议独立验证 generator，并单独核验数据增强恢复策略。
- **LR 日志口径：** scheduler 在 optimizer 更新后推进，日志写的是下次更新所用 LR；warmup 初始两个更新使用相同起始比例。这不属于当前主要结果失效的证据，但应明确记录“本步使用 LR”与“下步 LR”。

## 3. 通过检查的核心逻辑

- 对比损失的 paired targets 与双向 CE 对应关系正确；queue 是 detached 历史负例。
- 每个 microbatch 的 loss 除以累积次数，凑满后才更新 optimizer；CPU验证等于各 microbatch 目标的平均梯度。batch16、累积4不会扩大 InfoNCE 到64候选，当前 queue=0 时每个目标仍为16候选。
- 参数分组有非重叠与完整覆盖检查，视觉 backbone 不进入允许训练范围。
- 冻结 backbone 同时保持 eval，并在反向后检查没有梯度。
- sampler 在消费后推进，保存 shuffle generator、order 和 offset；CPU恢复后的剩余索引一致。
- 验证使用 eval/inference_mode，并恢复训练模式；forward_batch_size=64 仍拆为原定16候选 loss，相关分组一致性测试通过。
- checkpoint 只保存可训练模型参数及恢复状态，并没有每次重复保存整个冻结大模型。

以上没有证明真实 CUDA 大模型的所有路径无 bug。本轮未重跑真实训练、GPU AMP或数据增强恢复验证。

## 4. 验证与 best 的含义

当前 best 依据 `validation loss`（trainer.py:721）选择。每次验证返回 loss、样本数、logit_scale、耗时等，没有完整候选池的双向 Recall。完整检索主要由外部脚本对保存的阶段权重运行。

因此“保留日志、不保留中间权重”可记录现有 loss 曲线，但不能在事后重新获得被删除 checkpoint 的检索指标。若希望每个验证点记录检索，需要在权重仍在内存中时完成检索并写日志；可复用验证前向输出以减少重复编码。

下一轮在开跑前明确 best 的选择指标：可以继续采用验证 loss，也可以采用同源验证集 mean recall，但不能将 best_loss 与 best_retrieval 混用。外部/测试集不用于调 checkpoint。初始化也保留为可选 best 候选，避免将退化最少的训练点误称为改善。

## 5. 建议的存储策略

### 两文件最简策略

- `step_0000000.pt`：初始化权重，固定保留。
- `best.pt`：按预先声明的验证指标原子覆盖。
- 每次验证记录 step、loss、双向检索指标（新增）、best step、LR与训练诊断；不创建长期保留的每步编号快照。

它支持“初始化与最佳状态”复测。best 不等于最后一步；最终状态如不同于 best，只能从日志查看已计算的指标，不能事后加载复测。当前 best payload 已包含 optimizer/scheduler/RNG 等，理论上可从它恢复，但可能退回较早进度，且需要先处理第1.3节的日志回滚问题。

### 三文件滚动策略

另增加 `latest.pt`，按恢复间隔覆盖，保留完整 optimizer/scheduler/scaler/sampler/RNG 状态；运行正常完成后，它代表最终状态。文件数量固定，恢复时只损失 latest 之后的进度。若用户只需要初始化与best，latest 的最终保留可配置。

step0 无须携带训练后才存在的 Adam moments。用于评测的 best 也可以只保留模型参数与必要元数据，但这样会失去从 best 精确恢复的能力，并需要修改当前只接受format_version=2完整checkpoint的加载逻辑。第一版优先保持现有可恢复格式，靠固定数量节省空间。

**量级：** 当前 adapter 精确参数量1,054,976。CPU实测其模型状态+Adam状态约12.081 MiB（不含完整sampler/provenance等），step0模型参数约4.024 MiB。约25.33M参数的全量视觉 head，按FP32权重与两个Adam moment估计约289.8 MiB/训练后checkpoint。日志和元数据、队列容量、序列化实现会影响实际文件大小；这不是服务器产物实测。

3 epoch=1710 step、每50 step保存，若持续改进并补终点，约有35份训练后快照；单独head状态量级约9.9 GiB，加初始化。改为初始化+best，或初始化+best+latest，模型状态的占用保持在固定数量级。硬链接的 best 本身不要重复计费；备份或复制到不保留硬链接的介质时可能重新占一份。

## 6. 建议验证频率与迁移顺序

下一轮长训练建议：训练日志每10 step，完整验证每200 step，并强制step0与正常训练终点验证。200 step约0.35 epoch，3epoch日程仍有约9个训练后验证点，可观察学习曲线。200是起始选择，不保证所有短暂退化都被捕获；数值异常与缺失梯度继续每次更新检查。

减少验证主要节省前向评测时间；固定文件数量主要节省存储，两者应独立配置。验证 loss 的候选 batch size保持16，不随验证频率改变。

实现顺序：

1. 修正seed时机、终点验证、同分排名，并保持历史结果说明。
2. 定义best指标、记录last_validation_step与可选检索日志。
3. 加入滚动保存策略、可靠的原子写入、恢复日志回滚规则。
4. 更新加载器、summary与验证工具。旧 `tools/verify_training_run.py:403` 强制要求best的编号源文件；旧阶段脚本依赖step100/250/500文件，不能直接沿用“两文件”协议。
5. 用小型CPU训练验证step0/best/latest、无改善、末步非整除、恢复回滚和写入中断；再做真实模型短功能检查，最后进入充分训练。

本轮没有删除任何历史产物。下一轮应使用新的配置/输出协议，旧阶段脚本与证据保留其原有解释范围。
