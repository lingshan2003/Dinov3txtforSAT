# SAT adapter + 文本LoRA的negative queue审查（2026-10-04）

用户提出：正在进行三个候选的多seed复现时，能否重新引入queue_size，给当前最好的adapter + 全文本LoRA增加负例。本次只审查现有代码和提出实验方案，尚未修改训练实现、复现配置或启动queue实验。

## 当前负例与queue含义

我们一直在进行对比负例学习：物理batch16，每个查询有1个正例和15个同批负例。queue_size=0仅表示不加入历史向量，梯度累积4不把单次候选集合扩大到64。

现有实现位于`src/dinotxt_rs/losses/contrastive.py`：FIFO存储detach后的图像/文本向量，并在两个方向分别拼接到当前候选中。Trainer每个物理batch在backward后enqueue，再按4次累积进行optimizer更新；不是每个optimizer step只入队16个。

| queue_size | 队列满时每方向候选数 | 单查询名义负例数 | 队列跨度（当前batch16×累积4） |
| ---: | ---: | ---: | --- |
| 0 | 16 | 15 | 无历史向量 |
| 256 | 272 | 271 | 16个物理batch，约4次optimizer更新 |
| 1024 | 1040 | 1039 | 64个物理batch，约16次optimizer更新 |
| 4096 | 4112 | 4111 | 256个物理batch，约64次optimizer更新 |

候选数是满队列且没有屏蔽时的数量。刚开始队列为空，随后逐渐填充。Queue改变负例池，不增加独立训练数据，也不是单独的hard-negative mining算法。

## Caption去重已经发挥作用，但不是充分条件

当前unique-caption清单去掉了明确的同文多图冲突；相似语义但不同字符串仍可能成为假负例。这与adapter/LoRA的参数更新范围是不同问题，模型变强不会自动使标签关系或历史向量正确。

现有queue不保存sample ID或caption group，loss也没有known-positive mask。三个epoch中，同一个样本会再次出现；如果其上个epoch的向量仍在队列，代码会把它作为负例，即使清单caption全唯一。

为核验该问题，本地只重放实际`ResumableBatchSampler`（36,495条、batch16、shuffle=true、drop_last、3epoch），用数据索引追踪FIFO，没有加载模型、GPU或重算SHA256。统计在当前query入队之前，其相同样本是否已在历史队列：

| seed | queue256碰撞查询数 | queue1024碰撞查询数 | queue4096碰撞查询数 |
| ---: | ---: | ---: | ---: |
| 11 | 4 | 32 | 454 |
| 23 | 0 | 26 | 459 |
| 47 | 2 | 33 | 462 |

均发生于第二或第三epoch的跨轮次窗口；这里统计的是同样本碰撞，不包括未知同义/近重复关系，也不是实际损失或性能变化测量。Sample ID屏蔽可以直接处理这一确定冲突，不能靠当前seed23在queue256恰好零碰撞而省略。

## Adapter + LoRA的向量过时问题

当前最佳方案两侧都更新：image adapter与text LoRA。所以队列中的图像和文本向量都可能来自旧参数；detach意味着它们不会随最新模型重算，也不会对缓存向量反传。比文本完全冻结时多了一侧历史漂移。

本项目目前没有EMA/momentum encoder。[MoCo论文](https://arxiv.org/abs/1911.05722)使用queue和移动平均encoder构造一致的特征字典；我们的普通FIFO不应被称作MoCo。Momentum不是尝试小queue的绝对前提，但队列长度和向量年龄需要进入解释，不能假定queue越大越好。

## 建议的第一组独立对照

有理由把queue作为下一条实验轴，但不建议仅把4096写回正在运行的配置。

- 保留当前三个方法的多seed复现，先完成其原定queue0协议。
- 为adapter + 全文本LoRA单独做queue256、seed11的完整三轮对照；训练/验证清单、模块范围、LR、warmup171、1710step、batch16×累积4、增强等沿用queue0参照。
- 在启用前，queue保存稳定sample ID与规范化caption身份，loss屏蔽这些已知正例冲突；queue状态和identity同步保存/恢复。记录occupancy、optimizer年龄、被屏蔽的候选数与已有in_batch_loss。未知同义/地点关系仍需单独的数据诊断。
- 小FIFO明确作为仍有历史漂移的基线，不同时新增EMA、改batch、改caption或扩数据。是否尝试1024由该结果与诊断决定，不设置历史短预算的强制止损门槛。
- 保持验证不使用queue。训练queue-loss的候选数发生变化，不能直接同旧train loss比高低；应使用相同定义的validation loss及完整SkyScript/RSICD双向Recall，特别检查RSICD图→文R@1。

该首组建议从同一官方权重重新开始，与已有queue0三轮seed11比较。若改为加载现有best后开启queue继续训练，就是另一个“分阶段训练”实验，需要明确warm-start、重设optimizer/scheduler，并有同样额外预算的queue0续训对照。现有严格resume不能通过修改queue_size冒充同一次恢复；已经到终点的余弦学习率也不能直接用作新增训练日程。

这里的工程补充是为使queue对照符合其标签含义，不是把历史AI制定的queue禁令当作不可突破的规则。当前队列能够机械运行，但还不具备上述identity屏蔽和年龄诊断，尚未宣称第一组queue实验已可直接启动。
