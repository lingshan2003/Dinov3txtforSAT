# Web adapter-only 同预算对照审阅

审阅日期：2026-10-08。来源：用户提供的`/Users/wangyue/Downloads/web_adapter_control_seed11_reports.tar.gz`。
本地读取42个文件，核对配置、训练/验证日志、来源记录、21份检索报告及比较汇总；未登录服务器、重跑GPU或重新计算SHA256。
机器汇总保存在忽略目录`outputs/web_adapter_control_seed11_review.json`。

结论：本轮seed11中，adapter＋完整文本LoRA在SkyScript的两种验证池与RSICD mean Recall上均优于adapter-only。
这支持保留联合方案作为当前主候选；上轮“文本LoRA可能加重RSICD退化”的猜测未获得本轮对照支持。
两种适配后的RSICD MR仍低于原生，联合方案仅减小退化，不能称跨数据源全面提升。

## 1. 训练完成性与可比性

两组均完整完成5269个optimizer step、21075个microbatch，337199张训练图恰好曝光一轮，36495个caption组全部覆盖。
图片池覆盖率1.0，epoch offset归零，无resume、跳过更新或非有限loss/gradient记录。
固定seed11、物理batch16/accum4、mask_same_caption、queue0、无随机增强；不能将梯度累计解释为64候选对比损失。

两份配置仅四字段不同：experiment.name/output_dir、model.text_lora_rank/include_projection。
关闭LoRA是实质方法变化；数据、adapter结构/学习率、训练预算、调度与验证协议相同。
模型资产及train/val manifest的已有来源记录完全相同；527个日志窗口的caption-group采样诊断和曝光进度完全相同。
采样日志吻合不等同于本次直接重新核验全部图片像素或逐样本顺序。

adapter-only的optimizer仅包含6个image_adapter参数张量，共1054976参数；text_lora=null。
联合组共5013760参数，其中adapter相同、文本LoRA3958784参数，覆盖24个文本block的qkv/proj/fc1/fc2及末端projection。
记录显示视觉backbone永久冻结，adapter-only没有意外加入文本或视觉head参数组。
报告包未含权重张量，以上为配置/优化器/运行元数据核验，不是逐权重差分检查。

主组训练commit为a446e975，控制组5c74e5ca；两commit的`src/dinotxt_rs`没有差异，新增内容为控制入口、报告打包及文档/测试。
与上轮原生包逐字节比较，主组7个训练文件、9份检索及3份官方报告共19个文件完全相同，确认复用了既有主组。

验证均为step0、每200step及5269终点，共28次。
两组best均在5269，三池best/latest指标完全一致，主比较使用完整覆盖latest。
两组step0与原生官方三池指标完全一致，共6组起点检查通过。
18份checkpoint报告身份均match且无blocking/advisory mismatch；三池manifest、候选数、正例定义和tie policy一致。
比较JSON的所有增量已从单独报告重算核对。

## 2. 主结果

数值为mean Recall百分比；增量为百分点。MR是两个检索方向R@1/5/10六项平均。

| 验证指标 | 原生 | adapter-only | adapter＋LoRA | 联合−adapter-only |
| --- | ---: | ---: | ---: | ---: |
| SkyScript unique | 8.6313% | 14.5006% | 17.6860% | +3.1854pp |
| SkyScript多图按图 | 6.6031% | 13.5257% | 17.0671% | +3.5414pp |
| SkyScript多图按caption组均衡 | 7.3741% | 12.9117% | 15.8317% | +2.9201pp |
| RSICD-val | 14.4759% | 12.2578% | 13.9305% | +1.6728pp |

候选池：unique4055图/4055文，多图33118图/4055文，RSICD1094图/5470文。
多图按图与按组均衡是同一个池的不同加权方式，不能作为两个独立数据集计算证据量。
组均衡增益也明显，收益并非仅由图片数量最多的caption组推动。

| 检索方向 | adapter-only R@1/5/10 | adapter＋LoRA R@1/5/10 |
| --- | --- | --- |
| unique图→文 | 3.822 / 14.106 / 23.033 | 5.401 / 18.594 / 29.199 |
| unique文→图 | 4.612 / 16.202 / 25.228 | 5.869 / 18.594 / 28.459 |
| 多图图→文 | 3.699 / 15.714 / 25.959 | 5.659 / 21.004 / 33.891 |
| 多图文→图 | 4.661 / 12.700 / 18.422 | 5.894 / 14.797 / 21.159 |
| RSICD图→文 | 3.748 / 10.603 / 17.093 | 3.748 / 12.797 / 20.750 |
| RSICD文→图 | 3.693 / 13.729 / 24.680 | 3.729 / 14.790 / 27.770 |

同源两池所有六项Recall均提高。unique双向median rank从39/39改善至30/32，多图从31/111至21/92。
RSICD相对adapter-only六项Recall五升一平，但图→文R@1完全相同，文→图R@1仅+0.0366pp，主要增益在R@5/10。
不能将RSICD MR+1.6728pp描述为精确首位检索能力有同等程度提高。

相对原生RSICD：adapter-only MR下降2.2182pp，联合方案下降0.5454pp。
原生图→文R@1为4.6618%、文→图为5.4296%，两种适配的R@1均低于原生。
当前联合方案同时取得更高同源检索和更小外部MR损失；不支持“冻结文本就更能保留跨域能力”的简单推断。
训练只改视觉adapter同样会改变图文间几何关系，冻结原始权重并不保证最终检索能力保持。
联合更新提供更多适配自由度是可能解释，但本轮没有直接分析embedding分布或逐查询变化，不能当作已验证机制。

## 3. Loss与资源

| 项目 | adapter-only | adapter＋LoRA |
| --- | ---: | ---: |
| 可训练参数 | 1054976 | 5013760 |
| step0 unique-val loss | 1.600889 | 1.600889 |
| 终点/最佳unique-val loss | 0.688097 | 0.590825 |
| 最后一个optimizer更新训练loss | 0.501976 | 0.400583 |
| 最后9更新日志窗口平均loss | 0.431204 | 0.380279 |
| 训练日志区间累计 | 57.12分钟 | 78.99分钟 |
| CUDA峰值allocated | 3.91GiB | 5.68GiB |

联合方案验证loss相对低14.14%；本轮更低loss对应真实全局检索收益。
summary末更新loss与metrics末窗口loss是不同统计口径，不能混用或据此认定日志错误。
训练日志elapsed区间求和包含周期验证，未包括完整流程的所有起止及全局检索开销，不是严格端到端性能基准。
本次RTX3090记录中，联合组多约21.87分钟（38.3%）、峰值allocated多约1.77GiB；参数数约4.75倍不意味着实际时间/显存也同倍增加。

两组后段验证loss总体降低并有小幅波动，没有该验证loss持续反弹的证据。
adapter-only在4800/5000/5200/5269为0.689490/0.689514/0.688105/0.688097，联合为0.591606/0.592296/0.590993/0.590825。
后段变化趋缓同时伴随cosine学习率衰减至0，不能由此断定模型容量上限，也不能保证继续追加epoch必然有效。
全局检索只测了step0/best/latest，不能从这些loss点推断中途RSICD最佳checkpoint或断言更早停止一定保留更多迁移能力。

## 4. 研究结论与下一步

本轮可写：在相同SkyScript图片覆盖、监督和训练预算下，加入文本Transformer及projection的LoRA进行联合适配，相比仅训练视觉adapter，提高了同源全局检索，并减小了RSICD mean Recall下降。
该结论限定seed11及当前验证协议；不代表LoRA单独优于adapter，也不能拆分Transformer与projection的独立贡献。
联合组的adapter训练轨迹也会随文本更新而改变，不能把组间差异全部解释成文本encoder独立变强。

下一步优先固定配方，将adapter-only和adapter＋LoRA都补seed23/47，报告配对差异及各方向Recall，避免只复现赢家而无法核验消融稳定性。
若预算仅够两次新训练，可先补联合方案seed23/47核验主候选稳定性，但尚不能宣称LoRA相对优势跨seed稳定。
随后若要回答adapter是否必要，再做无adapter的文本LoRA-only或既有head＋LoRA，前者隔离adapter的加入效应，后者比较不同视觉更新模块，研究问题不同。
不优先同时扩大epoch、改batch、换caption和加更多更新模块；当前先建立可信重复结果。

尚缺独立最终test、跨seed重复与逐查询rank，不能推断统计显著性或通用能力保持。
历史M4旧RSICD原生15.8775%不用于本轮增量；当前共同原生基线为14.4759%。

运行协议见[adapter-only对照说明](WEB_ADAPTER_CONTROL_PLAN_2026-10-06.md)，主组与SAT比较见[上轮原生审阅](WEB_NATIVE_SEED11_ANALYSIS_2026-10-06.md)。
