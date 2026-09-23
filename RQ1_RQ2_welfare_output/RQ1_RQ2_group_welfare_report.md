# RQ1–RQ2 群体收益分析

## 口径

- 原始群体收益：每次独立运行中，所有参与者最终 payoff 的总和；主表换算为“人均每个 population round 的最终收益”。
- 机制成本只扣一次：优先读取 `final_payoffs` 或日志中已经调整后的角色 payoff。
- 标准化净福利：相对于各博弈无合作起点、除以完全合作可创造的基础社会剩余。该指标允许惩罚导致负值，也允许奖励创造额外收益而超过 100%。
- 信任博弈奖励条件包含第三方奖励者及额外 10 单位禀赋；因此跨条件判断优先看资源调整后的标准化净福利，而不是只看原始最终 payoff。
- 差值统一定义为 `RQ2 − RQ1`；正值表示机制组更高。每个模型等权。

## 附件数值与原始文件核验

共分析 93 个 RQ1/RQ2 单元格；按附件均值与 SD 四舍五入到两位核验后，92 个由原始日志精确匹配，1 个没有精确匹配日志而采用博弈恒等式重建，0 个仍不匹配。

没有精确匹配原始日志的项目：

| RQ | 博弈 | 模型 | 处理方式 | 搜索结果 |
|---|---|---|---|---|
| RQ1 | PD | Grok-4.3 | 博弈收益恒等式重建 | Current attachment reports 0.00% +/- 0.00%; available RQ1_result/result/prisoners_dilemma/072701_grok-4.3 logs give 0.69% +/- 0.24%. Exact DD payoff reconstructed from the current attachment value. |

## 主要发现

- 囚徒困境：三种机制的跨模型平均标准化净福利差均非负；惩罚、声誉、奖励依次为 +12.470、+25.734、+37.986 个百分点，奖励的平均改善最大。
- 公共物品：声誉的平均改善最大（+45.278 个百分点），其次为惩罚（+22.777）和奖励（+15.532）。但惩罚不是稳健改善：Grok-4.3 虽达到 24.65% 贡献率，标准化净福利却下降 107.532 个百分点，人均每轮最终收益下降 184.956。
- 信任博弈：三个机制均纳入 7 个模型；惩罚、声誉、奖励的平均标准化净福利差分别为 47.912、59.157、57.914 个百分点。
- 信任博弈奖励增加了第三方奖励者和额外预算。扣除这部分初始资源后，奖励的人均社会剩余平均增量为 +2.882，低于惩罚的 +4.791 和声誉的 +5.916。

## RQ2 − RQ1：逐模型标准化净福利差值

单位为完全合作基础社会剩余的百分点。这个指标是跨机制比较的主要结论口径。

| 博弈 | 机制 | 模型 | RQ1 | RQ2 | 差值 | 方向 |
|---|---|---|---:|---:|---:|---|
| PD | punishment | GLM-5.1 | 0.000 | 1.875 | 1.875 | 提高 |
| PD | punishment | GPT-5.4 | 0.000 | 0.000 | 0.000 | 不变 |
| PD | punishment | Gemini-3.1-Pro-Preview | 4.792 | 12.917 | 8.125 | 提高 |
| PD | punishment | Claude-Opus-4-6 | 100.000 | 100.000 | 0.000 | 不变 |
| PD | punishment | Kimi-K2.6 | 0.000 | 1.389 | 1.389 | 提高 |
| PD | punishment | Grok-4.3 | 0.000 | 43.472 | 43.472 | 提高 |
| PD | punishment | DeepSeek-V4-Pro | 0.208 | 32.639 | 32.431 | 提高 |
| PD | reputation | GLM-5.1 | 0.000 | 8.681 | 8.681 | 提高 |
| PD | reputation | GPT-5.4 | 0.000 | 0.208 | 0.208 | 提高 |
| PD | reputation | Gemini-3.1-Pro-Preview | 4.792 | 100.000 | 95.208 | 提高 |
| PD | reputation | Claude-Opus-4-6 | 100.000 | 100.000 | 0.000 | 不变 |
| PD | reputation | Kimi-K2.6 | 0.000 | 1.944 | 1.944 | 提高 |
| PD | reputation | Grok-4.3 | 0.000 | 0.000 | 0.000 | 不变 |
| PD | reputation | DeepSeek-V4-Pro | 0.208 | 74.306 | 74.097 | 提高 |
| PD | reward | GLM-5.1 | 0.000 | 11.806 | 11.806 | 提高 |
| PD | reward | GPT-5.4 | 0.000 | 0.000 | 0.000 | 不变 |
| PD | reward | Gemini-3.1-Pro-Preview | 4.792 | 93.194 | 88.403 | 提高 |
| PD | reward | Claude-Opus-4-6 | 100.000 | 100.000 | 0.000 | 不变 |
| PD | reward | Kimi-K2.6 | 0.000 | 82.014 | 82.014 | 提高 |
| PD | reward | Grok-4.3 | 0.000 | 5.486 | 5.486 | 提高 |
| PD | reward | DeepSeek-V4-Pro | 0.208 | 78.403 | 78.194 | 提高 |
| PGG | punishment | GLM-5.1 | 0.000 | 93.622 | 93.622 | 提高 |
| PGG | punishment | GPT-5.4 | 0.000 | 0.000 | 0.000 | 不变 |
| PGG | punishment | Gemini-3.1-Pro-Preview | 0.000 | 99.780 | 99.780 | 提高 |
| PGG | punishment | Claude-Opus-4-6 | 0.000 | 75.000 | 75.000 | 提高 |
| PGG | punishment | Kimi-K2.6 | 0.000 | 0.139 | 0.139 | 提高 |
| PGG | punishment | Grok-4.3 | 0.000 | -107.532 | -107.532 | 降低 |
| PGG | punishment | DeepSeek-V4-Pro | 0.000 | -1.568 | -1.568 | 降低 |
| PGG | reputation | GLM-5.1 | 0.000 | 2.100 | 2.100 | 提高 |
| PGG | reputation | GPT-5.4 | 0.000 | 98.207 | 98.207 | 提高 |
| PGG | reputation | Gemini-3.1-Pro-Preview | 0.000 | 100.856 | 100.856 | 提高 |
| PGG | reputation | Claude-Opus-4-6 | 0.000 | 51.163 | 51.163 | 提高 |
| PGG | reputation | Kimi-K2.6 | 0.000 | 0.776 | 0.776 | 提高 |
| PGG | reputation | Grok-4.3 | 0.000 | 0.000 | 0.000 | 不变 |
| PGG | reputation | DeepSeek-V4-Pro | 0.000 | 63.846 | 63.846 | 提高 |
| PGG | reward | GLM-5.1 | 0.000 | 12.706 | 12.706 | 提高 |
| PGG | reward | GPT-5.4 | 0.000 | 2.318 | 2.318 | 提高 |
| PGG | reward | Gemini-3.1-Pro-Preview | 0.000 | 3.632 | 3.632 | 提高 |
| PGG | reward | Claude-Opus-4-6 | 0.000 | 18.031 | 18.031 | 提高 |
| PGG | reward | Kimi-K2.6 | 0.000 | 0.493 | 0.493 | 提高 |
| PGG | reward | Grok-4.3 | 0.000 | 0.000 | 0.000 | 不变 |
| PGG | reward | DeepSeek-V4-Pro | 0.000 | 71.544 | 71.544 | 提高 |
| TG | punishment | GLM-5.1 | 30.213 | 48.167 | 17.954 | 提高 |
| TG | punishment | GPT-5.4 | 10.602 | 73.671 | 63.069 | 提高 |
| TG | punishment | Gemini-3.1-Pro-Preview | 61.065 | 99.986 | 38.921 | 提高 |
| TG | punishment | Claude-Opus-4-6 | 32.287 | 87.181 | 54.894 | 提高 |
| TG | punishment | Kimi-K2.6 | 23.454 | 78.060 | 54.606 | 提高 |
| TG | punishment | Grok-4.3 | 7.407 | 74.532 | 67.125 | 提高 |
| TG | punishment | DeepSeek-V4-Pro | 40.519 | 79.333 | 38.815 | 提高 |
| TG | reputation | GLM-5.1 | 30.213 | 94.065 | 63.852 | 提高 |
| TG | reputation | GPT-5.4 | 10.602 | 86.954 | 76.352 | 提高 |
| TG | reputation | Gemini-3.1-Pro-Preview | 61.065 | 94.907 | 33.843 | 提高 |
| TG | reputation | Claude-Opus-4-6 | 32.287 | 92.148 | 59.861 | 提高 |
| TG | reputation | Kimi-K2.6 | 23.454 | 91.463 | 68.009 | 提高 |
| TG | reputation | Grok-4.3 | 7.407 | 68.704 | 61.296 | 提高 |
| TG | reputation | DeepSeek-V4-Pro | 40.519 | 91.407 | 50.889 | 提高 |
| TG | reward | GLM-5.1 | 30.213 | 88.333 | 58.120 | 提高 |
| TG | reward | GPT-5.4 | 10.602 | 28.944 | 18.343 | 提高 |
| TG | reward | Gemini-3.1-Pro-Preview | 61.065 | 146.472 | 85.407 | 提高 |
| TG | reward | Claude-Opus-4-6 | 32.287 | 108.153 | 75.866 | 提高 |
| TG | reward | Kimi-K2.6 | 23.454 | 69.875 | 46.421 | 提高 |
| TG | reward | Grok-4.3 | 7.407 | 58.847 | 51.440 | 提高 |
| TG | reward | DeepSeek-V4-Pro | 40.519 | 110.319 | 69.801 | 提高 |

## 跨模型描述性汇总

| 博弈 | 机制 | 模型数 | 平均人均最终收益差 | 平均资源调整后人均剩余差 | 平均标准化净福利差 | 中位数差 | 正/零/负 |
|---|---|---:|---:|---:|---:|---:|---:|
| PD | punishment | 7 | 0.249 | 0.249 | 12.470 | 1.875 | 5/2/0 |
| PD | reputation | 7 | 0.515 | 0.515 | 25.734 | 1.944 | 5/2/0 |
| PD | reward | 7 | 0.760 | 0.760 | 37.986 | 11.806 | 5/2/0 |
| PGG | punishment | 7 | 39.177 | 39.177 | 22.777 | 0.139 | 4/1/2 |
| PGG | reputation | 7 | 77.879 | 77.879 | 45.278 | 51.163 | 6/1/0 |
| PGG | reward | 7 | 26.715 | 26.715 | 15.532 | 3.632 | 6/1/0 |
| TG | punishment | 7 | 4.791 | 4.791 | 47.912 | 54.606 | 7/0/0 |
| TG | reputation | 7 | 5.916 | 5.916 | 59.157 | 61.296 | 7/0/0 |
| TG | reward | 7 | 4.549 | 2.882 | 57.914 | 58.120 | 7/0/0 |

## 解释限制

- 最新 RQ1 附件不再包含 Qwen3。Grok-4.3 囚徒困境在附件中为 0.00% ± 0.00%，但现有原始日志为 0.69% ± 0.24%；本次按用户指定的最新附件零合作值，用标准 DD 收益恒等式重建，并在 CSV 中明确标记。
- 老模型的公共物品实验使用的乘数可能与新模型不同；逐模型 RQ2−RQ1 比较使用各自对应模型的日志，跨模型原始 payoff 不应直接横比。
- 本报告核验保存结果与附件数值的一致性，但不自动剔除包含 API 调用失败或默认决策的日志；这类运行在用于正式推断前仍需单独做数据质量审查。
- 当前差值是描述性均值差。正式显著性检验应以独立 game-log 为样本，并根据是否共享随机种子选择配对检验或 Welch/Bootstrap 区间。

## 机器可读文件

- `source_match_and_welfare_summary.csv`：文件匹配、附件数值核验和每个单元格的收益汇总。
- `run_level_welfare.csv`：每个独立 game-log 的收益。
- `rq2_minus_rq1_welfare.csv`：逐模型、逐博弈、逐机制差值。
- `mechanism_aggregate_welfare.csv`：跨模型描述性汇总。
