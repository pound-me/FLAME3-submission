# 夜间作业晨报（2026-08-24）

## 作业状态

| Job | 状态 | 事实证据 |
|---|---|---|
| Job 0 | 完成，允许继续 | `case2_test107_consistency_low_with_no_large_geometric_offset`；baseline aggregate IoU `0.234093`；最佳平移 `dy=+2, dx=+0`，增益 `0.006533`。 |
| Job 1 | 完成 | 图像层 bootstrap 10,000 次；三 seed 配对统计已生成；C-A 与 D-C 缺少逐图 C 臂指标，image-level CI 标记 unavailable。 |
| Job 2 | 完成，有一个缺失种子 | Fusion/RGB-only/Thermal-only 完整种子数为 3/3/2；Thermal seed 201 原始运行与同协议 retry1 均出现 non-finite gradient，未替代。 |
| Job 3 | 自检未通过而停止 | `Ignore pixels changed full training loss: 2.200896978378296 != 2.222754716873169`；`stop_job4=true`。 |
| Job 4 | 未启动 | Job 3 未写出 `job4_allowed=true`，按硬门槛跳过。 |
| Job 5 | 完成 | 已从冻结结果生成 CSV 表和 SVG 图；无训练、无推理、未读取 test107 图像、掩膜或预测。 |

## 关键数字

- Job 1：v1.1-baseline 的 val47 Fire/Heat 图像差 `+0.013151`，95% CI `[+0.004137, +0.022787]`；test107 对应差 `-0.006879`，95% CI `[-0.010903, -0.002760]`。
- Job 2：Fusion/RGB-only/Thermal-only 的 mean S 分别为 `0.722277` / `0.383231` / `0.638501`。
- Job 3：Ignore 断言中的两次全损失为 `2.2008969784` 与 `2.2227547169`；失败发生于正式比较网络训练前。

## 异常与停止原因

- RGB-only seed 200 原始产物仅到 epoch 27；保留原目录，汇总使用完整 retry1。
- Thermal-only seed 201 在原始运行和同协议 retry1 中均出现 non-finite gradient；汇总按缺失处理。
- Job 3 的 Ignore 排除门未通过；完整堆栈见 `staging/remote_night_evidence_20260824/job3/TRACEBACK.txt`。Job 4 因此未启动。

## 待人工决定事项

- Job 3 Ignore 排除失败的处置方式，以及修复后是否重新执行工程门。
- Thermal-only seed 201 是否需要单独调查数值失败，或保留当前缺失种子口径。
- 对本次新生成表格与矢量图的人工版式和引用文字复核。
