# FLAME3 夜间 Job 5：论文用冻结素材

本作业只从已保存的 CSV/JSON 结果生成表格与矢量图，不读取图像、掩膜、checkpoint 或模型预测，不训练、不推理。Job 1 的配对差与可用 CI 已并入 2×2 归因表。

已生成四个工单列明端点（val47、val134、FLAME2、test107）的 Fire/Heat 对照、FLAME2 零样本表、Job 2 模态消融表、test107 既有事后描述性模态失效与鲁棒性曲线，以及效率表。Job 3 因 Ignore 排除自检失败；Job 4 未启动，因此效率表按工单只含 v1.1 实测值。

所有图均为 SVG，源 CSV 位于 `tables/`；审计记录位于 `audit/`。
