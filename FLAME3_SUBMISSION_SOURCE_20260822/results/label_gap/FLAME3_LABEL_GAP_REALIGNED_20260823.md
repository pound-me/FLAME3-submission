# FLAME3 标签鸿沟对齐重算与勘误依据

日期：2026-08-23

本工单是只读的坐标口径核验、标签一致性重算与方法学敏感性审计。未训练、未推理、未读取任何模型预测，所有对齐仅在内存中完成，未生成或改写任何掩膜。本报告取代原标签鸿沟报告中 `A001–A101逐图IoU均值0.309806/中位数0.107363`、`人工新增低于80°C占93.7%`、以及“该开发子集证明温度伪标签存在实质标签鸿沟并主要解释test端点差0.39”的结论；原报告保留不改，仅由并置勘误约束使用。

工单 SHA256：`a754d48fe542ab3308cf2c12acee105dda20d6f72b7580c35c3cfda33801fad5`。

## 冻结判定

A001–A101 在修正口径下逐图 IoU 中位数为 `0.906013`，达到预先冻结的 `>=0.80` 条件。因此，原“实质标签鸿沟”贡献不成立，改为几何对齐方法学告诫；此判定没有在看到结果后调整阈值。

test107 的决定性检验独立归入第二种情况：人工与机器 Fire/Heat 一致性显著低，但没有大尺度错位。它支持“test107 使用更宽 Fire/Heat 定义”，适用范围仅限 test107，不得推广回 A001–A101。

## 任务一：对齐与 test107

| 批次 | 坐标状态 | 偏移估计 | 置信度 | 证据 |
|---|---|---:|---|---|
| A001–A047 Smoke/Ignore | 修正后 | 0 px | 高 | 150/150 当前掩膜哈希匹配修复 manifest |
| A001–A101 找回 Fire 备份 | 修正前 | 约 +44 px 峰值；冻结修复 +46 px | 高 | +46 px 内存差分修复与候选 101/101 一致 |
| val47 三分类冻结掩膜 | 修正后 | 0 px | 高 | 内存合并与冻结掩膜 47/47 一致 |
| A048–A150 Smoke/Ignore | 修正后 | 0 px | 高 | 机器 Fire 固定，Smoke/Ignore 已下移修正 |
| test107 人工三分类 | 修正后 | 最佳 dy=+2px, dx=+0px | 中 | aggregate 仅 `0.234093`→`0.240626` |

### test107 完整判定

- baseline aggregate / 逐图均值 / 逐图中位数：`0.234093 / 0.188505 / 0.177662`；仅非空图中位数 `0.233964`。
- 人工 / 机器 / 交集 Fire 像素：`459,557 / 205,329 / 126,121`；人工独有 `333,436`，机器独有有效 `79,208`。
- 最佳平移为 `dy=+2, dx=+0`，aggregate 增益仅 `0.006533`；边界均距也仅约 `15.1918→15.1795 px`。不满足第三种情况。
- 人工独有像素中低于80°C为 `207,137/333,436=62.122%`；机器独有有效像素中不低于80°C为 `78,259/79,208=98.802%`。test107 的差异是标签定义差异，不是 46 px 事故。

## 任务二：三个范围的重算摘要

| 范围 | Aggregate IoU | 逐图均值 | 逐图中位数 | 人工独有有效 | 其中<80°C | 比例 | 机器独有有效 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A001-A047 | 0.865510 | 0.842101 | 0.883398 | 49,260 | 49,045 | 99.564% | 1,204 |
| A048-A101 | 0.902244 | 0.692078 | 0.941475 | 12,828 | 12,768 | 99.532% | 556 |
| A001-A101 | 0.875330 | 0.761891 | 0.906013 | 62,088 | 61,813 | 99.557% | 1,760 |

原 `93.7%` 在统一对齐、排除 Ignore 的口径下变为 `99.557%`（`61,813/62,088`）。完整逐图值、分位数、四格表和温度分布分别见 `task2/PER_IMAGE_ALIGNED_FIRE_IOU.csv`、`task2/RANGE_SUMMARY.json`、`task2/TEMPERATURE_BIN_CONFUSION_ALIGNED.csv` 和 `task2/FIRE_TEMPERATURE_DISTRIBUTION_ALIGNED.csv`。

### A001–A101 温度分箱

| 温度 | TP | FP | FN | TN | Precision | Recall | 一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| <50 | 0 | 0 | 50,867 | 30,279,815 | — | 0.000000 | 0.998323 |
| 50-80 | 2,064 | 9 | 10,946 | 184,461 | 0.995658 | 0.158647 | 0.944526 |
| 80-100 | 20,711 | 122 | 231 | 6,281 | 0.994144 | 0.988970 | 0.987091 |
| 100-150 | 103,775 | 457 | 44 | 1,375 | 0.995616 | 0.999576 | 0.995258 |
| 150-200 | 132,853 | 424 | 0 | 32 | 0.996819 | 1.000000 | 0.996819 |
| 200-300 | 119,146 | 493 | 0 | 0 | 0.995879 | 1.000000 | 0.995879 |
| >=300 | 69,741 | 255 | 0 | 0 | 0.996357 | 1.000000 | 0.996357 |

对齐后，`100–150°C` 仍是 FP 首次超过 FN 的错误方向转折箱；但 `80–100°C` 已成为 precision 与 recall 同时达到 0.95 的高一致性起点（A001–A101 为 0.994144/0.988970），不再是旧报告中 recall 仅约 47.4% 的明显漏标箱。机器标签仍高度集中于 >=80°C；人工新增总量由旧口径的 465,736 大幅降为 62,088，但剩余新增几乎全部低于80°C。这支持一个规模有限、方向明确的低温边界差异，不支持把 A001–A101 描述为整体标签体系鸿沟。三个范围的转折定义与结果见 `task2/TEMPERATURE_TRANSITION_SUMMARY.csv`。

### Fire 温度分布

| 范围 | 标签 | 像素 | 中位数°C | Q1°C | Q3°C | >=80°C |
|---|---|---:|---:|---:|---:|---:|
| A001-A047 | machine | 325,966 | 186.331 | 143.530 | 254.249 | 0.995561 |
| A001-A047 | manual | 374,022 | 180.450 | 119.859 | 239.574 | 0.865027 |
| A048-A101 | machine | 124,084 | 186.331 | 142.182 | 248.721 | 0.994955 |
| A048-A101 | manual | 136,356 | 180.750 | 127.318 | 239.574 | 0.901772 |
| A001-A101 | machine | 450,050 | 186.331 | 143.119 | 252.692 | 0.995394 |
| A001-A101 | manual | 510,378 | 180.450 | 122.193 | 239.574 | 0.874844 |

### 分歧分类与案例边界

A001–A101 有效人工独有 `62,088`，有效机器独有 `1,760`；另有 `0` 个机器 Fire 位于人工 Ignore，必须与有效 FP 分开。

| 请求分类 | 证据级别 | 代理图像数 | 方向像素 | 可否作像素原因占比 |
|---|---|---:|---:|---|
| machine_overlabel_on_valid_manual_nonfire | direct_exclusive_pixel_count | 28 | 1,760 | True |
| machine_fire_on_manual_ignore | direct_exclusive_ignore_decomposition | 0 | 0 | True |
| hot_ground_or_residual_heat_combined | image_level_multi_label_proxy | 57 | 905 | False |
| residual_heat_but_manual_nonburning_explicit | image_level_proxy_explicit_residual_field | 46 | 556 | False |
| hot_ground_separate_from_residual_heat | unavailable_in_frozen_fields | 0 | — | False |
| reflection_or_highlight | unavailable_in_frozen_fields | 0 | — | False |
| unrelated_heat_source | unavailable_in_frozen_fields | 0 | — | False |

热地面/余热只能依据冻结图像级字段给出重叠代理；旧字段将二者合并，因此不能强行拆成互斥像素原因。反光和无关热源没有独立冻结字段。机器低于80°C有效多标仅出现在1张图，Ignore内机器Fire为0；这些类别无法满足3–5例时保留真实可用例数，不重复或虚构案例。完整分类、限制与代表ID见 `task2/DISAGREEMENT_REASON_AVAILABILITY_ALIGNED.csv`，代表图见 `task2/representative_cases/`。

## 任务三：错位敏感性

在 A001–A047 上，保持修正后 Ignore 不变、仅把人工 Fire 编辑差分从 `+46` 向上撤回到 `0` 时，aggregate 从 `0.865510` 降至 `0.403535`。既有固定顺序审计的历史锚点为 `0.314334`；它还使用事故时的旧 Ignore 几何，因此是“Fire 错位 + 历史 Ignore 口径”的复合端点，不能伪装成纯刚性位移曲线值。实际事故方向为向上撤回 46 px。

完整人工 Fire 的刚性位移曲线更陡：实际方向 8 px 已降至 `0.456748`，16 px 为 `0.253157`；正交方向 23 px 为 `0.324191`，已接近历史 `0.314334`。这说明把 0.87 显示成约 0.31 所需的位移取决于错误机制：全掩膜刚移约十几到二十几像素即可，真实事故的编辑差分错位为 46 px。

对完整人工 Fire 做刚性位移时，实际方向 1/2/4 px 的 aggregate 变化分别为 `-0.063535`、`-0.121770`、`-0.225790`。完整实际方向、正交方向与历史机制重放见 `task3/MISALIGNMENT_SENSITIVITY_A001_A047.csv` 和曲线图。

工单给定的模型侧 test107 post-hoc `thermal_shift` 变化为 1/2/4 px：`-0.0006 / -0.003 / -0.013`。二者对象不同，不作数值等同；可发表观察是：标签一致性测量会被几何坐标错误系统性压低，且 46 px 标注事故的影响远大于典型 1–4 px 模态配准扰动。

## 影响范围

| 产物 | 使用口径 | 是否受 46 px 事故影响 | 处置 |
|---|---|---|---|
| FLAME3_LABEL_GAP_QUANTIFICATION_20260822 | A001-A101 recovered Fire in pre-correction coordinates | yes | IoU, temperature/disagreement values and broad label-gap claim superseded by erratum |
| FLAME3_PAPER_READY_ABLATION_AND_LABEL_GAP_20260822 | inherits old A001-A101 label-gap analysis | yes | label-gap and test-gap attribution paragraphs superseded; ablation metrics unchanged |
| val47 frozen audit | aligned frozen A001-A047 three-class masks | no | retain; exact 47/47 in-memory reproduction |
| 2x2 attribution | uses aligned val47 endpoint and machine endpoint | no direct metric change | retain model metrics; remove reliance on old label-gap narrative |
| V1/V2 read-only validation | frozen evaluation artifacts; no recovered pre-alignment Fire consumed | no | retain |
| test107 frozen machine-label evaluation | test107 machine masks | no | retain |
| test107 frozen manual-label evaluation | test107 corrected-coordinate frozen manual masks | no | retain; interpretation narrowed to test107 label definition |
| FLAME2 zero-shot | FLAME2 endpoint; no A001-A101 recovered Fire | no | retain |
| 20260823 endpoint audit | explicitly separates old and aligned A001-A101 scopes | no | retain; this work order completes its requested correction |
| training runs | machine Fire plus corrected-coordinate manual Smoke/Ignore | no | confirmed no training consumed misaligned manual Fire |

没有训练使用未对齐人工 Fire。密集监督训练的 Fire 来自机器标签，人工 Smoke/Ignore 使用修正后坐标；本工单未训练、未推理。A001–A101 找回人工 Fire 在另行预注册并生成正式对齐修正掩膜前，禁止用于任何训练、评估或标签构建，包括此前排队的 v1.2。

## 文件与可追溯性

全部输入 SHA256 见 `audit/INPUT_SHA256.csv`；最终同步后的全部输出 SHA256 见 `audit/FINAL_OUTPUT_SHA256.csv`（该清单自排除），包内清单见 `SHA256_MANIFEST.txt`。原报告前后哈希见 `audit/ORIGINAL_REPORT_READONLY_GUARD.csv`，四份原报告均保持字节不变。
