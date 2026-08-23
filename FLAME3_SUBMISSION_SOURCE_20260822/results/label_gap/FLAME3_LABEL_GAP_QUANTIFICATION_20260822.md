# FLAME3机器标签—人工Fire/Heat标签鸿沟量化

日期：2026-08-22

## 结论先行

- A001–A101中，机器Fire与恢复的人工Fire/Heat逐图IoU均值为`0.309806`，中位数仅`0.107363`，说明两套标签并非同一语义口径。
- 人工新增（机器漏标）`465,736`像素，其中`436,208`像素低于80°C，占`93.7%`。
- 冻结二值Fire比较中的机器only为`178,743`像素，其中`177,871`像素不低于80°C；但其中`87,721`像素落在人工255 Ignore上，不能作为有效FP。排除Ignore后，人工有效非Fire上的机器only为`91,022`像素，其中`90,500`像素不低于80°C。
- 机器Fire像素不低于80°C的逐像素比例实测为`0.995394`（`99.539%`），不是此前口头值约`0.954`；人工Fire/Heat对应比例为`0.406535`（`40.654%`）。
- 因此，机器标签更接近“温度阈值产生的高温核心”，人工标签更接近“人工判定的Fire/Heat语义区域”；两者差异同时包含低温/烟下火区的人工扩展、高温有效非Fire上的过覆盖，以及大量被人工明确置为Ignore的不可判热足迹。

## 数据范围与坐标口径

- 仅分析A001–A101，覆盖47张冻结validation开发图和54张train人工审查图。
- 主口径严格使用`per_image_fire_label_comparison.csv`及与其逐像素计数完全一致的`source_pre_alignment_masks_A001_A101`。
- 46px显示对齐修复产生的`provisional_aligned_fire_candidates_A001_A101`仍要求人工复核，本报告没有悄悄替换为该候选。因此五列联系图中若见人工Fire位置偏移，它反映的是被审计的原坐标口径，而不是本次渲染再引入的位移。
- 未读取107张test图像、人工掩膜或预测，未重新前向test；仅引用用户已给出的冻结汇总端点。未训练模型，未修改checkpoint或冻结标签。

## 核心像素计数

| 指标 | 像素数 |
|---|---:|
| 机器Fire | 450,050 |
| 人工Fire/Heat | 737,043 |
| 双方一致Fire | 271,307 |
| 人工新增、机器漏标 | 465,736 |
| 机器only（二值比较，含Ignore） | 178,743 |
| 其中落在人工255 Ignore上的机器Fire | 87,721 |
| 排除Ignore后的有效机器FP | 91,022 |
| 全部改变Fire像素 | 644,479 |

## 温度分箱四格表（人工为参照）

| 温度(°C) | TP | FP | FN | TN | 可评估像素 | 可评估占比 | Ignore像素 | 机器Fire∩Ignore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| <50 | 0 | 0 | 378,722 | 29,953,944 | 30,332,666 | 97.345% | 1,277,270 | 0 |
| 50-80 | 1,201 | 522 | 57,486 | 266,967 | 326,176 | 1.047% | 299,803 | 350 |
| 80-100 | 12,719 | 3,964 | 14,129 | 48,455 | 79,267 | 0.254% | 126,422 | 4,150 |
| 100-150 | 64,999 | 19,945 | 11,773 | 46,348 | 143,065 | 0.459% | 134,459 | 19,288 |
| 150-200 | 67,490 | 29,260 | 3,626 | 16,260 | 116,636 | 0.374% | 70,281 | 36,527 |
| 200-300 | 78,053 | 24,972 | 0 | 0 | 103,025 | 0.331% | 16,614 | 16,614 |
| >=300 | 46,845 | 12,359 | 0 | 0 | 59,204 | 0.190% | 10,792 | 10,792 |

| 温度(°C) | Precision | Recall | 一致率 | 机器Fire | 人工Fire/Heat |
|---|---:|---:|---:|---:|---:|
| <50 | — | 0.000% | 98.751% | 0 | 378,722 |
| 50-80 | 69.704% | 2.046% | 82.216% | 2,073 | 58,687 |
| 80-100 | 76.239% | 47.374% | 77.175% | 20,833 | 26,848 |
| 100-150 | 76.520% | 84.665% | 77.830% | 104,232 | 76,772 |
| 150-200 | 69.757% | 94.901% | 71.805% | 133,277 | 71,116 |
| 200-300 | 75.761% | 100.000% | 75.761% | 119,639 | 78,053 |
| >=300 | 79.125% | 100.000% | 79.125% | 69,996 | 46,845 |

### 转折箱位

- `<50°C`与`50–80°C`是系统性漏标区：Recall分别为0与约2.0%，低温人工Fire/Heat几乎不可能被80°C规则覆盖。
- `80–100°C`仍以漏标为主（排除Ignore后FN 14,129 > FP 3,964，Recall约47.4%）。
- `100–150°C`是错误方向的转折箱：排除Ignore后机器FP首次超过人工FN（19,945 > 11,773）；从这里开始，主要矛盾由低温漏标转为高温足迹过覆盖。
- `150–200°C`排除Ignore后的Precision约69.8%；200°C以上Recall达到100%，但仍有37,331个有效FP，另有27,406个机器Fire落在人工Ignore上，说明“温度足够高”既不等于人工语义必然接受，也不应把不可判区域强记为FP。

温度分解能直接回答“分歧发生在哪个温度区间”，但不能单独把机器多标高温像素细分成余热、热地面、反光或配准泄漏；现有人工字段只支持图像级代理，不能冒充像素级原因真值。

## Fire像素温度分布摘要

| 标签口径 | 像素数 | 中位数(°C) | Q1(°C) | Q3(°C) | ≥80°C比例 |
|---|---:|---:|---:|---:|---:|
| 机器Fire | 450,050 | 186.331 | 143.119 | 252.692 | 99.539% |
| 人工Fire/Heat | 737,043 | 47.370 | 24.568 | 157.914 | 40.654% |

同轴直方图与逐图温度中位数散点分别见`temperature_fire_histogram_overlay.png`和`per_image_temperature_median_scatter.png`。

## 逐图一致性

- IoU均值：`0.309806`；中位数：`0.107363`；最小值：`0.000000`；最大值：`1.000000`。
- 最低5张：A044=0.000；A049=0.000；A051=0.000；A053=0.000；A054=0.000。
- 最高5张：A085=1.000；A087=1.000；A088=1.000；A089=1.000；A090=1.000。
- 完整最低/最高15张清单分别见`lowest_15.csv`和`highest_15.csv`。

## 图像级属性与分歧原因

- A001–A047使用旧70项错误审查生成的图像级、多标签strata；A048–A101使用semantic_class与余热字段。
- `image_level_cause_proxy_summary.csv`允许类别重叠，像素数不可相加为100%；`attribute_correlations.csv`只作描述性相关，不允许因果解释。
- 可直接确定的四种互斥分歧已经写入`disagreement_type_summary.csv`，并精确覆盖全部改变Fire像素。
- `visible_flame_yes_no_uncertain`在本批101张中有值的图像数为0，因此无法给出“低温且逐像素视觉可见火焰”的可靠像素规模。报告只给出更弱的图像级代理：活动火图像上的低温人工新增像素。

| 请求的原因/类型 | 证据级别 | 直接像素（原始或子分解） | 涉及/代理图像数 | 图像上的方向像素 | 代表图 |
|---|---|---:|---:|---:|---|
| manual_added_below_80C | direct_exclusive_pixel_count | 436,208 | 79 | 436,208 | A080;A079;A042;A003;A018 |
| manual_added_at_or_above_80C | direct_exclusive_pixel_count | 29,528 | 57 | 29,528 | A080;A079;A018;A003;A005 |
| machine_only_ge80_raw_binary_including_ignore | direct_exclusive_pixel_count | 177,871 | 79 | 177,871 | A080;A079;A009;A016;A078 |
| high_temperature_on_valid_manual_nonfire | direct_exclusive_pixel_count_excluding_ignore | 90,500 | 62 | 90,500 | A080;A079;A078;A011;A008 |
| high_temperature_machine_fire_on_manual_ignore | direct_exclusive_ignore_decomposition | 87,371 | 38 | 87,371 | A009;A016;A010;A015;A018 |
| machine_seed_below_80C_residue_raw_including_ignore | direct_exclusive_pixel_count | 872 | 37 | 872 | A079;A080;A008;A031;A015 |
| machine_below80_on_valid_manual_nonfire | direct_exclusive_pixel_count_excluding_ignore | 522 | 23 | 522 | A079;A080;A008;A046;A078 |
| machine_below80_on_manual_ignore | direct_exclusive_ignore_decomposition | 350 | 22 | 350 | A031;A015;A016;A040;A023 |
| low_temperature_but_visually_visible_combustion | image_level_active_fire_proxy_not_pixel_visibility | — | 40 | 313,576 | A080;A079;A042;A003;A018 |
| smoke_occlusion | image_level_multi_label_proxy | — | 6 | 46,577 | A011;A006;A010;A034;A001 |
| hot_ground_or_residual_heat | image_level_multi_label_proxy | — | 57 | 45,641 | A080;A079;A078;A048;A021 |
| registration_thermal_leakage | image_level_multi_label_proxy | — | 2 | 63 | A032 |
| label_ambiguity | image_level_multi_label_proxy | — | 1 | 0 | — |
| reflection_or_highlight | unavailable_in_frozen_fields | — | 0 | 0 | — |
| unrelated_heat_source | unavailable_in_frozen_fields | — | 0 | 0 | — |

- “低温但视觉可见燃烧”的可报告代理为：40张已判活动火图像，其上共有313,576个低于80°C的人工新增像素。这不是逐像素可见火焰真值，论文中必须称为active-fire image proxy。
- 热地面/余热、RGB/IR错位和标签不确定只支持图像级重叠代理；反光高亮与无关热源在冻结A001–A101字段中没有独立证据，因此不得编造像素占比。
- 已生成12张类型化代表案例联系图；数据不足的类别保留实际可用例数，不为满足3–5例而重复或虚构样本。

## 与2×2消融证据的连接

- v1.1（D）相对机器监督baseline（A），冻结best checkpoint人工Fire/Heat IoU配对平均提高`+0.023448`，3/3 seed提高。
- 同一对比在机器val134 Fire/Heat IoU上配对平均提高`+0.026506`，3/3 seed提高。
- 这说明人工Smoke/Ignore监督并非简单用人工口径换取机器口径下降：它在人工端点与机器val端点上均为正向。但这仍是开发验证证据，不替代独立test。

## 对一次性test端点的解释边界

用户给定及既有冻结摘要中的一次性test端点为机器定义约`0.689`、人工三分类Fire/Heat约`0.298`，差约`0.391`。本轮没有读取test图像、人工掩膜或预测，也没有重新前向；A001–A101的标签鸿沟量化表明，两套标签逐图IoU均值仅约0.310，且机器/人工Fire像素的≥80°C比例约为99.5%/40.7%。因此，现有证据强烈支持该约0.391差值主要由标签定义和坐标口径差异贡献，不能把它直接表述为同一真值下模型性能从0.689跌到0.298。
- 既有人工test对比中v1.1相对baseline的Fire/Heat IoU回退`−0.010139`，但本轮validation只读重评显示人工Fire/Heat为`+0.023448`（3/3）且机器val134为`+0.026506`（3/3）。两个validation口径方向一致，因此test上的−0.010139应报告为独立test权衡，不能泛化成整体Fire能力退化。

## 可以主张与不可以主张

可以主张：

- 温度阈值机器标签与人工Fire/Heat语义标签存在大规模、方向明确的系统差异。
- 人工标签主要补入低于80°C的区域；在≥80°C机器only中，90,500像素位于有效人工非Fire，另有87,371像素位于人工Ignore，两者必须分开报告。
- 人工Smoke/Ignore监督在三个随机种子的人工与机器validation端点上均提高Fire/Heat IoU。
- 在本FLAME3子集上，标签定义差异会显著改变数值评估结论。

不可以主张：

- 不能把所有机器多标高温像素都叫作余热、热地面、反光或配准泄漏。
- 不能把图像级strata当作像素级原因ROI，也不能将重叠代理类别相加为100%。
- 不能用A001–A047开发集结果替代107张test的一次性结论。
- 在46px候选未人工复核前，不能宣称恢复的原人工Fire掩码已完成空间对齐修正。
- 不能把本数据集结论推广到所有RGB-T火灾数据集，不能断言所有温度伪标签方法均不可用，也不能把全部标签鸿沟归因于任何单一因素。

## 后续排队但本轮不执行

1. v1.2：以v1.1为唯一基线，补A102–A137共36张人工Fire/Heat训练标签；人工未标为火但机器标为火的≥80°C区域按冻结协议标Ignore、禁止改为Background，并统计受影响像素。A001–A047人工三分类为主开发端点，同时报告机器val134；S规则及0.03对称灾难线、0.001 No-Fire误报阀不变；30轮三seed通过后才精确恢复至100轮。
2. FLAME2零样本迁移：按三分类协议映射；若FLAME2 Fire只覆盖可见火焰，未标余热与烟下热足迹一律Ignore、禁止当Background，结果明确称为覆盖差异下的下界估计。
3. 工程清理：将torchsummary硬依赖移入`__main__`，消除`sys.path.insert`，把硬编码D盘路径改为命令行参数；效率统一augment=False、batch 1、640×512、固定warm-up，报告参数量/FLOPs/平均与P95延迟/显存；归档resolved config并记录ImageNet加载与四通道conv1初始化；旧脚本和无关config移入archive。
4. Test-v2 limitation：FLAME3其他场次已停止对外提供，split v2内没有未使用图像；跨场景证据只能采用FLAME2零样本、现有test的事后分层/扰动分析及数据可得性声明，不承诺不存在的新采集测试集。

## 可视化与审计产物

- 全量五列联系图共11页：Corrected RGB、模型实际`convert('L')`热JPG灰度、固定0–500°C Celsius TIFF、机器标签、人工标签。
- 类型化代表案例图共12页，索引见`representative_case_index.csv`。
- 图表：温度分布、逐图IoU直方图、温度中位数散点、互斥分歧构成、属性相关性。
- 完整输出哈希见实验目录中的`SHA256_MANIFEST.txt`，完成审计见`LABEL_GAP_COMPLETION_AUDIT.json`。
