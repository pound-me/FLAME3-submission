# RTX4060Ti 基线承接失败：只读诊断

日期：2026-09-22，Asia/Shanghai。状态：诊断完成；存在明确历史证据缺口；没有启动任何修复、训练或评估。

## 1. 结论先行

1. **承接门失败是真实停止原因，不是训练崩溃。** 三种子各30轮完成，正式训练完成3次、失败/中断0次；R1四个新点未启动。原判定和26-30窗口全部保留。[E1][E2]
2. **确认了环境参照不同。** 原始v1.1目录保存的环境是Python3.9.21 / Torch2.6.0+cu118 / CUDA11.8 / cuDNN90100；本次是Python3.11.5 / Torch2.1.0+cu121 / CUDA12.1 / cuDNN8801。新启动程序核对的是9月18日R1环境，而非原始v1.1环境。配置审计PASS不能解释为历史软件栈相同。[H1][N1][N2]
3. **不能把环境差异直接写成唯一因果原因。** 本次同时改变机器与软件栈；缺原始四通道stem初始张量、最初启动过程和可配对训练RNG快照，也没有本机同seed独立重复训练。现有证据不能区分每个因素的贡献。
4. **本次未发现overflow跳步或额外BN训练累计的证据。** 15个固定窗口checkpoint均通过读取前后SHA核验；其scaler/BN状态支持每seed1830次正常更新的推导。逐步跳步日志没有保存，不能伪装成逐步实测。[K1][K2]
5. **同checkpoint回放一致不等于独立训练可重复，更不等于新旧机器可承接。** 三项证据在第4节分别判定，不能相互替代。

依据报告的本地副本：`evidence_4060/docs/FLAME3_R1_SWEEP_REPORT_REV2_20260922.md`。

实际SHA256与用户给定值一致：

`3BFD177A95AE085F0FBEDD13FD030D5B9A14F9B0BE5382D40A6BC0FCCA7600FC`

## 2. 范围与证据等级

本次只读取已保存的报告、审计、配置、源码、train/val标量日志，以及CPU反序列化的现有checkpoint/预训练权重元数据。没有构建模型、前向、反向、重放预测或训练；没有读取任何图像或掩膜，更没有读取test107的图像、掩膜、预测、统计量或样本清单。

证据分级：**实存**是文件内直接记录；**静态**是已取得代码的可见行为；**推导**依赖明确前提；**缺失/不可核验**不得按默认值填补。

已取得：4060三个种子的完整90轮标量日志、配置/初始化/环境、最终审计；15个epoch26-30 checkpoint的CPU元数据；实际安装Torch2.1.0的DataLoader/worker/sampler/SGD/GradScaler源码。拷贝的13份项目源码逐文件匹配本次冻结manifest。

旧4090首次连接成功，取得原始seed200的完整100轮日志、配置、环境和run_summary，以及9月21日补跑seed200的环境/配置/RESULT。原始seed202环境也曾通过短命令读取，与seed200的软件栈一致。随后直连和临时SSH中转均超时，未修改网络、服务或远端文件。

**历史证据缺口必须保留：**

- 原始seed201/202完整逐轮日志和配置未拉取成功；这两种子的原26-30窗口数值仅使用既有冻结承接审计，不伪称此次重算原始日志。
- 原始epoch26-30 checkpoint在已有精确路径检查中缺失；本次不使用epoch100或best_S替代。最初初始化张量/哈希和逐batch RNG/BN轨迹也未取得。[E3]
- 原始目录现在是100轮最终记录，名称仍含30e。保存的environment可证明该目录记录的软件栈，但缺完整启动/恢复历史，无法额外保证它未在恢复至100轮时被覆盖。
- 原始训练源码只有汇总SHA，未取得与该历史SHA绑定的完整逐文件快照；本次冻结的共享训练入口不能自动当作原始执行快照。
- 9月21日补跑seed200的`metrics.jsonl`在传输中被截断，不参与分析。其完整RESULT和已有9月21日报告只作辅助上下文；未取得另两seed完整原始日志及checkpoint元数据。
- 原始seed200拷贝可完整解析、100轮连续且字节数与远端目录列表一致，但远端二次SHA未在断线前完成。本地SHA可供追踪，不冒充远端双端复核。
- 原始训练时的驱动、TF32开关、完整依赖、实际cuDNN/SGD kernel选择、首轮初始化RNG状态：缺失/不可核验。

## 3. 新旧实际训练协议差异表

“原始”列以取得的seed200实存记录为主；未取得的其他seed细节不外推。9月21日补跑与原始v1.1是两个不同证据对象。

| 项目 | 原始v1.1 / 4090记录 | 本次4060Ti记录 | 判断与限制 |
|---|---|---|---|
| 模型/参数 | pidnet_s、fusion、7,717,095参数 | 同结构及参数量，无R2/R3门控 | 原始环境与新初始化审计均记录；不是容量变化证据。[H1][H2][E4] |
| 预训练文件 | ImageNet权重SHA为F96E2C96…A5F359 | 同一完整SHA，实存加载301个tensor | 旧实际匹配tensor数未取得；不能把新301当旧日志。[H1][E4] |
| 四通道stem | 配置跳过`conv1.0.weight` | 同样跳过整个weight，shape=[32,4,3,3] | 不是只随机初始化热通道；历史初始化张量缺失。[H2][N3][K1] |
| 独立初始化/恢复 | seed200配置明确；最初启动与恢复链缺失 | 三seed均`formal_from_seed_initialization=true`、`prior_checkpoint_loaded=false` | 新任务未从工程或其他seed checkpoint热启动。历史不可完全核验。[E5] |
| 物理batch | BATCHSIZE=8 | batch8，单GPU，无梯度累计 | 旧为有效配置；新为实际启动源码及计数。[H2][N1] |
| workers | NUM_WORKERS=4 | 训练与逐轮验证均4；事后评估0 | 事后评估0不是训练workers变更。[N1] |
| sampler | targeted sampling=false；历史实际对象/状态缺失 | 无放回RandomSampler，shuffle=true，独立CPU generator(seed)，drop_last=true | 当前共享旧入口也是此模式，但原执行快照不可完全核验。[N1][D1] |
| worker初始化 | 原运行额外hook记录缺失 | Torch默认worker seed后加只做访问审计的hook；prefetch_factor=2，persistent_workers=false | 新hook静态未发现RNG调用；未取得旧worker状态，不能宣称逐worker相同。[N1][D1][D2] |
| 每轮样本/batch | 原seed200日志每轮Fire+NoFire=488，损失分项计数与61 batch相符；总池493来自冻结协议 | 池493，实际488样本/61 batch；每轮drop_last丢5个 | 新有直接计数；旧61缺独立batch计数字段，属配置+日志支持的推导。[H3][E6] |
| 验证 | 原日志完整134张，manual dev47，NoFire35 | 同134/47/35；17 batch、batch8 | 不读取test107；验证样本数没有发现变化。[H3][E6] |
| 几何增强 | 配置multi_scale=true、flip=true、brightness=false，512×640，scale0.8-1.5 | 相同配置；每轮有顺序和几何tensor摘要SHA | 原seed200前30轮8项目标聚合统计全部相同，但不等于逐样本/顺序SHA相同。[A1] |
| R1热退化 | 原基线无R1配置 | B0明确绕过ThermalLoader | `apply_arm(...,'R1')`在此是结构不变的deepcopy，不表示启用了热退化。[N1][N4] |
| 训练长度 | 原目录最终100轮；比较仍取26-30 | 实际30轮，取26-30 | 不能用原epoch100换窗口；100对30的总长度差不能单独解释前30轮差异。[H2][H3] |
| 优化器 | 配置SGD、lr=.001、momentum=.9、WD=1e-5 | 相同；checkpoint还记录dampening=0、nesterov=false、foreach=null | 原实际optimizer backend/foreach路径缺失；不伪称内核一致。[K1] |
| 更新次数 | 原前30轮计划1830；实际成功更新/跳步历史状态缺失 | 61 scaler.step/轮，合计1830；状态支持1830次成功更新 | 新“成功更新”是代码+scaler状态推导，不是独立step计数器日志。[K1][G1] |
| LR日程 | 配置和逐轮日志为poly100；逐步LR实存缺失 | base=.001，power=.9，总步数6100，30轮结束LR=.0007255707435222272 | 新90个lr_end均与公式精确一致；没有错用poly30的证据。旧逐步轨迹不可直接核验。[N5][E6] |
| AMP/GradScaler | AMP_INIT_SCALE=128；历史有效AMP开关/逐轮scaler实存未取得 | AMP启用；scale128、growth_factor2、backoff.5、interval2000 | 新1-25轮独立scaler快照缺失，26-30实存；历史不以默认值填补。[K1][G1] |
| 梯度裁剪 | 配置L2 max_norm5；seed200日志记录已启用、clip统计 | 同L2 max_norm5；先unscale，再clip，非有限梯度报错，再step/update | 裁剪统计可查90轮CSV；不把“loss下降”当重复性证据。[N5][H3] |
| BN | 历史逐层状态/计数轨迹未取得 | train()训练、eval()+inference_mode验证；79个BN计数对预训练起点增量均为61×epoch | 新26-30未见额外training-mode累计；不能据此证明历史BN逐位相同。[K2][N5] |
| TF32 | 原始environment未记录 | cudnn.allow_tf32=true；matmul.allow_tf32=false | 原始不可核验，不能按库默认值宣称相同。[H1][E5] |
| cuDNN模式 | benchmark=false、deterministic=true；deterministic_algorithms_enabled=false | benchmark=false、deterministic=true；全局deterministic_algorithms未记入运行environment | 开关不是相同kernel或跨机器逐位相等的证明。[H1][E5] |
| Python/Torch/CUDA/cuDNN | 3.9.21 / 2.6.0+cu118 / 11.8 / 90100 | 3.11.5 / 2.1.0+cu121 / 12.1 / 8801 | 已确认保存记录不同；新环境匹配9月18日R1及9月21日补跑环境，不匹配原始记录。[H1][N2][H4] |
| 驱动 | 训练时驱动缺失 | 本次诊断时查询为560.94；训练时未单独保存 | “当前驱动”不写成历史实测；旧当前驱动因断线未取得。[K3] |
| 源码身份 | 汇总source_sha256=931abf5f…90bec59；缺对应完整快照 | manifest=BB74F0C4…32E1；13个相关源码逐文件核验通过 | 不同hash算法/绝对路径参与，不能仅因汇总SHA不同便认定训练内核改动。[H1][N5] |

seed200实际配置差异仅五项：EPOCHS、PRETRAINED路径、ROOTDATASET路径、人工验证包路径、FORMAL_TRAINING_GPU。完整结构化差异见`OBSERVED_CONFIG_DIFFERENCES_SEED200.json`。**软件栈不是这些配置键的一部分，因此配置字段相同不代表执行协议完全相同。**

## 4. 三种证据必须分开

| 命题 | 现有证据 | 结论 |
|---|---|---|
| 同checkpoint回放一致 | 新3seed×5checkpoint，在既有逐轮验证和事后clean评估之间，冻结9项指标最大绝对差均0；评估前后model/BN摘要相同 | **已支持本次相同权重的回放一致**。本诊断只检查记录，没有再回放。[E1][E7] |
| 同机独立训练可重复 | 本次每seed只有一次训练，没有同配置同seed从头两次的记录。三种不同seed不是重复实验 | **缺失/不可判定**。任务0初始化检查不是完整训练重复性证据。9月21日4090补跑与历史软件环境不同，也不构成同协议重复证明。 |
| 新旧机器训练结果可承接 | 保持既有26-30窗口和门限，三个seed均未同时通过承接条件 | **本次承接不通过**，不得自动恢复任务2。[E1] |

固定窗口数据如下，均为比例而非百分数：

| seed | 原Smoke | 新Smoke | 差值 | 原No-Fire FP | 新No-Fire FP | 失败项 |
|---:|---:|---:|---:|---:|---:|---|
| 200 | 0.7005964004 | 0.6850686193 | -0.0155277811 | 0.000115234375 | 0.00000244140625 | abs(Smoke差)不小于0.010 |
| 201 | 0.7028778687 | 0.7078245246 | +0.0049466558 | 0.001738124302 | 0.000337890625 | abs(FP差)=0.001400233677，不小于0.001 |
| 202 | 0.6967310785 | 0.6486443259 | -0.0480867526 | 0.000074881417 | 0.000549473354 | abs(Smoke差)不小于0.010 |

三个seed的No-Fire绝对FP均不超过0.005。seed201的FP实际改善，但承接检验要求双向接近，因此仍失败；不能把它说成误报恶化。原门限没有改动。[E1]

9月21日4090补跑的完整既有报告也记载历史恢复失败、环境不一致。seed200完整RESULT窗口Smoke=0.681498408832834，原历史为0.7005964004049379。因此，**偏离历史结果并非首次出现在换4060后**。这反驳“只因换卡才出现偏差”的简化叙述，但不能单独归因到某个Torch/cuDNN版本。[H4]

## 5. 四通道stem、RNG与BN专项

### 5.1 初始化

新链路是：读取/核验环境与文件 -> `seed_everything(seed)` -> 构建train/val数据集与验证目标 -> 独立train generator -> 构建DataLoader但尚未迭代 -> 构建PIDNet -> 加载预训练 -> R1结构恒等deepcopy -> 校验初始state SHA -> 转CUDA -> criterion/SGD/scaler -> 首个训练batch。[N1:64-101]

PIDNet先构造整个网络，再对Conv权重调用Kaiming normal。四通道stem的整个`conv1.0.weight`被排除在ImageNet加载之外；不是RGB三通道预训练加一个随机热通道。预训练文件原stem是[32,3,3,3]，本次是[32,4,3,3]。模型构建所消耗的CPU RNG可能影响未被预训练覆盖的张量；当前保存证据不能证明历史相同seed获得完全相同的初始张量。[N3][K3]

本次任务0保存的随机stem SHA：

| seed | random_stem_sha256 |
|---:|---|
| 200 | 7FB0E4E0AE23F3D3FE02D9497CB8EA7A2D07BB076FC7BA5679387D3CAF17E5C9 |
| 201 | 635A3713CA7D02CE2E64C930510162362CE1687478C525676A105E17E46263C4 |
| 202 | 1A5C8C3894BED501957659450040B00DAE31A747A93FB210BC8171B8BA36470C |

正式运行整体初始state SHA逐seed等于该任务0记录，并检查301个匹配tensor。此处支持的是**本次准备与正式初始化一致**，不支持历史stem一致，也不支持完整训练可重复。[E4][E5]

`apply_arm(...,'R1')`只deepcopy，不添加模块；内部CPU随机种子作用域由`fork_rng(devices=[])`包围后恢复。未看到R1分支额外前向或CUDA随机数调用。B0后续明确不使用ThermalLoader。[N4][N1:138]

### 5.2 数据加载与随机状态

已取得代码中，dataset构造只读CSV与配置；随机尺度/裁剪/翻转出现在取样增强，而非构造函数。训练DataLoader的sampler与worker base_seed使用独立train generator。Torch安装源码明确：迭代器的base_seed使用`loader.generator`；worker随后设置Python、Torch及NumPy随机种子；本次额外hook只建立访问审计。[D1:601][D2:223][N6]

验证DataLoader未给独立generator，**创建验证迭代器会消耗主进程Torch CPU RNG**。因此“eval不消耗任何随机状态”是错误的描述。不过本次训练sampler/worker使用另一个独立generator，模型在验证前已初始化，且事后扰动评估在另一进程；不能仅凭这一CPU RNG消耗就认定训练被污染。[D1][N1][N7]

每轮顺序/几何摘要的计算只读取CPU tensor字节，没有静态可见的随机抽样或原地修改。新B0三seed各30轮均有对应摘要；旧摘要没有保存，不能对旧新做逐轮哈希相等检验。[N1:37][E6]

额外检查历史seed200前30轮8项模型预测无关的训练目标聚合统计：Fire/NoFire样本数、NoFire有效像素及dense背景/烟/火、fire_core、hard_background像素，30/30轮均相同。它支持“没有发现粗粒度输入构成错位”，**不是**同样本次序、每张几何或初始化一致的充分证明。[A1]

### 5.3 验证、扰动与BN

训练内核每轮先`model.train()`；逐轮验证使用`@torch.inference_mode()`与`model.eval()`。事后六条件评估由队列等待训练子进程结束后启动独立子进程，并从已保存checkpoint加载模型；它不能修改已经结束的训练进程RNG。随后另一个seed训练再次独立设种子。[N5:664][N5:945][N7:67][N1:200]

本次15个checkpoint，每个79个BN计数，在扣除该层实际匹配的预训练计数（无对应项则0）后，全部等于61×epoch：1586、1647、1708、1769、1830。较大绝对计数727311等包含725725的预训练起点，不表示此次训练多跑了数十万batch。[K1][K2][K3]

已有事后评估记录model/BN摘要前后一致。结合正常的train/eval切换和计数增量，没有发现新运行额外training-mode验证累积的证据；但未保存每次逐轮验证前后的完整BN/RNG摘要，不能宣称所有瞬时状态已逐一核验，更不能替代历史BN状态。[E7]

## 6. 三种子逐轮汇总与数值状态

完整90轮表：`EPOCH_SCALARS_4060.md`。全精度可机读表：`EPOCH_SCALARS_4060_90_ROWS.csv`。包含train/val loss、Smoke IoU、No-Fire联合/烟/火FP、LR、裁剪统计、scaler实存/推导、跳步缺失/推导、batch/样本计数、数据摘要及原日志行号。

| seed | epoch1 train loss | epoch30 train loss | epoch26-30 Smoke均值 | epoch26-30 No-Fire FP均值 | 实际轮/batch |
|---:|---:|---:|---:|---:|---|
| 200 | 6.23090780 | 0.17734036 | 0.6850686193 | 0.00000244140625 | 30 / 1830 |
| 201 | 8.29347667 | 0.19427580 | 0.7078245246 | 0.000337890625 | 30 / 1830 |
| 202 | 5.89811238 | 0.19531619 | 0.6486443259 | 0.000549473354 | 30 / 1830 |

表中首末loss只作轨迹描述，不用于选checkpoint、调整窗口、提前停止或声称创新有效。固定窗口平均由已有日志重新聚合，与既有WINDOW逐项核对，无新的推理。

**LR：** 全部90条实存lr_end与`0.001*(1-step/6100)^0.9`一致，epoch1末值0.0009911431715319025，epoch30末值0.0007255707435222272。step按0起，30轮最后一次为1829；不是poly30归零。

**Scalers：** 初始scale=128、tracker=0；三seed的epoch26-30均实存scale=128，tracker分别为1586/1647/1708/1769/1830，growth_interval=2000。1-25轮独立checkpoint已不保留，故对应“scaler实存”栏填缺失。

**跳步与更新次数：** 逐step的found_inf/skip标志没有保存，历史运行也未取得。新训练源码每batch一次step/update，90轮均61batch，未恢复运行，末tracker=1830且小于首次growth阈值2000。依据实际安装GradScaler对“连续未跳步次数”的定义，可推导这1830次没有overflow跳步、scale保持128，每epoch61次更新；CSV明确标记为推导，不将其写为逐步日志实测。[G1:572]

梯度先unscale再L2裁剪，非有限梯度会抛错；三seed没有此类训练失败记录。现有信息不支持用“GradScaler跳过许多更新”解释本次掉分。但完整逐step梯度及优化器状态仍缺失，不能反向重建历史训练。

## 7. 能认定与不能认定的原因

**已证实的事实：** 环境审计基准是阶段2 R1，不是原始v1.1；两份保存的软件栈不同；新训练完整；既定承接门失败；同checkpoint回放通过；新初始化与本次任务0一致；新scaler/BN状态与1830次正常训练相容。

**可作为后续排查项，但目前不能认定因果：** Torch/CUDA/cuDNN版本、原始未保存的TF32/内核选择、CPU初始化随机流及未加载预训练的随机层、旧启动/恢复路径、两机器计算路径、旧初始BN状态与数据RNG。没有受控对照，不能给这些因素分配贡献比例。

**目前不支持：** 改成poly30、误用R1增强、batch或workers暗改、运行缺轮、事后扰动评估改了已结束训练、很多overflow跳步、额外BN训练累计等解释。这里的“不支持”受证据边界限制，不表示对旧运行所有细节的排除证明。

**禁止归因：** 不把结果写成“正常显卡误差”“4060精度必然差”“模型容量不足”，不由单seed曲线推定学习率最优，更不通过反复重跑挑接近历史分数的结果。

## 8. 下一阶段建议：仅提案，未执行

### A. 修复已明确的协议参照差异，再决定是否重建基线

先确定究竟要承接“原始Torch2.6 v1.1”还是建立“当前Torch2.1结构筛选协议”。把二者环境分别锁定，审计显式列出双重参照，不能只验证R1环境就写历史协议一致。

若目标是原历史承接，应先补取4090原始首30轮启动/恢复记录、对应源码快照及可获得的依赖/驱动/TF32记录，再评估能否独立部署历史环境。历史stem或checkpoint缺失是客观边界，即使软件版本恢复也不能保证重建旧随机轨迹。不得改写冻结源码、覆盖旧环境/结果或放宽原门限；任何正式补跑要单列新授权和预算。

### B. 固定seed202，两次各50 batch同机诊断

**推荐的下一项小预算实验，但本次没有执行。** 两个全新进程、同机同软件栈、同seed202、同batch8/workers4、相同预训练文件；不从任一已有训练checkpoint热启动，不调参数、不换seed。训练取已授权train池，不碰test107。

在模型构建前、构建后、预训练加载后、首个DataLoader迭代前分别记录Python/NumPy/Torch CPU/CUDA/train-generator状态SHA及模型/stem/BN摘要。逐batch记录输入顺序/增强摘要、loss、LR、unscale后梯度范数、scale前后值、found_inf/真实optimizer更新标志及BN增量。完整保留两次结果，按预先冻结的容差比较，不选择更接近历史成绩的一次。

**重要日程陷阱：不能直接传`max_batches=50`给当前内核。** 当前内核以`epoch_batches=min(len(loader), max_batches)`计算总步数，会把poly100分母从6100改成5000。将来诊断必须保持原61batch/epoch的日程索引，仅在第50个batch后终止诊断记录，不能改变LR分母。[N5:695]

诊断的目标是确定同机短程重复性及首个分歧位置，不是恢复历史成绩，也不能凭50batch证明30轮泛化表现。B通过与否都应先提交结果，不自动触发正式30轮或C。

### C. 建立4060Ti独立实验组

若历史状态不可恢复而同机重复性可接受，可申请新的预注册：以同一4060软件栈/训练链路建立本机配对基线，再对四个R1点按固定200/201/202种子扫描。是否可复用本次3个baseline必须在新授权中预先决定，不能根据候选成绩决定。旧4090 R1/R3只能作历史上下文，不能当成本机同协议对照。

C不是“放宽当前承接门后恢复任务2”。必须保留本次失败结论、单独报告数据反复使用的开发集限制、固定26-30规则以及Smoke/鲁棒收益/No-Fire护栏/置零副作用。新增训练预算另行审批，不沿用原15次额度自动启动。

**建议顺序：A的证据与参照修正 -> 经授权执行B -> 审核B后再决定A的正式重建或C。此处仅提案，三者都没有启动训练。**

## 9. 完整性与停止

- 原报告SHA已匹配；原报告、标签、冻结源码、配置和checkpoint没有改写。
- 本次15个4060 checkpoint及预训练权重均做CPU读取前后SHA核对，全部不变；没有把checkpoint拉到本地或另存为可训练副本。
- 原运行最终审计记载1932项输入核验通过、source manifest不变、test107相关允许读取为0；这些是**引用原审计**，本诊断没有重读其图像/掩膜来重新验证。[E2]
- 本地只新增本诊断目录的证据副本、只读提取脚本、汇总表与报告；没有修改原报告。远端没有新写入诊断脚本或结果文件。
- 不启动R1扫描、任务3、100轮确认、新结构训练或B/C。没有重新创建心跳、计划任务或服务；报告与哈希交付后停止。
- 最终本地文件列表、大小和SHA见`DELIVERY_SHA256.csv`；可读入口是本报告和`EPOCH_SCALARS_4060.md`。

## 10. 证据索引

下面路径均相对于本诊断目录。`路径:行号`用于定位；未标行号的JSON引用整个命名对象，不把缺失内容补全为默认状态。

| 编号 | 文件路径及定位 |
|---|---|
| E1 | `evidence_4060/audit/BASELINE_REPRODUCTION_CHECK.json:1`；三seed的seeds数组、conditions及replay_max_abs |
| E2 | `evidence_4060/audit/COMPLETION_AUDIT.json:1`；`evidence_4060/audit/FINAL_INPUT_SOURCE_CHECK.json:1` |
| E3 | `evidence_4060/audit/OLD_HOST_PRESTART_CHECK.json:1`；精确15个路径检查未找到历史窗口 |
| E4 | `evidence_4060/audit/INITIALIZATION_CHECK.json:1`；三seed初始化、stem哈希、scaler、BN模式 |
| E5 | `evidence_4060/baseline/B0-rep-20260922-v2/seed{200,201,202}/INITIALIZATION.json:1`与`environment.json:1` |
| E6 | `evidence_4060/baseline/B0-rep-20260922-v2/seed{200,201,202}/metrics.jsonl`；行号等于epoch，1-30 |
| E7 | `evidence_4060/evaluation/B0-rep-20260922-v2/seed{200,201,202}/WINDOW.json`；clean_replay_max_abs及model_and_bn_unchanged |
| H1 | `evidence_4090/original/seed200/environment.json:2`；版本、pretrained_sha256、source_sha256和参数量 |
| H2 | `evidence_4090/original/seed200/resolved_config.json:1` |
| H3 | `evidence_4090/original/seed200/metrics.jsonl:1`；原日志100行，本诊断仅用前30行作轨迹对照 |
| H4 | `evidence_4090/replay/seed200/environment.json:1`、`RESULT.json:1`；本地既有`docs/FLAME3_BASELINE_REPLAY_AND_WINDOW_EVALUATION_20260921.md:21`，副本见`context/` |
| N1 | `evidence_4060/source/run_experiment.py:22`（环境参照）、`:64`（训练）、`:200`（独立评估） |
| N2 | `evidence_4060/source/runtime_support.py:52`（版本检查与TF32设置） |
| N3 | `code_evidence/frozen/runtime/source/src/baseline_runtime.py:257`（预训练加载）、`code_evidence/frozen/runtime/source/third_party/RoboFireFuseNet/models/pidnet.py:21`与`:97`（四通道构建及初始化） |
| N4 | `code_evidence/frozen/runtime/implementation/structure_arms.py:142` |
| N5 | `code_evidence/frozen/runtime/source/src/train_baseline_v11.py:37`（scaler）、`:48`（clip）、`:634`（源码hash算法）、`:650`（LR）、`:664`（train）、`:826`（step）、`:945`（eval）、`:1359`（共享入口构建顺序） |
| N6 | `code_evidence/frozen/runtime/source/src/flame3_dataset.py:38`、`code_evidence/frozen/runtime/source/third_party/RoboFireFuseNet/datasets/base_dataset.py:100`与`:245` |
| N7 | `evidence_4060/source/run_queue.py:21`（独立子进程）、`:67`与`:82`（训练后评估） |
| D1 | `code_evidence/torch_2_1_0/utils/data/dataloader.py:233`、`:349`、`:601`；`utils/data/sampler.py:148` |
| D2 | `code_evidence/torch_2_1_0/utils/data/_utils/worker.py:223` |
| G1 | `code_evidence/torch_2_1_0/cuda/amp/grad_scaler.py:92`、`:314`、`:422`、`:572` |
| K1 | `CHECKPOINT_METADATA_4060.json:1`；15个checkpoint的scaler、optimizer、BN与前后SHA |
| K2 | `BN_SCALER_CHECKPOINT_CHECK.json:1`；15×79个BN计数对预训练起点的差值核验 |
| K3 | `LIBRARY_PRETRAIN_METADATA_4060.json:1`；预训练BN元数据、SHA和诊断时驱动；包含Torch安装源码字节的压缩证据 |
| A1 | `SEED200_TRAIN_TARGET_COUNT_COMPARISON.json:1`；8个聚合字段×30轮；不是逐样本哈希 |

最终判定：**本次承接失败维持；已确认参照环境不同；掉分的唯一原因不可判定；同机独立训练重复性尚未建立。等待新授权，不自动继续。**
