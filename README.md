# TailScope

面向广告转化率预测的多行为域推荐模型。TailScope 将用户属性、候选广告、行为序列和记忆槽位组织为统一 Token，通过候选条件尾部摘要控制长历史的输入规模，再利用带拓扑约束的注意力提取匹配信号。

## 核心设计

1. **双向条件查询**：用户属性以广告属性为条件更新，广告属性以用户属性为条件更新；两侧分别池化出多个查询，供各行为域读取。请求小时与各域活跃状态参与查询构造。
2. **候选条件尾部摘要**：近期事件保留逐事件表示；较早事件按位置分箱，在每个箱内按候选相关性加权，并与均值表示门控混合。每个行为域拥有独立的评分网络和门控。
3. **拓扑约束骨干**：行为读取本域局部邻居、查询和记忆；查询读取本域完整的压缩历史及其他域近期窗口。全局记忆聚合整体信号，分域记忆聚合本域信号。共享注意力与带类型低秩适配器的 FFN 联合更新全部 Token。
4. **分层预测**：拼接查询均值、广告均值、全局记忆均值及各域记忆均值，通过 MLP 输出单个转化 logit。

记忆槽位在每次前向中按样本初始化、在层间更新。注意力采用稠密 SDPA 与结构化掩码，计算量取决于压缩后的总 Token 数。行为顺序由时间降序排列，时间差桶直接加入事件表示。

## 候选条件尾部摘要

令 `x_i` 为行为特征与时间桶组成的事件向量，`c` 为候选广告属性 Token 的均值。先对事件与候选分别做 LayerNorm，再在各箱内计算：

```text
e_i   = MLP([LN(x_i), LN(c), LN(x_i) * LN(c), LN(x_i) - LN(c)])
a_i   = masked_softmax(e_i)
h_att = sum_i(a_i * x_i)
h_avg = masked_mean(x_i)
g     = sigmoid(Linear([LN(c), LN(h_avg)]))
h_bin = h_avg + g * (h_att - h_avg)
```

评分网络最后一层采用零初始化，初始注意力在有效事件间均匀分布；门控初始值为 `0.1`。因此模型从均值摘要起步，再学习候选相关的加权摘要。门控按样本与区间自适应调整，保留整体兴趣与目标匹配信号。空箱输出零向量并标记为 padding，Softmax 在 FP32 中计算；近期事件的表示与掩码原样保留。

这一模块的优先级来自压缩阶段的信息瓶颈：尾部均值会将少量相关行为与大量其他行为混合，后续网络接收的是汇总结果。提前引入候选条件，可以在同一摘要预算内调整哪些历史证据得到保留。

| 候选改动 | 解决的问题 | 本项目的选择依据 |
|---|---|---|
| 候选条件尾部摘要 | 历史汇总阶段的相关性稀释 | 直接作用于信息压缩入口，具有单开关对照 |
| 扩展 FFN 或专家网络 | 压缩后表示的拟合容量 | 对已有摘要增加建模能力，参数与优化成本同步增加 |
| 增强预测读出 | 已生成的查询和记忆之间的融合 | 作用于最终表征，效果依赖上游保留的证据 |

研究依据包括 [DIN](https://arxiv.org/abs/1706.06978) 的候选条件兴趣激活、[SIM](https://arxiv.org/abs/2006.05639) 的目标相关长历史筛选，以及 [TWIN](https://arxiv.org/abs/2302.02352) 对筛选阶段相关性质量的研究。这里采用分箱软聚合与门控混合；它们对当前模型的收益属于待验证假设。上述论文主要研究 CTR，当前输出任务为点击后的转化率。

## 文件

```text
model.py        特征 Token 化、模块组装与预测读出
blocks.py       条件查询、类型适配 FFN、记忆与注意力路由
history.py      候选条件尾部摘要与均值对照
dataset.py      schema 驱动的 Parquet 流式读取
train.py        训练、时间切分、评测、早停和 checkpoint
infer.py        独立推理并按输入行输出概率
config.json     模型配置
make_demo.py    合成样本生成器
tests/          压缩语义、拓扑、梯度及数据契约测试
requirements.txt
LICENSE
```

## 运行

使用 Python 3.10+。按设备安装合适的 PyTorch 构建，再安装依赖。在本目录执行：

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python make_demo.py --output outputs/demo
python train.py --data outputs/demo/train --valid-data outputs/demo/valid --schema outputs/demo/schema.json --config outputs/demo/config.json --output outputs/demo_target --device cpu --epochs 1 --batch-size 16
python infer.py --checkpoint outputs/demo_target/best.pt --data outputs/demo/infer --output outputs/predictions.csv --device cpu
```

合成数据用于检查训练与推理流程。每次实验使用独立输出目录；推理文件包含 `row_index,user_id,timestamp,probability`，顺序为文件名排序、行组顺序和原始行顺序。

## 输入格式

`schema.json` 描述以下字段组，`make_demo.py` 提供完整可运行示例。

| 字段组 | schema 格式 | Parquet 列名 |
|---|---|---|
| 用户／广告离散属性 | `user_int`、`item_int` 中的 `[fid, vocab_size, width]` | `user_int_feats_ID`、`item_int_feats_ID` |
| 用户／广告连续属性 | `user_dense`、`item_dense` 中的 `[fid, width]` | `user_dense_feats_ID`、`item_dense_feats_ID` |
| 行为域 | `seq_a` 至 `seq_d` 的 `prefix`、`ts_fid`、`features: [[fid, vocab_size], ...]` | `prefix_ID`，同一事件的字段位置对齐 |
| 请求时间 | 秒级 Unix 时间戳 | `timestamp` |
| 训练标签 | `label_type == 2` 映射为正样本，其余映射为负样本 | `label_type` |

离散 ID 使用 `1..vocab_size`，无效或缺失值映射到 `0`。多值字段做 masked mean 后进行分组投影；连续属性按 schema 字段投影，非有限值清零。广告连续属性可为空。超过 `max_embedding_vocab` 的字段采用零 embedding，训练前应结合词表大小确认此配置。

行为字段与时间戳联合排序，保留最近 `seq_max_lens` 个事件；缺少时间戳时使用输入顺序。请求时间减去行为时间得到时间差桶。小时特征按 UTC+8 计算。当前额外域统计输入为零，查询时间上下文使用请求小时和各域活跃状态。

## 默认配置

| 配置 | 值 |
|---|---|
| 隐藏维度／Embedding 维度 | 64 / 32 |
| 注意力头／骨干层数 | 4 / 2 |
| 用户／广告离散属性 Token | 5 / 4 |
| 每域用户侧／广告侧查询 | 4 / 2 |
| 全局／每域记忆 | 8 / 2 |
| FFN 扩展倍数／适配器秩 | 4 / 8 |
| 局部邻居窗口／跨域近期窗口 | 8 / 32 |
| 输入截断预算 a/b/c/d | 512 / 256 / 512 / 768 |
| 近期保留预算 a/b/c/d | 96 / 64 / 96 / 128 |
| 每域尾部摘要 | 4 |
| 行为 Token 总预算 | 384 个近期位置 + 16 个摘要 = 400 |
| 候选条件摘要新增参数 | 四域合计 34,568 |
| 优化器／学习率／Weight decay | AdamW / 1e-4 / 1e-4 |
| Batch size／最大 Epoch／早停耐心 | 128 / 5 / 2 |
| Dropout／梯度范数裁剪 | 0.02 / 1.0 |
| 目标与评测 | BCEWithLogits；AUC、LogLoss |

表中的长度是配置预算，实际有效行为数由样本决定。候选摘要在原始尾部事件上增加逐事件评分计算，骨干中的 Token 数保持固定。各域尾部为空时，该模块在对应样本上输出被掩码的空摘要。

## 单模块对照实验

准备按时间划分的训练集和验证集，运行以下两组命令。两组共享特征、输入长度、Token 预算、骨干、读出和训练超参数，唯一算法开关为 `tail_pooling`。

```bash
python train.py --data data/train --valid-data data/valid --schema data/schema.json --output outputs/mean_s42 --device cuda --amp --seed 42 --tail-pooling mean
python train.py --data data/train --valid-data data/valid --schema data/schema.json --output outputs/target_s42 --device cuda --amp --seed 42 --tail-pooling target
```

也可在同一数据目录上用 `--split-timestamp <秒级边界>` 替代 `--valid-data`：训练样本时间小于边界，验证样本时间大于等于边界。BF16 训练使用 `--amp`。Checkpoint 保存模型参数、schema 和配置，推理自动恢复对应的摘要模式。

建议以种子 `42、43、44` 分别训练，比较总体 AUC、LogLoss，以及至少一个域存在尾部行为的长历史子集。使用同一验证集逐样本配对比较；同时记录训练耗时与峰值显存。共同网络参数在同种子下具有相同初始化，候选摘要评分从均匀分布开始。

| 方案 | 验证 AUC | LogLoss | 长历史子集 AUC |
|---|---|---|---|
| 均值摘要 | 待测 | 待测 | 待测 |
| 候选条件摘要 | 待测 | 待测 | 待测 |

验证 AUC 用于早停和模型选择；单类别验证集使用 LogLoss 选择。验证集与测试集承担独立职责，最终测试集用于确认选定配置的泛化表现。

## 验证与许可

本地已通过 18 项测试，覆盖摘要初始化、候选敏感性、分箱隔离、padding 与空历史、评分及门控梯度、骨干路由、保存加载和 CUDA BF16。默认四域长度下已完成 CUDA 前反向检查，行为 Token 数为 400。两种摘要模式均完成合成数据训练、checkpoint 恢复和独立推理检查，各输出 32 行预测；独立加载的预测差异小于 `1e-6`。真实数据 AUC 与收益幅度待实测。

代码采用 [MIT License](LICENSE)。
