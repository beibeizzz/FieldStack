# FieldStack

面向广告转化率预估的单模型项目。联合建模用户和广告的离散字段、稠密语义特征与多域行为历史，通过目标条件化兴趣提取、字段级特征交互和双流融合输出转化概率。

## 模型

行为域 `seq_a`、`seq_b`、`seq_c` 使用 DIN 聚合。`seq_d` 按 newest-first 顺序截取 1024 个事件，拆分为近期和较早两个 512 窗口，分别由独立的两层 STCA 提取器处理。

每个 STCA 层用 SwiGLU 和 LayerNorm 编码历史，通过 4 头目标交叉注意力获取兴趣摘要；后续查询融合目标广告与前层摘要。最终读出加入原始查询残差，保持每窗口 64 维输出；两个窗口拼接为 128 维。空窗口返回查询表示，padding 位置通过掩码隔离。注意力采用查询侧与聚合后投影，序列长度方向的计算复杂度为线性。

其余结构包括 `(id, weight)` 加权池化、时间桶 Embedding、输入 BatchNorm、ChannelSENET、字段级语义 Tokenizer，以及两层 RankMixer 特征交互。默认字段分组形成 24 个 Token，Token 宽度为 1152，逐 Token pSwiGLU expansion 为 2。深层表示与浅层 MLP 通过分组双线性模块融合。

训练配方使用加权 BCE（`pos_weight=2`）、稀疏参数 Adagrad、稠密参数 Muon/AdamW、SAM 和稠密参数 SWA。数据字段分组由 schema 决定；`run.sh` 的字段编号和分组配置需要与实际数据对应。

## 文件

| 文件 | 用途 |
| --- | --- |
| `model.py` | STCA、DIN、字段 Tokenizer、特征交互和预测头 |
| `train.py` | 参数解析与单卡／DDP 训练入口 |
| `trainer.py` | 优化器、SAM、SWA、评估和检查点保存 |
| `dataset.py` | Parquet 解析、序列截断、时间桶和数据分片 |
| `infer.py` | 根据检查点配置恢复模型并生成预测 |
| `build_dense_stats.py` | 稠密特征统计工具 |
| `utils.py` | 日志、随机种子和早停等辅助函数 |
| `run.sh` | 完整训练配置，自动选择单进程或多卡 DDP |

## 数据与环境

建议使用 Python 3.10+、PyTorch 2.4+。完整配置具有较大的参数和激活预算；按显存容量设置 `--batch_size`。

```bash
python -m pip install -r requirements.txt
```

训练目录应包含 `*.parquet` 和 `schema.json`，schema 描述用户／广告 int、dense 字段，以及各行为域的特征列和时间戳列。数据使用 `user_int_feats_<fid>`、`item_int_feats_<fid>`、`user_dense_feats_<fid>`、`item_dense_feats_<fid>` 等列名。训练标签由 `label_type == 2` 生成；`timestamp` 为样本时间。推理使用 `user_id` 作为输出键，输入应为每个键提供唯一的样本标识。

## 训练

在项目目录运行：

```bash
bash run.sh \
  --data_dir /absolute/path/to/train_data \
  --ckpt_dir /absolute/path/to/checkpoints \
  --log_dir /absolute/path/to/logs
```

脚本优先使用 `PYTHON_BIN` 指定的解释器；默认使用 `python`。传入的参数可覆盖脚本配置，例如 `--batch_size 128`、`--num_epochs 5`。环境变量 `TRAIN_DATA_PATH`、`TRAIN_CKPT_PATH`、`TRAIN_LOG_PATH` 优先于对应 CLI 路径。

STCA 与窗口配置：

| 参数 | 推荐值 | 含义 |
| --- | --- | --- |
| `--seq_stca_domains` | `seq_d` | 使用 STCA 的域，CSV；空字符串选择全部 DIN |
| `--seq_dual_window_domains` | `seq_d` | 分别提取近期／较早窗口的域 |
| `--stca_layers` | `2` | 目标交叉注意力层数 |
| `--stca_heads` | `4` | 注意力头数，需整除 `d_model` |
| `--stca_expansion` | `2` | STCA SwiGLU 扩展倍数 |
| `--seq_max_lens` | `seq_a:256,seq_b:256,seq_c:512,seq_d:1024` | 各域截断长度 |

固定其余配置，可使用 `bash run.sh ... --seq_stca_domains ""` 做 DIN 对照，使用 `--stca_layers 1` 做层数消融。比较时固定数据划分、训练预算与种子，分别记录普通权重和 SWA 权重的结果。训练器报告 ROC-AUC 和 logloss；任务指定的 weighted-AUC 使用相应评估协议计算。STCA 的效果由实际数据上的消融实验评估。

Windows 可直接调用 `python train.py` 并传入上述模型参数；完整配置见 `run.sh`。数据读取和模型实现共用同一套 Python 脚本。

## 推理

选择同时包含 `model.pt`、`schema.json`、`train_config.json` 的检查点子目录：

```bash
MODEL_OUTPUT_PATH=/absolute/path/to/checkpoints/<checkpoint_dir> \
EVAL_DATA_PATH=/absolute/path/to/test_data \
EVAL_RESULT_PATH=/absolute/path/to/results \
python infer.py
```

推理从 `train_config.json` 恢复 STCA 域、层数、头数、扩展倍数和窗口配置，并严格加载权重。输出文件 `predictions.json` 的格式为 `{"predictions": {"<user_id>": 0.5}}`。启用标准差缩放时，相应 `dense_stats.json` 与检查点共同保存和恢复。

## 方法参考与许可

- [STCA：Make It Long, Keep It Fast](https://arxiv.org/abs/2511.06077)
- [DIN：Deep Interest Network](https://arxiv.org/abs/1706.06978)
- [RankMixer](https://arxiv.org/abs/2507.15551)

本项目采用 MIT 许可证，许可条款和版权声明见 `LICENSE`。
