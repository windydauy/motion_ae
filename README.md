# motion_ae

`motion_ae` 是一个面向 `motion.npz` 数据的窗口级运动重建项目，核心结构是 `MLP Encoder + iFSQ + MLP Decoder`。它不使用 VAE，也不包含 KL loss，训练目标只有重建误差。

当前实现适合做两类事情：

- 把运动序列编码为离散 latent，方便后续做检索、压缩、先验建模或 token 化分析
- 在固定长度时间窗上做稳定的自编码重建，检查 motion 数据表达是否充分

项目默认处理长度为 `10` 的滑动窗口，每个时间步的特征由 `joint_pos`、`joint_vel`、以及 pelvis 在 anchor frame 下的姿态与速度信息拼接而成。

## 方法概览

模型流程如下：

1. 从单个 `motion.npz` 中读取帧级特征，构造成 `m_t`
2. 用滑动窗口得到 `m_{t:t+9}`，shape 为 `[B, 10, D]`
3. 将窗口 flatten 为 `[B, 10 * D]`
4. 送入 `MLPEncoder` 得到连续 latent `z_c`
5. 通过 `iFSQ` 做真实量化，得到离散 latent `z_d`
6. 使用 dequantized continuous latent `z_dequant` 送入 `MLPDecoder`
7. 重建出 `[B, 10, D]`

其中 iFSQ 的前向使用真实离散化，反向使用 STE，因此梯度不会被 `round` 截断。

## 当前特征定义

单帧特征默认按下面顺序拼接：

1. `joint_pos`
2. `joint_vel`
3. `pelvis_quat_b`
4. `pelvis_lin_vel_b`
5. `pelvis_ang_vel_b`

这里的 `pelvis_*_b` 不是直接从 NPZ 里读出的 body frame 数据，而是由世界坐标系量通过 yaw-only anchor frame 变换得到。实现位于 [motion_ae/feature_builder.py](/home/humanoid/yzh/TextOp/motion_ae/motion_ae/feature_builder.py:1)。

默认约定：

- pelvis 使用 body 维度的第 `0` 个实体
- quaternion 格式默认是 `(w, x, y, z)`
- 输入 `motion.npz` 中 body 相关张量默认是世界坐标系下的量

如果你的数据格式不同，优先检查这些位置：

- [configs/default.yaml](/home/humanoid/yzh/TextOp/motion_ae/configs/default.yaml:10) 里的 `npz_keys`
- [configs/default.yaml](/home/humanoid/yzh/TextOp/motion_ae/configs/default.yaml:20) 里的 `pelvis.body_index`
- [motion_ae/feature_builder.py](/home/humanoid/yzh/TextOp/motion_ae/motion_ae/feature_builder.py:45) 里的 `_ensure_wxyz`
- [motion_ae/feature_builder.py](/home/humanoid/yzh/TextOp/motion_ae/motion_ae/feature_builder.py:58) 里的 `extract_pelvis_data`

## 目录结构

```text
motion_ae/
├── configs/
│   └── default.yaml
├── motion_ae/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluator.py
│   ├── feature_builder.py
│   ├── losses.py
│   ├── trainer.py
│   ├── models/
│   │   ├── autoencoder.py
│   │   ├── decoder.py
│   │   ├── encoder.py
│   │   └── ifsq.py
│   └── utils/
├── scripts/
│   ├── cli_args.py
│   ├── eval.sh
│   ├── evaluate.py
│   ├── infer.py
│   ├── train.py
│   └── train.sh
└── tests/
```

## 安装

建议使用 Python 3.10+ 和独立虚拟环境。

```bash
cd /home/humanoid/yzh/TextOp/motion_ae
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

如果你使用 conda，也可以：

```bash
conda activate text_tracker
cd /home/humanoid/yzh/TextOp/motion_ae
python -m pip install -r requirements.txt
```

## 数据格式

项目支持两种输入方式：

- 一个单独的 `motion.npz`
- 一个根目录，脚本会递归搜索其中所有名为 `motion.npz` 的文件

默认读取这些 key：

- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`
- `fps`

典型 shape 约定如下：

- `joint_pos`: `(T, J)`
- `joint_vel`: `(T, J)`
- `body_pos_w`: `(T, N, 3)`
- `body_quat_w`: `(T, N, 4)`
- `body_lin_vel_w`: `(T, N, 3)`
- `body_ang_vel_w`: `(T, N, 3)`

数据集构建逻辑位于 [motion_ae/dataset.py](/home/humanoid/yzh/TextOp/motion_ae/motion_ae/dataset.py:1)。

## 配置

主配置文件是 [configs/default.yaml](/home/humanoid/yzh/TextOp/motion_ae/configs/default.yaml:1)。

最常改的配置项有：

- `data.data_path`: 数据根目录或单个 NPZ 所在目录
- `data.val_ratio`: 验证集比例，按文件级划分
- `npz_keys.*`: NPZ 字段名映射
- `pelvis.body_index`: pelvis 在 body 维度中的索引
- `window_size`, `stride`: 滑窗参数
- `model.encoder_hidden_dims`, `model.decoder_hidden_dims`
- `model.ifsq_levels`: 同时决定 latent 维度
- `training.batch_size`, `training.learning_rate`, `training.num_epochs`
- `training.device`: `auto` / `cpu` / `cuda` / `cuda:0`
- `debug`: 打开后会打印 key 和 shape 等调试信息

## 训练

### 方式一：直接调用 Python

```bash
cd /home/humanoid/yzh/TextOp/motion_ae
python scripts/train.py \
  --config configs/default.yaml \
  --experiment_name motion_ae \
  --run_name baseline \
  --log_project_name motion_ae
```

### 方式二：使用 shell 包装脚本

```bash
cd /home/humanoid/yzh/TextOp/motion_ae
bash scripts/train.sh \
  --config configs/default.yaml \
  --experiment_name motion_ae \
  --run_name baseline
```

训练时会：

- 递归扫描数据并按文件级划分 train/val
- 从 train set 统计 mean/std
- 将归一化统计保存为 `stats.npz`
- 保存 `best_model.pt` 和 `last_checkpoint.pt`
- 按 `save_every` 周期保存额外 checkpoint

### 恢复训练

```bash
cd /home/humanoid/yzh/TextOp/motion_ae
python scripts/train.py \
  --config configs/default.yaml \
  --experiment_name motion_ae \
  --run_name baseline \
  --resume \
  --load_run 2026-04-14_12-00-00_baseline \
  --checkpoint last_checkpoint.pt
```

## 评估

```bash
cd /home/humanoid/yzh/TextOp/motion_ae
python scripts/evaluate.py \
  --config configs/default.yaml \
  --experiment_name motion_ae \
  --run_name 2026-04-14_12-00-00_baseline \
  --checkpoint best_model.pt \
  --split val
```

或者：

```bash
cd /home/humanoid/yzh/TextOp/motion_ae
bash scripts/eval.sh \
  --config configs/default.yaml \
  --experiment_name motion_ae \
  --run_name 2026-04-14_12-00-00_baseline \
  --checkpoint best_model.pt \
  --split val
```

评估结果会输出总 MSE 和各特征组的 MSE，并保存到：

```text
outputs/<experiment_name>/<run_name>/eval/<split>_<run_eval_name>.json
```

默认文件名通常是：

```text
val_eval.json
```

## 推理

对单个 `motion.npz` 做窗口级重建：

```bash
cd /home/humanoid/yzh/TextOp/motion_ae
python scripts/infer.py \
  --config configs/default.yaml \
  --experiment_name motion_ae \
  --run_name 2026-04-14_12-00-00_baseline \
  --checkpoint best_model.pt \
  --npz_path /path/to/motion.npz \
  --output infer_output.npz
```

输出文件会包含：

- `original`
- `reconstructed`
- `z_c`
- `z_d`
- `z_dequant`

这对于分析重建误差、量化行为和 latent 离散分布很方便。

## 输出目录

训练 run 的目录结构如下：

```text
outputs/
└── motion_ae/
    └── 2026-04-15_12-00-00_baseline/
        ├── artifacts/
        │   └── stats.npz
        ├── checkpoints/
        │   ├── best_model.pt
        │   ├── last_checkpoint.pt
        │   └── checkpoint_epochXXX.pt
        ├── eval/
        │   └── val_eval.json
        └── params/
            └── config.yaml
```

## 日志与 wandb

默认配置启用了 wandb。相关参数在：

- [configs/default.yaml](/home/humanoid/yzh/TextOp/motion_ae/configs/default.yaml:56)
- [scripts/train.sh](/home/humanoid/yzh/TextOp/motion_ae/scripts/train.sh:1)

如果你不想使用 wandb，可以在命令里显式关闭：

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --logger none
```

或者：

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint /abs/path/to/best_model.pt \
  --logger none
```

## 调试模式

打开 `--debug` 或在配置中设置 `debug: true` 后，数据管线会打印：

- NPZ 中所有 key
- 每个 key 的 shape
- 单帧特征维度
- 特征组切片
- 窗口 shape

这在排查字段名不一致、body 索引错误、quat 排列错误时很有用。

示例：

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --data_path /path/to/data \
  --debug \
  --logger none
```

## 测试

项目已经带了基础 pytest 覆盖，包括：

- 数据集滑窗构造
- iFSQ shape 与梯度传播
- 编码器 / 解码器 shape
- 训练 smoke test
- wandb 生命周期与评估脚本路径解析

运行方式：

```bash
cd /home/humanoid/yzh/TextOp/motion_ae
pytest
```

如果你只想先快速检查核心训练链路：

```bash
pytest tests/test_train_smoke.py -q
```

## 需要特别注意的约定

这个项目是“通用 motion.npz 自编码器”，为了兼容不同来源的数据，代码里故意保留了几个可修改点。真正上你的数据前，建议优先核对下面几项：

1. `joint_pos` / `joint_vel` 的列顺序是否已经与你要建模的关节顺序一致
2. `body_quat_w` 是否真的是 `(w, x, y, z)` 而不是 `(x, y, z, w)`
3. `pelvis.body_index` 是否确实对应根节点 body
4. `body_lin_vel_w` / `body_ang_vel_w` 是否真的是世界坐标系速度
5. 你的“anchor frame”定义是否与当前的 yaw-only 版本一致

如果这些前提不成立，模型仍然能训练，但学到的 latent 语义可能会偏掉，评估结果也会失真。

## 代码入口

如果你准备继续扩展，最值得先看的几个文件是：

- [motion_ae/models/autoencoder.py](/home/humanoid/yzh/TextOp/motion_ae/motion_ae/models/autoencoder.py:1)
- [motion_ae/models/ifsq.py](/home/humanoid/yzh/TextOp/motion_ae/motion_ae/models/ifsq.py:1)
- [motion_ae/feature_builder.py](/home/humanoid/yzh/TextOp/motion_ae/motion_ae/feature_builder.py:1)
- [motion_ae/dataset.py](/home/humanoid/yzh/TextOp/motion_ae/motion_ae/dataset.py:1)
- [motion_ae/trainer.py](/home/humanoid/yzh/TextOp/motion_ae/motion_ae/trainer.py:1)
- [scripts/cli_args.py](/home/humanoid/yzh/TextOp/motion_ae/scripts/cli_args.py:1)

## 一句话总结

如果你想快速上手，这个项目最常用的流程就是：

1. 在 `configs/default.yaml` 里改好数据路径和字段映射
2. 先用 `--debug` 跑一次，确认特征维度和 key 都对
3. 跑 `scripts/train.py` 训练
4. 用 `scripts/evaluate.py` 看重建误差
5. 用 `scripts/infer.py` 导出 `z_c / z_d / z_dequant` 做分析
