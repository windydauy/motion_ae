# TextOpRobotMDAR Motion VAE 模块说明

本文档总结 `/pfs/pfs-ilWc5D/yzh/TextOp/TextOpRobotMDAR` 中的 Motion VAE，也就是 `AutoMldVae` / MVAE 训练与推理链路。结论基于当前代码读取结果；当前默认配置是 G1 23-DoF 机器人、`FeatureVersion = 3`、`nfeats = 57`、2 帧 history 预测 8 帧 future。

## 1. 代码入口与文件索引

核心文件：

- `robotmdar/model/mld_vae.py`：VAE 网络主体 `AutoMldVae`，包含 `encode()` / `decode()`。
- `robotmdar/model/operator/cross_attention.py`：Skip Transformer Encoder/Decoder 与基础 Transformer layer。
- `robotmdar/model/operator/position_encoding.py`：1D sine/learned 位置编码。
- `robotmdar/dataloader/data.py`：`SkeletonPrimitiveDataset`，负责 motion primitive 采样、特征转换、归一化、重建。
- `robotmdar/dtype/motion.py`：motion feature 表示与 `MotionDict <-> feature tensor` 转换。
- `robotmdar/train/train_mvae.py`：MVAE 训练主循环。
- `robotmdar/train/manager.py`：`MVAEManager`，负责训练状态、loss、评估、保存/加载 checkpoint。
- `robotmdar/eval/vis_mvae.py`：MVAE 重构可视化推理。
- `robotmdar/wrapper/vae_decode.py`：仅导出/包装 decoder 的 `DecoderWrapper`。
- `robotmdar/cli.py`：Hydra CLI 路由，`train-mvae` 与 `vis-mvae` 都从这里分发。

默认配置：

- `robotmdar/config/train_mvae.yaml`：设置 `task: train-mvae`。
- `robotmdar/config/vis_mvae.yaml`：设置 `task: vis-mvae`。
- `robotmdar/config/vae/def.yaml`：VAE 网络超参。
- `robotmdar/config/train/mvae.yaml`：训练 manager、优化器、loss 权重。
- `robotmdar/config/data/babel.yaml`：数据集、batch、history/future/primitive 长度。

## 2. 模块定位

Motion VAE 在整个 RobotMDAR 中有两个角色：

1. 作为独立自编码器训练：输入真实 history + future，编码 future 条件分布，再解码重构 future。
2. 作为 DAR 扩散模型的 motion latent autoencoder：DAR 训练/推理时扩散模型预测 latent，VAE decoder 将 latent + history 解码成未来 motion feature。

MVAE 本身不使用文本条件。数据集返回 `(motion, cond)`，其中 `cond` 是 CLIP text embedding；在 `train_mvae.py` 中会被移动到 device，但没有送入 VAE。文本条件主要服务于 DAR denoiser。

## 3. 数据表示

### 3.1 原始 MotionDict

`robotmdar/dtype/motion.py` 中定义的原始运动字典包含：

- `root_trans_offset`: root 平移，最后维度 3。
- `root_rot`: root 四元数，xyzw，最后维度 4。
- `dof`: 23 个机器人关节自由度。
- `contact_mask`: 双脚接触 mask，最后维度 2。

当前 `FeatureVersion = 3`。数据配置中 `nfeats = 57`，来自：

```text
4   roll/pitch 的 sin/cos 表示
1   delta yaw
2   foot contact mask
3   当前 yaw 坐标下的 root delta translation
1   root height
23  dof position
23  delta dof
= 57
```

`motion_dict_to_feature_v3()` 输入 `T+1` 帧 `MotionDict`，输出 `T` 帧 feature。这样每一帧 feature 中可以携带从当前帧到下一帧的 delta 信息。反向的 `motion_feature_to_dict_v3()` 需要 `abs_pose` 作为第一帧绝对位姿，用 delta yaw / delta translation 累积回世界位姿。

### 3.2 数据集切片

`SkeletonPrimitiveDataset` 是 `IterableDataset`。默认配置：

```yaml
nfeats: 57
history_len: 2
future_len: 8
num_primitive: 8
batch_size: 512
```

因此一个训练样本段长度是：

```text
segment_len = history_len + future_len * num_primitive + 1
            = 2 + 8 * 8 + 1
            = 67 原始帧
```

每个 primitive 的原始抽取区间是：

```text
prim_start = seg_start + primitive_idx * future_len
prim_end   = prim_start + future_len + history_len + 1
```

经过 `motion_dict_to_feature_v3()` 后，每个 primitive 得到 `history_len + future_len = 10` 帧 feature。训练时再拆成：

- `history_motion = motion[:, :2, :]`
- `future_motion_gt = motion[:, -8:, :]`

相邻 primitive 以 `future_len` 为 stride，因此 primitive 之间天然连续；第 i 个 primitive 的后续帧会覆盖第 i+1 个 primitive 的 history。

### 3.3 归一化与重建

数据集使用 `meanstd.pkl` 做特征归一化：

```text
normalized = (feat - mean) / std
denormalized = feat * std + mean
```

如果缓存不存在，会在 train split 上采样约 10000 条 batch primitive 统计 mean/std。训练、VAE forward、loss 的直接输入输出都是归一化后的 feature；几何 loss 和可视化会通过 `reconstruct_motion(..., need_denormalize=True)` 反归一化并做 forward kinematics。

## 4. VAE 网络结构

### 4.1 默认超参

默认 `robotmdar/config/vae/def.yaml`：

```yaml
_target_: robotmdar.model.mld_vae.AutoMldVae
nfeats: ${..nfeats}       # 当前为 57
latent_dim: [1, 128]      # latent_size=1, latent_dim=128
h_dim: 512
ff_size: 1024
num_layers: 9
num_heads: 4
dropout: 0.1
arch: all_encoder
normalize_before: false
activation: gelu
position_embedding: learned
```

关键层：

- `skel_embedding`: `Linear(nfeats, h_dim)`，把每帧 motion feature 投影到 Transformer hidden dim。
- `global_motion_token`: 可学习 token，形状 `[2 * latent_size, h_dim]`。前半部分预测 `mu`，后半部分预测 `logvar`。
- `encoder`: `SkipTransformerEncoder`，9 层，带 U-Net 式 skip。
- `encoder_latent_proj`: `Linear(h_dim, latent_dim)`。
- `decoder_latent_proj`: `Linear(latent_dim, h_dim)`。
- `decoder`: 默认也是 `SkipTransformerEncoder`，因为 `arch: all_encoder`。
- `final_layer`: `Linear(h_dim, nfeats)`，输出 future motion feature。
- `latent_mean` / `latent_std`: buffer，当前初始化为 0/1；DAR 加载 VAE 时会兼容 checkpoint 中是否存在这些 buffer。

### 4.2 Skip Transformer

`SkipTransformerEncoder` 要求 `num_layers` 为奇数。以默认 9 层为例：

```text
input_blocks: 4 层 encoder layer
middle_block: 1 层 encoder layer
output_blocks: 4 层 encoder layer
```

forward 时会保存前半部分 block 的输出，后半部分每层把当前 hidden 与对应 skip hidden 在最后维拼接，再用 `Linear(2*h_dim, h_dim)` 压回 hidden dim。每个 `TransformerEncoderLayer` 是标准 self-attention + FFN + residual + LayerNorm，attention head 为 4，FFN hidden 为 1024，激活为 GELU。

### 4.3 Encoder 流程

`encode(future_motion, history_motion)` 的输入：

```text
history_motion: [B, H=2, D=57]
future_motion:  [B, F=8, D=57]
```

流程：

```text
1. 拼接输入
   x = concat(history_motion, future_motion, dim=1)
   x: [B, H+F=10, D]

2. 帧级投影
   x = skel_embedding(x)
   x: [B, 10, 512]

3. Transformer 维度转换
   x = x.permute(1, 0, 2)
   x: [10, B, 512]

4. 准备分布 token
   dist_token = tile(global_motion_token[:, None, :])
   dist_token: [2*latent_size=2, B, 512]

5. token 拼接与位置编码
   xseq = concat(dist_token, x, dim=0)
   xseq: [12, B, 512]

6. SkipTransformerEncoder
   encoded = encoder(query_pos_encoder(xseq))
   dist_hidden = encoded[:2]
   dist_hidden: [2, B, 512]

7. 投影到 latent distribution
   dist = encoder_latent_proj(dist_hidden)
   dist: [2, B, 128]

8. 拆分参数
   mu     = dist[:latent_size]     -> [1, B, 128]
   logvar = dist[latent_size:]     -> [1, B, 128]
   logvar clamp 到 [-10, 10]

9. 重参数采样
   std = exp(logvar) ** 0.5
   q(z|history,future) = Normal(mu, std)
   z = q.rsample()
   z: [1, B, 128]
```

`scale_latent=True` 时会执行 `z = z / latent_std`。当前 MVAE 独立训练没有打开；注释说明它主要用于后续 denoiser 训练。

### 4.4 Decoder 流程

`decode(z, history_motion, nfuture)` 的输入：

```text
z:              [latent_size=1, B, latent_dim=128]
history_motion: [B, H=2, D=57]
nfuture:        8
```

默认 `arch = all_encoder` 时：

```text
1. latent 投影
   z = decoder_latent_proj(z)
   z: [1, B, 512]

2. future query 初始化为全 0
   queries: [F=8, B, 512]

3. history 投影
   history_embedding = skel_embedding(history_motion).permute(1, 0, 2)
   history_embedding: [2, B, 512]

4. 拼接 decoder 序列
   xseq = concat(z, history_embedding, queries, dim=0)
   xseq: [1+2+8=11, B, 512]

5. 位置编码 + SkipTransformerEncoder
   output = decoder(query_pos_decoder(xseq))[-nfuture:]
   output: [8, B, 512]

6. 输出投影
   output = final_layer(output)
   output: [8, B, 57]

7. 维度还原
   future_motion_pred = output.permute(1, 0, 2)
   future_motion_pred: [B, 8, 57]
```

代码还支持 `arch = encoder_decoder`：decoder 输入为 `history_embedding + queries`，latent `z` 作为 memory 走 cross-attention。但当前默认不是这个分支。

### 4.5 Patcher 分支

`AutoMldVae` 支持 `use_patcher=True`，通过 `Patcher1D`/`UnPatcher1D` 做 1D Haar 或 rearrange patching；默认配置未启用。需要注意当前 `decode()` 末尾在 `use_patcher` 分支中对 `output` 而不是最终 `feats` 做 inverse transform，且返回仍是 `feats`，这个分支在当前默认设置下不会触发，但如果未来启用 patcher，需要重新验证维度和返回值。

## 5. 训练流程

### 5.1 CLI 与 Hydra

`robotmdar/cli.py` 通过 `task` 路由：

- `train-mvae` -> `robotmdar.train.train_mvae.main`
- `vis-mvae` -> `robotmdar.eval.vis_mvae.main`

典型训练命令：

```bash
cd /pfs/pfs-ilWc5D/yzh/TextOp/TextOpRobotMDAR
robotmdar --config-name train_mvae
```

或者：

```bash
python -m robotmdar.cli --config-name train_mvae
```

常用 override 示例：

```bash
robotmdar --config-name train_mvae \
  device=cuda \
  data.batch_size=256 \
  train.manager.learning_rate=1e-4 \
  ckpt.vae=/path/to/ckpt_80000.pth
```

日志与 checkpoint 默认保存到：

```text
logs/RobotMDAR/VAE/train-mvae-<timestamp>/
```

这是由 `base.yaml` 中 `experiment_dir` 和 `train_mvae.yaml` 中 `expname: VAE`、`task: train-mvae` 组合出来的。

### 5.2 训练主循环

训练初始化：

```text
train_data = instantiate(cfg.data.train)
val_data   = instantiate(cfg.data.val)
vae        = instantiate(cfg.vae)
optimizer  = Adam(vae.parameters(), lr=manager.learning_rate)
manager    = MVAEManager(...)
manager.hold_model(vae, optimizer, train_data)
```

每个 outer loop 从数据集取一个 batch。这个 batch 是长度为 `num_primitive=8` 的列表，每个元素是一个 primitive 的 `(motion, cond)`。训练时按 primitive 顺序循环：

```text
for pidx in range(num_primitive):
    motion: [B, H+F, D] = [B, 10, 57]
    future_motion_gt = motion[:, -8:, :]
    gt_history       = motion[:, :2, :]

    history_motion = manager.choose_history(gt_history, prev_motion, history_len)

    latent, dist = vae.encode(future_motion_gt, history_motion)
    future_motion_pred = vae.decode(latent, history_motion, nfuture=8)

    loss_dict, extras = manager.calc_loss(...)
    loss.backward()
    检查梯度 NaN/Inf
    grad clip
    optimizer.step()

    prev_motion = future_motion_pred.detach()
```

默认 `use_rollout: False`，所以 `choose_history()` 仍使用 ground-truth history。若打开 rollout，则从 stage 1 开始按线性升高概率使用上一 primitive 的预测尾部作为下一个 history。`use_static_pose: True` 且 `static_prob: 0.001`，从后续 stage 开始小概率把 history 替换成静态 zero pose，再归一化后输入模型。

### 5.3 训练阶段与学习率

默认 `stages`：

```yaml
stages: [15000, 25000, 30000, 30000]
```

总训练步数为四段累加：

```text
max_steps = 100000
```

`BaseManager.pre_step()` 每步根据 `step / max_steps` 线性衰减学习率：

```text
lr_now = (1 - step / max_steps) * learning_rate
```

默认：

- `learning_rate: 1e-4`
- `anneal_lr: True`
- `max_grad_norm: 1.0`
- `save_every: 20000`
- `eval_every: 2000`
- `eval_steps: 10`
- `use_ema: False`

### 5.4 Loss 组成

`MVAEManager.calc_loss()` 的基础项：

```text
rec = HuberLoss(future_motion_pred, future_motion_gt)
kl  = mean(KL(Normal(mu, std) || Normal(0, 1)))
```

当前 `FeatureVersion = 3`，所以走通用 `calc_geometry_loss()`：

1. 将预测和 GT 的 normalized future feature 反归一化并重建 motion。
2. 通过 robot skeleton forward kinematics 得到全局关节/旋转/速度等几何量。
3. 计算：
   - `body_trans`: FK 后扩展 body 全局平移 Huber loss。
   - `body_rot`: FK 后全局旋转 Huber loss。
   - `dof_pos`: dof 位置 Huber loss。
   - `dof_vel`: dof velocity Huber loss。
   - `foot_contact`: 只在 GT foot contact mask 为 1 的脚部位置 Huber loss。
   - `smooth`、`quantize_*`、`drift_*` 是可配置项；当前权重为 0 时不进 total，但 `drift_xy`/`drift_yaw` 会作为 extras 记录。

默认 loss 权重：

```text
total =
  1.0   * rec
+ 1e-4  * kl
+ 0.05  * body_trans
+ 1e-2  * body_rot
+ 0.03  * dof_pos
+ 1e-5  * dof_vel
+ 0.01  * foot_contact
```

配置中还保留了 `FeatureVersion == 4/5` 对应的 `fk_joints_rec`、`joints_consistency`、`*_delta` 等权重，但当前 v3 默认不会使用这些项。

### 5.5 Checkpoint

`MVAEManager.save_model()` 保存：

```python
{
    "vae": vae.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
    # 可选: "ema_models"
}
```

文件名为：

```text
ckpt_<step>.pth
```

加载时如果配置 `ckpt.vae` 不为空，会恢复 VAE、optimizer 和 step。独立 `vis-mvae` 中调用 `manager.hold_model(vae, None, val_data)`，如果指定了 `ckpt.vae`，也会只加载 VAE state；因为 optimizer 为 `None`，不会恢复 optimizer。

## 6. 推理流程

### 6.1 MVAE 自重构推理

`vis_mvae.py` 做的是 validation batch 的 teacher-forcing 重构可视化，并不从随机先验无条件采样。流程：

```text
1. val_data = instantiate(cfg.data.val)
2. vae = instantiate(cfg.vae)
3. manager.hold_model(vae, None, val_data)  # 通过 ckpt.vae 加载权重
4. 对每个 primitive:
   future_motion_gt = motion[:, -future_len:, :]
   history_motion   = motion[:, :history_len, :]
   latent, dist = vae.encode(future_motion_gt, history_motion)
   future_motion_pred = vae.decode(latent, history_motion, nfuture=future_len)
   pred_dict = val_data.reconstruct_motion(cat(history, pred), abs_pose=pd_abs_pose, ret_fk=False)
   gt_dict   = val_data.reconstruct_motion(cat(history, gt),   abs_pose=gt_abs_pose, ret_fk=False)
   更新 abs_pose，转换成 MuJoCo qpos，可视化 pred/gt
```

典型命令：

```bash
cd /pfs/pfs-ilWc5D/yzh/TextOp/TextOpRobotMDAR
robotmdar --config-name vis_mvae ckpt.vae=/path/to/ckpt_100000.pth
```

`vis_mvae.yaml` 默认把 batch size 改成 16，并将训练平台设为 `NoPlatform`。

### 6.2 作为 DAR decoder 使用

DAR 训练中，VAE 先被加载并冻结。当前逻辑：

```text
latent_gt = vae.encode(future_motion_gt, history_motion)[0]  # [1, B, 128]
x_start = latent_gt.permute(1, 0, 2)                         # [B, 1, 128]
denoiser 学习扩散/去噪 latent
latent_pred = x_start_pred.permute(1, 0, 2)                  # [1, B, 128]
future_motion_pred = vae.decode(latent_pred, history_motion, nfuture=8)
```

DAR 推理中，denoiser 根据文本 embedding 和 normalized history 生成 latent，VAE decoder 负责把 latent 还原到 normalized future feature，再通过 dataset 重建成 motion dict / qpos。

### 6.3 DecoderWrapper

`robotmdar/wrapper/vae_decode.py` 抽出 decoder 相关子模块：

- `decoder`
- `decoder_latent_proj`
- `skel_embedding`
- `final_layer`
- `query_pos_decoder`

`forward(z, history_motion)` 固定 `nfuture = 8`，输出 `[B, 8, D]`。这个 wrapper 适合只部署/导出 decoder，但当前代码中它没有处理 `scale_latent`，也不包含 encoder。

## 7. 关键张量形状速查

默认设置下：

| 名称 | 形状 | 说明 |
|---|---:|---|
| `motion` | `[B, 10, 57]` | 一个 primitive 的 normalized feature |
| `history_motion` | `[B, 2, 57]` | 条件历史 |
| `future_motion_gt` | `[B, 8, 57]` | 监督未来 |
| `xseq` encoder 输入 | `[12, B, 512]` | 2 个分布 token + 10 帧 motion token |
| `mu/logvar` | `[1, B, 128]` | latent distribution |
| `latent` | `[1, B, 128]` | VAE sampled latent |
| `decoder xseq` | `[11, B, 512]` | 1 latent token + 2 history token + 8 query token |
| `future_motion_pred` | `[B, 8, 57]` | decoder 输出 |
| DAR latent layout | `[B, 1, 128]` | denoiser 使用 batch-first，需要和 VAE layout 互相 permute |

## 8. 实现注意点

1. MVAE 训练中 `cond` 未使用；如果要做 text-conditioned VAE，需要改 `AutoMldVae` 和训练循环。
2. `AutoMldVae.forward(self, z, history_motion)` 调用 `decode(z, history_motion)`，但 `decode()` 需要 `nfuture` 参数；直接调用 `vae(z, history)` 会报缺参。当前训练/推理都显式调用 `decode(..., nfuture=future_len)`，所以主路径不受影响。
3. 默认 `arch=all_encoder`，decoder 实际是一个 encoder-style self-attention 序列模型，不是 PyTorch 意义上的 encoder-decoder cross-attention。`encoder_decoder` 分支存在，但默认未用。
4. `use_patcher` 默认关闭；若打开，需要额外验证输入 layout、inverse patching 和返回值。
5. `FeatureVersion` 在 `motion.py` 中是全局常量。改 feature version 时，需要同步 `data.nfeats`、loss 分支、zero feature、重建逻辑和已有 checkpoint。
6. VAE 输出仍是 normalized motion feature。任何几何评估、MuJoCo 可视化或部署给控制器前，都要通过 dataset 的 `denormalize` 和 motion feature 反变换链路。

## 9. 一句话总览

当前 Motion VAE 是一个 MLD 风格的 Transformer VAE：用 2 个 learnable distribution token 从 `history + future` 序列中读出 `mu/logvar`，采样 1 个 128 维 latent token，再用 `latent + history + future query` 通过 Skip Transformer 生成 8 帧 normalized robot motion feature；训练时用 feature 重构、KL 和 FK 几何约束共同监督，推理时既可做 validation 重构可视化，也可作为 DAR 文本扩散模型的冻结 decoder。
