好的，我已经完成了所有必要的调研。以下是基于数据格式、参考代码和 iFSQ 公式的完整头脑风暴计划。

---

## 项目总体计划：`motion_ae`（AutoEncoder + iFSQ）

---

### 一、数据分析总结

**NPZ 文件结构**（以 `WALK36` 为例）：

| 键名 | Shape | 含义 |
|---|---|---|
| `fps` | `(1,)` int64 | 帧率 |
| `joint_pos` | `(T, 29)` float32 | 29 个关节角度 |
| `joint_vel` | `(T, 29)` float32 | 29 个关节角速度 |
| `body_pos_w` | `(T, 37, 3)` float32 | 37 个刚体世界坐标位置 |
| `body_quat_w` | `(T, 37, 4)` float32 | 37 个刚体世界坐标四元数 |
| `body_lin_vel_w` | `(T, 37, 3)` float32 | 37 个刚体世界坐标线速度 |
| `body_ang_vel_w` | `(T, 37, 3)` float32 | 37 个刚体世界坐标角速度 |

数据集总量约 **2135 个** motion.npz，分布在 7 个子目录中。

---

### 二、坐标变换方案（世界系 → Anchor 系）

参考 `commands_multi.py` 和 `observations.py` 中的逻辑，我们需要将 pelvis（body[0]）的数据从**世界坐标系**转到**身体坐标系（robot anchor frame）**：

**核心思路**：anchor frame = pelvis 的 **yaw-only** 朝向帧（去除 pitch/roll，只保留航向角）

```python
anchor_yaw_quat = yaw_quat(pelvis_quat_w)

pelvis_quat_anchor_b = quat_mul(quat_inv(anchor_yaw_quat), pelvis_quat_w)
pelvis_rot6d_b = quat_to_rot6d(pelvis_quat_anchor_b)
pelvis_lin_vel_b = quat_apply(quat_inv(anchor_yaw_quat), pelvis_lin_vel_w)
pelvis_ang_vel_b = quat_apply(quat_inv(anchor_yaw_quat), pelvis_ang_vel_w)
```

其中 `yaw_quat()` 提取四元数的 yaw 分量，参考代码中已有现成实现。这样做的好处是：
- 特征对全局朝向不变
- 保留了 pelvis 的 pitch/roll 信息
- lin_vel / ang_vel 在局部坐标系下表达，物理含义更明确

**单帧特征维度**：
| 分量 | 维度 |
|---|---|
| `joint_pos` | 29 |
| `joint_vel` | 29 |
| `pelvis_rot6d_b` | 6 |
| `pelvis_lin_vel_b` | 3 |
| `pelvis_ang_vel_b` | 3 |
| **总计** | **70** |

---

### 三、iFSQ 实现方案

根据图片中的公式：

\[
z_d = \text{round}\left(\frac{L-1}{2}(f(z_c)+1)\right), \quad f(x)=2\sigma(1.6x)-1
\]

**四步流程**：

1. **Bounded mapping**：`f(z_c) = 2σ(1.6·z_c) - 1`，将连续 latent 映射到 `(-1, 1)`
2. **Scale to grid**：`(L-1)/2 · (f(z_c) + 1)` → 映射到 `[0, L-1]`
3. **Quantize**：`z_d = round(...)`，前向做真实离散化，反向用 STE
4. **Dequantize**：`z_dequant = 2·z_d/(L-1) - 1`，反映射回连续空间 `[-1, 1]`

**设计要点**：
- `ifsq_levels` 为 list，如 `[8, 8, 8, 8]`，每个元素对应一个 latent 维度的量化级别数
- `latent_dim` = `len(ifsq_levels)`
- 编码本大小 = `∏ levels`，如 `8^4 = 4096`
- Decoder 输入使用 dequantized continuous latent（而非 integer index）

---

### 四、模型架构设计

```
Input: [B, 10, 70]
       ↓ flatten
     [B, 700]
       ↓
  ┌─ MLP Encoder ─┐
  │ 700 → 512 → 256 → latent_dim │
  └────────────────┘
       ↓
  ┌─── iFSQ ────┐
  │ z_c → z_d → z_dequant │
  └──────────────┘
       ↓
  ┌─ MLP Decoder ─┐
  │ latent_dim → 256 → 512 → 700 │
  └────────────────┘
       ↓ reshape
Output: [B, 10, 70]
```

**默认超参数**：
- `window_size = 10`
- `stride = 1`
- `latent_dim = 8`（由 `ifsq_levels` 长度决定）
- `ifsq_levels = [8, 8, 8, 8, 8, 8, 8, 8]`
- `encoder_hidden_dims = [512, 256]`
- `decoder_hidden_dims = [256, 512]`
- `batch_size = 256`
- `learning_rate = 1e-3`
- `weight_decay = 1e-5`
- `num_epochs = 100`
- 激活函数: ReLU（简单稳定）
- 使用 LayerNorm（对 MLP 更友好）

---

### 五、训练流程设计

```
1. 扫描 npz → 计算全局 mean/std → 保存 stats.npz
2. 构建 Dataset（滑窗 stride=1）
3. 按 npz 文件级别做 train/val split（默认 90/10）
4. 训练循环：
   - normalize → encode → iFSQ → decode → denormalize → MSE loss
   - 每 epoch 保存 checkpoint + 评估 val loss
   - 保存 best model
5. 支持 resume from checkpoint
```

**关键设计决策**：
- **Split 策略**：按文件（而非窗口）划分，避免同一条 motion 的窗口同时出现在 train/val 中
- **归一化时机**：在 Dataset `__getitem__` 中做，保证全流程一致
- **std 保护**：`std = max(std, 1e-6)`

---

### 六、评估与推理

**评估输出**（按特征组）：
| 特征组 | 维度范围 | 含义 |
|---|---|---|
| `joint_pos` | `[0:29]` | 关节位置 MSE |
| `joint_vel` | `[29:58]` | 关节速度 MSE |
| `pelvis_rot6d_b` | `[58:64]` | 骨盆朝向 MSE |
| `pelvis_lin_vel_b` | `[64:67]` | 骨盆线速度 MSE |
| `pelvis_ang_vel_b` | `[67:70]` | 骨盆角速度 MSE |

**推理输出**（保存为 npz）：
- 原始窗口 `original`
- 重建窗口 `reconstructed`
- 连续 latent `z_c`
- 离散 latent `z_d`（integer indices）
- dequantized latent `z_dequant`

---

### 七、项目文件结构

```
motion_ae/
├── configs/
│   └── default.yaml
├── motion_ae/
│   ├── __init__.py
│   ├── config.py           # YAML 配置加载 + dataclass
│   ├── dataset.py           # Dataset + 滑窗构造
│   ├── feature_builder.py   # npz → 单帧特征拼接 + 坐标变换
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py       # MLP Encoder
│   │   ├── decoder.py       # MLP Decoder
│   │   ├── ifsq.py          # iFSQ 量化模块
│   │   └── autoencoder.py   # 组合模型
│   ├── losses.py            # 可扩展的重建损失
│   ├── trainer.py           # 训练逻辑
│   ├── evaluator.py         # 评估逻辑
│   └── utils/
│       ├── __init__.py
│       ├── io.py            # npz 加载 + 路径扫描
│       ├── seed.py          # 随机种子
│       ├── normalization.py # mean/std 统计与归一化
│       ├── logging.py       # 日志工具
│       ├── metrics.py       # 分组 MSE 计算
│       └── quaternion.py    # 四元数工具函数
├── train.py
├── evaluate.py
├── infer.py
├── tests/
│   ├── test_dataset.py
│   ├── test_ifsq.py
│   ├── test_model_shapes.py
│   └── test_train_smoke.py
├── pytest.ini
└── requirements.txt
```

> 注：相比原 spec 多了一个 `quaternion.py`，用于存放 yaw_quat / quat_mul / quat_apply / quat_inv 等四元数工具。

---

### 八、需要你确认/注意的点

1. **Pelvis = body[0]**：默认取 `body_*_w[:, 0, :]` 作为 pelvis，如果你的数据里 pelvis 不是第一个 body，需要改配置中的 `pelvis_body_index`。

2. **Anchor frame 的定义**：我计划用 **yaw-only** 的 pelvis 朝向作为 anchor。这意味着 `pelvis_rot6d_b` 表达去除 yaw 后的 pelvis 朝向，lin_vel/ang_vel 在航向对齐的局部坐标系下表达。这与参考代码的做法一致。

3. **joint_pos 的含义**：npz 中 `joint_pos` shape 为 `(T, 29)`，看起来已经是关节角度（1D per joint），而不是 3D 位置。如果实际是 3D 坐标需要调整。

4. **不包含 `body_pos_w`**：根据需求说明，单帧特征中不包含 pelvis 的世界位置（因为在 anchor frame 下位置为零），如果需要添加 pelvis 的高度或相对位移，可后续扩展。

5. **iFSQ levels 的选择**：默认 `[8]*8` 给出 `8^8 ≈ 1.6M` 的编码本，这对于 70 维输入可能过大或过小。你可以根据实验调整，例如 `[5]*4` → 625 个 code，或 `[8]*6` → 262K。

---

### 九、实施阶段划分

| 阶段 | 内容 | 预计文件数 |
|---|---|---|
| **P1** | 基础设施：config, io, seed, quaternion, normalization | 6 |
| **P2** | 数据管线：feature_builder, dataset | 2 |
| **P3** | 模型：encoder, decoder, ifsq, autoencoder, losses | 5 |
| **P4** | 训练/评估/推理：trainer, evaluator, train.py, evaluate.py, infer.py | 5 |
| **P5** | 测试 + 配置文件 | 5 |
| **P6** | Debug 模式 + 日志 + metrics | 3 |

---

以上就是完整的计划方案。如果你确认没有问题（特别是坐标变换方案和特征构成），我可以立即开始按阶段生成完整代码。有什么需要调整的地方吗？
