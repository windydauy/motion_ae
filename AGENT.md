请帮我在当前目录下创建一个完整的 Python 项目，项目名为 `motion_ae`。本项目实际上实现的是 **AutoEncoder + iFSQ**，**不使用任何 VAE KL loss**，只使用重建损失。

### 项目目标

对 motion npz 数据做一个基于窗口的运动编码与重建系统，结构为：

* Encoder: 输入 `m_{t:t+9}`，输出连续 latent `z_c`
* iFSQ: 对 `z_c` 做有限标量量化，得到离散 latent `z_d`
* Decoder: 输入量化后的 latent（建议使用 dequantized continuous representation），输出重建的 `\hat{m}_{t:t+9}`

训练目标是仅使用标准重建损失监督 encoder 和 decoder，不使用任何 VAE 损失。

---

### 数据说明

输入数据来自类似如下路径的 npz 文件：

`/root/yzh/optitrack_npz/converted_1031/BOXING1_Skeleton 004_z_up_x_forward_gym/motion.npz`

你可以先解析一下每个npz有哪些内容，请把数据加载逻辑设计成既支持单个 npz 文件，也支持递归遍历某个根目录下所有 `motion.npz` 文件。

每个时间步的输入特征 `m_t` 由以下部分拼接构成：

* `joint_pos`
* `joint_vel`
* pelvis (body 根节点) 的 `quat` , 是在 robot anchor frame（身体坐标系） 下表示的，不是世界坐标系
* pelvis (body 根节点) 的 `lin_vel`, 是在 robot anchor frame（身体坐标系） 下表示的，不是世界坐标系
* pelvis (body 根节点) 的 `ang_vel`, 是在 robot anchor frame（身体坐标系） 下表示的，不是世界坐标系

请先将这些字段从 npz 中解析出来，(npz 中包含这些键值，默认 body的第一个就是pelvis，默认均为全局坐标系下，你需要参考/root/yzh/TextOp/TextOpTracker/source/textop_tracker 这个文件夹中的有关坐标变换处理)然后拼接为单帧特征。再使用长度为 10 的滑动窗口构造 `m_{t:t+9}`。

请在代码中把“字段名映射”和“pelvis 的提取逻辑”写成可配置项，便于我后续按真实 npz 格式修改。

---

### 滑窗与样本构造

* 窗口长度固定为 `10`
* 默认 stride = `1`
* 输入为 `m_{t:t+9}`
* target 为同一个窗口本身
* 长度不足 10 的尾部样本直接丢弃

---

### 模型要求

请优先实现一个**简单、稳定、容易训练**的版本，第一版使用 **MLP Encoder + iFSQ + MLP Decoder**，不要上 Transformer。

建议默认设计为：

* 输入 shape: `[B, 10, D]`
* 先 flatten 为 `[B, 10*D]`
* Encoder 输出连续 latent `z_c`
* iFSQ 输出离散 latent `z_d`
* Decoder 从量化 latent 重建出 `[B, 10, D]`

请将以下超参数写入配置文件，并提供合理默认值：

* `window_size`
* `stride`
* `latent_dim`
* `ifsq_levels`
* `encoder_hidden_dims`
* `decoder_hidden_dims`
* `batch_size`
* `learning_rate`
* `weight_decay`
* `num_epochs`
* `num_workers`
* `seed`

---

### iFSQ 细节

请严格按如下要求实现 iFSQ：

1. Encoder 先输出连续 latent `z_c`
2. iFSQ 对 `z_c` 做真实量化
3. 前向必须使用真实离散化
4. 反向使用 straight-through estimator (STE)，把 `round` 当作恒等映射传梯度
5. Decoder 的输入建议使用量化后的 dequantized continuous latent，而不是纯 integer index
6. 请把 iFSQ 实现单独放到模块里，写清注释，并尽量让代码便于后续替换 quantizer

如果需要，请在代码中预留位置让我手动填入我图片中的 iFSQ 公式；同时在实现里提供一个默认版本：

* bounded mapping
* quantization
* STE
* dequantization

---

### 损失函数

* 只使用标准重建损失
* 默认使用 `MSELoss`
* 不使用 KL loss
* 不使用任何 VAE 相关项
* 请把损失设计成后续可扩展到分组加权（joint_pos / joint_vel / pelvis_xxx），但第一版默认统一权重

---

### 归一化与统计

请实现特征归一化：

* 从 train set 统计每个 feature 维度的 mean/std
* 保存到文件，例如 `stats.npz`
* train / eval / inference 必须复用同一套统计量
* 避免除零，std 过小时做保护

---

### 训练与评估

请实现完整训练与评估流程，包括：

1. `train.py`

   * 训练主入口
   * 保存 checkpoint
   * 保存 best model
   * 输出 train / val loss
   * 支持从 checkpoint 恢复训练

2. `evaluate.py`

   * 在验证集或测试集上评估重建效果
   * 输出总 MSE
   * 输出各特征组的 MSE：

     * joint_pos
     * joint_vel
     * pelvis_pos_w
     * pelvis_quat_w
     * pelvis_lin_vel
     * pelvis_ang_vel

3. `infer.py`

   * 对单个 npz 做重建推理
   * 保存原始窗口、重建窗口、latent、量化 index 等结果，方便分析

---

### 工程结构

请在当前目录下创建如下项目结构，并补齐完整代码：

`motion_vae/`

* `configs/`
* `motion_vae/`

  * `__init__.py`
  * `config.py`
  * `dataset.py`
  * `feature_builder.py`
  * `models/`

    * `__init__.py`
    * `encoder.py`
    * `decoder.py`
    * `ifsq.py`
    * `autoencoder.py`
  * `losses.py`
  * `trainer.py`
  * `evaluator.py`
  * `utils/`

    * `__init__.py`
    * `io.py`
    * `seed.py`
    * `normalization.py`
    * `logging.py`
    * `metrics.py`
* `train.py`
* `evaluate.py`
* `infer.py`
* `tests/`

  * `test_dataset.py`
  * `test_ifsq.py`
  * `test_model_shapes.py`
  * `test_train_smoke.py`
* `pytest.ini`
* `requirements.txt`
* `README.md`

---

### 测试要求

请补充 pytest 测试，至少覆盖：

* dataset 能正确从 npz 构造窗口
* iFSQ 前向 shape 正确
* iFSQ 的 STE 不会让梯度断掉
* encoder / decoder 输入输出 shape 正确
* 一个最小样例的训练 smoke test 可以跑通 1~2 step

---

### 代码风格要求

* 使用 Python 3.10+
* 使用 PyTorch
* 添加充分注释
* 对关键 shape 做断言
* 尽量写清楚每个 tensor 的 shape
* 不要只给片段代码，要给完整可运行项目
* 优先保证可读性和可维护性，而不是过度抽象
* README 里写清楚如何安装、训练、评估、测试

---

### 额外要求

1. 请先默认实现一个通用版本，不要假设我当前 npz 的键名百分百固定，把键名映射做成配置。
2. 对 pelvis 的提取逻辑写成一个独立函数，后续我可以按 body 名称或 index 修改。
3. 对 quaternion 部分，先默认直接使用 4D quaternion。
4. 输出一个 `debug` 模式，能打印：

   * npz 中所有键名
   * 每个键的 shape
   * 单帧特征维度
   * 窗口 shape
5. 如果某些 npz 字段名存在歧义，请在代码注释和 README 中明确指出我需要手动核对的位置。

请直接生成完整项目代码，而不是只给方案。
onda activate text_tracker
cd /root/yzh/motion_ae
python scripts/train.py \
  --config configs/default.yaml \
  --experiment_name motion_ae \
  --run_name baseline \
  --log_project_name motion_ae
python scripts/evaluate.py \
  --config configs/default.yaml \
  --experiment_name motion_ae \
  --run_name 2026-04-14_12-00-00_baseline \
  --checkpoint best_model.pt
python scripts/train.py \
  --config configs/default.yaml \
  --experiment_name motion_ae \
  --run_name baseline \
  --resume \
  --load_run 2026-04-14_12-00-00_baseline \
  --checkpoint last_checkpoint.pt