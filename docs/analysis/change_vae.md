1. 我现在想把刚刚你分析过的 transformer motion vae 迁移到本目录下，主要model部分写在 /pfs/pfs-ilWc5D/yzh/motion_ae/transformer_vae
2. 使用 /pfs/pfs-ilWc5D/yzh/g1_soma/npz_part 中同款数据，motion_ae 中的feature 构造方式如下： 单帧特征默认按下面顺序拼接：
`joint_pos`
`joint_vel`
`pelvis_rot6d_b`
`pelvis_lin_vel_b`
`pelvis_ang_vel_b`

这里的 `pelvis_*_b` 不是直接从 NPZ 里读出的 body frame 数据，而是由世界坐标系量通过 yaw-only anchor frame 变换得到。实现位于 [motion_ae/feature_builder.py]。
`pelvis_rot6d_b` 使用旋转矩阵前两列拼接的 6D 表示。
3. 完善完整的motion transformer vae 的训练 inference 和 eval 流程，全部都参考 /pfs/pfs-ilWc5D/yzh/motion_ae/docs/motion_vae_TextOpRobotMDAR.md 这个文档里总结的部分。
4. 一句话总结：使用 /pfs/pfs-ilWc5D/yzh/motion_ae/motion_ae 的数据读取和特征构造方式，和 /pfs/pfs-ilWc5D/yzh/TextOp/TextOpRobotMDAR/robotmdar 中的模型架构 训练推理验证方式。来构造一个新的 motion transformer vae
5. 为我进行测试，并且给出新的环境安装建议。我希望在这个环境下motion ae 和 transformer vae 都能训练， transformer 相关环境 参考 /pfs/pfs-ilWc5D/yzh/TextOp/TextOpRobotMDAR/pyproject.toml，进行测试