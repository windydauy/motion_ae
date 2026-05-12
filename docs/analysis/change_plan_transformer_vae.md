1. 从motion_transformer_vae的latent_space当中去进行rl tracker。 motion_transformer_vae 的相关路径：/home/humanoid/yzh/TextOp/motion_ae/transformer_vae,他的input feature 也是
`joint_pos`
`joint_vel`
`pelvis_rot6d_b`
`pelvis_lin_vel_b`
`pelvis_ang_vel_b`
2. config参考 Tracking-Flat-G1-ProjGravAnchorObs-NMMLP-v0 只是把 command motion 修改为 motion_transformer_vae 当中的 latent_space 然后同样去使用 further_step 来配置，以对其 motion_transformer_vae input 的windows dimension。其余奖励函数部分都不做改变。
3. 训练数据采用全部的 /home/humanoid/yzh/TextOp/trip_npz_filtered
4. 给我修改好完整的代码，对齐 motion_transformer_vae的接口。使用 这个 ckpt ：/home/humanoid/yzh/TextOp/motion_ae/outputs/transformer_vae/2026-05-11_00-19-00_optitrack_npz_trip_filtered/checkpoints/best_model.pt 为我做测试。
5. 整体思路就是reference motion-> motion_transformer_vae -> latent_space 在latent_space(z_c) 这里进行 rl 训练一个 符合动力学的解码器，从discret latent z_c 当中进行决策，而非直接给予 motion 本体进行决策。
6.请你为我修改好代码，进行smoke 测试，并且给完整的新的训练脚本。开启一次训练。 motion data 是：env 开启 8192 次 其余同  /home/humanoid/yzh/TextOp/scripts/train.sh