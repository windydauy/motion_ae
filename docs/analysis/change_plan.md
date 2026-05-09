1. 从motion_ae的z_c当中去进行rl tracker。
2. config参考 Tracking-Flat-G1-ProjGravAnchorObs-NMMLP-v0 只是把 command motion 修改为 motion_ae 当中的 z_c 然后同样去使用 further_step 来配置，以对其 motion_ae input 的windows dimension。其余奖励函数部分都不做改变。
3. 训练数据采用全部的 optitrack——clean
4. 给我修改好完整的代码，对其motion-ae 的接口。使用 这个 ckpt 为我做测试。
5. 整体思路就是reference motion-> motion encoder -> zd-> zc 在zc 这里进行 rl 训练一个 符合动力学的解码器，从discret latent zc 当中进行决策，而非直接给予 motion 本体进行决策。
6.请你为我修改好代码，进行smoke 测试，并且给完整的新的训练脚本。开启一次训练。 motion data 是：env 开启 8192 次 其余同 