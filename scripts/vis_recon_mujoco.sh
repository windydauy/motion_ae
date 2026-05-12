
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python -m scripts.visualize_recon_mujoco \
  --model_type ae \
  --config /home/humanoid/yzh/TextOp/motion_ae/outputs/motion_ae/2026-05-08_20-28-41_opti_clean_our_20/params/config.yaml \
  --checkpoint outputs/motion_ae/2026-05-08_20-28-41_opti_clean_our_20/checkpoints/last_checkpoint.pt \
  --npz_path "/home/humanoid/yzh/TextOp/optritrack_dataset/optitrack_npz_filtered_all/balance_014_Skeleton 006_z_up_x_forward_gym/motion.npz"

# python -m scripts.visualize_recon_mujoco \
#   --model_type vae \
#   --vae_config configs/transformer_vae.yaml \
#   --checkpoint outputs/transformer_vae/2026-05-11_00-19-00_optitrack_npz_trip_filtered/checkpoints/best_model.pt \
#   --npz_path "/home/humanoid/yzh/TextOp/optritrack_dataset/optitrack_npz_filtered_all/balance_014_Skeleton 006_z_up_x_forward_gym/motion.npz"
