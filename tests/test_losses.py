import pytest

from motion_ae.losses import ReconstructionLoss


def test_reconstruction_loss_rejects_unknown_group_weights():
    group_slices = {
        "joint_pos": (0, 29),
        "joint_vel": (29, 58),
        "pelvis_rot6d_b": (58, 64),
        "pelvis_lin_vel_b": (64, 67),
        "pelvis_ang_vel_b": (67, 70),
    }

    with pytest.raises(ValueError, match="pelvis_quat_b"):
        ReconstructionLoss(
            group_slices=group_slices,
            group_weights={"pelvis_quat_b": 1.0},
        )
