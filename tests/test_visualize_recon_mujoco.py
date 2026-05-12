from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from scripts.visualize_recon_mujoco import (
    add_ghost_robot_pose,
    build_parser,
    default_mujoco_xml,
)


def test_parser_accepts_ae_model_type():
    args = build_parser().parse_args(
        [
            "--model_type",
            "ae",
            "--checkpoint",
            "ae.pt",
            "--npz_path",
            "motion.npz",
        ]
    )

    assert args.model_type == "ae"
    assert args.checkpoint == "ae.pt"
    assert args.config == "configs/default.yaml"


def test_parser_accepts_vae_model_type():
    args = build_parser().parse_args(
        [
            "--model_type",
            "vae",
            "--checkpoint",
            "vae.pt",
            "--npz_path",
            "motion.npz",
        ]
    )

    assert args.model_type == "vae"
    assert args.vae_config == "configs/transformer_vae.yaml"


def test_default_mujoco_xml_prefers_textoptracker_g1_act_xml():
    path = default_mujoco_xml()

    assert path.endswith("TextOpTracker/source/textop_tracker/textop_tracker/assets/unitree_description/mjcf/g1_act.xml")
    assert os.path.exists(path)


def test_script_file_execution_can_import_project_package():
    result = subprocess.run(
        [sys.executable, "scripts/visualize_recon_mujoco.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--model_type" in result.stdout


def test_add_ghost_robot_pose_uses_python_mujoco_add_geoms_signature():
    calls = {}

    class FakeMujoco:
        class MjvOption:
            def __init__(self):
                self.geomgroup = np.ones(6, dtype=np.uint8)

        class MjvPerturb:
            pass

        mjtCatBit = SimpleNamespace(mjCAT_DYNAMIC=2, mjCAT_DECOR=4)
        mjtObj = SimpleNamespace(mjOBJ_GEOM=5)

        @staticmethod
        def mjv_addGeoms(model, data, opt, pert, catmask, scene):
            calls["args"] = (model, data, opt, pert, catmask, scene)
            scene.ngeom = 1

        @staticmethod
        def mj_id2name(_model, _objtype, _objid):
            return "pelvis"

    scene = SimpleNamespace(
        ngeom=0,
        geoms=[SimpleNamespace(objid=0, rgba=np.zeros(4), category=None)],
    )
    model = SimpleNamespace(ngeom=1, geom_group=np.array([2]))

    add_ghost_robot_pose(
        FakeMujoco,
        model,
        data=object(),
        scene=scene,
        rgba=np.array([0.45, 1.0, 0.55, 0.28], dtype=np.float32),
    )

    assert isinstance(calls["args"][2], FakeMujoco.MjvOption)
    assert isinstance(calls["args"][3], FakeMujoco.MjvPerturb)
    assert calls["args"][4] == FakeMujoco.mjtCatBit.mjCAT_DYNAMIC
    np.testing.assert_allclose(scene.geoms[0].rgba, [0.45, 1.0, 0.55, 0.28])
