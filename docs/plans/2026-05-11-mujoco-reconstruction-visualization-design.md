# MuJoCo Reconstruction Visualization Design

**Goal:** Add a `motion_ae` MuJoCo viewer that compares one reconstruction model, selected by `--model_type ae|vae`, against the original `motion.npz` sequence.

**Design:** The viewer loads a single G1 MuJoCo XML. The reconstructed motion drives the normal solid robot in `MjData`; the original motion drives a second `MjData` for forward kinematics, then its body tree is drawn as a wireframe overlay through `viewer.user_scn`. This avoids generating a second robot XML while still showing original and reconstructed poses in one MuJoCo window.

**Inputs:** The script accepts `--npz_path`, `--model_type`, `--checkpoint`, and optional config/stat/xml/device arguments. `--model_type ae` uses `motion_ae.config` and `MotionAutoEncoder`; `--model_type vae` uses `transformer_vae.config` and `MotionTransformerVAE`.

**Data Flow:** The script builds normalized windows with the existing `MotionWindowDataset`, runs reconstruction in batches, denormalizes the result, and converts feature windows to MuJoCo qpos. Root position and root quaternion come from the original pelvis world trajectory; joint qpos comes from either the original or reconstructed feature `joint_pos` slice.

**Controls:** Space toggles autoplay, left/right changes frame, `N/M` changes window, `R` resets to the first frame of the current window, and `Q`/Esc exits.

**Testing:** Unit tests cover root/joint feature-to-qpos conversion and window/frame indexing. The live viewer remains manual because CI may not have `mujoco` or a display.
