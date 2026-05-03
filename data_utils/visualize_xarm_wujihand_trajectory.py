"""
Visualize a Manus-converted ManipTrans trajectory in IsaacGym using the
xArm7 + WujiHand combined robot (fixed base) instead of a free-floating hand.

Mirrors `data_utils/visualize_shadow_trajectory.py` structurally, but every
frame:
  - solves IK for the 7 xArm joints to put palm_link at the demo wrist pose
  - drives all 27 DOFs (7 arm + 20 finger) via the dof state tensor
  - keeps the arm base fixed at (-0.2, 0, 0) so geometry matches the GP
    reference setup in scripts/data_processing/replay_xarm_wujihand_trajectory.py

The trajectory pickles produced by convert_manus_to_maniptrans.py have the
mujoco->isaac shift baked in (object/wrist z ≈ 0.413 because the existing
floating-hand env has its table at z=0.4). To put the trajectory on a
GP-style table whose top sits at z=0.1, we shift z by `-0.315 m` (= 0.1
- 0.415). Object xy is unchanged.

Usage:
    python data_utils/visualize_xarm_wujihand_trajectory.py \\
        --data_idx <idx> --hand_type xarm_wujihand --side right \\
        --input_dir data/manus_marker_pen \\
        --output_dir visualization_output_xarm
"""

import os
import sys
import argparse
import pickle

# IsaacGym must be imported before torch.
from isaacgym import gymapi, gymtorch
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


# GP placement values are defined in *Genesis* (Mujoco-style Z-up) frame.
# data_utils/convert_manus_to_maniptrans.py shows that the converter maps
# Genesis -> IsaacGym via:
#   p_gym = R_z(-90deg) @ p_genesis + [0, 0, 0.415]
#   R_gym = R_z(-90deg) @ R_genesis
# The +0.415 lift is a *trajectory-only* hack to land objects on the
# existing IG 0.415m table — it does NOT apply to static actors (the arm
# is bolted to the IG ground at z=0). The R_z(-90) rotation IS a global
# frame change and must be applied to every static placement coming from
# GP.
#
# Genesis values from replay_xarm_wujihand_trajectory.py (RIGHT_ARM_BASE_POS
# and the table in `replay()`):
#     arm base       (-0.2, 0,    0   )
#     table center   ( 0.5, 0,    0.09)
#     table size     ( 0.6, 1.0,  0.02)
#     trajectory offset (0, 0, 0.1) → trajectory zero point at z=0.1
#
# Mapped to IsaacGym (R_z(-90) only; ground stays at z=0; xy-size swaps):
#     arm base       ( 0,   0.2,  0   )         orient = R_z(-90)
#     table center   ( 0,  -0.5,  0.09)         top at z=0.10
#     table size     ( 1.0, 0.6,  0.02)         (x↔y swap)
DEFAULT_ARM_BASE_POS = (0.0, 0.2, 0.0)
# R_z(-90deg) as quaternion (x, y, z, w): rotates the arm's local +X to world -Y,
# matching GP's "arm forward = +X" rendered through the Genesis->IG rotation.
import math as _math
_HALF = _math.sin(-_math.pi / 4)   # sin(-45°) = -sqrt(2)/2
_HALF_C = _math.cos(-_math.pi / 4) # cos(-45°) = +sqrt(2)/2
DEFAULT_ARM_BASE_QUAT = (0.0, 0.0, _HALF, _HALF_C)
DEFAULT_TABLE_CENTER = (0.0, -0.5, 0.09)   # top face at z = 0.10
DEFAULT_TABLE_SIZE = (1.0, 0.6, 0.02)      # x and y swapped vs Genesis
# Existing converter bakes mujoco->isaac z-shift = 0.415 into pickles. To
# place the action on a GP-style table whose top is at z=0.10, shift by:
DEFAULT_Z_SHIFT = 0.10 - 0.415   # = -0.315


def aa_to_quat_xyzw(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) -> IsaacGym quaternion (x, y, z, w)."""
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return R.from_rotvec(aa).as_quat()


def aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    return R.from_rotvec(aa).as_matrix()


def rotmat_to_quat_xyzw(rotmat: np.ndarray) -> np.ndarray:
    return R.from_matrix(rotmat).as_quat()


class XArmWujiHandVisualizer:
    def __init__(self, args):
        self.args = args
        self.gym = gymapi.acquire_gym()
        os.makedirs(args.output_dir, exist_ok=True)

        self.has_object = False

        self.load_trajectory_data()
        self.setup_simulation()
        self.create_environment()
        self.setup_ik()
        self.setup_camera()

    # --------------------------------------------------------------- data
    def load_trajectory_data(self):
        """Load a Manus-converted trajectory (sequences/ + retargeting/)."""
        demo_path = os.path.join(
            self.args.input_dir, "sequences", f"{self.args.data_idx}.pkl"
        )
        retarget_path = os.path.join(
            self.args.input_dir, "retargeting", f"{self.args.data_idx}.pkl"
        )
        if not (os.path.exists(demo_path) and os.path.exists(retarget_path)):
            raise FileNotFoundError(
                f"Need both:\n  {demo_path}\n  {retarget_path}"
            )

        with open(demo_path, "rb") as f:
            demo = pickle.load(f)
        with open(retarget_path, "rb") as f:
            retarget = pickle.load(f)

        # Demo carries an absolute wrist rotation matrix per frame; the rest
        # of this script wants axis-angle for quat conversion symmetry.
        wrist_rotmat = np.asarray(demo["wrist_rot"])
        wrist_aa = R.from_matrix(wrist_rotmat).as_rotvec().astype(np.float32)

        z_shift = self.args.z_shift
        wrist_pos = np.asarray(demo["wrist_pos"], dtype=np.float32).copy()
        wrist_pos[:, 2] += z_shift

        obj_traj = np.asarray(demo["obj_trajectory"], dtype=np.float32).copy()
        obj_traj[:, 2, 3] += z_shift

        finger_dofs = np.asarray(retarget["opt_dof_pos"], dtype=np.float32)

        # Optional: MANO finger keypoints (for marker spheres). Apply the
        # same z shift so they line up with the simulated hand.
        mano_joints = {}
        for k, v in demo.get("mano_joints", {}).items():
            arr = np.asarray(v, dtype=np.float32).copy()
            if arr.ndim == 2 and arr.shape[1] == 3:
                arr[:, 2] += z_shift
            mano_joints[k] = arr

        self.wrist_pos = wrist_pos                 # (T, 3) world frame
        self.wrist_rotmat = wrist_rotmat            # (T, 3, 3)
        self.wrist_aa = wrist_aa                    # (T, 3)
        self.obj_traj = obj_traj                    # (T, 4, 4)
        self.finger_dofs = finger_dofs              # (T, n_finger_dofs)
        self.mano_joints = mano_joints              # name -> (T, 3)

        self.metadata = {
            "obj_urdf_path": demo["obj_urdf_path"],
            "obj_id": demo.get("obj_id", "object"),
            "side": demo["side"],
            "fps": demo.get("fps", 60),
        }

        self.num_frames = obj_traj.shape[0]
        print(f"Loaded trajectory: {self.num_frames} frames (z_shift={z_shift:.3f} m)")
        print(f"  initial wrist (post-shift): {wrist_pos[0]}")
        print(f"  initial obj   (post-shift): {obj_traj[0, :3, 3]}")

    # --------------------------------------------------------------- sim
    def setup_simulation(self):
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = True

        self.sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params,
        )
        if self.sim is None:
            raise RuntimeError("Failed to create sim")

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.distance = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def create_environment(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1)

        # ---------- xArm7 + WujiHand ----------
        dexhand = DexHandFactory.create_hand(self.args.hand_type, self.args.side)
        self.dexhand = dexhand
        urdf_abs = os.path.normpath(os.path.abspath(dexhand.urdf_path))
        asset_root = os.path.dirname(urdf_abs)
        asset_file = os.path.basename(urdf_abs)

        opts = gymapi.AssetOptions()
        opts.fix_base_link = True
        opts.disable_gravity = True
        opts.flip_visual_attachments = False
        opts.collapse_fixed_joints = False
        opts.thickness = 0.001
        opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
        print(f"Loading robot: {urdf_abs}")
        self.hand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, opts)
        self.num_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        assert self.num_hand_dofs == 27, f"expected 27 dofs, got {self.num_hand_dofs}"

        dof_props = self.gym.get_asset_dof_properties(self.hand_asset)
        for i in range(self.num_hand_dofs):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            # Tracking-only: stiff PD so the actor follows targets without lag.
            dof_props["stiffness"][i] = 800.0
            dof_props["damping"][i] = 40.0
        # Grab arm DOF metadata (lower/upper, used for clamping during IK)
        self.arm_dof_lower = np.array([dof_props["lower"][i] for i in range(7)])
        self.arm_dof_upper = np.array([dof_props["upper"][i] for i in range(7)])

        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(*self.args.arm_base_pos)
        hand_pose.r = gymapi.Quat(*self.args.arm_base_quat)

        NO_COLLISION = -1
        self.hand_actor = self.gym.create_actor(
            self.env, self.hand_asset, hand_pose, "xarm_wujihand", 0, NO_COLLISION,
        )
        self.gym.set_actor_dof_properties(self.env, self.hand_actor, dof_props)

        # Cache world->arm_base frame transform for IK targets.
        self.arm_base_pos_np = np.array(self.args.arm_base_pos, dtype=np.float64)
        # Quaternion → rotation matrix (R_arm_base = world->arm orientation).
        # IsaacGym quat order is (x, y, z, w); scipy expects (x, y, z, w) too.
        self.arm_base_R = R.from_quat(self.args.arm_base_quat).as_matrix()
        # World->base inverse rotation (used to transform IK targets):
        self.world_to_base_R = self.arm_base_R.T

        n_bodies = self.gym.get_actor_rigid_body_count(self.env, self.hand_actor)
        for i in range(n_bodies):
            self.gym.set_rigid_body_color(
                self.env, self.hand_actor, i,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.9, 0.7, 0.5),
            )

        # ---------- Object ----------
        obj_urdf_abs = os.path.normpath(os.path.abspath(self.metadata["obj_urdf_path"]))
        obj_asset_root = os.path.dirname(obj_urdf_abs)
        obj_asset_file = os.path.basename(obj_urdf_abs)
        obj_opts = gymapi.AssetOptions()
        obj_opts.fix_base_link = True       # kinematic playback
        obj_opts.disable_gravity = True
        try:
            self.obj_asset = self.gym.load_asset(
                self.sim, obj_asset_root, obj_asset_file, obj_opts
            )
            obj_pose0 = gymapi.Transform()
            obj_pose0.p = gymapi.Vec3(*self.obj_traj[0, :3, 3].tolist())
            obj_pose0.r = gymapi.Quat(*rotmat_to_quat_xyzw(self.obj_traj[0, :3, :3]).tolist())
            self.obj_actor = self.gym.create_actor(
                self.env, self.obj_asset, obj_pose0, "object", 0, NO_COLLISION,
            )
            for i in range(self.gym.get_actor_rigid_body_count(self.env, self.obj_actor)):
                self.gym.set_rigid_body_color(
                    self.env, self.obj_actor, i,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.2, 0.6, 0.9),
                )
            self.has_object = True
            print(f"Loaded object: {obj_urdf_abs}")
        except Exception as e:
            print(f"Object load failed: {e}; continuing without object")
            self.obj_actor = None

        # ---------- MANO marker spheres ----------
        self.create_mano_markers()

        # ---------- Table ----------
        # Visualize-only (collisionFilter=-1). Geometry mirrors the GP table
        # with a small xy enlargement so ManipTrans trajectories fit on it.
        sx, sy, sz = self.args.table_size
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, sx, sy, sz, table_opts)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(*self.args.table_center)
        self.table_actor = self.gym.create_actor(
            self.env, table_asset, table_pose, "table", 0, NO_COLLISION,
        )
        self.gym.set_rigid_body_color(
            self.env, self.table_actor, 0, gymapi.MESH_VISUAL,
            gymapi.Vec3(0.15, 0.15, 0.15),
        )
        print(
            f"Table: center={self.args.table_center} size={self.args.table_size} "
            f"top_z={self.args.table_center[2] + sz / 2:.3f}"
        )

    def create_mano_markers(self):
        sphere_opts = gymapi.AssetOptions()
        sphere_opts.density = 1000.0
        sphere_opts.fix_base_link = True
        sphere_opts.disable_gravity = True
        self.sphere_asset = self.gym.create_sphere(self.sim, 0.008, sphere_opts)

        finger_colors = {
            "thumb": gymapi.Vec3(1.0, 0.0, 0.0),
            "index": gymapi.Vec3(0.0, 1.0, 0.0),
            "middle": gymapi.Vec3(0.0, 0.0, 1.0),
            "ring": gymapi.Vec3(1.0, 1.0, 0.0),
            "pinky": gymapi.Vec3(1.0, 0.0, 1.0),
        }
        self.mano_joint_names = list(self.mano_joints.keys())
        self.mano_marker_actors = []
        for joint_name in self.mano_joint_names:
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, 0.5)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            actor = self.gym.create_actor(
                self.env, self.sphere_asset, pose,
                f"mano_marker_{joint_name}", 0, -1,
            )
            color = finger_colors.get(joint_name.split("_")[0], gymapi.Vec3(0.5, 0.5, 0.5))
            self.gym.set_rigid_body_color(
                self.env, actor, 0, gymapi.MESH_VISUAL, color
            )
            self.mano_marker_actors.append(actor)
        print(f"Created {len(self.mano_marker_actors)} MANO marker actors")

    # --------------------------------------------------------------- ik
    def setup_ik(self):
        ik_dir = os.path.join(
            os.path.dirname(os.path.abspath(self.dexhand.urdf_path)), "ik"
        )
        if ik_dir not in sys.path:
            sys.path.insert(0, ik_dir)
        from xarm7_kinematics import XArm7Kinematics
        # palm_link is offset (0, 0, 0.057) from link_eef, with yaw 2.3562
        # (right) or 0.7854 (left) per the URDF wujihand_fix joint.
        self.ik = XArm7Kinematics(tcp_offset=[0, 0, 57, 0, 0, self.dexhand.tcp_yaw])
        self._prev_arm_q = self.dexhand.arm_seed.copy()
        self._ik_failures = 0

    def solve_ik(self, wrist_pos_world: np.ndarray, wrist_rotmat_world: np.ndarray):
        """One-frame IK. Wrist pose is given in IsaacGym world frame; we
        rotate it into the arm-base frame before handing it to the C IK
        solver (which expects targets in the base of the manipulator).

        Returns (arm_angles[7], succeeded: bool).
        """
        # World -> arm-base translation, in arm-base frame:
        delta_world = wrist_pos_world - self.arm_base_pos_np   # (3,)
        target_pos_base = self.world_to_base_R @ delta_world   # (3,)
        target_pos_mm = target_pos_base * 1000.0
        # World -> arm-base rotation:
        target_R_base = self.world_to_base_R @ wrist_rotmat_world

        target_mat = np.eye(4)
        target_mat[:3, :3] = target_R_base
        target_mat[:3, 3] = target_pos_mm
        try:
            arm_q = self.ik.inverse_kinematics_mat(target_mat, q_ref=self._prev_arm_q)
            self._prev_arm_q = arm_q
            return arm_q, True
        except RuntimeError:
            self._ik_failures += 1
            return self._prev_arm_q.copy(), False

    # --------------------------------------------------------------- camera
    def setup_camera(self):
        cam_props = gymapi.CameraProperties()
        cam_props.width = self.args.width
        cam_props.height = self.args.height
        cam_props.enable_tensors = True
        self.camera_handle = self.gym.create_camera_sensor(self.env, cam_props)

        # Frame the action area.
        scene_center = 0.5 * (
            self.wrist_pos.mean(axis=0) + self.obj_traj[:, :3, 3].mean(axis=0)
        )
        if self.args.camera_view == "side":
            cam_offset = np.array([0.6, 0.6, 0.0])
            cam_target = scene_center.copy()
            cam_target[2] = self.args.table_center[2] + self.args.table_size[2] / 2
            cam_pos = scene_center + cam_offset
            cam_pos[2] = cam_target[2] + 0.005
        else:
            cam_offset = np.array([0.55, 0.55, 0.35])
            cam_target = scene_center
            cam_pos = scene_center + cam_offset

        self.gym.set_camera_location(
            self.camera_handle, self.env,
            gymapi.Vec3(*cam_pos.tolist()),
            gymapi.Vec3(*cam_target.tolist()),
        )

    # --------------------------------------------------------------- run
    def setup_state_tensors(self):
        _root = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof = self.gym.acquire_dof_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(_root)
        self.dof_tensor = gymtorch.wrap_tensor(_dof)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # Actor index layout:
        #   0: hand (xarm_wujihand) - root is fixed, do not write
        #   1: object (if has_object)
        #   1 or 2 .. : MANO markers
        #   last: table - leave at spawn pose
        self.obj_actor_idx = 1 if self.has_object else None
        self.mano_marker_start = (2 if self.has_object else 1)

    def set_frame_state(self, frame_idx: int):
        # Solve IK for arm joints, take finger DOFs straight from the pickle.
        wrist_pos = self.wrist_pos[frame_idx]
        wrist_rot = self.wrist_rotmat[frame_idx]
        arm_q, ok = self.solve_ik(wrist_pos, wrist_rot)
        finger_q = self.finger_dofs[frame_idx]
        n_f = min(len(finger_q), self.num_hand_dofs - 7)
        full_q = np.concatenate([arm_q, finger_q[:n_f]])
        # Pad if pickle has fewer than 20 finger DOFs (unlikely for wuji)
        if full_q.shape[0] < self.num_hand_dofs:
            pad = np.zeros(self.num_hand_dofs - full_q.shape[0])
            full_q = np.concatenate([full_q, pad])

        self.dof_tensor[:self.num_hand_dofs, 0] = torch.tensor(full_q, dtype=torch.float32)
        self.dof_tensor[:self.num_hand_dofs, 1] = 0.0

        if self.has_object:
            obj_mat = self.obj_traj[frame_idx]
            obj_pos = obj_mat[:3, 3]
            obj_quat = rotmat_to_quat_xyzw(obj_mat[:3, :3])
            self.root_tensor[self.obj_actor_idx, 0:3] = torch.tensor(obj_pos, dtype=torch.float32)
            self.root_tensor[self.obj_actor_idx, 3:7] = torch.tensor(obj_quat, dtype=torch.float32)
            self.root_tensor[self.obj_actor_idx, 7:13] = 0.0

        for i, jname in enumerate(self.mano_joint_names):
            actor_idx = self.mano_marker_start + i
            joint_pos = self.mano_joints[jname][frame_idx]
            self.root_tensor[actor_idx, 0:3] = torch.tensor(joint_pos, dtype=torch.float32)
            self.root_tensor[actor_idx, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
            self.root_tensor[actor_idx, 7:13] = 0.0

        return ok

    def apply_states(self):
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_tensor))

    def render_frame(self, frame_idx: int) -> np.ndarray:
        eff = self.args.freeze_frame if self.args.freeze_frame >= 0 else frame_idx
        self.set_frame_state(eff)
        self.apply_states()
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        img = self.gym.get_camera_image(
            self.sim, self.env, self.camera_handle, gymapi.IMAGE_COLOR
        )
        img = img.reshape(self.args.height, self.args.width, 4)
        return img[:, :, :3]

    def visualize(self):
        self.gym.prepare_sim(self.sim)
        self.setup_state_tensors()

        every = max(1, self.args.render_every)
        frames = list(range(0, self.num_frames, every))
        print(f"Rendering {len(frames)} of {self.num_frames} frames")

        images = []
        for frame_idx in tqdm(frames, desc="Render"):
            images.append(self.render_frame(frame_idx))
            if self.args.save_frames:
                Image.fromarray(images[-1].astype(np.uint8)).save(
                    os.path.join(self.args.output_dir, f"frame_{frame_idx:04d}.png")
                )

        if self._ik_failures:
            print(f"IK failures during playback: {self._ik_failures} / {len(frames)}")

        if self.args.save_video:
            self.save_video(images, every)
        print(f"Done. Output dir: {self.args.output_dir}")

    def save_video(self, images, every):
        try:
            import imageio
        except ImportError:
            print("imageio missing; falling back to PNG sequence")
            for i, img in enumerate(images):
                Image.fromarray(img.astype(np.uint8)).save(
                    os.path.join(self.args.output_dir, f"frame_{i:04d}.png")
                )
            return
        fps = max(1, 60 // every)
        path = os.path.join(self.args.output_dir, f"trajectory_{self.args.data_idx}.mp4")
        print(f"Writing {path} at {fps} fps")
        writer = imageio.get_writer(path, fps=fps, quality=8)
        for img in images:
            writer.append_data(img)
        writer.close()
        print("Video saved")

    def cleanup(self):
        if hasattr(self, "sim") and self.sim is not None:
            self.gym.destroy_sim(self.sim)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_idx", type=str, required=True)
    p.add_argument("--input_dir", type=str, default="data/manus_marker_pen")
    p.add_argument("--output_dir", type=str, default="visualization_output_xarm")
    p.add_argument("--hand_type", type=str, default="xarm_wujihand")
    p.add_argument("--side", type=str, default="right", choices=["right", "left"])

    p.add_argument("--arm_base_pos", type=float, nargs=3, default=DEFAULT_ARM_BASE_POS,
                   help="xArm base xyz in IG world frame. Default (0, 0.2, 0) = "
                        "GP's (-0.2, 0, 0) rotated by R_z(-90).")
    p.add_argument("--arm_base_quat", type=float, nargs=4, default=DEFAULT_ARM_BASE_QUAT,
                   help="xArm base orientation as IG quaternion (x, y, z, w). "
                        "Default = R_z(-90) so the arm's local +X points along world -Y, "
                        "matching GP's +X-forward arm rendered through the Genesis->IG "
                        "frame rotation.")
    p.add_argument("--table_center", type=float, nargs=3, default=DEFAULT_TABLE_CENTER)
    p.add_argument("--table_size", type=float, nargs=3, default=DEFAULT_TABLE_SIZE)
    p.add_argument("--z_shift", type=float, default=DEFAULT_Z_SHIFT,
                   help="Added to wrist/object/mano z. Default -0.315 takes ManipTrans pickles "
                        "(z baked +0.415) onto a GP-style table (top at z=0.10).")

    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--render_every", type=int, default=2)
    p.add_argument("--save_frames", action="store_true")
    p.add_argument("--save_video", action="store_true", default=True)
    p.add_argument("--camera_view", type=str, default="iso", choices=["iso", "side"])
    p.add_argument("--freeze_frame", type=int, default=-1)

    p.add_argument("--compute_device_id", type=int, default=0)
    p.add_argument("--graphics_device_id", type=int, default=0)

    args = p.parse_args()

    viz = XArmWujiHandVisualizer(args)
    try:
        viz.visualize()
    finally:
        viz.cleanup()


if __name__ == "__main__":
    main()
