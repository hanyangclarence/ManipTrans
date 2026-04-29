"""
Visualize Dexterous Hand and Object Trajectory in IsaacGym (Headless Mode)

This script loads a retargeted dexterous hand trajectory and object trajectory,
then visualizes them in IsaacGym with headless rendering.
Outputs are saved as images or video.
Supports all hand types: shadow, inspire, inspireftp, allegro, artimano, xhand, wujihand.

Usage:
    python visualize_shadow_trajectory.py --data_idx g0 --hand_type shadow --side right --output_dir visualization_output
"""

import os
import argparse
import numpy as np
import pickle
from isaacgym import gymapi, gymtorch, gymutil
import torch
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def aa_to_quat(aa):
    """Convert axis-angle to quaternion (x, y, z, w) for IsaacGym"""
    if isinstance(aa, torch.Tensor):
        aa = aa.cpu().numpy()

    angle = np.linalg.norm(aa)
    if angle < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion in [x, y, z, w]

    quat_xyzw = R.from_rotvec(aa).as_quat()  # Returns [x, y, z, w]
    return quat_xyzw  # IsaacGym uses [x, y, z, w] format


def rotmat_to_quat(rotmat):
    """Convert rotation matrix to quaternion (x, y, z, w) for IsaacGym"""
    if isinstance(rotmat, torch.Tensor):
        rotmat = rotmat.cpu().numpy()

    quat_xyzw = R.from_matrix(rotmat).as_quat()  # Returns [x, y, z, w]
    return quat_xyzw  # IsaacGym uses [x, y, z, w] format


class DexHandTrajectoryVisualizer:
    def __init__(self, args):
        self.args = args
        self.gym = gymapi.acquire_gym()

        # Initialize flags
        self.has_object = False

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load trajectory data
        self.load_trajectory_data()

        # Setup simulation
        self.setup_simulation()

        # Create environment
        self.create_environment()

        # Setup camera
        self.setup_camera()

    def load_trajectory_data(self):
        """Load trajectory data from pickle file.

        Supports two formats:
          (A) Legacy: <input_dir>/<hand_type>_hand_trajectory_<data_idx>.pkl
              produced by extract_shadow_trajectory.py.
          (B) Converted: <input_dir>/sequences/<data_idx>.pkl plus
              <input_dir>/retargeting/<data_idx>.pkl produced by
              convert_manus_to_maniptrans.py.
        """
        converted_demo = os.path.join(
            self.args.input_dir, "sequences", f"{self.args.data_idx}.pkl"
        )
        legacy_path = os.path.join(
            self.args.input_dir,
            f"{self.args.hand_type}_hand_trajectory_{self.args.data_idx}.pkl",
        )

        if os.path.exists(converted_demo):
            self._load_converted_format(converted_demo)
        elif os.path.exists(legacy_path):
            self._load_legacy_format(legacy_path)
        else:
            raise FileNotFoundError(
                "Trajectory file not found in either format:\n"
                f"  converted: {converted_demo}\n"
                f"  legacy:    {legacy_path}"
            )

        # Common fields the rest of the visualizer expects
        self.num_frames = self.obj_traj["pose_matrices"].shape[0]
        self.n_dofs = self.hand_traj["dof_positions"].shape[1]

        print("Loaded trajectory:")
        print(f"  - Frames: {self.num_frames} ({self.num_frames/60:.2f}s at 60 FPS)")
        print(f"  - {self.args.hand_type.capitalize()} Hand DOFs: {self.n_dofs}")
        print(f"  - Hand side: {self.metadata['side']}")
        coord_system = self.metadata.get("coordinate_system", None)
        if coord_system != "IsaacGym":
            raise ValueError(
                f"Expected coordinate_system='IsaacGym', got '{coord_system}'."
            )
        print(f"  - Coordinate system: {coord_system}")

    def _load_legacy_format(self, traj_file):
        print(f"Loading trajectory (legacy format) from: {traj_file}")
        with open(traj_file, "rb") as f:
            self.data = pickle.load(f)
        self.obj_traj = self.data["object_trajectory"]
        self.hand_traj = self.data["hand_trajectory"]
        self.mano_ref = self.data["mano_reference"]
        self.metadata = self.data["metadata"]

    def _load_converted_format(self, demo_path):
        retarget_path = os.path.join(
            self.args.input_dir, "retargeting", f"{self.args.data_idx}.pkl"
        )
        if not os.path.exists(retarget_path):
            raise FileNotFoundError(
                f"Converted demo present but retargeting file is missing:\n"
                f"  {retarget_path}"
            )
        print(f"Loading trajectory (converted format) from:")
        print(f"  demo:      {demo_path}")
        print(f"  retarget:  {retarget_path}")
        with open(demo_path, "rb") as f:
            demo = pickle.load(f)
        with open(retarget_path, "rb") as f:
            retarget = pickle.load(f)

        # convert wrist_rot (T,3,3) -> axis-angle for the rest of the pipeline
        wrist_rotmat = np.asarray(demo["wrist_rot"])
        wrist_aa = R.from_matrix(wrist_rotmat).as_rotvec().astype(np.float32)

        self.obj_traj = {
            "pose_matrices": np.asarray(demo["obj_trajectory"]),
        }
        self.hand_traj = {
            "wrist_positions": np.asarray(demo["wrist_pos"]),
            "wrist_rotations_aa": wrist_aa,
            "dof_positions": np.asarray(retarget["opt_dof_pos"]),
        }
        self.mano_ref = {
            "finger_joints": {
                k: np.asarray(v) for k, v in demo["mano_joints"].items()
            },
        }
        self.metadata = {
            "obj_urdf_path": demo["obj_urdf_path"],
            "obj_id": demo.get("obj_id", "object"),
            "side": demo["side"],
            "coordinate_system": "IsaacGym",
            "fps": demo.get("fps", 60),
        }
        self.data = demo  # keep for downstream introspection

    def setup_simulation(self):
        """Setup IsaacGym simulation parameters"""
        sim_params = gymapi.SimParams()

        # Set common parameters
        sim_params.dt = 1.0 / 60.0  # 60 FPS
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        # Set PhysX parameters
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = True

        # Create sim
        self.sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params
        )

        if self.sim is None:
            raise Exception("Failed to create sim")

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.distance = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def create_environment(self):
        """Create environment with dexterous hand and object"""

        # Create environment
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1)

        # Get hand URDF path using DexHandFactory
        dexhand = DexHandFactory.create_hand(self.args.hand_type, self.args.side)
        hand_urdf_raw = dexhand.urdf_path

        # Normalize path to resolve ../../../ which IsaacGym doesn't like
        hand_urdf_abs = os.path.normpath(os.path.abspath(hand_urdf_raw))

        # Split into asset_root and relative path for IsaacGym
        # IsaacGym expects: asset_root + asset_file
        asset_root = os.path.dirname(hand_urdf_abs)
        hand_urdf = os.path.basename(hand_urdf_abs)

        asset_options = gymapi.AssetOptions()
        # Kinematic playback: pin the hand base so PhysX doesn't integrate
        # an apparent floating-base velocity from each set_actor_root_state
        # teleport (same fix as the object). The finger DOFs are still
        # set every frame via set_dof_state_tensor.
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        print(f"\nLoading {self.args.hand_type.capitalize()} Hand from: {hand_urdf_abs}")
        self.hand_asset = self.gym.load_asset(self.sim, asset_root, hand_urdf, asset_options)

        # Get DOF properties
        dof_props = self.gym.get_asset_dof_properties(self.hand_asset)
        self.num_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        print(f"{self.args.hand_type.capitalize()} Hand DOFs: {self.num_hand_dofs}")

        # Set DOF properties for position control
        for i in range(self.num_hand_dofs):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = 20.0
            dof_props['damping'][i] = 1.0

        # Create dexterous hand actor
        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(0, 0, 0.5)
        hand_pose.r = gymapi.Quat(0, 0, 0, 1)

        # collisionFilter=-1 disables all collision contacts for this actor.
        # We are doing kinematic replay (states overwritten every frame), so
        # any PhysX contact resolution between hand/object/markers/table
        # would corrupt the recorded poses with collision impulses, showing
        # up as the rotation jitter the user observed.
        NO_COLLISION = -1

        self.hand_actor = self.gym.create_actor(
            self.env,
            self.hand_asset,
            hand_pose,
            f"{self.args.hand_type}_hand",
            0,
            NO_COLLISION,
        )

        # Set DOF properties
        self.gym.set_actor_dof_properties(self.env, self.hand_actor, dof_props)

        # Set hand color
        num_bodies = self.gym.get_actor_rigid_body_count(self.env, self.hand_actor)
        for i in range(num_bodies):
            self.gym.set_rigid_body_color(
                self.env, self.hand_actor, i,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.9, 0.7, 0.5)  # Skin-like color
            )

        # Load object asset
        obj_urdf_raw = self.metadata['obj_urdf_path']

        # Normalize object path
        obj_urdf_abs = os.path.normpath(os.path.abspath(obj_urdf_raw))
        obj_asset_root = os.path.dirname(obj_urdf_abs)
        obj_urdf = os.path.basename(obj_urdf_abs)

        obj_asset_options = gymapi.AssetOptions()
        # Kinematic playback: pin the object so PhysX does not integrate its
        # state. set_actor_root_state_tensor still teleports it each frame.
        obj_asset_options.fix_base_link = True
        obj_asset_options.disable_gravity = True

        print(f"Loading object from: {obj_urdf_abs}")

        # Try to load object asset
        try:
            self.obj_asset = self.gym.load_asset(self.sim, obj_asset_root, obj_urdf, obj_asset_options)

            # Create object actor
            obj_pose = gymapi.Transform()
            obj_pose.p = gymapi.Vec3(0, 0, 0.5)
            obj_pose.r = gymapi.Quat(0, 0, 0, 1)

            self.obj_actor = self.gym.create_actor(
                self.env,
                self.obj_asset,
                obj_pose,
                "object",
                0,
                NO_COLLISION,
            )

            # Set object color
            obj_num_bodies = self.gym.get_actor_rigid_body_count(self.env, self.obj_actor)
            for i in range(obj_num_bodies):
                self.gym.set_rigid_body_color(
                    self.env, self.obj_actor, i,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.2, 0.6, 0.9)  # Blue
                )

            self.has_object = True
            print("✓ Object loaded successfully")

        except Exception as e:
            print(f"⚠️  Warning: Failed to load object: {e}")
            print("   Continuing with hand-only visualization...")
            self.obj_actor = None
            self.has_object = False

        # Create MANO joint markers (spheres)
        self.create_mano_markers()

        # Create the training-env table so the rendered scene matches what
        # ManipTrans actually instantiates. Specs match dexhandmanip_sh.py:
        #   box 1.0 x 1.6 x 0.03 m, fix-base, center at (-0.1, 0, 0.4),
        #   so its top face sits at z = 0.415 (the IsaacGym table surface).
        # Placed last so existing actor indices (0=hand, 1=obj, 2+=markers)
        # are preserved.
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 1.0, 1.6, 0.03, table_asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(-0.1, 0.0, 0.4)
        table_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.table_actor = self.gym.create_actor(
            self.env, table_asset, table_pose, "table", 0, NO_COLLISION,
        )
        self.gym.set_rigid_body_color(
            self.env, self.table_actor, 0, gymapi.MESH_VISUAL,
            gymapi.Vec3(0.1, 0.1, 0.1),
        )
        print("✓ Table actor added (matches training-env table)")

    def create_mano_markers(self):
        """Create sphere markers for MANO finger joints"""
        print(f"\nCreating MANO joint markers...")

        # Create sphere asset for markers
        sphere_radius = 0.008  # 8mm radius spheres
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000.0
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        self.sphere_asset = self.gym.create_sphere(self.sim, sphere_radius, asset_options)

        # Get joint names from mano_reference
        self.mano_joint_names = list(self.mano_ref['finger_joints'].keys())
        self.mano_marker_actors = []

        # Define colors for different fingers
        finger_colors = {
            'thumb': gymapi.Vec3(1.0, 0.0, 0.0),      # Red
            'index': gymapi.Vec3(0.0, 1.0, 0.0),      # Green
            'middle': gymapi.Vec3(0.0, 0.0, 1.0),     # Blue
            'ring': gymapi.Vec3(1.0, 1.0, 0.0),       # Yellow
            'pinky': gymapi.Vec3(1.0, 0.0, 1.0),      # Magenta
        }

        # Create a sphere actor for each joint
        for joint_name in self.mano_joint_names:
            marker_pose = gymapi.Transform()
            marker_pose.p = gymapi.Vec3(0, 0, 0.5)
            marker_pose.r = gymapi.Quat(0, 0, 0, 1)

            marker_actor = self.gym.create_actor(
                self.env,
                self.sphere_asset,
                marker_pose,
                f"mano_marker_{joint_name}",
                0,
                -1,  # NO_COLLISION
            )

            # Set color based on finger
            finger_name = joint_name.split('_')[0]  # Extract finger name (thumb, index, etc.)
            color = finger_colors.get(finger_name, gymapi.Vec3(0.5, 0.5, 0.5))

            self.gym.set_rigid_body_color(
                self.env, marker_actor, 0,
                gymapi.MESH_VISUAL,
                color
            )

            self.mano_marker_actors.append(marker_actor)

        print(f"✓ Created {len(self.mano_marker_actors)} MANO joint markers")
        print(f"  Colors: Thumb=Red, Index=Green, Middle=Blue, Ring=Yellow, Pinky=Magenta")

    def setup_camera(self):
        """Setup camera for rendering"""
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.args.width
        camera_props.height = self.args.height
        camera_props.enable_tensors = True

        self.camera_handle = self.gym.create_camera_sensor(self.env, camera_props)

        # Auto-frame the camera around the trajectory's mean position so the
        # action is in view regardless of where the demo's table-frame lands.
        wrist_xyz = np.asarray(self.hand_traj["wrist_positions"])
        obj_xyz = np.asarray(self.obj_traj["pose_matrices"])[:, :3, 3]
        scene_center = 0.5 * (wrist_xyz.mean(axis=0) + obj_xyz.mean(axis=0))

        if self.args.camera_view == "side":
            # Horizontal view, eye-level with the table top. Useful for
            # eyeballing whether the object actually sits on the table.
            cam_offset = np.array([0.6, 0.6, 0.0])
            cam_target_xyz = scene_center.copy()
            cam_target_xyz[2] = 0.415  # look at table top
            cam_pos_xyz = scene_center + cam_offset
            cam_pos_xyz[2] = 0.42  # ~5 mm above table top
        else:  # "iso"
            cam_offset = np.array([0.55, 0.55, 0.35])
            cam_target_xyz = scene_center
            cam_pos_xyz = scene_center + cam_offset

        cam_pos = gymapi.Vec3(*cam_pos_xyz.tolist())
        cam_target = gymapi.Vec3(*cam_target_xyz.tolist())
        self.gym.set_camera_location(self.camera_handle, self.env, cam_pos, cam_target)

        print(f"\nCamera setup: {self.args.width}x{self.args.height}")

    def setup_state_tensors(self):
        """Setup state tensors for efficient state updates"""
        # Acquire tensors
        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state = self.gym.acquire_dof_state_tensor(self.sim)

        # Wrap as torch tensors
        self.root_tensor = gymtorch.wrap_tensor(_root_state)
        self.dof_tensor = gymtorch.wrap_tensor(_dof_state)

        # Pull the *actual* current poses (set by create_actor) into the
        # wrapped tensor. Without this the tensor is all zeros, and
        # apply_states' set_actor_root_state_tensor call would teleport every
        # actor we don't explicitly update (notably the fixed-base table)
        # back to (0, 0, 0).
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        print(f"State tensors initialized:")
        print(f"  Root state shape: {self.root_tensor.shape}")
        print(f"  DOF state shape: {self.dof_tensor.shape}")

    def set_hand_state(self, frame_idx):
        """Set dexterous hand state for a given frame"""
        # Get wrist pose
        wrist_pos = self.hand_traj['wrist_positions'][frame_idx]
        wrist_rot_aa = self.hand_traj['wrist_rotations_aa'][frame_idx]
        wrist_quat = aa_to_quat(wrist_rot_aa)

        # Get DOF positions
        dof_pos = self.hand_traj['dof_positions'][frame_idx]

        # Set root state (wrist) - hand is actor 0 in env 0
        self.root_tensor[0, 0:3] = torch.tensor(wrist_pos, dtype=torch.float32)
        self.root_tensor[0, 3:7] = torch.tensor(wrist_quat, dtype=torch.float32)
        self.root_tensor[0, 7:13] = 0  # Velocities

    def set_object_state(self, frame_idx):
        """Set object state for a given frame"""
        if not self.has_object:
            return  # Skip if object wasn't loaded

        # Get object pose
        obj_pose_mat = self.obj_traj['pose_matrices'][frame_idx]
        obj_pos = obj_pose_mat[:3, 3]
        obj_rot_mat = obj_pose_mat[:3, :3]
        obj_quat = rotmat_to_quat(obj_rot_mat)

        # Set root state - object is actor 1 in env 0
        self.root_tensor[1, 0:3] = torch.tensor(obj_pos, dtype=torch.float32)
        self.root_tensor[1, 3:7] = torch.tensor(obj_quat, dtype=torch.float32)
        self.root_tensor[1, 7:13] = 0  # Velocities

    def set_mano_markers_state(self, frame_idx):
        """Set MANO marker positions for a given frame"""
        # Actor indices: 0=hand, 1=object (if exists), 2+=mano markers
        marker_start_idx = 2 if self.has_object else 1

        for i, joint_name in enumerate(self.mano_joint_names):
            # Get joint position from mano_reference
            joint_pos = self.mano_ref['finger_joints'][joint_name][frame_idx]

            # Set root state for this marker
            actor_idx = marker_start_idx + i
            self.root_tensor[actor_idx, 0:3] = torch.tensor(joint_pos, dtype=torch.float32)
            self.root_tensor[actor_idx, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)  # Identity quaternion
            self.root_tensor[actor_idx, 7:13] = 0  # Velocities

    def apply_states(self, frame_idx):
        """Apply both hand and object states, then step simulation"""
        # Get DOF positions
        dof_pos = self.hand_traj['dof_positions'][frame_idx]

        # Set DOF positions for hand (first num_hand_dofs)
        self.dof_tensor[:self.num_hand_dofs, 0] = torch.tensor(dof_pos, dtype=torch.float32)
        self.dof_tensor[:self.num_hand_dofs, 1] = 0  # Zero velocity

        # Apply states
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_tensor))

    def render_frame(self, frame_idx):
        """Render a single frame and return the image"""
        # Optionally pin every step to a single source frame (debug aid).
        eff = self.args.freeze_frame if self.args.freeze_frame >= 0 else frame_idx
        # Set states for hand, object, and MANO markers
        self.set_hand_state(eff)
        self.set_object_state(eff)
        self.set_mano_markers_state(eff)
        self.apply_states(eff)

        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Render
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # Get camera image
        img = self.gym.get_camera_image(
            self.sim,
            self.env,
            self.camera_handle,
            gymapi.IMAGE_COLOR
        )

        # Convert to numpy array and reshape
        img = img.reshape(self.args.height, self.args.width, 4)

        # Convert RGBA to RGB
        img_rgb = img[:, :, :3]

        return img_rgb

    def visualize(self):
        """Main visualization loop"""
        print(f"\n{'='*80}")
        print(f"Starting visualization")
        print(f"{'='*80}\n")

        # Prepare simulation
        self.gym.prepare_sim(self.sim)

        # Setup state tensors
        self.setup_state_tensors()


        # Render every nth frame to save time/space
        render_every = self.args.render_every
        frames_to_render = range(0, self.num_frames, render_every)

        print(f"Rendering {len(frames_to_render)} frames (every {render_every} frame)")

        # Render frames
        images = []
        for i, frame_idx in enumerate(tqdm(frames_to_render, desc="Rendering frames")):
            img = self.render_frame(frame_idx)
            images.append(img)

            # Optionally save individual frames
            if self.args.save_frames:
                img_pil = Image.fromarray(img.astype(np.uint8))
                img_path = os.path.join(self.args.output_dir, f"frame_{frame_idx:04d}.png")
                img_pil.save(img_path)

        # Save as video if requested
        if self.args.save_video:
            self.save_video(images, render_every)

        print(f"\n✓ Visualization complete!")
        print(f"  Output directory: {self.args.output_dir}")

    def cleanup(self):
        """Clean up simulation resources"""
        if hasattr(self, 'sim') and self.sim is not None:
            self.gym.destroy_sim(self.sim)

    def save_video(self, images, render_every):
        """Save images as video using imageio"""
        try:
            import imageio

            video_path = os.path.join(self.args.output_dir, f"trajectory_{self.args.data_idx}.mp4")

            # Calculate FPS based on render_every
            fps = 60 // render_every

            print(f"\nSaving video to: {video_path}")
            print(f"  FPS: {fps}, Total frames: {len(images)}")

            try:
                writer = imageio.get_writer(video_path, fps=fps, quality=8)
                for img in images:
                    writer.append_data(img)
                writer.close()
                print(f"✓ Video saved successfully!")
            except (ValueError, RuntimeError) as e:
                print(f"\n⚠ Could not create video: {e}")
                print("  Install ffmpeg support with: pip install imageio-ffmpeg")
                print("  Or use imageio[pyav]: pip install 'imageio[pyav]'")
                print("\n  Saving frames as images instead...")
                self.save_image_sequence(images)

        except ImportError:
            print("\n⚠ imageio not installed. Install with: pip install imageio imageio-ffmpeg")
            print("  Saving frames as images instead...")
            self.save_image_sequence(images)

    def save_image_sequence(self, images):
        """Save images as PNG sequence"""
        print(f"\nSaving {len(images)} frames as PNG images...")

        for i, img in enumerate(images):
            img_pil = Image.fromarray(img.astype(np.uint8))
            img_path = os.path.join(self.args.output_dir, f"frame_{i:04d}.png")
            img_pil.save(img_path)

        print(f"✓ Saved {len(images)} frames to {self.args.output_dir}")
        print(f"\nTo create a video from these frames, use:")
        print(f"  ffmpeg -framerate 30 -i {self.args.output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {self.args.output_dir}/trajectory.mp4")


def main():
    parser = argparse.ArgumentParser(description="Visualize dexterous hand trajectory in IsaacGym")

    # Data parameters
    parser.add_argument("--data_idx", type=str, default="g0",
                        help="Data index (e.g., g0 for grab_demo/102)")
    parser.add_argument("--hand_type", type=str, default="shadow",
                        choices=["shadow", "inspire", "inspireftp", "allegro", "artimano", "xhand", "wujihand"],
                        help="Hand type (default: shadow)")
    parser.add_argument("--side", type=str, default="right", choices=["right", "left"],
                        help="Hand side")
    parser.add_argument("--input_dir", type=str, default="output_trajectories",
                        help="Directory containing trajectory data")
    parser.add_argument("--output_dir", type=str, default="visualization_output",
                        help="Directory to save visualization output")

    # Rendering parameters
    parser.add_argument("--width", type=int, default=1920,
                        help="Image width")
    parser.add_argument("--height", type=int, default=1080,
                        help="Image height")
    parser.add_argument("--render_every", type=int, default=2,
                        help="Render every nth frame (1=all frames, 2=half, etc.)")
    parser.add_argument("--save_frames", action="store_true",
                        help="Save individual frames as images")
    parser.add_argument("--save_video", action="store_true", default=True,
                        help="Save as video (requires imageio)")
    parser.add_argument("--camera_view", type=str, default="iso",
                        choices=["iso", "side"],
                        help="Camera angle: 'iso' (default 3/4 view) or "
                             "'side' (horizontal, eye-level with table top).")
    parser.add_argument("--freeze_frame", type=int, default=-1,
                        help="If >=0, force every rendered timestep to use "
                             "this single source frame's pose. Useful for "
                             "diagnosing physics-induced jitter (a frozen "
                             "trajectory should produce a perfectly still "
                             "video; any motion = physics drift).")

    # Device parameters
    parser.add_argument("--compute_device_id", type=int, default=0,
                        help="Compute device ID")
    parser.add_argument("--graphics_device_id", type=int, default=0,
                        help="Graphics device ID")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Dexterous Hand Trajectory Visualizer")
    print(f"{'='*80}\n")
    print(f"Configuration:")
    print(f"  Hand: {args.hand_type.capitalize()}")
    print(f"  Data: {args.data_idx} ({args.side} hand)")
    print(f"  Input: {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Render every: {args.render_every} frame(s)")

    # Create visualizer and run
    visualizer = DexHandTrajectoryVisualizer(args)
    try:
        visualizer.visualize()
    finally:
        visualizer.cleanup()


if __name__ == "__main__":
    main()
