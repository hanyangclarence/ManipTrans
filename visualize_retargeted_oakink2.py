"""
Visualize a retargeted OakInk2 hand-object trajectory in IsaacGym.

Loads the trajectory through the OakInk2 RH dataset (which merges raw OakInk2
annotations + the dumped retargeting pickle) and plays back kinematically:
the floating dexhand root pose comes from `opt_wrist_pos`/`opt_wrist_rot`,
the finger DOFs from `opt_dof_pos`, and the object pose from `obj_trajectory`.
Optional sphere markers at the source MANO joint positions for visual
sanity-check of the fit.

Default mode: headless, dumps PNG frames + an mp4 (if ffmpeg is on PATH) under
--output_dir. With --viewer, opens an interactive IsaacGym viewer instead.

Usage:
    python visualize_retargeted_oakink2.py --data_idx 03865@0 --dexhand inspire
    python visualize_retargeted_oakink2.py --data_idx 03865@0 --dexhand inspire --viewer
    python visualize_retargeted_oakink2.py --data_idx 03865@0 --dexhand inspire --no-mano-markers
"""

import argparse
import os
import shutil
import subprocess

# IsaacGym must be imported before torch.
from isaacgym import gymapi, gymtorch  # noqa: F401

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from main.dataset.factory import ManipDataFactory
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def aa_to_quat_xyzw(aa):
    aa = aa.cpu().numpy() if isinstance(aa, torch.Tensor) else np.asarray(aa)
    n = np.linalg.norm(aa)
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return R.from_rotvec(aa).as_quat().astype(np.float32)


def rotmat_to_quat_xyzw(rot):
    rot = rot.cpu().numpy() if isinstance(rot, torch.Tensor) else np.asarray(rot)
    return R.from_matrix(rot).as_quat().astype(np.float32)


# Color per finger group for the MANO markers.
FINGER_COLORS = {
    "thumb": (1.0, 0.0, 0.0),
    "index": (0.0, 1.0, 0.0),
    "middle": (0.0, 0.0, 1.0),
    "ring": (1.0, 1.0, 0.0),
    "pinky": (1.0, 0.0, 1.0),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_idx", required=True,
                    help="OakInk2 index '<5hex>@<stage>', e.g. '03865@0'")
    ap.add_argument("--dexhand", default="inspire")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--data_dir", default="data/OakInk-v2")
    ap.add_argument("--output_dir", default="visualization_output")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--render_every", type=int, default=1,
                    help="render every Nth frame (default 1, all frames)")
    ap.add_argument("--viewer", action="store_true",
                    help="interactive IsaacGym viewer instead of PNG/mp4")
    ap.add_argument("--no-mano-markers", action="store_true",
                    help="skip the colored sphere markers at MANO joints")
    ap.add_argument("--save_frames", action="store_true",
                    help="keep individual PNG frames after building the mp4")
    ap.add_argument("--cam_pos", nargs=3, type=float, default=[0.8, 0.8, 0.9],
                    help="camera position (world frame). Trajectories live in "
                         "the mano2dexhand 'table-top' frame, so action is at "
                         "z~0.4-0.6.")
    ap.add_argument("--cam_target", nargs=3, type=float, default=[0.0, 0.0, 0.5])
    return ap.parse_args()


def _mano2dex_transf(device, table_surface_z=0.415):
    """The same frame transform mano2dexhand.fitting() and dexhandimitator's
    training env apply: R_z(-pi/2) @ R_x(pi/2), then translate to the
    table-top z. opt_wrist_pos / opt_wrist_rot / opt_dof_pos in the retargeting
    pickle are emitted in this frame, so obj_trajectory + mano_joints must be
    transformed the same way for the viz to line up."""
    from main.dataset.transform import aa_to_rotmat
    M = np.eye(4)
    M[:3, :3] = (aa_to_rotmat(np.array([0, 0, -np.pi / 2]))
                 @ aa_to_rotmat(np.array([np.pi / 2, 0, 0])))
    M[:3, 3] = np.array([0, 0, table_surface_z])
    return torch.tensor(M, dtype=torch.float32, device=device)


def load_trajectory(args, dexhand):
    """Build an OakInk2 RH dataset and pull a single primitive task."""
    transf = _mano2dex_transf(device="cuda:0")
    ds = ManipDataFactory.create_data(
        manipdata_type="oakink2",
        side=args.side,
        data_dir=args.data_dir,
        device="cuda:0",
        mujoco2gym_transf=transf,
        dexhand=dexhand,
        verbose=True,
    )
    data = ds[args.data_idx]
    n_frames = data["obj_trajectory"].shape[0]
    n_dofs = data["opt_dof_pos"].shape[1]
    print(f"\nloaded '{args.data_idx}': {n_frames} frames, {n_dofs} dofs, "
          f"obj_id={data['obj_id']}")
    print(f"  wrist_pos range: x=[{data['wrist_pos'][:,0].min():.2f},"
          f"{data['wrist_pos'][:,0].max():.2f}] "
          f"y=[{data['wrist_pos'][:,1].min():.2f},{data['wrist_pos'][:,1].max():.2f}] "
          f"z=[{data['wrist_pos'][:,2].min():.2f},{data['wrist_pos'][:,2].max():.2f}]")
    return data


def make_sim(args):
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.dt = 1.0 / 60.0
    sp.substeps = 1
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, 0.0)   # kinematic playback only
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 1
    sp.physx.use_gpu = True
    sp.use_gpu_pipeline = True
    graphics_id = 0 if args.viewer else 0
    sim = gym.create_sim(0, graphics_id, gymapi.SIM_PHYSX, sp)
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)
    return gym, sim


def load_hand_asset(gym, sim, dexhand):
    urdf_abs = os.path.normpath(os.path.abspath(dexhand.urdf_path))
    asset_root = os.path.dirname(urdf_abs)
    asset_file = os.path.basename(urdf_abs)
    opts = gymapi.AssetOptions()
    opts.fix_base_link = False
    opts.disable_gravity = True
    opts.flip_visual_attachments = False
    opts.collapse_fixed_joints = False
    opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
    print(f"loading hand URDF: {urdf_abs}")
    return gym.load_asset(sim, asset_root, asset_file, opts)


def load_object_asset(gym, sim, urdf_path):
    urdf_abs = os.path.normpath(os.path.abspath(urdf_path))
    asset_root = os.path.dirname(urdf_abs)
    asset_file = os.path.basename(urdf_abs)
    opts = gymapi.AssetOptions()
    opts.fix_base_link = False
    opts.disable_gravity = True
    print(f"loading object URDF: {urdf_abs}")
    return gym.load_asset(sim, asset_root, asset_file, opts)


def color_actor(gym, env, actor, rgb):
    nb = gym.get_actor_rigid_body_count(env, actor)
    for i in range(nb):
        gym.set_rigid_body_color(env, actor, i, gymapi.MESH_VISUAL,
                                  gymapi.Vec3(*rgb))


def build_env(args, gym, sim, hand_asset, obj_asset, mano_joint_names):
    spacing = 1.0
    env = gym.create_env(sim,
                         gymapi.Vec3(-spacing, -spacing, 0.0),
                         gymapi.Vec3( spacing,  spacing,  spacing),
                         1)

    # Hand actor (idx 0).
    hand_pose = gymapi.Transform()
    hand_pose.p = gymapi.Vec3(0, 0, 0.5)
    hand_actor = gym.create_actor(env, hand_asset, hand_pose, "dexhand", 0, 0)
    dof_props = gym.get_asset_dof_properties(hand_asset)
    n_dofs = gym.get_asset_dof_count(hand_asset)
    # Low PD gains for kinematic playback. With high gains, the DOF position
    # controller fights the per-frame teleport and applies large reaction
    # torques that propagate through the joints and lift the floating root.
    for i in range(n_dofs):
        dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
        dof_props["stiffness"][i] = 20.0
        dof_props["damping"][i] = 1.0
    gym.set_actor_dof_properties(env, hand_actor, dof_props)
    color_actor(gym, env, hand_actor, (0.85, 0.70, 0.50))

    # Object actor (idx 1).
    obj_pose = gymapi.Transform()
    obj_pose.p = gymapi.Vec3(0, 0, 0.5)
    obj_actor = gym.create_actor(env, obj_asset, obj_pose, "object", 0, 0)
    color_actor(gym, env, obj_actor, (0.20, 0.55, 0.85))

    # MANO joint markers (idx 2..2+N).
    marker_actors = []
    if mano_joint_names:
        sphere_opts = gymapi.AssetOptions()
        sphere_opts.fix_base_link = True
        sphere_opts.disable_gravity = True
        sphere_asset = gym.create_sphere(sim, 0.006, sphere_opts)
        for jn in mano_joint_names:
            mp = gymapi.Transform()
            mp.p = gymapi.Vec3(0, 0, 0.5)
            ma = gym.create_actor(env, sphere_asset, mp, f"mano_{jn}", 0, 0)
            finger = jn.split("_")[0]
            color_actor(gym, env, ma, FINGER_COLORS.get(finger, (0.6, 0.6, 0.6)))
            marker_actors.append(ma)

    return env, hand_actor, obj_actor, marker_actors, n_dofs


def write_states(root_tensor, dof_tensor, n_dofs, has_markers,
                 wrist_pos, wrist_quat, dof_pos,
                 obj_pos, obj_quat,
                 marker_positions):
    root_tensor[0, 0:3] = torch.as_tensor(wrist_pos, dtype=torch.float32)
    root_tensor[0, 3:7] = torch.as_tensor(wrist_quat, dtype=torch.float32)
    root_tensor[0, 7:13] = 0
    root_tensor[1, 0:3] = torch.as_tensor(obj_pos, dtype=torch.float32)
    root_tensor[1, 3:7] = torch.as_tensor(obj_quat, dtype=torch.float32)
    root_tensor[1, 7:13] = 0
    if has_markers:
        for i, p in enumerate(marker_positions):
            root_tensor[2 + i, 0:3] = torch.as_tensor(p, dtype=torch.float32)
            root_tensor[2 + i, 3:7] = torch.tensor([0., 0., 0., 1.])
            root_tensor[2 + i, 7:13] = 0
    dof_tensor[:n_dofs, 0] = torch.as_tensor(dof_pos, dtype=torch.float32)
    dof_tensor[:n_dofs, 1] = 0


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dexhand = DexHandFactory.create_hand(args.dexhand, args.side)
    data = load_trajectory(args, dexhand)

    n_frames = data["obj_trajectory"].shape[0]
    mano_joint_names = (None if args.no_mano_markers
                        else list(data["mano_joints"].keys()))

    gym, sim = make_sim(args)
    hand_asset = load_hand_asset(gym, sim, dexhand)
    obj_asset  = load_object_asset(gym, sim, data["obj_urdf_path"])
    env, hand_actor, obj_actor, marker_actors, n_dofs = build_env(
        args, gym, sim, hand_asset, obj_asset, mano_joint_names or []
    )

    # Camera (image dump) or viewer.
    cam_pos = gymapi.Vec3(*args.cam_pos)
    cam_target = gymapi.Vec3(*args.cam_target)
    if args.viewer:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)
        camera_handle = None
    else:
        viewer = None
        cam_props = gymapi.CameraProperties()
        cam_props.width = args.width
        cam_props.height = args.height
        camera_handle = gym.create_camera_sensor(env, cam_props)
        gym.set_camera_location(camera_handle, env, cam_pos, cam_target)

    gym.prepare_sim(sim)
    root_tensor = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof_tensor  = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))

    # Pull trajectory tensors to CPU once.
    opt_wrist_pos = data["opt_wrist_pos"].detach().cpu().numpy()
    opt_wrist_rot = data["opt_wrist_rot"].detach().cpu().numpy()  # axis-angle
    opt_dof_pos   = data["opt_dof_pos"].detach().cpu().numpy()
    obj_traj_se3  = data["obj_trajectory"].detach().cpu().numpy()  # (T, 4, 4)
    if mano_joint_names:
        mano = {jn: data["mano_joints"][jn].detach().cpu().numpy()
                for jn in mano_joint_names}

    frame_dir = os.path.join(args.output_dir, f"frames_{args.data_idx.replace('@','_')}")
    if not args.viewer:
        os.makedirs(frame_dir, exist_ok=True)

    frame_iter = range(0, n_frames, args.render_every)
    rendered_paths = []

    for f in tqdm(frame_iter, desc="rendering"):
        wrist_quat = aa_to_quat_xyzw(opt_wrist_rot[f])
        obj_T = obj_traj_se3[f]
        obj_pos = obj_T[:3, 3]
        obj_quat = rotmat_to_quat_xyzw(obj_T[:3, :3])
        marker_pos = ([mano[jn][f] for jn in mano_joint_names]
                      if mano_joint_names else [])

        write_states(root_tensor, dof_tensor, n_dofs, bool(mano_joint_names),
                     opt_wrist_pos[f], wrist_quat, opt_dof_pos[f],
                     obj_pos, obj_quat, marker_pos)
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_tensor))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_tensor))
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_tensor[:, 0].contiguous()))

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)

        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
            if gym.query_viewer_has_closed(viewer):
                print("viewer closed by user")
                break
        else:
            gym.render_all_camera_sensors(sim)
            img = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
            img = img.reshape(args.height, args.width, 4)[:, :, :3]
            # Sequential numbering (not source-frame index) so ffmpeg's %05d
            # pattern works regardless of --render_every.
            out_path = os.path.join(frame_dir, f"frame_{len(rendered_paths):05d}.png")
            Image.fromarray(img.astype(np.uint8)).save(out_path)
            rendered_paths.append(out_path)

    # Build mp4 from PNG frames if ffmpeg is available.
    if not args.viewer and rendered_paths:
        mp4_path = os.path.join(args.output_dir,
                                f"{args.data_idx.replace('@','_')}.mp4")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print(f"\n(ffmpeg not on PATH; PNG frames left in {frame_dir})")
        else:
            fps = max(1, int(round(60 / args.render_every)))
            cmd = [ffmpeg, "-y", "-framerate", str(fps),
                   "-i", os.path.join(frame_dir, "frame_%05d.png"),
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4_path]
            print(f"\nbuilding mp4: {mp4_path}")
            try:
                subprocess.run(cmd, check=True,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                print(f"  wrote {mp4_path}")
                if not args.save_frames:
                    shutil.rmtree(frame_dir, ignore_errors=True)
            except subprocess.CalledProcessError as e:
                print(f"  ffmpeg failed: {e}; PNG frames preserved")

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("done.")


if __name__ == "__main__":
    main()
