#!/usr/bin/env python3
"""Convert Manus marker_pen pickles (Genesis Z-up frame, table at z=0, 60 Hz)
into ManipTrans IsaacGym-frame demo + retargeting pickles.

Per-frame transform (Genesis -> ManipTrans IsaacGym):
    p_gym = R_z(-pi/2) @ p_user + [0, 0, 0.415]
    R_gym = R_z(-pi/2) @ R_user

Output layout (under --output_dir):
    ObjURDF/<obj_id>.urdf
    ObjURDF/<obj_id>_collision.obj          (copied from input)
    sequences/<stem>.pkl                    (loaded by your ManipData subclass)
    retargeting/<stem>.pkl                  (loaded by load_retargeted_data)
    sequences.json                          (index)

sequences/<stem>.pkl keys (everything in IsaacGym frame, except wrist_rot which
is a rotation matrix as ManipTrans's __getitem__ contract requires):
    data_path        : str
    obj_id           : str
    side             : 'right' | 'left'
    fps              : 60
    obj_urdf_path    : str (absolute path to ObjURDF/<obj_id>.urdf)
    obj_verts        : (1000, 3) float32, mesh-local
    obj_trajectory   : (T, 4, 4) float32, world SE(3)
    wrist_pos        : (T, 3)    float32
    wrist_rot        : (T, 3, 3) float32 rotation matrices
    mano_joints      : dict with ONLY the 5 fingertip keys:
                       thumb_tip / index_tip / middle_tip / ring_tip / pinky_tip
                       each (T, 3) float32

retargeting/<stem>.pkl keys:
    opt_wrist_pos : (T, 3)  float32
    opt_wrist_rot : (T, 3)  float32 axis-angle
    opt_dof_pos   : (T, 20) float32 (frame-invariant)

Loader-side caveats your custom ManipData subclass should handle:
  * Set self.mujoco2gym_transf = torch.eye(4, ...) so process_data does NOT
    re-rotate the data (it's already in IsaacGym frame).
  * Set self.skip = 2 so process_data computes velocities with
    time_delta = self.skip / 120 = 1/60 s, matching the 60 Hz data. Do NOT
    actually subsample (your data is already 60 Hz).
  * mano_joints carries only the 5 *_tip keys, so patch dexhandmanip_sh.py's
    pack_data and reward to use those 5 keys instead of iterating
    dexhand.body_names. The base.py process_data already iterates
    data['mano_joints'].keys() for tips_distance + mano_joints_velocity, so
    those work unchanged.
"""

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R


TABLE_SURFACE_Z = 0.415  # ManipTrans IsaacGym table-top height
USER_TO_GYM_R = R.from_euler("z", -90, degrees=True).as_matrix().astype(np.float64)
USER_TO_GYM_T = np.array([0.0, 0.0, TABLE_SURFACE_Z], dtype=np.float64)

FINGERTIP_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="{obj_id}">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.05"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
"""


# -------------------- transforms --------------------

def transform_positions(p):
    """Rotate -90 deg about Z, then lift by TABLE_SURFACE_Z."""
    return p @ USER_TO_GYM_R.T + USER_TO_GYM_T


def transform_axis_angle_to_matrix(aa):
    """(T, 3) axis-angle (user frame) -> (T, 3, 3) rotation matrix (gym frame)."""
    R_user = R.from_rotvec(aa).as_matrix()                 # (T, 3, 3)
    return USER_TO_GYM_R @ R_user                          # broadcast: (3,3) @ (T,3,3)


def transform_pose_matrices(M):
    """(T, 4, 4) SE(3) (user frame) -> (T, 4, 4) SE(3) (gym frame)."""
    out = np.zeros_like(M)
    out[:, 3, 3] = 1.0
    out[:, :3, :3] = USER_TO_GYM_R @ M[:, :3, :3]
    out[:, :3, 3] = M[:, :3, 3] @ USER_TO_GYM_R.T + USER_TO_GYM_T
    return out


# -------------------- helpers --------------------

def sample_obj_verts(mesh_path, n=1000, seed=0):
    """Uniform-area surface sampling (mesh-local frame)."""
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=int(seed))
    return np.asarray(pts, dtype=np.float32)


def write_object_urdf(obj_dir, obj_id, mesh_filename):
    urdf_path = os.path.join(obj_dir, f"{obj_id}.urdf")
    with open(urdf_path, "w") as f:
        f.write(URDF_TEMPLATE.format(obj_id=obj_id, mesh_filename=mesh_filename))
    return urdf_path


def convert_one(in_path, demo_dir, retarget_dir, urdf_abs_path, obj_verts, n_dofs):
    with open(in_path, "rb") as f:
        d = pickle.load(f)

    md = d["metadata"]
    T = int(md["num_frames"])
    obj_id = md["obj_id"]
    side = md["hand_side"]

    ht = d["hand_trajectory"]
    obj = d["object_trajectory"]
    mref = d["mano_reference"]

    # --- core fields ---
    obj_traj = transform_pose_matrices(obj["pose_matrices"].astype(np.float64))
    wrist_pos_user = ht["wrist_positions"].astype(np.float64)
    wrist_aa_user = ht["wrist_rotations_aa"].astype(np.float64)
    wrist_pos_gym = transform_positions(wrist_pos_user)
    wrist_R_gym = transform_axis_angle_to_matrix(wrist_aa_user)
    wrist_aa_gym = R.from_matrix(wrist_R_gym).as_rotvec()

    dof_pos = ht["dof_positions"].astype(np.float32)
    if dof_pos.shape[1] != n_dofs:
        raise ValueError(
            f"{in_path}: expected {n_dofs} DOFs in dof_positions, got {dof_pos.shape[1]}"
        )

    # --- fingertip-only mano_joints ---
    src_tips = mref["finger_joints"]
    missing = [k for k in FINGERTIP_NAMES if k not in src_tips]
    if missing:
        raise KeyError(f"{in_path}: missing fingertip keys {missing}")
    mano_joints = {
        k: transform_positions(src_tips[k].astype(np.float64)).astype(np.float32)
        for k in FINGERTIP_NAMES
    }

    # --- shape sanity ---
    for name, arr, expected in [
        ("obj_trajectory", obj_traj, (T, 4, 4)),
        ("wrist_pos", wrist_pos_gym, (T, 3)),
        ("wrist_rot", wrist_R_gym, (T, 3, 3)),
        ("dof_pos", dof_pos, (T, n_dofs)),
    ]:
        if arr.shape != expected:
            raise ValueError(f"{in_path}: {name} shape {arr.shape} != expected {expected}")
    for k, v in mano_joints.items():
        if v.shape != (T, 3):
            raise ValueError(f"{in_path}: mano_joints[{k}] shape {v.shape} != ({T}, 3)")

    demo = {
        "data_path": str(in_path),
        "obj_id": obj_id,
        "side": side,
        "fps": 60,
        "obj_urdf_path": str(urdf_abs_path),
        "obj_verts": obj_verts,                                  # (1000, 3) float32
        "obj_trajectory": obj_traj.astype(np.float32),           # (T, 4, 4)
        "wrist_pos": wrist_pos_gym.astype(np.float32),           # (T, 3)
        "wrist_rot": wrist_R_gym.astype(np.float32),             # (T, 3, 3) rotation matrices
        "mano_joints": mano_joints,                              # 5 fingertip keys
    }

    retarget = {
        "opt_wrist_pos": wrist_pos_gym.astype(np.float32),       # (T, 3)
        "opt_wrist_rot": wrist_aa_gym.astype(np.float32),        # (T, 3) axis-angle
        "opt_dof_pos": dof_pos,                                  # (T, n_dofs) frame-invariant
    }

    stem = Path(in_path).stem
    with open(os.path.join(demo_dir, f"{stem}.pkl"), "wb") as f:
        pickle.dump(demo, f, protocol=4)
    with open(os.path.join(retarget_dir, f"{stem}.pkl"), "wb") as f:
        pickle.dump(retarget, f, protocol=4)
    return stem, T


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input_dir", required=True,
                    help="Dir of input *.pkl (also expected to contain "
                         "<obj_id>_collision.obj unless --obj_mesh is given).")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write ObjURDF/, sequences/, retargeting/.")
    ap.add_argument("--obj_mesh", default=None,
                    help="Override path to the collision .obj. Defaults to "
                         "<input_dir>/<obj_id>_collision.obj from the first pickle.")
    ap.add_argument("--n_dofs", type=int, default=20,
                    help="Expected hand DOF count (wujihand=20).")
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, only convert the first N pickles (for testing).")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    obj_dir = out_dir / "ObjURDF"
    demo_dir = out_dir / "sequences"
    retarget_dir = out_dir / "retargeting"
    for d in (obj_dir, demo_dir, retarget_dir):
        d.mkdir(parents=True, exist_ok=True)

    pickles = sorted(p for p in in_dir.glob("*.pkl"))
    if args.limit is not None:
        pickles = pickles[: args.limit]
    if not pickles:
        print(f"No pickles found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    # peek first pickle for obj_id / side
    with open(pickles[0], "rb") as f:
        first = pickle.load(f)
    obj_id = first["metadata"]["obj_id"]
    side_first = first["metadata"]["hand_side"]
    print(f"Object: {obj_id}   first-pickle hand_side: {side_first}")

    # mesh + URDF
    mesh_src = (Path(args.obj_mesh).resolve() if args.obj_mesh
                else (in_dir / f"{obj_id}_collision.obj"))
    if not mesh_src.exists():
        print(f"Mesh not found: {mesh_src}", file=sys.stderr)
        sys.exit(1)
    mesh_filename = f"{obj_id}_collision.obj"
    mesh_dst = obj_dir / mesh_filename
    if not mesh_dst.exists():
        shutil.copy2(mesh_src, mesh_dst)
        print(f"Copied mesh -> {mesh_dst}")
    urdf_path = write_object_urdf(str(obj_dir), obj_id, mesh_filename)
    print(f"Wrote URDF -> {urdf_path}")

    # sample obj verts (mesh-local, used by BPS encoder; same for every sequence)
    obj_verts = sample_obj_verts(str(mesh_dst), n=1000, seed=0)
    print(
        f"Sampled {len(obj_verts)} surface points "
        f"(x[{obj_verts[:,0].min():.3f},{obj_verts[:,0].max():.3f}] "
        f"y[{obj_verts[:,1].min():.3f},{obj_verts[:,1].max():.3f}] "
        f"z[{obj_verts[:,2].min():.3f},{obj_verts[:,2].max():.3f}])"
    )

    # Show the transform applied to the first pickle's first frame so the
    # user can eyeball the rotation/lift before processing the whole dir.
    print("\nSanity check on first pickle, frame 0:")
    p_user = first["hand_trajectory"]["wrist_positions"][0]
    o_user = first["object_trajectory"]["positions"][0]
    p_gym = transform_positions(p_user[None])[0]
    o_gym = transform_positions(o_user[None])[0]
    print(f"  wrist user={p_user}  ->  gym={p_gym}")
    print(f"  obj   user={o_user}  ->  gym={o_gym}")

    # convert
    converted = []
    failed = []
    print(f"\nConverting {len(pickles)} sequences...")
    for p in pickles:
        try:
            stem, T = convert_one(
                p, str(demo_dir), str(retarget_dir), urdf_path, obj_verts, args.n_dofs
            )
        except Exception as e:
            failed.append((p.name, str(e)))
            print(f"  SKIP {p.name}: {e}")
            continue
        converted.append({"idx": stem, "T": T})
        print(f"  ok   {stem}  T={T}")

    index = {
        "obj_id": obj_id,
        "urdf_path": urdf_path,
        "n_dofs": args.n_dofs,
        "fps": 60,
        "frame_convention": "ManipTrans IsaacGym (Z up, table z=0.415)",
        "sequences": converted,
        "failed": failed,
    }
    with open(out_dir / "sequences.json", "w") as f:
        json.dump(index, f, indent=2)

    print(
        f"\nDone. {len(converted)} ok, {len(failed)} failed. "
        f"Index: {out_dir / 'sequences.json'}"
    )


if __name__ == "__main__":
    main()
