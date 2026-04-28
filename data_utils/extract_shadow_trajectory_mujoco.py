"""
Extract retargeted dexterous hand trajectory and object pose trajectory in MUJOCO coordinates.
This script transforms the data back to the original MUJOCO coordinate system.
Both hand and object trajectories are saved in MUJOCO coordinates.
Supports all hand types: shadow, inspire, inspireftp, allegro, artimano, xhand, wujihand.
"""
from isaacgym import gymapi, gymtorch, gymutil
import os
import numpy as np
import pickle
from main.dataset.factory import ManipDataFactory
from main.dataset.transform import aa_to_rotmat, rotmat_to_aa
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
import torch
from scipy.spatial.transform import Rotation as R
import shutil
import xml.etree.ElementTree as ET


def extract_dexhand_trajectory_mujoco(data_idx="g0", hand_type="shadow", side="right", output_dir="output_trajectories_mujoco"):
    """
    Extract dexterous hand retargeted trajectory and object pose trajectory in MUJOCO coordinates.

    Args:
        data_idx: Data index (e.g., "g0" for grab_demo/102)
        hand_type: Hand type (shadow, inspire, inspireftp, allegro, artimano, xhand, wujihand)
        side: "right" or "left"
        output_dir: Directory to save output files

    Returns:
        dict: Contains retargeted hand trajectory and object trajectory in MUJOCO coordinates
    """

    print("="*80)
    print(f"EXTRACTING {hand_type.upper()} HAND TRAJECTORY - MUJOCO COORDINATES")
    print("="*80)

    # Create dexterous hand config
    dexhand = DexHandFactory.create_hand(hand_type, side)
    print(f"\nDexterous hand: {hand_type.capitalize()} Hand ({side})")
    print(f"Number of DOFs: {dexhand.n_dofs}")

    # Determine dataset type
    dataset_type = ManipDataFactory.dataset_type(data_idx)
    print(f"Dataset type: {dataset_type}")

    # Load dataset WITHOUT transformation to keep MUJOCO coordinates
    print("\nLoading dataset in MUJOCO coordinates...")
    demo_dataset = ManipDataFactory.create_data(
        manipdata_type=dataset_type,
        side=side,
        device="cpu",
        mujoco2gym_transf=torch.eye(4, device="cpu"),  # No transformation - keep MUJOCO coords
        dexhand=dexhand,
        verbose=True,
    )

    # Get processed data
    print(f"\nLoading data for index: {data_idx}")
    data = demo_dataset[data_idx]

    # Check if retargeted data exists
    has_retargeted = 'opt_wrist_pos' in data

    print("\n" + "="*80)
    print("DATA STATUS")
    print("="*80)

    if has_retargeted:
        print(f"\n✓ Retargeted {hand_type.capitalize()} Hand trajectory FOUND")
    else:
        print(f"\n✗ Retargeted {hand_type.capitalize()} Hand trajectory NOT FOUND")
        print("\nYou need to run the preprocessing first:")
        # Recommend iteration count based on hand type
        iter_count = 3000 if hand_type == "shadow" else 1000
        print(f"  python main/dataset/mano2dexhand.py --data_idx {data_idx} --dexhand {hand_type} --side {side} --headless --iter {iter_count}")
        print(f"\nNote: {hand_type.capitalize()} Hand uses {iter_count} iterations")
        return None

    # Get sequence length
    T = data['obj_trajectory'].shape[0]
    print(f"\nSequence length: {T} frames ({T/60:.2f} seconds at 60 FPS)")

    # =========================================================================
    # INVERSE TRANSFORMATION: IsaacGym → MUJOCO
    # =========================================================================

    # Build the transformation matrix (same as in mano2dexhand.py)
    mujoco2gym_transf = np.eye(4)
    mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    table_pos_z = 0.4
    table_half_height = 0.015
    table_surface_z = table_pos_z + table_half_height
    mujoco2gym_transf[:3, 3] = np.array([0, 0, table_surface_z])

    # Inverse transformation: IsaacGym → MUJOCO
    gym2mujoco_transf = np.linalg.inv(mujoco2gym_transf)

    print("\n" + "="*80)
    print("COORDINATE TRANSFORMATION")
    print("="*80)
    print(f"\nTable surface Z (IsaacGym): {table_surface_z}")
    print("\nMUJOCO → IsaacGym transformation:")
    print(mujoco2gym_transf)
    print("\nIsaacGym → MUJOCO inverse transformation:")
    print(gym2mujoco_transf)

    # =========================================================================
    # OBJECT TRAJECTORY (already in MUJOCO coordinates)
    # =========================================================================

    # Object data from dataset (already in MUJOCO coords since we loaded with identity transform)
    # Just use it as-is, no transformation needed
    object_trajectory = {
        'pose_matrices': data['obj_trajectory'].numpy(),  # (T, 4, 4)
        'positions': data['obj_trajectory'][:, :3, 3].numpy(),  # (T, 3)
        'rotations': data['obj_trajectory'][:, :3, :3].numpy(),  # (T, 3, 3)
        'orientations_aa': np.array([rotmat_to_aa(data['obj_trajectory'][i, :3, :3].numpy()) for i in range(T)]),  # (T, 3)
        'velocities': data['obj_velocity'].numpy(),  # (T, 3)
        'angular_velocities': data['obj_angular_velocity'].numpy(),  # (T, 3)
    }

    print(f"\nObject trajectory (MUJOCO coordinates - no transformation needed):")
    print(f"  Sample object position: {object_trajectory['positions'][0]}")

    # =========================================================================
    # ADDITIONAL ROTATION: Rotate 90° around X-axis
    # =========================================================================

    # Create 90° rotation around X-axis transformation
    additional_rot = np.eye(4)
    additional_rot[:3, :3] = aa_to_rotmat(np.array([np.pi / 2, 0, 0]))  # 90° around X-axis

    print(f"\nApplying additional 90° rotation around X-axis to both object and hand...")

    # Apply to object trajectory
    obj_poses_rotated = np.zeros_like(object_trajectory['pose_matrices'])
    for i in range(T):
        obj_poses_rotated[i] = additional_rot @ object_trajectory['pose_matrices'][i]

    # Update object trajectory
    object_trajectory = {
        'pose_matrices': obj_poses_rotated,
        'positions': obj_poses_rotated[:, :3, 3],
        'rotations': obj_poses_rotated[:, :3, :3],
        'orientations_aa': np.array([rotmat_to_aa(obj_poses_rotated[i, :3, :3]) for i in range(T)]),
        'velocities': (additional_rot[:3, :3] @ object_trajectory['velocities'].T).T,
        'angular_velocities': (additional_rot[:3, :3] @ object_trajectory['angular_velocities'].T).T,
    }

    print(f"  Object position after rotation: {object_trajectory['positions'][0]}")

    # =========================================================================
    # DEXTEROUS HAND TRAJECTORY (convert from IsaacGym to MUJOCO)
    # =========================================================================

    # Retargeted data in IsaacGym coordinates (from opt_* fields in data)
    opt_wrist_pos_gym = data['opt_wrist_pos'].numpy()  # (T, 3)
    opt_wrist_rot_gym = data['opt_wrist_rot'].numpy()  # (T, 3) axis-angle
    opt_dof_pos = data['opt_dof_pos'].numpy()  # (T, n_dofs)

    print(f"\nTransforming {hand_type.capitalize()} Hand trajectory from IsaacGym to MUJOCO...")
    print(f"  Sample hand wrist position (IsaacGym): {opt_wrist_pos_gym[0]}")

    # Convert wrist pose from IsaacGym to MUJOCO
    opt_wrist_pos_mujoco = np.zeros_like(opt_wrist_pos_gym)
    opt_wrist_rot_mujoco = np.zeros_like(opt_wrist_rot_gym)

    for i in range(T):
        # Build wrist pose in IsaacGym
        wrist_pose_gym = np.eye(4)
        wrist_pose_gym[:3, :3] = aa_to_rotmat(opt_wrist_rot_gym[i])
        wrist_pose_gym[:3, 3] = opt_wrist_pos_gym[i]

        # Transform to MUJOCO
        wrist_pose_mujoco = gym2mujoco_transf @ wrist_pose_gym

        # Extract position and rotation
        opt_wrist_pos_mujoco[i] = wrist_pose_mujoco[:3, 3]
        opt_wrist_rot_mujoco[i] = rotmat_to_aa(wrist_pose_mujoco[:3, :3])

    print(f"  Sample hand wrist position (MUJOCO): {opt_wrist_pos_mujoco[0]}")

    # Apply additional 90° X-axis rotation to hand trajectory
    print(f"\nApplying additional 90° rotation around X-axis to hand...")

    hand_wrist_pos_rotated = np.zeros_like(opt_wrist_pos_mujoco)
    hand_wrist_rot_rotated = np.zeros_like(opt_wrist_rot_mujoco)

    for i in range(T):
        # Build wrist pose in MUJOCO
        wrist_pose = np.eye(4)
        wrist_pose[:3, :3] = aa_to_rotmat(opt_wrist_rot_mujoco[i])
        wrist_pose[:3, 3] = opt_wrist_pos_mujoco[i]

        # Apply additional rotation
        wrist_pose_rotated = additional_rot @ wrist_pose

        # Extract position and rotation
        hand_wrist_pos_rotated[i] = wrist_pose_rotated[:3, 3]
        hand_wrist_rot_rotated[i] = rotmat_to_aa(wrist_pose_rotated[:3, :3])

    print(f"  Hand wrist position after rotation: {hand_wrist_pos_rotated[0]}")

    # Calculate velocities
    opt_wrist_velocity_rotated = np.zeros((T, 3))
    opt_wrist_velocity_rotated[1:] = (hand_wrist_pos_rotated[1:] - hand_wrist_pos_rotated[:-1]) * 60
    opt_wrist_velocity_rotated[0] = opt_wrist_velocity_rotated[1]

    opt_wrist_angular_velocity_rotated = np.zeros((T, 3))
    opt_wrist_angular_velocity_rotated[1:] = (hand_wrist_rot_rotated[1:] - hand_wrist_rot_rotated[:-1]) * 60
    opt_wrist_angular_velocity_rotated[0] = opt_wrist_angular_velocity_rotated[1]

    opt_dof_velocity = np.zeros_like(opt_dof_pos)
    opt_dof_velocity[1:] = (opt_dof_pos[1:] - opt_dof_pos[:-1]) * 60
    opt_dof_velocity[0] = opt_dof_velocity[1]

    hand_trajectory = {
        'wrist_positions': hand_wrist_pos_rotated,  # (T, 3)
        'wrist_rotations_aa': hand_wrist_rot_rotated,  # (T, 3) axis-angle
        'dof_positions': opt_dof_pos,  # (T, n_dofs) - joint angles don't change
        'wrist_velocities': opt_wrist_velocity_rotated,  # (T, 3)
        'wrist_angular_velocities': opt_wrist_angular_velocity_rotated,  # (T, 3)
        'dof_velocities': opt_dof_velocity,  # (T, n_dofs)
    }

    # =========================================================================
    # MANO REFERENCE (apply same rotation)
    # =========================================================================

    # Apply the same additional rotation to MANO reference for consistency
    mano_wrist_pos = data['wrist_pos'].numpy()
    mano_wrist_rot = data['wrist_rot'].numpy()

    mano_wrist_pos_rotated = (additional_rot[:3, :3] @ mano_wrist_pos.T).T + additional_rot[:3, 3]
    mano_wrist_rot_rotated = np.zeros_like(mano_wrist_rot)

    for i in range(T):
        rot_mat = aa_to_rotmat(mano_wrist_rot[i])
        rot_mat_rotated = additional_rot[:3, :3] @ rot_mat
        mano_wrist_rot_rotated[i] = rotmat_to_aa(rot_mat_rotated)

    # Transform MANO finger joints
    mano_joints_rotated = {}
    for joint_name, joint_positions in data['mano_joints'].items():
        joint_pos = joint_positions.numpy()  # (T, 3)
        # Apply the additional rotation
        joint_pos_rotated = (additional_rot[:3, :3] @ joint_pos.T).T + additional_rot[:3, 3]
        mano_joints_rotated[joint_name] = joint_pos_rotated

    # Transform MANO finger joint velocities
    mano_joints_velocity_rotated = {}
    for joint_name, joint_velocities in data['mano_joints_velocity'].items():
        joint_vel = joint_velocities.numpy()  # (T, 3)
        # Apply the additional rotation (rotation only, no translation for velocities)
        joint_vel_rotated = (additional_rot[:3, :3] @ joint_vel.T).T
        mano_joints_velocity_rotated[joint_name] = joint_vel_rotated

    mano_reference = {
        'wrist_positions': mano_wrist_pos_rotated,  # (T, 3)
        'wrist_rotations_aa': mano_wrist_rot_rotated,  # (T, 3)
        'finger_joints': mano_joints_rotated,
        'finger_joints_velocity': mano_joints_velocity_rotated,
        'fingertip_distances': data['tips_distance'].numpy(),  # (T, 5)
    }

    # =========================================================================
    # METADATA AND OBJECT MESH EXTRACTION
    # =========================================================================

    n_dofs = opt_dof_pos.shape[1]

    # Extract and copy object mesh files
    obj_urdf_path = data['obj_urdf_path']
    obj_mesh_paths = []
    obj_mesh_files_copied = []

    print(f"\nExtracting object mesh from URDF: {obj_urdf_path}")

    if os.path.exists(obj_urdf_path):
        try:
            # Parse URDF to find mesh files
            tree = ET.parse(obj_urdf_path)
            root = tree.getroot()

            # Find all mesh references in the URDF
            for mesh in root.iter('mesh'):
                mesh_filename = mesh.get('filename')
                if mesh_filename:
                    # Handle different path formats (package://, file://, or relative)
                    if mesh_filename.startswith('package://'):
                        mesh_filename = mesh_filename.replace('package://', '')
                    elif mesh_filename.startswith('file://'):
                        mesh_filename = mesh_filename.replace('file://', '')

                    # Resolve mesh path relative to URDF directory
                    urdf_dir = os.path.dirname(obj_urdf_path)
                    mesh_path = os.path.join(urdf_dir, mesh_filename)

                    if os.path.exists(mesh_path):
                        obj_mesh_paths.append(mesh_path)

            print(f"  Found {len(obj_mesh_paths)} mesh file(s)")

        except Exception as e:
            print(f"  Warning: Could not parse URDF: {e}")
    else:
        print(f"  Warning: URDF file not found at {obj_urdf_path}")

    metadata = {
        'sequence_length': T,
        'fps': 60,
        'duration_seconds': T / 60,
        'data_path': data['data_path'],
        'obj_id': data['obj_id'],
        'obj_urdf_path': obj_urdf_path,
        'obj_mesh_paths': obj_mesh_paths,
        'obj_mesh_files': [],  # Will be updated after copying
        'dexhand': hand_type,
        'side': side,
        'n_dofs': n_dofs,
        'coordinate_system': 'MUJOCO',
        'note': 'Data transformed to MUJOCO coordinates with additional 90° rotation around X-axis',
        'additional_rotation': '90° around X-axis',
    }

    # Combine all data
    output_data = {
        'object_trajectory': object_trajectory,
        'hand_trajectory': hand_trajectory,
        'mano_reference': mano_reference,
        'metadata': metadata,
    }

    # =========================================================================
    # PRINT STATISTICS
    # =========================================================================

    print("\n" + "="*80)
    print("TRAJECTORY STATISTICS (MUJOCO COORDINATES)")
    print("="*80)

    print(f"\nCoordinate System: MUJOCO")
    print(f"  - This is the ORIGINAL coordinate system from GRAB dataset")
    print(f"  - Object rotation is around Z-axis (vertical/up)")
    print(f"  - Units: meters, radians")

    print(f"\n1. Object Trajectory:")
    obj_pos = object_trajectory['positions']
    print(f"   - Position range: X[{obj_pos[:,0].min():.6f}, {obj_pos[:,0].max():.6f}], "
          f"Y[{obj_pos[:,1].min():.6f}, {obj_pos[:,1].max():.6f}], "
          f"Z[{obj_pos[:,2].min():.6f}, {obj_pos[:,2].max():.6f}]")
    print(f"   - Total displacement: {np.linalg.norm(obj_pos[-1] - obj_pos[0]):.6f} m")

    # Analyze rotation
    obj_rot = object_trajectory['orientations_aa']
    rotation_axes = []
    for i in range(T):
        angle = np.linalg.norm(obj_rot[i])
        if angle > 1e-6:
            axis = obj_rot[i] / angle
        else:
            axis = np.array([0, 0, 1])
        rotation_axes.append(axis)
    rotation_axes = np.array(rotation_axes)

    print(f"   - Rotation axis (mean): X={rotation_axes[:,0].mean():.4f}, "
          f"Y={rotation_axes[:,1].mean():.4f}, Z={rotation_axes[:,2].mean():.4f}")
    print(f"   - Expected: rotation around Z-axis [0, 0, 1] in MUJOCO coords")

    # Calculate total rotation
    R_total = aa_to_rotmat(obj_rot[-1]) @ aa_to_rotmat(obj_rot[0]).T
    total_angle = R.from_matrix(R_total).magnitude() * 180 / np.pi
    print(f"   - Total rotation: {total_angle:.2f}°")

    print(f"\n2. {hand_type.capitalize()} Hand Trajectory (MUJOCO coords):")
    wrist_pos = hand_trajectory['wrist_positions']
    print(f"   - Wrist position range: X[{wrist_pos[:,0].min():.6f}, {wrist_pos[:,0].max():.6f}], "
          f"Y[{wrist_pos[:,1].min():.6f}, {wrist_pos[:,1].max():.6f}], "
          f"Z[{wrist_pos[:,2].min():.6f}, {wrist_pos[:,2].max():.6f}]")
    print(f"   - Total wrist displacement: {np.linalg.norm(wrist_pos[-1] - wrist_pos[0]):.6f} m")
    print(f"   - Number of DOFs: {n_dofs}")

    dof_pos = hand_trajectory['dof_positions']
    print(f"   - DOF range: [{dof_pos.min():.3f}, {dof_pos.max():.3f}] radians")

    print(f"\n3. Retargeting Quality:")
    mano_wrist = mano_reference['wrist_positions']
    wrist_error = np.linalg.norm(wrist_pos - mano_wrist, axis=1)
    print(f"   - Wrist deviation from MANO:")
    print(f"     * Mean: {wrist_error.mean():.4f} m")
    print(f"     * Max: {wrist_error.max():.4f} m")
    print(f"     * Std: {wrist_error.std():.4f} m")

    print(f"\n4. Contact Information:")
    tips_dist = mano_reference['fingertip_distances']
    print(f"   - Fingertip-to-object distance:")
    print(f"     * Mean: {tips_dist.mean():.4f} m")
    print(f"     * Min: {tips_dist.min():.4f} m (closest contact)")
    print(f"     * Max: {tips_dist.max():.4f} m")

    print(f"\n5. Hand-Object Spatial Relationship:")
    hand_obj_diff = wrist_pos - obj_pos
    print(f"   - Initial positions:")
    print(f"     * Object: [{obj_pos[0,0]:.6f}, {obj_pos[0,1]:.6f}, {obj_pos[0,2]:.6f}]")
    print(f"     * Hand wrist: [{wrist_pos[0,0]:.6f}, {wrist_pos[0,1]:.6f}, {wrist_pos[0,2]:.6f}]")
    print(f"   - Hand-object distance:")
    print(f"     * Mean: {np.linalg.norm(hand_obj_diff, axis=1).mean():.4f} m")
    print(f"     * Initial: {np.linalg.norm(hand_obj_diff[0]):.4f} m")

    # =========================================================================
    # SAVE OUTPUT FILES
    # =========================================================================

    os.makedirs(output_dir, exist_ok=True)

    # Copy object mesh files to output directory with object_id-based naming
    if obj_mesh_paths:
        print(f"\nCopying object mesh files to output directory...")
        obj_id_safe = data['obj_id']

        for mesh_path in obj_mesh_paths:
            mesh_filename = os.path.basename(mesh_path)

            # Determine if this is visual or collision mesh
            if '_coacd' in mesh_filename:
                # Collision mesh - always .obj format
                new_filename = f"{obj_id_safe}_collision.obj"
            else:
                # Visual mesh - preserve original extension (.ply or .obj)
                ext = os.path.splitext(mesh_path)[1]
                new_filename = f"{obj_id_safe}_visual{ext}"

            output_mesh_path = os.path.join(output_dir, new_filename)
            try:
                shutil.copy2(mesh_path, output_mesh_path)
                obj_mesh_files_copied.append(output_mesh_path)
                print(f"  ✓ Copied: {mesh_filename} → {new_filename}")
            except Exception as e:
                print(f"  ✗ Failed to copy {mesh_filename}: {e}")

        # Update metadata with copied mesh file paths
        metadata['obj_mesh_files'] = obj_mesh_files_copied

    # Save complete data as pickle
    output_path = os.path.join(output_dir, f"{hand_type}_hand_trajectory_{data_idx}_mujoco.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"\n✓ Saved complete trajectory to: {output_path}")

    print("\n" + "="*80)
    print("OUTPUT FILES")
    print("="*80)
    print(f"\n1. Complete data (pickle): {output_path}")
    if obj_mesh_files_copied:
        print(f"\n2. Object mesh files:")
        for mesh_file in obj_mesh_files_copied:
            print(f"   - {os.path.basename(mesh_file)}")
    else:
        print(f"\n2. Object mesh files: None found or copied")

    print("\n" + "="*80)
    print("COORDINATE SYSTEM: MUJOCO")
    print("="*80)
    print("\nAll trajectories are in MUJOCO coordinates:")
    print("  - This is the ORIGINAL coordinate system from GRAB dataset")
    print("  - Object rotation is around Z-axis (vertical/up)")
    print("  - Units: meters, radians")

    print("\n" + "="*80)

    return output_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract dexterous hand and object trajectories in MUJOCO coordinates"
    )
    parser.add_argument("--data_idx", type=str, default="g0", help="Data index (default: g0)")
    parser.add_argument("--hand_type", type=str, default="wujihand",
                        choices=["shadow", "inspire", "inspireftp", "allegro", "artimano", "xhand", "wujihand"],
                        help="Hand type (default: shadow)")
    parser.add_argument("--side", type=str, default="right",
                        choices=["right", "left"],
                        help="Hand side (default: right)")
    parser.add_argument("--output_dir", type=str, default="output_trajectories_mujoco",
                        help="Output directory (default: output_trajectories_mujoco)")
    args = parser.parse_args()

    # Extract trajectories in MUJOCO coordinates
    result = extract_dexhand_trajectory_mujoco(
        data_idx=args.data_idx,
        hand_type=args.hand_type,
        side=args.side,
        output_dir=args.output_dir
    )

    if result is None:
        print("\n" + "="*80)
        print("ERROR")
        print("="*80)
        print("\nCould not extract trajectories. Please check the error messages above.")
    else:
        print(f"\n✓ SUCCESS! {args.hand_type.capitalize()} Hand and object trajectories extracted in MUJOCO coordinates.")
