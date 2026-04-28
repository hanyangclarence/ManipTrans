"""
Extract retargeted dexterous hand trajectory and object pose trajectory.
This script will load the data, run retargeting if needed, and save the trajectories.
Both hand and object trajectories are saved in IsaacGym coordinates.
Supports all hand types: shadow, inspire, inspireftp, allegro, artimano, xhand, wujihand.
"""
from isaacgym import gymapi, gymtorch, gymutil
import os
import numpy as np
import pickle
from main.dataset.factory import ManipDataFactory
from main.dataset.transform import aa_to_rotmat
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
import torch


def extract_dexhand_trajectory(data_idx="g0", hand_type="shadow", side="right", output_dir="output_trajectories"):
    """
    Extract dexterous hand retargeted trajectory and object pose trajectory.

    Args:
        data_idx: Data index (e.g., "g0" for grab_demo/102)
        hand_type: Hand type (shadow, inspire, inspireftp, allegro, artimano, xhand, wujihand)
        side: "right" or "left"
        output_dir: Directory to save output files

    Returns:
        dict: Contains retargeted hand trajectory and object trajectory
    """

    print("="*80)
    print(f"EXTRACTING {hand_type.upper()} HAND TRAJECTORY")
    print("="*80)

    # Create dexterous hand config
    dexhand = DexHandFactory.create_hand(hand_type, side)
    print(f"\nDexterous hand: {hand_type.capitalize()} Hand ({side})")
    print(f"Number of DOFs: {dexhand.n_dofs}")

    # Determine dataset type
    dataset_type = ManipDataFactory.dataset_type(data_idx)
    print(f"Dataset type: {dataset_type}")

    # Create mujoco2gym transformation (same as used during retargeting)
    mujoco2gym_transf = torch.eye(4, device="cpu")
    mujoco2gym_transf[:3, :3] = torch.tensor(
        aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0])),
        dtype=torch.float32
    )
    table_pos_z = 0.4
    table_half_height = 0.015
    table_surface_z = table_pos_z + table_half_height
    mujoco2gym_transf[:3, 3] = torch.tensor([0, 0, table_surface_z], dtype=torch.float32)

    print("\nMUJOCO to IsaacGym Transformation:")
    print(f"  Table surface Z: {table_surface_z}")
    print(f"  Using same transformation as during retargeting")

    # Load the dataset with transformation
    print("\nLoading dataset...")
    demo_dataset = ManipDataFactory.create_data(
        manipdata_type=dataset_type,
        side=side,
        device="cpu",
        mujoco2gym_transf=mujoco2gym_transf,
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

    # Extract trajectories
    T = data['obj_trajectory'].shape[0]
    print(f"\nSequence length: {T} frames ({T/60:.2f} seconds at 60 FPS)")

    # Object trajectory
    obj_trajectory = {
        'pose_matrices': data['obj_trajectory'].numpy(),  # (T, 4, 4)
        'positions': data['obj_trajectory'][:, :3, 3].numpy(),  # (T, 3)
        'rotations': data['obj_trajectory'][:, :3, :3].numpy(),  # (T, 3, 3)
        'velocities': data['obj_velocity'].numpy(),  # (T, 3)
        'angular_velocities': data['obj_angular_velocity'].numpy(),  # (T, 3)
    }

    # Dexterous hand retargeted trajectory
    hand_trajectory = {
        'wrist_positions': data['opt_wrist_pos'].numpy(),  # (T, 3)
        'wrist_rotations_aa': data['opt_wrist_rot'].numpy(),  # (T, 3) axis-angle
        'dof_positions': data['opt_dof_pos'].numpy(),  # (T, n_dofs) joint angles
        'wrist_velocities': data['opt_wrist_velocity'].numpy(),  # (T, 3)
        'wrist_angular_velocities': data['opt_wrist_angular_velocity'].numpy(),  # (T, 3)
        'dof_velocities': data['opt_dof_velocity'].numpy(),  # (T, n_dofs)
    }

    # MANO reference (original human hand trajectory)
    mano_reference = {
        'wrist_positions': data['wrist_pos'].numpy(),  # (T, 3)
        'wrist_rotations_aa': data['wrist_rot'].numpy(),  # (T, 3)
        'finger_joints': {k: v.numpy() for k, v in data['mano_joints'].items()},
        'fingertip_distances': data['tips_distance'].numpy(),  # (T, 5)
    }

    # Metadata
    metadata = {
        'sequence_length': T,
        'fps': 60,
        'duration_seconds': T / 60,
        'data_path': data['data_path'],
        'obj_id': data['obj_id'],
        'obj_urdf_path': data['obj_urdf_path'],
        'dexhand': hand_type,
        'side': side,
        'n_dofs': dexhand.n_dofs,
        'coordinate_system': 'IsaacGym',
        'table_surface_z': table_surface_z,
    }

    # Combine all data
    output_data = {
        'object_trajectory': obj_trajectory,
        'hand_trajectory': hand_trajectory,
        'mano_reference': mano_reference,
        'metadata': metadata,
    }

    # Print statistics
    print("\n" + "="*80)
    print("TRAJECTORY STATISTICS (IsaacGym Coordinates)")
    print("="*80)

    print(f"\nCoordinate System: IsaacGym")
    print(f"  - X: forward")
    print(f"  - Y: left")
    print(f"  - Z: up")
    print(f"  - Ground plane at Z=0")
    print(f"  - Table surface at Z={table_surface_z:.3f}")

    print(f"\n1. Object Trajectory:")
    obj_pos = obj_trajectory['positions']
    print(f"   - Position range: X[{obj_pos[:,0].min():.3f}, {obj_pos[:,0].max():.3f}], "
          f"Y[{obj_pos[:,1].min():.3f}, {obj_pos[:,1].max():.3f}], "
          f"Z[{obj_pos[:,2].min():.3f}, {obj_pos[:,2].max():.3f}]")
    print(f"   - Total displacement: {np.linalg.norm(obj_pos[-1] - obj_pos[0]):.3f} m")
    print(f"   - Max velocity: {np.linalg.norm(obj_trajectory['velocities'], axis=1).max():.3f} m/s")

    print(f"\n2. {hand_type.capitalize()} Hand Trajectory:")
    wrist_pos = hand_trajectory['wrist_positions']
    print(f"   - Wrist position range: X[{wrist_pos[:,0].min():.3f}, {wrist_pos[:,0].max():.3f}], "
          f"Y[{wrist_pos[:,1].min():.3f}, {wrist_pos[:,1].max():.3f}], "
          f"Z[{wrist_pos[:,2].min():.3f}, {wrist_pos[:,2].max():.3f}]")
    print(f"   - Total wrist displacement: {np.linalg.norm(wrist_pos[-1] - wrist_pos[0]):.3f} m")
    print(f"   - Number of DOFs: {hand_trajectory['dof_positions'].shape[1]}")

    dof_pos = hand_trajectory['dof_positions']
    print(f"   - DOF range: [{dof_pos.min():.3f}, {dof_pos.max():.3f}] radians")
    print(f"   - DOF mean: {dof_pos.mean():.3f} radians")

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
    print(f"     * Object: [{obj_pos[0,0]:.3f}, {obj_pos[0,1]:.3f}, {obj_pos[0,2]:.3f}]")
    print(f"     * Hand wrist: [{wrist_pos[0,0]:.3f}, {wrist_pos[0,1]:.3f}, {wrist_pos[0,2]:.3f}]")
    print(f"   - Hand-object distance:")
    print(f"     * Mean: {np.linalg.norm(hand_obj_diff, axis=1).mean():.4f} m")
    print(f"     * Initial: {np.linalg.norm(hand_obj_diff[0]):.4f} m")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save complete data as pickle
    output_path = os.path.join(output_dir, f"{hand_type}_hand_trajectory_{data_idx}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"\n✓ Saved complete trajectory to: {output_path}")

    # Save individual components as numpy arrays for easy loading
    np.save(os.path.join(output_dir, f"object_poses_{data_idx}.npy"),
            obj_trajectory['pose_matrices'])
    np.save(os.path.join(output_dir, f"object_positions_{data_idx}.npy"),
            obj_trajectory['positions'])
    np.save(os.path.join(output_dir, f"object_velocities_{data_idx}.npy"),
            obj_trajectory['velocities'])
    np.save(os.path.join(output_dir, f"{hand_type}_wrist_pos_{data_idx}.npy"),
            hand_trajectory['wrist_positions'])
    np.save(os.path.join(output_dir, f"{hand_type}_wrist_rot_{data_idx}.npy"),
            hand_trajectory['wrist_rotations_aa'])
    np.save(os.path.join(output_dir, f"{hand_type}_dof_pos_{data_idx}.npy"),
            hand_trajectory['dof_positions'])

    print(f"✓ Saved individual arrays to: {output_dir}/")

    print("\n" + "="*80)
    print("OUTPUT FILES")
    print("="*80)
    print(f"\n1. Complete data (pickle): {output_path}")
    print(f"\n2. Numpy arrays:")
    print(f"   - object_poses_{data_idx}.npy           : (T, 4, 4) - Object pose matrices")
    print(f"   - object_positions_{data_idx}.npy       : (T, 3) - Object positions")
    print(f"   - object_velocities_{data_idx}.npy      : (T, 3) - Object velocities")
    print(f"   - {hand_type}_wrist_pos_{data_idx}.npy  : (T, 3) - {hand_type.capitalize()} wrist positions")
    print(f"   - {hand_type}_wrist_rot_{data_idx}.npy  : (T, 3) - {hand_type.capitalize()} wrist rotations (axis-angle)")
    print(f"   - {hand_type}_dof_pos_{data_idx}.npy    : (T, {dexhand.n_dofs}) - {hand_type.capitalize()} joint angles")

    print("\n" + "="*80)
    print("USAGE EXAMPLE")
    print("="*80)
    print(f"""
import numpy as np
import pickle

# Load complete data
with open('{output_path}', 'rb') as f:
    data = pickle.load(f)

# Access object trajectory
obj_poses = data['object_trajectory']['pose_matrices']  # (T, 4, 4)
obj_positions = data['object_trajectory']['positions']   # (T, 3)

# Access {hand_type.capitalize()} Hand trajectory
hand_wrist_pos = data['hand_trajectory']['wrist_positions']  # (T, 3)
hand_dof_pos = data['hand_trajectory']['dof_positions']      # (T, {dexhand.n_dofs})

# Or load individual numpy arrays
obj_poses = np.load('{output_dir}/object_poses_{data_idx}.npy')
hand_dof = np.load('{output_dir}/{hand_type}_dof_pos_{data_idx}.npy')
    """)

    return output_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract dexterous hand and object trajectories")
    parser.add_argument("--data_idx", type=str, default="g0", help="Data index (default: g0)")
    parser.add_argument("--hand_type", type=str, default="shadow",
                        choices=["shadow", "inspire", "inspireftp", "allegro", "artimano", "xhand", "wujihand"],
                        help="Hand type (default: shadow)")
    parser.add_argument("--side", type=str, default="right",
                        choices=["right", "left"],
                        help="Hand side (default: right)")
    parser.add_argument("--output_dir", type=str, default="output_trajectories",
                        help="Output directory (default: output_trajectories)")
    args = parser.parse_args()

    # Extract trajectories
    result = extract_dexhand_trajectory(
        data_idx=args.data_idx,
        hand_type=args.hand_type,
        side=args.side,
        output_dir=args.output_dir
    )

    if result is None:
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Make sure you have downloaded the imitator checkpoints:")
        print("   Download from: https://huggingface.co/LiKailin/ManipTrans")
        print("   Place in: assets/")
        print("   Required files:")
        print(f"     - imitator_rh_{args.hand_type}.pth  (or imitator_pid_rh_{args.hand_type}.pth)")
        print(f"     - imitator_lh_{args.hand_type}.pth  (or imitator_pid_lh_{args.hand_type}.pth)")
        print("\n2. Run the retargeting preprocessing:")
        iter_count = 3000 if args.hand_type == "shadow" else 1000
        print(f"   python main/dataset/mano2dexhand.py --data_idx {args.data_idx} --dexhand {args.hand_type} --side {args.side} --headless --iter {iter_count}")
        print("\n3. Then run this script again:")
        print(f"   python data_utils/extract_shadow_trajectory.py --data_idx {args.data_idx} --hand_type {args.hand_type} --side {args.side}")
    else:
        print(f"\n✓ SUCCESS! {args.hand_type.capitalize()} Hand and object trajectories extracted.")
