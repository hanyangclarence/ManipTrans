#!/usr/bin/env python3
"""
Batch process OakInk-v2 objects through CoACD and generate URDF files.

This script automates the creation of URDF files for OakInk-v2 objects by:
1. Calling coacd_process.py to decompose object meshes into collision meshes
2. Automatically generating URDF files that reference both visual and collision meshes

Usage:
    # Process specific objects
    python data_utils/create_oakink_urdfs.py --objects S20005 O02@0032@00001

    # Process all objects in object_preview
    python data_utils/create_oakink_urdfs.py --all

    # Use custom CoACD parameters
    python data_utils/create_oakink_urdfs.py --objects S20005 --threshold 0.02 --max-convex-hull 64

    # Skip CoACD and only generate URDFs (if CoACD files already exist)
    python data_utils/create_oakink_urdfs.py --objects S20005 --skip-coacd
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path


def generate_urdf(
    obj_id,
    visual_mesh_filename,
    collision_mesh_filename,
    output_path,
):
    """
    Generate a URDF file for an object.

    Args:
        obj_id: Object identifier
        visual_mesh_filename: Filename of visual mesh (relative to URDF)
        collision_mesh_filename: Filename of collision mesh (relative to URDF)
        output_path: Path to save URDF file
    """
    urdf_content = f"""<?xml version="1.0"?>
<robot name="{obj_id}">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.000059" ixy="0" ixz="0" iyy="0.000058" iyz="0" izz="0.000106"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{visual_mesh_filename}" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{collision_mesh_filename}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
"""

    with open(output_path, 'w') as f:
        f.write(urdf_content)

    print(f"  ✓ Created URDF: {output_path}")


def process_object(
    obj_id,
    object_preview_dir,
    coacd_output_dir,
    skip_coacd=False,
    coacd_args=None,
):
    """
    Process a single object through CoACD and generate URDF.

    Args:
        obj_id: Object identifier (e.g., "S20005")
        object_preview_dir: Base directory for object_preview
        coacd_output_dir: Base directory for coacd_object_preview
        skip_coacd: If True, skip CoACD processing (assumes files exist)
        coacd_args: List of additional arguments for coacd_process.py

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Processing object: {obj_id}")
    print(f"{'='*80}")

    # Find source mesh file
    obj_dir = os.path.join(object_preview_dir, obj_id)
    if not os.path.exists(obj_dir):
        print(f"  ✗ Object directory not found: {obj_dir}")
        return False

    # Look for mesh files (.obj or .ply)
    mesh_files = []
    for ext in ['.obj', '.ply']:
        mesh_files.extend(list(Path(obj_dir).glob(f'*{ext}')))

    # Filter out material files
    mesh_files = [f for f in mesh_files if not f.name.endswith('.mtl')]

    if not mesh_files:
        print(f"  ✗ No mesh files found in {obj_dir}")
        return False

    # Use the first mesh file found
    visual_mesh_path = str(mesh_files[0])
    visual_mesh_name = os.path.basename(visual_mesh_path)
    mesh_base_name = os.path.splitext(visual_mesh_name)[0]
    mesh_ext = os.path.splitext(visual_mesh_name)[1]

    print(f"  Found visual mesh: {visual_mesh_name}")

    # Create output directory
    output_dir = os.path.join(coacd_output_dir, obj_id)
    os.makedirs(output_dir, exist_ok=True)

    # Define collision mesh path
    collision_mesh_name = f"{mesh_base_name}_coacd.obj"
    collision_mesh_path = os.path.join(output_dir, collision_mesh_name)

    # Copy visual mesh to output directory
    visual_mesh_output = os.path.join(output_dir, visual_mesh_name)
    if not os.path.exists(visual_mesh_output):
        shutil.copy2(visual_mesh_path, visual_mesh_output)
        print(f"  ✓ Copied visual mesh to: {visual_mesh_output}")
    else:
        print(f"  ○ Visual mesh already exists: {visual_mesh_output}")

    # Run CoACD processing
    if not skip_coacd:
        if os.path.exists(collision_mesh_path + ".done"):
            print(f"  ○ CoACD already processed (found .done file)")
        else:
            print(f"  Running CoACD decomposition...")

            # Build command to call coacd_process.py
            cmd = [
                sys.executable,
                "maniptrans_envs/lib/utils/coacd_process.py",
                "-i", visual_mesh_path,
                "-o", collision_mesh_path,
            ]

            # Add additional CoACD arguments
            if coacd_args:
                cmd.extend(coacd_args)

            print(f"  Command: {' '.join(cmd)}")

            try:
                result = subprocess.run(cmd, check=True)
                print(result.stdout)
                print(f"  ✓ CoACD completed: {collision_mesh_name}")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ CoACD failed with error:")
                print(e.stderr)
                return False
    else:
        if not os.path.exists(collision_mesh_path):
            print(f"  ✗ Collision mesh not found: {collision_mesh_path}")
            print(f"     Run without --skip-coacd to generate it")
            return False
        else:
            print(f"  ○ Using existing collision mesh")

    # Generate URDF
    urdf_path = os.path.join(output_dir, f"{mesh_base_name}.urdf")
    generate_urdf(
        obj_id=obj_id,
        visual_mesh_filename=visual_mesh_name,
        collision_mesh_filename=collision_mesh_name,
        output_path=urdf_path,
    )

    print(f"\n  ✓ Successfully processed {obj_id}")
    return True


def find_all_objects(object_preview_dir):
    """
    Find all object directories in object_preview.

    Args:
        object_preview_dir: Base directory for object_preview

    Returns:
        list: List of object IDs
    """
    if not os.path.exists(object_preview_dir):
        return []

    obj_ids = []
    for item in os.listdir(object_preview_dir):
        item_path = os.path.join(object_preview_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains mesh files
            has_mesh = False
            for ext in ['.obj', '.ply']:
                if list(Path(item_path).glob(f'*{ext}')):
                    has_mesh = True
                    break
            if has_mesh:
                obj_ids.append(item)

    return sorted(obj_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Process OakInk-v2 objects through CoACD and generate URDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--objects",
        nargs="+",
        help="Specific object IDs to process (e.g., S20005 O02@0032@00001)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all objects in object_preview directory",
    )

    parser.add_argument(
        "--object-preview-dir",
        type=str,
        default="data/OakInk-v2/object_preview/align_ds",
        help="Directory containing source object meshes (default: data/OakInk-v2/object_preview/align_ds)",
    )

    parser.add_argument(
        "--coacd-output-dir",
        type=str,
        default="data/OakInk-v2/coacd_object_preview/align_ds",
        help="Directory to save CoACD processed meshes and URDFs (default: data/OakInk-v2/coacd_object_preview/align_ds)",
    )

    parser.add_argument(
        "--skip-coacd",
        action="store_true",
        help="Skip CoACD processing and only generate URDFs (assumes collision meshes already exist)",
    )

    # CoACD parameters
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.07,
        help="CoACD threshold (0.01=fine-grained, 1=coarse) (default: 0.07)",
    )

    parser.add_argument(
        "-c", "--max-convex-hull",
        type=int,
        default=32,
        help="Max number of convex hulls (default: 32)",
    )

    parser.add_argument(
        "-mi", "--mcts-iteration",
        type=int,
        default=2000,
        help="MCTS iterations (default: 2000)",
    )

    parser.add_argument(
        "-md", "--mcts-max-depth",
        type=int,
        default=5,
        help="MCTS max depth (default: 5)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for CoACD (default: 1)",
    )

    args = parser.parse_args()

    # Determine which objects to process
    if args.all:
        obj_ids = find_all_objects(args.object_preview_dir)
        if not obj_ids:
            print(f"No objects found in {args.object_preview_dir}")
            return 1
        print(f"\nFound {len(obj_ids)} objects to process")
    elif args.objects:
        obj_ids = args.objects
        print(f"\nProcessing {len(obj_ids)} specified objects")
    else:
        print("Error: Must specify either --objects or --all")
        parser.print_help()
        return 1

    # Build CoACD arguments
    coacd_args = [
        "--max-convex-hull", str(args.max_convex_hull),
        "--seed", str(args.seed),
        "-mi", str(args.mcts_iteration),
        "-md", str(args.mcts_max_depth),
        "-t", str(args.threshold),
    ]

    # Print configuration
    print("\nConfiguration:")
    print(f"  Object preview dir: {args.object_preview_dir}")
    print(f"  CoACD output dir: {args.coacd_output_dir}")
    print(f"  Skip CoACD: {args.skip_coacd}")
    if not args.skip_coacd:
        print(f"\nCoACD parameters:")
        print(f"  Threshold: {args.threshold}")
        print(f"  Max convex hulls: {args.max_convex_hull}")
        print(f"  MCTS iterations: {args.mcts_iteration}")
        print(f"  MCTS max depth: {args.mcts_max_depth}")
        print(f"  Seed: {args.seed}")

    # Process objects
    print("\n" + "="*80)
    print("STARTING PROCESSING")
    print("="*80)

    success_count = 0
    failed_objects = []

    for i, obj_id in enumerate(obj_ids, 1):
        print(f"\n[{i}/{len(obj_ids)}]")
        success = process_object(
            obj_id=obj_id,
            object_preview_dir=args.object_preview_dir,
            coacd_output_dir=args.coacd_output_dir,
            skip_coacd=args.skip_coacd,
            coacd_args=coacd_args if not args.skip_coacd else None,
        )
        if success:
            success_count += 1
        else:
            failed_objects.append(obj_id)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Successfully processed: {success_count}/{len(obj_ids)}")

    if failed_objects:
        print(f"✗ Failed: {len(failed_objects)}")
        print("\nFailed objects:")
        for obj_id in failed_objects:
            print(f"  - {obj_id}")
        return 1

    print("\nAll objects processed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
