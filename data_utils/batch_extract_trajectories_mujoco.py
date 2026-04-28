"""
Batch extract all retargeted dexterous hand trajectories in MUJOCO coordinates.
This script finds all retargeted data files and extracts them using the extract_shadow_trajectory_mujoco module.
"""

import os
import glob
import argparse
from pathlib import Path
from extract_shadow_trajectory_mujoco import extract_dexhand_trajectory_mujoco


def parse_retargeted_filename(pkl_path, hand_type, dataset_type):
    """
    Parse retargeted pkl filename to extract data index.

    Args:
        pkl_path: Path to the retargeted pkl file
        hand_type: Hand type (shadow, wujihand, etc.)
        dataset_type: Dataset type (grab_demo, OakInk-v2)

    Returns:
        Data index string (e.g., "g0" for grab_demo, "cffb9@0" for OakInk-v2)
    """
    filename = os.path.basename(pkl_path)

    if dataset_type == "grab_demo":
        # For grab_demo: 102_sv_dict.pkl -> "g0"
        return "g0"

    elif dataset_type == "OakInk-v2":
        # For OakInk-v2: scene_03__A007++seq__cffb94f7d9de086d3604__2023-04-20-15-33-14@0.pkl
        # Format: scene_XX__AYYY++seq__HASH__TIMESTAMP@STAGE.pkl
        parts = filename.replace(".pkl", "").split("__")
        if len(parts) >= 4:
            # parts[2] is the sequence hash (e.g., "cffb94f7d9de086d3604")
            seq_hash = parts[2]
            # parts[3] contains timestamp and stage (e.g., "2023-04-20-15-33-14@0")
            timestamp_stage = parts[3]
            # Extract stage number after @
            if "@" in timestamp_stage:
                stage = timestamp_stage.split("@")[1]
            else:
                stage = "0"
            # Use first 5 characters of hash + @ + stage
            return f"{seq_hash[:5]}@{stage}"

    return None


def get_dataset_type_from_path(pkl_path):
    """Extract dataset type from path."""
    if "grab_demo" in pkl_path:
        return "grab_demo"
    elif "OakInk-v2" in pkl_path or "oakink2" in pkl_path.lower():
        return "OakInk-v2"
    return None


def get_hand_type_from_path(pkl_path):
    """Extract hand type from path (e.g., mano2shadow_rh -> shadow)."""
    dirname = os.path.basename(os.path.dirname(pkl_path))
    # dirname should be like "mano2shadow_rh" or "mano2wujihand_rh"
    if "mano2" in dirname:
        parts = dirname.split("_")
        if len(parts) >= 1:
            hand_part = parts[0].replace("mano2", "")
            return hand_part
    return None


def get_side_from_path(pkl_path):
    """Extract side from path (e.g., mano2shadow_rh -> right)."""
    dirname = os.path.basename(os.path.dirname(pkl_path))
    if dirname.endswith("_rh"):
        return "right"
    elif dirname.endswith("_lh"):
        return "left"
    return "right"  # default


def find_all_retargeted_files(retargeting_dir):
    """
    Find all retargeted pkl files in the retargeting directory.

    Args:
        retargeting_dir: Root retargeting directory

    Returns:
        List of tuples: (pkl_path, hand_type, side, dataset_type, data_idx)
    """
    pkl_files = glob.glob(os.path.join(retargeting_dir, "**/*.pkl"), recursive=True)

    results = []
    for pkl_path in pkl_files:
        dataset_type = get_dataset_type_from_path(pkl_path)
        hand_type = get_hand_type_from_path(pkl_path)
        side = get_side_from_path(pkl_path)

        if dataset_type and hand_type:
            data_idx = parse_retargeted_filename(pkl_path, hand_type, dataset_type)
            if data_idx:
                results.append((pkl_path, hand_type, side, dataset_type, data_idx))

    return results


def batch_extract_trajectories(
    retargeting_dir="data/retargeting",
    output_dir="/scratch4/workspace/hanyang_umass_edu-maniptrans/data/output_trajectories_mujoco",
    hand_types=None,
    dataset_types=None,
    max_count=None,
    skip_existing=True,
    start_idx=0.0,
    end_idx=1.0
):
    """
    Batch extract all retargeted trajectories.

    Args:
        retargeting_dir: Directory containing retargeted data
        output_dir: Output directory for extracted trajectories
        hand_types: List of hand types to process (None = all)
        dataset_types: List of dataset types to process (None = all)
        max_count: Maximum number of files to process (None = all)
        skip_existing: Skip files that already exist in output directory
    """

    print("="*80)
    print("BATCH TRAJECTORY EXTRACTION - MUJOCO COORDINATES")
    print("="*80)

    # Find all retargeted files
    print(f"\nScanning retargeting directory: {retargeting_dir}")
    all_files = find_all_retargeted_files(retargeting_dir)

    # Filter by hand types and dataset types if specified
    if hand_types:
        all_files = [f for f in all_files if f[1] in hand_types]
    if dataset_types:
        all_files = [f for f in all_files if f[3] in dataset_types]

    print(f"Found {len(all_files)} retargeted files to process")
    start_index = int(len(all_files) * start_idx)
    end_index = int(len(all_files) * end_idx)
    all_files = all_files[start_index:end_index]
    print(f"Processing files from index {start_index} to {end_index} (total {len(all_files)})")

    # Group by hand type and dataset type for summary
    summary = {}
    for _, hand_type, _, dataset_type, _ in all_files:
        key = f"{dataset_type}/{hand_type}"
        summary[key] = summary.get(key, 0) + 1

    print("\nSummary:")
    for key, count in sorted(summary.items()):
        print(f"  {key}: {count} files")

    # Limit if specified
    if max_count is not None:
        all_files = all_files[:max_count]
        print(f"\nLimited to first {max_count} files")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each file
    print("\n" + "="*80)
    print("PROCESSING FILES")
    print("="*80)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, (pkl_path, hand_type, side, dataset_type, data_idx) in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] Processing: {dataset_type}/{hand_type}/{data_idx}")

        # Check if output file already exists
        output_filename = f"{hand_type}_hand_trajectory_{data_idx}_mujoco.pkl"
        output_path = os.path.join(output_dir, output_filename)

        if skip_existing and os.path.exists(output_path):
            print(f"  ⊘ Skipping (already exists): {output_filename}")
            skip_count += 1
            continue

        try:
            result = extract_dexhand_trajectory_mujoco(
                data_idx=data_idx,
                hand_type=hand_type,
                side=side,
                output_dir=output_dir
            )

            if result is not None:
                print(f"  ✓ Success: {output_filename}")
                success_count += 1
            else:
                print(f"  ✗ Failed: No retargeted data found")
                fail_count += 1

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            fail_count += 1

    # Print final summary
    print("\n" + "="*80)
    print("BATCH EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nTotal files: {len(all_files)}")
    print(f"  ✓ Successfully extracted: {success_count}")
    print(f"  ⊘ Skipped (existing): {skip_count}")
    print(f"  ✗ Failed: {fail_count}")
    print(f"\nOutput directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch extract all retargeted dexterous hand trajectories in MUJOCO coordinates"
    )
    parser.add_argument(
        "--retargeting_dir",
        type=str,
        default="data/retargeting",
        help="Directory containing retargeted data (default: data/retargeting)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch4/workspace/hanyang_umass_edu-maniptrans/data/output_trajectories_mujoco",
        help="Output directory for extracted trajectories"
    )
    parser.add_argument(
        "--hand_types",
        type=str,
        nargs="+",
        choices=["shadow", "inspire", "inspireftp", "allegro", "artimano", "xhand", "wujihand"],
        help="Hand types to process (default: all)"
    )
    parser.add_argument(
        "--dataset_types",
        type=str,
        nargs="+",
        choices=["grab_demo", "OakInk-v2"],
        help="Dataset types to process (default: all)"
    )
    parser.add_argument(
        "--max_count",
        type=int,
        help="Maximum number of files to process (default: all)"
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Don't skip existing files (re-extract all)"
    )
    parser.add_argument(
        "--start_idx",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--end_idx",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    batch_extract_trajectories(
        retargeting_dir=args.retargeting_dir,
        output_dir=args.output_dir,
        hand_types=args.hand_types,
        dataset_types=args.dataset_types,
        max_count=args.max_count,
        skip_existing=True,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
