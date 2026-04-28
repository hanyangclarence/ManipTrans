#!/usr/bin/env python3
"""
Batch retargeting for OakInk-v2 dataset.

This script processes multiple OakInk-v2 sequences with options to:
- Skip already retargeted sequences
- Process a subset of data (e.g., 0.0-0.5 for parallel jobs)
- Specify hand type and side

Usage:
    # Process all sequences
    python data_utils/batch_retarget_oakink2.py --hand_type wujihand --side right

    # Process first half (0.0-0.5)
    python data_utils/batch_retarget_oakink2.py --hand_type wujihand --side right --start 0.0 --end 0.5

    # Process second half (0.5-1.0)
    python data_utils/batch_retarget_oakink2.py --hand_type wujihand --side right --start 0.5 --end 1.0

    # Force reprocess all (no skipping)
    python data_utils/batch_retarget_oakink2.py --hand_type wujihand --side right --force
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def get_all_oakink2_sequences(data_dir="data/OakInk-v2", hand_side="right"):
    """Get all OakInk-v2 sequence identifiers with valid stages for the specified hand"""
    import json

    anno_dir = os.path.join(data_dir, "anno_preview")
    program_dir = os.path.join(data_dir, "program", "program_info")

    if not os.path.exists(anno_dir):
        print(f"Error: Directory not found: {anno_dir}")
        return []

    files = sorted(os.listdir(anno_dir))
    pkl_files = [f for f in files if f.endswith('.pkl')]

    # Determine which object list key to check
    obj_list_key = "obj_list_rh" if hand_side == "right" else "obj_list_lh"

    sequences = []
    for filename in pkl_files:
        # Extract the 5-digit hash from filename
        # Format: scene_XX__AXXX++seq__HASH__TIMESTAMP.pkl
        parts = filename.split("__")
        if len(parts) >= 3:
            full_hash = parts[2]  # The full hash
            short_hash = full_hash[:5]  # First 5 digits

            # Get stages from program_info and filter by hand side
            program_file = os.path.join(program_dir, filename.replace('.pkl', '.json'))

            if os.path.exists(program_file):
                try:
                    with open(program_file, 'r') as f:
                        prog = json.load(f)

                    # Check each stage to see if it's valid for this hand
                    for stage_idx, (stage_key, stage_info) in enumerate(prog.items()):
                        # Check if this stage has objects for the requested hand
                        obj_list = stage_info.get(obj_list_key)

                        # Only add stage if it has objects (not None and not empty)
                        if obj_list is not None and len(obj_list) > 0:
                            seq_id = f"{short_hash}@{stage_idx}"
                            sequences.append(seq_id)
                except Exception:
                    # If can't read, add stage 0 by default
                    sequences.append(f"{short_hash}@0")
            else:
                # If no program file, add stage 0 by default
                sequences.append(f"{short_hash}@0")

    return sequences


def check_retargeting_exists(seq_id, hand_name, data_dir="data/OakInk-v2"):
    """Check if retargeting result already exists for a sequence"""

    # Find the full filename for this sequence
    anno_dir = os.path.join(data_dir, "anno_preview")
    short_hash = seq_id.split('@')[0]
    stage = seq_id.split('@')[1]

    # Find matching annotation file
    files = os.listdir(anno_dir)
    matching_file = None
    for f in files:
        if f.endswith('.pkl') and short_hash in f:
            matching_file = f
            break

    if not matching_file:
        return False

    # Expected output path
    output_path = f"data/retargeting/OakInk-v2/mano2{hand_name}/{matching_file.replace('.pkl', f'@{stage}.pkl')}"

    return os.path.exists(output_path)


def run_retargeting(seq_id, hand_type, side, iterations, headless=True):
    """Run retargeting for a single sequence"""

    # Determine hand name (e.g., "wujihand_rh")
    side_suffix = "rh" if side == "right" else "lh"
    hand_name = f"{hand_type}_{side_suffix}"

    print(f"\n{'='*80}")
    print(f"Retargeting: {seq_id} → {hand_name}")
    print(f"{'='*80}")

    cmd = [
        "python", "main/dataset/mano2dexhand.py",
        "--data_idx", seq_id,
        "--dexhand", hand_type,
        "--side", side,
        "--iter", str(iterations),
    ]

    if headless:
        cmd.append("--headless")

    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Retargeting failed for {seq_id} with error: {e}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch retargeting for OakInk-v2 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all sequences with wujihand
  python data_utils/batch_retarget_oakink2.py --hand_type wujihand --side right

  # Process first half (for parallel jobs)
  python data_utils/batch_retarget_oakink2.py --hand_type wujihand --side right --start 0.0 --end 0.5

  # Process second half
  python data_utils/batch_retarget_oakink2.py --hand_type wujihand --side right --start 0.5 --end 1.0

  # Process specific range with different hand
  python data_utils/batch_retarget_oakink2.py --hand_type inspire --side right --start 0.2 --end 0.4

  # Force reprocess everything (skip existing)
  python data_utils/batch_retarget_oakink2.py --hand_type wujihand --side right --force
        """
    )

    parser.add_argument(
        '--hand_type',
        type=str,
        required=True,
        choices=['shadow', 'inspire', 'inspireftp', 'allegro', 'artimano', 'xhand', 'wujihand'],
        help='Hand type to retarget to'
    )

    parser.add_argument(
        '--side',
        type=str,
        required=True,
        choices=['right', 'left'],
        help='Hand side (right or left)'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/OakInk-v2',
        help='OakInk-v2 data directory (default: data/OakInk-v2)'
    )

    parser.add_argument(
        '--start',
        type=float,
        default=0.0,
        help='Start ratio (0.0-1.0) for processing subset (default: 0.0)'
    )

    parser.add_argument(
        '--end',
        type=float,
        default=1.0,
        help='End ratio (0.0-1.0) for processing subset (default: 1.0)'
    )

    parser.add_argument(
        '--iter',
        type=int,
        default=3000,
        help='Number of optimization iterations (default: 3000 for shadow, 1000 for others recommended)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocess even if output exists'
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        default=True,
        help='Run in headless mode (default: True)'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Only show what would be processed, do not run retargeting'
    )

    args = parser.parse_args()

    # Validate start/end ratios
    if not (0.0 <= args.start < args.end <= 1.0):
        print("Error: Must have 0.0 <= start < end <= 1.0")
        return 1

    # Get hand name
    side_suffix = "rh" if args.side == "right" else "lh"
    hand_name = f"{args.hand_type}_{side_suffix}"

    print("=" * 80)
    print("OakInk-v2 Batch Retargeting")
    print("=" * 80)
    print(f"Hand: {hand_name}")
    print(f"Data directory: {args.data_dir}")
    print(f"Processing range: {args.start:.2f} - {args.end:.2f}")
    print(f"Iterations: {args.iter}")
    print(f"Force reprocess: {args.force}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80)

    # Get all sequences
    print("\nScanning for OakInk-v2 sequences...")
    all_sequences = get_all_oakink2_sequences(args.data_dir, args.side)

    if not all_sequences:
        print("Error: No sequences found!")
        return 1

    print(f"Found {len(all_sequences)} total sequences (valid for {args.side} hand)")

    # Apply range filter
    start_idx = int(len(all_sequences) * args.start)
    end_idx = int(len(all_sequences) * args.end)
    sequences_to_process = all_sequences[start_idx:end_idx]

    print(f"Processing subset: indices {start_idx} to {end_idx} ({len(sequences_to_process)} sequences)")

    # Filter out already processed sequences (unless force)
    if not args.force:
        print("\nChecking for existing retargeting results...")
        sequences_needing_processing = []
        skipped = 0

        for seq_id in sequences_to_process:
            if check_retargeting_exists(seq_id, hand_name, args.data_dir):
                skipped += 1
            else:
                sequences_needing_processing.append(seq_id)

        print(f"Skipping {skipped} already processed sequences")
        sequences_to_process = sequences_needing_processing

    print(f"\n{len(sequences_to_process)} sequences need processing")

    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN - Sequences to process:")
        print("=" * 80)
        for i, seq_id in enumerate(sequences_to_process, 1):
            print(f"{i:4d}. {seq_id}")
        print(f"\nTotal: {len(sequences_to_process)} sequences")
        return 0

    # Process sequences
    print("\n" + "=" * 80)
    print("Starting batch retargeting...")
    print("=" * 80)

    success_count = 0
    fail_count = 0
    failed_sequences = []

    for i, seq_id in enumerate(sequences_to_process, 1):
        print(f"\n[{i}/{len(sequences_to_process)}] Processing {seq_id}...")

        success = run_retargeting(
            seq_id,
            args.hand_type,
            args.side,
            args.iter,
            args.headless
        )

        if success:
            success_count += 1
        else:
            fail_count += 1
            failed_sequences.append(seq_id)

    # Summary
    print("\n" + "=" * 80)
    print("BATCH RETARGETING SUMMARY")
    print("=" * 80)
    print(f"Total processed: {len(sequences_to_process)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")

    if failed_sequences:
        print(f"\nFailed sequences:")
        for seq_id in failed_sequences:
            print(f"  - {seq_id}")

    print("=" * 80)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
