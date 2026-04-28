"""
List available OakInk-v2 sequences

This script helps you find valid data indices for OakInk-v2 dataset.
"""

import os
import pickle

def list_oakink2_sequences(data_dir="data/OakInk-v2", show_details=False):
    """List all available OakInk-v2 sequences"""

    anno_dir = os.path.join(data_dir, "anno_preview")

    if not os.path.exists(anno_dir):
        print(f"Error: Directory not found: {anno_dir}")
        return []

    files = sorted(os.listdir(anno_dir))
    files = [f for f in files if f.endswith('.pkl')]

    print(f"Found {len(files)} sequences in OakInk-v2")
    print("="*80)

    sequences = []

    for i, filename in enumerate(files):
        # Extract the 5-digit hash from filename
        # Format: scene_XX__AXXX++seq__HASH__TIMESTAMP.pkl
        parts = filename.split("__")
        if len(parts) >= 3:
            full_hash = parts[2]  # The full hash
            short_hash = full_hash[:5]  # First 5 digits

            sequences.append({
                'index': short_hash,
                'filename': filename,
                'full_hash': full_hash
            })

            if show_details or i < 10:  # Show first 10 or all if requested
                print(f"Index: {short_hash}@0  (or @1, @2, etc. for different stages)")
                print(f"  File: {filename}")

                # Try to load and get stage info
                try:
                    anno_path = os.path.join(anno_dir, filename)
                    with open(anno_path, 'rb') as f:
                        anno = pickle.load(f)

                    # Get number of stages if available
                    if 'program_info' in anno or 'task_target' in anno:
                        program_file = os.path.join(
                            data_dir, "program", "program_info",
                            filename.replace('.pkl', '.json')
                        )
                        if os.path.exists(program_file):
                            import json
                            with open(program_file, 'r') as f:
                                prog = json.load(f)
                            # Number of stages = number of top-level keys in JSON
                            n_stages = len(prog.keys())
                            print(f"  Stages: 0 to {n_stages-1} ({n_stages} total)")

                    # Get sequence length
                    if 'mocap_frame_id_list' in anno:
                        n_frames = len(anno['mocap_frame_id_list'])
                        print(f"  Frames: {n_frames} (at 120Hz)")

                except Exception as e:
                    print(f"  (Could not load details: {e})")

                print()

    if not show_details and len(files) > 10:
        print(f"\n... and {len(files) - 10} more sequences")
        print(f"\nRun with --details to see all sequences")

    print("="*80)
    print("\nUsage Examples:")
    print(f"  # Retarget first sequence (stage 0) with shadow hand:")
    if sequences:
        print(f"  python main/dataset/mano2dexhand.py --data_idx {sequences[0]['index']}@0 --dexhand shadow --side right --headless --iter 3000")
        print(f"\n  # Extract trajectory:")
        print(f"  python data_utils/extract_shadow_trajectory.py --data_idx {sequences[0]['index']}@0 --hand_type shadow --side right")
        print(f"\n  # Visualize:")
        print(f"  python data_utils/visualize_shadow_trajectory.py --data_idx {sequences[0]['index']}@0 --hand_type shadow --side right")

    print("\n" + "="*80)
    print("IMPORTANT NOTES:")
    print("  - Index format: {5_digit_hash}@{stage_number}")
    print("  - Stage number starts from 0")
    print("  - Different stages represent different manipulation phases")
    print("  - Use 3000 iterations for shadow hand, 1000 for other hands")
    print("="*80)

    return sequences


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="List OakInk-v2 sequences")
    parser.add_argument("--data_dir", type=str, default="data/OakInk-v2",
                        help="Path to OakInk-v2 data directory")
    parser.add_argument("--details", action="store_true",
                        help="Show details for all sequences")
    parser.add_argument("--search", type=str,
                        help="Search for sequences containing this text")

    args = parser.parse_args()

    sequences = list_oakink2_sequences(args.data_dir, args.details)

    if args.search:
        print(f"\nSearching for: {args.search}")
        print("="*80)
        matches = [s for s in sequences if args.search.lower() in s['filename'].lower()]
        for seq in matches:
            print(f"Index: {seq['index']}@0")
            print(f"  File: {seq['filename']}\n")
        print(f"Found {len(matches)} matches")
