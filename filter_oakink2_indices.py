#!/usr/bin/env python3
"""Filter OakInk2 right-hand primitive tasks by target object id, intersecting
with the user's reference trajectory set under output_trajectories_mujoco_xarm_
transformed (so the ManipTrans baseline trains on the same trajectory subset
the user's own method consumes).

Match logic mirrors the OakInk2 RH loader (oakink2_dataset_dexhand_rh.py:215):
the simulated manipulation target is `obj_list_rh[0]` of the primitive task,
so we only keep stages where THAT entry matches the requested object id.

Usage:
    python filter_oakink2_indices.py
    python filter_oakink2_indices.py --user-dir /path/to/wujihand_pkls
    python filter_oakink2_indices.py --target O02@0011@00003=cup --target O02@0030@00002=spoon

Writes one '<5hex>@<stage>' per line to indices_<name>.txt for each target.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict


DEFAULT_USER_DIR = (
    "/scratch4/workspace/hanyang_umass_edu-maniptrans/data/"
    "output_trajectories_mujoco_xarm_transformed"
)
DEFAULT_ANNO_DIR = "data/OakInk-v2/anno_preview"
DEFAULT_PROG_DIR = "data/OakInk-v2/program/program_info"

DEFAULT_TARGETS = {
    "O02@0011@00003": "cup",
    "O02@0030@00002": "spoon",
    "O02@0033@00001": "stick",
}

# Files in the user's reference set are named:
#   wujihand_hand_trajectory_<5hex>@<stage>_mujoco.pkl
USER_FILE_PATTERN = re.compile(
    r"^wujihand_hand_trajectory_([0-9a-f]{5})@(\d+)_mujoco\.pkl$"
)


def parse_targets(spec_list):
    if not spec_list:
        return DEFAULT_TARGETS
    out = {}
    for spec in spec_list:
        if "=" not in spec:
            sys.exit(f"--target must be 'OBJID=name', got {spec!r}")
        obj_id, name = spec.split("=", 1)
        out[obj_id.strip()] = name.strip()
    return out


def collect_user_pairs(user_dir):
    pairs = set()
    for fn in os.listdir(user_dir):
        m = USER_FILE_PATTERN.match(fn)
        if m:
            pairs.add((m.group(1), int(m.group(2))))
    return pairs


def build_hash5_map(anno_dir):
    """Mirrors loader logic (oakink2_dataset_dexhand_rh.py:80-86): sorted
    listdir of anno_preview/*.pkl, with 5-hex prefix from filename split."""
    files = sorted(os.listdir(anno_dir))
    return {os.path.split(p)[-1].split("_")[5][:5]: p for p in files}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user-dir", default=DEFAULT_USER_DIR)
    ap.add_argument("--anno-dir", default=DEFAULT_ANNO_DIR)
    ap.add_argument("--prog-dir", default=DEFAULT_PROG_DIR)
    ap.add_argument("--out-prefix", default="indices",
                    help="output files are <prefix>_<name>.txt")
    ap.add_argument("--target", action="append", default=None,
                    help="OBJID=name (repeatable). Default: cup/spoon/stick.")
    ap.add_argument("--combined", default=None,
                    help="if set, also write the union of all target indices "
                         "(deduped, sorted) to this path -- useful for one "
                         "single sbatch array submission.")
    args = ap.parse_args()

    targets = parse_targets(args.target)

    if not os.path.isdir(args.user_dir):
        sys.exit(f"user-dir not found: {args.user_dir}")
    if not os.path.isdir(args.anno_dir):
        sys.exit(f"anno-dir not found: {args.anno_dir}  "
                 f"(must run from repo root)")
    if not os.path.isdir(args.prog_dir):
        sys.exit(f"prog-dir not found: {args.prog_dir}")

    user_pairs = collect_user_pairs(args.user_dir)
    print(f"user reference set: {len(user_pairs)} <5hex>@<stage> trajectories")

    hash5 = build_hash5_map(args.anno_dir)

    found = defaultdict(list)        # obj_id -> [(h5, stage)]
    unresolved = []
    for h5, stage in sorted(user_pairs):
        if h5 not in hash5:
            unresolved.append((h5, stage, "no_anno_for_hash"))
            continue
        base = os.path.splitext(hash5[h5])[0]
        pj = os.path.join(args.prog_dir, base + ".json")
        if not os.path.exists(pj):
            unresolved.append((h5, stage, "no_program_info"))
            continue
        with open(pj) as f:
            info = json.load(f)
        keys = list(info.keys())
        if stage >= len(keys):
            unresolved.append((h5, stage,
                               f"stage_oob (have {len(keys)})"))
            continue
        rh = info[keys[stage]].get("obj_list_rh") or []
        if not rh:
            continue
        primary = rh[0]
        if primary in targets:
            found[primary].append((h5, stage))

    print()
    for obj_id, name in targets.items():
        out_path = f"{args.out_prefix}_{name}.txt"
        with open(out_path, "w") as f:
            for h5, stage in found[obj_id]:
                f.write(f"{h5}@{stage}\n")
        print(f"  {out_path:30s}  {len(found[obj_id]):>4}  ({obj_id} = {name})")

    if args.combined:
        union = sorted({(h5, st) for entries in found.values()
                                  for (h5, st) in entries})
        with open(args.combined, "w") as f:
            for h5, st in union:
                f.write(f"{h5}@{st}\n")
        print(f"\n  {args.combined:30s}  {len(union):>4}  (union, deduped)")

    if unresolved:
        print(f"\n  unresolved user pairs: {len(unresolved)}")
        for h5, stage, why in unresolved[:5]:
            print(f"    {h5}@{stage}  {why}")
        if len(unresolved) > 5:
            print(f"    ... and {len(unresolved) - 5} more")


if __name__ == "__main__":
    main()
