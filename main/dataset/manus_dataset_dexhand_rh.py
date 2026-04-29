import os
import pickle
from functools import lru_cache

import numpy as np
import torch

from .base import ManipData
from .decorators import register_manipdata


# Index format: "manus@<dataset_root>@<stem>".
# - <dataset_root> is the directory under data/ that holds sequences/ + retargeting/
#   (e.g. "manus_marker_pen", "manus_hammer_mixed").
# - <stem> is the file name without extension under sequences/ (e.g. "20260201_154446_552807").
#
# Example: dataIndices=["manus@manus_marker_pen@20260201_154446_552807"]


def _parse_index(index):
    if not (isinstance(index, str) and index.startswith("manus@")):
        raise ValueError(f"manus dataset expects 'manus@<root>@<stem>', got {index!r}")
    parts = index.split("@")
    if len(parts) != 3:
        raise ValueError(
            f"manus index must be 'manus@<root>@<stem>'. got {index!r} "
            f"(parts={parts})"
        )
    _, root, stem = parts
    return root, stem


@register_manipdata("manus_rh")
class ManusDexHandRH(ManipData):
    side = "right"

    def __init__(
        self,
        *,
        data_dir: str = "data",
        split: str = "all",
        skip: int = 2,
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        verbose=True,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            skip=skip,
            device=device,
            mujoco2gym_transf=mujoco2gym_transf,
            max_seq_len=max_seq_len,
            dexhand=dexhand,
            verbose=verbose,
            **kwargs,
        )

        # The converter already baked the Genesis -> IsaacGym transform into the
        # pickles, so we override mujoco2gym_transf to identity. base.process_data
        # will still convert wrist_rot from rotation matrix to axis-angle, which is
        # what the env consumes.
        self.mujoco2gym_transf = torch.eye(4, dtype=torch.float32, device=self.device)
        self.transf_offset = torch.eye(4, dtype=torch.float32, device=self.device)

        # Discover available sequences across all manus_* subdirs of data_dir so
        # __len__ is meaningful, but the actual data is fetched by index string.
        self.data_pathes = []
        if os.path.isdir(self.data_dir):
            for entry in sorted(os.listdir(self.data_dir)):
                root_dir = os.path.join(self.data_dir, entry)
                seq_dir = os.path.join(root_dir, "sequences")
                if not os.path.isdir(seq_dir):
                    continue
                for fn in sorted(os.listdir(seq_dir)):
                    if fn.endswith(".pkl"):
                        self.data_pathes.append(os.path.join(seq_dir, fn))

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        root, stem = _parse_index(index)

        seq_path = os.path.join(self.data_dir, root, "sequences", f"{stem}.pkl")
        ret_path = os.path.join(self.data_dir, root, "retargeting", f"{stem}.pkl")

        with open(seq_path, "rb") as f:
            raw = pickle.load(f)

        side = raw.get("side", "right")
        assert side == self.side, (
            f"manus sequence {seq_path} is side={side!r}, "
            f"but loader is registered as {self.side!r}"
        )

        def _t(x, dtype=torch.float32):
            return torch.tensor(np.ascontiguousarray(x), dtype=dtype, device=self.device)

        obj_verts = _t(raw["obj_verts"])  # (1000, 3)
        obj_traj = _t(raw["obj_trajectory"])  # (T, 4, 4)
        wrist_pos = _t(raw["wrist_pos"])  # (T, 3)
        wrist_rot = _t(raw["wrist_rot"])  # (T, 3, 3) rotation matrices

        mano_joints = {k: _t(v) for k, v in raw["mano_joints"].items()}
        # The converter writes only the 5 *_tip keys. The env uses
        # useFingertipsOnly=True to consume them; downstream callers reuse this dict.

        data = {
            "data_path": seq_path,
            "obj_id": raw["obj_id"],
            "obj_verts": obj_verts,
            "obj_urdf_path": raw["obj_urdf_path"],
            "obj_trajectory": obj_traj,
            "scene_objs": [],
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": mano_joints,
        }

        self.process_data(data, index, obj_verts)
        self.load_retargeted_data(data, ret_path)
        return data
