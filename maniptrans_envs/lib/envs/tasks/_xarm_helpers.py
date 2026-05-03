"""Shared helpers for arm-mounted dexhands (e.g. xarm_wujihand).

Used by both DexHandImitatorRHEnv (Stage 1) and DexHandManipRHEnv (Stage 2)
so that scene placement and IK-at-reset are implemented in exactly one
place. The mixin's methods are no-ops when the active dexhand has
``n_arm_dofs == 0`` (floating-hand path), so it's safe for the env classes
to always inherit from it.
"""

from __future__ import annotations

import os
import sys

import torch

from ...asset_root import ASSET_ROOT


# Trajectory-zero z-shift for arm-mounted dexhands.
# data_utils/convert_manus_to_maniptrans.py bakes a +0.415 m z-lift into
# every Manus pickle so the action lands on the legacy floating-hand
# table at z=0.415. With the xArm setup we want the action to land on a
# GP-style table whose top is at z=0.10 (matching
# replay_xarm_wujihand_trajectory.py); compensate by adding -0.315 m.
ARM_TRAJECTORY_Z_SHIFT: float = 0.10 - 0.415   # = -0.315


def load_xarm7_kinematics():
    """Lazy-load the xArm7 IK wrapper from the asset dir.

    Imported only when an arm-mounted dexhand is in use, so headless
    wuji-only training stays free of the ctypes / .so dependency.
    """
    ik_dir = os.path.join(ASSET_ROOT, "xarm_wujihand", "ik")
    if ik_dir not in sys.path:
        sys.path.insert(0, ik_dir)
    from xarm7_kinematics import XArm7Kinematics  # noqa: WPS433
    return XArm7Kinematics


class XArmIKMixin:
    """IK + scene-placement helpers shared by Stage 1 / Stage 2 envs.

    Expects the host class to expose:
      - ``self.dexhand``  with ``arm_base_pos``, ``arm_base_quat``,
        ``arm_seed``, ``tcp_yaw``, ``n_arm_dofs``
      - ``self.sim_device``  torch.device
    """

    def _init_arm_ik(self):
        """Load the C IK library and cache world<->base frame tensors.

        IK runs on CPU and accepts torch tensors of any device, returning
        results on the same device. We keep all cached tensors on
        sim_device so per-reset IK calls don't roundtrip through .cpu().
        """
        XArm7Kinematics = load_xarm7_kinematics()
        self._ik_solver = XArm7Kinematics(
            tcp_offset=[0.0, 0.0, 57.0, 0.0, 0.0, self.dexhand.tcp_yaw],
        )
        device = self.sim_device

        self._arm_base_pos_t = torch.tensor(
            list(self.dexhand.arm_base_pos), dtype=torch.float32, device=device,
        )
        # arm_base_quat is (x, y, z, w); build the corresponding world<-base
        # rotation matrix. Inverse (world->base) is its transpose.
        from main.dataset.transform import quat_to_rotmat as _q2r
        # quat_to_rotmat in transform.py expects (w, x, y, z) — re-order.
        qx, qy, qz, qw = self.dexhand.arm_base_quat
        q_wxyz = torch.tensor([qw, qx, qy, qz], dtype=torch.float32, device=device).unsqueeze(0)
        R_arm_base = _q2r(q_wxyz)[0]                    # (3, 3) world<-base
        self._world_to_base_R_t = R_arm_base.T.contiguous()
        self._arm_seed_t = torch.tensor(
            self.dexhand.arm_seed.tolist(), dtype=torch.float32, device=device,
        )

    def _solve_arm_ik(self, wrist_pos_world, wrist_rotmat_world, q_refs=None):
        """Batch IK for the 7 arm joints.

        wrist_pos_world: (B, 3) target palm_link world-frame position
        wrist_rotmat_world: (B, 3, 3) target palm_link world-frame rotation
        q_refs: (B, 7) optional warm-start (defaults to arm_seed broadcast)

        Returns:
            arm_q: (B, 7) joint angles. Failed rows are filled with arm_seed.
            success: (B,) bool — True where IK converged.
        """
        B = wrist_pos_world.shape[0]
        # World -> arm-base translation
        delta = wrist_pos_world - self._arm_base_pos_t[None]      # (B, 3)
        target_pos_base = (self._world_to_base_R_t @ delta.T).T   # (B, 3)
        target_pos_mm = target_pos_base * 1000.0
        # World -> arm-base rotation
        target_R_base = self._world_to_base_R_t[None] @ wrist_rotmat_world   # (B, 3, 3)

        target_mat = torch.zeros(B, 4, 4, device=wrist_pos_world.device, dtype=torch.float32)
        target_mat[:, 3, 3] = 1.0
        target_mat[:, :3, :3] = target_R_base
        target_mat[:, :3, 3] = target_pos_mm

        if q_refs is None:
            q_refs = self._arm_seed_t[None].expand(B, -1).contiguous()

        arm_q, ret_codes = self._ik_solver.inverse_kinematics_mat_batch(
            target_mat, q_refs=q_refs,
        )
        arm_q = arm_q.to(torch.float32)
        success = (ret_codes == 0)
        if (~success).any():
            seed_b = self._arm_seed_t[None].expand(B, -1).contiguous()
            arm_q = torch.where(success.unsqueeze(-1), arm_q, seed_b)
        return arm_q, success
