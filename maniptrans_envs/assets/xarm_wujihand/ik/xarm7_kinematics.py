"""Python wrapper for the UFactory xArm7 offline kinematics library.

Usage::

    from gs_env.sim.robots.xarm.xarm7_kinematics import XArm7Kinematics

    kin = XArm7Kinematics()
    # or with TCP offset (mm, rad): kin = XArm7Kinematics(tcp_offset=[0, 0, 172, 0, 0, 0])

    # Forward kinematics
    pose = kin.forward_kinematics(joint_angles)       # -> (x,y,z,roll,pitch,yaw)
    mat  = kin.forward_kinematics_mat(joint_angles)    # -> 4x4 numpy array

    # Inverse kinematics
    angles = kin.inverse_kinematics(pose)                          # pose = [x,y,z,r,p,y]
    angles = kin.inverse_kinematics(pose, q_ref=current_angles)    # with reference
    angles = kin.inverse_kinematics_mat(mat_4x4)                   # from 4x4 matrix

    # Batch inverse kinematics (OpenMP parallelized, torch tensor support)
    #   mats: (B, 4, 4) in mm — numpy or torch tensor (any device)
    #   q_refs: (B, 7) or (7,)
    angles, ret_codes = kin.inverse_kinematics_mat_batch(mats, q_refs=refs)
"""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Default library path — co-located .so next to this Python file
_DEFAULT_LIB_PATH = Path(__file__).parent / "libxarm7_kinematics.so"

# Joint limits from the header (radians)
XARM7_ANGLE_MIN = np.array(
    [-2.0 * math.pi, -2.059, -2.0 * math.pi, -0.19198, -2.0 * math.pi, -1.69297, -2.0 * math.pi]
)
XARM7_ANGLE_MAX = np.array(
    [2.0 * math.pi, 2.0944, 2.0 * math.pi, 3.927, 2.0 * math.pi, math.pi, 2.0 * math.pi]
)
XARM7_ANGLE_MIN_LIMITED = np.array(
    [-math.pi, -2.059, -math.pi, -0.19198, -math.pi, -1.69297, -math.pi]
)
XARM7_ANGLE_MAX_LIMITED = np.array(
    [math.pi, 2.0944, math.pi, 3.927, math.pi, math.pi, math.pi]
)

_c_double_p = ctypes.POINTER(ctypes.c_double)


def _to_c_array(arr: Sequence[float], n: int) -> ctypes.Array:
    """Convert a python sequence to a ctypes double array."""
    c_arr = (ctypes.c_double * n)()
    for i in range(n):
        c_arr[i] = float(arr[i])
    return c_arr


class XArm7Kinematics:
    """Offline FK/IK solver for xArm7 via UFactory's kinematics library.

    Args:
        lib_path: Path to ``libxarm7_kinematics.so``.
        tcp_offset: Optional 6-element [x, y, z, roll, pitch, yaw] TCP offset
            in mm and radians.  Use this to set the tool-center-point relative
            to the flange (e.g. the WujiHand palm center).
        world_offset: Optional 6-element world offset [x, y, z, r, p, y].
        limited: If True (default), use the +-180 degree joint limits instead
            of the full +-360 range.  Avoids strange IK solutions.
    """

    def __init__(
        self,
        lib_path: str | Path | None = None,
        tcp_offset: Optional[Sequence[float]] = None,
        world_offset: Optional[Sequence[float]] = None,
        limited: bool = True,
    ):
        lib_path = Path(lib_path) if lib_path else _DEFAULT_LIB_PATH
        if not lib_path.exists():
            raise FileNotFoundError(f"Kinematics library not found: {lib_path}")

        self._lib = ctypes.CDLL(str(lib_path))

        # Bind C functions
        self._config = self._lib.xarm7_config_c
        self._config.restype = ctypes.c_int
        self._config.argtypes = [_c_double_p, _c_double_p, _c_double_p, _c_double_p]

        self._fk_mat = self._lib.xarm7_fk_mat_c
        self._fk_mat.restype = ctypes.c_int
        self._fk_mat.argtypes = [_c_double_p, _c_double_p]

        self._fk_pose = self._lib.xarm7_fk_pose_c
        self._fk_pose.restype = ctypes.c_int
        self._fk_pose.argtypes = [_c_double_p, _c_double_p]

        self._ik_mat = self._lib.xarm7_ik_mat_c
        self._ik_mat.restype = ctypes.c_int
        self._ik_mat.argtypes = [_c_double_p, _c_double_p, _c_double_p]

        self._ik_pose = self._lib.xarm7_ik_pose_c
        self._ik_pose.restype = ctypes.c_int
        self._ik_pose.argtypes = [_c_double_p, _c_double_p, _c_double_p]

        # Batch functions (OpenMP parallelized)
        _c_int_p = ctypes.POINTER(ctypes.c_int)
        self._ik_mat_batch = self._lib.xarm7_ik_mat_batch_c
        self._ik_mat_batch.restype = ctypes.c_int
        self._ik_mat_batch.argtypes = [_c_double_p, _c_double_p, _c_double_p,
                                       _c_int_p, ctypes.c_int]

        self._ik_pose_batch = self._lib.xarm7_ik_pose_batch_c
        self._ik_pose_batch.restype = ctypes.c_int
        self._ik_pose_batch.argtypes = [_c_double_p, _c_double_p, _c_double_p,
                                        _c_int_p, ctypes.c_int]

        self._fk_mat_batch = self._lib.xarm7_fk_mat_batch_c
        self._fk_mat_batch.restype = ctypes.c_int
        self._fk_mat_batch.argtypes = [_c_double_p, _c_double_p, _c_int_p, ctypes.c_int]

        # Configure joint limits
        if limited:
            q_min = _to_c_array(XARM7_ANGLE_MIN_LIMITED, 7)
            q_max = _to_c_array(XARM7_ANGLE_MAX_LIMITED, 7)
        else:
            q_min = _to_c_array(XARM7_ANGLE_MIN, 7)
            q_max = _to_c_array(XARM7_ANGLE_MAX, 7)

        c_tcp = _to_c_array(tcp_offset, 6) if tcp_offset is not None else None
        c_world = _to_c_array(world_offset, 6) if world_offset is not None else None

        ret = self._config(q_max, q_min, c_tcp, c_world)
        if ret != 0:
            raise RuntimeError(f"xarm7_config failed with code {ret}")

    # ------------------------------------------------------------------
    # Forward Kinematics
    # ------------------------------------------------------------------

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute FK, returning [x, y, z, roll, pitch, yaw].

        Args:
            joint_angles: 7 joint angles in radians.

        Returns:
            pose: shape (6,) — [x(mm), y(mm), z(mm), roll(rad), pitch(rad), yaw(rad)].
        """
        joint_angles = np.asarray(joint_angles, dtype=np.float64).ravel()
        assert joint_angles.shape == (7,), f"Expected 7 joint angles, got {joint_angles.shape}"

        theta = _to_c_array(joint_angles, 7)
        pose = (ctypes.c_double * 6)()

        ret = self._fk_pose(theta, pose)
        if ret != 0:
            raise RuntimeError(f"xarm7_forward_kinematics (pose) failed with code {ret}")

        return np.array([pose[i] for i in range(6)])

    def forward_kinematics_mat(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute FK, returning 4x4 homogeneous transformation matrix.

        Args:
            joint_angles: 7 joint angles in radians.

        Returns:
            mat: shape (4, 4) — homogeneous transformation matrix.
        """
        joint_angles = np.asarray(joint_angles, dtype=np.float64).ravel()
        assert joint_angles.shape == (7,), f"Expected 7 joint angles, got {joint_angles.shape}"

        theta = _to_c_array(joint_angles, 7)
        mat = (ctypes.c_double * 16)()

        ret = self._fk_mat(theta, mat)
        if ret != 0:
            raise RuntimeError(f"xarm7_forward_kinematics (mat) failed with code {ret}")

        return np.array([mat[i] for i in range(16)]).reshape(4, 4)

    # ------------------------------------------------------------------
    # Inverse Kinematics
    # ------------------------------------------------------------------

    def inverse_kinematics(
        self,
        pose: np.ndarray,
        q_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute IK from [x, y, z, roll, pitch, yaw] pose.

        Args:
            pose: 6-element [x(mm), y(mm), z(mm), roll(rad), pitch(rad), yaw(rad)].
            q_ref: Optional 7-element reference joint angles (radians).
                IK returns the solution closest to this.  Defaults to zeros.

        Returns:
            joint_angles: shape (7,) in radians.

        Raises:
            RuntimeError: If no IK solution is found.
        """
        pose = np.asarray(pose, dtype=np.float64).ravel()
        assert pose.shape == (6,), f"Expected 6-element pose, got {pose.shape}"

        if q_ref is None:
            q_ref = np.zeros(7)
        q_ref = np.asarray(q_ref, dtype=np.float64).ravel()
        assert q_ref.shape == (7,), f"Expected 7-element q_ref, got {q_ref.shape}"

        c_pose = _to_c_array(pose, 6)
        c_qref = _to_c_array(q_ref, 7)
        c_theta = (ctypes.c_double * 7)()

        ret = self._ik_pose(c_pose, c_qref, c_theta)
        if ret != 0:
            raise RuntimeError(
                f"xarm7_inverse_kinematics (pose) failed with code {ret}. "
                f"Pose may be unreachable: {pose}"
            )

        return np.array([c_theta[i] for i in range(7)])

    def inverse_kinematics_mat(
        self,
        mat: np.ndarray,
        q_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute IK from a 4x4 homogeneous transformation matrix.

        Args:
            mat: 4x4 homogeneous transformation matrix.
            q_ref: Optional 7-element reference joint angles (radians).

        Returns:
            joint_angles: shape (7,) in radians.

        Raises:
            RuntimeError: If no IK solution is found.
        """
        mat = np.asarray(mat, dtype=np.float64).ravel()
        assert mat.shape == (16,), f"Expected 4x4 matrix, got shape that flattens to {mat.shape}"

        if q_ref is None:
            q_ref = np.zeros(7)
        q_ref = np.asarray(q_ref, dtype=np.float64).ravel()
        assert q_ref.shape == (7,), f"Expected 7-element q_ref, got {q_ref.shape}"

        c_mat = _to_c_array(mat, 16)
        c_qref = _to_c_array(q_ref, 7)
        c_theta = (ctypes.c_double * 7)()

        ret = self._ik_mat(c_mat, c_qref, c_theta)
        if ret != 0:
            raise RuntimeError(
                f"xarm7_inverse_kinematics (mat) failed with code {ret}. "
                f"Target may be unreachable."
            )

        return np.array([c_theta[i] for i in range(7)])

    # ------------------------------------------------------------------
    # Batch Operations
    # ------------------------------------------------------------------

    @staticmethod
    def _to_contiguous_f64(arr):
        """Convert numpy or torch tensor to C-contiguous float64 numpy array."""
        try:
            import torch
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().to(torch.float64).contiguous().numpy()
        except ImportError:
            pass
        return np.ascontiguousarray(arr, dtype=np.float64)

    def inverse_kinematics_mat_batch(
        self,
        mats,
        q_refs=None,
    ):
        """Batch IK from (B, 4, 4) matrices.

        Uses the **exact same** UFactory C solver per element (sequential
        execution — the C library has internal state that is not thread-safe).
        Accepts numpy arrays or torch tensors (any device); GPU tensors are
        transferred to CPU automatically.

        Units: mm and radians (same as the single-element API).

        Args:
            mats: (B, 4, 4) target transforms in mm.
            q_refs: (B, 7) or (7,) reference joint angles.  Defaults to zeros.

        Returns:
            joint_angles: (B, 7) — joint angles in radians.
            ret_codes: (B,) — int, 0 = success per element.
            If input was a torch tensor, outputs are tensors on the same device.
        """
        _input_torch_device = None
        try:
            import torch
            if isinstance(mats, torch.Tensor):
                _input_torch_device = mats.device
        except ImportError:
            pass

        mats_np = self._to_contiguous_f64(mats).reshape(-1, 16)
        B = mats_np.shape[0]

        if q_refs is None:
            q_refs_np = np.zeros((B, 7), dtype=np.float64)
        else:
            q_refs_np = self._to_contiguous_f64(q_refs)
            if q_refs_np.ndim == 1:
                q_refs_np = np.tile(q_refs_np, (B, 1))
            q_refs_np = np.ascontiguousarray(q_refs_np.reshape(B, 7))

        mats_flat = np.ascontiguousarray(mats_np, dtype=np.float64)
        thetas = np.zeros((B, 7), dtype=np.float64)
        ret_codes = np.zeros(B, dtype=np.int32)

        self._ik_mat_batch(
            mats_flat.ctypes.data_as(_c_double_p),
            q_refs_np.ctypes.data_as(_c_double_p),
            thetas.ctypes.data_as(_c_double_p),
            ret_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(B),
        )

        if _input_torch_device is not None:
            import torch
            return (
                torch.from_numpy(thetas).to(_input_torch_device),
                torch.from_numpy(ret_codes.astype(np.int64)).to(_input_torch_device),
            )

        return thetas, ret_codes

    def forward_kinematics_mat_batch(
        self,
        joint_angles,
    ):
        """Batch FK: (B, 7) -> (B, 4, 4).  OpenMP parallelized.

        Args:
            joint_angles: (B, 7) joint angles in radians.

        Returns:
            mats: (B, 4, 4) transforms (mm).
            ret_codes: (B,) int, 0 = success.
            If input was a torch tensor, outputs are tensors on the same device.
        """
        _input_torch_device = None
        try:
            import torch
            if isinstance(joint_angles, torch.Tensor):
                _input_torch_device = joint_angles.device
        except ImportError:
            pass

        angles_np = self._to_contiguous_f64(joint_angles).reshape(-1, 7)
        B = angles_np.shape[0]

        mats = np.zeros((B, 16), dtype=np.float64)
        ret_codes = np.zeros(B, dtype=np.int32)

        self._fk_mat_batch(
            angles_np.ctypes.data_as(_c_double_p),
            mats.ctypes.data_as(_c_double_p),
            ret_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(B),
        )

        mats = mats.reshape(B, 4, 4)

        if _input_torch_device is not None:
            import torch
            return (
                torch.from_numpy(mats).to(_input_torch_device),
                torch.from_numpy(ret_codes.astype(np.int64)).to(_input_torch_device),
            )

        return mats, ret_codes
