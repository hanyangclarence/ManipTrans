"""xArm7 + WujiHand right-hand dexhand registration.

The combined URDF (`assets/xarm_wujihand/right.urdf`) is a fixed-base 7-DOF
xArm7 with the WujiHand bolted to the end-effector. Total: 27 actuated DOFs
(7 arm + 20 finger) and 35 rigid bodies (9 arm bodies including link_eef +
26 wuji bodies starting from palm_link).

Body indices for the wuji portion are shifted by ARM_OFFSET=9 relative to
the standalone wujihand class, so weight_idx/bone_links are derived
programmatically rather than hand-edited.
"""
import numpy as np
from abc import ABC

from main.dataset.transform import aa_to_rotmat
from .base import DexHand
from .decorators import register_dexhand


# Number of rigid bodies in the xArm7 chain that precede palm_link, with
# collapse_fixed_joints=False. Counted from xarm7_with_wujihand_v5.urdf:
# link_base, link1..link7, link_eef = 9 bodies.
ARM_OFFSET = 9

# Right-arm warm-start seed (radians) used for IK at env reset. Mirrored
# from RIGHT_ARM_SEED in GenesisPlayground replay_xarm_wujihand_trajectory.py.
RIGHT_ARM_SEED = np.array([
    -0.2276704, 0.17346721, -0.30160318, 0.53979892,
    2.35584248, 1.32476175, 2.97201047,
])

# wujihand_fix joint yaw — 135° for right, 45° for left. Used to build the
# IK TCP offset (palm-relative-to-flange) so the IK targets palm_link, not
# the bare xArm flange.
RIGHT_TCP_YAW = 2.3562

# xArm base placement in *IsaacGym* world frame. The GP reference puts the
# arm base at (-0.2, 0, 0) in Genesis (Mujoco-style) coordinates with the
# arm pointing +X. ManipTrans pickles are in IsaacGym frame, related to
# Genesis by R_z(-90°) (see data_utils/convert_manus_to_maniptrans.py).
# Applying that rotation:
#     pos:   R_z(-90) @ (-0.2, 0, 0)         = ( 0,   0.2, 0)
#     orient: R_z(-90) (arm forward = world -Y)
# Quaternion is (qx, qy, qz, qw) = (0, 0, sin(-45°), cos(-45°)).
RIGHT_ARM_BASE_POS = (0.0, 0.2, 0.0)
_HALF_S = -0.7071067811865475   # sin(-45°)
_HALF_C =  0.7071067811865475   # cos(-45°)
RIGHT_ARM_BASE_QUAT = (0.0, 0.0, _HALF_S, _HALF_C)


class XArmWujiHand(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "xarm_wujihand"

        # Body names in IsaacGym order. Arm chain comes first because the
        # URDF declares it first (link_base is the root, palm_link is a
        # downstream fixed-joint child of link_eef).
        arm_body_names = [
            "link_base",
            "link1", "link2", "link3", "link4",
            "link5", "link6", "link7",
            "link_eef",
        ]
        wuji_body_names = [
            "palm_link",
            "finger1_link1", "finger1_link2", "finger1_link3", "finger1_link4", "finger1_tip_link",
            "finger2_link1", "finger2_link2", "finger2_link3", "finger2_link4", "finger2_tip_link",
            "finger3_link1", "finger3_link2", "finger3_link3", "finger3_link4", "finger3_tip_link",
            "finger4_link1", "finger4_link2", "finger4_link3", "finger4_link4", "finger4_tip_link",
            "finger5_link1", "finger5_link2", "finger5_link3", "finger5_link4", "finger5_tip_link",
        ]
        assert len(arm_body_names) == ARM_OFFSET
        self.body_names = arm_body_names + wuji_body_names

        # Arm bodies are not in any MANO mapping — exposed so the env can
        # filter them out of reward / observation joint state.
        self.arm_body_names = arm_body_names

        # Bodies whose rigid_body_state is consumed by the imitation reward
        # and target_joints obs. For wuji-only this is just `body_names`.
        # For xarm we drop the 9 arm bodies so that `joints_state[:, 1:]`
        # still spans the 25 finger bodies (skipping palm at index 0).
        self.joint_state_body_names = wuji_body_names

        # DOF names: 7 arm joints, then 20 finger joints (matching wujihand).
        arm_dof_names = [f"joint{i}" for i in range(1, 8)]
        finger_dof_names = [
            f"finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)
        ]
        self.dof_names = arm_dof_names + finger_dof_names
        assert len(self.dof_names) == 27

        # MANO -> dex body mapping is unchanged in name (the wuji finger
        # link names do not collide with the arm link names).
        self.hand2dex_mapping = {
            "wrist": ["palm_link"],
            "thumb_proximal": ["finger1_link2", "finger1_link1"],
            "thumb_intermediate": ["finger1_link3"],
            "thumb_distal": ["finger1_link4"],
            "thumb_tip": ["finger1_tip_link"],
            "index_proximal": ["finger2_link2", "finger2_link1"],
            "index_intermediate": ["finger2_link3"],
            "index_distal": ["finger2_link4"],
            "index_tip": ["finger2_tip_link"],
            "middle_proximal": ["finger3_link2", "finger3_link1"],
            "middle_intermediate": ["finger3_link3"],
            "middle_distal": ["finger3_link4"],
            "middle_tip": ["finger3_tip_link"],
            "ring_proximal": ["finger4_link2", "finger4_link1"],
            "ring_intermediate": ["finger4_link3"],
            "ring_distal": ["finger4_link4"],
            "ring_tip": ["finger4_tip_link"],
            "pinky_proximal": ["finger5_link2", "finger5_link1"],
            "pinky_intermediate": ["finger5_link3"],
            "pinky_distal": ["finger5_link4"],
            "pinky_tip": ["finger5_tip_link"],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        # arm bodies are not in hand2dex; they have no MANO counterpart.
        # dex2hand only covers wuji bodies, so length check uses wuji subset.
        assert len(self.dex2hand_mapping.keys()) == len(wuji_body_names)

        # Contact bodies: same fingertip links as wujihand.
        self.contact_body_names = [
            "finger1_link4", "finger2_link4",
            "finger3_link4", "finger4_link4",
            "finger5_link4",
        ]

        # Bone links / weight_idx index into `joint_state_body_names`
        # (palm-relative), NOT `body_names` (full chain). This mirrors the
        # wujihand class exactly so the imitation reward — which assumes
        # palm at index 0 — works without any per-dexhand branching.
        self.bone_links = [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],          # thumb
            [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],         # index
            [0, 11], [11, 12], [12, 13], [13, 14], [14, 15], # middle
            [0, 16], [16, 17], [17, 18], [18, 19], [19, 20], # ring
            [0, 21], [21, 22], [22, 23], [23, 24], [24, 25], # pinky
        ]
        self.weight_idx = {
            "thumb_tip": [5],
            "index_tip": [10],
            "middle_tip": [15],
            "ring_tip": [20],
            "pinky_tip": [25],
            "level_1_joints": [1, 6, 11, 16, 21],
            "level_2_joints": [2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24],
        }

        # PID-control wrist gains — unused in the xarm path (no free wrist
        # to integrate), but required by the base class API.
        self.Kp_rot = 0.5
        self.Ki_rot = 0.001
        self.Kd_rot = 0.01
        self.Kp_pos = 20
        self.Ki_pos = 0.005
        self.Kd_pos = 0.1

        # Per-DOF position-drive gains. Arm gains from XARM_WUJI_kp/kd_dict
        # in GenesisPlayground registry; finger gains identical to wujihand.
        self.dof_kp = [
            1414.0, 3015.0, 2065.0, 3015.0,           # joint1..4
            1414.0, 1414.0, 1414.0,                   # joint5..7
            96.86, 96.86, 141.42, 141.42,             # finger1
            96.86, 96.86, 141.42, 141.42,             # finger2
            96.86, 96.86, 141.42, 141.42,             # finger3
            96.86, 96.86, 141.42, 141.42,             # finger4
            96.86, 96.86, 141.42, 141.42,             # finger5
        ]
        self.dof_kd = [
            19.0, 37.0, 27.0, 37.0,
            19.0, 19.0, 19.0,
            5.18, 5.18, 7.20, 7.20,
            5.18, 5.18, 7.20, 7.20,
            5.18, 5.18, 7.20, 7.20,
            5.18, 5.18, 7.20, 7.20,
            5.18, 5.18, 7.20, 7.20,
        ]
        assert len(self.dof_kp) == 27 and len(self.dof_kd) == 27

        self.self_collision = False

        # IK + scene placement metadata consumed by the env. n_arm_dofs /
        # n_arm_bodies are set on the instance so they shadow the (= 0)
        # defaults inherited from DexHand.__init__.
        self.n_arm_dofs = 7
        self.n_arm_bodies = ARM_OFFSET   # number of bodies before palm_link
        self.tcp_yaw = None              # subclass sets per side
        self.arm_seed = None
        self.arm_base_pos = None
        self.arm_base_quat = None        # IsaacGym-frame quat (x, y, z, w)

    def __str__(self):
        return self.name


@register_dexhand("xarm_wujihand_rh")
class XArmWujiHandRH(XArmWujiHand):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/xarm_wujihand/right.urdf"
        self.side = "rh"

        # MANO -> palm_link rotation is the same as standalone wujihand:
        # palm_link's geometry is unchanged by being mounted on the arm.
        self.relative_rotation = (
            aa_to_rotmat(np.array([-np.pi / 2, 0, 0]))
            @ aa_to_rotmat(np.array([-np.pi / 2, 0, 0]))
            @ aa_to_rotmat(np.array([0, -np.pi / 2, 0]))
        )
        self.relative_translation = np.array([0, 0, 0.0592])

        self.tcp_yaw = RIGHT_TCP_YAW
        self.arm_seed = RIGHT_ARM_SEED.copy()
        self.arm_base_pos = RIGHT_ARM_BASE_POS
        self.arm_base_quat = RIGHT_ARM_BASE_QUAT

    def __str__(self):
        return super().__str__() + "_rh"
