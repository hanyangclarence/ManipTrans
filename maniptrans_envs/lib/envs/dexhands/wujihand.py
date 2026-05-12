from .base import DexHand
from .decorators import register_dexhand
from abc import ABC
import numpy as np
from main.dataset.transform import aa_to_rotmat


class WujiHand(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "wujihand"

        # Body names in IsaacGym order (from URDF)
        # Note: finger1 is thumb, finger2-5 are index, middle, ring, pinky
        self.body_names = [
            "palm_link",
            # Thumb (finger1)
            "finger1_link1",
            "finger1_link2",
            "finger1_link3",
            "finger1_link4",
            "finger1_tip_link",
            # Index (finger2)
            "finger2_link1",
            "finger2_link2",
            "finger2_link3",
            "finger2_link4",
            "finger2_tip_link",
            # Middle (finger3)
            "finger3_link1",
            "finger3_link2",
            "finger3_link3",
            "finger3_link4",
            "finger3_tip_link",
            # Ring (finger4)
            "finger4_link1",
            "finger4_link2",
            "finger4_link3",
            "finger4_link4",
            "finger4_tip_link",
            # Pinky (finger5)
            "finger5_link1",
            "finger5_link2",
            "finger5_link3",
            "finger5_link4",
            "finger5_tip_link",
        ]

        # DOF names (actuated joints only, excluding fixed tip joints)
        self.dof_names = [
            # Thumb (finger1) - 4 DOF
            "finger1_joint1",
            "finger1_joint2",
            "finger1_joint3",
            "finger1_joint4",
            # Index (finger2) - 4 DOF
            "finger2_joint1",
            "finger2_joint2",
            "finger2_joint3",
            "finger2_joint4",
            # Middle (finger3) - 4 DOF
            "finger3_joint1",
            "finger3_joint2",
            "finger3_joint3",
            "finger3_joint4",
            # Ring (finger4) - 4 DOF
            "finger4_joint1",
            "finger4_joint2",
            "finger4_joint3",
            "finger4_joint4",
            # Pinky (finger5) - 4 DOF
            "finger5_joint1",
            "finger5_joint2",
            "finger5_joint3",
            "finger5_joint4",
        ]

        # Mapping from MANO hand keypoints to WujiHand bodies
        # This is critical for retargeting
        self.hand2dex_mapping = {
            "wrist": ["palm_link"],
            # Thumb (finger1 in URDF)
            "thumb_proximal": ["finger1_link2", "finger1_link1"],
            "thumb_intermediate": ["finger1_link3"],
            "thumb_distal": ["finger1_link4"],  # map to both link3 and link4
            "thumb_tip": ["finger1_tip_link"],
            # Index (finger2 in URDF)
            "index_proximal": ["finger2_link2", "finger2_link1"],  # link1 is abduction
            "index_intermediate": ["finger2_link3"],
            "index_distal": ["finger2_link4"],
            "index_tip": ["finger2_tip_link"],
            # Middle (finger3 in URDF)
            "middle_proximal": ["finger3_link2", "finger3_link1"],
            "middle_intermediate": ["finger3_link3"],
            "middle_distal": ["finger3_link4"],
            "middle_tip": ["finger3_tip_link"],
            # Ring (finger4 in URDF)
            "ring_proximal": ["finger4_link2", "finger4_link1"],
            "ring_intermediate": ["finger4_link3"],
            "ring_distal": ["finger4_link4"],
            "ring_tip": ["finger4_tip_link"],
            # Pinky (finger5 in URDF)
            "pinky_proximal": ["finger5_link2", "finger5_link1"],
            "pinky_intermediate": ["finger5_link3"],
            "pinky_distal": ["finger5_link4"],
            "pinky_tip": ["finger5_tip_link"],
        }

        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)

        # Bodies that are expected to contact objects (fingertips)
        # Note: in v5 URDF, the tip geometry is baked into finger*_link4, and
        # finger*_tip_link has no visual/collision — contacts register on link4.
        self.contact_body_names = [
            "finger1_link4",  # thumb
            "finger2_link4",  # index
            "finger3_link4",  # middle
            "finger4_link4",  # ring
            "finger5_link4",  # pinky
        ]

        # Bone links for visualization (connect body indices)
        # Format: [start_body_idx, end_body_idx]
        self.bone_links = [
            # Thumb chain
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
            # Index chain
            [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],
            # Middle chain
            [0, 11], [11, 12], [12, 13], [13, 14], [14, 15],
            # Ring chain
            [0, 16], [16, 17], [17, 18], [18, 19], [19, 20],
            # Pinky chain
            [0, 21], [21, 22], [22, 23], [23, 24], [24, 25],
        ]

        # Weight indices for training loss
        # Higher weights for fingertips and critical joints
        self.weight_idx = {
            "thumb_tip": [5],
            "index_tip": [10],
            "middle_tip": [15],
            "ring_tip": [20],
            "pinky_tip": [25],
            # Level 1: Critical joints (base joints)
            "level_1_joints": [1, 6, 11, 16, 21],
            # Level 2: Other joints
            "level_2_joints": [2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24],
        }

        # PID control parameters (if using PID mode)
        self.Kp_rot = 0.5
        self.Ki_rot = 0.001
        self.Kd_rot = 0.01
        self.Kp_pos = 20
        self.Ki_pos = 0.005
        self.Kd_pos = 0.1

        # Per-DOF position-drive gains, ordered to match self.dof_names. Read by
        # dexhandimitator / dexhandmanip_sh in place of the global 500/30 defaults
        # because WujiHand's URDF effort limits (~0.15-0.65 N·m, since released to
        # 100) and small finger inertias don't tolerate the default damping; with
        # Kd=30, finger DOFs saturate before they can move.
        self.dof_kp = [
            96.86, 96.86, 141.42, 141.42,  # finger1 (thumb) joint1..4
            96.86, 96.86, 141.42, 141.42,  # finger2 (index)
            96.86, 96.86, 141.42, 141.42,  # finger3 (middle)
            96.86, 96.86, 141.42, 141.42,  # finger4 (ring)
            96.86, 96.86, 141.42, 141.42,  # finger5 (pinky)
        ]
        self.dof_kd = [
            5.18, 5.18, 7.20, 7.20,
            5.18, 5.18, 7.20, 7.20,
            5.18, 5.18, 7.20, 7.20,
            5.18, 5.18, 7.20, 7.20,
            5.18, 5.18, 7.20, 7.20,
        ]

        # Enable self-collision checking if needed
        self.self_collision = False

    def __str__(self):
        return self.name


@register_dexhand("wujihand_rh")
class WujiHandRH(WujiHand):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/wujihand/urdf/right.urdf"
        self.side = "rh"

        # Relative rotation from MANO wrist to WujiHand wrist
        # Calibrated to match Inspire hand retargeting results
        # Sequence: -90° Y → -90° X → -90° X (correction)
        self.relative_rotation = (
            aa_to_rotmat(np.array([-np.pi / 2, 0, 0]))  # -90° X (correction)
            @ aa_to_rotmat(np.array([-np.pi / 2, 0, 0]))  # -90° X
            @ aa_to_rotmat(np.array([0, -np.pi / 2, 0]))  # -90° Y
        )

        # Relative translation to correct height offset
        # WujiHand wrist is 6.92cm lower, so offset upward
        self.relative_translation = np.array([0, 0, 0.0592])

    def __str__(self):
        return super().__str__() + "_rh"


@register_dexhand("wujihand_lh")
class WujiHandLH(WujiHand):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/wujihand/urdf/left.urdf"
        self.side = "lh"

        # Mirror rotation for left hand
        self.relative_rotation = (
            aa_to_rotmat(np.array([0, 0, np.pi / 2]))
        )

        self.relative_translation = np.zeros(3)

    def __str__(self):
        return super().__str__() + "_lh"
