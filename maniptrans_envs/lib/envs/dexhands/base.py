from abc import ABC, abstractmethod
import os
import numpy as np


class DexHand(ABC):
    def __init__(self):
        self._urdf_path = None
        self.side = None
        self.name = None
        self.body_names = None
        self.dof_names = None
        self.hand2dex_mapping = None
        self.dex2hand_mapping = None
        self.relative_rotation = None
        self.relative_translation = np.zeros(3)
        self.contact_body_names = None
        self.bone_links = None
        self.weight_idx = None
        self.self_collision = False

        # Defaults for arm-mounted-chain support; see the docstring above.
        # Subclasses may overwrite these in __init__ after super().__init__().
        self.joint_state_body_names = None  # falls back to body_names below
        self.arm_body_names = []
        self.n_arm_dofs = 0
        self.n_arm_bodies = 0

        # ? >>>>>>>>>>>
        # ? Used only in PID-controlled wrist pose mode (reference only, not our main method).
        # ? More stable in highly dynamic scenarios but requires careful tuning.
        self.Kp_rot = None
        self.Ki_rot = None
        self.Kd_rot = None
        self.Kp_pos = None
        self.Ki_pos = None
        self.Kd_pos = None
        # ? <<<<<<<<<<

    @abstractmethod
    def __str__(self):
        pass

    def to_dex(self, hand_body):
        return self.hand2dex_mapping[hand_body]

    def to_hand(self, dex_body):
        return self.dex2hand_mapping[dex_body]

    @property
    def n_dofs(self):
        return len(self.dof_names)

    @property
    def n_bodies(self):
        return len(self.body_names)

    # --- arm-mounted-chain support (default = floating-hand) --------------
    # Subclasses representing an arm + dexhand combo (e.g. xarm_wujihand)
    # override these in __init__ so the env can branch its scene/control
    # logic without hard-coding hand names.
    #
    # joint_state_body_names: subset of body_names whose rigid_body_state
    #   is consumed by the imitation reward / joint obs. Defaults to ALL
    #   bodies; arm-mounted classes drop arm bodies so the reward's
    #   `joints_state[:, 1:]` slice still spans wrist-skipped finger bodies.
    # n_arm_dofs: arm DOFs preceding finger DOFs in dof_state. 0 = floating.
    # n_arm_bodies: arm rigid bodies preceding the wrist (palm). 0 = floating.

    @property
    def urdf_path(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(project_root, "../../..", self._urdf_path)

    @staticmethod
    def reverse_mapping(mapping):
        reverse = {}
        for key, values in mapping.items():
            if values is None:
                continue
            for value in values:
                if value in reverse:
                    reverse[value].append(key)
                else:
                    reverse[value] = [key]

        return reverse
