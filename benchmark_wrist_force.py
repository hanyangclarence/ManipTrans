"""
Compare the floating-wrist response of two dexhands under IDENTICAL force/torque
commands. Replicates exactly the wrist-control pipeline of the Stage-1 imitator
(`dexhandimitator.py:958-974`):

    force  = actions[:, 0:3] * dt * translationScale * 500
    torque = actions[:, 3:6] * dt * orientationScale  * 200
    gym.apply_rigid_body_force_tensors(...)   # applied for one step

Also matches the same wrist asset options the env loads with:
    fix_base_link=False, disable_gravity=True,
    angular_damping=20, linear_damping=20,
    max_linear_velocity=50, max_angular_velocity=100

Spawns one env per hand in a shared sim so frame timing / solver settings are
identical across hands, then steps physics with a fixed unit input on +X (force)
and +Z (torque) separately. Prints per-hand wrist mass, terminal velocity,
displacement after N steps, and time-to-50%-of-terminal-velocity.

Usage:
    python benchmark_wrist_force.py
    python benchmark_wrist_force.py --hands inspire,wujihand --steps 120 \
                                    --action 1.0 --translation_scale 1.0 \
                                    --orientation_scale 0.1
"""

import argparse
import os

# IsaacGym must be imported before torch.
from isaacgym import gymapi, gymtorch  # noqa: F401

import numpy as np
import torch

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", default="inspire,wujihand",
                    help="comma-separated list of dexhand names")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--steps", type=int, default=120,
                    help="physics steps for each test (default 120 = 2s)")
    ap.add_argument("--action", type=float, default=1.0,
                    help="action magnitude in [-1,1]; 1.0 is the max policy can send")
    ap.add_argument("--translation_scale", type=float, default=1.0)
    ap.add_argument("--orientation_scale", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    return ap.parse_args()


def make_sim(dt):
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.dt = dt
    sp.substeps = 1
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 1
    sp.physx.use_gpu = True
    sp.use_gpu_pipeline = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    return gym, sim


def load_hand_asset(gym, sim, urdf_path):
    asset_root, asset_file = os.path.split(os.path.abspath(urdf_path))
    opts = gymapi.AssetOptions()
    # Match dexhandimitator.py:241-251
    opts.thickness = 0.001
    opts.angular_damping = 20
    opts.linear_damping = 20
    opts.max_linear_velocity = 50
    opts.max_angular_velocity = 100
    opts.fix_base_link = False
    opts.disable_gravity = True
    opts.flip_visual_attachments = False
    opts.collapse_fixed_joints = False
    opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
    opts.use_mesh_materials = True
    return gym.load_asset(sim, asset_root, asset_file, opts)


def main():
    args = parse_args()
    hand_names = [h.strip() for h in args.hands.split(",") if h.strip()]

    # Build dexhand objects (for urdf_path + wrist body name + n_dofs).
    dexhands = {name: DexHandFactory.create_hand(name, args.side) for name in hand_names}

    gym, sim = make_sim(args.dt)

    # One env per hand; envs are spaced apart in y so the hands don't see each other.
    SPACING = 2.0
    assets = {}
    envs = {}
    hand_actors = {}
    wrist_body_idx = {}
    for i, name in enumerate(hand_names):
        dh = dexhands[name]
        asset = load_hand_asset(gym, sim, dh.urdf_path)
        env = gym.create_env(sim,
                             gymapi.Vec3(-SPACING / 2, i * SPACING - SPACING / 2, 0.0),
                             gymapi.Vec3( SPACING / 2, i * SPACING + SPACING / 2, SPACING),
                             1)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, i * SPACING, 0.5)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        actor = gym.create_actor(env, asset, pose, f"{name}_actor", i, 0)

        # Match imitator: set DOF driveMode + per-DOF stiffness/damping (using the
        # same logic as the real task — including the new dof_kp/dof_kd override).
        dof_props = gym.get_asset_dof_properties(asset)
        _kp = getattr(dh, "dof_kp", None)
        _kd = getattr(dh, "dof_kd", None)
        kp_list = list(_kp) if _kp is not None else [500] * dh.n_dofs
        kd_list = list(_kd) if _kd is not None else [30] * dh.n_dofs
        for j in range(dh.n_dofs):
            dof_props["driveMode"][j] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][j] = float(kp_list[j])
            dof_props["damping"][j] = float(kd_list[j])
        gym.set_actor_dof_properties(env, actor, dof_props)

        # Locate the wrist body index (first body of dexhand.body_names — palm or hand_base).
        wrist_name = dh.to_dex("wrist")[0]
        widx = gym.find_actor_rigid_body_index(env, actor, wrist_name, gymapi.DOMAIN_SIM)
        if widx < 0:
            raise RuntimeError(f"{name}: couldn't find wrist body '{wrist_name}'")
        wrist_body_idx[name] = widx

        assets[name] = asset
        envs[name] = env
        hand_actors[name] = actor

        # Print wrist body mass (raw URDF inertial.mass — the IsaacGym wrist body).
        rb_props = gym.get_actor_rigid_body_properties(env, actor)
        wrist_local_idx = gym.find_actor_rigid_body_index(env, actor, wrist_name, gymapi.DOMAIN_ACTOR)
        m = rb_props[wrist_local_idx].mass
        com = rb_props[wrist_local_idx].com
        print(f"  {name:10s}  wrist_body='{wrist_name}'  mass={m:.4f} kg  "
              f"com=({com.x:+.3f},{com.y:+.3f},{com.z:+.3f})  n_dofs={dh.n_dofs}")

    gym.prepare_sim(sim)

    # Acquire root + rigid-body state tensors.
    rb_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    n_envs = len(hand_names)
    # rb_state shape: (n_bodies_total, 13) where 13 = [pos(3), quat(4), lin_vel(3), ang_vel(3)]

    # Force tensor: one row per rigid body in the sim.
    n_bodies_total = rb_state.shape[0]
    apply_forces = torch.zeros((n_bodies_total, 3), device="cuda:0", dtype=torch.float32)
    apply_torques = torch.zeros((n_bodies_total, 3), device="cuda:0", dtype=torch.float32)

    def reset():
        # Hard-reset all rigid body states by re-setting root + step once with zero force.
        # Simpler: just reset the entire sim by recreating env states from sim defaults.
        # For this benchmark we rely on starting from rest and not driving DOFs.
        apply_forces.zero_()
        apply_torques.zero_()
        gym.refresh_rigid_body_state_tensor(sim)

    def run_test(label, axis, mode):
        """mode in {'force','torque'}; axis in {0,1,2}."""
        # Per-step physical magnitude matching dexhandimitator.py:958-967.
        if mode == "force":
            mag = args.action * args.dt * args.translation_scale * 500.0
        else:
            mag = args.action * args.dt * args.orientation_scale * 200.0

        apply_forces.zero_()
        apply_torques.zero_()
        for name in hand_names:
            target = apply_forces if mode == "force" else apply_torques
            target[wrist_body_idx[name], axis] = mag

        # Record trajectories.
        log = {name: {"pos": [], "lin_v": [], "ang_v": []} for name in hand_names}

        for step in range(args.steps):
            # apply_rigid_body_force_tensors applies for ONE step — must call every step
            gym.apply_rigid_body_force_tensors(
                sim,
                gymtorch.unwrap_tensor(apply_forces),
                gymtorch.unwrap_tensor(apply_torques),
                gymapi.ENV_SPACE,
            )
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_rigid_body_state_tensor(sim)

            for name in hand_names:
                row = rb_state[wrist_body_idx[name]]
                log[name]["pos"].append(row[:3].cpu().numpy().copy())
                log[name]["lin_v"].append(row[7:10].cpu().numpy().copy())
                log[name]["ang_v"].append(row[10:13].cpu().numpy().copy())

        print(f"\n  === {label}  ({mode} on axis={axis}, action={args.action}, "
              f"per-step magnitude={mag:.4f} {'N' if mode=='force' else 'N·m'}) ===")
        print(f"  {'hand':<10s} {'final_pos_axis':>15s} {'final_lin_v':>22s} "
              f"{'final_ang_v':>22s} {'|v|_term':>9s} {'t_to_50%':>9s}")
        for name in hand_names:
            pos = np.array(log[name]["pos"])
            lv  = np.array(log[name]["lin_v"])
            av  = np.array(log[name]["ang_v"])
            # Relative pos (origin per hand differs by y offset).
            init = pos[0]
            disp = pos[-1] - init
            v_norm = np.linalg.norm(lv, axis=1)
            term_v = v_norm[-1]
            # Time to 50% of terminal |v| (in steps).
            t_half = int(np.searchsorted(v_norm, 0.5 * term_v)) if term_v > 0 else -1
            print(f"  {name:<10s} disp=({disp[0]:+.3f},{disp[1]:+.3f},{disp[2]:+.3f}) m  "
                  f"v=({lv[-1,0]:+.2f},{lv[-1,1]:+.2f},{lv[-1,2]:+.2f}) m/s  "
                  f"w=({av[-1,0]:+.2f},{av[-1,1]:+.2f},{av[-1,2]:+.2f}) rad/s  "
                  f"|v|={term_v:7.3f}  t50={t_half:3d}step")
        # Also reset (zero out forces, settle).
        apply_forces.zero_()
        apply_torques.zero_()
        for _ in range(5):
            gym.apply_rigid_body_force_tensors(
                sim,
                gymtorch.unwrap_tensor(apply_forces),
                gymtorch.unwrap_tensor(apply_torques),
                gymapi.ENV_SPACE,
            )
            gym.simulate(sim)
            gym.fetch_results(sim, True)

    print("\nWrist body summary:")
    print(f"  (asset_options.linear_damping=20, angular_damping=20, "
          f"max_lin_v=50, max_ang_v=100)")
    print(f"  dt={args.dt:.4f}s  steps={args.steps}  total_t={args.steps*args.dt:.2f}s\n")

    print("== FORCE TEST: +X axis ==")
    run_test("force_+X", axis=0, mode="force")
    print("\n== FORCE TEST: +Z axis ==")
    run_test("force_+Z", axis=2, mode="force")
    print("\n== TORQUE TEST: +Z axis (yaw) ==")
    run_test("torque_+Z", axis=2, mode="torque")
    print("\n== TORQUE TEST: +X axis (roll) ==")
    run_test("torque_+X", axis=0, mode="torque")

    gym.destroy_sim(sim)
    print("\ndone.")


if __name__ == "__main__":
    main()
