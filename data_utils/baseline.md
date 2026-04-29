# Manus → ManipTrans baseline pipeline

End-to-end recipe for taking Manus / Genesis-frame hand-object trajectory
pickles and turning them into ManipTrans IsaacGym demo + retargeting files,
plus a sanity-check visualizer.

## 1. Source pickle format

Each input pickle (e.g. `data_v2/marker_pen/pickles_filtered/<stem>.pkl`)
must contain:

```
hand_trajectory.wrist_positions     (T, 3)   float64       wujihand palm world pos
hand_trajectory.wrist_rotations_aa  (T, 3)   float64       axis-angle, world frame
hand_trajectory.dof_positions       (T, 20)  float32       wujihand finger DOFs
object_trajectory.pose_matrices     (T, 4, 4) float64      object SE(3), world frame
mano_reference.finger_joints        dict of (T, 3)         must include 5 *_tip keys
metadata.obj_id                      str
metadata.hand_side                   'right' | 'left'
metadata.num_frames                  int
```

Frame convention: **Z up, table top at z = 0, 60 Hz**. The directory must
also contain `<obj_id>_collision.obj` (or pass `--obj_mesh`).

## 2. Convert

```bash
python data_utils/convert_manus_to_maniptrans.py \
  --input_dir <DIR_OF_PICKLES> \
  --output_dir data/<DATASET_NAME>
```

Per-frame transform applied (Genesis → IsaacGym):

```
p_gym = R_z(-π/2) · p_user + [0, 0, 0.415]
R_gym = R_z(-π/2) · R_user
```

The `+0.415` lift puts your `z = 0` (your table) on top of the training
env's table at IsaacGym `z = 0.415`.

Output layout:

```
data/<DATASET_NAME>/
├── ObjURDF/
│   ├── <obj_id>.urdf                generated 5-line wrap
│   └── <obj_id>_collision.obj       copied
├── sequences/<stem>.pkl             demo pickle (gym frame, see below)
├── retargeting/<stem>.pkl           opt_wrist_pos / opt_wrist_rot / opt_dof_pos
└── sequences.json                   index
```

`sequences/<stem>.pkl` keys (consumed by your `ManipData` subclass):

```
data_path, obj_id, side, fps,
obj_urdf_path, obj_verts (1000, 3),
obj_trajectory   (T, 4, 4)
wrist_pos        (T, 3)
wrist_rot        (T, 3, 3)         rotation matrix — ManipTrans's contract
mano_joints      {5 *_tip keys: (T, 3)}
```

`retargeting/<stem>.pkl`:

```
opt_wrist_pos (T, 3), opt_wrist_rot (T, 3) axis-angle, opt_dof_pos (T, 20)
```

## 3. Loader-side glue (still to write)

A `ManipData` subclass at `main/dataset/manus_dataset_dexhand_rh.py`:

- Read `sequences/<stem>.pkl` and return its dict; call
  `self.process_data(...)` then
  `self.load_retargeted_data(data, retargeting/<stem>.pkl)`.
- In `__init__`, set `self.mujoco2gym_transf = torch.eye(4, ...)` so
  `process_data` doesn't re-rotate (data is already gym frame).
- Set `self.skip = 2` so `process_data`'s velocity time-delta
  `self.skip / 120 = 1/60 s` matches 60 Hz; **don't** subsample.
- Add a dispatch case in `main/dataset/factory.py:dataset_type` (e.g.
  `manus@<stem>` → `"manus"`).
- `mano_joints` only carries the 5 `*_tip` keys, so patch
  `dexhandmanip_sh.py:533-545` (`pack_data`) and the reward to use those
  five keys instead of iterating all `dexhand.body_names`.

Then:

```bash
# stage 1: imitator
python main/rl/train.py task=DexHandImitator dexhand=wujihand side=RH \
  dataIndices=[manus@<stem>] num_envs=4096 ...

# stage 2: residual
python main/rl/train.py task=ResDexHand dexhand=wujihand side=RH \
  dataIndices=[manus@<stem>] \
  rh_base_model_checkpoint=runs/.../imit_*.pth \
  lh_base_model_checkpoint=runs/.../imit_*.pth ...
```

## 4. Visualize

```bash
python data_utils/visualize_shadow_trajectory.py \
  --data_idx <stem> --hand_type wujihand --side right \
  --input_dir data/<DATASET_NAME> --output_dir <OUT> \
  --width 960 --height 540 --render_every 4 \
  --save_video --camera_view iso          # or --camera_view side
```

Auto-detects the converted format (`<input_dir>/sequences/<stem>.pkl`).
Falls back to the legacy `<hand_type>_hand_trajectory_<stem>.pkl` schema
otherwise. The rendered scene includes the same 1.0 × 1.6 × 0.03 m table
the training env spawns, so you see the pen/hammer resting on the same
surface PhysX uses at training time.

Useful flags:

- `--camera_view side` — eye-level with the table top, useful for
  checking object-on-table contact.
- `--render_every N` — sample every Nth source frame (15 fps at N=4).
- `--freeze_frame K` — pin every rendered step to source frame K (debug).

### Kinematic playback note

The visualizer pins both hand and object with `fix_base_link=True` and
disables inter-actor collision (`filter=-1`). This is essential for
**replay only** — without it, PhysX integrates an apparent angular
velocity from each per-frame `set_actor_root_state_tensor` teleport and
adds visible jitter (~2.5× the source data's per-step rotation delta).
**Training does not have this problem** because resets happen once per
episode, not every step; the hand and object remain free dynamic bodies
under policy control during rollout.

## 5. Known caveats

- Source data has **real noise of 1–13°/frame** in object rotation, even
  when the object is supposed to be at rest on the table. The conversion
  is bit-exact (per-frame Δθ matches between original and converted), so
  any rotation jitter you see is in the source pickles. Add a SLERP /
  Savitzky-Golay smoothing pass to the converter if it bothers training.
- The pen/hammer mesh origin is at the **geometric center**, so its
  bottom face sits ~1 cm below the recorded center. Frame-0 has ~10 mm
  of table interpenetration; PhysX resolves it harmlessly during the
  first few sim steps.
- Object mass in the generated URDF defaults to 50 g with diagonal
  inertia 1e-5; the training env overrides via density (200 kg/m³) so
  this is fine for marker_pen / hammer.
