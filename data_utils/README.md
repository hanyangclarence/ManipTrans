# Dexterous Hand Trajectory Extraction and Visualization

This directory contains utilities for extracting and visualizing retargeted dexterous hand trajectories with object manipulation data.

## Supported Hand Types

All scripts support the following dexterous hands:
- `shadow` - Shadow Hand
- `inspire` - Inspire Hand
- `inspireftp` - Inspire FTP Hand
- `allegro` - Allegro Hand
- `artimano` - ArtiMano Hand
- `xhand` - XHand
- `wujihand` - Wuji Hand

## Supported Datasets

The scripts work with multiple datasets, each with different index formats:

### GRAB Demo Dataset
- **Index format**: `g{number}` (e.g., `g0`, `g1`, `g2`)
- **Location**: `data/grab_demo/`
- **Example**: `g0` refers to `data/grab_demo/102/`

### OakInk-v2 Dataset
- **Index format**: `{5_digit_hash}@{stage}` (e.g., `07bb1@0`, `07bb1@1`)
- **Location**: `data/OakInk-v2/`
- **Notes**:
  - The 5-digit hash is extracted from the sequence filename
  - Stage number starts from 0 and represents different manipulation phases
  - Use `list_oakink2_sequences.py` to find available sequences
- **Example**: `07bb1@0` refers to stage 0 of sequence `07bb164dc3d3873d6389`

### Finding Available Data

**For OakInk-v2:**
```bash
# List all available sequences
python data_utils/list_oakink2_sequences.py

# Show detailed information for all sequences
python data_utils/list_oakink2_sequences.py --details

# Search for specific sequences
python data_utils/list_oakink2_sequences.py --search "A001"
```

**For GRAB Demo:**
```bash
# List grab demo sequences
ls data/grab_demo/
```

## Scripts Overview

### 1. `extract_shadow_trajectory.py`

Extracts retargeted dexterous hand trajectories and object pose trajectories from preprocessed data. Saves trajectories in IsaacGym coordinates as both pickle files and individual numpy arrays.

### 2. `visualize_shadow_trajectory.py`

Visualizes extracted trajectories in IsaacGym with headless rendering. Outputs rendered frames as images or video.

### 3. `list_oakink2_sequences.py`

Lists all available OakInk-v2 sequences with their indices and metadata. Helps find valid data indices for processing.

### 4. `create_oakink_urdfs.py`

Processes OakInk-v2 object meshes using CoACD decomposition and generates URDF files for IsaacGym simulation. Only needs to be run once to prepare the dataset.

### 5. `debug_hand_orientation.py`

Debugging tool for fixing hand orientation issues. Tests different rotation matrices and analyzes trajectory palm orientation.

---

## Usage Guide

### Step 1: Extract Trajectories

First, you need to extract the retargeted hand and object trajectories from your preprocessed data.

#### Basic Usage

```bash
python data_utils/extract_shadow_trajectory.py --data_idx g0 --hand_type shadow --side right
```

#### Command-Line Arguments

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--data_idx` | str | `g0` | - | Data index (e.g., `g0` for grab_demo/102) |
| `--hand_type` | str | `shadow` | shadow, inspire, inspireftp, allegro, artimano, xhand, wujihand | Type of dexterous hand |
| `--side` | str | `right` | right, left | Hand side |
| `--output_dir` | str | `output_trajectories` | - | Output directory for trajectory files |

#### Examples

**Extract Shadow Hand trajectory (right hand):**
```bash
python data_utils/extract_shadow_trajectory.py \
    --data_idx g0 \
    --hand_type shadow \
    --side right \
    --output_dir my_trajectories
```

**Extract Inspire Hand trajectory (left hand):**
```bash
python data_utils/extract_shadow_trajectory.py \
    --data_idx g0 \
    --hand_type inspire \
    --side left
```

**Extract Wuji Hand trajectory:**
```bash
python data_utils/extract_shadow_trajectory.py \
    --data_idx g0 \
    --hand_type wujihand \
    --side right
```

**Extract from OakInk-v2 dataset:**
```bash
# First, find available sequences
python data_utils/list_oakink2_sequences.py

# Then extract (using the index from the list)
python data_utils/extract_shadow_trajectory.py \
    --data_idx 07bb1@0 \
    --hand_type shadow \
    --side right
```

#### Output Files

The script generates the following files in the output directory:

**Pickle file (complete data):**
- `{hand_type}_hand_trajectory_{data_idx}.pkl` - Contains all trajectory data and metadata

**Numpy arrays (individual components):**
- `object_poses_{data_idx}.npy` - Object pose matrices (T, 4, 4)
- `object_positions_{data_idx}.npy` - Object positions (T, 3)
- `object_velocities_{data_idx}.npy` - Object velocities (T, 3)
- `{hand_type}_wrist_pos_{data_idx}.npy` - Hand wrist positions (T, 3)
- `{hand_type}_wrist_rot_{data_idx}.npy` - Hand wrist rotations in axis-angle (T, 3)
- `{hand_type}_dof_pos_{data_idx}.npy` - Hand joint angles (T, n_dofs)

#### Loading Extracted Data

**Load complete data from pickle:**
```python
import pickle
import numpy as np

# Load complete data
with open('output_trajectories/shadow_hand_trajectory_g0.pkl', 'rb') as f:
    data = pickle.load(f)

# Access object trajectory
obj_poses = data['object_trajectory']['pose_matrices']  # (T, 4, 4)
obj_positions = data['object_trajectory']['positions']   # (T, 3)
obj_velocities = data['object_trajectory']['velocities'] # (T, 3)

# Access hand trajectory
hand_wrist_pos = data['hand_trajectory']['wrist_positions']  # (T, 3)
hand_wrist_rot = data['hand_trajectory']['wrist_rotations_aa']  # (T, 3)
hand_dof_pos = data['hand_trajectory']['dof_positions']      # (T, n_dofs)

# Access metadata
hand_type = data['metadata']['dexhand']  # e.g., 'shadow'
n_dofs = data['metadata']['n_dofs']
fps = data['metadata']['fps']  # 60
```

**Load individual numpy arrays:**
```python
import numpy as np

# Load individual arrays
obj_poses = np.load('output_trajectories/object_poses_g0.npy')
hand_dof = np.load('output_trajectories/shadow_dof_pos_g0.npy')
hand_wrist = np.load('output_trajectories/shadow_wrist_pos_g0.npy')
```

---

### Step 2: Visualize Trajectories

After extracting trajectories, you can visualize them in IsaacGym.

#### Basic Usage

```bash
python data_utils/visualize_shadow_trajectory.py --data_idx g0 --hand_type shadow --side right
```

#### Command-Line Arguments

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--data_idx` | str | `g0` | - | Data index (must match extraction) |
| `--hand_type` | str | `shadow` | shadow, inspire, inspireftp, allegro, artimano, xhand, wujihand | Type of dexterous hand |
| `--side` | str | `right` | right, left | Hand side |
| `--input_dir` | str | `output_trajectories` | - | Directory containing trajectory data |
| `--output_dir` | str | `visualization_output` | - | Directory to save rendered output |
| `--width` | int | `1920` | - | Image width in pixels |
| `--height` | int | `1080` | - | Image height in pixels |
| `--render_every` | int | `2` | - | Render every nth frame (1=all frames, 2=half) |
| `--save_frames` | flag | `False` | - | Save individual frames as PNG images |
| `--save_video` | flag | `True` | - | Save as video (requires imageio) |
| `--compute_device_id` | int | `0` | - | Compute device ID |
| `--graphics_device_id` | int | `0` | - | Graphics device ID |

#### Examples

**Visualize Shadow Hand with default settings:**
```bash
python data_utils/visualize_shadow_trajectory.py \
    --data_idx g0 \
    --hand_type shadow \
    --side right
```

**High-resolution visualization (4K):**
```bash
python data_utils/visualize_shadow_trajectory.py \
    --data_idx g0 \
    --hand_type inspire \
    --side right \
    --width 3840 \
    --height 2160 \
    --render_every 1
```

**Save individual frames instead of video:**
```bash
python data_utils/visualize_shadow_trajectory.py \
    --data_idx g0 \
    --hand_type allegro \
    --side left \
    --save_frames \
    --save_video False
```

**Custom input/output directories:**
```bash
python data_utils/visualize_shadow_trajectory.py \
    --data_idx g0 \
    --hand_type wujihand \
    --side right \
    --input_dir my_trajectories \
    --output_dir my_visualizations
```

**Fast preview (render every 5th frame, lower resolution):**
```bash
python data_utils/visualize_shadow_trajectory.py \
    --data_idx g0 \
    --hand_type shadow \
    --side right \
    --width 1280 \
    --height 720 \
    --render_every 5
```

#### Output Files

The visualization script generates:

**If `--save_video` is enabled (default):**
- `trajectory_{data_idx}.mp4` - Video of the complete trajectory

**If `--save_frames` is enabled:**
- `frame_0000.png`, `frame_0001.png`, ... - Individual rendered frames

---

## Complete Workflow Example

Here's a complete workflow from preprocessing to visualization:

### 1. Run Retargeting Preprocessing

First, ensure you have run the retargeting preprocessing:

```bash
# For Shadow Hand (use 3000 iterations)
python main/dataset/mano2dexhand.py \
    --data_idx g0 \
    --dexhand shadow \
    --side right \
    --headless \
    --iter 3000

# For other hands (use 1000 iterations)
python main/dataset/mano2dexhand.py \
    --data_idx g0 \
    --dexhand inspire \
    --side right \
    --headless \
    --iter 1000
```

### 2. Extract Trajectories

```bash
python data_utils/extract_shadow_trajectory.py \
    --data_idx g0 \
    --hand_type shadow \
    --side right \
    --output_dir output_trajectories
```

**Expected output:**
```
================================================================================
EXTRACTING SHADOW HAND TRAJECTORY
================================================================================

Dexterous hand: Shadow Hand (right)
Number of DOFs: 22
Dataset type: grab

✓ Retargeted Shadow Hand trajectory FOUND

Sequence length: 180 frames (3.00 seconds at 60 FPS)

TRAJECTORY STATISTICS (IsaacGym Coordinates)
...

✓ Saved complete trajectory to: output_trajectories/shadow_hand_trajectory_g0.pkl
✓ Saved individual arrays to: output_trajectories/
```

### 3. Visualize Trajectories

```bash
python data_utils/visualize_shadow_trajectory.py \
    --data_idx g0 \
    --hand_type shadow \
    --side right \
    --input_dir output_trajectories \
    --output_dir visualization_output
```

**Expected output:**
```
================================================================================
Dexterous Hand Trajectory Visualizer
================================================================================

Configuration:
  Hand: Shadow
  Data: g0 (right hand)
  Input: output_trajectories
  Output: visualization_output
  Resolution: 1920x1080
  Render every: 2 frame(s)

Loading trajectory from: output_trajectories/shadow_hand_trajectory_g0.pkl
Loaded trajectory:
  - Frames: 180 (3.00s at 60 FPS)
  - Shadow Hand DOFs: 22
  - Hand side: right

Rendering 90 frames (every 2 frame)...
[Progress bar]

✓ Video saved successfully!
✓ Visualization complete!
  Output directory: visualization_output
```

---

## OakInk-v2 Workflow Example

Here's a complete workflow for processing OakInk-v2 data:

### 0. Generate Object URDFs (First Time Only)

OakInk-v2 objects need to be processed with CoACD to create URDF files for simulation:

```bash
# Process all objects (this may take a while)
python data_utils/create_oakink_urdfs.py

# Process specific objects only
python data_utils/create_oakink_urdfs.py --objects O02@0032@00001 O02@0032@00002

# Use finer decomposition for better collision
python data_utils/create_oakink_urdfs.py --threshold 0.02 --max-convex-hull 64
```

**What this does:**
1. Uses CoACD to decompose object meshes into convex parts
2. Creates URDF files in `data/OakInk-v2/coacd_object_preview/`
3. Each URDF references both the original mesh (visual) and decomposed mesh (collision)

**Note:** This only needs to be done once. The generated URDFs are reused for all sequences.

### 1. Find Available Sequences

```bash
python data_utils/list_oakink2_sequences.py
```

**Output:**
```
Found 1 sequences in OakInk-v2
================================================================================
Index: 07bb1@0  (or @1, @2, etc. for different stages)
  File: scene_01__A001++seq__07bb164dc3d3873d6389__2023-04-27-20-45-29.pkl
  Frames: 10449 (at 120Hz)
================================================================================
```

### 2. Run Retargeting for OakInk-v2

```bash
# Retarget with wujihand
python main/dataset/mano2dexhand.py \
    --data_idx 07bb1@0 \
    --dexhand wujihand \
    --side right \
    --headless \
    --iter 1000
```

**Note**: OakInk-v2 sequences can be very long (10,000+ frames at 120Hz, downsampled to 60Hz). The retargeting may take some time.

### 3. Extract Trajectory

```bash
python data_utils/extract_shadow_trajectory.py \
    --data_idx 07bb1@0 \
    --hand_type wujihand \
    --side right \
    --output_dir oakink_trajectories
```

### 4. Visualize

```bash
python data_utils/visualize_shadow_trajectory.py \
    --data_idx 07bb1@0 \
    --hand_type wujihand \
    --side right \
    --input_dir oakink_trajectories \
    --output_dir oakink_visualizations \
    --render_every 5  # Render every 5th frame for faster processing
```

### 5. Process Different Stages

If the sequence has multiple stages (manipulation phases):

```bash
# Stage 0
python main/dataset/mano2dexhand.py --data_idx 07bb1@0 --dexhand wujihand --side right --headless --iter 1000

# Stage 1
python main/dataset/mano2dexhand.py --data_idx 07bb1@1 --dexhand wujihand --side right --headless --iter 1000

# And so on...
```

Then extract and visualize each stage separately.

---

## Batch Processing Multiple Trajectories

Process multiple trajectories in a loop:

```bash
#!/bin/bash

# Define hand types and data indices
HAND_TYPES=("shadow" "inspire" "allegro")
DATA_INDICES=("g0" "g1" "g2")

# Extract all trajectories
for hand in "${HAND_TYPES[@]}"; do
    for data_idx in "${DATA_INDICES[@]}"; do
        echo "Extracting ${hand} hand for ${data_idx}..."
        python data_utils/extract_shadow_trajectory.py \
            --data_idx ${data_idx} \
            --hand_type ${hand} \
            --side right \
            --output_dir trajectories_${hand}
    done
done

# Visualize all trajectories
for hand in "${HAND_TYPES[@]}"; do
    for data_idx in "${DATA_INDICES[@]}"; do
        echo "Visualizing ${hand} hand for ${data_idx}..."
        python data_utils/visualize_shadow_trajectory.py \
            --data_idx ${data_idx} \
            --hand_type ${hand} \
            --side right \
            --input_dir trajectories_${hand} \
            --output_dir visualizations_${hand}
    done
done
```

---

## Troubleshooting

### Error: "Retargeted trajectory NOT FOUND"

**Solution:** Run the preprocessing first:
```bash
python main/dataset/mano2dexhand.py \
    --data_idx g0 \
    --dexhand shadow \
    --side right \
    --headless \
    --iter 3000  # Use 3000 for shadow, 1000 for others
```

### Error: "Trajectory file not found"

**Solution:** Make sure the `--input_dir` for visualization matches the `--output_dir` from extraction, and that `--hand_type`, `--data_idx`, and `--side` are the same.

### Video encoding fails

**Solution:** Install video encoding support:
```bash
pip install imageio-ffmpeg
# or
pip install 'imageio[pyav]'
```

### Out of memory during visualization

**Solution:** Reduce resolution or increase `--render_every`:
```bash
python data_utils/visualize_shadow_trajectory.py \
    --data_idx g0 \
    --hand_type shadow \
    --side right \
    --width 1280 \
    --height 720 \
    --render_every 4
```

---

## Requirements

- Python 3.8+
- IsaacGym
- PyTorch
- NumPy
- Pillow
- tqdm
- scipy
- imageio (optional, for video output)
- imageio-ffmpeg or imageio[pyav] (optional, for video encoding)

---

## Coordinate System

All trajectories are in **IsaacGym coordinates**:
- **X**: forward
- **Y**: left
- **Z**: up
- **Ground plane**: Z=0
- **Table surface**: Z≈0.415

Wrist rotations are stored as axis-angle representations (3D vectors).

---

## Notes

- **Shadow Hand** requires more optimization iterations (3000) compared to other hands (1000)
- Trajectory extraction creates both complete pickle files and individual numpy arrays for flexibility
- Visualization uses headless rendering, suitable for remote servers
- The `--render_every` parameter helps reduce output size and rendering time
- All scripts support both left and right hands
