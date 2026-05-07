#!/bin/bash
# Evaluate a Stage-2 OakInk2 policy with per-trajectory progress stats.
#
# For the run's training trajectories (read back from runs/<run>/config.yaml),
# spins up one env per dataIndex starting at frame 0 with deterministic
# actions, lets each env terminate exactly once, and prints/dumps:
#   - per-trajectory progress, progress_ratio, status
#   - aggregate mean steps, mean progress ratio, success/fail/timeout rate
#
# Usage:
#   ./eval_oakink2_stage2.sh <run_dir> [checkpoint] [stats_out]
#
#     <run_dir>     : runs/oakink2_inspire_<obj>_n<N>_<ts>__... (i.e. the
#                     full path produced by submit_train_stage2_oakink2.sh)
#     [checkpoint]  : optional path to a specific .pth in the run's nn/.
#                     Default: runs/<run>/nn/<experiment>.pth (best-by-reward).
#     [stats_out]   : optional path to write the per-traj/overall JSON.
#                     Default: runs/<run>/eval_progress_stats.json
#
# Environment overrides (existing training conventions): test=true,
# randomStateInit=false, headless=true, num_envs=len(dataIndices),
# eval_progress_stats=true, wandb_activate=False.

set -eo pipefail

RUN_DIR="${1:?usage: $0 <run_dir> [checkpoint] [stats_out]}"
CKPT="${2:-}"
STATS_OUT="${3:-}"

if [ ! -d "$RUN_DIR" ]; then
    echo "Run directory '$RUN_DIR' not found." >&2
    exit 1
fi
CONFIG="$RUN_DIR/config.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "Config '$CONFIG' not found (was the run cancelled before write?)." >&2
    exit 1
fi

# Extract experiment + dexhand + dataIndices from the saved hydra config.
EXPERIMENT=$(grep -E "^experiment:" "$CONFIG" | head -1 | awk '{print $2}')
DEXHAND=$(grep -E "^dexhand:" "$CONFIG" | head -1 | awk '{print $2}')
SIDE=$(grep -E "^side:" "$CONFIG" | head -1 | awk '{print $2}')
ACTIONS_MA=$(grep -E "^actionsMovingAverage:" "$CONFIG" | head -1 | awk '{print $2}')
USE_PID=$(grep -E "^usePIDControl:" "$CONFIG" | head -1 | awk '{print $2}')
RH_BASE=$(grep -E "^rh_base_model_checkpoint:" "$CONFIG" | head -1 | awk '{print $2}')
LH_BASE=$(grep -E "^lh_base_model_checkpoint:" "$CONFIG" | head -1 | awk '{print $2}')

# Hydra dumps dataIndices as a YAML list of "- 03865@1" lines. Pull the
# values, strip the bullet, comma-join.
INDICES=$(awk '
    /^dataIndices:/ { in_list=1; next }
    in_list {
        if ($0 ~ /^[a-zA-Z_]/) { in_list=0; next }
        sub(/^[[:space:]]*-[[:space:]]*/, "", $0)
        if ($0 != "") print $0
    }
' "$CONFIG" | paste -sd ,)

if [ -z "$INDICES" ]; then
    echo "Could not parse dataIndices from $CONFIG." >&2
    exit 1
fi
N_TRAJ=$(echo "$INDICES" | tr ',' '\n' | wc -l)

# pack_data() does torch.stack(...).squeeze() which collapses the env dim
# when num_envs == 1 and breaks _create_obj_actor. Pad to >= 2; the second
# env round-robins back to dataIndices[0] (a duplicate rollout) and is
# ignored by the per-traj aggregation in player._run_eval_progress_stats.
NUM_ENVS=$N_TRAJ
if [ "$NUM_ENVS" -lt 2 ]; then NUM_ENVS=2; fi

if [ -z "$CKPT" ]; then
    CKPT="$RUN_DIR/nn/${EXPERIMENT}.pth"
fi
if [ ! -f "$CKPT" ]; then
    echo "Checkpoint '$CKPT' not found." >&2
    exit 1
fi

if [ -z "$STATS_OUT" ]; then
    STATS_OUT="$RUN_DIR/eval_progress_stats.json"
fi

echo "Run dir:        $RUN_DIR"
echo "Experiment:     $EXPERIMENT"
echo "Dexhand:        $DEXHAND  (side=$SIDE, usePIDControl=$USE_PID)"
echo "Trajectories:   $N_TRAJ"
echo "Checkpoint:     $CKPT"
echo "Stats out:      $STATS_OUT"

cd ~/code/humanoid/ManipTrans

# Same env scrubbing as the other launchers.
unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS CPP
unset CMAKE_ARGS CMAKE_PREFIX_PATH CONDA_BUILD_SYSROOT
unset AR AS LD NM RANLIB STRIP OBJCOPY OBJDUMP READELF
unset GCC GXX GCC_AR GCC_NM GCC_RANLIB
unset CC_FOR_BUILD CXX_FOR_BUILD NVCC_PREPEND_FLAGS
unset DEBUG_CFLAGS DEBUG_CXXFLAGS DEBUG_CPPFLAGS

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXX=/usr/bin/g++
export SETUPTOOLS_USE_DISTUTILS=stdlib

PID_ARG=""
if [ "$USE_PID" = "true" ]; then
    PID_ARG="usePIDControl=True"
fi

python main/rl/train.py \
    task=ResDexHand \
    "dexhand=${DEXHAND}" \
    "side=${SIDE}" \
    headless=true \
    "num_envs=${NUM_ENVS}" \
    test=true \
    randomStateInit=false \
    "rh_base_model_checkpoint=${RH_BASE}" \
    "lh_base_model_checkpoint=${LH_BASE}" \
    "dataIndices=[${INDICES}]" \
    "actionsMovingAverage=${ACTIONS_MA}" \
    ${PID_ARG} \
    "checkpoint=${CKPT}" \
    eval_progress_stats=true \
    "eval_stats_out=${STATS_OUT}" \
    wandb_activate=False
