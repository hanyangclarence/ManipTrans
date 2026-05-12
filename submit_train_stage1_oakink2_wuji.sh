#!/bin/bash
#SBATCH --job-name=oakink2-stage1-wuji
#SBATCH --output=logs/slurm_outputs/stage1-wuji-%j.out
#SBATCH --error=logs/slurm_outputs/stage1-wuji-%j.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a16

# Stage 1 (DexHandImitator) on the full OakInk2 right-hand subset for the wuji
# floating-hand baseline. Reads indices_combined.txt (204 trajectories =
# 93 cup + 98 spoon + 13 stick, produced by filter_oakink2_indices.py) and
# round-robins them across `num_envs` per `dexhandimitator.py:320`.
#
# The output checkpoint is the prerequisite for `submit_train_stage2_oakink2.sh`
# and `submit_train_partition_array.sh` (both Stage 2 launchers expect
# `assets/imitator_rh_wujihand.pth`). After training finishes, copy:
#
#   cp runs/imitator_oakink2_wuji_rh_<ts>__*/nn/imitator_oakink2_wuji_rh_<ts>.pth \
#      assets/imitator_rh_wujihand.pth
#
# Single-side (RH) only because indices_combined.txt is the RH subset of
# OakInk2 (filter_oakink2_indices.py uses obj_list_rh — see
# oakink2_dataset_dexhand_rh.py:215). If you later need bimanual stage-2,
# train an LH imitator separately on an LH index list.
#
# Usage:
#   sbatch submit_train_stage1_oakink2_wuji.sh                 # all 204 trajectories
#   sbatch submit_train_stage1_oakink2_wuji.sh 50              # first 50 only
#   sbatch submit_train_stage1_oakink2_wuji.sh all <resume_ckpt>

set -eo pipefail

N_TRAJ_ARG="${1:-all}"
RESUME_CKPT="${2:-}"

if [ "$N_TRAJ_ARG" != "all" ] && ! [[ "$N_TRAJ_ARG" =~ ^[1-9][0-9]*$ ]]; then
    echo "n_traj must be 'all' or a positive integer, got '$N_TRAJ_ARG'." >&2
    exit 1
fi

if [ -n "$RESUME_CKPT" ] && [ ! -f "$RESUME_CKPT" ]; then
    echo "Resume checkpoint '$RESUME_CKPT' not found." >&2
    exit 1
fi

cd ~/code/humanoid/ManipTrans
mkdir -p logs/slurm_outputs

INDEX_FILE="indices_combined.txt"
if [ ! -f "$INDEX_FILE" ]; then
    echo "Index file '$INDEX_FILE' not found. Run filter_oakink2_indices.py first." >&2
    exit 1
fi

TOTAL=$(wc -l < "$INDEX_FILE")
if [ "$N_TRAJ_ARG" = "all" ]; then
    N=$TOTAL
    N_TAG="all"
else
    N=$N_TRAJ_ARG
    if [ "$N" -gt "$TOTAL" ]; then
        echo "Requested n_traj=$N exceeds available (${TOTAL}). Capping to ${TOTAL}." >&2
        N=$TOTAL
    fi
    N_TAG="n${N}"
fi

INDICES=$(head -n "$N" "$INDEX_FILE" | paste -sd ,)

# Sanity: confirm the matching retargeting pickle exists for each picked index.
# Path comes from oakink2_dataset_dexhand_rh.py:218 via str(WujiHandRH()) = 'wujihand_rh'.
DUMP_DIR="data/retargeting/OakInk-v2/mano2wujihand_rh"
MISSING=0
while read -r IDX; do
    HASH5="${IDX%@*}"
    STAGE="${IDX#*@}"
    EXPECTED=$(ls "data/OakInk-v2/anno_preview/" | sort | grep "++seq__${HASH5}" | head -n1 | sed "s/\.pkl\$//")
    if [ -z "$EXPECTED" ] || [ ! -f "${DUMP_DIR}/${EXPECTED}@${STAGE}.pkl" ]; then
        echo "  WARNING: no retargeted pickle for ${IDX}" >&2
        MISSING=$((MISSING + 1))
    fi
done < <(echo "$INDICES" | tr ',' '\n')
if [ "$MISSING" -gt 0 ]; then
    echo "  ${MISSING}/${N} indices have no retargeted pickle on disk." >&2
    echo "  Stage-1 will train but those envs will use zero DOF init, hurting convergence." >&2
fi

echo "Stage:     1 (DexHandImitator)"
echo "Dexhand:   wujihand (floating, RH)"
echo "N traj:    ${N} (requested ${N_TRAJ_ARG}, available ${TOTAL})"
echo "Resume:    ${RESUME_CKPT:-<none>}"
echo "Job ID:    ${SLURM_JOB_ID:-local}"

module load conda/latest
conda activate maniptrans

unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS CPP
unset CMAKE_ARGS CMAKE_PREFIX_PATH CONDA_BUILD_SYSROOT
unset AR AS LD NM RANLIB STRIP OBJCOPY OBJDUMP READELF
unset GCC GXX GCC_AR GCC_NM GCC_RANLIB
unset CC_FOR_BUILD CXX_FOR_BUILD NVCC_PREPEND_FLAGS
unset DEBUG_CFLAGS DEBUG_CXXFLAGS DEBUG_CPPFLAGS

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXX=/usr/bin/g++
export SETUPTOOLS_USE_DISTUTILS=stdlib

export WANDB_DIR="./wandb"
mkdir -p "$WANDB_DIR"

nvidia-smi -L

EXPERIMENT="imitator_oakink2_wuji_rh_${N_TAG}_$(date +%Y%m%d_%H%M%S)"
echo "Run name:  ${EXPERIMENT}"

CKPT_ARG=""
if [ -n "$RESUME_CKPT" ]; then
    CKPT_ARG="checkpoint=${RESUME_CKPT}"
fi

# Stage 1 doesn't read rh/lh_base_model_checkpoint (those are Stage-2 only).
# wuji-specific knobs:
#   - translationScale=2.7 / orientationScale=0.4 compensate the 2.7x linear and
#     ~4x angular wrist-inertia disparity vs inspire (see benchmark_wrist_force.py).
#   - dof_kp / dof_kd are read from the WujiHand class via the getattr fallback
#     added to dexhandimitator.py:254-271.
python main/rl/train.py \
    task=DexHandImitator \
    dexhand=wujihand \
    side=RH \
    headless=true \
    num_envs=4096 \
    learning_rate=2e-4 \
    test=false \
    randomStateInit=true \
    "dataIndices=[${INDICES}]" \
    actionsMovingAverage=0.4 \
    translationScale=2.7 \
    orientationScale=0.4 \
    experiment="${EXPERIMENT}" \
    ${CKPT_ARG} \
    wandb_activate=True \
    wandb_project=maniptrans-oakink2-baseline \
    wandb_name="${EXPERIMENT}" \
    "wandb_tags=[stage1,wujihand,${N_TAG}]"
