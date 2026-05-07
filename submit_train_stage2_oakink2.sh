#!/bin/bash
#SBATCH --job-name=oakink2-stage2
#SBATCH --output=logs/slurm_outputs/stage2-%j.out
#SBATCH --error=logs/slurm_outputs/stage2-%j.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a16

# Stage 2 (residual manipulation) on OakInk2 inspire trajectories. Reads the
# per-object index file produced by filter_oakink2_indices.py and feeds the
# first N lines into `dataIndices=[...]`. The env round-robins these across
# `num_envs` per `dexhandimitator.py:320`/`dexhandmanip_sh.py:334`.
#
# The training has no hard cap and no plateau early-stop (root-config defaults
# are 9999999999999 for both `max_iterations` and `early_stop_epochs`), so the
# job runs until SLURM time limit or you `scancel` it.
#
# Usage:
#   sbatch submit_train_stage2_oakink2.sh <object> [n_traj] [resume_ckpt]
#
#     <object>     : one of {cup, spoon, stick}
#     [n_traj]     : 'all' (default) | positive integer (e.g. 1, 5, 10).
#                    If integer, only the first N lines of indices_<object>.txt
#                    are passed to dataIndices.
#     [resume_ckpt]: optional path to a Stage-2 checkpoint to resume from
#                    (i.e. the residual-policy ckpt, NOT the imitator base).
#
# Examples:
#   sbatch submit_train_stage2_oakink2.sh cup
#   sbatch submit_train_stage2_oakink2.sh spoon 10
#   sbatch submit_train_stage2_oakink2.sh stick 1
#   sbatch submit_train_stage2_oakink2.sh cup all runs/oakink2_inspire_cup_all_*/nn/oakink2_inspire_cup_all_*.pth

set -eo pipefail

OBJECT="${1:?usage: $0 <cup|spoon|stick> [n_traj=all] [resume_ckpt]}"
N_TRAJ_ARG="${2:-all}"
RESUME_CKPT="${3:-}"

case "$OBJECT" in
    cup|spoon|stick) ;;
    *)
        echo "Unsupported object '$OBJECT' — expected 'cup', 'spoon', or 'stick'." >&2
        exit 1
        ;;
esac

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

INDEX_FILE="indices_${OBJECT}.txt"
if [ ! -f "$INDEX_FILE" ]; then
    echo "Index file '$INDEX_FILE' not found. Run filter_oakink2_indices.py first." >&2
    exit 1
fi

# Slice the index file to the requested count.
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
DUMP_DIR="data/retargeting/OakInk-v2/mano2inspire_rh"
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
    echo "  Stage-2 will train but those envs will use zero DOF init, hurting convergence." >&2
fi

echo "Object:    ${OBJECT}"
echo "N traj:    ${N} (requested ${N_TRAJ_ARG}, available ${TOTAL})"
echo "Resume:    ${RESUME_CKPT:-<none>}"
echo "Job ID:    ${SLURM_JOB_ID:-local}"

module load conda/latest
conda activate maniptrans

# Same compiler-var unsets as the other submit scripts (maniptrans conda env
# injects build vars that conflict with system gcc on extension rebuilds).
unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS CPP
unset CMAKE_ARGS CMAKE_PREFIX_PATH CONDA_BUILD_SYSROOT
unset AR AS LD NM RANLIB STRIP OBJCOPY OBJDUMP READELF
unset GCC GXX GCC_AR GCC_NM GCC_RANLIB
unset CC_FOR_BUILD CXX_FOR_BUILD NVCC_PREPEND_FLAGS
unset DEBUG_CFLAGS DEBUG_CXXFLAGS DEBUG_CPPFLAGS

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXX=/usr/bin/g++
export SETUPTOOLS_USE_DISTUTILS=stdlib

# All jobs share ./wandb (wandb run subdirs are unique per experiment name,
# which embeds a timestamp, so concurrent jobs don't collide).
export WANDB_DIR="./wandb"
mkdir -p "$WANDB_DIR"

nvidia-smi -L

EXPERIMENT="oakink2_inspire_${OBJECT}_${N_TAG}_$(date +%Y%m%d_%H%M%S)"
echo "Run name:  ${EXPERIMENT}"

CKPT_ARG=""
if [ -n "$RESUME_CKPT" ]; then
    CKPT_ARG="checkpoint=${RESUME_CKPT}"
fi

# We deliberately omit `early_stop_epochs` and `max_iterations` so the root
# config's 9999999999999 defaults take effect — the job runs until SLURM
# kills it or you scancel. README pattern preserved otherwise.
python main/rl/train.py \
    task=ResDexHand \
    dexhand=inspire \
    side=RH \
    headless=true \
    num_envs=4096 \
    learning_rate=2e-4 \
    test=false \
    randomStateInit=false \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    "dataIndices=[${INDICES}]" \
    actionsMovingAverage=0.4 \
    experiment="${EXPERIMENT}" \
    ${CKPT_ARG} \
    wandb_activate=True \
    wandb_project=maniptrans-oakink2-baseline \
    wandb_name="${EXPERIMENT}" \
    "wandb_tags=[stage2,inspire,${OBJECT},${N_TAG}]"
