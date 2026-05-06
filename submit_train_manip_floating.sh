#!/bin/bash
#SBATCH --job-name=maniptrans-manip-float
#SBATCH --output=logs/slurm_outputs/slurm-%j.out
#SBATCH --error=logs/slurm_outputs/slurm-%j.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a16

# Stage 2 manipulation on a floating dexhand (ResDexHandManus, the floating-hand
# Manus stage-2 task). Default hand is `inspire` — the ManipTrans canonical hand
# whose reward weights and PD gains the upstream paper actually tuned against.
#
# Stage 2 is a residual policy on top of a frozen Stage-1 imitator. The base
# imitator (4th arg, REQUIRED) must have been trained on the SAME pipeline
# (DexHandImitatorManus + same dexhand) — the shipped assets/imitator_*_shadow.pth
# were trained on the full-body pipeline (obs_dim=367) and won't load against
# the Manus fingertip-only pipeline (obs_dim=169). The 5th arg (optional) is a
# warm-start for the residual policy itself, separate from the frozen base.
#
# Memory bumped to 16G vs Stage 1's 8G because Stage 2 also instantiates the
# dynamic object actor + VHACD collision meshes.
#
# Usage:
#   sbatch submit_train_manip_floating.sh inspire marker_pen all <stage1_ckpt.pth>
#   sbatch submit_train_manip_floating.sh shadow  marker_pen 10  <stage1_ckpt.pth>
#   sbatch submit_train_manip_floating.sh xhand   hammer_mixed all <stage1_ckpt.pth> <residual_warmstart.pth>

set -eo pipefail

DEXHAND="${1:-inspire}"
DATASET="${2:-marker_pen}"
N_TRAJ_ARG="${3:-all}"
BASE_CKPT="${4:?usage: $0 <hand> <dataset> <n_traj> <stage1_imitator_ckpt> [residual_warmstart_ckpt]}"
CKPT="${5:-}"

case "$DEXHAND" in
    inspire|wujihand|shadow|xhand) ;;
    *)
        echo "Unsupported dexhand '$DEXHAND' — expected 'inspire', 'wujihand', 'shadow', or 'xhand'" >&2
        exit 1
        ;;
esac

if [ ! -f "$BASE_CKPT" ]; then
    echo "Stage-1 base checkpoint '$BASE_CKPT' not found." >&2
    exit 1
fi

case "$DATASET" in
    marker_pen)   ROOT="manus_marker_pen" ;;
    hammer_mixed) ROOT="manus_hammer_mixed" ;;
    *)
        echo "Unknown dataset '$DATASET' — expected 'marker_pen' or 'hammer_mixed'" >&2
        exit 1
        ;;
esac

if [ "$N_TRAJ_ARG" != "all" ] && ! [[ "$N_TRAJ_ARG" =~ ^[1-9][0-9]*$ ]]; then
    echo "n_traj must be 'all' or a positive integer, got '$N_TRAJ_ARG'" >&2
    exit 1
fi

if [ -n "$CKPT" ] && [ ! -f "$CKPT" ]; then
    echo "Checkpoint '$CKPT' not found." >&2
    exit 1
fi

cd ~/code/humanoid/ManipTrans

mkdir -p logs/slurm_outputs

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

export WANDB_API_KEY=28b3c634497c0dc6c16767729d4719b1012a94f2
export SETUPTOOLS_USE_DISTUTILS=stdlib

export WANDB_DIR="wandb_${SLURM_JOB_ID:-local}"
mkdir -p "$WANDB_DIR"

nvidia-smi

SEQ_DIR="data/${ROOT}/sequences"
if [ ! -d "$SEQ_DIR" ]; then
    echo "Sequence dir '$SEQ_DIR' not found." >&2
    exit 1
fi

if [ "$N_TRAJ_ARG" = "all" ]; then
    SELECTED=$(ls "$SEQ_DIR"/*.pkl)
    N_TAG="all"
else
    SELECTED=$(ls "$SEQ_DIR"/*.pkl | head -n "$N_TRAJ_ARG")
    N_TAG="n${N_TRAJ_ARG}"
fi

INDICES=$(echo "$SELECTED" \
    | xargs -n1 basename -s .pkl \
    | sed "s|^|manus@${ROOT}@|" \
    | paste -sd ,)

N_TRAJ=$(echo "$INDICES" | tr ',' '\n' | wc -l)
echo "Hand: ${DEXHAND}    Dataset: ${ROOT}    n_trajectories=${N_TRAJ} (requested=${N_TRAJ_ARG})"
echo "Stage-1 base ckpt: ${BASE_CKPT}"
echo "Residual warm-start: ${CKPT:-<none>}"

SELECTED_PATHS="$SELECTED" python - <<'PY'
import os, pickle, sys
paths = [p for p in os.environ["SELECTED_PATHS"].splitlines() if p]
lens = [pickle.load(open(p, "rb"))["obj_trajectory"].shape[0] for p in paths]
print(f"  frames: min={min(lens)} max={max(lens)} mean={sum(lens)//len(lens)}")
# Stage 2 episodeLength is 1200 by default (ResDexHandManus.yaml).
if max(lens) > 1200:
    print(f"  WARNING: max length {max(lens)} > episodeLength=1200; bump ++task.env.episodeLength", file=sys.stderr)
PY

EXPERIMENT="manip_${DEXHAND}_${DATASET}_${N_TAG}_$(date +%Y%m%d_%H%M%S)"
echo "Run: ${EXPERIMENT}"

CKPT_ARG=""
if [ -n "$CKPT" ]; then
    CKPT_ARG="checkpoint=${CKPT}"
fi

python main/rl/train.py \
    task=ResDexHandManus \
    dexhand=${DEXHAND} \
    side=RH \
    rh_base_model_checkpoint="${BASE_CKPT}" \
    lh_base_model_checkpoint="${BASE_CKPT}" \
    "dataIndices=[${INDICES}]" \
    num_envs=4096 \
    experiment="${EXPERIMENT}" \
    ${CKPT_ARG} \
    wandb_activate=True \
    wandb_project=maniptrans-manus \
    wandb_name="${EXPERIMENT}" \
    "wandb_tags=[manip,${DEXHAND},${DATASET}]"
